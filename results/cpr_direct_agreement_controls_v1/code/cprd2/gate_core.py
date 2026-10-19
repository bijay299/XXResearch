"""Direct native-agreement gate: shared machinery.

The REAL (patched) `GEM.__call__` and the PRISTINE pinned-source `GEM.__call__` are driven through a minimal
`FakeWrapper` on identical cloned FP32 operands so that the actual native loss branch executes inside the
actual native loop (teachers under no_grad + disable_adapter, `wrapper.train()`, per-state backward,
optimizer.step, `(gamma*sum)/n_latents` reduction). The fake supplies ONLY the three network outputs.
Harness stubs (disclosed): `memory_cleanup` -> no-op (CUDA cache management only, no numerics).
"""
import contextlib
import hashlib
import importlib
import io
import json
import os
import random
import re
import subprocess
import types

import numpy as np
import torch

from cprd2 import losses as LS
from cprd2 import protocol as P
from cprd2.common import ROOT, dump
from cprd2.flux_run import rng_digest

GEM_SRC = os.path.join(ROOT, "GEM_src")
GEM_PIN = "c328fc6bec687efe60223498d1b6bed2e5fbcffd"
ERASURE_PATH = "gem/operators/erasure/erasure_gem.py"
ETA = P.SETTINGS["eta"]
GAMMA = P.SETTINGS["gamma"]
N_LATENTS = P.SETTINGS["n_latents_per_window"]
TOL = P.TOL


class GateFailure(RuntimeError):
    pass


class Recorder:
    def __init__(self, path=None):
        self.R, self.path = {}, path

    def chk(self, name, ok, **details):
        self.R[name] = {"pass": bool(ok), **details}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(details, default=str)[:300]}", flush=True)
        if not ok:
            self.flush(failed=name)
            raise GateFailure(f"required gate check failed: {name}")
        return True

    def note(self, name, **details):
        self.R[name] = {"note": True, **details}

    def flush(self, failed=None):
        if self.path:
            dump({"checks": self.R, "failed": failed,
                  "all_pass": failed is None and all(v.get("pass", True) for v in self.R.values())}, self.path)


# ----------------------------------------------------------------------------- pristine + patched modules
def pristine_source():
    return subprocess.check_output(["git", "-C", GEM_SRC, "show", f"{GEM_PIN}:{ERASURE_PATH}"]).decode()


def load_pristine_module():
    src = pristine_source()
    mod = types.ModuleType("gem_pristine_erasure_gem")
    mod.__file__ = f"<{GEM_PIN}:{ERASURE_PATH}>"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod, src


def load_patched_module():
    return importlib.import_module("gem.operators.erasure.erasure_gem")


def native_block_lines(src):
    """The original three native-loss source lines + the scaling line, taken verbatim from the pinned source."""
    want = {"dist_pos": r"^\s*dist_pos = .*pairwise_distance\(pred_flat, pos_flat, p=2\)\s*$",
            "dist_neg": r"^\s*dist_neg = .*pairwise_distance\(pred_flat, neg_flat, p=2\)\s*$",
            "target_loss": r"^\s*target_loss = torch\.relu\(dist_pos - config\.eta \* dist_neg\)\.sum\(\)\s*$",
            "scale": r"^\s*loss = \(config\.gamma \* loss\) / n_latents\s*$"}
    out = {}
    for k, rx in want.items():
        m = [ln for ln in src.splitlines() if re.match(rx, ln)]
        if len(m) != 1:
            raise GateFailure(f"pinned source line {k} found {len(m)} times")
        out[k] = m[0].strip()
    return out


# ----------------------------------------------------------------------------- fake wrapper
class FakeWrapper(torch.nn.Module):
    """Supplies only the three network outputs; every other trainer call is the real trainer's."""

    def __init__(self, V, A, B, student_dtype=torch.float32):
        super().__init__()
        self.V, self.A, self.B = V, A, B
        self.w = torch.nn.Parameter(torch.zeros(V.shape, dtype=torch.float32))
        self.pipe_config = types.SimpleNamespace(num_inference_steps=28, image_size=512)
        self.device = torch.device("cpu")
        self.scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))
        self.student_dtype = student_dtype
        self._off = 0
        self.teacher_calls = 0
        self.student_calls = 0
        self.modes_seen = []

    def encode_prompt(self, prompt):
        return (torch.zeros(1) if prompt == "" else torch.ones(1),)

    def set_timesteps(self, *a, **k):
        return None

    @contextlib.contextmanager
    def disable_adapter(self):
        self._off += 1
        try:
            yield
        finally:
            self._off -= 1

    def partially_denoise_until_timestep(self, **kw):
        d = self.V.shape[1]
        return None, [torch.full((1, 1, d), float(j)) for j in range(N_LATENTS)]

    def predict_noise(self, idx, latents, embeds):
        rows = latents[:, 0, 0].long()
        tag = float(embeds[0].flatten()[0])
        if self._off:
            assert not torch.is_grad_enabled(), "teacher call outside no_grad"
            self.teacher_calls += 1
            return (self.A if tag == 0.0 else self.B)[rows].clone()
        assert tag == 1.0 and torch.is_grad_enabled(), "student call must be grad-enabled on the target prompt"
        self.student_calls += 1
        self.modes_seen.append(bool(self.training))
        return (self.V[rows] + self.w[rows]).to(self.student_dtype)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class RecordingHook:
    """Wraps a loss hook; verifies return value, input immutability and RNG invariance at every call."""

    def __init__(self, inner):
        self.inner, self.calls, self.returns_none, self.ps, self.violations = inner, 0, [], [], []
        self.arg_dtypes = []

    def __call__(self, pred, pos, neg):
        self.calls += 1
        before = (pred.detach().clone(), pos.clone(), neg.clone())
        rng0 = rng_digest()
        self.arg_dtypes.append([str(t.dtype) for t in (pred, pos, neg)])
        out = self.inner(pred, pos, neg)
        if rng_digest() != rng0:
            self.violations.append("rng_changed")
        if not (torch.equal(before[0], pred.detach()) and torch.equal(before[1], pos) and torch.equal(before[2], neg)):
            self.violations.append("input_mutated")
        self.returns_none.append(out is None)
        self.ps.append(None if out is None else out.detach().clone())
        self.last_in = tuple(t.detach().clone() for t in (pred, pos, neg))
        return out


class RecordingCapture:
    def __init__(self):
        self.calls, self.violations, self.idx = 0, [], []

    def __call__(self, processed_index, global_step, batch_start, pred, anchor, target):
        self.calls += 1
        self.idx.append(processed_index)
        rng0 = rng_digest()
        _ = (pred.sum(), anchor.sum(), target.sum())
        if rng_digest() != rng0:
            self.violations.append("rng_changed")
        return "ignored-return-value"


def run_trainer(module, V, A, B, n_rows, hook=None, capture=None, student_dtype=torch.float32,
                budget=None, parallel=None, lr=1e-3, seed=0):
    """Run `module.GEM().__call__` for ONE window prefix. Returns dict with w.grad, scalar loss, target_loss."""
    module.memory_cleanup = lambda *a, **k: None
    W = FakeWrapper(V.clone(), A.clone(), B.clone(), student_dtype=student_dtype)
    cfg = module.GEMConfig(targets=["target"], max_step_budget=budget or n_rows,
                           max_parallel_timesteps=parallel or n_rows, train_seed=seed, val_seed=seed,
                           use_wandb=False, learning_rate=lr, eta=ETA, gamma=GAMMA)
    if hook is not None:
        cfg.cpr_loss_hook = hook
    if capture is not None:
        cfg.cpr_capture_hook = capture
    op = module.GEM()
    logs = []
    orig = op.log_loss_and_metrics_to_wandb

    def spy(loss, step, metrics=None, **k):
        logs.append((loss, metrics))
        return orig(loss=loss, step=step, metrics=metrics, **k)

    op.log_loss_and_metrics_to_wandb = spy
    rng0 = None
    with _quiet():
        op(W, cfg)
    per_state = [l for l in logs if l[1] is not None]
    return {"grad": W.w.grad.detach().clone(), "w": W.w.detach().clone(), "loss": [l[0] for l in per_state],
            "target_loss": [l[1]["target_loss"] for l in per_state], "wrapper": W, "all_logs": logs}


# ----------------------------------------------------------------------------- independent FP64 references
def ref64(arm, v, a, b, eta=ETA, eps=1e-6):
    """Independent FP64 reference (separate code path from losses.py). Returns (per_sample, args) with graph on v."""
    v = v.detach().double().clone().requires_grad_(True)
    a, b = a.double(), b.double()
    d = v.shape[1]
    dp = torch.sqrt(((v - a + eps) ** 2).sum(1))
    dm = torch.sqrt(((v - b + eps) ** 2).sum(1))
    Lv = torch.sqrt(((b - a) ** 2).sum(1))
    live = (Lv > 1e-6).double()
    e = (b - a) / torch.where(Lv > 1e-6, Lv, torch.ones_like(Lv)).unsqueeze(1) * live.unsqueeze(1)
    alpha = ((b - v) * e).sum(1)
    table = {
        "native_gem": (dp - eta * dm, True),
        "cpr": (dp - eta * alpha, True),
        "centered_radial": (dp - eta * torch.sqrt(((v - b) ** 2).sum(1) + eps ** 2 * d), True),
        "truncated_anchor": (dp - (eta * Lv / (1 + eta)).detach(), True),
        "anchor_only": (dp, False),
        "axial_margin": (Lv - (1 + eta) * alpha, True),
    }
    arg, hinge = table[arm]
    ps = torch.relu(arg) if hinge else arg
    scale = Lv if arm == "axial_margin" else dp
    return ps, v, arg.detach(), scale.detach(), hinge


def ref64_full(arm, v, a, b, n_latents=N_LATENTS):
    ps, vv, arg, scale, hinge = ref64(arm, v, a, b)
    total = GAMMA * ps.sum() / n_latents
    g, = torch.autograd.grad(total, vv)
    return {"ps": ps.detach(), "grad": g, "scalar": float(total.detach()), "arg": arg, "scale": scale, "hinge": hinge}


def close(x, y, kind):
    t = TOL[kind]
    x, y = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    if x.shape != y.shape or not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        return False, float("nan")
    ok = bool(torch.allclose(x, y, rtol=t["rtol"], atol=t["atol"]))
    return ok, float((x - y).abs().max()) if x.numel() else 0.0


def away_from_boundary(ref, margin=0.1):
    """True per row where the hinge argument is >= margin*scale away from 0 (always True when no hinge)."""
    if not ref["hinge"]:
        return torch.ones_like(ref["arg"], dtype=torch.bool)
    return ref["arg"].abs() >= margin * ref["scale"]


# ----------------------------------------------------------------------------- fixtures (FP64-reference based selection)
def _arm_refs(v, a, b):
    return {arm: ref64_full(arm, v, a, b, N_LATENTS) for arm in LS.ARMS}


def build_fixture(seed, d, n, mode, margin=0.1, min_grad_frac=0.01, max_try=40000):
    """Rows v = a + t(b-a) + sigma*z. Accept rows that every arm sees >= margin away from its hinge boundary
    and whose active gradient has no near-zero element (so FP32-vs-FP64 comparison is well-conditioned).
    mode: 'active' (every hinge arm active), 'inactive' (every hinge arm inactive), 'mixed' (alternating).
    Selection depends only on FP64 references, never on the system under test."""
    rng = np.random.default_rng(seed)
    rows, tries = [], 0
    want = {"active": [True] * n, "inactive": [False] * n, "mixed": [(i % 2 == 0) for i in range(n)]}[mode]
    while len(rows) < n and tries < max_try:
        tries += 1
        want_active = want[len(rows)]
        a = rng.normal(size=d)
        L = float(rng.uniform(0.5, 2.0)) * np.sqrt(d)
        u = rng.normal(size=d)
        b = a + L * u / np.linalg.norm(u)
        t = float(rng.uniform(0.62, 1.6)) if want_active else float(rng.uniform(-0.6, 0.38))
        sig = float(rng.uniform(0.05, 0.5)) * L / np.sqrt(d)
        v = a + t * (b - a) + sig * rng.normal(size=d)
        V, A, B = (torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in (v, a, b))
        ok = True
        for arm in LS.ARMS:
            r = ref64_full(arm, V, A, B)
            if r["hinge"]:
                if not bool(away_from_boundary(r, margin).all()):
                    ok = False
                    break
                if (float(r["ps"][0]) > 0) != want_active:
                    ok = False
                    break
            if float(r["ps"][0]) > 0:
                g = r["grad"].abs()
                if float(g.min()) < min_grad_frac * float(g.max()):
                    ok = False
                    break
        if ok:
            rows.append((V[0], A[0], B[0]))
    if len(rows) < n:
        raise GateFailure(f"fixture {mode} d={d}: only {len(rows)}/{n} rows after {tries} tries")
    return tuple(torch.stack([r[i] for r in rows]) for i in range(3)), tries


def fixtures():
    out = {}
    (V, A, B), t1 = build_fixture(7001, 32, 8, "mixed")
    out["mixed_d32"] = (V, A, B, {"tries": t1})
    (V, A, B), t2 = build_fixture(7002, 256, 6, "mixed", min_grad_frac=0.0)  # REPAIR-1: filter infeasible at d=256 (0/3000 rows pass), see artifacts/repairs/repair_1.md
    out["mixed_d256"] = (V, A, B, {"tries": t2})
    (V, A, B), t3 = build_fixture(7003, 64, 4, "inactive")
    out["all_inactive_d64"] = (V, A, B, {"tries": t3})
    (V, A, B), t4 = build_fixture(7004, 64, 4, "active")
    out["all_active_d64"] = (V, A, B, {"tries": t4})
    return out


def exact_boundary_fixtures():
    """d=1 operands where the computed hinge argument is exactly 0 in FP32 for the stated arm. A claim is made
    ONLY if the arm's computed hinge is exactly 0.0; no root search."""
    e32 = float(torch.tensor(1e-6, dtype=torch.float32))
    f = lambda *x: tuple(torch.tensor([[float(t)]], dtype=torch.float32) for t in x)  # noqa: E731
    return {
        "native_gem": f(1.5, 0.0, 0.0),                 # a == b -> Dplus == Dminus exactly
        "truncated_anchor": f(0.0, 0.0, 2 * e32),        # Dplus = eps32 ; L/2 = eps32
        "cpr": f(32.0, 0.0, 64.0),                       # eps absorbed: Dplus = 32 = alpha
        "axial_margin": f(1.0, 0.0, 2.0),                # L - 2 alpha = 2 - 2 = 0
        "centered_radial": f(0.0, 0.0, 0.0),             # Dplus vs sqrt(eps^2 d): exact zero NOT guaranteed
    }


def digest_tensors(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
