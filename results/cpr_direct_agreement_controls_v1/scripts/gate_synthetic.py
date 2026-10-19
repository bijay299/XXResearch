#!/usr/bin/env python3
"""CPU half of the direct native-agreement gate (synthetic fixtures through the REAL native trainer loop).
Any required failure raises and stops immediately (artifact written first)."""
import os
import sys

import torch
import torch.nn.functional as F

from cprd2 import losses as LS
from cprd2 import protocol as P
from cprd2.common import ROOT, utc
from cprd2.flux_run import rng_digest
from cprd2.gate_checks import hooked_arm, native_three_way
from cprd2.gate_core import (ETA, GAMMA, N_LATENTS, GateFailure, Recorder, RecordingCapture, RecordingHook,
                             build_fixture, close, exact_boundary_fixtures, fixtures, load_patched_module,
                             load_pristine_module, native_block_lines, ref64_full, run_trainer, pristine_source)


def arg32(arm, v, a, b, src):
    """Computed FP32 hinge argument (pre-ReLU) for the exact-boundary claim."""
    if arm == "native_gem":
        ns = {"torch": torch, "pred_flat": v, "pos_flat": a, "neg_flat": b,
              "config": type("C", (), {"eta": ETA})()}
        L = native_block_lines(src)
        exec(L["dist_pos"], ns)
        exec(L["dist_neg"], ns)
        return float((ns["dist_pos"] - ETA * ns["dist_neg"])[0])
    g = LS.geometry(v, a, b)
    if arm == "cpr":
        return float((g["d_plus"] - ETA * g["alpha"])[0])
    if arm == "truncated_anchor":
        return float((g["d_plus"] - (ETA * g["L"] / (1 + ETA)))[0])
    if arm == "axial_margin":
        return float((g["L"] - (1 + ETA) * g["alpha"])[0])
    if arm == "centered_radial":
        diff = g["v"] - g["b"]
        rad = torch.sqrt((diff * diff).sum(1) + (LS.EPS ** 2) * g["d"])
        return float((g["d_plus"] - ETA * rad)[0])
    raise KeyError(arm)


def main():
    out_path = os.path.join(ROOT, "artifacts", "gate", "synthetic_gate.json")
    rec = Recorder(out_path)
    rec.note("meta", started_utc=utc(), tolerances=P.TOL, eta=ETA, gamma=GAMMA, n_latents=N_LATENTS,
             harness_stubs=["memory_cleanup -> no-op (CUDA cache management only)"],
             scope="synthetic FP32 fixtures through the real native trainer loop on CPU (FakeWrapper supplies only "
                   "the three network outputs); real-model operands are replayed in gate_replay.py")
    patched = load_patched_module()
    pristine, psrc = load_pristine_module()
    rec.chk("pristine_source_native_block_present", len(native_block_lines(psrc)) == 4)
    pt = open(patched.__file__).read()
    rec.chk("patched_retains_original_native_block_verbatim",
            all(l in pt for l in native_block_lines(psrc).values()),
            patched_file=patched.__file__)
    # ---------------- fixtures
    fx = fixtures()
    rec.note("fixtures", names=list(fx), meta={k: v[3] for k, v in fx.items()},
             shapes={k: list(v[0].shape) for k, v in fx.items()},
             selection="rows accepted by FP64-reference margin>=0.1*scale for every arm and no near-zero active gradient "
                       "element; independent of the system under test")
    for name, (V, A, B, _) in fx.items():
        ok, o = native_three_way(rec.R, name, V, A, B, patched, pristine, psrc)
        rec.chk(f"native_agreement::{name}", ok, **{k: o[k] for k in o if k in ("fp64_ref", "hook_protocol")},
                bitwise={k: v for k, v in o.items() if k.endswith("_vs_A")})
        for arm in LS.ARMS:
            ok, o = hooked_arm(rec.R, name, arm, V, A, B, patched)
            rec.chk(f"arm_vs_fp64::{arm}::{name}", ok, **o)
        # activity coverage (nontrivial active AND inactive rows)
        cov = {}
        for arm in LS.ARMS:
            r = ref64_full(arm, V, A, B)
            cov[arm] = {"active": int((r["ps"] > 0).sum()), "inactive": int((r["ps"] <= 0).sum())}
        rec.note(f"activity::{name}", rows=cov)
        if name.startswith("mixed"):
            for arm in LS.ARMS:
                if arm != "anchor_only":
                    rec.chk(f"activity_mixed::{arm}::{name}", cov[arm]["active"] >= 2 and cov[arm]["inactive"] >= 2, **cov[arm])
    # ---------------- teacher detachment / FP32 preparation
    V, A, B, _ = fx["mixed_d32"]
    for arm in LS.ARMS:
        a2, b2, v2 = A.clone().requires_grad_(True), B.clone().requires_grad_(True), V.clone().requires_grad_(True)
        ps = LS.per_sample(arm, v2, a2, b2, ETA)
        s = (F.pairwise_distance(v2, a2) - F.pairwise_distance(v2, b2)).relu().sum() if ps is None else ps.sum()
        s.backward()
        # native arm is evaluated by the trainer (pristine lines); its detachment is enforced by the trainer's no_grad
        if ps is not None:
            rec.chk(f"teacher_detached::{arm}", a2.grad is None and b2.grad is None and v2.grad is not None
                    and torch.isfinite(v2.grad).all().item(), ps_requires_grad=ps.requires_grad)
    vb, ab, bb = (t.to(torch.bfloat16) for t in (V, A, B))
    for arm in LS.ARMS:
        if arm == "native_gem":
            continue
        ps = LS.per_sample(arm, vb.requires_grad_(True), ab, bb, ETA)
        rec.chk(f"fp32_preparation::{arm}", ps.dtype == torch.float32 and torch.isfinite(ps).all().item(), dtype=str(ps.dtype))
    # BF16 student through the real trainer: hook receives FP32
    for arm in LS.ARMS:
        hk = RecordingHook(LS.make_dispatch(arm, ETA))
        r = run_trainer(patched, V, A, B, V.shape[0], hook=hk, student_dtype=torch.bfloat16)
        pred_in, pos_in, neg_in = hk.last_in
        ref = ref64_full(arm, pred_in, pos_in, neg_in)
        ok_sc, e_sc = close(r["loss"][0], ref["scalar"], "loss")
        g, rg = r["grad"].double().flatten(), ref["grad"].flatten()
        cos = float(g @ rg / (g.norm() * rg.norm() + 1e-30))
        rec.chk(f"bf16_student_path::{arm}",
                all(d == "torch.float32" for d in hk.arg_dtypes[0]) and ok_sc and cos > 0.999 and torch.isfinite(r["grad"]).all().item(),
                arg_dtypes=hk.arg_dtypes[0], scalar_err=e_sc, grad_cosine_vs_fp64=cos,
                note="forward value at the loss tolerance; gradient passes through BF16 rounding at the student output, "
                     "so only direction (cosine>0.999) is asserted here")
    # ---------------- zero and small contrast
    g_ = torch.Generator().manual_seed(7100)
    Vz = torch.randn(4, 16, generator=g_)
    Az = torch.randn(4, 16, generator=g_)
    for arm in ("cpr", "truncated_anchor", "anchor_only"):
        ok, o = hooked_arm(rec.R, "zero_contrast", arm, Vz, Az, Az.clone(), patched)
        rec.chk(f"zero_contrast::{arm}", ok, **o)
    hk0 = RecordingHook(LS.make_dispatch("centered_radial", ETA))
    r0 = run_trainer(patched, Vz, Az, Az.clone(), 4, hook=hk0)
    rec.chk("zero_contrast::centered_radial_finite", torch.isfinite(r0["grad"]).all().item() and torch.isfinite(hk0.ps[0]).all().item(),
            note="hinge is O(eps) by construction at zero contrast (boundary regime); finiteness only, no FP64 value claim")
    ok, o = native_three_way(rec.R, "zero_contrast", Vz, Az, Az.clone(), patched, pristine, psrc)
    rec.chk("zero_contrast::native_gem", ok, fp64_ref=o["fp64_ref"])
    try:
        run_trainer(patched, Vz, Az, Az.clone(), 4, hook=RecordingHook(LS.make_dispatch("axial_margin", ETA)))
        raised = False
    except LS.AxialDomainViolation:
        raised = True
    rec.chk("zero_contrast::axial_raises_arm_failure_via_trainer", raised)
    try:
        LS.per_sample("axial_margin", Vz, Az, Az.clone(), ETA)
        raised2 = False
    except LS.AxialDomainViolation:
        raised2 = True
    rec.chk("zero_contrast::axial_raises_direct", raised2)
    Vs = torch.randn(4, 16, generator=g_)
    As = torch.zeros(4, 16)
    Bs = torch.full((4, 16), 2.0e-6 / 4.0)          # L = 2e-6 > tau_L: live but tiny
    for arm in LS.ARMS:
        hk = RecordingHook(LS.make_dispatch(arm, ETA))
        r = run_trainer(patched, Vs, As, Bs, 4, hook=None if arm == "native_gem" else hk)
        rec.chk(f"small_contrast_finite::{arm}", torch.isfinite(r["grad"]).all().item() and all(map(lambda x: x == x, r["loss"])),
                L=float(torch.linalg.vector_norm(Bs - As, dim=1)[0]))
    # ---------------- exact boundary / ReLU at zero
    z = torch.zeros((), requires_grad=True)
    torch.relu(z).backward()
    rec.chk("relu_scalar_semantics_at_zero", float(torch.relu(torch.zeros(()))) == 0.0 and float(z.grad) == 0.0)
    claimed, unclaimed = [], {}
    for arm, (v, a, b) in exact_boundary_fixtures().items():
        h = arg32(arm, v, a, b, psrc)
        if h == 0.0:
            hk = RecordingHook(LS.make_dispatch(arm, ETA))
            r = run_trainer(patched, v, a, b, 1, hook=None if arm == "native_gem" else hk)
            ok = r["target_loss"][0] == 0.0 and float(r["grad"].abs().max()) == 0.0
            rec.chk(f"exact_boundary::{arm}", ok, computed_hinge=h, target_loss=r["target_loss"][0],
                    grad_abs_max=float(r["grad"].abs().max()))
            claimed.append(arm)
        else:
            unclaimed[arm] = h
    rec.note("exact_boundary_coverage", claimed=claimed, not_exactly_zero=unclaimed,
             anchor_only="no hinge (N/A)",
             statement="claims restricted to arms whose computed FP32 hinge argument was exactly 0.0; no root search")
    # ---------------- window-level invariance (hooks/capture are read-only); CPU deterministic
    (Vw, Aw, Bw), _ = build_fixture(7005, 32, 11, "mixed")
    base = run_trainer(patched, Vw, Aw, Bw, 11, budget=11, parallel=1)
    rng_base = rng_digest()
    hk = RecordingHook(LS.make_dispatch("native_gem", ETA))
    cap = RecordingCapture()
    inst = run_trainer(patched, Vw, Aw, Bw, 11, hook=hk, capture=cap, budget=11, parallel=1)
    rng_inst = rng_digest()
    rec.chk("window_invariance_hooks_on_vs_off",
            torch.equal(base["w"], inst["w"]) and base["loss"] == inst["loss"] and rng_base == rng_inst,
            states=len(base["loss"]), hook_calls=hk.calls, capture_calls=cap.calls, capture_indices=cap.idx,
            note="optimizer state, update and RNG identical with the capture and native-dispatch hooks attached")
    rec.chk("hook_protocol_over_window", hk.calls == 11 and all(hk.returns_none) and not hk.violations and
            cap.calls == 11 and not cap.violations and cap.idx == list(range(11)),
            hook_violations=hk.violations, capture_violations=cap.violations)
    rec.chk("window_one_update_divisor_eleven", len(base["loss"]) == 11
            and inst["wrapper"].student_calls == 11 and inst["wrapper"].teacher_calls == 22,
            student_calls=inst["wrapper"].student_calls, teacher_calls=inst["wrapper"].teacher_calls,
            student_modes_train=inst["wrapper"].modes_seen[:3])
    rec.chk("trainer_mode_train_during_student", all(inst["wrapper"].modes_seen))
    rec.note("summary", finished_utc=utc())
    rec.flush()
    print("SYNTHETIC_GATE_PASS")


if __name__ == "__main__":
    try:
        main()
    except GateFailure as e:
        print("SYNTHETIC_GATE_FAIL:", e)
        sys.exit(2)
