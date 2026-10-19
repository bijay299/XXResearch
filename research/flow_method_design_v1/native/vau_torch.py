"""STATIC, NOT EXECUTED.

torch is not installed in the sprint environment, so this file has never been imported or run. It is an
integration sketch of the VAU step for GEM c328fc6 (gem/operators/erasure/erasure_gem.py). Its arithmetic
mirrors the numpy prototype one-to-one (function names in brackets), which is what the numpy contract
tests actually cover:
    adamw_peek            [code/vau/adamw.py: peek]
    weighted_null_project [code/vau/projections.py: weighted_null_project / metric_project]
    VAUAdamW.build_probe  [code/vau/projections.py: sketch_probe]
    VAUAdamW.rho_hat      [code/vau/methods.py: Gated.rho_hat]
    VAUAdamW.step         [code/vau/methods.py: Gated.step around VAUNoGate.propose]
Any torch-API detail here (optimizer state keys, autograd.grad arguments, peft adapter toggles) is from
memory of those libraries and is UNVERIFIED until run. See native/INTEGRATION_NOTES.md.

NOTE: the numpy sketch draws Z (k_max x N_r d) row-major from ONE stream; this torch version draws per output
tensor per row, so the two produce different (equally distributed) sketches for the same seed.

Usage inside GEM (see native/gem_integration.patch): build VAUAdamW once after the optimizer, and at
EG:288 call `vau.step()` instead of `optimizer.step()`. The erasure gradient is read from p.grad exactly
as torch.optim.AdamW would read it (after GEM's backward calls and optional clipping).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

RIDGE_REL = 1e-6


class ContractViolation(ValueError):
    pass


# --------------------------------------------------------------------------------------------------
# configuration and retained data
# --------------------------------------------------------------------------------------------------
@dataclass
class VAUConfig:
    k: int = 16                 # sketch rows used
    k_max: int = 64             # sketch rows drawn per refresh (rows k..k_max-1 discarded: same stream for every k)
    refresh: int = 5            # K: probe refresh period (in step() calls, never rolled back)
    tau2: float = 0.0           # drift budget on mean ||f(theta) - f(theta0)||^2 (fp32, flattened output)
    M: float = 1.0              # per-sample clip / assumed bound
    n2: int = 8                 # gate batch size
    delta: float = 0.05         # confidence
    T: int = 23                 # planned optimizer steps (GEM default budget 250 / 11 latents -> 23)
    R: int = 4                  # max gate tries, s = 1, 1/2, ..., 2^-(R-1)
    reject_nonprogress: bool = True
    seed: int = 0               # sketch / pool generator seed (separate from GEM's train_generator)
    probe_mode: str = "eval"    # dropout policy for probes and gate: 'eval' = LoRA dropout OFF

    @property
    def slack(self) -> float:
        return self.M * math.sqrt(math.log(self.T * self.R / self.delta) / (2 * self.n2))


@dataclass
class RetainedItem:
    """One retained evaluation point: a prompt's embeddings, one latent, its timestep index."""
    embeds: tuple               # wrapper.encode_prompt(retained_prompt) output
    latent: torch.Tensor        # (seq, C) packed latent, as stored by the GEM rollout
    timestep_idx: int
    base_out: Optional[torch.Tensor] = None   # cached f(theta0) (adapter disabled), fp32 flattened


@dataclass
class RetainedStream:
    """Sequential, NON-REUSING stream of retained items for the gate (fresh batches). Raises when
    exhausted instead of silently recycling items (reuse would void the Hoeffding argument)."""
    items: List[RetainedItem]
    pos: int = 0

    def take(self, n: int) -> List[RetainedItem]:
        if self.pos + n > len(self.items):
            raise ContractViolation("gate pool exhausted: enlarge the retained pool (need T*R*n2 items)")
        out = self.items[self.pos:self.pos + n]
        self.pos += n
        return out


@torch.no_grad()
def build_retained_pool(wrapper, prompts, n_steps, end_idx, image_size, seed, guidance_scale=3.0):
    """Teacher (adapter disabled) prefix rollouts of the retained prompts, base outputs cached.
    Uses its own generator so GEM's train_generator stream is untouched."""
    gen = torch.Generator(device=wrapper.device).manual_seed(seed)
    items = []
    for prompt in prompts:
        embeds = wrapper.encode_prompt(prompt)
        with wrapper.disable_adapter():
            _, lats = wrapper.partially_denoise_until_timestep(
                prompts=None, embeds=embeds, image_size=image_size, num_inference_steps=n_steps,
                guidance_scale=guidance_scale, end_at_timestep_idx=end_idx, generator=gen,
                return_all_latents=True)
            for idx, lat in enumerate(lats):
                item = RetainedItem(embeds, lat.squeeze(0), idx)
                wrapper.set_timesteps(lat.shape[1], n_steps)
                out = wrapper.predict_noise(torch.tensor([idx]), lat, embeds)   # guidance 1.0 as GEM loss
                item.base_out = out.float().reshape(-1).clone()
                items.append(item)
    return items


# --------------------------------------------------------------------------------------------------
# pure arithmetic (mirrors of the numpy package)
# --------------------------------------------------------------------------------------------------
def adamw_peek(m, v, step, g, theta, lr, beta1, beta2, eps, wd):
    """[adamw.peek] torch AdamW single-tensor formula on flat fp32 tensors, no mutation.
    Returns (delta0, m_new, v_new, step_new, D)."""
    t = step + 1
    w = 1.0 - beta1
    m_new = m + w * (g - m) if abs(w) < 0.5 else g - (g - m) * (1.0 - w)   # lerp_ semantics
    v_new = v * beta2 + (1.0 - beta2) * (g * g)
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    denom = v_new.sqrt() / math.sqrt(bc2) + eps
    D = 1.0 / denom
    delta0 = -(lr * wd) * theta - (lr / bc1) * (m_new / denom)
    return delta0, m_new, v_new, t, D


def weighted_null_project(x, S, w):
    """[projections.weighted_null_project] x1 = x - w*(S^T (S diag(w) S^T + mu I)^{-1} S x),
    mu = 1e-6 tr(G0)/k. k x k Gram and solve in fp64; P-sized products in fp32."""
    if S.shape[1] != x.shape[0] or w.shape != x.shape:
        raise ContractViolation("shape mismatch in projection")
    if not (torch.isfinite(x).all() and torch.isfinite(S).all() and (w > 0).all()):
        raise ContractViolation("nonfinite input or nonpositive metric")
    G0 = ((S * w) @ S.T).double()
    tr = float(torch.trace(G0))
    if tr == 0.0:
        return x.clone(), 0.0
    mu = RIDGE_REL * tr / S.shape[0]
    L = torch.linalg.cholesky(G0 + mu * torch.eye(S.shape[0], dtype=G0.dtype, device=G0.device))
    y = torch.cholesky_solve((S @ x).double().unsqueeze(1), L).squeeze(1).to(x.dtype)
    return x - w * (S.T @ y), mu


# --------------------------------------------------------------------------------------------------
# the optimizer wrapper
# --------------------------------------------------------------------------------------------------
class VAUAdamW:
    """VAU around an existing torch.optim.AdamW (the AdamW object remains the owner of m, v, step).

    Never calls optimizer.step(); reads/writes optimizer.state[p]['exp_avg' | 'exp_avg_sq' | 'step']
    directly so that a later plain optimizer.step() or state_dict() round trip stays consistent.
    """

    def __init__(self, optimizer: torch.optim.AdamW, wrapper, cfg: VAUConfig,
                 R1: List[RetainedItem], gate_stream: RetainedStream, n_steps: int):
        if len(optimizer.param_groups) != 1:
            raise ContractViolation("expected one param group (GEM passes wrapper.parameters())")
        grp = optimizer.param_groups[0]
        if grp.get("amsgrad", False) or grp.get("maximize", False):
            raise ContractViolation("amsgrad / maximize are not supported")
        self.opt, self.wrapper, self.cfg = optimizer, wrapper, cfg
        self.params = [p for p in grp["params"] if p.requires_grad]   # LoRA A, B only (base frozen)
        self.R1, self.gate_stream, self.n_steps = R1, gate_stream, n_steps
        self.theta0 = self._flat([p.detach() for p in self.params]).clone()
        self.gen = torch.Generator(device=self.theta0.device).manual_seed(cfg.seed)
        self.calls = 0
        self.S: Optional[torch.Tensor] = None
        self.ops = {"probe_build": 0, "probe_vjp": 0, "gate_forward_items": 0, "rejected": 0,
                    "nonprogress_zeroed": 0, "solve": 0}

    # ---- flat views -----------------------------------------------------------------------------
    @staticmethod
    def _flat(ts):
        return torch.cat([t.reshape(-1).float() for t in ts])

    def _write_params(self, flat):
        with torch.no_grad():
            off = 0
            for p in self.params:
                n = p.numel()
                p.copy_(flat[off:off + n].view_as(p).to(p.dtype))
                off += n

    def _state_flat(self):
        ms, vs, steps = [], [], set()
        for p in self.params:
            st = self.opt.state[p]
            if len(st) == 0:
                ms.append(torch.zeros_like(p, dtype=torch.float32)); vs.append(torch.zeros_like(p, dtype=torch.float32))
                steps.add(0)
            else:
                ms.append(st["exp_avg"]); vs.append(st["exp_avg_sq"]); steps.add(int(st["step"]))
        if len(steps) != 1:
            raise ContractViolation(f"inconsistent per-parameter AdamW steps {steps}")
        return self._flat(ms), self._flat(vs), steps.pop()

    def _commit_state(self, m, v, t):
        off = 0
        for p in self.params:
            n = p.numel()
            st = self.opt.state[p]
            if len(st) == 0:
                st["step"] = torch.tensor(0.0)
                st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st["exp_avg"].copy_(m[off:off + n].view_as(p))
            st["exp_avg_sq"].copy_(v[off:off + n].view_as(p))
            st["step"].fill_(float(t))
            off += n

    # ---- model evaluation -------------------------------------------------------------------------
    @contextlib.contextmanager
    def _probe_mode(self):
        """Probes and gate: adapter ON, LoRA dropout OFF (module eval mode). Restores train mode,
        which GEM set at EG:142. Stated policy; the erasure gradient itself keeps dropout 0.1."""
        was_training = self.wrapper.training
        if self.cfg.probe_mode == "eval":
            self.wrapper.eval()
        try:
            yield
        finally:
            if was_training:
                self.wrapper.train()

    def _outputs(self, items: List[RetainedItem]):
        """Adapter-ON predictions, fp32 flattened, one predict_noise call per item (GEM's predict_noise
        takes one prompt per call; batching by prompt is an optimisation left to the integrator)."""
        outs = []
        for it in items:
            lat = it.latent.unsqueeze(0)
            self.wrapper.set_timesteps(lat.shape[1], self.n_steps)
            outs.append(self.wrapper.predict_noise(torch.tensor([it.timestep_idx]), lat, it.embeds).float().reshape(-1))
        return outs

    def build_probe(self):
        """[projections.sketch_probe] S = Sigma J_R1(theta): one graph-building forward, k VJPs."""
        with self._probe_mode(), torch.enable_grad():
            outs = self._outputs(self.R1)
            rows = []
            for i in range(self.cfg.k_max):
                cot = [torch.randn(o.shape, generator=self.gen, device=o.device) / math.sqrt(self.cfg.k) for o in outs]
                if i >= self.cfg.k:
                    continue            # advance the generator only (common random numbers across k)
                grads = torch.autograd.grad(outs, self.params, grad_outputs=cot,
                                            retain_graph=i < self.cfg.k - 1, allow_unused=False)
                rows.append(self._flat(grads))
        self.ops["probe_build"] += 1
        self.ops["probe_vjp"] += self.cfg.k
        return torch.stack(rows)        # (k, P) fp32; k*P*4 bytes

    @torch.no_grad()
    def rho_hat(self, theta_trial, items):
        """[methods.Gated.rho_hat] mean_x min(||f_x(theta_trial) - f_x(theta0)||^2, M)."""
        self._write_params(theta_trial)
        with self._probe_mode():
            outs = self._outputs(items)
        vals = [torch.clamp(((o - it.base_out) ** 2).sum(), max=self.cfg.M) for o, it in zip(outs, items)]
        self.ops["gate_forward_items"] += len(items)
        return float(torch.stack(vals).mean())

    # ---- the step ---------------------------------------------------------------------------------
    def step(self):
        grp = self.opt.param_groups[0]
        lr, (b1, b2), eps, wd = grp["lr"], grp["betas"], grp["eps"], grp["weight_decay"]
        if any(p.grad is None for p in self.params):
            raise ContractViolation("a trainable parameter has no gradient")
        g = self._flat([p.grad for p in self.params])
        theta = self._flat([p.detach() for p in self.params]).clone()
        if not (torch.isfinite(g).all() and torch.isfinite(theta).all()):
            raise ContractViolation("nonfinite gradient or parameters")
        m, v, t = self._state_flat()
        delta0, m1, v1, t1, D = adamw_peek(m, v, t, g, theta, lr, b1, b2, eps, wd)
        if self.S is None or self.calls % self.cfg.refresh == 0:
            self.S = self.build_probe()
        delta1, mu = weighted_null_project(delta0, self.S, D)
        self.ops["solve"] += 1
        info = {"progress_lin": float(-(g @ delta1)), "mu": mu, "nonprogress_zeroed": False}
        if self.cfg.reject_nonprogress and float(g @ delta1) >= 0.0:
            delta1 = torch.zeros_like(delta1)
            self.ops["nonprogress_zeroed"] += 1
            info["nonprogress_zeroed"] = True
        s_acc, rhos = 0.0, []
        for i in range(self.cfg.R):
            s = 0.5 ** i
            rho = self.rho_hat(theta + s * delta1, self.gate_stream.take(self.cfg.n2))
            rhos.append(rho)
            if rho + self.cfg.slack <= self.cfg.tau2:
                s_acc = s
                break
        self.calls += 1
        if s_acc > 0:
            self._write_params(theta + s_acc * delta1)
            self._commit_state(m1, v1, t1)
        else:
            self._write_params(theta)        # bit-exact restore of theta; m, v, step untouched
            self.ops["rejected"] += 1
        info.update(s=s_acc, rho_hat=rhos, slack=self.cfg.slack, accepted=s_acc > 0)
        return info

    # ---- serialization -------------------------------------------------------------------------------
    def state_dict(self):
        """VAU additions next to optimizer.state_dict() (which already holds m, v, step)."""
        return {"calls": self.calls, "S": self.S, "gen_state": self.gen.get_state(), "theta0": self.theta0,
                "gate_pos": self.gate_stream.pos, "cfg": self.cfg.__dict__.copy(), "ops": dict(self.ops)}

    def load_state_dict(self, sd):
        self.calls, self.S, self.theta0 = sd["calls"], sd["S"], sd["theta0"]
        self.gen.set_state(sd["gen_state"])
        self.gate_stream.pos = sd["gate_pos"]
        self.ops = dict(sd["ops"])
