"""ATTACK CORE -- a SPECIFICATION of the full-UNet relearning attacker's update, plus one
implementation of it used by defences that simulate the attacker (`erase_utrap_v2.py`).

SCOPE CORRECTION (Agent-3 fidelity audit, 2026-10-26). An earlier version of this docstring said
this file was "shared by the EVALUATION attack (`attack_relearn.py`)". IT IS NOT. `attack_relearn.py`
does not import `attack_core` and never has (`grep -n attack_core scripts/attack_relearn.py` ->
no match); it runs its own inline loop at `attack_relearn.py:333-362`. What this file provides is:

  * `EVAL_FULL_UNET` -- a hand-transcribed MIRROR of that inline loop's configuration. Verified
    line-by-line against it on 2026-10-26: Adam (`attack_relearn.py:333`), lr 1e-5 (`:209`),
    300 steps (`:208`), batch 1 (`:335` samples a single index), clip_grad_norm 1.0 (`:362`),
    full-UNet scope (`:332`), persistent optimizer state (`opt` built once outside the loop).
    Because it is a mirror and not the same code object, it can DRIFT from the evaluator. Anyone
    editing `attack_relearn.py`'s loop must re-check this dict by hand.
  * `AttackSpec.divergence_from_eval()` -- an accurate divergence report AGAINST THAT MIRROR.
  * `run_attack` -- the implementation that `erase_utrap_v2.py` (and only it) actually calls.

Two axes the mirror does NOT capture, so `divergence_from_eval()` cannot report them:
RNG provenance (`attack_relearn.py` draws noise/timesteps/indices from the GLOBAL torch stream;
`diffusion_mse` below takes an explicit `generator`) and the attacker's DATA, which lives in
`AttackData` rather than `AttackSpec`.

WHY THIS FILE EXISTS (audit section 5). At the audited snapshot the two diverged on seven axes at
once, while the code and `PREREG_H84` both called the inner process "the ACTUAL full-UNet relearning
attack (the one attack_relearn.py runs)". It was not:

| axis | UTRAP inner (snapshot) | evaluation attack | reconciled here |
|---|---|---|---|
| optimizer | SGD, rebuilt every outer step | Adam | `AttackSpec.optimizer` |
| grad clipping | none | global norm 1.0 | `AttackSpec.clip_grad_norm` |
| batch size | 2 | 1 | `AttackSpec.batch` |
| horizon | 4 / 8 / 16 | 300 | `AttackSpec.steps` (+ `horizon_role`) |
| target images from | the ALREADY-ERASED teacher | the UNMODIFIED base model | `AttackData.provenance` |
| target prompts | `["a photo of a church"] * n` (an EVAL prompt) | 8 `T` prompts (disjoint from eval) | `AttackData.prompts` |
| optimizer state | discarded each outer step | persistent across 300 steps | `AttackSpec.persistent_state` |

Nothing here decides which configuration is right. It makes the configuration EXPLICIT, puts it in
one place, and records it in the output JSON, so a defence can no longer silently claim to train
against an attacker it does not implement.

`horizon_role` is mandatory and has no default. A defence that unrolls K < the evaluation budget
must declare `SHORT_HORIZON_SURROGATE`, and that string is written into the artifact. That is the
audit's requirement: a shortened horizon may be a deliberate design choice, but it must not be
called the real 300-step attack at its real budget.

Pure PyTorch, no diffusers import, so the gradient tests run on CPU in under a second.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F

CORE_VERSION = "attack_core/v1"

#: the evaluation attacker, exactly as attack_relearn.py has always run it. Changing any value here
#: changes the measured adversary and therefore invalidates comparison with every historical record.
EVAL_FULL_UNET = dict(
    optimizer="adam", lr=1e-5, steps=300, batch=1, clip_grad_norm=1.0,
    scope="full_unet", precision="fp32", persistent_state=True,
    horizon_role="EVALUATION_REFERENCE",
)


@dataclass
class AttackSpec:
    """Every knob that distinguishes one relearning attacker from another."""
    optimizer: str = "adam"
    lr: float = 1e-5
    steps: int = 300
    batch: int = 1
    clip_grad_norm: Optional[float] = 1.0
    scope: str = "full_unet"
    precision: str = "fp32"
    persistent_state: bool = True
    momentum: float = 0.0            # SGD only
    horizon_role: str = ""           # must be set explicitly; see validate()
    notes: str = ""

    def validate(self, eval_steps: int = 300):
        roles = {"EVALUATION_REFERENCE", "SHORT_HORIZON_SURROGATE", "MATCHED_HORIZON_SURROGATE",
                 "ABLATION_K0"}
        if self.horizon_role not in roles:
            raise ValueError(
                f"horizon_role must be one of {sorted(roles)}, got {self.horizon_role!r}. "
                "It has no default on purpose: the audit's section-5 finding was that a 4-16 step "
                "surrogate was described as the real 300-step attack.")
        if self.horizon_role == "EVALUATION_REFERENCE" and self.steps != eval_steps:
            raise ValueError(
                f"horizon_role=EVALUATION_REFERENCE requires steps=={eval_steps}, got {self.steps}")
        if self.horizon_role == "SHORT_HORIZON_SURROGATE" and self.steps >= eval_steps:
            raise ValueError(
                f"steps={self.steps} is not short relative to the {eval_steps}-step evaluation; "
                "use MATCHED_HORIZON_SURROGATE")
        if self.horizon_role == "ABLATION_K0" and self.steps != 0:
            raise ValueError("ABLATION_K0 requires steps==0")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}")
        return self

    def divergence_from_eval(self):
        """Named list of every axis on which this spec differs from the evaluation attacker."""
        out = {}
        for k, v in EVAL_FULL_UNET.items():
            if k == "horizon_role":
                continue
            mine = getattr(self, k)
            if mine != v:
                out[k] = {"this_attacker": mine, "evaluation_attacker": v}
        return out

    def as_record(self, eval_steps: int = 300):
        return {"attack_core_version": CORE_VERSION, "spec": asdict(self),
                "evaluation_reference": EVAL_FULL_UNET,
                "divergence_from_evaluation_attacker": self.divergence_from_eval(),
                "is_the_evaluation_attacker": not self.divergence_from_eval()
                                             and self.steps == eval_steps}


@dataclass
class AttackData:
    """The attacker's data, with provenance recorded as a first-class field.

    `provenance` must say which model generated the latents and which prompt pool was used, because
    the audit's section-5 finding included 'teacher-generated images are not the same base-generated
    target images supplied to the attacker'.
    """
    latents: torch.Tensor
    embeddings: torch.Tensor
    prompts: List[str]
    provenance: str = ""
    split_role: str = ""             # DEF_TRAIN | ATK_TRAIN | DEV | FINAL

    def validate(self):
        if not self.provenance:
            raise ValueError("AttackData.provenance is mandatory (which model generated these?)")
        if self.split_role not in ("DEF_TRAIN", "ATK_TRAIN", "DEV", "FINAL"):
            raise ValueError(f"split_role must be a declared split, got {self.split_role!r}")
        if len(self.latents) != len(self.embeddings):
            raise ValueError("latents and embeddings must align")
        return self

    def as_record(self):
        return {"n": int(len(self.latents)), "prompts": list(self.prompts),
                "provenance": self.provenance, "split_role": self.split_role}


def make_optimizer(spec: AttackSpec, params):
    if spec.optimizer == "adam":
        return torch.optim.Adam(params, lr=spec.lr)
    return torch.optim.SGD(params, lr=spec.lr, momentum=spec.momentum)


def diffusion_mse(unet, sched, latents, embeddings, idx, generator, num_train_timesteps=None):
    """The relearning loss as used by defences that simulate the attacker.

    CORRECTED (Agent-3 audit): this is NOT called by `attack_relearn.py`, which computes the same
    algebraic loss inline at `attack_relearn.py:355-357` from its own global-RNG draws. The two are
    algebraically identical (DDPM add_noise + MSE against the sampled noise) but are separate code,
    so drift in the loss itself is possible and is not prevented by this file."""
    z0 = latents[idx]
    noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
    nt = num_train_timesteps or sched.config.num_train_timesteps
    ts = torch.randint(0, nt, (z0.shape[0],), device=z0.device, generator=generator).long()
    zt = sched.add_noise(z0, noise, ts)
    return F.mse_loss(unet(zt, ts, encoder_hidden_states=embeddings[idx]).sample, noise)


def run_attack(unet, sched, data: AttackData, spec: AttackSpec, params=None, generator=None,
               loss_fn: Callable = None, on_step: Callable = None, eval_steps: int = 300):
    """Run `spec.steps` relearning steps in place on `unet`. Returns a telemetry dict.

    CORRECTED (Agent-3 audit): the only caller is `erase_utrap_v2.py`'s K-step inner surrogate.
    `attack_relearn.py` does NOT call this -- it runs its own inline 300-step loop. The audit's
    requirement that update norms, optimizer configuration and inner progress be logged is met by
    the returned record, which callers write into their artifact.
    """
    spec.validate(eval_steps)
    data.validate()
    params = list(params if params is not None else unet.parameters())
    loss_fn = loss_fn or (lambda idx: diffusion_mse(unet, sched, data.latents, data.embeddings,
                                                    idx, generator))
    opt = make_optimizer(spec, params) if spec.steps > 0 else None
    n = len(data.latents)
    losses, gnorms, unorms = [], [], []
    for step in range(spec.steps):
        if not spec.persistent_state:
            opt = make_optimizer(spec, params)       # discards moment estimates every step
        idx = torch.randint(0, n, (spec.batch,), device=data.latents.device, generator=generator)
        loss = loss_fn(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, spec.clip_grad_norm)
                   if spec.clip_grad_norm else
                   sum(float(p.grad.norm()) ** 2 for p in params if p.grad is not None) ** 0.5)
        pre = [p.detach().clone() for p in params] if on_step or step in (0, spec.steps - 1) else None
        opt.step()
        if pre is not None:
            unorms.append(sum(float((p.detach() - q).norm()) ** 2
                              for p, q in zip(params, pre)) ** 0.5)
        losses.append(float(loss.detach()))
        gnorms.append(gn)
        if on_step:
            on_step(step, float(loss.detach()), gn)
    return {"attack_core_version": CORE_VERSION, "config": spec.as_record(eval_steps),
            "data": data.as_record(),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_trace": losses[:: max(1, len(losses) // 32)] if losses else [],
            "grad_norm_first": gnorms[0] if gnorms else None,
            "grad_norm_last": gnorms[-1] if gnorms else None,
            "update_norms_sampled": unorms,
            "steps_run": spec.steps}
