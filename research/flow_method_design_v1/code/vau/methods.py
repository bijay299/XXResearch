"""Optimisation arms sharing one interface:

    arm = make_arm(name, task, hp, **cfg)
    ms  = arm.init_state(theta0, R1, seed)          # R1 = fixed retained probe/penalty batch
    theta_new, info = arm.step(ms, theta, batch)    # batch = Batch(erase=..., gate=[R2_1..R2_R])

Every arm is split into `propose` (no commit: proposal step + tentative AdamW state) and `step`
(commit). `Gated(arm, gate_cfg)` wraps ANY arm with the fresh-batch backtracking gate and full
rollback of (theta, m, v, t) on rejection. VAU = Gated(VAUNoGate).

Arms: adamw, adamw_stepmatch, penalty, pre_proj, post_euclid, vau_nogate, vau (+ gate wrapper).
Operation counts go to ms.ops (see contracts.Ops).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import adamw as aw
from . import losses
from .contracts import ContractViolation, Ops, check_array, check_vec, dev_rng
from .projections import euclid_project, exact_probe, metric_project, sketch_probe


@dataclass
class Batch:
    erase: object                 # erasure batch (task-specific: z or initial noise)
    gate: tuple = ()              # fresh retained batches R2, one per gate try (>= R when gated)
    match_norm: float | None = None   # optional per-step target norm for adamw_stepmatch


@dataclass
class MethodState:
    adam: aw.AdamState
    theta0: np.ndarray
    R1: object
    rng: np.random.Generator
    ops: Ops = field(default_factory=Ops)
    calls: int = 0                # step() calls (data index; drives probe refresh, never rolled back)
    probe: np.ndarray | None = None
    cache: dict = field(default_factory=dict)


@dataclass
class Proposal:
    delta: np.ndarray
    adam_next: aw.AdamState
    g: np.ndarray
    loss: float
    info: dict


@dataclass(frozen=True)
class GateConfig:
    tau2: float          # drift budget tau^2 on E||f(theta) - f(theta0)||^2
    M: float             # per-sample clip / assumed bound
    n2: int              # gate batch size
    delta: float         # confidence
    T: int               # planned number of steps (union bound)
    R: int = 4           # max tries (s = 1, 1/2, ..., 2^-(R-1))

    def __post_init__(self):
        if not (self.M > 0 and self.n2 >= 1 and 0 < self.delta < 1 and self.T >= 1 and self.R >= 1):
            raise ContractViolation(f"invalid GateConfig {self}")

    @property
    def slack(self):
        return self.M * math.sqrt(math.log(self.T * self.R / self.delta) / (2 * self.n2))


class Arm:
    name = "base"
    uses_probe = False

    def __init__(self, task, hp=aw.AdamWHP(), probe="sketch", k=16, K=5, reject_nonprogress=True):
        if probe not in ("sketch", "exact"):
            raise ContractViolation(f"probe must be 'sketch' or 'exact', got {probe!r}")
        if not (isinstance(K, int) and K >= 1):
            raise ContractViolation("K must be a positive int")
        self.task, self.model, self.hp = task, task.model, hp
        self.probe_kind, self.k, self.K = probe, k, K
        self.reject_nonprogress = reject_nonprogress

    # --- lifecycle ---------------------------------------------------------------------------
    def init_state(self, theta0, R1, seed):
        check_vec("theta0", theta0, self.model.P)
        self.model.check_input(R1)
        return MethodState(adam=aw.init_state(self.model.P), theta0=theta0.copy(), R1=R1, rng=dev_rng(seed))

    def _check_step_inputs(self, ms, theta, batch):
        if not isinstance(ms, MethodState) or not isinstance(batch, Batch):
            raise ContractViolation("step needs a MethodState and a Batch")
        check_vec("theta", theta, self.model.P)

    # --- shared pieces -----------------------------------------------------------------------
    def _grad(self, ms, theta, batch):
        loss, g = self.task.erasure_loss_grad(theta, batch.erase, ms.ops)
        check_vec("g", g, self.model.P)
        return loss, g

    def _probe(self, ms, theta):
        if ms.probe is None or ms.calls % self.K == 0:
            if self.probe_kind == "exact":
                ms.probe = exact_probe(self.model, ms.R1, theta, ms.ops)
            else:
                ms.probe = sketch_probe(self.model, ms.R1, theta, self.k, ms.rng, ms.ops)
        return ms.probe

    def _nonprogress(self, ms, g, d1, info):
        info["progress_lin"] = float(-g @ d1)
        if self.reject_nonprogress and g @ d1 >= 0:
            ms.ops.add("nonprogress_zeroed")
            info["nonprogress_zeroed"] = True
            return np.zeros_like(d1)
        info["nonprogress_zeroed"] = False
        return d1

    def propose(self, ms, theta, batch) -> Proposal:
        raise NotImplementedError

    # --- ungated step ------------------------------------------------------------------------
    def step(self, ms, theta, batch):
        self._check_step_inputs(ms, theta, batch)
        prop = self.propose(ms, theta, batch)
        ms.adam = aw.commit(ms.adam, prop.adam_next)
        ms.calls += 1
        ms.ops.add("step")
        info = dict(prop.info, arm=self.name, loss=prop.loss, s=1.0, accepted=True,
                    step_norm=float(np.linalg.norm(prop.delta)))
        return theta + prop.delta, info


class AdamWArm(Arm):
    """Native GEM optimisation: plain AdamW on the erasure gradient (no retention information)."""
    name = "adamw"

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, _ = aw.peek(ms.adam, g, theta, self.hp)
        return Proposal(d0, nxt, g, loss, {"progress_lin": float(-g @ d0)})


class AdamWStepMatch(Arm):
    """AdamW with the global step rescaled to a reference per-step norm (state committed from raw g).
    Target norm: batch.match_norm if given, else ref_norms[ms.calls]."""
    name = "adamw_stepmatch"

    def __init__(self, task, hp=aw.AdamWHP(), ref_norms=None, **kw):
        super().__init__(task, hp, **kw)
        self.ref_norms = None if ref_norms is None else list(ref_norms)

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, _ = aw.peek(ms.adam, g, theta, self.hp)
        target = batch.match_norm
        if target is None:
            if self.ref_norms is None or ms.calls >= len(self.ref_norms):
                raise ContractViolation("adamw_stepmatch needs batch.match_norm or ref_norms[step]")
            target = self.ref_norms[ms.calls]
        if not (np.isfinite(target) and target >= 0):
            raise ContractViolation(f"invalid target norm {target!r}")
        n0 = float(np.linalg.norm(d0))
        d = d0 * (target / n0) if n0 > 0 else d0
        return Proposal(d, nxt, g, loss, {"progress_lin": float(-g @ d), "target_norm": float(target)})


class PenaltyArm(Arm):
    """AdamW on L_erase + lam/2 ||f_R1(theta) - f_R1(theta0)||^2 (same retained batch R1)."""
    name = "penalty"

    def __init__(self, task, hp=aw.AdamWHP(), lam=1.0, **kw):
        super().__init__(task, hp, **kw)
        self.lam = lam

    def init_state(self, theta0, R1, seed):
        ms = super().init_state(theta0, R1, seed)
        ms.cache["fR1_0"] = self.model.forward(R1, theta0)
        ms.ops.add("teacher", self.model.n(R1))   # one-off base outputs (cacheable offline)
        return ms

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        n = self.model.n(ms.R1)
        fr = self.model.forward(ms.R1, theta); ms.ops.add("forward", n)
        pen, gr = losses.retention_penalty(fr, ms.cache["fR1_0"], self.lam)
        g_tot = g + self.model.vjp(ms.R1, theta, gr); ms.ops.add("vjp", n)
        d0, nxt, _ = aw.peek(ms.adam, g_tot, theta, self.hp)
        return Proposal(d0, nxt, g, loss, {"penalty": pen, "progress_lin": float(-g @ d0)})


class PreProjArm(Arm):
    """GPM/OGD-style: Euclidean projection of the raw gradient onto null(S^J) BEFORE AdamW
    (the Adam state accumulates the projected gradient)."""
    name = "pre_proj"
    uses_probe = True

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        S = self._probe(ms, theta)
        gp, pinfo = euclid_project(g, S); ms.ops.add("solve")
        d0, nxt, _ = aw.peek(ms.adam, gp, theta, self.hp)
        return Proposal(d0, nxt, g, loss, {"progress_lin": float(-g @ d0), "mu": pinfo["mu"]})


class PostEuclidArm(Arm):
    """Adam-NSCL-style: Euclidean projection of the ACTUAL AdamW step (VAU-Euclid ablation)."""
    name = "post_euclid"
    uses_probe = True

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, _ = aw.peek(ms.adam, g, theta, self.hp)
        S = self._probe(ms, theta)
        d1, pinfo = euclid_project(d0, S); ms.ops.add("solve")
        info = {"mu": pinfo["mu"]}
        d1 = self._nonprogress(ms, g, d1, info)
        return Proposal(d1, nxt, g, loss, info)


class VAUNoGate(Arm):
    """D-metric projection of the actual AdamW step, s = 1 always."""
    name = "vau_nogate"
    uses_probe = True

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, D = aw.peek(ms.adam, g, theta, self.hp)
        S = self._probe(ms, theta)
        d1, pinfo = metric_project(d0, S, D); ms.ops.add("solve")
        info = {"mu": pinfo["mu"]}
        d1 = self._nonprogress(ms, g, d1, info)
        return Proposal(d1, nxt, g, loss, info)


class Gated(Arm):
    """Fresh-batch backtracking gate around any arm.

    For s in 1, 1/2, ..., 2^-(R-1), on FRESH retained batch batch.gate[i] (size n2):
        rho_hat = mean_x min(||f_x(theta + s Delta) - f_x(theta0)||^2, M)
        accept iff rho_hat + M sqrt(ln(T R / delta) / (2 n2)) <= tau^2.
    Accept: theta += s Delta and the inner arm's tentative AdamW state is committed.
    Reject all tries: (theta, m, v, t) unchanged (rollback); ms.calls still advances (data advances).
    """

    def __init__(self, inner: Arm, gate: GateConfig):
        if not isinstance(inner, Arm) or isinstance(inner, Gated):
            raise ContractViolation("Gated wraps one ungated Arm")
        self.inner, self.gate = inner, gate
        self.task, self.model, self.hp = inner.task, inner.model, inner.hp
        self.name = "vau" if isinstance(inner, VAUNoGate) else f"gated_{inner.name}"
        self.uses_probe = inner.uses_probe

    def init_state(self, theta0, R1, seed):
        return self.inner.init_state(theta0, R1, seed)

    def propose(self, ms, theta, batch):
        return self.inner.propose(ms, theta, batch)

    def rho_hat(self, ms, theta_trial, x2):
        n = self.model.n(x2)
        if n != self.gate.n2:
            raise ContractViolation(f"gate batch size {n} != n2 {self.gate.n2}")
        f = self.model.forward(x2, theta_trial); ms.ops.add("gate_forward", n)
        f0 = self.model.forward(x2, ms.theta0); ms.ops.add("gate_base", n)   # cacheable offline
        return float(np.mean(np.minimum(np.sum((f - f0) ** 2, axis=1), self.gate.M)))

    def step(self, ms, theta, batch):
        self._check_step_inputs(ms, theta, batch)
        if len(batch.gate) < self.gate.R:
            raise ContractViolation(f"gated step needs {self.gate.R} fresh gate batches, got {len(batch.gate)}")
        prop = self.propose(ms, theta, batch)
        s_acc, rhos = 0.0, []
        for i in range(self.gate.R):
            s = 0.5 ** i
            ms.ops.add("gate_try")
            rho = self.rho_hat(ms, theta + s * prop.delta, batch.gate[i])
            rhos.append(rho)
            if rho + self.gate.slack <= self.gate.tau2:
                s_acc = s
                break
        ms.calls += 1
        ms.ops.add("step")
        info = dict(prop.info, arm=self.name, loss=prop.loss, s=s_acc, accepted=s_acc > 0, rho_hat=rhos,
                    slack=self.gate.slack)
        if s_acc > 0:
            ms.adam = aw.commit(ms.adam, prop.adam_next)
            d = s_acc * prop.delta
            info.update(step_norm=float(np.linalg.norm(d)), progress_applied=float(-prop.g @ d))
            return theta + d, info
        ms.ops.add("rejected_step")
        info.update(step_norm=0.0, progress_applied=0.0)
        return theta.copy(), info


ARMS = {"adamw": AdamWArm, "adamw_stepmatch": AdamWStepMatch, "penalty": PenaltyArm,
        "pre_proj": PreProjArm, "post_euclid": PostEuclidArm, "vau_nogate": VAUNoGate}


def make_arm(name, task, hp=aw.AdamWHP(), gate: GateConfig | None = None, **cfg):
    """'vau' = Gated(VAUNoGate); 'gated_<arm>' = Gated(<arm>); otherwise the plain arm."""
    if name == "vau":
        if gate is None:
            raise ContractViolation("vau needs a GateConfig")
        return Gated(VAUNoGate(task, hp, **cfg), gate)
    if name.startswith("gated_"):
        if gate is None:
            raise ContractViolation(f"{name} needs a GateConfig")
        return Gated(ARMS[name[6:]](task, hp, **cfg), gate)
    if name not in ARMS:
        raise ContractViolation(f"unknown arm {name!r}")
    return ARMS[name](task, hp, **cfg)
