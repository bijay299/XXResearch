"""Optimisation arms sharing one interface:

    arm = make_arm(name, task, hp, gate=GateConfig(...), **cfg)
    ms  = arm.init_state(theta0, R1, sketch_rng)     # R1 fixed retained probe/penalty batch; sketch_rng = stream 3
    theta_new, info = arm.step(ms, theta, batch)     # batch = Batch(erase=..., gate=(R2_1, ..., R2_R))

Each arm is split into `propose` (proposal step + tentative AdamW state, nothing committed) and `step`
(commit). `Gated(arm, gate)` wraps ANY ungated arm with the fresh-batch backtracking gate and full rollback
of (theta, m, v, t) on rejection; VAU = Gated(VAUNoGate). `EarlyStop(arm, gate)` is penalty_earlystop's
wrapper (s = 1 tries only, the run stops permanently at the first failure).

Arms: adamw, adamw_stepmatch, penalty, pre_proj, post_euclid, vau_nogate; vau, gated_<arm>, earlystop_<arm>.
Common config: probe ('sketch' | 'exact'), k (0..64; k = 0 means no probes), K (refresh period in step
calls; K = None means build once at the first call), ridge_rel (default 1e-6), reject_nonprogress,
inject (hook (ms, theta, delta) -> delta applied to the final proposed step, e.g. the G5 gauge injection).
Operation counts go to ms.ops (contracts.Ops).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import adamw as aw
from . import losses
from .contracts import ContractViolation, Ops, check_vec
from .projections import RIDGE_REL, euclid_project, exact_probe, metric_project, sketch_probe


@dataclass
class Batch:
    erase: object                       # erasure batch (task-specific: z or initial noise)
    gate: tuple = ()                    # fresh retained batches R2, one per gate try (>= R when gated)
    match_norm: float | None = None     # per-step target norm for adamw_stepmatch


@dataclass
class MethodState:
    adam: aw.AdamState
    theta0: np.ndarray
    R1: object
    rng: np.random.Generator            # sketch stream (protocol stream 3); arms draw nothing else
    ops: Ops = field(default_factory=Ops)
    calls: int = 0                      # step() calls = data index; drives probe refresh, never rolled back
    probe: np.ndarray | None = None
    cache: dict = field(default_factory=dict)


@dataclass
class Proposal:
    delta: np.ndarray
    adam_next: aw.AdamState
    g: np.ndarray                       # raw erasure gradient
    D: np.ndarray                       # Adam diagonal of the tentative state (for H-norms)
    loss: float
    info: dict


@dataclass(frozen=True)
class GateConfig:
    tau2: float          # drift budget on mean min(||f(theta) - f(theta0)||^2, M); may be +inf
    M: float             # per-sample clip / assumed bound
    n2: int              # gate batch size
    delta: float         # confidence
    T: int               # planned number of steps (union bound)
    R: int = 4           # max tries, s = 1, 1/2, ..., 2^-(R-1)

    def __post_init__(self):
        if not (np.isfinite(self.M) and self.M > 0 and self.n2 >= 1 and 0 < self.delta < 1
                and self.T >= 1 and self.R >= 1 and not np.isnan(self.tau2)):
            raise ContractViolation(f"invalid GateConfig {self}")

    @property
    def slack(self):
        return self.M * math.sqrt(math.log(self.T * self.R / self.delta) / (2 * self.n2))


def h_norm(x, D):
    """||x||_H with H = diag(1/D) = diag(sqrt(vhat)+eps)."""
    return float(np.sqrt(np.sum(x * x / D)))


class Arm:
    name = "base"
    uses_probe = False

    def __init__(self, task, hp=aw.AdamWHP(), probe="sketch", k=16, K=5, ridge_rel=RIDGE_REL,
                 reject_nonprogress=True, inject=None):
        if probe not in ("sketch", "exact"):
            raise ContractViolation(f"probe must be 'sketch' or 'exact', got {probe!r}")
        if not (K is None or (isinstance(K, int) and K >= 1)):
            raise ContractViolation("K must be a positive int or None (build once)")
        if not (np.isfinite(ridge_rel) and ridge_rel >= 0):
            raise ContractViolation("ridge_rel must be >= 0")
        self.task, self.model, self.hp = task, task.model, hp
        self.probe_kind, self.k, self.K, self.ridge_rel = probe, k, K, ridge_rel
        self.reject_nonprogress, self.inject = reject_nonprogress, inject

    # --- lifecycle ---------------------------------------------------------------------------
    def init_state(self, theta0, R1, sketch_rng):
        check_vec("theta0", theta0, self.model.P)
        self.model.check_input(R1)
        if not isinstance(sketch_rng, np.random.Generator):
            raise ContractViolation("sketch_rng must be a numpy Generator (protocol stream 3)")
        return MethodState(adam=aw.init_state(self.model.P), theta0=theta0.copy(), R1=R1, rng=sketch_rng)

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
        due = ms.probe is None or (self.K is not None and ms.calls % self.K == 0)
        if due:
            if self.probe_kind == "exact":
                ms.probe = exact_probe(self.model, ms.R1, theta, ms.ops)
            else:
                ms.probe = sketch_probe(self.model, ms.R1, theta, self.k, ms.rng, ms.ops)
        return ms.probe

    def _project(self, ms, x, S, w=None):
        if S.shape[0] == 0:
            return x.copy(), {"mu": 0.0}
        ms.ops.add("solve")
        y, info = (euclid_project(x, S, ridge_rel=self.ridge_rel) if w is None
                   else metric_project(x, S, w, ridge_rel=self.ridge_rel))
        return y, {"mu": info["mu"]}

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

    def propose_final(self, ms, theta, batch) -> Proposal:
        prop = self.propose(ms, theta, batch)
        if self.inject is not None:
            prop.delta = self.inject(ms, theta, prop.delta)
            check_vec("injected delta", prop.delta, self.model.P)
            prop.info["injected"] = True
        return prop

    # --- ungated step ------------------------------------------------------------------------
    def step(self, ms, theta, batch):
        self._check_step_inputs(ms, theta, batch)
        prop = self.propose_final(ms, theta, batch)
        ms.adam = aw.commit(ms.adam, prop.adam_next)
        ms.calls += 1
        ms.ops.add("step")
        info = dict(prop.info, arm=self.name, loss=prop.loss, s=1.0, accepted=True,
                    step_norm=float(np.linalg.norm(prop.delta)), step_Hnorm=h_norm(prop.delta, prop.D))
        return theta + prop.delta, info


class AdamWArm(Arm):
    """Native GEM optimisation: plain AdamW on the erasure gradient (no retention information)."""
    name = "adamw"

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, D = aw.peek(ms.adam, g, theta, self.hp)
        return Proposal(d0, nxt, g, D, loss, {"progress_lin": float(-g @ d0)})


class AdamWStepMatch(Arm):
    """AdamW step rescaled to a target norm (AdamW state still committed from the raw g).
    Target: batch.match_norm if given, else c * ref_norms[ms.calls]. metric='H' (protocol: ||Delta||_H of
    the paired vau step, using this arm's own D) or 'euclid'. Target 0 (paired vau rejected) -> zero step."""
    name = "adamw_stepmatch"

    def __init__(self, task, hp=aw.AdamWHP(), ref_norms=None, c=1.0, metric="H", **kw):
        super().__init__(task, hp, **kw)
        if metric not in ("H", "euclid"):
            raise ContractViolation("metric must be 'H' or 'euclid'")
        self.ref_norms = None if ref_norms is None else list(ref_norms)
        self.c, self.metric = c, metric

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, D = aw.peek(ms.adam, g, theta, self.hp)
        target = batch.match_norm
        if target is None:
            if self.ref_norms is None or ms.calls >= len(self.ref_norms):
                raise ContractViolation("adamw_stepmatch needs batch.match_norm or ref_norms[step]")
            target = self.c * self.ref_norms[ms.calls]
        if not (np.isfinite(target) and target >= 0):
            raise ContractViolation(f"invalid target norm {target!r}")
        n0 = h_norm(d0, D) if self.metric == "H" else float(np.linalg.norm(d0))
        d = d0 * (target / n0) if n0 > 0 else d0
        return Proposal(d, nxt, g, D, loss, {"progress_lin": float(-g @ d), "target_norm": float(target)})


class PenaltyArm(Arm):
    """AdamW on L_erase + lam/2 mean_{x in R1} ||f_x(theta) - f_x(theta0)||^2 (full R1 outputs)."""
    name = "penalty"

    def __init__(self, task, hp=aw.AdamWHP(), lam=1.0, **kw):
        super().__init__(task, hp, **kw)
        self.lam = lam

    def init_state(self, theta0, R1, sketch_rng):
        ms = super().init_state(theta0, R1, sketch_rng)
        ms.cache["fR1_0"] = self.model.forward(R1, theta0)
        ms.ops.add("teacher", self.model.n(R1))   # one-off base outputs (cacheable, not charged by protocol)
        return ms

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        n = self.model.n(ms.R1)
        fr = self.model.forward(ms.R1, theta); ms.ops.add("forward", n)
        pen, gr = losses.retention_penalty(fr, ms.cache["fR1_0"], self.lam)
        g_tot = g + self.model.vjp(ms.R1, theta, gr); ms.ops.add("vjp", n)
        d0, nxt, D = aw.peek(ms.adam, g_tot, theta, self.hp)
        return Proposal(d0, nxt, g, D, loss, {"penalty": pen, "progress_lin": float(-g @ d0)})


class PreProjArm(Arm):
    """GPM/OGD-style: Euclidean projection of the raw gradient onto null(S^J) BEFORE AdamW (the Adam
    state accumulates the projected gradient). No reject rule (it is not part of this control)."""
    name = "pre_proj"
    uses_probe = True

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        gp, pinfo = self._project(ms, g, self._probe(ms, theta))
        d0, nxt, D = aw.peek(ms.adam, gp, theta, self.hp)
        return Proposal(d0, nxt, g, D, loss, dict(pinfo, progress_lin=float(-g @ d0)))


class PostEuclidArm(Arm):
    """Adam-NSCL-style / VAU-Euclid: Euclidean projection of the ACTUAL AdamW step, with the reject rule."""
    name = "post_euclid"
    uses_probe = True

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, D = aw.peek(ms.adam, g, theta, self.hp)
        d1, info = self._project(ms, d0, self._probe(ms, theta))
        d1 = self._nonprogress(ms, g, d1, info)
        return Proposal(d1, nxt, g, D, loss, info)


class VAUNoGate(Arm):
    """D-metric projection of the actual AdamW step, reject rule, s = 1 always."""
    name = "vau_nogate"
    uses_probe = True

    def propose(self, ms, theta, batch):
        loss, g = self._grad(ms, theta, batch)
        d0, nxt, D = aw.peek(ms.adam, g, theta, self.hp)
        d1, info = self._project(ms, d0, self._probe(ms, theta), w=D)
        d1 = self._nonprogress(ms, g, d1, info)
        return Proposal(d1, nxt, g, D, loss, info)


class Gated(Arm):
    """Fresh-batch backtracking gate around any ungated arm.

    For try i = 0..R-1, s = 2^-i, on FRESH retained batch batch.gate[i] (size n2):
        rho_hat = mean_x min(||f_x(theta + s Delta) - f_x(theta0)||^2, M)
        accept iff rho_hat + M sqrt(ln(T R / delta) / (2 n2)) <= tau^2.
    Accept: theta += s Delta and the inner arm's tentative AdamW state is committed (from the raw g).
    All tries fail: (theta, m, v, t) unchanged (rollback); ms.calls still advances (data advances).
    tau^2 = +inf accepts s = 1 and is bitwise identical to the ungated arm (I6).
    """
    prefix = "gated_"

    def __init__(self, inner: Arm, gate: GateConfig):
        if not isinstance(inner, Arm) or isinstance(inner, Gated):
            raise ContractViolation("Gated wraps one ungated Arm")
        if not isinstance(gate, GateConfig):
            raise ContractViolation("gate must be a GateConfig")
        self.inner, self.gate = inner, gate
        self.task, self.model, self.hp = inner.task, inner.model, inner.hp
        self.name = "vau" if (isinstance(inner, VAUNoGate) and type(self) is Gated) else self.prefix + inner.name
        self.uses_probe = inner.uses_probe

    def init_state(self, theta0, R1, sketch_rng):
        return self.inner.init_state(theta0, R1, sketch_rng)

    def propose(self, ms, theta, batch):
        return self.inner.propose_final(ms, theta, batch)

    def rho_hat(self, ms, theta_trial, x2):
        n = self.model.n(x2)
        if n != self.gate.n2:
            raise ContractViolation(f"gate batch size {n} != n2 {self.gate.n2}")
        f = self.model.forward(x2, theta_trial); ms.ops.add("gate_forward", n)
        f0 = self.model.forward(x2, ms.theta0); ms.ops.add("gate_base", n)   # cacheable offline
        return float(np.mean(np.minimum(np.sum((f - f0) ** 2, axis=1), self.gate.M)))

    def _n_tries(self):
        return self.gate.R

    def step(self, ms, theta, batch):
        self._check_step_inputs(ms, theta, batch)
        if len(batch.gate) < self._n_tries():
            raise ContractViolation(f"gated step needs {self._n_tries()} fresh gate batches, got {len(batch.gate)}")
        prop = self.propose(ms, theta, batch)
        s_acc, rhos = 0.0, []
        for i in range(self._n_tries()):
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
            info.update(step_norm=float(np.linalg.norm(d)), step_Hnorm=h_norm(d, prop.D),
                        progress_applied=float(-prop.g @ d))
            return theta + d, info
        ms.ops.add("rejected_step")
        info.update(step_norm=0.0, step_Hnorm=0.0, progress_applied=0.0)
        return theta.copy(), info


class EarlyStop(Gated):
    """penalty_earlystop wrapper: gate with s = 1 only; after the first failure the run is stopped and
    every later step returns theta unchanged with no computation (info['stopped'] = True)."""
    prefix = "earlystop_"

    def _n_tries(self):
        return 1

    def step(self, ms, theta, batch):
        if ms.cache.get("stopped"):
            self._check_step_inputs(ms, theta, batch)
            ms.calls += 1
            return theta.copy(), {"arm": self.name, "stopped": True, "accepted": False, "s": 0.0,
                                  "step_norm": 0.0, "step_Hnorm": 0.0}
        theta_new, info = super().step(ms, theta, batch)
        if not info["accepted"]:
            ms.cache["stopped"] = True
        info["stopped"] = bool(ms.cache.get("stopped"))
        return theta_new, info


ARMS = {"adamw": AdamWArm, "adamw_stepmatch": AdamWStepMatch, "penalty": PenaltyArm,
        "pre_proj": PreProjArm, "post_euclid": PostEuclidArm, "vau_nogate": VAUNoGate}


def make_arm(name, task, hp=aw.AdamWHP(), gate: GateConfig | None = None, **cfg):
    """'vau' = Gated(VAUNoGate); 'gated_<arm>' = Gated(<arm>); 'earlystop_<arm>' = EarlyStop(<arm>)."""
    for prefix, wrapper in (("gated_", Gated), ("earlystop_", EarlyStop)):
        if name.startswith(prefix):
            if gate is None:
                raise ContractViolation(f"{name} needs a GateConfig")
            inner = name[len(prefix):]
            if inner not in ARMS:
                raise ContractViolation(f"unknown arm {inner!r}")
            return wrapper(ARMS[inner](task, hp, **cfg), gate)
    if name == "vau":
        if gate is None:
            raise ContractViolation("vau needs a GateConfig")
        return Gated(VAUNoGate(task, hp, **cfg), gate)
    if name not in ARMS:
        raise ContractViolation(f"unknown arm {name!r}")
    return ARMS[name](task, hp, **cfg)
