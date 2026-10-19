"""Functional AdamW with torch.optim.AdamW semantics (amsgrad=False, maximize=False, single-tensor path).

Torch reference (single-tensor, non-capturable path). Source location: torch/optim/adamw.py,
function `_single_tensor_adamw` (torch <= 2.5); in newer releases AdamW is implemented by
torch/optim/adam.py `_single_tensor_adam` with `decoupled_weight_decay=True`. Line numbers: UNVERIFIED
(torch is not installed here, the source was not re-read; the formula below is from memory of the
source and matches the documented algorithm in the torch AdamW docstring):

    step_t += 1
    param.mul_(1 - lr * weight_decay)                    # decoupled WD, uses the PRE-step param
    exp_avg.lerp_(grad, 1 - beta1)                       # m = m + (1-b1)(g-m)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr / bias_correction1
    bias_correction2_sqrt = sqrt(bias_correction2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)

Hence the actual step is
    Delta0 = -lr*wd*theta - lr * mhat * D,   mhat = m/bc1,   D = 1/(sqrt(v)/sqrt(bc2) + eps).
Defaults are torch's (lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2); GEM passes only lr
(A_gem_code_facts.md section 5). The foreach/fused torch kernels can differ from this in the last ulp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import ContractViolation, check_array, check_vec


@dataclass(frozen=True)
class AdamWHP:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2

    def __post_init__(self):
        for name in ("lr", "eps"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ContractViolation(f"{name} must be > 0, got {v}")
        for name in ("beta1", "beta2"):
            v = getattr(self, name)
            if not (0.0 <= v < 1.0):
                raise ContractViolation(f"{name} must be in [0,1), got {v}")
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ContractViolation(f"weight_decay must be >= 0, got {self.weight_decay}")


@dataclass(frozen=True)
class AdamState:
    """Immutable AdamW state; arrays are never mutated in place by this module."""
    m: np.ndarray
    v: np.ndarray
    t: int

    @property
    def P(self):
        return self.m.shape[0]


def init_state(P):
    return AdamState(np.zeros(P), np.zeros(P), 0)


def _lerp(a, b, w):
    # torch CPU lerp kernel: |w| < 0.5 ? a + w*(b-a) : b - (b-a)*(1-w)
    return a + w * (b - a) if abs(w) < 0.5 else b - (b - a) * (1.0 - w)


def peek(state: AdamState, g, theta, hp: AdamWHP):
    """Tentative AdamW step without mutation.

    Returns (delta0, tentative_state, D): theta_new = theta + delta0 is the step torch would take,
    tentative_state the state torch would hold afterwards, D = 1/(sqrt(vhat)+eps) Adam's diagonal.
    """
    if not isinstance(state, AdamState):
        raise ContractViolation("state must be an AdamState")
    P = state.P
    check_vec("g", g, P)
    check_vec("theta", theta, P)
    t = state.t + 1
    m = _lerp(state.m, g, 1.0 - hp.beta1)
    v = state.v * hp.beta2 + (1.0 - hp.beta2) * (g * g)
    bc1 = 1.0 - hp.beta1 ** t
    bc2 = 1.0 - hp.beta2 ** t
    denom = np.sqrt(v) / math.sqrt(bc2) + hp.eps
    D = 1.0 / denom
    delta0 = -(hp.lr * hp.weight_decay) * theta - (hp.lr / bc1) * (m / denom)
    if not np.all(np.isfinite(delta0)):
        raise ContractViolation("nonfinite AdamW step")
    return delta0, AdamState(m, v, t), D


def mhat(state: AdamState, hp: AdamWHP):
    return state.m / (1.0 - hp.beta1 ** state.t)


def commit(state: AdamState, tentative: AdamState):
    """Functional commit: validates that `tentative` is the successor of `state` and returns it."""
    if not isinstance(tentative, AdamState) or tentative.t != state.t + 1 or tentative.P != state.P:
        raise ContractViolation("tentative state is not the successor of the current state")
    check_array("m", tentative.m, shape=(state.P,))
    check_array("v", tentative.v, shape=(state.P,))
    return tentative


def step(state, g, theta, hp):
    """Convenience: one plain AdamW step. Returns (theta_new, new_state)."""
    delta0, tent, _ = peek(state, g, theta, hp)
    return theta + delta0, commit(state, tent)
