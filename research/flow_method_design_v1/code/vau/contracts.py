"""Input contracts and operation counters shared by the whole package.

Contract policy: invalid inputs (wrong type/shape/dtype, nonfinite values, bad hyper-parameters,
non-development seeds) raise ContractViolation. Nothing is silently repaired.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEV_SEEDS = range(0, 100)  # development seeds only; validation seeds are chosen in a later wave


class ContractViolation(ValueError):
    """Raised for invalid shapes, nonfinite values or invalid configuration."""


def check_array(name, x, shape=None, ndim=None):
    """Return x unchanged after validating it is a finite float64 ndarray of the given shape."""
    if not isinstance(x, np.ndarray):
        raise ContractViolation(f"{name} must be a numpy array, got {type(x).__name__}")
    if x.dtype != np.float64:
        raise ContractViolation(f"{name} must be float64 (FP64 prototype), got {x.dtype}")
    if ndim is not None and x.ndim != ndim:
        raise ContractViolation(f"{name} must have ndim {ndim}, got shape {x.shape}")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ContractViolation(f"{name} shape mismatch: expected {tuple(shape)}, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ContractViolation(f"{name} contains nonfinite values")
    return x


def check_vec(name, x, P):
    return check_array(name, x, shape=(P,))


def check_pos(name, val, allow_zero=False):
    if not np.isfinite(val) or val < 0 or (val == 0 and not allow_zero):
        raise ContractViolation(f"{name} must be {'>=' if allow_zero else '>'} 0 and finite, got {val!r}")
    return val


def dev_rng(seed):
    """numpy Generator for a DEVELOPMENT seed (0..99) only."""
    if not isinstance(seed, (int, np.integer)) or int(seed) not in DEV_SEEDS:
        raise ContractViolation(f"seed {seed!r} is not a development seed (0..99)")
    return np.random.default_rng(int(seed))


# Operation names. Network ops are counted as (calls, samples): a call is one batched pass,
# samples is the number of per-sample evaluations inside it. Event ops count calls only.
NET_OPS = ("forward", "vjp", "jvp", "teacher", "rollout_forward", "gate_forward", "gate_base",
           "probe_forward", "probe_vjp")
EVENT_OPS = ("solve", "probe_build", "gate_try", "rejected_step", "nonprogress_zeroed", "step")


@dataclass
class Ops:
    calls: dict = field(default_factory=lambda: {k: 0 for k in NET_OPS + EVENT_OPS})
    samples: dict = field(default_factory=lambda: {k: 0 for k in NET_OPS})

    def add(self, name, n=1):
        if name in NET_OPS:
            self.calls[name] += 1
            self.samples[name] += int(n)
        elif name in EVENT_OPS:
            self.calls[name] += int(n)
        else:
            raise ContractViolation(f"unknown op {name!r}")

    def snapshot(self):
        return {"calls": dict(self.calls), "samples": dict(self.samples)}

    def diff(self, before):
        now = self.snapshot()
        return {g: {k: now[g][k] - before[g][k] for k in now[g]} for g in now}
