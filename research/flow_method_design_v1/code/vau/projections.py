"""Probe blocks and (weighted) null-space projections.

Weighted projection with weights w > 0 (w = D for the Adam metric, w = 1 for Euclidean):
    G0 = S diag(w) S^T,  mu = 1e-6 * tr(G0) / k,  y = (G0 + mu I)^{-1} S x,  x1 = x - w * (S^T y)
x1 minimises ||x1 - x||_{diag(1/w)} subject to S x1 = 0 when mu = 0 (Lagrange/KKT; C_analysis T2).
With mu > 0 the residual is S x1 = mu (G0 + mu I)^{-1} S x (exact identity, tested).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .contracts import ContractViolation, Ops, check_array

RIDGE_REL = 1e-6


K_MAX = 64     # sketch rows drawn per refresh (eval_protocol stream 3); an arm uses the first k


def ridge_mu(G0, ridge_rel=RIDGE_REL):
    return ridge_rel * float(np.trace(G0)) / G0.shape[0]


def weighted_null_project(x, S, w, mu="auto", ridge_rel=RIDGE_REL):
    """Returns (x1, info). info: mu, tr(G0), degenerate (S == 0 or k == 0 -> no constraint, x returned
    as a copy, bitwise equal). mu='auto' -> ridge_rel * tr(G0)/k; ridge_rel = 0 is the exact projection."""
    P = x.shape[0]
    check_array("x", x, shape=(P,))
    check_array("S", S, ndim=2)
    check_array("w", w, shape=(P,))
    if S.shape[1] != P:
        raise ContractViolation(f"S has {S.shape[1]} columns, expected {P}")
    if np.any(w <= 0):
        raise ContractViolation("metric weights must be > 0")
    G0 = (S * w) @ S.T
    tr = float(np.trace(G0))
    if tr == 0.0:
        return x.copy(), {"mu": 0.0, "trG0": 0.0, "degenerate": True}
    mu_v = ridge_mu(G0, ridge_rel) if mu == "auto" else float(mu)
    try:
        cf = sla.cho_factor(G0 + mu_v * np.eye(S.shape[0]), lower=True)
    except np.linalg.LinAlgError as e:   # only reachable with mu=0 and rank-deficient S
        raise ContractViolation(f"Gram matrix not positive definite: {e}") from e
    y = sla.cho_solve(cf, S @ x)
    return x - w * (S.T @ y), {"mu": mu_v, "trG0": tr, "degenerate": False}


def metric_project(delta0, S, D, mu="auto", ridge_rel=RIDGE_REL):
    """VAU: min ||x - delta0||_{D^{-1}} s.t. S x ~ 0 (Adam metric H = diag(sqrt(vhat)+eps))."""
    return weighted_null_project(delta0, S, D, mu, ridge_rel)


def euclid_project(x, S, mu="auto", ridge_rel=RIDGE_REL):
    """Euclidean projection onto null(S) (Adam-NSCL-style post step / GPM-OGD-style pre step)."""
    return weighted_null_project(x, S, np.ones_like(x), mu, ridge_rel)


def sketch_probe(model, x_r, theta, k, rng, ops: Ops, k_max=K_MAX):
    """S = Sigma J_r(theta), Sigma = Z[:k]/sqrt(k), Z (k_max x N_r d) iid N(0,1) drawn from rng on EVERY
    refresh (so the stream is identical for every k: common random numbers). k VJPs; the graph-building
    forward is counted once. k = 0 returns an empty (0 x P) block with no network ops."""
    if not (isinstance(k, (int, np.integer)) and 0 <= k <= k_max):
        raise ContractViolation(f"k must be an int in [0, {k_max}], got {k!r}")
    N, d = model.n(x_r), model.out_dim
    Z = rng.standard_normal((k_max, N * d))
    S = np.empty((k, model.P))
    if k == 0:
        return S
    ops.add("forward", N)
    for i in range(k):
        S[i] = model.vjp(x_r, theta, Z[i].reshape(N, d) / np.sqrt(k))
        ops.add("vjp", N)
    ops.add("probe_build")
    return S


def exact_probe(model, x_r, theta, ops: Ops, max_entries=None):
    """S = J_r(theta) dense rows (small cases only); counted as N_r*d single-sample VJPs (one per row)."""
    kw = {} if max_entries is None else {"max_entries": max_entries}
    J = model.jacobian(x_r, theta, **kw)
    N = model.n(x_r)
    ops.add("forward", N)
    for _ in range(J.shape[0]):
        ops.add("vjp", 1)
    ops.add("probe_build")
    return J
