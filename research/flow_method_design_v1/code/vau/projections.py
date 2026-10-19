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


def ridge_mu(G0):
    return RIDGE_REL * float(np.trace(G0)) / G0.shape[0]


def weighted_null_project(x, S, w, mu="auto"):
    """Returns (x1, info). info: mu, tr(G0), degenerate (S == 0 -> no constraint, x returned)."""
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
    mu_v = ridge_mu(G0) if mu == "auto" else float(mu)
    try:
        cf = sla.cho_factor(G0 + mu_v * np.eye(S.shape[0]), lower=True)
    except np.linalg.LinAlgError as e:   # only reachable with mu=0 and rank-deficient S
        raise ContractViolation(f"Gram matrix not positive definite: {e}") from e
    y = sla.cho_solve(cf, S @ x)
    return x - w * (S.T @ y), {"mu": mu_v, "trG0": tr, "degenerate": False}


def metric_project(delta0, S, D, mu="auto"):
    """VAU: min ||x - delta0||_{D^{-1}} s.t. S x ~ 0 (Adam metric H = diag(sqrt(vhat)+eps))."""
    return weighted_null_project(delta0, S, D, mu)


def euclid_project(x, S, mu="auto"):
    """Euclidean projection onto null(S) (Adam-NSCL-style post step / GPM-OGD-style pre step)."""
    return weighted_null_project(x, S, np.ones_like(x), mu)


def sketch_probe(model, x_r, theta, k, rng, ops: Ops):
    """S = Sigma J_r(theta), Sigma in R^{k x N_r d} with iid N(0, 1/k) entries; k VJPs (one forward
    for the linearisation point is counted, as a native backward needs the graph)."""
    if not (isinstance(k, (int, np.integer)) and k >= 1):
        raise ContractViolation(f"k must be a positive int, got {k!r}")
    N, d = model.n(x_r), model.out_dim
    ops.add("forward", N)
    S = np.empty((k, model.P))
    for i in range(k):
        U = rng.normal(scale=1.0 / np.sqrt(k), size=(N, d))
        S[i] = model.vjp(x_r, theta, U)
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
