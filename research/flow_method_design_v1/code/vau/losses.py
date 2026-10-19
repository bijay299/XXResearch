"""Native GEM target loss (numpy, FP64) with analytic gradients wrt the student prediction v.

Transcribed from GEM c328fc6 gem/operators/erasure/erasure_gem.py:234-249 (VERIFIED by reading;
same block as REF/results/cpr_gem_cpu_prototype/code/cpr_gem/losses.py:10-22):

    dist_pos = F.pairwise_distance(pred_flat, pos_flat, p=2)      # ||v - a + eps*1||, eps = 1e-6
    dist_neg = F.pairwise_distance(pred_flat, neg_flat, p=2)      # ||v - b + eps*1||
    target_loss = relu(dist_pos - eta * dist_neg).sum()
    loss = gamma * target_loss / n_latents

Teachers a, b are constants (detached): no gradient flows to them. Gradient conventions follow torch
autograd: relu'(0) = 0 (strict >0 mask) and the L2-norm backward is 0 where the norm is exactly 0.
GEM casts to float32 (`.float()`); this prototype keeps FP64 (deviation, see README_E.md).
CPR / anchor-regression formulas match REF losses.py:119-187 (cpr_alpha, cpr_loss, anchor_regression).
"""
from __future__ import annotations

import numpy as np

from .contracts import ContractViolation, check_array

EPS_N = 1e-6   # F.pairwise_distance default eps (all-ones shift)
TAU_L = 1e-6   # CPR contrast-degeneracy threshold (REF frozen design choice)


def _prep(v, a, b, n_latents):
    for name, t in (("v", v), ("a", a), ("b", b)):
        check_array(name, t)
        if t.ndim < 2:
            raise ContractViolation(f"{name} needs a batch dim plus >=1 feature dim, got {t.shape}")
    if not (v.shape == a.shape == b.shape):
        raise ContractViolation(f"shape mismatch: v{v.shape} a{a.shape} b{b.shape}")
    if not isinstance(n_latents, (int, np.integer)) or isinstance(n_latents, bool) or n_latents <= 0:
        raise ContractViolation(f"n_latents must be a positive int, got {n_latents!r}")
    n = v.shape[0]
    return v.reshape(n, -1), a.reshape(n, -1), b.reshape(n, -1)


def pairwise_distance(x, y, eps=EPS_N):
    """Exact F.pairwise_distance(x, y, p=2, eps) semantics: ||x - y + eps*1|| per row."""
    return np.linalg.norm(x - y + eps, axis=1)


def _unit(diff, dist):
    # torch norm backward: diff/norm, 0 where norm == 0
    out = np.zeros_like(diff)
    nz = dist > 0
    out[nz] = diff[nz] / dist[nz, None]
    return out


def gem_loss(v, a, b, n_latents, eta=1.0, gamma=100.0):
    """Native GEM target loss. Returns (loss, dloss/dv with v's shape)."""
    vf, af, bf = _prep(v, a, b, n_latents)
    dp_vec, dn_vec = vf - af + EPS_N, vf - bf + EPS_N
    dp, dn = np.linalg.norm(dp_vec, axis=1), np.linalg.norm(dn_vec, axis=1)
    h = dp - eta * dn
    c = gamma / n_latents
    loss = c * np.maximum(h, 0.0).sum()
    act = (h > 0).astype(np.float64)[:, None]
    grad = c * act * (_unit(dp_vec, dp) - eta * _unit(dn_vec, dn))
    return float(loss), grad.reshape(v.shape)


def anchor_regression_loss(v, a, b, n_latents, eta=0.0, gamma=100.0):
    """eta = 0 in the native loss (pure attraction to the anchor teacher); eta != 0 is rejected."""
    if eta != 0.0:
        raise ContractViolation("anchor_regression_loss is defined only at eta=0")
    return gem_loss(v, a, b, n_latents, eta=0.0, gamma=gamma)


def cpr_alpha(vf, af, bf, tau_L=TAU_L):
    contrast = bf - af
    L = np.linalg.norm(contrast, axis=1)
    live = L > tau_L
    e = np.zeros_like(contrast)
    e[live] = contrast[live] / L[live, None]
    alpha = ((bf - vf) * e).sum(axis=1)
    return alpha, e, L, live


def cpr_loss(v, a, b, n_latents, eta=1.0, gamma=100.0, tau_L=TAU_L):
    """CPR: D- replaced by alpha = (b-v).e_tau, e_tau = (b-a)/||b-a|| (0 if ||b-a|| <= tau_L)."""
    vf, af, bf = _prep(v, a, b, n_latents)
    dp_vec = vf - af + EPS_N
    dp = np.linalg.norm(dp_vec, axis=1)
    alpha, e, _, _ = cpr_alpha(vf, af, bf, tau_L)
    h = dp - eta * alpha
    c = gamma / n_latents
    loss = c * np.maximum(h, 0.0).sum()
    act = (h > 0).astype(np.float64)[:, None]
    grad = c * act * (_unit(dp_vec, dp) + eta * e)       # d alpha / dv = -e
    return float(loss), grad.reshape(v.shape)


def retention_penalty(fr, fr0, lam):
    """lam/2 * ||fr - fr0||^2 (sum over all retained samples and entries). Returns (value, d/dfr)."""
    check_array("fr", fr)
    check_array("fr0", fr0, shape=fr.shape)
    r = fr - fr0
    return float(0.5 * lam * np.sum(r * r)), lam * r


LOSSES = {"gem": gem_loss, "cpr": cpr_loss, "anchor": anchor_regression_loss}
