"""Independent FP64 mathematical oracle, kept SEPARATE from the FP32 losses.

The implemented losses force FP32 (native GEM calls ``.float()``).  Applying a
naive double-precision ``gradcheck`` to a function that internally forces FP32
measures the FP32 rounding floor, not the mathematics -- so it is NOT done.
Instead this module re-derives the same formulas independently in FP64 and the
suite compares values/gradients at a disclosed FP32-limited tolerance.

These oracles are written from the mathematical statement of the contract, not
by calling the FP32 implementations.
"""

import math
import torch

EPS_N = 1e-6
TAU_L = 1e-6


def _as64(x):
    return x.detach().double().reshape(x.shape[0], -1)


def o_d_plus(v, a, eps_n=EPS_N):
    """``||v - a + eps_n*ones||_2`` in FP64 (native's SHIFTED norm)."""
    return torch.linalg.vector_norm(v - a + eps_n, dim=1)


def o_d_minus(v, b, eps_n=EPS_N):
    return torch.linalg.vector_norm(v - b + eps_n, dim=1)


def o_native(v, a, b, eta, gamma_scale=1.0, n_latents=1, eps_n=EPS_N):
    per = torch.relu(o_d_plus(v, a, eps_n) - eta * o_d_minus(v, b, eps_n))
    return per.sum() * (gamma_scale / n_latents)


def o_alpha(v, a, b, tau_L=TAU_L):
    contrast = b - a
    L = torch.linalg.vector_norm(contrast, dim=1)
    live = L > tau_L
    safe = torch.where(live, L, torch.ones_like(L))
    e = contrast / safe.unsqueeze(1)
    e = torch.where(live.unsqueeze(1), e, torch.zeros_like(e))
    return ((b - v) * e).sum(dim=1), L, e, live


def o_cpr(v, a, b, eta, gamma_scale=1.0, n_latents=1, eps_n=EPS_N, tau_L=TAU_L):
    alpha, _, _, _ = o_alpha(v, a, b, tau_L)
    per = torch.relu(o_d_plus(v, a, eps_n) - eta * alpha)
    return per.sum() * (gamma_scale / n_latents)


def o_centered(v, a, b, eta, gamma_scale=1.0, n_latents=1, eps_n=EPS_N):
    d = v.shape[1]
    delta_d = eps_n * math.sqrt(d)
    dm = torch.sqrt(((v - b) ** 2).sum(dim=1) + delta_d ** 2)
    per = torch.relu(o_d_plus(v, a, eps_n) - eta * dm)
    return per.sum() * (gamma_scale / n_latents)


# --------------------------------------------------------------------------
# closed-form analytical gradients w.r.t. v (smooth ACTIVE region only)
# --------------------------------------------------------------------------

def grad_native_analytic(v, a, b, eta, gamma_scale=1.0, n_latents=1, eps_n=EPS_N):
    """d/dv of the native loss where the hinge is strictly active.

    ``d(D_plus)/dv = (v - a + eps)/D_plus``;  likewise for ``D_minus``.
    The ``+eps*ones`` shift makes ``d(D_minus)/dv`` point along the ALL-ONES
    direction as ``v -> b`` -- this is the transverse artifact the centered
    control removes.
    """
    dp = o_d_plus(v, a, eps_n).unsqueeze(1)
    dm = o_d_minus(v, b, eps_n).unsqueeze(1)
    active = (dp.squeeze(1) - eta * dm.squeeze(1) > 0).double().unsqueeze(1)
    g = (v - a + eps_n) / dp - eta * (v - b + eps_n) / dm
    return active * g * (gamma_scale / n_latents)


def grad_cpr_analytic(v, a, b, eta, gamma_scale=1.0, n_latents=1,
                      eps_n=EPS_N, tau_L=TAU_L):
    """d/dv of CPR where the hinge is strictly active.  ``d(alpha)/dv = -e_tau``."""
    dp = o_d_plus(v, a, eps_n).unsqueeze(1)
    alpha, _, e, _ = o_alpha(v, a, b, tau_L)
    active = (dp.squeeze(1) - eta * alpha > 0).double().unsqueeze(1)
    g = (v - a + eps_n) / dp + eta * e        # -eta * (-e_tau) = +eta*e_tau
    return active * g * (gamma_scale / n_latents)


def grad_centered_analytic(v, a, b, eta, gamma_scale=1.0, n_latents=1, eps_n=EPS_N):
    d = v.shape[1]
    delta_d = eps_n * math.sqrt(d)
    dp = o_d_plus(v, a, eps_n).unsqueeze(1)
    dm = torch.sqrt(((v - b) ** 2).sum(dim=1) + delta_d ** 2).unsqueeze(1)
    active = (dp.squeeze(1) - eta * dm.squeeze(1) > 0).double().unsqueeze(1)
    g = (v - a + eps_n) / dp - eta * (v - b) / dm
    return active * g * (gamma_scale / n_latents)


# --------------------------------------------------------------------------
# finite differences (FP64), used AWAY from hinges and norm cusps
# --------------------------------------------------------------------------

def fd_grad(fn, v, h=1e-6):
    """Central finite differences in FP64.

    Disclosed precision: central differences carry truncation O(h^2) and
    FP64 cancellation O(eps64/h); at h=1e-6 the achievable accuracy is
    ~1e-9..1e-10 relative for well-scaled smooth functions.  This is used ONLY
    away from hinge boundaries and norm cusps.
    """
    v = v.detach().double()
    g = torch.zeros_like(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            vp = v.clone(); vp[i, j] += h
            vm = v.clone(); vm[i, j] -= h
            g[i, j] = (fn(vp) - fn(vm)) / (2 * h)
    return g
