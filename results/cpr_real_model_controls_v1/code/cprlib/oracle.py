"""Independent FP64 oracle for the six arms.

Written from the mathematical statement of the contract, NOT by calling
`cprlib.losses`. Kept separate from the FP32 production losses: the production
path forces FP32 (native GEM calls `.float()`), so a naive double-precision
gradcheck against it would measure the FP32 rounding floor, not the maths.

Each oracle returns both the value and the closed-form gradient w.r.t. `v` in
the smooth, strictly-active region, so the suite can compare against genuinely
independent expected values rather than against a restatement of the code.
"""

import math
import torch

EPS = 1e-6
TAU_L = 1e-6


def _g64(v, a, b):
    v = v.double(); a = a.detach().double(); b = b.detach().double()
    n = v.shape[0]
    v = v.reshape(n, -1); a = a.reshape(n, -1); b = b.reshape(n, -1)
    dp = torch.linalg.vector_norm(v - a + EPS, dim=1)
    dm = torch.linalg.vector_norm(v - b + EPS, dim=1)
    contrast = b - a
    L = torch.linalg.vector_norm(contrast, dim=1)
    live = L > TAU_L
    safe = torch.where(live, L, torch.ones_like(L))
    e = contrast / safe.unsqueeze(1)
    e = torch.where(live.unsqueeze(1), e, torch.zeros_like(e))
    alpha = ((b - v) * e).sum(dim=1)
    return v, a, b, dp, dm, L, e, alpha, live


def value(arm, v, a, b, eta=1.0, gamma=100.0, n_latents=1):
    v, a, b, dp, dm, L, e, alpha, live = _g64(v, a, b)
    if arm == "native_gem":
        per = torch.relu(dp - eta * dm)
    elif arm == "cpr":
        per = torch.relu(dp - eta * alpha)
    elif arm == "centered_radial":
        d = v.shape[1]
        rad = torch.sqrt(((v - b) ** 2).sum(dim=1) + (EPS ** 2) * d)
        per = torch.relu(dp - eta * rad)
    elif arm == "margin_truncated":
        per = torch.relu(dp - eta * L / (1.0 + eta))
    elif arm == "anchor_only":
        per = dp
    elif arm == "axial_margin":
        if not bool(live.all()):
            raise ValueError("axial_margin undefined for L <= TAU_L")
        per = torch.relu(L - (1.0 + eta) * alpha)
    else:
        raise ValueError(arm)
    return (gamma * per.sum()) / n_latents


def grad_v(arm, v, a, b, eta=1.0, gamma=100.0, n_latents=1):
    """Closed-form d(loss)/dv in the strictly-active smooth region.

    d||v-a+eps||/dv = (v-a+eps)/||v-a+eps||;  d(alpha)/dv = -e.
    """
    v, a, b, dp, dm, L, e, alpha, live = _g64(v, a, b)
    k = gamma / n_latents
    u_plus = (v - a + EPS) / dp.unsqueeze(1)
    if arm == "native_gem":
        act = (dp - eta * dm > 0).double().unsqueeze(1)
        g = u_plus - eta * (v - b + EPS) / dm.unsqueeze(1)
    elif arm == "cpr":
        act = (dp - eta * alpha > 0).double().unsqueeze(1)
        g = u_plus + eta * e                      # -eta * d(alpha)/dv = +eta*e
    elif arm == "centered_radial":
        d = v.shape[1]
        rad = torch.sqrt(((v - b) ** 2).sum(dim=1) + (EPS ** 2) * d)
        act = (dp - eta * rad > 0).double().unsqueeze(1)
        g = u_plus - eta * (v - b) / rad.unsqueeze(1)
    elif arm == "margin_truncated":
        act = (dp - eta * L / (1.0 + eta) > 0).double().unsqueeze(1)
        g = u_plus                                 # threshold is detached
    elif arm == "anchor_only":
        act = torch.ones_like(dp).unsqueeze(1)
        g = u_plus
    elif arm == "axial_margin":
        act = (L - (1.0 + eta) * alpha > 0).double().unsqueeze(1)
        g = (1.0 + eta) * e                        # -(1+eta)*d(alpha)/dv
    else:
        raise ValueError(arm)
    return k * act * g


def fd_grad(fn, v, h=1e-6):
    """FP64 central differences.

    Disclosed precision: truncation O(h^2) plus cancellation O(eps64/h); at
    h=1e-6 the usable floor for well-scaled smooth functions is ~1e-9 relative.
    Used ONLY away from hinge boundaries and norm cusps.
    """
    v = v.detach().double()
    g = torch.zeros_like(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            vp = v.clone(); vp[i, j] += h
            vm = v.clone(); vm[i, j] -= h
            g[i, j] = (fn(vp) - fn(vm)) / (2 * h)
    return g
