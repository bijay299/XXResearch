"""Six per-sample losses, supplied to the pinned GEM trainer as a small
attributed dispatch hook.

ATTRIBUTION / SCOPE
-------------------
The pinned GEM trainer at
`multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd`
(`gem/operators/erasure/erasure_gem.py`) owns the training loop, the trajectory
sampler, the teachers, the optimizer and the reduction. This module is ONLY a
per-sample loss dispatch:

  * arm `native_gem` returns `None`, and the trainer executes **its own
    original loss block** unchanged.
  * every other arm returns a per-sample tensor that the trainer reduces with
    its own native scalar order `(gamma * per_sample.sum()) / n_latents`.

Nothing here reimplements LoRA, the sampler, AdamW, or the reduction.

Definitions (teachers `a`, `b` already detached by the trainer; `v` the
student; all cast to FP32 and flattened over nonbatch dims):

    eps    = 1e-6
    Dplus  = ||v - a + eps*ones||
    Dminus = ||v - b + eps*ones||
    L      = ||b - a||
    e      = (b - a)/L  when L > 1e-6, else exactly zero
    alpha  = (b - v) . e

A safe denominator is formed BEFORE masking, so no eager 0/0 is evaluated.
`torch.relu` boundary semantics are used as-is; `clamp` is never substituted.
"""

import torch

EPS = 1e-6
TAU_L = 1e-6

ARMS = ("native_gem", "cpr", "centered_radial", "truncated_anchor",
        "anchor_only", "axial_margin")

NATIVE_ARM = "native_gem"


class ContractViolation(ValueError):
    pass


class AxialDomainViolation(ContractViolation):
    """Axial margin evaluated where L <= 1e-6.

    An explicit ARM FAILURE: never a skipped sample, never an invented
    fallback value.
    """


def prepare(v, a, b):
    """FP32, flattened over nonbatch dims, teachers detached (defensively)."""
    if not (v.shape == a.shape == b.shape):
        raise ContractViolation(
            f"shape mismatch v{tuple(v.shape)} a{tuple(a.shape)} b{tuple(b.shape)}")
    n = v.shape[0]
    vf = v.float().reshape(n, -1)
    af = a.detach().float().reshape(n, -1)
    bf = b.detach().float().reshape(n, -1)
    if not torch.isfinite(vf).all():
        raise ContractViolation("student contains nonfinite values")
    return vf, af, bf


def geometry(v, a, b):
    vf, af, bf = prepare(v, a, b)
    d_plus = torch.linalg.vector_norm(vf - af + EPS, dim=1)
    d_minus = torch.linalg.vector_norm(vf - bf + EPS, dim=1)
    contrast = bf - af
    L = torch.linalg.vector_norm(contrast, dim=1)
    live = L > TAU_L
    safe_L = torch.where(live, L, torch.ones_like(L))     # safe denom BEFORE masking
    e = contrast / safe_L.unsqueeze(1)
    e = torch.where(live.unsqueeze(1), e, torch.zeros_like(e))
    alpha = ((bf - vf) * e).sum(dim=1)
    return {"v": vf, "a": af, "b": bf, "d_plus": d_plus, "d_minus": d_minus,
            "L": L, "e": e, "alpha": alpha, "live": live, "d": vf.shape[1]}


def per_sample(arm, v, a, b, eta=1.0):
    """Per-sample loss for one arm, or None for the native arm."""
    if arm == NATIVE_ARM:
        return None                      # trainer runs its ORIGINAL loss block
    if arm not in ARMS:
        raise ContractViolation(f"unknown arm {arm!r}")
    g = geometry(v, a, b)

    if arm == "cpr":
        # attraction is RETAINED at degenerate contrast (alpha is exactly 0 there)
        return torch.relu(g["d_plus"] - eta * g["alpha"])

    if arm == "centered_radial":
        diff = g["v"] - g["b"]
        rad = torch.sqrt((diff * diff).sum(dim=1) + (EPS ** 2) * g["d"])
        return torch.relu(g["d_plus"] - eta * rad)

    if arm == "truncated_anchor":
        # threshold DETACHED. Ideal-unshifted-collinear interpretation ONLY;
        # no native first-stop matching is claimed.
        k_t = (eta * g["L"] / (1.0 + eta)).detach()
        return torch.relu(g["d_plus"] - k_t)

    if arm == "anchor_only":
        return g["d_plus"]               # native shifted norm. NOT MSE, NOT ESD.

    if arm == "axial_margin":
        if not bool(g["live"].all()):
            n_bad = int((~g["live"]).sum())
            raise AxialDomainViolation(
                f"axial_margin undefined for L <= {TAU_L}: {n_bad}/{g['live'].numel()} "
                f"samples degenerate")
        return torch.relu(g["L"] - (1.0 + eta) * g["alpha"])

    raise ContractViolation(arm)


def make_dispatch(arm, eta=1.0, log=None):
    """Build the hook handed to the patched GEM trainer.

    Returns a callable `(pred, anchor, target) -> per_sample tensor or None`.
    """
    def hook(pred_flat, pos_flat, neg_flat):
        out = per_sample(arm, pred_flat, pos_flat, neg_flat, eta=eta)
        if log is not None and out is not None:
            log(arm, float(out.detach().sum()))
        return out
    hook.arm = arm
    hook.eta = eta
    return hook
