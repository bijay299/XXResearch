"""The six comparison arms, implemented afresh for this pilot.

ATTRIBUTION
-----------
The native GEM target loss (arm 1) reproduces the arithmetic of the small
target-loss block in GEM, pinned at
`multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd`,
`gem/operators/erasure/erasure_gem.py` (lines 233-249), read as a reference and
re-implemented here. GEM is third-party work by its authors. The GEM operator
and pipeline modules are NEVER imported.

`torch.nn.functional.pairwise_distance(x, y, p=2, eps=1e-6)` computes
`||x - y + eps*ones||_2` -- an ALL-ONES SHIFT, not a centered smoothed norm.
Rather than depend on that behaviour implicitly, the shift is written out
explicitly below so the formula is visible and testable.

Shared contract for every arm
-----------------------------
    v   student output          (requires grad)
    a   detached anchor teacher
    b   detached target teacher
Cast to FP32 and flatten all nonbatch dimensions. Teachers are detached BEFORE
every norm, contrast and direction calculation.

    eps    = 1e-6
    Dplus  = ||v - a + eps*ones||
    Dminus = ||v - b + eps*ones||
    L      = ||b - a||
    e      = (b-a)/L   if L > TAU_L   else exactly zero
    alpha  = (b-v) . e

Reduction, shared by every arm and preserved from native GEM:

    loss = (gamma * per_sample.sum()) / n_latents

`n_latents` is the native independently supplied divisor -- NOT the batch size.
`torch.relu` boundary behaviour is used as-is; `clamp` is never substituted.
"""

import torch

EPS = 1e-6
TAU_L = 1e-6

ARMS = (
    "native_gem",
    "cpr",
    "centered_radial",
    "margin_truncated",
    "anchor_only",
    "axial_margin",
)


class ContractViolation(ValueError):
    """Invalid shape, nonfinite input, invalid divisor, or a production-domain
    violation. Rejected, never silently repaired."""


class AxialDomainViolation(ContractViolation):
    """Axial margin evaluated where `L <= TAU_L`.

    The contract defines arm 6 only for `L > 1e-6`. In production this is an
    EXPLICITLY RECORDED ARM FAILURE: it is never skipped silently and no
    fallback value is invented.
    """


# --------------------------------------------------------------------------

def _prepare(v, a, b, n_latents):
    """Validate, cast to FP32, flatten nonbatch dims, detach teachers."""
    for name, t in (("v", v), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor):
            raise ContractViolation(f"{name} is not a torch.Tensor")
        if t.dim() < 2:
            raise ContractViolation(f"{name} needs a batch dim plus >=1 feature dim, got {tuple(t.shape)}")
    if not (v.shape == a.shape == b.shape):
        raise ContractViolation(f"shape mismatch: v{tuple(v.shape)} a{tuple(a.shape)} b{tuple(b.shape)}")
    if not isinstance(n_latents, int) or n_latents <= 0:
        raise ContractViolation(f"n_latents must be a positive int, got {n_latents!r}")
    n = v.shape[0]
    vf = v.float().reshape(n, -1)
    af = a.detach().float().reshape(n, -1)
    bf = b.detach().float().reshape(n, -1)
    for name, t in (("v", vf), ("a", af), ("b", bf)):
        if not torch.isfinite(t).all():
            raise ContractViolation(f"{name} contains nonfinite values")
    return vf, af, bf


def _shifted_norm(diff):
    """||diff + eps*ones||_2 -- the native all-ones shift, written explicitly."""
    return torch.linalg.vector_norm(diff + EPS, dim=1)


def geometry(v, a, b, n_latents=1):
    """Everything the six arms are built from. Teachers detached throughout."""
    vf, af, bf = _prepare(v, a, b, n_latents)
    d_plus = _shifted_norm(vf - af)
    d_minus = _shifted_norm(vf - bf)
    contrast = bf - af                                   # detached
    L = torch.linalg.vector_norm(contrast, dim=1)
    live = L > TAU_L
    # safe denominator BEFORE masking: no eager 0/0 is ever evaluated
    safe_L = torch.where(live, L, torch.ones_like(L))
    e = contrast / safe_L.unsqueeze(1)
    e = torch.where(live.unsqueeze(1), e, torch.zeros_like(e))
    alpha = ((bf - vf) * e).sum(dim=1)                   # exactly 0 when degenerate
    return {"v": vf, "a": af, "b": bf, "d_plus": d_plus, "d_minus": d_minus,
            "L": L, "e": e, "alpha": alpha, "live": live, "d": vf.shape[1]}


def reduce_native(per_sample, gamma, n_latents):
    """Native reduction and evaluation order: (gamma * sum) / n_latents."""
    return (gamma * per_sample.sum()) / n_latents


# --------------------------------------------------------------------------
# the six arms -- per-sample losses
# --------------------------------------------------------------------------

def per_sample_loss(arm, g, eta):
    """Per-sample loss for one arm, given the shared geometry dict `g`."""
    if arm == "native_gem":
        # 1. relu(Dplus - eta*Dminus)
        return torch.relu(g["d_plus"] - eta * g["d_minus"])

    if arm == "cpr":
        # 2. relu(Dplus - eta*alpha). At L <= TAU_L alpha is exactly zero, so
        #    repulsion is disabled while attraction (Dplus) remains.
        return torch.relu(g["d_plus"] - eta * g["alpha"])

    if arm == "centered_radial":
        # 3. relu(Dplus - eta*sqrt(||v-b||^2 + eps^2*d))
        diff = g["v"] - g["b"]
        dm = torch.sqrt((diff * diff).sum(dim=1) + (EPS ** 2) * g["d"])
        return torch.relu(g["d_plus"] - eta * dm)

    if arm == "margin_truncated":
        # 4. relu(Dplus - eta*L/(1+eta)), threshold DETACHED.
        #    Ideal unshifted collinear stopping interpretation ONLY. No claim of
        #    equal native first stopping, off-axis equivalence, or empirical
        #    suppression follows from this threshold.
        k_t = (eta * g["L"] / (1.0 + eta)).detach()
        return torch.relu(g["d_plus"] - k_t)

    if arm == "anchor_only":
        # 5. Dplus -- the native eta=0 control (native shifted norm).
        #    This is NOT MSE and NOT ESD.
        return g["d_plus"]

    if arm == "axial_margin":
        # 6. relu(L - (1+eta)*alpha), defined only for L > TAU_L.
        if not bool(g["live"].all()):
            n_bad = int((~g["live"]).sum())
            raise AxialDomainViolation(
                f"axial_margin is undefined for L <= {TAU_L}: {n_bad} of "
                f"{g['live'].numel()} samples are degenerate")
        return torch.relu(g["L"] - (1.0 + eta) * g["alpha"])

    raise ContractViolation(f"unknown arm {arm!r}")


def arm_loss(arm, v, a, b, eta=1.0, gamma=100.0, n_latents=1):
    """Full scalar loss for one arm, with the native reduction and order."""
    g = geometry(v, a, b, n_latents)
    return reduce_native(per_sample_loss(arm, g, eta), gamma, n_latents)
