"""Six per-sample losses supplied to the pinned GEM trainer through a small loss-dispatch hook.

Scope: the pinned trainer (multimodal-ai-lab/GEM@c328fc6b...) owns the loop, sampler, teachers,
optimizer and reduction. This module is ONLY a per-sample loss dispatch:
  * arm `native_gem` returns None -> the trainer executes its ORIGINAL loss block;
  * other arms return a per-sample FP32 tensor that the trainer reduces with its own native
    scalar order `(gamma * per_sample.sum()) / n_latents`.

Definitions (v student, a anchor teacher, b target teacher; flattened over nonbatch dims, FP32):
    eps = 1e-6; Dplus = ||v - a + eps*1||; Dminus = ||v - b + eps*1||; L = ||b - a||
    e = (b - a)/L if L > 1e-6 else 0; alpha = (b - v) . e; d = flattened dimension.
A safe denominator is formed BEFORE masking. torch.relu semantics are used; clamp is never used.
"""
import torch

EPS = 1e-6
TAU_L = 1e-6
ARMS = ("native_gem", "cpr", "centered_radial", "truncated_anchor", "anchor_only", "axial_margin")
NATIVE_ARM = "native_gem"


class ContractViolation(ValueError):
    pass


class AxialDomainViolation(ContractViolation):
    """axial_margin evaluated with L <= 1e-6: an explicit ARM FAILURE (never skipped/fallback)."""


def prepare(v, a, b):
    if not (v.shape == a.shape == b.shape):
        raise ContractViolation(f"shape mismatch v{tuple(v.shape)} a{tuple(a.shape)} b{tuple(b.shape)}")
    n = v.shape[0]
    vf = v.float().reshape(n, -1)               # student keeps its autograd graph
    af = a.detach().float().reshape(n, -1)      # teachers detached
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
    safe_L = torch.where(live, L, torch.ones_like(L))        # safe denominator BEFORE masking
    e = contrast / safe_L.unsqueeze(1)
    e = torch.where(live.unsqueeze(1), e, torch.zeros_like(e))
    alpha = ((bf - vf) * e).sum(dim=1)
    return {"v": vf, "a": af, "b": bf, "d_plus": d_plus, "d_minus": d_minus, "L": L, "e": e,
            "alpha": alpha, "live": live, "d": vf.shape[1]}


def per_sample(arm, v, a, b, eta=1.0):
    if arm == NATIVE_ARM:
        return None
    if arm not in ARMS:
        raise ContractViolation(f"unknown arm {arm!r}")
    g = geometry(v, a, b)
    if arm == "cpr":
        return torch.relu(g["d_plus"] - eta * g["alpha"])
    if arm == "centered_radial":
        diff = g["v"] - g["b"]
        rad = torch.sqrt((diff * diff).sum(dim=1) + (EPS ** 2) * g["d"])
        return torch.relu(g["d_plus"] - eta * rad)
    if arm == "truncated_anchor":
        k_t = (eta * g["L"] / (1.0 + eta)).detach()
        return torch.relu(g["d_plus"] - k_t)
    if arm == "anchor_only":
        return g["d_plus"]
    if arm == "axial_margin":
        if not bool(g["live"].all()):
            raise AxialDomainViolation(
                f"axial_margin undefined for L <= {TAU_L}: {int((~g['live']).sum())}/{g['live'].numel()} degenerate")
        return torch.relu(g["L"] - (1.0 + eta) * g["alpha"])
    raise ContractViolation(arm)


def make_dispatch(arm, eta=1.0):
    """Hook `(pred_flat, pos_flat, neg_flat) -> per_sample tensor | None`, with call counting."""
    def hook(pred_flat, pos_flat, neg_flat):
        hook.calls += 1
        return per_sample(arm, pred_flat, pos_flat, neg_flat, eta=eta)
    hook.calls = 0
    hook.arm = arm
    hook.eta = eta
    return hook


def reference_native(v, a, b, eta=1.0):
    """HOOKED-FORMULA expression of the native objective relu(D+ - eta*D-) (linalg-norm form, used for CHECKS only;
    training for `native_gem` executes the trainer's own `_cpr_native_loss_block`)."""
    g = geometry(v, a, b)
    return torch.relu(g["d_plus"] - eta * g["d_minus"])


def hooked(arm, v, a, b, eta=1.0):
    """Per-sample hooked formula for ANY arm, including native (check use)."""
    return reference_native(v, a, b, eta) if arm == NATIVE_ARM else per_sample(arm, v, a, b, eta=eta)
