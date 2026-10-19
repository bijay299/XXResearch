"""Source-faithful native GEM target loss, the CPR candidate, and the declared controls.

NATIVE REFERENCE ATTRIBUTION
----------------------------
The native baseline below is a transcription of the small target-loss block in
GEM, pinned at commit c328fc6bec687efe60223498d1b6bed2e5fbcffd:

    https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/gem/operators/erasure/erasure_gem.py

The upstream block (lines 233-249 of that file) reads:

    pred_flat = predicted_target_score.float().reshape(predicted_target_score.shape[0], -1)
    pos_flat  = anchor_score.float().reshape(anchor_score.shape[0], -1)
    neg_flat  = target_score.float().reshape(target_score.shape[0], -1)

    dist_pos = torch.nn.functional.pairwise_distance(pred_flat, pos_flat, p=2)
    dist_neg = torch.nn.functional.pairwise_distance(pred_flat, neg_flat, p=2)

    target_loss = torch.relu(dist_pos - config.eta * dist_neg).sum()

    loss = target_loss
    loss = (config.gamma * loss) / n_latents

GEM is third-party work by its authors; only the loss arithmetic is reproduced
here, transcribed by reading.  The GEM operator / pipeline modules are NEVER
imported -- this prototype has no GEM, diffusers, transformers or PEFT
dependency.

IMPORTANT SEMANTICS OF ``F.pairwise_distance``
----------------------------------------------
``F.pairwise_distance(x, y, p=2, eps=1e-6)`` computes ``||x - y + eps*ones||_2``
-- an ALL-ONES SHIFT of the difference, not a centered smoothed norm.  Verified
empirically on this environment: ``pairwise_distance(0, 0) == eps*sqrt(d)``.
This shift is preserved exactly, because it is what native GEM actually
optimises and it induces a real transverse gradient component at ``v == b``.
"""

import torch
import torch.nn.functional as F

# Frozen native defaults (upstream dataclass defaults / torch defaults).
EPS_N = 1e-6        # F.pairwise_distance default eps -- the all-ones shift.
TAU_L = 1e-6        # CPR contrast-degeneracy threshold. A FROZEN DESIGN CHOICE,
                    # not an empirically validated scale.


class ContractViolation(ValueError):
    """Raised for invalid shapes, nonfinite inputs or invalid divisors.

    The contract requires rejection, never silent repair.
    """


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def _validate(v, a, b, n_latents):
    """Reject invalid shapes, nonfinite inputs and invalid divisors."""
    for name, t in (("v", v), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor):
            raise ContractViolation(f"{name} is not a torch.Tensor")
        if t.is_cuda:
            raise ContractViolation(f"{name} is a CUDA tensor; this prototype is CPU-only")
        if t.dim() < 2:
            raise ContractViolation(f"{name} must have a batch dim plus >=1 feature dim, got {tuple(t.shape)}")
        if not torch.isfinite(t).all():
            raise ContractViolation(f"{name} contains nonfinite values")
    if not (v.shape == a.shape == b.shape):
        raise ContractViolation(
            f"shape mismatch: v{tuple(v.shape)} a{tuple(a.shape)} b{tuple(b.shape)}")
    if not isinstance(n_latents, int) or n_latents <= 0:
        raise ContractViolation(f"n_latents must be a positive int, got {n_latents!r}")


def _flatten(v, a, b):
    """Cast to FP32 and flatten all nonbatch dims exactly as native GEM does.

    Teachers are detached BEFORE every norm / contrast / direction calculation.
    """
    n = v.shape[0]
    v_f = v.float().reshape(n, -1)
    a_f = a.detach().float().reshape(n, -1)   # detached anchor teacher
    b_f = b.detach().float().reshape(n, -1)   # detached target teacher
    return v_f, a_f, b_f


def _scale(per_sample, gamma_scale, n_latents):
    """Native reduction and multiplier: ``.sum()`` then ``gamma/n_latents``."""
    loss = per_sample.sum()
    return loss * (gamma_scale / n_latents)


# --------------------------------------------------------------------------
# 1. native GEM  (the baseline)
# --------------------------------------------------------------------------

def native_gem_loss(v, a, b, eta, gamma_scale=1.0, n_latents=1):
    """Source-faithful native GEM target loss.

    ``D_plus  = pairwise_distance(v, a)``  == ``||v - a + eps_n|| ``
    ``D_minus = pairwise_distance(v, b)``  == ``||v - b + eps_n|| ``
    ``loss = relu(D_plus - eta*D_minus).sum() * gamma_scale / n_latents``

    ``relu``'s boundary derivative is zero; ``clamp`` is NOT substituted.
    ``n_latents`` is supplied independently and is NOT the batch size.
    """
    _validate(v, a, b, n_latents)
    v_f, a_f, b_f = _flatten(v, a, b)
    d_plus = F.pairwise_distance(v_f, a_f, p=2)
    d_minus = F.pairwise_distance(v_f, b_f, p=2)
    return _scale(torch.relu(d_plus - eta * d_minus), gamma_scale, n_latents)


# --------------------------------------------------------------------------
# 2. CPR  (the candidate) -- changes ONLY D_minus
# --------------------------------------------------------------------------

def cpr_alpha(v, a, b, tau_L=TAU_L):
    """Directional repulsion coordinate ``alpha = (b - v) . e_tau``.

    ``L = ||b - a||`` per sample (a plain, UNSHIFTED norm: this is a contrast
    magnitude, not one of native GEM's two shifted distances).
    ``e_tau = (b - a)/L`` when ``L > tau_L``, otherwise EXACTLY zero.

    A safe denominator is used before masking so no eager ``0/0`` is evaluated.
    """
    v_f, a_f, b_f = _flatten(v, a, b)
    contrast = b_f - a_f                                   # detached (teachers)
    L = torch.linalg.vector_norm(contrast, dim=1)          # (n,)
    live = L > tau_L
    safe_L = torch.where(live, L, torch.ones_like(L))      # safe denom BEFORE masking
    e_tau = contrast / safe_L.unsqueeze(1)
    e_tau = torch.where(live.unsqueeze(1), e_tau, torch.zeros_like(e_tau))
    alpha = ((b_f - v_f) * e_tau).sum(dim=1)               # exactly 0 when degenerate
    return alpha, L, e_tau, live


def cpr_loss(v, a, b, eta, gamma_scale=1.0, n_latents=1, tau_L=TAU_L):
    """CPR: native attraction, hinge, reduction and multiplier unchanged;
    ``D_minus`` replaced by the signed directional coordinate ``alpha``.

    At small contrast (``L <= tau_L``) repulsion is disabled (``alpha == 0``)
    but attraction remains.
    """
    _validate(v, a, b, n_latents)
    v_f, a_f, b_f = _flatten(v, a, b)
    d_plus = F.pairwise_distance(v_f, a_f, p=2)            # native attraction, unchanged
    alpha, _, _, _ = cpr_alpha(v, a, b, tau_L=tau_L)
    return _scale(torch.relu(d_plus - eta * alpha), gamma_scale, n_latents)


# --------------------------------------------------------------------------
# 3. controls
# --------------------------------------------------------------------------

def centered_smooth_radial_loss(v, a, b, eta, gamma_scale=1.0, n_latents=1):
    """Centered smooth radial control: replace ONLY ``D_minus`` by
    ``sqrt(||v-b||^2 + delta_d^2)`` with fixed ``delta_d = eps_n*sqrt(d)``.

    ``delta_d`` is chosen so the control MATCHES native repulsion's VALUE at
    ``v == b`` (native gives ``||eps_n*ones|| = eps_n*sqrt(d)``) while removing
    native's all-ones DIRECTIONAL shift.

    This is a DIAGNOSTIC control.  It is neither native GEM nor a complete
    causal isolation of the shift: it changes the smoothing geometry globally,
    not only at the degenerate point.
    """
    _validate(v, a, b, n_latents)
    v_f, a_f, b_f = _flatten(v, a, b)
    d = v_f.shape[1]
    delta_d = EPS_N * (d ** 0.5)
    d_plus = F.pairwise_distance(v_f, a_f, p=2)            # native attraction, unchanged
    diff = v_f - b_f
    d_minus = torch.sqrt((diff * diff).sum(dim=1) + delta_d ** 2)
    return _scale(torch.relu(d_plus - eta * d_minus), gamma_scale, n_latents)


def anchor_regression_loss(v, a, b, eta=0.0, gamma_scale=1.0, n_latents=1):
    """Ordinary anchor regression: ``eta = 0`` in the NATIVE loss.

    ``eta`` is accepted only to keep one common call signature; a nonzero value
    is a contract violation for this control.
    """
    if eta != 0.0:
        raise ContractViolation("anchor_regression_loss is defined only at eta=0")
    return native_gem_loss(v, a, b, 0.0, gamma_scale=gamma_scale, n_latents=n_latents)


NOT_APPLICABLE = "NOT_APPLICABLE"


def axial_margin_loss(v, a, b, eta, gamma_scale=1.0, n_latents=1, tau_L=TAU_L):
    """Axial margin control: ``relu(L - (1+eta)*alpha)``, defined for ``L > tau_L``.

    Returns ``(loss, status)``.  If ANY sample is degenerate (``L <= tau_L``)
    the control is marked ``NOT_APPLICABLE`` for that fixture and the returned
    loss is ``None`` -- degenerate cases are not silently repaired.
    """
    _validate(v, a, b, n_latents)
    alpha, L, _, live = cpr_alpha(v, a, b, tau_L=tau_L)
    if not bool(live.all()):
        return None, NOT_APPLICABLE
    per_sample = torch.relu(L - (1.0 + eta) * alpha)
    return _scale(per_sample, gamma_scale, n_latents), "OK"


def margin_truncated_anchor_loss(v, a, b, k_t, gamma_scale=1.0, n_latents=1):
    """Margin-truncated anchor control: ``relu(D_plus - k_t)``.

    ``k_t`` is a DETACHED threshold (scalar or per-sample tensor), supplied by
    :func:`ideal_k_t` or :func:`matched_k_t`.
    """
    _validate(v, a, b, n_latents)
    v_f, a_f, b_f = _flatten(v, a, b)
    d_plus = F.pairwise_distance(v_f, a_f, p=2)
    if isinstance(k_t, torch.Tensor):
        k_t = k_t.detach()
    return _scale(torch.relu(d_plus - k_t), gamma_scale, n_latents)


# --------------------------------------------------------------------------
# thresholds for the margin-truncated control
# --------------------------------------------------------------------------

def ideal_k_t(L, eta):
    """``k_t = eta*L/(1+eta)``.

    For the IDEAL UNSHIFTED COLLINEAR reference only: there, CPR's hinge
    vanishes at ``alpha = L/(1+eta)``.  This identity is verified ONLY inside
    that domain and is NOT an exact native match off-axis.
    """
    return eta * L / (1.0 + eta)


# Frozen bounded bisection policy for the stronger native collinear match.
BISECT_MAX_ITER = 200
BISECT_XTOL = 1e-12
ROOT_RESIDUAL_TOL = 1e-5     # FP32 matching tolerance on the hinge interior

NOT_BRACKETED = "NOT_BRACKETED"
ROOT_UNRESOLVED = "ROOT_UNRESOLVED"


def matched_k_t(a_s, b_s, eta, tau_L=TAU_L):
    """Stronger native collinear match for a SINGLE sample (1-D tensors).

    Walks ``v(t) = b - t*e_tau`` for ``t in [0, L]`` and finds a root of the
    ACTUAL CPR hinge interior

        phi(t) = D_plus(v(t)) - eta*((b - v(t)) . e_tau)

    using the SAME numerical direction and precision as the implemented losses
    (FP32, native shifted ``D_plus``).  Requires a STRICT valid bracket
    (``phi(0) > 0 > phi(L)``) and a frozen bounded bisection.

    Returns ``(k_t, status, info)``.  ``k_t`` is detached.
    This scalar calculation DEFINES A CONTROL; it is not a parameter search.
    """
    a_s = a_s.detach().float().reshape(-1)
    b_s = b_s.detach().float().reshape(-1)
    contrast = b_s - a_s
    L = torch.linalg.vector_norm(contrast)
    info = {"L": float(L)}
    if float(L) <= tau_L:
        return None, NOT_BRACKETED, {**info, "reason": "degenerate contrast L <= tau_L"}
    e_tau = contrast / L

    def phi(t):
        v_t = (b_s - t * e_tau).reshape(1, -1)
        d_plus = F.pairwise_distance(v_t, a_s.reshape(1, -1), p=2)
        alpha = ((b_s.reshape(1, -1) - v_t) * e_tau.reshape(1, -1)).sum(dim=1)
        return float(d_plus - eta * alpha)

    lo, hi = 0.0, float(L)
    f_lo, f_hi = phi(lo), phi(hi)
    info.update({"phi_lo": f_lo, "phi_hi": f_hi})
    if not (f_lo > 0.0 > f_hi):                     # STRICT bracket required
        return None, NOT_BRACKETED, info

    for _ in range(BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = phi(mid)
        if f_mid > 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
        if (hi - lo) <= BISECT_XTOL:
            break

    t_star = 0.5 * (lo + hi)
    residual = phi(t_star)
    info.update({"t_star": t_star, "root_residual": residual,
                 "matching_tol": ROOT_RESIDUAL_TOL, "bracket_width": hi - lo})
    if abs(residual) > ROOT_RESIDUAL_TOL:
        return None, ROOT_UNRESOLVED, info

    v_star = b_s - t_star * e_tau
    alpha_star = float(((b_s - v_star) * e_tau).sum())
    k_t = torch.tensor(eta * alpha_star).detach()   # detached threshold
    info["alpha_star"] = alpha_star
    return k_t, "OK", info


LOSSES = {
    "native_gem": native_gem_loss,
    "cpr": cpr_loss,
    "centered_smooth_radial": centered_smooth_radial_loss,
    "anchor_regression": anchor_regression_loss,
}
