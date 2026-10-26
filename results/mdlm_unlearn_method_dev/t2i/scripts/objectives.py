"""The regularisers and meta-objective terms, as PURE FUNCTIONS of parameter lists and losses.

Separated out so the signs and the gradient directions can be tested deterministically on CPU in
milliseconds, with no diffusers, no CUDA and no checkpoint. The audit asked for exactly that:
"add a small deterministic test verifying the intended regularizer direction and gradient" and
"implement and test each required term and its gradient sign".

Two weight-space terms live here and they are NOT variants of one sign flip:

  anchor_penalty   -- OURS. +lam * ||theta - theta_base||^2 / ||theta_base||^2.
                      Minimised => PULLS TOWARD the base weights. This is what erase_wsr.py has
                      always computed, and what FINDING61/62 measured. It is a legitimate method;
                      it is simply not the cited paper's.

  wdr_forget_loss  -- FAITHFUL to 'From Dormant to Deleted' (arXiv 2505.22310) section 6.1 and its
                      released code (shoaibahmed/vision_relearning @7b9a1cc, vision_unlearner.py
                      L366-373):
                          weight_dist_loss = - sqrt(mean(square(theta_pre - theta)) + eps)
                          loss             = alpha * weight_dist_loss
                      Minimised => PUSHES AWAY FROM the base weights.

Four differences beyond the sign, which is why the audit's "do not blindly negate the existing
squared penalty" is correct:

  1. ROOT-MEAN-SQUARE, not a sum of squares. The reference takes sqrt(mean(.)) over the flattened
     parameter vector, so the term is O(per-coordinate RMS) and does not grow with model size.
  2. NO normalisation by ||theta_base||. The reference divides by the parameter COUNT (via mean),
     not by the base weight norm. Our anchor divides by ||theta_base||^2.
  3. It REPLACES the forget objective rather than being added to one. In the reference,
     `weight_dist_loss` IS the forget-iteration loss (`forget_loss_holder = weight_dist_loss`);
     there is no separate ascent-on-forget-data term at all. Our anchor is a THIRD term added to an
     erase term and a retain term.
  4. The retain term is unscaled (coefficient exactly 1.0) while the distance term carries alpha;
     the reference alternates forget and retain iterations rather than summing them in one step.

An eps inside the sqrt is load-bearing: at theta == theta_base the gradient of sqrt(x) at x=0 is
infinite, and the reference guards it with eps=1e-8. Reproduced.
"""
import torch

EPS_REFERENCE = 1e-8


def flat_diff(params, ref):
    """theta_pre - theta, flattened, exactly as the reference's flatten_param_dict difference."""
    return torch.cat([(r - p).reshape(-1) for p, r in zip(params, ref)])


def wdr_forget_loss(params, ref, alpha=1.0, eps=EPS_REFERENCE):
    """FAITHFUL 'Weight Dist Reg' forget loss. Minimising this MAXIMISES RMS distance from `ref`.

    Returns alpha * ( - sqrt( mean( (theta_pre - theta)^2 ) + eps ) ), so the value is <= 0 and
    becomes more negative as the model moves away from the reference.
    """
    d = flat_diff(params, ref)
    return alpha * (-torch.sqrt(torch.mean(d * d) + eps))


def wdr_rms_distance(params, ref, eps=EPS_REFERENCE):
    """The positive RMS distance itself, for logging. wdr_forget_loss == -alpha * this."""
    d = flat_diff(params, ref)
    return torch.sqrt(torch.mean(d * d) + eps)


def anchor_penalty(params, ref, lam=1.0, ref_sq=None):
    """OURS (erase_wsr.py). Minimising this PULLS TOWARD `ref`. Preserved unchanged so FINDING61
    and FINDING62 remain reproducible bit-for-bit; renamed, not deleted."""
    if ref_sq is None:
        ref_sq = sum(float((r.double() ** 2).sum()) for r in ref)
    tot = 0.0
    for p, r in zip(params, ref):
        tot = tot + ((p - r.to(p.device)) ** 2).sum()
    return lam * tot / ref_sq


def meta_terms(l_target_prime, l_retain_prime, l_retain_theta, zeta=1.0):
    """Meta-Unlearning (arXiv 2410.12777) Eq. 8 meta term, every component explicit:

        meta = - L_target(theta')  -  zeta * [ L_retain(theta') - L_retain(theta) ]

    Verified against the released implementation (sail-sg/Meta-Unlearning @5a2a760, src/meta.py
    L920-1030), which accumulates exactly this signed structure into `sum_grads`:

        + gamma1_1 * grad L(irt @ theta)      <- pre-attack retain, POSITIVE
        + gamma1_2 * grad L(tgt @ theta)      <- pre-attack retain (related concepts), POSITIVE
          ... one simulated malicious fine-tuning step on the harmful pool, full-UNet scope ...
        - gamma2_1 * grad L(hrm @ theta')     <- post-attack TARGET/forget, NEGATIVE
        - gamma2_2 * grad L(tgt @ theta')     <- post-attack retain, NEGATIVE
        - gamma2_3 * grad L(irt @ theta')     <- post-attack retain, NEGATIVE

    so the paper's R is the union of the reference's `irt` (irrelevant) and `tgt` (related) pools,
    and the paper's F is the reference's `hrm` (harmful) pool. The reference lets each pool carry
    its own coefficient; `zeta` here is the paper's single shared one.

    Returns the scalar meta term. Sign convention: this is ADDED to the at-theta objective, and the
    whole thing is MINIMISED.
    """
    return -l_target_prime - zeta * (l_retain_prime - l_retain_theta)


def meta_terms_as_implemented_variantA(l_retain_prime, lam):
    """What meta_unlearn.py (variant A) actually computes for its meta term: -lam*L_retain(theta').

    Preserved as a named function so the gap is testable rather than asserted. Relative to
    `meta_terms` it is missing BOTH
        -L_target(theta')      (the post-attack target term)
    and
        +zeta*L_retain(theta)  (the matching pre-attack retain diffusion term)
    and its at-theta terms are teacher distillation, which is not algebraically either of them.
    """
    return -lam * l_retain_prime
