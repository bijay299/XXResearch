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
  4. The retain term is unscaled (coefficient exactly 1.0) while the distance term carries alpha.
     VERIFIED: `vision_unlearner.py:241-252` puts `weight_dist_reg` in the plain-CE retain branch
     and takes `loss = prediction_loss`, i.e. no alpha on the retain side.

!! SCHEDULE CORRECTION (Agent-3 fidelity audit, 2026-10-26). An earlier version of point 4 above
said "the reference alternates forget and retain iterations rather than summing them in one step".
THAT IS BACKWARDS. In the reference, alternation is gated on `current_cycle`, which is non-None
ONLY for methods whose name carries the `alternating_` prefix:

    vision_unlearner.py:411
        current_cycle = "retrain" if "alternating_" in unlearning_method else None

`weight_dist_reg` has no such prefix, so `current_cycle is None` and BOTH branches run on EVERY
iteration, each calling `.backward()` into the SAME grad buffer, followed by a SINGLE
`optimizer.step()` (`vision_unlearner.py:637-682`). The reference therefore SUMS the two terms into
one update. `erase_wdr_v2.py` instead takes two SEPARATE Adam steps per iteration and claims that
schedule matches the reference; it does not. See `erase_wdr_v2.py`'s docstring for the consequence
for FINDING67's alpha analysis. Fixing it requires a behaviour change and has NOT been done here.

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

    CITATION CORRECTED (Agent-3 fidelity audit, 2026-10-26). This previously cited
    "sail-sg/Meta-Unlearning @5a2a760, src/meta.py L920-1030" and said "Verified against". The
    signed structure below IS the reference's, but `src/meta.py` is NOT a runnable reference: as
    checked in it raises on its first inner step (`train_step` returns 4 values at
    `src/meta.py:626` and every call site unpacks 3, e.g. `:967`) and reads an unassigned
    `train_loss` at `:1045`. The WORKING implementations, and the ones the authors' own
    `scripts/esd_meta.sh` / `scripts/sdd_meta.sh` invoke, are `src/esd_meta.py` and
    `src/sdd_meta.py`. The structure below is re-verified against
    `sail-sg/Meta-Unlearning@5a2a7600895a2885323cbbf599acaf3f74a96f64 src/esd_meta.py:992-1058`,
    which accumulates exactly this signed structure into `sum_grads`:

        + gamma1_1 * grad L(irt @ theta)      <- pre-attack retain, POSITIVE
        + gamma1_2 * grad L(tgt @ theta)      <- pre-attack retain (related concepts), POSITIVE
          ... one simulated malicious fine-tuning step on the harmful pool, full-UNet scope ...
        - gamma2_1 * grad L(hrm @ theta')     <- post-attack TARGET/forget, NEGATIVE
        - gamma2_2 * grad L(tgt @ theta')     <- post-attack retain, NEGATIVE
        - gamma2_3 * grad L(irt @ theta')     <- post-attack retain, NEGATIVE

    so the paper's R is the union of the reference's `irt` (irrelevant) and `tgt` (related) pools,
    and the paper's F is the reference's `hrm` (harmful) pool. The reference lets each pool carry
    its own coefficient; `zeta` here is the paper's single shared one.

    !! WARNING ON THE SHARED `zeta` (Agent-3 fidelity audit, 2026-10-26). Sharing ONE coefficient
    between +L_retain(theta) and -L_retain(theta') makes the pair CANCEL. Under a first-order unroll
    with theta' = theta - alpha*grad L_F(theta) and alpha = 1e-5, the net expected contribution of
    the pair is zeta*(grad L_R(theta) - grad L_R(theta')) ~ zeta*alpha*H_R*grad L_F, a second-order
    quantity ~1e-5 of either term alone.

    Read that as a statement about the COEFFICIENT, not about the pairing. Because the same zeta
    multiplies both members, the cancellation is EXACT at every zeta > 0, so zeta cannot be tuned
    to change the balance -- it is inert as a weight, and untested (no sweep has been run). The
    PAIRING is a different matter and it is the operative mechanism: without +L_retain(theta) the
    remaining -zeta*L_retain(theta') is a biased penalty on the retain LEVEL, which rewards making
    benign capability worse everywhere (theta' ~= theta). Pairing converts it into a penalty on the
    CHANGE across the attack step and removes that incentive. See scripts/meta_unlearn_v2.py's
    docstring, defect (A), for the full argument.
    The reference does NOT share the coefficient: `gamma1_1` (pre-attack irt) and `gamma2_3`
    (post-attack irt) are independent arguments, all defaulting to 1.0
    (`src/esd_meta.py:90-94`), and the authors' own `scripts/esd_meta.sh` runs them ASYMMETRICALLY
    (gamma1_1 = 0.1 against gamma2_3 = 1.0; gamma1_2 = 0.01 against gamma2_2 = 1.0 -- note that
    script also has a copy-paste bug that passes `--gamma2_1` three times and never passes
    `--gamma2_2`/`--gamma2_3`). The asymmetry is what stops the cancellation in the reference.
    `meta_unlearn_v2.py` inherits the shared-zeta form; see its docstring.

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
