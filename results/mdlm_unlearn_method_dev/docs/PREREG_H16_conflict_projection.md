# Pre-registration — H16: retain-conflict projection in parameter space

**Registered:** 2026-10-21, before any H16 arm ran (smoke test excepted)
**Script:** `scripts/train_h16.py` · **Queue:** `scripts/queue_h16.sh`
**Triggered by:** the decision rule in `docs/H16_DESIGN_conflict_projection.md` §6, branch
"H15 clears neither"

---

## 1. Why H15 is closed, and what it leaves

H15 localized MDU's anchor loss to positions chosen by a selector that **is validated**: it
moves only the positions it selects, and its complement carries nothing forgettable (the
anti-control's `p_true` *rises* 0.606 → 0.733 and at generation level it reaches DISCLOSE 0.725
at retain 0.719 — no unlearning at all).

But localization **does not move the frontier**. Matched `λ_forget`, generation-level:

| λ_forget | MDU DISC / retain | H15 DISC / retain |
|---|---|---|
| 1 | 0.317 / **0.524** | 0.092 / 0.085 |
| 2 | 0.225 / **0.403** | 0.150 / 0.094 |
| 3 | pending | 0.133 / 0.013 |
| 8 | 0.067 / 0.091 | 0.067 / **0.199** |

The curves cross. H15 wins only at DISCLOSE 0.067, where both models are unusable
(retain 0.199 / 0.091 against θ₀'s 0.762). In the usable region (retain > 0.4) MDU is strictly
better and H15 never arrives: at DISCLOSE ≈ 0.15 H15 gives retain 0.094 while MDU interpolates
to ≈ 0.25. Localization makes the loss more aggressive per unit λ, on the same or a worse
frontier.

`λ_retain` is also closed as a lever: at `λ_retain ∈ {4,16}` the forgetting is cancelled and
`p_true` ends *above* θ₀ (0.675, 0.701, 0.757 vs 0.606).

## 2. The frontier to beat, fixed now

**MDU at `λ_forget = 1`: DISCLOSE 0.317, DEGENERATE 0.075, REPLACE 0.175, IGNORANT 0.433,
retain ASCR 0.524, TV→pristine 0.407.** This is the best point any arm has reached, baseline or
candidate. Reference points: θ₀ at DISCLOSE 0.792 / retain 0.762; pristine target at
DISCLOSE 0.030 / DEGENERATE 0.000 / REPLACE 0.470 / IGNORANT 0.388.

"Beats the baseline" means **strictly dominating this point or the MDU curve through it** —
lower DISCLOSE at equal-or-better retain, or higher retain at equal-or-better DISCLOSE — not
winning on one coordinate while losing the other.

## 3. H16

Act on the update direction rather than the loss's support, because Finding 14/15's diagnosis is
that the damage transfers across **inputs** through shared parameters, which position selection
cannot reach.

```
g_f = ∇θ L_forget            (MDU's anchor-first KL; --localize adds H15's selector)
g_r = ∇θ L_retain            (computed every step already)
R   = orthonormal basis of the last k retain gradients
g_f⊥ = g_f − R Rᵀ g_f
step on  g_f⊥ + λ_r·g_r
```

To first order a step along `g_f⊥` leaves the retain loss unchanged while still reducing the
forget loss by `‖g_f⊥‖²`. The forgetting that survives is the forgetting that does not require
moving retain — ATWU's retain-conflict criterion (arXiv 2606.06320 Eq. 2) in parameter space
instead of at the token level.

Arms: `none` (= the H15/MDU baseline in this script), `proj` with `k ∈ {1, 8, 32}`, `pcgrad`,
and `anti` (projects **onto** R, keeping only the conflicting part).

## 4. Pre-registered predictions and kill rules

Thresholds identical to H15's so the two are directly comparable, plus a dominance rule.

| # | prediction | kill condition |
|---|---|---|
| **Q1** | retain ASCR ≥ **0.40** | < 0.40 → parameter-space projection does not protect retain either |
| **Q2** | DISCLOSE ≤ **0.15** and DEGENERATE ≤ **0.25** | either exceeded → forgetting was bought away, the λ_retain failure again |
| **Q3** | `cos(g_f, g_r)` after projection is < half its magnitude before | not reduced → the projection is not doing what it claims and Q1/Q2 are uninterpretable |
| **Q4** | `anti` fails Q1+Q2 together | if the anti-projection matches, the effect is not conflict removal |
| **Q5** | H16 **strictly dominates** MDU λ=1 (§2) | if it only matches or trades one coordinate for the other, there is no contribution |

Q1 **and** Q2 must hold together. Q5 is the one that decides whether this is a method or a
finding.

## 5. Q3 makes the run informative either way

Q3 is measured directly, before and after projection, every step. If gradient alignment
**cannot** be driven toward zero at any rank, then forget and retain gradients are not
geometrically separable in this model, and the entire projection/orthogonalization class —
orthogonal gradient descent, GEM/A-GEM, PCGrad, O-LoRA applied to MDLM unlearning — is closed
by measurement rather than by one failed arm. That is a publishable negative independent of Q1.

The main design risk is stated in advance: projecting one direction out of a 42 M-dimensional
LoRA space removes almost nothing, and one minibatch gradient is a noisy estimate of the retain
direction. Hence the rank sweep `k ∈ {1, 8, 32}` with `k=1` included as the degenerate case.

## 6. Novelty, stated plainly before any result

The projection machinery is **standard** in continual learning (orthogonal gradient descent,
GEM/A-GEM, PCGrad, O-LoRA) and nothing about it is new. What the primary-source sweep did not
find is the pairing with an **anchor-matching** forget loss on a **masked diffusion LM**, with
the motivation coming from a measurement rather than an analogy. If H16 works, the contribution
is **the diagnosis plus the combination, not the projection** — and the write-up will say so.
ATWU remains a required baseline and is composable with this, not rival to it.

## 7. Caveats carried forward

One backbone, one dataset, one seed, 120 forget / 80 retain probes; multi-seed only if an arm
passes. H15's retain ASCR was non-monotonic in λ_forget (0.085, 0.094, 0.013, 0.199 at
λ = 1, 2, 3, 8), so single points are not to be trusted — curves are read, not points. No
robustness claim: anchor-matching families recover under finetuning attack (arXiv 2506.06278)
and our own study found MDLM unlearning reversible 0.049 → 0.605 in 10 steps.
