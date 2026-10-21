# Pre-registration — H15: MDU's anchor, applied only where the oracle disagrees with θ₀

**Registered:** 2026-10-21, before any H15 arm was run
**Script:** `scripts/train_lam.py`, arms `lam_mdu` / `lam_mdu_anti` / `mdu`
**Rests on:** `outputs/ignorance_shape_main.json`, `outputs/arm_*.json`

---

## 0. A refuted hypothesis, stated first

H14's premise was that a *form-preserving* (entity-masked) anchor would avoid degeneracy because
its own generations are non-degenerate (DEGEN 0.0030). **Refuted.** Students trained against it
are 69–75% DEGENERATE. Anchor non-degeneracy does not transfer to the student, and prediction
P1 is dead.

I then hypothesised that genuine ignorance is low-entropy "confident confabulation" close to θ₀,
and pre-registered the test `|H(oracle) − H(θ₀)| < 0.5` nats. **Also refuted**, verdict
`IGNORANCE_IS_UNCERTAINTY_hypothesis_dies`: the true retrain oracle sits at **1.839** nats
against θ₀'s **1.024**. Entropy-raising is directionally correct. My framing was wrong and is
withdrawn.

## 1. The post-hoc observation this registers a test for

Measured at answer positions on 120 TOFU forget facts, 13,229 positions per model, same noise
draws and positions throughout:

| model | entropy | top1_p | p_true | true-rank median |
|---|---|---|---|---|
| `base_pristine` (saw neither) | 1.039 | 0.807 | 0.716 | 0 |
| **`retain_ref` (saw retain, NOT forget — the true oracle)** | **1.839** | **0.711** | **0.507** | **0** |
| `theta0` (saw both) | 1.024 | 0.799 | 0.631 | 0 |
| MDU λ=8 | **1.848** | 0.679 | 0.501 | 0 |
| EMDU λ=8 | 8.170 | 0.054 | 0.022 | 8 |
| entity-uniform λ=8 | 4.256 | 0.253 | 0.073 | 25 |
| entity-localized λ=8 | 4.259 | 0.261 | 0.103 | 10 |

Two readings, neither pre-registered, which is why this document exists:

**(i) There is a specific target shape and the entropy-raising family cannot stop at it.** The
oracle is at 1.839 nats / top1 0.711. MDU at λ=8 lands at 1.848 / 0.679 — essentially on it.
EMDU overshoots by **4.4×**, the entity anchors by **2.3×**. "Raise entropy" is directionally
right and quantitatively unbounded; FAM and uniform-prior targets have no stopping rule.

**(ii) MDU's failure is its SUPPORT, not its shape.** The oracle's true-token rank is **0 at the
median answer position** — the oracle and θ₀ *agree* at most positions, because only ~13% of
answer positions carry the fact (≈4.6 of ≈36 masked positions per state, consistent with the
earlier ordering measurements). MDU applies its loss with uniform weight `1/|M_t|`, so ~87% of
its force lands where there is nothing to forget. In the partition that shows up as the best
forget profile of any arm (DISCLOSE 0.067, DEGENERATE 0.150, REPLACE 0.392, IGNORANT 0.392,
TV→pristine 0.191) **paired with retain ASCR 0.091 against θ₀'s 0.762** — retain destroyed.

## 2. H15

> Keep MDU's anchor and temperature exactly as published — it already lands on the oracle's
> distributional shape — and change only **where** the loss is applied, concentrating it on the
> positions where the conditional-vs-anchor divergence is largest. Those are the positions where
> the oracle and θ₀ disagree. This should recover the retain utility MDU destroys without
> giving up its forget profile.

`lam_mdu` = anchor-last KL (MDU's direction), random-answer anchor (MDU's construction),
τ=1, top-k localization by `d_i = KL(p_anchor,i ‖ p_θ₀(·|x,y_t)_i)` at frozen θ₀,
weights normalized to sum to 1 so total force per sequence is unchanged from MDU's `1/|M_t|`.

The single difference from the `mdu` arm is `weighting`. No other flag moves.

## 3. Pre-registered predictions and kill rules

Measured on the generation-level six-cell partition (`eval_arm_partition.py`, 120 forget /
80 retain) and on the answer-position diagnostic, at λ=8 to match the arms already run.

| # | prediction | kill condition |
|---|---|---|
| **R1** | retain ASCR ≥ **0.40** (MDU: 0.091; θ₀: 0.762) | retain ASCR < 0.40 → localization does not protect retain and H15 is dead |
| **R2** | forget profile stays near MDU's: DISCLOSE ≤ 0.15 **and** DEGENERATE ≤ 0.25 | either exceeded → the forget profile was not preserved; the retain gain was bought by not forgetting |
| **R3** | answer-position entropy stays in **[1.3, 2.5]** nats (oracle 1.839) | outside → the arm left the oracle's shape and is in the overshoot regime with EMDU |
| **R4** | `lam_mdu_anti` (identical total force on the LOWEST-divergence positions) does **not** satisfy R1 and R2 together | if the anti-control matches, the effect is not localization and H15 is a reparameterized learning rate |

R1 and R2 must hold **together**. An arm that preserves retain by failing to forget is not a
result, and R2 exists to forbid exactly that. R4 is the attribution control and is run in the
same batch, not afterwards.

## 4. What would make this a real contribution, and what would not

Would: MDU's forget profile at materially better retain, with the anti-control failing, on a
qualified backbone with the frozen-anchor check passing — i.e. the trade-off curve moves, not a
single point.

Would not: beating MDU at a single λ. Both arms must be swept (λ ∈ {2,4,8} is already running
for `mdu`) and compared as curves, because the λ=8 comparison across the existing arms was
uninformative precisely because every arm sat at a different operating point.

## 5. Standing caveats carried forward

- `retain_ref` is an evaluation asset only. A method requiring a retain-only retrained reference
  at deployment would not transfer to real pretrained knowledge, and it is used here solely to
  locate the target, never inside any arm's loss.
- The relearning hazard is unaddressed by anything here: oracle-matching families recover under
  finetuning attack (arXiv 2506.06278), and our own study found MDLM unlearning reversible
  0.049 → 0.605 in 10 steps. No robustness claim is made.
- ATWU (arXiv 2606.06320) is SOTA on the weighting axis with a *learned* scorer and is explicitly
  composable with any forget loss. H15's divergence weighting is a free, non-learned alternative
  in the family ATWU reports beating. ATWU adapted to MDLM remains a required baseline, and the
  strongest expected configuration is ATWU's weighting on MDU's anchor.
