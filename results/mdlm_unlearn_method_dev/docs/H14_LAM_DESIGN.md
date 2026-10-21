# H14 — LAM: Localized Anchor Matching on a form-preserving anchor

**Status:** under test (`queue_lam2`, `queue_partition`)
**Date opened:** 2026-10-21
**Script:** `scripts/train_lam.py` · **Supersedes:** H12 (ESD), closed as anticipated
**Rests on:** `docs/FINDING13_mdu_anchor_is_degenerate.md`

---

## 1. The precise limitation addressed

Stated against primary sources, not against a general difficulty.

MDU ([2605.18253](https://arxiv.org/abs/2605.18253)) Eq. (5) and EMDU
([2610.25447](https://arxiv.org/abs/2610.25447)) Eq. (3)–(4) both minimise a KL between the
prompt-conditional prediction and a frozen prompt-masked anchor, averaged with **uniform weight
`1/|M_t|` over every masked response position** — EMDU says so explicitly ("the same `M_t`
masking and uniform averaging as MDU's forget loss"). The only knob either offers for forgetting
harder is the global temperature τ, and MDU states what τ does: τ=1 "sometimes under-forgets",
τ=0 "becomes a uniform distribution, which is exactly the same as the Maximum Entropy (ME)
loss" and "might suffer from over-forgetting".

Finding 13 measured why that knob cannot win. The all-masked anchor is **two different targets
at the two ends of the noise schedule** (23,450 positions, θ₀):

| answer still masked | `p_true` | `punct` argmax | entropy | character |
|---|---|---|---|---|
| 0.75–1.00 (early) | 0.064 | **0.422** | 4.41 | fact-free but **formless** |
| 0.00–0.25 (late) | **0.651** | 0.288 | 0.82 | well-formed but **knows the fact** |

and decoded without a scaffold it is **97.91% DEGENERATE** (335 facts), the furthest of any arm
from the pristine profile. MDU draws `t~U[0,1]` and weights all of it equally, so it averages a
target that is good-for-forgetting-bad-for-form against one that is the reverse. **The trade-off
is a property of the target, not of forgetting.** This is the same bimodality this project
measured independently: 9 of 15 baseline grid points at DISCLOSE 0.000 / DEGENERATE 1.000, the
fluent non-disclosing cell holding 4 of 960 generations.

## 2. Operation

Two changes to EMDU, each addressing one half of the measured conflict.

**(a) A form-preserving anchor.** Mask only the entity-mention span of the prompt rather than
all of it. Measured at the generation level (335 facts): DEGENERATE **0.0030** (vs 0.9791
all-masked), REPLACE **0.4597**, and the smallest TV to pristine of any arm (**0.3825**). It
keeps the relation and the syntax, so it writes a fluent plausible answer instead of
punctuation. Its cost is that it is *less* ignorant — `p_true` 0.462 vs 0.318 per position —
which (b) is there to pay for.

**(b) Localized application.** Replace the uniform `1/|M_t|` with weights from the per-position
anchor divergence that MDU's own Figure 3 already computes and interprets — verbatim: "These
high-KL positions pinpoint the exact knowledge the forget objective aims to suppress."

```
d_i = KL( p_anchor,i ‖ p_θ₀(·|x,y_t)_i )          at FROZEN θ₀
w_i = top-k over i∈M_t by d_i, value 1/k           ( Σ w_i = 1 )
L   = Σ_{i∈M_t} w_i · KL( p_anchor,i ‖ p_θ,i )     anchor-first, per EMDU Thm 2
```

Three deliberate properties:

| property | why |
|---|---|
| `w` computed at **frozen θ₀**, stop-gradiented | keeps this an ordinary fixed scalar loss rather than a sequence of surrogates. H10 was closed partly over that distinction and it is not being reintroduced quietly. `--weight-source current` exists to test the adaptive variant; it is not the default. |
| `w` **normalized to sum to 1**, exactly as `1/|M_t|` is | LAM changes only the *allocation* of force across positions, not the total per sequence. This is what makes `ent_uniform` a fair matched-force control rather than a straw man. |
| the signal is **free** | `p_anchor` is already required by the loss; one extra frozen forward gives `p_θ₀`. An MDLM scores every masked position in parallel with bidirectional context, so a per-position conditional/unconditional contrast costs O(1) forwards — not one per position. This is why WGA weights by token probability and ATWU must *learn* a scorer over hidden states. |

Anchor-first KL is adopted on EMDU's authority, not ours: their Prop. 1 gives
`∇_z L_AF = q − p` against `∇_z L_AL = q ⊙ (log(q/p) − KL(q‖p)1)`, and Thm 2 bounds the latter
by `C(p)·ε·log(V/ε)`, so at a memorized near-one-hot prediction the anchor-last gradient
vanishes and the anchor-first one does not. Direction is therefore held fixed at anchor-first
for all LAM arms and is never varied in the same comparison as the anchor or the weighting.

## 3. Arms, and the controls that can void the claim

| label | direction | anchor prompt | anchor answer | weighting | role |
|---|---|---|---|---|---|
| `mdu_pub` | anchor-last | all | random | uniform | MDU as published |
| `emdu_pub` | anchor-first | all | FAM | uniform | EMDU as published — the strongest published baseline |
| `ent_uniform` | anchor-first | **entity** | FAM | uniform | isolates (a): the anchor swap alone |
| `ent_localized` | anchor-first | **entity** | FAM | **top-k** | **LAM** — adds (b) |
| `ent_antilocalized` | anchor-first | **entity** | FAM | **bottom-k** | attribution control |

`ent_uniform` → `ent_localized` is a one-flag difference, so any gain is attributable to the
weighting. `ent_antilocalized` applies **identical total force to the lowest-divergence
positions**; if it forgets as well as `ent_localized`, the effect is not localization and LAM is
a reparameterized learning rate. Because `w` is normalized, the matched-force control is already
in the design rather than requiring a tuned λ.

## 4. Observable separating predictions

Each is falsifiable and none restates the objective.

**P1 — the degeneracy channel stays shut.** Sweeping forget strength, MDU/EMDU trace DISCLOSE↓
with DEGENERATE↑. LAM predicts DISCLOSE↓ with **DEGENERATE ≤ 0.05 throughout**, because its
anchor's own generations are 0.30% degenerate. *Refuted if LAM's sweep traces the same curve.*

**P2 — the mass lands in REPLACE.** ΔREPLACE ≥ 0.5·(−ΔDISCLOSE). The entity anchor's own
profile (REPLACE 0.4597) is where the student is being pulled. *Refuted if the mass lands in
DEGENERATE or ABSTAIN.*

**P3 — the localization signature.** `p_true` falls at **selected** (high-divergence) positions
and does **not** fall at unselected ones. Uniform weighting predicts both fall together, which
is the degeneracy channel. Instrumented directly as `p_true_selected` / `p_true_unselected` on
the author-disjoint probe. *This is the prediction that distinguishes the mechanism from a
learning-rate change, and `ent_antilocalized` must fail it.*

**P4 — EMDU is more degenerate than MDU at matched forget quality** (Finding 13 §4). FAM makes
the anchor fact-free at every noise level but formless at every noise level too. *If EMDU is not
more degenerate, the Finding 13 account is incomplete and must be revised.*

**P5 — shared-attribute retention.** At matched forget quality, LAM's retain ASCR exceeds every
baseline's, because the target is on-topic for the question rather than a global flattening.
Neither MDU nor EMDU evaluates shared-attribute retention — the "forget Alice lives in Paris,
keep Bob lives in Paris" case — and it is an evaluation gap in both.

## 5. Failure conditions, each with its observable

| # | condition | observable |
|---|---|---|
| F1 | the entity anchor is too disclosing to be a target (0.4090 vs pristine 0.030), so the student converges to a still-disclosing distribution | final DISCLOSE floor across strengths |
| F2 | top-k selection concentrates on positions that are high-divergence for reasons unrelated to the fact (rare tokens, formatting), so forgetting does not follow | `student_fact_argmax` movement vs `p_true_selected` movement |
| F3 | entity-span masking requires an entity mention: 335 of 400 facts qualify, 20/20 authors resolved at ≥0.90 question coverage — a strictly added requirement MDU does not have | eligibility fraction, reported in every run |
| F4 | the per-position result that entity ≈ control (Finding 13 §5) means the anchor swap buys nothing where the loss is actually applied, and only the generation-level difference is real | `ent_uniform` vs a control-anchor arm |
| F5 | relearning: anchor-matching families recover under finetuning attack ([2506.06278](https://arxiv.org/abs/2506.06278)), and our own study found MDLM unlearning reversible 0.049 → 0.605 in 10 steps | RTT, if the method survives the trade-off test |

## 6. What is not claimed

No training result is in hand yet. Specifically not claimed: that LAM beats MDU or EMDU; that
MDU's published numbers are wrong (§5 of Finding 13); that entity anchoring is established at
the per-position level (it is not — it matches the control there); or that the generation-level
DISCLOSE separation from the control is strong (0.110 against a 0.10 pre-registered threshold is
marginal, on one backbone, one dataset, one seed).

## 7. Novelty position

Taken, and cited: the prompt-masked frozen self-anchor with stop-gradient and CE on retain
(MDU); anchor-first KL and answer-span masking of the anchor (EMDU); token-level weighting for
AR unlearning by token probability (WGA) and by a learned scorer (ATWU
[2606.06320](https://arxiv.org/abs/2606.06320), mandatory baseline, orthogonal and composable).

Not found in any of these: **using the conditional-vs-anchor divergence as the position
weighting of the anchor-matching loss itself**, nor **a form-preserving (entity-selective) prompt
anchor**. "ablation" appears 0 times in MDU's text, and EMDU states it keeps uniform averaging.
Both gaps are recorded with the primary-source quotations supporting them, and the position is
provisional on the two appendix caveats in `docs/NOVELTY_ESD.md` §5 — now resolved for MDU and
EMDU, whose full PDFs have been read.
