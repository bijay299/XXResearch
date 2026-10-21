# Finding 14 — H15 fails R1: position localization works, but the damage leaks through parameters, not positions

**Date:** 2026-10-21
**Status:** H15 as specified is **DEAD** by its own pre-registered rule. The mechanism is
verified and the diagnosis is new; the retain axis is being swept as the stated completion.
**Attempts:** `p124` (train), `p125` (anti-control), `p126` (partition)
**Pre-registration:** `docs/PREREG_H15_lam_mdu.md`, fixed before any arm ran
**Outputs:** `adapters/h15_lammdu_l8/report.json`, `outputs/arm_h15_lammdu_l8.json`

---

## 1. Verdict against the rules as registered

| rule | threshold | measured | |
|---|---|---|---|
| R2 forget profile preserved | DISCLOSE ≤ 0.15 **and** DEGENERATE ≤ 0.25 | DISC **0.0667**, DEG **0.2000** | **PASS** |
| R3 stay on the oracle's shape | entropy ∈ [1.3, 2.5] nats | **2.313** | **PASS** |
| **R1 retain recovered** | retain ASCR ≥ **0.40** | **0.199** | **FAIL** |

R1 and R2 were required to hold **together**. They do not. H15 is closed as specified.

## 2. What nevertheless improved, stated precisely

At **identical forget quality** — DISCLOSE 0.0667 for both arms, the same number to four
decimals — localization more than doubles retain:

| arm | DISC | DEG | ABST | REP | IGN | ASCR forget | **ASCR retain** | TV→pristine |
|---|---|---|---|---|---|---|---|---|
| θ₀ | 0.792 | 0.008 | 0.000 | 0.092 | 0.108 | 0.717 | **0.762** | 0.770 |
| MDU (uniform `1/|M_t|`) | 0.067 | 0.150 | 0.000 | 0.392 | 0.392 | 0.140 | **0.091** | 0.191 |
| **H15 (localized)** | **0.067** | 0.200 | 0.000 | 0.342 | 0.392 | 0.113 | **0.199** | 0.241 |

So the trade-off does move — 2.2× retain at matched forget — but 0.199 against θ₀'s 0.762 is
not a usable model, and the threshold was set at 0.40 precisely so that a partial improvement
could not be reported as success.

## 3. The localization mechanism is verified, and that is what makes the failure informative

Measured on the author-disjoint frozen probe, with the anchor confirmed bit-stable
(`anchor_p_true` 0.41921 → 0.41921, PASS):

| quantity | θ₀ → final | change |
|---|---|---|
| `p_true` at **selected** (high-divergence) positions | 0.637 → 0.335 | **−0.302** |
| `p_true` at **unselected** positions | 0.595 → 0.610 | **+0.015** |

The loss moved the positions it was applied to and left the others untouched — cleaner than the
entity arms, where the ratio was 0.723 vs 0.244 and *both* fell. Position-level selectivity is
therefore not the bottleneck: it works, and it works almost perfectly.

## 3a. R4 — the attribution control passes decisively, so the selector is validated

`lam_mdu_anti` (`p125`) is identical to H15 in every respect except that its normalized weights
are placed on the **lowest**-divergence positions. Same total force, same anchor, same τ, same
direction, same split, same steps.

| quantity | H15 (high-divergence) | anti-control (low-divergence) |
|---|---|---|
| `p_true` selected | 0.637 → **0.335** (−0.302) | 0.716 → **0.756** (**+0.039**) |
| `p_true` unselected | 0.595 → 0.610 | 0.569 → 0.725 |
| `p_true` overall | 0.606 → 0.541 | 0.606 → **0.733** (**rises**) |
| entropy | 1.103 → 2.313 | 1.103 → **0.853** (**sharpens**) |
| `student_fact_argmax` | 0.087 → 0.075 | 0.087 → 0.083 |

The anti-control does not merely forget less — it does not forget **at all**. Its answer
probability *rises* and its entropy *falls*, moving away from the anchor rather than toward it.

**This is the first validated component this project has produced.** Stated as a finding in its
own right, independent of H15's failure:

> The conditional-vs-anchor KL evaluated at frozen θ₀ identifies the answer positions whose
> suppression actually reduces disclosure. Its complement carries no forgettable content:
> placing the entire loss budget there produces no forgetting under any instrument measured.

That is a reusable selection criterion, it costs one extra frozen forward pass, it requires no
learned scorer and no annotation, and it is available in an MDLM because every masked position
is scored in parallel under both the conditional and the prompt-masked context. Whether it beats
ATWU's *learned* selector (arXiv 2606.06320) is a separate, open question with the head-to-head
already implemented in `scripts/train_atwu.py`.

## 4. The diagnosis

Within forget sequences, unselected positions are preserved to within 0.015. Yet the **retain
split** degraded badly:

```
H15 retain profile:  DISCLOSE 0.188  DEGENERATE 0.263  REPLACE 0.263  IGNORANT 0.288
```

Retain queries are different *inputs*, not different positions. So:

> **The damage does not transfer across positions — it transfers across inputs.** It leaks
> through the shared LoRA parameters, which position selection cannot reach. Selecting *where
> in a forget sequence* to apply the loss bounds the forget-side footprint; it does nothing to
> bound the function change induced on inputs the loss never sees.

This also reinterprets MDU's collapse. Finding 13 attributed its retain destruction (ASCR
0.091) to uniform weighting spending ~87% of its force where the oracle and θ₀ agree. That
account is **incomplete**: removing that waste recovers only 0.091 → 0.199, so most of MDU's
retain damage is *not* mis-targeted position force either. It is the same parameter-space leak.

## 5. The stated completion, not a change of goalposts

`λ_retain` was held at **1.0** in every arm run so far; only `λ_forget` was swept. Retain is
the axis that failed, so it must be swept before the mechanism can be called dead at all
operating points. `scripts/queue_retain.sh` runs `λ_retain ∈ {4, 16}` with `λ_forget ∈ {4, 8}`,
**and the matched `mdu` arm at `λ_retain = 16`** so the comparison stays like-for-like rather
than crediting localization with a gain that a stronger retain term gives both arms.

R1, R2 and R3 keep their pre-registered values. If H15 passes only at a λ_retain where MDU
also passes, there is no contribution and that will be recorded.

If the sweep does not lift retain ASCR past 0.40, the conclusion is that **no reweighting of an
anchor-matching loss over answer positions can protect retain**, and the next candidate must
act in parameter or function space on retain inputs rather than on the forget set's support.
That is a different class of intervention and would be registered separately.

## 6. Caveats

- One backbone (`theta0_tofu`), one dataset (TOFU forget10-style split), one seed, 120 forget /
  80 retain probes. No multi-seed claim.
- R4 is decided at the loss level (§3a) and passes. Its generation-level partition is still
  running; until it reports, the claim "the anti-control cannot satisfy R1 and R2 together" is
  supported by its training instruments (no forgetting on any of them) rather than by its
  six-cell profile.
- `retain_ref` is an evaluation asset only and appears in no arm's loss.
- ATWU (arXiv 2606.06320) adapted to MDLM is implemented and queued but has not run, so no
  claim is made against the state of the art on the weighting axis.
