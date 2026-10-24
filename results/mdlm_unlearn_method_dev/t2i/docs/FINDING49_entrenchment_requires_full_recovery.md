# Finding 49 — entrenchment has a THRESHOLD: it appears only once the attack achieves FULL recovery. Partial recovery leaves the concept exactly as removable as the base model's.

**Date:** 2026-10-23 · **Arm:** `frp_church_r3_l05` (FRP port), church, lr 1e-5, seed 0, 16 images
**Evidence:** `outputs/atk_DOSE_frp_church_s*.json`, `outputs/redundancy_dose_s*_church.json`
**Scope:** ONE arm. The replication on a second arm is **running**, and this finding is deliberately
written at single-arm scope until it reports.

---

## 1. The curve

Attack budget swept over eight values; entrenchment read as target accuracy at the removal fraction
where `retain8` is nearest 0.45 (base model reference: **0.333**).

| steps | T rise | gate | H_after | **entrenchment** |
|---|---|---|---|---|
| 5 | +0.125 | **FAIL** | 0.333 | 0.333 — *= base* |
| 10 | +0.458 | PASS | 0.467 | **0.333 — = base** |
| 20 | +0.750 | PASS | 0.867 | **0.333 — = base** |
| **30** | +0.708 | PASS | **1.000** | **0.733** |
| 50 | +0.750 | PASS | 1.000 | **0.733** |
| 75 | +0.792 | PASS | 1.000 | **1.000** |
| 150 | +0.792 | PASS | 1.000 | **1.000** |
| 300 | +0.792 | PASS | 1.000 | 0.867 |

> **Entrenchment switches on exactly where recovery completes.** At 10 and 20 steps the attack is
> valid and has brought the concept most of the way back (H 0.467, 0.867) — yet the recovered
> concept is **exactly as removable as the base model's** (0.333 at both). From 30 steps, where
> H reaches 1.000, entrenchment appears and then saturates.

So entrenchment is **not** a generic byproduct of fine-tuning on a concept. It tracks *complete*
recovery, and partial recovery does not produce it at all.

## 2. ⚠ I called this hypothesis unsupported one interval ago, on an incomplete range

I raised the dose-response idea from a single observation (AdvUnlearn seed 1: weaker attack H 0.733,
weaker entrenchment 0.600), then measured budgets **30–300**, found no monotone trend
(0.733 → 1.000 → 1.000 → 0.867), and reported the hypothesis as **not supported**.

That was premature. The 30–300 range sits **entirely above the threshold**, so it could only ever
show saturation plus noise. The informative regime was 10–20 steps, which I had not yet measured.
The relation is real and is a **threshold, not a monotone gradient** — which is why looking for
monotonicity found nothing.

Two things I got right and one I got wrong, stated plainly: the non-monotonicity was correctly
observed; labelling it "noise at the 1/15 quantisation" was correct for 30–300; concluding the
hypothesis was unsupported **before measuring the low end** was the error.

## 3. What this adds to the threat model

- **Attacker cost for this arm is 20–30 steps, not 300.** H reaches 0.867 at 20 steps and 1.000 at
  30 — an order of magnitude below the budget used throughout this phase. Per `ERRATUM_r2` C3 the
  defensible phrasing is **"30 steps suffices"**, not that 30 is a minimum; 10 steps already yields
  H 0.467, and 5 fails the validity gate so measures nothing.
- **A partially-successful attack does not entrench.** A defender facing an attack that got the
  concept most of the way back (H 0.867) is still in the base model's removal regime. The harder
  problem `FINDING48` describes arrives only with complete recovery.
- **It constrains the mechanism.** `FINDING46` §3 proposed overfitting to 16 images as the cause. A
  threshold at completion fits that loosely at best — 20 steps is already substantial fine-tuning on
  16 images. Something about reaching the *ceiling* of the metric, or about the final stage of
  recovery, is doing the work. **I do not have a mechanism for the threshold**, and will not assert
  one.

## 4. Scope and the check in flight

One arm, one concept, one seed, one learning rate, one removal rule (random), entrenchment read at a
single matching level. The threshold location (between 20 and 30 steps) is bracketed by two points
and not resolved more finely.

**The second-arm replication is running** (`h68v2_rf_a0.3_church` at 10/20/50/100 steps). If the
threshold does not appear there, this finding stays a single-arm observation about the FRP port. That
check was launched **before** this document was written, which is the ordering `FINDING47` §4 and
`FINDING48` §1a both concluded should be standard — after two corrections today caused by publishing
ahead of a queued control.

Note also that `H_after` saturates at 1.000 because the metric ceiling is 15/15; a finer recovery
measure might place the threshold differently or reveal it as a steep gradient rather than a step.
