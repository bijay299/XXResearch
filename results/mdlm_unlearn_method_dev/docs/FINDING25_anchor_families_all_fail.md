# Finding 25 — every anchor-based MDLM unlearning method fails a validated relearning attack

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM with an **exact** retrain oracle (`FINDING21`)
**Scripts:** `train_small_unlearn.py`, `train_small_rtt.py`
**Status:** the clearest result in the project. A uniform negative, replicated, with ground truth.

---

## 1. The measurement

Every published MDLM unlearning family we could implement, attacked under one protocol: RTT with
an author-disjoint T/H split, recovery measured on held-out forget authors the attacker never
supplied, at the **adversarially-optimal** lr 3e-5, with **every attack passing `FINDING20`'s
validity gate**.

| family | forget target source | retain | **recovery** |
|---|---|---|---|
| MDU λ=0.5 | model's own distribution, prompt-masked anchor | 0.6733 ± 0.00 | **0.5788 ± 0.02** |
| AFC λ=1 | same, anchor-first KL | 0.6992 ± 0.02 | **0.6274 ± 0.01** |
| GA λ=0.5 | gradient ascent on the forget span | 0.6417 ± 0.01 | **0.6752 ± 0.05** |
| EMDU λ=1 | same, full answer masking | 0.7433 ± 0.01 | **0.7183 ± 0.04** |
| MDU+FAM λ=1 | same | 0.7275 *(n=1)* | **0.7263** *(n=1)* |
| **UNDO plain α=0.5** | **damage + distillation** | 0.3650 ± 0.01 | **0.0513 ± 0.02** |
| **UNDO retain-Fisher α=0.5** | **damage + distillation** | 0.4567 ± 0.03 | **0.0498 ± 0.08** |

n=3 seeds except where noted; 13 attack runs across the anchor/GA families.

> **The two bands do not overlap.** Anchor and gradient-ascent methods recover **0.5608–0.7538**
> of the forgotten knowledge. Damage+distillation recovers **0.0498–0.0513**. The separation is
> **10.9×** at the closest point, and the within-family standard deviations are 0.01–0.05.

## 2. Why this is the two walls, confirmed from the other side

`FINDING18` and `FINDING19` argued — from measurements on a different backbone, at 65× the scale —
that a forget target drawn from the model's own distributions cannot work: form and fact-identity
are the same information (output space), and fluency-preserving activation targets carry no
signal (activation space). Their joint prediction was that a working method must source its target
from **outside** those distributions.

Every family in the top block of §1 draws its target from the model's own distribution on a
modified input. **All five fail.** The only family that resists is the only one that sources its
target elsewhere. The prediction was made before any of these runs and is not retrofitted.

## 3. And every failing method looks successful behaviourally

| family | TV→oracle | DEGENERATE | recovery |
|---|---|---|---|
| MDU | 0.0725 | 0.0000 | 0.5788 |
| AFC | 0.0750–0.0900 | 0.0000 | 0.6274 |
| EMDU / MDU+FAM | 0.3225 | 0.0325–0.1200 | 0.7183 / 0.7263 |
| UNDO (both) | 0.0850 | 0.0000 | 0.0513 / 0.0498 |

MDU at TV 0.0725 is **behaviourally closer to the exact oracle than UNDO is** (0.0850) and
recovers **11× more knowledge**. A study ranking these methods on behavioural metrics — which is
what the MDLM unlearning literature reports — would rank MDU above UNDO and be exactly wrong
about which one removed the knowledge.

This is `FINDING23`'s suppression–removal gap in its sharpest form: at α=0.5, TV cannot separate
six methods at all (0.072–0.100, DEG ≈ 0) while the ranking probe separates them by 0.10 and the
attack separates them by 11×.

## 3a. The λ test: pushing harder makes it WORSE, not better

§4 lists "λ swept over only {0.5, 1, 2}" as the finding's main weakness — if a stronger forget
weight bought robustness, the finding would be wrong. Tested at λ ∈ {4, 8}, n=3 seeds, same gated
protocol:

| arm | retain | **recovery** | TV→oracle |
|---|---|---|---|
| MDU λ=0.5 | 0.6733 ± 0.00 | 0.5788 ± 0.02 | 0.0725 |
| MDU λ=4 | 0.6533 ± 0.03 | **0.6442 ± 0.01** | 0.0783 |
| MDU λ=8 | 0.5417 ± 0.03 | **0.6966 ± 0.01** | 0.0767 |
| AFC λ=1 | 0.6992 ± 0.02 | 0.6274 ± 0.01 | 0.0750 |
| AFC λ=4 | 0.6642 ± 0.05 | **0.6541 ± 0.02** | 0.0775 |
| AFC λ=8 | 0.5717 ± 0.06 | **0.6869 ± 0.02** | 0.0758 |
| MDU+FAM λ=1 | 0.7333 ± 0.01 | 0.7411 ± 0.02 | 0.3225 |
| **UNDO retain-Fisher α=0.5** | 0.4567 ± 0.03 | **0.0498 ± 0.08** | 0.0850 |

> **Recovery rises with λ for MDU, AFC and GA while retain falls.** MDU 0.579 → 0.644 → 0.697 as
> retain drops 0.673 → 0.542; AFC 0.627 → 0.654 → 0.687 as retain drops 0.699 → 0.572; GA
> 0.675 → 0.749 → 0.800.

⚠ **CORRECTION — the monotonicity is family-specific, not general.** An earlier version of this
section said recovery rises with λ for anchor objectives as a class. **EMDU is a counterexample**:
0.718 → 0.697 → 0.662 as λ goes 1 → 4 → 8, i.e. the opposite direction. So "harder forgetting is
less robust" holds for the prompt-masked anchor (MDU, AFC) and for gradient ascent, and **not** for
full-answer masking. Generalising from the first three families would have been an overclaim.

What is unaffected: no λ in {0.5…8} rescues any of these families — the knob that looks like "forget more" produces *more
recoverable* suppression. The weakness §4 named has been tested and it does not save them; the
band is now **0.5608–0.7568 over 27 gated runs**, still with no overlap against UNDO's
0.0498–0.0513.

`mdu_fam` is now n=3 (0.7411 ± 0.02), so no cell in this finding rests on a single seed.

Note the TV column: every λ leaves TV at 0.0725–0.0783. **Behavioural distance to the oracle is
flat across a λ range that moves recovery by 0.12.** The behavioural metric is not merely
insensitive to the knowledge question — it is insensitive to a 2× change in how much an attacker
can recover.

## 4. What this does and does not license

**Supports:**
- Reporting a relearning attack is not optional for MDLM unlearning. Behavioural metrics alone
  cannot distinguish a method that removes knowledge from one that hides it.
- The two walls generalise across backbone and scale, from 8B to 124M.
- Damage+distillation is, on this evidence, the only tested family that survives.

**Does not support:**
- **Not a SOTA claim.** MDU/EMDU/AFC were re-implemented by us on our backbone; the original
  authors' tuning is not reproduced, and λ was swept over only {0.5, 1, 2}. A fairer statement is
  *"our implementations of these families, at the λ we could afford to sweep, are all
  non-robust"*.
- **One corpus, one task shape, one attack family.** Four attributes, closed answer pools, 200
  authors, SFT-style relearning. A different attack could order these differently.
- The λ weakness named above is now **tested and closed** (§3a): λ ∈ {0.5…8} across all five
  families, 37 gated runs, band 0.5608–0.8642. Higher λ makes robustness worse for MDU/AFC/GA and
  *better* for EMDU — family-specific, as §3a now records.
- **One scale.** 124M. The 8B programme could not test UNDO at all for feasibility reasons.
- ~~`mdu_fam` is n=1~~ — now n=3 (0.7411 ± 0.02).

## 5. The honest size of it

This is a **negative** result about five published method families plus a confirmation of a sixth
that is not ours. It does not deliver the method the project set out to find. What it does deliver
is a reason to distrust the field's standard evaluation, measured against a true retrain reference
rather than argued — and the walls that predicted it.
