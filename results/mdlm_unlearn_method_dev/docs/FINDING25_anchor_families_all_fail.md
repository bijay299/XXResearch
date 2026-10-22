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
- **One scale.** 124M. The 8B programme could not test UNDO at all for feasibility reasons.
- **`mdu_fam` is n=1** and is reported as such.

## 5. The honest size of it

This is a **negative** result about five published method families plus a confirmation of a sixth
that is not ours. It does not deliver the method the project set out to find. What it does deliver
is a reason to distrust the field's standard evaluation, measured against a true retrain reference
rather than argued — and the walls that predicted it.
