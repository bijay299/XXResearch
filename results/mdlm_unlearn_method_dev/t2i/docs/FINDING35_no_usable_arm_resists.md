# Finding 35 — 10 of 11 usable T2I arms are fully reversed; the one exception could not be rebuilt

**Date:** 2026-10-23 (**substantially corrected same day** — see §0) · **Model:** SD v1.4 (pinned)
**Concepts:** church, parachute · **Protocol:** `attack_relearn.py`, prompt-disjoint T/H,
`FINDING20` gate, max-over-**valid** attacks

---

## 0. ⚠ CORRECTION NOTICE — the first version of this finding was wrong in three ways

An independent adversarial audit of this document against the raw JSONs found three errors, all
confirmed by re-derivation. They are listed before anything else because one of them was a false
published claim and the cause was **my own selection**, not a measurement.

**(1) The headline was false. I omitted the one arm that contradicts it.** The first version said
*"every one of the six arms with retain ≥ 0.78 is restored to exactly H = 1.000 — six arms, six
identical values, no exceptions"* and claimed to rest on *"every gated arm accumulated"*.

In fact **11** arms meet the stated inclusion criterion, not six. Five were missing, and one of the
missing arms **contradicts the claim**: `h47_adv_a0.3` has retain **0.8667**, a **gate-passing**
attack (`sd0`, T 0.208 → 0.458, rise +0.250), and max-over-valid H = **0.533**. I had published
that arm's numbers myself hours earlier. Excluding it is the exact failure mode this project's
record says clusters in my work — selection, not measurement.

**(2) Both correlation coefficients were wrong, and one had the wrong sign.**

| statistic | published | **correct** (tie-corrected, same table) |
|---|---|---|
| Spearman(retain, max-valid H), n=10 | −0.103 | **+0.499** |
| Spearman(retain, gate-pass rate), n=11 | +0.309 | **+0.648** |

The n's were also swapped (max-valid H is undefined for the α=0.7 arm, so that one is n=10).

**This means §2 of the first version argued against a statistic that does not exist.** It claimed
the correlation was "essentially nothing" and that the structure was therefore "a threshold, not a
correlation". With the real coefficient the relationship is a **moderate positive** one — and that
is a *stronger* result for `PREREG_H40` than the null I reported, not a weaker one. The "threshold"
framing is withdrawn: on n=10–11 convenience arms, a Spearman cannot adjudicate
threshold-versus-monotone in either direction, and presenting it as settling that question was
rationalisation.

**(3) Two of the six "independent" arms were the same model.** `gen_h47_adv_a0.0` is
**bit-identical to `gen_advunlearn_official`** across all 24 per-concept values, and
`gen_uce_church_tuned` is bit-identical to `gen_undo_a0.0_retain_fisher`. The α=0 control is a
mathematically exact no-op, so this is expected behaviour — but it means the "six identical values"
unanimity was **partly double-counted**, and `FINDING32` §3's "two independent α=0 arms agree to
four decimal places" is near-tautological rather than a validation of the measure. That check is
withdrawn; `FINDING32`'s trainability numbers themselves all re-derive exactly and stand.

---

## 1. The corrected measurement

All arms with a utility measurement and ≥ 1 gate-passing attack:

| arm | retain8 | **max-over-valid H** | valid/total |
|---|---|---|---|
| AdvUnlearn parachute (official) | 1.000 | **1.000** | 3/3 |
| adv2_church (*failed erasure, church 0.933*) | 1.000 | 1.000 | 3/3 |
| AdvUnlearn church (official) | 0.992 | **1.000** | **6/6** |
| ESD (our re-impl) | 0.992 | **1.000** | 3/3 |
| composite α=0.0 *(= AdvUnlearn, exact no-op)* | 0.992 | 1.000 | 3/3 |
| adv3_church (*failed erasure, church 0.800*) | 0.992 | 1.000 | 2/2 |
| parachute composite α=0.3 | 0.933 | *attacks in flight* | — |
| **composite α=0.3** | **0.867** | **0.533** | **1/3** ← the exception |
| UCE tuned (our re-impl) | 0.850 | **1.000** | 3/3 |
| undo α=0.0 *(= UCE teacher, exact no-op)* | 0.850 | 1.000 | 3/3 |
| composite α=0.3 (rebuild) | 0.783 | **1.000** | 4/6 |
| — | | | |
| composite α=0.5 | 0.558 | 0.267 *(under re-test)* | 1/3 |
| damage+distil α=0.3 (UCE teacher) | 0.458 | **1.000** | 3/3 |
| UCE at matched utility | 0.350 | **1.000** | 4/4 |
| damage+distil α=0.5 (UCE teacher) | 0.317 | 0.467 | **3/10** |
| damage+distil α=0.7 (UCE teacher) | 0.000 | *no valid attack* | 0/3 |

> **Corrected claim: of 11 arms with retain ≥ 0.78 and a valid attack, 10 are restored to
> H = 1.000. The single exception sits at retain 0.867 with H 0.533 — and it could not be
> reproduced.**

**Note on the metric's ceiling, which the first version traded on improperly:** H is 3 prompts × 5
images = 15 samples quantised to 1/15, with the base model at the 1.000 ceiling. For any arm the
attack fully reverses, 1.000 is the *only attainable value*. The unanimity of the 1.000s is
therefore arithmetically forced, not an observed coincidence, and the first version's emphasis on
"exactly 1.000, no variance at all" was rhetoric.

## 2. The exception, and why it is a reproducibility failure rather than a retraction

`H48` was run to verify the 0.533 and it reported H = 1.000 — which I published as a retraction of
the 0.533. **The audit shows that inference was confounded.** The 0.533 was measured on
`h47_adv_a0.3` (retain **0.867**); every H48 attack, including the "different seed at the same lr"
that reached 0.933, ran against `h48_adv_a0.3` — a **rebuilt and materially different arm** at
retain **0.783**. Arm, seed and learning rate all changed together, so "the 0.533 was an artifact of
seeds and learning rate" is unsupported.

The correct statement is narrower and less convenient: **the arm that produced 0.533 could not be
rebuilt** (the construction was non-deterministic until the fix in `08742c1`; the same config and
seed gave retain 0.867 vs 0.783). Its datapoint therefore stands as the sole observation of
non-trivial resistance at usable utility, resting on **one valid attack of three**, and unreproduced.

## 3. What the corrected numbers support

- **Spearman(retain, max-valid H) = +0.499.** Higher-utility arms are recovered *more* completely.
  This is the direction `PREREG_H40` predicts and is stronger evidence for it than the null first
  reported.
- **Spearman(retain, gate-pass rate) = +0.648.** Higher-utility arms are substantially *easier to
  attack validly* — `FINDING32`'s trainability effect seen from the other side. The arm admitting
  no valid attack at all has retain 0.000.
- **Low utility does not buy resistance.** `uce_matched` at retain 0.350 and damage+distil α=0.3 at
  0.458 are both fully reversed. Resistance appears only in **damaged** arms, inconsistently.
- **Resistance and usability have been observed together exactly once**, unreproduced, at 1 valid
  attack of 3 — not "never", as the first version implied, and not reliably either.

## 4. ⚠ A methodological problem that limits this finding and PREREG_H40 itself

The audit's sharpest point, and I accept it. **`max-over-valid-attacks` is monotone non-decreasing
in the number of attacks run, and no stopping rule was ever declared.** The attack counts are
asymmetric in exactly the damaging direction:

| arm type | attacks run |
|---|---|
| candidate arms I hoped would resist (α=0.3 composite, α=0.5 damage+distil) | **9–10** |
| already-reversed high-retain arms | 2–6 |

Combined with the rule I stated in `PREREG_H47` — *"a robustness number measured at one learning
rate with fewer than ~5 valid attacks has never survived"* — this is a ratchet: keep strengthening
the adversary against candidates until the number breaks, while leaving the already-broken arms
alone. **Applied that way, `H40` cannot be refuted by any amount of evidence, which makes it
unfalsifiable in practice regardless of its wording.**

This is not corrected by reanalysis. It requires a **pre-declared, symmetric, fixed attack budget
per arm**, applied identically to candidates and baselines, before the next arm is measured. That
is now a precondition on any further robustness claim in this project, including `H51`'s.

## 5. Other corrections from the same audit

- **Attack counts were inconsistent across documents** for the same experiment: AdvUnlearn-church
  is reported as 6/6 (`FINDING34`), 4/4 (this file, first version), and 2/2 (`PREREG_H47`). **6/6
  is correct**; the 4/4 dropped the two UNet+text-encoder attacks.
- **`3/7` for damage+distil α=0.5 was wrong; it is `3/10`** (`T2I_REPORT` had it right). The
  inflated pass rate fed the wrong Spearman.
- **`adapters/uce_church/unet.pt` was overwritten** between a retain-0.350 build and a retain-0.850
  build, so `T2I_REPORT`'s "5/5" for UCE tuned pooled attacks on two different models. The correct
  count is **3/3**, and "UCE and ESD likewise, 8 of 8" should read **6 of 6**.
- **"four published erasure methods" is three** (AdvUnlearn, ESD, UCE). RECE and Receler were never
  started.
- `clip_align` **cannot confirm erasure depth**: an arm that generates nothing (α=0.7, every concept
  at 0.000) shows a *smaller* align drop (−17%) than AdvUnlearn (−42%), and UCE at church 0.067
  shows the same −40% as AdvUnlearn at church 0.400. `FINDING34`'s "the 42% drop confirms the
  erasure is real" does not follow and is withdrawn.
- `FINDING34`'s "it erases more than our ESD" rests on 0.400 vs 0.467 — a **one-image** difference
  on a 15-image metric. Withdrawn as a comparative ranking.

## 6. Scope, unchanged

One model, two concepts, one attack family, our own protocol and its three documented gate failure
modes. The 12-way closed-set CLIP metric is coarse and top-saturated. Arm construction was
non-deterministic until `08742c1`. The three competing defences for this threat model
(Meta-Unlearning 2410.12777, IMMA 2311.18815, GIFT 2507.13598) remain unrun, so this bounds what we
tested, not the field. And the threat-model claim itself is **anticipated** — Meta-Unlearning and
`Erased or Dormant?` (2505.16174) established it first.
