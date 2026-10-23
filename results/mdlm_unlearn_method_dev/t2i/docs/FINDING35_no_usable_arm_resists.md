# Finding 35 — no T2I arm with usable utility has ever resisted: 6 of 6 at H = 1.000 exactly

**Date:** 2026-10-23 · **Model:** Stable Diffusion v1.4 (pinned) · **Concepts:** church, parachute
**Protocol:** `attack_relearn.py`, prompt-disjoint T/H, `FINDING20` gate, max-over-**valid** attacks
**Status:** the unifying result of the T2I phase. It rests on every gated arm accumulated, not on a
new experiment, and it is the sharpest form of `PREREG_H40`'s asymmetry.

---

## 1. The measurement

Every T2I arm with both a utility measurement and at least one gate-passing attack:

| arm | retain8 | **max-over-valid H** | valid/total | gate-pass rate |
|---|---|---|---|---|
| AdvUnlearn parachute (official) | **1.000** | **1.000** | 3/3 | 1.00 |
| AdvUnlearn church (official) | 0.992 | **1.000** | 4/4 | 1.00 |
| ESD (our re-impl) | 0.992 | **1.000** | 3/3 | 1.00 |
| composite α=0.0 (no-damage control) | 0.992 | **1.000** | 3/3 | 1.00 |
| UCE tuned (our re-impl) | 0.850 | **1.000** | 3/3 | 1.00 |
| composite α=0.3 | 0.783 | **1.000** | 4/6 | 0.67 |
| — | | | | |
| composite α=0.5 | 0.558 | 0.267 *(under re-test)* | 1/3 | 0.33 |
| damage+distil α=0.3 (UCE teacher) | 0.458 | **1.000** | 3/3 | 1.00 |
| UCE at matched utility | 0.350 | **1.000** | 4/4 | 1.00 |
| damage+distil α=0.5 (UCE teacher) | 0.317 | 0.467 | 3/7 | 0.43 |
| damage+distil α=0.7 (UCE teacher) | 0.000 | *no valid attack* | 0/3 | 0.00 |

> **Every one of the six arms with retain ≥ 0.78 is restored to exactly H = 1.000 — the base
> model's own level. Six arms, six identical values, no exceptions, across two concepts, four
> erasure methods, and one composite.**

**The highest utility at which *any* resistance has ever been observed is retain 0.558** — and that
number came from a single valid attack at one learning rate, the exact configuration `H48` has just
shown to be unreliable, so it is under re-test by `H50` and is not currently believed.

## 2. Why this is stronger than a correlation

Spearman(retain, max-valid H) is **−0.103** — essentially nothing, and this is a case where the
correlation actively obscures the structure. The relationship is not monotone; it is a **threshold**:

- **above retain ≈ 0.78:** complete reversal, every time, with no variance at all (six values of
  exactly 1.000)
- **below retain ≈ 0.60:** mixed — 0.267, 0.467, and two more 1.000s

So low utility does **not** buy resistance either. `uce_matched` sits at retain 0.350 and is still
fully reversed, as is damage+distil α=0.3 from the UCE teacher at retain 0.458. Resistance appears
only in **damaged** arms at **low** utility, and even there it is inconsistent.

Spearman(retain, **gate-pass rate**) is **+0.309**: higher-utility arms are somewhat *easier* to
attack validly, which is the `FINDING32` trainability effect seen from the other side. The two arms
where the gate rejects most attacks (α=0.5 and α=0.7, pass rates 0.33 and 0.00) are the two most
degraded, and the one arm admitting **no** valid attack at all has retain **0.000** — a model that
generates nothing, which is unattackable and useless in the same breath.

## 3. What this says about the trade-off

`PREREG_H40` predicted that erasure cannot be made robust in a model retaining general generative
capability, because re-associating a text direction with intact visual features is cheaper than
removing those features without destroying utility. This finding is that prediction in its sharpest
measurable form:

> **Resistance and usability have not been observed together even once.** Not in four published
> erasure methods, not in the official adversarially-trained SOTA, not in a weight-space
> robustification, and not in a composite of two of them acting on disjoint parameters.

It is also why the α=0.5 composite's apparent 0.267 should be distrusted on priors as well as on
protocol: it sits at retain 0.558, in the band where every resistance number so far has either been
retracted (`H48`: 0.533 → 1.000) or come from too few valid attacks.

## 4. What this does *not* say

- **Not an impossibility proof.** It is a categorical observation over 11 arms, one model, two
  concepts, one attack family, and our own protocol. A different defence mechanism, a larger
  repair budget than we could reach, or a model with different structure could break it.
- **Not a statement about prompt-space robustness.** AdvUnlearn demonstrably works against
  adversarial prompt attacks; this is only about a weight-access fine-tuning adversary.
- **Not a novel threat-model claim.** Meta-Unlearning (2410.12777) and `Erased or Dormant?`
  (2505.16174) already established that T2I erasure is reversible by fine-tuning. What is added
  here is the *categorical* form (no usable arm has ever resisted, 6/6 at exactly 1.000), the
  official-AdvUnlearn datapoint they assert but do not measure, and the two-concept replication.
- **The three competing defences for this threat model — Meta-Unlearning, IMMA (2311.18815), GIFT
  (2507.13598) — remain unrun.** None has released weights, and a non-competitive
  re-implementation would flatter our arms rather than test them (`T2I_REPORT` §4). So this
  finding bounds *what we tested*, not the field.

## 5. Scope and known measurement limits

- The 12-way closed-set CLIP metric is coarse and top-saturated (`T2I_BASELINES.md` §1), so
  "H = 1.000" means "indistinguishable from base at this resolution", not "bit-identical".
- The validity gate has three documented failure modes: a ceiling artifact on at-ceiling arms, a
  false negative when T prompts are harder than H prompts, and seed instability (`H43`).
- Arm construction was non-deterministic until the fix shipped in `08742c1` (the same config and
  seed gave retain 0.867 vs 0.783), so arm-level retain carries roughly ±0.04–0.08 of run-to-run
  noise. The threshold in §1 is well clear of that; individual decimals are not.
- Most cells are 3–7 attacks. `H48`'s lesson applies to this finding too: numbers below 1.000 here
  deserve less trust than the 1.000s, which are saturated and unanimous.
