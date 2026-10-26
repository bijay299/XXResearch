# FINDING 76 — At one attack budget (300 steps, lr 1e-5), **all four tested unroll depths K ∈ {0, 1, 8, 32} reached full measured recovery**: 12/12 gate-valid, every attack at H = 1.0000. The run has **no adequate mechanism check**, so this is an outcome at one budget, not a result about depth.

**Date:** 2026-10-26 (rewritten 2026-10-26 by Agent 4 under `AUDIT_REPAIR_REGISTER_v1.md` D-005)
**Experiment:** E4m builds + E4mg grid · **Preregistered:** `PREREG_H90` (§1–§3), with the §8
compute-accounting amendment applied **before** these arms were built.
**Evidence:** `outputs/atk_E4MG_*.json` (12), `outputs/utrap2_utrap2*_church_l0.15k*.json` (4),
`outputs/gen_utrap2_church_l0.15k0.json`, `outputs/gen_utrap2m_church_l0.15k{1,8,32}.json`,
`logs/e4m_matched.log`, `logs/e4mg_grid.log`, `run_e4m_matched.sh`.

> **This document supersedes the version that headlined "unroll depth buys nothing" and claimed a
> passing mechanism check.** Both claims are withdrawn. Every number below was re-verified against
> the JSON artifacts named above; §9 lists exactly what changed and why.

---

## 1. What was measured

`erase_utrap_v2.py`, concept church, teacher `frp_church_r3_l05`, λ = 0.15, outer lr 1e-6.
Attacks: `attack_relearn.py`, 300 steps, lr 1e-5, seeds {0, 1, 6}, gate T_rise ≥ 0.20 absolute.

| arm | K | outer steps | samples | as-is target | retain8 (n=8) | all-non-target (n=11) | gate-valid | per-seed H_after |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `utrap2_church_l0.15k0` | **0** | 400 | 1600 | **0.0000** | 0.8667 | 0.8727 | 3/3 | 1.0000, 1.0000, 1.0000 |
| `utrap2m_church_l0.15k1` | 1 | 320 | 1600 | 0.1333 | 0.9667 | 0.9697 | 3/3 | 1.0000, 1.0000, 1.0000 |
| `utrap2m_church_l0.15k8` | 8 | 133 | 1596 | 0.3333 | 0.9667 | 0.9697 | 3/3 | 1.0000, 1.0000, 1.0000 |
| `utrap2m_church_l0.15k32` | 32 | 44 | 1584 | 0.2667 | 0.9500 | 0.9455 | 3/3 | 1.0000, 1.0000, 1.0000 |

Per-attack detail (all twelve, verified from `atk_E4MG_*.json`):

| K | seed | T_before → T_after (T_rise) | valid | H_before → H_after |
|---:|---:|---|---:|---|
| 0 | 0 / 1 / 6 | 0.0000 → 1.0000 (1.0000) ×3 | ✅✅✅ | 0.0000 → 1.0000 ×3 |
| 1 | 0 | (T_rise 0.8750) | ✅ | 0.1333 → 1.0000 |
| 1 | 1 | (T_rise 0.8333) | ✅ | 0.0667 → 1.0000 |
| 1 | 6 | (T_rise 0.9167) | ✅ | 0.0000 → 1.0000 |
| 8 | 0 | (T_rise 0.7083) | ✅ | 0.3333 → 1.0000 |
| 8 | 1 | (T_rise 0.6667) | ✅ | 0.4000 → 1.0000 |
| 8 | 6 | (T_rise 0.7500) | ✅ | 0.4667 → 1.0000 |
| 32 | 0 | (T_rise 0.8333) | ✅ | 0.2667 → 1.0000 |
| 32 | 1 | (T_rise 0.7500) | ✅ | 0.2000 → 1.0000 |
| 32 | 6 | (T_rise 0.6250) | ✅ | 0.3333 → 1.0000 |

### The claim this supports, stated at its actual scope

**All four tested settings reached full measured recovery at the specified attack budget.** Twelve
of twelve attacks cleared the validity gate and every one landed at H_after = 1.0000.

`PREREG_H90`'s §1 prediction — *"max-over-valid H is non-increasing in K, and at least one K reaches
max-over-valid H < 1.000"* — **did not occur** at this budget. Its second half is contradicted:
no K reached max-over-valid H < 1.000. Its first half is vacuous here, because a flat series at the
ceiling is trivially non-increasing.

### What this does **not** support

- **Not "unroll depth buys nothing."** Four settings, one concept, one teacher, one λ, one defence
  build per K, and **one attack budget**. "Buys nothing" is a claim about all budgets; this
  measured one.
- **Nothing about ordering.** H_after = 1.0000 is the **ceiling** of a metric quantised to 1/15
  (15 images: 3 held-out DEV prompts × 5 images). Four arms pinned at the ceiling carry **zero**
  information about their relative ordering. A saturated metric cannot rank.
- **Nothing about depth as a mechanism**, because the mechanism check is inadequate (§3).

---

## 2. What this run fixed relative to `FINDING63`

`FINDING63` §3a(a) announced a similar conclusion and was **withdrawn**: its sweep varied the
horizon of a surrogate that differed from the evaluation attacker on six other axes
(`ERRATUM_utrap_inner_attacker.md`). Two of those defects are genuinely fixed here.

| defect | `FINDING63` | here |
|---|---|---|
| inner attacker | SGD, rebuilt per step, batch 2, no clipping, teacher-generated images, one **evaluation** prompt | **Adam, persistent, batch 1, clip 1.0, base-model images, DEF_TRAIN prompts** — the evaluator's own configuration, via `attack_core.py` |
| compute accounting | claimed matched at K+1; actually **1600 → 468 samples**, falling with K (verified: the unmatched `utrap2_*` builds are 400/200/45/13 outer steps → 1600/1000/540/468 samples) | **sample-matched to ≈1600: 1600 / 1600 / 1596 / 1584** (see §5 for what "matched" does and does not mean) |
| attack seeds | single build, attacks at n = 1–3 | **n = 3 at every cell** |
| K = 0 control | absent | **present** |

These are real improvements. They do not add a mechanism check, and `FINDING63` did not have one
either.

---

## 3. The mechanism check — **absent, not passed**

`PREREG_H90` §1 declared a **second falsifier**: *"if the inner-attacker probe delta does not grow
with K … no conclusion about depth may be drawn from the run at all."*

The earlier version of this document presented three mean probe deltas (−0.00124, −0.00269,
−0.00638), stated that "the delta grows with K, roughly in proportion", and concluded that the
second falsifier "did not fire" and that "the mechanism check passed". **All of that is
withdrawn.** The raw measurements follow in §4. Three reasons the check is not adequate:

1. **It is not a test of what the design assumes.** The design assumes the inner loop
   **approximates the evaluation attacker**. The probe measures only that the fixed-noise target
   loss **moves** during the unroll. A deeper unroll takes more gradient steps, so a larger
   movement is close to tautological — it is consistent with the inner loop approximating the
   attacker and equally consistent with it drifting in an unrelated direction.
2. **The samples are not replicates.** The per-K probe counts are **8 / 7 / 3 / 1** and they are
   serial observations **inside a single training run**, not independent draws. A mean over them is
   a mean over a non-stationary time series (§4 shows K=1's series is non-monotone), so it carries
   no dispersion that an ordering claim could be tested against.
3. **The counts are confounded with the thing being compared.** `--probe-every 50` against
   different outer-step counts is why n differs; the matched-compute design *causes* the unequal n.

**An adequate check would need to be defined and is not in this run.** At minimum it would have to
(a) compare the inner loop's parameter update against the evaluation attacker's update over the
same data from the same point — a *direction/overlap* measurement, not a loss-movement
measurement; (b) take n ≫ 1 independent measurements per K; and (c) be measured at **matched points
in training** across arms. None of (a), (b), (c) holds here.

**Consequence, per the prereg's own wording:** no conclusion about **depth** may be drawn from this
run. §1's result stands as an **empirical outcome at one budget** and nothing more.

**The one part of §3 that does survive:** the K=0 control is **exactly 0.00000 on all 8 probes**,
which confirms `ABLATION_K0` is a true no-op. That is verified and retained.

---

## 4. The raw probe measurements, with **where in training** each was taken

`--probe-every 50` (`run_e4m_matched.sh`), so probes land on outer steps 0, 50, 100, … The step
column is the part the earlier version omitted, and it is the decisive one.

| arm | K | outer steps | probes **n** | probe at outer step → delta |
|---|---:|---:|---:|---|
| `utrap2_church_l0.15k0` | 0 | 400 | **8** | 0 → 0.00000 · 50 → 0.00000 · 100 → 0.00000 · 150 → 0.00000 · 200 → 0.00000 · 250 → 0.00000 · 300 → 0.00000 · 350 → 0.00000 |
| `utrap2m_church_l0.15k1` | 1 | 320 | **7** | 0 → −0.00093 · 50 → −0.00020 · 100 → −0.00011 · 150 → −0.00044 · 200 → −0.00018 · 250 → −0.00222 · 300 → **−0.00457** |
| `utrap2m_church_l0.15k8` | 8 | 133 | **3** | 0 → −0.00320 · 50 → −0.00219 · 100 → −0.00268 |
| `utrap2m_church_l0.15k32` | 32 | 44 | **1** | 0 → **−0.00638** |

Three things this table makes visible that the means concealed:

**(a) The arms were not probed over comparable stretches of training.** K=1 was probed across
outer steps 0–300; K=8 across 0–100; **K=32 only at outer step 0**, i.e. before a single outer
update was applied. Any between-arm comparison of the means therefore compares *early-training
K=32* against *all-of-training K=1*. The probe counts and the training coverage are confounded
together.

**(b) K=1's series is non-monotone and grows late.** −0.00093 → −0.00020 → −0.00011 → −0.00044 →
−0.00018 → −0.00222 → −0.00457. Its largest value is its **last** one. The mean −0.00124 is not a
central estimate of a stable quantity; the quantity drifts within the run.

**(c) At n = 1, what is determinable is only the measurement itself.** For K=32 the sole datum is:
*the single probe taken at outer step 0 of the single K=32 build measured −0.00638.* No dispersion,
no interval, no comparison test, and no mean distinguishable from the datum. That is the whole of
what n = 1 supports.

### The non-overlap, corrected

The previous §7 asserted that K=1's range **"overlaps"** K=32's point. **That is false.**
K=1's deltas span **[−0.00457, −0.00011]**. K=32's point is **−0.00638**, which is *more negative
than K=1's extreme* and therefore lies **outside** K=1's observed range. The two are **disjoint**.

The correct statement is the opposite of the one that was made — and it still establishes no
ordering, for the reason in §4(e) below.

### The only like-for-like slice available

One probe per arm was taken at the **same** point in training (outer step 0), so this is the single
comparison not confounded by training coverage. Each cell is **n = 1**:

| K | probe at outer step 0 |
|---:|---:|
| 0 | 0.00000 |
| 1 | −0.00093 |
| 8 | −0.00320 |
| 32 | −0.00638 |

These four values happen to be monotone in K. **This is not evidence of an established ordering.**
With n = 1 per cell there is no variance estimate, no test, and no basis for excluding that a
re-run would reorder them; and the pattern is close to tautological, since more inner gradient
steps from the same starting point should move the loss further whether or not the inner loop
resembles the evaluation attacker. It is reported because it is the only matched slice in the data,
and it is labelled for exactly what it is: four single measurements.

### (e) No ordering is inferred from, or denied by, a range ratio

The previous §7 argued from the ratio of K=1's within-arm spread (≈40×, −0.00011 to −0.00457) to
the between-arm differences, and concluded "**no ordering survives that**". **That inference is
also withdrawn.** A within-arm range ratio is not a test statistic: a wide range does not establish
that an ordering is absent, any more than a narrow one would establish that it is present. The
spread in question is a drift over training (§4b), not sampling noise around a mean, so it does not
license either conclusion.

**What is actually determinable:** at n = 1 for K=32 — and with K=1, K=8 probed over different and
unmatched stretches of training — the ordering of probe deltas across K is **undetermined**. Not
"suggestive", not "absent". Undetermined. The data cannot adjudicate it either way.

---

## 5. Three different replication counts, kept separate

The earlier version used "n = 3", "n = 7/3/1" and "one build per K" in ways that ran together.
They are three distinct and non-substitutable things, so they get three tables. **Pooling records
that vary different things into one table does not merely lose information — it can invert a
conclusion** (see `AUDIT_REPAIR_REGISTER_v1.md` D-012, where de-pooling two campaigns from one
table retracted a mandate claim). Each table below states what was varied and what was held fixed,
and carries its own denominator.

### 5a. Defence-build replication — **n = 1 per K**

*Varied:* nothing. *Held fixed:* everything — concept, teacher, λ, outer lr, sample budget, and the
single RNG seed / data order of the defence training run.

| K | builds | build labels |
|---:|---:|---|
| 0 | **1** | `utrap2_church_l0.15k0` |
| 1 | **1** | `utrap2m_church_l0.15k1` |
| 8 | **1** | `utrap2m_church_l0.15k8` |
| 32 | **1** | `utrap2m_church_l0.15k32` |

**Denominator: 1 build per cell.** Supports **nothing** about build-to-build variability. A second
build per K is **unrun**. Every per-K number in §1 and §7 rests on a single defence training run,
so no per-K quantity here has an error bar of any kind.

### 5b. Attack-seed replication — **n = 3 per build**

*Varied:* the attack seed ∈ {0, 1, 6}. *Held fixed:* the defence build (one per K), attack steps
(300), attack lr (1e-5), n_images (16), gate (0.20).

| K | build | attack seeds | gate-valid | H_after |
|---:|---|---|---:|---|
| 0 | `utrap2_church_l0.15k0` | 0, 1, 6 | 3/3 | 1.0000, 1.0000, 1.0000 |
| 1 | `utrap2m_church_l0.15k1` | 0, 1, 6 | 3/3 | 1.0000, 1.0000, 1.0000 |
| 8 | `utrap2m_church_l0.15k8` | 0, 1, 6 | 3/3 | 1.0000, 1.0000, 1.0000 |
| 32 | `utrap2m_church_l0.15k32` | 0, 1, 6 | 3/3 | 1.0000, 1.0000, 1.0000 |

**Denominator: 3 attacks per build, 12 total.** Supports a statement about the **attacker's**
variability *conditional on that one build*. This — and only this — is what "12/12 at 1.0000"
measures. **It is not defence-build replication.** Twelve attacks at the ceiling say the attacker
reliably reverses *these four specific builds*; they say nothing about whether a differently-seeded
K=32 build would behave the same, because no such build exists.

### 5c. Within-training probes — **n = 8 / 7 / 3 / 1**

*Varied:* the outer step at which the probe is taken, **inside one build**. *Held fixed:* the
build, and therefore everything else.

| K | build | probes | outer steps probed |
|---:|---|---:|---|
| 0 | `utrap2_church_l0.15k0` | **8** | 0, 50, 100, 150, 200, 250, 300, 350 |
| 1 | `utrap2m_church_l0.15k1` | **7** | 0, 50, 100, 150, 200, 250, 300 |
| 8 | `utrap2m_church_l0.15k8` | **3** | 0, 50, 100 |
| 32 | `utrap2m_church_l0.15k32` | **1** | 0 |

**Denominator: not a replication count at all.** These are serial, non-independent observations of
a drifting quantity within a single run (§4b), and the counts differ *because* the arms differ in
outer steps. They cannot be averaged as if they were independent draws, and the per-K counts cannot
be compared as if they were sample sizes of the same thing. Any between-K comparison of these
numbers is additionally confounded by the arms having been probed over different stretches of
training (§4a).

**Why the separation matters here specifically:** the withdrawn §3 claim read a between-arm trend
off three *means* of 5c, implicitly treating 5c's counts as 5b-style replication. They are not the
same quantity, and the probe counts are not independent of K. That substitution is what made a
confounded time series look like a passing mechanism check.

---

## 6. "Sample-matched" — what the matching actually is

The matching is a **count of training samples consumed by the defence objective**, computed as
`outer_steps × (4 + K)`: four samples per outer step for the objective's own batches, plus one per
inner step. Verified:

| K | outer steps | 4 + K | samples |
|---:|---:|---:|---:|
| 0 | 400 | 4 | **1600** |
| 1 | 320 | 5 | **1600** |
| 8 | 133 | 12 | **1596** |
| 32 | 44 | 36 | **1584** |

Three things this does **not** mean:

1. **It is not exact.** 1600 / 1600 / 1596 / 1584 spans a **16-sample (1.0%)** range; integer outer
   steps cannot hit 1600 exactly at every K.
2. **It excludes probe overhead, which is unequal and inversely related to K.** The probes are
   forward-only and are **not** in the counts above: **8** probes at K=0, 7 at K=1, 3 at K=8, **1**
   at K=32 (§4). Each probe is 2 forward passes (`probe_before`, `probe_after`), so K=0 carries ~16
   extra forward passes against K=32's ~2. This residual is declared, not argued away — but it
   means the arms did **not** consume equal compute.
3. **No claim of wall-clock, FLOP, or memory equality is made or supported.** A K=32 unroll and a
   K=0 step are not equal-cost operations at equal sample count; backward-pass structure, optimizer
   state traffic and activation memory all differ. "Sample-matched" is a matched **sample budget**
   and nothing broader.

So §2's row should be read as: the sample budget was matched to within 1%, which is a large
improvement on `FINDING63`'s 1600 → 468, and is not compute equality.

---

## 7. The K = 0 control — informative, but it does **not** dominate

With K = 0 the objective degenerates to `L_retain(θ) − λ·L_target(θ)`: erasure with **no
adversarial anticipation whatever**. It is the most useful arm in the grid, and the previous
version mis-stated what it showed.

**Verified from `outputs/gen_utrap2_church_l0.15k0.json` and `outputs/gen_utrap2m_church_l0.15k8.json`:**

| | K = 0 | K = 8 | which is better |
|---|---:|---:|---|
| as-is target suppression (lower = better) | **0.0000** | 0.3333 | **K = 0** |
| retain8 (n = 8, higher = better) | **0.8667** | **0.9667** | **K = 8** |
| all-non-target utility (n = 11) | 0.8727 | 0.9697 | **K = 8** |
| post-attack H (3 seeds) | 1.0000 ×3 | 1.0000 ×3 | tie (both at ceiling) |

**K = 0 has better target suppression *and* lower retained utility. It does not dominate K = 8 —
the two trade off.** The previous version's reading — that K=0 "achieves the best as-is suppression
of the four" and therefore "the unrolling does not merely fail to buy robustness, it **costs**
suppression" — took the suppression column alone and omitted the utility column that moves the
other way. Corrected:

**Both differences are well above measurement granularity.** The image denominators (verified from
`gen_eval.py`: 3 prompts × `--n-per-prompt` 5 = **15 images per concept**, and the per-concept
accuracies are then averaged):

- **Target suppression**: 15 images, step 1/15. The gap is 0.0000 vs 0.3333 = **0/15 vs 5/15 —
  5 images**, five quantisation steps. **K = 0 is better by 5 images.**
- **retain8**: 8 concepts × 15 = **120 images**, step **1/120 = 0.00833** (*not* 1/8 — "n=8"
  counts concepts, not images). The gap is 0.8667 vs 0.9667 = **104/120 vs 116/120 — 12 images**,
  twelve quantisation steps. **K = 8 is better by 12 images.**
- **all-non-target**: 11 × 15 = **165 images**, step **1/165 = 0.00606**. The gap is 144/165 vs
  160/165 = **16 images**. Same direction, same conclusion.
- Both arms are at the **ceiling** post-attack, so neither is more robust by any measurement here.
- **The utility reference is 1.000, not something lower.** The unmodified base model scores
  `clip_acc = 1.0000` on **all twelve** DEV concepts (`outputs/gen_base_sd14.json`, independently
  reconfirmed by `outputs/gen_E6cal_BASE_sd14_on_DEV.json`). So retained utility is a **loss**
  against a clean 120/120 reference: **K = 0 gives up 16 of 120 images, K = 8 gives up 4.**

So this is a **genuine, well-resolved two-axis trade-off**, not a near-tie and not a domination:
K = 0 buys 5 images of suppression and pays 12 images of retained utility. The earlier version
reported only the first half.

> **Erratum within this rewrite (2026-10-26).** An intermediate draft of this section asserted the
> retain8 gap was "less than one image", from a mistaken 1/8 step. That was wrong by 15×: retain8
> averages eight **15-image** concepts, so its step is 1/120 and the gap is **12 images**. The
> mistaken figure was anti-conservative in the dismissive direction — it would have licensed
> discarding the largest utility difference in the grid as quantisation noise. Corrected above and
> in §8. No other section depended on it.

**What the K=0 control legitimately shows:** plain erasure, at the same sample budget, reached a
*different* and not obviously worse pre-attack operating point, and was reversed to the same
ceiling. Spending the budget on anticipating an adversary did not buy a better pre-attack point on
suppression, and did not buy any post-attack robustness that this budget can detect. It is not
established that it bought nothing on utility; the utility gap runs the other way and is about one
image wide.

---

## 8. Limits

- **One concept (church), one teacher, one λ (0.15), one attack budget (300 steps / lr 1e-5), one
  defence build per K.** See §5.
- **A saturated ceiling cannot rank depths.** All twelve attacks land at 1.0000. This says *depth
  did not prevent reaching the ceiling at this budget* — not that depth has no value at any budget
  (`ERRATUM_interpretations_r5.md` §1).
- **Quantisation, with the correct image denominators** (all verified from `attack_relearn.py` and
  `gen_eval.py`, not from any prior document):

  | metric | images | step | where |
  |---|---:|---:|---|
  | **H** (recovery) | 3 H-prompts × `eval_n` 5 = **15** | **1/15 = 0.0667** | §1 |
  | **T** (the gate's statistic) | 8 T-prompts × n_per 3 = **24** | **1/24 = 0.0417** | §1 |
  | **retain8** | 8 concepts × 15 = **120** | **1/120 = 0.00833** | §1, §7 |
  | **all-non-target** | 11 concepts × 15 = **165** | **1/165 = 0.00606** | §1, §7 |

  The "n=8" and "n=11" labels count **concepts, not images**; the image denominators are 120 and
  165. §7 applies these.
- **The gate threshold is not representable on its own grid.** `--gate 0.20` is tested as
  `T_after − T_before >= 0.20` against a 24-image T, and 0.20 × 24 = 4.8 — so the operative gate is
  **≥ 5/24 = 0.2083**, with the adjacent failing value 4/24 = 0.1667. All twelve T_rise values here
  are 0.625–1.0000, i.e. 15–24 images, far from the boundary, so **no verdict in this finding turns
  on that rounding**. It is recorded because it is undocumented project-wide.
- **No mechanism check** (§3). This is the governing limit: per `PREREG_H90`'s own second
  falsifier, no depth conclusion is licensed by this run.
- **Recovery-versus-budget curves per K are unmeasured.** See §8a — they, not the probe, are the
  measurement this finding most needs.
- **The trajectory is still detached.** A fully differentiated unroll remains untested and is not
  claimed either way.
- **The λ=1.2 arm** (`PREREG_H90` §7 amendment) is built but is not in this grid.
- **DEV prompts throughout** (`ERRATUM_split_leak.md`). `"a photo of a church"` — H[0] — is a
  defence-training prompt for the UTRAP family, so H is a development measurement, not held out.
- **The K=0 arm is from the `utrap2_*` build family, the other three from `utrap2m_*`.** They were
  produced by different runner scripts (`run_e4_h90_utrap_v2.sh` vs `run_e4m_matched.sh`); K=0's
  400 outer steps already sat at the 1600-sample budget, so it was not rebuilt. The arm names in
  §1 reflect this, and it is a difference in provenance, not only in K.

### 8a. Two measurements that would actually advance this, and what each can and cannot do

**Post-hoc probes (queued).** The probe can be re-run on a frozen checkpoint at any K — load,
run K inner steps, measure the fixed-noise target loss before and after — giving n ≫ 1 per K
cheaply. **But these would be NEW diagnostics about the checkpoints they are run on.** They
**cannot retrospectively validate the training trajectories** that produced these four builds: the
quantity of interest during training is how the inner loop behaved *at each outer step, from that
step's parameters*, under that run's optimizer and data order. Those states are gone. Post-hoc
probing answers "what does a K-step inner loop do from *here*", which is a different question from
"what did the inner loop do *during* the run", and it can neither confirm nor refute §1's four
builds. It is worth running on its own terms; it is not a repair of this run.

**Recovery-versus-budget curves (unrun, and the higher-value measurement).** Sweeping the attack
budget per K and recording where H crosses each level measures **empirical recovery speed
directly** — how many gradient steps each depth costs the attacker. This is **independent of any
mechanism probe**: it does not require knowing whether the inner loop approximates the evaluation
attacker, because it measures the outcome that matters without routing through the mechanism. It
also escapes the ceiling problem that makes §1 unrankable, since a curve has resolution where a
saturated point does not. Precedent exists in this corpus (`FINDING50`, `FINDING69`,
`FINDING74`). If one measurement is to be run on this question, it is this one, not the probe.

---

## 9. What changed in this rewrite

| § | previous claim | status |
|---|---|---|
| title / §1 | "unroll depth buys nothing" | **withdrawn.** Replaced with the observed outcome: all four tested settings reached full measured recovery at the specified attack budget. |
| title / §3 | "The mechanism check passed, so the null is interpretable" | **withdrawn.** No adequate check exists; §3 defines what one would require and records its absence. |
| §3 | "the delta grows with K, roughly in proportion"; `PREREG_H90`'s second falsifier "did not fire" | **withdrawn.** Undetermined. |
| §3 | three means quoted without n, without raw values, without probe locations | **corrected.** §4 gives n = 8/7/3/1, every raw delta, and the outer step each was taken at. |
| old §7 | K=1's range "**overlaps**" K=32's point | **corrected — the claim was false.** K=1 spans [−0.00457, −0.00011]; K=32's −0.00638 lies outside it. They are **disjoint**. |
| old §7 | "No ordering survives that" (argued from a 40× within-arm range ratio) | **withdrawn.** A range ratio cannot establish the absence of an ordering any more than its presence. The ordering is **undetermined** (§4e). |
| §1 / §3 / §5 | "n = 3", "n = 7/3/1" and "one build per K" used interchangeably | **separated.** §5 distinguishes defence-build replication (n=1), attack seeds (n=3), and within-training probes (n=8/7/3/1). |
| §2 | "sample-matched" stated as compute matching | **qualified.** §6: matched to within 1% on a sample count; probe overhead unequal (8 vs 1 probes) and excluded; no wall-clock/FLOP equality claimed. |
| §4 (old) | K=0 "achieves the best as-is suppression" → unrolling "**costs** suppression" | **corrected.** §7: K=0 has better suppression (0.0000 vs 0.3333 = **5 of 15 images**) **and lower** retained utility (0.8667 vs 0.9667 = **12 of 120 images**). It **does not dominate**; it is a well-resolved two-axis trade-off. |
| §7 / §8 | (intermediate draft of this rewrite) retain8 step stated as 1/8, so the utility gap read as "less than one image" | **corrected.** retain8 averages eight **15-image** concepts → step **1/120**; the gap is **12 images**, not sub-step. Wrong by 15× and anti-conservative. §8 now tabulates all four true denominators (H 1/15, T 1/24, retain8 1/120, 11-set 1/165). |
| — | gate stated only as "0.20 absolute" | **qualified.** §8: on a 24-image T, 0.20 is unreachable; the operative gate is ≥ 5/24. No verdict here turns on it. |
| old §7 | post-hoc probes framed as "the proper measurement", implying they would validate §1 | **corrected.** §8a: they are new diagnostics on frozen checkpoints and cannot validate the training trajectories. |
| — | recovery curves listed only as "unmeasured" | **promoted.** §8a: they measure recovery speed independently of any mechanism probe, and are the higher-value unrun measurement. |

Unchanged and re-verified: §1's table values, all twelve gate outcomes and H values, the
K=0 probe being exactly 0.00000 on all 8 probes, and §2's `FINDING63` comparison including the
1600 → 468 sample figure.

---

## 10. Standing of UTRAP

UTRAP is **our proposed method**. Authorship establishes neither novelty nor effectiveness. The
record shows it reversed at every setting tested on SD-1.4 (`FINDING63`), on SD-1.5 and BK-SDM
(`FINDING64`), on held-out prompts (`FINDING66`), and — with the inner attacker reconciled to the
evaluation attacker and the sample budget matched to within 1% — **at all four unroll depths
tested here, including K = 0**, at this one attack budget. That is a consistent and unflattering
pattern across many settings. It is not a demonstration that no unroll depth could ever help at
any budget, and this document does not claim one.
