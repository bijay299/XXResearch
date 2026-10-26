> ### ⚠ §6 AMENDS THIS FINDING'S EMPHASIS (E11k, same day)
>
> A **third** concept, `tench`, is the **slowest** of the three, not the fastest. So "generalises
> **and gets worse**" was an over-read from the one extra concept I happened to test — which turned
> out to be the easiest. What generalises is that **all three saturate far below 300 steps**; the
> budget needed **varies substantially by concept**. See §6.

# FINDING 73 — The budget correction **generalises to a second concept and gets worse**. At a matched method and a matched learning rate, FRP on `garbage_truck` reaches the **ceiling on all three seeds in 25 steps**, where FRP on `church` is still at 0.8667 — and it does so from a *deeper* starting suppression.

**Date:** 2026-10-26 · **Experiment:** E11j · **n = 3 at every cell, held until all cells returned**
**Closes** the "one concept (church)" limit named in `FINDING68`, `FINDING69`, `FINDING70` **and**
`FINDING71`.
**Prediction recorded before the run** (`run_e11j_second_concept.sh`): *"if garbage_truck needs
materially more budget than church did, then the ≤100-step cost figure is concept-specific and
FINDING68/70's cost claims must be narrowed to church."*
**Evidence:** `outputs/atk_E11J_*.json`, `logs/e11j_second_concept.log`

---

## 1. The result

`frp_garbage_truck_r5_l05` (target suppression 0.0000, retain8 0.7750), lr 1e-5, n = 3:

| steps | valid | mean H | **max-over-valid** | post-attack utility |
|---:|---:|---:|---:|---:|
| 10 | 3/3 | 0.7556 | 0.8667 | 0.9097 |
| **25** | 3/3 | **1.0000** | **1.0000** | 0.8681 |
| 100 | 3/3 | 1.0000 | 1.0000 | 0.8750 |

**At 25 steps the concept is fully recovered on all three seeds** — mean 1.0000, so sd = 0.

## 2. The matched-method comparison, which is the defensible one

`FINDING72` §1 established that comparing *different* methods confounds the result. So the
comparison that matters here holds the method fixed — **FRP on both concepts, lr 1e-5, n = 3:**

| arm | concept | H_before | 10 steps (mean / max) | **25 steps (mean / max)** |
|---|---|---:|---|---|
| `frp_church_r3_l05` | church | **0.1333** | 0.5000 / 0.5333 *(2/3 valid)* | 0.8667 / 0.9333 |
| `frp_garbage_truck_r5_l05` | garbage_truck | **0.0000** | 0.7556 / 0.8667 *(3/3)* | **1.0000 / 1.0000** |

**The garbage_truck arm starts from a *deeper* suppression (0.0000 vs 0.1333) and is nonetheless
recovered faster and completely.** So within one defence family, deeper pre-attack erasure bought
the attacker nothing — if anything the reverse.

**The prediction's falsifier did not fire.** garbage_truck did not need more budget than church; it
needed less. So `FINDING68` and `FINDING70`'s cost corrections **do not narrow to church** — they
hold across two concepts and, on this evidence, understate the attacker on the second one.

## 3. The cheapest reproducible full recovery in the project, restated

| arm | concept | lr | steps | result |
|---|---|---:|---:|---|
| `utrap_church_l0.15k8` | church | 5e-5 | 25 | H = 1.0000, 3/3 seeds, sd 0 (`FINDING70`) |
| **`frp_garbage_truck_r5_l05`** | garbage_truck | **1e-5** | **25** | **H = 1.0000, 3/3 seeds, sd 0** |

**Same step count at a five-times lower learning rate.** Against this project's published
"300 gradient steps at lr 1e-5", the garbage_truck figure is a **12× overstatement measured at the
very same learning rate** — no lr substitution needed to get there.

## 4. Utility: no damage, consistent with `FINDING71`

Post-attack utility stays **0.8681–0.9097** across all three budgets, in the band `FINDING71`
identified for lr 1e-5 (corpus mean 0.9284, n = 169). So the attacker recovers the concept
completely and leaves the measured non-target panel intact — the same pattern as church, and further
evidence that the damage regime is a property of **high lr**, not of budget.

## 5. Limits

- **Two concepts, one defence family in the matched comparison** (FRP), three seeds. E11k is running
  a **third** concept (`frp_tench_r3_l05`, target 0.1333 / retain8 0.7917) in the same family.
- **The arms differ in retain8** — 0.7750 for garbage_truck against 0.9417 for FRP-church. A
  less-retentive arm may simply sit closer to a state the attacker can exploit, and that is not
  separated here. §2's claim is therefore about *these two arms*, not about the concepts in the
  abstract.
- `FINDING72` §2's gate bias applies: FRP-church's 2/3 validity at 10 steps means its mean is over
  two runs, garbage_truck's over three.
- **H quantised to 1/15**; the 10-step gap (0.7556 vs 0.5000) is ≈ 3.8 images, the largest effect
  in this finding and the only one comfortably above quantisation.
- **DEV prompts** throughout (`ERRATUM_split_leak.md`); utility is the 48-image closed-set CLIP
  panel, not an image-quality measure.
- **No robustness verdict changes.** Both arms were already known to reverse to the ceiling at 300
  steps; this says only that the ceiling arrives much sooner, on a second concept too.

## 6. Amendment (E11k) — a third concept is the **slowest**, so "gets worse" was an over-read. Concept is a substantial cost axis in its own right.

`frp_tench_r3_l05` (target 0.1333 / retain8 0.7917), same FRP family, same lr 1e-5, n = 3:

| steps | valid | mean H | max-over-valid | utility |
|---:|---:|---:|---:|---:|
| 10 | **2/3** | 0.4667 | 0.5333 | **0.9861** |
| 25 | 3/3 | **0.6889** | **0.7333** | **0.9861** |

**All three concepts, matched method (FRP) where possible, lr 1e-5, 25 steps, n = 3:**

| arm | concept | H_before | mean H | **max-over-valid** | utility |
|---|---|---:|---:|---:|---:|
| `frp_tench_r3_l05` | tench | 0.1333 | **0.6889** | **0.7333** | 0.9861 |
| `frp_church_r3_l05` | church | 0.1333 | 0.8667 | 0.9333 | 0.9722 |
| `frp_garbage_truck_r5_l05` | garbage_truck | 0.0000 | **1.0000** | **1.0000** | 0.8681 |

**Ordering: tench < church < garbage_truck**, spanning **0.6889 to 1.0000** in mean H at identical
method, learning rate and budget. **tench has not reached the ceiling at 25 steps** (max 0.7333),
where garbage_truck is fully recovered on every seed.

**What this amends.** §2's matched-method comparison stands as a fact about those two arms, and §3's
"25 steps at lr 1e-5" result for garbage_truck stands. What is withdrawn is the **generalising
direction**: I wrote that the correction "generalises **and gets worse**" on the strength of one
additional concept, and that concept turns out to be the **easiest** of the three. With tench in
hand the defensible statement is narrower and more useful:

> **All three concepts saturate far below the reported 300 steps, but the budget required varies
> substantially by concept — roughly a 0.31 spread in mean H at 25 steps.**

So `FINDING68`/`FINDING70`'s cost corrections are **not church-specific** (the original point, which
holds), but neither is any single number from them transportable to a new concept.

**Concept is now the largest between-condition cost spread the project has measured** that is *not*
explained by the lr-damage mechanism of `FINDING71` — note tench's utility is **0.9861**, the highest
of any arm here, so nothing about its slower recovery is a damage artifact. It is also consistent
with `FINDING72` §2's gate bias: tench shares church's `H_before` of 0.1333 and shows the same 2/3
validity at 10 steps.

**E11l is running tench at 50 and 100 steps** to find where it does saturate; until then "tench is
slower" is bounded only below.

### 6a. Process

This is the third consecutive cycle in which the new measurement amended the previous one
(`FINDING71` narrowed 70, `FINDING72` corrected 68 §6, this narrows 73). In each case the amendment
came from a replication or extension **I chose to run against my own claim**, which is the intended
behaviour — but it also means `FINDING68`–`FINDING73` should be read as one young, self-correcting
block, and the generalising clauses in each are the parts most likely to move next.

## 7. Completion (E11l) — three concepts, full curves at n = 3. **Every concept is fully reversed within 100 steps at lr 1e-5.** Concept varies the required budget by ~4×, not by an order of magnitude.

`frp_tench_r3_l05`, lr 1e-5, complete:

| steps | valid | mean H | max-over-valid | utility | per-seed H |
|---:|---:|---:|---:|---:|---|
| 10 | **2/3** | 0.4667 | 0.5333 | 0.9861 | 0.4000, *0.6667*, 0.5333 |
| 25 | 3/3 | 0.6889 | 0.7333 | 0.9861 | 0.7333, 0.6667, 0.6667 |
| **50** | 3/3 | 0.9556 | **1.0000** | 0.9722 | 1.0000, 1.0000, 0.8667 |
| **100** | 3/3 | **1.0000** | **1.0000** | 0.9722 | 1.0000, 1.0000, 1.0000 |

**The three-concept table, matched method (FRP) and lr (1e-5), n = 3:**

| concept | arm | H_before | max-over-valid first = 1.000 | **all three seeds = 1.000** | utility there |
|---|---|---:|---:|---:|---:|
| garbage_truck | `frp_garbage_truck_r5_l05` | 0.0000 | **25** | **25** | 0.8681 |
| tench | `frp_tench_r3_l05` | 0.1333 | **50** | **100** | 0.9722 |
| church | `frp_church_r3_l05` | 0.1333 | 50 *(n = 1)* | — | 0.9722 |

### 7a. What the completed series supports

1. **Every concept tested is fully reversed within 100 steps at lr 1e-5**, against this project's
   reported **300**. The cost correction of `FINDING68`/`FINDING70` holds on **three** concepts and
   is not church-specific — §6's amendment stands, and the ordering it reported is confirmed with
   tench's curve complete.
2. **Concept varies the budget by ~4× for reproducible recovery** (25 steps for garbage_truck,
   100 for tench) and ~2× for max-over-valid (25 vs 50). So concept matters, and it matters *less*
   than the 12× the 300-step figure was wrong by. "Slowest concept" means 100 steps, not 300.
3. **tench is the cleanest recovery measured anywhere in this project**: complete restoration on all
   three seeds while post-attack utility stays **0.9722–0.9861**, the highest band of any arm. Full
   concept recovery with essentially no collateral damage, on a concept the defence had suppressed
   to 0.1333.
4. **Every cell across all three concepts fails the mandate's robustness half.** There is no
   (concept, budget) combination at which any arm reaches max-over-valid H ≤ 0.40 while holding
   retain8 ≥ 0.85.

### 7b. What it still does not support

- **One defence family in the matched comparison** (FRP), **one learning rate**, **three seeds**,
  **three concepts**, and the arms still differ in `H_before` (0.0000 vs 0.1333) and retain8
  (0.7750 / 0.7917 / 0.9417). Concept, arm quality and starting suppression remain **confounded**;
  separating them needs builds matched on pre-attack suppression, which is disk-blocked.
- **The concept ordering is not explained.** Why garbage_truck is 4× cheaper than tench is unknown;
  frequency in the pretraining distribution, visual distinctiveness and the CLIP classifier's own
  per-class behaviour are all candidates and none is tested.
- **The classifier is the measuring instrument for all of this.** A 12-label closed-set CLIP argmax
  may simply be more willing to call an image a garbage truck than a tench. That alone could produce
  the ordering, and it is **not** controlled for. This is the single most likely alternative
  explanation and it needs an independent detector to rule out (`ERRATUM_interpretations_r5.md` §8).
- **DEV prompts** throughout.

**E11m is running a fourth concept** (`frp_parachute_r5_l05`, target 0.2000 / retain8 0.7417) to see
whether the ~4× spread holds or widens.
