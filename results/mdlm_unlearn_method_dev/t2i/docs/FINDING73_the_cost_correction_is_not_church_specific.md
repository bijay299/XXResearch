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
