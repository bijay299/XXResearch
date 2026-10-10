# Amendment 01 — the fixed early grid at steps 10…90: results

**Detector-based and PROVISIONAL.** The blinded human annotation is a
prerequisite for any final scientific conclusion. No labels exist yet; the
packet is built with every label cell empty and its key stored outside it.

Execution commit `0cd9dc611b737e32774c631930ddf5a4199edc27`. Output root
`/data/bijaypandey/cuig_pilot/seq_pilot/diag_v2_early_grid`. The original run
(`diag_v1`) is untouched. Protocol:
[`AMENDMENT_01_EARLY_GRID.md`](../../docs/research/AMENDMENT_01_EARLY_GRID.md).

---

## 1. The hypothesis this tested, and the answer

The original 100-step grid found no dump within 5 pp of its seed's L2 endpoint
on either seed: the measured points **straddled** the target (0 pp at step 0,
38.75/42.50 pp at step 100). Straddling does **not** guarantee a matching
discrete checkpoint, so this amendment **tested** whether some step on the fixed
grid 10, 20, …, 90 lands inside the band.

**It does, on both seeds, at step 70 — on the development set.** On the frozen
test set the matching survived for seed 17 (1.25 pp) but **not** for seed 29
(6.25 pp), which makes seed 29's matched comparison inconclusive. See §7.1.

| seed | fixed L2 target | step-70 suppression | mismatch | within 5 pp |
|---|---|---|---|---|
| 17 | 30.00 pp (eligible, exactly on the gate) | 31.25 pp | **1.25 pp** | yes |
| 29 | 33.75 pp (eligible) | 31.25 pp | **2.50 pp** | yes |

Granularity is 1.25 pp (one detection in 80 dog pairs).

---

## 2. Training: the two declared changes, and nothing else

Each seed's trajectory was retrained from its **saved MA** (bound by digest
`3cb72ee162b9…` s17, `fbf49199a306…` s29) for **100 optimizer steps** with a
dump **every 10**. Both reached `optimizer_steps_completed = 100`, and both
wrote **10/10 dumps**, each validating against the checkpoint contract with all
digests distinct.

Every setting that determines the first 100 updates was preserved and
**verified from the reports**, not assumed; the learning-rate schedule is
`constant`, which diffusers builds without reference to the horizon, so the
shortened stopping limit cannot move the learning rate at any early step. The
declared contract was enforced by value: seeds 17/29, 100 steps, cadence 10,
bridge step 100, tolerance 5 pp, same parent digest on both runs.

---

## 3. The bridge at step 100: HELD

The one step both runs write, compared three ways with the stop condition
declared in advance.

| seed | original step-100 | rerun step-100 | difference | ≤ 5 pp |
|---|---|---|---|---|
| 17 | 38.75 pp | 37.50 pp | **1.25 pp** | yes |
| 29 | 42.50 pp | 43.75 pp | **1.25 pp** | yes |

Weights at step 100 are **not** bitwise identical (0/32 tensors; max abs
3.297e-04 s17, 3.340e-04 s29; relative Frobenius ~1.2e-03). Two things follow,
and only these two:

* the early dumps are a **second realisation**, not points on the original run's
  path, so they are never mixed with the original run's dumps in the match;
* **the cause of the difference is not established.** This comparison cannot
  separate nondeterministic kernels from library, driver or upstream-code state,
  from the changed save cadence, or from dataloader ordering. No controlled
  repeat was run and none is proposed.

Equally, **identical step-100 weights would not have proved identical earlier
states**: one shared later point is not a shared path, and the candidates come
from the unmeasured interior at steps 1–99.

---

## 4. The reference target could not move

MA and MAB_L2 were **not regenerated**. They were read read-only from the
original run's verified development evaluations and bound four ways before
anything was selected (`reference_bindings.json`, `ok: true`):

| check | result |
|---|---|
| generating checkpoint == the registered saved artifact | 4/4 slots |
| manifest == the frozen development manifest `1cd1f902…` | 4/4 slots |
| detections and image-report digests == the published `development_slots_index.json` | 4/4 slots |
| measured L2 suppression == the declared target | 30.00 pp (s17), 33.75 pp (s29) |

The development scan therefore scored **20 slots — the ten dumps per seed only**
and wrote no reference evaluation beside the candidates.

---

## 5. Selection on the development set

Dog-only, 80 pairs per checkpoint, t=0.5, against each seed's own fixed MA
evaluation. Gates ≥30 pp suppression and ≤60% residue, applied to the candidates
**and to the L2 reference**; both references were eligible (seed 17 exactly on
the suppression gate).

| step | seed 17 suppression (gates) | seed 29 suppression (gates) |
|---|---|---|
| 10 | 7.50 pp (fail) | 10.00 pp (fail) |
| 20 | 10.00 pp (fail) | 16.25 pp (fail) |
| 30 | 16.25 pp (fail) | 20.00 pp (fail) |
| 40 | 21.25 pp (fail) | 26.25 pp (fail) |
| 50 | 26.25 pp (fail) | 30.00 pp (PASS) |
| 60 | 28.75 pp (fail) | 28.75 pp (fail) |
| **70** | **31.25 pp (PASS) — SELECTED** | **31.25 pp (PASS) — SELECTED** |
| 80 | 35.00 pp (PASS) | 38.75 pp (PASS) |
| 90 | 36.25 pp (PASS) | 40.00 pp (PASS) |

Both seeds `MATCHED`; step 70 is the qualifying dump closest to its seed's L2
level. Two dumps lay inside the band for seed 17 and three for seed 29; the
prespecified ambiguity rule (separated dumps inside the band with a ≥10 pp
residue spread) was evaluated and **not** triggered — the spread was 8.75 pp for
seed 29.

**Worth stating plainly:** suppression is **not monotone** in steps on seed 29
(30.00 pp at step 50, 28.75 pp at step 60, 31.25 pp at step 70). The selection
rule is indifferent to that, but it is a reminder that these are noisy discrete
measurements at 1.25 pp granularity, not a smooth curve.

Every development input the decision read is recorded with its sha256
(11 per seed), and the decision was **re-bound** to those bytes immediately
before the frozen test was touched.

---

## 6. Gates that had to pass before the frozen test was evaluated

All three were recomputed at that moment, not read from disk:

1. the bridge, **recomputed** and then re-bound — `BRIDGE BINDINGS OK` on both seeds;
2. the selection record, re-bound to its development inputs — `SELECTION BINDINGS OK`;
3. both seeds matched.

Selection is now refused outright while the frozen-test output holds any file,
so there is no path to an automatic reselection after test access.

---

## 7. Frozen test, evaluated once

6 slots (MA, L2, U*=step 70 per seed) × 560 pairs = **3,360 images, 3,360
detector rows**. The manifest identity `5dd87dbb5a77c0f4…` is unchanged, the set
was evaluated **once**, and selection is now locked out of running at all.

### 7.1 The matching check, re-reported on TEST — and it does not hold on both seeds

The protocol requires the matching check to be re-reported on the test set
without reselection. Matching was established on the **development** set; whether
it still holds on the **frozen test** set is a separate question, and it is the
one that licenses the word "matched".

| seed | dog deletion, L2 | dog deletion, U* | mismatch on TEST | dev mismatch | verdict |
|---|---|---|---|---|---|
| 17 | 17.50 pp | 16.25 pp | **1.25 pp** | 1.25 pp | **MATCHED ON TEST** |
| 29 | 22.50 pp | 28.75 pp | **6.25 pp** (U* deleted more) | 2.50 pp | **NOT MATCHED ON TEST** |

**Consequence, by the prespecified rule:** seed 29's retention contrast is
**INCONCLUSIVE as a matched comparison** — the arms are 6.25 pp apart in target
deletion on the very set being reported, beyond the 5 pp tolerance. The selected
checkpoint stands; **no reselection was performed**, the tolerance was not
widened, and nothing was retrained. Since the design requires **both** seeds, the
confirmatory matched claim is **not** satisfied across seeds.

> This check was missing from the analysis script and was added here (CPU only,
> no reselection, previous analysis record preserved as
> `analysis_test.20261010T205055.superseded.json`). It is the single most
> consequential number in this run, and the run would have over-claimed without
> it.

### 7.2 Primary endpoint: paired bird retention at matched deletion, t=0.5

Per training seed, never pooled. Frozen scene-cluster bootstrap
`2ba0e8df5034d011…`, 70 clusters, 7 category strata, 10,000 draws, RNG
2026100901, 95% percentile intervals.

| seed | D_bird(L2) | D_bird(U*) | contrast L2 − U* | 95% interval | reading | matched on TEST |
|---|---|---|---|---|---|---|
| 17 | 73.75% | 72.50% | **+1.25 pp** | [−2.50, +5.00] | practical equivalence; a ≥10 pp benefit **ruled out** | **yes** |
| 29 | 76.25% | 77.50% | **−1.25 pp** | [−6.25, +3.75] | practical equivalence; a ≥10 pp benefit **ruled out** | **no — inconclusive** |

**What this supports, on the one seed where matching holds on the test set:** at
matched dog deletion, the L2-SP arm shows **no material retention advantage** on
bird. The interval lies wholly inside ±10 pp and its upper bound is +5.0 pp, so a
benefit of 10 pp or more is ruled out at that margin, conditionally.

**What it does not support:** a two-seed confirmatory claim. Seed 29's arms were
not matched on the test set, so its numerically similar result cannot be read as
a matched comparison — even though it points the same way.

### 7.3 Every category separately, never averaged (t=0.5, contrast L2 − U*)

| category | role | seed 17 | seed 29 |
|---|---|---|---|
| **bird** | **primary retained** | +1.25 [−2.50, +5.00] | −1.25 [−6.25, +3.75] |
| horse | retained (anchor) | +1.25 [−2.50, +5.00] | +1.25 [0.00, +3.75] |
| chair | retained | −2.50 [−8.75, +3.75] | −3.75 [−10.00, +2.50] |
| bicycle | retained | +2.50 [0.00, +6.25] | +2.50 [−2.50, +8.75] |
| sandwich | never-targeted control | −2.50 [−7.50, +2.50] | 0.00 [−7.50, +6.25] |
| cat | earlier deletion target | −2.50 [−8.75, +2.50] | +1.25 [−3.75, +6.25] |
| dog | current target (matching check) | −1.25 [−6.25, +3.75] | **+6.25 [−1.25, +13.75]** |

Thresholds 0.3 and 0.7, and the literal/paraphrase split, are reported in
`analysis_test.json`; 0.5 is primary. Every interval is **descriptive**: no
multiplicity correction is applied, an interval including zero is not evidence of
no effect, and a difference in labels between seeds is not a significant
difference.

### 7.4 Blinded human-audit packet

**228 items** (SET R 168 stratified-random, SET D 60 enriched for scores in
[0.3, 0.7] and parent→child verdict flips), shuffled once under RNG 20261011.
Every label cell is **empty**. The key lives outside the packet at
`annotation_v2/KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv`; the packet directory
contains no arm, seed, score or set label — verified by probing it. Key-free
metadata is committed at
[`annotation_packet_v2/`](annotation_packet_v2/).

### 7.5 Compute

2.9014 device-hours recorded in total across both runs (1.2022 carried forward
from the original run, 1.6992 for this amendment), 56 stages, **0 failed, 0
orphaned**. No ceiling applied. GPUs 0 and 1 only, both verified idle before
assignment; GPUs 2 and 3 were never touched.

---

## 8. What no outcome here can support

* **No mechanism claim.** Nothing here speaks to *why* either arm behaves as it
  does, and none was sought.
* **No causal account of run-to-run differences** (§3).
* **Asymmetric completion evidence, disclosed not corrected.** The saved L2
  endpoints' training completion is **UNVERIFIED** legacy evidence; the new
  unregularised arms' completion is counter-verified at 100 steps. Nothing was
  backfilled.
* **The grid was chosen after seeing development results** (disclosed in the
  amendment, §1). The development set is a selection instrument and no
  confirmatory claim rests on it, but the grid lacks the "fixed before looking"
  property the original 100-step grid had, and any report must say so in the
  same sentence as the result.
* **Detector output is a proxy.** Every number above is provisional until the
  blinded human annotation is complete.
