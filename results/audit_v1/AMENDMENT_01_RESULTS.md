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

**It does, on both seeds, at step 70.**

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

## 7. Frozen test and analysis

<!-- filled in when the run completes -->

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
