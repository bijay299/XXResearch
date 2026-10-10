# Amendment 01 — a fixed early dump grid at steps 10…90

**Status: PREPARED AND CPU-VERIFIED, NOT EXECUTED.** No GPU work has been done
for this amendment and none is authorised by it. It awaits the PI's separate
amendment decision.

> The resource clarification of 2026-10-10 (no GPU-hour ceiling; see
> [`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md)) **does not
> approve this amendment.** Free devices are not a protocol decision. Nothing
> here was widened because compute became cheaper, and nothing was narrowed to
> fit the retired budget.

Parent protocol: [`MATCHED_EFFECTIVENESS_PROTOCOL.md`](MATCHED_EFFECTIVENESS_PROTOCOL.md).
Original run: commit `3a5a69c`, output `/data/bijaypandey/cuig_pilot/seq_pilot/diag_v1`.
Implementation: `scripts/seq/run_early_grid.sh`, `scripts/seq/bridge_check.py`.

---

## 1. Why, and the disclosure that comes with it

The approved run completed training and the full development scan, and **stopped
at selection by the pre-declared infeasibility rule**. All ten dumps passed both
reference gates on both seeds, and the L2 reference was itself eligible on both
seeds, but no dump landed within 5 pp of its seed's L2 suppression:

| seed | L2 suppression | step-100 dump | closest mismatch | lower bracket |
|---|---|---|---|---|
| 17 | 30.0 pp (eligible, exactly on the gate) | 38.8 pp | 8.8 pp | MA, step 0, 0 pp |
| 29 | 33.8 pp (eligible) | 42.5 pp | 8.8 pp | MA, step 0, 0 pp |

So on both seeds the measured points **straddle** the L2 level: 0 pp at step 0
(the parent, by definition) and above it by step 100.

> **Corrected statement of what that implies.** An earlier version of this
> document said "the match point for each seed lies between step 0 and step
> 100". That claims more than straddling supports, and it is withdrawn.
> Suppression here is a **discrete** measurement on a **discrete** grid of
> steps, granular to 1.25 pp (one detection in 80 pairs), and it is not known to
> be monotone in steps. Straddling therefore does **not** guarantee that any
> step between 1 and 99 has suppression within 5 pp of the L2 level: the
> sequence may step over the band entirely, or wander in and out of it.
>
> What this amendment is, then, is a **test of a hypothesis** — that some step on
> the grid 10…90 lands inside the band — not the retrieval of a point already
> known to exist. **It may legitimately fail**, and a failure is reported as the
> result rather than treated as a reason to look harder.

> **Disclosure, stated plainly because it matters for how any result from this
> amendment must be read.** This early grid is **motivated by the development
> results**. It was not pre-registered independently of the data: we chose
> 10…90 *because* the completed scan showed both measured points to straddle the
> L2 level within the first 100 steps. The grid is therefore **post-hoc with
> respect to the development set**.
>
> What that does and does not cost:
> * The **development set is a selection instrument**, and selecting a grid on
>   it is the use it was frozen for. No confirmatory claim rests on it.
> * The **frozen test set is unaffected**: it has never been evaluated, its
>   manifest identity `5dd87dbb5a77c0f4…` is unchanged, and nothing in this
>   amendment inspects it.
> * But the *grid* no longer has the "fixed before looking" property the
>   original 100-step grid had, and any eventual report must say so in the same
>   sentence as the result.

---

## 2. What changes, and what does not

**Changes — exactly two, both in how training is stopped and saved, neither in
how training proceeds.** For each seed, retrain from that seed's saved `MA`:

| # | setting | original run | this rerun |
|---|---|---|---|
| 1 | **stopping limit** (`--iterations`) | 1000 optimizer steps | **100** |
| 2 | **save cadence** (`--checkpoint_every`) | every 100 steps | **every 10** |

That yields dumps at 10, 20, …, 90 (the amendment's grid) plus **100**, which is
scanned as the bridge to the original run and is **not offered for matching**.
`scripts/seq/run_early_grid.sh` enforces that separation: the scan list and the
match list are distinct and both are printed in preflight.

### 2.1 Everything that determines the first 100 updates is preserved

These are the settings that enter updates 1…100. Every one of them must be
identical, and `scripts/seq/bridge_check.py` **verifies** it from the two train
reports rather than taking it on trust; a difference in any of them is reported
as a defect of the rerun, not as a finding about training.

| group | preserved settings |
|---|---|
| start point | the same saved `MA` per seed, bound by digest: `3cb72ee162b9…` (s17), `fbf49199a306…` (s29) |
| randomness | the same `--seed` per trajectory (17, 29); evaluation generation seeds untouched and never reused as training seeds |
| learning rate | `learning_rate` 2e-06 before scaling, `scale_lr` on, `effective_learning_rate` **8e-06**, same `scale_lr_formula` |
| schedule | `lr_scheduler = constant`, `lr_warmup_steps = 500` (inert — see 2.3) |
| batching | `anchor_batch_size 4`, `gradient_accumulation_steps 1`, `num_processes 1`, `steps_per_epoch 50` |
| optimizer | AdamW, β₁ 0.9, β₂ 0.999, weight decay 0.01, ε 1e-08, `max_grad_norm 1.0` |
| objective | target dog, anchor horse, `l2sp_weight 0`, `l1sp_weight 0`, no anchor preservation, no gradient projection, no SelFT |
| data & precision | the same anchor image set and prompt file, `resolution 512`, `hflip` on, `noaug` on, fp32 |
| what is trained | `parameter_group = kv-xattn`, 32 tensors, 19,169,280 parameters |

### 2.2 What upstream derives from the stopping limit, declared

Changing `--iterations` makes upstream recompute
`epochs = ceil(iterations / steps_per_epoch)`, so `epochs_cap` goes 20 → 2 and
`epoch_capacity_steps` 1000 → 100. These are **consequences** of change 1, not
separate changes, and the bridge check classifies them as such rather than
letting them pass unnoticed.

### 2.3 Why a shorter horizon does not change the early updates — checked, not assumed

Upstream passes `num_training_steps = iterations × gradient_accumulation_steps`
into diffusers' `get_scheduler` (`UnlearningMethods/ConAbl/src/data.py:938`).
For `SchedulerType.CONSTANT`, `get_scheduler` returns
`get_constant_schedule(optimizer, last_epoch=…)` and **discards both
`num_warmup_steps` and `num_training_steps`** (verified in the installed
diffusers 0.30.2). Our runs use `lr_scheduler = constant`, so the learning rate
at update *k* does not depend on the stopping limit, and the recorded
`lr_warmup_steps = 500` never takes effect.

Under a cosine, linear or polynomial schedule this would be false — shortening
the horizon would change the learning rate at **every one of the first 100
updates** and the rerun would not be comparable at all. `bridge_check.py`
therefore refuses any rerun whose schedule is not horizon-free, and
`test_bridge_check.py` exercises that refusal with a cosine schedule.

### 2.4 Unchanged, explicitly

| | |
|---|---|
| training horizon of the arms being **compared** | 1000 steps for the L2 endpoint; the new dumps are *earlier points on the same trajectory shape*, not a shortened arm |
| L2 references | the same saved `MAB_L2` endpoints: `8ee7d56f62b0…` (s17), `c4f3ecdcb945…` (s29). **No L2 retraining.** |
| development selection | dog-only, 80 pairs per checkpoint, t=0.5, against that seed's own MA |
| eligibility gates | ≥30 pp suppression **and** ≤60% residue, applied to candidate dumps **and to the L2 reference itself** |
| match tolerance | **≤5 pp**, not widened |
| tie-break | earliest step |
| both-seed requirement | both seeds must match or the paired test does not run |
| frozen test manifest | `5dd87dbb5a77c0f4…`, 140 texts × 4 seeds = 560 pairs; evaluated once, no reselection |
| reporting | every category separately and never averaged; literal/paraphrase separately; thresholds 0.3/0.5/0.7 with 0.5 primary |
| uncertainty | per-training-seed intervals, never pooled; frozen grouping `2ba0e8df5034d011…`, scene-cluster unit, category strata, 10 000 draws, RNG 2026100901 |
| annotation | human labels stay manual and blinded; the arm key stays closed; detector results stay provisional |

**Excluded, as before:** iterative refinement (this is **one** grid, and if it
fails that is the end of this line), tolerance widening, L2 retraining,
additional training seeds, any test-based reselection, any sandwich-branch work,
any new M0 evaluation, any unmatched reference arm on the test set. None of
these becomes acceptable because compute is cheaper.

---

## 3. The rerun, and how the two runs are shown comparable

**A rerun is required.** Dumps can only be written while training runs; there is
no way to recover step-10…90 states from a finished trajectory.

**The honest problem with that.** The planned CPU comparison of the new
1000-step `U` endpoints against the saved `MAB` endpoints — same recorded
configuration, same parent, same seed — found them numerically different
([`u_vs_mab_comparison.json`](../../results/audit_v1/diag_v1_evidence/u_vs_mab_comparison.json)):

| seed | configuration identical | same saved parent | bitwise-identical tensors | max abs weight diff | overall relative Frobenius |
|---|---|---|---|---|---|
| 17 | yes, 32/32 hyperparameters | yes | **0 of 32** | 2.045e-03 | 5.652e-03 |
| 29 | yes, 32/32 hyperparameters | yes | **0 of 32** | 1.778e-03 | 5.717e-03 |

> **Corrected interpretation.** That comparison previously concluded "the
> divergence is therefore RUN-TO-RUN NONDETERMINISM on this stack". **Withdrawn:
> the cause is not isolated.** Equal *recorded* configuration and an equal
> parent do not identify a mechanism. The two runs also differ in library and
> driver state, in upstream code path (the new run trained with periodic
> checkpointing switched on), in dataloader worker ordering, and in anything
> outside the 32 recorded hyperparameters. Separating those would need a
> controlled repeat — same host, same commit, same cadence, twice — which was not
> run and **is not requested here**.
>
> What **is** established is the part that matters for this design: a re-run
> does not reproduce the original endpoint bitwise, so checkpoints from a new
> run are **not interchangeable** with the original run's, and a design that
> mixes them must measure the difference rather than assume it away.

**How comparability is established rather than assumed.**

1. **All dumps used for the match come from ONE run per seed.** Selection under
   this amendment uses only the new run's early grid (10…90). The original run's
   100…1000 dumps are *not* mixed into the match.
2. **The reference arms are unchanged saved artifacts.** `MA` and `MAB_L2` are
   reused by path and bound by hash. They are rescored in the new output
   directory so the new root is self-contained; because the scoring pipeline is
   seeded and manifest-bound, that rescoring doubles as a free reproducibility
   cross-check against the original run's detections for those four slots.
3. **A measured bridge at step 100 — the one step both runs write.**
   `scripts/seq/bridge_check.py` answers three questions separately:
   * **settings** — every setting in §2.1 identical, changes exactly those in
     §2.2 plus the cadence, schedule horizon-free;
   * **weights** — the two step-100 dumps compared tensor by tensor: bitwise
     identity, max absolute difference, relative Frobenius. Bitwise identity
     would mean the first 100 updates were reproduced exactly and the early
     dumps lie on the original path. Anything else is a measured distance whose
     **cause is not attributed**;
   * **behaviour** — development dog suppression at step 100 in both runs
     (original: 38.75 pp s17, 42.50 pp s29).
4. **A stop condition on the bridge, declared now, before the measurement.** If
   the two realisations' step-100 dog suppression differs by **more than 5 pp** —
   the same tolerance the match itself uses — then run-to-run variation is of the
   same order as the quantity being matched, the early grid cannot be trusted to
   locate a match point, and **this amendment stops and reports that instead**.
   `run_early_grid.sh` refuses to evaluate the frozen test set unless the bridge
   verdict is `BRIDGE_HELD`.
5. **The original run is retained in full.** Its trajectory, ten dumps, 24
   development slots, selection records, ledger and evidence bundle are
   preserved untouched; the new run writes to its own directory
   (`diag_v2_early_grid`).

**What this still does not establish.** That either realisation's step-*k* state
is "the" state of that configuration at step *k*. Run-to-run variation is
measured at the endpoint and bridged at step 100; it is **not** characterised
across the early grid, and no claim is made that it is small there.

---

## 4. Resources — informational

There is no GPU-hour ceiling and no reserve to protect
([`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md)). The figures
below are planning estimates from the measured, conservative cost model
([`cost_model_measured.json`](../../results/audit_v1/diag_v1_evidence/cost_model_measured.json)):
maximum observed per-image generation time, maximum observed per-slot overhead,
plus a 25 % margin. They are recorded for provenance and efficiency and they
**gate nothing**.

| line | device-hours |
|---|---|
| already recorded (original run, 27 stages) | 1.2022 |
| retrain 100 steps with dumps every 10, both seeds | 0.0883 |
| development scoring, 24 slots (MA + L2 + 10 dumps, × 2 seeds) | 0.9528 |
| frozen test, **only if both seeds match and the bridge holds** | 1.3474 |
| cumulative if the amendment runs to completion | **3.5907** |

Work starts only on verified-idle devices and waits otherwise; no other user's
job is touched.

---

## 5. What a result from this amendment could and could not support

* **If both seeds match** within 5 pp on the early grid **and** the step-100
  bridge holds, the frozen test set is evaluated **once** and the primary paired
  bird contrast is reported per seed with its interval, every category
  separately, and the matching check re-reported on TEST without reselection.
* **If either seed does not match**, the paired test does not run. That is a
  result and the end of this line of work; no further grid refinement follows.
* **If the bridge does not hold**, that is the reported result: the two
  realisations differ at step 100 by as much as the quantity being matched, so
  the early grid cannot locate a match point. No selection, no test evaluation.
* **In either case** the report must carry: that the grid was chosen after
  seeing development results; that the arms' completion evidence is asymmetric
  (the L2 endpoints' training completion is **UNVERIFIED** legacy evidence, the
  new unregularised arms' is counter-verified); that run-to-run variation was
  measured and bridged, **not** eliminated and **not** explained; and that
  detector results are **provisional** until the blinded human annotation is
  complete.
* **No mechanism claim** follows from any outcome, and none is sought.

### The scientific decision this enables

Whether a **matched-effectiveness** comparison of the two regularisation regimes
is feasible at all on this stack. If an early dump matches the L2 endpoint's
deletion level on both seeds, the frozen test set can answer the question the
project actually needs — *at equal deletion of the target, does L2-SP retain more
of the earlier concept?* — which is the gap the submission would rest on. If
nothing matches, or the bridge fails, then matched comparison by trajectory
truncation is not available here, and the gap has to be approached a different
way; that is equally decision-relevant and costs one short rerun to learn.

---

## 6. Decision requested

Approve, modify, or decline **this single amendment**. Nothing runs until a
separate decision is recorded.

```bash
# everything below is CPU-gated and starts only on verified-idle devices
bash scripts/seq/run_early_grid.sh preflight   # identities, reuse bindings, capacity
bash scripts/seq/run_early_grid.sh all         # smoke -> train -> dev -> bridge -> select
# the frozen test set stays a separate, explicitly gated invocation:
bash scripts/seq/run_early_grid.sh test        # refuses unless the bridge HELD
```

Configuration, for the record (`scripts/seq/run_early_grid.sh`):

| variable | value |
|---|---|
| `SEQ_DIAG_ROOT` | `${SEQ_ROOT}/diag_v2_early_grid` (new; the original run is read-only) |
| `SEQ_DIAG_ITERATIONS` | `100` (declared change 1) |
| `SEQ_DIAG_DUMP_EVERY` | `10` (declared change 2) |
| `SEQ_DIAG_MATCH_STEPS` | `10,20,30,40,50,60,70,80,90` (step 100 is the bridge, never a candidate) |
| `SEQ_EG_BRIDGE_STEP` / `SEQ_EG_BRIDGE_TOL_PP` | `100` / `5.0` |
| `SEQ_DIAG_SEEDS` | `17 29` (unchanged) |

**CPU validation already done** (no GPU, trainer and detector mocked, real
runner, real generator, real validator, real selection):

| suite | assertions | what it pins down here |
|---|---|---|
| `test_diagnostic_runner.sh` | 104 | the early-grid profile end to end: 100 steps at cadence 10, ten dumps per seed, 24 development slots, nine candidates offered for matching, the original run untouched, the frozen test set refused before the bridge is checked |
| `test_bridge_check.py` | 38 | the bridge holds only when exactly the declared changes are present; violations, horizon-dependent schedules, unclassified settings, behaviour drift, missing dumps and a one-seed failure all refuse |
| `test_gpu_budget.py` | 59 | usage recording without a cap, and the preserved historical record |

What is **not** yet verified, and cannot be on CPU: that upstream writes dumps
at cadence 10 on real hardware for a 100-step run (the smoke stage checks
exactly this, at 20 steps with dumps every 10, before either full trajectory
starts), and the actual numbers.
