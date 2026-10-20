# VALIDATION_SPEC — locked final CPU checks

Status: **LOCKED** (coordinator, 05:35Z, SPEC v1.3.1). Binding together with `configs/LOCK.json` (sha256 of this file,
SPEC.md, the code, tests and `configs/validation_locked.json`); the lock commit hash is recorded in RESULTS.md. After the lock: no tuning of the candidate, no change of success criteria, at most one
documented correctness repair (G_BRIEF §4).

## 1. Method and arms (SPEC v1.3 §4)
Candidate **FX-Mix**. Decisive comparators **SC-TF** (exclusion target on random teacher-forced states) and **SC**
(MDU loop with its rollout flag on, which replaces the TF state by one random-stop greedy rollout state). Ablations **FX-Mix-greedy** (pool = P1 only) and **SC-TF-cm** (compute-matched SC-TF).
Literature context **EMDU**, **MDU τ=1**, **NPO** (+ retain SFT). References (not arms): base, retrain, decode-time F-ban.

## 2. Configuration (fixed from development; see results/dev and DEV_LOG.md)
Substrate: `mdmu` toy world (24 entities, forget {city, year} of 6 entities), 2-layer mask predictor (d 64), base and
retrain models trained 3000 steps from the same init (`python -m mdmu.base_train --world-seed W --init-seed W --steps 3000`).
Unlearning (all arms): LoRA r=8, α=16 on all 7 linear types of both blocks, AdamW (wd 0), clip 1.0, forget batch 32
(SC-TF-cm 52 = compute-matched to FX-Mix's 136 row-forwards/step), retain batch 32, λ_f = λ_r = 1, ε = 1e-3; FX arms
α = 0.25, J = 4 (+ decision state), refresh every R = 4 steps, 1 rollout per forget prompt per refresh.
**Strength grid (tuning decision from dev worlds 0–1, where Cut-A picks sat at the top of the old grid):
L* = {2e-3, 4e-3} for every arm × checkpoints every 50 steps up to S* = 400** (16 strength points per unit).
Evaluation: configs P1–P3, H1–H6 (SPEC §3); forget 8 samples/prompt for stochastic configs, retain 4, copy 1; eval seed
1234; mechanism statistics (off-manifold fraction, certificate) at every checkpoint with 4 samples.
Everything is in `configs/validation_locked.json`; the runner refuses placeholders.

## 3. Fresh validation instances
Worlds **W ∈ {2, 3, 4}** (world seed = init seed = W; never used in development), unlearning seeds **{0, 1}** per arm and
world ⇒ 6 units per arm. G trains base/retrain for each world with the locked command.

## 4. Metrics (per checkpoint)
held-out max leak_any over H1–H5 (primary), H6 (extrapolation), in-pool max; leak_full; mean F_wf; mean retained-fact
accuracy over 9 configs (R), shared-value-token slice accuracy (S), min copy accuracy; off-manifold fraction of
first-commit decision states; certificate B̂ (H configs); compute counters and wall time.

## 5. Operating points and decision rules (applied mechanically by G)
- Feasible checkpoint: R ≥ R_base − 0.05 and min copy ≥ copy_base − 0.02 (per world, from that world's base).
- **Cut A** (per unit): min held-out max leak_any over feasible checkpoints (ties → higher R); INFEASIBLE if none.
- **Cut B** (per unit): max R over checkpoints with held-out max leak_any ≤ that world's retrain held-out max leak_any;
  INFEASIBLE if none.
- Δ_A(X) = CutA(X) − CutA(FX-Mix) per unit (INFEASIBLE comparator ⇒ Δ_A = +1; INFEASIBLE FX-Mix ⇒ Δ_A = −1).
- **R1 (distinctness, decisive):** PASS iff for X ∈ {SC-TF, SC}: mean Δ_A(X) ≥ 0.03 over the 6 units AND Δ_A(X) > 0 in
  ≥ 5 of 6 units AND FX-Mix's Cut B is INFEASIBLE in no more units than SC-TF's.
- **R2 (pool ablation, descriptive):** mean Δ_A(FX-Mix-greedy) reported; not a pass/fail input.
- **R3 (mechanism, descriptive with threshold):** mean off-manifold first-commit fraction of SC-TF at Cut A over H1–H5;
  "coverage gap present" iff ≥ 0.5.
- **R4 (target vs degeneration, literature context):** at Cut A, mean F_wf(FX-Mix) − max over {NPO, MDU τ=1, EMDU} of
  mean F_wf ≥ 0.2 ⇒ "exclusion target avoids degenerate forgetting" (a property shared by every arm with the same
  target; credited to the TNT/UNDIAL target family, not to FX-Mix's state design).
- **R5 (sanity, must hold for a world to count):** base held-out max leak_any ≥ 0.9; retrain held-out max leak_full = 0.
- **R2b (pre-registered secondary, decision-irrelevant):** the R1 rule with FX-Mix-greedy as the candidate (dev worlds
  showed it lowest, within one-prompt granularity); reported only.
- Coverage: 8 arms × 3 worlds × 2 seeds × 2 LRs = 96 runs must complete; a unit that cannot be completed after the single
  permitted repair counts as INFEASIBLE for that arm.
- Interpretations confirmed (implemented in `experiments/summarize_locked.py`): I1 ties — Cut A: higher R, fewer steps,
  lower LR; Cut B: lower leak, fewer steps, lower LR. I2 both INFEASIBLE ⇒ Δ_A = 0 (flagged). I3 a world failing R5 is
  excluded and the 5-of-6 rule becomes ⌈5n/6⌉ of n. I4 R3 averages the off-manifold fraction over H1–H5 cells with ≥ 1
  decision state, then over units. I5 R4 means over units whose Cut A exists. I6 missing unit ⇒ INFEASIBLE.

## 6. Decision mapping (sprint outcome)
- R1 PASS and integration dry-run test passes ⇒ **READY_FOR_BOUNDED_REAL_MODEL_TEST** (dependencies in HYPERPLANE_PLAN §1).
- R1 PASS but integration dry-run fails ⇒ **DESIGNED_WITH_SPECIFIC_BLOCKER**.
- R1 FAIL ⇒ **NO_DISTINCT_METHOD_AFTER_CONSTRUCTION**: the implemented best candidate is reported with its exact
  reduction (SC-TF = TNFF/UNDIAL-γ=∞ target on MDU-style random TF states), and the Hyperplane plan is offered only as
  the bounded falsification test of whether the CPU negative transfers (its S1 zero-training gate first).

## 7. Commands (G runs exactly these from `$R`, through the ledger wrapper with `--agent G`; ≤ 4 concurrent 1-thread processes)
```
PY=/tmp/claude-0/env-llada/bin/python; L="$PY tools/ledger_run.py --agent G --threads 1 --"
# 0. lock check: sha256 of every file listed in configs/LOCK.json must match
# 1. tests
$L $PY -m pytest -q tests/
# 2. substrates (one process per world, in parallel), run from src/
cd src && for W in 2 3 4; do $L $PY -m mdmu.base_train --world-seed $W --init-seed $W --steps 3000 --out ../results/final/substrate/w${W}_i${W} & done; wait; cd ..
# 3. references
$L $PY experiments/run_locked.py --refs
# 4. all units: 4 shards in parallel
for i in 0 1 2 3; do $L $PY experiments/run_locked.py --shard $i/4 & done; wait
# 5. assemble and summarise (integration status from the dry-run test in step 1)
$L $PY experiments/run_locked.py --assemble
$L $PY experiments/summarize_locked.py --integration-dryrun <pass|fail>
```
Outputs: `results/final/locked/{refs,runs,units}/`, `results/final/locked/final_summary.json`.
