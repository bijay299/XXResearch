# VALIDATION_SPEC — locked final CPU checks (DRAFT until `configs/LOCK.json` exists)

Status: **DRAFT**. Becomes binding when committed together with `configs/LOCK.json` (sha256 of this file, SPEC.md, the
code and the locked config). After the lock: no tuning of the candidate, no change of success criteria, at most one
documented correctness repair (G_BRIEF §4).

## 1. Method and arms (SPEC v1.3 §4)
Candidate **FX-Mix**. Decisive comparators **SC-TF** (exclusion target on random teacher-forced states) and **SC**
(SC-TF + MDU's rollout flag). Ablations **FX-Mix-greedy** (pool = P1 only) and **SC-TF-cm** (compute-matched SC-TF).
Literature context **EMDU**, **MDU τ=1**, **NPO** (+ retain SFT). References (not arms): base, retrain, decode-time F-ban.

## 2. Configuration (fixed from development; see results/dev and DEV_LOG.md)
Substrate: `mdmu` toy world (24 entities, forget {city, year} of 6 entities), 2-layer mask predictor (d 64), base and
retrain models trained 3000 steps from the same init. Unlearning: LoRA r=8 on all 7 linear types, AdamW (wd 0), clip 1.0,
forget batch 32 (SC-TF-cm 52), retain batch 32, λ_r = 1, ε = 1e-3, FX α = 0.25, J, R and rollout settings as in
`configs/validation_locked.json`. Strength grid: LR ∈ L* (locked list) × checkpoints every 25 steps up to S* steps.
Evaluation: configs P1–P3, H1–H6 (SPEC §3), eval seeds and sample counts fixed in the locked config.

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
- Coverage: 7 arms × 3 worlds × 2 seeds × |L*| runs must complete; a unit that cannot be completed after the single
  permitted repair counts as INFEASIBLE for that arm.

## 6. Decision mapping (sprint outcome)
- R1 PASS and integration dry-run test passes ⇒ **READY_FOR_BOUNDED_REAL_MODEL_TEST** (dependencies in HYPERPLANE_PLAN §1).
- R1 PASS but integration dry-run fails ⇒ **DESIGNED_WITH_SPECIFIC_BLOCKER**.
- R1 FAIL ⇒ **NO_DISTINCT_METHOD_AFTER_CONSTRUCTION**: the implemented best candidate is reported with its exact
  reduction (SC-TF = TNFF/UNDIAL-γ=∞ target on MDU-style random TF states), and the Hyperplane plan is offered only as
  the bounded falsification test of whether the CPU negative transfers (its S1 zero-training gate first).

## 7. Commands
To be filled from `experiments/` once the locked runner exists (G runs exactly these, `--agent G`, ≤ 3 threads total).
