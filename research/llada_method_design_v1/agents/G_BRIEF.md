# G brief — independent verifier (final locked checks + audit)

You did not author the method, the implementation or the validation spec. Your separation is procedural only: you share
files and a base model with the other agents (no blinding). Say so in your report.

Inputs (read in order): `$R/VALIDATION_SPEC.md` (locked; its commit hash is in `$R/configs/LOCK.json`), `$R/SPEC.md`,
`$R/THEORY.tex`, `$R/METHOD.md`, the code (`$R/src/mdmu/`, `$R/experiments/`, `$R/integration/`), tests (`$R/tests/`).

## Tasks
1. **Lock integrity.** Verify that the files hashed in `configs/LOCK.json` are unchanged (sha256) before you run anything.
2. **Tests.** Run the full test suite once; record counts and any failure verbatim. Test counts are not efficacy evidence.
3. **Code ↔ spec audit (static).** Check, citing file:line, that: the exclusion target equals Lemma T (log-space, fp32, ε
   handling); training states for every arm match SPEC §2/§4 (fact-free TF with fact positions forced; rollout states
   from the *current* student with first-commit truncation and decision state kept; SC/SC-TF exactly as defined); the
   evaluation counts Λ_F over all answer positions; held-out configs are never used for training states; compute
   counters count what they claim; the selection rule implements VALIDATION_SPEC exactly.
4. **Locked run.** Execute exactly the commands in VALIDATION_SPEC §6 on the fresh worlds it names (train base/retrain
   per world, then every arm), with the locked hyperparameters, thread caps and ledger wrapper (`--agent G`). Do not
   tune, re-seed, or drop arms/configs. If a run crashes, record it; you may make **at most one correctness repair**
   (a bug that makes code disagree with SPEC/VALIDATION_SPEC), only after preserving the original outputs, documenting
   the defect, and listing which claims the repair invalidates; then rerun only the affected units.
5. **Decision.** Apply VALIDATION_SPEC §5 rules mechanically to the outputs; report every rule's inputs and outcome,
   coverage (units run / planned) and infeasible cases.
6. **Claims audit.** Compare every quantitative or qualitative claim in START_HERE.md, RESULTS.md, METHOD.md,
   PRIOR_ART.md, HYPERPLANE_PLAN.md and THEORY.tex against outputs and SPEC; list each mismatch as (file, quote,
   problem, required fix). Check that no document claims superiority on LLaDA, retraining equivalence, or novelty from
   an absent search hit.

## Outputs
`$R/results/final/` (raw JSON per unit + `final_summary.json`), `$R/results/final/G_REPORT.md` (≤200 lines: lock check,
tests, audit findings with file:line, decision-rule table, coverage, repairs if any, claims-audit list).
Use ≤ 3 threads total. Reply ≤ 25 lines.
