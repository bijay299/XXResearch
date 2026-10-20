# SPEC_ERRATA — post-lock corrections to locked documents (no effect on the locked computation)

`SPEC.md` and `VALIDATION_SPEC.md` are hashed in `configs/LOCK.json` and are left byte-identical. The corrections
below were found by the independent verifier G after the locked run (`results/final/G_REPORT.md`) or by the coordinator
while writing the results. None changes a configuration, a rule or a result of the locked CPU validation.

| # | locked text | correction | found by |
|---|---|---|---|
| E1 | SPEC.md §2 (LLaDA ε): "certificate floor is ≤ 2.1 % for low-confidence configs" | holds for gen 128 / block 32 / 128 steps (Σ_n\|E_n\| = 2112); for single-block 128-step low-confidence decoding Σ_n\|E_n\| = 8256 ⇒ floor ≈ 8.3 % at ε = 1e-5; for random remasking ≈ 0.13 %. The certificate is informative mainly for random-remasking configs and for the greedy exact-zero check. | G (K4) |
| E2 | SPEC.md §6: retrain "leak_any 0.08–0.13 is chance from shared value pieces" | world-dependent: dev world 0 0.13, world 1 0.26; locked worlds 2/3/4 held-out max 0.250 / 0.234 / 0.167 (per config up to 0.292). | G (K7) |
| E3 | SPEC.md §2 and THEORY.tex Algorithm 1: "config π_k ~ w (uniform over the pool)" per rollout | implementation: stratified round-robin over the 24 unique forget prompts at each refresh (each config used for exactly 1/3 of prompts); equal in expectation. | G (A5) |
| E4 | VALIDATION_SPEC.md §7: `L="$PY tools/ledger_run.py --agent G --threads 1 --"` | `tools/ledger_run.py` requires `--label`; G added `--label "G stepN …"` and changed nothing else (`results/final/g_check/orchestrate.sh`). | G (A14) |
| E5 | configs/validation_locked.json `base_train_command`: "2 threads" | VALIDATION_SPEC §7 (1 thread per process) governed; G used 1 thread. | G (A15) |
| E6 | SPEC.md §4 SC row (v1.3.1) — correct; but PRIOR_ART/METHOD drafts wrote SC/SC-TF with r_0 (`logits[F] = −inf`) | code and SPEC use r_ε with ε = 1e-3 for SC-TF and SC (toy); fixed in PRIOR_ART/METHOD. | G (K6) |
| E7 | `src/mdmu/fxmix.py` module docstring ("SPEC v0.1", "SC (r_0 on MDU's random TF states)"), `unlearn.py` line 1 | stale docstrings in locked code files; behaviour matches SPEC v1.3.1 (G A1–A12). Left unedited to keep the lock hashes. | G (A16) |
| E8 | (process) `experiments/run_locked.py` edited after the lock at 05:36:25Z | no numeric effect (bit-identical re-run); locked version restored; edited copy kept in `results/final/g_check/`. DECISION_LOG #20. | G (A18) |
| E9 | (test gap) `tests/test_fxmix.py::test_selection_rules_cut_A_and_B` tests `dev_compare.py`, not `summarize_locked.py` | the decisive summariser has no unit test; G's independent re-implementation (`results/final/g_check/g_recompute.py`) reproduced every selection and rule outcome. | G (A13) |
