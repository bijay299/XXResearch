# EVIDENCE_INDEX — claim → evidence → status

Status legend: **FINAL** = locked validation run by G; **DEV** = development (seen while designing/tuning);
**EXACT** = exact/tabular computation; **STATIC** = code/paper reading; **PROJ** = arithmetic projection, not measured.

| # | claim | evidence (path) | status |
|---|---|---|---|
| 1 | LLaDA SFT trains only teacher-forced states; native sampler = blocks/steps/lc-or-random/Gumbel-T/no re-mask | `/tmp/claude-0/src/LLaDA` @ 9182493 (GUIDELINES.md, generate.py); `sources/SOURCE_PINS.md`; port `src/mdmu/diffusion.py`; `tests/test_substrate.py`, `tests/test_integration_dryrun.py::test_generate_port_matches_upstream` | STATIC + tested |
| 2 | MDU objective, states, anchor; rollout flag replaces TF state; released `run_main.sh` omits objective flags | MDU PDF (sha256 82e1d2ad…); MDU code @ e5e8e82 L1102–1147, L1005–1100, L1355–1371; `agents/D_prior_art.md` §2.1 | STATIC |
| 3 | EMDU objective (anchor-first, fully-masked anchor), EOS-only forgetting mode, robustness grid | EMDU PDF (sha256 d8059beb…); `PRIOR_ART.md` §2 | STATIC |
| 4 | Exclusion target = TNT/TNFF target (UNDIAL γ→∞) | google/t5patches @ 09285b3 `models.py` L565–641; open-unlearning @ b299290 `undial.py`; `agents/A_proposal.md` §5 | STATIC |
| 5 | MDU's anchor can predict the fact via visible answer tokens | `agents/A_scripts/tabular_check_out.json`; `results/dev/coord_probe/probe_baselines.json` (MDU τ=1 leak_any 0.10–0.54) | EXACT + DEV |
| 6 | In the zero-sharing class, TF-only exact training does not identify config-robust forgetting; own-visitation training does | `THEORY.tex` Prop. 2; `agents/E_scripts/prop2_check_out.json`; `agents/B_scripts/tabular_commit_hazard_out.json` | EXACT (proof + check) |
| 7 | Sampler-level first-commit decomposition and hazard bound (standard tools) | `THEORY.tex` Prop. 1; `agents/E_critique.md` §5; `tests/test_fxmix.py::test_certificate_upper_bounds_leak_in_expectation` | EXACT |
| 8 | Lemma T (target closed form, KL value, gradient at θ0, coarse-graining bound) | `THEORY.tex` Lemma 1; `tests/test_fxmix.py` (target/I-projection/gradient tests) | EXACT + tested |
| 9 | MSDA (C) cannot pass the anchor's fact-mass floor; too close to the closed projection family | `agents/OBJ_A_to_C.md`, `agents/RESP_C.md`, `agents/A_scripts/objection_msda_anchor_out.json` | EXACT |
| 10 | Under sharing, exclusion edits act like a token ban (no retention mechanism from the state design) | `agents/OBJ_C_to_B.md`, `agents/RESP_B.md`, `agents/C_scripts/objection_fchmix_tokenban_out.json` | EXACT (log-linear toy) |
| 11 | GD/NPO/MDU degenerate forget answers on the neural toy | `results/dev/coord_probe/` | DEV |
| 12 | Dev worlds 0–1: SC-TF ≈ FX-Mix on held-out leakage; EMDU/NPO degenerate | `results/dev/compare_w{0,1}/`, `results/dev/DEV_LOG.md` | DEV |
| 13 | Locked comparison FX-Mix vs SC-TF/SC (R1), secondary R2b, mechanism R3, degeneration R4, sanity R5 | `results/final/locked/final_summary.json`, `results/final/G_REPORT.md`, `VALIDATION_SPEC.md`, `configs/LOCK.json` | FINAL — R1 FAIL ⇒ NO_DISTINCT_METHOD_AFTER_CONSTRUCTION; G's independent recomputation matches (`results/final/g_check/`) |
| 14 | CPU costs per arm; LLaDA FLOP/memory projections | `results/dev/COST.md`; `ledger/resources.jsonl` | measured (CPU) / PROJ (LLaDA) |
| 15 | Integration module (FX-Mix + comparator arms SC-TF/EMDU/MDU/NPO) matches pinned interfaces on a tiny model; real model untested | `integration/llada_fxmix.py`, `configs/hyperplane_fxmix.yaml`, `tests/test_integration_dryrun.py` (locked), `tests/test_integration_comparators.py` (post-validation) | tested (tiny) / UNTESTED (real) |
| 16 | Decision log, agent roles and exchanges | `DECISION_LOG.md`, `agents/AGENT_LEDGER.md`, `agents/OBJ_*`, `agents/RESP_*` | record |
| 17 | Resource use | `ledger/resources.jsonl`, `ledger/manual.jsonl`, `ledger/README.md`, `ledger/SUMMARY.md` | measured (wrapped processes) |
| 18 | Post-lock corrections to locked text and process issues (no effect on locked computation) | `SPEC_ERRATA.md`, `DECISION_LOG.md` #20–21, `results/final/G_REPORT.md` §1, §6–7 | record |
| 19 | Compute-matched SC-TF equals FX-Mix's held-out leakage at higher retention (alternative explanation E2) | `results/final/locked/final_summary.json` (units `sc_tf_cm_*`), RESULTS.md §3 | FINAL |
