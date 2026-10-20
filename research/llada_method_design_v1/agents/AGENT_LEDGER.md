# Agent ledger

Real agent instances launched with the Agent tool (general-purpose subagents, same base model as coordinator).
Exchanges between agents are mediated by files in `agents/` plus resumption of the same agent instance
(SendMessage), so each response is authored by the agent that holds the original context.
Shared files and a shared base model mean there is **no blinding**; independence of G is procedural only
(G did not author the implementation and runs locked checks from the committed spec).

| role | instance | wave | launched (UTC) | task | status / output |
|---|---|---|---|---|---|
| A objective designer | A#1 | 1 | 03:40 | proposal FFX-OP | done 04:05 → `A_proposal.md`, `A_scripts/` |
| B denoising-process designer | B#1 | 1 | 03:40 | proposal FCH-Mix | done ~04:17 → `B_proposal.md`, `B_scripts/` |
| C parameterisation designer | C#1 | 1 | 03:40 | proposal MSDA | done 04:15 → `C_proposal.md`, `C_scripts/` |
| A objective designer | A#1 (resumed) | 2 | 04:16 | strongest objection to C | running → `OBJ_A_to_C.md` |
| D prior-art & controls | D#1 | 2 | 04:17 | reduction to known methods, controls | running → `D_prior_art.md` |
| B denoising-process designer | B#1 (resumed) | 2 | 04:23 | strongest objection to A | done 04:27 → `OBJ_B_to_A.md`, `B_scripts/objection_A_coverage*` |
| A objective designer | A#1 (resumed) | 2 | 04:16 | (row above) | done 04:22 → `OBJ_A_to_C.md`, `A_scripts/objection_msda_anchor*` |
| E mathematical critic | E#1 | 2 | 04:23 | proofs, counterexamples, spec corrections | running → `E_critique.md` |
| F implementation engineer | F#1 | 3 | 04:28 | reference impl, tests, dev comparison, LLaDA integration, cost | running |
| D prior-art & controls | D#1 | 2 | 04:17 | (row above) | done 04:38 → `D_prior_art.md` (4 repos cloned read-only: t5patches, dllm, open-dlu, dlm-priming-vulnerability) |
| E mathematical critic | E#1 | 2 | 04:23 | (row above) | done 04:39 → `E_critique.md`, `E_scripts/` |
| C parameterisation designer | C#1 (resumed) | 2 | 04:39 | response to A; strongest objection to B | running → `RESP_C.md`, `OBJ_C_to_B.md` |
| A objective designer | A#1 (resumed) | 2 | 04:40 | response to B's objection + E verdicts | running → `RESP_A.md` |
| A objective designer | A#1 (resumed) | 2 | 04:40 | (row above) | done 04:43 → `RESP_A.md` |
| C parameterisation designer | C#1 (resumed) | 2 | 04:39 | (row above) | done 04:47 → `RESP_C.md`, `OBJ_C_to_B.md`, `C_scripts/objection_fchmix_tokenban*` |
| E mathematical critic | E#1 (resumed) | 2 | 04:46 | check THEORY.tex incl. Prop 2, SPEC consistency | running → `E_theory_check.md` |
| B denoising-process designer | B#1 (resumed) | 2 | 04:47 | response to C's objection + E verdicts | running → `RESP_B.md` |
| E mathematical critic | E#1 (resumed) | 2 | 04:46 | (row above) | done 04:50 → `E_theory_check.md`, `E_scripts/prop2_check*` |
| B denoising-process designer | B#1 (resumed) | 2 | 04:47 | (row above) | done 04:51 → `RESP_B.md`, `B_scripts/resp_norm_ablation*` |
| D prior-art & controls | D#1 (resumed) | 2 | 04:54 | verbatim evidence for snippet-only sources (EMDU not reproduced by coordinator search) | running → `D_prior_art.md` §6 |
| F implementation engineer | F#1 (resumed after container restart) | 3 | 05:16 | finish dev world 1; locked runner + summariser; draft locked config | running |
| F implementation engineer | F#1 (resumed) | 3 | 05:16 | (row above) | done 05:32 → world-1 dev, `experiments/run_locked.py`, `summarize_locked.py`, draft config |
| G independent verifier | G#1 | 4 | 05:36 | lock check, tests, static audit, locked run (worlds 2–4 × seeds 0–1), decision, claims audit | running → `results/final/G_REPORT.md` |
| G independent verifier | G#1 | 4 | 05:36 | (row above) | done 06:25 → `results/final/G_REPORT.md`, `results/final/locked/`, `results/final/g_check/` (independent rule re-implementation) |
| F implementation engineer | F#1 (resumed) | post-validation | 06:26 | LLaDA comparator arms (SC-TF, EMDU, MDU, NPO) for the Hyperplane plan (G finding A17); new test file only | running |
| F implementation engineer | F#1 (resumed) | post-validation | 06:26 | (row above) | done 06:30 → comparator arms in `integration/llada_fxmix.py`, `tests/test_integration_comparators.py` (10/10) |
| G independent verifier | G#1 (resumed) | reconciliation | 06:38 | re-audit of the outcome-bearing documents | done 06:41 → `results/final/G_REPORT.md` §9 |
