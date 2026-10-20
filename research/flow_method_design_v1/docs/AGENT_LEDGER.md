# Agent and resource ledger (coordinator-maintained)

Model serving the coordinator and all subagents: configured `claude-sonnet-5-5` (agents inherit unless noted).
Agent spawning was AVAILABLE; roles ran as genuine agents. Concurrency cap: 3 specialists at a time.

| Wave | Agent | Role | Output path | Start (UTC) | End (UTC) | Notes |
|---|---|---|---|---|---|---|
| 1 | A | Sources / competitive evidence | agents/A_* | 2026-10-19T22:55 | | |
| 1 | B | Loss geometry | agents/B_* | 2026-10-19T22:55 | ~23:01 | 105.9k subagent tokens, 11 tool uses, 314 s; verdict NONE (diagnostic H-B1) |
| 1 | C | Optimizer / shared params | agents/C_* | 2026-10-19T22:55 | | |
| 1b | D | Trajectories / representations | agents/D_* | ~23:02 | | spawned when B finished |

Coordinator model switched by user from claude-sonnet-5-5 to claude-opus-5-5 at ~23:02; agents spawned after that inherit it.
| 1 | A (end) | | | | ~23:01 | 153.1k subagent tokens, 49 tool uses, 342 s; arXiv/HF/S2 unreachable (Bash 403, WebFetch DNS) — paper equations UNVERIFIED this sprint |

Environment events: torch install attempted — pytorch CPU index blocked (403); PyPI CUDA wheel download aborted at 2.7 GB pip cache (deleted, uncounted toward final files). Sprint is numpy/scipy only.
Git push to origin failed with HTTP 403 (no GitHub App write access) at ~22:58; commits accumulate locally.
| 1 | C (end) | | | | ~23:10 | verdict: method novelty NONE; testable ablation C-H1 "VAU-AdamW"; ~5 core-min (ledger C_*) |
| 1b | D (end) | | | | ~23:17 | 143.0k tokens, 18 tool uses, 819 s; verdict NONE (diagnosis H-D; uniform-t ablation P-D4 INCONCLUSIVE); ~2.8 core-min |
| — | Coordinator | Selection (C-H1 → VAU), CANDIDATES.md, DERIVATION.tex (self-found error in Thm 7 proof fixed before review), sign pre-check | | 23:15 | 23:22 | coord_sign_check 2.8 core-s |
| 2 | E | Prototype engineer (core library, tests, native interface) | code/vau, code/tests, native/ | 23:19 | | |
| 2 | G | Evaluation protocol + real-model protocol | configs/eval_protocol.json, docs/EVAL_PROTOCOL.md, docs/REAL_MODEL_PROTOCOL.md | 23:19 | | |
| 2 | F | Adversarial proof/novelty review | review/ | 23:22 | | |
Git push access restored by the user (GitHub reconnected) at ~23:23Z on 2026-10-19; first successful push of 5b99c2d.
| 2 | G (end) | | | | ~23:32 | 164.3k tokens, 18 tool uses, 746 s; documents only, 0 compute |
| 2 | F (end) | | | | ~23:40 | 179.4k tokens, 36 tool uses, 1027 s; ~17 core-min (over the ~10 requested); VAU practical claims not supported vs matched controls |
| — | Coordinator | Early pivot to FNS considered and DECLINED (collapses under zero-init B; working form = known null-A/InfLoRA → control na_lora); F narrowings applied to CANDIDATES.md and DERIVATION.tex | | 23:41 | 23:55 | coord_factor_exact_check, coord_factor_collapse_check |
| 2 | G (A1 amendment end) | | | | ~00:01 | +212.6k tokens, 8 tool uses, 294 s; protocol v1.1; G's 'file reworded' note checked: no committed diff to REAL_MODEL_PROTOCOL.md between G's writes (benign) |
| 3 | E (resumed) | Runner, integrity I1-I15, dev phase, compute projection; stops pre-freeze | code/run_validation.py, results/dev | ~23:46 | | |
| 3 | F (resumed) | ONE bounded full-algorithm novelty assessment (user scope clarification received 2026-10-20T00:18Z; study NOT yet frozen at that time; no spec change made in response) | review/F_FULL_ALGORITHM_NOVELTY.md | 00:18 | | |
| 3 | F (assessment end) | | | | 00:21 | bounded assessment written; one sympy check, no models run |
| 3 | E (end) | | | | 00:21 | 495.8k tokens total, 112 tool uses, 5167 s; integrity I1-I15 pass; dev phase ~30 core-min; re-delivered final report at ~00:45 (no new content) |

## Final reconciliation (coordinator, 2026-10-20, after validation)
**Timing correction.** Rows above marked "~" were coordinator estimates written by hand and some are inconsistent
(e.g. G's A1 "~00:01" vs the 23:5x events around it). They are preserved, not rewritten. Authoritative times are the
UTC stamps in `results/ledger.jsonl` and the git commit times. Key milestones: wave 1 spawned 22:55Z (2026-10-19);
selection 23:15Z; F review returned ≈23:40Z; freeze record written 2026-10-20T00:23:15Z and committed in `aec0f38`;
validation 00:24:08–01:24:01Z (one failed launch at 00:24:07Z ran no units); post-hoc table 01:27:42Z.

**Agents (all genuine subagents; ≤ 3 concurrent specialists).** Token/tool/duration figures are from harness notifications.
| agent | role | subagent tokens | tool uses | wall |
|---|---|---|---|---|
| A | sources, GEM code facts | 153.1k | 49 | 342 s |
| B | loss geometry (spec: NONE + H-B1 diagnostic) | 105.9k | 11 | 314 s |
| C | optimizer/shared params (spec selected → VAU) | 109.1k | 13 | 532 s |
| D | trajectories/timesteps (spec: NONE + H-D) | 143.0k | 18 | 819 s |
| E | engineer: library, tests, native patch, runner, dev phase | 495.8k (cumulative over resumptions) | 112 | 5167 s |
| F | adversarial proof/novelty review + bounded full-algorithm assessment | 179.4k + 203.0k | 36 + 5 | 1027 s + 151 s |
| G | evaluation protocol + real-model proposal (+ A1 amendment) | 164.3k + 212.6k | 18 + 8 | 746 s + 294 s |
Coordinator: synthesis, selection, derivation, pivot check, freeze, validation launch, analysis, write-up, git.

**Undisclosed-by-agent event found by the coordinator:** the first `E_run_dev` (development phase, pre-freeze) FAILED
after 1,577 core-s; a second 5-s invocation regenerated the dev summary from the written raw outputs. E's report did not
label this as a repair. It is development-only and pre-freeze; disclosed here.

**Compute (charged once; nested `E_run_validation` excluded as it is inside `VALIDATION_PHASE_v1_2`).**
Pre-validation 3,470.6 core-s; validation 13,920.5 core-s; post-hoc analysis 22.1 core-s ⇒ **17,413 core-s ≈ 4.84
core-h of 48**, of which failed attempts 1,682 core-s. Peak RSS 1.53 GiB (cap 16). New files: 158 MiB for this directory (cap 5 GiB), mostly gzipped raw validation records. Unmetered: short interactive shell/Python calls, pip installs, the 2.7 GB aborted torch
download (deleted). Dollar/credit spend: not observable or cappable from inside the session; no paid API, no top-up.
Deadline: work ran 2026-10-19T22:50Z – 2026-10-20 ≈ 02:00Z, inside the 5-day timebox.
