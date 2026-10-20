# G_REPORT — independent verifier: locked CPU validation, code/spec audit, claims audit

Agent G, 2026-10-20 05:36–06:25Z. Lock commit `13ec821e549b093bc04c7aac6366bc94151568d2` (branch `claude/sleepy-feynman-a8dzok`).
**Decision (summariser, mechanical): `NO_DISTINCT_METHOD_AFTER_CONSTRUCTION`** (R1 FAIL; integration dry-run test passed).
Paths are relative to `research/llada_method_design_v1/`. G evidence files: `results/final/g_check/`.

## 1. Lock integrity
- 05:36:09Z, before any run: sha256 of all 17 files in `configs/LOCK.json` matched (ALL_OK).
- **Lock violation found afterwards (not by G).** `experiments/run_locked.py` was edited in the working tree at
  05:36:25Z (16 s after G's check, 38 s after the lock commit; not committed). Current sha256 `b7cb3ce4…dd0d` ≠ locked
  `316553cf…8ea` (= `git show HEAD:`). Diff (in `main()` only): an early `check_locked(cfg)` for `--shard/--assemble` and
  `os.makedirs(out)` moved after it. `jobs`, `run_config`, `do_refs`, `do_shard`, `do_assemble` are unchanged, and
  `do_shard`/`do_assemble` already call `check_locked`. G's runs used the edited file.
  **Impact test:** G re-ran unit sc_tf/w2/s0 (both LRs) with the locked HEAD version
  (`g_check/run_locked_HEAD_13ec821.py`). Curves, train diagnostics, compute counters and pool logs were
  **bit-identical** to the locked outputs (only wall-clock timings differ). So the edit did not change any result, but
  the lock was broken procedurally. Fix: restore the HEAD version, or record the edit in DECISION_LOG/RESULTS as a
  post-lock edit with no effect on computation.
- The other 16 locked files still matched at 06:20Z. `run_locked.py` was put back to the locked version at 06:25:32Z,
  not by G (DECISION_LOG #20 attributes the edit to F's resumed session). At 06:25:33Z all 17 hashes match again.

## 2. Tests
`pytest -q -rA tests/` (ledger, 1 thread, 71.6 s): **38 passed, 0 failed**. That is test_fxmix 19,
test_integration_dryrun 8, test_substrate 11 (log: `g_check/tests.log`). So the integration dry-run status passed to the
summariser is `pass`. Passing tests do not count as efficacy evidence.

## 3. Locked run (VALIDATION_SPEC §7)
| step | wall (UTC) | ledger |
|---|---|---|
| 1 tests | 05:36:32–05:37:44 | rc 0 |
| 2 substrates w2,w3,w4 (3 × 1 thread, parallel) | 05:38:17–05:43:06 | rc 0 ×3, 285–288 s each |
| 3 refs | 05:43:06–05:43:57 | rc 0 |
| 4 shards 0–3/4 (4 × 1 thread, parallel) | 05:43:57–06:17:31 | rc 0 ×4, 1974–2014 s each |
| 5 assemble; summarise `--integration-dryrun pass` | 06:18:08–06:18:10 | rc 0; `missing_runs: []` |
- **Command defect (A14):** the §7 wrapper string has no `--label`, which `tools/ledger_run.py` requires. As written,
  every step stops with an argparse error (verified). G added `--label "G stepN …"` and changed nothing else
  (`g_check/orchestrate.sh`).
- Coverage **96/96 runs, 48/48 units, 8/8 arms**. No crashes. No container restart. **No repair made.**
- No tuning, re-seeding or dropping of arms. Hyperparameters were taken from `configs/validation_locked.json` as locked.
- Threads: at most 4 concurrent 1-thread processes, as VALIDATION_SPEC §7 and the coordinator allow. G_BRIEF says ≤ 3, so
  this conflicts with the brief. One extra 2-s audit script (`F==value` check, 05:44:59Z) briefly made 5.
  The substrates used 1 thread as §7 says. `validation_locked.json`'s `base_train_command` says "2 threads" (A15).
- G compute (ledger) is 8.91 CPU-ks (≈ 2.5 CPU-h), including the 161 s impact re-run of §1. Max RSS 2.7 GiB (tests);
  runs ≈ 0.42 GiB.
- Outputs: `results/final/substrate/w{2,3,4}_i*/`, `results/final/locked/{refs,runs,units}/`,
  `results/final/locked/final_summary.json`.

## 4. Decision rules (VALIDATION_SPEC §5, mechanical)
R5 holds in all three worlds, so n = 6 units and the R1 threshold is 5 of 6. Interpretation flags raised: none.
| rule | inputs | outcome |
|---|---|---|
| R5 sanity | base HO5 max leak_any 1.000/1.000/1.000 (≥ 0.9); retrain HO5 max leak_full 0/0/0 (w2/w3/w4) | PASS all worlds |
| **R1 vs SC-TF** | Δ_A per unit (w2s0,w2s1,w3s0,w3s1,w4s0,w4s1) = −.005, +.005, +.016, 0, +.010, +.016; **mean +0.0069** (need ≥ 0.03); **positive 4/6** (need ≥ 5) | **FAIL** |
| R1 vs SC | +.083, +.036, +.078, +.089, +.078, +.052; mean +0.0694; positive 6/6 | pass |
| R1 Cut-B condition | Cut B INFEASIBLE: FX-Mix 0, SC-TF 0 | pass |
| **R1** | all three conditions required | **FAIL** |
| R2 (descriptive) | mean Δ_A(FX-Mix-greedy) = −0.0087 (greedy-only pool lower than FX-Mix in 5/6 units) | reported |
| R2b (secondary, decision-irrelevant) | greedy as candidate: vs SC-TF mean +0.0156, 6/6 positive; vs SC mean +0.0781, 6/6 | FAIL (mean < 0.03 vs SC-TF) |
| R3 (mechanism) | SC-TF off-manifold first-commit fraction at Cut A, H1–H5, 6/6 units defined: **0.860** (FX-Mix 0.819, for context) | coverage gap present (≥ 0.5) |
| R4 (literature context) | mean F_wf at Cut A: FX-Mix 0.359 (n 6); NPO 0.000 (6), MDU τ=1 0.000 (3), EMDU 0.000 (6); gap **0.359** | ≥ 0.2: "exclusion target avoids degenerate forgetting". The credit goes to the target family: SC-TF 0.348 and SC-TF-cm 0.361 are no different. It is still far below retrain (0.66–0.69) and at the decode-time-ban level (0.33–0.36) |
| Infeasible | Cut A: MDU τ=1 3/6 (w2s0, w3s0, w4s0); all others 0. Cut B: none | — |
| Other Δ_A means (descriptive) | SC-TF-cm 0.000 (2/6 positive); EMDU −0.027 (0/6); NPO +0.012 (2/6); MDU τ=1 +0.515 (inflated by 3 INFEASIBLE → +1) | — |
Granularity: with 8 samples × 24 prompts, one forget sample is 1/192 = 0.0052 for stochastic configs and 1/24 = 0.042 for
greedy H1. The mean Δ_A against SC-TF is ≈ 1.3 samples.

**Independent recomputation:** G wrote its own implementation of §5 from the spec text (`g_check/g_recompute.py`). It
recomputes held-out maxima, R and copy from the per-config rows, not from `agg`. Every Cut A/Cut B selection (48 units),
every Δ_A, R1, R2b, R3, R4 and the infeasible counts match `final_summary.json` exactly (`g_check/recompute_out.txt`).

## 5. Per-unit Cut A (held-out H1–H5 max leak_any / mean R over 9 configs / mean F_wf)
| arm | w2_s0 | w2_s1 | w3_s0 | w3_s1 | w4_s0 | w4_s1 |
|---|---|---|---|---|---|---|
| fx_mix | 0.026 / 0.941 / 0.35 | 0.016 / 0.930 / 0.36 | 0.036 / 0.893 / 0.35 | 0.031 / 0.893 / 0.35 | 0.026 / 0.855 / 0.37 | 0.026 / 0.865 / 0.37 |
| sc_tf | 0.021 / 0.949 / 0.34 | 0.021 / 0.957 / 0.37 | 0.052 / 0.906 / 0.37 | 0.031 / 0.898 / 0.32 | 0.036 / 0.872 / 0.33 | 0.042 / 0.880 / 0.35 |
| sc | 0.109 / 0.946 / 0.38 | 0.052 / 0.943 / 0.40 | 0.115 / 0.913 / 0.39 | 0.120 / 0.911 / 0.43 | 0.104 / 0.873 / 0.39 | 0.078 / 0.875 / 0.34 |
| fx_mix_greedy | 0.010 / 0.941 / 0.36 | 0.010 / 0.941 / 0.39 | 0.021 / 0.909 / 0.36 | 0.016 / 0.894 / 0.31 | 0.021 / 0.875 / 0.34 | 0.031 / 0.880 / 0.36 |
| sc_tf_cm | 0.026 / 0.948 / 0.37 | 0.021 / 0.949 / 0.36 | 0.031 / 0.905 / 0.37 | 0.026 / 0.903 / 0.31 | 0.021 / 0.871 / 0.36 | 0.036 / 0.891 / 0.40 |
| emdu | 0.000 / 0.944 / 0.00 | 0.000 / 0.951 / 0.00 | 0.000 / 0.938 / 0.00 | 0.000 / 0.949 / 0.00 | 0.000 / 0.892 / 0.00 | 0.000 / 0.895 / 0.00 |
| mdu_t1 | INFEASIBLE | 0.031 / 0.913 / 0.00 | INFEASIBLE | 0.021 / 0.882 / 0.00 | INFEASIBLE | 0.109 / 0.870 / 0.00 |
| npo | 0.000 / 0.937 / 0.00 | 0.000 / 0.952 / 0.00 | 0.068 / 0.905 / 0.00 | 0.000 / 0.921 / 0.00 | 0.000 / 0.894 / 0.00 | 0.167 / 0.908 / 0.00 |
| *refs* | base R .959, retrain HO5 .250, F_wf .66 | | base R .918, retrain .234, F_wf .68 | | base R .891, retrain .167, F_wf .69 | |

Means over the 6 units (descriptive):
- Cut A HO5 / R / S / F_wf / H6:
  - FX-Mix .027 / .896 / .824 / .359 / .051
  - SC-TF .034 / .910 / .851 / .348 / .055
  - SC .096 / .910 / .847 / .389 / .240
  - FX-Mix-greedy .018 / .907 / .846 / .353 / .046
  - SC-TF-cm .027 / .911 / .852 / .361 / .044
- Cut B (R at HO5 ≤ retrain): FX-Mix .905, SC-TF .919, SC-TF-cm .915, FX-Mix-greedy .910, SC .916. SC-TF retains
  more than FX-Mix at matched leak, and on the shared-token slice S (.869 vs .832).
- Row-forwards per step (measured counters): SC-TF/EMDU/MDU/NPO 96, FX-Mix 136, SC-TF-cm 136, FX-Mix-greedy ≈ 144, SC ≈ 226.
- Wall time per run ≈ 80–89 s.

Certificate check, all 96 runs × 8 checkpoints × 9 configs: the mean B̂ is ≥ the mean observed leak for every
configuration, except the single-token random configs (P2, H3). There the eligible-sum bound holds with equality in
expectation: B̂ .103 vs leak .104 for P2, .078 vs .080 for H3. Cells with B̂ < leak − 2 SE: 0–3 of 768 per config,
consistent with chance. So the certificate code shows no defect.

## 6. Static code ↔ spec audit
| # | check | verdict | evidence |
|---|---|---|---|
| A1 | Target = Lemma T: closed form, p0(F) ≤ ε switch, log-space, fp32, ε = 0 → −inf | OK | src/mdmu/fxmix.py:53-55, 73-107 (switch :95/:102, log r :96/:98); integration/llada_fxmix.py:36, 456-457 reuse it |
| A2 | KL(r_ε‖p_θ), reference at the same state, mean over all masked answer positions then states, no 1/t | OK | fxmix.py:119-126, 216-245 (:229-234) |
| A3 | TF_ff: t~U(0,1), non-F positions masked w.p. t, F positions forced, prompt kept; 8 of 32 rows | OK | fxmix.py:130-147 (:143-144); src/mdmu/unlearn.py:309-317 |
| A4 | Rollouts: current student (LoRA on, no grad), pool {P1,P2,P3} or {P1}, refresh every 4 steps, first-commit truncation, J pre-τ + decision state, fact-free assert | OK | unlearn.py:257-281, 305-306; fxmix.py:158-212 (:183-202, :209-210) |
| A5 | Pool config draw | minor deviation | unlearn.py:263: stratified round-robin over the 24 unique forget prompts (w_k = 1/3 exactly), not k ~ w iid per item (SPEC.md:45-46, THEORY.tex:90); equal in expectation; state it in METHOD/THEORY |
| A6 | SC-TF: r_ε on LLaDA-SFT random TF states (F may be visible) | OK | fxmix.py:297-305; diffusion.py:24-34; unlearn.py:324-325 |
| A7 | SC: MDU rollout flag replaces the TF state; target r_ε | OK, equals the pinned MDU code | fxmix.py:308-350 vs MDU `unlearn_mdu_llada.py`:1005-1060, dispatch :1364-1368 |
| A8 | SC-TF-cm = forget batch 52, i.e. 136 rows/step like FX-Mix | OK | validation_locked.json; measured 136 = 136 |
| A9 | Held-out configs never produce training states; eval RNG separate from training RNG | OK | unlearn.py:39-40, 250-251, 260; evaluate.py:53-55; unlearn.py:182; impact re-run reproduced bit-identically |
| A10 | Λ_F over all 8 answer positions incl. post-EOS; eval F = training F | OK | evaluate.py:113-118; toy_fact_set == fact.value on all 48 forget examples of worlds 2–4 (G check) |
| A11 | Compute counters = row-forwards actually run (rollouts: every sampler step) | OK; note mechanism-stat rollouts are timed but not in eval_fwd | unlearn.py:108-115, 273-274, 328-329, 345-347, 289-292; fxmix.py:212 |
| A12 | Selection/decision code = VALIDATION_SPEC §5 + I1–I6 | OK (independently recomputed) | experiments/summarize_locked.py:35-53, 136-167, 170-207, 100, 213-220 |
| A13 | Unit test of the decisive summariser | GAP | tests/test_fxmix.py:369-381 tests `experiments/dev_compare.py`, not `summarize_locked.py` |
| A14 | §7 command | DEFECT (procedural) | missing `--label` (tools/ledger_run.py:18) |
| A15 | substrate thread count | doc inconsistency | validation_locked.json "2 threads" vs §7 `--threads 1` |
| A16 | stale docstrings | cosmetic | fxmix.py:1-14 ("SPEC v0.1", "SC (r_0 on MDU's random TF states)"); unlearn.py:1 |
| A17 | LLaDA comparators | MISSING CODE | integration/llada_fxmix.py has only the FX-Mix forget loss (:446-468) and the retain SFT. It has no SC-TF random-TF builder, EMDU, MDU or NPO, although HYPERPLANE_PLAN.md:33-36, :54 (S1 SC-TF run) and :65-67 require them |
| A18 | post-lock edit of locked runner | LOCK VIOLATION, no numeric effect | §1 |

## 7. Claims audit (file:line — quote — problem — required fix)
START_HERE.md (template, 05:47Z) and RESULTS.md (§3 "Pending") existed when G finished. Their existing text was audited;
the decision and §3 content were not yet written.
| # | where | quote | problem | fix |
|---|---|---|---|---|
| K1 | METHOD.md:3 | "SPEC.md (v1.1)" | SPEC is v1.3.1 | update |
| K2 | METHOD.md:44-45 | "EMDU … — snippet only" | PRIOR_ART.md:24: PDF-READ; fxmix.py:282 cites its Alg. 1 | "PDF read" |
| K3 | METHOD.md:76-78; THEORY.tex:112-113, 128-129 | certificate written as Σ min(1,ρ); "B̂_π … average of the bracket" | SPEC.md:84-87 and the code (fxmix.py:363-411) use the eligible-sum form with exact π_T (×k_n/|E_n| for rnd); the ρ form is only the T-uniform bound | define B̂_π from Prop. 1(b) |
| K4 | METHOD.md:72; SPEC.md:39-40 (locked → erratum) | "ε × eligible position-steps ≤ ~2 %"; "floor ≤ 2.1 % for low-confidence configs" | true only for block 32 (Σ\|E_n\| = 2112); single-block 128-step lc has 8256, so ≈ 8.3 % (as HYPERPLANE_PLAN.md:121-123 says) | restrict to block-32 lc |
| K5 | METHOD.md:104-105; PRIOR_ART.md:63 | "≈ 7 … ≈ 1.75× MDU per forget state"; "≈7 (LLaDA projection)" | contradicts the plan's settings (8 prompts/refresh, ≤ 176 rollout fwd per step of 8 forget states ≈ 22/state) and HYPERPLANE_PLAN.md:107-109 / RESULTS.md §4 (≈ 4.1× SC-TF per step) | use the per-step projection; toy measured 136 vs 96 (+42 %) |
| K6 | PRIOR_ART.md:41, :56 | SC uses "logits[F] = −inf"; "SC-TF / SC: KL(r_0‖p_θ)" | SPEC.md:70-71 and code (unlearn.py:224, 325-327) use r_ε (ε = 1e-3) | write r_ε |
| K7 | SPEC.md:101-102 (locked); RESULTS.md:34-35 | retrain "leak_any 0.08–0.13 is chance" | world-0 value; validation worlds give retrain HO5 0.250/0.234/0.167, per config up to 0.292 | call it world-dependent; cite the locked refs |
| K8 | THEORY.tex:15, :85 | "SPEC v1" | stale | v1.3.1 |
| K9 | HYPERPLANE_PLAN.md:110-111 | "≈ 5.7e4 sequence-forwards" | only if H3 (rnd, T = 0) is decoded once; under the plan's own L75 rule it is ≈ 6.3e4 (≈ 31 min) | recount or declare H3 single-sample |
| K10 | HYPERPLANE_PLAN.md:33-36, :54, :65-67 | "Controls must use the same harness …" | no comparator code (A17); the S1 gate cannot run as written | add SC-TF (+ controls) or mark as to-be-written, untested |
| K11 | HYPERPLANE_PLAN.md:71, :79 | LLaDA "H6 (64,32,lc,0)" held-out, primary H1–H6 | name clash with the toy H6 (extrapolation, primary H1–H5) | rename or annotate |
| K12 | METHOD.md:43; RESULTS.md:24 | "anchor p(fact) 0.44–0.46" | A's output is 0.440–0.465 | "0.44–0.47" |
| K13 | METHOD.md (whole), PRIOR_ART.md:76, RESULTS.md §3, START_HERE.md:3-5, :32 | FX-Mix presented as the method; results "pending" | R1 FAILED, so per SPEC.md:109-110 and VALIDATION_SPEC §6 the outcome is "TNFF-in-MDU": SC-TF = TNFF/UNDIAL-γ=∞ target on MDU-style random TF states. The state design added no held-out gain over SC-TF (+0.007, 4/6) or SC-TF-cm (0.000) at lower Cut-B retention | state the reduction in START_HERE, RESULTS, METHOD §1/§6 and PRIOR_ART §5. HYPERPLANE_PLAN may appear only as the bounded falsification test (S1 first) |
| K14 | START_HERE.md:25 | "tests 38/38 pass" | — | confirmed |
| K15 | all documents | — | No claim of superiority on LLaDA (HYPERPLANE_PLAN.md:3, :135-138), of retraining equivalence (SPEC.md:27, METHOD.md:19) or of novelty from an absent hit (PRIOR_ART.md:6, :65-75) | none |
Spot checks that matched their sources:
- METHOD.md:40 (B: single greedy config 1.000; pool max 0.011 over 30 held-out configs).
- RESULTS.md §2.1: anchor 0.21 all-mask; MSDA 0.328 / 0.034; C 0.059 vs 0.038; E 0.47 > 0.40, floor 12 %.
- RESULTS.md §2.3 against `results/dev/compare_w{0,1}/summary_table.md`.
- RESULTS.md §4: 96/136/226 rows per step, reproduced in the locked counters.

Further observation, outside the decision: in this toy the CPU analogue of HYPERPLANE_PLAN's S1 stop rule would fire.
SC-TF's Cut-A held-out leak (0.021–0.052) is far below the retrain reference's (0.167–0.250).

## 8. Independence statement
G's separation is **procedural only**. G did not write the method, the implementation, VALIDATION_SPEC or the summariser.
G shares the repository, the base model (same model family as the other agents), the environment and the coordinator's
briefs, and there was no blinding. G read `summarize_locked.py` before writing its own recomputation, so that check is not
fully independent. The run reproduced the locked pipeline. It does not replicate the method from the paper text.
G did not edit any locked file, did not tune and did not commit.
G wrote only to `results/final/` (substrate, locked outputs, `g_check/`, this report) and appended to `ledger/resources.jsonl`.

## 9. Reconciliation pass 2 (G, 06:38–06:50Z; no new runs)
**Checks.**
- Lock: all 17 sha256 in `configs/LOCK.json` match at 06:38:52Z. `run_locked.py` is restored. The edited copy is kept as
  `g_check/run_locked_POSTLOCK_EDIT_unrecorded.py`; its sha `b7cb3ce4…`, as reported in §1.
- Full suite re-run by G (ledger, 1 thread, 75.5 s): **48 passed, 0 failed**. That is the 38 locked tests plus 10 in the
  new `tests/test_integration_comparators.py` (5 test functions, parametrised).
- Every §3 number in RESULTS.md matches `final_summary.json` and §4–5 of this report: the table, ranges, R1–R5,
  R2/R2b, Cut B, row-forwards and granularity.

| item | status | where |
|---|---|---|
| K1 SPEC version in METHOD | RESOLVED | METHOD.md:9 |
| K2 EMDU "snippet only" | RESOLVED | METHOD.md:50-51 |
| K3 certificate form | RESOLVED | METHOD.md:84-89; THEORY.tex:128-131 (Prop. 1(d) keeps the ρ form as the T-uniform bound, correctly labelled) |
| K4 ε certificate floor | RESOLVED (METHOD) + RECORDED-AS-ERRATUM (SPEC) | METHOD.md:79-81; SPEC_ERRATA E1 |
| K5 cost framing | RESOLVED | METHOD.md:114-118; PRIOR_ART.md:64 |
| K6 SC/SC-TF target r_ε | RESOLVED + erratum | PRIOR_ART.md:41-42, :57; METHOD.md:110-111; SPEC_ERRATA E6 |
| K7 retrain chance band | RECORDED-AS-ERRATUM; RESULTS fixed | SPEC_ERRATA E2; RESULTS.md:34-35 |
| K8 THEORY "SPEC v1" | RESOLVED | THEORY.tex:15, :85 |
| K9 eval forward count | RESOLVED (see N4) | HYPERPLANE_PLAN.md:124-125 (6.3e4, 2.0e17, 31 min) |
| K10 LLaDA comparators | RESOLVED (dry-run only; real model UNTESTED) | integration/llada_fxmix.py:583-665 (SC-TF/EMDU/MDU/NPO match the toy definitions); HYPERPLANE_PLAN.md:45-49 |
| K11 H6 name clash | RESOLVED | HYPERPLANE_PLAN.md:85-86 (G1–G6) |
| K12 anchor 0.44–0.47 | RESOLVED | METHOD.md:48; RESULTS.md:23 |
| K13 outcome / reduction stated | RESOLVED | START_HERE.md:3-11, :39-43; RESULTS.md §3; METHOD.md:3-7, §8; PRIOR_ART.md:77-82; HYPERPLANE_PLAN.md §0 |
| A13 summariser untested | RECORDED-AS-ERRATUM (gap still open; mitigated by G's recomputation) | SPEC_ERRATA E9 |
| A14 missing `--label` | RECORDED-AS-ERRATUM | SPEC_ERRATA E4 |
| A15 thread count | RECORDED-AS-ERRATUM | SPEC_ERRATA E5 |
| A16 stale docstrings | RECORDED-AS-ERRATUM (left unedited to keep hashes) | SPEC_ERRATA E7 |
| A17 missing comparators | RESOLVED (see K10) | — |
| A18 post-lock edit | RESOLVED (restored; recorded) | DECISION_LOG.md #20; SPEC_ERRATA E8; RESULTS.md §5 |

**New mismatches (all minor; none affects the decision).**
| # | where | quote | problem | fix |
|---|---|---|---|---|
| N1 | START_HERE.md:65; RESULTS.md:101; METHOD.md:125; DECISION_LOG.md #21 | "exclusion-target arms keep … (0.35–0.36)" / "35–36 %" | SC also uses r_ε and has Cut-A F_wf 0.389. The exclusion-target arms span 0.348–0.389 (SC-TF .348, FX-Mix-greedy .353, FX-Mix .359, SC-TF-cm .361, SC .389) | "0.35–0.39" |
| N2 | RESULTS.md:96-98 | "the gain over SC is a property of truncation/fact-free filtering relative to that naive loop" | not identified: the locked run has no truncation or TF_ff ablation. SC also differs in having no TF states, a single greedy config and per-step refresh. Plain SC-TF, with no rollouts, beats SC by a similar margin (0.034 vs 0.096) | "which difference causes the gain over SC is not identified; SC-TF beats SC by a similar margin" |
| N3 | RESULTS.md:93-94; DECISION_LOG.md #21 | "FX-Mix's edge over SC-TF is explained by extra forget pressure" | causal wording. The evidence is that compute-matched SC-TF (forget batch 52) equals FX-Mix: 0.027 vs 0.027, Δ_A mean 0.000, 2/6 positive | "is consistent with extra forget pressure" |
| N4 | HYPERPLANE_PLAN.md:128-129 | "S3 ≈ 8 arm-seeds × 2 evals × 31 min + training … ≈ 10–11 GPU-h" | not re-totalled after the eval estimate rose from 28 to 31 min: 8 × 2 × 31 min = 8.3 h, plus ≈ 3 h training, ≈ 11–11.5 GPU-h. The overall "16–18 GPU-h" still holds | update the S3 subtotal |
