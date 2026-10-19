# PROGRESS (append-only; resume from here after compaction)
- 2026-10-19T19:29Z task received; no prior recovery work existed. Repo HEAD 0a40315. Work dir created.

## 2026-10-19T19:41:23Z supervisor ledger written + smoke-tested (CPU-only jobs; test ledger deleted)
- code/cprx/ledger.py, scripts/sup.py. Smoke: ok/timeout-kill/exit-fail charged; CPU pool exhaustion refused; supervisor kill -9 -> next launch reconciled+killed orphan+charged.
- Real ledger (artifacts/ledger) not yet created; opens with 725.4364926815033 GPU-s (eng), 2037.790186 CPU-s.

## 2026-10-19T19:48:49Z USER UPDATE: bounded blinded visual audit (VISUAL_AUDIT_ADDENDUM_01) received mid-turn
- Arrived while engineering was in progress; no test-model outputs, scores, images from this study existed or had been inspected by anyone at arrival (only the synthetic GPU gate had run).
- Plan: commit addendum (sampling rule + rubric + order/mapping commitment) BEFORE any judgment; run after test images exist; does not alter the primary comparison.
- Added synthetic GPU gate script scripts/run_synth_gate.py; build_manifests.py rewritten (new case IDs x.*); run_trajectory.py train() rewritten (spec file + Probe).
- 19:52:24Z eng1 (native 132 st / cpr 44 st, seed 6901, probe idx [0,11,24]) FAILED gate checks (not model failures): (a) my optimizer check counted frozen params in the native optimizer group (1971 params in group, only 152 require grad, 152 state tensors) -> check rewritten to 'requires-grad params == LoRA set'; (b) grad-scale consistency used atol=0 elementwise rtol 1e-6 -> replaced by frozen RMS/max-abs criteria. Probe agreement (graph vs hooked/native source/FP64) PASSED at all captures. eng1 wall 183s native (peak host RSS 3.1 GiB). Rerun as eng2.
- 19:55:42Z eng2: native 132 st all checks PASS (wall 184s, peak RSS 1.6 GiB). CPR 44 st: only probe idx0 failed: reduction_bitwise recomputed on CPU (CUDA scalar divide is reciprocal-multiply, 1-ulp) -> now same-device recompute. Added saved-vs-restored key/shape check in load_adapter, AxialDomainViolation -> exit 7 failed-row handling. eng3 (native/cpr) + techbatch (4 dev technical images + score) launched.

## 2026-10-19T20:06Z FREEZE_01 prepared: eng3/eng4 gates PASS (all six arms on real trajectories); protocol/addendum/sample written; test_analysis re-run on CPU outside supervisor (gap). Next: commit+push freeze, then final validation valid.* (6 arms).
- 2026-10-19T20:08Z FREEZE_01 pushed as c8d58bf8224f9d9ab8eca7092c3b407a1faea9a2; launching final validation valid.* (6 arms)
- 20:11Z final validation valid.* PASS (6 arms; fresh seeds 6301-6306; indices native [0,12,121], others [0,12,33]; freeze commit was c8d58bf at 20:06:05Z — earlier 20:08Z note was approximate). Launching dev pipeline (dev_m0, dev_train, dev_select).
- 2026-10-19T20:50:41Z dev pipeline done: selection lock written (configs/selection_lock.json); committing lock BEFORE any test-model output; then ho_m0/ho_run.
