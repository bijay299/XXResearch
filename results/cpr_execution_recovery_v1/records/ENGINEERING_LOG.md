# Engineering log (cpr_execution_recovery_v1) — every attempt, failure, patch and allocation

Engineering data below is SEEN data (engineering seed 6901, probe indices [0,11,24]); none of it is final-validation evidence.
GPU seconds are ledger-charged held seconds (`artifacts/ledger/attempts.jsonl`). The ledger opened with the carried 725.4364926815033 GPU-s.

| # | UTC (approx.) | job | device | GPU-s | result | cause / action |
|---|---|---|---|---|---|---|
| 0 | 19:41 | supervisor smoke test (CPU only, test ledger deleted) | – | 0 | ok | `ledger.py`/`sup.py` written; kill/timeout/reconcile/pool-exhaustion tested. Not charged to the real ledger (CPU-only, unsupervised-by-real-ledger; listed under enforcement gaps). |
| 1 | 19:49 | `eng.synth_gate_gpu_a1` | 0 | 5.6 | ok | Full-dimensional (d=65,536) synthetic FP64 gate on GPU for all six objectives, plus the trainer's native loss block. PASS, one disclosed branch-ambiguity case. |
| 2 | 19:52 | `eng1.train.native_gem` (132 st) | 0 | 182.9 | FAIL exit 3 | My gate-check bugs, not model/objective failures: (a) optimizer check counted the 1,971 frozen params in the native AdamW group (only 152 require grad, 152 state tensors) and was rewritten to "requires-grad params == LoRA set, 152 state tensors, frozen-base digest unchanged"; (b) grad-scale consistency used atol 0 / elementwise rtol 1e-6 and was replaced by the frozen RMS/max-abs criteria. Probe agreement (graph vs hooked, native source, FP64) PASSED at all captures. |
| 3 | 19:52 | `eng1.train.cpr` (44 st) | 1 | 73.5 | FAIL exit 3 | same two check bugs |
| 4 | 19:55 | `eng2.train.native_gem` | 0 | 183.7 | PASS | all checks pass |
| 5 | 19:55 | `eng2.train.cpr` | 1 | 73.4 | FAIL exit 3 | only `reduction_bitwise` at index 0: recomputed `(gamma*loss)/n` on the CPU; CUDA scalar division is a reciprocal-multiply (1 ulp). Fixed: same-device recomputation. Also added saved-vs-restored key/shape verification in `load_adapter`, AxialDomainViolation -> exit-7 failed-row handling. |
| 6 | 19:57 | `gen.tech` + `score.tech` | 0 | 35.6 + 7.5 | ok | 4 DEVELOPMENT technical images from the eng2 native terminal adapter; 4 technical images charged. ~5 s/image generation, 8 s scoring held. |
| 7 | 19:58 | `eng3.train.native_gem` / `eng3.train.cpr` | 1 / 2 | 184.7 / 75.7 | PASS / PASS | fixed checks pass on both arms |
| 8 | 20:03 | `eng4.train.{centered_radial,truncated_anchor,anchor_only,axial_margin}` (44 st) | 0–3 | ~76 each | PASS ×4 | control arms run through the same probes on real trajectories (hooked vs native-source vs FP64 agreement at post-update indices) before freezing |

Patches to the pinned GEM source (all attributed, additive, diffs in `patches/`): P1a/P1b/P1c (native loss block moved verbatim into module-level `_cpr_native_loss_block`; live-graph probe hook before `loss.backward()`), P2 (preview-decode skip with counter), P3 (step hook), P4–P9 (see `patches/ALL_PATCHES.diff`).

Stale files removed before freeze (never executed in this study): `code/cprx/project.py`, `scripts/run_cpu_job.py`.
No objective was changed and nothing was tuned for efficacy. Engineering GPU total at this point: see `artifacts/ledger/state.json` (well under the 4 GPU-h engineering cap).
