# AMENDMENT_01 — scope and protocol corrections to `cpr_direct_agreement_controls_v1`

Dated 2026-10-19 (UTC, task `cpr_execution_recovery_v1`). This amendment corrects the *description* of the old study. It does not change
its terminal status: **`ENTRY_NOT_PASSED/CLOSED` is retained**, every old artifact is preserved byte for byte, and no historical failure
becomes a pass. The old payload/tag target remains `50c8900b00449839e8ae90e917af58a4ef7b7628`; its release `cpr-direct-agreement-controls-v1` keeps all four assets.
Continuation work is in `results/cpr_execution_recovery_v1/` (a new study with its own ledger; the old ledger was not rewritten and no new 24-hour allowance was granted).

## Corrections

1. **Detector qualification failed, with exact counts.** Primary RT-DETR failed on **oven 15/20**; secondary DETR failed on **refrigerator 15/20** and **couch 14/20** (frozen floor ≥16/20). These failures are retained and the detectors are not described as accuracy-qualified.
2. **Strict stopping was not achieved.** Qualification failed at **18:41:47Z**, yet another GPU trajectory was launched at **18:47:57Z** (followed by the replay and post-hoc numerical tests). The report's statement that execution stopped at the entry gates is corrected: the *comparison* never started, but GPU work continued after the first failed entry check, so strict stopping compliance was **not** met.
3. **Repair 1 changed an acceptance parameter.** It changed the fixture acceptance `min_grad_frac` from **0.01 to 0.0** (not a neutral harness fix; it altered what the synthetic gate accepted).
4. **What "exact agreement" was.** The exact native/hooked/pristine agreement was a **CPU replay at index 0**, not a GPU gradient agreement. GPU gradients were **not captured**, and **both captured indices (0 and 7) preceded the first optimizer update**, so the student equalled the target there and the replay was a weak discriminator.
5. **Post-hoc numerical work was additional testing.** The post-hoc Torch/backward/FP64 execution was additional numerical testing, not "metadata-only" work.
6. **Accounting scope.** Recorded GPU (725.4364926815033 device-s) and CPU (2,037.790186 core-s) totals are **supervised subtotals**. Whole-task CPU time and peak storage were **incomplete**, including **unsupervised CPU unit-test runs made before the supervisor existed**. An OS thread count (264) does **not** establish active BLAS concurrency.
7. **Base-model integrity claims were narrower than worded.** Aggregate base statistics are **not** hashes of every weight byte, and meta-device skeleton restoration is **not** full-model forward parity.
8. **Repair-time figures were estimates.** The "10 and 15 = 25 of 30 minutes" figures were declared estimates without wall timers, not measurements; the allowance was **30 minutes per repair cycle** (not 30 in total).
9. **Licensing verification was not completed before the old publication.** The old report stated licensing terms (FLUX.1 [dev] non-commercial; no base weights redistributed) without a completed verification of the official redistribution terms before publication.

## What did and did not change
- Terminal status, all counts (0 generated images, no comparison), release assets and tag target: unchanged.
- The old study's FINAL_REPORT.md now begins with a dated correction notice; its original body follows verbatim. EVIDENCE_INDEX.md carries the same pointer to the current record.
