# Modern-model FLUX.1-schnell DP/UCE reproduction (cassette player / chain saw)

First reproduction of the released DP and UCE FLUX.1-schnell adaptations from
`czhang024/CVPR_DouleProjection` (pinned commit
`3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`), with a conditional sequential
extension that did not run because neither method cleared the pre-registered
eligibility gate.

| start here | for |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | observed recognition counts, the gate arithmetic that blocked the sequential extension, retention, and the 4-case qualitative rendering-panel observation |
| [`RUN_MANIFEST.md`](RUN_MANIFEST.md) | source/model commit hashes, environment, hardware, native settings — frozen before any edit or generation |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | exact path, byte count and sha256 of every file published here |
| [`sequential_diagnostic_execution/FINAL_REPORT.md`](sequential_diagnostic_execution/FINAL_REPORT.md) | **EXECUTED result** of the revision-02 protocol — chain saw -> parachute, A->B. Outcome **STOPPED_AT_FROZEN_GATE**: `C=0, U=0` on the historical target for both methods (no detector-discovered relapse, not an absence certificate), parachute deleted 100->0 directly and sequentially, DP passes every retention bound while UCE fails the cumulative one; measured preservation residual 0.42% (DP) vs 38.7% (UCE) with no relapse difference. 4,000 of 6,000 images; neither batch comparator triggered |
| [`sequential_diagnostic_protocol/PROTOCOL.md`](sequential_diagnostic_protocol/PROTOCOL.md) | **the protocol that was executed, revision 02.** A prospective chain-saw -> parachute sequential-erasure diagnostic built from this reproduction's existing tables; no model/encoder/GPU work ran to produce it or either correction pass. Correction mappings: [`REVISION_01.md`](sequential_diagnostic_protocol/REVISION_01.md) (design/mechanism corrections), [`REVISION_02.md`](sequential_diagnostic_protocol/REVISION_02.md) (decision-rule and numerical corrections, including the authoritative ambiguity-aware decision table). Execution requires a separate future instruction |

## What is here

- **`evidence/`** — per-image classifier predictions for all 5,000 generated
  images (`{M0,DP_A,DP_B,UCE_A,UCE_B}_per_image.csv`), the recognition/gate/
  retention computation (`recognition_summary.json`, `gates_and_retention.txt`),
  the frozen 1,000-case manifest (`full_case_manifest.csv`), the four native
  edit logs, the pip-frozen dependency list, the downloaded-weight hash
  table, the rendering-panel file hashes, and the exact scripts used
  (`code/`).
- **`images/`** — the fixed 20-case rendering panel (variation 0 of the
  first two CSV rows per class) at full 512x512 resolution, for all five
  endpoints (M0, DP_A, DP_B, UCE_A, UCE_B).

## What is NOT here (host-only, see `RUN_MANIFEST.md` for locations/hashes)

The full 5,000-image generation output, the four edited `.safetensors`
checkpoints, the FLUX.1-schnell model snapshot, the isolated venv, and the
pinned (unmodified) upstream source tree — all on
`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/` on host "hyperplane".
