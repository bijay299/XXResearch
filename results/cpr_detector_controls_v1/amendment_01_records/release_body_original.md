# CPR detector-defined controls screen v1 — entry-gate artifacts

**Terminal status: `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`**

Production never started. **No development training, no held-out training, no
six-arm comparison, and ZERO images generated.** No scientific decision is
available or claimed; **CPR's detector-defined retention behaviour remains
untested.**

**The blocker.** The required gate — native path with loss dispatch
disabled/enabled must agree under controlled inputs/RNG — was frozen with an
**exact-equality** criterion on the trained native PEFT adapter hash. It
failed (`c05be7b1…` vs `ed373817…`). The patch is inert by code inspection and
deterministic algorithms were never enabled, so GPU non-determinism is the
likely cause — **but that was not verified**: the OFF-vs-OFF control was not
run, because the single technical GPU retry was already spent and loosening a
frozen criterion after seeing the outcome is forbidden. This is a
gate-specification defect.

**What passed first:** categories by metadata alone (dev `stop sign`; held-out
`bench`, `fire hydrant`); **both evaluators QUALIFIED, 7/7 classes** at
≥16/20 positives and negatives; CPU implementation gate 64 rows / 0 failures;
native GEM + **native PEFT** bring-up (76 `qk_dual` modules, base frozen,
save/reload); budget projected 8.16 of 16 usable GPU-hours.

Report: [`FINAL_REPORT.md`](https://github.com/bijay299/XXResearch/blob/00bc8ae0a3d98640cb49cbbe616d2101e827e64f/results/cpr_detector_controls_v1/FINAL_REPORT.md)

## Assets

| Asset | Contents |
|---|---|
| `entry_gate_peft_adapter_init.pt` | native PEFT `qk_dual` adapter at initialisation (76 modules). Entry-gate artifact, **not a trained erasure model** |
| `qualification_raw_detections.tar.gz` | both detectors' raw boxes/scores over all 280 qualification pairs |
| `allocation_ledger.tar.gz` | durable allocation ledger + append-only journal |
| `frozen_configs.tar.gz` | frozen case matrix and category/qualification selection |
| `RECONSTRUCTION.md`, `SHA256SUMS` | restore instructions and hashes |

**Licensing:** no FLUX base weights, no detector weights and no COCO
photographs are redistributed — COCO is published as image ids plus
provenance. No credentials.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01BrMSBQygJfiRi5LiiGkskj
