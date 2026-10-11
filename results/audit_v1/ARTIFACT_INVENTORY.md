# ARTIFACT_INVENTORY — what exists, what is reusable, what is gone

Audit of everything the two-seed snapshot `a9de625` left on disk, and what it
permits the next experiment to reuse. Paths are on **hyperplane**; large assets
live under `/data/bijaypandey/cuig_pilot/` and are not in the repository.

Compiled 2026-10-09. No job was running and all four A100s were idle during
this audit; nothing was interrupted, retrained or regenerated.

---

## 1. Final checkpoints — all 10 present

One `delta.bin` per training request, holding **absolute updated values** for
the 32 trainable kv-xattn tensors (19,169,280 parameters, 76,686,190 bytes each,
~73 MiB; 730 MiB total). Base SD-1.5 weights are unmodified and separate.

`/data/bijaypandey/cuig_pilot/seq_pilot/models/seed<N>/<ckpt>/delta.bin`

| checkpoint | sha256[:12] | train s | checkpoint | sha256[:12] | train s |
|---|---|---|---|---|---|
| seed17/MA | `3cb72ee162b9` | 828.6 | seed29/MA | `fbf49199a306` | 829.4 |
| seed17/MAB | `63c9a8232cfd` | 828.5 | seed29/MAB | `d9f28f0fc6b8` | 828.8 |
| seed17/MAB_L2 | `8ee7d56f62b0` | 831.3 | seed29/MAB_L2 | `c4f3ecdcb945` | 832.7 |
| seed17/MAC | `e7b6324d3209` | 828.3 | seed29/MAC | `45e910d2fb11` | 831.2 |
| seed17/MAC_L2 | `c39e03ff85a4` | 831.7 | seed29/MAC_L2 | `06b62d00a55b` | 832.8 |

All ten sha256 values are distinct. Both `MA` parents verified loadable with
`max_abs_deviation_after_load = 0.0`, and every L2-SP arm recorded
`l2sp_reference_check.max_abs_deviation_from_parent = 0.0`, i.e. the
regularisation reference was asserted equal to MA before the first optimizer
step.

## 2. Intermediate checkpoints — **none exist**

Every run was launched with `eval_interval: null` and no periodic dump. There
is **no checkpoint at any step between 0 and 1000** for any of the ten runs.

Searched for and absent: optimizer state, LR-scheduler state, RNG/generator
state, gradient-accumulation state, `checkpoint-*` or `step_*` directories,
any `.pt` file. The only per-run state is the final `delta.bin` plus the JSON
report.

**Consequence.** An early-stopping or smaller-update control cannot be built
from existing artifacts. It requires a new trajectory. This is the single
binding constraint on the next experiment and is costed in
`docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md`.

## 3. Training logs and reports — complete

- `models/seed<N>/<ckpt>/train_report.json` — 10 files. Each carries the full
  effective hyperparameter set, `upstream_argv`, parent verification, the
  L2-SP reference check, an `l2sp_trace` summary (`steps_recorded`,
  `raw_first`, `raw_last`, `weighted_last` — see the note below: non-zero only
  on regularised arms), runtime, peak memory and the GPU UUID. Copied into the repo at `results/seq<N>/train/<ckpt>.json`.
- `seq_pilot/logs/` — 4.4 MB of stdout/stderr per stage, including
  `orchestrator.log` and `seed29_orchestrator.log`.

The `l2sp_trace` is a **summary, not a per-step series**: first/last raw value
and the weighted last value only. A per-step regularisation curve is not
recoverable from it.

> **Legacy metadata gap, found by AUDIT-01b and reported rather than worked
> around.** `l2sp_trace` is a *regularisation-loss* trace: it is populated only
> when `l2sp_weight > 0` and is legitimately `steps_recorded: 0` on all six
> unregularised runs (MA, MAB, MAC in both seeds). So for an unregularised run
> **this schema records no direct count of completed optimizer steps** — only
> `iterations_requested`, which is an input rather than an outcome, and the
> runtime-implied estimate `train_seconds / seconds_per_optimizer_step`.
> `validate_stage.py` therefore treats the trace as step evidence only for
> regularised arms and says so in its output. **Recommendation:** the next
> trajectory should record an explicit `optimizer_steps_completed` field. No
> migration of the existing reports was performed.

## 4. Per-image predictions — complete, 11 distinct evaluations

`seq_pilot/eval/<ckpt>/detections.jsonl` (seed 17) and
`seq_pilot/eval_seed29/<ckpt>/detections.jsonl` (seed 29). Every file has
exactly 280 rows with full per-detection boxes, scores and labels, so rates at
any threshold are recomputable without re-running the detector.

| | slots | rows | images on disk |
|---|---|---|---|
| seed 17 (`eval/`) | 6 | 1680 | 1680 + 8 reload-check |
| seed 29 (`eval_seed29/`) | 6 | 1680 | 1400 |
| **distinct** | **11** | **3080 generated** | **3088 jpg total** |

`eval_seed29/M0` is a **symlink** to `eval/M0`; the two M0 detection files are
byte-identical (md5 `df04381d336465cd47e332234775e64d`). M0 is therefore one
evaluation set appearing under both seed labels — 11 distinct evaluations and
3080 generated images, not 12 and 3360. Image storage: 266 MB + 159 MB.

Also present: `image_report.json` and `detect_report.json` per checkpoint, each
with `complete: true`, `expected_images: 280`, measured `seconds_per_image` and
peak memory.

## 5. Caches and fixed assets — reusable as-is

| asset | path | notes |
|---|---|---|
| Frozen evaluation manifest | `seq_pilot/eval_manifest.json` | 280 records, sha256 `0020c81c4a4d…`, `frozen_before_any_editing: true`, 0 collisions / 0 near-duplicates against 999 training strings |
| Anchor cache, horses | `seq_pilot/anchor_caches/Horses/` | 200 images + prompts, `anchor_seed: 17` |
| Anchor cache, flowers | `seq_pilot/anchor_caches/Flowers/` | 200 images + prompts, `anchor_seed: 17` |
| Calibration set | `seq_pilot/calib/` | 6.1 MB; M0 positive rates and negative controls |
| No-op reload check | `seq_pilot/eval/_reload_check/` | 8/8 regenerated images byte-identical |
| Base SD-1.5 | `Generators_substitute/sd-v1-5` | unmodified backbone |
| Detector | COCO `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` | sha256 `258fb6c638b1…`, in `torch_home` |

Anchor caches total 17 MB and are shared **byte-identically across both
training seeds and every matched arm**. The anchor-generation seed is 17 and is
independent of the training seed — a point the report generator previously
rendered as a bare "(seed 17)" inside the seed-29 report.

## 6. Can the trajectories be reproduced or continued?

**Continued from an intermediate step: no.** No optimizer, scheduler or RNG
state was saved, and no intermediate weights exist. Resuming a run at step 500
is impossible for every one of the ten runs.

**Re-run from scratch: possible, bit-exactness unverified.** `--seed <N>` is
passed through to the upstream trainer and every effective hyperparameter and
the full `upstream_argv` are recorded, so a re-run is fully specified. Whether
it reproduces the same weights bit-for-bit has **never been tested**: no run
was ever repeated under a fixed seed, and no determinism flags
(`torch.use_deterministic_algorithms`, cuDNN determinism) appear in the
training path.

> **Three distinct claims, not to be conflated** when that check is eventually
> run. (1) *File hashes equal* — byte-identical serialisation; a mismatch can
> come from serialisation order or metadata alone and is **not** evidence of a
> training difference. (2) *Tensors numerically equal* — the weights agree to a
> stated tolerance; this is what a reproducibility question is about.
> (3) *Training equivalent* — stronger than either, and established by neither.
> `scripts/seq/compare_checkpoints.py` answers (1) and (2) on CPU, per tensor
> (max abs, mean abs, relative Frobenius), and explicitly refuses (3).
>
> For calibration: the two existing seed-17 and seed-29 `MAB` checkpoints differ
> by max abs **2.86e-03** and relative Frobenius **2.69e-02** — a measured
> size for "same configuration, different training seed" on these tensors.

What *is* verified is **inference** determinism: the no-op reload check
regenerated 8/8 images byte-identically through the real checkpoint-loading
path. That covers generation, not training.

> A re-run of an unregularised arm with checkpoint dumps gets a free
> reproducibility test: compare its final `delta.bin` sha256 against the value
> tabulated in §1. The protocol records this as a secondary check, not as an
> assumption.

## 7. Absent, and not reconstructed

Recorded rather than filled in. Nothing below was estimated, inferred or
substituted from any other quantity.

| missing | consequence |
|---|---|
| Intermediate checkpoints (all 10 runs) | early-stopping / smaller-update controls need a new trajectory |
| Optimizer, scheduler and RNG state | no run can be resumed mid-trajectory |
| Per-step L2-SP regularisation series | only first/last summary values available |
| Per-step loss curves as data | present in logs as text, never parsed into a series |
| Any L2-SP coefficient other than 25000 | one tested value; no coefficient search was run |
| Human annotation | none performed; see §8 |
| UnlearnCanvas assets, generator, classifiers | blocked; no UA/IRA/CRA, no baseline reproduction |
| Blinded grids for seed 29 | only seed 17 had grids built (`analysis/seed17/grids/`) |
| A direct completed-optimizer-step counter for unregularised runs | step completion is inferred from a runtime cross-check only (see §3) |
| Development prompt split | the pilot's 280-prompt set was the *only* evaluation set and has now informed hypotheses, so it is exploratory from here on |

## 8. Human-annotation assets

- **Pre-existing (seed 17 only):** `analysis/seed17/grids/blinded/` — five
  per-category grid JPEGs with shuffled rows, an empty `annotation_sheet.csv`
  and a separate key. Row-level blinding across checkpoints, 2 generation seeds.
- **Built by this audit:** `seq_pilot/annotation_v1/packet/` — 236 per-image
  items (176 stratified-random + 60 enriched diagnostic) across cat, bird, dog
  and sandwich, both seeds, with the key stored outside the packet at
  `seq_pilot/annotation_v1/key/`. See
  `results/audit_v1/annotation_packet/README.md`.

**No human annotation has been performed.** Both label sheets are empty by
design. Every visual statement in the current reports is AI inspection and is
labelled as such; it is not human review and must not be cited as validation.

## 9. Disk and repository footprint

| | size |
|---|---|
| `seq_pilot/` total | 1.2 GB |
| └ final checkpoints (10 × 73 MiB) | 730 MiB |
| └ evaluation images | 425 MB |
| └ anchor caches | 17 MB |
| └ logs | 4.4 MB |
| new annotation packet | 26 MB |
| `/data` free | 1.4 TB of 14 TB |

Committed to the repository: JSON reports, per-image CSVs, rate tables,
contrasts, figures, the empty annotation sheet, the frozen draft manifests, and
the checkpoint/generation contracts. **No weights and no image archives are
committed**, by `.gitignore` and by this audit's own choice.

### Contracts and validation added by AUDIT-01b

| file | purpose |
|---|---|
| `configs/checkpoint_contract.json` | the 32 kv-xattn tensor names, shapes and dtypes, 19,169,280 params, expected file size — derived from the 10 saved checkpoints, which all agree |
| `configs/generation_settings.json` | the evaluation-generation settings every checkpoint of a comparison must share — derived from the 12 pilot image reports, which all agree |
| `scripts/seq/validate_stage.py` | contract validation for a training or evaluation stage, replacing existence/row-count checks |
| `scripts/seq/compare_checkpoints.py` | CPU tensor-level checkpoint comparison, keeping hash / tensor / training claims separate |

**All 10 training runs and all 12 evaluation slots were revalidated on CPU
against these contracts and pass.**

### Added by AUDIT-01d

| file | purpose |
|---|---|
| `configs/legacy_training_artifacts.json` | the **exact ten** saved artifacts permitted a limited completed-step exception, each by `train_report.json` sha256 **and** `delta.bin` sha256, with provenance. Replaces the seed-number list, which also exempted new runs using seeds 17/29. All ten in-repo report copies verified **byte-identical** to the saved reports |
| `configs/base_model_contract.json` → `identity_policy` | the exact launch-time base-model identity policy and its limits: what the cheap identity covers, what it does **not** (weight contents), and that the full weight digest is checked only under `--verify_base_model_weight_sha` |
| `scripts/seq/legacy_artifact_policy.py` | decides the policy for one artifact from its content (`decide`), protects registered checkpoints from being retrained over (`protect`), builds the registry (`freeze`), re-checks it (`verify`) |
| `scripts/seq/inspect_training_logs.py` | bounded CPU inspection of the saved training logs and their linkage to the registered checkpoints — **not** a completed-step counter |
| `scripts/seq/freeze_bootstrap_grouping.py` | derives and freezes the bootstrap resampling grouping from the actual manifests |
| `results/audit_v1/draft_manifests/bootstrap_grouping.json` | the frozen grouping: scene-cluster unit, category strata, 10 000 draws, recorded RNG seed, measured cross-cluster template dependence |
| `results/audit_v1/training_log_inspection.json`, `TRAINING_LOG_INSPECTION.md` | what the terminal progress logs do and do not establish |
| `scripts/seq/test_image_provenance.sh`, `scripts/seq/testlib/fake_models/` | CPU regressions for image-reuse provenance through the real launcher and the real generator, with fake torch/diffusers |

**The saved pilot was not modified.** No checkpoint, report, image, detection or
log was edited, moved or deleted in this round; the registry and the log
inspection only read and hash them.

### Added by the Amendment 01 closeout (2026-10-10)

| file | purpose |
|---|---|
| `scripts/seq/diag_analysis.py` (repaired) | re-applies protocol §5's **reference gates** on the frozen test set alongside the match, and reports **three separate** fields — `test_matching_check` (tolerance agreement), `test_eligibility_check` (each arm's own gates), `test_protocol_validity` (both). Where validity fails, every protocol-level claim is withheld with `withdrawn_claims` populated, while `interval_*` keeps the arithmetic facts and the contrast survives as descriptive |
| `scripts/seq/test_diag_test_conditions.py` | **59 CPU checks, 0 failures.** Gate boundaries; **arms close but both under-suppressed** (tolerance passes, eligibility fails, no claim available, interval retained); a **valid positive** case where the §7 table does apply; a residue-only failure; and **source-bound** assertions against the committed `analysis_test.json` |
| `scripts/seq/build_paired_annotation_packet.py` | builds the **prescribed 180-item paired** audit from existing images — 2 seeds × {cat,dog,bird} × 10 tuples × 3 arms. Reads only `image_report.json` and the frozen manifest, never `detections.jsonl`, so detector independence is structural; declared sampling seed; labels asserted empty; key and overlap report written outside the packet; a blinding probe aborts the build on a leak |
| `results/audit_v1/annotation_packet_paired180/` | that packet's key-free metadata: empty sheet, instructions, manifest, overlap summary, README |
| `scripts/seq/build_amendment01_evidence.py` | completes the evidence archive: supplies `unattended.log` checked against the identity the published manifest recorded, adds the raw development and frozen-test detector rows with their stage reports as deterministic archives, preserves every analysis version, recomputes the device-hour ledger without altering it, and records access limitations explicitly |
| `results/audit_v1/amendment01_evidence/` | 15 hash-listed payloads, 0 access limitations; `MANIFEST.published_at_7e01fbd.json` keeps the earlier listing verbatim |
| `results/audit_v1/AMENDMENT_01_CLOSEOUT.md`, `CLAIM_CORRECTIONS_V5.md` | the one-page closeout, and every withdrawn claim with what replaced it |
| `docs/research/NEXT_SCREEN_PROPOSAL.md` | the next scientific screen — CPU-only, **not authorised to run** |

**Nothing was reselected, retrained or re-evaluated**, no tolerance or gate was
changed, the frozen test set was not touched again, and both step-70 checkpoints
and all run outputs are preserved. The only new computation is a CPU re-analysis
of the existing detector rows; the two superseded analysis records are kept
verbatim beside the current one.

### Added by the closeout publication round (2026-10-10)

| file | purpose |
|---|---|
| `scripts/seq/audit_packet_blinding.py` | **attacks** a blinded packet using only its committed key-free records, two channels: identifier enumeration (`sha256(salt \| image_path)` over deterministic paths) and row-order replay from a published sampling RNG. Opens no key, prints no salt; exit 0 repelled, 1 broken, 2 not modelled |
| `results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md`, `blinding_audit.log` | the result for all three packets. The paired packet's **first build was de-blinded 180/180** on the row-order channel despite a secret salt; fixed by a separate secret-seeded order RNG. The **228-item packet is BROKEN** on the identifier channel, reproducibly from files committed at `7e01fbd` — preserved as delivered, 0 labels filled, re-salting left to the PI |
| `results/audit_v1/PUBLISHED_EVIDENCE_INDEX.md` | exact path and sha256 of every published file, so a reviewer can verify any row with `git show <commit>:<path> \| sha256sum` without trusting the table |
| `docs/research/NEXT_SCREEN_PROPOSAL.md` (revision 2) | prior-work account corrected against CEASE §3.2/§3.3/A.5 and LACU; eligibility separated from iso-deletion with residue as the matching quantity; the four concept/anchor cells and their identification limits; stagewise relapse defined against each target's own post-erasure checkpoint |

**No GPU work, and no change to any measurement.** The paired packet was rebuilt
from the same existing images with new blinded ids; no labels existed in any
packet, so nothing was lost.

### Added by the delivery-gate round (2026-10-10)

| file | purpose |
|---|---|
| `scripts/seq/audit_packet_blinding.py` (repaired) | the delivery gate. **Bound to the published sheet row-for-row**: missing, extra, duplicate or unreadable entries and an empty comparison corpus are **REJECTED** (exit 3) with no blinding claim, where the previous version scored them 0/0 and printed a pass. Five statuses — `holds` / `BROKEN` / `UNRESOLVED` / `NOT MODELLED` / `REJECTED` — and only one is a pass. Attack 3 reports file **and** distinct-hash counts, and **grades** digest occurrences in committed text: within two lines of a checkpoint/arm/slot identifier is `BROKEN`, a bare mention is `UNRESOLVED` requiring assessment |
| `scripts/seq/test_packet_delivery_gate.py` | **27 CPU checks, 0 failures**, synthetic fixtures only. Regressions for both demonstrated false passes (empty directory, one unrelated file), the inventory faults beside them, the copied-source control (BROKEN) and fresh-PNG control (holds), a digest→arm association (BROKEN), a bare digest mention (UNRESOLVED), duplicate delivered content (UNRESOLVED), and `export_image`'s pixel/byte/EXIF behaviour including refusal to trust an existing destination |
| `scripts/seq/build_paired_annotation_packet.py` (hardened) | a **missing source image is fatal** instead of silently producing no file; an **existing destination is re-exported and re-verified** rather than skipped; the byte guard now covers the **stated published corpus — 6,880 digests** from the committed archives plus the 180 selected originals — and aborts on any collision, duplicate exported digest, or count mismatch. Rebuilt with the stored secret: the published sheet is byte-identical, so **item ids are unchanged** |
| `results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md`, `blinding_audit.log` | 3 attacks × 6 cases, with the threat model stated. The paired packet holds; the two false passes now REJECT; the **pilot 236-item packet is BROKEN on 2 of 236 items**, whose digests sit beside `sha256_m0_direct` in `results/seq17/calibration/reload_check.json`, naming **M0** — one of its six blinded checkpoints. Found by the graded sweep, assessed by hand, now matched automatically. Preserved as delivered |
| `docs/research/NEXT_SCREEN_PROPOSAL.md` (revision 4) | request-1 reuse made **conditional** on qualification against the new frozen sets, with checkpoint ownership corrected to **per method**; `LOW-PARENT` separated from causal prior damage, which now needs a paired `M0`→parent contrast with uncertainty; **cumulative** preservation against `M0` added alongside stagewise; native success defined over each method's own sequences; and the decision rules **ordered** so a material interaction resolves before any main-effect reading |

**No artifact was deleted this round**, no measurement changed, and the 41/0 and
59/0 suites were not re-run — `diag_analysis.py` is byte-identical.
