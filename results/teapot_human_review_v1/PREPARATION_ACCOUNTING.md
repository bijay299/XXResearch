# Preparation accounting

Append-only ledger: [`ledger/prep_ledger.jsonl`](ledger/prep_ledger.jsonl) ·
clock: [`ledger/clock.json`](ledger/clock.json) ·
helper: [`scripts/prep_ledger.py`](scripts/prep_ledger.py)

## Clock — recorded once

| | |
|---|---|
| Started | `2026-10-18T13:05:57−04:00` |
| Deadline | `2026-10-20T13:05:57−04:00` (48 elapsed hours) |
| Elapsed at publication | under 1 h |

The clock was recorded **once**, at the first preparation action, and never
reset.

## Model and compute — all zero, as required

| | used | limit |
|---|---|---|
| **GPU allocation** | **0** | 0 |
| **GPU seconds** | **0.0** | 0 |
| **Training runs** | **0** | 0 |
| **Model inference runs** | **0** | 0 |
| **Models loaded** | **0** | — |
| **New model-generated images** | **0** | 0 |
| **Automated judging / scoring** | **0** | forbidden |
| **Paid services / hosting / deployment** | **none** | forbidden |
| **External messages sent** | **0** | forbidden |

No model was downloaded or loaded at any point. No image was generated,
regenerated, cropped, resized, enhanced or annotated.

## Images

| | count |
|---|---|
| **Images imported** (read-only reuse of approved material) | **40** |
| — verified against accepted source hashes | 40 / 40 |
| — published in the Task-A bundle, byte-unchanged | 40 |
| — placed in the private Task-B package, byte-unchanged | 40 (20 pairs) |
| **New model-generated images** | **0** |
| Synthetic placeholder images (clearly labelled, UI preview only) | 1 |

The same 40 source images populate both stages; they were copied, not created.

## Work and failures

8 ledger steps recorded, **0 failures**:

| step | outcome |
|---|---|
| `source_verification` | 40/40 PNGs matched accepted hashes; pairing records intact |
| `taskA_bundle` | 40 neutral images, byte-identical, viewer + items.json |
| `taskB_private_package` | 20 pairs / 40 images, byte-identical, private `0700` |
| `validator_mechanical_qa` | 18/18 synthetic cases pass |
| `quarantine_leaking_screenshots` | two previews showing a real image beside its neutral ID moved to private; replaced with synthetic-placeholder previews |
| `browser_qa` | 30/30 real-browser checks pass |
| `build_public_tree_and_zip` | deterministic ZIP, masking audit clean |
| `private_backup_verified` | 49 files, host-performed hash comparison |

Four non-blocking obstacles were worked through during preparation and are noted
for completeness rather than as failures:

1. The system Firefox is a snap and cannot write its profile while the root
   filesystem is full; a Playwright **standalone** Firefox was installed instead.
2. That standalone build also needed `HOME` redirected to `/data` for the same
   reason.
3. The first browser-QA run stopped on a **test-harness** issue — the export
   controls sit inside a collapsed `<details>` — not a product defect.
4. The first masking audit flagged a **real** problem (two joinable hash row
   sets) and a batch of **false positives** (field names in source code). The
   real problem was fixed by redaction; the audit's scoping was narrowed and the
   narrowing is documented so it can be challenged.

## Workers and storage

| | used | limit |
|---|---|---|
| CPU/BLAS worker threads | capped at **8** | ≤ 8 |
| Project directory | 0.228 GiB | — |
| Playwright browser (installed for browser QA) | 0.307 GiB | — |
| **Total additional storage under `/data`** | **≈ 0.54 GiB** | ≤ 10 GiB |

CPU and browser work only. No allowance from any completed project was spent:
this package has its own clock, its own ledger and its own limits, and the
`modern_object_assets_v1` accounting is untouched.

## What was *not* done

Not done and **not authorized**:

* recruiting or contacting reviewers,
* collecting any response,
* releasing Task B,
* unblinding,
* analyzing outcomes,
* building or reusing any statistical-analysis engine or bootstrap,
* introducing any automated judge,
* repairing or repopulating any earlier viewer, schema or analyzer,
* touching any other closed panel, label, prediction or checkpoint.

**No reviewer response exists**, and no file in this package is, or may be used
as, a human judgement about any image.
