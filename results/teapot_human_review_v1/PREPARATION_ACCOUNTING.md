# Preparation accounting

Append-only ledger: [`ledger/prep_ledger.jsonl`](ledger/prep_ledger.jsonl) ·
clock: [`ledger/clock.json`](ledger/clock.json) ·
helper: [`scripts/prep_ledger.py`](scripts/prep_ledger.py)

## Clock — recorded once

| | |
|---|---|
| Started | `2026-10-18T13:05:57−04:00` |
| Deadline | `2026-10-20T13:05:57−04:00` (48 elapsed hours) |
| Elapsed at first publication | under 1 h |
| Elapsed after the patch01 repair | see ledger (same clock, not reset) |

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

12 ledger steps recorded, **0 failures** (one recorded defect, found by the user and fixed — see below):

| step | outcome |
|---|---|
| `source_verification` | 40/40 PNGs matched accepted hashes; pairing records intact |
| `taskA_bundle` | 40 neutral images, byte-identical, viewer + items.json |
| `taskB_private_package` | 20 pairs / 40 images, byte-identical, private `0700` |
| `validator_mechanical_qa` | 18/18 synthetic cases pass |
| `quarantine_leaking_screenshots` | two previews showing a real image beside its neutral ID moved to private; replaced with synthetic-placeholder previews |
| `browser_qa` | 88/88 real-browser checks pass (3 engine configurations) |
| `build_public_tree_and_zip` | deterministic ZIP, masking audit clean |
| `private_backup_verified` | 49 files, host-performed hash comparison |
| `published_and_verified` | release created, fresh-download verification passed |
| `taskA_items_loader_fix` | **defect found by the user after publication** — see below |
| `browser_qa_dual_engine` | suite extended to 3 engine configurations, 88/88 pass |
| `republish_and_reverify` | asset replaced on the same tag, re-verified in Chromium |

### One real defect, found by the user after publication

The first published bundle **did not open in Chrome or Edge**. The viewer
fetched its item list with `XMLHttpRequest` over `file://`, which every Chromium
browser blocks by CORS policy; reviewers saw *"Could not load items.json"*
instead of the task. It was fixed by shipping `items.js` and loading it with a
plain `<script>` tag, which is not subject to that restriction.

**The QA suite had reported 30/30 green on that broken bundle because it ran
Firefox only.** Firefox permits the request; Chrome does not. The suite now runs
**Chromium, Firefox and strict-file-origin Firefox** on every check (88/88), and
the fresh-download verification runs in Chromium. This is a genuine QA gap, not
an environment quirk: testing one engine was not sufficient to claim the bundle
worked for reviewers.

### Non-blocking obstacles

Four obstacles were worked through during preparation and are noted for
completeness rather than as failures:

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

## Repair accounting (patch01) — appended, not rewritten

The original clock was **not reset** and the original ledger entries were **not
rewritten**. Repair work is appended.

| | |
|---|---|
| GPU allocation / model loading / training / generation | **still 0** |
| New model-generated images | **still 0** |
| Images imported | **still 40** (the same files; none re-imported) |
| Synthetic placeholder images | 1 (UI preview only) |
| Browser engines exercised | 3 (Chromium, Firefox, strict-file-origin Firefox) |
| Focused browser checks | **150 / 150 pass** |

### Successful ledger steps vs unsuccessful setup attempts

Distinguished deliberately, because the earlier report blurred them:

**Unsuccessful setup / harness attempts (no research effect):**

1. System Firefox is a snap and could not write a profile while `/` was full.
2. The same standalone build needed `HOME` redirected to `/data`.
3. First browser-QA run stopped on a collapsed `<details>` in the harness.
4. First masking audit produced false positives on field names in code.
5. **patch01:** the first run of the new suite reported 6 failures in the
   provenance-conflict section. Investigation showed the **product was correct**
   and the **test helper** was reading an arbitrary `localStorage` key instead of
   the active session. The helper was fixed; no product change resulted.

**Genuine defects found and repaired:** the five listed in Amendment 01 — the
Chromium loader failure (found by the user), weak import validation, truthiness
completion, the validator's hard-coded path, and a mis-stated QA claim.

### Storage

| | used | limit |
|---|---|---|
| Project directory | ~0.5 GiB | — |
| Playwright browsers (Firefox + Chromium) | ~0.43 GiB | — |
| **Total additional storage under `/data`** | **well under 1 GiB** | ≤ 10 GiB |
| CPU/BLAS worker threads | capped at **8** | ≤ 8 |

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
