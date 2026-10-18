# measurement_study_v1 — blinded image-review package

> # ⚠ `DATA_ACCESS_BLOCKED` — PRACTICE/UI PREVIEW ONLY
>
> **Zero study items; not ready for independent study review.**
> **Do not collect research responses from this package.**
>
> Corrections to the originally published text are in
> **[`AMENDMENT_01_ACCESS_AND_READINESS.md`](AMENDMENT_01_ACCESS_AND_READINESS.md)**, which is
> authoritative wherever this directory disagrees with itself.

The published ZIP contains **0 study images** and **12 non-study practice images**; its
study-item array is empty. It exists to show how the review interface works, not to gather
data. No participants, no human labels, no evaluator-error findings and no research
contribution exist.

## What this study asks (planned; not performed)

> Do automated target-presence evaluators make different errors on **original** versus
> **concept-erased** modern-model images — enough to change estimated removal effects or
> comparisons?

The object of study is the **evaluator**, not any erasure method. No outcome is reported here,
and none exists.

## Why the panel is empty — corrected conclusion

> **No eligible matched non-sensitive object assets were identified during the bounded access
> check.**

**Missing item:** matched original/erased outputs — or released original *and* erased
checkpoints — for a **modern backbone (FLUX.1-dev or SD3.5)**, targeting **concrete
non-sensitive objects**, from a genuine erasure method rather than inference-time guidance
avoidance.

**Withdrawn** from the original publication: the claims that no modern erased checkpoints
exist, that all relevant modern papers are code-only, and that no eligible data source exists.
Zero GitHub releases and zero metadata-filtered search hits are absence of evidence; they
cannot establish nonexistence.

**Corrected example.** GEM **does** publish FLUX erasure checkpoints —
[`MAI-Lab/GEM-Flux-Nudity`](https://huggingface.co/MAI-Lab/GEM-Flux-Nudity) and
[`MAI-Lab/GEM-Flux-Bloody-Gore`](https://huggingface.co/MAI-Lab/GEM-Flux-Bloody-Gore), linked
from the [GEM README](https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/README.md).
As the model cards state, they are **reproductions, not the original paper checkpoints**. They
are **ineligible here** because their targets are sensitive content, not concrete non-sensitive
objects. The original access check recorded GEM as "code only", which was wrong; the bounded
search missed these because HuggingFace's `search=` matches the model **name**, and neither
name contains "erase" or "unlearn" even though both repositories are correctly tagged
`concept-erasure` and `base_model:black-forest-labs/FLUX.1-dev`.

This correction does **not** resolve the missing matched object assets, so the status stands.
Candidate-by-candidate record: [`ACCESS_CHECK.md`](ACCESS_CHECK.md) (see its supersession
notice).

## The package — practice/UI preview

**[`package/measurement_study_v1_taskA_v1.0.0.zip`](package/measurement_study_v1_taskA_v1.0.0.zip)** — 17.8 MB

| | |
|---|---|
| bytes | `18,641,021` |
| SHA256 | `604e3227d28bcc3c1548a59da33fd5a01f27fed9f95e35409035a547c3fb2246` |
| study images | **0** |
| practice images | 12 |

Unzip and open `index.html` in any browser to **preview the interface**. It runs offline with
no account, no install, no server and no upload.

> **Do not distribute this as a study questionnaire and do not collect Final JSON from it.**
> The generic completion instructions **inside** the archive — its `README.txt` and the
> viewer's own "send the downloaded JSON to your coordinator" wording — are **superseded by
> this amendment**. They remain present only because the archive is preserved byte-for-byte
> and is deliberately **not** rebuilt or repackaged.
>
> The viewer itself correctly **disables Final submission at zero items**, so it does not
> invite an empty study return. An empty **draft** export is a UI artefact, not a human study
> response.

GitHub's rendering of `index.html` is not an interactive viewer; the ZIP must be downloaded
and extracted. Screenshots: [`preview/`](preview/).

## What the interface shows (Task A, as designed)

One full-resolution image at a time with the target name and a short frozen object definition,
then: target **present / absent / uncertain**; **coherent scene / severe rendering failure /
unable to judge**; and an optional short explanation. Model, method, prompt, filename, seed and
all machine scores are withheld by design; ids are neutral and order is derived from the
reviewer's alias and an assigned variant.

The 12 practice examples are instructional aids only — not study items, not scored, not an
accuracy gate — on a practice object (*bicycle*) deliberately outside any study target set.

## Planned designs versus the actual panel

The **planned** design was 180 images (3 targets × 20 cases × 3 conditions), with a 120-image
fallback if only one erased condition were obtainable and ranking claims dropped. Those are
**plans**. The **actual** study panel is **empty**. Do not read the planned numbers as a
description of what exists.

## Readiness

The reported **47/47** browser checks are **synthetic-fixture UI checks**. They do not
establish statistical correctness, independent reviewer admission, real-arm blinding or study
readiness. Five prospective implementation limitations in the analysis code and plan — reviewer
admission, uncertainty bounds, bootstrap clustering, unimplemented suppression contrasts, and an
internal inconsistency about partial returns — are documented in
[`AMENDMENT_01_ACCESS_AND_READINESS.md`](AMENDMENT_01_ACCESS_AND_READINESS.md) §4. **The code
was not repaired**; these require separate authorisation before any future collection or
analysis.

## Blinding and privacy, stated precisely

No actual Task-B reference associations, arm keys or automated scores were found in the
published package. Generic shared viewer code containing an unused reference-file field is not
itself a mapping leak. [`COMMITMENTS.json`](COMMITMENTS.json) explicitly commits the arm-key,
case-provenance and prediction material as **empty JSON objects**, because no study cases
exist. Task B is an **empty private scaffold**, not a populated matched-reference package.

Private seed/salt storage and backups are **host-reported**; the coordinator did not
independently verify their contents or permissions. **Meaningful masking of actual study arms
has not yet been exercised**, because there are no study arms.

Any future actual study requires separately authorised prospective case freezing, new
commitments and a new package version. This frozen empty panel must not be silently populated.

## Contents

| | |
|---|---|
| [`AMENDMENT_01_ACCESS_AND_READINESS.md`](AMENDMENT_01_ACCESS_AND_READINESS.md) | **authoritative corrections** |
| [`PROTOCOL.md`](PROTOCOL.md) | frozen protocol (historical; see its notice) |
| [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) | frozen analysis plan (historical; see its notice) |
| [`VALIDATION.md`](VALIDATION.md) | validation record (historical; see its notice) |
| [`ACCESS_CHECK.md`](ACCESS_CHECK.md) | access scope (historical; see its notice) |
| [`schema/`](schema/), [`analysis/`](analysis/), [`code/`](code/) | schema and code, **unchanged** |
| [`COMMITMENTS.json`](COMMITMENTS.json), [`MASKING_AUDIT.json`](MASKING_AUDIT.json) | **unchanged** |
| [`ledgers/`](ledgers/), [`validation/`](validation/) | raw ledgers and validation log, **unchanged** |

## Accounting (recorded totals)

**0.042485 GPU-hours** (≈ 0.0425), **12** generated practice images, **0** study images,
**0** recorded failed GPU jobs. The rounded 152.9-second duration is consistent with the
rounded GPU-hour figure; there is no discrepancy. These are this study's own recorded totals,
not independent proof of complete host history. The original 12 GPU-hour / 240-image /
72-hour figures are historical ceilings, not renewed permission to spend unused allowance.

## Licensing and attribution

The 12 practice images were generated for this study with **FLUX.1-dev**
(`black-forest-labs/FLUX.1-dev`, revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`) and are
distributed under that model's terms for generated outputs. No third-party image is
redistributed, because the study panel is empty.
