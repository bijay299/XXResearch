# measurement_study_v1 — blinded image-review package

**Status: `DATA_ACCESS_BLOCKED`.**
The review instrument is complete and browser-validated. **The study panel is empty**: no
eligible data source exists, so there are **0 study images** in this package. The exact
missing item is stated below and evidenced in [`ACCESS_CHECK.md`](ACCESS_CHECK.md).

## What this study asks

> Do automated target-presence evaluators make different errors on **original** versus
> **concept-erased** modern-model images — enough to change estimated removal effects or
> comparisons?

The object of study is the **evaluator**, not any erasure method. This package is not
evidence that a judge is wrong or that a method is better, and no outcome is reported here.

## Why the panel is empty

**Missing item:** a publicly released **concept-erased checkpoint, or a matched
original/erased image set, for a modern backbone (FLUX.1-dev or SD3.5), targeting concrete
non-sensitive objects**, produced by a genuine erasure method rather than inference-time
guidance avoidance.

Every modern-backbone erasure paper located releases **code only**. Every released *erased
weight* located is **SD1.x**, which this study excludes. The one training-free FLUX method
located works by safety-guidance steering — negative-concept avoidance — on a sensitive
target. The best-provenanced artifact found (9 methods × 15 concrete object concepts, with
per-artifact SHA256 and reload proofs) is built on **SD v1.5** and was therefore **not**
substituted. The scientific task was not quietly changed to fit the data that happened to
exist. Full candidate-by-candidate record: [`ACCESS_CHECK.md`](ACCESS_CHECK.md).

## Download the reviewer package

**[`package/measurement_study_v1_taskA_v1.0.0.zip`](package/measurement_study_v1_taskA_v1.0.0.zip)** — 17.8 MB

Unzip it and open `index.html` in any browser. No account, no install, no network, no
server, nothing uploaded. File list and hashes: [`package/PACKAGE_MANIFEST.json`](package/PACKAGE_MANIFEST.json).

> GitHub's rendering of `index.html` is **not** an interactive viewer. The ZIP must be
> downloaded and extracted for the viewer to work.

Screenshots: [`preview/`](preview/).

## What a reviewer sees (Task A)

One full-resolution image at a time, with the target name and a short frozen object
definition. For each image, three questions:

1. target **present / absent / uncertain**;
2. **coherent scene / severe rendering failure / unable to judge**;
3. an optional short explanation.

Reviewers are **not** shown the model, the method, the generation prompt, the source
filename, the seed, any machine score, any other reviewer's answers, or any desired outcome.
Item ids are neutral and opaque, and presentation order is derived from the reviewer's own
alias and an assigned order variant (A/B/C), independently of any private mapping.

The package also carries **12 non-study practice examples**. They are instructional aids
only — not study items, not scored, not a reviewer-accuracy gate — and they use a practice
object (*bicycle*) that is deliberately not a study target.

## Task B is prepared but **not** published

A second, matched-reference stage exists: it shows the original-model reference beside each
erased-condition candidate and asks whether the candidate shows less / about the same / more
structural deformation, or cannot be fairly compared. It is **built and held privately** and
will be released only after first-stage collection closes, all submitted responses are
locked, and separate explicit authorisation is given. Only its salted commitment and its
procedure are published here. Stage-A labels are locked once a reference has been seen; any
later revision is recorded separately and never overwrites an original.

## Contents

| | |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | frozen public protocol |
| [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) | frozen analysis specification |
| [`schema/response_schema_v1.json`](schema/response_schema_v1.json) | versioned response schema |
| [`analysis/`](analysis/) | response validator and analysis code (synthetic fixtures only) |
| [`ACCESS_CHECK.md`](ACCESS_CHECK.md) | source/access scope and the blocker |
| [`VALIDATION.md`](VALIDATION.md) | preparation-validation record, including real browser results |
| [`COMMITMENTS.json`](COMMITMENTS.json) | salted blinding commitments |
| [`MASKING_AUDIT.json`](MASKING_AUDIT.json) | masking audit of the public build and the ZIP |
| [`ledgers/`](ledgers/) | resource, job and failure ledgers |
| [`validation/`](validation/) | validation suite and results |

## Blinding, honestly stated

Held outside this repository, the ZIP, the viewer and git history: the arm/method key, the
randomisation seed, case-level source provenance, source-to-review hash associations, all
automated predictions and raw answers, and all Task-B reference associations. A salted
SHA256 commitment to each is in [`COMMITMENTS.json`](COMMITMENTS.json); the salt is retained
privately with a verified backup.

**Masking limitation:** public source images can in principle be reverse-identified by
someone who goes looking for them. Blinding here means reviewers are not *told* the arm. It
is not a guarantee that an arm cannot be inferred, and no stronger claim is made. Case-level
provenance is withheld until review lock where licensing permits.

## Licensing and attribution

The 12 practice images were generated for this study with **FLUX.1-dev**
(`black-forest-labs/FLUX.1-dev`, revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`) and are
distributed under that model's licence terms for generated outputs. No third-party image is
redistributed in this package, because the study panel is empty. When a source is adopted,
its licence and attribution will be published with it, and only permitted copies or stable
source identifiers will be distributed.

## Status if and when a source becomes available

Nothing here authorises a new experiment. This package records a blocked study with a ready
instrument. Filling the panel requires an eligible source and a separate decision.
