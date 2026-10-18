# teapot_human_review_v1 — two-stage human review package

**Status: `READY_FOR_INDEPENDENT_REVIEW` — Task A only.**

> ### What this status does and does not mean
>
> It means: the Task-A package is built, complete (40/40 images), masked with
> neutral IDs, verified against the accepted source hashes, mechanically
> validated, and **tested in a real browser**.
>
> It does **not** establish that any independent label has been collected, that
> the evaluation is scientifically valid, or that erasure succeeded.
>
> Distribution of the corrected Task A is handled by the user independently of
> this repository. **This package collects no responses and reports no
> outcomes**, and nothing here describes any participant, label or result.

Task B is **prepared and fully populated but held privately** and is **not
released**.

| | |
|---|---|
| Protocol | [`TEAPOT-HR-PROTO-1.0.0`](protocol/PROTOCOL.md) |
| Export schema | [`THR-EXPORT-SCHEMA-1.1.0`](protocol/SCHEMA.md) (app 1.1.0, patch01) |
| Task-A package | `THR-V1-A` — **public**, 40 individual images |
| Task-B package | `THR-V1-B` — **private**, 20 matched pairs, not released |
| Reviewer instructions | [`protocol/INSTRUCTIONS_TASK_A.md`](protocol/INSTRUCTIONS_TASK_A.md) |
| Prepared | `2026-10-18` (clock started 13:05:57 −04:00; 48 h window) |

## What is being reviewed

The **40 already-published teapot images** (20 matched M0/edited pairs) from the
completed [`modern_object_assets_v1`](../modern_object_assets_v1/) asset build,
reused **read-only**. Those images and their provenance are already public in
the release [`modern-object-assets-v1`](https://github.com/bijay299/XXResearch/releases/tag/modern-object-assets-v1).

That asset build's status remains **technical-only `ASSETS_DELIVERED`**, and its
chronology qualifications (Amendment 01) stand unchanged. **This package adds a
human-review instrument; it changes nothing about that build and reports no
result from it.**

## Source integrity

Before any packaging, every file was checked against the accepted records:
**40/40 PNGs matched their accepted sha256**, all 20 pairs carry **byte-identical
initial-noise tensors** across conditions with 20 distinct noise hashes, all
images are native **512×512**, and the local copy of the published image archive
matched its accepted hash `8b254ab3…` exactly.

**No model was loaded, no image was regenerated, and no image was cropped,
resized, enhanced or annotated.** The PNG containers carry only `IHDR`/`IDAT`/
`IEND` — no text or EXIF chunks — so the files are published **byte-unchanged**
and merely renamed to neutral IDs. Decoded-pixel identity was verified
separately.

See [`verify/source_verification_summary.json`](verify/source_verification_summary.json)
and [`verify/taskA_completeness_summary.json`](verify/taskA_completeness_summary.json).

## Task A — what reviewers do

40 images, shown **one at a time** at native 512×512 with zoom, under neutral
opaque IDs (`A-001`…`A-040`). No pairing, counterpart link, condition,
filename, prompt, seed, machine score or other reviewer's answer appears
anywhere in the bundle.

Two required questions per image, no preselected answer, plus an optional
reason:

1. **Target** — present / absent / uncertain
2. **Rendering** — coherent / localized structural deformation / whole-frame
   structural breakdown / unable to judge

Ordering: three fixed variants (`V1`/`V2`/`V3`) or a reproducible alias-derived
shuffle.

* **`V1`/`V2`/`V3` are pair-aware.** They were built **using private pair
  membership** to keep paired items ≥4 positions apart, so they are **not**
  computed from the public IDs and alias alone and **not** independent of
  pairing. They do **not** explicitly use which member is M0 and which is
  edited; the spacing rule reveals negative pairing constraints, not the key.
* **Only the alias-derived shuffle is pairing-independent** (alias + public IDs
  only), and it carries **no minimum-spacing guarantee**.

The bundle runs **entirely offline**: no account, no network, no upload, no
tracking. Progress is saved in the reviewer's own browser and leaves it only
when they export a file.

**Download (corrected — use this one):**
`teapot-human-review-v1-taskA-patch01.zip` from the release
[`teapot-human-review-v1-taskA-patch01`](https://github.com/bijay299/XXResearch/releases/tag/teapot-human-review-v1-taskA-patch01)
— 13,923,703 bytes, sha256
`837770ae23e6b1fb56b7269a87e2a2bc23f40f6e2090e5e172ea57a8719ab7e7`,
47 members, 40 PNGs. The archive build is deterministic.

The original release
[`teapot-human-review-v1-taskA`](https://github.com/bijay299/XXResearch/releases/tag/teapot-human-review-v1-taskA)
(13,917,868 bytes, sha256 `45e3eb7a…`) is **preserved for the record**. It does
**not** open in Chrome or Edge — do not send it to reviewers.

UI preview: published as the release assets `preview_A1_instructions.png`,
`preview_A2_item_view_synthetic.png`, `preview_A3_mobile_instructions.png` and
`preview_A4_mobile_item_synthetic.png` (this repository gitignores `*.png` by
standing policy, so no PNG is committed to ordinary Git). The instructions pages
are real; **the item-view previews use a clearly synthetic placeholder, never a
research image** — see [masking](MASKING_AND_LIMITATIONS.md) §6.

## Planned participation

The protocol plans for three independent reviewers. **Recruitment and
distribution are handled by the user, independently of this repository**, and no
participant information, response or outcome is held or reported here.

Primary Task-A reference judgments require **at least two distinct, complete,
independent returns**; future Task-B primary reporting likewise requires two
complete returns from the locked Task-A cohort. Fewer ⇒ no primary reference
labels, reported as a shortfall. Missing and partial participation is preserved,
never dropped.

Primary eligibility is prespecified: independent completion, no prior access to
these source pairings or condition labels, no image-specific coaching before
Task A. Other exposure is recorded separately and is **not** an automatic
exclusion — generic research familiarity in particular is fine. Non-primary
reviews are preserved with their exposure status.

> **An alias, a file count or a declaration alone does not prove distinct humans
> or independence.** Exposure answers are self-declared and unverifiable from
> the artifacts.

## Frozen reporting conventions

Fixed in [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md) §4 **before any label
exists**, so no convention can be chosen after seeing results. In brief:
presence is three-valued; a reference label needs **two agreeing definite
votes** or the image is **unresolved**; **uncertainty and disagreement are never
converted to `absent`**; every raw label is preserved and agreement/coverage
reported; rendering is per reviewer with consensus needing two agreeing votes;
the **full 3×3 presence table summing to 20** is reported; the conditional
present→absent count is taken **only among definite M0-positive cases** and is
**`not estimable`** if that denominator is zero; absence-with-coherent-rendering,
localized deformation, whole-frame breakdown and uncertainty are **never merged
into one success score**; and **`absent → absent` is not demonstrated removal**.
Strata are descriptive. **Twenty paired cases are not 40 independent comparisons
or 120 independent reviewer observations.**

## Analysis plan — plan only

[`protocol/PROTOCOL.md`](protocol/PROTOCOL.md) §7 states prospectively what
would be computed if authorized returns ever exist. **No analysis engine,
bootstrap or statistical machinery was built or reused, no outcome analysis was
run, and no automated judge was introduced.**

## Validator and QA

* **Schema/validator:** [`protocol/SCHEMA.md`](protocol/SCHEMA.md),
  [`scripts/validate_export.py`](scripts/validate_export.py). Enforces the
  trusted package/protocol/schema/stage/ID sets, requires exactly 40 (A) or 20
  (B) distinct expected IDs in a Final with every required answer, allows
  partial drafts, treats `uncertain` as a completed answer, rejects duplicate
  and extra IDs, wrong stage/package, invalid enums and malformed JSON, and
  **recomputes completion rather than trusting any supplied count**.
* **Mechanical QA:** 18/18 synthetic cases pass —
  [`qa/validator_qa_report.json`](qa/validator_qa_report.json).
* **Validator maintenance (r2):** the companion Python validator crashed on
  several malformed JSON *types*. Explicit type guards were added and a 45-check
  CPU-only regression suite now covers them —
  [`maint/validator_r2_report.json`](maint/validator_r2_report.json). Acceptance
  of correctly generated schema-1.1.0 exports is unchanged, verified by
  before/after comparison. **The questionnaire was not touched.**
* **Real-browser QA (patch01):** **150/150 focused checks pass** across **three engine
  configurations** — Chromium (the Chrome engine), Firefox, and Firefox with a
  strict file-origin policy — loading the bundle from `file://` with every
  non-`file://` request aborted. They assert **actual restored state** — alias,
  declarations, order and raw answers — not just a success message: incomplete
  draft export → fresh-browser import → reload/resume; complete 40-item Final
  export → **fresh-session re-import** → reload/resume → companion-validator
  acceptance; conflicting-provenance imports following the documented policy;
  14 rejection cases each asserted to leave **memory and `localStorage`
  unchanged**; uncertainty counting as complete; isolation, navigation, zoom and
  mobile. [`qa/qa_patch01_report.json`](qa/qa_patch01_report.json).
* **Historical reports.** The original 18-fixture validator report
  (`qa/validator_qa_report.json`) is unchanged at its original path. The
  original 30-check browser report **was overwritten in place** at
  `qa/browser_qa_report.json` when the suite was re-run, and
  `verify/published_verification.json` **was removed** from the current tree.
  Those earlier versions remain in Git at
  [`d7921df`](https://github.com/bijay299/XXResearch/tree/d7921df8f23609d4ce709fed4a5d78df6c103631/results/teapot_human_review_v1),
  and exact byte-identical copies are restored under
  [`archive/d7921df/`](archive/d7921df/) with their source commit, source path
  and hash recorded. They were **not** preserved unchanged at their original
  paths, and no such claim is made.

All fixtures are **explicitly synthetic**, generated from a fixed rotation. **No
fixture is a human judgement and none may be used as reviewer evidence.** No
answer key or completed test response is present in any reviewer bundle. The
public QA reports contain **mechanical results only** — no image-specific
judgement.

## Amendment 02 — coordinator-side cleanup (current)

[`AMENDMENT_02_COORDINATOR_CLEANUP.md`](AMENDMENT_02_COORDINATOR_CLEANUP.md) —
documentation and companion-validator corrections only, made **while reviews are
in progress**. No questionnaire byte, release, tag or asset was changed.

## Amendment 01 — Task-A software repair (patch01)

Read **[`AMENDMENT_01_TASKA_PATCH01.md`](AMENDMENT_01_TASKA_PATCH01.md)**; it is
authoritative on these points.

Five defects were found after the first publication and repaired:

1. **The bundle would not open in Chrome or Edge** (found by the user) — it
   fetched its item list with `XMLHttpRequest` over `file://`, which Chromium
   browsers block. Now loaded with a plain `<script src="items.js">`.
2. **Import validation was far too weak** — it checked almost nothing and then
   *merged* a file into whatever session was open, keeping that session's alias
   and declarations. Imports are now fully validated first and **restore their
   own session**; they are never merged into another alias.
3. **`answered()` counted truthiness, not valid choices**, so a malformed file
   could enable a Final export.
4. **The Python validator had a hard-coded host path** and could not run from
   the documented location.
5. **A mis-stated QA claim** — a Final export plus Python validation was
   described as a Final *re-import* test. It was not; that claim is withdrawn
   and a genuine re-import test now exists.

Also corrected: the **ordering-provenance claims** (the fixed variants are
pair-aware, not public-data-only), the **exposure wording** ("Before starting
this Task A review", covering same-day exposure), and the **scope of the
published commitments**.

**Versioning:** export schema `THR-EXPORT-SCHEMA-1.0.0` → **`1.1.0`**, app
**1.1.0 (patch01)**. Legacy `1.0.0` exports are **rejected, not migrated** — no
reviewer return has ever been collected. Package and protocol IDs are unchanged;
scientific definitions and reporting conventions are unchanged.

**Nothing about the study material changed:** the 40 image bytes, neutral IDs,
pairings, private mapping, `V1`/`V2`/`V3` permutations, alias-order algorithm and
the published commitments are all **unchanged and verified identical** to the
reviewed publication. `freeze_ids.py` was not re-run.

## Masking, privacy and limitations

Read [`MASKING_AND_LIMITATIONS.md`](MASKING_AND_LIMITATIONS.md). The essential
point, stated plainly:

> **The images and their condition/pair provenance are already public.** Neutral
> IDs reduce *accidental* disclosure during review. They **cannot prevent
> reverse matching and do not guarantee blinding**, and nothing here makes the
> already-public source provenance private.

## Private material

The neutral-to-source mapping, randomization salt, the populated Task-B package
and the quarantined previews are held outside Git and outside every release, at
the locations recorded in [`PRIVATE_RECOVERY.md`](PRIVATE_RECOVERY.md), with a
verified backup. **Salted commitments to the actual populated artifacts** — not
to placeholders — are published in
[`protocol/COMMITMENTS.json`](protocol/COMMITMENTS.json).

## Preparation accounting

[`PREPARATION_ACCOUNTING.md`](PREPARATION_ACCOUNTING.md): **zero GPU allocation,
zero training, zero model inference, zero automated judging, zero new
model-generated images, 40 imported images.**

## File manifest

[`MANIFEST_SHA256.md`](MANIFEST_SHA256.md) — sha256 of every published file and
release asset.

## Stop point

Preparation ends here. **Not done and not authorized:** recruiting reviewers,
collecting responses, releasing Task B, unblinding, and analyzing outcomes.
