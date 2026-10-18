# teapot_human_review_v1 — two-stage human review package

**Status: `READY_FOR_INDEPENDENT_REVIEW` — Task A only.**

> ### What this status does and does not mean
>
> It means: the Task-A package is built, complete (40/40 images), masked with
> neutral IDs, verified against the accepted source hashes, mechanically
> validated, and **tested in a real browser**.
>
> It does **not** establish that any participant exists, that any independent
> label has been collected, that the evaluation is scientifically valid, or that
> erasure succeeded. **No reviewer has been recruited or contacted, and no
> response exists.** Nothing here reports any outcome.

Task B is **prepared and fully populated but held privately** and is **not
released**.

| | |
|---|---|
| Protocol | [`TEAPOT-HR-PROTO-1.0.0`](protocol/PROTOCOL.md) |
| Export schema | [`THR-EXPORT-SCHEMA-1.0.0`](protocol/SCHEMA.md) |
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
shuffle — all computed from the public IDs and alias only, **independent of the
arm mapping**. Paired items are kept ≥4 positions apart in every fixed variant.

The bundle runs **entirely offline**: no account, no network, no upload, no
tracking. Progress is saved in the reviewer's own browser and leaves it only
when they export a file.

**Download:** `teapot-human-review-v1-taskA.zip` from the release
[`teapot-human-review-v1-taskA`](https://github.com/bijay299/XXResearch/releases/tag/teapot-human-review-v1-taskA)
— 13,919,126 bytes, sha256
`e239f2d4d52ce75bd914a4c6073cb98da6c702b825f64f0394afbe9130a7f456`,
47 members, 40 PNGs. The archive build is deterministic: rebuilding from the
same inputs reproduces the same hash.

UI preview: published as the release assets `preview_A1_instructions.png`,
`preview_A2_item_view_synthetic.png`, `preview_A3_mobile_instructions.png` and
`preview_A4_mobile_item_synthetic.png` (this repository gitignores `*.png` by
standing policy, so no PNG is committed to ordinary Git). The instructions pages
are real; **the item-view previews use a clearly synthetic placeholder, never a
research image** — see [masking](MASKING_AND_LIMITATIONS.md) §6.

## Planned participation — none yet

Three independent reviewers are **planned**. **No participant is assumed and no
recruitment is authorized at this time.**

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
* **Real-browser QA:** **88/88 checks pass** across **three engine
  configurations** — Chromium (the Chrome engine), Firefox, and Firefox with a
  strict file-origin policy — loading the bundle from `file://` with every
  non-`file://` request aborted — offline opening, navigation, zoom, save/resume,
  alias/package/stage storage isolation, draft export + import round-trip,
  complete Final round-trip accepted by the validator, Final correctly
  unavailable while answers are missing, malformed and wrong-stage imports
  rejected without crashing, and no horizontal overflow at 390 px.
  [`qa/browser_qa_report.json`](qa/browser_qa_report.json).

All fixtures are **explicitly synthetic**, generated from a fixed rotation. **No
fixture is a human judgement and none may be used as reviewer evidence.** No
answer key or completed test response is present in any reviewer bundle. The
public QA reports contain **mechanical results only** — no image-specific
judgement.

## Correction — bundle v2 (2026-10-18)

The **first published bundle did not open in Chrome or Edge.** It loaded its
item list with `XMLHttpRequest` over `file://`, which every Chromium browser
blocks by CORS policy ("Cross origin requests are only supported for protocol
schemes: … http, https"), and which Firefox's file-origin policy can block too.
Reviewers saw *"Could not load items.json"* instead of the task.

**Fixed:** the item list now ships as `items.js` and is loaded with a plain
`<script src="items.js">`, which is not subject to that restriction and works
from `file://` in every browser. No `XMLHttpRequest` or `fetch` remains in
either viewer. `items.json` is retained for reference and tooling.

**Why QA missed it:** the browser suite ran **only Firefox**, which permits the
request, so it reported every check green against a bundle that was broken for
most reviewers. The suite now runs **Chromium, Firefox and strict-file-origin
Firefox** on every check, and the fresh-download verification runs in Chromium —
the engine that failed.

The replaced asset keeps the same tag and filename; its sha256 changed from
`45e3eb7a…` to `e239f2d4…`. No image, ID, mapping, ordering or commitment
changed — only the loader.

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
