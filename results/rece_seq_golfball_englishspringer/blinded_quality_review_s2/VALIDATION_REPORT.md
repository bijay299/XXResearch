# Validation report — blinded_quality_review_s2 package

No GPU, no model computation, no image generation, no classifier run was
used to produce this package or this report. Every check below is
read-only verification of already-published, already-generated files,
or static/logic inspection of the package's own code.

## 1. Source-image verification (before package construction)

`prep_code/verify_source_images.py` (a copy of the same script run at
`/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2_verify_sources.py`)
checked, for all 40 source images (case IDs 60–79 × 2 endpoints):

- Panel manifest `FRESH_PANEL_MANIFEST.json` sha256 matches the pinned
  value `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`. **Pass.**
- Each image's actual bytes (re-hashed now, not trusted from an earlier
  record) match BOTH the host-side `CASE_LEVEL_COMPLETION_RECORD.json`
  entry AND the published `PUBLISHED_EVIDENCE_INDEX.md` row for that
  file. **Pass, all 40.**
- Each image is exactly 512×512 pixels. **Pass, all 40.**
- Prompt text and seed match between the two endpoints for every
  case_id, and match the panel manifest's own row. **Pass, all 20
  cases, no mismatches.**
- Exactly 20 unique case IDs (60–79), exactly one image per endpoint per
  case, 40 images total. **Pass.**

Full machine-readable result:
`blinded_quality_review_s2_SOURCE_VERIFICATION.json` (host-side; not
republished in full here since it contains host file paths — the pass/
fail summary above is the public record).

## 2. PNG metadata inspection (identifying-content check)

Inspected PNG chunk structure of representative source images (one per
endpoint) via direct chunk parsing (not just Pillow's `.info`, which can
miss chunks Pillow doesn't surface): chunks present were exactly
`IHDR, IDAT×N, IEND` — **no `tEXt`, `zTXt`, `iTXt`, or any other
metadata chunk was found**, and Pillow's own `.info`/`.text` were empty.
No EXIF. This is consistent with the generation pipeline (PIL
`Image.save()` with no metadata arguments) and was independently
re-checked, not assumed. All 40 package images are exact byte-copies of
these same-pipeline source files (see §4), so this finding is
representative of all of them — the limitation is disclosed in
`FROZEN_PROTOCOL.md`: this is a metadata/filename/source-check, not a
guarantee against visual recognition of pipeline-specific rendering
characteristics by a reviewer already familiar with this results line.

## 3. Reviewer-interface content scan

Direct case-insensitive text search of the shipped `index.html` for:
`springer, golf, stage1, both_stages, parachute, horn, case_id,
classifier`, every original case ID pattern (`60_0`…`79_0`), and
specific disputed-result phrases (`6/20`, `2/20`, `4 newly`,
`regression`). **No match found** except the script's own comment
*stating* that no such data is present. The neutral image filenames
(`Q01_A.png` … `Q20_B.png`) carry no such terms either.

## 4. Package construction fidelity

`prep_code/build_reviewer_package.py`'s own build-time assertions
(re-verified after the fact, not merely trusted): every neutral package
image's sha256, computed immediately after `shutil.copyfile`, was
asserted equal to the corresponding verified source image's sha256
before the build completed — i.e. the 40 package images are **exact byte
copies**, not re-encoded, cropped, rescaled, or annotated. A-position
balance (each endpoint as "A" in exactly 10 of 20 pairs) was asserted at
build time: **10/10, confirmed**.

## 5. JavaScript logic validation (synthetic data, not published)

No browser or headless-browser tool was available in this environment —
**visual rendering, layout, and real click/keyboard event wiring were
NOT verified**, and this report does not claim otherwise.

What *was* verified: `index.html`'s inline script (1) passes
`node --check` (valid JavaScript syntax), and (2) was evaluated in a
Node.js sandbox with stub `document`/`localStorage`/`Blob`/`URL` objects
to exercise the pure state-management and export-schema logic, using
clearly-marked synthetic responses (e.g. `"comment":"SYNTHETIC TEST
ROW"`, placeholder answers on `Q01`/`Q02`) that exist **only** in a
scratch location (`/data/bijaypandey/claude_tmp/scratch_check/`,
outside this results line and outside the git checkout) and are **not**
included in this deliverable. Confirmed:

- `PAIRS` has exactly 20 entries, IDs `Q01`–`Q20`, referencing exactly
  40 unique image paths that match the 40 files actually built.
- `freshState()` initializes all 20 response slots with every field
  empty (`""`) and `setupComplete: false` — **human fields start
  empty**, as required.
- Completion detection (`isPairComplete`, `isIndividualPhaseComplete`,
  `pairCompleteCount`) correctly distinguishes a fully-answered
  synthetic row from a partially-answered one.
- The export payload contains all required fields (package/protocol
  version, reviewer provenance, per-pair neutral IDs, responses,
  comments, completion flag) and correctly reports `isComplete: false`
  for an incomplete synthetic draft.
- An explicit `"uncertain"` answer and a genuinely missing (`""`) answer
  remain distinguishable in the exported payload — missing is never
  silently treated as "no degeneration."
- A re-imported export round-trips through `JSON.stringify`/`JSON.parse`
  without data loss, and the package-version check that gates import
  would reject a mismatched version.

## 6. Zip deliverable check

`BLINDED_REVIEW_PACKAGE.zip` extracted cleanly to a scratch directory;
contains exactly `index.html` + `images/` (40 PNGs); every image path
referenced by `index.html`'s `PAIRS` array resolves to a present file
after extraction; a spot-checked extracted file's sha256 matches the
private mapping's recorded hash for that neutral filename.

## What this report does not claim

- No claim that the interface renders correctly in a real browser, that
  navigation/export buttons work when actually clicked, or that the
  layout is usable on any specific device — **not checked, no browser
  tool was available**.
- No claim of blinding against a reviewer recognizing pipeline-specific
  visual characteristics independent of labels (disclosed in
  `FROZEN_PROTOCOL.md`).
- No human response data exists yet. This package has collected none.
