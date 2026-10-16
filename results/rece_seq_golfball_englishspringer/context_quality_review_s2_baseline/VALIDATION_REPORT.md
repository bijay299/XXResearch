# Validation report — context_quality_review_s2_baseline package

No GPU, no model computation, no image generation, no classifier run,
no reviewer recruitment, no v2 builder re-run was used to produce this
package or this report. Every check below is read-only verification of
already-published/already-generated files, or static/logic inspection
of this package's own code.

## 1. Edited-image reuse (unchanged from v2)

The 40 `R01`–`R20` `_A.png`/`_B.png` images are exact byte copies of the
already-verified, already-repaired `context_quality_review_s2_v2`
package's images (hashes re-checked against that package's own recorded
hashes during the copy; all 40 match). Pair order, A/B assignment, and
the v2 private mapping are **unchanged** — the v2 builder was never
re-run.

## 2. New M0 reference images

`prep_code/verify_baseline_references.py` checked all 20 pristine M0
images (case IDs 60–79):

- Panel manifest sha256 matches the pinned value
  `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`. **Pass.**
- Each image's bytes (re-hashed now) match both the M0 `image_report.json`
  entry and the published evidence index. **Pass, all 20.**
- Each image is exactly 512×512. **Pass, all 20.**
- Prompt and seed match the frozen panel manifest for every case. **Pass,
  no mismatches.**
- M0's own `image_report.json` records `edit_ckpt: null` — confirmed
  programmatically that M0 is the pristine, pre-editing checkpoint, not
  `EnglishSpringer_parent` or any edited checkpoint. **Pass.**
- Generation settings (512×512, 50 steps, guidance 7.5, float32,
  `LMSDiscreteScheduler` with the stated beta/schedule/train-step
  parameters) were compared across M0's, `S2_STAGE1_HISTORY`'s, and
  `S2_HISTORY_BOTH_STAGES`'s own `image_report.json` files: **identical
  across all three.** No material mismatch was found; none was silently
  worked around.

Full result:
`context_quality_review_s2_baseline_SOURCE_VERIFICATION.json` (host-side).

## 3. Baseline association (private, derived read-only from v2's mapping)

`prep_code/build_baseline_association.py`:

- Verified the v2 private mapping's hash
  (`bd360bc35c45d5eb68f5ff6e91aedb1db30cd70e10a9b55023bb2d330fbbc529`)
  matches the required/published commitment **before** reading it, and
  **again after** finishing — confirmed byte-identical both times. The
  v2 mapping file was opened read-only; nothing was written to it.
- Built a **separate** private file, `BASELINE_ASSOCIATION_PRIVATE.json`,
  associating each existing `R01`–`R20` ID with its same-case M0
  reference (filename, hash, prompt) — kept outside the git checkout,
  mode `0600`, never published. Its own public SHA256 commitment,
  `333e9230740ce9b636bb561e50cdcf9bc7ab81b9f071e0023a6e4f587df74edd`, is
  published in `BASELINE_ASSOCIATION_COMMITMENT.json`, before any
  response is collected.
- Every one of the 60 package images' bytes, copied into
  `reviewer_package/images/`, was re-hashed immediately after copying
  and asserted equal to the verified source hash — exact copies, no
  re-encoding.

## 4. Reviewer-interface content scan

Case-insensitive search of the shipped `index.html` for endpoint names,
case-ID/seed patterns, `EnglishSpringer_parent`, classifier/label terms,
and v1/v2's neutral IDs (`Q01`–`Q20`): **no match** except this
package's own descriptive code comments stating that no such data is
present.

## 5. Independent logic validation (synthetic data, not published)

No browser or headless-browser tool was available in this environment
— **visual rendering, layout, and real click/keyboard event wiring were
NOT verified**, same disclosed limitation as v1 and v2.

What *was* verified, with a freshly written Node.js harness (not reused
from any external bundle) using clearly-marked synthetic data (never
part of this deliverable):

- `index.html` passes `node --check`; brace/paren/backtick counts and
  `id=`/`getElementById` cross-references are balanced and consistent.
- `PAIRS` has exactly 20 entries, each with `imgRef`/`imgA`/`imgB`
  referencing exactly 60 unique paths matching the 60 files actually
  built.
- `freshState()` initializes every field empty, including the four new
  relative-deformation fields — **human fields start empty.**
- The initial three-image render executes without error and the
  rendered HTML contains all three image paths for the current pair.
- The paired "more breakdown" question stays hidden until **every**
  mandatory field is answered, **now including** each relative-
  deformation field and its conditional reason (required only for
  "More"/"Unable to compare," not "Less"/"About the same") — confirmed
  both that a missing required relative reason keeps it hidden, and
  that entering it reveals the question **immediately**, in the same
  handler call, exactly as the v2 repair fixed for the absolute-
  breakdown reason (`saveReasonAndRefresh`, reused unmodified here).
  Clearing the reason re-hides it again.
- A full synthetic final export round-trips through `restoreExport()`
  without throwing; the imported state's relative-deformation answer
  and its reason are preserved exactly; clicking Begin afterward does
  not throw (the exact v2 import defect, reproduced and fixed there,
  does not reappear here since the same validated `restoreExport()`
  logic — extended to recognize the new field suffix
  `_vs_original_deformation` — is reused).
- An invalid relative-deformation value (not one of
  less/same/more/unable) is rejected by import, not silently accepted.
- A draft import with a missing relative-deformation answer leaves it
  missing after restore — not auto-filled.

## 6. Zip deliverable check

`BASELINE_CONTEXT_REVIEW_PACKAGE.zip` extracted cleanly; contains
exactly `index.html` + `images/` (60 PNGs); all three image paths
referenced for a sampled pair resolve to present files after extraction.

## What this report does not claim

- No claim that the interface renders correctly in a real browser, that
  the three-column layout actually displays as described on any given
  device, or that native file dialogs/downloads work — **not checked,
  no browser tool was available.**
- No claim of blinding against a reviewer recognizing which case a
  group belongs to from the now-visible Original image combined with
  already-public material — disclosed as partial masking in
  `FROZEN_PROTOCOL_BASELINE.md`, not claimed as full blinding.
- No human response data exists yet for this package. None was
  simulated as if real.
