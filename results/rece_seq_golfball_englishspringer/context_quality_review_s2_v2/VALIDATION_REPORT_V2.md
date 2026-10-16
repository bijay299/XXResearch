# Validation report — context_quality_review_s2_v2 package

No GPU, no model computation, no image generation, no classifier run,
no reviewer recruitment was used to produce this package or this
report. Every check below is read-only verification of already-
published/already-generated files, or static/logic inspection of the
package's own code.

## 1. Source-image and prompt verification

`prep_code/verify_source_and_extract_prompts.py` re-checked all 40
source images (same files used in v1, case IDs 60–79 × 2 endpoints):

- Panel manifest sha256 matches the pinned value
  `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`. **Pass.**
- Each image's bytes (re-hashed now) match both the host completion
  record and the published `PUBLISHED_EVIDENCE_INDEX.md` row. **Pass,
  all 40.**
- Each image is exactly 512×512. **Pass, all 40.**
- The prompt and seed recorded for each case are **identical** between
  the two endpoints **and** identical to the frozen panel manifest's own
  row for that case — the prompt shown to reviewers is the manifest's
  own text, not reconstructed or paraphrased. **Pass, all 20 cases, no
  mismatches.**

Full result: `context_quality_review_s2_v2_SOURCE_VERIFICATION.json`
(host-side; contains host file paths, not republished in full).

## 2. PNG metadata (identifying-content check)

Spot-checked two of the 40 v2 package images (`R01_A.png`, `R20_B.png`
— exact byte copies of v1's already-checked source files) at the chunk
level: only `IHDR`, `IDAT`, `IEND` present, no `tEXt`/`iTXt`/`zTXt`, no
EXIF. Consistent with v1's full-scope finding on the same source
pipeline.

## 3. Reviewer-interface content scan

Case-insensitive search of the shipped `index.html` for endpoint names
(`S2_STAGE1_HISTORY`, `S2_HISTORY_BOTH_STAGES`, `stage1`, `both_stages`),
v1's neutral IDs (`Q01`–`Q20`), case-ID/seed patterns, classifier/label
terms, and the other three panel classes (`golf`, `parachute`, `french
horn` — which would hint at the wider original panel). **No match
found** except the script's own descriptive comment stating that no
such data is present. `golf`/`parachute`/`french horn`: **no match**.

## 4. Prompt injection fidelity

`prep_code/inject_prompts.py` replaced each `__PROMPT_Rnn__` placeholder
with the manifest-verbatim prompt text for that R-slot (reading the
R-id↔prompt association from the private mapping, nothing else from
it), then asserted no endpoint-identifying substring was introduced as
a side effect of injection. **Pass** — script output confirms all 20
placeholders replaced, no forbidden terms found.

## 5. Package construction fidelity

`prep_code/build_v2_reviewer_package.py`'s own build-time assertions:
every neutral package image's sha256, computed immediately after
`shutil.copyfile`, equals the corresponding verified source image's
sha256 — exact byte copies, no re-encoding, cropping, rescaling, or
annotation. A-position balance asserted: **10/10, confirmed.**

**v1 isolation, explicitly checked**: the build script hashes v1's
`PRIVATE_MAPPING.json` before and after the v2 build and asserts they
are byte-identical. Confirmed unchanged
(`bbf8a9eb8b6459bdb2e4ff33ed8638c5708fe4ecb10e58c09a02ecbdcb47689a`,
matching v1's published commitment) — this run never read from or wrote
to v1's mapping.

## 6. JavaScript logic validation (synthetic data, not published)

No browser or headless-browser tool was available in this environment
— **visual rendering, layout, and real click/keyboard event wiring were
NOT verified**, exactly as disclosed for v1, and this report does not
claim otherwise.

What *was* verified: `index.html`'s inline script (1) passes
`node --check`, and (2) was evaluated in a Node.js sandbox with stub
DOM objects, using clearly-marked synthetic responses (e.g.
`"comment":"SYNTHETIC TEST ROW"`, placeholder answers on `R01`/`R02`)
that exist only in `/data/bijaypandey/claude_tmp/scratch_check/` and are
**not** included in this deliverable. Confirmed:

- `PAIRS` has exactly 20 entries (`R01`–`R20`), each with a non-empty
  prompt mentioning "springer" (sanity check that injection used the
  right source field), referencing 40 unique image paths matching the
  40 files actually built.
- `freshState()` initializes every field empty, including the three
  fields new to v2 (`a/b_widespread_reason`, `a/b_style_mismatch`,
  `caseSpecificFeedback`) — **human fields start empty.**
- **New v2-specific logic verified**: a "yes" or "unable" widespread-
  breakdown answer without an accompanying reason is correctly treated
  as an **incomplete** pair by `isPairComplete`/`isIndividualPhaseComplete`
  as standalone functions, and once the reason is supplied in `state`
  directly, the same predicates correctly return complete.
  **Correction**: this check exercised the predicates in isolation, not
  the actual `reason-a-box`/`reason-b-box` `input` event handlers — the
  real handlers updated `state` but never called `render()`, so the
  page did not visibly unlock the paired question until the reviewer
  navigated away and back. The predicates being correct did not catch
  this, because the defect was in the handler wiring, not the
  completion logic. See `closeout/REPAIR_NOTE.md` for the handler-level
  fix and verification.
- The export payload contains all required fields and correctly
  reports `isComplete: false` for an incomplete synthetic draft; the
  required reason text is present in the export for a "yes" answer.
- An explicit answer and a genuinely missing one remain distinguishable
  in the export.
- **Correction (see the repair section below)**: this section originally
  claimed "a re-imported export round-trips without data loss." That
  claim tested only `JSON.parse(JSON.stringify(payload))` — plain data
  serialization fidelity — never the app's actual `import-file` change
  handler. The real handler (`state = imported;`) was broken and did
  not round-trip; clicking Begin after importing a real export threw
  `TypeError: Cannot read properties of undefined (reading 'id')`. This
  was not caught because the original check exercised a standalone
  reimplementation of round-tripping, not the shipped handler. See
  `closeout/REPAIR_NOTE.md` for the defect, the fix, and the corrected,
  handler-level verification that replaces this claim.

## 7. Zip deliverable check

`CONTEXT_REVIEW_PACKAGE_V2.zip` extracted cleanly to a scratch
directory; contains exactly `index.html` + `images/` (40 PNGs); spot-
checked extracted files are present at the paths `index.html`'s `PAIRS`
array references.

## What this report does not claim

- No claim that the interface renders correctly in a real browser, or
  that it is usable on any specific device — **not checked, no browser
  tool was available in this environment.**
- No claim of blinding beyond what is disclosed in
  `FROZEN_PROTOCOL_V2.md` — the prompt and erasure context are shown
  deliberately; endpoint/case identity is withheld but the images
  themselves are already public and potentially recognizable.
- No human response data exists yet for v2. This package has collected
  none, and none was simulated as if real.
- If further visual calibration beyond the three generic written
  examples were judged essential before a review could proceed, that
  need would be reported here rather than resolved by generating or
  selecting additional images — no such need was identified during this
  preparation; the three written examples are offered as sufficient for
  the user to judge instruction clarity, consistent with the task's own
  framing of this as a bounded preparation step, not a validated
  calibration.
