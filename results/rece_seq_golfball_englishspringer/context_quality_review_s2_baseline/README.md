# context_quality_review_s2_baseline — README

**Status: preparation deliverable, ready for the user to inspect — not
a scientific result, and no review has been collected yet.**

## What this is

A third, separate human-review package for the same 20 English-springer
image groups used in v1 (`blinded_quality_review_s2/`) and v2
(`context_quality_review_s2_v2/`), adding a **visible pristine-M0
Original image** alongside each existing A/B pair, so a reviewer can
judge whether an edited image's faces, bodies, or objects are well
formed relative to what the unedited model already produced — not just
whether the scene is recognizable. This package **incorporates** the
already-repaired v2 viewer logic (`-v2`: live reason-entry refresh,
validated import/restore) rather than publishing that repair
separately. v1 and the original v2 package, including their own
responses/unblinding, are **unchanged**.

## What to do first (reading only — no review yet)

1. Read `INSTRUCTIONS_PREVIEW.md` (or `FROZEN_PROTOCOL_BASELINE.md` for
   full detail) and judge whether the questions are clear.
2. If useful, open `reviewer_package/index.html` directly in a browser
   to see the live setup screen and the three-image layout — this does
   not commit you to submitting anything.

## What to do when ready to arrange one review

1. Download `BASELINE_CONTEXT_REVIEW_PACKAGE.zip`.
2. Extract it.
3. Open `index.html` — no server, login, or installation required.
4. Have one reviewer (preferably someone unfamiliar with this project's
   prior claims, and not separately coached on any of these images
   beforehand) go through all 20 groups.
5. Export the responses (JSON) and return the file.

## What's different from v2

- Each group now shows **three** images — Original (pristine M0), A,
  and B — instead of two, at equal size on wide screens, stacked
  (readable) on narrow ones.
- The Original's identity is **intentionally visible** (it's explained
  as a quality reference, not ground truth); only its association to a
  specific case/endpoint stays private, in a new, separate private
  manifest with its own published commitment hash.
- A new question per edited image: "Compared with the Original-model
  image, how much visible structural deformation does this image
  have?" (Less / About the same / More / Unable to compare), kept
  strictly separate from the absolute breakdown/local-defect/style
  questions, with a required reason for More/Unable.
- Instruction wording added: the Original may itself contain defects; a
  recognizable subject does not mean anatomy/objects are well formed;
  equal prompt/seed doesn't require identical geometry after editing.
- New package/protocol identifiers and local-storage namespace
  (`context_quality_review_s2_baseline-v1`); incompatible with v1 or v2
  exports.

## Validation performed and its limits

See `VALIDATION_REPORT.md`. In short: all 40 edited images reused
byte-identical from the (repaired) v2 package; all 20 M0 references
re-verified against the M0 image_report and the published evidence
index (hashes, 512×512, prompt/seed, no edit checkpoint); generation
settings confirmed identical across M0 and both edited checkpoints; the
v2 private mapping confirmed byte-unchanged before and after this
build; package logic (including the new relative-deformation
completion gating) validated in a Node sandbox with synthetic data
excluded from the deliverable. **No browser tool was available in this
environment — visual rendering/layout was not checked**, disclosed
rather than assumed.

## Files in this directory

- `BASELINE_CONTEXT_REVIEW_PACKAGE.zip` — the reviewer deliverable (60 images)
- `FROZEN_PROTOCOL_BASELINE.md` — full frozen protocol
- `INSTRUCTIONS_PREVIEW.md` — short, readable instruction preview
- `PACKAGE_MANIFEST.json` — file list + hashes + version
- `BASELINE_ASSOCIATION_COMMITMENT.json` — public commitment to the
  (still private) baseline-association manifest
- `VALIDATION_REPORT.md` — what was checked and what wasn't
- `prep_code/` — preparation scripts (read the v2 mapping read-only;
  never rerandomize; generate no new seed of their own)

## Bounded later plan (not executed now)

After the instructions are confirmed understandable, the user arranges
one complete assessment. When returned: preserve and hash the response
verbatim before opening any private mapping, verify against the
published commitments, report all results and uncertainty separately,
and end the round with whatever it finds. No further model experiment
is authorized or scheduled by this preparation.
