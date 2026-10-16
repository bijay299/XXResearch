# context_quality_review_s2_v2 — README

**Status: preparation deliverable, ready for the user to inspect — not
a scientific result, and no review has been collected yet.**

## What this is

A second, separate human-review package for the same 20 matched
English-springer image pairs used in v1 (`blinded_quality_review_s2/`),
adding the generation-prompt context v1 did not show, and refining the
question wording after v1's unblinded result (C34) disagreed sharply
with the assistant's own per-case labels. v1, its selected response, and
its verified unblinding result are **unchanged** — v2 is additive, not a
replacement or correction of v1's data.

## What to do first (reading only — no review yet)

1. Read `INSTRUCTIONS_PREVIEW.md` (or `FROZEN_PROTOCOL_V2.md` for full
   detail) and judge whether the questions are clear.
2. If useful, open `reviewer_package/index.html` directly in a browser
   to see the live setup screen — this does not commit you to
   submitting anything; you can close it without answering.

## What to do when ready to arrange one review

1. Download `CONTEXT_REVIEW_PACKAGE_V2.zip`.
2. Extract it.
3. Open `index.html` — no server, login, or installation required.
4. Have one reviewer (preferably someone unfamiliar with this project's
   prior claims, and not separately coached on any of these 20 specific
   images beforehand) go through all 20 pairs.
5. Export the responses (JSON) and return the file.

## What's different from v1

- The exact shared generation prompt is shown above each pair (taken
  verbatim from the frozen panel manifest, not paraphrased).
- Explicit context: the dog concept was intentionally erased; its
  absence is expected, not a defect.
- "Whole-frame degeneration" is renamed "widespread rendering
  breakdown," with a required brief reason whenever the answer is "yes"
  or "unable to judge."
- A new, optional style-mismatch field, tracked separately from
  corruption.
- A new provenance question asks directly whether the reviewer has
  discussed or received feedback about any of these 20 specific images
  before — the exact gap disclosed in v1's C34 report.
- New randomization: neutral IDs `R01`–`R20` (not v1's `Q01`–`Q20`), a
  fresh independent seed, verified not to touch v1's mapping.

## Validation performed and its limits

See `VALIDATION_REPORT_V2.md`. In short: all 40 source images and their
shared prompts re-verified against the published evidence index and the
frozen manifest; no identifying metadata or HTML-source leaks found;
v1's mapping confirmed byte-unchanged by this build; package logic
(including the new required-reason rule) validated in a Node sandbox
with synthetic data excluded from the deliverable. **No browser tool
was available in this environment — visual rendering/layout was not
checked**, disclosed rather than assumed.

## Files in this directory

- `CONTEXT_REVIEW_PACKAGE_V2.zip` — the reviewer deliverable
- `FROZEN_PROTOCOL_V2.md` — full frozen protocol
- `INSTRUCTIONS_PREVIEW.md` — short, readable instruction preview
- `PACKAGE_MANIFEST_V2.json` — file list + hashes + version
- `MAPPING_COMMITMENT_V2.json` — public commitment to the (still
  private) v2 mapping
- `VALIDATION_REPORT_V2.md` — what was checked and what wasn't
- `prep_code/` — preparation scripts (generate a fresh random seed
  every run; never hard-code or reveal any specific run's mapping)

## Bounded later plan (not executed now)

After the instructions are confirmed understandable, the user arranges
one complete assessment. When returned: preserve and hash the response
verbatim before opening the mapping, verify against
`MAPPING_COMMITMENT_V2.json`, report all results and uncertainty
separately, and end the round with whatever it finds — not repeated
until a preferred direction appears. No further model experiment is
authorized or scheduled by this preparation.
