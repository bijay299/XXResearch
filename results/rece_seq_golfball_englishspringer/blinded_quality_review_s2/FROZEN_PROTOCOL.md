# Blinded human quality review: S2 springer-class matched pairs — frozen protocol

**Package version**: `blinded_quality_review_s2-v1`
**Protocol version**: `FROZEN_PROTOCOL-v1`
**Frozen**: 2026-10-15, before any response is collected.

## Purpose

`history_both_stages_s2/FINAL_REPORT.md` (`CORRECTIONS.md` C32) reports
that, on the assistant's own visual-review labels, English-springer
(historical-class) degeneration rises from 2/20 (`S2_STAGE1_HISTORY`) to
6/20 (`S2_HISTORY_BOTH_STAGES`) — i.e. the assistant's claim is that
full-pipeline refinement reintroduces some rendering corruption that the
Stage-1-only checkpoint did not show. **This review exists to determine
whether that claimed regression survives consistent, independent human
assessment** — not to confirm either assistant's prior interpretation.
The reviewer is not shown, and should not be asked to confirm, any
assistant conclusion.

## Scope

All 20 matched springer-class pairs, case IDs 60–79, not only the 6
cases the assistant labeled degenerate. Matching is by identical prompt
text and identical seed between the two endpoints (verified
programmatically before package construction; see
`VALIDATION_REPORT.md`).

- Endpoint 1: `S2_STAGE1_HISTORY` (input checkpoint sha256
  `797739999bc29c2978dc10f3b7ebb7c6f983e7e1738732a5ad47b7d2c4c6bf41`)
- Endpoint 2: `S2_HISTORY_BOTH_STAGES` (final checkpoint sha256
  `7fdc01c44e247ccfa23ebabd5f663283943501832894ff2e6fb23c475f1cd5c6`)
- Panel: `FRESH_PANEL_MANIFEST.json`, sha256
  `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`
  (reused, not reconstructed)

Prior assistant labels for these 40 images are **preserved unchanged**
as historical records in their original publication paths. Human
judgments collected through this package form a **new, explicitly
identified adjudication layer** — they do not overwrite, and are not
merged into, the prior labels.

## Degeneration rubric — source and version

**Source**: the "whole-frame degeneration" rubric already in continuous
use throughout this results line since its first use (e.g.
`solver_only_intervention/FINAL_REPORT.md`, `preservation_content_control/FINAL_REPORT.md`,
`history_preservation_confirmation/FINAL_REPORT.md`, and both
`stage1_history_intervention*/FINAL_REPORT.md` reports) — pervasive
sparkle/mesh/checkerboard/blocky-pixelation or blur/painterly-distortion
corruption across the frame counts as degenerate; a coherent image of an
unexpected or off-topic subject, by itself, does not.

**Version for this review**: the rubric as most recently clarified in
`history_both_stages_s2/closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` item
0 and applied in `history_both_stages_s2/FINAL_REPORT.md` — localized
defects (a single distorted face, hand, or other local structural flaw
in an otherwise clean frame) are recorded **separately** from whole-frame
degeneration, and whether a defect is *inherited* from an earlier
checkpoint in the lineage **does not** change its degeneration label for
the image being judged — only what is visible in that specific image
counts.

## Instructions given to the reviewer (verbatim, as shown in the package)

> Judge visible rendering corruption in each image — not whether the
> image shows an expected subject, or matches any particular
> description.
> - Pervasive mesh, checkerboard, sparkle/static, smearing, or
>   comparable rendering breakdown spread across the frame can count as
>   whole-frame degeneration.
> - An unusual or unexpected subject, a coherent abstract or patterned
>   image, stylization, and ordinary depth-of-field blur do **not**
>   automatically count as degeneration.
> - Localized anatomical, facial, or structural defects are recorded
>   **separately** — their presence does not by itself mean the whole
>   frame is degenerate.
> - Use **"Uncertain"** whenever you cannot confidently judge the
>   boundary. There is no penalty for using it.
> - Apply the same standard to every image.
>
> **A/B is a neutral, randomized label only — it does not mean
> "before" and "after," and the same letter does not carry meaning
> from one pair to the next.**

## Per-pair data collected

For each of the 20 pairs, in this order (paired-comparison question
enabled only after all four individual-image judgments are given):

1. Image A — whole-frame degeneration: yes / no / uncertain
2. Image A — localized structural/anatomical/facial defects: yes / no / uncertain
3. Image B — whole-frame degeneration: yes / no / uncertain
4. Image B — localized structural/anatomical/facial defects: yes / no / uncertain
5. *(revealed after 1–4)* Which image has more widespread rendering
   corruption? A / B / no clear difference / uncertain
6. Optional free-text comment

No field is preselected or defaulted to an answer. No severity
threshold, success criterion, or new primary metric is introduced by
this protocol beyond the rubric above.

## Reviewer provenance fields

- **Alias**: free text, reviewer's choice; real personal identifiers are
  not requested.
- **Prior exposure disclosure** (voluntary, self-reported): none / read
  summaries or labels describing this work / previously inspected these
  specific images / uncertain.
- **Acknowledgement**: a checkbox confirming a human is performing the
  review. This records stated provenance; it is not independent proof
  of viewing — software cannot establish that.

**Blinding limitation, disclosed, not concealed**: a reviewer who has
seen the assistant/user discussion that produced this package (including
this protocol document, if read in full) is not fully naive to the
study's existence or its general subject matter, even though the
per-pair endpoint identity, case ID, and prior labels are withheld from
the package itself. This review is reported as blinded **to
case/endpoint identity**, not as a fully naive first exposure to the
research question.

## Number of reviewers

One human reviewer is sufficient for this bounded first adjudication.
If more than one reviewer's responses are returned, each is preserved
and reported **separately** — no consensus is manufactured across
reviewers.

## Randomization

- **Pair-position randomization**: the 20 case IDs are mapped to neutral
  slots Q01–Q20 via a single Fisher-Yates shuffle, seeded at build time
  from `secrets.randbits(64)` (cryptographically-sourced, not a
  hard-coded or reproducible value).
- **A/B assignment**: for each pair, which endpoint is shown as "A" is
  randomized and balanced so each endpoint is "A" in exactly 10 of the
  20 pairs.
- The seed and the full neutral-ID↔case-ID↔endpoint mapping are recorded
  in a **private** file (`PRIVATE_MAPPING.json`, outside the git
  checkout, under `/data/bijaypandey/cuig_pilot/review_private/`) and are
  **not published** until after human responses are returned and frozen
  (Phase 7). A SHA256 **commitment** to that private file is published
  now, before any response is collected, so the mapping cannot be
  altered after the fact without detection — see `MAPPING_COMMITMENT.json`.
- The published preparation code (`prep_code/build_reviewer_package.py`)
  generates a **new, different** random seed every time it is run; it
  does not hard-code or reveal this specific run's mapping.

## Blinding limitations, disclosed

- Filenames (`Q01_A.png` etc.) and the HTML source contain no endpoint
  names, case IDs, prompts, seeds, classifier outputs, prior labels, or
  proposed conclusions — verified by direct text search of the shipped
  `index.html` and by inspecting PNG metadata chunks (see
  `VALIDATION_REPORT.md`; no identifying PNG text/EXIF chunks were
  found in the source images).
- This is a software/content blinding check, not a guarantee against
  recognition: a reviewer already broadly familiar with this results
  line's checkpoints and generation style could, in principle,
  recognize visual characteristics specific to one pipeline stage from
  the images themselves, independent of any label. That risk is
  disclosed, not something this package can eliminate.
- Public research images are not provably unrecognizable or
  non-reverse-matchable; the commitment hash provides an audit trail on
  the mapping's integrity, not a blinding guarantee.

## What this protocol explicitly does not do

- It does not ask the reviewer to confirm or dispute any assistant
  interpretation, conclusion, or prior correction.
- It does not introduce a new severity threshold, pass/fail criterion,
  or primary metric beyond the whole-frame rubric above.
- It does not use any of the 20 study images as calibration/example
  images shown to the reviewer beforehand.
- It does not transmit any response automatically; export/return is
  manual, by the reviewer and the user.
