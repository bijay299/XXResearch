# Context-aware human quality review, v2 — frozen protocol

**Package version**: `context_quality_review_s2_v2-v2`
**Protocol version**: `FROZEN_PROTOCOL_V2-v2`
**Frozen**: 2026-10-16, before any v2 response is collected.
**Replaces**: the proposal to reuse the unchanged v1 questionnaire for a
second review round. v1 itself, its selected published response, and
its verified unblinding result (`blinded_quality_review_s2/`) are
**unchanged and preserved** — this is a new, separate package, not a
revision of v1.

> **Repair note (`-v2`, 2026-10-16).** This revision fixes two
> interaction/resume software defects found in the first published v2
> package (`-v1`) and clarifies several instruction-wording points — it
> does **not** change the underlying qualitative criterion, model
> endpoints, prompts, image pixels, pair membership, pair order, A/B
> assignment, or any success threshold, and it is not a new research
> experiment. See `closeout/REPAIR_NOTE.md` for the full list of fixes
> and verification. No `-v1` response is known to exist; `-v1` and
> `-v2` are **not** interchangeable — the viewer rejects an import whose
> `packageVersion`/`protocolVersion` do not match exactly.

## Why a v2 package

v1's unblinded result (C34) disagreed sharply with the assistant's own
per-case labels (16/20 and 15/20 case-level agreement, overlapping on
exactly one flagged case) while also reversing the claimed aggregate
direction. One plausible contributor, among others not excluded: v1
reviewers saw the two images with **no prompt context** — no indication
of what the generation prompt asked for, or that the historical concept
was intentionally erased and its absence therefore expected. v2 adds
that context and refines the question wording and severity vocabulary
(see §3). **This is a separate development evaluation on the same
already-inspected 20-pair panel — not a fresh holdout, not an
independent replication, and no change in future labels may be
attributed specifically to "the addition of context" without further
evidence.**

## Scope — identical source material to v1

Same 20 matched case IDs (60–79), same 40 original images, same two
endpoints, same frozen panel. No regeneration, no substitution, no
pixel alteration.

- Endpoint 1: `S2_STAGE1_HISTORY`
- Endpoint 2: `S2_HISTORY_BOTH_STAGES`
- Panel: `FRESH_PANEL_MANIFEST.json`, sha256
  `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`

All 40 source images and their hashes were re-verified against both the
host completion records and the published evidence index before this
package was built (`context_quality_review_s2_v2_SOURCE_VERIFICATION.json`).
**New in v2**: the exact shared generation prompt for each pair,
extracted **verbatim from the frozen panel manifest itself** (not an
assistant's description of what either image appears to show), is shown
to the reviewer above each pair. Prompt and seed were re-verified
identical between the two endpoints and against the manifest for every
one of the 20 cases before being shown. No original-base (pre-erasure)
reference image is included — intentionally, so the task cannot become
"restore the erased concept" by implication.

## Context given to the reviewer (new in v2)

> These images come from AI image-generation models that were
> intentionally edited to remove ("erase") a specific dog breed concept
> from what they can draw. Each pair shows two images generated from
> the **same text prompt** (shown above the pair) by two different
> edited versions of the model. **The erased dog does NOT need to
> appear in either image; its absence is expected. A coherent
> substitute subject, even if unrelated to the prompt, is not a
> rendering defect by itself.** This questionnaire scores rendering
> breakdown, local defects, and optional style mismatch. It does not
> score subject or scene compliance. Style mismatch applies only when
> the prompt explicitly requests a style.
>
> **A/B is a neutral, randomized label only** — it does not mean
> "before"/"after," and the same letter does not mean the same thing
> from one pair to the next.

## Revised rubric and question set

**Term change**: v1's "whole-frame degeneration" is renamed
**"widespread rendering breakdown"** in v2, with a more explicit
qualitative anchor: *visible structural breakdown extending across much
of the scene or its major structures — pervasive fragmentation, broken
geometry, static/pixel artifacts, or smearing that destroys structure.*
This is stated explicitly as **a qualitative boundary, not a newly
validated objective metric** — no claim of improved measurement
precision is made by renaming or re-anchoring the term.

Explicitly **not** automatically a breakdown:
- A coherent abstract, patterned, or painting-like image, even where a
  photograph might otherwise be assumed — **if the prompt does not
  specify a style, photography is not assumed to have been required.**
- An unexpected subject, by itself.
- Ordinary depth-of-field blur or ordinary stylization.

Explicitly recorded **separately**, never combined into the corruption
judgment:
- Localized structural/anatomical/facial defects, affecting a specific
  part of the image (independent yes/no/unable-to-judge; local defects
  can occur with or without widespread breakdown, and both questions
  can receive "yes").
- Style mismatch — only meaningful if the prompt names a style; answered
  "style not specified" otherwise. Optional, and never merged into the
  severity score.

## Per-pair data collected (individual judgments before paired comparison)

1. Image A — widespread rendering breakdown: yes / no / unable to judge
2. **If A is "yes" or "unable"**: one brief, required reason identifying
   the visible error and its extent, or the specific ambiguity — not
   rewritten or inferred by anyone else
3. Image A — localized structural/anatomical/facial defects: yes / no /
   unable to judge
4. Image A — style mismatch (optional): yes / no / unable to judge /
   style not specified
5–8. The same four questions for Image B
9. *(revealed once every **mandatory** field above is answered — the
   two breakdown questions, the two local-defects questions, and each
   reason field whenever it applies — not the optional style fields and
   not a reason field that isn't applicable)* Which image has **more**
   widespread rendering breakdown? A / B / no clear difference / unable
   to judge — **the selected letter is the worse image, not a
   preference**
10. Optional free-text comment

No field is preselected. A draft (incomplete) export is permitted at
any time; a final export requires every mandatory field (all of 1, 3,
5, 7, 9, plus the reason field whenever a widespread answer is "yes" or
"unable") to be filled — missing answers and explicit "unable to judge"
remain distinguishable in every export, and a missing answer is never
treated as "no breakdown."

## Illustrative examples shown to the reviewer

Three short, **generic, non-study** illustrations accompany the
definitions (verbatim in `index.html` and reproduced in
`INSTRUCTIONS_PREVIEW.md`):
1. A coherent painting-like image when no style was specified in the
   prompt — not a breakdown.
2. An otherwise coherent scene with one malformed hand — a localized
   defect, not a widespread breakdown.
3. An image where fragmented, broken geometry and static-like artifacts
   cover most of the frame — a widespread breakdown.

**None of these examples uses a study image, a prior count, an
assistant label, or an endpoint-specific explanation.** Q10_B (the
image a v1 reviewer separately asked another assistant about) is
explicitly **not** used as a calibration example here. These
illustrations clarify the written definitions; they do not validate
measurement reliability, and no claim to that effect is made.

## Reviewer provenance fields (expanded in v2)

- **Alias**: free text, reviewer's choice.
- **Prior exposure to this material**: none / read summaries or labels /
  previously inspected these specific images / uncertain.
- **Case-specific feedback disclosure (new in v2)**: has the reviewer
  discussed or received feedback about any of *this review's 40 study
  images (20 pairs)* before this review? none / yes / uncertain. This targets
  exactly the kind of exposure the v1 reviewer disclosed for image Q10
  (C34).
- **Acknowledgement**: a checkbox confirming a human is performing the
  review.

## Blinding, honestly scoped (unchanged limitation from v1, restated)

The exact shared prompt and the general erasure objective are
**intentionally shown** — that is the whole point of v2. Endpoint
names, v1's neutral IDs (`Q01`–`Q20`), case IDs, seeds, prior labels
(either assistant's), classifier outputs, and conclusions are withheld
from the package, its filenames, source, and payload, exactly as in v1.
**All 40 images are already public** (published in v1's evidence
index) **and potentially recognizable or reverse-matchable** by anyone
who cross-references them; a fresh randomization does not restore
naivety or guarantee blinding against that. This is disclosed, not
claimed away.

## Randomization — independent of v1

New neutral IDs (`R01`–`R20`, distinct from v1's `Q01`–`Q20`), a fresh
Fisher-Yates shuffle seeded from a new `secrets.randbits(64)` value, and
a fresh, separately balanced (10/10) A-position assignment — all
generated independently of, and verified not to disturb, v1's private
mapping (`build_v2_reviewer_package.py` asserts v1's mapping file is
byte-identical before and after the v2 build). The v2 private mapping
and seed live at
`/data/bijaypandey/cuig_pilot/review_private/context_quality_review_s2_v2/PRIVATE_MAPPING_V2.json`,
outside the git checkout, not published until genuine v2 responses are
frozen. A public SHA256 commitment is published now
(`MAPPING_COMMITMENT_V2.json`).

## What this protocol explicitly does not do

- Does not ask any reviewer to label study images yet — this is a
  preparation deliverable only, for the user to inspect the
  instructions and arrange one human review afterward.
- Does not pool with v1, retroactively relabel v1's responses, or claim
  an independent holdout.
- Does not attribute any future change in labels specifically to the
  addition of context, without further evidence.
- Does not introduce a new severity threshold or success criterion
  beyond the qualitative rubric above.
- Does not contact, recruit, or message any reviewer.

## Bounded later plan (not executed now)

After the user has read the instructions and judged them understandable,
the user arranges one complete assessment — preferably from a human
unfamiliar with the prior claims and without study-image coaching
beforehand. When a genuine response is returned: it will be preserved
verbatim and hashed before the mapping is opened; the precommitted
mapping will be verified against `MAPPING_COMMITMENT_V2.json`; all
results and uncertainty will be reported separately (missing ≠
uncertain ≠ a specific answer); and the round will end with whatever it
finds, not be repeated to seek a preferred direction. If the quality
question remains ambiguous after this round, it will be reported as
unresolved. Any further model experiment motivated by the result would
need its own separate scientific decision and authorization — none is
implied or scheduled by this document.
