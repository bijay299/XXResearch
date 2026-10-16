# Baseline-context human quality review — frozen protocol

**Package version**: `context_quality_review_s2_baseline-v1`
**Protocol version**: `FROZEN_PROTOCOL_BASELINE-v1`
**Frozen**: 2026-10-16, before any baseline-package response is collected.
**Supersedes**: the earlier instruction to publish the v2 viewer repair
(`context_quality_review_s2_v2-v2`) as a standalone package. That repair
is **incorporated here**, not published as an intermediate step — see
§4. v1 (`blinded_quality_review_s2/`), its selected response and
verified unblinding, and the original v2 package
(`context_quality_review_s2_v2/`, including its own `-v2` repair) are
all **preserved, unchanged**. This is a new, separate, third package.

## Why a baseline-context package

The user explicitly requested this: recognizing what a scene depicts
does not establish that its faces, bodies, or objects are well formed.
Neither v1 nor v2 showed reviewers any reference for what the
*unedited* model could already produce on the same prompt, so a
reviewer had no way to judge whether something in an edited image
looked different from, or worse than, what the base model itself was
capable of. This package adds that reference, without turning it into
a competing method or a template that rewards restoring the erased
concept — see the explicit instructions reviewers are given below.

## Scope — same 40 edited images and pairing as v2, plus 20 new originals

- The same 20 matched pairs, same `R01`–`R20` neutral IDs, same pair
  order, same A/B assignment as the (now-repaired) v2 package. **The v2
  builder was not re-run; nothing was rerandomized.** The v2 private
  mapping was read **once, read-only**, to derive which original
  corresponds to each existing pair — its own hash was independently
  verified unchanged immediately before and immediately after that read
  (`bd360bc35c45d5eb68f5ff6e91aedb1db30cd70e10a9b55023bb2d330fbbc529`,
  matching the already-published commitment exactly).
- **New**: the pristine **M0** image for the same case and prompt —
  explicitly **not** `EnglishSpringer_parent` or any edited checkpoint.
  M0's own `image_report.json` records no edit checkpoint
  (`edit_ckpt: null`), confirmed programmatically, not assumed. All 20
  M0 originals for cases 60–79 were already published at commit
  `321b83679f2b4b81b1db5e51a59abff8dd7e1a59`
  (`images/history_preservation_confirmation/target_role_full_res/M0_case60_English_springer.png`
  through `..._case79_...png`); their bytes were re-hashed now (not
  trusted from the earlier record) and verified against both the M0
  `image_report.json` and the published evidence index — all 20 match,
  all are exactly 512×512, and every recorded prompt/seed matches the
  frozen panel manifest exactly.
- Generation settings were compared across all three checkpoints' own
  `image_report.json` files (M0, `S2_STAGE1_HISTORY`,
  `S2_HISTORY_BOTH_STAGES`): 512×512, 50 steps, guidance 7.5, float32,
  `LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
  beta_schedule=scaled_linear, num_train_timesteps=1000)` — **identical
  across all three, no material mismatch found.** Had one existed, it
  would have been disclosed here rather than silently substituting a
  different reference or regenerating anything.
- Panel: `FRESH_PANEL_MANIFEST.json`, sha256
  `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`
  (reused unmodified).

## Baseline identity: intentionally visible — partial masking, not full blinding

Unlike `A`/`B`, the **Original** image's role is not hidden — the
reviewer is explicitly told it is the pristine, pre-editing model
output. What stays private is only its *association* to a specific
case/endpoint pair, recorded in a new, separate, private
`BASELINE_ASSOCIATION_PRIVATE.json` (derived read-only from the v2
mapping, never published; its own SHA256 commitment,
`333e9230740ce9b636bb561e50cdcf9bc7ab81b9f071e0023a6e4f587df74edd`, is
published now, before any response, in
`BASELINE_ASSOCIATION_COMMITMENT.json`). **This is explicitly disclosed
as partial masking, not a blinding guarantee**: the 40 edited images
were already public (v1's and v2's evidence indices) and potentially
recognizable or reverse-matchable; adding a visible, genuine baseline
image does not undo or improve on that exposure, and a reviewer who
cross-references published material could in principle narrow down
which case a group belongs to. Neutral filenames (`R01_REFERENCE.png`
etc.) and the viewer's source avoid stating this directly, but cannot
prevent it.

## Three-image display

Each of the 20 groups shows **Original model | Image A | Image B**
with the shared prompt displayed above the group. On a sufficiently
wide screen the three images display side by side at equal size; on
narrower screens they stack (readable, full-width) rather than
shrinking to thumbnails. Every image (including the Original) opens
full-resolution on click. No image is cropped, enhanced, annotated, or
otherwise pixel-altered at any stage of this preparation.

## Context explained to the reviewer (verbatim, in `index.html`)

> The Original is a generated quality reference, not ground truth — it
> may itself contain defects. ... A and B's model identities remain
> hidden — A/B is a neutral, randomized label only... The erased dog
> does NOT need to appear in A or B; its absence is expected. A
> coherent substitute subject, even if unrelated to the prompt, is not
> a rendering defect by itself. Changed subjects, composition, or pose
> — or the Original and an edited image simply looking different from
> each other — do not by themselves constitute corruption: equal
> prompts and seeds do not require identical objects or geometry after
> editing. An image can depict a clearly recognizable person or object
> while still containing severe deformities — recognizing the scene
> does not mean its faces, bodies, or objects are well formed. ...
> Style mismatch applies only when the prompt explicitly requests a
> style — a painting-like image alone is not a breakdown.

## Questions retained unchanged from v2 (per image, A and B)

1. Widespread rendering breakdown: yes / no / unable to judge (reason
   required on yes/unable, live-refresh fixed — see §4)
2. Localized structural/anatomical/facial defects: yes / no / unable to
   judge — independent of (1); both can be "yes"
3. Style mismatch (optional): yes / no / unable to judge / style not
   specified

Subject/scene compliance remains explicitly **outside** this
questionnaire — not scored, and not claimed to be tracked separately
anywhere in this revision.

## New question (per image, A and B only — not asked of the Original)

> **"Compared with the Original-model image, how much visible
> structural deformation does this image have?"** Less / About the
> same / More / Unable to compare.

**Structural deformation** is defined to include malformed anatomy or
objects, broken geometry, and destructive fragmentation or smearing —
whether **local or widespread**. A brief reason is **required** for
"More" or "Unable to compare" (not for "Less"/"About the same"). If the
subjects or scenes differ too much from the Original to judge fairly,
the reviewer is instructed to choose "Unable to compare" rather than
guess. **This field is never inferred from, nor merged into, the
absolute widespread-breakdown count** — it is collected, exported, and
reported as a separate judgment. A recognizable subject does not force
a "No" on widespread breakdown, and a "No" on breakdown does not
constrain this relative question either; nothing here is auto-filled
from another field.

## The A-vs-B comparison is unchanged in meaning

The existing "which image has more widespread rendering breakdown? A /
B / no clear difference / unable to judge" question remains **the
refinement comparison** — A vs. B, not either vs. the Original. It is
unlocked by the same mandatory individual fields as before, now
including the new relative-deformation fields and their conditional
reasons (never by the optional style field or an inapplicable reason).

## What a baseline comparison can and cannot establish

Comparisons against the Original describe a reviewer's **perceived
cumulative difference after editing** — they **cannot identify which
specific editing stage** (Stage 1, refinement, or their interaction)
caused any given defect, since the Original predates every edit in this
pipeline. This is a **separate, baseline-assisted development
assessment on the already-inspected panel** — not independent
confirmation, not a fresh holdout, and not evidence that any future
change in labels was caused specifically by providing a baseline.
Exposure disclosures and "uncertain"/"unable" answers are preserved and
reported exactly as given, never resolved into a forced answer.

## Reviewer provenance fields (unchanged from v2)

Alias, prior exposure to this material, whether the reviewer has
discussed or received feedback about any of this review's **60**
images (20 groups) before now, and an acknowledgement that a human is
performing the review.

## No calibration on study images, no desired outcome implied

The same three generic, non-study illustrative examples from v2
accompany the definitions (a coherent painting when no style was
requested; a local hand defect; a widespread fragmented-structure
breakdown) — extended with one sentence noting the same definitions
apply to the new Original-comparison question too. No study image,
prior count, assistant label, or endpoint-specific explanation appears
in them. Nothing in the instructions implies a desired direction for
any answer.
