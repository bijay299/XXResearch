# Corrections bundled ahead of the S2 both-stages build (2026-10-15)

Bounded correction pass over existing artifacts only. No new images, no
regeneration, no broad re-audit of already-closed endpoints.

## 0. Objective reframing: English springer's continued absence is desired, not a failure

`stage1_history_intervention_s2/FINAL_REPORT.md` and the prior turn's
chat summary characterized unchanged 0/20 springer recognition as a
"quality-vs-recognition dissociation" and treated it as something an
explanatory diagnostic should address. This mischaracterizes the
objective: **English springer is the historical, deliberately erased
concept in this sequence** (exactly as golf ball was for S1) — its
continued absence under `S2_STAGE1_HISTORY` is the *correct*, desired
outcome (erasure holding), not an incomplete fix needing explanation.
The prior report's "failure to restore correct recognition" framing,
and the chat summary's proposed next-step diagnostic aimed at explaining
*why* recognition wasn't restored, are both **withdrawn** — restoring
springer recognition was never the goal. "Reverse sequence" is also
withdrawn as a description of S2 relative to S1: S2 (English
springer→golf ball) is a second, independently selected sequence, not
the structural reverse of S1 (golf ball→French horn) — the two do not
share a concept pair in reverse order. The correct framing, carried into
`history_both_stages_s2/FINAL_REPORT.md`: did Stage-1 historical
preservation improve image quality on historical-class prompts **while
correctly maintaining historical erasure** (springer stays absent) and
**without weakening new-target (golf ball) suppression**? Measured
separately: exact hits, suppression gates vs. their correct references
(parent for golf, M0 for springer), and the ≥10pp historical-**return**
flag (which exists to catch the opposite failure mode — the erased
concept coming back, not staying away).

## 1. Springer cases 63, 77, 79: inherited parent content vs. added damage, reinspected

Direct side-by-side reinspection at native resolution,
`EnglishSpringer_parent` vs. `S2_Stage1` vs. `S2_STAGE1_HISTORY`:

- **Case 63** ("curled up asleep on an indoor dog bed", paraphrase, seed
  6230): `EnglishSpringer_parent`'s own image for this exact case is
  already a glowing neon-abstract scene with floating blobby white
  shapes and indistinct tiny figures — the *same* visual character as
  `S2_STAGE1_HISTORY`'s case 63, which I had labeled degenerate. This is
  an **inherited rendering pattern for this specific prompt/seed
  combination**, present before any sequential editing, not damage
  introduced by the Stage-1-with-historical-preservation edit.
  Corrected: `S2_STAGE1_HISTORY` case 63 → **non-degenerate**
  (consistent with `EnglishSpringer_parent`'s own established label for
  the same visual pattern). `S2_Stage1`'s own case 63 is a materially
  different, genuinely corrupted image (a visible dog in a bed under
  heavy sparkle/static overlay) — already correctly counted among its
  established dog-present/degenerate cases; unaffected by this
  correction.
- **Cases 77, 79** ("riding in a car... head out the window", seed
  6230): `EnglishSpringer_parent`'s versions show reasonably clear,
  coherent smiling faces (only minor structural blur on the car/roof
  rail). `S2_STAGE1_HISTORY`'s versions show pronounced facial
  warping/blurring beyond that baseline (grimacing expression, blurred
  features). This is **added distortion**, not inherited — both labels
  **stand as degenerate**, unchanged.

**Corrected count**: `S2_STAGE1_HISTORY` springer degeneration **3/20 →
2/20** (cases 77, 79 only). Paired `S2_Stage1`(20/20 established)
→`S2_STAGE1_HISTORY`: **18 recovered, 2 persisted, 0 newly damaged, 0
neither** (was 17 recovered/3 persisted).

## 2. Retained cases 32 (parachute), 54 (French horn): object-presence, not just coherence

Both were labeled `object_present: true` from grid-level coherence
alone, without per-image object verification — exactly the gap this
correction targets. Individually reinspected at native resolution:

- **Case 32** (parachute): `S2_STAGE1_HISTORY`'s image is a coherent
  lake/mountain landscape with **no parachute visible anywhere**.
  `EnglishSpringer_parent`'s own case 32 is *also* a lake/mountain scene
  with no parachute (same absence, inherited). `S2_Stage1`'s case 32
  shows a clear, unambiguous red parachute canopy. Corrected:
  `S2_STAGE1_HISTORY` case 32 `object_present` **true → false**. The
  image remains non-degenerate (coherent, no corruption) — absence of
  the retained object is a real, disclosed recognition loss, not
  excused by scene coherence.
- **Case 54** (French horn): `S2_STAGE1_HISTORY`'s image is a classroom
  scene with no French horn visible anywhere; one child in the
  foreground has a blocky, multi-colored glitch artifact covering their
  face — a genuine **local** corruption signature (not whole-frame:
  every other child, the desks, and the room itself render cleanly).
  Corrected: `object_present` **true → false**; `degenerate` stays
  **false** under the whole-frame rubric (the glitch is local, not
  pervasive), with the local artifact now explicitly noted rather than
  silently absorbed into a blanket "clean" label.

**These two corrections are scoped to cases 32 and 54 only.** The
remaining 38 retained-class (`parachute`/`French horn`) images in
`S2_STAGE1_HISTORY` were grid-reviewed for whole-frame coherence/
degeneration but **not individually verified for object presence** —
this limitation is now disclosed, not assumed away. (The new
`S2_HISTORY_BOTH_STAGES` endpoint below is individually reviewed for
all 80 cases, retained classes included, specifically to avoid
repeating this gap.)

## 3. S1 macro-case correction (C29), arithmetic propagation only

C29 corrected `S1_Stage1` golf-ball degeneration from 18/20 to 14/20
(cases 12,13,14,15,17,18 non-degenerate). Holding the existing,
unreopened final labels for `S1_BLANK` (19/20 degenerate, non-degenerate
only case 18) and `S1_HISTORY` (20/20 degenerate) fixed, the corrected
paired transitions from the corrected Stage-1 baseline are, by direct
set arithmetic (no new review):

- **Stage1 → S1_BLANK**: 14 persisted, **5 newly damaged** (cases
  12,13,14,15,17), 1 neither (case 18), 0 recovered. (14+5+1+0=20.)
- **Stage1 → S1_HISTORY**: 14 persisted, **6 newly damaged** (cases
  12,13,14,15,17,18), 0 neither, 0 recovered. (14+6+0+0=20.)

So Stage 1 accounts for **14/19** of `S1_BLANK`'s final degenerate count
and **14/20** of `S1_HISTORY`'s — not the "18/18 = 100%" originally
claimed, nor exactly "18/19 or 18/20" as C29's own text still said
(C29 corrected the *denominator* but this propagates the *numerator*
through to the corrected Stage-1 baseline fully). This is pure
arithmetic over already-fixed labels, not authorization to reopen
`S1_BLANK`/`S1_HISTORY`'s own final images.

## 4. Verification-claim honesty: the Stage-1 build script and the classifier guard

Two claims from the prior two turns are corrected, not retrofitted:

- **Reload verification never actually compared against the intended
  saved object.** `stage1_history_intervention/make_stage1_history_checkpoint.py`
  and `stage1_history_intervention_s2/make_stage1_history_checkpoint_s2.py`
  (both built on the same pattern) compare the reloaded checkpoint only
  against the **untouched parent** (to confirm which keys changed and
  that the unchanged ones are byte-identical to it) — neither script
  ever asserts that the reloaded tensor values equal `final_state_cpu`,
  the object that was actually written to disk. This is very likely true
  in practice (`torch.save`/`load` round-trips these dtypes losslessly),
  but it was never explicitly checked, and no report should have implied
  it was. `history_both_stages_s2/build_both_stages_s2.py` (this
  assignment) fixes this directly: it independently clones the intended
  final CPU state **before** saving and asserts exact key/shape/dtype/
  value equality against that specific clone after reload — see its own
  build log for the result.
- **The S2_STAGE1_HISTORY classifier guard was an after-the-fact
  check, not a true precondition gate.** `CLASSIFIER_INVOCATION_GUARD_RESULT.json`
  (prior turn) was a *separate* script run *after* classification had
  already completed, re-verifying hashes retroactively — it did not sit
  in the classification invocation path and could not have refused
  classification on failure. This assignment's own classifier guard
  (below) is built as a single wrapper that validates preconditions and
  only then calls classification, in the same guarded execution path,
  refusing on failure — a genuine fix, not a relabeling of the old one.
- **Sequencing requirement, recalibrated**: the original standing
  requirement was publication of existing evidence *before* the S1
  intervention began, not independent rereading of every prior check.
  `stage1_history_intervention/closeout/CORRECTIONS_BEFORE_S2.md` item 5
  described this gap in stricter terms than the actual requirement
  called for; this note recalibrates that language without retracting
  the substance (the publish did bundle work from before the disk-space
  recovery completed, which remains disclosed).

## What this does not establish

- Items 1–3 are precision corrections found entirely from existing
  artifacts; no new GPU computation, generation, or classification was
  used to produce them.
- None of these items reopens `S1_BLANK`, `S1_HISTORY`,
  `preservation_content_control`, or any other closed finding in this
  results line.
- Item 0 is a framing correction, not a numerical one — no counts change
  as a result of it.
