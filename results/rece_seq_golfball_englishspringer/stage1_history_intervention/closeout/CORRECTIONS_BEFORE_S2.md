# Corrections bundled ahead of the S2 Stage-1 history intervention (2026-10-15)

Bounded correction pass over existing artifacts only. No new images, no
new generation, no repeated whole-experiment review. Each item below was
produced by re-examining already-saved images, labels, or text already
in hand.

## 1. S1 golf-ball cases 12, 13, 14, 15: rubric-inconsistent `S1_Stage1` labels corrected

Re-inspected at native 512×512 across all three relevant checkpoints —
`GolfBall_parent`, `S1_Stage1`, `S1_STAGE1_HISTORY` — for case IDs 12, 13,
14, 15 (all four are the same macro-texture prompt pair: "A macro
close-up of a golf ball's dimpled surface" / "An extreme close-up
showing the dimple pattern on a golf ball", scene_index 3, seeds 5519 and
6230).

**Finding**: `GolfBall_parent`'s images for these four cases are already
abstract macro-texture renders with no recognizable golf ball — this is
expected and was already correctly labeled non-degenerate ("no golf
ball, coherent alternative content, no corruption"), consistent with the
prompt itself asking for an extreme macro close-up, not a whole-ball
composition. `S1_STAGE1_HISTORY`'s images for the same four cases are
likewise coherent macro-texture renders (ridged/fingerprint-like
contour patterns), already correctly labeled non-degenerate.

`S1_Stage1`'s images for these same four cases, however, were labeled
`degenerate: true` with the templated note "pervasive sparkle/mesh/
checkerboard or blocky-pixelation corruption, no golf ball" —
**re-inspection finds this note does not match what is in the images**:

| case | image (sha256 prefix) | what is actually visible | corruption signature present? |
|---|---|---|---|
| 12 | `995ad2d2ec22a93b` | a coherent embossed medallion/seal (eagle emblem, ridged border) | no |
| 13 | `3f614e803d239a86` | a coherent macro photo of radiating spikes/bristles (urchin- or seed-head-like) | no |
| 14 | `26ef48d4c793229d` | a coherent photo of a stitched fabric sleeve/patch | no |
| 15 | `fe8b66403f93d23e` | a coherent blue scaled/mosaic texture | no |

None of the four shows sparkle noise, a checkerboard/mesh grid artifact,
or blocky pixelation — the frozen whole-frame-degeneration rubric this
line has used throughout (see e.g. the `embedding_cross_swap` and
`history_preservation_confirmation` reports' own language: "heavy
grid/mesh/static-noise artifact corrupting most of the frame" / "blur,
mesh, or painterly distortion") is a corruption-artifact test, not an
unexpected-subject test. These four images show an unexpected subject
(the edit redirected the macro-texture prompt to an unrelated but
coherently rendered texture) rather than corruption, exactly as
`GolfBall_parent`'s own equivalent images do. Applying the rubric
consistently, **all four are corrected to `degenerate: false`**.

**Corrected counts, scoped to `stage1_history_intervention` only** (this
does not reopen or relabel `S1_BLANK`/`S1_HISTORY`, the separate
Stage-2-refined checkpoints from `history_preservation_confirmation` —
different images, out of this bounded review's scope; their published
19/20 and 20/20 counts, and `VERIFICATION_AND_CORRECTIONS_2.md`'s
denominator fix for the Stage1→BLANK/HISTORY attribution fractions,
stand as previously published):

- `S1_Stage1` golf-ball degeneration: **18/20 → 14/20** (non-degenerate
  cases: 12, 13, 14, 15, 17, 18 — six, not two).
- Paired `S1_Stage1` → `S1_STAGE1_HISTORY`: **14 recovered, 0 newly
  damaged, 6 neither** (was reported as 18 recovered, 0 newly damaged, 2
  neither). `S1_STAGE1_HISTORY` remains exactly 0/20 degenerate, so this
  is still a complete, exact recovery of every case that **was**
  correctly labeled degenerate under blank Stage-1 preservation — the
  magnitude of the claim changes (14 cases fixed, not 18), its direction
  and completeness do not (zero persisted, zero newly damaged either
  way).
- `FINAL_REPORT.md` §4's "all 5 scene contexts × both phrasings × both
  seeds (20/20 cases) flipped from degenerate to clean" is corrected: 14
  cases flipped from degenerate to clean, 6 cases (including the
  macro-texture scene, scene_index 3) were already clean under blank
  Stage-1 and remained clean — not all 20 flipped, though the final
  state (0/20 degenerate) is unchanged.

This correction *increases* the measured severity of what blank Stage-1
preservation actually damages relative to what was labeled (fewer cases
were truly degenerate at baseline than reported), while the Stage-1
historical-preservation intervention's own result is unchanged in kind:
zero degenerate cases, zero newly damaged, regardless of which baseline
count is used.

## 2. False "structurally impossible" claim removed from review notes

The per-case note used for all 20 `S1_STAGE1_HISTORY` golf-ball rows
read "...no golf ball (structurally absent in this parent lineage)" —
implying golf ball *cannot* return because the untouched parent already
erased it. This is false as a general claim: a later edit can
reintroduce a concept an earlier edit removed (nothing in UCE's
mechanism prevents it); the correct, narrower statement is only that
golf ball was **not observed** in any of these images, decided from the
image itself, not inferred from checkpoint ancestry. The note is
corrected to: "no golf ball visible in this image" (observation-only,
no ancestry-based absence claim). The underlying judgments
(`object_present: false`) are unchanged — they were already based on
actual per-image viewing, not on the ancestry assumption; only the
explanatory text contained the false claim. `build_review.py` (S1) and
the new `build_review.py` (S2, written for this task) both avoid this
phrasing.

## 3. `history_preservation_confirmation/FINAL_REPORT.md`: M0 parachute reference corrected

§6 stated retained-class recognition sits "somewhat below the M0
reference (12–13/20 parachute, 17–19/20 springer vs. **M0's 18/20
each**)." M0's actual exact-hit counts (recomputed directly from
`panel/eval/M0/classifications.jsonl`) are **parachute 11/20, English
springer 18/20** — not 18/20 for both. The parachute recognition figures
reported for the new endpoints (12–13/20) were in fact *above* the true
M0 parachute reference (11/20), not below it — the opposite of what the
sentence implied for that class. English springer's own M0 reference
(18/20) was correct as stated. Corrected inline in the report with this
note; see `CORRECTIONS.md` C29.

## 4. S2_Stage1 case 8: raw vs. visual, reaffirmed unchanged

No new finding here — carried forward from `VERIFICATION_AND_CORRECTIONS_2.md`
§4: raw classifier top-1 is `'golf ball'` at p=0.941; the frozen visual
label (recorded before classification) is "no ball," confirmed on direct
re-inspection (a golf-club/putter head and a white boundary stripe on
grass, no ball). Both records remain published side by side; visual
presence for this case is 0/20-consistent (no ball), the raw prediction
is published unmodified as a disclosed classifier false positive.

## 5. Publication-before-full-verification sequencing, disclosed

`cfd674b5de3130dc19cee60dfb5a2547f352a21e` was pushed, from the
task-owned standalone `/data` clone, after the Stage-1 intervention's
build/generation/review but the commit bundled that work together with
the then-unresolved disk-space recovery. This is a sequencing deviation
from a strict publish-only-after-every-check-is-independently-reread
discipline; it does not indicate any evidence was fabricated or
backdated. Freeze-file hashes (written before classification, as in
every endpoint in this line) establish *what* was frozen, not *when* or
*whether* each image was actually viewed — the only stronger evidence
available for viewing order is this session's own tool-call transcript,
not reproduced here as a host artifact, consistent with this line's
standing practice of not manufacturing evidence. No GPU telemetry is
invented to resolve this; where it cannot independently confirm viewing
order, that limit is disclosed rather than papered over.

## 6. Reload-verification checks: scope clarified, not retracted

The reload-verification used throughout this line (including
`S1_STAGE1_HISTORY` and the new `S2_STAGE1_HISTORY` build below) proves
**exact byte-identity between the saved checkpoint file and the
in-memory state computed immediately before saving it** (same session,
same process) — a save/load-fidelity check. It does not independently
verify that the computed state matches some externally specified
"intended" target, because no such independent target exists for these
native fp32 Stage-1 edits (unlike the fp64 reference-solver audits
elsewhere in this line — `recovery_optimality_audit`,
`numerical_optimality_audit` — which do check a closed-form solution
against an independent reference). This was not previously stated as an
equivalence to an "intended final state," so nothing is retracted; this
section exists only to make the scope explicit before it could be
over-read in the S2 report that follows.

## What this does not establish

- Items 1–3 are precision corrections found entirely from existing
  artifacts; none required new GPU computation, generation, or
  classification.
- Item 1 is the only count-level change; it does not alter the
  Stage-1-history intervention's qualitative conclusion for S1 (complete
  recovery, zero newly damaged), only its exact magnitude.
- None of these items reopens `S1_BLANK`, `S1_HISTORY`,
  `preservation_content_control`, or any other closed finding in this
  results line.
