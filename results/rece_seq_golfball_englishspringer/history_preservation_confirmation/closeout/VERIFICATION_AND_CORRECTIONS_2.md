# Verification and corrections to the `history_preservation_confirmation` report (round 2)

This note verifies the Stage-1 premise against saved labels and corrects
several imprecise statements from the prior turn's report and chat
summary. No new GPU computation was used to produce it — all numbers are
recomputed from already-frozen files (`panel/CASE_LEVEL_COMPLETION_RECORD_NEW_ENDPOINTS.json`,
`panel/eval/*/image_report.json`, file mtimes).

## 1. Exact counts, verified against frozen labels

| Sequence | Parent | Stage 1 | BLANK | HISTORY |
|---|---:|---:|---:|---:|
| S1: golf ball → French horn | 0/20 | **18/20** | 19/20 | 20/20 |
| S2: English springer → golf ball | 0/20 | **20/20 (exact, not approximate)** | 20/20 | 20/20 |

S2's Stage-1 count was reported as "approximately 20/20" in the prior
chat summary. Recomputing directly from
`panel/CASE_LEVEL_COMPLETION_RECORD_NEW_ENDPOINTS.json` (the frozen,
hashed label file) gives **exactly 20/20** for `S2_Stage1`, `S2_BLANK`,
and `S2_HISTORY` alike — every one of the 20 springer cases was recorded
degenerate in all three S2 endpoints. The hedge was unwarranted; the
underlying record was already exact. No case label changes.

Paired transitions (unchanged from the prior report, restated for the
premise check below): S1 parent→Stage1 introduces 18 new damaged cases;
S2 parent→Stage1 introduces 20. S1 Stage1→BLANK: 1 more newly damaged
(case 18), 0 recovered. S1 BLANK→HISTORY: 0 recovered, 1 newly damaged
(case 18) — net −1. S2 Stage1→BLANK→HISTORY: the identical 20-case
degenerate set throughout — net 0 at every step.

## 2. Premise verified: most damage is introduced at Stage 1, not refinement

- Parent → Stage-1 accounts for **18/18 (100%)** of S1's final degenerate
  count and **20/20 (100%)** of S2's — refinement (BLANK/HISTORY) adds at
  most one further case (S1) or none (S2).
- The parent is materially cleaner than Stage-1 in both sequences: 0/20
  vs. 18/20 (S1) and 0/20 vs. 20/20 (S2) — not a marginal difference.
- `S1_stage1.uce_only_log.json` / `S2_stage1.uce_only_log.json` confirm
  the native Stage-1 edit function, hyperparameters (λ=0.5, erase=1,
  preserve=1.0, guide=' ', retain=[''], replace), and exactly-32-keys-
  changed provenance are present and internally consistent.
- Comparator (`S1_Stage1` images/labels/predictions) and manifest
  (`panel/FRESH_PANEL_MANIFEST.json`, sha256 `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`)
  are both available and already verified.

**All four conditions for proceeding to the intervention hold.** Per the
task's own framing, this is not grounds for re-running the confirmation
experiment — it is the premise the intervention is designed to test.

## 3. Interpretation corrected: narrow, not general

The prior report's interpretation overstated its conclusion. The
correct, narrow statement:

**Refinement-only historical preservation did not rescue these two
extensions, with most damage already introduced at Stage 1, before
refinement (and its preservation choice) ever ran.**

This does **not** establish, and the prior report should not have
implied:
- that history preservation generally fails (only this one Stage-2-only
  application, on these two sequences, was tested);
- that the original `preservation_content_control` benefit is solely
  pair-specific (both the sequence *and* the fresh evaluation panel
  changed between that experiment and this one, so the two are not a
  matched comparison);
- that any novelty claim is established either way (no new matched
  direct-edit control was run in either experiment).

The prior report's §7 ("closing large recovery-solve numerical error...
the original dramatic result should be read as specific to the
(golf ball, English springer) pair... identifies a possible method
target") is corrected to this narrower reading. The repeatable,
consequential fact that survives is structural, not interpretive: in
both new sequences, by the time refinement's preservation choice can act
at all, the historical-class images are already almost entirely
degenerate. That is exactly the premise motivating the Stage-1
intervention below — it is not evidence about preservation in general.

## 4. Raw classifier hits vs. visual judgment — preserved, reconciled

Both records remain published side by side (`panel/eval/*/classifications.jsonl`
for raw predictions; `panel/CASE_LEVEL_COMPLETION_RECORD_NEW_ENDPOINTS.json`
for frozen visual judgments). One divergence, already caught and
disclosed: `S2_Stage1` case 8 (golf ball, new-target role) classified
top-1 `'golf ball'` at p=0.941, while the frozen visual label (recorded
before classification) was "no ball." Direct re-inspection confirms a
golf club/putter head and a white boundary stripe on grass — confirmed
classifier false positive, visual presence remains 0/20. No other
divergence was found between raw predictions and visual judgments.

## 5. Generic dogs vs. categorical historical-object absence

Kept distinct throughout, as in the original report: `object_present`
(an actual englishspringer/golf-ball instance) is recorded separately
from the narrower `dog-like animal visible` notes used only for S2's
springer cases. A dog-like shape appearing in a still-corrupted frame
(S2_Stage1: cases 62,63,66,67; S2_BLANK: 74,77; S2_HISTORY: 76,77) is
never counted as `object_present=True` for the historical class — none
of these reads as an unambiguous, undamaged springer recognition, and
none is a categorical claim that the historical object is "present" in
the sense the historical-return flag checks for. The flag itself uses
`object_present`/exact-recognition, not generic-dog presence, and was
correctly not triggered in either sequence.

## 6. Retained recognition vs. retained image quality — kept separate

Unchanged from the original report: exact-match recognition on
parachute/French-horn (S1) and parachute/French-horn (S2) sits below the
M0 reference (9–17/20 vs. 18/20) in every new endpoint, while visual
degeneration on those same retained classes is **0/40 in every
endpoint** (grid-reviewed, no escalation needed in any case). These are
reported as two separate numbers, not collapsed into one "quality"
statement, consistent with this line's established practice.

## 7. Individual vs. grid review — chat-summary overclaim corrected

The prior turn's **chat response** stated "All 720 new images... were
individually reviewed before classification." This overclaims. The
actual, already-correct figure is in `FINAL_REPORT.md` §5: **380 images
were individually inspected at native 512×512** (every historical-target
and new-target case, both target-role classes, all 9 checkpoints); the
remaining **340 purely-retained-class images were grid-reviewed** (18
fixed-order 160px contact sheets, escalated to full resolution only on
uncertainty — no case required escalation). The chat summary's "720...
individually reviewed" is corrected here; `FINAL_REPORT.md` itself did
not make this overclaim and needs no change.

## 8. Chronology — invocation-log evidence, not hashes alone

File-mtime evidence (not fabricated; gathered from the host filesystem),
extending `CORRECTIONS.md` C20's disclosure pattern:

| artifact | mtime |
|---|---|
| shared-input generation complete (`EnglishSpringer_parent/image_report.json`) | 2026-10-15 01:38:27 |
| shared-input review script run (`build_shared_input_review.py`) | 2026-10-15 01:43:34 (**+5m07s**) |
| shared-input classification complete | 2026-10-15 01:44:21 |
| 6-endpoint generation complete (`S2_HISTORY/image_report.json`) | 2026-10-15 02:24:20 |
| new-endpoint review script run (`build_new_endpoints_review.py`) | 2026-10-15 08:27:13 (**+6h02m53s**) |
| new-endpoint classification complete | 2026-10-15 08:28:08 |

Both gaps are consistent with the claimed order (generate → review →
freeze → classify). The ~6-hour gap for the 480-image batch reflects
real elapsed session time spent on image-by-image inspection (plus an
intervening disk-space incident, during which no classification was run
and no file in this record was touched). As before: **file-write
timestamps are consistent with, but do not independently prove,** that
every image was actually opened — the only stronger evidence available
is the Claude Code tool-call transcript itself, which is not a host
artifact and is not reproduced here, consistent with the instruction not
to manufacture evidence.

## 9. GPU allocation by phase, corrected

The prior chat summary said "GPU 2 used throughout." Corrected:

| phase | device |
|---|---|
| shared-input generation (M0, GolfBall_parent, EnglishSpringer_parent; 3×80 images) | **GPU 0** |
| shared-input classification (3×80) | **GPU 0** |
| Stage-1 builds ×2 | **GPU 2** |
| recovery + retain-encoding ×2 | **GPU 2** |
| 6-endpoint generation (6×80 images) | **GPU 2** |
| 6-endpoint classification (6×80) | **GPU 2** |

The switch happened because GPU 0 (and GPU 1) were claimed by another
user's job between the shared-input phase and the Stage-1-build phase;
GPU 2 was verified idle before that allocation and every allocation
after it, and GPU 0/1 were never touched again. `FINAL_REPORT.md`'s own
resource table already reports per-step seconds without naming a device
per row; this correction applies specifically to the chat summary's
"GPU 2 throughout" shorthand, not to any published artifact.

## 10. Unchanged (no new evidence to revisit)

- `AB_RETAIN_HORN` golf-ball degeneration: **32/40**, generic dogs
  **6/20** (`CORRECTIONS.md` C25) — unchanged.
- GOLF cases 8, 9, 11, 12: ambiguity preserved, unchanged.
- `AB_RETAIN_GOLF` case 30: raw classifier hit and visual false-positive
  judgment both preserved separately, unchanged.
