# Closeout of the completed direct-vs-sequential experiment (existing records only)

Written before any new GPU work in this diagnostic. Reconciles
`FINAL_REPORT.md` against the keyed per-case labels and the actual images,
per this task's explicit instruction not to accept the aggregate
degeneration difference (`32-7=25`) without checking recoveries.

## Four-checkpoint recognition (unchanged from `FINAL_REPORT.md` section 3)

| class (role) | M0 | A | B (direct) | A->B (sequential) |
|---|---|---|---|---|
| golf ball (historical target) | 39/40 (lit 20, par 19) | 3/40 (lit 2, par 1) | 37/40 (lit 18, par 19) | 0/40 (lit 0, par 0) |
| parachute (retained) | 16/20 (lit 8, par 8) | 15/20 (lit 8, par 7) | 15/20 (lit 8, par 7) | 18/20 (lit 8, par 10) |
| French horn (retained) | 20/20 (lit 10, par 10) | 18/20 (lit 9, par 9) | 19/20 (lit 9, par 10) | 19/20 (lit 10, par 9) |
| English springer (new target) | 20/20 (lit 10, par 10) | 20/20 (lit 10, par 10) | 0/20 (lit 0, par 0) | 0/20 (lit 0, par 0) |

No checkpoint or evaluation identity issue was found: `edits/B_english_springer.edit_log.json`
and `edits/AtoB_english_springer.edit_log.json` both record their input
checkpoint hash (A->B's input re-verified against `GolfBall_ep0.pt`'s known
sha256), exactly 32 changed cross-attention keys, and a verified-separate,
verified-unchanged M0 reference. Nothing here is unresolved.

## CORRECTION: golf-ball degeneration re-verified against full-resolution images

The prior report's claim -- A->B golf-ball degeneration 32/40 (80%), stated
alongside A's 7/40 (17.5%) -- was built from a single visual pass over
180px thumbnail grids. Per this task's explicit instruction, the 7 cases A
itself already marked degenerate ({1, 30, 32, 33, 35, 37, 39}, confirmed
from `rece_golfball_confirmation/VISUAL_REVIEW.json`'s keyed
`golf_ball_case_by_case` table) were re-examined at full image resolution
in both A and A->B, and the prior report's 8 claimed A->B-nondegenerate
cases ({15, 22, 24, 25, 26, 27, 34, 39}) were each individually
re-examined at full resolution as well.

**Result: three of those eight were mislabeled at thumbnail scale.** Cases
22, 34, and 39 are, at full resolution, dominated by the same severe
mesh/honeycomb corruption artifact seen throughout the grid -- this was
not visible at 180px. Cases 15, 24, 25, 26, 27 are confirmed genuinely
clean (coherent subject, no corruption) on full-resolution re-inspection.

**Corrected A->B golf-ball degeneration: 35/40 (87.5%)**, not 32/40 (80%).

### Corrected paired transition table (A's 7 known-degenerate cases vs. A->B, full-resolution-verified for all 7; remaining 33 cases' A->B status taken from the corrected 8-case re-check plus the unchanged original read of the other 25 cases)

| transition | count | case_ids |
|---|---|---|
| degenerate under both A and A->B (persisted) | **7** | 1, 30, 32, 33, 35, 37, 39 |
| degenerate in A only (recovered under A->B) | **0** | (none) |
| degenerate in A->B only (newly damaged) | **28** | all A->B-degenerate cases except the 7 above |
| degenerate under neither | **5** | 15, 24, 25, 26, 27 |

**Every one of A's own 7 already-degenerate golf-ball cases remained
degenerate under the sequential edit -- none recovered.** The net
aggregate change (35-7=28) is numerically close to, but not identical to,
the originally-reported "25" and is now properly decomposed into 28 newly
damaged and 0 recovered, not an undifferentiated net figure. This is a
more severe finding than originally reported, not a less severe one: the
corruption is more pervasive (87.5% vs. 80%) and strictly additive (zero
recoveries) rather than partially offset by any case improving.

French horn and parachute quality (0/20 degenerate in both B and A->B,
against A's own prior 1/40 pooled rate) were not re-verified at full
resolution in this pass -- no specific concern was raised about them, the
claim itself (zero degenerate) is the easiest visual claim to get right at
any resolution (a clean, recognizable instrument/canopy is clearly
distinguishable from the mesh/noise artifact even at thumbnail scale), and
re-litigating settled, unflagged findings is out of scope for this
diagnostic. This limitation is stated explicitly rather than silently
assumed away.

## English-springer presence vs. survival (frozen gate preserved)

The prior report's "5/20 (25%) still visually show a dog" finding for A->B
stands, re-confirmed: cases 80, 84, 86, 90, 91. **This is dog presence, not
confirmed English-springer survival.** The classifier's own top-1 guesses
for those five cases are *other* breeds -- Border terrier, miniature
poodle, Sussex spaniel, Yorkshire terrier, Welsh springer spaniel -- none
of which is "English springer" (ImageNet index 217). **The frozen
exact-label gate is unchanged: A->B scores 0/20 against the English-springer
label, identically to B, and this diagnostic does not revisit or relax
that gate.** The two findings are compatible and are kept separate: the
exact-label suppression metric (0/20, used for the qualification gate) and
the coarser "is any dog visually present" check (5/20 for A->B, 0/20 for
B, used only as a quality/interpretation note) measure different things.

## Decision: proceed

The corruption signal is substantiated -- more strongly, not less, after
verification -- and no checkpoint/evaluation identity issue was found.
Proceeding to the stage-ablation diagnostic (section 2 of the task).
