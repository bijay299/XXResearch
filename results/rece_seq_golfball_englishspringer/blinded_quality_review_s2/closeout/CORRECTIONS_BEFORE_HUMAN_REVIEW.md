# Bounded factual corrections, ahead of the human review package (2026-10-15)

Text-only corrections from existing records. No new GPU computation, no
image generation, no re-review of disputed quality counts — those stay
exactly as published pending the human responses this package exists to
collect.

## 1. Paired-transition label corrected

`history_both_stages_s2/FINAL_REPORT.md` §5 states the paired transition
from `S2_STAGE1_HISTORY` (direct input) to `S2_HISTORY_BOTH_STAGES`
(final) as **0 recovered, 2 persisted, 4 newly damaged, 14 neither**.
This is already the correct, published number for that specific
transition. "18 recovered" belongs to a **different** comparison —
the earlier `S2_Stage1` (blank Stage-1) → `S2_STAGE1_HISTORY`
(Stage-1-with-preservation) transition reported in
`stage1_history_intervention_s2/FINAL_REPORT.md` (18 recovered, 2
persisted, after the C31 correction). The two transitions measure
different checkpoint pairs and must not be conflated; this note exists
to make that explicit for anyone reading both reports together.

## 2. Golf-ball suppression: not uniquely best

`history_both_stages_s2/FINAL_REPORT.md`'s claim that `S2_HISTORY_BOTH_STAGES`'s
0/20 exact golf-ball hit rate is "the best of any S2 checkpoint tested"
is corrected: it **ties** the existing `S2_BLANK` and `S2_HISTORY`
refinement-only comparators, which were also 0/20. It is not uniquely
best — it matches the established floor, while additionally avoiding the
false positive `S2_Stage1` showed and the genuine single leak
`S2_STAGE1_HISTORY` showed.

## 3. Old `S2_HISTORY` parachute recognition corrected

`history_both_stages_s2/FINAL_REPORT.md`'s reference table states
`S2_HISTORY` (the old, refinement-only comparator) parachute exact-hit
count as 11/20. Re-checked directly against
`history_preservation_confirmation/panel/eval/S2_HISTORY/classifications.jsonl`:
**11/20 is correct as published** (this item was flagged for
verification; no correction was needed — recorded here so the check is
documented, not silently skipped).

## 4. Suppression numerator/denominator kept separate; the 87.5% estimate withdrawn

`history_both_stages_s2/FINAL_REPORT.md`'s golf-ball section states
"suppression even counting both ambiguous cases as present:
(16−2)/16 = 87.5%." This mixes a classifier **denominator** (the
reference 16/20) with a **visual** numerator (2 ambiguous cases from
human/assistant visual review, not classifier output) into a single
percentage. **Withdrawn.** The classifier-based suppression figure
(16→0 exact hits, 100%) and the visual observation (0 confirmed present,
2 ambiguous) are reported **separately** going forward; no single mixed
percentage is computed from them.

## 5. French horn and parachute: case-level composition, not just totals

`history_both_stages_s2/FINAL_REPORT.md` reports French horn exact
recognition as 14/20 in both `S2_STAGE1_HISTORY` and
`S2_HISTORY_BOTH_STAGES`, and parachute as 13/20 in both — true at the
**total** level, but the **case-level identity is not the same set**.
Rechecked directly from each endpoint's `classifications.jsonl`:

- French horn: `S2_HISTORY_BOTH_STAGES` **gains** cases 46 and 58
  (exact hit in BOTH_STAGES, not in STAGE1_HISTORY) and **loses** cases
  44 and 50 (exact hit in STAGE1_HISTORY, not in BOTH_STAGES) relative
  to `S2_STAGE1_HISTORY`. Net count is unchanged (14/20 both), but the
  underlying case set shifts by 4 cases, not 0.
- Parachute: the 13/20 exact-hit case identities are **unchanged**
  between the two endpoints — the same 13 cases are exact hits in both.

Neither finding changes any degeneration count; both are disclosed so
the "14/20 both, 13/20 both" totals are not read as "the identical cases
every time" when that is only true for parachute, not French horn.

## 6. Retained-class visual distribution: not claimed as independently verified beyond cases 32/54

No report in this line claims the full retained-class (parachute/French
horn) visual presence distribution was independently verified case-by-
case beyond what is actually documented. For `S2_STAGE1_HISTORY`
specifically: `closeout/CORRECTED_LABELS.json` (C31) individually
corrected **only** cases 32 and 54, and explicitly discloses that "the
remaining 38 retained-class images... were grid-reviewed for coherence/
degeneration only — not individually verified for object presence." Any
reading of a specific count like "16 present / 2 ambiguous / 2 absent"
for `S2_STAGE1_HISTORY`'s retained classes as independently verified
would overstate what was actually checked; this note exists to foreclose
that reading. (`S2_HISTORY_BOTH_STAGES`'s own retained-class review, by
contrast, *was* individually verified for all 40 cases — see
`history_both_stages_s2/FINAL_REPORT.md` §4 — and is not affected by
this item.)

## 7. The seed-concentration pattern is not statistically distinguishable from the existing saturated counts

`history_both_stages_s2/FINAL_REPORT.md`'s recommended-next-decision
note observed that 5 of 6 `S2_HISTORY_BOTH_STAGES` degenerate springer
cases use seed 6230. This is restated with an explicit limit: the
**existing** seed/scene check on this panel is already settled and
closed — `S2_Stage1`, `S2_BLANK`, and `S2_HISTORY` each show **20/20**
historical degeneration, i.e. **10/10 per seed and 4/4 per scene** in
every one of those three checkpoints. These are **saturated binary
counts** (every case degenerate) and carry no information that could
distinguish a seed effect, a scene effect, or any other mechanism — they
are consistent with literally any explanation, including none.
**Corrected**: the 6/20 `S2_HISTORY_BOTH_STAGES` pattern is not the
*first* non-saturated count in this comparison — `S2_STAGE1_HISTORY`'s
own corrected 2/20 (C31: cases 77, 79) was already non-saturated, and
precedes it. Both non-saturated counts are observational only (2 and 6
cases respectively, from two checkpoints, on one reused panel); neither
by itself establishes a seed-driven or any other mechanism, and no such
claim is made or should be inferred. This item does not reopen or
repeat the seed diagnostic — it clarifies what the already-published,
already-closed saturated counts (`S2_Stage1`/`S2_BLANK`/`S2_HISTORY`)
can and cannot show, and corrects an overclaim about which count was
first non-saturated.

## What this does not do

- Does not change any disputed quality count (`S2_STAGE1_HISTORY`'s
  2/20 vs. `S2_HISTORY_BOTH_STAGES`'s 6/20 springer degeneration, or any
  individual case label) — those are exactly what the human review
  package exists to independently assess.
- Does not reopen any closed audit, baseline, or prior correction.
- Does not run or request any new GPU computation, generation, or
  classification.
