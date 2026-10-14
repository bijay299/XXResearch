# Eligibility resolution (from existing records only)

Resolved before any new generation, editing, or GPU use. No replacement
baseline examples were generated and no additional concepts were searched.

## 1. Parachute: feasibility decision preserved, not overridden

Source: `rece_golfball_confirmation/VISUAL_REVIEW.json` ->
`parachute_summary` and `RUN_MANIFEST.md` section 2.

- Parachute recognition under the GolfBall parent (checkpoint A): **15/20**,
  below the fixed `>=16/20` second-concept feasibility requirement.
- The one point below threshold is explained by a *visually favorable*
  detector disagreement (case 55: visually an unambiguous parachute on both
  checkpoints, edited-side classifier called it "balloon" at low confidence
  0.214). That disagreement is **not** used to round 15 up to 16 or to lower
  the threshold. The fixed `>=16/20` requirement was set before any result
  was seen, and it is honored as written.
- **Decision: parachute remains INELIGIBLE as the second-edit target.**
  This is unchanged from the prior task's finding.

## 2. English springer: both required counts checked

Source: `rece_golfball_confirmation/ANALYSIS_RESULTS.json` ->
`classes["English springer"]` (classifier-based recognition counts, not the
visual-review labels, since the frozen criterion is a recognition count).

| condition | combined | literal | paraphrase |
|---|---|---|---|
| M0 | 20/20 | 10/10 | 10/10 |
| GolfBall parent (A) | 20/20 | 10/10 | 10/10 |

- Required: `>=16/20` under M0 -> **20/20, MET**.
- Required: `>=16/20` under the GolfBall parent -> **20/20, MET**.
- Both requirements are met with no ambiguity and no missing records.

## 3. Retained-quality criteria re-checked against existing labels

Source: `rece_golfball_confirmation/VISUAL_REVIEW.json` ->
`pooled_retained_degeneration_gate`.

- Pooled edited-minus-M0 degeneration rate across the 60 retained cases
  (parachute + French horn + English springer) in the existing confirmation
  panel: **+1.67pp** (1/60 newly degenerate). Requirement `<=10pp`: **MET**.
- Per-class maximum increase: French horn +5.0pp (the largest of the three).
  Requirement: no retained class may increase by more than 20pp: **MET**.
- These figures describe the *existing* panel (GolfBall parent only, since
  that is the only edited checkpoint evaluated so far); they establish that
  the existing record contains no disqualifying retained-quality failure,
  which is what this eligibility step checks. They are not re-measured or
  re-opened here.

## 4. Decision

All existing-record requirements for substituting English springer as the
development-stage second-edit target are met, and the parachute failure is
preserved unchanged. **English springer is selected as a development-stage
replacement for parachute.**

This selection **used observed development results** (English springer's
existing 20/20 / 20/20 recognition counts, found in the course of resolving
why parachute failed its own feasibility check). It is explicitly **not**
the originally selected sequence -- the original design intended parachute
as the second-edit target, before its own feasibility check failed. All
results in this task's directory are development evidence produced after
that substitution, not an independent confirmatory panel; see
`RUN_MANIFEST.md` section 4 for how that status is carried through
reporting.

Golf ball remains the historical (first) deletion target throughout.
French horn and parachute remain retention categories; parachute's limited
baseline recognition (16/20 M0, 15/20 under the GolfBall parent) stays
visible in all reporting below -- it is a retained class in this task, not
the thing being deleted, and its borderline recognition is not treated as
evidence of anything by itself.
