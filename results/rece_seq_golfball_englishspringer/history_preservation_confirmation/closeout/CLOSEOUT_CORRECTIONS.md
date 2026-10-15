# Closeout of `preservation_content_control` (no new GPU/replay/audit)

This document closes the previous experiment using only existing artifacts
already on disk (`preservation_content_control/*`), per the task that
authorized this closeout. No image was regenerated, no checkpoint was
rebuilt, no recovery or audit was re-run. The only new computation is two
image reinspections and two file copies (120 originals).

## 1. All 120 target originals published

`ORIGINAL_IMAGES_HASH_MAPPING.json` lists all 120 target-class originals
(golf_ball cases 0-39 and English_springer cases 80-99, both
`AB_RETAIN_GOLF` and `AB_RETAIN_HORN`), copied byte-identical from
`eval/*/imgs/` -- each copy's own sha256 is checked against the sha256
already recorded for that case in `eval/*/image_report.json` at copy time
(script asserts equality; no mismatch occurred). Only four individual crops
were previously published (`disputed_cases/`); this closes that gap.

## 2. Two case reinspections, applied as corrections (not silent edits)

**HORN case 16 (golf ball), originally labeled non-degenerate ("recovered"
vs. `AB_STABLE_SOLVE`).** Reinspected at native 512x512
(`original_images/AB_RETAIN_HORN/case16_golf_ball.png`): the upper ~35-40%
of the frame (sky) carries pervasive white speckle/sparkle texture, the
same signature this results line has used throughout to call whole-frame
degeneration; the frame overall (an incoherent red canister object, no
recognizable scene matching the prompt) does not read as successfully
recovered. **Corrected: degenerate.**

With this one label flipped and every other label unchanged:
`AB_RETAIN_HORN` golf-ball degeneration becomes **32/40 (80.0%)**, not
31/40. The blank-to-unrelated-control paired transition becomes **31
persisted, 0 recovered, 1 newly damaged (case 15), 8 neither** — the
unrelated control no longer shows even the single nominal "recovery" it
appeared to show before; it is now strictly net-negative by one case, not
net-zero. Full recomputation: `JOINT_ANALYSIS_CORRECTIONS.json`.

**HORN case 94 (English springer), originally labeled dog-absent.**
Reinspected (`original_images/AB_RETAIN_HORN/case94_English_springer.png`):
a hairy, paw-shaped limb enters water in a lake scene; it is not a clean
human foot (no clear five-toe human anatomy) and is consistent with a dog
mid-leap into water, a common water-retrieval framing for this class. This
is read, honestly, as visually ambiguous between human and animal limb, but
the rubric in use throughout this results line is "generic dog presence,"
not "unambiguous dog identity" — scored as a positive under that rubric.
**Corrected: generic dog present.** `AB_RETAIN_HORN` generic-dog presence
becomes **6/20** (cases 80, 84, 86, 90, 91, 94), not 5/20.

**GOLF cases 8, 9, 11, 12 (golf ball), reinspected, label unchanged.** All
four remain heavily hallucinated golf-course scenes (fabricated text,
flame-colored artifacts, invented skyscrapers); cases 9 and 11 show a round
pale blob near a hole/green that could be a golf ball or could be a
stylized hole/flag marker. The original-image evidence does not resolve
this either way, so the ambiguous label stands, per the task's own
instruction to keep it ambiguous absent resolving evidence. None of these
four is in `AB_RETAIN_GOLF`'s degenerate set (only case 39 is) — this is
semantic/stylistic hallucination, not the pixel-level sparkle/mesh
corruption signature used for whole-frame degeneration, and that
distinction is unaffected by this reinspection.

## 3. English-springer "0/20 in both arms" claim corrected

The published `FINAL_REPORT.md` (S5.3) and `JOINT_ANALYSIS.json` already
record the raw classifier result correctly: `AB_RETAIN_GOLF` 0/20 exact,
`AB_RETAIN_HORN` **1/20 exact (case 86, p=0.079)**. No file requires
correction here. The overstated "0/20 in both new arms" phrasing appeared
only in this assistant's own end-of-task chat summary to the user (not in
any published artifact) and is corrected here for the record: raw
classifier predictions establish **GOLF 0/20, HORN 1/20 (case 86)** — the
suppression gate still passes for both (100% / 95%), but "0/20 in both"
was never the full picture and should not be repeated.

## 4. GOLF case 30: raw hit preserved separately from the visual judgment

This was already done correctly in the published materials and is
reaffirmed here without change: `JOINT_ANALYSIS.json`'s
`golf_ball_recognition.AB_RETAIN_GOLF` carries `exact_classifier_hits: 1`
(case 30) as a distinct field from `visual_unambiguous_ball: 0`, with an
explicit note that case 30 is a confirmed classifier false positive on
direct visual re-inspection. Both numbers remain published and are not
collapsed into one "0" or one "1" anywhere.

## 5. Mathematical description corrected: R enters both H and N

`FINAL_REPORT.md` S3 states: *"the two arms differ only in N's preserve
term, not in solve conditioning"* and attributes `cond(H)`'s near-identity
between arms (1.607365e7 vs. 1.607097e7 — agreeing to 4 significant
figures, not exactly) to `H`'s dependence on `X`, "not on R." This
overstates the point. **`H = λI + e·XᵀX + p·RᵀR` depends on `R` in every
arm** — the `p·RᵀR` term is R-dependent by construction, and the small
`cond(H)` difference between `AB_RETAIN_GOLF` and `AB_RETAIN_HORN`
(16073647.42 vs. 16070966, a ~0.017% relative difference, confirmed from
`RESULT_GOLF.json`/`RESULT_HORN.json`'s own `H_diagnostics.cond` fields) is
itself the (small) footprint of that R-dependence — it is not evidence
that R is absent from H. The correct statement is: in this specific
configuration, the `p·RᵀR` term (coefficient 0.1, `‖R‖_F≈253-254`) is small
relative to the dominant `e·XᵀX` term (coefficient 1, `‖X‖_F≈1285`), so
`cond(H)` happens to be nearly — not exactly — the same across arms. **This
numerical similarity does not establish that the two arms apply equal
"effective preservation strength."** Effective preservation strength is
about how well each arm's own `V·Rᵀ≈W·Rᵀ` constraint ends up satisfied in
the solved `V` (the `preserve` component of `J_at_V_star`, which differs
between arms and is already reported per-layer in `RESULT_GOLF.json`/
`RESULT_HORN.json`), not about `H`'s conditioning, which is dominated by
the shared erase term in both arms regardless of which text is preserved.
No numeric result changes; this is a correction to the published causal
explanation only.

## 6. Unsupported "noise-level" / statistical-equivalence language removed

Two phrases in `FINAL_REPORT.md` and `JOINT_ANALYSIS.json` claim more than
a plain count supports, because no significance test was run on these
n=20/n=40 panels:

- S5.1 / `JOINT_ANALYSIS.json`: *"net zero... statistically
  indistinguishable from the blank comparator"* (blank→HORN transition).
  **Withdrawn.** After the case-16 correction (section 2 above) this
  transition is no longer even a nominal wash — it is net **-1** (one
  newly-damaged case, zero recovered). Report it as a plain count: 31
  persisted, 0 recovered, 1 newly damaged, 8 neither. No equivalence claim,
  statistical or otherwise, is supported or needed.
- S5.4: *"noise-level, not a directional cost for either arm"* (retained
  parachute/French-horn recognition, within 1 case of `AB_STABLE_SOLVE`
  across 20-case panels). **Withdrawn.** The raw counts (16/20 or 17/20 vs.
  17/20 or 16/20) are reported as-is; "noise-level" asserts a
  significance conclusion this single-draw comparison cannot support.
  No flag is triggered by the task's own retained-class degeneration gate
  (0/40 both classes, both arms) — that gate result stands untouched.

## 7. Chronology: what the available evidence does and does not show

Hashes verify that a recorded file's *content* matches what is claimed —
they say nothing about when or whether an image was actually looked at.
The only available evidence for review-before-classification chronology is
**file-write timestamps**, gathered from the host filesystem (not
manufactured for this closeout):

| artifact | mtime (America/New_York) |
|---|---|
| `eval/AB_RETAIN_GOLF/image_report.json` (generation done) | 00:30:45 |
| `eval/AB_RETAIN_HORN/image_report.json` (generation done) | 00:37:44 |
| `build_visual_review.py` (compiles the hand-written per-case judgment dicts) | 00:44:56 |
| `CASE_LEVEL_COMPLETION_RECORD.json` / `CLASSIFIER_PRECONDITION_GATE.json` / `VISUAL_REVIEW_INITIAL.json` | 00:45:02 |
| `eval/AB_RETAIN_GOLF/classify_report.json` | 00:45:28 |
| `eval/AB_RETAIN_HORN/classify_report.json` | 00:45:41 |

This ordering is **consistent with** the claimed sequence (generate both →
review → freeze labels → classify), and the ~7-minute gap between
generation finishing and the review script running is consistent with
manual image-by-image inspection having happened in that window. **It is
not independent proof of viewing**: a script that wrote judgment values
without any image ever being opened would produce the same timestamp
pattern. No separate, independent per-image viewing log exists on this
host (the Claude Code tool-call transcript that would show each image
actually being opened is not a host artifact and is not reproduced or
summarized here, consistent with the instruction not to manufacture
evidence). This limitation is disclosed plainly, matching the disclosure
already made for every earlier endpoint in this line (`CORRECTIONS.md`
C20).

## 8. Resource arithmetic, corrected from existing records

Precise logged figures (not rounded) from `RETAIN_ENCODING_PROVENANCE.json`,
`eval/*/image_report.json`, `eval/*/classify_report.json`, and
`RESULT_{GOLF,HORN}.json`:

| step | device | logged elapsed |
|---|---|---|
| retain-text encoding | GPU 0 | 1.3083s |
| generate `AB_RETAIN_GOLF` (100 images) | GPU 0 | 386.6965s |
| generate `AB_RETAIN_HORN` (100 images) | GPU 0 | 386.5433s |
| classify `AB_RETAIN_GOLF` | GPU 0 | 8.2747s |
| classify `AB_RETAIN_HORN` | GPU 0 | 8.2672s |
| **GPU total (5 allocations, logged compute only)** | | **791.0900s ≈ 13.18 min** |
| build `AB_RETAIN_GOLF` (fp64 Cholesky, 32 layers) | CPU, 8 threads | 21.7268s |
| build `AB_RETAIN_HORN` (fp64 Cholesky, 32 layers) | CPU, 8 threads | 22.0779s |
| **CPU total** | | **43.8047s** |

The previously reported "≈783s... total ~783s GPU-held across 5
allocations" in this assistant's chat summary was close but imprecise;
791.09s is the correct sum of the five logged figures. Both figures above
are **logged computation time only** — each allocation also carries
additional real GPU-occupied wall-clock time for pipeline/model loading
that is not separately logged per step, so 791.09s is a lower bound on
total GPU occupancy, not an exact wall-clock total. GPU 0 was verified
idle immediately before each of the 5 allocations and confirmed idle again
after; no other user's job was touched.

## What this closeout changes and does not change

- **Changes**: `AB_RETAIN_HORN` golf-ball degeneration 31/40 → **32/40**;
  its paired transition from blank goes from a nominal net-zero to a
  genuine net **-1**; its generic-dog presence 5/20 → **6/20**. The
  mathematical explanation for similar `cond(H)` values is corrected (R
  does enter H; the similarity is a magnitude effect, not evidence of
  R's absence). Two unsupported "noise-level"/statistical-equivalence
  phrases are withdrawn.
- **Does not change**: the headline finding. `AB_RETAIN_GOLF` (historical)
  remains 1/40, still a 96.9% relative reduction from the 31/40 blank
  baseline (unchanged, since this correction touches only the HORN arm).
  `AB_RETAIN_HORN` (unrelated control) was already shown to be no real
  improvement; it is now shown to be, if anything, slightly worse than the
  blank comparator on this panel — which strengthens, not weakens, the
  content-specificity conclusion the original report reached.
- **No new finding, no new computation beyond two reinspections and two
  file copies.** This closeout does not reopen the comparison; it is
  closed as of this document.
