# Direct-vs-sequential RECE: does English-springer deletion behave differently after golf-ball deletion?

One bounded, authorized experiment, carried through to completion without
further approval cycles. **Direct edit (B) passed its own gate cleanly. The
sequential edit (A->B) also passed its numeric gates, but the visual review
surfaces two concrete, sequential-specific quality costs that the numeric
gates cannot see** -- this report states them plainly rather than reporting
only the clean numbers.

## 1. Eligibility decision and the recorded concept substitution

Full detail: `ELIGIBILITY.md`. Parachute's `>=16/20` second-concept
feasibility check **failed** under the GolfBall parent (15/20) and that
failure is **preserved unchanged**, not overridden by its own favorable
visual disagreement (case 55, a visually-clear parachute the classifier
called "balloon"). English springer was checked against the same existing
records: `20/20` under M0 and `20/20` under the GolfBall parent (both
`>=16/20`, MET), and the existing pooled retained-degeneration gate from
the prior confirmation panel was re-checked (pooled +1.67pp <=10pp MET,
max per-class +5.0pp <=20pp MET). **English springer was selected as a
development-stage replacement for parachute, using observed development
results** -- explicitly not the originally intended sequence. All results
below are development evidence, not an independent confirmatory panel,
because the manifest's existing results informed this substitution.

## 2. Frozen settings, references, checkpoint identities, commands

Full detail: `RUN_MANIFEST.md`. Published operating point (directly from
`repo_src/RECE/scripts/object.sh`, pinned commit
`403a67fd86dce6c124b88798340e001e625b28c9`):

```
python train_object.py --concepts 'English Springer' --concept_type object \
  --target_ckpt "UCE/erased-english springer-...-1-1.0.pt" --regular_scale 1e-1
```

**Correction surfaced by re-reading the pinned source for this task**:
Stage 1 (preliminary UCE) is implemented by `erase_methods/train_uce.py`'s
own `edit_model()` (single-token slicing), not `edit_model_adversarial()`
(full 77-token sum) as the prior task's design doc stated for Stage 1. Only
Stage 2 (adversarial refinement) calls `edit_model_adversarial()`. Both
functions were vendored byte-for-byte from the pinned repo
(`seq_edit/vendored_functions.py`, drift-checked against the pinned source
at every run) specifically to avoid live-importing an unrelated
nudity/NSFW-detector (onnxruntime) dependency chain pulled in by their
respective driver modules -- the same bounded-dependency decision already
established earlier in this project.

Four checkpoints: **M0** (pristine SD1.4, reused), **A** (`GolfBall_ep0.pt`,
reused, sha256 unchanged), **B** (new: direct M0->EnglishSpringer,
`edits/B_english_springer.pt`, sha256 in `edits/B_english_springer.edit_log.json`),
**A->B** (new: same procedure from A, `edits/AtoB_english_springer.pt`).
Reference policy (frozen before execution): **M0 is retained as the
adversarial-embedding reference in both branches** -- B initializes its
working model from M0, A->B initializes from A, but both compute
`close_form_emb_regzero`'s adversarial embedding against the SAME pristine
M0 copy, never against A. This is a documented sequential adaptation, not
an author-validated protocol. Verified at runtime, not assumed: working
model and reference copy never share tensor storage (`data_ptr()` check);
the reference is bit-identical to its pre-edit snapshot after Stage 1 and
again after Stage 2, in both edits (see `edits/*.edit_log.json`); exactly
32 cross-attention `to_k`/`to_v` projections differ from each edit's own
starting point, in both edits.

**CPU dry-run status**: did not complete within a 600s cap (the authors'
own `edit_model()` recomputes a full inner 32-layer loop once per *outer*
layer index -- confirmed O(n^2) redundancy in the pinned source itself, not
assumed -- which is slow on CPU, irrelevant on GPU). No logic failure was
observed. The actual GPU run's own built-in assertions (listed above) are
what verified correctness, and all passed on both real edits.

## 3. Four-checkpoint recognition table

| class (role) | M0 | A (GolfBall parent) | B (direct) | A->B (sequential) |
|---|---|---|---|---|
| golf ball (historical target) | 39/40 (lit 20, par 19) | 3/40 (lit 2, par 1) | 37/40 (lit 18, par 19) | 0/40 (lit 0, par 0) |
| parachute (retained, limited baseline) | 16/20 (lit 8, par 8) | 15/20 (lit 8, par 7) | 15/20 (lit 8, par 7) | 18/20 (lit 8, par 10) |
| French horn (retained) | 20/20 (lit 10, par 10) | 18/20 (lit 9, par 9) | 19/20 (lit 9, par 10) | 19/20 (lit 10, par 9) |
| English springer (new target) | 20/20 (lit 10, par 10) | 20/20 (lit 10, par 10) | 0/20 (lit 0, par 0) | 0/20 (lit 0, par 0) |

(M0, A columns reused unchanged from `rece_golfball_confirmation`; B, A->B
newly generated and classified on the identical 100-case manifest.)

## 4. Direct and sequential second-target qualification

- **Direct (B) gate**: English-springer recognition must fall `>=75%`
  relative to its own M0 input baseline (20/20). Result: **20/20 -> 0/20,
  100% relative suppression. PASS.**
- **Sequential (A->B) gate**: measured relative to **A's own**
  English-springer baseline (20/20, not M0's), same `>=75%` threshold.
  Result: **20/20 -> 0/20, 100% relative suppression. PASS.**
- **Both branches pass the same threshold, but passing does not establish
  matched effectiveness** (per this task's explicit instruction) -- see
  section 7: the visual review shows these two "100%" results are not
  equally clean.

## 5. Historical-return (golf ball) counts and paired visual evidence

Measured relative to **A** (3/40), not M0, per the frozen design:

- A->B golf-ball recognition: **0/40** (down from A's 3/40). Relapse
  `-7.5pp` vs. A -- **no material return** (`material_return_flag_ge_10pp:
  false`). Cumulative suppression vs. M0: **100%**, well above the
  original 75% requirement -- `lost_first_deletion_qualification_flag:
  false`. By the frozen numeric definitions, there is **no relapse
  signal** at all; if anything, golf-ball recognition fell slightly
  further.
- **Visual check (this is the material finding the numbers hide):**
  although no golf ball is visually present in any of the 40 A->B images
  (full agreement with the 0/40 count), **32 of those 40 (80%) are
  severely degenerate** -- a heavy grid/mesh/static-noise artifact
  corrupting most of the frame, a qualitatively different and far more
  severe failure mode than anything seen on golf-ball prompts before: A's
  own golf-ball degeneration rate on the equivalent earlier panel was
  7/40 (17.5%), and B's golf-ball images on *this same* panel (untouched
  by B's edit) are 0/40 degenerate. The artifact is concentrated
  specifically on golf-ball prompts within A->B -- it does not appear in
  A->B's parachute or French-horn grids (0/20 degenerate each). Full
  detail and representative case citations: `VISUAL_REVIEW.json` ->
  `AtoB_golf_ball_historical_target_relapse_check`.

**This is not "the golf ball came back"** -- it is a severe, sequential-
specific collapse in generation quality on historical-target prompts,
co-occurring with (not contradicting) a clean recognition-count result.

## 6. Incremental / cumulative preservation and quality changes

| class | direct damage (B vs M0) | incremental (A->B vs A) | cumulative (A->B vs M0) |
|---|---|---|---|
| French horn | +5.0pp | -5.0pp (improved) | +5.0pp |
| parachute | +5.0pp | -15.0pp (improved) | -10.0pp (net improved) |

All differences are within ordinary single-sample noise at n=20 (one image
flipping = 5pp); none is attributed to the second edit specifically without
this incremental (A->B vs A) comparison backing it, and none crosses the
20pp per-class damage concern. **Visual quality for both retained classes
in both new checkpoints: 0/20 degenerate each** (B and A->B alike) --
pooled retained (French horn + parachute) degeneration stayed at or below
A's own prior rate (A: 1/40 = 2.5%; A->B: 0/40). No retained-class quality
concern was found.

**New-target (English springer) quality, both branches:**
- **B: 0/20 dog-present, 0/20 degenerate.** Checked against
  `eval/B/classifications.jsonl`: all 20 top-1 misses are non-animal
  categories (corn, cannon, football helmet, lipstick, sombrero, canoe,
  gown, miniskirt, etc.) -- a genuinely clean, category-level redirection
  away from "dog," not just a label flip.
- **A->B: 5/20 (25%) still visually show a dog**, despite also scoring
  0/20 on the exact-label metric. The classifier's own top-1 guesses for
  those five are *other dog breeds* (Border terrier, miniature poodle,
  Sussex spaniel, Yorkshire terrier, Welsh springer spaniel) at low
  confidence (0.056-0.234) -- two of them (Sussex spaniel, Welsh springer
  spaniel) are themselves close spaniel relatives of the erased concept.
  **The sequential edit moved these outputs to an adjacent point in
  dog-space rather than away from dogs entirely**, something the
  single-label metric cannot distinguish from full removal. None of these
  5 images are themselves degenerate. Full case-level citations:
  `VISUAL_REVIEW.json` -> `AtoB_english_springer_new_target`.

## 7. Verdict

- **Sequential-specific effect found, not a sequential-specific failure
  and not a direct-edit limitation.** Both branches *pass* their numeric
  qualification gates (suppression, retention, no-relapse), so this is not
  a gate failure. But the comparison is not "two matched successes": the
  sequential branch carries two concrete, reproducible quality costs that
  the direct branch does not --
  1. **80% severe visual degeneration on historical-target (golf-ball)
     prompts**, vs. 0% for the direct branch on the same prompts and 17.5%
     for checkpoint A alone, and
  2. **25% residual dog content on new-target (English-springer) prompts**
     that the exact-label metric scores as fully suppressed, vs. 0% for
     the direct branch.
  Retained-class damage and golf-ball relapse are, by contrast, **not**
  materially different between the two branches -- this is a specific,
  targeted finding, not a general "sequential editing is worse" claim.
- This result is **inconclusive as a causal mechanism finding** -- a
  single run, no repeated seeds on the edit itself, and no ablation
  isolating which of Stage 1 vs. Stage 2's reuse of a perturbed starting
  point produces the golf-ball-prompt artifact specifically. **A positive
  signal here motivates mechanism testing; it does not yet establish
  causation or novelty**, per this task's framing.
- The original screen's label, `NUMERICAL_PASS_VISUAL_CONCERN`, is
  **unchanged** and not reopened. This experiment's own pattern --
  clean numeric gates, concrete visual concerns underneath -- is
  consistent with, and reinforces, that original characterization; it is
  not described as full quality validation of either branch.

## 8. Execution time, GPU usage, release confirmation

| stage | time |
|---|---|
| GPU idle wait (background poll, 90s interval) | ~45 min (15:44:11 -> 16:29:08; other users' jobs, not touched) |
| Edit B (Stage 1 + Stage 2) | 18.9s |
| Generate B (100 images) | 6m46s |
| Classify B | 13s |
| Edit A->B (Stage 1 + Stage 2) | 41s (incl. GolfBall_ep0.pt load+verify) |
| Generate A->B (100 images) | 6m47s |
| Classify A->B | 13s |
| **Total GPU-held time** | **~15m06s, one GPU (GPU 0), CUDA_VISIBLE_DEVICES=0** |

GPU released immediately after the final classification step (process
exited; no manual release needed). Confirmed idle again via
`scripts/gpu_select.sh` at `2026-10-14T16:45:00`: GPU 0 and GPU 1 both
IDLE, GPUs 2/3's other-user occupant (`jibekgupta`) untouched throughout.

**Artifacts**: `ELIGIBILITY.md`, `RUN_MANIFEST.md`,
`seq_edit/vendored_functions.py` (drift-checked vendored source),
`seq_edit/run_edit.py` (executed edit driver, per-edit logs in
`edits/*.edit_log.json`), `orchestrate.py`, `ANALYSIS_RESULTS.json`,
`VISUAL_REVIEW.json`, `review_grids/{B,AtoB}/*.png` (fixed-order review
sheets), `eval/{B,AtoB}/` (images, classifications, reports),
`FINAL_REPORT.md` (this file).

## 9. One next scientific decision

**Investigate the mechanism behind the golf-ball-prompt-specific
degeneration under sequential editing** -- e.g., whether it stems from
Stage 2's `close_form_emb_regzero` solving an ill-conditioned regression
when the "live" basis (`mat1`) is already a once-edited matrix rather than
pristine M0, concentrated specifically along directions associated with
the first-erased concept's own embedding. A minimal next step: repeat the
A->B edit with a different `regular_scale` (diagnostic only, not a
re-litigation of this result) to see whether the golf-ball-prompt artifact
is a conditioning effect of the regularization term, or inspect the
Stage-2 adversarial embedding's own norm/direction relative to M0's golf-
ball embedding. This is explicitly **not authorized or run here** -- it is
the recommended next step, not a continuation of this task.

No alternate checkpoint, no parameter tuning, no additional prompt
families, no sample expansion after seeing results, no reverse branch
ordering, no extra epochs, no reference-policy sweep, no new concepts, no
additional checkpoints, and no automatic confirmatory follow-up followed
from this task. Stopping here as scoped.
