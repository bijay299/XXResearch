# Direct-vs-sequential RECE experiment: English springer (frozen before execution)

Written and frozen before any edit, generation, or classification in this
directory. Eligibility resolution: `ELIGIBILITY.md` (English springer
selected as a development-stage replacement for parachute; parachute's
feasibility failure preserved unchanged).

## 1. Four checkpoints compared

| id | description | source |
|---|---|---|
| M0 | pristine SD1.4 | existing, reused from `rece_golfball_confirmation/` |
| A | verified author GolfBall checkpoint (`GolfBall_ep0.pt`) | existing, reused, sha256 `62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b` |
| B | direct English-springer edit from M0 | **new, this task** |
| A->B | same English-springer edit procedure, starting from A | **new, this task** |

Only B and A->B are newly produced. M0 and A's images/predictions are
reused unchanged from `rece_golfball_confirmation/eval/{M0,rece_golfball}/`.

## 2. Published operating point (pinned source, re-verified by direct reading)

Source: `repo_src/RECE/scripts/object.sh` (unchanged from the prior task's
pin, commit `403a67fd86dce6c124b88798340e001e625b28c9`):

```
python train_object.py --concepts 'English Springer' --concept_type object \
  --target_ckpt "UCE/erased-english springer-towards_uncond-preserve_false-sd_1_4-method_replace-1-1.0.pt" \
  --regular_scale 1e-1
```

This is a **directly published** command for English springer (not an
extrapolation across concepts, unlike parachute's case) -- the same pattern
already used for Golf Ball and Parachute in the same script, with the same
`--regular_scale 1e-1` override and all other `train_object.py` defaults
(`--seed 42 --epochs 1 --emb_computing close_regzero --reg_item 1st
--ddim_steps 50 --base 1.4 --technique replace --erase_scale 1
--preserve_scale 0.1 --lamb 0.1`).

### Correction to the prior task's design doc

Re-reading `erase_methods/train_uce.py` for this task (not merely assumed
from the earlier summary) establishes that **Stage 1 (preliminary UCE) is
implemented by `train_uce.py`'s own `edit_model()` function, which uses
SINGLE-TOKEN slicing** (`final_token_idx = attention_mask.sum()-2`), the
same convention as the "official" UCE script used earlier in this project
-- **not** `erase_methods/uce.py`'s `edit_model_adversarial()` (full
77-token sum), as the prior task's `SEQUENTIAL_EDIT_DESIGN.md` stated for
Stage 1. Only **Stage 2** (adversarial refinement, inside
`train_object.py`'s epoch loop) actually calls `edit_model_adversarial()`.
Both functions edit the same 32 cross-attention to_k/to_v projections;
they differ in which token positions of the padded embedding are summed.
This correction is used throughout this experiment; see
`seq_edit/vendored_functions.py` for the verbatim vendored source of both
functions and the drift check run against the pinned repo before every
edit.

## 3. Component-by-component reference policy (frozen, explicit)

Per this task's instruction: "retain original M0 as the adversarial
embedding reference in both branches, while initializing the working model
from M0 for B and from A for A->B." This is a **sequential adaptation
retaining an M0 reference** -- a documented, scoped choice for this
development-stage test, not an author-validated sequential protocol, and
not asserted to be the only mathematically possible reference choice.

| component | B (direct) | A->B (sequential) |
|---|---|---|
| Initial working weights | M0 | A (GolfBall_ep0.pt) |
| Preliminary UCE guide | `' '` (single space) | `' '` (identical) |
| Preliminary UCE preservation outputs | `retain_texts=['']` | identical |
| Preliminary UCE hyperparameters | lamb=0.5, erase_scale=1, preserve_scale=1.0 (auto), technique=replace | identical |
| Regularization term | `regular_scale=0.1` (hyperparameter, not a checkpoint) | identical |
| Adversarial embedding reference (`model_copy` in `close_form_emb_regzero`) | pristine M0 | pristine M0 (**not** A -- explicit instruction) |
| Refinement guide | `' '` | identical |
| Refinement preservation outputs | `retain_texts=['']` | identical |
| Refinement hyperparameters | lamb=0.1, erase_scale=1, preserve_scale=0.1, technique=replace | identical |

The only difference between the two branches' entire procedure is the
**starting weights loaded into the working model before Stage 1**. Every
other input (guide text, preservation text, regularization scale,
adversarial-embedding reference model, all five hyperparameters in both
stages) is identical, by design, to isolate the effect of the starting
point alone.

### Verified branch mechanics (checked at runtime, not assumed)

`seq_edit/run_edit.py` verifies, before any editing:
- the working model's and the reference copy's cross-attention weight
  tensors do not share underlying storage (`data_ptr()` inequality) --
  edits to the working model cannot silently corrupt the reference;
- for the B branch, the freshly-loaded working model's probe weight
  equals the reference's (both are pristine M0 at that point);
- after Stage 1 and again after Stage 2, the reference copy's probe
  weight is re-checked to still equal its pre-edit value (full tensor
  equality, not just the probe slice is logged, the full eight-key
  retrieval below for the probe).

And after editing: the final checkpoint is diffed key-by-key against its
own starting point (M0 for B, A for A->B); only the 32 cross-attention
`attn2.to_k`/`attn2.to_v` projections are expected to differ, and this is
asserted, not assumed.

## 4. Frozen evaluation panel (reused, not regenerated)

Exact same 100-case manifest as `rece_golfball_confirmation/manifest/frozen_manifest.json`
(sha256 `ca1401c0064441969fdf3ab329fb9ab8eaab980b63ade62b6f14bd9bd1ef43ad`):
40 golf ball, 20 parachute, 20 French horn, 20 English springer cases,
literal/paraphrase x seeds as previously frozen. Same inference mechanics
(float32, `LMSDiscreteScheduler`, 512px, 50 steps, guidance 7.5,
`torch.cuda.manual_seed`+`generator=None`), same ResNet50_Weights.DEFAULT
evaluator and class-index map. M0 and A's images and predictions are reused
unchanged (not regenerated); only B and A->B are newly generated and
classified -- at most 200 new images total.

**Role reassignment for this task**: English springer is now the deletion
target (was: retained class). Golf ball is the historical deletion target
(tracked for relapse, not re-deleted). French horn and parachute remain
retention categories; parachute's limited baseline recognition (16/20 M0,
15/20 under A) stays visible in all reporting.

Because this manifest's existing results informed the second-concept
substitution (English springer's strong existing recognition counts were
what made it eligible), **all results from this comparison are
development evidence**, not an independent confirmatory panel. This is
stated explicitly in the final report and not described otherwise.

## 5. Frozen qualification gates (before execution, not changed after results)

- **Direct control (B) gate**: English-springer recognition must fall by
  `>=75%` relative to its OWN M0 input baseline (20/20 on this manifest).
  If this fails, the result is preserved and reported, and the sequential
  branch (A->B) is **not** executed.
- **Sequential (A->B) gate**: English-springer suppression is measured
  relative to **A's own English-springer baseline** (20/20, not M0's),
  same `>=75%` threshold. If this fails, it is reported as a sequential
  deletion failure, not reframed as a comparison of two successes.
- **Historical-relapse descriptive flags** (screening thresholds, not
  significance tests), measured relative to **A**, not M0:
  - *Material return signal*: A->B's golf-ball presence count is `>=10pp`
    higher than A's (A: 3/40 -> at least 4 additional net positives on
    this 40-case panel would cross this flag, i.e. >=7/40 combined).
  - *Loss of first-deletion qualification*: A->B's golf-ball recognition
    no longer meets the original `>=75%` suppression requirement relative
    to **M0** (A's own historical gate, re-checked, not re-derived).
  Both flags are reported regardless of whether they trip; smaller changes
  are reported too, and any apparent return is checked visually before
  being called a return.
- **Preservation (French horn, parachute)**: reported three ways for each
  class -- B vs. M0 (direct damage), A->B vs. A (incremental damage from
  the second edit alone), A->B vs. M0 (cumulative damage). No class's
  damage is attributed to the second edit without the A->B-vs-A
  comparison specifically backing that attribution.
- **Visual-quality limits** (same coarse development thresholds as the
  prior confirmation panel, reused unchanged): pooled edited-minus-input
  degeneration rate across retained+new-target cases must not be silently
  waived; corrupt images are never treated as verified deletion; ambiguity
  is preserved rather than forced.
- Missing outputs or predictions are execution errors, never successful
  deletion. No criterion is changed after seeing results, and the sample
  is not expanded after results.

## 6. Execution order (frozen)

1. Produce B from M0, generate its 100 images, classify, check the direct
   gate.
2. If the direct gate fails: stop. Report B's actual result; do not run
   A->B.
3. If the direct gate passes: produce A->B from A, generate its 100
   images, classify, check the sequential gate (vs. A's own baseline).
4. Regardless of the sequential gate's outcome, compute and report golf-ball
   relapse, French-horn/parachute incremental and cumulative preservation
   changes, and the visual-quality review for whatever was actually
   generated.

## 7. Resource plan

At most 1 genuinely idle GPU (checked via
`/home/bijaypandey/FINMLResearch/scripts/gpu_select.sh --require 1`
immediately before launch), released immediately after generation and
classification complete. All new artifacts under `/data`. No alternate
checkpoint, no parameter tuning, no additional prompt families, no sample
expansion after results, no reverse branch ordering, no extra epochs, no
reference-policy sweep, no new concepts, no additional checkpoints, no
automatic confirmatory follow-up continuation.
