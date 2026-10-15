# Reference-only RECE diagnostic: does using parent A instead of M0 as the adversarial-embedding reference rescue the sequential refinement corruption?

One bounded reference-only experiment, completed without pausing between
authorized stages. **Answer: no unqualified rescue. The reference swap
produces a small, real reduction in golf-ball-prompt corruption (35/40 to
31/40, 87.5% to 77.5%) at the cost of a measurable weakening of
English-springer deletion (0/20 to 1/20 exact-label hits, suppression 100%
to 95%). Corruption remains the majority outcome on golf-ball prompts in
both branches, and retained classes (parachute, French horn) show no
quality cost in either branch.** This is a causal diagnostic of this
specific sequential adaptation, not a new-method claim; it implicates the
adversarial-reference choice as a *contributing* factor, not the whole
story.

## 0. Carried-forward verified state (not re-derived here)

From `../refinement_ablation/FINAL_REPORT.md` and `../CORRECTIONS.md`
(commit `f368ab2030457e71f0e029fb12483591223f9757`):

- Sequential golf-ball degeneration: 2/40 (AB-UCE, Stage 1 only) -> 35/40
  (A->B, final) -- 33 newly damaged, 0 recovered, same mesh/honeycomb
  corruption artifact.
- English-springer exact recognition: 0/20 at both AB-UCE and A->B.
- Generic-dog presence on English-springer prompts: AB-UCE 6/20, A->B 6/20
  (corrected), with churn (2 recovered, 2 newly appeared, 4 persisted).
- Golf-ball exact recognition: 0/40 at both AB-UCE and A->B.
- Retained classes improve numerically through refinement: parachute
  15->18/20 (AB-UCE->A->B), French horn 18->19/20 -- descriptive, within
  ordinary single-sample noise at n=20, not estimated noise bands.

None of this was re-measured or re-derived. No eligibility, first-parent
confirmation, stage-ablation generation, or baseline search was repeated.

## 1. The single change

Both branches start Stage 2 from the identical, verified AB-UCE checkpoint
(`refinement_ablation/edits/AB_UCE.pt`, sha256
`3a9d86de1339f9c516399fde1a9ab3c6b4ff8851062fb371712d473e7862b550`) and run
the identical `close_form_emb_regzero` + `edit_model_adversarial` procedure
(`regular_scale=0.1`, one refinement epoch, `lamb=0.1`, `erase_scale=1`,
`preserve_scale=0.1`, guide `' '`, retain `['']`, technique `replace`,
K/V-only editing). **The only thing that changes is the `model_copy`
argument supplied to `close_form_emb_regzero`**: a fresh pipeline loaded
with parent A (`GolfBall_ep0.pt`, sha256
`62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b`) instead
of a fresh pristine-M0 pipeline. AB-UCE itself is never used as a
reference; the working model is never restarted from A or re-run through
preliminary UCE.

Frozen manifest (written before execution):
`FROZEN_MANIFEST.json`. Intervention script:
`run_reference_A_intervention.py`.

## 2. Verification

### 2.1 Corrected description of the published checks' actual scope

The prior experiments' own verification code (`seq_edit/run_edit.py`,
`refinement_ablation/make_uce_only_checkpoint.py`) checked less than their
logged fields implied:

| check | what the prior code actually did | strengthened in this run |
|---|---|---|
| working/reference storage separation | compared `data_ptr()` of **one** probe tensor (`up_blocks.1.attentions.1...attn2.to_k.weight`) | compared `data_ptr()` of **all 686** UNet state-dict tensors |
| reference invariance through refinement | compared the **first 8 elements** of that same one probe tensor (`probe_vector(...)[0, :8]`) before/after | compared **all 686** tensors, full values, before/after |
| changed-key count | **logged** `n_keys_changed_vs_start` and a boolean `changed_keys_are_only_cross_attn_kv`, never asserted | **asserted** the changed-key set equals an explicit 32-key whitelist (derived independently from the A-vs-AB-UCE Stage-1 diff); mismatches would raise, not just log |

No prior finding is invalidated by this -- the narrower checks happened not
to miss anything in the runs that used them (re-verified in Phase 1 below,
where the full 686-tensor comparison against the existing A->B checkpoint
came back with zero differing keys). This section corrects the
*description* of what was checked, per this task's explicit instruction.

### 2.2 Corrected description of preliminary UCE's slicing

`vendored_functions.py`'s own docstring and `refinement_ablation/FINAL_REPORT.md`
describe Stage 1's `edit_model()` as using "single-token slicing." Re-reading
the vendored code (`vendored_functions.py` lines 216-223): `final_token_idx`
is computed once per text, and the embedding is sliced as
`old_emb[final_token_idx : len(old_emb) - max(0, farthest-final_token_idx)]`
-- a **suffix slice** of the token embedding sequence (bounded by the
`farthest` adjustment between the old/new text lengths), not an index into
a single token vector. For a single-word concept this slice is often one
token wide in practice, but the mechanism itself is a token-suffix slice,
not a single-token-only operation. This corrects the characterization;
the completed Stage-1 procedure itself is unchanged and not reopened.

### 2.3 Phase 1 -- image-free replay (required gate before any new evaluation)

Because AB-UCE was reconstructed (not saved) during the original
experiment, the gate was: replay the *original* (M0-reference) Stage-2
procedure starting from the untouched AB-UCE checkpoint, and compare the
result against the existing `AtoB_english_springer.pt` final checkpoint,
tensor values (not file hash), across all 686 UNet tensors.

**Result: exact match. 0 differing keys out of 686.** (`VERIFICATION_LOG.json`,
`phase1_replay`.) This confirms AB-UCE faithfully represents the Stage-1
intermediate the original A->B run actually used -- the reconstruction is
sound, and Phase 2 (the actual intervention) was authorized to proceed.
Had this not matched exactly, this report would stop here with a
discrepancy record; no tolerance or substitute comparator was available or
used, and none was needed.

Checks performed in Phase 1, all full-tensor (not probe-based):

| check | result |
|---|---|
| AB-UCE loaded tensors verified against source checkpoint (sha256 + per-key equality) | PASS |
| Working(AB-UCE) / reference(M0) storage separation, all 686 tensors | PASS, 0 shared |
| Reference(M0) tensors unchanged through refinement, all 686 tensors | PASS, 0 changed |
| Changed keys vs. AB-UCE input = expected 32-key cross-attn K/V whitelist | PASS, exact match |
| Replay result vs. existing `AtoB_english_springer.pt`, all 686 tensors | **PASS, 0 differing keys** |

### 2.4 Phase 2 -- AB_REF_A build

A fresh, untouched AB-UCE working model and a fresh A-loaded reference
model were used (the Phase-1 replayed object was discarded, not reused or
re-refined).

| check | result |
|---|---|
| A (`GolfBall_ep0.pt`) loaded tensors verified against source checkpoint (sha256 + per-key equality) | PASS |
| Working(AB-UCE) / reference(A) storage separation, all 686 tensors | PASS, 0 shared |
| Reference(A) tensors unchanged through refinement, all 686 tensors | PASS, 0 changed |
| Changed keys vs. AB-UCE input = expected 32-key cross-attn K/V whitelist | **PASS, exact match** (32/32, no extra, no missing) |

Output: `edits/AB_REF_A.pt`, sha256
`49c8517bf5b171df34b55209b660510f852012107ff47be6ddf34df60af4582c`.

### 2.5 Diagnostic traces

| quantity | M0-reference (replay) branch | A-reference (AB_REF_A) branch |
|---|---|---|
| `G = 0.1*I + sum(W_l^T W_l)` (fixed AB-UCE working weights) | -- | -- |
| G identical across branches (bit-for-bit) | **true**, max abs diff = 0.0 | |
| concept_embedding norm | 248.758 | 248.758 (identical, as expected -- depends only on the shared working model) |
| adversarial-embedding norm | 1285.169 | 1285.682 |
| adversarial-embedding difference norm (replay vs. A-reference) | 284.265 | |
| total Stage-2 weight-update norm (32 keys, quadrature sum) | 33.003 | 33.640 |

Full per-key update norms: `UPDATE_NORMS.json`. G's equality is expected
by construction (it depends only on the working model, not the reference)
and was verified, not assumed, so this is a sanity check that nothing was
accidentally broken -- **it does not show the reference choice is
immaterial**. The reference changes `sum(W_l^T R_l)`, which is exactly
where the two branches' adversarial embeddings diverge (284-norm
difference against a ~1285-norm embedding, roughly 22% relative), and
that divergence is the only mechanistic channel through which the
downstream image differences below can arise. These traces support
interpretation; they do not by themselves establish which equation
(the embedding computation vs. the erase step) or which reference
choice is "responsible" for image-level outcomes -- no regularization or
layer sweep was run to localize further.

## 3. Evaluation (100 new images, AB_REF_A only; all other endpoints reused unchanged)

Manifest: `frozen_manifest.json`, sha256
`ca1401c0064441969fdf3ab329fb9ab8eaab980b63ade62b6f14bd9bd1ef43ad`
(unchanged). Inference settings, classifier, and per-case seeds unchanged
(`generate_confirmation.py` / `classify_confirmation.py`, reused
unmodified). GPU: idle-verified GPU 0, `CUDA_VISIBLE_DEVICES=0`.
Classification complete: 100/100, 0 errors.

### 3.1 Recognition table (exact classifier top-1 match)

| class | M0 | A | B-UCE | B (final) | AB-UCE | A->B (final, M0-ref) | **AB_REF_A (A-ref, new)** |
|---|---|---|---|---|---|---|---|
| golf ball | 39/40 | 3/40 | 40/40 | 37/40 | 0/40 | 0/40 | **0/40** (0 lit, 0 par) |
| parachute | 16/20 | 15/20 | 16/20 | 15/20 | 15/20 | 18/20 | **16/20** (7 lit, 9 par) |
| French horn | 20/20 | 18/20 | 20/20 | 19/20 | 18/20 | 19/20 | **18/20** (10 lit, 8 par) |
| English springer | 20/20 | 20/20 | 1/20 | 0/20 | 0/20 | 0/20 | **1/20** (1 lit, 0 par) |

Full per-family counts: `ANALYSIS_RESULTS.json`.

### 3.2 English springer (new-target deletion)

- **Exact recognition: 1/20 (case 84, literal, top1 "English springer",
  p=0.271).** Suppression relative to A's 20/20 baseline: 19/20 = **95%**.
  **Passes the 75% gate; does not match the existing 0/20 result** --
  this is the measured deletion cost of the reference swap.
- Breed-specific ambiguity: case 84's dog has a short, smooth coat and
  does not show the long pendant ears typical of an English springer
  spaniel -- visually it reads more like a pointer- or boxer-type hunting
  dog, even though the classifier's top-1 guess happens to land on the
  erased label. The inverse also holds (case 90, floppy-eared and more
  visually spaniel-like, is classified "Yorkshire terrier," not a hit) --
  the same breed/label decoupling already documented for the M0-reference
  branch.
- Generic dog presence: **6/20 (30%)**, cases {84, 86, 88, 90, 91, 92} --
  the identical *rate* as the corrected A->B figure (6/20), though the
  specific cases differ (different images). All 20 cases individually
  inspected at full resolution; classifier near-misses (85, 87, 93, 94,
  95) were specifically re-checked and show no dog.
- Severe degeneration: none observed on English-springer prompts (all 20
  images are coherent photographs, whether or not a dog is present).

### 3.3 Golf ball (historical target)

- **Exact recognition: 0/40 (0 literal, 0 paraphrase) -- unchanged from
  AB-UCE and A->B.** No material net return (threshold >=7/40, not met;
  actual 0). No loss of the original suppression qualification against
  M0's 39/40 (threshold >=10/40, far from met).
- **Paired presence transitions**: no case recognized as "golf ball" at
  any of AB-UCE, A->B, or AB_REF_A. The reference swap changes image
  *quality* (see 3.4), not recognized object presence.
- The 5 cases whose *quality* improved relative to A->B (28, 29, 30, 31,
  33 -- see below) were individually inspected visually: **none shows a
  recognizable golf ball.** All show a coherent hand/arm/person consistent
  with the "golfer/hand holding ball" prompt family, with the ball itself
  absent or indeterminate. The quality improvement is not an object
  return by any measure, visual or classifier-based.

### 3.4 Golf-ball quality (degeneration), three separate comparisons

| comparison | result |
|---|---|
| AB_REF_A vs. **AB-UCE** (shared Stage-1 input) | 2/40 -> 31/40 degenerate. Paired: 2 persisted, 0 recovered, **29 newly damaged**, 9 neither. |
| AB_REF_A vs. **A->B** (existing M0-reference final) | 35/40 -> 31/40 degenerate. Paired: 30 persisted, **5 recovered** (28,29,30,31,33), **1 newly damaged** (15), 4 neither. Net **+4** non-degenerate cases. |
| AB_REF_A vs. **A** (parent; context only, conflates Stage 1+2) | 7/40 -> 31/40 degenerate. Paired: 5 persisted, 2 recovered, 26 newly damaged, 7 neither. |

**AB_REF_A's own rate: 31/40 (77.5%) degenerate**, full-resolution-verified
for all 40 cases (not sampled). This is a real, modest reduction from
A->B's 87.5% -- the reference swap is not inert -- but corruption remains
the majority outcome and is the identical mesh/honeycomb artifact type in
both branches, not a different (milder) failure mode. **This is not an
unqualified rescue**: going from "almost all corrupted" to "most
corrupted" is a measurable partial mitigation, reported as such, not
rounded up to "fixed."

### 3.5 Retention (French horn, parachute)

| class | AB_REF_A recognition | AB_REF_A degeneration | vs. A->B (M0-ref final) | vs. relevant input/baseline |
|---|---|---|---|---|
| French horn | 18/20 | **0/20**, fully coherent studio/ensemble photography throughout; the 2 misses are brass-instrument-family confusions, not corruption | A->B: 19/20 recognized, 0/20 degenerate | A: 18/20 recognized, 1/20 degenerate (pre-existing, different experiment, not re-measured); AB-UCE: 18/20 recognized, 0/20 degenerate |
| parachute | 16/20 | **0/20**, fully coherent skydiving/canopy photography throughout; the 4 misses are plausible scene-framing effects (packed-canopy close-ups), not corruption | A->B: 18/20 recognized, 0/20 degenerate | A: 15/20 recognized; AB-UCE: 15/20 recognized, 0/20 degenerate |

Both retained classes show **zero quality cost in either branch**.
Recognition-count swings (parachute 18->16, French horn 19->18 going from
A->B to AB_REF_A) are single- or double-case flips at n=20, within
ordinary sampling noise, consistent with the completed experiment's own
treatment of such swings -- reported, not filtered, and not treated as a
significant effect. The corruption problem remains concentrated
specifically on golf-ball prompts in both reference conditions; this is
not a general quality collapse in either branch.

## 4. Interpretation (conditional, per this task's framing)

A substantial quality rescue with preserved measured deletion would
implicate the adversarial-reference choice as correctable. **That is not
what was found.** Instead:

- The reference swap produces a **real but partial** quality improvement
  on the historical target (35/40 -> 31/40 degenerate, net +4 clean
  cases, all in one prompt sub-family) -- not noise, not nothing, and not
  a rescue.
- That improvement carries a **measurable deletion cost** on the new
  target: English-springer exact suppression weakens from 100% (0/20) to
  95% (1/20). It still passes the 75% gate, but passing the gate does not
  establish matching the existing result, and this report does not round
  "passes the gate" up to "equivalent."
- Retained classes carry **no cost** in either branch.
- G is verified identical across reference choices at fixed working
  weights, confirming the only mechanistic channel is
  `sum(W_l^T R_l)` (the adversarial embedding), which measurably diverges
  (~22% relative norm difference) between the two reference choices. This
  is consistent with the reference choice being a *contributing* factor
  to the corruption, but the embedding-computation vs. erase-step
  localization proposed as the next step in `refinement_ablation/FINAL_REPORT.md`
  section 9 remains undone, and this result alone does not disentangle
  them.
- First interpretation, per this task's instruction: **a correctable
  issue in this specific sequential adaptation, partially but not fully
  addressed by this one intervention** -- not evidence of a new method,
  and not a failure of the author-validated sequential protocol in
  general. The panel remains development evidence (n=20/40 per class, one
  concept pair, one classifier, Assistant-only visual review with stated
  blinding limitations). Generality and novelty remain unestablished. No
  alternative reference, tuning, or follow-up experiment was launched --
  the result does not clear the bar this task set for pursuing one.

## 5. Resource usage and release

| stage | time |
|---|---|
| GPU idle check immediately before allocation (Phase 1+2) | GPU 0 confirmed IDLE at `2026-10-14T19:45:xx` (`gpu_select.sh --pick`) |
| Phase 1 (image-free replay) | 21.6s |
| Phase 2 (AB_REF_A build) | 36.5s |
| Combined verification script wall time | 71.6s |
| GPU idle re-check before generation | GPU 0 confirmed IDLE at `2026-10-14T19:47:xx` |
| Generate AB_REF_A (100 images) | 385.6s (~6m26s) |
| Classify AB_REF_A | 1.9s |
| **Total GPU-held time** | **~7m55s, one GPU (GPU 0), `CUDA_VISIBLE_DEVICES=0`** |

GPU released immediately after classification; confirmed idle again
(`gpu_select.sh`, all 4 GPUs IDLE) at `2026-10-14T19:54:53-04:00`. No
other users' jobs were present or touched throughout.

**Artifacts**: `FROZEN_MANIFEST.json`, `run_reference_A_intervention.py`,
`VERIFICATION_LOG.json`, `UPDATE_NORMS.json`, `edits/AB_REF_A.pt` +
`AB_REF_A.ref_a_intervention_log.json`, `eval/AB_REF_A/` (image report,
classifications, classify report), `review_grids/AB_REF_A/*.png`,
`VISUAL_REVIEW.json`, `ANALYSIS_RESULTS.json`, `FINAL_REPORT.md` (this
file).

## 6. One next scientific decision

**Localize within Stage 2 which equation the reference choice actually
acts through**, as already proposed (not run) in
`refinement_ablation/FINAL_REPORT.md` section 9: compare the adversarial
embedding's *direction* (not just norm, already measured here) against
each branch's own erase-step sensitivity, to determine whether the
residual ~77.5% corruption rate under the A-reference is dominated by
`close_form_emb_regzero`'s embedding computation still being
contaminated by sequential composition, or by `edit_model_adversarial`'s
erase step being inherently fragile on this weight trajectory regardless
of reference. A minimal next diagnostic (not run here, not authorized by
this task): project both branches' adversarial embeddings onto the
retained/erased concept subspaces already computable from existing
embeddings, no new images. No new final-RECE runs, no reverse ordering,
no additional checkpoints, no automatic follow-up were run beyond this
bounded diagnostic. Stopping here as scoped.
