# S2 Both-Stages: Does the Stage-1 Quality Gain Survive Full-Pipeline Refinement?

**Question.** `stage1_history_intervention_s2` showed that applying
historical preservation at Stage 1 substantially improves image quality
on S2's historical-class (English springer) prompts while correctly
maintaining springer's erasure (the desired outcome — springer is the
historical, deliberately erased concept, not a target for recovery).
This report completes the pipeline: applying the same, already-validated
Stage-2 refinement (fp64 stable solve, historical preservation) to the
improved `S2_STAGE1_HISTORY` checkpoint. Does the Stage-1 quality gain
survive full refinement, while golf-ball deletion holds or improves and
retained classes (parachute, French horn) are not damaged? Existing-
method control, using the already-validated numerical implementation;
not a new method, not a claim about unmodified upstream RECE.

**Answer: mostly yes, with one genuine, disclosed cost.** Golf-ball
(new-target) suppression is now **complete at the classifier level**
(0/20, improving on `S2_STAGE1_HISTORY`'s 1/20 genuine leak and
`S2_Stage1`'s own 1/20 false-positive) and visually near-complete (0
confirmed, 2 ambiguous). English springer stays correctly erased (0/20
throughout, as desired — this is not a failure). French horn recognition
is tied for best among every S2 checkpoint tested (14/20), and parachute
improves over the refinement-only `S2_HISTORY` comparator (13/20 vs.
9/20). But **springer degeneration on historical-class prompts rises
from 2/20 (corrected `S2_STAGE1_HISTORY`) to 6/20** — the full pipeline
does not simply inherit Stage-1's quality gain unchanged; refinement
reintroduces some corruption, concentrated in 3 of 5 scenes and 5 of 6
cases at one specific seed. This is a genuine tradeoff, not a clean
"fix persists" result.

---

## 0. Objective correction carried forward

See `closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` item 0. English
springer's continued 0/20 recognition throughout this results line is
the **desired** outcome (erasure holding), not a shortfall — the
question this report answers is whether refinement preserves image
*quality* on springer prompts while keeping springer correctly absent
and golf ball correctly suppressed, not whether springer "comes back."

## 1. Inputs verified, nothing rebuilt

| | sha256 | match |
|---|---|---|
| `S2_STAGE1_HISTORY.pt` (working UNet) | `797739999bc29c2978dc10f3b7ebb7c6f983e7e1738732a5ad47b7d2c4c6bf41` | exact |
| `B_english_springer.pt` (provenance only) | `c711e97127d3dcca9d6752e86d64fba8240ed20347c699179f9fe0859267b9dc` | exact |
| `S2_stage1.pt` (blank comparator, reused) | `76c8740e543b90b5b065aceb7bb7772f186b9f3a9fd38f81d4d16b1c6d3895ee` | exact |
| `S2_HISTORY.pt` (refinement-only comparator, reused) | `1b6b4c9ee79436b6b3bfc4e6a1cf90b17c84c4d0b639f2c5d64ac193efa26b68` | exact |
| `guide_S2.pt` (reused, not re-encoded) | `3d1919279165c3e1ba25e09224e6d13af6301b7fe2341f65e07d9516725a032c` | exact |
| `retain_S2_historical.pt` (reused, not re-encoded) | `6dec8c005e362bffa89d55a2775491c62e675e58e46d1bad67a69a2dc0fe6672` | exact |
| `FRESH_PANEL_MANIFEST.json` (reused) | `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2` | exact |

SD1.4 snapshot and RECE commit confirmed unchanged via
`vf.assert_pinned_sources_unchanged()`. The old `E_S2_newtarget.pt` (from
the corrupted `S2_Stage1` checkpoint) was never read; a fresh embedding
was recovered from `S2_STAGE1_HISTORY` instead (below).

## 2. Recovery (GPU, once) — storage separation and invariance verified

Working UNet = `S2_STAGE1_HISTORY` (loaded, hash-verified). Reference =
separate pristine M0. `close_form_emb_regzero(concept='golf ball',
regular_scale=0.1, with_to_k=True, seed=42)`. Verified: 0 shared tensors
between working/reference pipelines; neither pipeline's weights changed
during recovery; recovered embedding all-finite. 33.4s, GPU 0 (verified
idle immediately before allocation, released after).
`E_S2_BOTH_STAGES_newtarget.pt`, sha256
`34955a096af2c0dfc92c820f1af97c2aca5484b0d6d7396675d8555aa4073021`,
norm 938.01. Guide and retain operands reused unmodified (hashes above),
not re-encoded.

## 3. Refinement (CPU-only fp64, once) — build checks

One update, replace, all 32 cross-attention K/V weights. λ=0.1,
erase_scale=1, preserve_scale=0.1. W = each layer's untouched
`S2_STAGE1_HISTORY` weight; X = the freshly recovered embedding; G/R =
the reused guide/retain operands. H/N assembled in fp64, solved via
Cholesky.

- **Reference backward error**: 2.57e-17 to 3.29e-17 on all 32 layers
  (≪1e-10 tolerance) — `reference_solve_fully_valid: true`.
- **Quadratic-gap identity**: passes on all 32 layers (candidate = input
  W) — `quadratic_gap_identity_fully_valid: true`.
- **Reload verification, corrected this time** (see
  `closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` item 4): before saving,
  the intended final CPU state was independently cloned; after reload,
  every key/shape/dtype/value was asserted **exactly equal** to that
  specific clone — `exact_equality_vs_independently_cloned_intended_final_state.all_keys_match:
  true`, 0 mismatched keys. Separately: changed-key set equals the
  explicit 32-key whitelist exactly (0 extra, 0 missing), the other 654
  tensors are byte-identical to the `S2_STAGE1_HISTORY` input, all 686
  tensors finite.
- Checkpoint sha256
  `7fdc01c44e247ccfa23ebabd5f663283943501832894ff2e6fb23c475f1cd5c6`.
  18.3s total, CPU-only (`CUDA_VISIBLE_DEVICES=""`, asserted).

## 4. Generation and review — mandatory order honored

80 images on the reused, hash-verified manifest, 309.8s, GPU 0 verified
idle immediately before allocation, confirmed released (0 MiB/0% on
every GPU) after. Inference model's loaded state confirmed to match the
saved checkpoint exactly (`post_load_values_match_saved_tensors: true`).

**All 80 images individually inspected at native 512×512, including
both retained classes** — not grid-only, correcting the gap that missed
cases 32/54 in the prior endpoint (see
`closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` item 2). Labels frozen and
hashed (`CLASSIFIER_PRECONDITION_GATE.json`, `VISUAL_REVIEW_INITIAL.json`,
`CASE_LEVEL_COMPLETION_RECORD.json`) **before** classification ran.

**Classifier guard, genuinely enforced this time**: `guarded_classify.py`
is a single executed wrapper — it re-verifies the frozen hashes, the
exact 80-case ID set with completed judgments, every image file's hash
against the completion record, and the classification input's agreement
with that record, **in this same process**, logging the precondition
pass with a timestamp, and only then invokes classification in the same
guarded path (`CLASSIFIER_GUARD_INVOCATION_LOG.json`). This corrects the
prior endpoint's guard, which was a separate script run after
classification had already completed (disclosed in
`closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` item 4) — this one could
have refused and did not need to.

**Assistant-review disclosure**: all 80 judgments above were recorded by
this assistant viewing each image; there is no independent human
validation layer in this pipeline, consistent with every prior endpoint
in this results line.

## 5. Joint outcomes

Full tables: `CASE_LEVEL_COMPLETION_RECORD.json`,
`eval/S2_HISTORY_BOTH_STAGES/classifications.jsonl`.

### Reference exact-hit counts, all checkpoints, all /20

| | golf ball | English springer | French horn | parachute |
|---|---:|---:|---:|---:|
| M0 | 18 | 18 | 18 | 11 |
| EnglishSpringer_parent | 16 | 0 | 14 | 15 |
| S2_Stage1 (blank Stage-1) | 1 (false positive) | 0 | 11 | 17 |
| S2_STAGE1_HISTORY (Stage-1 only) | 1 (genuine) | 0 | 14 | 13 |
| S2_BLANK (refinement only, blank) | 0 | 0 | 11 | 12 |
| S2_HISTORY (refinement only, historical) | 0 | 0 | 9 | 11 |
| **S2_HISTORY_BOTH_STAGES** | **0** | **0** | **14** | **13** |

### Historical springer: quality and continued erasure

**Exact/visual recognition**: 0/20 in every checkpoint from
`EnglishSpringer_parent` onward, including this one — unchanged, as
desired. The ≥10pp historical-**return** flag (≥2 net additional hits
vs. parent on 20 cases) is **not triggered** — 0 vs. 0. Suppression
vs. M0's 18/20 remains at its full 100% throughout.

**Degeneration (quality)**: **6/20** (cases 61, 63, 68, 71, 77, 79),
corrected-baseline paired against `S2_STAGE1_HISTORY`'s 2/20 (cases 77,
79 only — see `closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md` item 1):

| transition | count | case IDs |
|---|---:|---|
| persisted (degenerate in both) | 2 | 77, 79 |
| newly damaged (clean→degenerate) | 4 | 61, 63, 68, 71 |
| recovered (degenerate→clean) | 0 | — |
| neither (clean in both) | 14 | — |

**Refinement reintroduces some corruption that Stage-1 alone had fixed
— the Stage-1 gain does not fully survive.** All 6 degenerate cases
fall in 3 of 5 scenes (scene 0 "dog bed," scene 2 "grooming," scene 4
"car window"); scenes 1 ("stick-fetching") and 3 ("snow") show zero
degeneration in either arm. 5 of 6 degenerate cases use seed 6230 (only
case 68 uses 5519) — a seed-concentrated pattern, disclosed as an
observation, not a claim that the two seeds are statistically
distinguishable from 20 cases. Inherited-vs-added was checked
case-by-case against `EnglishSpringer_parent`/`S2_Stage1` (item 1 of the
corrections); none of the 6 is an inherited parent pattern — all are
genuinely new to this checkpoint or (for 77/79) a continuation of
`S2_STAGE1_HISTORY`'s own added distortion. Still no generic dog-like
shape in any of the 20 cases (one ambiguous indistinct form in case 63,
recorded as uncertain, not asserted as a dog).

### Golf-ball (new-target) deletion

Gate: ≥75% suppression against the **original `EnglishSpringer_parent`'s
16/20** (not `S2_Stage1`'s own 1/20, which was itself already a false
positive — see `closeout/CORRECTIONS_BEFORE_S2.md`).

- **Exact (classifier)**: 0/20 — **100% suppression**, the strongest of
  any S2 checkpoint tested including the all-blank arms.
- **Visual**: 0 confirmed present. Two cases remain ambiguous: case 4
  (small reddish/orange round object floating in water, unusual color
  for a golf ball but shape/behavior-consistent with the "floating in a
  water hazard" prompt) and case 11 (a dark round shape near golf clubs
  in an illustrated-style scene, not confidently a golf ball). Neither
  is forced to a verdict. Case 10 — `S2_STAGE1_HISTORY`'s one confirmed,
  genuine golf-ball return, same prompt/seed — shows **no ball** here (a
  cluttered pro-shop interior instead): this specific leak did not
  survive refinement, though this is one case, not a general claim that
  refinement eliminates leaks.
- Raw classifier predictions do **not** flag cases 4 or 11 as golf ball
  at all (not even as a secondary candidate) — consistent with keeping
  these as genuinely ambiguous rather than either dismissing or
  overclaiming them from the classifier alone.
- Suppression even counting both ambiguous cases as present:
  (16−2)/16 = **87.5%**, still comfortably above the 75% gate.

### French horn / parachute (retained)

**Degeneration**: 0/40 in this endpoint (individually confirmed, not
grid-inferred) — two local, non-pervasive defects disclosed rather than
absorbed into a blanket "clean" label: case 52 (local facial
warping on one figure, instrument itself clear) and case 54 (a local
blocky glitch artifact on one child's face; see object-absence below).
Paired against `S2_STAGE1_HISTORY`'s own 0/40: **0/40 → 0/40, no change,
no new degeneration from refinement.**

**Object presence, individually verified (not grid-inferred)**:

| | French horn present | parachute present (confirmed/ambiguous/absent) |
|---|---:|---|
| S2_STAGE1_HISTORY (grid-level, 2 cases individually corrected) | 19/20 | 16 / 2 / 2 |
| **S2_HISTORY_BOTH_STAGES** | **19/20** | **16 / 2 / 2** |

French horn absent only at case 54 (persists from `S2_STAGE1_HISTORY`,
same classroom scene, same local glitch — an inherited, not newly
introduced, defect). Parachute absent at cases 28 (gear close-up, no
canopy) and 32 (same lake/mountain scene with no parachute, persisting
from `EnglishSpringer_parent` onward — an inherited absence, not new);
ambiguous at 29 and 31 (packed/folded fabric, not a flying canopy).

**Exact recognition**: French horn 14/20 (tied with `S2_STAGE1_HISTORY`,
up from `S2_Stage1`'s 11/20 and well above the refinement-only
`S2_HISTORY`'s 9/20). Parachute 13/20 (tied with `S2_STAGE1_HISTORY`, up
from the refinement-only `S2_HISTORY`'s 11/20, below `S2_Stage1`'s
17/20). These are recognition-confidence changes with zero associated
whole-frame degeneration in every case.

## 6. Interpretation — a genuine, qualified result, not a clean "fix persists"

The strong existing control (Stage-1 preservation, carried through a
validated Stage-2 refinement) **works jointly on three of four
dimensions**: golf-ball suppression is the best of any S2 checkpoint
tested, springer stays correctly erased with no historical-return
signal, and both retained classes hold their Stage-1-level recognition
with zero new degeneration. It does **not** work cleanly on the fourth:
**springer-prompt image quality partially regresses** under refinement
(2/20→6/20 degenerate), concentrated in specific scenes and one seed.
This narrows, rather than confirms, the corruption-only novelty reading
from C27/C30: refinement is not a free pass-through of Stage-1's gain on
this class.

**This experiment does not isolate why.** Without a refinement-blank
arm built from this same improved `S2_STAGE1_HISTORY` input (explicitly
not authorized here), it cannot distinguish whether the 4 newly damaged
springer cases trace to the refinement step generally, to the specific
recovered embedding, or to an interaction between the two — nor can it
claim this pattern is sequential-editing-specific, universal, or novel
on this one reused development panel. Passing the golf-deletion gate
does not by itself establish matched semantic-deletion effectiveness
with any other checkpoint; the differing suppression mechanisms (one
confirmed false positive, one genuine leak now gone, one checkpoint with
zero raw hits at all) are kept distinct above, not collapsed into a
single number.

## 7. Resources and protocol compliance

| phase | device | elapsed |
|---|---|---:|
| embedding recovery | GPU 0 (verified idle before, released after) | 33.4s |
| refinement (fp64 solve) | CPU only (`CUDA_VISIBLE_DEVICES=""`, asserted) | 18.3s |
| generation | GPU 0 (verified idle before, released after) | 309.8s |
| classification | GPU 0 (inside the guarded wrapper) | 1.9s |

**Total GPU: 345.1s ≈ 5.75 min. Total CPU (fp64 solve): 18.3s.** No
other GPU was touched; all four GPUs were checked idle before each
allocation via `nvidia-smi`, not inferred from utilization alone.
Review-before-classify order followed and independently enforced (not
merely checked after the fact) via `guarded_classify.py` and
`CLASSIFIER_GUARD_INVOCATION_LOG.json`.

## 8. Artifacts

- `FROZEN_MANIFEST.json` — written before any computation
- `recover_newtarget_s2_both_stages.py`, `RECOVER_S2_BOTH_STAGES_RESULT.json`, `E_S2_BOTH_STAGES_newtarget.pt`
- `build_both_stages_s2.py`, `S2_HISTORY_BOTH_STAGES.build_log.json`, `S2_HISTORY_BOTH_STAGES.pt`
- `eval/S2_HISTORY_BOTH_STAGES/{image_report.json,classifications.jsonl,classify_report.json}`
- `build_review_both_stages.py`, `CASE_LEVEL_COMPLETION_RECORD.json`, `VISUAL_REVIEW_INITIAL.json`, `CLASSIFIER_PRECONDITION_GATE.json`
- `guarded_classify.py`, `CLASSIFIER_GUARD_INVOCATION_LOG.json` — the genuinely-enforced, single-path precondition-then-invoke wrapper
- `closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md`, `closeout/CORRECTED_LABELS.json` — bundled corrections to the prior S2 Stage-1-only endpoint and verification-claim honesty fixes
