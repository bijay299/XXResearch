# Preservation-content control: can RECE's existing preservation term prevent golf-ball corruption, and does historical preservation beat an unrelated control?

One controlled existing-method experiment, not a novelty claim. Three
frozen comparison arms, all starting from the untouched `AB_UCE.pt`
working checkpoint and the untouched, previously-saved `E_S.pt` embedding:
the already-published blank-preservation comparator (`AB_STABLE_SOLVE`,
retain text `''`), and two new endpoints differing from it in exactly one
frozen setting -- the retain text -- replaced, not appended: `AB_RETAIN_
GOLF` (`'golf ball'`, historical/self preservation) and `AB_RETAIN_HORN`
(`'French horn'`, an unrelated-content control).

## 1. Starting point

Commit `95d5c881ea5aa11569bbe950317203bf80c32b4d`. Large recovery-solve
error is closed as the explanation for the direct/sequential embedding
contrast (`recovery_optimality_audit/`). The corrected blank-preservation
comparator `AB_STABLE_SOLVE` (`CORRECTIONS.md` C17): golf degeneration
**31/40**, recognition golf 0/40 / springer 0/20 / parachute 17/20 / horn
16/20, two generic dogs (cases 86, 90, exact breed unconfirmed). Not
repeated: eligibility, baseline searches, parent confirmation, Stage-1
reconstruction, both diagonal replays, both numerical audits.

## 2. Inputs, provenance, frozen scope

All four required hashes verified live: `AB_UCE.pt`
(`3a9d86de...62b550`), `E_S.pt`
(`e6ea282a...94af89bdb8`), `guide_embedding.pt`
(`af70a52c...011d4695da7`), `AB_STABLE_SOLVE.pt`
(`3b491531...278366e8829`) -- all MATCH. `guide[0]`'s **raw tensor-byte**
sha256 (distinct from the serialized-file hash above) also matches the
required `24a6d0ee...895fbdbb93f` exactly.

Equations confirmed against the pinned source (RECE
`403a67fd86dce6c124b88798340e001e625b28c9`,
`erase_methods/uce.py::edit_model_adversarial`): `J(V) = lambda||V-W||^2
+ e||VX^T-WG^T||^2 + p||VR^T-WR^T||^2`, `H = lambda*I + e*X^TX + p*R^TR`,
`N = lambda*W + e*W*G^TX + p*W*R^TR`, solved via `HV^T=N^T`. `W` is the
**untouched AB_UCE weight** (the already-once-erased, pre-refinement
state) for every arm -- preservation targets `WR^T` are computed from
this same state, never from pristine M0, per task instruction (preserving
M0's golf response would test a different intervention and risk
restoring the erased object).

**New retain-text encoding** (GPU, ~1.3s, idle-verified before
allocation): `'golf ball'` and `'French horn'`, native batch `[t,t]`
exactly as `edit_model_adversarial`'s own loop. Both texts: 3 non-pad
tokens, token-id rows identical, **embedding rows bit-identical to each
other** (max abs diff 0.0) for both texts -- gate passed, common value
`R` used once per arm. Guide/blank encoding, embedding recovery, Stage 1,
and `AB_STABLE_SOLVE` construction were not rerun.

## 3. Build and validate (CPU, fp64)

Both arms solved via fp64 Cholesky, no explicit inverse, minimal
generalization of the already-validated `solver_only_intervention`
script (only the retain operand `R` changes).

| | AB_RETAIN_GOLF | AB_RETAIN_HORN |
|---|---|---|
| `\|\|X\|\|_F` (E_S) | 1285.17 | 1285.17 |
| `\|\|G\|\|_F` (guide) | 254.14 | 254.14 |
| `\|\|R\|\|_F` | 254.24 | 252.89 |
| min/max eig(H) | 0.1 / 1.607e6 | 0.1 / 1.607e6 |
| cond(H) | 1.607e7 | 1.607e7 |
| reference backward error (all 32 layers) | 2.6e-17 - 3.0e-17 | 2.6e-17 - 3.0e-17 |
| tolerance (<=1e-10) | PASS, all 32 | PASS, all 32 |
| quadratic-gap identity | PASS, all 32 | PASS, all 32 |

`H`'s dominant term is `e*X^TX` (same `X=E_S` for both arms, `erase_
scale=1` vs `preserve_scale=0.1`), so `cond(H)` is essentially unchanged
by which retain text is used -- the two arms differ only in `N`'s
preserve term, not in solve conditioning. **`\|\|R\|\|_F` is almost
identical between arms (254.24 vs 252.89, both close to `\|\|G\|\|_F`'s
254.14 -- all three are short common-phrase CLIP embeddings of similar
scale)**: the dramatically different outcome (S5) is not explained by one
arm's retain operand being numerically larger or imposing a stronger
constraint by norm -- it is a content/direction effect, consistent with
the task's own caution that equal token/coefficient budget does not
imply equal effective constraint strength. **Reference solve fully valid
for both arms, no layer excluded.** The saved `AB_STABLE_SOLVE` checkpoint,
scored under each arm's own (golf- or horn-specific) objective as a
diagnostic "candidate," shows a far larger excess objective under the
golf-specific objective (median ~140x `J(V*)`) than under the horn-specific
one (median ~24x `J(V*)`) -- expected, since `AB_STABLE_SOLVE` was built
with blank preservation and the two new objectives pull in different
directions; this is a solve-diagnostic only, not an image-level claim.
**Reload verification**: both checkpoints show exactly 32 keys changed,
654 unchanged tensors byte-identical to `AB_UCE`, all 686 finite, solved
weights exact fp32 casts. Full per-layer tables: `RESULT_GOLF.json`,
`RESULT_HORN.json`.

## 4. Generation and review (GPU for generation only; CPU/no-GPU review)

GPU 0 verified idle before each of 2 generation allocations (100 images
each, 386.7s / 386.5s, mechanics identical to every prior endpoint:
fp32, 512x512, 50 LMS steps, guidance 7.5, frozen per-case seeds, frozen
manifest sha256 `ca1401c0...f43ad`). GPU released and confirmed idle
after each.

**Mandatory execution order honored**: both 100-image sets were generated
first; the GPU was released; **all 200 images (120 target + 80
retained-class, both endpoints in full) were then individually opened and
visually judged at native 512x512 resolution** -- not a subset, not a
thumbnail/grid substitute for the required rows. Judgments (degeneration,
target presence/absence/ambiguity, generic-dog presence, breed
uncertainty) were recorded per case, then `VISUAL_REVIEW_INITIAL.json`
and the 200-row `CASE_LEVEL_COMPLETION_RECORD.json` were written and
hashed (`build_visual_review.py`, `CLASSIFIER_PRECONDITION_GATE.json`
records both hashes and confirms all 120 target rows present) **before**
either endpoint was classified. Classification was run only after the
gate passed. This is the one concrete, checkable protocol improvement
over prior endpoints in this line (`CORRECTIONS.md` C20: classification
preceded review for every earlier endpoint) -- it is disclosed as
assistant (Claude) visual review, not independent human validation, and
anonymous filenames alone do not constitute blinding; this task's order
guarantee is about sequencing, not about who is reviewing.

**One classifier/visual divergence this ordering caught**: `AB_RETAIN_
GOLF` case 30 classified top-1 `'golf ball'` (the only exact-match hit in
either arm's golf-ball class) -- but case 30 was already recorded,
*before* classification ran, as two people on a golf green with no ball
visible anywhere in frame (`disputed_cases/AB_RETAIN_GOLF_case30_
classifier_false_positive.png`). Direct re-inspection after classification
confirms the frozen visual label: this is a classifier false positive,
not an object return. Had review followed classification (as in every
prior endpoint), this case would plausibly have been counted as a visual
hit under anchoring.

## 5. Joint outcomes

Full tables: `JOINT_ANALYSIS.json`.

### 5.1 Golf-ball degeneration (the primary question)

| | degenerate | rate |
|---|---|---|
| AB_STABLE_SOLVE (blank) | 31/40 | 77.5% |
| **AB_RETAIN_GOLF (historical)** | **1/40** | **2.5%** |
| AB_RETAIN_HORN (unrelated) | 31/40 | 77.5% |
| *context, not regenerated: AB_UCE (Stage-1 only)* | *2/40* | *5%* |
| *context, not regenerated: native AtoB* | *35/40* | *87.5%* |

**Paired transition, blank -> historical**: 1 persisted degenerate (case
39, the one case still corrupted under either arm), **30 recovered**, 0
newly damaged, 9 neither -- a **96.8% relative reduction**, landing
degeneration almost exactly at Stage-1's own 2/40 (5%) baseline.
**Paired transition, blank -> unrelated**: 30 persisted, 1 recovered
(case 16), 1 newly damaged (case 15), 8 neither -- **net zero**, a single
case flipping each direction, statistically indistinguishable from the
blank comparator.

### 5.2 Golf-ball recognition (object return check)

Visual unambiguous golf-ball presence: **0/40 in all three arms**.
`AB_RETAIN_GOLF`'s one classifier exact hit (case 30) is a confirmed
visual false positive (S4). Four `AB_RETAIN_GOLF` cases (8, 9, 11, 12)
show an ambiguous round shape near a flag/hole, not counted as a ball on
visual grounds, consistent across arms. **Historical preservation's large
degeneration improvement is not achieved by restoring the golf ball** --
both the material-return flag (>=7/40 vs A) and the suppression-loss flag
(>=10/40 vs M0's 39/40) are nowhere close to triggered in either new arm.

### 5.3 English springer (erasure target)

| | exact recognition | suppression vs A (20/20) | degeneration | generic-dog presence |
|---|---|---|---|---|
| AB_STABLE_SOLVE | 0/20 | 100% | 0/20 | 2/20 (86, 90) |
| AB_RETAIN_GOLF | 0/20 | 100% (unchanged) | 0/20 | **7/20** (80,84,86,90,92,93,94) |
| AB_RETAIN_HORN | 1/20 (case 86) | 95% | 0/20 | 5/20 (80,84,86,90,91) |

Both gates pass (>=75% suppression). Per task framing: `AB_RETAIN_HORN`'s
one exact hit is an increase above the blank arm's 0/20 and **does**
weaken its exact-label result even though the gate passes, disclosed
plainly rather than minimized because the gate holds. **`AB_RETAIN_GOLF`'s
real cost is elsewhere**: generic-dog leakage on the erasure target is
**more than triple** the blank arm's rate (7/20 vs 2/20) -- a genuine,
non-trivial tradeoff for the large golf-ball improvement, not a free
lunch. Breed identifications (classifier top-1, low confidence throughout:
Irish setter p=0.450, Yorkshire terrier p=0.279/0.342, bull mastiff,
redbone, American Staffordshire terrier, Brittany spaniel) do not
establish exact breed; they are cited only as cross-validation that these
are genuinely dog-shaped content, consistent with the direct visual
judgment.

### 5.4 Retained classes

Parachute / French horn: **0/40 degeneration in both new arms**, no
quality cost. Recognition: `AB_RETAIN_GOLF` 16/20 parachute, 17/20 horn;
`AB_RETAIN_HORN` 16/20 parachute, 17/20 horn -- both within 1 case of
`AB_STABLE_SOLVE`'s 17/20 / 16/20 (noise-level, not a directional cost
for either arm).

## 6. Interpretation

**Historical preservation helps far more than the unrelated control, with
a good joint outcome.** Protecting `AB_UCE`'s own (already-once-erased)
response to the literal erased-concept text during Stage-2 refinement
very largely prevents the golf-ball corruption this line has
characterized since the first direct-vs-sequential report -- 31/40 ->
1/40, without restoring the golf ball (0/40 visual recognition in every
arm) and without weakening English-springer exact deletion (0/20 in both
new arms). The unrelated control (`French horn`) produces **no
improvement distinguishable from noise** (31/40 -> 31/40, a single case
flipping each way) -- preservation *content* matters, not merely the
presence of a nonzero preserve term, supporting the existing RECE
preservation mechanism specifically when it targets the response actually
at risk, in this one development setting. This is an **existing-method
control result**, not a new method: nothing about `edit_model_
adversarial`'s objective, solver, or hyperparameters was changed: only
which text fills the existing `retain_texts` argument.

**This does not come free.** Generic-dog leakage on the erasure target
more than triples (2/20 -> 7/20) under historical preservation -- a real
cost the exact-match recognition metric cannot see. One case (39) resists
repair under either preservation choice. This is one development concept
pair (golf ball / English springer); generality to other concept pairs,
to the sequential-vs-direct asymmetry itself, or to a principled
"preserve what's actually at risk" procedure (as opposed to this task's
oracle choice of retain text) is not established by this single control.

## 7. Correcting a prior session's own accounting error (no regeneration)

`CORRECTIONS.md` C21 (written in the previous session, `recovery_
optimality_audit/`) claimed two *additional* full-resolution rows for the
120-row embedding-cross-swap record (`D_WITH_SEQ_EMB` cases 38 and 87),
giving "16/120 full-resolution, 104 thumbnail-only." This double-counted
case 38: `images/corrected_cases/D_WITH_SEQ_EMB_case38.png` already
existed and was already one of that record's original 14 full-resolution
rows (`solver_only_intervention/FINAL_REPORT.md` S5's own accounting).
Only case 87 was genuinely new. **Corrected: 15/120 full-resolution, 105
thumbnail-only** -- `CORRECTIONS.md` C23 (below) fixes this without
touching the underlying case labels (unaffected; this is purely a
row-counting correction). The old cross-swap review remains provisional;
this task does not claim its closeout is complete, and no further pass
over it was made a prerequisite for this experiment, which uses only the
already-reviewed `AB_STABLE_SOLVE` endpoint as its comparator.

## 8. Resources and release

GPU 0 idle-verified immediately before each of 3 allocations (retain-text
encoding ~1.3s; two generation runs, 386.7s + 386.5s; two classification
runs, 8.3s + 8.3s -- total GPU-held time ~783s across 5 allocations, one
GPU, `CUDA_VISIBLE_DEVICES=0`, confirmed idle again within seconds after
each release). GPU 2 was occupied by another user's job throughout (100%
util, ~3.9GB) and was never touched. CPU-only for both fp64 solves (~22s
each, 8 threads, `CUDA_VISIBLE_DEVICES=""`). **No GPU polling beyond the
idle checks already logged. No baseline, comparator, or prior endpoint
was regenerated. Exactly 2 new endpoints, exactly 200 new images (100
each), as scoped.**

## 9. Artifacts

`FROZEN_MANIFEST.json` (written before any computation),
`RETAIN_ENCODING_PROVENANCE.json`, `native_retain_{GOLF,HORN}_row{0,1}.pt`,
`encode_retain_texts.py`, `run_build_arm.py`, `RESULT_{GOLF,HORN}.json`
(full per-layer fp64 tables), `edits/AB_RETAIN_{GOLF,HORN}.pt`,
`eval/AB_RETAIN_{GOLF,HORN}/` (100 images each, `image_report.json`,
`classifications.jsonl`, `classify_report.json`),
`VISUAL_REVIEW_INITIAL.json` and `CASE_LEVEL_COMPLETION_RECORD.json`
(frozen before classification, hashes in `CLASSIFIER_PRECONDITION_
GATE.json`), `JOINT_ANALYSIS.json`, `review_grids/AB_RETAIN_{GOLF,HORN}/`,
`disputed_cases/` (case 30's classifier false positive, case 39's lone
persisting degeneration, case 16's lone horn-arm recovery, one
representative horn-arm degenerate case), this `FINAL_REPORT.md`.
`CORRECTIONS.md` gained C23 (the 15/120 fix) and C24 (indexing this
experiment).

---

**No new recovery, replay, comparator generation, prompt/weight sweep,
additional endpoint, layer search, or automatic follow-up was run. Two
endpoints, 200 images, as scoped. Finishing this bounded comparison here.**
