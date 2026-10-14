# Stage-ablation diagnostic: does the sequential corruption predate or follow RECE refinement?

One bounded diagnostic, completed without pausing between authorized
stages. **Answer: the severe golf-ball-prompt corruption found in the
completed experiment's sequential branch (A->B) arises during Stage 2
(adversarial refinement), not at Stage 1 (preliminary UCE).** Stage 1 alone
(AB-UCE) is close to checkpoint A's own baseline corruption rate; refinement
is where the collapse happens, and only in the sequential branch.

## 1. Provenance

Diagnostic for `/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/FINAL_REPORT.md`
(the completed direct-vs-sequential English-springer experiment). Same
pinned RECE source (commit `403a67fd86dce6c124b88798340e001e625b28c9`),
same vendored `edit_model()`/`close_form_emb_regzero()`
(`seq_edit/vendored_functions.py`, drift-checked at every run), same
checkpoints M0/A (`GolfBall_ep0.pt`, sha256
`62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b`), same
100-case frozen manifest (sha256 `ca1401c0064441969fdf3ab329fb9ab8eaab980b63ade62b6f14bd9bd1ef43ad`),
same inference/classifier settings. No concept change, no new method, no
baseline search.

## 2. Completed-run reconciliation (existing-evidence closeout)

Full detail: `CLOSEOUT.md`. Two corrections were found on verification
against keyed labels and full-resolution images -- **both make the original
finding stronger, not weaker**:

- **Golf-ball degeneration in A->B: corrected from 32/40 (80%) to 35/40
  (87.5%).** Three of the prior report's 8 claimed-clean cases (22, 34, 39)
  were mislabeled at 180px thumbnail scale; at full resolution they show
  the same severe mesh/honeycomb artifact as the rest. **Paired transitions
  against A's own 7 known-degenerate cases: all 7 persisted (none
  recovered), confirming the aggregate difference is not "25 newly damaged"
  but 28 newly damaged and 0 recovered.**
- **English-springer dog-presence in A->B: corrected from 5/20 (25%) to
  6/20 (30%).** Case 94 ("redbone," a hound breed, p=0.05) was a
  dog-breed classifier guess the prior pass's scan overlooked; visually
  confirmed as a dog diving into water.
- No checkpoint or evaluation identity issue was found; both corrections
  came from re-examining existing images and labels, not from new
  generation.

## 3. Saved-versus-reconstructed endpoint identities

**No Stage-1-only checkpoint was saved by the completed experiment** --
`edits/*.edit_log.json` record only final (post-Stage-2) checkpoints.
Verified, not assumed, before reconstructing. Both intermediates were
therefore reconstructed once, executing ONLY the exact preliminary-UCE
procedure already used and verified in the completed experiment (same
vendored `edit_model()`, same hyperparameters: lamb=0.5, erase_scale=1,
preserve_scale=1.0 auto, guide=`' '`, retain=`['']`, technique=replace),
starting from the same verified M0 / `GolfBall_ep0.pt`:

| endpoint | starting point | status | sha256 (out) | verification |
|---|---|---|---|---|
| B-UCE | M0 | **reconstructed** | `23b691460d...` | 32/32 cross-attn keys changed, nothing else |
| AB-UCE | A (`GolfBall_ep0.pt`, re-verified `62720e14...`) | **reconstructed** | `3a9d86de13...` | 32/32 cross-attn keys changed, nothing else |
| B | M0 | already-evaluated final (reused) | (completed experiment) | -- |
| A->B | A | already-evaluated final (reused) | (completed experiment) | -- |

No component of the original reference policy was changed: Stage 1 never
used an adversarial embedding or M0 reference copy to begin with (that
machinery belongs to Stage 2 only), so omitting Stage 2 required no
reference-policy decision at all -- it is simply unexercised. This is
recorded explicitly, not silently: the pristine-M0 reference copy,
`close_form_emb_regzero`, and `edit_model_adversarial` were never
instantiated for either reconstruction (`make_uce_only_checkpoint.py`,
`stage2_omitted_components_unused` in each edit log).

## 4. Exact settings (unchanged from the completed experiment)

Stage 1 only: `edit_model(old_text_=['English Springer'], new_text_=[' '],
retain_text_=[''], lamb=0.5, erase_scale=1, preserve_scale=1.0,
technique='replace')`, vendored byte-for-byte from
`erase_methods/train_uce.py`, single-token-slicing convention. No tuning,
no concept change, no reference sweep.

## 5. Four/six-endpoint recognition table

| class (role) | M0 | A | B-UCE | B (final) | AB-UCE | A->B (final) |
|---|---|---|---|---|---|---|
| golf ball (historical target) | 39/40 | 3/40 | 40/40 | 37/40 | **0/40** | 0/40 |
| parachute (retained) | 16/20 | 15/20 | 16/20 | 15/20 | 15/20 | 18/20 |
| French horn (retained) | 20/20 | 18/20 | 20/20 | 19/20 | 18/20 | 19/20 |
| English springer (new target) | 20/20 | 20/20 | 1/20 | 0/20 | 0/20 | 0/20 |

(M0, A, B, A->B columns reused unchanged; B-UCE, AB-UCE newly generated and
classified on the identical manifest, 200 new images total.)

## 6. Deletion strength at every endpoint (correct baseline per branch)

| endpoint | baseline used | suppression |
|---|---|---|
| B-UCE (Stage 1 only, direct) | M0 (20/20) | **95%** (20->1) |
| B (final, direct) | M0 (20/20) | **100%** (20->0) |
| AB-UCE (Stage 1 only, sequential) | A (20/20) | **100%** (20->0) |
| A->B (final, sequential) | A (20/20) | **100%** (20->0) |

Both intermediates were evaluated regardless of pass/fail, per the frozen
diagnostic design; both happen to already clear the 75% threshold at
Stage 1 alone. **This does not reopen the completed experiment's own
direct-first gate**, which remains passed as originally reported. Passing
the same exact-label threshold at multiple stages still does not establish
matched effectiveness -- see section 7 for why.

## 7. Stage comparisons: where does the corruption come from?

### Golf-ball-prompt degeneration

| change | value |
|---|---|
| Sequential refinement change (A->B minus AB-UCE) | **+82.5pp** (5.0% -> 87.5%) |
| Direct refinement change (B minus B-UCE) | **+0.0pp** (0.0% -> 0.0%) |
| Difference between those two changes | **82.5pp** |
| Each intermediate's change from its own input | AB-UCE vs. A: **-12.5pp** (7/40 -> 2/40, i.e. Stage 1 alone slightly *reduces* A's own pre-existing degeneracy -- 5 of A's 7 degenerate cases recover, 0 new ones appear); B-UCE vs. M0: **0.0pp** (0/40 -> 0/40, golf ball untouched) |

**Paired counts, not just rates** (full detail: `VISUAL_REVIEW.json`):
- A -> AB-UCE (Stage 1 only): 2 persisted degenerate (37, 39), **5 recovered** (1, 30, 32, 33, 35), 0 newly damaged.
- AB-UCE -> A->B (adding Stage 2): 2 persisted degenerate (37, 39), 0 recovered, **33 newly damaged**.
- B-UCE -> B (direct, adding Stage 2): 0 persisted, 0 recovered, 0 newly damaged (40/40 clean throughout).

**Interpretation, narrowly stated**: the corruption is not present before
refinement (AB-UCE's 5.0% is close to -- and in fact lower than -- A's own
17.5%, and both of its degenerate cases are the same pre-existing
flaming-streak artifact type seen in A, not a new corruption signature).
It appears specifically when Stage 2 is added, specifically in the
sequential branch. This is a candidate composition-dependent refinement
effect in this setting: a difference of 82.5pp between the sequential and
direct branches' own refinement-stage changes, on the same prompts, same
manifest. It does not, by itself, establish which equation
(`edit_model_adversarial`'s erase step vs. `close_form_emb_regzero`'s
embedding computation) or which reference choice is responsible, nor
whether this generalizes beyond this golf-ball/English-springer pair.

### English-springer (new-target) dog-content persistence

| branch | Stage 1 dog-present | final dog-present | refinement change |
|---|---|---|---|
| Direct (B-UCE -> B) | 8/20 (40%) | 0/20 (0%) | **-8** (complete cleanup) |
| Sequential (AB-UCE -> A->B) | 6/20 (30%) | 6/20 (30%) | **0 net**, but churns: 2 recovered (88, 92), 2 newly appeared (84, 94), 4 persisted (80, 86, 90, 91) |

This is dog presence, not confirmed English-springer survival (two of
AB-UCE's/A->B's dog cases, 92 and 94, were only found by directly viewing
the image after the classifier assigned them low-confidence non-dog labels
-- detector misses, not breed-confusion misses). The frozen exact-label
gate (classifier top-1 == English springer) is unaffected and still reads
0/20 for both final endpoints. **Refinement's "residual-category cleanup"
function works completely in the direct branch and not at all, net, in the
sequential branch** -- a second, independent composition-dependent
refinement effect alongside the golf-ball corruption, on the opposite
(new-target) side of the comparison.

### Retained classes, carried through

Parachute and French horn both stayed clean (0/20 degenerate) at every one
of the six endpoints, including both new intermediates -- the corruption
remains concentrated on golf-ball prompts specifically; it is not a
general quality collapse. Recognition-count swings at Stage 1 vs. final
(parachute: B-UCE 16->B 15, AB-UCE 15->A->B 18; French horn: B-UCE 20->B
19, AB-UCE 18->A->B 19) are all within ordinary single-sample noise at
n=20 (one flipped case = 5pp) and are reported, not filtered, per this
task's instruction; none is treated as a significant effect.

**Better quality with weaker deletion is a tradeoff, not demonstrated
matched effectiveness**: at Stage 1, both branches pass the suppression
gate (95% and 100%) while being visually far cleaner on the historical
target (AB-UCE: 5% degenerate) than after refinement is added (A->B:
87.5% degenerate). This is reported as a tradeoff observation, not as
evidence that Stage-1-only would be a preferable configuration -- Stage 1
alone still leaves substantial new-target dog content (30-40%), which
Stage 2 exists specifically to address, and succeeds at addressing only in
the direct branch.

## 8. Resource use and release

| stage | time |
|---|---|
| GPU idle check immediately before allocation | GPU 0 confirmed IDLE at `2026-10-14T17:18:52` |
| Build B-UCE (Stage 1 only) | 16.4s |
| Generate B-UCE (100 images) | 6m48s |
| Classify B-UCE | 13s |
| Build AB-UCE (Stage 1 only, incl. `GolfBall_ep0.pt` load+verify) | 31.3s |
| Generate AB-UCE (100 images) | 6m46s |
| Classify AB-UCE | 14s |
| **Total GPU-held time** | **~15m03s, one GPU (GPU 0), CUDA_VISIBLE_DEVICES=0** |

GPU released immediately after the final classification step. Confirmed
idle again via `scripts/gpu_select.sh` at `2026-10-14T17:34:20`: GPU 0 and
GPU 1 both IDLE; GPUs 2/3's other-user occupant (`jibekgupta`, different
PIDs than the completed experiment's run, same user) untouched throughout.

**Artifacts**: `CLOSEOUT.md`, `make_uce_only_checkpoint.py`,
`orchestrate_ablation.py`, `edits/{B_UCE,AB_UCE}.pt` +
`*.uce_only_log.json`, `ANALYSIS_RESULTS.json`, `VISUAL_REVIEW.json`,
`review_grids/{B_UCE,AB_UCE}/*.png`, `eval/{B_UCE,AB_UCE}/`,
`FINAL_REPORT.md` (this file).

## 9. One next scientific decision

**Localize further within Stage 2 itself**: determine whether the
golf-ball-prompt corruption originates in `close_form_emb_regzero`'s
adversarial-embedding computation (which reads the Stage-1-edited weights
as its live basis) or in `edit_model_adversarial`'s own erase step applied
on top of that embedding. A minimal next diagnostic (not run here, not
authorized by this task): save the Stage-2 adversarial embedding itself
for both the direct and sequential branches and compare their norms/
directions relative to M0's own golf-ball embedding, without generating
any images -- a CPU-only, no-GPU check that would show whether the
embedding computation alone already diverges abnormally for the sequential
branch, before implicating the erase step.

No new final-RECE runs, no extra epochs, no reverse ordering, no
reference-policy variants, no new concepts, no additional checkpoints, and
no automatic follow-up were run beyond this bounded diagnostic. Stopping
here as scoped.
