# Corrections to the published RECE sequential-editing reports

Both reports in this directory are published as delivered. This file
documents three discrepancies found while preparing publication, following
this repository's standing practice ([`audit_v1/CLAIM_CORRECTIONS.md`](../audit_v1/CLAIM_CORRECTIONS.md)
and successors): claims are corrected in place here, not silently rewritten
in the original report text.

## C1 — Sequential golf-ball degeneration: 32/40 is superseded by 35/40

**What was claimed.** `FINAL_REPORT.md` (the direct-vs-sequential report)
reported the A->B (sequential) checkpoint's golf-ball-prompt degeneration
rate as **32/40 (80%)**, built from a single visual pass over 180px
thumbnail grids.

**What changed.** `refinement_ablation/CLOSEOUT.md` re-examined, at full
image resolution, the 8 cases that pass originally called non-degenerate
(15, 22, 24, 25, 26, 27, 34, 39). **Three were mislabeled at thumbnail
scale: cases 22, 34, and 39.** At full resolution all three are dominated
by the same severe mesh/honeycomb corruption artifact visible throughout
the rest of the grid; at 180px this was not visible.

**Status:**

| claim | status |
|---|---|
| A->B golf-ball degeneration is 32/40 (80%) | **SUPERSEDED** |
| A->B golf-ball degeneration is 35/40 (87.5%), cases 22/34/39 added | **STANDS** — confirmed at full resolution; see `images/corrected_cases/AtoB_case{22,34,39}.png` |
| Cases 15, 24, 25, 26, 27 are genuinely clean | **STANDS** — independently re-confirmed at full resolution |

**Resolution.** The corrected figure is 35/40 (87.5%), not 32/40 (80%).
Paired against checkpoint A's own 7 known-degenerate golf-ball cases
(1, 30, 32, 33, 35, 37, 39 — from
`evidence/upstream/rece_golfball_confirmation_VISUAL_REVIEW.json`'s keyed
table), **all 7 persisted under A->B; none recovered.** The aggregate
difference (35-7=28) is **not** "28 newly damaged" read naively either —
it decomposes exactly into 28 newly damaged and 0 recovered, which is
poorer than the uncorrected report's own implied reading (32-7=25, which
would understate both the rate and conceal that zero cases recovered).
Full paired table: `refinement_ablation/CLOSEOUT.md` and
`evidence/refinement_ablation/VISUAL_REVIEW.json`.

**This correction makes the original finding (severe, sequential-specific
golf-ball-prompt corruption) stronger, not weaker.** It does not change
which report section it supports, and it does not affect the
stage-localization conclusion in `refinement_ablation/FINAL_REPORT.md`
(corruption absent at the Stage-1-only AB-UCE endpoint, 2/40 = 5%;
appearing during refinement) — that comparison was computed directly
against the corrected 35/40 figure, not the superseded 32/40 one.

## C2 — Sequential final generic-dog presence: 5/20 is superseded by 6/20

**What was claimed.** `FINAL_REPORT.md` reported that 5 of A->B's 20
English-springer-prompt images still visually show a dog (cases 80, 84,
86, 90, 91), despite the exact-label classifier scoring all 20 as
suppressed.

**What changed.** While building the stage-ablation diagnostic's dog-
presence table, a full re-scan of A->B's own `classifications.jsonl`
top-1 category field for all 20 English-springer cases found **case 94's
top-1 guess is "redbone"** (confidence 0.05) — redbone is itself a hound
dog breed. The original report's case-by-case scan of the same file
overlooked this one row. Direct visual inspection of
`images/corrected_cases/AtoB_case94.png` confirms a dog's face is clearly
visible, diving into water.

**Status:**

| claim | status |
|---|---|
| A->B generic dog presence is 5/20 (25%): cases 80, 84, 86, 90, 91 | **SUPERSEDED** |
| A->B generic dog presence is 6/20 (30%): cases 80, 84, 86, 90, 91, 94 | **STANDS** |

**Resolution.** The corrected count is 6/20 (30%), not 5/20 (25%). This
does not change the qualitative finding (generic dog content persists in
roughly a third, not a quarter, of the sequential branch's "fully
suppressed" new-target images; the frozen exact-label gate is unaffected
either way — both the superseded and corrected reading score 0/20 against
the exact "English springer" label). The stage-ablation diagnostic's own
paired-transition table (`refinement_ablation/VISUAL_REVIEW.json`) was
built directly against the corrected 6-case set, not re-derived from the
superseded 5-case one.

## C3 — "French horn/parachute stayed clean throughout" needs a checkpoint qualifier

**What was claimed.** `refinement_ablation/FINAL_REPORT.md` §7 states:
*"Parachute and French horn both stayed clean (0/20 degenerate) at every
one of the six endpoints, including both new intermediates."*

**The issue.** Checkpoint **A** is one of the six endpoints in that
table, and A does **not** have a 0/20 French-horn degeneration rate: the
already-closed, unchanged completed screen
(`evidence/upstream/rece_golfball_confirmation_VISUAL_REVIEW.json` ->
`french_horn_summary`) recorded **1 degenerate French-horn case under A**
(case 70, "structurally confused double-bell shape with an attached
purple ornament"). The sentence as written is imprecise about which
endpoints it covers.

**Status:**

| claim | status |
|---|---|
| French horn and parachute are 0/20 degenerate at M0, B-UCE, B, AB-UCE, and A->B | **STANDS** — independently confirmed for both new intermediates (B-UCE, AB-UCE) during this diagnostic; M0/B/A->B values are unchanged from the completed experiment |
| French horn and parachute are 0/20 degenerate at **A** | **WRONG as stated** — A carries 1/20 (case 70), from the already-closed completed screen, not re-opened or re-measured here |

**Resolution.** The precise statement is: **French horn and parachute
degeneration is 0/20 at every endpoint newly or previously evaluated in
the direct-vs-sequential and stage-ablation experiments (M0, B-UCE, B,
AB-UCE, A->B) — five of the six endpoints in the table.** Checkpoint A
alone carries forward its own pre-existing 1/20 French-horn degeneration
rate (2.5% pooled across French horn + parachute, already reported and
gated as MET against a <=10pp pooled / <=20pp per-class limit in the
completed confirmation panel) from a different, already-closed experiment;
it is cited here for context, not re-measured, and this correction does
not reopen it. `refinement_ablation/FINAL_REPORT.md` itself is left as
written, per this repository's practice of correcting in place rather than
editing published report text after the fact.

## C4 — Verification-check scope was narrower than its logged fields implied

**What was claimed.** `seq_edit/run_edit.py` and
`refinement_ablation/make_uce_only_checkpoint.py` log
`"working_reference_separate_storage_verified": true`,
`"changed_keys_are_only_cross_attn_kv": true`, and
`"n_keys_changed_vs_start": 32` for every edit in this results line,
presented without qualification.

**What the code actually did.** Storage separation was checked via a
single probe tensor's `data_ptr()`
(`up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight`), not
all UNet tensors. Reference invariance was checked via the first 8
elements of that same probe tensor
(`unet.state_dict()[key][0, :8]`), not full tensor values. The
changed-key count was **logged**, and
`changed_keys_are_only_cross_attn_kv` **computed and logged as a boolean**,
but neither was ever asserted — a run that violated either would have
logged `false` and continued rather than raising.

**Status:**

| claim | status |
|---|---|
| Storage separation / reference invariance / changed-key-set were fully verified in every prior edit in this line | **IMPRECISE AS STATED** — they were checked on a narrow probe, not the full tensor set, and not asserted in the changed-key case |
| Any prior edit in this line actually violated separation, reference invariance, or the 32-key whitelist | **NOT FOUND** — re-verified with full-686-tensor, asserted checks in `reference_A_intervention/run_reference_A_intervention.py` (see that directory's `FINAL_REPORT.md` §2.1–2.4): the Phase-1 replay reproduced the existing `AtoB_english_springer.pt` checkpoint exactly (0/686 differing tensors), which would not hold if an earlier narrow check had missed real drift |

**Resolution.** No prior finding is invalidated. This is a description
correction: the prior checks were real but narrower than their logged
field names suggested, and have now been independently confirmed
sufficient (via the Phase-1 exact-replay result) and superseded going
forward by full-tensor, asserted checks (`reference_A_intervention/`
onward).

## C5 — Preliminary UCE's embedding slice is a token suffix, not "single-token slicing"

**What was claimed.** `seq_edit/vendored_functions.py`'s own docstring and
`refinement_ablation/FINAL_REPORT.md` describe Stage 1's `edit_model()` as
using a "single-token slicing convention," contrasted with Stage 2's
"full-77-token-sum convention."

**What the code actually does.** `vendored_functions.py` lines 216–223:
`final_token_idx = text_input.attention_mask[0].sum().item() - 2`, then
`old_emb = old_emb[final_token_idx : len(old_emb) - max(0, farthest - final_token_idx)]`
— a **suffix slice** of the token embedding sequence, bounded by a
`farthest` adjustment between the old/new text lengths. For a single-word
concept this slice is often one token wide in practice, but the mechanism
itself slices a variable-length suffix, not a single fixed token index.

**Status:**

| claim | status |
|---|---|
| Stage 1 uses "single-token slicing" | **IMPRECISE AS STATED** — it is a token-suffix slice that happens to be narrow for the single-word concepts used in this results line |
| The completed Stage-1 procedure (hyperparameters, behavior, outputs) | **UNCHANGED, not reopened** — this is a description-only correction |

**Resolution.** Future descriptions in this line should say "token-suffix
slice (single-token-wide for the concepts tested here)" rather than
"single-token slicing." No code, hyperparameter, or result changes.

## C6 — `reference_A_intervention/evidence/VISUAL_REVIEW.json`'s published `persisted_degenerate` list for AtoB→AB_REF_A wrongly included the four recovered cases

**What was claimed.** The `AtoB_M0_reference_to_AB_REF_A` paired-transition
table's `persisted_degenerate` array listed 34 case ids, including
28, 29, 30, 31, and 32 — but 28, 29, 30, and 31 are *also* listed in that
same table's `recovered` array (`[28, 29, 30, 31, 33]`). A case cannot be
both persisted-degenerate and recovered; this is an internal
contradiction in the published JSON, not a visual-judgment question.

**What the correct count is, under the labels as originally published**
(golf-ball degenerate set for AB_REF_A = all 40 cases except
{24,25,26,27,28,29,30,31,33}; for A→B = all 40 except {15,24,25,26,27}):
`persisted_degenerate` should list exactly 30 cases —
`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,32,34,35,36,37,38,39]`
(the 34-case published list minus the erroneously-included 28, 29, 30,
31) — together with `recovered: [28,29,30,31,33]` (5, unchanged, correct
as published), `newly_damaged: [15]` (1, unchanged, correct as published),
and `neither: [24,25,26,27]` (4, unchanged, correct as published).
**30 persisted / 5 recovered / 1 newly damaged / 4 neither**, summing to
40. The headline aggregate counts in `FINAL_REPORT.md` §3.4 (35/40 →
31/40, net +4) were arithmetically correct despite this listing bug — the
bug was confined to one array's contents, not the totals computed from
it.

**Resolution.** Corrected list published in
`evidence/reference_A_intervention/CORRECTED_LABELS.json`
(`c6_persisted_degenerate_fix`). The original (buggy) list in the
published `VISUAL_REVIEW.json` is left as published, per this repository's
practice of correcting in place rather than silently editing published
files.

## C7 — Re-adjudication of golf-ball cases 28–31 and 33 at full resolution (independent re-review; unresolved disagreement noted)

**What was claimed.** `reference_A_intervention/FINAL_REPORT.md` and its
`VISUAL_REVIEW.json` labeled cases 28, 29, 30, 31, and 33 as
non-degenerate ("a coherent hand/person... none shows a recognizable golf
ball").

**Independent re-review finding.** Re-inspected all five at full
resolution under the same degeneration rubric, explicitly applying the
clarification that **recognizable foreground content alone does not
establish a usable image** when the surrounding frame is dominated by the
mesh/honeycomb corruption artifact:

| case | re-review finding | verdict |
|---|---|---|
| 28 | Coherent glove/arm; background is a repetitive tree/rigging pattern against sky — regular but structurally plausible as foliage/rigging, not the chaotic small-square mesh signature. | **non-degenerate (retained)**, low uncertainty |
| 29 | Coherent hand in the lower-right ~40% of frame, but the upper-left/upper-right ~60% is dominated by the same honeycomb/mesh corruption seen throughout the degenerate cases, plus an indistinct ghost second hand and unnatural dark banding across the main hand itself. | **DEGENERATE — label changed.** Majority of frame is corrupted; the clean hand does not make the image usable as a whole. |
| 30 | Coherent hand/arm; blurred green background carries a faint, regular diamond-grid texture, fainter and more localized than case 29's. | **non-degenerate (retained)**, **explicit uncertainty**: this faint background texture could be the same artifact family at low amplitude, or ordinary bokeh; not resolved here. |
| 31 | Coherent hand; mostly smooth green-blur background with one anomalous horizontal line. | **non-degenerate (retained)**, low uncertainty (single minor artifact, not pervasive) |
| 33 | A small coherent face/arm fragment occupies the bottom-left corner (~15–20% of frame); the remaining ~80% is the same mesh/mosaic corruption as the clearly-degenerate golf-ball cases. | **DEGENERATE — label changed.** This matches the original report's own stated rubric almost exactly; labeling it non-degenerate in the published report was a reviewer error, not a close call. |

This re-review was conducted independently, without being instructed
which specific verdict to reach, and disagrees with the originally
published labels for 2 of the 5 cases (29, 33) while retaining the other
3. It also disagrees, in the other direction, with an initial outside
characterization that case 30 "has conspicuous background artifacts"
comparable to 29/33 — on direct full-resolution inspection, case 30's
artifact is real but visibly fainter and more localized, hence recorded
as retained-with-uncertainty rather than flipped. **This disagreement is
left explicit, not adjudicated to false certainty.**

**Recomputed golf-ball quality figures, under the re-reviewed labels**
(non-degenerate set becomes {24,25,26,27,28,30,31}, 7 cases; degenerate
becomes the other 33):

| quantity | as originally published | under re-reviewed labels |
|---|---|---|
| AB_REF_A degeneration | 31/40 (77.5%) | **33/40 (82.5%)** |
| AtoB→AB_REF_A: persisted / recovered / newly damaged / neither | 30 / 5 / 1 / 4 (after C6's listing fix) | **30 / 3 (28,30,31) / 1 (15) / 3 (24,25,26,27)** — 29 and 33 move from "recovered" to "persisted" (both were already degenerate in A→B, so they do not become "newly damaged"; they simply stop being counted as recovered) |
| AB_UCE→AB_REF_A: persisted / recovered / newly damaged / neither | 2 / 0 / 29 / 9 | **2 / 0 / 31 / 7** (29, 33 move from "neither" to "newly damaged," since both are non-degenerate at AB-UCE) |
| Net quality change vs. A→B (recovered − newly damaged) | +4 | **+2** |

**This does not change the qualitative conclusion in
`reference_A_intervention/FINAL_REPORT.md`** — corruption remains the
majority outcome under either labeling (77.5% or 82.5%), and the
reference-A swap is still a small, partial mitigation at best, not a
rescue. It does roughly halve the size of the already-modest quality
improvement (net +4 → net +2), and confirms the originally published
"not an unqualified rescue" framing was, if anything, too generous.

**Resolution.** Both label sets, and the full per-case reasoning above,
are published in
`evidence/reference_A_intervention/CORRECTED_LABELS.json`. The originally
published `VISUAL_REVIEW.json` and `FINAL_REPORT.md` are left as
published; this entry supersedes their golf-ball-quality headline figures
for any downstream comparison.

## C8 — Prompt-family mischaracterization: case 33 is not in the "golfer/hand holding ball" family

**What was claimed.** `reference_A_intervention/FINAL_REPORT.md` §3.4
states "All 5 recovered cases are in the 'golfer/hand holding ball'
family and show a coherent hand/arm/person consistent with the
'golfer/hand holding ball' prompt family."

**What is actually true**, per `frozen_manifest.json`'s own per-case
`family`/`prompt` fields: cases 28, 29, 30, and 31 share the **paraphrase**
prompt "A person's hand gripping a golf ball before teeing off." Case 33
is a **literal** prompt from a *different* family entirely — "A golf ball
flying through the air after being struck" — which has nothing to do
with a hand or a golfer; it depicts a ball in flight. The claim that all
5 share one prompt family is incorrect for case 33.

**Resolution.** Corrected statement: cases 28–31 share the hand-gripping
paraphrase family; case 33 is an unrelated literal "flying golf ball"
prompt whose recovery (under the original labels) or non-recovery (under
C7's re-reviewed labels) is not explained by prompt-family clustering.
No other claim in that section depended on this grouping.

## C9 — Adversarial-embedding norm wording conflated direction change with magnitude change

**What was claimed.** `reference_A_intervention/FINAL_REPORT.md` §2.5
reports the M0-reference and A-reference adversarial-embedding norms
(1285.169 and 1285.682) and their difference norm (284.265, "~22%
relative... against a ~1285-norm embedding") without separating how much
of that relative difference is a magnitude change versus a direction
change.

**Corrected figures**, computed directly from the already-saved norms in
`VERIFICATION_LOG.json` (pure arithmetic, no new computation): relative
difference-vector norm = 284.2654 / 1285.1686 = **22.12%**. Magnitude
(scalar norm) change = |1285.6818 − 1285.1686| / 1285.1686 = **0.03993%**
— three orders of magnitude smaller. Using the law of cosines on the two
saved norms and the saved difference norm, the aggregate angle between
the M0-reference and A-reference adversarial embeddings is **≈12.70°**.

**Resolution.** The 22% figure is overwhelmingly a **directional**
difference, not a magnitude difference — the two adversarial embeddings
are nearly the same length and point in measurably different directions
(~12.7°). The original wording did not make this distinction and could
be misread as implying a scale difference. No underlying number was
wrong; the interpretation is sharpened. See
`evidence/reference_A_intervention/CORRECTED_LABELS.json` for the
worked computation.

## C10 — The published script's 32-key whitelist check was logged, not enforced, despite being described as an assertion

**What was claimed.** `reference_A_intervention/FINAL_REPORT.md` §2.3–2.4
lists "Changed keys vs. AB-UCE input = expected 32-key cross-attn K/V
whitelist | **PASS, exact match**" as one of several checks performed,
in a table whose heading describes them collectively in language implying
enforcement.

**What `run_reference_A_intervention.py` actually did.** The storage-
separation check (`assert_full_storage_separation`) and the reference-
unchanged check (`assert_reference_unchanged`) both genuinely `raise
RuntimeError(...)` on failure — real, enforced assertions. The 32-key
whitelist check did not: it computed
`changed_set == EXPECTED_32_KEYS` and stored the boolean in the result
dict (`result["phase2_ab_ref_a"]["changed_keys_equal_expected_32"]`)
without an `assert` or `raise` — a run where the sets differed would have
recorded `false` and continued, identical in kind to the C4 pattern this
same report claimed to have corrected.

**Status:** no actual violation occurred — the logged sets did match
(32/32, confirmed in `VERIFICATION_LOG.json`) — so no finding is
affected. This is a description correction only: the check was real and
its result was correct, but it was not an enforced assertion as
described.

**Resolution.** Fixed going forward: `embedding_cross_swap`'s new
intervention script (§3 of this task) uses a literal `assert` (raising
`AssertionError`) for the 32-key whitelist check, not a logged boolean.

## C11 — Preliminary-UCE's token-suffix slice is usually NEAR-FULL-LENGTH for short concepts, not narrow (supersedes C5's "often one token wide" claim)

**What was claimed.** C5 (above) corrected "single-token slicing" to
"token-suffix slice," but added: "For a single-word concept this slice is
often one token wide in practice" — itself incorrect.

**Correct general formula**, re-derived from `vendored_functions.py`
lines 216–223: for padded sequence length `L`, old-text final index `i`
and new-text final index `j` (`farthest = max(i,j)`), both the old-text
and new-text embedding slices have length **`L − max(i, j)`** — not a
function of which text is older/newer, and not generally close to 1.

**Concrete measurement for this experiment's own concept** (CPU-only
tokenization check, SD1.4 CLIP tokenizer, no GPU): for
`old_text="English Springer"`, `new_text=" "`, `L=77`: attention-mask
sums are 4 and 2, giving `i=2`, `j=0`, `farthest=2`, and **slice length
= 77 − 2 = 75 tokens** — the suffix spans nearly the *entire* padded
sequence (75 of 77 positions), not a single token.

**Resolution.** C5 is superseded on this specific point: the slice is a
token suffix whose length is `L − max(i,j)`, which for this experiment's
short concept text is *near-full-length*, the opposite of "narrow." C5's
correction of "single-token slicing" → "token-suffix slice" stands; its
"often one token wide in practice" aside is itself wrong and is retracted
here. No code or hyperparameter changes; description-only, for the
second time on this same mechanism.

## C12 — "Within ordinary noise" language for retained-class recognition swings was not a statistical claim and should not have read as one

**What was claimed.** Both `refinement_ablation/FINAL_REPORT.md` and
`reference_A_intervention/FINAL_REPORT.md` describe single- or double-case
recognition-count swings on French horn/parachute (n=20) as "within
ordinary single-sample noise," reported alongside a claim of "zero
quality cost."

**Clarification.** "Zero severe degeneration" (the visual-review finding)
and "recognition-count swing is noise" (an implicit statistical claim)
are different kinds of statements. The former was directly observed
(every image individually inspected). The latter was asserted by
analogy ("a single flipped case = 5pp at n=20") without a variance
estimate, confidence interval, or repeated-seed measurement to actually
establish a noise floor. Recognition losses at this panel size are
**panel counts**, not independently validated noise estimates.

**Resolution.** Going forward in this results line (starting with
`embedding_cross_swap/FINAL_REPORT.md`), retained-class recognition
swings are reported as raw panel counts without the "ordinary noise"
characterization unless an actual noise estimate (e.g. repeated seeds) is
computed. The "zero severe degeneration" finding itself is unaffected and
not in question.

## What must not be concluded from any of this

- **These are not new findings** — all three corrections were found by
  re-examining existing images and labels already in hand; no new GPU
  computation was run to produce them, and none is authorized by this
  publication task.
- **Corrections 1 and 2 both increase the measured severity** of the
  sequential branch's quality problems; neither is a retraction of the
  core conclusion in either report.
- **Correction 3 is a precision fix to a summary sentence**, not a finding
  that French horn/parachute quality degraded in this experiment — A's
  1/20 rate was already known, already gated as passing, and predates both
  reports published here.
- **C6–C12 were likewise found by re-examining existing artifacts already
  in hand** (JSON labels, saved diagnostic norms, the frozen manifest's
  own prompt text, this line's own published code, and one CPU-only
  tokenizer check) — no image generation or GPU computation was used to
  produce any of them, consistent with the task that authorized this
  correction pass.
- **C7 moves the net golf-ball quality improvement from the reference-A
  swap from +4 to +2 cases** (under the independently re-reviewed labels)
  — a reduction in an already-modest, already-qualified finding, not a
  reversal of it; `reference_A_intervention/FINAL_REPORT.md`'s "not an
  unqualified rescue" conclusion holds, more strongly, under either
  labeling.
- **C7 explicitly preserves disagreement** rather than forcing a single
  verdict: this reviewer's case 30 re-read (retained, with uncertainty)
  disagrees with one external characterization of that case, and the
  reasoning for both the originally published and the re-reviewed labels
  is published side by side in `CORRECTED_LABELS.json` so a future
  reviewer can adjudicate independently rather than inherit a silently
  resolved call.
