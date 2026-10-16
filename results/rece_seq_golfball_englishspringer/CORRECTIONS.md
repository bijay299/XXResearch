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

## C13 — C7's own recomputed persisted/recovered/newly-damaged/neither row does not sum to 40

**What was claimed.** C7 (above) states the re-reviewed
AtoB→AB_REF_A paired-transition figures as **"30 / 3 (28,30,31) / 1 (15) /
3 (24,25,26,27)"**. This sums to 30+3+1+3 = **37**, not 40, and the
"neither" cell lists four case numbers (24,25,26,27) while labeling the
count "3."

**What the correct figures are.** Under the originally published labels
(after C6's listing fix), AtoB→AB_REF_A was 30 persisted / 5 recovered
{28,29,30,31,33} / 1 newly damaged {15} / 4 neither {24,25,26,27} = 40.
C7's re-review moves cases 29 and 33 from "recovered" to "persisted" —
both were already degenerate in A→B itself, so losing their "recovered"
status makes them persisted-degenerate, not newly damaged (a case can
only be "newly damaged" if it was non-degenerate in the comparator, A→B,
which neither 29 nor 33 was). That gives:

| quantity | C7 as published (buggy) | corrected |
|---|---|---|
| persisted | 30 | **32** |
| recovered | 3 (28, 30, 31) | 3 (28, 30, 31) — unchanged |
| newly damaged | 1 (15) | 1 (15) — unchanged |
| neither | "3" (24,25,26,27) | **4** (24,25,26,27) — the published number was a miscount of its own 4-item list |
| **total** | **37** | **40** |

**AB_REF_A's own rate (33/40, 82.5%) and the net quality change (+2, from
recovered(3) − newly damaged(1)) were already correct in C7 and are
unaffected by this fix** — the error was confined to the persisted/neither
split of the paired-transition table, not the headline degeneration rate
or net-improvement figure.

**Resolution.** C7's table is left as published, per this repository's
practice of correcting in place via new entries rather than silently
editing prior text. The corrected row is: **32 persisted / 3 recovered
(28, 30, 31) / 1 newly damaged (15) / 4 neither (24, 25, 26, 27)**,
summing to 40.

## C14 — C7 attributed a case-30 severity judgment to an unverified "outside characterization" that should not have been stated as a specific claim

**What was claimed.** C7 states the case-30 re-review "disagrees, in the
other direction, with an initial outside characterization that case 30
'has conspicuous background artifacts' comparable to 29/33."

**The issue.** This attributes a specific comparative severity judgment
("comparable to 29/33") to an "outside characterization" that is not
independently sourced or quoted from any identified reviewer in this
results line's evidence. Stating it as a concrete prior claim invents a
reviewer disagreement that cannot be verified, rather than simply
reporting this reviewer's own uncertainty about case 30.

**Resolution.** The corrected statement, with the invented attribution
removed: case 30 shows a faint, regular diamond-grid background texture
that is visibly fainter and more localized than the pervasive
mesh/honeycomb corruption seen in cases 29 and 33. Whether this faint
texture is the same artifact family at low amplitude, or ordinary bokeh,
is **not resolved** by this reviewer — recorded as **non-degenerate,
with explicit uncertainty**, on its own merits, not as a rebuttal to any
specific prior claim. No other case's re-review finding in C7 is
affected.

## C15 — Cases 32 and 33 are both the "flying golf ball" literal prompt family, not "golfer/hand holding ball"

**What was claimed.** C8 (above) corrected case 33's prompt-family
mischaracterization but did not mention case 32, leaving it ambiguous
whether 32 belongs to the same family.

**What the frozen manifest actually says.** Per
`rece_golfball_confirmation/manifest/frozen_manifest.json`, golf-ball
cases 28–31 (`scene_index` 3, paraphrase) share the prompt *"A person's
hand gripping a golf ball before teeing off."* Cases **32 and 33**
(`scene_index` 4, literal) share a **different** prompt entirely: *"A
golf ball flying through the air after being struck."* Both 32 and 33
are flying-ball literal-family cases; neither is part of the
hand-gripping family.

**Resolution.** Extending C8: whenever cases 28–33 are discussed as a
block in this results line, 28–31 (hand-gripping paraphrase) and 32–33
(flying-ball literal) are two distinct prompt families and should not be
described together as all "golfer/hand holding ball." This affects
description only; no quality or recognition figure changes.

## C16 — `embedding_cross_swap/FINAL_REPORT.md`'s claim that "any cell using E_S is severely corrupted regardless of working checkpoint" is false

**What was claimed.** The report's opening summary and §6 state: *"On
the erasure target (English springer), corruption cleanly follows the
embedding — any cell using E_S is severely corrupted regardless of
working checkpoint, and any cell using E_D is clean regardless of
working checkpoint."*

**What the preserved evidence actually shows.** The existing sequential
final `A→B` checkpoint is cell **(W_S, E_S)** — it uses E_S — and its
own already-published, unchanged `evidence/VISUAL_REVIEW.json` records
`"AtoB_english_springer_new_target": {... "degenerate": 0 ...}`: **0/20
English-springer degeneration**, with the existing review grid showing
20 coherent images (6 of which still show visible generic-dog content
under the corrected count, C2 — a target-persistence concern, not a
quality/corruption one). This cell is **not** severely corrupted. The
claim that "any cell using E_S is severely corrupted" is directly
contradicted by this cell, which is reused unchanged in this very report
(§2 design table, row W_S/column E_S) and was never in question.

**The actual four-cell pattern for English-springer degeneration**
(independently re-confirmed by viewing both new cells' full
`review_grids/*/English_springer.png` contact sheets during this audit,
alongside the preserved AtoB and existing-B records):

| cell | embedding | working checkpoint | degeneration |
|---|---|---|---|
| B (existing) | E_D | W_D | 0/20 |
| D_WITH_SEQ_EMB (new) | E_S | W_D | **20/20** |
| S_WITH_DIRECT_EMB (new) | E_D | W_S | 0/20 |
| A→B (existing, preserved) | E_S | W_S | **0/20** |

Severe corruption occurs in exactly **one** of the four cells —
(W_D, E_S) — not "any cell using E_S." E_S paired with its own native
working checkpoint (W_S, i.e. A→B) is clean; E_S paired with the
*other* row's checkpoint (W_D) is where corruption appears. This is the
mirror image of a clean column effect: it is a one-cell, direction-specific
mismatch between the sequentially-recovered embedding and the
*direct*-branch Stage-1 weights, not a property of E_S in isolation. No
explanation for why this specific pairing (and not its reverse,
(W_S, E_D), which is also an off-diagonal mismatch and is clean) is
offered here — this is reported as an open, narrower finding, not
resolved.

**What does still hold.** The report's **golf-ball** "interaction"
framing (§5.3/§6) is unaffected by this correction — golf-ball
degeneration really is severe (35/40) only at the (W_S, E_S) cell and
sharply reduced (≤5/40) at every single-factor substitution, which is
consistent across both the originally published and this corrected
reading.

**Resolution.** `embedding_cross_swap/FINAL_REPORT.md` is left as
published, per this repository's practice. §1 and §6 of
`numerical_optimality_audit/FINAL_REPORT.md` (published alongside this
correction) supersede its English-springer "corruption cleanly follows
the embedding" framing with the corrected one-cell-pattern description
above.

## C17 — `solver_only_intervention/FINAL_REPORT.md`'s "5 recovered" golf-ball cases is wrong: case 38 still shows pervasive mesh corruption under the line's own criterion

**What was claimed.** §4.2 reports `AB_STABLE_SOLVE`'s paired transition
vs. native `AtoB` as "30 persisted degenerate / **5 recovered** (28, 29,
30, 31, 38) / 0 newly damaged / 5 neither (15, 24, 25, 26, 27)," giving a
headline golf-ball degeneration rate of 30/40 (75.0%, down from native
`AtoB`'s 35/40).

**What direct full-resolution inspection shows.** Viewing
`solver_only_intervention/eval/AB_STABLE_SOLVE/imgs/38_0.png` at native
512x512 resolution (this audit; not previously done at this resolution
for this specific case/endpoint pairing) shows the same pervasive
mesh/sparkle corruption signature this results line has used throughout
as its whole-frame degeneration criterion (`VISUAL_REVIEW.json`'s own
note: "the familiar mesh/sparkle corruption signature... dominates
cases 0-23 and 32-39... plus case 38"): heavy checkerboard/moiré texture
across the flag backdrop and chromatic-fringed sparkle texture across
the lower-left region, covering the large majority of the frame. This is
**not** the milder, non-mesh "faint striping" / "minor edge artifact"
family that distinguishes this line's non-degenerate borderline cases.
For consistency, cases 15, 24, and 28 (the other cases this report or
its precedents flagged as borderline) were re-inspected at the same
resolution during this audit: all three remain correctly non-degenerate
under the existing rubric — case 15's artifact is a right-edge chromatic
fringe covering a minority of the frame with the central golf-green
subject clean; cases 24 and 28 show the milder regular-striping
background texture family (not mesh/sparkle) with a coherent, clean
foreground subject in both. Only case 38's label changes.

**Corrected paired table** (native `AtoB` vs. `AB_STABLE_SOLVE`, golf-ball,
40 cases): **31 persisted degenerate** (adds case 38) / **4 recovered**
(28, 29, 30, 31) / 0 newly damaged / 5 neither (15, 24, 25, 26, 27) — sums
to 40, consistent with the unchanged native-`AtoB` degenerate-case count.
Corrected `AB_STABLE_SOLVE` golf-ball degeneration: **31/40 (77.5%)**, not
30/40 (75.0%). Net improvement vs. native `AtoB`'s 35/40 is **4 cases**,
not 5 (35->31, 11.4% relative reduction, not 14.3%).

**What does still hold.** Every qualitative claim in §4.2/§4.4/§6 survives
this correction unchanged in direction and is, if anything, mildly
strengthened: golf-ball corruption remains the large majority outcome
under a verified fp64-accurate solve (77.5%, not 75.0%); the improvement
over native `AtoB` is real but even more modest than reported (4 of 35
cases, not 5); none of the recovered cases shows an actual returned golf
ball (unaffected by this correction, not re-examined here); and the
paired-vs-`AB_UCE` finding ("28 of 30 degenerate cases are
refinement-introduced, not inherited from Stage 1") becomes "29 of 31,"
the same conclusion more strongly. The English-springer and retained-class
tables (§4.1, §4.3) are untouched by this correction.

**Resolution.** `solver_only_intervention/FINAL_REPORT.md` is left as
published, per this repository's practice. This correction, plus the
independent recovery-optimality audit below, is the basis for not
carrying "35->30, five recoveries" forward as a settled headline in any
later report in this line.

## C18 — `solver_only_intervention/FINAL_REPORT.md`'s fp32-storage objective change is mislabeled and imprecise: "~5e-8 relative" should read ~5.04e-8 absolute / ~5.18e-10 relative

**What was claimed.** §3: "casting V* to fp32 and back to fp64 changes
the total objective by ≈5e-8 relative (97.388368776 ->
97.388368827)."

**What is actually true.** The quoted numbers' own difference,
97.388368827 - 97.388368776 = 5.1e-8, is the **absolute** change, not
the relative one (as printed, the sentence mislabels an absolute figure
as "relative"). Recomputing precisely from the full-precision per-layer
values already in `solver_only_intervention/RESULT.json` (summing
`J_at_V_star_fp64.total` and `J_at_V_star_fp32_roundtrip.total` across
all 32 layers, no new computation): **absolute change = 5.0418e-08**,
**relative change = 5.1770e-10** (sum fp64 = 97.38836877622424, sum
fp32-roundtrip = 97.38836882664246). The qualitative point the report
draws from this number is unaffected: at ~5e-10 relative, fp32 storage
remains roughly six orders of magnitude smaller than native's own
backward error (0.93e-3-2.74e-3) and is not a meaningful source of the
native/reference gap.

**Resolution.** Reported precisely here; `solver_only_intervention/
FINAL_REPORT.md`'s §3 text is left as published, per this repository's
practice, with this entry as the correction of record.

## C19 — "Accurate solving" in the solver-only report describes assembly-and-solve accuracy, not matrix-inversion accuracy alone

**What was imprecise.** `solver_only_intervention/FINAL_REPORT.md` and
`numerical_optimality_audit/FINAL_REPORT.md` describe the tested change
as replacing "the native float32 `torch.inverse` path" with an "fp64
Cholesky solve," which can read as isolating matrix-inversion accuracy
specifically. The actual tested change is broader: fp64 upcasting of the
operands **before** forming the Gram matrices (`H`/`N` or, in this
audit, `A`/`B`/`D`), fp64 accumulation across all 32 layers, and a
Cholesky solve with no explicit inverse formed -- i.e. **assembly-and-
solve accuracy**, not inversion accuracy in isolation. Neither report's
own methodology actually isolates inversion from accumulation (both
cited reports state this explicitly: "this audit assesses the combined
recovery computation... does not claim to separate accumulation,
inversion and final multiplication errors"), so no finding in either
report is affected -- this corrects only how the intervention is
*described*, not what was measured.

**Resolution.** Both reports are left as published. Future reports in
this line (including `recovery_optimality_audit/FINAL_REPORT.md` below)
use "assembly-and-solve accuracy" rather than "inversion accuracy" when
describing this class of intervention.

## C20 — Visual-review blinding: classification and aggregate inspection preceded visual review for every endpoint in this line, and this cannot be repaired retroactively

**What was disclosed, partially.** Several reports in this line (e.g.
`solver_only_intervention/FINAL_REPORT.md` S4, `embedding_cross_swap/
FINAL_REPORT.md`) disclose, per-endpoint, that "per-class aggregate
recognition counts were queried immediately after classification and
before the visual review began," describing this as a limitation on
independence but not always stating plainly that this means the
**classifier ran, and its aggregate output was seen, before any human
visual judgment was formed** -- for every endpoint published in this
results line, not a subset.

**Stated plainly here.** The frozen protocol this line's earlier work
(`RUN_MANIFEST.md`, `ELIGIBILITY.md`) describes as the intended practice
-- visual review before classification, to avoid anchoring labels to the
classifier's own (separately fallible) judgment -- was **not followed**
for any endpoint in this results line, including every endpoint audited
or corrected in this publication. This is a departure from the frozen
instruction, not a partial compliance, a blinding, or a "soft" blinding;
no rereading of the existing record can make the already-completed
visual passes have happened before classification. All visual-review
labels in this line, including the corrections in this file, should be
read as **prediction-informed retrospective review**, not independent
human validation -- consistent with how individual reports already
describe their own method sections, stated here as a line-wide fact
rather than a per-report footnote.

**Resolution.** No report is retracted on this basis; the quantitative
labels are not disputed by this disclosure alone (C17 above is a
separate, evidence-based relabeling of one specific case). This entry
exists so the limitation is visible at the collection level, not only
inside each report's own methods section.

## C21 — The 120-row embedding-cross-swap case-level record: full-resolution-vs-thumbnail accounting corrected, partial completion only

**What was claimed.** `solver_only_intervention/CASE_LEVEL_COMPLETION_
RECORD_120.json` (published in the solver-only report) records, for each
of 120 rows (40 golf-ball + 20 English-springer, x `D_WITH_SEQ_EMB` and
`S_WITH_DIRECT_EMB`), the resolution at which it was viewed: 14 rows
`full_512x512_crop` (10 via previously-saved `images/corrected_cases/`
crops, plus 4 more read directly from `eval/*/imgs/` during that
session), 106 rows `grid_thumbnail_180px`.

**What this audit adds.** Bounded additional full-resolution spot checks
this session (reading the actual `eval/*/imgs/<case>_0.png` files
directly -- these exist at full 512x512 resolution for every case
regardless of what resolution any prior pass used, so "viewing at full
resolution" only required opening the right file, not generating
anything new): `D_WITH_SEQ_EMB` case 38 (golf-ball) and case 87
(English-springer) were viewed at full resolution and both show the same
pervasive mesh-corruption signature already counted in that cell's
published 5/40 golf-ball and 20/20 springer degeneration totals --
**reconfirming**, not changing, those existing aggregate counts.

**What remains thumbnail-only.** The remaining ~104 rows of the 120-row
record were **not** re-viewed at full resolution in this bounded
session. Per the task scope that authorized this correction pass ("no
repeated baselines... no automatic follow-up"), a full case-by-case
re-review of all 120 rows was not undertaken. These rows' existing
thumbnail-grid-based labels are **not disputed** by this entry (no
inspection here contradicts any of them), but they remain exactly what
they were: thumbnail-resolution judgments, explicitly marked as such in
the existing record, not upgraded to full-resolution by this correction.
Any future claim that treats the full 120-row record as fully
full-resolution-reviewed would be incorrect; this entry is the record of
that limit.

**Resolution.** `CASE_LEVEL_COMPLETION_RECORD_120.json` is left as
published (its own `resolution_viewed` field per row is accurate and
unchanged). This entry records the two additional full-resolution views
performed during this audit and states plainly that the rest remain
provisional in the sense that they rest on thumbnail resolution only,
per this repository's own standing distinction between thumbnail and
full-resolution review.

## C22 — M0's provenance and the embedding-recovery operands' remaining documentation limits

**M0 provenance, newly recorded.** `embedding_cross_swap/edits/
M0_unet_state_dict.pt` was never independently hashed or provenance-
checked in any previously published report in this line (`embedding_
cross_swap/EMBEDDING_PROVENANCE.json` hashes `E_D`, `E_S`,
`concept_embedding`, `guide_embedding`, and `retain_embedding`, but not
M0 itself). `recovery_optimality_audit/` (below) establishes, for the
first time in this line: (a) M0's current sha256
(`ee60d14bf80c6f829ab1b9da5f6c0eb63338f8a29d7ec7f6a3895bae1b1611df`),
disclosed as newly recorded, not previously published; (b) its save-code
provenance, read directly from `embedding_cross_swap/
run_embedding_cross_swap.py` (an untouched `state_dict()` of a fresh
`StableDiffusionPipeline.from_pretrained` load of the pinned SD1.4
snapshot, saved before any edit is applied to that pipeline object); and
(c) an independent CPU-only, no-pipeline cross-check of all 32 audited
K/V projections directly against the pinned snapshot's own safetensors
file -- all 32 bit-identical. M0's provenance for this audit's purposes
is fully established.

**Remaining operand-provenance documentation limits, using only existing
artifacts.** The embedding-recovery operands' concept embedding `C` was
independently verified identical across the `W_D` and `W_S` recovery
pipelines (`EMBEDDING_PROVENANCE.json`, pre-existing). The retain/guide
text-embedding batching artifact documented in `solver_only_intervention/
FINAL_REPORT.md` S2 (a ~4.29e-6 max-abs-diff, batch-size-dependent
floating-point difference, not a content difference) concerns the
*Stage-2 refinement's* retain operand, not this audit's `close_form_
emb_regzero` computation, which has no retain term (confirmed in
`recovery_optimality_audit/FROZEN_MANIFEST.json`'s equation derivation --
`close_form_emb_regzero` has only a zero-regularizer term and the
erase/B term, no `preserve`/retain accumulation at all, unlike `edit_
model_adversarial`'s Stage-2 objective). No new text-encoding check,
endpoint build, or replay was run to further verify any operand in this
audit, per task scope; where an operand's provenance rests on a prior
report's own disclosure (e.g. the text-encoder batching note above), that
prior disclosure is cited, not re-verified.

- **C17–C22 were produced during `recovery_optimality_audit/`**, a CPU-
  only task auditing a different computation (`close_form_emb_regzero`,
  the embedding-recovery step) from either prior audit
  (`numerical_optimality_audit/` audited `edit_model_adversarial`'s
  Stage-2 refinement solve). **C17 is the most consequential of this
  batch**: it withdraws the "35->30, five recoveries" headline from
  `solver_only_intervention/FINAL_REPORT.md`, replacing it with "35->31,
  four recoveries" on direct full-resolution re-inspection of case 38 —
  a correction that strengthens, not weakens, that report's own
  "not an unqualified rescue" framing. C18 is a units/precision fix with
  no effect on any conclusion. C19 is a description-only correction. C20
  states a line-wide blinding limitation explicitly rather than only
  per-report. C21 reports two additional genuine full-resolution views
  and is explicit that the bulk of the 120-row record remains
  thumbnail-based, not newly completed (but see C23: C21 itself
  double-counted one of those two rows). C22 records M0's provenance for
  the first time and notes no further operand-provenance work was done
  beyond what already existed.
- **C23–C24 were produced during `preservation_content_control/`**, the
  existing-method preservation-content control. C23 fixes C21's own
  double-count (15/120 full-resolution, not 16/104) — a row-counting
  correction only, no case label affected. C24 indexes the control
  experiment's headline finding (see its own `FINAL_REPORT.md`): historical
  preservation (retain text = the erased concept itself) very largely
  prevents golf-ball corruption (31/40 → 1/40) where an unrelated-content
  control does not (31/40 → 31/40, net zero), at the real cost of tripled
  generic-dog leakage on the erasure target — an existing-method result,
  not a new method.

## C23 — C21 double-counted case 38: the 120-row cross-swap record is 15/120 full-resolution, not 16/104

**What was claimed.** C21 (written in the `recovery_optimality_audit`
session) states that "`D_WITH_SEQ_EMB` case 38 (golf-ball) and case 87
(English-springer) were viewed at full resolution this session,"
implying both were *additional* rows on top of the previously-published
14, giving 16/120 full-resolution and 104 thumbnail-only.

**What is actually true.** `images/corrected_cases/D_WITH_SEQ_EMB_
case38.png` already existed before that session and was already one of
the 120-row record's original 14 full-resolution rows
(`solver_only_intervention/FINAL_REPORT.md` S5's own accounting, which
C21 itself quotes: "10 via previously-saved `images/corrected_cases/`
crops, plus 4 more... 14 rows `full_512x512_crop`"). Case 38 was not a
new row; only case 87 was. **Corrected: 15/120 full-resolution, 105
thumbnail-only.** No underlying case label is affected -- this is a
row-counting correction only, found while preparing
`preservation_content_control/` (which does not depend on or re-review
this record; it uses only the already-closed `AB_STABLE_SOLVE` endpoint
as its comparator).

**Resolution.** C21's text is left as published, per this repository's
practice; this entry is the correction of record. The 120-row record's
own per-row `resolution_viewed` field was and remains accurate -- only
the *prose summary* in C21 miscounted.

## C24 — Preservation-content control: historical preservation vs. an unrelated control (index entry)

`preservation_content_control/FINAL_REPORT.md` (published alongside this
entry) tests whether RECE's existing preservation term, pointed at the
already-erased golf-ball response (`AB_UCE`'s own, not pristine M0's),
prevents the golf-ball corruption characterized throughout this results
line, and whether that beats an unrelated-content control (`'French
horn'`). Headline: golf degeneration 31/40 (blank) -> **1/40**
(historical) vs. 31/40 (unrelated, net zero vs. blank) -- without
restoring the golf ball (0/40 visual recognition in every arm) and
without weakening English-springer exact deletion (0/20 in both new
arms), but with generic-dog leakage on the erasure target more than
tripling (2/20 -> 7/20) under historical preservation. This is an
existing-method control result, not a new method -- only the `retain_
texts` argument changed. Full tables, paired transitions, and the one
classifier/visual divergence this task's review-before-classify ordering
caught (case 30, a confirmed false positive) are in the report itself,
not duplicated here.

## C25 — `preservation_content_control` closeout: two case reinspections, math-description and resource-arithmetic corrections

`history_preservation_confirmation/closeout/CLOSEOUT_CORRECTIONS.md`
(published alongside this entry) closes `preservation_content_control`
using only existing artifacts (no new generation, recovery, or audit).
Reinspecting `AB_RETAIN_HORN` case 16 (golf ball) at full resolution finds
pervasive sparkle/speckle sky corruption inconsistent with its original
"recovered" label; corrected to degenerate, moving `AB_RETAIN_HORN` golf
degeneration **31/40 → 32/40** and its paired transition from blank from a
nominal net-zero to a genuine **net -1** (0 recovered, 1 newly damaged).
Reinspecting case 94 (English springer) finds a hairy, paw-shaped limb
entering water, scored as generic-dog-present under this line's own
rubric; `AB_RETAIN_HORN` generic-dog presence **5/20 → 6/20**. The
`AB_RETAIN_GOLF` headline (1/40, a 96.8% relative reduction from the
31/40 blank baseline) is unaffected. Separately corrected: the
mathematical claim that `cond(H)`'s near-identity between arms reflects
`H` depending on `X` "not on R" (it depends on both; `p·R^TR`'s
contribution is merely small relative to `e·X^TX` here, and similar
`cond(H)` does not establish equal effective preservation strength); two
unsupported "noise-level"/statistical-equivalence phrases are withdrawn;
chronology evidence (file-write timestamps only, disclosed as
non-independent) and GPU/CPU resource arithmetic (791.09s GPU, 43.80s CPU,
precise logged figures) are corrected from existing records. All 120
target-class originals from both arms are published for the first time
(`history_preservation_confirmation/original_images/`), where previously
only 4 individual crops were available.

## C26 — History-preservation confirmation on two new sequences (index entry)

`history_preservation_confirmation/FINAL_REPORT.md` (published alongside
this entry) tests the same existing-method preservation control on two
fresh sequential-editing sequences (S1: golf ball → French horn, from the
author's own `GolfBall_ep0.pt`; S2: English springer → golf ball, from
this line's own `edits/B_english_springer.pt`), using a freshly frozen
80-case evaluation panel distinct from every prior panel in this line and
a BLANK-vs-HISTORY refinement comparison built the same way as
`preservation_content_control`. See that report for eligibility gates,
joint outcomes, and the qualified per-sequence result — not duplicated
here.

## C27 — `history_preservation_confirmation`: verification and corrections (round 2)

`history_preservation_confirmation/closeout/VERIFICATION_AND_CORRECTIONS_2.md`
(published alongside this entry) verifies the Stage-1 premise behind the
`stage1_history_intervention` (C28) against frozen labels and corrects
several imprecise statements from the prior turn, none of which changes
any published case label:

- S2's Stage-1 golf-ball-sequence historical (English springer)
  degeneration, previously reported as "approximately 20/20," is
  **exactly 20/20** in `S2_Stage1`, `S2_BLANK`, and `S2_HISTORY` alike,
  recomputed directly from `CASE_LEVEL_COMPLETION_RECORD_NEW_ENDPOINTS.json`.
- The report's interpretation is narrowed: **"refinement-only historical
  preservation did not rescue these two extensions, with most damage
  introduced earlier"** replaces the prior, overstated reading that
  implied history preservation fails generally or that the original
  `preservation_content_control` benefit is pair-specific — neither is
  established (the sequence *and* the evaluation panel both changed
  between the two experiments, so they are not a matched comparison).
- GPU allocation is corrected from "GPU 2 throughout" to by-phase: GPU 0
  for the shared-input generation/classification, GPU 2 for everything
  from the Stage-1 builds onward (after GPU 0/1 were claimed by another
  user's job).
- The prior chat summary's claim that all 720 images were "individually
  reviewed" is corrected: 380 were individually inspected at native
  512×512, 340 purely-retained-class images were grid-reviewed (no
  escalation needed in any case) — `FINAL_REPORT.md` itself already
  stated this correctly and required no change.
- Chronology evidence (file-write timestamps, not proof of viewing) is
  added for the new-endpoint review step, extending the same disclosure
  already made for the shared-input step and for every earlier endpoint
  in this line (C20).
- Raw classifier hits vs. visual judgment (S2 golf case 8), generic-dog
  vs. categorical historical-object-absence, and retained recognition
  vs. retained image quality were already kept distinct in the published
  record; VERIFICATION_AND_CORRECTIONS_2.md confirms no change needed.
- `AB_RETAIN_HORN` 32/40 degeneration and 6/20 generic dogs (C25), and
  GOLF cases 8/9/11/12 ambiguity, are reaffirmed unchanged — no new
  evidence warranted revisiting them.

## C28 — Stage-1 history intervention: preserving earlier prevents the damage (index entry)

`stage1_history_intervention/FINAL_REPORT.md` (published alongside this
entry) tests whether applying RECE's existing preservation mechanism AT
Stage 1 -- before the golf-ball-prompt damage occurs -- rather than only
at Stage-2 refinement (after it, where C26/`history_preservation_confirmation`
found it did not help), can prevent that damage. Built from the
untouched `GolfBall_ep0.pt` parent with exactly one changed argument
(`retain_texts=['']` → `retain_texts=['golf ball']`) on the native
Stage-1 UCE edit erasing French horn. Result: golf-ball degeneration
**18/20 (blank Stage-1) → 0/20** -- a complete, exact paired recovery of
every degenerate case, zero newly damaged -- with French-horn deletion
unchanged (0/20, 100% suppression against the parent's own verified
16/20) and retained-class recognition slightly higher than the blank
comparator, zero degeneration either way. A narrow, qualified result:
one sequence, one fixed preservation choice, no refinement stage added
on top (not run, per task instruction). See the report itself for full
tables and the explicit scope limits.

## C29 — Corrections bundled ahead of the S2 Stage-1 history intervention

`stage1_history_intervention/closeout/CORRECTIONS_BEFORE_S2.md` and
`CORRECTED_LABELS.json` (published alongside this entry) bundle a
bounded correction pass over `stage1_history_intervention` (S1), found
entirely from existing artifacts, no new GPU computation:

- **S1 golf-ball cases 12, 13, 14, 15**: `S1_Stage1`'s labels for these
  four macro-texture-prompt cases were `degenerate: true` with a
  templated "sparkle/mesh/checkerboard" note that does not match what
  the images show (a coherent embossed medallion, radiating macro
  spikes, a stitched fabric patch, a scaled/mosaic texture — none show
  the corruption signature; `GolfBall_parent`'s own equivalent images
  were already correctly labeled non-degenerate on the same basis).
  Corrected to non-degenerate, consistent with this line's own
  `GolfBall_parent`/`EnglishSpringer_parent` precedent for erased-concept
  classes. `S1_Stage1` golf-ball degeneration moves **18/20 → 14/20**;
  the paired `S1_Stage1`→`S1_STAGE1_HISTORY` recovery moves **18
  recovered/2 neither → 14 recovered/6 neither**. `S1_STAGE1_HISTORY`
  itself (still exactly 0/20 degenerate) and its qualitative conclusion
  (complete recovery, zero newly damaged) are unchanged — only the
  magnitude. `S1_BLANK`/`S1_HISTORY` (separate Stage-2-refined
  checkpoints, different images) are explicitly **not** reopened here.
- **False "structurally impossible" claim removed**: the per-case note
  used for all 20 `S1_STAGE1_HISTORY` golf-ball rows implied golf ball
  cannot return because the untouched parent already erased it — false
  as a general claim (a later edit can in principle restore an
  earlier-erased concept). Corrected to an observation-only note; the
  underlying `object_present: false` judgments are unchanged.
- **`history_preservation_confirmation/FINAL_REPORT.md` M0 parachute
  reference corrected**: stated "vs. M0's 18/20 each" for both parachute
  and springer; M0's actual parachute exact-hit count (recomputed from
  `panel/eval/M0/classifications.jsonl`) is **11/20**, not 18/20 — the
  new-endpoint parachute figures reported (12–13/20) were in fact above,
  not below, the true M0 parachute reference.
- **S2_Stage1 case 8** (raw classifier hit `p=0.941`, visual judgment
  "no ball," confirmed false positive): reaffirmed unchanged, both
  records published side by side, no new finding.
- **Publication sequencing**, disclosed: `cfd674b` bundled the Stage-1
  intervention's build/generation/review together with the
  then-unresolved disk-space recovery rather than publishing strictly
  after every check was independently reread. Freeze-file hashes
  establish *what* was frozen, not *when* each image was viewed; this
  limit is disclosed, not resolved by invented telemetry.
- **Reload-verification scope clarified**: the save/reload checks used
  throughout this line (S1 and the new S2 build below) prove exact
  byte-identity between the saved file and the in-memory state computed
  immediately before saving — not equivalence to an independent
  "intended" target, since none exists for these native fp32 Stage-1
  edits. Nothing is retracted; this makes the existing scope explicit.

## C30 — Stage-1 history intervention on S2: quality improves substantially, recognition does not (index entry)

`stage1_history_intervention_s2/FINAL_REPORT.md` (published alongside
this entry) tests the same Stage-1 preservation intervention as C28, on
the reverse sequence (S2: English springer → golf ball), explicitly left
open by C28. Built from the untouched `B_english_springer.pt` parent
with exactly one changed argument (`retain_texts=['']` →
`retain_texts=['English springer']`) on the native Stage-1 UCE edit
erasing golf ball. **Result does not replicate C28's pattern**: English
springer (historical) degeneration drops from an established **20/20
(blank Stage-1) to 3/20** — a large quality improvement — but exact
classifier recognition and visual breed-specific presence stay at
**0/20 in every checkpoint from `EnglishSpringer_parent` onward**,
identical to blank Stage-1; even generic dog-like shapes (2–4/20 under
blank Stage-1) drop to 0/20. Golf-ball (new-target) deletion shows one
genuine, independently confirmed return (case 10, a clear dimpled ball
on a fairway) plus two visually ambiguous cases (small round objects in
a water-hazard scene, not flagged by the raw classifier) — a materially
different picture from the blank comparator's own single raw hit, which
was already a confirmed false positive (C27). Suppression remains well
above the ≥75% gate either way (81.25–93.75%). Retained classes: 0/40
degeneration in both arms. A genuine, narrower, dissociated result —
quality repaired, recognition not — not a second confirmation of C28.
See the report itself for full tables, scene/family breakdowns, and
explicit scope limits.

## C31 — Corrections bundled ahead of the S2 both-stages build

`history_both_stages_s2/closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md`
(published alongside this entry) bundles corrections found entirely
from existing artifacts, no new GPU computation:

- **Objective reframing**: English springer is the historical,
  deliberately erased concept in S2 — its continued 0/20 absence is the
  *desired* outcome, not a "failure to restore correct recognition."
  `stage1_history_intervention_s2/FINAL_REPORT.md`'s "quality-vs-
  recognition dissociation" framing and the prior turn's proposed
  recognition-restoration diagnostic are withdrawn; "reverse sequence"
  as a description of S2 relative to S1 is withdrawn (S2 does not share
  a concept pair with S1 in reverse order).
- **Springer case 63**: re-inspected against `EnglishSpringer_parent`'s
  own identical-prompt/seed image and found to be an **inherited**
  rendering pattern (present before any sequential edit), not added
  damage. `S2_STAGE1_HISTORY` springer degeneration corrected **3/20 →
  2/20** (cases 77, 79 only); paired recovery corrected **17/3 →
  18/2**. Cases 77 and 79 (genuinely added distortion, confirmed by
  direct comparison) are unaffected.
- **Retained cases 32 (parachute), 54 (French horn)**: individually
  reinspected and found to have no object present despite coherent,
  non-degenerate images (grid-level review had inferred presence from
  coherence alone). `object_present` corrected `true → false` for both;
  no degeneration count changes. The remaining 38 retained-class images
  in `S2_STAGE1_HISTORY` were not individually re-verified for object
  presence — disclosed as a scope limit, not assumed complete.
- **S1 macro-case arithmetic, propagated**: holding C29's corrected
  `S1_Stage1` baseline (14/20) and the existing final `S1_BLANK`/
  `S1_HISTORY` labels fixed, Stage 1 accounts for **14/19** of
  `S1_BLANK`'s final degenerate count (14 persisted, 5 newly damaged:
  12,13,14,15,17; 1 neither: 18) and **14/20** of `S1_HISTORY`'s (14
  persisted, 6 newly damaged: 12,13,14,15,17,18) — not "100%," and not
  exactly C29's own "18/19 or 18/20" either. Pure set arithmetic over
  already-fixed labels; `S1_BLANK`/`S1_HISTORY` themselves remain
  unreopened.
- **Verification-claim honesty**: the Stage-1 build scripts (S1 and S2)
  never asserted reload equality against the actual saved object
  (`final_state_cpu`), only against the untouched parent — a disclosed
  rigor gap, not a proven defect. The S2_STAGE1_HISTORY classifier guard
  was an after-the-fact check run post-classification, not a true
  precondition gate. Both are fixed, not merely relabeled, in the new
  build below.

## C32 — S2 both-stages: the Stage-1 quality gain mostly survives full refinement, with one disclosed cost (index entry)

`history_both_stages_s2/FINAL_REPORT.md` (published alongside this
entry) applies the already-validated Stage-2 refinement (fp64 stable
solve, historical preservation) to the improved `S2_STAGE1_HISTORY`
checkpoint — one freshly recovered `'golf ball'` embedding (working
UNet = `S2_STAGE1_HISTORY`, reference = pristine M0), one refinement
update (λ=0.1, erase=1, preserve=0.1), reload verified against an
independently cloned intended final state (not just the parent, fixing
the gap disclosed in C31). **Result: golf-ball suppression reaches
0/20 exact (the best of any S2 checkpoint tested, improving on
`S2_STAGE1_HISTORY`'s 1/20 genuine leak and `S2_Stage1`'s own 1/20 false
positive); English springer correctly stays erased (0/20 throughout, as
desired); French horn recognition ties the best of any S2 checkpoint
(14/20, vs. `S2_HISTORY`'s 9/20) and parachute improves over that same
refinement-only `S2_HISTORY` comparator (13/20 vs. 11/20 — corrected
from a mis-cited 9/20, French horn's figure, not parachute's);
retained-class degeneration holds at 0/40.
But springer-prompt degeneration rises from 2/20 (corrected
`S2_STAGE1_HISTORY`) to 6/20** (18 recovered, 2 persisted, 4 newly
damaged, 0 neither vs. that baseline) — refinement reintroduces some
corruption, concentrated in 3 of 5 scenes and 5 of 6 cases at one seed.
A genuine, qualified result on four dimensions, not a clean "the Stage-1
gain simply persists" finding. The classification guard used here is a
single executed wrapper that validates preconditions and invokes
classification in the same guarded path, refusing on failure — see the
report itself for full tables, scene/family breakdowns, and explicit
scope limits.

## C33 — Blinded human review package published for the S2 springer quality question (index entry; no scientific finding yet)

`blinded_quality_review_s2/` (published alongside this entry) prepares,
but does not yet answer, a direct human check of C32's claimed springer
degeneration regression (`S2_STAGE1_HISTORY` 2/20 → `S2_HISTORY_BOTH_STAGES`
6/20). All 20 matched case-60–79 pairs (not only the 6 disputed cases)
are included, randomized once (pair position and A/B assignment,
balanced 10/10), with a private neutral-ID↔endpoint mapping kept outside
the git checkout and a public SHA256 commitment to it
(`MAPPING_COMMITMENT.json`) published before any response is collected.
The frozen protocol applies the same whole-frame-degeneration rubric
used throughout this results line, collects individual-image judgments
before any paired comparison, and does not ask the reviewer to confirm
either assistant's interpretation. Bounded factual corrections
(`closeout/CORRECTIONS_BEFORE_HUMAN_REVIEW.md`), verified against raw
classification records, accompany the package: a mixed visual-numerator/
classifier-denominator suppression estimate (87.5%) is withdrawn; the
French-horn 14/20 totals are shown to involve a 4-case identity shift
(gains 46/58, loses 44/50) between the two endpoints despite the equal
total, while parachute's 13/20 case identities are unchanged; the
existing seed/scene check (`S2_Stage1`/`S2_BLANK`/`S2_HISTORY`, each
20/20 historical degeneration) is noted as a saturated binary count that
cannot distinguish any mechanism, unlike the new, non-saturated 6/20
pattern. **No disputed quality count is changed by this entry.** Status
is `READY_FOR_HUMAN_REVIEW`, not a scientific finding — see
`READY_FOR_HUMAN_REVIEW.md` for exact reviewer instructions.

## C34 — First human response, unblinded: does not reproduce the claimed springer-degeneration regression

`blinded_quality_review_s2/closeout/HUMAN_REVIEW_REPORT.md` (published
alongside this entry) unblinds and analyzes one selected human response
to the C33 package (reviewer alias "Bijay," submitted after a rubric
clarification; an earlier submission was explicitly withdrawn by the
user and is not published or analyzed here). Mapping verified against
the published commitment before opening (`bbf8a9eb8b6459bdb2e4ff33ed8638c5708fe4ecb10e58c09a02ecbdcb47689a`,
exact match); response file verified against its claimed hash/size
before analysis. **Disclosed, not concealed**: the reviewer had prior
exposure to the assistant's aggregate claim and received case-specific
guidance on one study image (Q10) from another assistant before this
submission was frozen — the endpoint mapping itself was not revealed
first. This is reported as blinded to case/endpoint identity with
disclosed prior exposure, not as a fully naive validation.

**Unblinded result: whole-frame degeneration goes `S2_STAGE1_HISTORY`
2/20 → `S2_HISTORY_BOTH_STAGES` 1/20** (1 recovered — case 72, 1
persisted — case 63, 0 newly damaged, 18 neither) — the opposite
direction from C32's claimed regression (2/20→6/20, 4 newly damaged).
**Case-level agreement with the assistant's own current labels is low**:
16/20 for `S2_STAGE1_HISTORY`, 15/20 for `S2_HISTORY_BOTH_STAGES`, with
the two flagged case sets overlapping in exactly one case
(`S2_HISTORY_BOTH_STAGES` case 63) across both endpoints — close or
matching totals should not be read as agreement on which cases. Local
defects are flagged on 39/40 images by this reviewer, far more liberally
than the assistant's own sparse free-text local-defect notes; no
structured comparison is computed for that dimension since the
assistant's schema has no directly comparable field. Three further
textual errata (a mis-cited `S2_HISTORY` parachute comparator, an
overclaim about which degeneration count was "first non-saturated," and
an overstated retained-object-presence verification claim) are corrected
directly in `history_both_stages_s2/FINAL_REPORT.md` and
`blinded_quality_review_s2/closeout/CORRECTIONS_BEFORE_HUMAN_REVIEW.md`,
verified against raw classification records before being applied. **No
disputed quality count from C32 is altered** — this entry adds an
independent, lower-weight, non-reproducing data point against it. One
human reviewer; per the frozen protocol, this does not establish broad
improvement, equivalence, or a new method, and does not end the
question — see the report itself for the full joined table, severity
comparison, and recommendation to recruit further independent reviewers
on the existing, unmodified package before treating either direction as
settled.

## C35 — Second (v2), context-aware human review package prepared — not yet answered

`context_quality_review_s2_v2/` (published alongside this entry) is a
**separate, additive** preparation deliverable, not a revision of v1
(`blinded_quality_review_s2/`, C33–C34) and not yet a scientific result.
It reuses the identical 20 matched case-60–79 pairs and 40 original
images as v1 (re-verified against the evidence index and the frozen
panel manifest `cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2`
before building), adding the generation-prompt context v1 did not show
(the exact shared prompt per pair, taken verbatim from the frozen
manifest) and explicit erasure-objective framing: the erased dog's
absence is expected, not a defect. The rubric term is renamed "widespread
rendering breakdown" (from v1's "whole-frame degeneration"), now
requiring a brief written reason whenever the answer is "yes" or
"unable to judge"; a new optional style-mismatch field is tracked
separately from corruption; a new provenance question asks directly
whether the reviewer discussed or received feedback about any of these
20 specific images beforehand — the exact gap C34 disclosed for v1's
Q10. New, independent randomization (neutral IDs `R01`–`R20`, a fresh
seed, a separate private mapping and public commitment
`MAPPING_COMMITMENT_V2.json`); the build script asserts v1's private
mapping is byte-unchanged before and after the v2 build, confirmed.
**No study image, prior count, assistant label, or endpoint-specific
explanation is used in the three generic illustrative examples shown to
the reviewer; Q10_B is explicitly not used as a calibration example.**
No browser tool was available to verify rendering/layout — disclosed,
not assumed, in `VALIDATION_REPORT_V2.md`. **No response has been
collected for v2; v1's own selected response and its verified
unblinding result (C34) are unchanged and not reanalyzed here.** Any
future difference between v1 and v2 results must not be attributed
specifically to "the addition of context" without further evidence —
v2 is a second development evaluation on the same already-inspected
panel, not an independent replication.

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
- **C7 explicitly preserves uncertainty** rather than forcing a single
  verdict: this reviewer's case 30 re-read is retained, with uncertainty
  about whether its faint background texture is the same artifact family
  at low amplitude — not a disagreement with any specific prior claim
  (see C14, which removes an invented attribution on this point) — and
  the reasoning for both the originally published and the re-reviewed
  labels is published side by side in `CORRECTED_LABELS.json` so a future
  reviewer can adjudicate independently rather than inherit a silently
  resolved call.
- **C13–C16 were found the same way**: re-examining this repository's own
  already-published arithmetic, attributions, and prompt-family metadata,
  and (for C16) re-viewing the already-saved `review_grids/` contact
  sheets and the already-published, unchanged `evidence/VISUAL_REVIEW.json`
  for the existing `A→B` checkpoint — no new GPU computation, image
  generation, or embedding recovery. **C16 is the most consequential**:
  it corrects a false mechanism claim in `embedding_cross_swap/
  FINAL_REPORT.md` (the English-springer "clean column effect") to the
  narrower, direction-specific one-cell pattern actually supported by the
  preserved evidence; it does not change the golf-ball interaction
  finding or any recognition/degeneration count.
