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
