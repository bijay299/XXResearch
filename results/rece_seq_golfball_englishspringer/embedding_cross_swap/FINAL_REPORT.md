# Embedding cross-swap: does sequential refinement corruption follow the recovered embedding, the working checkpoint, or their interaction?

One bounded 2x2 diagnostic, completed without pausing between authorized
stages. **Answer: it depends on which prompt family. On the erasure
target (English springer), corruption cleanly follows the embedding —
any cell using E_S is severely corrupted regardless of working checkpoint,
and any cell using E_D is clean regardless of working checkpoint. On the
historical, non-target class (golf ball), corruption shows a genuine
interaction — it is severe only when the sequential working checkpoint
and the sequential embedding are combined; substituting either factor
alone sharply reduces it, but not identically.** This is a causal
diagnostic of this specific sequential adaptation's mechanics, not a
new-method claim.

## 0. Carried-forward verified state (not re-derived here)

From commit `5b2550c89084dcf7a12a203ad190c4be80e890ec`:

- Original M0-reference refinement: direct golf-prompt degeneration
  0/40→0/40; sequential 2/40→35/40 (AB-UCE→A→B).
- Original final direct/sequential English-springer recognition: 0/20
  each.
- The parent-A-reference intervention did not rescue the sequential
  model; its reported 35/40→31/40 improvement is **provisional** pending
  `CORRECTIONS.md` C7 (full-resolution re-review moves this to a net +2,
  not +4 — see §1 below).
- Parent-A-reference recognition: English springer 1/20, golf ball 0/40,
  parachute 16/20, French horn 18/20 (vs. original sequential final 0/20,
  0/40, 18/20, 19/20).
- The original sequential M0-reference replay already matched the saved
  final checkpoint exactly across all 686 UNet tensors — closed, not
  repeated.

Eligibility, first-parent confirmation, preliminary-UCE construction,
baseline searches, the completed sequential replay, and existing endpoint
image generation were **not** repeated.

## 1. Narrow reporting corrections (no new GPU computation)

Six description/listing corrections were made to the previously published
`reference_A_intervention` evidence, all from re-examining existing
artifacts already in hand (no images regenerated): see `CORRECTIONS.md`
C6–C12. Headline consequences:

- **C6**: a listing bug in the published paired-transition table (four
  recovered cases were also listed as persisted) is fixed; the correct
  figure under the originally published labels is 30 persisted / 5
  recovered / 1 newly damaged / 4 neither (unchanged aggregate totals,
  35/40→31/40).
- **C7**: independent full-resolution re-review of the 5 cases called
  "recovered" (28, 29, 30, 31, 33) found 2 (29, 33) are actually dominated
  by the same corruption artifact as the degenerate majority —
  recognizable foreground content alone does not make an image usable.
  **AB_REF_A's golf-ball degeneration is 33/40 (82.5%), not 31/40
  (77.5%); its net quality improvement over the native sequential final
  is +2 cases, not +4.** Disagreement with one external read of case 30
  is left explicit, not forced to resolution.
- **C8**: case 33 is a "flying golf ball" literal prompt, not part of the
  "hand gripping" paraphrase family the prior report grouped it with.
- **C9**: the prior ~22%-relative-embedding-difference figure is
  overwhelmingly directional (≈12.70° angle), not magnitude (0.03993%
  magnitude change) — relevant context for interpreting this report's own
  much larger embedding differences in §4 below.
- **C10**: the prior script's 32-key whitelist check was logged, not
  enforced (no actual violation occurred, but the check as implemented
  could not have stopped one). **This report's own script uses a literal,
  raising `assert`** for the same check (§3).
- **C11**: the preliminary-UCE token-suffix slice length is
  `L − max(i,j)`; for this experiment's actual concept it is 75 of 77
  tokens (near-full-length), not "often one token wide" as previously
  (incorrectly) stated.
- **C12**: "within ordinary noise" language for retained-class recognition
  swings is retracted as an unestablished statistical claim; this report
  uses plain panel counts throughout (see §5).

Full reasoning and recomputed tables: `CORRECTED_LABELS.json`, published
alongside this report.

## 2. Design

| | E_D (recovered from W_D) | E_S (recovered from W_S) |
|---|---|---|
| **W_D** (B_UCE.pt, direct Stage-1, from M0) | existing **B** — reused | **NEW: D_WITH_SEQ_EMB** |
| **W_S** (AB_UCE.pt, sequential Stage-1, from A) | **NEW: S_WITH_DIRECT_EMB** | existing **A→B** — reused |

Both E_D and E_S are recovered via `close_form_emb_regzero` with
**pristine M0 as the reference in both cases** — this experiment holds
the reference fixed (unlike the prior task) and instead varies which
working checkpoint supplies the embedding. The receiving checkpoint's own
native guide (`' '`) and retain (`''`) embeddings and weight regularizer
(`lamb=0.1`) are used unmodified in every cell; only the recovered
embedding itself is transferred, complete and untrimmed (77 tokens,
`old_embeddings=[E[0]]`, matching `run_edit.py`'s own convention). All 32
cross-attention K/V projections, `erase_scale=1`, `preserve_scale=0.1`,
technique `replace`, precision, and layer traversal are unchanged. Frozen
manifest: `FROZEN_MANIFEST.json`, written before any computation.

## 3. Verification

### 3.1 Input checkpoints

| checkpoint | sha256 |
|---|---|
| W_D (`B_UCE.pt`) | `23b691460d7189b7e8c78ceb5ea8441a0400f61c6a79abf8a6e8afd89a161bc4` |
| W_S (`AB_UCE.pt`) | `3a9d86de1339f9c516399fde1a9ab3c6b4ff8851062fb371712d473e7862b550` |
| existing B (`B_english_springer.pt`) | `c711e97127d3dcca9d6752e86d64fba8240ed20347c699179f9fe0859267b9dc` |
| existing A→B (`AtoB_english_springer.pt`) | `ec16c9f748035b832d57dbe6b2eaa33d0e26279fe9a852eb171837f48a7a2469` |

All four verified against the expected hashes and against their own
loaded tensors (per-key equality after load), before any computation.

### 3.2 Phase 1 — recovery (reference M0, both E_D and E_S)

| check | E_D recovery (from W_D) | E_S recovery (from W_S) |
|---|---|---|
| Working/reference storage separation, all 686 tensors | PASS, 0 shared | PASS, 0 shared |
| Reference (M0) unchanged through recovery, all 686 tensors | PASS, 0 changed | PASS, 0 changed |
| Working checkpoint itself unchanged by `close_form_emb_regzero` (it only reads weights) | PASS, 0 changed | PASS, 0 changed |

Concept embedding ("English Springer"), guide embedding (`' '`), and
retain embedding (`''`) were each computed from **two different
pipelines** (the W_D-loaded and W_S-loaded pipelines) and verified
bit-identical — confirming the text encoder is genuinely untouched by any
UCE edit in this line, as assumed throughout.

### 3.3 Phase 2 — gate: image-free direct compatibility check

E_D was applied natively (fresh, untouched W_D, native guide/retain,
unchanged hyperparameters) via `edit_model_adversarial` and compared
against the existing `B_english_springer.pt`, all 686 UNet tensors,
**values not hashes**.

**Result: exact match. 0 differing keys.** This is the reciprocal of the
sequential-branch replay already verified in the prior task — it confirms
`B_UCE.pt` (W_D) is a faithful reconstruction of what the original direct
run's Stage-1 output actually was, the direct-branch analogue of a check
that had not previously been performed. Had this not matched exactly,
this report would stop here with a discrepancy record; none was needed.

### 3.4 Phase 3 — building the two new cells

Each cell starts from a fresh, untouched load of its row's working
checkpoint (never the gate-check object, never re-used across cells).

| cell | changed-key check | out checkpoint sha256 |
|---|---|---|
| D_WITH_SEQ_EMB (W_D + E_S) | **enforced `assert`** (raises on mismatch, per C10's fix) — exactly the 32 expected cross-attn K/V keys, no extra, no missing | `234c5cf39410a00f74a1c7ae7d3d412a6da172582c31c24d6c522ba58e466801` |
| S_WITH_DIRECT_EMB (W_S + E_D) | **enforced `assert`**, same result | `fa88928baab20db752cdf2b9bc88e8c9460b886e62ecb15e0c23865a6f04f59b` |

## 4. Diagnostics (descriptive; computed CPU-only from saved tensors after the GPU was released, per `FROZEN_MANIFEST.json`'s frozen formulas)

### 4.1 Embedding comparison

| | value |
|---|---|
| E_D norm | 329.54 |
| E_S norm | 1285.17 (identical to the prior task's own measurement of the same quantity — a consistency check across sessions) |
| Difference norm | 1348.12 |
| Relative difference (vs. E_D) | **409%** |
| Cosine similarity | **−0.0675** (near-orthogonal) |

This is a dramatically larger embedding difference than the prior
reference-A-intervention task found when varying only the *reference*
(22.12% relative difference, ≈12.70° angle, magnitude nearly fixed at
0.04% — C9). **Varying which working checkpoint recovers the embedding
moves it roughly an order of magnitude further than varying the
reference did**, in both direction (near-orthogonal vs. ~13°) and
magnitude (3.9x vs. ~1.0004x).

### 4.2 Per-cell diagnostics (2x2)

| cell | edit displacement | recovery-fit residual | erase-fit residual | retain-projection drift |
|---|---|---|---|---|
| W_D,E_D (B) | 37.11 | 1140.87 | **22.42** | **15.17** |
| W_S,E_D (S_WITH_DIRECT_EMB, new) | 30.37 | 1378.68 | **21.98** | **14.64** |
| W_S,E_S (A→B) | 33.00 | 533.68 | **401.58** | **55.95** |
| W_D,E_S (D_WITH_SEQ_EMB, new) | 42.07 | **12989.18** | **403.63** | **55.60** |

**Erase-fit residual and retain-projection drift both cleanly track the
embedding (column), not the working checkpoint (row)**: both E_D cells
sit at ≈22 / ≈15 regardless of row; both E_S cells sit at ≈402 / ≈56
regardless of row (roughly an 18x and 4x gap respectively, consistent
across rows). **Recovery-fit residual shows a specific interaction**: the
(W_D, E_S) cell is roughly 10–24x higher than any of the other three,
which are otherwise comparable in order of magnitude. Edit displacement
does not show a clean pattern either way. Full per-layer detail:
`PER_LAYER_DISPLACEMENT.json`; raw values: `DIAGNOSTICS.json`.

**These residuals are descriptive diagnostics of how well each cell's
closed-form fit succeeded, not semantic-deletion certificates.** Notably,
the recovery-fit residual's interaction signature does **not**
straightforwardly predict the §5 visual outcome: (W_D, E_S) has by far
the worst recovery-fit residual yet only mild (12.5%) golf-ball
degeneration, while (W_S, E_S) has a much better recovery-fit residual
yet catastrophic (87.5%) golf-ball degeneration. This mismatch is reported
as an open question, not resolved here.

## 5. Evaluation (200 new images; all other endpoints reused unchanged)

Manifest unchanged (sha256
`ca1401c0064441969fdf3ab329fb9ab8eaab980b63ade62b6f14bd9bd1ef43ad`),
inference/classifier settings unchanged. GPU: idle-verified GPU 0,
`CUDA_VISIBLE_DEVICES=0`. Both endpoints: 100/100 classified, 0 errors.

### 5.1 Recognition table

| class | M0 | A | B-UCE | B | AB-UCE | A→B | **D_WITH_SEQ_EMB** | **S_WITH_DIRECT_EMB** |
|---|---|---|---|---|---|---|---|---|
| golf ball | 39/40 | 3/40 | 40/40 | 37/40 | 0/40 | 0/40 | **22/40** (12 lit, 10 par) | **2/40** (2 lit, 0 par) |
| parachute | 16/20 | 15/20 | 16/20 | 15/20 | 15/20 | 18/20 | **16/20** | **14/20** |
| French horn | 20/20 | 18/20 | 20/20 | 19/20 | 18/20 | 19/20 | **20/20** | **16/20** |
| English springer | 20/20 | 20/20 | 1/20 | 0/20 | 0/20 | 0/20 | **0/20** | **0/20** |

### 5.2 English springer (erasure target) — follows the embedding

| | D_WITH_SEQ_EMB (W_D + E_S) | S_WITH_DIRECT_EMB (W_S + E_D) |
|---|---|---|
| Exact recognition | 0/20 | 0/20 |
| **Degeneration** | **20/20 (100%)** — pervasive corruption in every case inspected | **0/20 (0%)** — fully coherent in every case |
| Dog presence | cannot be confidently assessed under the corruption (not asserted as "absent") | **1/20** (case 90, visually a dog, classifier top-1 "Blenheim spaniel" p=0.333 — not equated with the targeted breed) |

Both cells achieve the same 0/20 exact-label suppression — **this metric
saturates and cannot by itself distinguish these two radically different
outcomes.** The quality difference is total: corruption is present
whenever E_S is the supplied embedding, absent whenever E_D is, on this
class, regardless of which working checkpoint received it.

### 5.3 Golf ball (historical, non-target) — interaction

| | D_WITH_SEQ_EMB (W_D + E_S) | S_WITH_DIRECT_EMB (W_S + E_D) |
|---|---|---|
| Exact recognition | 22/40 (vs. native B's 37/40 — a real 15-case collateral drop; golf ball is not erased on this lineage and "should remain available," so this is reported plainly, not against an erasure gate) | 2/40 (below A's own 3/40; no material return, threshold ≥7/40 not met; no loss of 75% suppression qualification, threshold ≥10/40 not met) |
| **Degeneration** | **5/40 (12.5%)**, concentrated in the "flying ball" family (34,35,37,38,39) | **0/40 (0%)** |
| vs. own row's Stage-1 input | B-UCE: 0/40 → 5/40 (5 newly damaged) | AB-UCE: 2/40 → 0/40 (2 recovered: cases 37, 39) |
| vs. own row's native diagonal cell | B: 0/40 → 5/40 (5 newly damaged) | A→B: 35/40 → 0/40 (all 35 recover) |
| vs. parent A (context only, not a paired partner for the direct row) | not a natural comparison (direct row never passes through A) | A: 7/40 → 0/40 — all 7 of A's own pre-existing degenerate cases recover; individually spot-checked (cases 1, 35) |

**This is the interaction**: golf-ball corruption is severe (87.5%) only
in the cell combining the sequential working checkpoint *and* the
sequential embedding. Substituting either factor alone — the direct
embedding on the sequential checkpoint (S_WITH_DIRECT_EMB), or the
sequential embedding on the direct checkpoint (D_WITH_SEQ_EMB) —
collapses most or all of it, but not to the same degree or in the same
direction: S_WITH_DIRECT_EMB goes further, recovering even A's own
pre-existing degeneracy, while D_WITH_SEQ_EMB still picks up a real,
if much smaller (12.5%), spillover concentrated in one prompt
sub-family.

### 5.4 Retention (French horn, parachute)

Both show **zero severe degeneration** in both new cells — the
quality effects above are concentrated on golf-ball and English-springer
prompts, consistent with every prior report in this line.
**S_WITH_DIRECT_EMB carries a real recognition-panel cost**: French horn
16/20 (vs. its row's own native 19/20) and parachute 14/20 (vs. 18/20) —
reported as plain panel counts, not characterized as noise (C12).
D_WITH_SEQ_EMB shows no comparable drop (French horn 20/20, parachute
16/20, both close to native B's 19/20 and 15/20).

## 6. Interpretation (conditional, as scoped)

**Within-row contrasts** (same working checkpoint, different embedding)
isolate the supplied embedding's own effect:
- Row W_D: E_D → B (clean on both classes) vs. E_S → D_WITH_SEQ_EMB
  (English-springer fully corrupted; golf ball mildly spillover-damaged,
  12.5%; retention intact).
- Row W_S: E_S → A→B (both classes severely affected) vs. E_D →
  S_WITH_DIRECT_EMB (both classes recover; retention takes a real but
  modest hit).

**Across-row/column contrasts** also carry the working checkpoint's own
native guide/preservation targets and prior edit history, so they are
not a pure embedding effect — reported as such, not claimed otherwise.

**On the erasure target (English springer), corruption follows the
embedding cleanly** — a main effect of column, invariant to row, both
directions. **On the historical class (golf ball), corruption follows an
interaction** — severe only under the (sequential-checkpoint,
sequential-embedding) combination; the quantitative diagnostics
(erase-fit residual, retain-projection drift) that cleanly track the
embedding for the *general* corruption signature do not, by themselves,
explain this class-specific interaction, and recovery-fit residual's own
interaction signature points in a direction the visual outcome does not
confirm. **Mixed pattern, reported as mixed — this does not establish
that the recovered embedding alone determines corruption, nor that the
working checkpoint alone does; both matter, and they matter differently
depending on which prompt class is examined.**

This remains one selected concept sequence, one classifier, one
Assistant-only visual review (stated blinding limitations above), on a
reused development panel. **No claim is made that this generalizes
beyond this golf-ball/English-springer pair, or establishes universal
properties of RECE's recovery or erase equations.** No additional
reference, sweep, concept pair, or follow-up was launched.

## 7. Resource usage and release

| stage | time |
|---|---|
| GPU idle check immediately before allocation | GPU 0 confirmed IDLE at `2026-10-14T21:31:23-04:00` |
| Phase 1–3 (recovery, gate, two new cells) | 107.5s |
| GPU idle re-check before generation | GPU 0 confirmed IDLE at `2026-10-14T21:33:5x` |
| Generate D_WITH_SEQ_EMB (100 images) | 387.1s |
| Generate S_WITH_DIRECT_EMB (100 images) | 387.0s |
| Classify both (100 each) | 1.9s + 1.8s |
| Diagnostics (`compute_diagnostics.py`) | CPU-only, after GPU release, not counted against GPU time |
| **Total GPU-held time** | **≈14m45s, one GPU (GPU 0), `CUDA_VISIBLE_DEVICES=0`** |

GPU released immediately after classification; confirmed idle again (all
4 GPUs IDLE) at `2026-10-14T21:48:05-04:00`. No other users' jobs were
present or touched throughout.

**Artifacts**: `FROZEN_MANIFEST.json`, `run_embedding_cross_swap.py`,
`compute_diagnostics.py`, `VERIFICATION_LOG.json`, `DIAGNOSTICS.json`,
`PER_LAYER_DISPLACEMENT.json`, `edits/{D_WITH_SEQ_EMB,S_WITH_DIRECT_EMB}.pt`
+ `*.cross_swap_log.json`, `edits/{E_D,E_S,concept_embedding,
guide_embedding,retain_embedding}.pt`, `eval/{D_WITH_SEQ_EMB,
S_WITH_DIRECT_EMB}/`, `review_grids/`, `VISUAL_REVIEW.json`,
`ANALYSIS_RESULTS.json`, `CORRECTED_LABELS.json` (§1 corrections),
`FINAL_REPORT.md` (this file).

## 8. One next scientific decision

**Localize the golf-ball-specific interaction**: determine whether it
originates in W_S's own pre-existing fragility (inherited from A's 17.5%
baseline degeneracy plus Stage-1 composition) interacting with E_S's
erase direction, or in some other property of the (row, column) pairing
not captured by the four diagnostics computed here. A minimal next
diagnostic (not run here, not authorized by this task): compute the same
four residuals restricted to just the golf-ball-prompt-relevant
projection directions (rather than the full embedding), which might
reconcile the recovery-fit residual's uninformative interaction signature
with the visual outcome — no new images required, CPU-only. No new
final-RECE runs, no reverse ordering, no additional checkpoints or
concepts, and no automatic follow-up were run beyond this bounded
diagnostic. Stopping here as scoped.
