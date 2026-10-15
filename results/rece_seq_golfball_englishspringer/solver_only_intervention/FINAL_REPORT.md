# Solver-only intervention: does accurately solving the existing RECE refinement equations remove the original sequential corruption?

One controlled numerical-implementation test, not a new method. Built one
new endpoint, `AB_STABLE_SOLVE`, from the untouched `AB_UCE.pt` working
checkpoint and the untouched, previously-saved `E_S.pt` embedding, solving
the *existing* RECE Stage-2 objective exactly (fp64 Cholesky) instead of
the native float32 `torch.inverse` path, then generating and classifying
the full 100-image panel for this one endpoint only.

## 1. Preserved state

Starting point: commit `ff48666750783ce1fa8e8af96b5f0ad237b050e0`. Native
sequential objective ≈161687.96 vs accurate reference ≈97.39, ≈12% median
relative weight discrepancy, all 128 prior reference solves valid
(`numerical_optimality_audit/`, unchanged, not repeated). Original
sequential final: 35/40 golf-ball degenerate, 0/20 English-springer
degenerate — **the two E_S cells (AtoB and D_WITH_SEQ_EMB) are each
severely corrupted on different prompt classes; AtoB is not described as
globally clean anywhere in this report.** Recognition: golf 0/40, springer
0/20, parachute 18/20, French horn 19/20; generic dogs in 6/20 springer
cases (`CORRECTIONS.md` C2). Eligibility, parent verification, both
diagonal replays, preliminary-UCE reconstruction, baseline searches, and
the numerical audit are closed and were not repeated. The layer/spatial
localization exercise proposed at the end of the prior audit was **not**
launched — causal localization to "quality-relevant layers" remains
unestablished.

## 2. Operand provenance (GPU, text-encoder only, ~1.3s)

GPU 0 verified idle (99% free) immediately before this allocation.
Recomputed, on `cuda:0` with the native SD1.4 pipeline/tokenizer/text
encoder (matching the device/backend of the original edits, not CPU):

- **Guide** (`tokenizer(" ")` → text_encoder): bit-identical (sha256
  match) to the published `embedding_cross_swap/edits/guide_embedding.pt`.
- **Retain** (`edit_model_adversarial`'s own `tokenizer(["",""])` batch of
  2 → text_encoder): the two batch rows are **bit-identical to each
  other** (max abs diff 0.0), but **do not** bit-match the previously
  published singleton `retain_embedding.pt` (max abs diff ≈4.29e-6).
  Separately confirmed by direct tokenization: `" "` and `""` produce the
  **identical** token-id sequence (`[BOS,EOS,PAD,...]`), so this is a
  genuine, small, **batch-size-dependent floating-point artifact** of the
  text-encoder forward pass, not a text/content difference.

Gate passed (guide exact; retain rows agree with each other). Per task
instruction, this build uses the **native** retain operand (the common
row0=row1 value) rather than the prior audit's singleton approximation —
disclosed here as superseding that approximation for this build only; the
prior audit's own publication is not retroactively edited.

**Projection-bias check**, done structurally (not from a key-set
whitelist): directly instantiated the SD1.4 UNet and read `.bias` on all
16 cross-attention modules' `to_v`/`to_k`/`to_q` `nn.Linear` layers —
confirmed `None` for all of them.

GPU released immediately after (idle again within seconds, confirmed).

## 3. Solve and build (CPU-only, 21.9s)

`CUDA_VISIBLE_DEVICES=""` set in-process; `torch.cuda.is_available()`
asserted `False` before solving. All 3 input hashes re-verified live
(`AB_UCE.pt`, `E_S.pt`, `AtoB_english_springer.pt` — all matched). `H`
(768×768, built from `E_S` + the native retain operand) is positive
definite (min eig 0.1, max eig 1.607099e6, **cond(H) = 1.60710e7** —
identical to the prior audit's E_S-cell conditioning, confirming the
tiny retain-operand change does not materially move `H`); Cholesky
succeeded. All 32 layers solved via one cached Cholesky factor.

**Reference-solve validation** (frozen tolerances, this build's actual
operands): normalized backward error for `V*` (fp64) ranged
2.72e-17–3.02e-17, **≤ the frozen 1e-10 tolerance on every layer**;
quadratic-gap identity held to tolerance on every layer. **Reference
solve fully valid, no layer excluded.**

**Native vs. reference** (same comparator checkpoint, `AtoB_english_
springer.pt`, under this build's operands): native backward error
0.93e-3–2.74e-3; excess objective (native − V\*) is **536×–3875×
`J(V*)`** per layer (median 1609×), consistent with the prior audit.
This is not re-litigated here — it was already established and closed;
cited only to confirm this build's operands reproduce it.

**Save + reload verification**: `AB_STABLE_SOLVE.pt` (clone of
`AB_UCE.pt`, 32 layers replaced with the fp32 cast of `V*`), sha256
`3b491531a71c86c3b97dad0a9a50cd52112e82334951c5e041c49278366e8829`.
Reload confirms: **exactly 32 keys changed** (the expected set, no extra,
none missing), all 686 tensors finite, the 32 changed tensors
bit-identical to the intended fp32 cast, and the other 654 tensors
bit-identical to `AB_UCE.pt`.

**fp32 storage cost, measured separately** (not required to meet the
1e-10 fp64 tolerance): casting `V*` to fp32 and back to fp64 changes the
total objective by ≈5e-8 relative (97.388368776 → 97.388368827); its own
backward error is 9.08e-10–9.97e-10 per layer — still ~6 orders of
magnitude better than native's 0.93e-3–2.74e-3. **fp32 storage is not a
meaningful source of error here**; the entire native/reference gap is in
the solve itself, as already established.

## 4. Image generation and review (GPU, ~390s total)

GPU 0 re-verified idle immediately before this second allocation. 100
images generated for `AB_STABLE_SOLVE` only (386.7s), identical mechanics
to every prior endpoint (float32, 512px, 50 LMS steps, guidance 7.5,
per-case seed, frozen manifest sha256
`ca1401c0064441969fdf3ab329fb9ab8eaab980b63ade62b6f14bd9bd1ef43ad`).
Classified 100/100, 0 errors (1.8s). GPU released and confirmed idle
within seconds. No comparator images were regenerated.

**Blinding limitation, disclosed** (same pattern already established for
`embedding_cross_swap`): per-class aggregate recognition counts were
queried immediately after classification and before the visual review
began; per-case predictions were consulted only after initial visual
labels were recorded. This is not independent human validation. Full
method statement: `VISUAL_REVIEW.json`.

### 4.1 English springer (erasure target)

| | native AtoB | AB_STABLE_SOLVE |
|---|---|---|
| Recognition | 0/20 | **0/20 (unchanged)** |
| Suppression vs. A's 20/20 | 100% | **100% (unchanged)** — the 75% gate remains met, not a claim of matching A's 20/20 |
| Degeneration | 0/20 | **0/20 (unchanged)** — already clean; nothing to fix |
| Generic dog presence | 6/20 (C2) | **2/20** (cases 86 "Irish setter" p=0.261, 90 "Yorkshire terrier" p=0.288 — real but non-target breeds, same pattern as native AtoB's) |

Exact-label deletion is preserved exactly. Quality was already good and
stays good. **Generic-dog leakage on the target class drops by two
thirds** (6/20→2/20) under accurate solving — a genuine, if secondary,
deletion-adjacent improvement the exact-match metric cannot see either
way.

### 4.2 Golf ball (historical, non-target)

| | native AtoB | AB_UCE (Stage-1) | A (context) | AB_STABLE_SOLVE |
|---|---|---|---|---|
| Recognition | 0/40 | — | 3/40 | **0/40 (unchanged)** |
| Degeneration | 35/40 (87.5%) | 2/40 (5%) | 7/40 (17.5%) | **30/40 (75.0%)** |

Recognition stays at 0/40 — far below both the ≥7/40 material-return and
≥10/40 suppression-loss flags; none of the 10 now-clean cases shows a
visible golf ball on individual inspection (not an object return, same
finding as `reference_A_intervention`).

**Paired transition vs. native AtoB**: 30 persisted degenerate / **5
recovered** (28, 29, 30, 31, 38) / 0 newly damaged / 5 neither (15, 24,
25, 26, 27). Net **+5** cases, no case moved the other direction — a
modest, real, one-directional improvement (87.5%→75.0%, 14.3% relative
reduction).

**Paired transition vs. AB_UCE (Stage-1 input) — inherited vs.
refinement-introduced damage**: AB_UCE's 2 pre-existing degenerate cases
(37, 39) **both persist unchanged** into AB_STABLE_SOLVE — accurate
Stage-2 solving does not repair Stage-1's own damage. The other **28 of
30** degenerate cases (93%) are newly introduced *during* Stage 2 itself,
despite Stage 2 now being solved to the fp64 optimum. **Refinement
introduces the large majority of golf-ball corruption, and accurate
solving does not eliminate it.**

Full case lists and paired table vs. A (context only): `VISUAL_REVIEW.json`.

### 4.3 Retained classes

| | native AtoB | AB_UCE | AB_STABLE_SOLVE |
|---|---|---|---|
| Parachute recognition / degeneration | 18/20 / 0/20 | 15/20 / — | **17/20 / 0/20** |
| French horn recognition / degeneration | 19/20 / 0/20 | 18/20 / — | **16/20 / 0/20** |

Zero severe degeneration either class, unchanged. A real, small,
negative recognition-panel cost vs. native AtoB (French horn −3,
parachute −1), reported as plain panel counts (`CORRECTIONS.md` C12),
not noise. French-horn misses are recognizable instruments at unusual
crops (strict-classifier-vs-visual divergence, already documented
elsewhere in this line); parachute misses match the pre-existing
packed-backpack ambiguity.

### 4.4 Every tradeoff, stated plainly

This is **not** an unqualified rescue. Accurate solving: (a) leaves
exact-label deletion on both target classes unchanged, (b) meaningfully
reduces generic-dog leakage on the erasure target (6/20→2/20), (c)
modestly reduces golf-ball degeneration (35/40→30/40) without any object
return, and (d) costs a small amount of retained-class recognition
(French horn −3, parachute −1) at zero quality cost. **Golf-ball
corruption remains the majority outcome (75%) even under a verified
fp64-accurate solve** — cleaner images were not achieved primarily by
"restoring golf balls" (none returned) or by materially weakening
new-target deletion (unchanged); the retention cost is real but small.

## 5. Closing the outstanding visual-review record (no new generation)

`CASE_LEVEL_COMPLETION_RECORD_120.json`: a genuine 120-row table (40
golf-ball + 20 English-springer, × `D_WITH_SEQ_EMB` and
`S_WITH_DIRECT_EMB`), each row carrying the actual image path, its
sha256 (recomputed live, 0 missing), the resolution at which it was
viewed (`full_512x512_crop` for the 14 previously-crop-saved cases,
`grid_thumbnail_180px` for the rest, consistent with this line's
established practice of judging the pervasive mesh signature reliable at
thumbnail resolution), and degeneration / recognition / dog-presence
fields. `D_WITH_SEQ_EMB`'s 22 golf-ball recognition case ids —
**[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25,
28, 29, 33]** (12 literal, 10 paraphrase) — were read directly from its
published `classifications.jsonl`, not inferred or claimed unavailable.
Three of these (1, 25, 28) were additionally viewed at full 512×512
resolution this session and show genuine, sharp, recognizable golf balls
— `D_WITH_SEQ_EMB`'s recognition is real visual return on those cases,
unlike `S_WITH_DIRECT_EMB`'s 2/40 (cases 17, 26, both previously
confirmed false positives — no visible ball). No case was unavailable;
no aggregate is provisional. Original labels are preserved unchanged;
this is an unblinded, prediction-informed retrospective review, stated
as such, not independent validation.

## 6. Interpretation

**Numerical implementation is a real, demonstrated contributor to the
original sequential endpoint's failure, but an insufficient one on its
own.** Holding working checkpoint, embedding, hyperparameters, and layer
scope fixed and only replacing the float32 assembly-and-inverse path with
an fp64 Cholesky solve of the *same* stated objective: English-springer
deletion is unaffected (already correct) and its generic-dog leakage
drops substantially; golf-ball degeneration improves by a real but modest
margin (14% relative) and remains the majority outcome; retained-class
recognition takes a small real cost. **Most of the golf-ball corruption
(28 of 30 degenerate cases) is refinement-introduced, not inherited from
Stage 1, and persists despite a verified-accurate solve** — this is
evidence against numerical solver inaccuracy being the dominant
mechanism for golf-ball corruption specifically, while being consistent
with it being a real, partial contributor (the 5 recovered cases, the
dog-leakage reduction). This is framed as an **implementation-accuracy
finding**, not a new method: the intervention changed only how the
existing equations are assembled and solved, nothing about what RECE
does conceptually. One development pair (golf ball / English springer)
cannot establish generality to other concepts, and the separate
embedding-recovery computation (`close_form_emb_regzero`, which produced
`E_S` in the first place) remains unchanged and untested by this
intervention — a numerically-inaccurate recovery step could still be
contributing to whatever is not explained here.

## 7. One next scientific decision (not executed)

Test whether the embedding-recovery step itself (`close_form_emb_
regzero`, which produced `E_S`) is accurately solving *its own* stated
objective, using the same fp64-reference-solve methodology as this task
and the numerical audit — this is a different computation (a different
objective, over the UNet's `to_v` weights used as a basis for the
adversarial embedding, not the Stage-2 refinement solved here) and was
explicitly excluded from both the numerical audit and this intervention.
**Not run here.**

## 8. Resources and release

GPU 0 idle-verified immediately before each of 2 allocations (text-encoder
check: free_frac 0.989; generation+classification: confirmed idle
immediately before, confirmed idle again within seconds after release).
Total GPU-held time ≈390s (1.3s + 386.7s + 1.8s), one GPU
(`CUDA_VISIBLE_DEVICES=0`), no other GPU touched, no other users' jobs
present. CPU-only for the solve (21.9s, 8 threads, `CUDA_VISIBLE_
DEVICES=""`). All outputs under `/data/bijaypandey/cuig_pilot/
rece_seq_golfball_englishspringer/solver_only_intervention/`.

## 9. Artifacts

`FROZEN_MANIFEST.json` (written before Section 4's solve/build),
`OPERAND_PROVENANCE_CHECK.json`, `native_guide_embedding.pt` /
`native_retain_row{0,1}.pt` (the verified native operands, small
float32 tensors), `run_operand_provenance_check.py`,
`run_solver_only_build.py`, `RESULT.json` (per-layer fp64/fp32 numerical
tables), `edits/AB_STABLE_SOLVE.pt`, `make_review_grids.py`,
`review_grids/AB_STABLE_SOLVE/*.png`, `eval/AB_STABLE_SOLVE/` (100
images, `image_report.json`, `classifications.jsonl`,
`classify_report.json`), `VISUAL_REVIEW.json`,
`CASE_LEVEL_COMPLETION_RECORD_120.json`, this `FINAL_REPORT.md`.

---

**No second endpoint, baseline repeat, embedding recovery, parameter
sweep, concept pair, layer search, or automatic follow-up was run.
Finishing this bounded assignment here.**
