# Numerical-optimality audit of `edit_model_adversarial`'s closed-form UCE solve, plus correction of the embedding-cross-swap report

CPU-only. No GPU allocated, no image generated, no model trained, no
embedding recovered, no baseline searched, no parameter swept, no new
evaluation checkpoint built. This closes out the bounded assignment:
correct one false mechanism claim in the latest published report, complete
the case-level visual-review record for the two new cross-swap cells using
only existing images/labels, fix a prose-arithmetic bug and an invented
attribution in `CORRECTIONS.md`, and audit whether the native closed-form
UCE solver actually solves its own stated objective, for all four
`embedding_cross_swap` cells and all 32 edited cross-attention K/V layers.

## 1. Corrected four-cell interpretation

`embedding_cross_swap/FINAL_REPORT.md`'s opening summary and §6 claim:
*"On the erasure target (English springer), corruption cleanly follows
the embedding — any cell using E_S is severely corrupted regardless of
working checkpoint."* **This is false.** The existing, unchanged
`A→B` checkpoint is cell (W_S, E_S) — it uses E_S — and its own
already-published `evidence/VISUAL_REVIEW.json` records **0/20**
English-springer degeneration, with a coherent 20-image grid (6 of which
still show generic-dog content under the corrected count, C2 — a
target-*persistence* concern, not a corruption one). This cell was
reused unchanged in that very report's own design table and was never in
question; the summary sentence simply contradicts the report's own data.
Full correction: `CORRECTIONS.md` C16.

**The actual pattern, all four cells:**

| | E_D | E_S |
|---|---|---|
| **W_D** | B (existing): **0/20** springer degenerate, 0/40 golf degenerate | D_WITH_SEQ_EMB (new): **20/20** springer degenerate, 5/40 golf degenerate |
| **W_S** | S_WITH_DIRECT_EMB (new): **0/20** springer degenerate, 0/40 golf degenerate | A→B (existing, preserved): **0/20** springer degenerate, 35/40 golf degenerate |

Severe English-springer degeneration occurs in exactly **one** of the
four cells — (W_D, E_S) — not in "any cell using E_S." Holding the
embedding fixed at E_S and swapping only the receiving checkpoint (A→B's
W_S → D_WITH_SEQ_EMB's W_D) flips the outcome from 0/20 to 20/20; holding
the embedding fixed at E_D and doing the same swap (B's W_D →
S_WITH_DIRECT_EMB's W_S) changes nothing (0/20 both times). This is a
**direction-specific transfer effect**, not a column (embedding) main
effect: E_S is well-behaved on its own native checkpoint and badly
behaved only when moved onto the *other* branch's checkpoint, while E_D
is well-behaved under the same kind of transfer in both directions. No
mechanism for this asymmetry is established here — it is reported as an
open, narrower finding (full per-case paired-transition tables:
`CASE_LEVEL_COMPLETION_RECORD.json`).

**Golf ball is unaffected by this correction** — the report's own
"interaction" framing there (severe only at (W_S,E_S), sharply reduced by
either single-factor substitution: 0/40, 5/40, 35/40, 0/40 across the
four cells respectively) is independently reconfirmed in this session by
viewing both new cells' full `golf_ball.png` contact sheets, and is
unchanged.

**Exact recognition/quality tradeoffs** (unchanged from
`embedding_cross_swap/FINAL_REPORT.md`, independently reconfirmed this
session at grid resolution for both new cells, full case-level detail in
`CASE_LEVEL_COMPLETION_RECORD.json`):

| | D_WITH_SEQ_EMB (W_D,E_S) | S_WITH_DIRECT_EMB (W_S,E_D) |
|---|---|---|
| Springer recognition / degeneration | 0/20 / **20/20** | 0/20 / **0/20** |
| Golf-ball recognition / degeneration | 22/40 (vs native B's 37/40) / **5/40** | 2/40 (vs A's 3/40, no material return) / **0/40** |
| French horn / parachute recognition | 20/20, 16/20 (no quality cost) | 16/20, 14/20 (real, modest recognition-panel cost; no quality cost) |

## 2. Visual-review completion

`CASE_LEVEL_COMPLETION_RECORD.json` records, per case, what was
inspected and at what resolution: all 120 target images (40 golf-ball +
20 English-springer, ×2 new cells) were viewed this session as full
contact-sheet grids (`embedding_cross_swap/review_grids/*/{golf_ball,
English_springer}.png`), independently reconfirming the already-published
aggregate counts; the 10 cases with an existing true full-resolution crop
(`images/corrected_cases/{D_WITH_SEQ_EMB,S_WITH_DIRECT_EMB}_case*.png`)
were cross-checked individually. No image was regenerated. Target
presence/absence/ambiguity, generic-dog presence, breed uncertainty, and
degeneration are recorded as separate fields, with paired transitions for
all four row/column pairs on the English-springer class (new — this
table did not previously exist) and the existing golf-ball
paired-transition table is reused unchanged. A small number of
individual-case judgments (6 "milder artifact, non-degenerate" golf-ball
cases in D_WITH_SEQ_EMB, and the exact case-id list behind
D_WITH_SEQ_EMB's 22/40 golf-ball recognition count) rest on grid
resolution only and are explicitly marked **provisional** — this does not
affect any published aggregate, all of which were independently
reconfirmed.

`CORRECTIONS.md` gained four new entries this session:
- **C13**: fixes a prose-arithmetic bug in C7's own recomputed
  AtoB→AB_REF_A row (published as 30/3/1/3 = 37, corrected to **32
  persisted / 3 recovered / 1 newly damaged / 4 neither = 40**).
  AB_REF_A's rate (33/40, 82.5%) and net improvement (+2) were already
  correct and are unaffected.
- **C14**: removes an invented attribution in C7 (an unverified "outside
  characterization" that case 30 was "comparable" to cases 29/33),
  replacing it with this reviewer's own plain uncertainty about case 30,
  not a rebuttal to any specific prior claim.
- **C15**: extends C8 — cases 32 *and* 33 (not just 33) are both the
  "flying golf ball" literal-prompt family (`scene_index` 4), distinct
  from cases 28–31's "hand gripping" paraphrase family (`scene_index` 3).
- **C16**: the correction in §1 above (full text in `CORRECTIONS.md`).

## 3. Numerical-optimality audit

### 3.1 Objective derivation, confirmed against the pinned source

Read `erase_methods/uce.py :: edit_model_adversarial` directly from the
pinned RECE repo (`/data/.../repo_src/RECE`, verified at commit
`403a67fd86dce6c124b88798340e001e625b28c9`, clean working tree). For a
layer with pre-edit weight `W` ([d_out,768]), supplied embedding `X`
([77,768], `old_embeddings[0]`), guide `G` ([77,768], `new_embeddings[0]`,
text `" "`), retain `R` ([77,768], text `""`):

```
mat2 (-> H) = lamb*I + erase_scale*(context^T @ context)       with context = X
mat1 (-> N) = lamb*W + erase_scale*((layer(new_emb))^T @ context)   with layer(new_emb) = W @ G (technique='replace')
            + preserve_scale*((layer(retain_new))^T @ context_retain)   with context_retain = retain_new = R (old_text==new_text=="")
new_weight  = mat1 @ inverse(mat2)
```

Algebraically this is `V = N @ H^-1` where `H = λI + erase·XᵀX +
preserve·RᵀR` and `N = λW + erase·W(GᵀX) + preserve·W(RᵀR)` — exactly the
normal equations for

```
J(V) = λ‖V−W‖²_F + erase‖VXᵀ−WGᵀ‖²_F + preserve‖VRᵀ−WRᵀ‖²_F
```

(confirmed by hand: `dJ/dV = 0` reduces to `V·H = N`). **No discrepancy
between the pinned source and the task's stated equations was found.**

- **Why each layer uses its own pre-edit W despite sequential traversal**:
  the 32 layers are 32 disjoint weight tensors (16 `to_v` + 16 `to_k` of
  different cross-attention modules); the outer loop only overwrites
  `projection_matrices[layer_num]` at the end of that same index's own
  iteration, so no layer's weight is touched by another layer's update
  within one call — not an ordering safeguard, just non-aliasing.
- **Shapes/batching/token sums**: `old_embeddings`/`new_embeddings` are
  single-element lists (`[adv_embedding[0]]`, `[new_embedding[0]]`), so
  the erase accumulation loop runs once; `retain_texts=[""]` zipped with
  itself likewise runs once. All three embeddings are the **full,
  untrimmed 77-token** tensor (Stage 2's own convention — distinct from
  Stage 1's suffix slice, C5/C11, not relevant here).
- **Projection bias**: the expected-32-key whitelist
  (`embedding_cross_swap/VERIFICATION_LOG.json`) contains only
  `to_k.weight`/`to_v.weight` entries, no `.bias` keys, and the native
  pipeline's own enforced changed-key assertion passed exactly against
  this whitelist for every cell — confirming `to_k`/`to_v` carry no bias
  term in this architecture, consistent with the objective as stated
  (no bias term anywhere in `J(V)`).
- **Retain-embedding batching, disclosed not re-verified**: the saved
  `retain_embedding.pt` was computed as a single-item tokenizer batch;
  the native call batches `["", ""]` together (batch of 2, since
  `old_text==new_text==""`). CLIP's text transformer is per-sequence
  (causal mask, per-sample LayerNorm), so these are *expected* to be
  bit-identical, but this was **not independently re-verified** here —
  per task scope, text encoders are not recomputed in this audit.

### 3.2 Method

fp32 tensors loaded from the saved checkpoints/embeddings, upcast to fp64
**before** forming H/N (never casting an already-rounded fp32 Gram
matrix). `H` depends only on the cell's `X` and the fixed `R` — identical
across all 32 layers in a cell — so it is Cholesky-factored **once per
cell** (2 distinct H's across the 4 cells, since E_D and E_S each define
one) and the factor reused for all 32 per-layer solves of `H·V*ᵀ = Nᵀ`.
No explicit inverse, no jitter, no truncation, no CUDA (`CUDA_VISIBLE_
DEVICES=""` set in-process; `torch.cuda.is_available()` asserted False
before any computation). All 10 required inputs were re-verified by
`sha256sum` against the frozen manifest at run time — live recheck, not
reuse of previously logged hashes — and matched exactly; no missing or
mismatched input, so no scoped stop was needed. Script, frozen manifest
(hashes/equations/scope written **before** any computation), and full
per-layer/per-cell `RESULT.json` are published alongside this report.

### 3.3 H diagnostics (per cell, 768×768)

| cell (embedding) | finite | symmetry error | min eig | max eig | cond(H) | Cholesky |
|---|---|---|---|---|---|---|
| E_D cells (B, S_WITH_DIRECT_EMB) | yes | 0.0 | 0.09999999999 | 1.00458e5 | **1.00458e6** | OK |
| E_S cells (D_WITH_SEQ_EMB, AtoB) | yes | 0.0 | 0.09999999989 | 1.60710e6 | **1.60710e7** | OK |

`cond(H)` for E_S-based cells is **~16× worse** than for E_D-based cells
— matching `‖E_S‖²/‖E_D‖² ≈ (1285.17/329.54)² ≈ 15.2`, since `H`'s
dominant term for erase_scale=1 is `XᵀX`. Both are positive-definite
(min eig ≈ λ = 0.1 exactly, as expected since `XᵀX`/`RᵀR` are PSD and
`λI` sets the floor); Cholesky succeeded for both, so the reference solve
is well-posed for all four cells.

### 3.4 Reference-solve validation (frozen before interpretation)

All 4 cells × 32 layers = 128 reference solves: **normalized backward
error for V\*** ranged from 2.7e-17 to 5.6e-17 — at fp64 machine-epsilon
scale, **≤ the frozen 1e-10 tolerance for every single layer**. The
quadratic-gap identity `J(V)-J(V*) = tr((V-V*)H(V-V*)ᵀ)` held to within
the frozen `1e-8×max(1,|J(V)|,|J(V*)|)` tolerance for every layer in every
cell. **Reference solve fully valid for all four cells — no unresolved
diagnostic, no cell excluded.**

### 3.5 Native vs. reference, per cell (aggregated over 32 layers)

| cell | Σ J(W) | Σ J(V_native) | Σ J(V\*) | Σ excess (V_native−V\*) | median / max backward err (native) | ‖V_native−V\*‖ median rel. | Σ output-rounding scale |
|---|---|---|---|---|---|---|---|
| B (W_D,E_D) | 2.049e6 | 663.4 | 184.1 | **479.3** | 3.1e-4 / 6.8e-4 | **0.97%** | 3.4e-9 |
| D_WITH_SEQ_EMB (W_D,E_S) | 1.595e8 | 163402 | 174.9 | **163227** | 1.6e-3 / 2.8e-3 | **12.1%** | 4.8e-8 |
| S_WITH_DIRECT_EMB (W_S,E_D) | 1.650e6 | 596.9 | 135.3 | **461.7** | 3.0e-4 / 6.8e-4 | **0.96%** | 3.4e-9 |
| AtoB (W_S,E_S) | 2.755e6 | 161688 | 97.4 | **161591** | 1.6e-3 / 2.7e-3 | **12.0%** | 5.1e-8 |

The excess objective is **11–13 orders of magnitude larger than the
output-rounding floor** (Σ excess / Σ output-rounding-scale ≈ 1.4e11 for
E_D cells, ≈ 3.3–3.4e12 for E_S cells) — this is not explained by simply
rounding the correct answer to fp32. **Native outputs differ materially
from V\* in every one of the four cells**, under the frozen decision rule
(§4 of the manifest).

**The severity tracks the embedding (E_D vs E_S), cleanly, by column —
but this is a numerical-accuracy pattern, not the image-quality
pattern from §1.** E_S-based cells carry ≈340× larger excess objective,
≈16× worse `cond(H)`, and ≈12.5× larger median forward relative error
than E_D-based cells, **in both rows** — i.e., both E_S cells (D_WITH_
SEQ_EMB *and* AtoB) are similarly numerically inaccurate relative to
their own V\*, even though only D_WITH_SEQ_EMB is visually corrupted and
AtoB is not. **Numerical accuracy and visual outcome do not coincide
here**: the (W_D,E_S) vs (W_S,E_S) image-quality split from §1 is not
explained by this audit's solver-accuracy numbers, which are nearly
identical between those two cells (163227 vs 161591 excess; 12.1% vs
12.0% median relative forward error). Worst single-layer excess ratios
reach ~7.8×–3875× `J(V*)` (always the smallest, 320-row `to_v` layers);
full per-layer table in `RESULT.json`.

### 3.6 Bounded scientific decision

Per the frozen decision rule: **native outputs differ materially from
the accurate optimum, with excess objective far beyond output-rounding
scale, in all four cells** — the native closed-form solver (float32
accumulation, explicit `torch.inverse`, no stabilization) does **not**
accurately solve its own stated regularized objective. This makes
solver numerical accuracy a **concrete candidate mechanism** worth a
separately authorized image-level test (not run here). At the same time:

- This is a **column (embedding) effect at the level of numerical
  accuracy** (driven by `cond(H)` tracking `‖X‖²`), cleanly separate from
  the **single-cell, direction-specific effect at the level of image
  quality** documented in §1. The two patterns do not line up — the
  numerical finding does not, by itself, explain why (W_S,E_S) stays
  visually clean while (W_D,E_S) does not.
- Backward error alone (the ~1e-3-scale native residual) would
  understate this: it looks "small," but `cond(H)` is poor enough (up to
  1.6e7) that forward error is amplified roughly in proportion —
  forward discrepancy and the objective gap, not backward error alone,
  are what make this material, consistent with the task's framing.
- Agreement/disagreement here says nothing about the separate
  `close_form_emb_regzero` embedding-recovery computation that produced
  E_D/E_S in the first place — not tested by this audit.
- No image-level claim follows from this: a numerical discrepancy does
  not establish that an accurate solve would change (or not change) any
  generated image. That is a distinct, separately authorized
  intervention, not run here.

## 4. One next scientific decision (not executed)

**Test whether the solver-accuracy gap is causally connected to the
image-level pattern**, by computing V\* for just the four cells already
audited here and comparing *only the quality-relevant layers'*
displacement from native V (no new checkpoint assembly, no image
generation) against where in the network the (W_D,E_S)-specific
corruption is known to concentrate (spatial resolution / up-block index)
— this could show whether the numerically worst layers are the same
layers already implicated by `embedding_cross_swap/PER_LAYER_DISPLACEMENT.
json`, or a different set entirely, which would argue for or against
numerical accuracy as part of the mechanism. **Not run here**: it would
require cross-referencing against the image-level diagnostics in a way
this task's scope does not authorize, and any follow-up image generation
to actually test the hypothesis is explicitly out of scope.

## 5. Resource usage

CPU-only throughout. `torch.set_num_threads(8)`; `CUDA_VISIBLE_DEVICES=""`
set before importing torch; `torch.cuda.is_available()` asserted `False`
before any computation (script would have aborted otherwise). Six UNet
checkpoints (~3.4GB fp32 each) loaded sequentially on a 503GB-RAM host,
freed between cells; peak usage well within available memory (no swap
touched). Total wall-clock time: **33.4s** of audit computation
(hash-verification + loading + solving + writing results), measured by
the script itself; **35.8s** real time for the full process including
interpreter startup. **No GPU was allocated at any point in this task.
No image was generated.**

## 6. Artifacts

`FROZEN_MANIFEST.json` (written before any computation), `run_
numerical_optimality_audit.py`, `RESULT.json` (full 4-cell ×
32-layer tables: H diagnostics, J components, backward errors, forward
errors, quadratic-gap identity checks, output-rounding comparison, and
the top-level `input_verification_recheck`/timing fields — every layer
published, not just aggregates), `CASE_LEVEL_COMPLETION_
RECORD.json`, this `FINAL_REPORT.md`. The raw console transcript
(`numerical_optimality_audit/run.log` on the host) is not published,
per this repository's existing convention of not vendoring raw `.out`/
`.log` run transcripts elsewhere in this results line — its substantive
content (input-hash recheck, H diagnostics, per-cell timings) is in
`RESULT.json`. `CORRECTIONS.md` (repo root of this
results directory) gained C13–C16.

---

**This closes the bounded assignment as scoped: no golf-subspace search,
no norm scaling, no alternative reference, no new concept, no stable-solve
image generation, no automatic follow-up. No replacement model checkpoint
was assembled from any diagnostic per-layer solution. Stopping here.**
