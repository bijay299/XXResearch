# Recovery-optimality audit: does the saved embedding-recovery output (`E_D`/`E_S`) accurately solve its own stated objective?

CPU-only. No GPU allocated, no image generated, no model edited, no
embedding recovered on GPU, no text encoded. This closes the "next
scientific decision" explicitly left open by both prior audits in this
line (`numerical_optimality_audit/`'s S4 and `solver_only_intervention/
FINAL_REPORT.md`'s S7): whether `close_form_emb_regzero` -- the
closed-form solve that produced `E_D` and `E_S` in the first place, a
different computation from the Stage-2 refinement solve audited
previously -- is itself solved accurately on the native GPU/fp32 path.
Alongside it, this task closes a bounded set of existing-evidence
corrections (`CORRECTIONS.md` C17-C22), the most consequential of which
withdraws a previously published "5 recovered" headline.

## 1. Preserved state

Starting point: commit `cbff93656ba95aa7ec89365cf0c293645a949e77`
(`solver_only_intervention`: `AB_STABLE_SOLVE`). Eligibility, baseline
searches, parent confirmation, Stage-1 reconstruction, both diagonal
replays, the downstream erase-optimality audit
(`numerical_optimality_audit/`), and `AB_STABLE_SOLVE` are closed and not
repeated. Established going in: accurate downstream (Stage-2) solving
preserves golf recognition at 0/40 and springer at 0/20, while retained
recognition falls (parachute 18->17, horn 19->16); golf-ball corruption
remains the majority outcome even under a verified fp64-accurate Stage-2
solve. The reported 35->30 degeneration improvement and its "5 recovered"
cases were flagged as disputed going into this task (case 38) -- resolved
in S5/C17 below, not carried forward as settled.

## 2. Scope, inputs, and provenance

All five supplied tensor hashes verified **live** against the actual
files (not reused from a prior log):

| tensor | path | sha256 | match |
|---|---|---|---|
| W_D | `refinement_ablation/edits/B_UCE.pt` | `23b69146...61bc4` | MATCH |
| W_S | `refinement_ablation/edits/AB_UCE.pt` | `3a9d86de...62b550` | MATCH |
| E_D | `embedding_cross_swap/edits/E_D.pt` | `32d1bd98...ebd4c3b` | MATCH |
| E_S | `embedding_cross_swap/edits/E_S.pt` | `e6ea282a...94af89bdb8` | MATCH |
| C (concept_embedding) | `embedding_cross_swap/edits/concept_embedding.pt` | `2e5ab9da...e14a4e5c6` | MATCH |

**M0 provenance** (no hash was supplied for M0; none had been previously
published anywhere in this line -- `CORRECTIONS.md` C22). Current sha256
of `embedding_cross_swap/edits/M0_unet_state_dict.pt`:
`ee60d14bf80c6f829ab1b9da5f6c0eb63338f8a29d7ec7f6a3895bae1b1611df`,
disclosed here as newly recorded. Save-code provenance confirmed by
reading `embedding_cross_swap/run_embedding_cross_swap.py` directly: it
is the untouched `.state_dict()` of a freshly-instantiated
`StableDiffusionPipeline.from_pretrained` load of the pinned SD1.4
snapshot (`133a221b8aa7292a167afc5127cb63fb5005638b`), detached/cloned/
cpu'd and saved **before** any edit is applied to that pipeline object.
Independently cross-checked (CPU-only, no pipeline, no inference: the
pinned snapshot's `unet/diffusion_pytorch_model.safetensors` read
directly) -- all 32 audited K/V projections are **bit-identical** between
`M0_unet_state_dict.pt` and the pinned snapshot file. M0's provenance for
this audit's purposes is fully established.

**Equations confirmed against the pinned source** (RECE repo at
`403a67fd86dce6c124b88798340e001e625b28c9`, `utils/embedding_calculation/
close_form_emb.py :: close_form_emb_regzero`, clean working tree):

```
mat1 (-> A) = r*I + sum_l W_l^T W_l            (W_l = working checkpoint's layer)
mat2 (-> B) = sum_l W_l^T W0_l                 (W0_l = M0's corresponding layer)
coefficient = inverse(mat1) @ mat2
adv_embedding = concept_embedding @ coefficient.T
```

Transposing (`mat1` symmetric): `adv_embedding^T = inverse(mat1) @ mat2 @
concept_embedding^T` -- i.e. with `A=mat1`, `B=mat2`, `D=B@C^T`,
`Y=adv_embedding`: **`A @ Y^T = D` exactly**, matching the task's stated
normal-equation form with no discrepancy. Confirmed call site
(`train_object.py` line 310) uses `with_to_k=True` (hardcoded at the call
site, not the function's own default) and `regular_scale=0.1` (confirmed
via `RUN_MANIFEST.md`'s directly-published `train_object.py` command,
`--regular_scale 1e-1`, and `embedding_cross_swap/FROZEN_MANIFEST.json`'s
own recorded recovery parameters) -- all **32** K/V projections, full
**77**-token embeddings (no suffix slice -- the slice is commented out in
the pinned source), M0 as `model_copy` in both branches (not A). This is
a materially different computation from the Stage-2 refinement audited
previously: `close_form_emb_regzero` has **no retain/preserve term at
all** (only the zero-regularizer and erase/B term), unlike `edit_model_
adversarial`'s three-term Stage-2 objective.

**32 audited layers**, identified structurally (the key set that differs
between `M0` and `W_D`, cross-checked identical to the `M0`-vs-`W_S` key
set): 16 `to_v.weight` + 16 `to_k.weight`. Shapes match between the
working checkpoint and M0 at every key; all 32x2 weight tensors plus `C`,
`E_D`, `E_S` are finite. Bias check (structural, not a whitelist): no
`.bias` key exists for any of the 32 audited `.weight` keys in any of the
three state dicts -- confirms no bias term, consistent with the objective
as stated.

**CPU lockdown**: `CUDA_VISIBLE_DEVICES=""` set before importing torch;
`torch.cuda.is_available()` asserted `False` before any computation;
`torch.set_num_threads(8)`. No `diffusers` import anywhere in this
script -- the UNet is never instantiated as a module; all three weight
sets are read as raw `state_dict` tensors and the 32 layers are the raw
`.weight` entries at matching keys (order of summation does not affect
`A`/`B`, which are permutation-invariant sums over `l`).

## 3. Validation (frozen tolerances, both branches)

| | branch D (W_D, M0, E_D) | branch S (W_S, M0, E_S) |
|---|---|---|
| min eig(A) | 0.100082 | 0.100072 |
| max eig(A) | 680.04 | 1392.06 |
| cond(A) | **6.79e3** | **1.39e4** |
| reference normalized backward error | 1.08e-16 | 4.14e-17 |
| tolerance (<=1e-10) | PASS | PASS |
| quadratic-gap discrepancy | 4.36e-10 | 3.69e-09 |
| quadratic-gap tolerance | 1.31e-02 | 4.50e-03 |
| quadratic-gap identity | PASS | PASS |

**Reference solve fully valid for both branches** -- no layer excluded,
no tolerance relaxed. `cond(A)` here (6.8e3-1.4e4) is **far better
conditioned** than the Stage-2 refinement's `cond(H)` audited previously
(1.0e6-1.6e7) -- `A`'s dominant term sums `W_l^T W_l` over 32 already-
edited UNet projection matrices (not a small token-embedding Gram
matrix), which is intrinsically better scaled.

## 4. Native vs. reference, per branch

| | branch D | branch S |
|---|---|---|
| F(native E) | 1,312,447.4348 | 449,976.3308 |
| F(Y\*) | 1,312,447.4339 | 449,976.3282 |
| native backward error | **1.14e-06** | **3.26e-07** |
| abs forward discrepancy \|\|E-Y\*\|\| | 0.01395 | 0.04831 |
| rel forward discrepancy | **4.23e-05 (0.0042%)** | **3.76e-05 (0.0038%)** |
| \|\|native E\|\| | 329.5411 | 1285.1685 |
| \|\|Y\*\|\| | 329.5436 | 1285.1851 |
| magnitude ratio (native/Y\*) | 0.999993 | 0.999987 |
| cosine(native, Y\*) | 0.99999999913 | 0.99999999938 |
| angle(native, Y\*) | **0.00239 deg** | **0.00202 deg** |
| per-token abs diff (min/mean/max) | 7.4e-4 / 1.50e-3 / 5.6e-3 | 2.6e-3 / 5.49e-3 / 6.6e-3 |
| fp32-cast-of-Y\* backward error (rounding floor, not held to 1e-10) | 2.30e-09 | 1.59e-09 |
| fp32-cast objective gap vs F(Y\*) | 3.96e-09 | 5.40e-08 |

Native's own backward error (1.1e-6 / 3.3e-7) is **~500x / ~200x** the
fp32-storage-rounding floor (2.3e-9 / 1.6e-9) -- so there is a small,
genuine solve-accuracy gap beyond pure output rounding (the native path
uses `torch.inverse` in fp32, not an fp64 Cholesky), consistent with
`CORRECTIONS.md` C19's "assembly-and-solve accuracy" framing. But in
absolute and forward terms this gap is tiny: **relative forward
discrepancy is ~0.004% and the angle between the native and accurate
embeddings is ~0.002 degrees, in both branches.** This is three to four
orders of magnitude smaller than the Stage-2 refinement's own native/
reference gap audited previously (12% median relative forward error,
0.93e-3-2.74e-3 backward error) -- **the embedding-recovery solve is
materially more accurate, in both absolute and relative terms, than the
downstream refinement solve that consumes its output.**

## 5. Direct-vs-sequential contrast, before and after accurate recovery

| | native (E_D vs E_S) | accurate (Y\*_D vs Y\*_S) | shift |
|---|---|---|---|
| \|\|D\|\|/\|\|S\|\| norm ratio | 0.256419 | 0.256417 | -1.38e-06 |
| cosine | -0.067502 | -0.067506 | -3.9e-06 |
| angle (deg) | 93.8705 | 93.8707 | **+0.00022** |

**The large direct/sequential embedding contrast (near-orthogonal
direction, ~3.9x magnitude difference) survives accurate recovery
essentially unchanged** -- the shift in both norm ratio and angle is six
orders of magnitude smaller than the contrast itself. This is a
numerical-accuracy result about the saved embeddings, not an image-level
claim: no image was generated from either `Y*_D` or `Y*_S`, and no
inference is drawn here about why this directional contrast exists.

## 6. One scientific decision

Per the frozen decision rule: **the saved native recovery is close to
the validated fp64 optimum (relative forward discrepancy ~0.004%, angle
~0.002 degrees, in both branches), and the large direct/sequential
embedding contrast survives accurate recovery essentially unchanged
(shift of ~2e-4 degrees against a ~94-degree angle).** Per the task's own
framing, this **closes a large recovery-solve numerical error as the
leading explanation** for the direct/sequential contrast and for
whatever residual failure in the sequential endpoint is not already
explained by the Stage-2 refinement audit. The embedding-recovery step
itself is not where a substantial numerical error lives in this
pipeline.

This does **not** mean the recovery solve is perfect, or that numerical
accuracy is irrelevant generally: native's backward error is real and
~200-500x the output-rounding floor (S3/S4 above), and the Stage-2
refinement solve (audited separately) *is* materially inaccurate at the
12%-relative-error scale. It means specifically that **this** computation,
at **this** scale, is not a plausible source of a large numerical error
for the direct/sequential contrast.

**Recommended next scientific action** (not executed here, per scope):
investigate the consequences of the *accurately recovered* constraints in
the sequential state -- i.e., treat `Y*_S` (or, equivalently, the
already-accurate native `E_S`) as a legitimate, correctly-computed
solution to `close_form_emb_regzero`'s own objective when the working
checkpoint is the sequential branch's already-once-edited `W_S`, and ask
what makes that correctly-computed solution qualitatively different
(near-orthogonal, ~3.9x larger norm) from the direct branch's correctly-
computed `Y*_D` -- a question about what the objective *asks for* given
`W_S`'s prior-edit history, not about how accurately it is solved. A
controlled test of this (e.g. an authorized, separately scoped image-
level intervention using the now-validated `Y*_S`/`Y*_D` reference
tensors saved in `diagnostics/`) is a reasonable next step but is **not**
authorized or run by this task.

Numerical accuracy does not prove semantic correctness, and numerical
accuracy here does not itself establish that any generated image would
be unaffected by this finding -- no image was generated from `Y*_D` or
`Y*_S`. Neither outcome of this audit establishes a new method.

## 7. Existing-evidence corrections (alongside the audit, no regeneration)

Full detail in `CORRECTIONS.md` C17-C22; summary:

- **C17 (most consequential)**: `solver_only_intervention/FINAL_REPORT.md`
  S4.2's "30/40 degenerate, 5 recovered (28,29,30,31,38)" is corrected to
  **31/40 degenerate, 4 recovered (28,29,30,31)** -- case 38, viewed at
  full 512x512 resolution this session, shows the same pervasive mesh/
  sparkle corruption signature this line uses as its own whole-frame
  degeneration criterion, not the milder non-mesh family. Borderline
  cases 15, 24, 28 were re-inspected at full resolution for consistency
  and keep their existing non-degenerate labels. The "35->30, five
  recoveries" headline is withdrawn; "35->31, four recoveries" replaces
  it. Every qualitative conclusion in that report's S4.2/S4.4/S6 survives
  this correction, slightly strengthened (corrected table:
  `CASE_LEVEL_AB_STABLE_SOLVE_GOLF_CORRECTED.json`).
- **C18**: the solver-only report's fp32-storage-rounding figure is
  corrected from a mislabeled "~5e-8 relative" to the precise
  **5.0418e-08 absolute / 5.1770e-10 relative** (recomputed exactly from
  already-published per-layer `RESULT.json` values -- no new
  computation). No conclusion is affected.
- **C19**: "fp64 Cholesky solve replacing `torch.inverse`" is reframed as
  **assembly-and-solve accuracy** (fp64 upcast-before-accumulate plus a
  Cholesky solve), not inversion accuracy in isolation -- matches what
  both audits actually measure and already disclose (neither separates
  accumulation/inversion/multiplication error).
- **C20**: stated plainly, line-wide: classification and aggregate
  inspection preceded visual review for every endpoint in this results
  line, not a subset -- a departure from the frozen blinding instruction
  that cannot be repaired retroactively. All visual-review labels in this
  line, old and new, are prediction-informed retrospective review.
- **C21**: two additional case rows (`D_WITH_SEQ_EMB` cases 38, 87) were
  viewed at genuine full 512x512 resolution this session and reconfirm
  (do not change) that cell's existing 5/40 golf / 20/20 springer
  degeneration counts. The remaining ~104 of the 120-row cross-swap
  record's rows were **not** re-reviewed at full resolution in this
  bounded session -- their thumbnail-based labels stand, undisputed but
  not upgraded; the record's own per-row `resolution_viewed` field
  remains the accurate account of what was actually viewed and at what
  resolution.
- **C22**: M0 provenance (S2 above) and a statement that no further
  operand-provenance work (no text-encoding check, no endpoint build, no
  replay) was run beyond what already existed in prior reports.

## 8. Resources and release

CPU-only throughout, **no GPU allocated at any point in this task**.
`torch.set_num_threads(8)`; `CUDA_VISIBLE_DEVICES=""` set before
importing torch; `torch.cuda.is_available()` asserted `False` before any
computation. Three UNet-sized fp32 state dicts (`W_D`, `W_S`, `M0`, ~3.4GB
each) loaded sequentially on a 503GB-RAM host; peak usage well within
available memory. Total audit wall-clock time: **18.1s** (script-measured
compute) / **19.5s** real time including interpreter startup. **No image
was generated. No model checkpoint was built or edited. No embedding was
recovered on GPU or via text encoding -- the two fp64 reference solutions
(`Y*_D`, `Y*_S`) are diagnostic-only, saved locally with hashes, never
applied to any model.**

## 9. Artifacts

`FROZEN_MANIFEST.json` (written before any computation), `run_recovery_
optimality_audit.py`, `run.log`, `RESULT.json` (full per-branch tables:
hash verification, M0 provenance/per-key comparison, shape/finiteness/
layer-correspondence checks, H/A diagnostics, objective components,
backward errors, quadratic-gap identity, per-token discrepancies, direct-
vs-sequential contrast, diagnostic-tensor hashes), `diagnostics/
Y_STAR_{D,S}_fp64.pt` and `Y_STAR_{D,S}_fp32cast.pt` (diagnostic-only
reference tensors, hashes in `RESULT.json`, not published to GitHub per
this repository's checkpoint-exclusion convention -- shape/dtype/sha256
published instead), `CASE_LEVEL_AB_STABLE_SOLVE_GOLF_CORRECTED.json`,
this `FINAL_REPORT.md`. `CORRECTIONS.md` (repo root of this results
directory) gained C17-C22.

---

**No second endpoint, baseline repeat, image generation, model editing,
tuning, layer search, new concept pair, or automatic follow-up was run.
Finishing this bounded assignment here.**
