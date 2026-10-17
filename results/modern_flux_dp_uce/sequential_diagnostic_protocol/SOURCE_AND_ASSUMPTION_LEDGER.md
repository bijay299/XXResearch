# Source and assumption ledger — chain saw -> parachute diagnostic protocol

Every claim below is tagged **ESTABLISHED** (directly inspected source, or
arithmetic on an already-published table, both reproducible without running
anything), **UNESTABLISHED** (a real gap, named rather than assumed away),
or **PREDICTION** (the untested empirical claim this protocol exists to
test). Nothing in this document required loading a model, running an
encoder, or generating/classifying an image.

## Source inspected (all already present on `/data`, cloned at the pinned commit for the completed reproduction — not re-cloned)

- `czhang024/CVPR_DouleProjection` @ `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`, `Flux/Double_proxy.py`, `Flux/UCE_original.py`, `Flux/generate_erased_flux.py`, `Flux/scripts/*.sh` — paths under `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo/`.
- Paper: `https://arxiv.org/abs/2604.10032` (DP/UCE-FLUX release). Not re-fetched in this task; cited by ID only, as already recorded in `../RUN_MANIFEST.md`.
- UCE's original formula: `https://arxiv.org/abs/2308.14761` (cited directly in `UCE_original.py`'s own code comment, line 85).

## ESTABLISHED (proved from source or from already-published tables)

1. Both edit scripts' `original_modules` is a fresh `copy.deepcopy` of
   `pipe.transformer`'s two targeted modules, loaded from whatever
   checkpoint `model_id`/the loaded pipeline currently holds, at the start
   of the function call (`Double_proxy.py:34-54`, `UCE_original.py:15-30`).
   A parent-loading adapter inserted before this point is therefore
   sufficient, by construction, to make every downstream reference/guide/
   preserve computation operate on the parent's edited weights rather than
   pristine M0's — the same mechanism already used by
   `generate_erased_flux.py:61-81` in the completed reproduction to merge a
   saved 2-key `.safetensors` into a freshly loaded pipeline.
2. Both edit scripts modify exactly two global weight matrices
   (`context_embedder.weight`, shape `[3072, 4096]`; `time_text_embed.text_embedder.linear_1.weight`,
   shape `[3072, 768]`), selected by the identical substring match
   (`Double_proxy.py:48`, `UCE_original.py:26`) regardless of which concept
   is being edited — confirmed both by source inspection and empirically in
   the completed reproduction (4/4 single edits matched exactly these 2
   keys, `../RUN_MANIFEST.md`). This is the structural precondition for any
   cross-concept interference: two edits to different concepts necessarily
   write to the identical parameter tensors.
3. DP's preserve constraint is an exact null-space projection
   (`Double_proxy.py:191-233`): `ΔW = Z @ U_2.T` where `U_2` spans the left
   null-space of the stacked preserve-concept embedding vectors, computed by
   SVD. By construction, `ΔW @ c_i = 0` exactly (to float32 precision) for
   every preserve-concept vector `c_i`. Given a loaded parent `W_0` and
   chain saw included in `preserve_concepts`, `(W_0 + ΔW_B) @ c_{chain saw}
   = W_0 @ c_{chain saw}` exactly, for the isolated string `"chain saw"`'s
   own token embeddings as encoded by `pipe.encode_prompt(prompt="chain saw", ...)`.
4. UCE's preserve term is a *soft*, weighted least-squares trade-off, not an
   exact constraint (`UCE_original.py:85-113`): `W_new = mat1 @
   inverse(mat2)`, where `mat1`/`mat2` sum weighted outer products over
   every erase and preserve concept plus an `λ·w_old` regularizer. The
   preserve term's target is explicitly `w_old`'s own output (so it *is*
   trying to match the parent's response for chain saw), but as one term in
   a single shared optimum — nothing in the closed form guarantees exact
   (or even bounded) preservation. This asymmetry with DP is native to the
   released code, not introduced by this protocol.
5. DP's native multi-concept loop (`Double_proxy.py:153-235`) is genuinely
   sequential/ordered even within a single call: each concept's update
   modifies `W_0_f32` in place before the next concept's own baseline (`b =
   W_0_f32 @ (...)`) is computed from the updated matrix.
6. UCE's native multi-concept loop (`UCE_original.py:96-113`) is a true
   simultaneous closed-form solve even within a single call: `mat1`/`mat2`
   are built from commutative sums over every listed concept and solved
   once — list order does not change the result. This is mechanistically
   different from DP's loop and must not be described the same way.
7. The seven candidate-B recognition counts in `PROTOCOL.md`'s table were
   independently recomputed from `evidence/recognition_summary.json`
   (sha256 `0dd43f81ed98fe53eecb26b72b0e423a643831df687fc9466ded3e2cedaa96d4`,
   commit `01f8832`) and match the assignment's supplied table exactly —
   verified, not merely copied.
8. `Flux/scripts/` contains ten released `Double_<Class>_flux.sh` examples
   (one per class) but only two released `UCE_<Class>_flux.sh` examples
   (cassette player, chain saw) — directly confirmed by directory listing,
   not inferred.
9. Parachute never appears as an erasure or historical target anywhere in
   the closed `results/rece_seq_golfball_englishspringer/` line — confirmed
   by grep across that line's `README.md` and `FINAL_REPORT.md` files, which
   name golf ball and English springer as the original pair and French horn
   as S1's erased new-target; parachute appears only in retention-panel
   mentions ("French horn/parachute stayed clean").

## UNESTABLISHED (named gaps, not assumed away)

10. Whether DP's exact null-space guarantee for the *isolated* string
    "chain saw" transfers, even approximately, to the *contextual* in-prompt
    token embeddings of "chain saw" inside the real evaluation prompt
    `"Image of chain saw"`. FLUX's T5 encoder is contextual (full
    self-attention over the whole prompt), so the two embeddings are not
    generally identical. No toy closed-form bound was derived for this gap
    in this protocol.
11. Whether UCE's soft preservation term keeps chain-saw's output
    sufficiently close to the parent's own response in practice — not
    bounded analytically here, left as the empirical question SEQ execution
    would measure.
12. Whether a positive classifier reading of "chain saw" after a B-edit
    reflects actual chain-saw content or a different object entirely (the
    completed reproduction's own chain-on-a-stick UCE_B spot-check shows the
    classifier can correctly *reject* a visually chain-adjacent image) —
    `PROTOCOL.md ## 9`'s verification rule exists precisely because this is
    unresolved, not because it is assumed to be reliable either way.

## PREDICTION (the untested claim this protocol is designed to test)

13. SEQ (both methods) will show some chain-saw reappearance beyond the
    completed reproduction's own M0-vs-parent noise floor, with DP's
    reappearance (if any) smaller than UCE's — motivated by established
    claim 3 (DP's exact guarantee) vs. established claim 4 (UCE's soft
    trade-off), but not proved, because of unestablished claims 10-11. This
    is exactly what `PROTOCOL.md ## 1`'s rejection criterion is built to
    falsify.

## CEASE and BLOCK — acknowledged prior art, not re-investigated here

- **CEASE** (`yongliang-wu/CEASE`, "Continual concept erasure in diffusion
  models by suppressing cross-edit interference") targets this exact
  mechanism by name. As last inspected (`docs/research/feasibility_sources_2026-10-11.json`,
  fetched 2026-10-11T05:13:57Z): repository tree contains no method code
  (`.nojekyll`, `README.md`, `assets/`, `index.html`, `style.css` only;
  description states "Code coming soon"; `contains_method_code: false`).
  Cited as directly relevant prior art for the predicted mechanism; its
  unavailability as an implementation is unchanged and was not re-checked
  by fetching anything new in this protocol-only task.
- **BLOCK** (`https://proceedings.mlr.press/v306/kumar26c.html`,
  `sprintml/block-concept-removal` @
  `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5`) already studies sequential
  removal on modern models, as recorded in the completed reproduction's
  `../FINAL_REPORT.md`. A bounded deficit found by this protocol, if any,
  would be a narrow DP/UCE-specific finding on one concept pair — it would
  not establish that BLOCK fails, and does not make sequential erasure a
  new research topic. No additional model or baseline search is proposed.

## What this ledger deliberately does not contain

No RECE/SD1.x mechanism claims (that line is closed and unrelated — DP/UCE's
FLUX edit targets two transformer linear layers, not SD1.x's UNet
cross-attention K/V projections RECE targets). No claim that Gemini's
proposed statistical protocol was evaluated or used; it was not read or
implemented here. No claim of novelty for the sequential-interference
question itself (CEASE and BLOCK already motivate it in the literature).
