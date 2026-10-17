# Source and assumption ledger — chain saw -> parachute diagnostic protocol (revision 01)

Every claim below is tagged **ESTABLISHED** (directly inspected source, or
arithmetic on an already-published table or on hypothetical numbers,
reproducible without running anything), **UNESTABLISHED** (a real gap,
named rather than assumed away), or **PREDICTION** (the untested empirical
claim this protocol exists to test). Nothing in this document required
loading a model, running an encoder, or generating/classifying an image.
Two existing checkpoint files were hashed at the file level only (no
tensor/safetensors deserialization) — see item 14.

This revision corrects the first published ledger per the coordinator's
NEEDS_CHANGES review; see `REVISION_01.md` for the full mapping.

## Source inspected (all already present on `/data`, cloned at the pinned commit for the completed reproduction — not re-cloned)

- `czhang024/CVPR_DouleProjection` @ `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`, `Flux/Double_proxy.py`, `Flux/UCE_original.py`, `Flux/generate_erased_flux.py`, `Flux/scripts/*.sh` — paths under `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo/`.
- Paper: `https://arxiv.org/abs/2604.10032` (DP/UCE-FLUX release). Not re-fetched; cited by ID only.
- UCE's original formula: `https://arxiv.org/abs/2308.14761` (cited directly in `UCE_original.py`'s own code comment, line 85).

## ESTABLISHED (proved from source, from already-published tables, or verified hand arithmetic)

1. Both edit scripts' `original_modules` is a fresh `copy.deepcopy` of
   `pipe.transformer`'s two targeted modules, loaded from whatever
   checkpoint is currently held, at the start of the function call
   (`Double_proxy.py:34-54`, `UCE_original.py:15-30`). A parent-loading
   adapter inserted before this point makes every downstream reference/
   guide/preserve computation operate on the parent's weights.
2. Both edit scripts modify exactly two global weight matrices
   (`context_embedder.weight` `[3072,4096]`,
   `time_text_embed.text_embedder.linear_1.weight` `[3072,768]`), selected
   by an identical substring match (`Double_proxy.py:48`,
   `UCE_original.py:26`) regardless of which concept is edited — confirmed
   both by source and empirically (4/4 completed single edits matched
   exactly these 2 keys).
3. DP's preserve constraint is an exact null-space projection in **ideal
   arithmetic** (`Double_proxy.py:191-233`): `ΔW = Z @ U_2.T`, `U_2` spanning
   the left null-space of the stacked preserve vectors via SVD.
4. **Correction (this revision): DP's computation departs from the ideal
   identity in three concrete, source-confirmed ways**, none stated in the
   prior ledger:
   - SVD rank cutoff `1e-6` (`Double_proxy.py:216`) — singular values at or
     below this are treated as zero, shrinking the *effective* null space
     relative to the ideal, infinite-precision one.
   - Per-token skip threshold `1e-10` on `x_norm_sq`
     (`Double_proxy.py:225-228`) — if a given erase token's projection onto
     the null space is too small, that token's update is **skipped**
     entirely (printed warning, not an error). A large nominal null-space
     dimension does not establish that a *specific* erase vector (e.g.
     parachute's) has a usable nonzero projection into it.
   - A single bf16 cast at the end of each module's edit loop
     (`Double_proxy.py:238`, within the fp32-arithmetic block spanning
     `Double_proxy.py:215-238`) can turn an exactly-preserved response
     (in fp32) into a measurably different one. Verified by hand (CPU
     bookkeeping, no model execution): toy parent row `[1,1]`, update
     `[3/1024,-3/1024]`, input `[1,1]` — exact response change before
     casting is `0`; bf16's representable spacing near `1.0` is `1/128`
     and near `0.997` is `1/256` (half-exponent bracket), so
     `1+3/1024 -> 1.0` and `1-3/1024 -> 255/256`; the rounded response is
     `511/256`, a `-1/256` deviation from the exact value. Reproduced here
     as an illustrative toy example only, explicitly not a measured FLUX
     effect.
5. UCE's preserve term is a *soft*, weighted least-squares trade-off, not
   an exact constraint (`UCE_original.py:85-113`): `W_new = mat1 @
   inverse(mat2)` with `mat1`/`mat2` summing weighted outer products over
   every erase and preserve concept plus a `λ·w_old` regularizer. This
   asymmetry with DP is native to the released code, not introduced by this
   protocol.
6. **Correction (this revision): DP's native multi-concept loop pools
   guides across every erase target, it does not assign one guide per
   target.** `Double_proxy.py:174-189` rebuilds `guide_vecs` from the
   **entire** `guide_concepts` list for every erase token of every erase
   concept; with `len(guide_vecs) > 1` the code takes the projection onto
   the *joint span* of all listed guides (`V_g @ pinv(V_g.T@V_g) @
   (V_g.T@v_i)`, lines 186-189). Passing `--edit_concepts "chain
   saw;parachute" --guide_concepts "stick;cloth"` in one call does **not**
   pair `stick` with chain saw and `cloth` with parachute — both erase
   concepts' tokens are projected toward the pooled `{stick,cloth}` span.
   The prior ledger's framing of DP's batch call as merely "ordered,
   pooled guides unmentioned" was incomplete and is corrected here.
7. DP's multi-concept loop is still genuinely sequential/ordered even
   within a single call for the *update* step itself (`Double_proxy.py:
   153-235`): each concept's update modifies `W_0_f32` in place before the
   next concept's baseline is computed from the updated matrix.
8. UCE's native multi-concept loop (`UCE_original.py:96-113`) assigns
   guides **correctly per-target** via `zip(edit_concepts, guide_concepts)`
   (line 97) — unlike DP, `"chain saw;parachute"` paired with
   `"stick;cloth"` does pair `stick` with chain saw and `cloth` with
   parachute. `mat1`/`mat2` are built from sums over every listed concept
   and solved once. **Correction (this revision): this order-invariance
   claim is about exact arithmetic only** — floating-point summation is not
   strictly associative, so a different evaluation order of the same
   concept set can produce a bit-different (not meaningfully different)
   numeric result. The prior ledger's unqualified "order of the list does
   not change the result" claim overstated this as bitwise invariance,
   which it is not.
9. **Correction (this revision): a batch call's `preserve_concepts`
   necessarily excludes both erase targets** — in a joint
   `edit_concepts="chain saw;parachute"` call, neither concept is preserved
   (both are being erased), unlike SEQ's second edit, which does preserve
   chain saw. The prior ledger did not state this preservation-membership
   difference, which is at least as significant as the guide-handling
   difference for interpreting any SEQ-vs-batch comparison.
10. **Correction (this revision): DP's native edit dtype is bf16, UCE's is
    fp32** (already recorded in `RUN_MANIFEST.md`, not previously connected
    to the batch-comparator precision question). DP's two-call SEQ
    therefore has one extra bf16-rounding step (after edit 1) that its
    one-call batch variant does not have; UCE's two-call SEQ does not have
    an analogous step, since fp32 safetensors save/reload is lossless. This
    precision-boundary asymmetry is DP-specific.
11. **Correction (this revision): protection concerns both conditioning
    streams, not only T5.** `context_embedder` consumes T5 sequence
    embeddings (contextual, full self-attention); `time_text_embed.
    text_embedder.linear_1` consumes the CLIP **pooled** embedding, itself
    computed over the whole input sequence and therefore also contextual
    despite being a single vector. Both streams' preserve-constraint inputs
    are built from the **isolated** string `"chain saw"`
    (`pipe.encode_prompt(prompt="chain saw", ...)` in both scripts), not
    from the actual evaluation prompt `"Image of chain saw"`.
12. The seven candidate-B recognition counts in `PROTOCOL.md`'s table were
    independently recomputed from `evidence/recognition_summary.json`
    (sha256 `0dd43f81ed98fe53eecb26b72b0e423a643831df687fc9466ded3e2cedaa96d4`)
    and match the supplied table exactly.
13. `Flux/scripts/` contains ten released `Double_<Class>_flux.sh` examples
    but only two released `UCE_<Class>_flux.sh` examples (cassette player,
    chain saw) — confirmed by directory listing.
14. **New this revision**: the two existing A-parent checkpoint files were
    hashed at the file level (sha256, no tensor loading): DP A-parent
    (`flux_double_proxy_ChainSaw.safetensors`) =
    `d9e3d91c1cb2b544ca8a70353146ce68bdf2d330b795223d01caea62911cfd0a`; UCE
    A-parent (`flux_UCE_proxy_ChainSaw.safetensors`) =
    `bd21369aed67f69f199d037fce27041da300aab17ba2179f23a9f08f9155ee20`. These
    pin parent identity for a future execution; matching the two weight-key
    *names* at load time is a separate, necessary-but-not-sufficient check.
15. **New this revision**: `Double_Parachute_flux.sh`'s released preserve
    list is `"cassette player;chain saw;church;english springer;gas
    pump;tench;garbage truck;golf ball;french horn"` — chain saw is
    **already** one of parachute's 9 preserve concepts in the released
    single-edit example, so "include A in B's preserve list" is satisfied
    by reusing this exact list and order unchanged, with no reordering
    needed, for both the direct-B and sequential endpoints.
16. **New this revision**: `SD/data/small_imagenet_prompts.csv` uses
    **identical prompt text across all 10 rows of a given class** — e.g.
    every chain-saw row reads `"Image of chain saw"`, differing only in
    `evaluation_seed`. There is exactly one literal prompt per target, not
    ten distinct contexts; the prior ledger's "10 CSV-row contexts" framing
    was imprecise and is corrected.
17. Parachute never appears as an erasure or historical target anywhere in
    the closed `results/rece_seq_golfball_englishspringer/` line (golf
    ball: historical target throughout; French horn: S1's erased
    new-target) — confirmed by grep; parachute appears only in that line's
    retention-panel mentions.

## UNESTABLISHED (named gaps, not assumed away)

18. Whether DP's exact null-space guarantee for the *isolated* strings
    "chain saw" transfers, even approximately, to the *contextual* in-
    prompt token embeddings (both T5 and, per item 11, the CLIP pooled
    stream) inside the real evaluation prompt `"Image of chain saw"`. No
    toy closed-form bound was derived for this gap.
19. Whether UCE's soft preservation term keeps chain-saw's output
    sufficiently close to the parent's own response in practice — not
    bounded analytically here.
20. Whether a positive classifier reading of "chain saw" after a B-edit
    reflects actual chain-saw content, or the reverse (a false negative on
    the A-parent itself) — the paired witness-verification rule in
    `PROTOCOL.md ## 11` exists precisely because this is unresolved.
21. Whether parachute's own erase-token vectors clear DP's `1e-10`
    skip-threshold (item 4) — not knowable without running the edit;
    flagged, not assumed either way.

## PREDICTION (the untested claim this protocol is designed to test)

22. SEQ (both methods) will show some chain-saw relapse beyond zero, with
    DP's relapse (if any) smaller than UCE's — motivated by established
    claims 3-5 (DP's ideal-arithmetic exactness vs. UCE's soft trade-off),
    but explicitly confounded by other native differences (DP's all-token
    vs. UCE's last-token-only encoding; DP's bf16-native vs. UCE's
    fp32-native precision pathway, item 10) that are independently
    plausible contributors to any observed ranking. A ranking, if observed,
    is reported as consistent with the hard/soft distinction, not as
    isolating it as the sole cause. **"Noise band" is withdrawn as a
    concept in this revision** — see `PROTOCOL.md ## 10` for the explicit
    replacement definitions (raw transitions, confirmed witnesses, matched
    SEQ-vs-batch counts) and the prospectively fixed
    meaningful-aggregate-deficit threshold (>=10/100 confirmed witnesses).

## CEASE and BLOCK — acknowledged prior art, not re-investigated here

- **CEASE** (`yongliang-wu/CEASE`) targets this exact mechanism by name. As
  last inspected (`docs/research/feasibility_sources_2026-10-11.json`,
  fetched 2026-10-11T05:13:57Z): no method code in the repository tree
  (`contains_method_code: false`; "Code coming soon"). Cited as directly
  relevant prior art; not re-checked in this protocol-only revision.
- **BLOCK** (`https://proceedings.mlr.press/v306/kumar26c.html`,
  `sprintml/block-concept-removal` @
  `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5`) already studies sequential
  removal on modern models. A bounded deficit found by this protocol, if
  any, would be a narrow DP/UCE-specific finding on one concept pair — it
  would not establish that BLOCK fails. No additional model or baseline
  search is proposed.

## What this ledger deliberately does not contain

No RECE/SD1.x mechanism claims (closed, unrelated line). No claim that
Gemini's proposed statistical protocol was evaluated or used. No claim of
novelty for the sequential-interference question itself. No new literature
or baseline survey was performed for this revision — only re-inspection of
already-cloned source and already-published tables, plus the two new
file-level checkpoint hashes (item 14).
