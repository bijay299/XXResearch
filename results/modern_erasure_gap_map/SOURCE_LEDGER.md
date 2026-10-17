# Source ledger — modern erasure gap map

Scoping date **2026-10-17**. Two evidence fields are tracked separately for every row:

- **Inspection depth** — `abstract` / `methods` / `evaluation` / `relevant code`.
- **Claim support** — `explicit source claim/result` (the source states it) /
  `deduction under stated assumptions` (my reasoning from it) / `unverified`.

**Reading a paper is not independent empirical verification.** Nothing here was reproduced.
No code was executed, no model/encoder/detector was run, no checkpoint tensor was loaded, no
image was viewed. **No exhaustive novelty search is claimed.**

## Primary sources inspected this pass

| # | Source | Version / date | Inspection depth | Used for | Claim support |
|---|---|---|---|---|---|
| S1 | **Mosaic / CoME-Bench**, `arxiv.org/abs/2605.25574v1` + `/html/2605.25574v1` | v1, fetched 2026-10-17 | **methods + evaluation** (HTML) | Inference-time vector-field blending with per-concept LoRA masks, "without additional optimization"; "We adopt FLUX.1-dev as our baseline T2I model" (§5.1); baselines **MACE** (weight-fused into DiT linear weights, App. A, Eq. 9) and **EraseAnything** (§5.1, Tbl 2); metrics ESR (VLM), C-CLIP, **Selective Alignment** (App. B), SSIM/MS-SSIM, FID, GFLOPs ratio (Tbl 4); limitations §6 — LoRA dependency, extra inference cost, object/character-level only | explicit source claim/result |
| S2 | **DVE**, `arxiv.org/abs/2602.01089v1` + `/html/2602.01089v1` | v1, fetched 2026-10-17 | **methods + evaluation** (HTML) | Training-free inference-time velocity projection: `Δv = v_anc − v_era` (Eq. 6), `s = ⟨v_user, Δv/‖Δv‖⟩` (Eq. 8), correction only when `s < τ` (Eq. 9); requires erase/anchor/user prompts, **no weight modification, LoRA or classifier** (Alg. 1); "We use Flux.1-dev as our base model" (§5.1); baselines ESD, UCE, CA, MACE, EAP, Meta-Unlearning, EraseAnything (§5.1); I2P 4,703 prompts, 12 ImageNet objects, 3 styles; ASR over 4 attack suites (Tbl 2); latency 5.45 → 16.37 → 5.49 s (Tbl 6); multi-concept via independent `Δv⁽ⁱ⁾` (§4.4, Tbl 7/11); **sequential not addressed**; App. A.1 Case 3 "the velocity space of flow models is not a perfectly idealized semantic space" | explicit source claim/result |
| S3 | **EMMA**, `arxiv.org/abs/2512.17320v2` | v2, fetched 2026-10-17 | **abstract only** | Concept-erasure benchmark; 13 metrics; 5 methods analysed; robustness to indirect descriptions and visually similar non-targets; gender/ethnicity bias. **Backbone coverage not stated in the abstract** — whether flow models are included is **unverified** | explicit (abstract) / coverage unverified |
| S4 | **M-ErasureBench**, `arxiv.org/abs/2512.22877v1` | v1, located 2026-10-17 via arXiv API listing | **listing/abstract only** | Multimodal erasure-evaluation benchmark across text prompts, learned embeddings and inverted latents; introduces an IRECE defense | explicit (listing) / details unverified |
| S5 | **CEASE**, `arxiv.org/html/2610.01989v1` | v1, fetched 2026-10-17 (reused from the prior assessment's corrected record) | **methods + evaluation** | Sequential closed-form updates against running weights (§3.1); AIC `Δᵢc∗=0` "up to numerical precision" on the anchor response (§3.3); HOC orthogonal projection (Eq. 12–13); one generic anchor per task `person`/`art`/`object` (§4.1); ~13 s/request (App. A.6) | explicit source claim/result |
| S6 | **LACU v2**, `arxiv.org/html/2512.02657v2` | v2 (reused corrected record) | **methods + evaluation** | §4.4 + Alg. 1 mapping selection recomputed each continual step with the current frozen teacher; §5.1 `N_f=100` forget prompts per concept; §5.3 target-invalidation discussion; soft distillation + ℓ2; ~60 min/update on 3×H100 | explicit source claim/result |
| S7 | **AGE**, `arxiv.org/html/2501.18950v3` | v3 (reused corrected record) | **methods + evaluation** | Minimax target selection (Eq. 3), Gumbel-Softmax relaxation (Eq. 4), `L₂` preserves the chosen target; **fixed erasure set assumed upfront** — not a continual method; 1000 steps, batch 1; `k=100` | explicit source claim/result |
| S8 | **Continual-unlearning regularizers**, `arxiv.org/abs/2511.07970v2` | v2 (reused corrected record) | **abstract only** | Sequential requests; "rapid utility collapse"; cumulative parameter drift; add-on regularizers + gradient projection | explicit (abstract) |
| S9 | **DP**, pinned source `czhang024/CVPR_DouleProjection@3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (+ paper `arxiv.org/abs/2604.10032`, cited by ID) | commit verified previously; reused | **relevant code** | Edits exactly two conditioning-path matrices (`context_embedder`, `time_text_embed.text_embedder.linear_1`); `Double_proxy.py:174-189` guides **pooled** across erase tokens, not per-target; `:295` `;`-split = **batch interface only** | explicit source inspection |

## Secondary mentions — named only inside S1/S2, **not independently inspected**

**MACE**, **EraseAnything**, **ESD**, **CA**, **EAP**, **Meta-Unlearning**. Recorded as
baselines reported by DVE §5.1 and Mosaic §5.1. Their own claims, scopes and limitations
were **not** read; nothing in this map treats them as audited.

## In-repo records reused as **context only** (never as fresh confirmation data)

| # | Record | Used for | Status |
|---|---|---|---|
| L1 | `results/modern_flux_dp_uce/sequential_diagnostic_execution/` — **CLOSED_CORRECTED at `0e88d26`** | Architectural facts about the pinned DP/UCE FLUX adaptation and the standing measurement cautions (a detector-defined `C=0,U=0` is not a semantic-absence certificate; input-normalised residuals are not output-presence measures) | Read only. **Not re-scored, not regenerated, unused image allowance not spent** |
| L2 | `results/changing_replacement_assessment/` — **NO_GO, CLOSED_CORRECTED at `5786864`** | Confirms the changing-replacement formulation is closed and must not be revived under renamed terminology; its corrected source records for CEASE/LACU/AGE reused | Read only, unchanged |
| L3 | `docs/research/FEASIBILITY_BRIEF_2026-10-11.md` | CEASE implementation availability **as of 2026-10-11, not re-verified here**; not used as evidence of a missing capability | Read only |

## Access limitations and explicit non-claims

- **Abstract-level only**: EMMA (S3), M-ErasureBench (S4), 2511.07970v2 (S8). Their
  coverage of flow backbones and of the weight-edit vs inference-time distinction is
  **unverified**, and that uncertainty is recorded as a limit — **unknown coverage is not
  failure**, and absence of a result in material I did not read establishes nothing.
- **No code audit** for Mosaic, DVE, CEASE, LACU or AGE; only the pinned DP repo was read
  at code level. **No imported research code was executed.**
- Mosaic and DVE were previously known here at abstract level only; this pass raised them
  to methods/evaluation depth via fetched HTML, which is **not** independent reproduction.
- Backbone naming: Mosaic and DVE both state **FLUX.1-dev**; our own closed records concern
  **FLUX.1-schnell**. These are different checkpoints and are not treated as interchangeable.
- No author or reviewer was contacted; no dataset, benchmark or evaluation package was
  created or downloaded.
