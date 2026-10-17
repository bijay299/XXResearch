# Modern text-to-image concept-erasure — capability & assumptions gap map

## Decision: **NONE — SCOPING PASS CLOSED WITH CORRECTIONS**

**No sufficiently specified and justified additional contribution was established in this
bounded scoping pass.** That is the whole of the claim. It does **not** mean the three
assessed directions are solved, impossible, or exhausted, and **no field-wide impossibility
or solvedness conclusion follows.**

> **Amended 2026-10-17** — see [`CORRECTIONS.md`](CORRECTIONS.md). The decision (**NONE**)
> is retained; several supporting claims published at `5b2d184` were wrong or overreaching
> and are corrected: a backwards description of DVE's Eq. 9 repair direction, an unsupported
> expressiveness separation, an over-general "the frontier is inference-time" characterisation,
> prior-art reasoning that treated existing comparisons as disproof, and blanket
> field-closure wording. **No new experimental evidence was produced or required.**

Why no candidate was selected, stated correctly:

- **C1 and C2**: repeating an existing comparison would not by itself establish novelty —
  DVE §5.1 contains relevant FLUX baseline comparisons and CoME-Bench evaluates fused MACE
  with non-target preservation metrics. But those comparisons **do not prove the task
  solved and do not rule out a stronger algorithmic or theoretical contribution**. The
  stronger hypothesis in each card was simply **not specified sharply enough to select**:
  "build a better weight-space editor" is too vague to act on, not inadmissible.
- **C3**: a sharper mechanistic claim about persistent non-activation might be worth
  stating, but none was developed here to a selectable level. DVE's own Table 5 already
  ablates threshold `τ` and strength `γ`, which covers an ordinary parameter sweep; it does
  not settle the sharper claim either way.

An **unresolved** essential gate prevents selection just as a failed one does. All three
cards are retained as assessed (`CANDIDATES.md`); no replacement, fourth candidate,
charter, dossier or successor assessment is created.

---

## 1. Capability and assumptions matrix

Deployment contract is tracked separately from mechanism, because a weight-space
modification ships an artifact that carries the erasure wherever the weights go, whereas an
inference-time intervention protects only a serving stack whose sampling loop the provider
controls. Neither is "better" in general.

| Method | Deployment contract | Mechanism / access needed | Scope (backbone, concepts, operation) | Evidence reported | Guarantee status | Inspection depth |
|---|---|---|---|---|---|---|
| **DP** (2604.10032; pinned `3fc3d8a`) | Modified checkpoint | Closed-form edit of **2 conditioning-path matrices**: `context_embedder` (T5 seq) + `time_text_embed.text_embedder.linear_1` (CLIP pooled); guides **pooled** across erase tokens | FLUX.1-**schnell** in our records; objects; `;`-separated list = batch **interface** | Our closed run (context only) | Hard null-space projection on preserve vectors; no semantic-absence guarantee | **relevant code** |
| **UCE** (as implemented in the pinned DP repo) | Modified checkpoint | Regularized closed-form least squares; last-content-token encoding | FLUX adaptation in the pinned repo; objects | Our closed run (context only) | Soft trade-off | **relevant code** |
| **Minimalist Concept Erasure** (2507.13386v1) | **Modified checkpoint** | **Parameter masking / neuron-level ablation**: `θ = M ⊙ θ′, M ∈ {0,1}^{\|θ\|}` (§3.6); hard-discrete sampling then binarisation | **FLUX.1-schnell** primary (§4.1: *"we focus on … FLUX.1-Schnell"*); SD-XL in appendix; IP characters + art styles, multiple concepts | ACC via LLaVA; CLIP, FID, SSIM; step-wise gradient checkpointing (§3.7) | Empirical | **methods/eval** |
| **EraseAnything** (2412.20413v1) | **Adapted checkpoint (LoRA)** | **Flow-specific LoRA optimization**; tunes `add_q_proj`/`add_k_proj` in dual-stream blocks (App. A) | **FLUX.1-dev**; single-concept primary, multi-concept via normalized LoRA blending (App. F.2, Eq. 12) | NudeNet on I2P (4,703); FID/CLIP on MS-COCO 10k; synonym + black-box robustness (App. B); 3.57 MB LoRA/concept, single A100 | Empirical | **methods/eval** |
| **CEASE** (2610.01989v1) | Modified checkpoint | Closed-form sequential update; AIC hard-constrains the anchor response; HOC projects off historical updates | SD 1.4/1.5/2.1 per prior record; generic anchor per task; continual | ~13 s/request (App. A.6) | `Δᵢc∗=0` "up to numerical precision" on the anchor | **methods/eval** (no code released as of the 2026-10-11 record) |
| **LACU v2** (2512.02657v2) | Fine-tuned checkpoint | Distillation from frozen teacher + ℓ2; mapping recomputed each continual step (§4.4, Alg. 1) | SD1.5 default; continual; `N_f=100` forget prompts/concept | ~60 min/update, 3×H100 | Soft losses | **methods/eval** |
| **AGE** (2501.18950v3) | Fine-tuned checkpoint | Minimax target selection (Eq. 3), Gumbel-Softmax (Eq. 4) | **Fixed erasure set upfront**; not continual | 1000 steps, batch 1 | Soft `L₁+λL₂` | **methods/eval** |
| **DVE** (2602.01089v1) | **Inference-time** | Velocity projection in the ODE loop; `Δv = v_anc − v_era` (Eq. 6), `s = ⟨v_user, Δv/‖Δv‖⟩` (Eq. 8); correction **only when `s < τ`** (Eq. 9). Needs prompts only — no weights, LoRA or classifier | **FLUX.1-dev**; NSFW/objects/styles; multi-concept via independent `Δv⁽ⁱ⁾` (§4.4); sequential not addressed | Deletion detectors; FID/CLIP; **ASR (Tbl 2)**; latency 5.45→16.37→5.49 s (Tbl 6); **τ/γ ablation (Tbl 5)** | Empirical; App. A.1 notes velocity space "is not a perfectly idealized semantic space" | **methods/eval** |
| **Mosaic + CoME-Bench** (2605.25574v1) | **Inference-time** | Vector-field blending with **per-concept pre-trained LoRA** masks; no optimization | **FLUX.1-dev**; compositional multi-concept in one scene; object/character-level | ESR (VLM), C-CLIP, **Selective Alignment** (App. B), SSIM/MS-SSIM, FID, GFLOPs (Tbl 4); baselines **MACE fused into DiT** (App. A) + EraseAnything (§5.1, Tbl 2) | Empirical | **methods/eval** |
| **Continual-unlearning regularizers** (2511.07970v2) | Fine-tuning | Add-on regularizers + gradient projection | Sequential requests; reports rapid utility collapse | — | — | **abstract only** |
| **EMMA** (2512.17320v2) | Benchmark | 13 metrics; 5 methods; indirect descriptions, similar non-targets, bias | Backbone coverage not stated in abstract | — | — | **abstract only** |
| **M-ErasureBench** (2512.22877v1) | Benchmark | Text prompts, learned embeddings, inverted latents; IRECE defense | — | — | — | **listing only** |

Secondary mentions **not independently inspected**: MACE, ESD, CA, EAP, Meta-Unlearning —
known only as baselines named inside DVE §5.1 and Mosaic §5.1.

### What the matrix shows (corrected)

**The modern flow-model erasure landscape is not characterised by inference-time methods.**
Of the flow-targeted work inspected here, **two are weight-space** —
Minimalist Concept Erasure (parameter masking, **FLUX.1-schnell**) and EraseAnything
(flow-specific LoRA, FLUX.1-dev) — and **two are inference-time** (DVE, Mosaic). The
earlier published statement that the frontier is inference-time, with weight editing
represented "mainly by ported U-Net-era methods", was an artifact of which papers that pass
happened to inspect and is **withdrawn**. Weight-space erasure on flow backbones is an
active, current line, not a vacancy.

The genuine distinction that survives is narrower and still useful: **DVE and Mosaic, as
inspected, do not modify weights**, so their erasure does not travel with a released
checkpoint — DVE states it needs "no weight modification, LoRA or external classifiers", and
Mosaic depends on per-concept LoRA masks plus extra vector-field evaluations. That is a
property of **those two inspected methods**, not of the field.

## 2. Gate results (corrected)

| Candidate | Relevance | Plausible novelty | Measurability | Feasibility | Outcome |
|---|---|---|---|---|---|
| **C1** — erasure carried by released weights on flow backbones | **PASS** | **UNRESOLVED — insufficiently specified.** Repeating DVE's existing FLUX baseline comparison would not establish novelty; but that comparison does not prove the task solved or exclude a stronger algorithmic/theoretical result. C1 as written reduces to "build a better weight-space editor", which is **too vague to select, not inadmissible** | PASS | UNRESOLVED | **Not selected** |
| **C2** — compositional selective removal under weight-space edits | PASS | **UNRESOLVED — insufficiently specified.** CoME-Bench evaluates fused MACE with non-target preservation metrics, so re-running that comparison adds nothing; it does **not** show weight-space methods cannot achieve compositional selectivity. **C2 specializes C1** (same weight-release objective, compositional setting) rather than being an independent mechanism | PASS | UNRESOLVED | **Not selected** |
| **C3** — persistent non-activation of thresholded velocity correction | PASS | **UNRESOLVED — insufficiently specified.** DVE Table 5 already ablates `τ` and `γ`, covering an ordinary sweep; a sharper mechanistic claim about *persistent* non-activation across a trajectory was **not developed here** to a selectable level, and existing ASR results neither establish nor exclude it | PASS | PASS (analysis-first) | **Not selected** |

**Three cards do not mean three independent technical mechanisms**: C2 is a specialization
of C1. An unresolved essential gate prevents selection exactly as a failure would, so the
outcome is NONE either way — but the *reason* is under-specification, not refutation.

## 3. What this pass could not settle

Recorded as limits, **not** as gaps and **not** as coverage:

- EMMA, M-ErasureBench and 2511.07970v2 were read at abstract/listing level only.
- No code was audited beyond the pinned DP repository; MACE, ESD, CA, EAP and
  Meta-Unlearning were not inspected at all.
- **DVE's reported comparisons are not uniformly first-party**: Table 1 states *"The results
  with * are sourced from EraseAnything"*, and its robustness table (Tbl 2) covers only
  CA (Model-base), ESD, EraseAnything and DVE — a **smaller set** than the full §5.1
  baseline list (which also names UCE, MACE, EAP, Meta-Unlearning). Nothing here should be
  read as every named baseline having been independently rerun across all metrics, attacks
  and costs.
- Backbone versions differ and are not interchangeable: Mosaic, DVE and EraseAnything use
  **FLUX.1-dev**; Minimalist Concept Erasure and our own closed records use
  **FLUX.1-schnell**.
- Source reading here is **not** independent reproduction, a comprehensive method audit, or
  proof that C1/C2 is solved. **No exhaustive novelty search is claimed.**
