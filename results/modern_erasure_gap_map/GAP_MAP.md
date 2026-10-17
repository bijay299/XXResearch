# Modern text-to-image concept-erasure — capability & assumptions gap map

## Decision: **NONE selected**

Three distinct candidates were formulated and gated (`CANDIDATES.md`). **None clears the
plausible-novelty gate**, and a strength elsewhere cannot compensate. The decision rests on
two different grounds, kept separate on purpose:

- **Established coverage** — C2's measurement already exists (CoME-Bench evaluates a
  weight-editing baseline, MACE fused into DiT linear weights, with a Selective-Alignment
  metric); C1's weight-edit-vs-inference-time comparison on FLUX.1-dev already exists
  (DVE §5.1 benchmarks ESD, UCE, CA, MACE, EAP, Meta-Unlearning, EraseAnything).
- **Triviality / insufficient class** — what remains of C1 is either "build a better weight
  edit" (method design) or an elementary expressiveness observation (a fixed linear map on
  the conditioning path is not an input- and timestep-dependent velocity correction); C3
  reduces to a threshold/attack study on a method whose own robustness is already measured
  across four attack suites.

**NONE does not mean no gap exists in this field.** It means no candidate I could justify
from the inspected evidence survived the bar that a model port, standard projection, generic
bound, extra detector/attack, or bookkeeping is insufficient. Inspection was bounded (see
`SOURCE_LEDGER.md`): no code was audited, several sources were read at abstract level only,
and **no exhaustive novelty search is claimed**.

No successor assessment, experiment or follow-up job is scheduled, and nothing here
authorises implementation.

---

## 1. Capability and assumptions matrix

Deployment contract is tracked separately from mechanism, because an inference-time
intervention and a weight edit ship different artifacts: a weight edit yields a standalone
checkpoint that carries the erasure wherever the weights go; an inference-time method
protects only a serving stack whose sampling loop the provider controls. Neither is
"better" in general — they answer different deployment questions.

| Method | Deployment contract | Mechanism / access needed | Scope (backbone, concepts, operation) | Evidence reported | Guarantee status | Inspection depth |
|---|---|---|---|---|---|---|
| **DP** (2604.10032; pinned `3fc3d8a`) | Standalone edited checkpoint | Closed-form edit of **2 global matrices** on the text-conditioning path: `context_embedder` (T5 seq) + `time_text_embed.text_embedder.linear_1` (CLIP pooled); guides **pooled** across erase tokens | FLUX.1-schnell in our records; objects; single or batch erase list (`;`-separated = batch **interface**) | Our closed run (context only): parachute 100→0; retention mixed | Hard null-space projection on preserve vectors; no semantic-absence guarantee | **relevant code** (lines cited) |
| **UCE** (as implemented in the pinned DP repo) | Standalone edited checkpoint | Regularized closed-form least squares; last-content-token encoding; fp32 edit | FLUX adaptation in the pinned repo; objects | Our closed run (context only): cumulative retention failed on 2 classes | Soft trade-off, no exact preservation | **relevant code** |
| **CEASE** (2610.01989v1) | Standalone edited checkpoint | Closed-form sequential update; AIC hard-constrains the anchor response, HOC projects off historical updates | SD 1.4/1.5/2.1 per prior record; one **generic** anchor per task (`person`/`art`/`object`); continual | ~13 s/request (App. A.6) | `Δᵢc∗=0` "up to numerical precision" on the anchor; not semantic absence over earlier prompts | **methods/eval** (no code; none released as of the 2026-10-11 record) |
| **LACU v2** (2512.02657v2) | Fine-tuned checkpoint | Distillation from frozen teacher + ℓ2 reg; per-prompt mapping from 10 LLM+CLIP candidates, **recomputed every continual step** (§4.4, Alg. 1) | SD1.5 default; continual; `N_f=100` forget prompts/concept (§5.1) | ~60 min/update, 3×H100 | Soft losses; §5.3 addresses target-invalidation for the **newly requested** concept | **methods/eval** |
| **AGE** (2501.18950v3) | Fine-tuned checkpoint | Minimax-selected concrete target (Eq. 3), Gumbel-Softmax relaxation (Eq. 4) | **Fixed erasure set assumed upfront**; not a continual method | 1000 steps, batch 1; `k=100` nearest concepts | Soft `L₁+λL₂` | **methods/eval** |
| **DVE** (2602.01089v1) | **Inference-time** — erasure does **not** travel with weights | Velocity-field projection inside the ODE loop: `Δv = v_anc − v_era` (Eq. 6), correct only when `s < τ` (Eq. 9). Needs erase/anchor/user prompts; **no weights, LoRA or classifier** | **FLUX.1-dev**; NSFW (I2P 4,703), 12 ImageNet objects, 3 styles; multi-concept via independent `Δv⁽ⁱ⁾` (§4.4; 12 groups of 3). **Sequential not addressed** | Deletion (NudeNet / ResNet50 / style clf); FID MSCOCO30k; CLIP; **ASR over 4 attack suites** (Tbl 2); latency 5.45→16.37 s, 5.49 s with preprocessing (Tbl 6) | Empirical only; appendix notes velocity space "is not a perfectly idealized semantic space" | **methods/eval** |
| **Mosaic + CoME-Bench** (2605.25574v1) | **Inference-time** | Vector-field blending with **pre-trained per-concept LoRA** masks; no optimization | **FLUX.1-dev**; compositional multi-concept within one scene; object/character-level | ESR (VLM), C-CLIP, **Selective Alignment**, SSIM/MS-SSIM, FID, GFLOPs ratio (Tbl 4); baselines **MACE (weight-fused into DiT)** and EraseAnything (§5.1, Tbl 2) | Empirical only | **methods/eval** |
| **Continual-unlearning regularizers** (2511.07970v2) | Fine-tuning | Add-on regularizers + gradient projection orthogonal to a subspace | Sequential requests; reports **rapid utility collapse** from cumulative drift | — | — | **abstract only** |
| **EMMA** (2512.17320v2) | Benchmark | 13 metrics; indirect descriptions, visually similar non-targets, gender/ethnicity bias; 5 methods | Architecture coverage **not stated in the abstract** | — | — | **abstract only** |
| **M-ErasureBench** (2512.22877v1) | Benchmark | Multimodal: text prompts, learned embeddings, inverted latents; introduces an IRECE defense | — | — | — | **listing/abstract only** |

Secondary mentions not independently inspected: **MACE**, **EraseAnything**, **ESD**, **CA**,
**EAP**, **Meta-Unlearning** — known to me only as baselines named inside DVE §5.1 and
Mosaic §5.1. They are recorded as such and not treated as audited.

### What the matrix shows

The recent flow-model frontier (DVE, Mosaic — both **FLUX.1-dev**) is **inference-time**.
Both require control of the sampling loop; Mosaic additionally requires a per-concept LoRA
and extra vector-field evaluations, DVE roughly 3× latency before its preprocessing
optimisation. The weight-editing line that yields a distributable checkpoint is represented
on flow backbones mainly by ported U-Net-era methods (UCE, MACE fused into DiT) plus the
pinned DP/UCE FLUX adaptation. Benchmarks exist on both sides (CoME-Bench verified;
EMMA and M-ErasureBench abstract-level only).

## 2. Gate results

| Candidate | Relevance | Plausible novelty | Measurability | Feasibility | Outcome |
|---|---|---|---|---|---|
| **C1** — erasure that survives weight release on flow backbones | **PASS** — open-weight release is a real contract (FLUX weights are distributed) | **FAIL** — DVE §5.1 already benchmarks weight-edit baselines on FLUX.1-dev; the residue is either method design ("a better weight edit") or an elementary expressiveness point | PASS — deletion/utility/cost are separately definable | UNRESOLVED — a fair comparison needs backbone-scale generation | **Rejected** |
| **C2** — compositional selective removal by weight-only edits | PASS | **FAIL** — established coverage: CoME-Bench evaluates MACE weight-fusion with a Selective-Alignment metric on exactly this task | PASS | UNRESOLVED | **Rejected** |
| **C3** — blind spot of thresholded velocity projection (`s ≥ τ` ⇒ no correction) | PASS | **FAIL** — DVE already reports ASR across four attack suites; τ and γ are its own knobs, so this reduces to a sweep or a new attack (insufficient class) | PASS | PASS (analysis-first) | **Rejected** |

No candidate passes every gate, so **NONE is selected**. Per the rules, a strong Relevance
or Feasibility score does not offset a failed novelty gate, and I have not required
experimental proof at this scoping stage — C1–C3 fail on *attribution to existing work* and
*insufficient claim class*, not on being untested.

## 3. What would change this

Honest statement of what my inspection could not settle, recorded as **limits, not gaps**:

- EMMA and M-ErasureBench were read at abstract/listing level. If either already covers
  flow backbones with the weight-edit vs inference-time distinction, that further
  *strengthens* NONE; if neither does, that absence still would not by itself establish a
  gap.
- No code was audited for DVE, Mosaic, CEASE, LACU or AGE; MACE and EraseAnything were not
  inspected at all. A deeper audit could reveal either prior coverage I missed or a sharper
  distinction — in the latter case a **new, separately declared** scoping pass would be
  required. Nothing here schedules one.
