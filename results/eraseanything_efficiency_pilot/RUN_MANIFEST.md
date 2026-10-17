# Run manifest — EraseAnything efficiency pilot

The research contract. §1–§7 were fixed before any edited-model evaluation; §8 is the
append-only deviation log. **Amended once** — see [`AMENDMENT_01.md`](AMENDMENT_01.md).

| | |
|---|---|
| Start (UTC) | **2026-10-17T22:19:05Z** |
| Deadline (UTC) | 2026-10-31T22:19:05Z (14 calendar days) |
| Work directory | `/data/bijaypandey/cuig_pilot/eraseanything_efficiency_pilot/` |
| Host | 4 × NVIDIA A100-SXM4-**40GB**, driver 550.144.03, CUDA 12.4, 255 CPU cores |

## 1. Research question

> Can one algorithmic change reduce adaptation resources for native EraseAnything while
> preserving its deletion, unrelated-generation preservation and prompt-variation robustness?

## 2. Pins

| item | pin |
|---|---|
| Author code | `tomguluson92/EraseAnything` @ `67d8c0bd5b153ff0e394b45e893b6671bc40e30b` |
| Paper | `arXiv:2412.20413v1` (EraseAnything, ICML 2025) |
| Base model | `black-forest-labs/FLUX.1-dev` @ `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| diffusers | 0.32.0 (released) + 2 ported author changes, see §6 |
| peft | the author's vendored 0.13.2 |
| torch | 2.4.0+cu121 |
| Scorer | `openai/clip-vit-large-patch14` |
| Comparator code (amendment) | EA++ @ `f73e5c1e5bb51b6a24b57039660e06dcd0ecec82`; BLOCK @ `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5` |

FLUX.1-dev is **gated**; access was verified privately under HF account `CaptIce`.
Credentials live only in `/data/bijaypandey/cuig_pilot/secrets/` (chmod 600) and are never
echoed, logged or committed.

## 3. Native reference, as implemented

Read from the author tree, not assumed:

- **LoRA** on `attn.add_k_proj`, `attn.add_q_proj` only; r = alpha = 8, gaussian init →
  **76 tensors / 1,867,776 params**; backbone frozen (1160 tensors), verified at runtime.
- **Bi-level alternation**: `global_step % 2 == 0` → upper (ESD + attention deactivation);
  else → lower (flow-matching LoRA loss + InfoNCE). `lamb1=1, lamb2=0.001, lamb3=0.1, lamb4=0.1`.
- bf16, resolution 512, batch 1, grad-accum 1, AdamW lr 1e-3 constant, no warmup,
  `max_train_steps=300`, `ddim_steps=28`, `start_guidance=3`, `max_sequence_length=256`,
  `gradient_checkpointing: false`.
- **K = 3 "irrelevant concepts" come from a GPT-4o call** (`tools/ir_concept.py: MoE`).
  One call per concept, **frozen to disk** and reused by every arm and seed.
- `CustomFlowMatchEulerDiscreteScheduler.set_train_timesteps` draws a **random** 28-point
  schedule per call (`torch.sigmoid(torch.randn(...))`), so trajectories are random in both
  noise and schedule.

**Paper/code discrepancies resolved explicitly:** the paper's App. A names
`add_q_proj`/`add_k_proj` and the code matches; the author's `config/config.yaml` targets an
NSFW concept, so the **non-sensitive, author-supplied** `config/disentangle/config_frida.yaml`
(artistic style) was used instead; the repository ships **no** `instance_data_dir` images, so
the training set was created (§5).

## 4. Task family, concepts, panels

**Artistic style**, author-supported (`config_frida.yaml`). Disjoint from every closed
study in this repository (those used only ImageNet objects).

| role | concept | chosen by |
|---|---|---|
| development | **Frida Kahlo** | the author's own config — not chosen by outcome |
| held-out | **Vincent van Gogh** | fixed before any model existed |
| held-out | **Katsushika Hokusai** | fixed before any model existed |

**120 frozen cases per concept** = 40 canonical + 40 paraphrase/perturbation + 40 retention
(20 related-but-retained artists + 20 generic objects). Deterministic, LLM-free, built from
a frozen 40-subject list and templates. Per-case generation seeds are derived by SHA-256 and
reused identically across arms; training seeds are separate.
`frozen/cases.json` sha256 `89673233d25951e45c5e5c2fce3df573824a4ed5af468d8d4774f636e84ad90a`.

Evaluation prompts are kept **out** of training: the trainer sees only `instance_prompt`,
`prompt_b` and the frozen LLM concepts.

## 5. Generation, training data, scoring

- **Generation (all endpoints, M0 included):** FLUX.1-dev, 512×512, 28 steps,
  `max_sequence_length=256`, `guidance_scale=3.5`, bf16, **author-faithful explicit
  attention**. Measured: the fused-SDPA path is *not* output-preserving here (mean |Δ| 1.5,
  max 135/255, 64% of pixels differ) because bf16 roundoff amplifies over 28 steps, so one
  path is used uniformly for every endpoint.
- **Training images:** 10 per concept from the config `instance_prompt` at frozen seeds
  (the repo ships none). Count is our choice, frozen here.
- **Scorer (frozen before any image existed):** CLIP ViT-L/14, argmax over the case's frozen
  label set. Target present iff argmax == target artist; retention success iff
  argmax == intended label. Missing image ⇒ counted as *present* on target panels and
  *failure* on retention, never in favour of the edited model. No case dropped.
- **Quality rule (frozen):** degenerate iff `gray_std < t_std` or `sat_frac > t_sat` or
  `lap_var < t_lap`, thresholds to be calibrated on M0 only.
- **Adequacy gate:** canonical and perturbed panels each need ≥80% M0 target presence.
  **This gate FAILED for all three concepts — see `AMENDMENT_01.md` §5.**

## 6. Documented patches to third-party code

| # | file | fix |
|---|---|---|
| 1 | diffusers `attention_processor.py`, `transformer_flux.py` | port the author's two method-critical changes (explicit materialised-softmax attention returning the matrix; `(sample, attn_maps)` from the last dual-stream block) onto released 0.32.0. The author's vendored diffusers is a **partial** tree (229 files, `utils/` and `schedulers/` absent) and does not import. Verified in float64: maps are row-stochastic and the consumed map is **bit-identical** whether or not earlier blocks materialise. `code/apply_ea_patch.py` |
| 2 | `utils/find_token.py` | it loaded a full FLUX.1-schnell pipeline **at module import** (only for its `__main__` demo) and `lora_dataset.py` imports it; moved under `__main__`. Also removed a hardcoded `HF_ENDPOINT=hf-mirror.com`. |
| 3 | `tools/ir_concept.py` | the non-Azure branch passed the whole `api_keys` dict as `api_key` → HTTP 401. Now passes `["api_key"]`, matching the Azure branch. |
| 4 | `tools/ir_concept.py` | `filt_response` ran `eval()` on the live model reply (arbitrary code execution on network content); replaced with `ast.literal_eval`, identical for the dict literal the prompt requests. |

Patches 1–2 are packaging/import fixes; 3 is a credential fix; 4 is a safety fix. None
changes the objective, update rule, LoRA targets or step counts.

## 7. Arms and screening rules (frozen, unchanged)

- **A** full native EraseAnything with shared practical engineering.
- **B** strongest ordinary cost-saving control.
- **C** the one proposed improvement (≤2 settings of the same mechanism).

Native must reach ≥75% relative suppression on both qualified target panels, retention
within 10 pp of M0, and the quality criterion. The improvement must cut allocated
GPU-device time to produce the adapter by ≥25% and be within 5 pp of A on canonical
suppression, perturbation suppression and retention, **per concept and seed**.

Planned images: M0 360 + development ≤480 + held-out 1440 = 2280 of 3000.

## 8. Deviation log (append-only)

| # | when | deviation |
|---|---|---|
| D1 | setup | Released diffusers 0.32.0 + ported patch instead of the author's vendored tree, which does not import (§6.1). |
| D2 | setup | Training images created (10/concept); the repo ships none. |
| D3 | setup | Standard OpenAI endpoint instead of the author's unavailable Azure deployment; same model (`gpt-4o`) and same MoE prompt. |
| D4 | setup | 2-GPU placement (transformer+VAE on d0, CLIP+T5 on d1). Measured peak **38.4 GB** on a 40 GB card; the paper used one 80 GB A100, and 4×40 GB is not one pool. |
| D5 | setup | Per-process thread caps (`OMP_NUM_THREADS=8`). Without them 3 concurrent jobs drove load 432 on 255 cores at 12–39% GPU utilisation. **0.695 GPU-h charged to the aborted attempt.** |
| D6 | setup | 2 failed LLM requests (HTTP 401, patch 3) charged against the 20-request cap. |
| D7 | amendment | **Algorithm development stopped**; arm C not trained; no comparator arm taken. `AMENDMENT_01.md`. |
| D8 | amendment | M0 adequacy gate failed for all 3 concepts; concepts disclosed as unqualified, **not replaced**. Two scorer recalibrations and one resolution probe tried, all reported, then stopped. |
