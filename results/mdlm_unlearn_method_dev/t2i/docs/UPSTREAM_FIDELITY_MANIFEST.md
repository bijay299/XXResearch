# Upstream source & fidelity manifest — official reference implementations

**Date:** 2026-10-26 · **Clone root:** `/data/bijaypandey/t2i_unlearn/upstream/` (**git-ignored**, see
§0.1) · **Machine-readable:** `outputs/upstream_manifest.json`

This is the fidelity ledger the implementation task requires: for each method, what upstream
actually is, what our wrapper actually does, and **every departure between them**. It makes no
performance claims. Nothing here is a reproduction until a run is recorded in §6.

---

## 0. Resolved sources

| repo | resolved commit | date | license | ref file | py files |
|---|---|---|---|---|---:|
| **IMMA** | `2f55041141705031d503134dec22f674f09e21bf` | 2025-01-17 | **Apache-2.0** | `train/defend_text_to_image_lora.py` | 11 |
| **Meta-Unlearning** | `5a2a7600895a2885323cbbf599acaf3f74a96f64` | 2025-04-22 | **NONE** | `src/meta.py` | 6 |
| **vision_relearning** | `7b9a1ccada35a4059604c2688d2822567b3cf40d` | 2025-05-30 | **README declares MIT** (no standalone LICENSE file) | `vision_unlearner.py` | 8 |
| **unified-concept-editing** | `7c724d9d7d19190e9bd33fe62fb656f1df87cd8e` | 2025-12-07 | **MIT** | `trainscripts/` | 17 |
| **FRAG** | `7fa3b5e8bcacd6b7711290843797c234d62eb64f` | 2026-08-27 | NONE | — | **0** |

All three audited revisions resolved **exactly as requested**. UCE and FRAG had no revision
specified, so current `HEAD` is recorded above and pinned in the manifest JSON.

### 0.1 Licensing constrains what we may redistribute

**Meta-Unlearning and vision_relearning ship no LICENSE file.** Absent a license, default copyright
applies and redistribution is not permitted. Therefore:

- `upstream/` is added to `.gitignore` and **no upstream source is committed to XXResearch**.
- Only **our own wrappers and patch files** are committed, with upstream referenced by commit hash
  and reproduced by the clone commands in §5.
- IMMA (Apache-2.0) and UCE (MIT) would permit vendoring; they are treated the same way for
  consistency and to keep upstream code unmixed with ours, as the task requires.

### 0.2 FRAG/FRP — availability recheck, recorded

The repository **exists and is non-empty, but contains no implementation**: `README.md` plus two
PNG assets, **0 Python files**. The README describes FRAG (a training-free predictor) and FRP, now
styled *EMNLP 2026 Main Conference*, but ships neither. **Status: implementation still
unavailable.** Our FRP port therefore remains a from-paper port with no reference to check against —
the one case where `ERRATUM_r2` C19's original "no released code" finding holds
(`T2I_BASELINES_v2.md`).

### 0.3 GIFT — no official implementation

Unchanged and unverified. **No repository is substituted**, and our GIFT port is **not** described as
an official reproduction anywhere.

---

## 1. IMMA — fidelity reconciliation

**Reference:** `train/defend_text_to_image_lora.py` @ `2f55041`. **Ours:** `scripts/erase_imma.py`.

### 1.1 The reference's actual mechanics (read from source, lines cited)

| aspect | reference |
|---|---|
| **defence scope S** | `xattn_layer_names = ['to_q','to_v','to_k','to_out']`, matched **by substring over every `named_parameter`**, excluding `'lora'` (L563-567) |
| **defence update** | `accelerator.backward(-loss)` — **ascent** on the diffusion loss w.r.t. S (L937, L954) |
| **inner adversary** | LoRA via `LoRAAttnProcessor`, own `optimizer_lora`, `accelerator.backward(loss)` (L946-952) |
| **alternation** | `freeze_grad`/`free_grad` toggling between `lora_layers` and `unet_attn_params` each step (L860-868) |
| **LoRA re-init** | `lora_layers.load_state_dict(initial_state_dict)` under `--reinitialize_lora` (L872, L366) |
| **optimizer** | `AdamW` (or `AdamW8bit` with `--use_8bit_adam`), **separate** optimizers for S and LoRA (L589-607) |
| **clipping** | `clip_grad_norm_(unet_attn_params, args.max_grad_norm)` (L940, L957) |
| **lr** | `--learning_rate` (S) and `--learning_rate_lora` separately; README uses `1e-5` / `1e-4` |
| **artifact** | `imma_unet_xatten_layer.pt`, saving only S (L977) |

### 1.2 **Deviation found: our scope is less than half the reference's**

Measured on the real 859.5M UNet (`outputs/imma_scope_fidelity.json`):

| scope | params | % of UNet | tensors |
|---|---:|---:|---:|
| full UNet | 859.5M | 100% | — |
| **reference** `to_q/k/v/out` over all params | **93.5M** | **10.88%** | 160 |
| — of which **attn1 (self-attention)** | **49.6M** | 5.77% | 80 |
| — of which attn2 (cross-attention) | 44.0M | 5.11% | 80 |
| **our `erase_imma.py`** (`.attn2.` only) | **44.0M** | 5.11% | 80 |

**The reference scope is 2.13× ours and includes self-attention, which our port excludes
entirely.** Our port's docstring says *"S = cross-attention layers only, per the paper"* — that
follows the paper's description, but the released code immunises self-attention as well. This is a
material fidelity gap and it was not previously recorded.

### 1.3 Other departures

| axis | reference | ours | status |
|---|---|---|---|
| LoRA re-initialisation | optional, `load_state_dict(initial_state_dict)` | **never** re-initialised (warm-started) | **deviation** |
| optimizers | two separate AdamW | Adam outer, one-step inner | **deviation** |
| alternation | per-step freeze/free toggling | alternating blocks | partial |
| gradient clipping on S | `max_grad_norm` | present | match |
| ascent sign on S | `backward(-loss)` | ascent on `L_A` | match |

**Conclusion: our IMMA arms are a *project reimplementation*, not a reproduction**, and the scope
gap alone is enough to make their numbers non-substitutable for the reference's.

---

## 2. Meta-Unlearning — fidelity reconciliation

**Reference:** `src/meta.py` @ `5a2a760`. **Ours:** `meta_unlearn.py` (variant A) and
`meta_unlearn_v2.py` (E2).

Already reconciled in `ERRATUM_meta_objective.md`; restated as a ledger with the new points:

| aspect | reference | variant A | **v2 (E2)** |
|---|---|---|---|
| `−L_F(θ′)` post-attack target | present (`−γ2_1`) | **absent** | **present** |
| `−ζL_R(θ′)` post-attack retain | present (`−γ2_2`, `−γ2_3`) | present | present |
| `+ζL_R(θ)` pre-attack retain | present (`+γ1_1`, `+γ1_2`) | **absent** | **present** |
| data pools | **three**: `hrm` / `irt` / `tgt`, each its own γ | two (forget / benign) | two | 
| meta-gradient scope | `gather_parameters` — selectable | full UNet, unrecorded | **xattn, recorded (44.0M)** |
| attack-simulation scope | `gather_parameters_full` | full UNet | **full UNet, recorded (859.5M)** |
| target (forget) data | `./dataset/**hrm**` imagefolder | **erased teacher's own output, unverified** | **base model, CLIP-checked 64/64 = 1.0000** |
| fixed-timestep option | `--fix_timesteps`; **the README states the paper used `fix_timesteps=False`**, the True path being only a training accelerant | absent | absent — **and therefore NOT a mismatch with the published configuration** |

### 2.1 **Two departures that remain in v2**

1. **Pool structure is 2, not 3.** The reference separates `irt` (unrelated-retain) from `tgt`
   (related-retain) with independent coefficients γ1_1/γ1_2 and γ2_2/γ2_3. v2 collapses both into
   one benign pool with a single ζ. The README's warning that example hyperparameters may differ
   from paper settings applies directly here.
2. **`--fix_timesteps` is not implemented.** The reference can accumulate the meta gradient over a
   *fixed* list of timesteps rather than random draws. v2 uses random timesteps throughout.

**Therefore E2 is our corrected implementation, not an official reproduction**, exactly as the task
stipulates. It restores the objective's signed structure; it does not restore the pool structure or
the timestep option.

---

## 3. Weight-space regularisation — fidelity reconciliation

**Reference:** `vision_unlearner.py` @ `7b9a1cc`. **Ours:** `erase_wsr.py` (anchor, wrong
direction) and `erase_wdr_v2.py` (faithful adaptation).

Reconciled in `ERRATUM_wsr_direction.md` and `FINDING67`; the ledger entry:

| aspect | reference (L366-373) | `erase_wsr.py` | `erase_wdr_v2.py` |
|---|---|---|---|
| direction | `−sqrt(mean(sq(θ_pre−θ))+1e-8)` → **maximise** distance | **minimise** distance | **maximise** |
| normalisation | RMS (÷ param count) | sum-of-squares ÷ `‖θ_base‖²` | RMS, eps 1e-8 |
| role | **is** the forget loss | third term beside erase+retain | **is** the forget loss |
| retain | CE, coefficient 1.0, alternating iteration | summed, same step | diffusion MSE, 1.0, alternating |
| scope | all `model.parameters()` | all | all |
| optimizer | Adam, `--clip-grad-norm` default **None** | Adam | Adam, no clip |
| α default | **1** | n/a | 1 (**measured inert — see below**) |
| domain | **vision classifiers, CIFAR-scale** | SD UNet | SD UNet |

**Domain adaptation, labelled as required:** the reference's experiments are example-level
unlearning in vision classifiers; the retain objective must become the diffusion MSE. Any SD result
is a **domain adaptation**, never a reproduction — including `erase_wdr_v2.py`.

**Scale caveat carried forward (`FINDING67`):** at the reference's own α=1 the term's gradient norm
is exactly **1/√N = 3.411e-5** against a retain gradient of ~0.137, i.e. **inert at 859.5M
parameters**. Parity is α ≈ 4.0e3. **Gradient-norm parity is not a claim of optimality** — under
Adam the update is preconditioned and per-coordinate normalised, so parity bounds where the term
*can* act, not where it acts *best*. The α sweep remains open.

**Native-domain sanity check is required** before the diffusion adaptation can be interpreted: the
reference must be run on its own CIFAR-scale setting to confirm our reading of `weight_dist_reg`
reproduces its published behaviour. Queued in §6 as **R3**, not yet run.

---

## 4. UCE — fidelity reconciliation

**Reference:** `unified-concept-editing` @ `7c724d9`. **Ours:** `scripts/erase_uce.py`.

Our port implements the closed form

```
W_new = (Σ v* cᵀ + s·(W c_pᵀ)c_p + λW) (c_eᵀc_e + s·(c_pᵀc_p) + λI)⁻¹
```

which was verified algebraically in `ERRATUM_uce_lambda_scale.md`. **Not yet reconciled against the
official code** — that is task item 4's requirement and is queued as **R4** in §6. Until matched-input
outputs are compared, our UCE numbers are a **project reimplementation**.

Known from our side: λ is measured inert below ~1e3 because `cᵀc` has max eigenvalue 85,850
(`ERRATUM_uce_lambda_scale.md`); λ=1000 is the first usable SD-1.5 setting.

---

## 5. Environments & reproduction commands

**`venv_t2i` is never modified** — it supports the running experiments. Each upstream env is an
isolated **copy** with its own interpreter; all pip operations target the copy.

```bash
# clone upstream at the pinned revisions (not redistributed; see §0.1)
mkdir -p /data/bijaypandey/t2i_unlearn/upstream && cd $_
git clone https://github.com/amberyzheng/IMMA && (cd IMMA && git checkout 2f55041141705031d503134dec22f674f09e21bf)
git clone https://github.com/sail-sg/Meta-Unlearning && (cd Meta-Unlearning && git checkout 5a2a7600895a2885323cbbf599acaf3f74a96f64)
git clone https://github.com/shoaibahmed/vision_relearning && (cd vision_relearning && git checkout 7b9a1ccada35a4059604c2688d2822567b3cf40d)
git clone https://github.com/rohitgandikota/unified-concept-editing
git clone https://github.com/Yi1-Chen/FRAG          # README + assets only, 0 .py files

# isolated environments
bash setup_upstream_envs.sh        # creates venv_imma, venv_uce by copy; never touches venv_t2i
```

### 5.1 Environment build — first attempt FAILED, errors recorded verbatim

Unpinned installs resolved to latest and broke both copies:

```
venv_imma:  pip install datasets torchvision
            -> torch 2.14.1+cu130, huggingface-hub 1.33.0
            -> ImportError: huggingface-hub>=0.23.2,<1.0 is required ... (transformers 4.46.3)
venv_uce:   pip install "diffusers>=0.34.0"
            -> huggingface-hub 1.33.0 + a diffusers calling torch.accelerator
            -> AttributeError: module 'torch' has no attribute 'accelerator'
               (diffusers/utils/torch_utils.py:245, torch 2.4.1 has no torch.accelerator)
```

**Root cause:** unpinned resolution against a copied 2024-era stack. **Fix:** a constraints file
(`torch==2.4.1`, `huggingface-hub<1.0`, `transformers==4.46.3`, `diffusers==0.31.0`) plus
`--no-deps` for torchvision. `venv_t2i` was never involved in either attempt.

**Second attempt succeeded:**

| env | result |
|---|---|
| `venv_uce` | **unmodified copy** — UCE's `trainscripts` import cleanly on our stack (12 modules, **0 failures**), so its `requirements.txt` pin of `diffusers>=0.34.0` is **not required** for `uce_sd_erase.py`. torch 2.4.1 / diffusers 0.31.0 / transformers 4.46.3 |
| `venv_imma` | copy + `torchvision==0.19.1` (`--no-deps`, matches torch 2.4.1) + `datasets` (constrained). torch 2.4.1 / tv 0.19.1 / datasets 5.0.1 / diffusers 0.31.0 / transformers 4.46.3. **`LoRAAttnProcessor` and `AttnProcsLayers` both import.** |

**Hardware:** both reference trainers target a single CUDA device; our cap is two GPUs at a time and
GPUs 2–3 are left to other users.

**Dependency findings (tested read-only against the current stack):** IMMA's 19 referenced modules
import cleanly on `torch 2.4.1 / diffusers 0.31.0 / transformers 4.46.3` **except `datasets` and
`torchvision`**; critically `LoRAAttnProcessor` and `diffusers.loaders.AttnProcsLayers` both resolve,
so IMMA's pinned 2023 stack (`diffusers 0.19.0.dev0`, `torch 2.0.1`) is **not** required. UCE's
`requirements.txt` asks `diffusers>=0.34.0` against our 0.31.0, so `venv_uce` upgrades diffusers in
isolation. `vision_relearning` pins `torch==2.4.0` (we have 2.4.1) and needs `wandb`.

---

## 6. Experiment matrix — declared before any new result is inspected

Finite, bounded, with stopping rules. Labels and output directories keep the three categories apart:

| prefix | category |
|---|---|
| `REF_*` | **official reproduction** — authors' code, authors' protocol |
| `ADAPT_*` | **domain adaptation** — authors' method, our domain, labelled as such |
| existing names | **historical project variants** — unchanged |

| id | method | what | arms | seeds | budget | primary outcome |
|---|---|---|---:|---:|---:|---|
| **R1** | IMMA | run `defend_text_to_image_lora.py` at an author-documented setting on SD-1.4, then attack the resulting checkpoint with **our** common protocol | 1 | 3 | ~4 GPU-h | author-protocol checkpoint + max-over-valid H |
| **R2** | Meta-Unlearning | run `src/meta.py` with its own three-pool data | 1 | 3 | ~6 GPU-h | as above |
| **R3** | WDR | **native-domain** sanity check of `weight_dist_reg` on the reference's own classifier setting | 1 | 1 | ~2 GPU-h | does our reading reproduce its published behaviour |
| **R4** | UCE | official code vs our port, **matched inputs and configuration**, compare edited weights and generations | 1 | — | ~1 GPU-h | discrepancy magnitude |

**Stopping rules.** R1/R2: three attack seeds at lr 1e-5, 300 steps, no escalation; arms failing
K1 (retain8 ≥ 0.60) are not attacked and are reported as utility failures. R4: numerical comparison
only, no attack. **Total ≈ 13 GPU-h, ≤ 10 GB storage.**

**Evaluation protocol.** Our common attack (`attack_relearn.py`, unmodified) and `gen_eval.py`;
`retain8`, all-non-target utility and the paired 48-image panel reported **separately**; all attempted
attacks reported with gate outcomes and **both gated and ungated** scores; per-seed values and
denominators shown; bounded intermediate budgets (25/100) alongside the 300-step endpoint.

**Final-prompt discipline.** The existing FINAL set **has been inspected** (`FINDING65`, `FINDING66`)
and is therefore **development data from here on**. It will **not** be reused for tuning and then
described as untouched. A **fresh** final set must be frozen before any new configuration is selected
against it; none of R1–R4 selects a configuration, so none requires it yet.

**Status: no R-experiment has been run.** §7 is written only when cells complete.

---

## 7. Corrections to this manifest, 2026-10-26

Verified against the upstream READMEs and source; the earlier entries were wrong.

**(a) Meta-Unlearning pool meanings — I had `tgt` labelled as the forget pool. It is not.** The
README states it verbatim:

> *"the **hrm** dataset contains the unlearned concept, while both the **irt** and **target**
> dataset contain retain concepts. **irt** dataset means the concept unrelated to unlearned
> concept, and the **target** dataset means the concept related to unlearned concept."*

So: **`hrm` = forget**, **`irt` = unrelated retain**, **`tgt` = related retain**. §2's data-pool row
is corrected. The *equation* reading in `ERRATUM_meta_objective.md` was already right — `F` = `hrm`,
`R` = `irt` ∪ `tgt` — so no objective conclusion changes; the manifest's pool table was the error.

**(b) `--fix_timesteps` is NOT a departure from the published configuration.** The README says
*"We use **fix_timesteps = False** in our paper, and the **fixed_timesteps = True** is just used for
accelerating training."* Our v2 uses random timesteps, i.e. `False`, which **matches the paper**.
Listing it as a mismatch was wrong. **The two-pool vs three-pool difference remains material** and
is unchanged.

**(c) WDR licence — a missing LICENSE file is not an absent licence.** `vision_relearning` has no
standalone LICENSE file, but its **README contains `## License` / `MIT`**. The earlier entry said
"NONE" and drew a redistribution conclusion from it. Corrected to: *README declares MIT; no
standalone LICENSE file.* We continue not to vendor upstream source — for consistency and to keep
upstream unmixed with ours, **not** because of a legal finding. **Meta-Unlearning** has neither a
LICENSE file nor a README licence declaration, and that remains accurate.

**(d) UCE embedding selection — our port does not match upstream, contrary to its own docstring.**
`erase_uce.py` says its selection is *"matching UCE's official implementation."* Verified false:

| | selection |
|---|---|
| upstream `uce_sd_erase.py:34-42` | **one** embedding at `attention_mask.sum()-2` |
| our `erase_uce.py:63-65` | positions **`attention_mask.sum()-2 .. 76`** |

**(e) UTRAP's label.** Described as **our proposed method**. Authorship establishes neither novelty
nor effectiveness, and `FINDING63`/`FINDING74` record that every UTRAP arm tested was reversed.
`erase_wsr.py` remains a **project variant with the opposite regularisation direction**;
`erase_wdr_v2.py` remains a **diffusion-domain adaptation**.
