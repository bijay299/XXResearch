# Upstream source & fidelity manifest — official reference implementations

> **Status pointer:** for the **one current verdict per issue**, read `STATUS_CURRENT_v2.md`. `AUDIT_REPAIR_REGISTER_v1.md` is chronological and records hypotheses that were later refuted — two of them ("IMMA composition unproven", "upstream UCE may have written an unedited model") are **refuted** and must not be quoted as live.

> **⚠ Corrections 2026-10-26:** `FINDING76` §3's probe mechanism check is **overstated** (n=7/3/1, within-arm spread 40×, ordering undetermined — see its §7). `FINDING77` §2's "B kills the accumulation-order hypothesis" is **wrong** — B holds accumulation order fixed and never varied it (see its §6).


**Date:** 2026-10-26 · **Clone root:** `/data/bijaypandey/t2i_unlearn/upstream/` (**git-ignored**, see
§0.1) · **Machine-readable:** `outputs/upstream_manifest.json`

This is the fidelity ledger the implementation task requires: for each method, what upstream
actually is, what our wrapper actually does, and **every departure between them**. It makes no
performance claims.

**Nothing here is a reproduction, and recording a run in §6 does not make it one.** Two cells have
now run upstream code (R1, R4) — both scored on **our** benchmark, which is §6's middle tier, not an
official reproduction. The previous revision's "nothing here is a reproduction *until* a run is
recorded in §6" wrongly implied that executing upstream code is sufficient; it is not.

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

### 2.2 What E2 measured, and what the retention gain is actually attributable to

E2 has now run (λ ∈ {0.5, 1.0}), plus an 18-cell frozen grid (E2g) and an attacker-lr sweep (E2h).
`outputs/gen_meta2_church_lt0.5.json`: as-is target **0.0000**, retain8 **0.9000**, against variant
A's **0.6250** (λ=0.6) and **0.7000** (λ=0.7) at the same 0.0000 suppression — **+0.20 to +0.27**
(`outputs/gen_h80_meta_church_lam{0.6,0.7}.json`; variant A at λ=1.0 collapses to 0.1167).

**Attribution, per `FINDING78` §1a:** the gain comes from restoring Eq. 8's **pairing** — the
restored terms cancel variant A's retain-level penalty — and **not** from the weight of the ζ
coefficient, which is **near-inert and has never been swept**. So the single-ζ collapse recorded in
§2.1 as a departure is a departure in *structure*, and the retention result should not be read as
evidence about ζ's value in either implementation. Nothing in this manifest credits the gain to ζ.

**Robustness is unchanged.** The arm is fully reversed by 300 steps (max-over-valid H 1.0000), and
`FINDING78`'s companion claim to have produced "the first arm to meet both mandate halves" is
**retracted** (audit defect `D-008`): at 25 steps it clears 0.40 at attacker lr 1e-5 (0.0667) and
5e-5 (0.2000) but reaches **0.5333** at lr 1e-4, and max-over-valid is a ratchet. **No arm has met
both halves at any budget.** The retention result survives; the mandate claim does not.

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

**Scale caveat, corrected 2026-10-26 (audit defect `D-017`).** The derivation stands: at the
reference's own α=1 the term's gradient norm is exactly **1/√N = 3.411e-5** against a retain
gradient of ~0.137. **But that is not why α is inert in our port, and the "~4,000×" figure is an
artifact.** Measured as the distance step's update magnitude as a fraction of `lr`: α=1 → **0.104**,
α=10 → **0.537**, α=4000 → **0.998** — so α 1→4000 buys **≈9.6×**, not 4,000×.

The real mechanism is **our optimiser plumbing**: both steps share **one Adam instance**, so the
second-moment buffer `v` is an EMA over both terms and is dominated by the larger (retain) one,
making the distance update ≈ `lr·|g_dist|/√v_retain`. α is therefore inert at **both** ends — below
≈10 because `v` is retain-dominated, above ≈4e3 because **Adam saturates**. The useful window
α ∈ (1, 4000) is real, but its upper bound is a **saturation point, not a frontier**.

**This is a property of our two-step port, not of the published method.** The reference **sums** the
forget and retain gradients into one `opt.step()` (`vision_unlearner.py:411,637-682`); under that
summed schedule the gradient-norm framing *would* be apt. `erase_wdr_v2.py`'s docstring claiming it
alternates "as the reference does" is itself inverted (audit defect `D-016`). So the faithfulness
defect and the coefficient finding are **entangled**, and `wdr2_church_a4000`'s retain8 **0.5917**
measures *our two-step shared-Adam schedule at saturation* — **not** "WDR at gradient parity". Any
"at gradient parity" reading of that number is withdrawn.

**Gradient-norm parity was never a claim of optimality** either — under Adam the update is
preconditioned and per-coordinate normalised. The α sweep remains open, and so does a re-run under
the reference's summed schedule. See `FINDING67` (header + §7) and
`AGENT3_FIDELITY_FINDINGS.md` §4.2–4.3.

**Native-domain sanity check is required** before the diffusion adaptation can be interpreted: the
reference must be run on its own CIFAR-scale setting to confirm our reading of `weight_dist_reg`
reproduces its published behaviour. Queued in §6 as **R3**, **not yet run** — verified 2026-10-26:
no `logs/r3*` and no `outputs/r3_*` exist. R3 is the one declared cell for which "blocked, not
started" is still accurate.

**The diffusion adaptation has since been run (E1), at two α values.** `outputs/gen_wdr2_church_a1.0.json`
and `gen_wdr2_church_a4000.json`: α=1.0 gives target 0.5333 / retain8 0.9500 (the term is inert, so
this is close to doing nothing); α=4000 gives target 0.2000 / retain8 **0.5917**, a K1 failure. The
preregistered grid was α ∈ {0.3, 1.0, 3.0}; what ran was {1.0, 4000}, so **α ∈ (1, 4000) — where a
usable arm would have to be — remains unmeasured.** None of this bears on the published method,
which R3 has not checked.

---

## 4. UCE — fidelity reconciliation

**Reference:** `unified-concept-editing` @ `7c724d9`. **Ours:** `scripts/erase_uce.py`.

Our port implements the closed form

```
W_new = (Σ v* cᵀ + s·(W c_pᵀ)c_p + λW) (c_eᵀc_e + s·(c_pᵀc_p) + λI)⁻¹
```

which was verified algebraically in `ERRATUM_uce_lambda_scale.md`.

**R4 has now run** (four phases, 2026-10-26 11:32→12:45) — the previous revision's "not yet
reconciled against the official code ... queued as R4" is withdrawn. What it establishes, and what
it does not:

| phase | artifact | result |
|---|---|---|
| R4 | `r4_uce_*.json` | **`VOID: true` — DO NOT CITE.** Char-split inputs (`list('church')` → six single letters, recorded as `n_erase_prompts: 6`); an embedding selection matching **neither** implementation; and **no upstream code executed** — two local transcriptions were compared. `FINDING75` rested on it and is **retracted** |
| R4b | `r4b_uce_lamb{0.1,0.5,1000.0}.json` | Upstream **executed** (`executed: true`; the shim binds only the module-level `device`/`torch_dtype` that upstream sets inside `__main__` at `uce_sd_erase.py:116-117`; the `UCE()` body is unmodified). **A** (full pipelines) max rel err **0.384 / 0.322 / 0.108**; **B** (our solver on **upstream's** embedding selection) **1.5e-3 / 3.0e-4 / 1.3e-7**; **C** (fp32 vs fp64, identical matrices) 1.0e-3 / 2.1e-4 / 3.7e-7 |
| R4c / R4d | `r4c_uce_lamb{0.1,0.5}.json`, `r4d_…` | Checkpoints + generations, with and without `preserve_concepts`. λ=0.1 upstream-vs-port label agreement **0/15**; λ=0.5 **2/15**; per-matrix max 0.384 / 0.322 |

**The divergence localises to embedding selection, not the solver** — B is three orders of magnitude
below A at every λ. That is a measured confirmation of §7(d) below, which was read from source
independently.

**The "upstream UCE may have written an unedited model" explanation is REFUTED.** At λ=0.1 the
**upstream** arm scores target **15/15** (`r4c` `generations`: BASE 15/15, UPSTREAM 15/15,
OUR_PORT 0/15), and an earlier revision read that as consistent with upstream's edit having been
**inert**. Measured against the pinned base across all **32** `attn2` K/V matrices, it was not:

| checkpoint | vs BASE max | vs BASE mean | bitwise identical to base |
|---|---:|---:|---:|
| `upstream_church_lamb0.1` | 9.7948e-02 | **5.3620e-02** | **0 / 32** |
| `upstream_church_lamb0.5` | 9.7899e-02 | **5.3593e-02** | **0 / 32** |

Upstream changed **every** matrix — **5.36 % mean / 9.79 % max** of weight norm, **0 of 32** identical
to base — and cross-attention K/V **output** for `"a photo of a church"` moves **43.4 %** (against
58.7 % for our port). **No "upstream" number in R4b/R4c/R4d measures the base model.** A **1.35×**
activation-space gap separates 15/15 from 0/15 in CLIP label, so **behaviour is not proportional to
the numerical gap**, which limits every weight-norm-based behavioural inference in `FINDING75` and
`FINDING77`.

**What is still open is narrower.** `erase_scale` 1→10 is inert to four digits (0.0536 → 0.0536), as
are `lamb` and the guide string (`''` ≡ `' '`); the one effective lever is `expand_prompts=true`
(6 erase prompts), which raises dW/W to **0.0925**. So the live invocation question is whether
upstream should have been given an **expanded erase-prompt set**, not whether it wrote anything.
Independently, the **port** side of R4c-vs-R4d is confounded — the port checkpoint was cached on an
incomplete key, so R4d re-used R4c's 10-preserve build and its recorded `"n_preserve": 0` is false for
that arm (`D-026`) — and R4d **overwrote** R4c's upstream checkpoint, so R4c is no longer reproducible
from disk (`D-031`). **R4d is a required re-run.** R4's status is therefore **"executed; upstream's
edit confirmed real; port-side comparison confounded pending a re-run."**

Note also that `FINDING77` §2's claim that column B "kills the accumulation-order hypothesis" is
**wrong** (its §6): B holds accumulation order **fixed** — it uses upstream's order (`mat1 = lamb*W`
first) on both sides, whereas `erase_uce.py:116-117` puts `lam*W` last — so accumulation order was
never varied and remains **unmeasured**.

Our UCE numbers remain a **project reimplementation** on their prior grounds (§7(d)'s source-level
selection mismatch), **not** on R4's numbers. Note also `D-034`: our port's 0/15 **scatters across
five different classes at 4× the edit size**, and **no retain axis was measured in R4c or R4d at
all**, so "our port erases and upstream does not" does **not** establish that our port erases *well*
— it is equally consistent with broad damage. No superiority claim may rest on R4 without a utility
panel.

Known from our side: λ is measured inert below ~1e3 because `cᵀc` is ill-conditioned. **Measured
(`D-033`):** max eigenvalue **3.2420e+05** for our port's 826-row system (rank 136 in 768 dims) and
**2.2171e+03** for upstream's 11-row system. The **85,850** figure previously carried here traces only
to `ERRATUM_uce_lambda_scale.md` and was never measured; the measurement is **3.8× larger**, so the
erratum's qualitative point is **strengthened**, not weakened. λ=1000 is the first usable SD-1.5
setting.

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
| `REF_*` | **upstream code, our benchmark** — the authors' released code executed faithfully, then measured by **our** `gen_eval`, **our** 12-label closed-set CLIP scorer and **our** prompt pools. **This is not an official reproduction** |
| `ADAPT_*` | **domain adaptation** — authors' method, our domain, labelled as such |
| existing names | **historical project variants** — unchanged |
| *(no prefix exists for)* | **official reproduction** — authors' code **and** authors' protocol, metric and data. **Nothing in this project qualifies**, and nothing should be given a prefix implying it |

**Corrected 2026-10-26.** The previous revision defined `REF_*` as "**official reproduction** —
authors' code, authors' protocol". That is unearned and is withdrawn. `REF_imma_vangogh`, the only
`REF_*` artifact that exists, used upstream's code and the authors' released ESD checkpoint — but
its 100 training images were **self-generated** (`upstream_runs/imma/data/vangogh/`, 11:39–11:43),
and it was scored by **our** `gen_eval` on **our** DEV prompts against **our** closed-set 12-label
CLIP classifier (`logs/r1c_eval.log:20-33`). The code is the authors'; the **protocol is ours**.

**The general rule, applied throughout this manifest:** an upstream-code run on our constructed
benchmark is **not** a reproduction of the paper's reported experiment, and its outcome is **not**
evidence for or against the published method. Three tiers are needed, not two.

| id | method | what | arms | seeds | budget | primary outcome |
|---|---|---|---:|---:|---:|---|
| **R1** | IMMA | run `defend_text_to_image_lora.py` at an author-documented **training** setting on SD-1.4, then attack the resulting checkpoint with **our** common protocol | 1 | 3 | ~4 GPU-h | checkpoint + max-over-valid H |
| **R2** | Meta-Unlearning | run `src/meta.py` with the three-pool **structure**, on pools **we** construct | 1 | 3 | ~6 GPU-h | as above |
| **R3** | WDR | **native-domain** sanity check of `weight_dist_reg` on the reference's own classifier setting | 1 | 1 | ~2 GPU-h | does our reading reproduce its published behaviour |
| **R4** | UCE | upstream code vs our port, **matched inputs and configuration**, compare edited weights and generations | 1 | — | ~1 GPU-h | discrepancy magnitude |

R1's and R2's descriptions were "author protocol" and "its own three-pool data" in the previous
revision. Both are corrected above: the ESD checkpoint is the authors', but the **training images,
the pools, the scorer and the prompts are ours**. `scripts/r2_build_meta_pools.py` builds the pools
from our own prompt lists — `a photo of a church` (hrm); cathedral / chapel / basilica (tgt);
fridge, door, firefly, oven, potted plant, coffee table (irt), 30 train + 30 test each. The
three-pool *structure* is the paper's; the *data* is not.

**Stopping rules.** R1/R2: three attack seeds at lr 1e-5, 300 steps, no escalation; arms failing
K1 (retain8 ≥ 0.60) are not attacked and are reported as utility failures. R4: numerical comparison
only, no attack. **Total ≈ 13 GPU-h, ≤ 10 GB storage.**

**Evaluation protocol.** Our common attack (`attack_relearn.py`, unmodified) and `gen_eval.py`;
`retain8`, all-non-target utility and the paired 48-image panel reported **separately**; all attempted
attacks reported with gate outcomes and **both gated and ungated** scores; per-seed values and
denominators shown; bounded intermediate budgets (25/100) alongside the 300-step endpoint.

**Final-prompt discipline.** The existing FINAL set **has been inspected, and its scores carry a
defect** (`D-022`): `gen_eval_final.py` prompted CLIP with `gen_eval.LAB_KEY` (`:50`) rather than
`LABELS` (`:48`). **Nine of the twelve label strings differ; `church`, `parachute` and `tench` are
spelled identically.** **The evaluator used inconsistent label text. The effect on scores and
conclusions is unresolved pending corrected evaluation** — for `FINDING65`'s depth figures and
`FINDING66`'s ceiling conclusion alike; neither the direction nor the size of the embedding effect is
established. **R1–R4 were scored through `gen_eval.py`'s `LABELS` path**, so no R-cell number was
scored with the inconsistent text. *(`r4c`/`r4d` are a separate case: they scored with
`list(ge.CONCEPTS)`, whose keys equal `LAB_KEY` — `D-006` — so their target accuracies are not
comparable to any other H number in this project, and the same unresolved-effect statement applies to
them.)* The set is
consumed regardless (`FINDING65`, `FINDING66`)
and is therefore **development data from here on**. It will **not** be reused for tuning and then
described as untouched. A **fresh** final set must be frozen before any new configuration is selected
against it; none of R1–R4 selects a configuration, so none requires it yet.

### 6.1 Status — **two of the four cells have run** (corrected 2026-10-26)

The previous revision said *"Status: no R-experiment has been run."* That was false when written.
Verified from artifacts and mtimes:

| id | status | evidence |
|---|---|---|
| **R1** | ✅ **RAN.** ESD checkpoint downloaded (`upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt`, 3,438,307,109 B, 11:36); 100 training images generated (11:39–11:43); `defend_text_to_image_lora.py` trained **5000/5000** (11:59→12:21), saving `REF_imma_vangogh/imma_unet_xatten_layer.pt`; eval composed base+ESD+IMMA (`overlaid 160/160 IMMA tensors`) and scored it (12:39→12:44). **The 3 attack seeds were never run.** | `logs/r1b_imma_train.log`, `logs/r1c_eval.log`, `outputs/gen_REF_imma_vangogh.json` |
| **R2** | ⏸ **PARTIAL.** Three pools built 12:39→12:47 (`[r2pools] done`), 30 train + 30 test each. A first attempt failed `rc=1` — `ModuleNotFoundError: No module named 'pandas'` at `r2_build_meta_pools.py:9` (12:21). **`src/meta.py` has never been invoked.** | `logs/r2_meta.log`, `upstream_runs/meta/dataset/{hrm,irt,tgt}/{train,test}` |
| **R3** | ❌ **NOT STARTED.** No `logs/r3*`, no `outputs/r3_*`. | — |
| **R4** | ✅ **EXECUTED, four phases. Upstream's edit confirmed real; the port-side comparison is confounded and R4d is a required re-run** (`D-026`, `D-031`). See §4. | `outputs/r4{,b,c,d}_uce_*.json` |

**R1's recorded outcome, with interpretation withheld.** `outputs/gen_REF_imma_vangogh.json`: target
(`van_gogh`) **0.8667**, all-non-target **0.0424** (n=11), legacy_retain8 **0.0333** (n=8); eleven of
twelve concepts score ≤ 0.20.

**The "checkpoint composition unproven" explanation is REFUTED. The composition is valid.**
Replicating `r1_eval_imma.py`'s matching loop offline gives **exact=160, fuzzy=0, reshaped=0,
skipped=0**. Auditing `adapters/REF_imma_vangogh/unet.pt`: **160/160** IMMA tensors byte-equal to
`half(IMMA)`, **526/526** others byte-equal to `half(ESD)`, 0 unexplained, 0 non-finite, exactly 160
tensors differing from ESD (80 `attn1` + 80 `attn2`, IMMA's own trained scope), and the **ESD delta
is not inverted**. The composition is byte-semantically identical to upstream's own
(`eval/text2image.py:65-69`, `train_text_to_image_lora.py:458-462`).

**`outputs/gen_REF_imma_vangogh.json` stays quarantined on different grounds:** the closed-set 12-way
CLIP argmax is **inapt** here — eight non-targets at *exactly* 0.000 with the target at 0.867 is a
single-label collapse of the metric, not a spread around the 1/12 chance line — and **the
configuration is not one the authors report** (IMMA never evaluates an immunised model standalone,
`upstream/IMMA/README.md:146-151`). It must **not** be cited as an IMMA result, and specifically not
as "IMMA destroys utility" — because the measure and the configuration are inapt, **not** because the
checkpoint is malformed.

Two remaining explanations are live and neither has been tested: IMMA behaving as designed (it
*immunises* against relearning rather than erasing, and here sits atop an ESD-erased UNet), and
inheritance from the **ESD checkpoint** — all **686** tensors of
`upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt` differ from pinned base SD-1.4, resnet blocks
reaching 0.20 relative, so it is not the cross-attention-only erasure "ESD-x1" denotes (`D-019`). The
decisive control — the ESD checkpoint scored **alone** on the same 12 DEV concepts — **has not been
run**, and the attack seeds were never run either. **No conclusion about IMMA-the-published-method
follows from this cell**, in either direction. Two further limits on the cell: it ran **5000** steps
rather than the documented 1000 (2500 ascent steps against the README's 500) and 3 of its 100 training
images are NSFW-blanked black frames (`D-021`).

This is consistent with §0's standing rule: a fidelity ledger makes **no performance claims**.

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

---

## 8. Change log — 2026-10-26 (Agent 6, provenance audit)

Every claim in this manifest was re-derived from `outputs/upstream_manifest.json`,
`outputs/imma_scope_fidelity.json`, the `outputs/r4*` artifacts, the upstream READMEs and source, or
file mtimes. Full traceability in `AGENT6_PROVENANCE_FINDINGS.md` §1.3. Banners were judged
insufficient: the incorrect sentences were rewritten.

### 8.1 Re-verified and found correct (unchanged)

| § | claim | verified against |
|---|---|---|
| §0 | all five commits, dates and `python_files` counts; "three audited revisions resolved exactly as requested" | `outputs/upstream_manifest.json` — three non-null `requested_pin`, each equal to its `resolved_commit` |
| §0 | `vision_relearning`: **README declares MIT, no standalone LICENSE file** | `ls -a upstream/vision_relearning` → no LICENSE/COPYING; `README.md:54-56` → `## License` / `MIT` |
| §0.2 | FRAG: non-empty, **0 Python files** | `upstream_manifest.json` `FRAG.python_files: 0` |
| §1.1 | every cited line number and mechanic of `defend_text_to_image_lora.py` | upstream source at `2f55041` |
| §1.2 | 859.5M / 93.5M (10.88%) / 49.6M (5.77%) / 44.0M (5.11%) / **2.13×**, 160 and 80 tensors | `outputs/imma_scope_fidelity.json` — exact, including tensor counts |
| §2 | `hrm` = unlearned concept, `irt` = **unrelated** retain, `tgt` = **related** retain | `upstream/Meta-Unlearning/README.md:18`, quoted verbatim and accurately |
| §2 | the README states the paper used **`fix_timesteps = False`**, so our random-timestep v2 **matches** the paper and this is **not** a departure | `README.md:43` — accurate |
| §5.1 | the verbatim first-attempt environment failures | `logs/upstream_envs.log` |
| §7(a)(b)(c) | all three corrections | re-verified at source; all three hold |
| §7(d) | upstream selects **one** embedding at `attention_mask.sum()-2`; ours selects `sum()-2 .. 76` | source, **and now independently corroborated by measurement**: R4b column B drops to 1.5e-3 when upstream's selection is used on both sides |
| §7(e) | UTRAP labelled "our proposed method", authorship ≠ novelty or effectiveness | holds |

### 8.2 Corrected

| § | what changed | why |
|---|---|---|
| header | "Nothing here is a reproduction **until a run is recorded in §6**" rewritten | it implied that executing upstream code suffices. R1 and R4 have run and are still not reproductions |
| §2.2 | **new** — E2's measured outcome, with the retention gain attributed to Eq. 8's **pairing** rather than to ζ's weight, and `FINDING78`'s mandate claim recorded as **retracted** (`D-008`) | §2 described v2's objective but recorded no result; and the gain must not be read as evidence about ζ, which is near-inert and never swept (`FINDING78` §1a) |
| §3 | R3 confirmed not started; the **E1 results added** (α ∈ {1.0, 4000}, K1 FAIL at 0.5917, α ∈ (1,4000) unmeasured) | the section implied no WDR adaptation had been measured |
| §3 | **scale caveat rewritten** (`D-017`): the ~4,000× factor withdrawn (measured ≈**9.6×**), the mechanism reattributed to our **shared-Adam two-step schedule**, the window's upper bound relabelled a **saturation point**, and "WDR at gradient parity" withdrawn as a reading of retain8 0.5917 | the derivation ‖∇L‖ = α/√N is correct but was **not** the cause. The reference **sums** both terms into one step, so this is a defect in our port (entangled with `D-016`), not a coefficient-scale property of "From Dormant to Deleted" |
| §4 | "**Not yet reconciled against the official code** ... queued as R4" → the four R4 phases, with A/B/C magnitudes | R4 ran 11:32→12:45 |
| §4 | **R4 phase 1 marked `VOID`**, with its three defects named, and `FINDING75`'s retraction recorded | the artifact self-flags `VOID: true` / "DO NOT CITE"; citing it would propagate a retracted number |
| §4 | **"no port-fidelity conclusion drawn"** paragraph added | upstream's λ=0.1 arm scores target 15/15, read at the time as consistent with an inert edit. **That reading has since been refuted by measurement and the paragraph is replaced** — see §8.1 |
| §4 | `FINDING77` §2's accumulation-order error recorded inline | the banner named it; the body did not, and §4 is where a reader would act on it |
| §4 | the 85,850 eigenvalue marked **single-sourced** | no artifact under `outputs/` recomputed it. **It has since been measured at 3.2420e+05** (`D-033`) and §4 now carries the measurement — see §8.1 |
| §6 | `REF_*` redefined from "**official reproduction** — authors' code, authors' protocol" to "**upstream code, our benchmark**", with a fourth row stating that nothing qualifies as an official reproduction | unearned. `REF_imma_vangogh` used self-generated training images and our scorer and prompts (`logs/r1c_eval.log:20-33`) |
| §6 | R1 relabelled off "author protocol"; R2 off "its **own** three-pool data", with the actual prompt lists given | the images, pools, scorer and prompts are all ours |
| §6.1 | "**Status: no R-experiment has been run**" → per-cell status table with evidence | false when written: R1 complete, R4 four phases, R2 pools built |
| §6.1 | R1's outcome recorded **with interpretation explicitly withheld** | the number should be visible, but its three candidate explanations are untested and the attack seeds were never run |

### 8.3 Defects in artifacts this document depends on — reported, not fixed

1. **`outputs/upstream_manifest.json` contradicts this manifest on the WDR licence.** The JSON says
   `vision_relearning.license: "NONE FOUND"`; §0 and §7(c) say *README declares MIT, no standalone
   LICENSE file*. **The markdown is right** (verified at source). The JSON is the machine-readable
   companion this document advertises, so the two must be reconciled — by whoever owns the JSON.
2. **`outputs/upstream_manifest.json` records `"pinned_as_requested": true` alongside
   `"requested_pin": null`** for `unified-concept-editing` and `FRAG`. Nothing was requested for
   either, so the flag asserts a vacuous truth and would mislead an automated consumer. §0's prose
   handles it correctly ("UCE and FRAG had no revision specified"); the schema does not.
3. **`outputs/registry_v1.json` is stale by ~150 JSON files**, so `TABLES_v1_generated.md` is too.
   The registry covers **696 of the 808** attack records now on disk (**112 absent**) and 1,278 of
   1,424 JSON files; **≥11 defence builds are missing**, so **the arms behind `FINDING71`–`FINDING78`
   appear in no generated table**, including `meta2_church_lt0.5`, the project's closest approach to
   the mandate. Not regenerated here: `scripts/registry_v1.py` sha256s many GB and `FREEZE_ACTIVE` is
   in force.

**Not changed, deliberately:** the `FINDING76`/`FINDING77` correction banner, independently
re-verified against `r4b_uce_lamb*.json` column B and found accurate; §0.1's non-redistribution
policy, which §7(c) already states rests on consistency rather than on a legal finding; every
fidelity verdict (IMMA reimplementation, Meta v2 ours, WDR adaptation, UCE reimplementation, GIFT
unsubstituted), all of which verified correct and none of which R1/R4 overturn.

### 8.1 Documentation reconciliation, 2026-10-26 — two refuted explanations removed

| what changed | why |
|---|---|
| §4's **"upstream's edit may have been inert"** paragraph **replaced with the refutation** | **REFUTED by measurement:** all **32** `attn2` K/V matrices changed, **5.36 % mean / 9.79 % max**, **0/32** identical to base, **43.4 %** change in K/V output. The open questions are erase-prompt expansion and the **confounded port side** (`D-026` re-run, `D-031` lost checkpoint) |
| §6's **"artifact of `r1c_eval`'s tensor composition"** removed as a live explanation; the composition recorded as **valid** | **REFUTED by direct audit:** `exact=160 fuzzy=0 reshaped=0 skipped=0`; 160/160 and 526/526 tensors byte-equal to their sources; ESD delta **not** inverted. `gen_REF_imma_vangogh.json` stays quarantined because the closed-set metric is **inapt** and the configuration is **not one the authors report** (`D-009`); `D-019`'s ESD checkpoint is the live candidate and **its control run is outstanding** |
| §4's λ-inertness eigenvalue **corrected to the measured 3.2420e+05** | the 85,850 figure was single-sourced to `ERRATUM_uce_lambda_scale.md` and never measured; the measurement is **3.8× larger** (`D-033`) |
| §4 and §6's **`D-022` statements de-directionalised** | the label-text mismatch is **established** (nine of twelve strings differ; `church`/`parachute`/`tench` identical); its **direction and size are not**, pending paired re-scoring. "Degraded distractors" and "target hits biased upward" are withdrawn as inferences |
| §7's staleness item given exact coverage, and the absent findings named | "stale by 150 files" did not say **which** arms a reader would not find |
