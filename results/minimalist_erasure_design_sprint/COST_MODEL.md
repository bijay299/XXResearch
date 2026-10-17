# Cost model — native reference, exact ordinary alternatives, and truncation

**NO EXPERIMENTS. NO PROFILING.** Every number here is either a **symbolic expression**, a
**source-supported configuration value**, or an **arithmetic consequence** of the two.

> **Nothing in this file is a measurement.** There are no seconds, no GPU-hours and no
> measured speedups. `F`, `c_b`, `D`, `F_dev` and all hardware constants are **unmeasured**
> and explicitly labelled. Any ratio below is a *predicted* work ratio under stated
> assumptions, and **must not be quoted as an observed speedup.** No runtime profiling was
> authorized or performed.

---

## 1. Units and unmeasured constants

| symbol | meaning | status |
|---|---|---|
| `F` | one FLUX.1-schnell transformer forward at the training configuration | **unmeasured unit**; all costs are expressed in it |
| `c_b` | cost of one reverse sweep over one step's graph, in units of `F` | **unmeasured**; carried symbolically, instantiated at `c_b ∈ {1, 2}` |
| `D` | one VAE decode | **unmeasured**; tracked separately, never folded into `F` |
| `F_dev` | one FLUX.1-**dev** forward (data preparation only) | **unmeasured**; assumed `≈ F` (same family/size), labelled as an assumption |
| `E_vae` | one VAE encode | **unmeasured**; negligible, tracked for completeness |

**Hardware assumptions:** single device (the paper's §3.7 limitation — *"lacks support for
multi-GPU training"* — and `cfg.trainer.device: "cuda:0"`); paper App. H.1 states 1× H100.
No memory capacity, bandwidth or occupancy figure is assumed or asserted.

**Training configuration** — the forward cost `F` is set by a resolution that is **implicit**:
`configs/flux.yaml` has **no `img_size` key**, so `getattr(cfg.data, "img_size", None)`
returns `None` (`scripts/train.py:274-275`) and
`inference_preparation_phase` falls back to `default_sample_size × vae_scale_factor`
(`models/flux.py:216-217`) — i.e. **1024×1024** for FLUX, giving `(1024/16)² = 4096` image
tokens plus `max_sequence_length = 512` text tokens (`models/flux.py:213`). By contrast
`configs/sdxl.yaml:12` *does* set `img_size: 512`. So `F` is a 12B DiT forward over ≈4608
tokens. **Deduction from source defaults, not a stated claim.**

---

## 2. Source-supported parameters

All from `configs/flux.yaml` at the pinned commit.

| quantity | value | source |
|---|---|---|
| `K` sampler steps in training | **5** | `:24` `num_intervention_steps: 5` (and `:80`, debug override) |
| epochs `E` | 5 | `:15` |
| `data.size` | 40 → **80 rows** after forget/retain merge | `:6`; `promptdataset.py:159-179` |
| `|D|` after CLIP filtering | **≈ 64** | `filter_ratio: 0.9` (`:10`) → ≈8 forget rows exceed the 0.9 quantile, each removing itself **and** its successor (`utils_train.py:249-252`). **Data-dependent; we did not compute the CLIP scores** |
| `N_iter` gradient-computing iterations | `E·|D|` = **320** | loop `train.py:259-260` |
| `N_upd` optimizer updates | **80** | `step()` when `(idx·batch_size) % 4 == 0` (`:335`), `batch_size: 1` (`:7`), `accumulate_grad_batches: 4` (`:35`) |
| `plot_interval` | 4 | `:59`, `:79` |
| `β` | 0.1 | `:16` |

**Disagreements with the paper (App. H.1, Table 6), carried as uncertainty, not resolved:**
batch size **4** vs `1`; `β` **0.01** vs `0.1`; **400** training steps vs `N_upd ≈ 80`. If
the paper's 400 updates is authoritative, `N_iter` is ~5× larger and all *totals* below
scale, while all *ratios* are unchanged.

---

## 3. Per-iteration accounting

Phases as implemented (`scripts/train.py`):

| phase | work | cost |
|---|---|---|
| **A** forward, `@torch.no_grad()`, stores all `z_k` (`:68-98`) | `K` transformer forwards | `K·F` |
| **B** terminal loss + `dL/dz_K` (`:282-300`) | MSE on **latents**; `output_type="latent"` ⇒ **no VAE decode in the training graph** (`models/flux.py:367-371`) | `≈ 0` |
| **C** reverse loop (`:306-333`) | per step: **1 recomputed forward** (`:313`) + **2 separate reverse sweeps** (`:320`, `:333`) | `K·F·(1 + 2c_b)` |

```
C_ref  =  K·F·(2 + 2c_b)
```

### 3.1 Teacher / reference / regularization work: zero per iteration

- **Teacher `ℱθ′`:** computed **once per dataset row and cached to disk**
  (`promptdataset.py:224-316`, safetensors; only the last 2 latents stored, `:54, 311-312`).
  **No teacher forward occurs during training** — it is entirely a one-time cost (§5).
- **Mask sampling:** `torch.rand` + `σ` + affine + `clamp` per hooked module per call
  (`utils/utils.py:213-223`) — elementwise, `O(#neurons)`. Negligible vs. matmuls.
- **Regularization:** **none in the trained loss.** `calculate_reg_loss` is never called
  from `scripts/train.py` (`DERIVATION.md` §1.2d). Cost zero because the term is absent.
- **Selection/estimation overhead, caches, refreshes, warm-up:** zero for the reference, and
  for SSMG these are exactly the costs that sink it ([`ALGORITHM.md`](ALGORITHM.md) §1.3(iii)).

### 3.2 Instrumentation — accounted separately, and it is large

`scripts/train.py:344-414`, fires when `(idx·batch_size) % plot_interval == 0`, i.e. **every
4 iterations**. Per firing:

- 2 prompt groups (3 validation prompts — `VALIDATION_PROMPT["concept"]`, a 3-element set at
  `unlearn_template.py:249-254` — plus 1 train prompt) × 2 functions
  (`save_image_seed` = continuous mask, `save_image_binarize_seed` = binarized mask)
  = **8 images**;
- each image: `K = 5` forwards **plus a VAE decode** (`save_image` default
  `output_type="pil"`, `utils_train.py:30-45`);
- plus `hooker.save()` × 3 hookers (`:412-413`) — disk I/O.

```
C_log  =  (8·K·F + 8·D) / 4  =  10·F + 2·D   per iteration
```

This runs in both logger modes: under `debug: true` (`:77`) `logger.type` becomes `csv`
(`:81`), which writes the images to disk rather than suppressing them.

**At `c_b = 1` instrumentation is 50% of the gradient cost; at `c_b = 2`, 33%.** It is pure
logging and contributes nothing to the gradient.

---

## 4. The four regimes, symbolically and numerically

```
C_ref       = K·F·(2 + 2c_b)                   native (two sweeps + recompute)
C_A1        = K·F·(2 +  c_b)                   fuse the two sweeps        — EXACT
C_A2        = K·F·(1 +  c_b)                   retain graph, no recompute — EXACT, = FLOOR
C_trunc(τ)  = K·F   +   τ·F·c_b                DRaFT-K style, τ ≤ K       — BIASED
C_fwd_floor = K·F                              irreducible: z_K is needed for the loss
```

| | `c_b = 1` | `c_b = 2` | exact? |
|---|---|---|---|
| `C_ref` | **20 F** | **30 F** | yes (of the relaxed surrogate) |
| `C_A1` | 15 F | 20 F | **bit-identical to `C_ref`** |
| `C_A2` | 10 F | 15 F | **exact; attains the floor** |
| `C_trunc(τ=1)` | 6 F | 7 F | **biased** |
| `C_fwd_floor` | 5 F | 5 F | — |

### 4.1 The exact-gradient floor, and why `A2` attains it

Computing the exact `∇_λ` of the relaxed objective requires, for each step whose direct term
is included: (i) one evaluation of `f_k` to materialize activations, and (ii) one reverse
sweep through `f_k`. The sweep in (ii) yields `∂L/∂z_k` **and**, by the lemma
(`DERIVATION.md` §4 (†)), **all** mask gradients at that step as an elementwise byproduct.
Hence `C_floor = K·F·(1 + c_b) = C_A2`. **There is no exact-gradient headroom below `A2`,
and in particular none that an estimator could reach.**

### 4.2 Peak memory — the real gate on `A2`

| regime | peak | note |
|---|---|---|
| `C_ref`, `C_A1` | weights + **one** step's activation graph + `K` stored latents | the paper's §3.7 goal: *"constant memory complexity regardless of the step"* |
| `C_A2` | weights + **`K = 5`** steps' activation graphs | **≈5× activation memory.** Whether this fits is **unmeasured** and is the binding constraint on `A2` |
| **data preparation** | **two 12B models resident simultaneously** — FLUX.1-schnell **and** FLUX.1-dev (`promptdataset.py:78-84`, freed at `:89-90`), then CLIP (`utils_train.py:225`) | **this, not training, is the pipeline's global peak memory** |

**`A1` is unconditionally free — same gradient, same memory, 25–33% less work. `A2` is free
only if the activation memory fits, which we cannot determine without profiling.** If it
does not fit, `A1` is the only no-cost exact win and the floor is not reachable.

### 4.3 The Amdahl ceiling on any gradient-side improvement

The terminal loss needs `z_K`, so the `K·F` forward is **irreducible**. Even a
**free** gradient would give at most:

- vs. `C_ref`: `20F → 5F` = **4.0×** (`c_b=1`) — and only `1.33×` beyond `A2`;
- with instrumentation left in: `30F → 15F` = **2.0×** only.

**A hypothetical zero-cost gradient estimator is worth at most 1.33× over ordinary correct
autograd usage.** That ceiling, computed before any design work, is what makes the
estimator question unattractive independent of the dominance lemma.

---

## 5. One-time costs, and the whole-run picture

| one-time item | cost | source |
|---|---|---|
| concept latents, all 80 rows | `80·K·F` = **400 F** + `80·D` | `promptdataset.py:238-241, 300-307` |
| **FlowEdit targets**, 40 forget rows, on **FLUX.1-dev** | `40 · ≈(24·2) · F_dev` ≈ **1920 F** + `40(E_vae + D)` | `:250-276`: `T_steps=28`, `n_max=24`, `n_avg=1`, src **and** tar guidance evaluations per active step |
| CLIP filtering over 80 rows | `≪ F` each; GPU memory only | `utils_train.py:217-261` |
| dummy init generation, initial/final image saves | `≈ 3F + 2·8·K·F` ≈ 83 F | `train.py:228, 197-208, 422-441` |
| **total preparation** | **≈ 2320 F** (+≈120 `D`, 40 `E_vae`) | |

**Whole run** (`N_iter = 320`, `c_b = 1`, VAE decodes tracked separately):

| regime | training | +prep | **total** | composition |
|---|---|---|---|---|
| **as published** | `320·(20+10)F` = 9600 F | 2320 F | **11 920 F** | gradient **54%**, instrumentation **27%**, preparation **19%** |
| **A1+A2+A4** (exact) | `320·10F` = 3200 F | 2320 F | **5 520 F** | → `2.16×` less total work, **zero bias** |
| **+ truncation `τ=1`** (biased) | `320·6F` = 1920 F | 2320 F | **4 240 F** | → `2.81×` |

At `c_b = 2`: 15 120 F → 7 120 F (`2.12×`, exact) → 4 560 F (`3.32×`).

**Read the composition row.** In the published configuration, **46% of total work is neither
the gradient nor anything an estimator touches.** The two largest reducible items are
**logging** and a **FlowEdit data-preparation pass on a second 12B model** — and the latter
is also where peak memory lives.

---

## 6. Break-even conditions

Let `ρ` = per-run work ratio (new / baseline) and `ν` = factor by which the number of
optimization updates must grow to reach **comparable** deletion, retained utility, rendering
quality and robustness. Net win requires

```
ν · ρ  <  1          ⟺          ν  <  1/ρ
```

**For `A1` and `A2`: `ν = 1` exactly.** They produce the *same* gradient, so the
optimization trajectory is unchanged and `ρ < 1` transfers to total cost with **no
convergence uncertainty**. This is the whole reason they are the right answer and an
estimator is not.

**For truncation, measured against the exact fixes** (the honest baseline — not against the
unfixed reference):

| `c_b` | `ρ = C_trunc / C_A1A2A4` | **update-count budget `ν < 1/ρ`** |
|---|---|---|
| 1 | 4240 / 5520 = **0.768** | **`ν < 1.30`** |
| 2 | 4560 / 7120 = **0.640** | **`ν < 1.56`** |

So at `c_b = 1`, **truncation must not require more than 30% additional updates** or it is a
net loss. Charging the full-gradient calibration/warm-up calls that any adaptive scheme
needs (`W = 20` warm-up iterations plus a refresh every `R = 50` ⇒ ≈26 extra
full-cost iterations ⇒ `≈ +106 F`) tightens the budget to **`ν < 1.23`**.

**`ν` is unknown and we have no evidence for it.** DRaFT's own reported finding is that
truncation depth trades against *result quality*, not merely speed
([`ALGORITHM.md`](ALGORITHM.md) §4) — i.e. `ν > 1` is the expected case, and it may not
converge to comparable quality at any `ν`. **A 23–30% margin is not a defensible place to
spend a biased estimator.** Carrying this uncertainty forward, the truncation route has **no
supported net saving**, and the `2.81×` figure above must not be read as achievable.

### 6.1 Why SSMG has no break-even at all

`C_SSMG = K·F + (K − min(S))·(1 + c_b)·F + O(ε)`
([`ALGORITHM.md`](ALGORITHM.md) §1.3(ii)). Unbiasedness requires `P(min(S) = 0) > 0`, so in
expectation `C_SSMG → C_A2` while `Var > 0` ⇒ `ν ≥ 1`. Therefore `ν·ρ ≥ 1`: **SSMG cannot
break even for any `q`, `S₀`, `R` or `W`.** No condition on the unmeasured constants
changes this, because the result does not depend on `F`, `c_b` or `D`.

---

## 7. What these costs do **not** establish

- **No measurement.** No profiling was run. `F`, `c_b`, `D`, `F_dev`, the 1024×1024
  resolution inference, `|D| ≈ 64`, and the FlowEdit step count are **derived or assumed**,
  and `ν` is **unknown**.
- **Work ratios are not wall-clock ratios.** They ignore memory-bandwidth limits, kernel
  efficiency, occupancy, the `K×` activation-memory feasibility of `A2` (§4.2), host-side
  overheads, and disk I/O from per-interval checkpoint saves.
- **Cheaper updates are not cheaper training.** Only `A1`/`A2` escape this, by being exact.
- **No cost statement is a semantic statement.** Equal gradients imply an equal optimization
  trajectory; they imply **nothing** about concept absence, retained utility, rendering
  quality or robustness — those are properties of the **binarized** mask
  (`DERIVATION.md` §5.4), and no deletion, utility, quality or robustness claim appears
  anywhere in this sprint.
- **Totals, not ratios, depend on the unresolved `N_upd`** (80 from config vs. 400 from the
  paper) and on the unresolved target construction (§2, `DERIVATION.md` §1.2a).
