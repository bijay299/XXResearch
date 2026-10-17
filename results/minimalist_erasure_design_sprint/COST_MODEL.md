# Cost model — corrected

**NO EXPERIMENTS. NO PROFILING.** Every quantity here is a **symbolic expression**, a
**source-supported configuration value**, or an **arithmetic consequence** of the two.

> **Nothing in this file is a measurement.** No seconds, no GPU-hours, no measured speedups.
> `F`, `F_txt`, `c_state`, `c_mask`, `c_joint`, `D`, `F_dev`, `ν` and all hardware constants
> are **unmeasured** and labelled. Any ratio is a *predicted* work ratio under stated
> assumptions and **must not be quoted as an observed speedup.** **Simplified arithmetic is
> not a measurement.**

> **Amended 2026-10-17 — see [`CORRECTIONS.md`](CORRECTIONS.md).** Withdrawn from the
> original publication: the **exact-gradient "floor" theorem** (X5), the **`1.33×` ceiling**
> (X9), the **`ν < 1.23–1.56` break-even** computed by multiplying shared preparation by
> update inflation (X10), **`A1` bit-identity / fixed runtime saving / identical trajectory**
> (X11), **`A2` as an exact substitute for the native estimator** (X12), the
> **`54%/27%/19%` composition** (X14), **preparation as global memory peak** (X15), the
> **whole-run ratios `2.16×/2.81×/2.12×/3.32×`** (X16), and **"only totals scale"** (X17).
> Cost terms omitted in the original are added or explicitly excluded (X18).

---

## 1. Units and unmeasured constants

| symbol | meaning | status |
|---|---|---|
| `F` | one FLUX.1-schnell transformer forward at the training configuration | **unmeasured unit** |
| `F_txt` | **prompt encoding** per training forward preparation — T5 + CLIP text encoders, invoked by `encode_prompt` inside `inference_preparation_phase` (`models/flux.py:250`) | **unmeasured; newly accounted (X18).** The original model ignored this entirely |
| `c_state` | reverse sweep cost for **state adjoints only**, in units of `F` | **unmeasured** |
| `c_mask` | reverse sweep cost for **mask terms only** | **unmeasured** |
| `c_joint` | reverse sweep cost returning **both** from one graph | **unmeasured**; `c_joint ≤ c_state + c_mask`, and the original model's single `c_b` collapsed all three (X18) |
| `D` | one VAE decode | **unmeasured** |
| `E_vae` | one VAE encode | **unmeasured** |
| `F_dev` | one FLUX.1-**dev** forward (data preparation only) | **unmeasured**; assumed `≈ F`, labelled |
| `ν` | update-count factor to reach **comparable** deletion, retained utility, rendering quality and robustness | **UNKNOWN**; not derivable here |

**Simplified instantiation.** Where a single number is needed below, we write `c_b` for a
common reverse-sweep constant with `c_b ∈ {1, 2}`, **as a simplification carried over from
the original model**, and flag it as such. Expressions distinguishing `c_state`/`c_mask`/
`c_joint` are the accurate form.

**Hardware assumptions:** single device (paper §3.7: *"lacks support for multi-GPU
training"*; `cfg.trainer.device: "cuda:0"`); paper App. H.1 states 1× H100. No capacity,
bandwidth or occupancy figure is assumed or asserted.

**Training resolution** (unchanged, a deduction): `configs/flux.yaml` has **no `img_size`
key**, so `getattr(cfg.data, "img_size", None)` returns `None` (`scripts/train.py:274-275`)
and `inference_preparation_phase` falls back to `default_sample_size × vae_scale_factor`
(`models/flux.py:216-217`) — **1024×1024** for FLUX, giving `4096` image tokens plus
`max_sequence_length = 512` text tokens (`models/flux.py:213`). `configs/sdxl.yaml:12` *does*
set `img_size: 512`. **Deduction from source defaults, not a stated claim.**

---

## 2. Source-supported parameters

All from `configs/flux.yaml` at the pinned commit. **These are retained unchanged.**

| quantity | value | source |
|---|---|---|
| `K` sampler steps in training | **5** | `:24`, and `:80` debug override |
| epochs `E` | 5 | `:15` |
| `data.size` | 40 → **80 rows** after forget/retain merge | `:6`; `promptdataset.py:159-179` |
| `|D|` after CLIP filtering | **≈ 64** | `filter_ratio: 0.9` (`:10`), `utils_train.py:249-252`. **Data-dependent; CLIP scores not computed** |
| `N` gradient-computing iterations | `E·|D|` = **320** | `train.py:259-260` |
| `N_upd` optimizer updates | **80** | `step()` when `(idx·batch_size) % 4 == 0` (`:335`), `batch_size: 1`, `accum: 4` |
| `plot_interval` | 4 | `:59`, `:79` |
| `β` | 0.1 | `:16` |

**Unresolved disagreements with paper App. H.1 / Table 6**, carried as uncertainty: batch
size **4** vs `1`; `β` **0.01** vs `0.1`; **400** training steps vs `N_upd ≈ 80`.

**Corrected (X17):** if the paper's 400 updates is authoritative, `N` is ~5× larger. Because
one-time preparation `P` is **fixed**, **whole-run ratios and composition also change** —
not only totals. The original statement that "all ratios are unchanged" is withdrawn.

---

## 3. Per-iteration accounting, by stochastic reference

**Each expression names the reference it concerns (X12).** Two references appear:

- **R_native** — fresh mask draws on every recompute (`DERIVATION.md` §5.3).
- **R_coherent** — one coherent draw per step used for both forward and backward.

Phases as implemented (`scripts/train.py`):

| phase | work | cost |
|---|---|---|
| **P0** prompt encoding + latent prep + timesteps (`:269` → `models/flux.py:246-291`) | `encode_prompt` (T5 + CLIP), `prepare_latents`, `retrieve_timesteps` | **`F_txt` + `O(ε)`** — *newly accounted* |
| **A** forward, `@torch.no_grad()`, stores all `z_k` (`:68-98`) | `K` transformer forwards | `K·F` |
| **B** terminal loss + `dL/dz_K` (`:282-300`) | MSE on **latents**; `output_type="latent"` ⇒ no VAE decode in the training graph | `≈ 0` |
| **C** reverse loop (`:306-333`) | per step: 1 recomputed forward (`:313`) + **two separate** sweeps (`:320`, `:333`) | `K·F·(1 + c_state + c_mask)` |

```
C_native  =  F_txt  +  K·F·(2 + c_state + c_mask)        [reference: R_native]
          =  F_txt  +  K·F·(2 + 2c_b)                     (simplified, c_state = c_mask = c_b)
```

### 3.1 Teacher, regularization, and selection work

- **Teacher `ℱθ′`:** computed **once per dataset row and cached to disk**
  (`promptdataset.py:224-316`). **No teacher forward per iteration.** *(Retained.)*
- **Mask sampling:** elementwise, `O(#neurons)` per hooked module per call. Small.
- **Regularization:** **absent from the trained loss**; `calculate_reg_loss` is never called
  from `scripts/train.py`. Cost zero because the term is absent. *(A source fact — not a
  paper/code divergence; the paper's Eq. 16 also has no such term, X20.)*
- **Selection/estimation overhead** for any sampling scheme: **unquantified**, carried as an
  explicit unknown, not as a disqualifying argument.

### 3.2 Explicitly excluded or unquantified (X18)

Stated rather than silently omitted: **optimizer work** (AdamW over all `λ` vectors each
update); **I/O** (`hooker.save()` per logging event, `:412-413`); **initialization**
(`:228` dummy generation; `:197-208` initial images); **batching effects**; **VAE decodes**
outside the training graph (tracked as `D`, never folded into `F`); **host-side overheads**;
**kernel efficiency, bandwidth and occupancy**. None is modelled numerically.

### 3.3 Instrumentation — accounted separately

`scripts/train.py:344-414`, when `(idx·batch_size) % plot_interval == 0`, i.e. every 4
iterations. Per firing: 2 prompt groups (3 validation prompts — `VALIDATION_PROMPT["concept"]`
is a 3-element set, `unlearn_template.py:249-254` — plus 1 train prompt) × 2 functions
(continuous and binarized mask) = **8 images**, each `K = 5` forwards **plus a VAE decode**
(`save_image` default `output_type="pil"`) **plus its own prompt encoding**.

```
C_log  =  ( 8·K·F  +  8·D  +  8·F_txt ) / 4  =  10·F + 2·D + 2·F_txt   per iteration
```

Runs in both logger modes: under `debug: true` (`:77`) `logger.type` becomes `csv` (`:81`),
which writes images to disk rather than suppressing them.

---

## 4. The regimes, with corrected status

```
C_native     = F_txt + K·F·(2 + c_state + c_mask)     R_native
C_A1         = F_txt + K·F·(2 + c_joint)              R_native   — same reference as native
C_A2         = F_txt + K·F·(1 + c_joint)              R_coherent — a DIFFERENT reference
C_trunc(τ)   = F_txt + K·F + τ·F·c_joint              biased w.r.t. its own reference
C_fwd_only   = F_txt + K·F                            forward alone (z_K needed for the loss)
```

Simplified at `c_b` (dropping `F_txt` for comparability with the original table):

| | `c_b = 1` | `c_b = 2` | status |
|---|---|---|---|
| `C_native` | **20 F** | **30 F** | reference computation, `R_native` |
| `C_A1` | 15 F | 20 F | **same reference**; both derivatives from one recomputed graph **in exact arithmetic**. **Not** bit-identical; **no** fixed runtime saving or trajectory claim (X11) |
| `C_A2` | 10 F | 15 F | **different reference** (`R_coherent`). **Not** an exact substitute for the native estimator (X12) |
| `C_trunc(τ=1)` | 6 F | 7 F | biased w.r.t. its own reference |
| `C_fwd_only` | 5 F | 5 F | — |

### 4.1 Withdrawn: the "floor" theorem (X5)

The original text asserted `C_floor = K·F·(1+c_b)` as a **lower bound** for the exact relaxed
gradient, **attained** by `A2`, concluding *"there is no exact-gradient headroom below `A2`,
and in particular none that an estimator could reach."*

**Withdrawn.** One forward plus one full reverse traversal per step is an **ordinary
reverse-mode cost model** for the coherent full-chain estimator — **not a universal
lower-bound theorem** over estimators or implementations. The expression is retained purely
as a cost model for that estimator. In particular, an estimator that **stops traversal at a
random depth** is not bounded below by it: under
`E[C] = K·F + (1+c_b)·F·Σ_i q_i(K−i) + overhead` with `K = 5`, `c_b = 1`, one draw and all
probabilities positive, `q = (1/20,1/20,1/20,1/20,4/5)` gives expected depth `3/2` and
**expected main cost `8F` < `10F`** — a saving is possible within this model if overhead is
below `2F`. **This refutes the proof; it is not a FLUX speedup, variance/convergence or
novelty claim.**

### 4.2 Peak memory — corrected

| regime | peak | note |
|---|---|---|
| `C_native`, `C_A1` | weights + **one** step's activation graph + `K` stored latents | the paper's §3.7 goal: *"constant memory complexity regardless of the step"* |
| `C_A2` | weights + **`K = 5`** steps' activation graphs | **≈5× activation memory. Feasibility unmeasured**, and the binding practical constraint on `A2` |
| data preparation | **two 12B models resident simultaneously** — FLUX.1-schnell and FLUX.1-dev (`promptdataset.py:78-84`, freed at `:89-90`), then CLIP (`utils_train.py:225`) | **Corrected (X15):** this establishes a **preparation memory burden**, **not** proof that it **exceeds the unknown training activation peak**. The earlier "global peak memory" claim is withdrawn |

`A1` does not change the reference or the memory profile. **`A2`'s viability cannot be
determined without profiling, which is not authorized.**

### 4.3 Conditional ceilings — corrected arithmetic (X9)

The original text reported a single **`1.33×`** "Amdahl ceiling" for a free gradient over
`A2`. **That number was the including-instrumentation figure, mislabeled.** Corrected, each
ceiling stated **with its actual included costs** (`c_b = 1`, `F_txt` excluded from both
sides):

| basis | baseline | free-gradient cost | ratio |
|---|---|---|---|
| vs `C_native`, **gradient only** | 20 F | 5 F | **`4`** |
| vs `C_A2`, **gradient only** | 10 F | 5 F | **`2`** (*not* `1.33`) |
| vs `C_native`, **incl. instrumentation** | 30 F | 15 F | **`2`** |
| vs `C_A2`, **incl. instrumentation** | 20 F | 15 F | **`4/3`** |

The terminal loss needs `z_K`, so the `K·F` forward is irreducible **for this objective** —
a statement about the forward, not a bound on estimators. **Conclusions that depended on the
`1.33×` figure are withdrawn**, including the claim that a free gradient is "worth at most
`1.33×` over ordinary correct autograd usage".

---

## 5. One-time costs and the whole-run picture

| one-time item | cost | source |
|---|---|---|
| concept latents, all 80 rows | `80·(K·F + F_txt)` ≈ **400 F** + `80·F_txt` + `80·D` | `promptdataset.py:238-241, 300-307` |
| **FlowEdit targets**, 40 forget rows, on **FLUX.1-dev** | `40 · ≈(24·2) · F_dev` ≈ **1920 F** + `40(E_vae + D)` | `:250-276`: `T_steps=28`, `n_max=24`, `n_avg=1`, src **and** tar evaluations per active step |
| CLIP filtering over 80 rows | `≪ F` each; GPU memory only | `utils_train.py:217-261` |
| dummy init generation, initial/final image saves | `≈ 3F + 2·8·K·F` ≈ 83 F + decodes | `train.py:228, 197-208, 422-441` |
| **total preparation `P`** | **≈ 2320 F** + `≈120·F_txt` + `≈120·D` + `40·E_vae` | |

### 5.1 Withdrawn: composition and whole-run ratios (X14, X16)

The original published a table giving **`11 920 F`** as published, **`5 520 F`** fixed and
**`4 240 F`** truncated, a composition of **gradient 54% / instrumentation 27% /
preparation 19%**, and predicted ratios **`2.16×`, `2.81×`, `2.12×`, `3.32×`**.

**All of these are withdrawn.** They omit `F_txt` on every iteration and every preparation
row, omit optimizer/I/O/initialization work, collapse `c_state`/`c_mask`/`c_joint` into one
constant, and — because `P` is fixed — **depend on the iteration count**, which is itself
unresolved (80 vs 400). What remains usable is only the **per-iteration structure of §3–§4**
and the **retained configuration facts of §2**.

As an **explicitly labelled illustration** of the terms actually modelled (not a composition
claim, not a speedup): at `c_b = 1`, `N = 320`, ignoring `F_txt`, `D` and all §3.2 terms, the
modelled per-iteration gradient work is `20F` against `10F` of modelled instrumentation,
with `≈2320F` of modelled fixed preparation. **Adding the omitted terms moves all three, and
their relative sizes are not established.**

---

## 6. Break-even conditions — corrected (X10)

Use consistent iteration/update units, with `P` once-per-run work and `H` **explicitly
defined** additional overhead:

```
T_base  =  P_base  +  N · c_base
T_new   =  P_new   +  H  +  ν · N · c_new
```

**Shared one-time preparation must not be multiplied by update inflation.** If preparation
is shared (`P_new = P_base`) and `H = 0`:

```
ν  <  c_base / c_new
```

Under the illustrative per-iteration values of §4:

| `c_b` | `c_base` (`A2`) | `c_new` (`trunc τ=1`) | **break-even** |
|---|---|---|---|
| 1 | 10 F | 6 F | **`ν < 10/6 = 5/3`** |
| 2 | 15 F | 7 F | **`ν < 15/7`** |

**Warm-up and refresh work is counted as replacing or adding iterations**, not as a
multiplier on `P`, and not double counted: `W` warm-up iterations at `c_base` plus
`⌈N/R⌉` refresh iterations at `c_base` enter `T_new` as `H = (W + ⌈N/R⌉)·(c_base − c_new)`
above the `ν·N·c_new` term, tightening the threshold by that amount.

**These corrected thresholds still do not establish useful actual-model savings.** `ν` is
**unknown**; selection overhead is unquantified; `F_txt` and the §3.2 terms are unmodelled.
And **we do not claim `ν ≥ 1`** (X8): update requirements in the actual nonlinear
optimization are unknown, and DRaFT's results do not establish that ordering for Minimalist.

### 6.1 Withdrawn: "SSMG has no break-even at all" (X3)

The original asserted `C_SSMG → C_A2` in expectation with `Var > 0` ⇒ `ν ≥ 1` ⇒ `ν·ρ ≥ 1`,
concluding SSMG *"cannot break even for any `q`, `S₀`, `R` or `W`."* **Withdrawn on both
legs**: the expected-cost claim ignores random stopping depth (§4.1), and the `ν ≥ 1`
inference is unsupported (X8).

**What replaces it:** break-even is a **nonvanishing, unresolved** condition — neither
established nor excluded.

---

## 7. What these costs do **not** establish

- **No measurement.** No profiling. `F`, `F_txt`, `c_state`/`c_mask`/`c_joint`, `D`,
  `F_dev`, the 1024×1024 deduction, `|D| ≈ 64`, and the FlowEdit step count are **derived or
  assumed**; `ν` is **unknown**.
- **Work ratios are not wall-clock ratios**, and **simplified arithmetic is not a
  measurement.**
- **No composition claim**, no whole-run speedup claim, and no claim that preparation is the
  global memory peak.
- **No lower-bound theorem.** §4.1's expression is a cost model for one estimator.
- **Cheaper updates are not cheaper training**, and `A1`/`A2` do **not** escape this by
  being "exact": `A1` preserves the reference but carries **no** verified runtime or
  trajectory claim, and `A2` **changes the reference**.
- **No cost statement is a semantic statement.** Nothing here speaks to concept absence,
  retained utility, rendering quality or robustness — properties of the **binarized** mask
  (`DERIVATION.md` §5.4). No such claim appears anywhere in this sprint.
- **Totals *and* ratios** depend on the unresolved `N_upd` (80 vs 400) and on the unresolved
  target construction.
