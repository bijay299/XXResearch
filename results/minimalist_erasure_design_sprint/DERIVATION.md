# Derivation — the reference objective and mask-parameter gradient

**NO EXPERIMENTS.** Everything below is static reading of the paper and of the author
source at the pinned commit, plus pencil derivation and two tiny exact rational/scalar
arithmetic checks. No model was loaded and no research code was imported or executed.

Pinned source: `YaNgZhAnG-V5/minimalist_concept_erasure` @
**`d178324ec4a48d0536a595b25fdad587b441add8`** (all code added in one commit,
`1bb7ca2`, 2025-11-24). Paper: `arxiv.org/html/2507.13386v1` (v1, 2025-07).
Line numbers refer to the pinned commit. See [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md).

---

## 1. Source / objective map

### 1.1 The paper's stated objective (the *ideal* objective)

Paper §3.6 defines the intervention as a binary mask on parameters:

> `θ = M ⊙ θ′,  M ∈ {0,1}^|θ|`

and §3.6 states the relaxation: *"we perform continuous relaxation using Hard-discrete
sampling (Louizos et al., 2018) to learn a continuous mask, and binarize the learned mask
to obtain a discrete mask."* The paper also states it masks **neurons, not weights**
(*"neuron masking instead of weight masking to reduce the number of trainable
parameters"*).

The loss (paper Eq. 16) is

```
L  =  E_{c~C_R, x_T} ‖ ℱ_θ(x_T, c) − ℱ_θ′(x_T, ∅) ‖²₂            (erase)
   +  β · E_{c~C_N, x_T} ‖ ℱ_θ(x_T, c) − ℱ_θ′(x_T, c) ‖²₂         (preserve)
```

with `ℱ` the full sampler, `θ′` the frozen original parameters, `C_R` the target
(erase) prompt set and `C_N` the neutral set. Paper Table 6 (App. H.1): batch size 4,
`lr_ffn = lr_norm = 0.5`, `β = 0.01`, Adam, **400 training steps**, weight decay `1e-2`,
1× H100.

Paper §3.7 states the gradient rule and the memory scheme:

> *"the mask gradient is `dL(X_0, X̃_0)/dM ≔ Σ_i [dL(X_0(X_i), X̃_0)/dX_i] · [dX_i/dM]`
> … We follow the approach in [Zhang et al. (2024d)] to perform step-wise gradient
> checkpointing to calculate long gradient chains with constant memory complexity
> regardless of the step. During forward propagation, we store only the step outcomes
> `X_i` of the model. During backward propagation, we recompute the forward before
> gradient calculation."*

and the limitation: *"our step-wise gradient checkpointing relies on a primitive
implementation that lacks support for multi-GPU training, limiting the method's
scalability."*

> **Notation defect (ours, flagged not relied upon).** As written, `Σ_i (dL/dX_i)(dX_i/dM)`
> double-counts if `dX_i/dM` is read as a **total** derivative, because `X_i` already
> depends on `M` through all earlier steps. The expression is correct only if `dX_i/dM`
> is the **partial/direct** Jacobian of step `i−1`'s map holding its input fixed. **The
> code implements the partial reading** (§3 below), which is the correct reverse-mode
> accumulation. We treat the code as definitive here and the paper's notation as loose.

### 1.2 What the implementation actually optimizes

Four substantive divergences from §1.1. All are **source-supported**, and the first is
**essential** — it changes the objective, not the estimator.

**(a) The erasure target is not `ℱθ′(x_T, ∅)`.** Targets are **precomputed once per
dataset row and cached to disk** as safetensors
(`src/diffsolver/data/promptdataset.py:224-316`), never recomputed during training. With
the default `with_flowedit=True` (`promptdataset.py:57`) the forget target is produced by
**FlowEdit on a second, different model — `black-forest-labs/FLUX.1-dev`** —
instantiated alongside FLUX.1-schnell (`promptdataset.py:78-84`), at `T_steps=28`,
`src_guidance_scale=1.5`, `tar_guidance_scale=5.5`, `n_max=24`
(`promptdataset.py:250-276`). The non-FlowEdit branch
(`promptdataset.py:283-298`) instead uses `ℱθ′(x_T, c_neutral)` with `c_neutral` from a
hand-written `CON_DECON_DICT` substitution table (`unlearn_template.py:282+`, e.g.
`"Hulk": "a strong man"` at `:292`). **Neither branch is `ℱθ′(x_T, ∅)`**, and the paper never
mentions FlowEdit as a target construction or FLUX.1-dev at all.

**(b) The two loss terms are not summed; one replaces the other.** `scripts/train.py:284-297`:

```python
loss, loss_reconstruct = pruning_loss(...)        # loss = β · ‖z_K − z*_retain‖²
if value:                                         # value == 1 → forget row
    intermediate_loss = reconstruction_loss_func(deconcept_image_pt[:, -1, ...], image[0])
    loss = intermediate_loss                      # ← REPLACES, discarding the β term
```

So per sample the objective is **either** the forget term **or** `β ×` the retain term,
chosen by the row's `value` flag. The dataset is built to alternate them exactly 50/50
(`promptdataset.py:177-179`: `df.iloc[::2] = deconceptdf`, `df.iloc[1::2] = df`). This is a
**stochastic (here: deterministically alternating) single-term surrogate** of Eq. 16's sum,
not Eq. 16. The retain term also takes `image` while the forget term takes `image[0]` —
batch element 0 only.

**(c) `x_T` is not sampled.** `scripts/train.py:268` re-seeds the generator to the *same*
`cfg.trainer.seed` **inside the per-sample loop**, and `promptdataset.py:182` does the same
when building targets. The initial latent is therefore **one fixed `x_T`** for every
iteration, every prompt and every epoch. The `E_{x_T}` in Eq. 16 is **not estimated**; the
implemented objective is conditional on a single noise draw.

**(d) There is no sparsity/`L0` term in the trained loss.** `calculate_reg_loss` and
`l0_complexity_loss` exist (`src/diffsolver/utils/utils.py:226-257`) and `configs/flux.yaml:45-53`
configures `reg`, `reg_alpha`, `reg_beta`, `lambda_reg`, `use_ffn_reg`, but **nothing in
`scripts/train.py` calls them** (verified by grep over all `*.py`; the only other reference
is `scripts/hyperparameter_tuning.py`, which merely sets the config value and shells out via
`os.system`, `:91`). Sparsity in the published FLUX path is produced **only** by the
post-hoc binarization threshold, not by the objective. The paper likewise shows no `L0`
term in Eq. 16.

### 1.3 The mask: parameterization, relaxation, and where it acts

Trainable parameters are **per-neuron scalars `λ`**, one vector per hooked module,
initialized to `init_lambda = 3` (`configs/flux.yaml:26`;
`hooks/linear_layer_hooks.py:85-88`). For the FLUX path, `init_hooker` builds **three**
hookers — cross-attention, feed-forward, norm (`hooks/init_hooker.py:26-81`).

All three apply the gate the **same way**: an elementwise multiplicative gate on the
module's **output activations**.

| hooker | site | line |
|---|---|---|
| feed-forward | `hidden_states * mask + randn·epsilon·(1−mask)` | `hooks/ff_hooks.py:87` |
| norm | same form | `hooks/norm_hook.py:92` |
| cross-attention | same form, per head, after the attention op | `hooks/attention_processor.py:64` |
| (generic linear, non-FLUX) | same form | `hooks/linear_layer_hooks.py:74` |

`configs/flux.yaml:17` sets `epsilon: 0.0`, so the additive noise branch vanishes and the
gate is **exactly** `h ↦ m ⊙ h`. **This single fact drives the whole result** (§4).

The relaxation (`utils/utils.py:213-223`), with `use_log: false`:

```
u ~ U(0,1)
n(u) = log(u + eps) − log(1 − u + eps)
s    = σ( (n(u) + λ) / β ),      β = 0.83
m    = clamp( s·(η − γ) + γ , 0, 1 ),   η = 1.1, γ = −0.1
```

Two source-supported consequences, both verified by exact scalar arithmetic:

- **`eps` is not a numerical-safety epsilon here.** `configs/flux.yaml:31` sets
  `masking_eps: 0.5`, which is plumbed into this `eps`
  (`init_hooker.py:45,57,74` → `base.py:110-111`). With `eps = 0.5` the noise `n(u)` is
  **bounded in `[−1.0986, +1.0986]`** instead of the unbounded logistic of Louizos et al.
  So the published FLUX config instantiates a **bounded-noise variant**, not the
  hard-concrete distribution the paper cites.
- **At initialization the gate is saturated and the gradient is mostly exactly zero.**
  The upper clamp binds when `σ((n+λ)/β) ≥ (1−γ)/(η−γ) = 11/12`. At `λ = 3`:
  `P(m = 1 exactly) = 0.9659`. Where the clamp binds, `∂m/∂λ = 0` **exactly**. So
  **≈96.6% of gate draws contribute an exact zero** to the mask gradient at init
  (at `λ = 1` this falls to 4.2%).

A fresh `u` is drawn on **every hook call** — i.e. per layer, **per denoising step**, and
again on **every recomputation**. There is **no RNG stashing**. This is the hinge of §5.3.

For the FLUX config `attn_lr: 0` (`configs/flux.yaml:19`), so attention gates receive
gradients but **never move**; only feed-forward and norm gates train
(`ff_lr = n_lr = 0.5`).

**Final binarization** is post-hoc and *not* the relaxation: `hooks/base.py:49-89`
(`binarize`, top-`ratio` by sorted `λ`, local or global scope),
`hooks/base.py:41` (`load`, `relu(λ − threshold) > 0`),
`hooks/cross_attn_hooks.py:107-117` (`λ ≥ threshold`), and in-training logging uses
`continues2binary`, `(λ > 0).float()` (`base.py:114-116`) with the threshold **hardcoded
to 0** and a `TODO` acknowledging it (`scripts/train.py:398-399`). The paper specifies
neither a ratio nor a threshold.

---

## 2. The discrete computation

Let the hooked gate vector be `m = m(λ, ξ)` with `ξ` the per-call uniform draws, and let
`K = cfg.trainer.num_intervention_steps = 5` (`configs/flux.yaml:24`) — the FLUX.1-schnell
path is genuinely **few-step**. Write the one-step map (`models/flux.py:314-365`:
transformer evaluation then `scheduler.step`):

```
z_{k+1} = f_k( z_k , m(λ, ξ_k) , c )            k = 0 … K−1
z_0     = x_T  (fixed; §1.2c)
```

`c` is the fixed per-row prompt embedding, computed once in
`inference_preparation_phase` (`models/flux.py:194-312`). The loss is **terminal only**:
`output_type="latent"` means `inference_aft_denoising` returns `z_K` unchanged
(`models/flux.py:367-385`), so **no VAE decode is in the training graph**, and the loss
compares `z_K` against the cached target latent `z*` (`scripts/train.py:282-297`; only the
last 2 latents are even stored — `promptdataset.py:54, 311-312`):

```
L(λ, ξ) = ℓ( z_K(λ, ξ) , z* )        ℓ = MSE   (configs/flux.yaml:47 → train.py:213-214)
```

There is **no per-step loss term** and **no trajectory coupling to a teacher**: the frozen
model `θ′` appears **only** through the cached constant `z*`. Nothing about `ℱθ′` is
recomputed during training.

---

## 3. The reference mask gradient, as implemented

`scripts/train.py:299-333` is a hand-rolled recompute-based reverse sweep:

```python
(loss / accum).backward()                     # :299  → dL/dz_K
grad = latents.grad.detach()                  # :300
for timesteps_idx, t in enumerate(reversed(timesteps)):            # :306
    current_latents = intermediate_latents[-(timesteps_idx+1)].detach()   # :307
    current_latents.requires_grad = True                                  # :308
    latents = pipe.inference_denoising_step(...)                 # :313  RECOMPUTE f_k
    lamb_grads = torch.autograd.grad(latents, trainable_lambs,
                                     grad_outputs=grad, retain_graph=True)   # :320  sweep 1
    for lamb, lamb_grad in zip(trainable_lambs, lamb_grads):     # :327-331  accumulate
        lamb.grad = lamb_grad if lamb.grad is None else lamb.grad + lamb_grad
    grad = torch.autograd.grad(latents, current_latents, grad_outputs=grad)  # :333  sweep 2
```

With `ḡ_k ≔ ∂L/∂z_k`, this computes

```
ḡ_K = ∂ℓ/∂z_K

for k = K−1 … 0:
    ∇_λ L  +=  ( ∂f_k/∂λ |_{z_k, ξ_k} )ᵀ  ḡ_{k+1}          (sweep 1, train.py:320)
    ḡ_k     =  ( ∂f_k/∂z |_{z_k, ξ_k} )ᵀ  ḡ_{k+1}          (sweep 2, train.py:333)
```

i.e. the standard reverse-mode accumulation

```
∇_λ L  =  Σ_{k=0}^{K−1}  ( ∂f_k/∂λ )ᵀ  ( Π_{j=k+1}^{K−1} ∂f_j/∂z )ᵀ  ∂ℓ/∂z_K .   (★)
```

This is the partial-Jacobian reading of paper §3.7, and it is the correct reverse-mode
gradient of `L(λ, ξ)` **given** a single fixed `ξ` consistently used throughout — a
condition the code does **not** satisfy (§5.3).

Three implementation facts that matter for cost (§[`COST_MODEL.md`](COST_MODEL.md)):

1. **Sweeps 1 and 2 are two separate reverse traversals of the same graph**
   (`retain_graph=True` at `:324` exists only to permit the second). They are fusable into
   one call with no change to the result.
2. The forward is done **twice** per step: once stored under `@torch.no_grad()`
   (`train.py:68-98`, storing *all* `z_k`), once recomputed with a graph (`:313`).
3. The loop runs to `k = 0`, so `ḡ_0 = ∂L/∂x_T` is computed and **discarded** — `x_T` is a
   fixed constant with no parameter dependence. One wasted state-VJP per iteration.

---

## 4. The structural lemma: the mask gradient is a free byproduct

This is the core result of the sprint.

**Lemma.** At every hooked module the gate acts as `h ↦ m ⊙ h` with `epsilon = 0`
(§1.3). Hence for neuron `j` of a hooked module at step `k`,

```
∂L/∂m_j^{(k)}  =  Σ_{b,t}  h_{b,t,j} · ḡ_{b,t,j} ,          ḡ ≔ ∂L/∂(m ⊙ h)       (†)
```

and then `∂L/∂λ_j = (∂m_j/∂λ_j) · ∂L/∂m_j` with `∂m_j/∂λ_j` a scalar from the relaxation.

**`ḡ` is exactly the activation gradient the reverse sweep must already compute** at that
module in order to keep propagating toward `z_k`. Therefore:

> The entire per-step mask gradient — across **all** hooked modules, attention, FFN and
> norm alike — costs `O(#neurons × #tokens)` multiply-accumulates on top of a reverse
> sweep that must happen anyway. Relative to the module's own matmuls, that is a rounding
> error.

**Corollary (why the sprint's hypothesis fails).** The two quantities the native loop
computes with *two separate sweeps* — `(∂f_k/∂λ)ᵀḡ_{k+1}` and `(∂f_k/∂z)ᵀḡ_{k+1}` —
are obtainable from **one** sweep, because the first is the elementwise byproduct (†) of the
activation gradients the second produces en route. It follows that:

- the exact gradient (★) costs **one forward + one reverse sweep per step**, and
- **no estimator that targets the mask-gradient terms can save anything**, since those
  terms are free riders. Subsampling them, importance-weighting them, or estimating them
  stochastically removes work that costs nothing and buys variance.

The only remaining lever is **not sweeping a step at all** — truncation — which is
published prior art (DRaFT-K, AlignProp; see [`ALGORITHM.md`](ALGORITHM.md) §4).

---

## 5. Four distinct gradient notions — kept separate

### 5.1 Ideal vs. relaxed

The ideal object is `∇_M` of Eq. 16 over `M ∈ {0,1}^{|θ|}`, which **does not exist**: the
domain is discrete. The relaxation replaces it with `∇_λ E_ξ[L(λ,ξ)]` for the *relaxed*
gate. (★) is a gradient of the **relaxed surrogate**, never of the discrete model.

### 5.2 Conditional pathwise vs. gradient of the expectation

Two different targets:

- **Conditional pathwise:** `∇_λ L(λ, ξ)` at one fixed realization `ξ`. This is what a
  single consistent reverse sweep computes.
- **Gradient of the expected objective:** `∇_λ J(λ)`, `J(λ) = E_ξ[L(λ,ξ)]`.

Exchanging differentiation and expectation, `∇_λ E_ξ[L] = E_ξ[∇_λ L]`, requires: (i) the
sampling density is **reparameterized** so `ξ` carries no `λ`-dependence — satisfied here,
`ξ` is `U(0,1)` independent of `λ`; (ii) `L(λ, ·)` is a.e. differentiable in `λ` and
**locally Lipschitz with an integrable dominating bound** near `λ` (dominated convergence).

(ii) holds but is **weak at the published settings**: the gate is clamped, so `L(·, ξ)` is
only piecewise differentiable in `λ`, with `∂m/∂λ = 0` on the clamped set. Since
`P(clamped at 1) = 0.9659` at `λ = 3` (§1.3), the single-sample estimator is an exact zero
for ~96.6% of gates at initialization, and unbiasedness for `∇J` holds only in an
average-over-`ξ` sense that the implementation does not realize — it uses **one** `ξ` per
iteration and **one fixed `x_T`** ever (§1.2c). The clamp set has measure zero in `λ` for
fixed `ξ`, so a.e. differentiability is not violated; the practical issue is variance and
signal sparsity, not formal bias.

### 5.3 What the code actually computes — neither of the above

`u` is redrawn on **every** hook call and there is **no RNG stashing** (§1.3). So:

- the stored trajectory `z_k` comes from draws `ξ^A` (the `@torch.no_grad()` pre-pass,
  `train.py:68-98`);
- `∂ℓ/∂z_K` is evaluated at `z_K^A` — consistent with `ξ^A`;
- but every recomputed step at `train.py:313` draws **fresh, independent** `ξ_k^B`, so both
  Jacobians at step `k` are evaluated at `(z_k^A, ξ_k^B)` with `ξ^B ⟂ ξ^A`.

The implemented quantity is therefore

```
Σ_k ( ∂f_k/∂λ |_{z_k^A, ξ_k^B} )ᵀ ( Π_{j>k} ∂f_j/∂z |_{z_j^A, ξ_j^B} )ᵀ ∂ℓ/∂z_K|_{z_K^A}
```

which is **not** the conditional pathwise gradient at `ξ^A` (wrong linearization noise) and
**not** an unbiased estimate of `∇_λ J` (the factors carry independent `ξ^B`, so their
expectation factorizes, whereas the true pathwise product shares one `ξ_k` per step). It is
a **mismatched-realization estimator** whose bias we have **not** characterized and do not
claim to bound.

> **So the native reference already pays the full price of an exact method for a gradient
> that is not the exact gradient of its own forward trajectory.** `torch.utils.checkpoint`
> exists precisely to avoid this: `preserve_rng_state` **defaults to `True`** and the
> recompute runs under `torch.random.fork_rng` with the stashed forward state restored
> (verified by static read of the locally installed
> `torch/utils/checkpoint.py:291-297, 477, 1507-1513`). The published trainer hand-rolls its
> own checkpointing and omits this.
>
> This is a **fidelity** finding, not a saving. It matters here because it **confounds**
> any future efficiency comparison: a "cheaper" estimator could appear better or worse
> purely through differing mask-noise consistency. See
> [`FALSIFICATION_PLAN.md`](FALSIFICATION_PLAN.md) §3.

### 5.4 The deployed artifact: the discrete mask

What ships is the **binarized** mask (§1.3), and `∇_λ` of the relaxed objective is **not**
the derivative of the discrete model. The relaxation→binarization gap is governed by
nothing in the trained objective — there is no `L0` term (§1.2d) and no
straight-through estimator anywhere (no custom `autograd.Function`, no
`detach()`-based pass-through in any hook; the gate is an ordinary differentiable
`clamp∘σ`). **Deletion, utility, image quality and robustness are properties of the
discrete artifact.** No statement in this document about gradient exactness, estimator
variance, or loss values says anything about them. See
[`COST_MODEL.md`](COST_MODEL.md) §7.

---

## 6. Assumptions used, and what is derived vs. assumed

**Derived from source** (traceable to file:line above): the computation (§2); the reverse
accumulation (★) and its two-sweep structure (§3); the gate form and hence the lemma (†)
and its corollary (§4); the four objective divergences (§1.2); `K = 5`; the bounded-noise
relaxation and `P(clamp) = 0.9659` at `λ = 3`; the RNG mismatch (§5.3); targets cached and
never recomputed; no VAE decode in the training graph; no `L0` term.

**Assumed** (and carried as uncertainty, never asserted):

1. `configs/flux.yaml` is representative of the published FLUX.1-schnell run. The paper's
   Table 6 **disagrees** with it (`β`: 0.01 vs `0.1`; batch 4 vs `1`; 400 steps vs ≈80
   updates — see [`COST_MODEL.md`](COST_MODEL.md) §2). Unresolved.
2. `with_flowedit=True` (the code default) is the published target construction. The paper
   says `ℱθ′(x_T, ∅)`. **Unresolved, and essential.**
3. A reverse sweep over one step's graph costs `c_b · F` with `c_b ∈ [1,2]`. **Unmeasured.**
4. Hook ordering stays aligned with `hook_counter`
   (`linear_layer_hooks.py:106-107`, modulo arithmetic over a fixed traversal order). A
   static-reading assumption we did not validate at runtime.
5. The lemma assumes `epsilon = 0`. At `epsilon > 0` the gate gains the additive branch
   `randn·epsilon·(1−m)` and (†) acquires a second term — still an elementwise byproduct,
   so the corollary survives, but we only checked this for the published `epsilon: 0.0`.

**Not established:** which of paper and code is authoritative; any runtime behavior, timing
or memory figure; whether the snapshot is the publication implementation; the bias of the
mismatched-realization estimator (§5.3).

---

## 7. Arithmetic checks

Two tiny deterministic checks were run (exact rationals / scalar `math`), with no tensors,
no model, no learned system, no simulation and no sweep:

- **Gate saturation** (§1.3): `P(m = 1)` at `λ ∈ {3, 1, 0}` = `0.9659 / 0.0417 / 0.0000`;
  noise range `[−1.0986, +1.0986]` at `eps = 0.5`.
- **Reverse accumulation (★)** reproduces the analytically substituted gradient exactly on a
  2-step rational chain, and the attempted design is unbiased but zero-saving on it. Full
  worked example, including the variance, in [`ALGORITHM.md`](ALGORITHM.md) §3.

Synthetic arithmetic is **not** FLUX evidence and certifies only the algebra.
