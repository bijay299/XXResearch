# Derivation — the reference objective and mask-parameter gradient

**NO EXPERIMENTS.** Everything below is static reading of the paper and of the author
source at the pinned commit, plus pencil derivation and tiny exact rational/scalar
arithmetic checks. No model was loaded and no research code was imported or executed.

> **Amended 2026-10-17 — see [`CORRECTIONS.md`](CORRECTIONS.md).** The **local
> multiplicative-gate derivative of §4 is retained**. The **impossibility corollary drawn
> from it is withdrawn** (X1), together with the measure-theoretic reading of the clamp
> (X22), the "not an unbiased estimate" assertion (X23), the multi-sample requirement
> (X21), the `image[0]` batch-index diagnosis (X24), the `L0` "divergence" (X20), and the
> "different empirical objective" reading of per-example term selection (X19).

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

A **source-defined empirical reference objective can be stated** from the code below. Its
**alignment with the publication is unverified** — that is the accurate claim, and the
earlier assertion that "there is no reference objective to be cheaper than" is withdrawn
(X25).

Of the four items originally listed as divergences, **(a) is a genuine and essential
difference** and **(c) is a genuine but narrower one**; **(b) and (d) are corrected below
and are *not* divergences** (X19, X20).

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

**(b) Per-example term selection — *not* a different empirical objective (corrected, X19).**
`scripts/train.py:284-297`:

```python
loss, loss_reconstruct = pruning_loss(...)        # loss = β · ‖z_K − z*_retain‖²
if value:                                         # value == 1 → forget row
    intermediate_loss = reconstruction_loss_func(deconcept_image_pt[:, -1, ...], image[0])
    loss = intermediate_loss                      # ← REPLACES, discarding the β term
```

Per sample the loss is **either** the forget term **or** `β ×` the retain term, chosen by the
row's `value` flag, and the dataset alternates them exactly 50/50
(`promptdataset.py:177-179`: `df.iloc[::2] = deconceptdf`, `df.iloc[1::2] = df`).

**Corrected reading (X19).** At fixed parameters, a **balanced uniform** forget/retain row
mixture **estimates `½(L_forget + β·L_retain)`** — a positive multiple of Eq. 16's sum.
Choosing one term per example is therefore **not automatically a different empirical
objective**, and the earlier "single-term surrogate, not Eq. 16" reading is withdrawn. What
*does* remain true, and is weaker: **alternating updates need not reproduce the finite-step
trajectory** of computing both terms together, since each update sees one term. The genuine
objective differences are **(a) target construction** and **(c) fixed-seed coverage**, kept
separate from this.

**Also withdrawn (X24):** the claim that the forget term "takes batch element 0 only".
`image` at `:294` is the `FluxPipelineOutput` returned at `:282`, and in the declared
compatible Diffusers (`pyproject.toml:19`, `diffusers>=0.31.0`) `BaseOutput.__getitem__` is
`return self.to_tuple()[k]` for non-`str` keys — so `image[0]` returns the **first declared
output field** (`.images`), i.e. the **whole latent tensor**. There is no batch-index
asymmetry.

**(c) Fixed-seed coverage of `x_T` (genuine, but narrower than published).**
`scripts/train.py:268` re-seeds the generator to the *same* `cfg.trainer.seed` **inside the
per-sample loop**, and `promptdataset.py:182` does the same when building targets. The
initial latent is therefore **one fixed `x_T`** for every iteration, every prompt and every
epoch.

**Corrected framing (X21).** The accurate statement is that the **fixed seed limits coverage
of `x_T` across iterations**. The earlier stronger framing — that `E_{x_T}` is therefore
"not estimated" and that unbiasedness needs an averaging the implementation "does not
realize" — is withdrawn: **a coherent single reparameterized sample can be unbiased** under
the required regularity conditions, and **multiple samples within an iteration are not
required**. **Initial-latent randomness and mask randomness are separate axes**, and the
fixed seed constrains only the former.

**(d) No sparsity/`L0` term in the trained loss — a source fact, *not* a paper/code
divergence (corrected, X20).** The cited paper objective (Eq. 16) **also contains no such
term**, so its absence in the code is not a divergence. What remains is the narrower source
fact: `calculate_reg_loss` and
`l0_complexity_loss` exist (`src/diffsolver/utils/utils.py:226-257`) and `configs/flux.yaml:45-53`
configures `reg`, `reg_alpha`, `reg_beta`, `lambda_reg`, `use_ffn_reg`, but **nothing in
`scripts/train.py` calls them** (verified by grep over all `*.py`; the only other reference
is `scripts/hyperparameter_tuning.py`, which merely sets the config value and shells out via
`os.system`, `:91`). Sparsity in the published FLUX path is produced **only** by the
post-hoc binarization threshold, not by the objective.

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

## 4. The local gate derivative (retained) — and the withdrawn corollary

> **Amended.** The **local identity below is retained**: the mask derivative is the incoming
> activation times the gate-output adjoint, with the appropriate reductions. The
> **impossibility corollary drawn from it is withdrawn** (X1). It was presented as the core
> result of the sprint; it is not a result.

**Local identity.** At every hooked module the gate acts as `h ↦ m ⊙ h` with `epsilon = 0`
(§1.3). Hence for neuron `j` of a hooked module at step `k`,

```
∂L/∂m_j^{(k)}  =  Σ_{b,t}  h_{b,t,j} · ḡ_{b,t,j} ,          ḡ ≔ ∂L/∂(m ⊙ h)       (†)
```

and then `∂L/∂λ_j = (∂m_j/∂λ_j) · ∂L/∂m_j` with `∂m_j/∂λ_j` a scalar from the relaxation.

`ḡ` is the activation gradient at that module. **Conditional on the state adjoint at that
site already being computed**, the mask term is an elementwise multiply-accumulate,
`O(#neurons × #tokens)` — small relative to the module's own matmuls.

### 4.1 What this supports, and what it does not (X1)

**Supported — regime (1).** *If all state adjoints are always computed*, then skipping the
cheap local mask terms offers little saving. Concretely: the two quantities the native loop
obtains with *two separate sweeps* — `(∂f_k/∂λ)ᵀḡ_{k+1}` and `(∂f_k/∂z)ᵀḡ_{k+1}` — can be
obtained from **one** sweep over the same recomputed graph, because the first is the
elementwise byproduct (†) of activation gradients the second produces en route. **This is a
statement about one implementation detail, not about estimators.**

**Withdrawn — the impossibility corollary.** The published record inferred that therefore
*"no estimator that targets the mask-gradient terms can save anything"*, that the exact
gradient's cost is a **floor**, and that truncation is *"the only remaining lever"*. **All
three are withdrawn.** The identity is **conditional on the state adjoint being computed at
that site**; it says nothing about estimators that **do not compute every state adjoint**.

**Regime (2) — the case the record ignored.** If reverse traversal **stops after the
earliest selected contribution**, cost depends on a **random stopping depth**, and a nonzero
probability of a full sweep does **not** force full expected cost. Under the record's own
simplified expression,
`E[C] = K·F + (1+c_b)·F·Σ_i q_i(K−i) + selection overhead`, with `K = 5`, `c_b = 1`, one
draw and all probabilities positive, `q = (1/20,1/20,1/20,1/20,4/5)` gives expected reverse
depth `3/2` and **expected main cost `8F` against `A2 = 10F`** — a saving is possible within
this model if overhead is below `2F`. See [`CORRECTIONS.md`](CORRECTIONS.md) §A.

**That counterexample refutes the universal proof only.** It is not evidence of a FLUX
speedup, of acceptable variance or convergence, or of novelty, and it is **not adopted as a
proposal**. Separately, **one forward plus one full reverse traversal per step is an
ordinary reverse-mode cost model, not a universal lower-bound theorem** over estimators or
implementations (X5).

Prior art bearing on the truncation lever is stated accurately in
[`ALGORITHM.md`](ALGORITHM.md) §4: **DRaFT** §4 covers **deterministic** suffix truncation
with checkpointing; **ARTBP** covers **unbiased reweighted variable-length** truncation.
Those raise a **novelty objection**, not an impossibility (X27).

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

**Corrected treatment of the clamp (X22).** The gate is clamped, so `L(·, ξ)` is only
piecewise differentiable in `λ`, with `∂m/∂λ = 0` on the clamped region. That region
**can have positive measure** — it is **not** a measure-zero set, and the earlier claim that
"the clamp set has measure zero in `λ` for fixed `ξ`" is withdrawn. What *is* measure-zero is
the **transition boundary** between clamped and unclamped regions; the two must be
distinguished. With `P(clamped at 1) = 0.9659` at `λ = 3` (§1.3), the per-gate derivative is
an exact zero for ~96.6% of draws at initialization — a real feature of the relaxation at
these settings, affecting signal density.

**Corrected treatment of sampling (X21).** A **coherent single reparameterized sample can be
unbiased** for `∇_λ J` under (i) and (ii); **multiple samples within an iteration are not
required**, and the earlier suggestion that unbiasedness needs an averaging "the
implementation does not realize" is withdrawn. **Initial-latent randomness and mask
randomness are separate axes**: the fixed seed (§1.2c) constrains coverage of `x_T`, which
is a coverage limitation, not a failure of per-sample unbiasedness in `ξ`.

### 5.3 What the code actually computes — fresh replay, and what that does and does not imply

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

**Corrected characterization (X23).** Fresh replay **does not enforce conditional pathwise
consistency**: the linearization noise differs from the noise that produced the stored
trajectory. It therefore **removes the general unbiasedness guarantee** that a coherent
single reparameterized sample would carry (§5.2). But **nonzero bias of this actual FLUX
system was not analytically established**, and none is claimed here. The earlier flat
assertion that the implemented quantity is "**not** an unbiased estimate of `∇_λ J`" is
withdrawn as unproven; so is the framing that the native reference computes a gradient
"that is not the exact gradient of its own forward trajectory" **as if the resulting bias
were derived**. No measured or derived native bias is claimed anywhere in this record.

> **On RNG-preserving replay (X13).** `torch.utils.checkpoint`'s `preserve_rng_state`
> **defaults to `True`** and the recompute runs under `torch.random.fork_rng` with the
> stashed forward state restored (static read of the locally installed
> `torch/utils/checkpoint.py:291-297, 477, 1507-1513`); the published trainer hand-rolls its
> own checkpointing and omits this. **RNG-preserving replay changes the reference contract
> — it is not proof of an unchanged native gradient**, and it does not establish what the
> native gradient "should" have been.
>
> This remains a **reference-definition** issue, not a saving, and it is a **confound** for
> any future efficiency comparison: two arms differing in mask-noise consistency are
> estimating with respect to **different stochastic references**. Every cost expression in
> [`COST_MODEL.md`](COST_MODEL.md) therefore names the reference it concerns.

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
accumulation (★) and its two-sweep structure (§3); the gate form and hence the **local
identity** (†) (§4 — **not** its withdrawn corollary); the **target-construction** and
**fixed-seed coverage** differences (§1.2a, §1.2c); `K = 5`; the bounded-noise relaxation
and `P(clamp) = 0.9659` at `λ = 3`; the **fresh-replay** RNG behavior (§5.3); targets cached
and never recomputed; no VAE decode in the training graph; no `L0` term in either the code's
trained loss or the paper's Eq. 16.

**Withdrawn from the "derived" list:** the impossibility corollary (X1); the exact-gradient
"floor" as a theorem (X5); the "four objective divergences" count — (b) and (d) are not
divergences (X19, X20); the `image[0]` batch-index reading (X24); and any derived native
bias (X23).

**Assumed** (and carried as uncertainty, never asserted):

1. `configs/flux.yaml` is representative of the published FLUX.1-schnell run. The paper's
   Table 6 **disagrees** with it (`β`: 0.01 vs `0.1`; batch 4 vs `1`; 400 steps vs ≈80
   updates — see [`COST_MODEL.md`](COST_MODEL.md) §2). Unresolved.
2. `with_flowedit=True` (the code default) is the published target construction. The paper
   says `ℱθ′(x_T, ∅)`. **Unresolved, and essential.**
3. A reverse sweep over one step's graph costs `c_b · F` with `c_b ∈ [1,2]`. **Unmeasured**,
   and a **single** `c_b` is a simplification: state-only, mask-only and joint VJP costs may
   differ, and [`COST_MODEL.md`](COST_MODEL.md) §1 now carries `c_state`, `c_mask`, `c_joint`
   separately.
4. Hook ordering stays aligned with `hook_counter`
   (`linear_layer_hooks.py:106-107`, modulo arithmetic over a fixed traversal order). A
   static-reading assumption we did not validate at runtime.
5. The local identity (†) assumes `epsilon = 0`. At `epsilon > 0` the gate gains the additive
   branch `randn·epsilon·(1−m)` and (†) acquires a second term; we only checked the published
   `epsilon: 0.0`.

**Not established:** which of paper and code is authoritative; any runtime behavior, timing
or memory figure; whether the snapshot is the publication implementation — **later
source-release dates alone do not prove a different algorithm was used** (X26); and the bias
of the fresh-replay estimator, which was **not** analytically established (X23).

---

## 7. Arithmetic checks

Tiny deterministic checks only (exact rationals / scalar `math`), with no tensors, no model,
no learned system, no simulation and no sweep:

- **Gate saturation** (§1.3): `P(m = 1)` at `λ ∈ {3, 1, 0}` = `0.9659 / 0.0417 / 0.0000`;
  noise range `[−1.0986, +1.0986]` at `eps = 0.5`.
- **Reverse accumulation (★)** reproduces the analytically substituted gradient exactly on a
  2-step rational chain. Full worked example in [`ALGORITHM.md`](ALGORITHM.md) §3, with the
  corrected variance reading.
- **Amendment counterexamples** (§4.1, [`CORRECTIONS.md`](CORRECTIONS.md) §G): expected
  reverse depth `3/2` and expected main cost `8F` vs `A2 = 10F`; and zero importance-sampling
  variance at `v = (10,15)`, `q = (2/5,3/5)`.

Synthetic arithmetic is **not** FLUX evidence and certifies only the algebra. **Scalar
counterexamples are not model evidence** and are not reinterpreted as such.
