# Attempted formulation and scoped stop

**NO EXPERIMENTS.** The pseudocode below is **unexecuted and deliberately not runnable**
as a training implementation. No model, tensor, dataset or research code was involved.

> **Outcome: no algorithm is proposed.** One formulation was developed to the point where
> it could be judged, and it is **exactly dominated**. This document records it in full,
> with the dominance proof, so the attempt is reviewable rather than merely asserted.

---

## 1. The formulation that was developed: SSMG

**Step-Subsampled Mask Gradient with importance weighting.**

Chosen *after* mapping the source graph (not from a shortlist), because it is the only
change that targets the one thing the native loop does `K` times with a **dedicated**
reverse traversal: the mask term `(∂f_k/∂λ)ᵀ ḡ_{k+1}` at `scripts/train.py:320`. The native
loop spends a **whole second sweep per step** on it (`DERIVATION.md` §3), so a method that
evaluates it at only a subset of steps looks, from the code alone, like a real saving.

### 1.1 Definition

Keep the exact reverse state recursion; subsample only the mask accumulation. With
reference gradient (`DERIVATION.md` ★)

```
g(λ, ξ)  =  Σ_{k=0}^{K−1}  Jᵏ_λ ᵀ  ḡ_{k+1},        Jᵏ_λ ≔ ∂f_k/∂λ |_{z_k, ξ_k}
```

fix a selection distribution `q = (q_0 … q_{K−1})`, `q_k > 0`, `Σ q_k = 1`. Draw
`S ⊂ {0…K−1}`, `|S| = S₀`, by `S₀` independent draws from `q` (with replacement), and set

```
ĝ(λ, ξ, S)  =  (1/S₀) · Σ_{k∈S}  (1/q_k) · Jᵏ_λ ᵀ ḡ_{k+1}                        (SSMG)
```

**Objective: unchanged.** SSMG alters neither the loss, the gate, the relaxation, the
binarization, nor the sampling distribution of `x_T` or `ξ`. It is purely an estimator of
the same `∇_λ L(λ, ξ)`.

### 1.2 Exactness, bias, variance

**Unbiased, conditionally on the trajectory.** `E_{S~q}[ĝ] = Σ_k q_k · (1/q_k) Jᵏ_λᵀḡ_{k+1} = g`.
(This inherits whatever fidelity the underlying `ḡ` has — at the published settings that is
the mismatched-realization quantity of `DERIVATION.md` §5.3, **not** the pathwise gradient.
SSMG does not fix that and does not make it worse.)

**Variance.** With per-step contributions `v_k ≔ Jᵏ_λᵀḡ_{k+1}`,

```
Var[ĝ]  =  (1/S₀) · ( Σ_k  ‖v_k‖² / q_k  −  ‖g‖² )
```

minimized over `q` at `q_k* ∝ ‖v_k‖` (Cauchy–Schwarz), giving
`Var_min = ( (Σ_k ‖v_k‖)² − ‖g‖² ) / S₀`. Zero only if all but one `v_k` vanish.

**Schedule / refresh / hyperparameters** (as the formulation would need them): `S₀ ∈ {1…K}`;
`q` refreshed every `R` iterations from an EMA of `‖v_k‖` with decay `ρ`; warm-up of `W`
full-gradient iterations to initialize the EMA. Cached: the EMA table (`K` scalars —
negligible). Retained/recomputed: unchanged from the native loop.

### 1.3 Why it fails — the dominance argument

Two independent reasons, both from `DERIVATION.md` §4.

**(i) The skipped work is free, so `S₀ < K` saves nothing.** By the lemma (†), `v_k` is an
elementwise byproduct of the activation gradients that the **state** sweep
`ḡ_k = (∂f_k/∂z)ᵀ ḡ_{k+1}` computes anyway. Once the native loop's two sweeps are fused into
one — exact, free, `A1` in [`COST_MODEL.md`](COST_MODEL.md) — the marginal cost of `v_k` is
`O(#neurons × #tokens)` multiply-accumulates against the module's matmuls. Skipping it
saves a rounding error.

**(ii) The state chain pins the cost floor regardless of `S`.** To obtain `ḡ_{k+1}` for the
smallest selected `k` you must sweep steps `K−1, K−2, …, k+1` anyway. So SSMG's cost is set
by `min(S)`, not `|S|`:

```
cost(SSMG)  =  K·F  +  (K − min(S))·(1 + c_b)·F  +  O(ε)
```

Choosing `q` to put mass on late steps makes it cheap **and** makes it truncation — at which
point it *is* DRaFT-K with an importance correction, and the correction is exactly what
forces the sweep back to step `min(S) = 0` with probability `1 − (1−q_0)^{S₀}`. **Unbiasedness
and cheapness are mutually exclusive here.** Importance weighting cannot be both unbiased
and truncated.

**(iii) The information it needs costs what it claims to save.** A good `q` needs `‖v_k‖`
for every `k` — obtainable only from the full sweep SSMG exists to avoid. The EMA +
warm-up dodge is self-defeating: `W` full-gradient iterations are charged in full
([`COST_MODEL.md`](COST_MODEL.md) §5), the EMA is stale exactly when `λ` is moving, and the
clamp dynamics guarantee it moves non-stationarily (96.6% → 4.2% saturation between
`λ = 3` and `λ = 1`, `DERIVATION.md` §1.3).

**Net:** SSMG is unbiased, adds variance, needs information it cannot afford, and saves
approximately nothing. It is **strictly dominated by the exact fused sweep (A1)**.

### 1.4 The condition under which SSMG *would* have been viable — and why this graph is not it

SSMG pays off only if the per-step **mask** Jacobian-vector product is expensive *relative
to* the state VJP, i.e. if `cost(Jᵏ_λᵀ v) ≳ cost((∂f_k/∂z)ᵀ v)`. That needs the mask to
enter through a map whose own linearization costs real arithmetic — e.g. a **dense weight
mask** `W ⊙ M` (a separate matmul-scale VJP per step), a low-rank or
hypernetwork-parameterized `λ ↦ m`, or a mask on a quantity not already on the state path.

The published method is the opposite case: **per-neuron scalar gates on activations already
on the state path** (`DERIVATION.md` §1.3). The paper chose this deliberately — *"neuron
masking instead of weight masking to reduce the number of trainable parameters"* (§3.6).
**The paper's own design choice is what removes the headroom SSMG would need.** Had
Minimalist used weight masking, SSMG would be worth analyzing; it does not.

---

## 2. What about the genuinely cheaper options?

They exist, they are **exact**, and they are **ordinary** — which is why they are
alternatives in the cost model and not contributions.

| | change | gradient | cost (per iter) |
|---|---|---|---|
| **A1** | one `autograd.grad(latents, trainable_lambs + [current_latents], …)` instead of two calls at `train.py:320,333` | **bit-identical** | `K·F·(2+c_b)` |
| **A2** | keep the graph across all `K` steps; drop the no-grad pre-pass and the recompute | **exact** | `K·F·(1+c_b)` = **the floor** |
| **A3** | stash/restore mask RNG across recompute (`preserve_rng_state`, torch's default) | **changes identity** → true pathwise | free |
| **A4** | raise `plot_interval`; stop regenerating 8 images every 4 iterations | no change | removes ≈`10·F`/iter |

A1 is a one-line correction of autograd usage. A2 is the textbook memory/time trade the
paper explicitly declined (§3.7: *"constant memory complexity regardless of the step"*) —
reversing that choice is a configuration decision, not a method. A3 is using
`torch.utils.checkpoint` as documented. A4 is turning logging down.

**Any of these would show up in a wall-clock table as a "speedup". None is a contribution,
and presenting one as such would be misreporting ordinary engineering as research.**

---

## 3. Worked analytic example — exact rational arithmetic

Smallest chain with non-degenerate per-step Jacobians. Scalar state `z`, scalar relaxed
gate `m`, additive gate injection `f_k(z,m) = a_k z + b_k m`, terminal loss
`ℓ = ½z_K²`, `K = 2`, `z_0 = 0`.

Parameters `a_1 = 2`, `b_0 = 1`, `b_1 = 3`, `m = 1`. Then `z_1 = 1`, `z_2 = 5`, `ℓ = 25/2`,
and by direct substitution `z_2 = (a_1b_0 + b_1)m` so

```
exact  dℓ/dm = z_2 · (a_1 b_0 + b_1) = 5 · 5 = 25
```

Reverse accumulation exactly as `train.py:300-333` sequences it:

| step `k` | mask term `v_k = ḡ_{k+1}·b_k` (`:320`) | state update `ḡ_k = ḡ_{k+1}·a_k` (`:333`) |
|---|---|---|
| `k=1` | `5 · 3 = 15` | `ḡ_1 = 5 · 2 = 10` |
| `k=0` | `10 · 1 = 10` | (discarded — `z_0` is constant) |

`Σ v_k = 15 + 10 = 25` ✓ **matches the exact gradient**, confirming (★).

**SSMG on this example**, `S₀ = 1`, `q = (½, ½)`:

| drawn `k` | estimate `v_k/q_k` |
|---|---|
| `1` | `30` |
| `0` | `20` |

`E[ĝ] = ½·30 + ½·20 = 25` — **unbiased** ✓.
`Var[ĝ] = ½(30−25)² + ½(20−25)² = 25`, i.e. a **relative standard deviation of
`5/25 = 20%`** on a 2-step chain.

**And the saving is zero.** Drawing `k = 0` requires `ḡ_1`, which requires sweeping step 1
— the same step whose `v_1` was just discarded. Drawing `k = 1` alone is truncation, not
SSMG. So on the example SSMG costs what the exact fused sweep costs and returns a
20%-noise estimate of it. **This is the dominance argument, instantiated.**

Verified with exact `fractions.Fraction` arithmetic — scalars only, no tensors, no model,
no simulation. **Synthetic correctness is not FLUX evidence**: it certifies the algebra of
(★) and the unbiasedness/variance claims, and nothing about deletion, utility, image
quality or robustness.

---

## 4. Closest prior art — targeted check around this design

Primary sources, checked for *this* design rather than as a field survey. No broad baseline
search was performed and **no novelty search is claimed.**

| work | what it actually does | distinction / why it bears on SSMG |
|---|---|---|
| **DRaFT / DRaFT-K / DRaFT-LV** — Clark et al., *Directly Fine-Tuning Diffusion Models on Differentiable Rewards*, `arXiv:2309.17400` | Backprops a **terminal** differentiable reward through the sampling chain; **DRaFT-K differentiates through only the last `K` steps**; *"while gradient checkpointing makes it possible to backpropagate through the full sampling chain, optimization speed and overall performance can be substantially improved by truncating backprop through only the last `K` sampling steps"*; `K=1` reported as the best reward-vs-compute tradeoff; DRaFT-LV adds last-step variance reduction | **This is the design space.** Terminal objective + checkpointed backprop through a sampler + truncation, with the compute/quality tradeoff already characterized. Any truncation variant here reduces to DRaFT-K. SSMG's only escape from DRaFT-K is unbiasedness, and §1.3(ii) shows unbiasedness forfeits the saving |
| **AlignProp** — Prabhudesai et al., *Aligning Text-to-Image Diffusion Models with Reward Backpropagation*, `arXiv:2310.03739` | *"end-to-end backpropagation of the reward gradient through the denoising process"*; *"finetunes low-rank adapter weight modules and uses gradient checkpointing, to render its memory usage viable"* | Checkpointed reward backprop through denoising is established practice. Native Minimalist's §3.7 scheme is an instance |
| **`torch.utils.checkpoint`** — verified by static read of the locally installed `torch/utils/checkpoint.py:291-297, 477, 1507-1513` | *"preserve_rng_state(bool, optional): Omit stashing and restoring the RNG state during each checkpoint. Default: `True`"*; recompute runs under `torch.random.fork_rng` with the stashed forward RNG state restored | A3 is **using the standard utility as documented**. The published trainer hand-rolls checkpointing and omits RNG preservation (`DERIVATION.md` §5.3) |
| **Sublinear-memory checkpointing** — Chen et al., `arXiv:1604.06174`, cited by Minimalist §3.7 via "Zhang et al. (2024d)" | store a subset of activations, recompute the rest | Native Minimalist already does the per-step version. A2 is the *other* end of the same trade. No headroom to claim by "adding checkpointing" |
| **Truncated BPTT, and unbiasing it** — Williams & Peng (1990); Tallec & Ollivier, *Unbiasing Truncated Backpropagation Through Time*, `arXiv:1705.08209` | randomized truncation lengths restore unbiasedness in expectation | **Closest prior art to SSMG specifically.** Importance-weighted random truncation of a backprop chain is published, in the general sequence setting. SSMG is the same idea applied to a 5-step sampler, and inherits the same bias/variance trade — with none of the saving, because here the subsampled terms are free |
| **Hard-concrete `L0` gates** — Louizos et al., `arXiv:1712.01312`; **concrete/Gumbel-softmax** — Maddison et al. `arXiv:1611.00712`, Jang et al. `arXiv:1611.01144` | stretched-sigmoid relaxation of binary gates, reparameterized | The relaxation Minimalist cites. Note the published config instantiates a **bounded-noise** variant (`eps = 0.5`), not Louizos' distribution (`DERIVATION.md` §1.3) — a deviation, not an improvement, and not ours to claim |
| Score-function / REINFORCE and low-variance discrete estimators (REBAR, RELAX, ARMS) | unbiased gradients for discrete variables without pathwise differentiability | Not cheaper: they still require a terminal loss evaluation per sample, i.e. a full `K`-step forward, and lose the pathwise variance reduction. Strictly worse on this graph |

**Conclusion of the prior-art check:** every lever available on this computation graph is
either (a) published (DRaFT-K, AlignProp, randomized-truncation unbiasing, checkpointing),
or (b) ordinary correct usage of existing tooling (A1–A3), or (c) turning off logging (A4).
SSMG is (a) applied to a graph that removes its benefit. **Absence from our gap map is not
evidence of novelty, and nothing here establishes novelty for anyone.**

---

## 5. Stopping conditions — which fire, and why

Against the four stated conditions:

| condition | fires? | reason |
|---|---|---|
| **1 — reduces to an ordinary existing technique without an additional useful claim** | **YES** | Truncation = DRaFT-K; importance-weighted truncation = Tallec & Ollivier applied to a sampler; A1–A3 = correct autograd/checkpoint usage; A4 = logging config. No additional technical claim survives |
| **2 — lacks plausible net savings after overhead and possibly more updates** | **YES** | Lemma (†): the subsampled terms are free byproducts; §1.3(ii): the state chain pins cost at `min(S)`; §1.3(iii): building `q` costs the sweeps it skips. Added variance ⇒ update count can only rise ([`COST_MODEL.md`](COST_MODEL.md) §5) |
| **3 — depends on weakening deletion, utility, image quality or robustness** | not reached | SSMG leaves the objective unchanged, so this was never the binding problem |
| **4 — requires unavailable information, or an essential unresolved source/objective mismatch** | **YES, independently** | The reference objective cannot be established: paper Eq. 16 forget target `ℱθ′(x_T, ∅)` vs. code default FlowEdit on **FLUX.1-dev** at `T_steps=28`; sum-of-two-terms vs. term replacement; `β` 0.01 vs 0.1; 400 steps vs ≈80 updates; `E_{x_T}` vs one fixed `x_T` (`DERIVATION.md` §1.2). **Condition 4 alone would halt the sprint before any design question.** |

Three of four fire independently. Condition 2 is the technically decisive one: it would hold
**even if** the paper and code agreed perfectly.

---

## STOP — the mask gradient is a free byproduct of the state-chain reverse sweep, so no estimator can reduce it; all real headroom is exact and ordinary (fused VJP, RNG-preserving checkpointing, published DRaFT-K truncation), and the reference objective cannot be established because the paper's erasure target and the code's default target are different objectives.

**Scope of this stop.** It applies to **this formulation** — a cheaper gradient computation
or estimator for native Minimalist mask training. It is **not** a claim that:

- Minimalist's training cost is irreducible — **A1, A2 and A4 are real reducible costs**, and
  A4 is the largest single one;
- gradient-estimator research for erasure, for few-step samplers, or for masked generative
  models is closed;
- weight-masking erasure methods lack this headroom — §1.4 identifies exactly the structure
  that would restore it;
- the paper is wrong. We identified a mismatch; we did **not** adjudicate it.

**No replacement project is appended.** No candidate was reranked, revived or created. No
follow-up job, monitor or scheduled task exists, and no future computation is authorized.
