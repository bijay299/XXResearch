# Attempted formulation and scoped stop

**NO EXPERIMENTS.** The pseudocode below is **unexecuted and deliberately not runnable**
as a training implementation. No model, tensor, dataset or research code was involved.

> **Amended 2026-10-17 — see [`CORRECTIONS.md`](CORRECTIONS.md).** The original publication
> ([`53d3fb1`](https://github.com/bijay299/XXResearch/tree/53d3fb1c287b735176a8a1a192ec94ca48d1d3b3/results/minimalist_erasure_design_sprint))
> declared this formulation **"exactly dominated"** and claimed it **"cannot break even for
> any `q`, `S₀`, `R` or `W`"**. **Both claims are withdrawn** (X1–X3), along with
> "unbiasedness and cheapness are mutually exclusive" (X2), the variance-vanishing condition
> (X6), the `ν ≥ 1` inference (X8), and the "reduces to DRaFT-K" attribution (X27).
>
> **The formulation is preserved here as a record. It is not promoted to a proposal, and no
> selection distribution is recommended.**

> **Outcome: no algorithm is proposed, and none is ruled out.** One formulation was developed
> far enough to be judged, and its **novelty, variance/convergence and actual total-cost
> benefit remain unresolved**. That is the ground for the STOP — **not** a dominance proof.

---

## 1. The formulation that was developed: SSMG

**Step-Subsampled Mask Gradient with importance weighting.**

Chosen *after* mapping the source graph (not from a shortlist), because it targets the one
thing the native loop does `K` times with a **dedicated** reverse traversal: the mask term
`(∂f_k/∂λ)ᵀ ḡ_{k+1}` at `scripts/train.py:320`.

### 1.1 Definition

Keep the reverse state recursion; subsample the mask accumulation. With the reference
accumulation (`DERIVATION.md` ★)

```
g(λ, ξ)  =  Σ_{k=0}^{K−1}  Jᵏ_λ ᵀ  ḡ_{k+1},        Jᵏ_λ ≔ ∂f_k/∂λ |_{z_k, ξ_k}
```

fix a selection distribution `q = (q_0 … q_{K−1})`, `q_k > 0`, `Σ q_k = 1`. Draw an
**ordered list (multiset) `S = (I_1 … I_{S₀})` of `S₀` independent draws from `q`, counted
with multiplicity** — not a deduplicated set (X7) — and set

```
ĝ(λ, ξ, S)  =  (1/S₀) · Σ_{s=1}^{S₀}  (1/q_{I_s}) · J^{I_s}_λ ᵀ ḡ_{I_s+1}             (SSMG)
```

**Objective: unchanged.** SSMG alters neither the loss, the gate, the relaxation, the
binarization, nor the sampling distribution of `x_T` or `ξ`. It is purely an estimator of
the same `∇_λ L(λ, ξ)`.

### 1.2 Exactness, bias, variance

**Unbiased, conditionally on the trajectory.**
`E_{S~q}[ĝ] = Σ_k q_k · (1/q_k) Jᵏ_λᵀḡ_{k+1} = g`, counting draws with multiplicity.

This inherits whatever fidelity the underlying `ḡ` has. At the published settings that is
the **fresh-replay** quantity of `DERIVATION.md` §5.3, for which the general unbiasedness
guarantee does **not** hold — though **no nonzero bias was analytically established**
(X23). SSMG neither fixes nor worsens that.

**Variance.** With per-step contributions `v_k ≔ Jᵏ_λᵀḡ_{k+1}`, and `S₀` draws with
multiplicity,

```
Var[ĝ]  =  (1/S₀) · ( Σ_k  ‖v_k‖² / q_k  −  ‖g‖² )
```

minimized over `q` at `q_k* ∝ ‖v_k‖`, giving `Var_min = ( (Σ_k ‖v_k‖)² − ‖g‖² ) / S₀`.
**The formula is retained with its assumptions.**

**Withdrawn (X6): "zero only if all but one contribution vanishes."** Several **nonzero**
contributions give **exactly zero** variance when they are positively collinear and
probabilities are proportional to their magnitudes. The record's own example suffices:

```
v = (10, 15),  q = (2/5, 3/5)   ⇒   estimates (25, 25),  mean 25,  Var = 0
```

Here `Σ‖v_k‖ = 25 = ‖g‖` because the `v_k` are positively collinear, so `Var_min = 0` — fully
consistent with the retained formula, and flatly inconsistent with the withdrawn condition.

**Schedule / refresh / hyperparameters** (as the formulation would need them): `S₀ ∈ {1…K}`;
`q` refreshed every `R` iterations from an EMA of `‖v_k‖` with decay `ρ`; `W` warm-up
iterations to initialize the EMA. Warm-up and refresh work is to be counted as **replacing
or adding iterations**, not as a multiplier on shared one-time preparation
([`COST_MODEL.md`](COST_MODEL.md) §6).

### 1.3 What is unresolved — the actual grounds for stopping

**(i) Novelty — unresolved, and the strongest objection.** SSMG is **ordinary importance
weighting over truncation depth**. **ARTBP** (`arXiv:1705.08209`) already obtains
**unbiased** gradients from **variable-length** truncation using *"carefully chosen
compensation factors"*, on exactly the principle *"if a flow has a probability `p` to occur,
multiplication of the flow by `1/p` restores balance on average"*. We did **not** establish
any additional useful technical claim for SSMG over that construction, and we claim **no**
novelty finding either way (§4).

**(ii) Variance and convergence — unresolved.** The variance formula holds, but the **number
of updates required to reach comparable deletion, retained utility, rendering quality and
robustness (`ν`) is unknown**. We do **not** claim added variance implies `ν ≥ 1` (X8):
update requirements in the actual nonlinear optimization are unknown, and DRaFT's reported
results do not establish that ordering for Minimalist.

**(iii) Actual total-cost benefit — unresolved.** Corrected break-even is
`ν < c_base/c_new` — `5/3` at `c_b = 1`, `15/7` at `c_b = 2`
([`COST_MODEL.md`](COST_MODEL.md) §6). Those thresholds are **not** vanishing, so no
impossibility follows; but **they do not establish useful actual-model savings** either,
and several cost terms (per-iteration prompt encoding, decodes, optimizer, I/O, distinct
VJP costs, memory feasibility) remain unmodelled or unmeasured.

**(iv) Selection overhead and the information requirement — a real cost, not a proof.**
A good `q` needs `‖v_k‖` per step, and obtaining it consumes reverse work. The EMA +
warm-up scheme is charged as iterations, and the EMA is stale while `λ` moves — and it does
move non-stationarily (96.6% → 4.2% gate saturation between `λ = 3` and `λ = 1`,
`DERIVATION.md` §1.3). **This is a genuine unresolved overhead, not the self-defeating
contradiction previously claimed.**

### 1.4 Withdrawn: the dominance argument

The published record argued dominance from three premises. All are withdrawn:

- *"The skipped work is free, so `S₀ < K` saves nothing."* — **Withdrawn (X1).** The local
  identity (`DERIVATION.md` §4) is **conditional on the state adjoint at that site already
  being computed**. It does not apply to an estimator that **stops traversal** at the
  earliest selected contribution.
- *"The state chain pins the cost floor regardless of `S`"*, and *"unbiasedness and
  cheapness are mutually exclusive."* — **Withdrawn (X2, X5).** Cost depends on a **random
  stopping depth**; nonzero probability of a full sweep does not force full expected cost.
  Under the record's own expression `E[C] = K·F + (1+c_b)·F·Σ_i q_i(K−i) + overhead`, with
  `K = 5`, `c_b = 1`, one draw and **all probabilities positive**,
  `q = (1/20,1/20,1/20,1/20,4/5)` gives expected depth `3/2` and **expected main cost `8F`
  against `A2 = 10F`** — a saving is possible within this model if overhead is below `2F`.
- *"Building `q` costs the sweeps it skips," hence no break-even for any parameters.* —
  **Withdrawn as a universal claim (X3).** It remains a real unresolved overhead, §1.3(iv).

**The counterexample refutes the universal proof and nothing more.** It is **not** evidence
of a FLUX speedup, **not** evidence of acceptable variance or convergence, **not** evidence
of novelty, and it is **not** adopted as a new algorithm proposal.

### 1.5 Where SSMG would be most and least attractive

Retained as **observation, not as a necessary condition** (X29). SSMG's appeal grows when
the per-step **mask** Jacobian-vector product is expensive relative to the state VJP — e.g.
a **dense weight mask** `W ⊙ M`, or a low-rank/hypernetwork map `λ ↦ m`. The published
method is a **per-neuron scalar gate on activations already on the state path**
(`DERIVATION.md` §1.3), chosen deliberately — *"neuron masking instead of weight masking to
reduce the number of trainable parameters"* (§3.6). That **reduces the per-term saving**
available in regime (1).

**It does not make savings impossible**, because regime (2) — stopping traversal early —
remains open. The published claim that a future design **must** change the gate or **must**
defeat the local identity to be "mathematically possible" is **withdrawn as invalid** (X29).

---

## 2. Cheaper options that are ordinary

Real and source-supported. **Not** contributions, and **not** proof that nothing else exists.

| | change | status | cost (per iter) |
|---|---|---|---|
| **A1** | one `autograd.grad(latents, trainable_lambs + [current_latents], …)` instead of two calls at `train.py:320,333` | returns both requested derivatives **from the same recomputed graph in exact arithmetic**; **same stochastic reference as native**. **No** bit-identity, fixed-runtime or identical-trajectory claim (X11) | `K·F·(2+c_b)` |
| **A2** | retain the graph across all `K` steps; drop the no-grad pre-pass and the recompute | **a different stochastic reference**: coherent forward graph vs. native fresh mask redraws. **Not** an exact substitute for the native estimator (X12). Ordinary reverse-mode cost, **not a floor theorem** (X5) | `K·F·(1+c_b)` |
| **A3** | stash/restore mask RNG across recompute (`preserve_rng_state`, torch's default) | **changes the reference contract**; **not** proof of an unchanged native gradient (X13) | ≈ free |
| **A4** | raise `plot_interval` | removes instrumentation work; no effect on the gradient | — |

A1 is correct autograd usage. A2 reverses a memory/time choice the paper made explicitly
(§3.7: *"constant memory complexity regardless of the step"*), and its feasibility is an
**unmeasured** memory question. A3 is using `torch.utils.checkpoint` as documented. A4 is
logging configuration.

**None of these is presented as a speedup measurement**, and none would be a contribution.

---

## 3. Worked analytic example — exact rational arithmetic

Smallest chain with non-degenerate per-step Jacobians. Scalar state `z`, scalar relaxed
gate `m`, additive gate injection `f_k(z,m) = a_k z + b_k m`, terminal loss `ℓ = ½z_K²`,
`K = 2`, `z_0 = 0`, parameters `a_1 = 2`, `b_0 = 1`, `b_1 = 3`, `m = 1`.

Then `z_1 = 1`, `z_2 = 5`, `ℓ = 25/2`, and `z_2 = (a_1b_0 + b_1)m`, so

```
exact  dℓ/dm = z_2 · (a_1 b_0 + b_1) = 5 · 5 = 25
```

Reverse accumulation exactly as `train.py:300-333` sequences it:

| step `k` | mask term `v_k = ḡ_{k+1}·b_k` (`:320`) | state update `ḡ_k = ḡ_{k+1}·a_k` (`:333`) |
|---|---|---|
| `k=1` | `5 · 3 = 15` | `ḡ_1 = 5 · 2 = 10` |
| `k=0` | `10 · 1 = 10` | (discarded — `z_0` is constant) |

`Σ v_k = 15 + 10 = 25` ✓ **matches the exact gradient**, confirming (★).

**SSMG on this example**, `S₀ = 1`, `q = (½, ½)`: estimates `30` and `20`, mean `25`
(**unbiased** ✓), `Var = 25` — a relative standard deviation of `20%`.

**Corrected reading of the same example.** The published record concluded from it that the
saving is *"zero"* and the design *"dominated"*. That conclusion is **withdrawn**:

- With `q = (2/5, 3/5)` — proportional to `v = (10, 15)` — **both estimates equal `25` and
  the variance is exactly `0`**, with two nonzero contributions (X6).
- The cost claim assumed the full traversal happens regardless. Under early-stopping
  traversal, drawing `k = 1` requires depth 1 and drawing `k = 0` requires depth 2, so
  expected cost depends on `q` (X1, §1.4).

Verified with exact `fractions.Fraction` arithmetic — scalars only, no tensors, no model, no
simulation. **Synthetic correctness is not FLUX evidence**, and **scalar counterexamples are
not model evidence**: they certify algebra and nothing about deletion, retained utility,
rendering quality or robustness.

---

## 4. Closest prior art — targeted check, corrected attribution

Primary passages consulted for *this* design only. **No broad literature survey**, and
**no novelty finding is claimed**.

| work | inspection depth | what it actually supports | bearing on SSMG |
|---|---|---|---|
| **DRaFT / DRaFT-K / DRaFT-LV** — Clark et al., `arXiv:2309.17400` | **§4 targeted** | **Deterministic suffix truncation and checkpointing**: *"truncates backpropagation to only the last K steps of sampling"*, implemented with `x_t = stop_grad(x_t)`, **no importance weighting or bias correction**. Checkpointing stores step input latents and recomputes activations. DRaFT-LV reduces variance by noising the generated image `n` times | Establishes **fixed-depth** truncation as existing practice. **Does not** cover randomized or reweighted truncation — so the earlier claim that any truncation variant here *"reduces to DRaFT-K"* is **withdrawn** (X27) |
| **ARTBP** — Tallec & Ollivier, *Unbiasing Truncated Backpropagation Through Time*, `arXiv:1705.08209` | **abstract + mechanism** | **Anticipated Reweighted Truncated Backpropagation**: *"keeps the computational benefits of truncated BPTT, while providing unbiasedness"*, via *"variable truncation lengths together with carefully chosen compensation factors in the backpropagation equation"*; *"if a flow has a probability p to occur, multiplication of the flow by 1/p restores balance on average"* | **The closest prior art, and the strongest objection to SSMG.** Unbiased reweighted randomized truncation is **published**, in the general sequence setting. This is a **novelty objection**, **not** an impossibility of expected savings, and **not** equivalence of every variant |
| **AlignProp** — Prabhudesai et al., `arXiv:2310.03739` | **abstract only** | *"end-to-end backpropagation of the reward gradient through the denoising process"*; *"finetunes low-rank adapter weight modules and uses gradient checkpointing"* | Context only. **Carries no decisive claim here** (X28) |
| **`torch.utils.checkpoint`** | **static source read** (`:291-297, 477, 1507-1513`) | `preserve_rng_state` defaults to `True`; recompute under `fork_rng` with stashed state restored | `A3` is standard usage. **Changes the reference contract** (X13) |
| **Hard-concrete / concrete gates** — Louizos et al. `arXiv:1712.01312`; Maddison et al. `arXiv:1611.00712`; Jang et al. `arXiv:1611.01144`; **sublinear checkpointing** — Chen et al. `arXiv:1604.06174` | **cited by identifier only; not inspected** | — | Named to **locate** the design. **No decisive claim rests on them** (X28) |

**Corrected conclusion.** The truncation lever is substantially anticipated — deterministic
by DRaFT, unbiased-and-reweighted by ARTBP. That leaves SSMG with **no established
additional useful claim**, which is a **novelty objection we did not resolve**. It is
**not** a demonstration that expected savings are impossible, and **absence from our gap map
is not evidence of novelty for anyone.**

---

## 5. Stopping conditions — corrected assessment

| condition | fires? | corrected reason |
|---|---|---|
| **1 — reduces to an ordinary existing technique without an additional useful claim** | **YES — the primary ground** | Ordinary importance weighting over truncation depth, substantially anticipated by **ARTBP** (unbiased reweighted variable-length truncation) and **DRaFT** §4 (deterministic suffix truncation with checkpointing). **No additional useful technical claim was established.** Unresolved, not disproven |
| **2 — lacks plausible net savings after overhead and possibly more updates** | **UNRESOLVED, not established** | Corrected break-even `ν < 5/3` (`c_b=1`) and `15/7` (`c_b=2`) is **not vanishing**, and the expected-cost counterexample shows savings are **possible within the model**. But `ν` is **unknown**, selection overhead is unquantified, and cost terms are missing — so **no useful actual-model saving is established either.** The earlier "YES" verdict, and its dominance reasoning, are **withdrawn** |
| **3 — depends on weakening deletion, utility, image quality or robustness** | not reached | SSMG leaves the objective unchanged |
| **4 — requires unavailable information, or an essential unresolved source/objective mismatch** | **YES** | A **source-defined reference objective can be stated**, but its **fidelity to the publication is unverified** (X25): paper Eq. 16's forget target `ℱθ′(x_T, ∅)` vs. the code default's **FlowEdit on FLUX.1-dev**, `T_steps=28`; plus unresolved `β`, batch size and update count. **Identified, not adjudicated** |

Conditions **1** and **4** fire. Condition **2** is **unresolved in both directions** — the
correction removes it as a ground for stopping *and* denies it as a ground for proceeding.

---

## STOP — no distinct, justified improvement has been established. The proposed estimator's novelty, variance/convergence and actual total-cost benefit remain unresolved; fidelity of the source-defined reference to the publication is not established. No experiment is authorized.

**STOP — DESIGN SPRINT CLOSED WITH CORRECTIONS.**

**Scope of this stop.** It applies to **this formulation**, closed as **not established** —
**not** as impossible. It is **not** a claim that:

- estimators cannot reduce expected gradient work for native Minimalist — the withdrawn
  proofs claimed that, and the counterexample refutes them;
- Minimalist's training cost is irreducible;
- ARTBP-style reweighted truncation could not be made to work here — only that **we
  established no additional useful claim** over it;
- gradient-estimator research for erasure, few-step samplers or masked generative models is
  closed;
- the paper is wrong. We identified a fidelity question; we did **not** adjudicate it.

**No replacement project is appended, no successor was selected, and the attempted
formulation is not converted into a proposal.** No follow-up job, monitor or scheduled task
exists, and no future computation is authorized.
