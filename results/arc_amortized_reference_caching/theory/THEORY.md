# SPEX — Self-Projected EXclusion: statements and proofs

All claims below are verified twice by `scripts/verify_theory.py`: symbolically with
sympy where tractable, and numerically in float64 on random instances. The script exits
nonzero if any claim fails.

**Notation.** A vocabulary `V`, a *fact set* `F ⊂ V` (the token ids whose probability we
want capped), `¬F = V \ F`. A distribution `p` on `V` with `P := p(F) = Σ_{f∈F} p(f)`.
Logits `z` with `p = softmax(z)`. Target cap `ε ∈ (0,1)`.

---

## Lemma 1 (I-projection onto a capped-mass set; known)

Let `C_ε = {q : q(F) ≤ ε}` and `r = argmin_{q ∈ C_ε} KL(q ‖ p)`. If `P > ε` then

```
r(f) = ε · p(f) / P            for f ∈ F
r(v) = (1-ε) · p(v) / (1-P)    for v ∈ ¬F
```

and if `P ≤ ε` then `r = p`.

*Proof.* For `P ≤ ε`, `p ∈ C_ε` and `KL(p‖p)=0` is the minimum. For `P > ε` the
constraint is active at `q(F) = ε` (decreasing `q(F)` below `ε` is feasible but, by the
block decomposition below, strictly increases the objective). Write any `q` with
`q(F)=ε` via its block masses and within-block conditionals `q = (ε·a, (1-ε)·b)` where
`a` is a distribution on `F` and `b` on `¬F`. Then

```
KL(q‖p) = ε log(ε/P) + (1-ε) log((1-ε)/(1-P)) + ε·KL(a ‖ p_F) + (1-ε)·KL(b ‖ p_¬F)
```

where `p_F, p_¬F` are `p` restricted and renormalised. The first two terms are constant
in `(a,b)`; the last two are nonnegative and vanish iff `a = p_F`, `b = p_¬F`. ∎

This is the target used by the pinned `exclusion_target` (TNT/TNFF; UNDIAL with γ→∞).
**Lemma 1 is not new** — it is restated because Lemma 2 depends on its exact form.

---

## Lemma 2 (rank-2 gradient) — the load-bearing claim

Define the **self-projection** `r̃ := Proj_{C_ε}(p_θ)` — i.e. Lemma 1 applied to the
*student's own* current distribution rather than to a frozen reference. Let
`L(z) = KL(sg(r̃) ‖ softmax(z))` with `sg` the stop-gradient, evaluated at the same `z`
that produced `p_θ`. Then

```
∇_z L = p_θ − r̃ = α · (p_θ ⊙ 1_F)  +  β · (p_θ ⊙ 1_¬F)
```

with the two **scalars**

```
α = 1 − ε/P            β = 1 − (1-ε)/(1-P).
```

Consequently `∇_z L ∈ span{ p_θ⊙1_F , p_θ⊙1_¬F }`, a subspace of dimension **at most 2**,
and `α·P + β·(1-P) = 0`, so the update is exactly a **mass transfer across the F/¬F
boundary that conserves total probability**.

*Proof.* For cross-entropy against a fixed target `t`, `∇_z KL(t ‖ softmax(z)) = softmax(z) − t`.
Here `t = r̃`. Substituting Lemma 1 componentwise: for `f ∈ F`,
`p_θ(f) − ε p_θ(f)/P = p_θ(f)(1 − ε/P) = α p_θ(f)`; for `v ∈ ¬F`,
`p_θ(v) − (1-ε)p_θ(v)/(1-P) = β p_θ(v)`. Summing,
`αP + β(1-P) = (P-ε) + ((1-P)-(1-ε)) = 0`. ∎

### Why this is the efficiency argument

**Corollary 2.1 (within-block shape is a null direction).** For any `v, v' ∈ ¬F`, the
gradient leaves the ratio `p_θ(v)/p_θ(v')` stationary to first order: the logit update is
`−η β p_θ(v)` and `−η β p_θ(v')`, and
`d/dη [ log p_θ(v) − log p_θ(v') ] = −β(p_θ(v) − p_θ(v')) + β(p_θ(v) − p_θ(v')) = 0`
after accounting for the softmax Jacobian's shared term (verified numerically).
So the relative shape **within** `¬F` is preserved by construction.

**This is what removes the frozen reference.** MDU, EMDU and SSD need a reference (or
anchor) forward pass to stop the forget term from reshaping the non-fact distribution.
Under Lemma 2 the target's own algebra forbids that reshaping, so no reference forward is
required — **1 forward/step instead of 2**.

**Scope limit, stated as part of the claim.** Lemma 2 is a statement about the gradient
*at the current parameters, per position*. Across many steps `p_θ` moves, and positions
share parameters, so trajectory-level shape preservation does **not** follow. Whether
degeneracy appears over a real training run is an empirical question, gated in the
protocol, not a corollary.

---

## Proposition 3 (monotone descent on the fact mass, and its rate)

Take one gradient step on the logits with learning rate `η > 0` (logit-space descent, the
idealisation used for the rate; the network case is the empirical question above). Then to
first order in `η`

```
dP/dη = −(P − ε) · ( Σ_{f∈F} p_θ(f)² / P² · ... )  ≤ 0,
```

and more usefully, the exact first-order expression is

```
dP/dη = −[ α·(S_F − P·S) + β·(−P·S_¬F) ]   with S_F = Σ_{f∈F} p(f)², S_¬F = Σ_{v∉F} p(v)², S = S_F + S_¬F
```

which simplifies to

```
dP/dη = −(P − ε) · [ S_F/P + S_¬F/(1−P) ]  ≤ 0, with equality iff P = ε.
```

So `P` decreases monotonically toward the cap `ε`, at a rate proportional to the current
violation `(P − ε)` times a positive curvature factor. `P = ε` is the unique fixed point
in `(0,1)`.

*Proof.* The softmax Jacobian is `∂p/∂z = diag(p) − p pᵀ`. With `∇_z L` from Lemma 2,
`dP/dη = −1_Fᵀ (diag(p) − p pᵀ) ∇_z L`. Expanding `1_Fᵀ diag(p) ∇_z L = α S_F + β·0`-type
terms and `1_Fᵀ p pᵀ ∇_z L = P·(α S_F + β S_¬F)/…` and substituting
`α = (P−ε)/P`, `β = −(P−ε)/(1−P)` gives the displayed form. Verified symbolically. ∎

---

## Proposition 4 (forward-pass cost)

Per optimizer step, per microbatch, counting full model forwards:

| method | student fwd | reference/anchor fwd | extra | total |
|---|---|---|---|---|
| **SPEX** | 1 | **0** | 0 | **1** |
| MDU τ=1 | 1 | 1 (prompt-masked anchor) | 0 | 2 |
| EMDU | 1 | 1 (fully-masked anchor) | 0 | 2 |
| SSD (control Z) | 1 | 1 teacher + 1 control canvas | 0 | 3 |
| TD-SSD (X) | 1 | 1 control canvas | 0 | 2 |
| FX-Mix | 1 | 1 | `J·n_prompts/R` sampler rollouts | 2 + rollouts |

*Proof.* By inspection of the pinned implementations (`llada_fxmix.py`,
`tdssd.py`) and of SPEX's definition, which references only `p_θ` at the states it
already computed. ∎

**Predicted speedup vs the 2-forward family: 2.0× in forwards**, less in wall time
because of optimizer and data overhead. The measured number is the claim; this is the
upper bound.

---

## Relation to prior art (checked before claiming novelty)

- The **target** `r` is the known TNT/TNFF exclusion target (UNDIAL with γ→∞); Lemma 1 is
  standard and appears in the pinned `mdmu/fxmix.py`.
- The **novel component is only** applying it to `p_θ` rather than `p0`, plus Lemma 2 as
  the justification for dropping the reference pass.
- This is a form of **self-distillation onto a constraint set**. If that exact
  construction exists in the unlearning literature, SPEX is not new and the contribution
  reduces to the cost measurement. That possibility is recorded here, not hidden.
- **No claim** of certified deletion, benchmark superiority, or established novelty.

---

# ⛔ DISPROOF AND PIVOT (recorded 05:05 ET, 0:08 into the sprint)

`scripts/verify_theory.py` → **10/13 claims pass**. Lemma 1 and Lemma 2 hold. **Corollary
2.1 is FALSE**, and Proposition 3's rate algebra was wrong. Nothing below is deleted; the
original statements stand above as what was attempted.

## Corollary 2.1 is false — and it was the load-bearing claim

The gradient `∇_z L = α(p⊙1_F) + β(p⊙1_¬F)` is rank-2 **as a vector in distribution
space**, but the *logit update* it induces is `Δz(v) = −η β p(v)` for `v ∈ ¬F`, which is
**proportional to `p(v)`, not constant on the block**. Measured on a random instance
(`P = 0.5353`): the ratios of `−g(v)` across `¬F` are
`[1, 2.391, 0.817, 6.696, 86.63, 2.218, 1.716, 15.81, 12.12]` — identical to the ratios of
`p(v)` itself. Central-difference derivative of `log p(v) − log p(v')` along `−g` reaches
**0.0594**, not 0.

So the self-projection target **does reshape the non-fact distribution**. The claim
"no reference forward is needed because the target protects ¬F shape" is **false**.

## The sharper negative result this gives (new, and worth keeping)

**Proposition 5 (only block-uniform logit shifts preserve within-block shape).**
A logit update `Δz` leaves every within-block ratio `p(v)/p(v')`, `v,v' ∈ B`, exactly
invariant **iff** `Δz` is constant on `B`. Since `softmax(z + c·1_B)` rescales block `B`'s
mass and leaves its internal conditionals untouched, and conversely any non-constant
`Δz|_B` changes some ratio, the exact shape-preserving edits are precisely
`Δz = δ_F 1_F + δ_¬F 1_¬F` with `δ_F = log(ε/P)`, `δ_¬F = log((1−ε)/(1−P))`.

**Consequence.** The *only* exactly shape-preserving realisation of the exclusion target
is a **token-level bias on `F`** — i.e. token banning. The pinned prior study measured
that this channel carries **91–96 % of the collateral damage**. So the shape-preserving
edit is exactly the damaging one, and any gradient-based realisation necessarily reshapes
`¬F` and therefore **needs an anchor**.

**This explains why MDU, EMDU, SSD and FX-Mix all carry a reference/anchor forward.** It
is not an implementation habit; it is forced. That is a negative result with a proof, and
it rules out the whole "drop the reference by algebra" family — including SPEX as
originally specified.

## Proposition 3's rate: corrected

The verifier's symbolic check gives the two expressions explicitly:

```
true  : (P-ε)·(P²·S_¬F + (P-1)·(P·(S_F+S_¬F) − S_F)) / (P·(P-1))
claimed: (P-ε)·(P·S_¬F − S_F·(P-1)) / (P·(P-1))
```

They differ, so my hand algebra was wrong (worst relative error 0.894). **Monotonicity
itself survives** — `P3_fact_mass_decreases_monotonically` and
`P3_P_equals_eps_is_a_fixed_point` both pass numerically over 300 random instances. Only
the closed-form *rate* was misderived. The corrected rate is the first expression above,
confirmed symbolically.

## Pivot: ARC — Amortized Reference Caching

The disproof says the reference pass cannot be removed by algebra. It does **not** say the
reference pass must be *recomputed every step*. The reference model is **frozen**, so
`p0(·|x)` is a pure function of the state `x`. The prior methods recompute it only because
they draw **fresh random states** each step.

**ARC:** fix a finite pool of `K` masked states up front, precompute and cache the frozen
reference's exclusion target on each, then train by sampling states from the pool. The
per-step forward count drops from 2 to **1**, with the reference cost paid **once**.

- **Exactness.** On a cached state the target is *bit-identical* to recomputing it — the
  reference is frozen and the state is fixed. There is **no approximation in the target**.
- **The one honest approximation** is statistical: training sees `K` states instead of
  fresh draws, so the state distribution is a finite sample. That changes the *estimator's
  variance*, not the target's correctness, and it is directly measurable by sweeping `K`.
- **Cost.** Baseline: `2·S` forwards for `S` steps. ARC: `K + S`. For the pinned recipe
  (`S = 150`, 8 states/step → 1200 reference forwards) a pool of `K = 256` gives
  `256 + 150` vs `1200 + 150` → **~3.3× fewer reference forwards**, and 1 vs 2 per step in
  steady state.

ARC's claims are therefore (a) target exactness — provable and bit-checkable, and (b) a
measured cost reduction, with (c) a measured efficacy-vs-`K` curve as the honest cost of
the approximation. This is what the remaining sprint time tests.

---

# ARC — Amortized Reference Caching: statements and proofs

**Setting.** A frozen reference model `M0` (here: the qualified memorized checkpoint), a
student `M_θ` carrying a LoRA. A *state* is a concrete masked token canvas `x`. The forget
loss of the reference-based family is, at masked answer positions `j` of state `x`,

```
L(θ; x) = mean_j  D( T(p0(·|x)_j) ‖ p_θ(·|x)_j )
```

for some fixed target map `T` (for the exclusion family, `T = Proj_{C_ε}` of Lemma 1; for
MDU/EMDU, `T` is the τ-scaled anchor). Baseline implementations draw a *fresh* random `x`
each step and therefore evaluate `p0(·|x)` with a **reference forward** every step.

## Proposition 6 (target exactness under caching)

Because `M0` is frozen and `T` is a deterministic function of `p0(·|x)`, the target
`T(p0(·|x))` is a **pure function of the state `x`**. Hence for any state `x` whose
reference target was computed once and stored, reusing the stored value yields a loss
value and gradient **bit-identical** to recomputing it.

*Proof.* `θ`-independence: `p0` does not depend on `θ`, and in all the arms above the
reference enters only through `sg(·)`/`no_grad`, so `∂L/∂θ` involves the target only as a
constant. Determinism: `M0` in eval mode with fixed dtype and no dropout is a deterministic
map `x ↦ logits`; `T` is deterministic. Therefore `T(p0(·|x))` cached equals
`T(p0(·|x))` recomputed, and both the value and the gradient agree exactly. ∎

**This is checkable, not just arguable**, and the protocol checks it: compute the target
twice (fresh vs cached) on the same states and require **max |Δ| = 0** in the stored dtype.

## Proposition 7 (cost)

Let `S` = optimizer steps, `m` = states per step, `K` = cached pool size. Counting full
model forwards:

```
baseline  :  S·m (student)  +  S·m (reference)        = 2·S·m
ARC       :  S·m (student)  +  K      (reference, once) = S·m + K
```

Steady-state forwards per step: **2m → m**, i.e. a factor **2**. Total-forward ratio
`2Sm / (Sm + K)`, which for the pinned recipe (`S = 150`, `m = 8`, `K = 256`) is
`2400 / 1456 ≈ **1.65×**` overall and exactly **2×** in the per-step limit `K ≪ S·m`.

*Proof.* By construction: ARC performs the reference pass once per pooled state and never
again; all other work is unchanged. ∎

## The one approximation, stated precisely

ARC does **not** approximate the target. It approximates the **state distribution**: the
baseline's per-step state is a fresh draw from the masking distribution `q(x)`, while ARC
draws uniformly from a fixed `K`-sample of `q`. So ARC optimises an empirical objective

```
L_K(θ) = (1/K) Σ_{i=1..K} L(θ; x_i)      instead of      L(θ) = E_{x~q} L(θ; x)
```

`L_K` is an unbiased estimator of `L` at every `θ` (the pool is drawn from `q` before
training), but a **fixed** sample: it has zero per-step resampling noise and a nonzero
generalisation gap in `K`. Two consequences, both measurable, neither hidden:

1. **Overfitting to the pool** is possible for small `K` — the student can reduce `L_K`
   by exploiting the particular masks sampled.
2. The **efficacy-vs-`K` curve** is the honest price of the speedup, and the protocol
   sweeps `K ∈ {64, 256, 1024}` plus the uncached control to locate it.

## What ARC does and does not claim

- **Claims:** (i) exact targets under caching (Prop 6, bit-checked); (ii) 2× fewer
  forwards per step and a measured wall-time reduction (Prop 7); (iii) a measured
  efficacy-vs-`K` curve.
- **Does not claim:** better forgetting than any published method, novelty of the
  exclusion target, certified deletion, or that a finite pool is free.
- **Prior-art honesty:** caching a frozen teacher's outputs is standard practice in
  knowledge distillation (precomputed teacher logits). ARC's content is therefore *not*
  the idea of caching but (a) the observation that the MDLM unlearning family's
  per-step reference pass is removable *because the state space can be pre-sampled*, and
  (b) the measured cost/efficacy trade-off on LLaDA-8B. If a published unlearning method
  already caches reference targets over a pre-sampled state pool, ARC is not new and the
  contribution reduces to the measurement. Recorded, not hidden.

### Correction to Proposition 7 (measured, 05:20 ET)

The smoke run shows **ARC: student=2, reference=0 forwards/step**. Prop 7 as first written
counted only the *forget* forwards and therefore overstated the ratio. The honest count
includes the retain term's own student forward, which both arms pay:

| per step | forget student | forget reference | retain student | **total** |
|---|---|---|---|---|
| uncached | 1 | 1 | 1 | **3** |
| **ARC** | 1 | **0** | 1 | **2** |

So the correct steady-state claim is **3 → 2 total forwards, a factor 1.5×**, not 2×. The
*reference* forwards specifically drop from `S` (one per step) to `K` (one per pooled
state, paid once): for `S = 150`, `K = 256` that is 150 → 256 in absolute terms, i.e.
**caching only wins on reference forwards when `K < S`**. With `K = 256 > S = 150` the
prefill actually costs *more* reference forwards than the baseline spends.

**This materially narrows the claim and is stated before the results:**
- the per-step steady-state saving is real and measured (3 → 2 forwards, and the wall-time
  effect is what the runs measure);
- the *total*-compute saving requires `K < S`, so `K = 64` is the only pool size in this
  sweep that can reduce total reference forwards (64 < 150), while `K = 256` and `K = 1024`
  trade more prefill for lower per-step cost;
- the honest framing is therefore **amortisation**, not free speedup: ARC moves reference
  compute out of the training loop, which pays off when `S ≫ K` (long runs, small pools) and
  loses when `K > S`.

The `K` sweep is accordingly also a *cost* sweep, not only an efficacy sweep.

### Correction #2 to Proposition 7 — the measured truth (05:25 ET)

My first correction was **also wrong**: it assumed one reference forward per *state*. In
both implementations the reference pass is **batched over `m` states at a time**, so

```
baseline reference forwards = S            (one batched pass per step)
ARC      reference forwards = ceil(K/m)    (one batched pass per pooled batch, once)
```

Measured on identical settings (`S = 150`, `m = 8`):

| arm | fwd/step (student + reference) | s/step | train wall | prefill | reference fwd total | peak GiB |
|---|---|---|---|---|---|---|
| uncached | 2 + 1 = **3** | 0.4840 | 74.9 s | 0 s | **150** | 23.685 |
| ARC `K=256` | 2 + **0** | 0.3924 | 60.6 s | 3.5 s | **32** | 25.838 |
| ARC `K=64` | 2 + **0** | 0.3906 | 60.2 s | 1.1 s | **8** | 24.361 |

So the corrected, measured claims are:

- **per-step forwards 3 → 2** (factor **1.5**), exactly as the batched count predicts;
- **in-loop s/step 0.484 → 0.392** = **1.23×** faster;
- **end-to-end wall** (train + prefill) 74.9 → 64.1 s = **1.17×** faster at `K=256`;
- **reference forwards 150 → 32** = **4.7×** fewer at `K=256`, and 150 → 8 = **18.8×** at
  `K=64`. Caching wins on reference compute whenever `ceil(K/m) < S`, i.e. `K < m·S = 1200`
  here — which *contradicts my first correction's* `K < S` condition. That condition was
  derived under the wrong (unbatched) model and is withdrawn.
- **memory cost is real**: peak rises 23.7 → 25.8 GiB at `K=256`, because the cached
  targets are held resident. That is the price of amortisation and is reported, not hidden.

Two mis-derivations of the same proposition are recorded above rather than overwritten.
The measured table is the claim; the arithmetic that preceded it was twice wrong.
