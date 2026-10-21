# H10 — FORGETTING BY COMMIT-ORDER CONTROL

**Owner:** objective and architecture design lead
**Date:** 2026-10-20
**Status:** Tier 2 (active development). Design + crux derivation + prerequisite measurements.
Companion to `OBJECTIVE_DESIGN.md` (§10 Amendment 1 supersedes parts of its body). **I have run no
GPU jobs.**

Scope note carried from the brief: the dataset is now **TOFU800** (400 forget / 400 retain QA,
author-disjoint, 400 distinct answers each side, free-form answers averaging ~27 words). Two
consequences are load-bearing and are developed in §10: the answer span is multi-token with
distinguishable token roles (which is what makes H10 *possible at all*), and the fact-bearing
subset is a **large fraction** of that span (which is what makes H10 *weak*). Both are new relative
to the single-word synthetic set.

Status labels: **[P]** proved here, **[A]** assumption, **[H]** hypothesis needing measurement,
**[M]** measured.

---

## 0. VERDICT FIRST, because the brief asks for the crux before the design

**The mechanism as stated — "lower the answer position's confidence without lowering
`p(answer)`" — is impossible. Proved in §2.1.** At `T = 0` with `low_confidence` remasking the
commit score at a position *is* `max_v p_θ(v|s,i)`, and Lemma 4 says disclosure at that position
requires the answer to *be* the argmax. So on exactly the set of (state, position) pairs where
disclosure can happen, confidence and answer probability are **the same function of θ**. There is no
objective that separates them. If H10 is read as absolute confidence reduction, it is dead, and it is
dead by identity, not by approximation.

**The mechanism survives in its relative form, and only there.** Commit order depends on the
*ranking* of confidences, not their levels. The answer position can be pushed down the ranking by
**raising the confidence of the other positions**, which touches `p_θ(answer|s,i*)` not at all
(§2.2). That reading is literally what the brief says — "relative to its neighbours" — so I take it
as the intended one. It is feasible only because the answer span is multi-token with structural
positions to raise; it would have been infeasible on the old single-word set.

**But the design's empirical premise is the opposite of the formalization's stated hazard, and that
is the real risk — bigger than the crux.** The brief's premise is that a late commit is safe because
"the surrounding text is model-written and no longer determines the answer". §4(f) says the
opposite: later commits condition on *more* context, and in a bidirectional denoiser the context the
model writes around a memorized fact is the context that **entails** it ("…lives in the capital of
France…"). Both are possible; which one obtains is an empirical fact about this model that **nobody
has measured**, and it decides H10 outright.

**It is measurable with zero training.** Run the author's sampler but force the fact-proposing
positions to be committed last — an evaluation-only decode intervention, which §3 explicitly permits
as a reference ("a decode-time `F`-ban is permitted only as an *evaluation reference*, never as the
method"). That is an **oracle for a perfectly-trained H10**: it realizes the target ordering exactly,
with the weights untouched.

> **Job H10-M0 — the oracle, and the single thing that should run before anything else.**
> 100 forget probes × {natural order, forced-fact-last} × {2 configs}. ≈ **7 GPU-minutes.**
> - forced-last disclosure ≈ natural disclosure ⇒ **H10 is dead**; no training can beat its own
>   oracle, and the premise is refuted.
> - forced-last disclosure ≈ 0 ⇒ H10's ceiling is the full gap, and the design below is worth
>   building.
> - forced-last disclosure **above** natural ⇒ re-entailment confirmed; H10 is not merely dead, it
>   is *harmful*, and that is itself a publishable negative about trajectory-aware unlearning.
>
> I am stating in advance, as asked: **if the oracle does not separate, H10 dies and I will say so.**

Everything below §2 is conditional on H10-M0 separating.

---

## 1. The sampler quantity, exactly

From the pinned sampler (`llada_fxmix.generate_traced:240-262`, line-identical to upstream
`generate`), at step `n` from state `s_n`, with `logits = f_θ(s_n)`:

```
logits_with_noise = add_gumbel_noise(logits, T)        # T = 0  ->  returns logits unchanged
x0                = argmax_v logits_with_noise          # proposed token per position
p                 = softmax(logits)                     # NOTE: un-noised
x0_p              = p[ x0 ]                             # the commit score
x0_p[:, beyond current block] = -inf
confidence        = where(masked, x0_p, -inf)
transfer_index    = topk(confidence, k = num_transfer_tokens[n])
```

So, writing `p_i := p_θ(·|s_n,i)`:

> **(1.1)** eligible set `E_n = M(s_n) ∩ B_b`;
> **commit score** `c_i(θ) = p_i(x̂_i)` where `x̂_i = argmax_v (gumbel-noised logits)`;
> at **`T = 0`**, `x̂_i = argmax_v p_i` and therefore **`c_i(θ) = max_v p_θ(v|s_n,i)`**;
> **commit set** `C_n = top-k_n of {c_i}_{i∈E_n}`, `k_n = ⌊|M|/N⌋ (+1 for the first remainder steps)`.

At the evaluation decoders (`gen_length = 64, steps = 64`, and `steps = 32`) `k_n = 1` and `2`
respectively. `remasking = "random"` replaces `x0_p` by `U(0,1)` and is treated separately in §7.4.

Two facts about `c_i` that the design turns on:

- **`c_i` is the max, not the answer probability.** `c_i = max_v p_i ≥ p_i(a)` for any token `a`.
- **`c_i` is computed from the un-noised softmax even at `T > 0`.** So the *score* is deterministic
  in θ at every temperature; only the *identity* `x̂_i` is randomized. This matters in §7.4.

---

## 2. THE CRUX: can confidence be lowered without lowering `p(answer)`?

### 2.1 Pointwise: no, and by identity [P]

Let `A ⊂ V` be the fact surface set at position `i` (for a multi-token answer, the distinctive
content tokens; `build_fact_set` constructs it). Write `q_i(θ) = p_θ(A|s,i)` and
`q̄_i(θ) = max_{v∈A} p_θ(v|s,i)`.

> **Proposition 2.1 (inseparability on the disclosure set) [P].**
> At `T = 0`, `c_i(θ) = max_v p_θ(v|s,i) ≥ q̄_i(θ)`, with equality **iff** `argmax_v p_i ∈ A`.
> By Lemma 4, position `i` emits a fact token at step `n` only if `argmax_v p_i ∈ A` and `i ∈ C_n`.
> Define the **live set** `D = {(s,i) : argmax_v p_θ(v|s,i) ∈ A}`. Then on `D`,
>
> `c_i(θ) ≡ q̄_i(θ)`
>
> as functions of θ. Hence `∂c_i/∂θ = ∂q̄_i/∂θ` everywhere on `D`, and **no objective can change one
> without an identical change to the other.** Off `D`, `c_i` and `q̄_i` are separable — but off `D`
> the position cannot disclose at `T = 0`, so the separation is available exactly where it is
> useless. ∎

**Corollary 2.2 [P].** Any objective that reduces `c_i` on the live set reduces `q̄_i` by the same
amount, i.e. **it is ordinary suppression, re-labelled.** An "absolute confidence penalty" at fact
positions is therefore not a new mechanism; it is `max`-pooled mass suppression, and it inherits
every §8.4 / Theorem 2.1 consequence of mass suppression. It is also *worse* than the exclusion
target as suppression, because `max_v` is a non-smooth pooling of a single coordinate and gives no
control over where the mass goes.

**This is the answer to the brief's item 1 and it is negative for the mechanism as stated.**

### 2.2 Relatively: yes, exactly, and the gradient support is the proof [P]

Commitment depends on `c_{i*}` only through its **rank** among `{c_j}_{j∈E_n}`.

> **Proposition 2.3 (relative separability) [P].** Let `i* ∈ I_f` be a fact-proposing position and
> `S ⊆ E_n \ I_f`. Define the **rank margin** against the `m`-th highest member of `S`:
> `g_{i*,m}(θ) = log c_{i*}(θ) − log c_{(m)}(θ)`, `c_{(m)} = m`-th largest of `{c_j}_{j∈S}`.
> `i*` is excluded from `C_n` for at least `m/k_n` steps once `g_{i*,m} < 0`.
> Now `c_j` for `j ∈ S` is a functional of the output head at position `j`, and `q̄_{i*}` of the head
> at position `i*`. These are **distinct coordinates of the same forward pass**. Therefore an
> objective term whose logit-space gradient is supported on `{(s,j) : j ∈ S}` has
> `∂L/∂z_θ(·|s,i*) = 0` identically, and changes `q̄_{i*}` only through the **shared-parameter
> (NTK) path**, not through the objective. ∎

So the separable mechanism is: **raise the structural positions' confidence until they outrank the
fact positions, and never differentiate the fact positions' head at all.** The brief's phrasing
("relative to its neighbours") is exactly this; the absolute reading is Prop. 2.1's casualty.

**The residual leakage, named and bounded.** Prop 2.3 gives zero *direct* gradient at `i*`, not zero
change: after one step at rate η,

> **(2.1)** `Δ log q̄_{i*} ≈ −η Σ_{j∈S} K_θ((s,i*,a), (s,j,v̂_j)) · ∂L/∂c_j`

with `K` the empirical NTK across **positions of the same state**. This is small only if
cross-position kernel similarity within a state is small, which is **not** obvious in a
bidirectional transformer and is unmeasured. `Δ log q̄` is therefore both the separating prediction
(§8) and its own test; I do not claim it is zero.

### 2.3 Feasibility is a hard inequality, and it is cheap to check [P]

The ordering is achievable only if `m` structural positions can clear the fact positions by a
margin:

> **(2.2)** need `c_{(m)} ≥ e^{μ} · max_{i∈I_f} c_i`, with `c ≤ 1`.
> ⇒ **necessary condition** `max_{i∈I_f} c_i ≤ e^{−μ}`.
> At a strongly memorized fact `c_{i*} ≈ 0.9`, even `μ = 0` requires `m` structural positions with
> `c_j > 0.9`; `μ = 0.1` requires `c_j > 0.995`, which is essentially unattainable.

So the achievable margin is bounded by the *headroom above the fact position's own confidence*. Two
consequences:

1. **`μ` must be tiny** (a rank flip, not a comfortable margin) — `μ ≈ 10⁻²` in log space, i.e. a
   1 % relative confidence gap. The design is a knife-edge by construction, which is the same
   knife-edge Corollary 2.2 of the formalization describes for the mass condition, re-appearing here
   as a rank condition. Declare it; do not pretend to a margin we cannot buy.
2. **The feasibility check is free** and must precede training: measure `c_{i*}` and the empirical
   distribution of `{c_j}` on non-fact positions at θ₀ (job H10-M1, §9).

---

## 3. What replaces Lemma 5

Lemma 5 (suppression lowers confidence, therefore delays commitment) was refuted by the critic in
the multi-surface regime — suppressing one surface of the answer moves mass to another surface of
the *same* answer, so `max_v p` need not fall. I do not use it. H10 needs only:

> **Lemma H10.A (order controllability) [P].** `c_i(θ)` is a continuous function of θ, and the
> commit order at `s_n` is the ranking of `{c_i(θ)}_{i∈E_n}`. Hence (i) θ *controls* the commit
> order, and (ii) because the ranking is piecewise constant in θ, any trainable objective for it must
> be a **margin surrogate** on the gaps whose sublevel set implies the ordering. ∎
>
> No directional claim is made about what suppression does to the order. In particular **H10 does not
> assume that lowering `q̄` delays commitment**, and it must not be defended with that argument.

A second thing I must *not* inherit: the formalization's §4(f) displacement argument also rested on
Lemma 5's mechanism. The *hazard* of re-entailment does not need Lemma 5 — it needs only that a
later state carries more model-written context — so §4(f) survives as the risk named in §0 and §8.

---

## 4. The objective

### 4.1 Role partition of the canvas

At an on-policy state `s` with masked set `M(s)`:

> **(4.1)** `I_f(s) = { i ∈ E_n : argmax_v p_θ(v|s,i) ∈ A }`  — positions *currently proposing* a
> fact token. This is the set Lemma 4 says can disclose; it is the correct set, and it is
> **θ-dependent** (declared in §5).
> `I_s(s) = E_n \ I_f(s)`;  `S_m(s) = ` the `m` members of `I_s(s)` of highest current `c_j`
> (stop-gradient selection).

`m` is the **delay budget** in commits. `A` comes from the declared `𝔉` via `build_fact_set`, so it
is θ-independent; only the *positions* proposing members of `A` move with θ.

### 4.2 The ordering term, one-sided by construction

> **(4.2)** `L_ord(θ; s) = (1/|I_f|) Σ_{i∈I_f} (1/m) Σ_{j∈S_m} [ μ + log sg(c_i) − log c_j(θ) ]₊`

The `sg(·)` on the fact side is **not** cosmetic and **not** optional: without it, descent would
reduce the hinge by lowering `c_i`, which Prop. 2.1 says is identical to suppression, and the
mechanism would silently collapse into the thing it claims to replace. The stop-gradient is what
makes the gradient support match Prop. 2.3.

Its logit-space gradient at a structural position `j`:

> **(4.3)** `∂L_ord/∂z_θ(·|s,j) = −(w_j / m) · ( e_{v̂_j} − p_θ(·|s,j) )`,
> `w_j = (1/|I_f|)·|{ i ∈ I_f : μ + log c_i − log c_j > 0 }|`

— i.e. **a hinge-gated cross-entropy toward position `j`'s own argmax**, weighted by how many fact
positions it still fails to outrank. Recognise it plainly: this is **confidence sharpening /
entropy minimisation** at selected positions, an established object, with a commit-margin gate.
Saying so is required by the ground rules; the novel part is the gate and the position set, not the
sharpening.

### 4.3 Anchor the sharpening to the reference, not to ourselves

Sharpening toward *our own* argmax is self-training and amplifies our own errors; it is also the
classic route to a canned, item-invariant template (degeneracy (a)). Since the C1 composite already
computes a forward through the retain-trained reference `θ_r`, replace `v̂_j` by the reference's
argmax at zero extra cost:

> **(4.4)** `v̂^r_j = argmax_v p_{θ_r}(v|s,j)`  — **θ-independent**.

Then H10 collapses into a modification of C1's own target, which is the cleanest statement of it:

> **(4.5)** `r^{H10}(v|s,j) ∝ p_{θ_r}(v|s,j) · exp( β_j · 1{v = v̂^r_j} )`,  `β_j = λ_ord · w_j`,
> for `j ∈ S_m(s)`; and `r^{H10}(·|s,i) = p_{θ_r}(·|s,i)` unchanged for every other position,
> **fact positions included.**
>
> **(4.6)** `L = Σ_{i∈M(s)} CE( r^{H10}(·|s,i), p_θ(·|s,i) ) + (retain term)`

So H10 is exactly: **boost the reference target's own mode at the `m` highest-confidence structural
positions, by an amount gated on whether they still fail to outrank the fact positions.** The fact
positions receive C1's target untouched. Three properties follow immediately:

- the gradient at fact positions is C1's and only C1's (Prop. 2.3 holds);
- the fixed point at a gated structural position is a **tilted** reference distribution, not a
  one-hot, so the sharpening is bounded by `λ_ord` and does not run away;
- the term switches itself off when the ordering is achieved (`w_j = 0`), so it cannot over-sharpen
  past the rank flip — the same hinge discipline as A6.

### 4.4 Where the term is evaluated

On-policy decode states from the author's sampler with first-fact-commit truncation
(`generate_traced(..., record=True, fact_mask=A, stop_when_all_leaked=True)`), re-collected every
`P` steps, because commit order is a property of the trajectory and cannot be read off a
teacher-forced corruption. `L_ord` is meaningless at a teacher-forced state at `t ~ U(0,1)`: the
visible tokens there are ground truth, not the model's own commits, so the confidence ranking is not
the one the sampler will face. This is a **hard** requirement for H10, unlike for C1.

---

## 5. The mathematical object, classified honestly

**Within a step:** ordinary descent on a well-defined scalar loss. Equation (4.6) is a soft
cross-entropy to a fully specified target; there is no stop-gradient inside the differentiated path,
and `λ_ord` multiplies a term whose gradient is (4.3).

**Across steps: not descent on any fixed function.** Three θ-dependencies in the target, each
declared with its cost:

| θ-dependency | why | cost |
|---|---|---|
| `I_f(s)` = positions whose argmax lies in `A` | this is the set Lemma 4 says can disclose; a θ-independent proxy would target the wrong positions | the index set moves during training; the objective is a sequence of surrogates `L_t`, and no convergence claim is available |
| `S_m(s)`, the top-`m` structural positions by current confidence | we need the positions that can actually overtake | non-differentiable top-`m` selection; stop-gradient |
| `w_j`, the hinge gate, through `sg(c_i)` and `c_j` | one-sidedness (§4.2) | the gate is piecewise constant; the loss is continuous but its *weights* jump |
| the state pool | commit order is a trajectory property | `P`-step staleness, declared and logged (§8.5) |

So the honest label is: **a scalar loss at each step, over a θ-dependent, stop-gradient-selected
index set and weighting — a sequence of surrogates, not a fixed objective.** It is the same class as
MDU/EMDU (frozen anchor) plus a gate, and it is *milder* than the cone projection of
`OBJECTIVE_DESIGN.md` §B.5, which is not a gradient map at all. I make no claim of convergence and
no claim that `L_ord`'s value at the end is meaningful.

---

## 6. The composite with C1, and the ablation that makes H10 attributable

**C1 (base, prior art — arXiv:2506.06278).** `L_C1 = Σ_{i∈M(s)} CE( p_{θ_r}(·|s,i), p_θ(·|s,i) )`
on forget states, plus a retain term, `θ_r` = a reference trained on retain only. It cannot carry a
claim and is not asked to.

**Composite.** Exactly (4.6): C1's target with the mode boost at `S_m`. **One knob: `λ_ord`.**
Everything else — states, reference, retain data, optimizer, steps, LoRA config, selection rule —
identical across arms.

### 6.1 The ablation ladder, and the control a reviewer will actually demand

| arm | `λ_ord` | structural positions | purpose |
|---|---|---|---|
| **A. C1** | 0 | C1 target | base; isolates everything |
| **B. C1 + uniform structural upweight (mass-matched)** | 0 | C1 target, positions in `I_s` upweighted by a **constant** chosen so the total extra gradient mass equals arm D's, measured | **the essential control**: removes the *commit-margin criterion* while holding the *position reweighting* fixed |
| **C. H10 alone** | λ | mode-boosted | the risky arm: predicted possibly **worse** than θ₀ on disclosure (§8.1) |
| **D. C1 + H10** | λ | mode-boosted | the candidate |

**Arm B is non-negotiable.** Without it, any benefit of D over A is attributable to "upweight the
structural positions", which PRIOR_ART says is *not* novel ("select which positions to forget is NOT
novel"; five published AR selectors). H10's claim is specifically that the *commit-margin gate* —
derived from the sampler's own top-`k` rule — beats an equal-mass uniform reweighting. D > B is the
claim; D > A is not.

**One distinguishing operation.** The operation is the gate `w_j` and its index set `S_m` vs `I_f`.
Nothing else is added. In particular I am **dropping the twin projection** of `OBJECTIVE_DESIGN.md`
§B.5/§10.2 from this composite, and the reason is a data change, not a change of mind: TOFU800 is
author-disjoint with 400 distinct answers per side, so **the shared-answer retain twin set is
empty** and the projection's constraint set does not exist. Carrying it forward would be a mechanism
with nothing to constrain. (If a shared-answer slice is wanted back, it has to be built; on free-form
27-word answers it cannot occur by construction.)

### 6.2 A confound specific to this composite, which must be controlled

If the reference's ignorant answer is **shorter** than the memorized one, most of the canvas is
`eos`, and `eos` positions carry very high confidence (the default decoder does **not** set
`confidence_eos_eot_inf`, so they compete normally). A short target therefore produces the ordering
effect *for free*, with `λ_ord = 0`. Control: **report and match the target-answer token length
across arms**, and report arm A's own fact-commit-step distribution. Without this, D-vs-A is
confounded by target length and the ablation is uninterpretable.

---

## 7. Scope restrictions that follow from the mechanism, not from taste

### 7.1 `T = 0`, `lc` remasking: full mechanism
Both channels operate: the **rank channel** (the fact position is excluded from `C_n`) and the
**state channel** (when it does commit, the canvas around it is model-written).

### 7.2 `T > 0`, `lc`: state channel only — and a sharp prediction
At `T > 0`, `x̂_i` is a Gumbel draw while `c_i` is still computed from the un-noised softmax (§1).
So the ranking still responds to θ, but emission at `i*` requires `x̂_{i*} ∈ A`, which happens with
probability `q_{i*}` at whatever state `i*` commits. Since H10 leaves `q` intact by construction, the
**only** route to lower disclosure at `T > 0` is the state channel.

> **Prediction 7.A (quantitative, and it bounds the claim).** H10's disclosure reduction at `T = 1`
> should be a **strict fraction** of its reduction at `T = 0`, and that fraction *is* the
> state-channel share. A suppression-family method shows no such drop (it lowers `q`, which is what
> `T > 0` reads). If H10's `T = 1` reduction is ≈ its `T = 0` reduction, then it is not working by
> ordering and something else is happening.

### 7.3 `k_n ≥ 2`: leverage falls as `1/k`
Delaying past `m` structural positions buys `m/k_n` steps. At the `steps = 32, gen_length = 64`
decoder (`k = 2`) the same `m` buys half the delay. Report per config.

### 7.4 `remasking = "random"`: the effect must be **identically zero** through the rank channel [P]
Under `rnd`, `x0_p = U(0,1)`, θ-independent, so the commit order is θ-independent and the rank
channel vanishes exactly. Only the state channel can survive.

> **Prediction 7.B (a free, decisive placebo).** H10's rank-channel benefit **must vanish under
> `remasking="random"`**. No training is needed to test it — just re-decode the trained arm with
> `rnd`. If D still beats A by the same margin under `rnd`, **H10's stated mechanism is not what is
> producing the effect**, and the result, whatever it is, is not an ordering result. I know of no
> other mechanism in this literature with a built-in placebo this clean, and it should be run on the
> first trained checkpoint.

**Consequently the claim must be indexed `Π = {T = 0, lc, declared k}`** (R1), with the `rnd` and
`T = 1` cells reported as the mechanism's own scope boundary rather than hidden.

---

## 8. Separating predictions, sharpened

The brief's prediction — disclosure falls while `p(answer)` at the teacher-forced state is unchanged
— is right in form but must be stated as an **arm contrast**, not against θ₀, because C1 *itself*
lowers `q` (its target is `p_{θ_r}`, which has little mass on the answer). The attributable version:

Define `q*(s) = log p_θ(A | s, i*)` at two frozen diagnostic states: the **step-1 state**
`s⁰ = (c, all-mask)` and a teacher-forced state `s^tf` at `t = 0.5` on the memorized answer. Let `D`
be generated disclosure and `τ_c` the fact positions' mean commit step.

> **Prediction 8.A (the separator).** Arm **D vs arm B**, matched budget, matched target length:
> 1. `D(D) < D(B)` — disclosure strictly lower;
> 2. `|q*(D) − q*(B)| ≤ 0.05` nats at both `s⁰` and `s^tf` — the answer marginal is
>    **statistically unchanged**;
> 3. `τ_c(D) > τ_c(B)` — fact positions commit strictly later;
> 4. the gap in (1) vanishes under `rnd` remasking (Prediction 7.B) and shrinks at `T = 1`
>    (Prediction 7.A).

**Why no suppression-family method can produce 1 + 2 together, with the number.** For MDU, EMDU,
NPO, UNDIAL or the exclusion target, the mechanism *is* `∇_z = p_θ − r` with `r(A) ≈ 0`, so `q*` must
fall; and for the rank to flip at all, `q*` must fall **below the runner-up**:

> **(8.1)** `Δq* ≤ log( p₀(runner-up) / p₀(A) )`.
> At a strongly memorized fact (`p₀(A) ≈ 0.9`, runner-up `≈ 0.03`) that is
> **`Δq* ≤ −3.4 nats`** on every item whose disclosure it flips.

So the prediction is a **two-orders-of-magnitude** separation on a directly measurable scalar:
suppression families `≤ −3.4` nats, H10 `|Δ| ≤ 0.05` nats, on the *same* flipped items. That is the
sharpest separator in either design document, and it is cheap (one forward per probe at two frozen
states).

**What makes it fail**, in the four interpretable outcomes:

| `ΔD` | `|Δq*|` | reading |
|---|---|---|
| < 0 | small | **H10 as advertised.** Order changed, marginal intact. |
| < 0 | large | H10 is suppressing through the shared-parameter path (2.1) — the gate leaked. Not distinguishable from C1; claim withdrawn. |
| ≈ 0 | small | the ordering changed and disclosure did not: **re-entailment** (§8.1). H10 dead, premise refuted *after* training — which is why H10-M0 must run first. |
| > 0 | small | ordering actively harmful. Publishable negative. |

---

## 8.1 Failure conditions, each with an observable

| # | condition | why | observable |
|---|---|---|---|
| **H1** | **Re-entailment.** A later commit conditions on model-written text that *pins* the answer (§4(f)), so delay raises the hazard. | bidirectional denoiser; the frame the model writes around a memorized fact is the frame that entails it | **H10-M0 oracle** (pre-training, 7 GPU-min); then per-probe disclosure vs `τ_c` |
| **H2** | **Feasibility (2.2) fails.** No `m` structural positions can clear `max_{i∈I_f} c_i`. | `c ≤ 1` and memorized facts sit near 0.9 | **H10-M1**: `c_{i*}` and `n_above = |{j ∉ I_f : c_j > c_{i*}}|` at step 1 |
| **H3** | **The fact set is too large a fraction of the canvas.** On 27-word answers `|I_f|` may be a third of the span; the margin must hold for the **weakest** fact position, so cost grows with `|I_f|`. | §10 | `|I_f|/|E_1|` per probe, and `min_{i∈I_f} n_above` |
| **H4** | **Gate leakage.** Prop. 2.3 gives zero direct gradient, not zero change (2.1). | cross-position NTK within a state | `Δq*` — this is Prediction 8.A(2) doing double duty |
| **H5** | **Entropy-minimisation pathology.** Sharpening structural positions produces a canned, item-invariant template. | (4.3) is entropy minimisation | `Î_item` on the forget panel against the unknown-subject panel; structural-position entropy; repetition with no length floor |
| **H6** | **Commitment diffuse.** No concentrated commit step to move, and `I_f` unstable across seeds/configs. | — | **H10-M2**; see §9. **Stated in advance: if diffuse, H10 dies.** |
| **H7** | **Already late.** If the fact positions already commit in the last decile there is no room. Two-sided kill: H10 needs the fact *early* and the distribution *concentrated*. | — | H10-M2's `rank_1(i*)` histogram |
| **H8** | **Config transfer.** The order is config-specific (§7.3); training on one `k` may not transfer. | `k_n` scales the delay | held-out config panel reported as a **max**, with a `k`-extrapolation cell |
| **H9** | **Target-length confound.** A short reference answer produces the ordering for free. | §6.2 | matched target token length across arms; arm A's own `τ_c` |
| **H10f** | **Placebo fails.** The benefit survives under `rnd`. | §7.4 | Prediction 7.B — free |
| **H11** | **Staleness.** The pool's commit order is `P` steps stale and the trained order drifts off it. | order is a trajectory property | per-step disclosure sawtooth between refreshes; `P` sweep |

---

## 9. Prerequisite measurements, with the exact instrumentation

All of these are requests. **Ordering: H10-M0 → H10-M1 → H10-M2.** Together under 20 GPU-minutes,
and any one of them can kill H10 before an arm is trained.

### H10-M0 — the forced-order oracle (≈7 GPU-min). **Run first.**
Add an evaluation-only branch to `generate_traced`: after `confidence` is computed and before
`topk`, set `confidence[i] = −inf` for every `i` with `x0[i] ∈ A`, **unless** no other masked
position remains eligible. That defers every fact-proposing position maximally — the exact ordering a
perfect H10 would learn, with weights untouched. §3 permits this as an evaluation reference and
forbids it as the method; assert that in code.
Report: disclosure and the four-way partition (DISCLOSE / DEGENERATE / REPLACE / IGNORANT) under
{natural, forced-last} × {decoder A, decoder B}, 100 forget probes.
**Kill rule (pre-registered): if forced-last disclosure is within the Wilson interval of natural
disclosure, H10 is dead.** No trained approximation can beat its own oracle.

### H10-M1 — feasibility and the confidence landscape (≈3 GPU-min)
One forward at the step-1 state `s⁰ = (c, all-mask)` per forget probe (400 probes, batched).
Report, per probe: `c_i` for all `i ∈ E_1`; `I_f(s⁰)`; `max_{i∈I_f} c_i`; `n_above`; `|I_f|/|E_1|`;
and `rank_1(i*) = |{j : c_j > c_{i*}}|`. Aggregate: the joint distribution of
`(max_{i∈I_f} c_i, n_above)`, which is condition (2.2) plotted directly, and the largest feasible
`m` at `μ = 0`.
**Kill rule: if the median `n_above` is below the delay budget needed for any effect (say `m ≥ 3`),
H2 has fired.**

### H10-M2 — the commitment distribution, which is what the brief's item 6 asks for (≈6 GPU-min)
Instrument the author's sampler via the existing `on_step` callback (`generate_traced:263`), which
already receives `(n, x, logits, candidate_mask, transfer_index, x0)` **before** the commit. Per
step and per eligible position record `x0[i]`, `x0_p[i]`, `transfer_index[i]`, and `1{x0[i] ∈ A}`.
Derive two distinct quantities the brief conflates:

- `τ_argmax(i)` = first step at which `x0[i] ∈ A` — when the answer token **becomes argmax**;
- `τ_commit(i)` = first step at which such a position is in `transfer_index` — when it **is kept**
  (this is the existing `first_commit`).

Report over 400 probes × 2 configs × 3 seeds: histograms of `τ_commit/N` and `τ_argmax/N`; the
**gap** `τ_commit − τ_argmax` (a position that is argmax-A at step 1 but committed at step 20 is
*already* being delayed by the confidence ranking, and that is free evidence about the headroom);
modal-bin mass; entropy in bits against `log₂ N`; and the per-probe count `|{i : x0[i] ∈ A}` over
steps.
**Kill rule, stated in advance as asked: if `H(τ_commit)` exceeds ~0.8·log₂N (diffuse) *or* the
median `τ_commit/N` is already above 0.9 (already late), H10 dies.** Unlike the hazard formulation
in `OBJECTIVE_DESIGN.md` §C, H10 cannot survive diffuseness: its entire content is moving a step
index, so a flat distribution leaves nothing to move.

### Downstream, only if the above pass
- **H10-M3**: `λ_ord` and `m` sweep on DEV, 2 points each; the feasibility map from M1 sets the
  range.
- **H10-M4**: the `rnd` placebo (Prediction 7.B) on the first trained checkpoint — re-decode only,
  ≈2 GPU-min, and it can invalidate the mechanism story at near-zero cost.
- **H10-M5**: `Δq*` at the two frozen diagnostic states for arms A/B/D — one forward per probe per
  arm; this is Prediction 8.A(2) and H4 simultaneously.

---

## 10. The multi-token regime: why it enables H10 and why it weakens it

Both directions follow from the same fact and must be stated together.

**It enables the mechanism.** With a single-word answer in a mostly-`eos` canvas there is no
structural span to raise: `I_s` is `eos` padding, and Prop. 2.3's separable channel has almost
nothing in it. With 27-word free-form answers there are ~30–40 non-fact positions per canvas, which
is the resource (4.2) spends. **H10 is only well-posed in the multi-token regime**, and the move to
TOFU800 is a precondition for it, not a detail.

**It weakens the mechanism, by the same arithmetic.** Three specific costs:

1. **`|I_f|` is large.** For a 27-word answer, `build_fact_set` leaves roughly 10–15 distinctive
   content tokens after the stoplist/prompt/special filters. The margin must hold for **every** fact
   position — the weakest link governs — so the requirement is
   `min_{i∈I_f} n_above(i) ≥ m`, not `n_above ≥ m` for one position. The leverage is
   `|I_s|/|I_f| ≈ 2–3` rather than `≈ 40`.
2. **Partial disclosure counts.** By §5.2/R12, disclosure is identifiability, not completeness:
   emitting *some* distinctive tokens can already be full disclosure (`"Berkshire"` identifies
   `"Berkshire Hathaway"`). So delaying 12 of 15 fact positions may buy nothing. **H10's unit of
   success is `min` over fact positions, while the metric's unit of failure is `max`** — the two are
   opposed, and this is the single biggest structural weakness of H10 on TOFU800.
3. **Pieces are emitted out of order and non-adjacently** (§5.2's MDLM twist), so there is no single
   "answer position" and no contiguous span to reason about. (4.1) handles this correctly by
   defining `I_f` pointwise, but it means `|I_f|` changes every step and the gate is noisy.

**This is a deflation and I am stating it before the measurement**: on free-form 27-word answers I
expect H10's effect to be substantially smaller than it would be on a short-answer benchmark, for
reason 2 above, and the quantity that decides it is `min_{i∈I_f} n_above` from H10-M1. If a
short-answer regime is available (RWKU-RQ answers are 1–3 tokens), **H10's mechanism is strictly
better posed there**, and that is worth knowing before committing TOFU800 budget to it.

---

## 11. Novelty position, per the Part 3 rubric

- **Closest prior art on the selection side:** MDU's own token-role diagnostic (in-context /
  structural / stored-knowledge, −20.6 % KL on stored-knowledge tokens) — *published, measured, and
  explicitly unused in its objective*; and the five AR position selectors (WGA/TNPO `p_θ^α`, SU's
  assistant-score difference, SPAN's EK-FAC influence, FUnDIAL's entity indicator, CUSE's
  counterfactual log-prob gap). **Every one of them weights the target/fact tokens.** H10 weights
  the **complement**, by a criterion read off the sampler's own top-`k` rule.
- **Closest prior art on the trajectory side:** CUSE and RULE (on-policy, autoregressive);
  TimeROME-DLM (trajectory-aware *localisation*, not a training objective); and the
  cited-but-unlocated Zhang et al. 2026, which must be located before any novelty statement.
- **Honest standing: category (e) at the outset.** The route to (d) is the A/B/C/D factorial of §6.1
  with ≥3 seeds *including the ablation cells* (the gap EMDU left), plus the `rnd` placebo as the
  negative control that the interaction is specific. **D > B is the claim**; D > A is not.
- One more conservative reading a reviewer will press and which I accept as possible: (4.5) is
  "C1 with a tilted target at selected positions", and tilting a distillation target is familiar.
  The defence is the *gate*, and if arm B matches arm D the defence fails and the correct verdict is
  category (c).

---

## 12. What I could not make work

1. **Absolute confidence control is impossible on the disclosure set** (Prop. 2.1): `c_i ≡ q̄_i`
   there, by identity. Any "confidence penalty" at fact positions is suppression under another name,
   and a worse-behaved one.
2. **No margin can be bought at a strongly memorized fact** (2.2): `c ≤ 1` and `c_{i*} ≈ 0.9` cap
   `μ` at ~10⁻², so H10 is a knife-edge rank flip, not a robust ordering. I could not find a
   formulation that escapes the ceiling, because the ceiling is the simplex's.
3. **H10 cannot work through the rank channel under `rnd` remasking** (§7.4) — exactly zero, proved.
   It therefore cannot be claimed for `Π` broader than `{T = 0, lc}` plus whatever the state channel
   contributes, and I could not construct an ordering mechanism that survives θ-independent
   confidences, because there is nothing for θ to act on.
4. **On TOFU800, H10's success unit (`min` over fact positions) is opposed to the metric's failure
   unit (`max`)** (§10.2). I could not remove this; it is a property of identifiability-based
   disclosure scoring in a multi-token regime, and it is the reason I expect the effect to be small
   here even if the premise holds.
5. **The twin projection from `OBJECTIVE_DESIGN.md` §B.5 is withdrawn from this composite** — not
   because it was wrong, but because TOFU800 is author-disjoint with 400 distinct answers per side,
   so its constraint set is empty. Any shared-answer preservation claim now requires new data.
