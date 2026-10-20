# OBJECTIVE AND ARCHITECTURE DESIGN

**Owner:** objective and architecture design lead
**Date:** 2026-10-20
**Status:** Tier 2 (active development). One primary candidate, one declared fallback. Nothing
here is a result. Every GPU job named in §8 is a request, not a record — **I have run no GPU
jobs.**

Inputs honoured: `FORMALIZATION.md` (R1–R17, Lemmas 1–5, §7.2 A1–A7, §8), 
`FINDING_answer_channel.md` **including the θ₀ control added after my brief**, `PRIOR_ART.md`
(Parts 3, 5, 8), `HYPOTHESIS_REGISTER.md` (H1 refuted; H2, H3 untested), `EVAL_PROTOCOL.md`
(gates C1–C6, DEV/HELD-OUT seal, two-coordinate reporting), and the empirical fact that our
strongest control reaches 0.031 disclosure / 1.000 retain / 0 degenerate **by replacement**.

Status labels: **[P]** proved here, **[A]** assumption, **[H]** hypothesis needing measurement,
**[M]** measured (cited with caveats).

> **AMENDMENT 1 (§10) supersedes parts of §0.2, §B.5 and §E.3.** It re-states the primary claim
> against the IGNORANT rate, corrects that rate by a measurement I ran on existing data, refutes one
> of my own predictions, and relaxes the Part B constraint to a declared damage budget. Read §10
> alongside the body.

---

## 0. Summary, and the three things that must be read before the equations

### 0.1 The candidate, in one box

**IHP — Ignorance target, Hazard index set, Projected update.**

Per step, with a state pool re-collected on-policy every `P` steps:

```
(1)  index set      I_t = { (s_n, i) : s_n ∈ pool, n ≤ τ(s), i ∈ Ĉ_n(θ_t) }      [stop-grad top-k]
(2)  target         log r̂(·|s,i) = log p_θ0( σ_S̃(·) | σ_S̃(s), i ),   S̃ ~ ν_S     [1 stranger/state/step]
(3)  forget term    L_ign(θ) = (1/|I_t|) Σ_{(s,i)∈I_t} CE( r̂(·|s,i) , p_θ(·|s,i) )
(4)  constraints    c_rank(s,i) = [ γ + max_{v∈F} z_θ(v) − max_{v∉F} z_θ(v) ]₊          (T = 0)
                    c_mass(s,i) = [ log p_θ(F|s,i) − log max(ε, r̂(F|s,i)) ]₊          (T > 0)
(5)  primal grad    g_t = ∇_θ [ L_ign + λ_r · mean c_rank + λ_m · mean c_mass ]
(6)  retain grads   v_j = ∇_θ L_ret^(j),  j = 1..J  (shared-answer twins; independent retain)
(7)  PROJECTION     u_t = argmin_u ½‖u − g_t‖²   s.t.  ⟨v_j, u⟩ ≥ 0  ∀j
(8)  update         θ_{t+1} = AdamW_step(θ_t ; grad := u_t + Σ_j β_j v_j)
(9)  dual ascent    λ ← [ λ + η_λ ( mean c − slack ) ]₊
```

**What kind of mathematical object this is, stated exactly.** It is **not** descent on any fixed
scalar loss, and I will not call it that. It is a **primal–dual projected-gradient method on a
constrained problem whose index set is itself a function of θ through a non-differentiable
top-`k`**. Three separate reasons:

- (1) makes the objective a *sequence* of surrogates `L_t(θ)`, not a fixed `L(θ)` — the committed
  set is selected by a stop-gradient top-`k` on the student's own confidences, so the function
  being minimised changes as the student changes (online hard-example mining, in that family).
- (7) makes the update a **Euclidean projection onto a polyhedral cone**. A cone projection is not
  a gradient map: there is in general no function whose gradient is `u_t`. This is the GEM/A-GEM
  construction (Lopez-Paz & Ranzato 2017); I name it as established, not invented.
- (9) is dual ascent, so the effective weights move. Fixed Lagrange multipliers cannot satisfy
  constraints (R: §7.3), and 2 extra scalars is the cheapest honest fix.

Only step (3), at a frozen `I_t`, is ordinary loss descent. Everything else is a declared custom
update, and every claim about convergence in this document is a claim about the *measured*
trajectory, not about a minimiser.

### 0.2 The three things that reframe the problem

**(i) Disclosure has no headroom. The headline metric is already at the ignorance floor.** Our
strongest control sits at **0.031** disclosure; the never-memorized control's held-out disclosure
is **0.023–0.046** [M]. Def. 1.2 subtracts the type-prior baseline precisely because a model that
never learned the fact still names the right city sometimes. **0.031 is inside the ignorance
band.** Therefore: any further reduction in disclosure is *over*-suppression, and every remaining
axis of improvement is in the *composition* (R17) — well-formedness, ignorance-matching,
shared-answer retain, paraphrase generality, relearning. A method that reports "lower disclosure"
on this benchmark is reporting over-suppression, not forgetting. This conclusion is forced and it
is the single most important design constraint in this document.

**(ii) The mask-rate marginal is already correct; it is the conditional content that is wrong.**
H2 asserts that "uniform `t` spends most of the training budget on states the decoder never
visits". The first half of that is **false** and the second half is **true**, and the distinction
decides what part C can offer. Under the frozen decoder (`gen_length = 32, steps = 32,
block = 32`), `num_transfer_tokens = 1` per step, so the visited states have mask rates exactly
`t_n = (32 − n)/32`, a deterministic ladder whose marginal **is** uniform on the grid. So
`t ~ U(0,1)` already matches the visited *rate* distribution. What it does not match is the
*conditional law of the visible content at that rate*: teacher-forced states show ground-truth
answer tokens (including `F` tokens — the σ_F = 1 partition of §5.6, where the correct behaviour
is the *opposite*), whereas on-policy states show only the model's own commits. **The
mask-rate-selection axis, which PRIOR_ART calls the least crowded, is therefore the weakest of the
two levers, not the strongest.** The lever with content is the *position* set and the *state*
conditional. §3 derives both from a union bound rather than from the mask rate.

**(iii) The answer-token coupling, re-derived, is probably not caused by the target — and my
candidate therefore does NOT predict what the coordinator's screen predicts.** This is §2.3 and
§2.4. It is the most consequential thing in this document and I state the conclusion up front: the
coordinator's prediction ("the ignorance closure is indexed by the entity, so same-answer coupling
should stay near 0.388") is, on my derivation, **predicted to fail**, because the error vector's
leading term is `Wᵀp_θ ≈ w_{v*}` — the model's *own current prediction*, not the target — and no
choice of target removes it. I give the algebra, the quantitative prediction (δ-level cosine
≈ 0.80 under the ignorance closure vs ≈ 0.97 under exclusion, not 0.388 vs 0.92), and a
**zero-training, ~12-GPU-minute measurement on four already-trained baseline adapters** that
decides between my account and the coordinator's before anything is built.

---

# PART A — THE TARGET

## A.1 The derivation, from (★), with the estimating closure

The ideal per-position target is `r*(v|s,i) = p(v|s,i ; D_{¬φ})`. The formalization's mixture
posit gives the deconvolution

> **(★)** `r*(v|s,i) = [ p_{θ0}(v|s,i) − w(s,i)·q_φ(v|s,i) ] / (1 − w(s,i))`

with `w` and `q_φ` the only two unknowns. I am instructed to close (★) by **estimating** rather
than **assuming**, i.e. A4 rather than A2+A3. I do so, and I note at once that this means
**`q_φ` is never constructed**. That is not an evasion; it is the content of the closure. Under A4
we estimate the *left* side directly and `q_φ` becomes a derived diagnostic (§A.5), because (★) has
two unknowns and one equation: `(w, q_φ)` is not identified, and any procedure that claims to build
`q_φ` has silently added an assumption. A2 is exactly such an assumption (`supp q_φ ⊆ F`), and §8.2
and §8.4 of the formalization are the price of it.

## A.2 The estimator, exactly

**Assets.** Memorized backbone = base LLaDA-8B-Instruct + a merged memorization LoRA
(`/data/bijaypandey/spex/adapters/memorized_big`, PEFT r=32, α=64, 224 modules, target regex
`.*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$`). `θ₀` is reached for free
through `model.disable_adapter()` on the *unlearning* LoRA stacked on top of the merged
memorization weights — exactly the pattern `arc_big.py:61-64` already uses. No second 8B copy.

**Stranger substitution, with the state-alignment problem actually solved.** §7.2 A4 flags that the
map between `s` and `s[S ← S̃]` "is a real implementation problem, not a detail". It is solved, and
the solution already exists on disk:

- Draw `S̃` from the length-matched stranger pool `/data/bijaypandey/llada_poc/data/name_pool.json`
  (**5655 names, bucketed by token length {3,4,5,6}**, built CPU-only by
  `llada_poc/scripts/make_name_pool.py` precisely because our fictional entities tokenize to 3–6
  pieces).
- Require `|Tok(S)| = |Tok(S̃)|` and that the pairwise map `Tok(S)_k ↔ Tok(S̃)_k` extends to an
  **involution** `σ_S̃ : V → V` (admissibility test `_swap_pairs` / `_involution` /
  `accept_name`, already implemented in `llada_poc/cand3/llada/tdssd.py:121-169`).
- Then `σ_S̃(s)` is obtained by applying `σ_S̃` elementwise to **every** token id of the state —
  prompt *and* canvas, so already-committed restatements of the name are covered — and the position
  alignment is the **identity map**. No length surgery, no re-tokenization, no interpolation.
- Because `σ_S̃` is an involution, the pull-back of the reference's prediction is a permutation of
  the vocabulary axis (`swap_perm`, `tdssd.py:179`), not a renormalization.

**The estimator.**

> `log r̂(v | s,i) = log p_{θ0}( σ_S̃(v) | σ_S̃(s), i ),   S̃ ~ ν_S`  — one sample
>
> `log r̂_m(v | s,i) = logsumexp_{k=1..m} [ log p_{θ0}( σ_k(v) | σ_k(s), i ) ] − log m`  — m samples

**Why the arithmetic mean and not the geometric mean.** `r*` is a posterior predictive:
`r* = E_{S̃}[ p₀(·|s[S←S̃]) ]` under the exchangeability assumption A4. That is an expectation of
distributions, so the Monte-Carlo estimator is the mixture (arithmetic mean), computed in log space
by `logsumexp`. A geometric mean (`softmax` of averaged log-probs, i.e. a product of experts) would
**sharpen** the target — and spread-matching is the entire purpose of this closure (§4(d),
ignorance matching, H3). Using a log-average here would reintroduce degeneracy (d) by arithmetic
error. Named, because it is the kind of mistake that is invisible once coded.

## A.3 Which divergence, and why linearity in the target is load-bearing

Apply the target with **soft cross-entropy**:

> `ℓ(s,i) = CE( r̂(·|s,i), p_θ(·|s,i) ) = − Σ_v r̂(v) log p_θ(v|s,i)`,
> `∇_{z_θ(·|s,i)} ℓ = p_θ(·|s,i) − r̂(·|s,i)`.

Three consequences, in increasing order of importance.

1. By EMDU's Proposition 1, `∇_z KL(p‖softmax(z)) = q − p`, so soft-CE against a frozen target is
   **exactly** anchor-first KL up to the additive constant `−H(r̂)` which has zero gradient. Our
   existing implementation already uses this (`soft_ce_kl`, `mdmu/fxmix.py:119`). It is EMDU's
   divergence, not MDU's. PRIOR_ART §8.3 item 4 asks us to stop mislabelling this; adopted.
2. Anchor-last KL (`KL(p_θ‖r̂)`) is excluded on the published analytic ground (gradient
   `q ⊙ (log(q/p) − KL·1)` vanishes at a memorized near-one-hot — exactly where we need it).
3. **The decisive reason, which is ours and not in either paper.** Soft-CE is **linear in the
   target**. Hence for a one-sample stranger draw,
   `E_{S̃}[ ∇_z ℓ ] = p_θ − E_{S̃}[r̂] = p_θ − r*`, i.e. **a single stranger per state per step gives
   an unbiased gradient of the full m→∞ expectation.** Anchor-last KL does not have this property
   (it is non-linear in the target through `log r̂`), so under anchor-last a one-sample stranger
   target is *biased* and the bias does not vanish with training. This fixes `m = 1` as the primary
   choice on a derivation rather than on cost grounds, and it is the single cleanest argument in
   this document for the divergence orientation.

## A.4 Variance of the estimator, exactly, and the trap in caching it

Write `p_k(v) = p_{θ0}(σ_k(v)|σ_k(s),i)`. Per coordinate,
`Var[r̂_m(v)] = Var_{S̃}[p_k(v)] / m`. The quantity that matters is the total logit-space gradient
noise at a state:

> `tr Cov[ ∇_z ℓ ] = (1/m) Σ_v Var_{S̃}[p_k(v)] = (1/m) ( E‖p_k‖²₂ − ‖r*‖²₂ )`.

Two closed forms worth carrying:

- **Bound.** `Σ_v Var ≤ E‖p_k‖²₂ ≤ E[max_v p_k] ≤ 1`, so `tr Cov ≤ 1/m`, independent of `|V|`.
- **Sharp form in the regime we are in.** If each stranger's prediction is near-one-hot on a
  candidate drawn from a mode distribution `π` over the type-compatible set `𝒪`, then
  `r* = π` and `Σ_v Var = 1 − ‖π‖²₂`. With `|𝒪| = 32` (our 32 distinct answers) and `π` roughly
  uniform, `tr Cov ≈ 0.97/m`. The gradient noise is therefore the **collision-complement of the
  ignorance distribution** — large when ignorance is genuinely spread, which is exactly when the
  target is most valuable. There is no regime in which this estimator is both informative and
  quiet.

**The caching trap, which is a real and non-obvious failure mode.** The obvious cost saving is to
draw `m` strangers once per fact and cache the target (this is the ARC construction,
`arc_big.py:182-243`, which is exact for a *deterministic* target). For a *sampled* target it is
not a cost saving, it is a change of fixed point: the student is then fit to the **empirical**
mixture `r̂_m` of `m` specific strangers, and that error is a **frozen bias**, not noise, so the
optimizer will fit it. For `π` uniform on `j` candidates, the empirical mode's probability exceeds
the true `1/j` by a factor `1 + O(√(j log j / m))`; keeping that within `(1+δ)` needs
`m ≳ j log j / δ²`, i.e. **m ≳ 450 at j = 32, δ = 0.5**. A cached pool of `m = 4` or `m = 16`
reinstates degeneracy (d) at a coarser grain — a memorized *small answer set* instead of a
memorized single answer, and `MAR` (mode agreement across seeds) will not be far from 1.

**Resolution, and it is forced:** resample the stranger **fresh every step**, so the expectation is
realized across optimization steps rather than within a step, and the effective `m` for a given
fact is the number of steps on which it appears (`steps × batch / 64` ≈ 150 × 8/64 ≈ 19 at our
scale, rising with `steps`). The gradient is unbiased at every step by §A.3(3), and Adam's
first-moment EMA does the averaging. Declared staleness: none on the target; the only stale object
is the state pool (§C.4).

## A.5 What `q_φ` and `w` become: a diagnostic, not a construction

`(w, q_φ)` is unidentified from (★). What *is* identified is the smallest `w` consistent with
`q_φ ⪰ 0`:

> `q_φ ⪰ 0  ⟺  p₀(v) ≥ (1−w) r̂*(v) ∀v  ⟺  w ≥ 1 − min_{v: r̂*(v)>0} p₀(v)/r̂*(v)`
>
> **`w_min(s,i) = 1 − exp( − max_v [ log r̂*(v|s,i) − log p₀(v|s,i) ] ) = 1 − exp( −D_∞(r̂* ‖ p₀) )`**

`w_min` is a free by-product of the two forwards we already do, and it is the self-consistency
check on the mixture model itself: if `w_min → 1`, the decomposition is degenerate and (★) carries
no information at that state. Report its distribution over the hazard index set. **[H]** I expect
`w_min` to be large at memorized answer positions (where `p₀(v*) ≈ 0.9` and `r̂*(v*) ≈ 0.05`, so
`max_v log(r̂*/p₀)` is attained at some *other* city and `w_min` is moderate) and near 0 at
structural positions. If it is near 1 everywhere, the mixture framing should be abandoned and the
target defended directly as a distillation target rather than as a deconvolution.

## A.6 Three ignorance references, one of which is gold and must not be used for training

This is an asset the brief did not name and it changes what we can prove.

| ref | construction | available on | role |
|---|---|---|---|
| `P_ign^sub` | stranger substitution on the memorized backbone (A4) | synthetic **and** RWKU | **the training target** |
| `P_ign^base` | base LLaDA at the *original* prompt | synthetic only (entities are fictional, so base is ignorant by construction) | validator |
| `P_ign^retrain` | base + memorization SFT on the **64 retain facts only** | synthetic only | **gold validator** |

`P_ign^retrain` is the genuine counterfactual `θ_ign` for the synthetic study. Theorem 6.4 and §3
rule out `θ_retrain` for *pretraining* knowledge, and rightly; they do **not** rule out retraining
the *SFT stage*, which is one run of `memorize_big.py` on half the data. The toy evaluator already
used a retrain reference (`F_wf` 0.66–0.69 against which NPO/MDU/EMDU scored 0.000) [M].

So A4, which the formalization lists as an untested **[A]**, becomes a **measurable** assumption:

> **Gate A4.** `TV( r̂*(·|s,i) , p_{θ_ign^retrain}(·|s,i) ) ≤ τ_A4` on the hazard index set, and
> `|H(r̂*) − H(p_{θ_ign^retrain})| ≤ τ_H`.

If Gate A4 fails, the target is wrong and we know it before spending the training budget. Hard
rule: `P_ign^retrain` and `P_ign^base` are **evaluation-only**. Training against either would
exploit the fictional-entity construction and would not transfer to RWKU, where neither exists.
This must be asserted in code, not just in prose.

## A.7 Cost of Part A, in forward passes

| item | forwards | note |
|---|---|---|
| target, per training step | **1** no-grad reference forward on `σ_S̃(s)`, batched over the state batch | `disable_adapter()`; `m = 1` |
| target, per step, `m > 1` variant | `m` | not recommended (§A.4) |
| Gate A4 validation | 1 SFT run (retain-only memorization, `memorize_big.py`) + 3 forwards per probed state | one-off, ≈ 0.4 GPU-h |
| stranger pool | **0** | CPU only, already built |

The reference forward is **not** cacheable here, unlike EMDU's FAM anchor: our anchor input depends
on the state's mask pattern *and* on the stranger draw. The cacheable variant is the declared
fallback (§7), and there the memory cost is bounded by truncating the target to top-`K` plus a
uniform smear of the residual: `64 facts × 32 positions × 512 × 8 B = 8.4 MB` per stranger draw.

---

# PART B — BREAKING THE ANSWER-TOKEN CHANNEL

## B.1 The channel, derived exactly — and the identity that makes it measurable

Everything in this section follows from one algebraic fact about where the measurement was taken.
`scripts/grad_cosine.py:86` trains only `lora_B` with `A` at its random init, so `dL/dA = 0` and
the measured gradient lives in the **B-subspace**. For a LoRA layer `h ↦ h + (α/r) B A h`, writing
`a = Ah` for the low-rank input activation and `δ_out` for the back-propagated error at that
layer's output,

> `∂L/∂B = Σ_{(s,i) ∈ μ} δ_out(s,i) · a(s,i)ᵀ`  — a sum of rank-one outer products.

For a single state, `⟨uvᵀ, u'v'ᵀ⟩_F = ⟨u,u'⟩⟨v,v'⟩` and `‖uvᵀ‖_F = ‖u‖‖v‖`, so

> **(B1) [P]  `cos( ∂L/∂B |_φ , ∂L/∂B |_{φ'} ) = cos( δ_φ , δ_{φ'} ) · cos( a_φ , a_{φ'} )`.**

The measured cosine **factorizes** into an *error-direction* term and an *activation* term. With a
frozen LM head `W`, the error direction at the top is
`δ(s,i) = Wᵀ e(s,i)`, `e = ∇_z ℓ`.

This identity explains the whole of Finding 1's table quantitatively, which is why I trust it:

| pair type | measured cos(g) at θ_u | `cos(a)` (template/name overlap) | implied `cos(δ)` |
|---|---|---|---|
| forget–forget, same answer | 0.916 | ≈ 0.9 (same relation **and** same surname: `Alira Ashcroft` vs `Corvan Ashcroft`) | ≈ 1.0 |
| forget–forget, same relation, diff answer | 0.067 | ≈ 0.7 (same template, different surnames) | ≈ 0.10 |
| forget–retain, same answer | 0.260 | ≈ 0.9 | ≈ 0.29 (diluted: retain side's `e ≈ 0`) |
| forget–retain, diff answer | 0.011 | ≈ 0.5 | ≈ 0.02 |

The contrast that Finding 1 attributes to "the answer token" is, under (B1), **entirely a
`cos(δ)` effect**: `cos(a)` is a property of the prompts and is high for same-relation pairs
whether or not the answers match. That is testable at near-zero cost (§8, M1) and it is the first
thing I would check, because if (B1) does not reproduce the table then my §B.3 derivation is void.

## B.2 Why an `F`-indexed target forces `cos(δ) → 1` [P]

For the exclusion target at `ε → 0`, §8.1 gives
`e = p₀ − r_ε = (q−ε)₊ · ( p₀|_F − p₀|_{F^c} )`, so

> `δ_excl(s,i) ∝ Wᵀp₀|_F − Wᵀp₀|_{F^c} ≈ w_{v*} − w̄_tail`

where `w_{v*}` is the unembedding row of the answer token and `w̄_tail` is a tail-weighted average
of many rows, whose norm is suppressed by `≈ 1/√N_eff` through cancellation. So `δ_excl ≈ w_{v*}`:
**a single fixed direction determined by `F` alone, identical at every active state and identical
for every fact sharing the answer.** Hence `cos(δ_excl) ≈ 1` for same-answer pairs, and by (B1)
`cos(g) ≈ cos(a)`, which is large by construction (Prop. 2.3: the two states differ only in the
subject span). This is the derivation of Finding 1's 0.916, and it is also the derivation of the
NTK transfer coefficient of §8.3(ii) — they are the same object seen in two bases.

## B.3 The confound: the measured amplification is probably a property of the *adversary*, not of the target

This is the part the coordinator asked me to derive rather than assert, and the derivation does not
come out the way the prediction expects.

**What was measured.** `grad_cosine.py:10` and `:100-113`: the gradient is of
`L_mem(f)` — the **memorization cross-entropy**, i.e. the *adversary's* SFT objective — evaluated
at θ_u (and now at θ₀). For that objective the error vector at an answer position is

> `e_mem = p_θ(·|s,i) − onehot(v*)`,  so  `δ_mem = Wᵀp_θ − w_{v*}`.

Now evaluate it at the two checkpoints:

- **At θ₀** the fact is fit, `p_θ ≈ onehot(v*)`, so `e_mem ≈ 0`. Gradient norm `4.32e-6` [M] —
  five orders below θ_u, exactly as the coordinator's caveat says. The *direction* is whatever
  residual survives, i.e. numerically delicate by construction.
- **At θ_u** the fact is suppressed, `p_θ(v*)` is small and `p_θ` is spread, so
  `δ_mem ≈ −w_{v*} + (small spread mean)`. **This is a clean, large, shared direction for every
  fact with the same answer — supplied by the adversary's own one-hot target, not by the
  unlearning method.**

> **Proposition B2 [P].** For any model in which `p_θ(v*|s,i)` is small at the answer position,
> the adversary's error direction satisfies `δ_mem → −w_{v*}` up to a term of norm
> `‖Wᵀp_θ‖ ≤ max_v ‖w_v‖ · ‖p_θ‖₁ / √N_eff`. Hence `cos(δ_mem) → 1` for any two facts sharing
> `v*`, **independently of the objective that produced the suppression.** By (B1),
> `cos(g_mem) → cos(a)`, which is ≈ 0.9 for a template- and surname-matched pair.

**Consequence.** The observed `0.388 → 0.916` is fully accounted for by "the facts went from fit to
suppressed", with **no causal role for the exclusion target's `F`-indexing**. Both arms of the
coordinator's prediction collapse: any method that lowers `p_θ(v*)` — including the ignorance
closure — is predicted to show the same amplification of `cos(g_mem)`.

This is the "simpler explanation" I am required to name, and it is not a quibble: it would mean the
causal account in `FINDING_answer_channel.md` §"This completes a causal account" steps 1→3 is
**not established** by this measurement, while step 4 (the behavioural 1.000 vs 0.514 split)
remains measured and real — it just has a different cause. Under Proposition B2 the behavioural
leak is a property of *the adversary + the shared unembedding row + the model class*, not of the
unlearning target. That is a stronger and more pessimistic claim than Finding 1's, and it has a
direct methodological consequence (§B.6).

## B.4 Does my candidate predict non-amplification? No. Here is the number instead.

Asked directly: **my candidate does not predict that same-answer coupling stays near 0.388.** I
derive what it does predict.

Under the ignorance closure, at a memorized state,

> `δ_ign = Wᵀp_θ − Wᵀr̂* ≈ w_{v*} − m_φ`,  `m_φ := E_{v∼r̂*_φ}[w_v]`.

Same-answer facts φ, φ' share the relation template, so `r̂*_φ` and `r̂*_{φ'}` are the *same*
relation-prior ignorance distribution up to the stranger draw. Decompose
`δ_ign = s + p_φ` with `s = w_{v*}(1 − r̂*(v*))` shared and `p_φ = −(m_φ − E[m])` private. Then

> **(B3)**  `cos(δ_ign, δ_ign') = ‖s‖² / ( ‖s‖² + σ²_priv )`,  `σ²_priv = E‖m_φ − E m‖²`.

Numbers. `‖s‖ ≈ (1 − 0.05)·‖w‖ ≈ 0.95‖w‖` at a memorized position, so `‖s‖² ≈ 0.90‖w‖²`. For the
private part, `m_φ` is a mixture mean over `j_eff` roughly-isotropic rows, so
`‖m_φ‖² ≈ ‖w‖²/j_eff`; the difference of two independent draws has
`σ²_priv ≈ 2‖w‖²/j_eff ≈ 0.20‖w‖²` at `j_eff ≈ 10`. Hence

> **predicted `cos(δ_ign) ≈ 0.90/(0.90+0.20) ≈ 0.82` versus `cos(δ_excl) ≈ 0.97`.**

By (B1) the parameter-space cosines would be `≈ 0.82·cos(a) ≈ 0.74` and `≈ 0.97·cos(a) ≈ 0.87`.
**A real but modest reduction, and nothing like 0.388.** Three riders:

1. The reduction comes from **stranger-sampling noise**, not from entity conditioning. Averaging
   over `m` strangers shrinks `σ²_priv` by `1/m` and pushes the cosine back toward 0.97. That is a
   derivation-level reason the `m = 1` choice of §A.4 is doubly right, and an honest admission that
   this particular benefit is decorrelation-by-noise, which a reviewer may legitimately call "it
   just adds noise".
2. The **leading term is `Wᵀp_θ`** — the model's own prediction. It is `w_{v*}` because the model is
   memorized. **No choice of target can remove it**, because the target only enters the subtracted
   term. The entity-vs-token framing of the *target* is therefore structurally incapable of
   delivering the predicted decoupling. If the entity is to enter, it must enter through `cos(a)`
   (the forward activations) or through the **update rule**, not through `r`.
3. Therefore the honest statement of my position is: **the entity-vs-token framing is incomplete,
   and the mechanism story in Finding 1 needs the §B.6 measurement before it can ground a method.**

## B.5 The mechanism I do propose, and what it guarantees

Given B.4, I do not propose a target-side fix for the channel. I propose the operation that has an
*exact first-order guarantee* on the half of the channel that is actually a preservation problem
(Alice/Bob), and I decline to propose anything for the half that Proposition B2 says is
irreducible.

**Derivation.** A step `θ ← θ − η u` changes a retain fact's loss by
`ΔL_ret(ψ) = −η⟨v_ψ, u⟩ + O(η²)`, `v_ψ = ∇_θ L_ret(ψ)`. Damage is `ΔL_ret > 0`, i.e. exactly the
event `⟨v_ψ, u⟩ < 0`. Finding 1's `+0.260` same-answer forget–retain coupling, read through the sign
of the relearning gradient, *is* the statement that this inner product is non-zero and 24× the
unrelated baseline. So the requirement is a half-space constraint, not a penalty:

> **(B4)**  `u_t = argmin_u ½‖u − g_t‖²₂`  subject to  `⟨v_j, u⟩ ≥ 0, j = 1..J`.

**Closed form, `J = 1`:** `u = g − min(0, ⟨g,v⟩/‖v‖²)·v`. **General `J`:** the dual is a
nonnegative least-squares problem, `u = g + Vα` with
`α = argmin_{α ⪰ 0} ½ αᵀ(VᵀV)α + αᵀ(Vᵀg)`, solved by projected gradient on `α ∈ R^J₊` using the
`J×J` Gram matrix — a handful of iterations, negligible cost at `J ≤ 8`. This is GEM
(Lopez-Paz & Ranzato 2017) with the retain twins as the memory set; **established, named, not
claimed as new.**

**What it guarantees, and what it costs.** To first order, `ΔL_ret(ψ_j) ≤ 0` for every constrained
twin, **exactly, with no weight to tune.** That matters because the formalization's condition C2
says the retain weight "that would cancel the NTK transfer ... is not derivable a priori" — the
projection removes the need to derive it. The cost in forgetting rate is exactly computable: the
directional derivative of the forget objective along the projected update is

> `⟨g, u⟩ = ‖g‖²(1 − cos²(g, v))` for an active single constraint.

At the measured `cos = 0.260`, **93.2 % of the descent rate survives.** For contrast, orthogonalizing
*across forget facts* (the H1(a)/(c) family) at `cos = 0.916` would retain only
`1 − 0.916² = 16 %` — a factor-6 slowdown. That asymmetry is why the retain-twin projection is in
the primary candidate and forget-forget decorrelation is not (and is also why I am not resurrecting
H1: as instructed, and because the numbers say it is six times more expensive for a benefit
Proposition B2 says is unavailable anyway).

**What it does NOT fix, stated plainly.** The projection removes the component of `g` along a
`J`-dimensional subspace. The shared channel is the pull-back of **one covector** `w_{v*}` through
`#states × #layers` Jacobians, so it spans far more than `J` dimensions. The energy removed is
`cos² ≈ 7 %`. Therefore **the projection is predicted to reduce same-answer coupling only slightly**
— consistent with §B.4 and inconsistent with any claim that this candidate "breaks" the channel. It
fixes first-order Alice/Bob damage; it does not fix cross-fact relearning transfer.

**Mathematical object:** constrained optimization, realized as a projection; the update is not the
gradient of any function (§0.1). It is **not** a scalar loss, **not** a stop-gradient trick, and I
make no claim of ordinary loss descent for it.

## B.6 The negative result, and the measurement that decides it

> **Proposition B5 (no target-only fix) [P].** Let the forget objective have logit-space error
> `e(s,i) = p_θ(·|s,i) − r(·|s,i)` for any target family `r`. Then
> `δ(s,i) = Wᵀp_θ(·|s,i) − Wᵀr(·|s,i)`. The first term is independent of `r`. At a memorized state
> it equals `w_{v*} + O(1/√N_eff)` and is therefore identical for all facts sharing `v*`.
> Consequently, for any two same-answer facts, `cos(δ, δ')` is bounded below by
> `‖s‖²/(‖s‖² + σ²_priv)` with `‖s‖² ≥ (1 − max_v r(v*))² ‖w_{v*}‖²`, and **no choice of `r`
> drives `cos(δ,δ')` to zero while the model is still memorized.** The only targets that could are
> those with `r(v*) ≈ p_θ(v*)` — i.e. targets that do not forget. ∎

So: *a target-only change cannot break the answer channel.* The levers are (i) the update rule
(§B.5, bounded by the energy it removes), (ii) `cos(a)` — the forward activations — which means
learning a subject-dependent readout, i.e. **degeneracy (e) by Corollary 2.4**, and (iii) where the
error lives, which is §B.7.

**The decisive measurement, and it needs no training [M-request].** Run the existing
`grad_cosine.py` pipeline, unmodified except for the adapter path, on the **four baseline adapters
that already exist**: `/data/bijaypandey/spex/adapters_baselines/{emdu, mdu_t1, npo, sc_tf}`.
None of EMDU, MDU(τ=1) or NPO uses an `F`-indexed target — their anchors are model-derived
null-prompt distributions and a preference loss respectively.

- If all four show same-answer coupling ≈ 0.9: **Proposition B2 stands**, the `F`-indexing story is
  dead, and the coordinator's screen measures the suppression level rather than the target's
  structure. (Note MDU-τ1 and EMDU reach 0.000 disclosure by emitting nothing [M, E2], so they are
  maximally suppressed and are the sharpest test.)
- If EMDU/MDU show ≈ 0.4 while our exclusion control shows 0.92 **at matched disclosure**: my
  derivation is wrong, `F`-indexing is causal, and §B.4's prediction is falsified.

Cost ≈ **12 GPU-minutes, zero training.** This is the highest-information-per-GPU-second job in the
whole design and it should run before anything is built.

**A better screen than the one proposed.** The coordinator's screen measures the *adversary's*
gradient geometry at a trained arm. The quantity that actually governs what a target does is the
cosine of **the forget objective's own gradient** across facts, at θ₀, where it is large and
well-conditioned (unlike `L_mem` at θ₀, norm `4.3e-6`). That comparison — `cos(∇L_excl)` vs
`cos(∇L_ign)` across same-answer pairs at θ₀ — requires **no training at all**, one forward+backward
per fact per target, ≈ 6 GPU-minutes, and answers the coordinator's question directly and before
the fact. Predicted: ≈ 0.97 vs ≈ 0.82 (§B.4). Both cosine-of-δ and cosine-of-activation should be
reported separately so that (B1) is checked rather than assumed.

Per the coordinator's instruction I add **no coupling penalty** to the objective. The prediction in
§B.4 falls out of (B3); it is not something the objective is steered toward, and I would regard a
coupling term as exactly the "optimize the correlate" error.

## B.7 The one genuinely MDLM-native route to an entity-indexed error — recorded, not adopted

For completeness, because it is the only construction I found that makes `e` entity-indexed rather
than answer-indexed, and because it is available **only** in a bidirectional denoiser.

Mask the *subject* and condition on the answer: the state
`s^rev = (c_rev, z)` with `c_rev = "Who lives in Paris?"` or canvas `"⟨mask⟩ lives in Paris."`. At the
subject position the error vector is supported on the **entity** surface set `A(S)`, so
`δ ≈ w_{Alice}` — not shared with Bob, whose preservation needs mass on `w_{Bob}`, a *different
coordinate*. The Alice/Bob conflict becomes coordinate-separable, which it never is in the forward
direction. This is P5's reverse-direction constraint used as a *mechanism* rather than a
side-condition, and an autoregressive model cannot do it (no conditioning on the suffix).

I do **not** adopt it as the primary, for three honest reasons: (a) whether reverse-direction
training transfers to forward disclosure is the reversal-curse question and is unmeasured — it may
transfer at zero; (b) it trades the shared-answer coupling for shared-*subject* coupling, which
damages ψ₂ (same entity, other relation) — and SPEX-128 contains **no** ψ₂ pairs (every entity
carries exactly one fact), so by Proposition 5.1 that damage is **unidentified in our data and
would not be seen**; (c) adding it would make the candidate a four-operation bundle. It is recorded
as the first extension to test if the §B.6 measurement kills the target-side story, and it comes
with a required data fix (§D.7).

---

# PART C — MASK-RATE AND POSITION TARGETING

## C.1 Derive the index set from R2's union bound, not from the mask rate

R2 forbids justifying a per-position divergence as a bound on `G^π_θ` (Corollary 3.1: a 0.05
output-level guarantee needs per-position `KL ≈ 1.1e-9`). The only permitted transfer is **a union
bound on the disclosure event**. Write it out for the sampler of §0:

> **(C1) [P]**  `Pr[ disclosure ] ≤ Σ_{n=1}^{N} E_{s_n ∼ G^π_θ} [ Σ_{i ∈ C_n(s_n)} h_i(s_n) ]`
>
> with `h_i(s) = 1{ argmax_v p_θ(v|s,i) ∈ F }` at `T = 0` (Lemma 4's rank condition) and
> `h_i(s) = p_θ(F|s,i)` at `T > 0` (Lemma 4's mass condition, Theorem 2.1(ii)).

Every ingredient of the training distribution now falls out, and nothing is chosen:

- **the states** are `s_n`, drawn from the model's own trajectory — on-policy, satisfying R4;
- **the positions** are `C_n`, the **committed set**, not `M(s_n)`;
- **the weight** is 1 on committed positions and **0 everywhere else** — every other
  `(state, position)` pair in the teacher-forced `U(0,1)` distribution contributes *nothing* to the
  bound;
- **the truncation** is at the first `F`-commit `τ`: beyond `τ` the event has already occurred, and
  states past it are σ_F = 1 states where §5.6 says the correct behaviour is the opposite.

So the answer to "which `(mask rate, position)` states should the objective be applied to" is:
**`(step n ≤ τ, position i ∈ C_n)` along the model's own decode trajectory, under each config in a
declared training pool.** The mask rate is not selected at all — it is whatever the trajectory
visits, which (§0.2(ii)) is already uniform on the grid. **This is a derivation, and it says the
mask-rate axis that PRIOR_ART identifies as unoccupied is unoccupied because it is the wrong
axis.** I would rather report that than take the one-line change because it is cheap.

I note the overlap with published work honestly: MDU's own §7 names exactly this gap as its next
goal; MDU publishes a stored-knowledge token-role diagnostic (−20.6 % KL) and **does not use it**;
CUSE and RULE do on-policy trajectory objectives for *autoregressive* models; TimeROME-DLM occupies
trajectory-aware *localisation*. Per PRIOR_ART Part 3 this lands at **category (e)** at the outset
and can only reach (d) through the factorial ablation of §F. And PRIOR_ART's warning applies to us
too: CUSE's Eq. 12 carries `E_{Y∼p_θ}` but its implemented gradient has no score-function term. I
am making the same choice and I declare it: **we differentiate through the per-position loss only,
and treat the state distribution and the commit selection as stop-gradient.** The gradient is
therefore a biased estimate of `∇` of (C1); the bias is the omitted score-function term and we do
not estimate it.

## C.2 How the commitment states are identified, exactly

Already implemented and pinned: `llada_fxmix.generate_traced(..., fact_mask=F_mask, record=True,
stop_when_all_leaked=True)` returns `{"trace": [...], "first_commit": τ, "n_steps_run": n}`, where
`first_commit[r]` is the first global step at which `transfer_index & gather(fact_mask, x0)` fires
for row `r` (`llada_fxmix.py:265-268`). The commitment **position** is the member of
`transfer_index` at step `τ` whose `x0` lies in `F`. `select_rollout_states` (`:274`) already
returns `J` pre-τ states plus the decision state `s_τ`.

So: no new sampler, no inference-time hook (§3 compliance), and the trajectory collection is a
read-only use of the author's `generate` with side-effect-free extras that were tested identical to
upstream.

## C.3 The commitment distribution is NOT measured. The measurement, and why correctness does not depend on it.

**[H]** I have no measurement of whether `(τ, i)` is concentrated. Proposed job (§8, M3):

> Instrument `generate_traced` over **64 forget facts × 3 training-pool configs × 3 seeds** at θ₀
> (the memorized model). Record `(τ, i_τ, conf_τ, |C_τ|)` per row. Report: the histogram of `τ`;
> `H(τ)` in bits against `log₂ 32 = 5`; the mass of the modal step and of the modal
> `(step, position)` cell; the mean `τ̄` (which fixes the collection cost); and the fraction of rows
> with `τ = −1` (no `F` commit, i.e. already non-disclosing).

**Two competing predictions, both derivable, which is why it must be measured.**

- `lc` remasking commits the most *confident* position first, and a memorized answer token has
  `p₀(v*) ≈ 0.9` while filler tokens are less certain ⇒ **`τ` concentrated at the earliest steps.**
- MDU's published token-role diagnostic says in-context tokens spike early and **stored-knowledge
  tokens late** (their Fig. 3) ⇒ **`τ` concentrated late**, the answer conditioning on
  model-written context — which is precisely Lemma 5's re-entailment hazard.

These have opposite operational consequences (`τ̄` small ⇒ collection is nearly free and the hazard
sum has few terms; `τ̄` large ⇒ collection costs most of a full decode and the displaced-disclosure
risk is real), so the measurement is decisive for cost but not for correctness.

**Correctness does not depend on concentration, and this is where I part company with H2 as
written.** H2 says "If disclosure turns out to be decided diffusely rather than at an identifiable
commitment step, there is nothing to target." That is wrong: (C1) is a **sum over all `n ≤ τ`**, so
it is valid whether or not `τ` concentrates. What concentration buys is a smaller state budget per
unit of bound, i.e. **cost**, not validity. If `τ` is diffuse the candidate does not die; it
becomes more expensive and its cost advantage over uniform-`t` disappears, leaving the
on-policy-content argument (§0.2(ii)) as the whole of part C's contribution. That is a weaker but
still non-empty position, and it should be stated that way in advance rather than after the
measurement.

## C.4 Cost relative to uniform `t`, measured where possible and estimated where not

**Per-step cost is lower, not higher.** Uniform `t` applies the loss at `|M(s)| ≈ L·E[t] = 16`
positions per state (at `L = 32`); the hazard index set applies it at `|C_n| = k = gen_length/steps
= 1`. The soft-target tensor shrinks correspondingly, and this removes the OOM site PRIOR_ART
§2.2 identifies:

| index set | fp32 soft-target tensor | note |
|---|---|---|
| all positions, batch 4 × seq 1024 | `4·1024·126464·4 B = 2.07 GB` (1.04 GB in bf16) | the predicted OOM site |
| `M_t` only, `t ~ U`, 8 states × 16 pos | `128 · 126464 · 4 B = 65 MB` | what `arc_big.py` already does |
| **hazard set, 8 states × k=1** | **`8 · 126464 · 4 B = 4.0 MB`** | **260× below the bf16 full-vocab figure** |

So the full-vocab target, named in the brief as the likely OOM site, is **structurally removed** by
the derivation rather than mitigated by chunking. The existing `[rows, cols]` gather pattern
(`fx_forget_loss_llada:455-462`) is retained unchanged.

**Amortized collection cost is the real price.** One refresh of the pool costs
`⌈64/B_gen⌉ · min(τ̄, N)` no-grad forwards per config. With `B_gen = 16`, three training configs,
and `τ̄` to be measured:

| `τ̄` | forwards per refresh | at refresh period `P = 25` | as a fraction of a 4-forward step |
|---|---|---|---|
| 4 | 48 | 1.9 / step | +48 % |
| 8 | 96 | 3.8 / step | +96 % |
| 16 | 192 | 7.7 / step | +192 % |
| 32 (no truncation) | 384 | 15.4 / step | +385 % |

**This table is why M3 is a gate and not a curiosity.** If `τ̄ ≥ 16`, part C costs more than the
rest of the method put together and the fallback (§7) should be taken.

**Staleness, declared (R4).** The state pool is `P` steps stale; the *position* set is **not**,
because `Ĉ_n(θ_t)` is recomputed each step from the student's current confidences on the stored
state (free — we already have those logits). Lemma 5 guarantees the edit pushes the fact position
later in the commit order, so a stale pool systematically under-covers the new decision states;
the observable signature is **disclosure rising between refreshes and falling at each refresh**,
which must be logged per step and reported. If that sawtooth is large, `P` must fall and the cost
table above must be re-read at the smaller `P`.

## C.5 The constraint family, and its collision with the target

(C1) gives the constraint functionals directly (R3: two config families, two constraint types):

> `c_rank(s,i;θ) = [ γ + max_{v∈F} z_θ(v|s,i) − max_{v∉F} z_θ(v|s,i) ]₊`   (`T = 0`, `lc`)
> `c_mass(s,i;θ) = [ log p_θ(F|s,i) − log ε_eff(s,i) ]₊`                     (`T > 0`)

Three deliberate choices:

1. **Logit margin, not renormalized mass.** A hinge on logits never *specifies* where the removed
   mass goes, so Proposition 8.4's amplification `(1−ε)/(1−q) → ∞` of an omitted surface variant
   **does not occur**: there is no `1/(1−q)` factor anywhere. The redistribution is specified by
   the ignorance target instead, which does not single out `p₀`'s runner-up. `F`
   mis-specification then costs us a variant left at the *ignorance* rate rather than one actively
   promoted — a milder, declared failure, and a partial answer to R5/Corollary 8.5.
2. **Hinges, so the constraint switches off.** Lemma 4 says mass reduction past the rank flip buys
   nothing for greedy and Corollary 2.2 says it is what zeroes Bob discontinuously
   (`ε = 1e-5` sits a factor 1.26 above the `1/|V| = 7.91e-6` cliff). A hinge stops pushing; a KL
   term does not. This is the direct implementation of A6.
3. **`ε_eff(s,i) = max(ε, r̂(F|s,i))` — the floor is the ignorance reference's own `F` mass.**
   Without this the constraint and the target *fight*: `r̂*` gives Paris real mass (Paris is a
   plausible city), and a hard `ε = δ/n̂ = 0.01/32 = 3.1e-4` would force the student below its own
   ignorance reference, which is over-suppression by construction and reintroduces the (c)/(e)
   failures. With the floor, the constraint is **inactive at the ignorance fixed point** and its
   only remaining job is the one it is actually needed for: **catching displaced disclosure at
   states the target happens not to be fit at** (Lemma 5, degeneracy (f)). Say that plainly — the
   constraint is a robustness device, not a rate-reduction device, which follows from §0.2(i).

`ε = δ/n̂` with `δ = 0.01`, `n̂ = L = 32` ⇒ `ε = 3.1e-4`. **Per R15 this is a design quantity three
orders below what any feasible battery can certify** (Corollary 6.2: `n = 1000`, `m = 500` cells
⇒ `δ̂ = 9.2e-3` per cell) and must never be reported as a leakage rate.

---

# PART D — THE COMPLETE PROCEDURE

## D.1 Data and assets (all verified on disk)

| role | path | note |
|---|---|---|
| base weights | `/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct` @ `08b83a6f…` | no downloads (`HF_HUB_OFFLINE=1`) |
| sampler | `/data/bijaypandey/llada_rwku/repos/llada_official` @ `9182493` | author's `generate`, read-only |
| memorized backbone | `/data/bijaypandey/spex/adapters/memorized_big` | PEFT r=32 α=64, 224 modules; merged at load |
| facts | `/data/bijaypandey/spex/data_big/{facts.json,forget.jsonl,retain.jsonl}` | 64 forget + 64 retain, 32 answers × (2F+2R) |
| stranger pool | `/data/bijaypandey/llada_poc/data/name_pool.json` | 5655 names, token-length buckets {3,4,5,6} |
| pinned target/loss/states | `llada_poc/scripts/{mdmu/fxmix.py, llada_fxmix.py}` | `soft_ce_kl`, `generate_traced`, `build_fact_set`, `sft_loss_llada`, `swap_perm`, `accept_name` |
| existing baselines | `/data/bijaypandey/spex/adapters_baselines/{emdu,mdu_t1,npo,sc_tf}` | for §B.6, no training needed |
| coupling pipeline | `/data/bijaypandey/mdlm_unlearn/scripts/grad_cosine.py` | reuse unmodified except adapter path |

## D.2 Parameterization, and where the edit is allowed to live (R8)

- **LoRA only.** Full-parameter 8B does not fit 2×40 GB (`PAPER_ALIGNED_RUN_BLOCKED` [M];
  PRIOR_ART §2.1: full-param steady state 128.3 GB vs 80 GB budget, ZeRO-3/2 still 64.2 GB).
  A second LoRA stacked on the merged memorized weights, `r = 32, α = 64`, dropout 0,
  `target_modules = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"`.
  The block-anchored regex is **required**, not stylistic: the bare leaf `ff_out` collides with the
  top-level LM head `model.transformer.ff_out` in this checkpoint (`common_big.py:158-161`).
  `check_lora_targets` must assert 7 wrapped modules per block and 0 at top level, and the run must
  abort otherwise.
- **The LM head is frozen.** This is forced, not preferred: Corollary 2.4 and Proposition 8.3 show a
  trainable output embedding gives degeneracy **(c)** (global vocabulary suppression) via a
  context-independent `Δw_{v*}`. Freezing it converts that into degeneracy **(e)** (a learned query
  gate). We choose (e) and must therefore *measure* it: the held-out-paraphrase gate `G_para`
  (promoted to a **gate**, not a report, per §4(e)1), the interpolation width `w` vs `w_ref`, and
  the hidden-state detector probe `ΔAUC`. R8 requires a term or constraint penalizing whichever
  degeneracy we made available; we do not have one, and I record that as a requirement **not
  satisfied** (§E.5).
- fp32 LoRA parameters, bf16 frozen weights (`arc_big.py:161-166`); gradient checkpointing on.
  Steady-state ≈ 17–18 GB, comfortably inside one 40 GB card at `L = 64`.

## D.3 Training states, in full

1. **Forget states — on-policy.** Every `P = 25` steps, for each of the 3 training-pool configs,
   run `generate_traced` from the all-mask canvas on the 64 forget prompts (DEV half only; §D.6)
   with `fact_mask`, `record=True`, `stop_when_all_leaked=True`. Keep, per row, `min(J, τ)` pre-τ
   states sampled uniformly plus the decision state `s_τ` (`select_rollout_states`, `J = 4`).
   Discard rows with `τ = −1`? **No** — keep them with the full trace, because a currently
   non-disclosing row is exactly where displacement would land; log their count.
2. **Index set per step.** From the sampled states, recompute `Ĉ_n` as the top-`k` of the student's
   current `conf_i = max_v p_θ(v|s,i)` over `E_n`, **under stop-gradient**, with `k` from the
   config's `num_transfer_tokens`. `I_t` = those positions.
3. **Retain states — three distinct families (R7).**
   - `ℜ_twin`: the 2 shared-answer retain twins of each forget answer. **This is the constraint set
     in (B4).** Present in SPEX-128 by construction.
   - `ℜ_ind`: independent retain facts (different answer, same relation). Scalar-weighted term.
   - `𝒮_neu`: neutral-use states where an `F` token is the legitimate continuation
     ("the capital of France is ___"). **Does not exist and must be built** (CPU-only, §D.7). The
     P3 selectivity contrast `SEL = λ(𝒮_φ) − λ(𝒮_neu)` cannot be computed without it, and no
     evaluator in any of our five stacks reads logits at all.
   Retain states are teacher-forced at `t ~ U(eps,1)` via the pinned `sft_loss_llada` /
   `random_tf_states_llada`. On-policy collection for retain is **not** done, and I declare the
   asymmetry: preservation is a statement about matching θ₀, and θ₀'s own trajectory is the right
   reference there — but we use the forward-corruption distribution instead, for cost. Declared
   approximation, cost unquantified.

## D.4 The step, in full

```
# once per P steps
if t % P == 0: pool ← collect_onpolicy(θ_t, forget_DEV, configs=[π1,π2,π3])   # §D.3.1

# per step t
S      ← sample B_f = 8 states from pool
Ĉ      ← stopgrad_topk_confidence(θ_t, S)                                      # §D.3.2
S̃      ← sample one admissible length-matched stranger per state              # fresh, §A.4
z0     ← no_grad, disable_adapter: logits(σ(S))        [rows,cols=Ĉ] .float()   # 1 fwd
log r̂  ← permute_vocab(log_softmax(z0), σ)                                      # involution pull-back
z      ← logits(S)                                     [rows,cols=Ĉ] .float()   # 1 fwd (grad)
L_ign  ← soft_ce_kl(log r̂, z).ce.mean()                                         # §A.3
c_r    ← relu(γ + max_F z − max_notF z).mean();  c_m ← relu(logsumexp_F(z) − log ε_eff).mean()
g      ← ∇_θ ( L_ign + λ_r c_r + λ_m c_m )                                      # 1 bwd
# retain branch
v      ← ∇_θ [ KL( p_θ0(·|S_twin) ‖ p_θ(·|S_twin) ) + ω·L_sft(S_ind) + ω_n·λ(𝒮_neu)² ]
                                                                 # 2 fwd (1 no-grad) + 1 bwd
u      ← cone_project(g, [v_twin])                                              # (B4), closed form
set .grad ← u + β v ;  clip_grad_norm_(1.0) ;  AdamW.step()
λ_r ← relu(λ_r + η_λ c_r) ;  λ_m ← relu(λ_m + η_λ c_m)                          # dual ascent
```

Hyperparameters, all declared before any run: `lr 1e-4` (matched to every prior arm), AdamW
`wd 0`, `betas (0.9,0.999)`, warmup 10, grad clip 1.0, `steps 300` with checkpoints at
`{50,100,150,200,300}`, `B_f = 8` states, `J = 4` pre-τ states per row, `P = 25`, `γ = 2.0` logits,
`δ = 0.01` ⇒ `ε = 3.1e-4`, `η_λ = 0.05`, `λ` init 0, `β = 1.0`, `ω = 1.0`, `ω_n = 1.0`,
`m = 1` stranger, seeds `{0,1,2}` × 3 memorized starting checkpoints (EVAL_PROTOCOL requires ≥3
unlearning seeds and ≥3 memorizations on Tier A).

## D.5 Measured cost per step

| item | forwards | backwards |
|---|---|---|
| student on forget states | 1 (grad) | shared |
| reference on `σ(S)` (adapter off) | 1 | — |
| student on retain states | 1 (grad) | shared |
| reference on retain states (for the KL anchor) | 1 | — |
| **gradients taken separately for the projection** | — | **2** |
| **per-step total** | **4** | **2** |
| amortized on-policy collection, `P = 25`, 3 configs | **+1.9 … +15.4** (τ̄-dependent, §C.4) | 0 |

Against MDU/EMDU's 3 forwards + 1 backward, that is **1.33× the forwards and 2× the backwards per
step, plus a τ̄-dependent collection overhead of 0.5× to 4×**. Total ≈ **2.2× MDU at `τ̄ = 8`.**
Extra memory: two fp32 gradient buffers over the 23.07 M-parameter B-subspace (46 M params at
r=32 for A+B) ≈ 2 × 185 MB. Projection arithmetic: a `J×J` Gram matrix, negligible.

Budget estimate against the ledger (prior total 4.117 GPU device-h): one IHP arm at 300 steps with
`τ̄ = 8` is ≈ 0.35 GPU-h; the full ablation ladder of §F at 3 seeds is ≈ 4.5 GPU-h; the evaluation
panel (896 probes per checkpoint at 1–2 s/probe) dominates, as EVAL_PROTOCOL says.

## D.6 Selection rule, and inference behaviour

- **Inference:** plain LLaDA sampler on the edited weights. No prompt prefix, no logit filter, no
  decoding hook, no post-processing. §3 compliant. (Note EMDU's "replace the empty completion with
  'I don't know'" is a wrapper they do not evaluate; we take none.)
- **Selection:** on **DEV only**, per EVAL_PROTOCOL's split (Tier A: 26F+26R DEV, 38F+38R sealed;
  answer clusters never split). Not a scalar-on-test rule, and explicitly not EMDU's
  "largest Forget/Retain gap on the reported metric".
  > Choose the earliest checkpoint satisfying **all** of: `C1 ≤ 0.10` shared-answer retain drop,
  > `C2 ≤ 0.05` independent retain drop, `C4 ≤ 0.03` MMLU, `C5 ≤ 0.02` degeneracy,
  > `G_para ≤ g*`, `TV(Ĝ_O, P_ign) ≤ τ_ign`, `|H(Ĝ_O) − H(P_ign)| ≤ τ_H`; among those, the one
  > minimising `D_any`. If none qualifies, the run is PRESERVATION-FAILED and reports no
  > forgetting number.
- **Reporting:** the two-coordinate pair `(static CF_strict, post-RTT CF_strict at k=10)` with CIs
  on both. No single forgetting scalar, no ranking on static disclosure (EVAL_PROTOCOL §8.1), and
  per §0.2(i) no claim of lower disclosure than the ignorance band at all.

## D.7 Non-GPU prerequisites that the objective cannot proceed without

1. **`P_ign` and `PLL_{θ0}` do not exist** anywhere (EVAL_PROTOCOL has zero occurrences of either)
   and R10 makes them prerequisites. `P_ign` is a by-product of §A.2. `PLL` must be built; note
   EVAL_PROTOCOL forbids any ELBO/pseudo-likelihood stand-in from being a *gate*, so `PLL` can only
   be a relative fluency indicator with its MC standard error shown.
2. **`𝒮_neu` neutral-state panel** (P3/`SEL`): does not exist. ~200 frozen contexts where each of
   the 32 answer tokens is the correct continuation. CPU-only.
3. **ψ₂ pairs (same entity, other relation) do not exist in SPEX-128** — each of the 128 entities
   carries exactly one fact. By Proposition 5.1 any damage to ψ₂ is **unidentified by any objective
   built from our `(𝔉, ℜ)`**, and EVAL_PROTOCOL's C3 gate is declared unevaluable below 20 pairs.
   Fix: extend the fact set with a second relation per entity for ≥20 entities, re-memorize. This
   is a data change with a GPU cost (re-memorization) and it blocks any claim about subject-side
   collateral.
4. **All 128 answers are single-word** (EVAL_PROTOCOL: "the weak regime"). R12's identifiability
   scoring has nothing to bite on, and "suppression of one token is not knowledge removal". Multi-
   token answers are needed before any claim about partial disclosure.
5. `grad_cosine.py` **already** takes `--unlearn-adapter` (`:62`), so job M1a needs no code change at
   all — only four invocations with different adapter paths. Verified on disk; this is the cheapest
   decisive job in the design and there is no engineering reason it has not already run.

---

# PART E — HONEST SELF-ASSESSMENT

## E.1 The precise limitation of existing methods this addresses

Not "they forget less". Three specific, documented defects:

1. **MDU, EMDU and our own exclusion control all evaluate a per-position divergence at teacher-
   forced states drawn from `t ~ U(0,1)` on the ground-truth answer.** Those states place `F` tokens
   in the bidirectional visible context (EMDU's own failure mode 1, and §5.6's σ_F = 1 partition
   where the correct behaviour is the opposite), and they are mostly off the reachable set. The
   candidate's index set is derived from the union bound on the disclosure event instead, which is
   the only transfer R2 permits.
2. **The exclusion target's fixed point is a memorized replacement** (Prop. 8.2: `argmax_{v∉F} p₀`,
   `MAR = 1`, `mode ∉ Σ(O)`), confirmed in our own control: 35/64 forget probes assert a different
   valid answer and **19 of them the same one ("Tbilisi")**, fluently justified, scored as success
   by every metric we have. The candidate replaces that fixed point with an estimate of the
   ignorance distribution, which is the only non-circular counterfactual our access model admits.
3. **The retain weight that would cancel NTK transfer to a shared-answer twin is not derivable**
   (condition C2). The candidate replaces it with a first-order-exact half-space constraint and so
   has no such weight to tune.

## E.2 The distinguishing algorithmic operation

**One operation, named:** *the forget objective is evaluated only at `(s_n, i)` pairs with
`s_n` on the model's own truncated decode trajectory and `i` in the step's committed set `Ĉ_n`,
with everything else carrying weight zero.* That is the index set of (C1).

The other two components are declared **not** distinguishing and are attributed to their prior art:
the divergence is EMDU's anchor-first KL (identical to soft-CE, PRIOR_ART category (a)); the
stranger-substituted target is our own TD-SSD's teacher (`research/llada-poc-v1` @ `c3578a908d6c`)
and, in the autoregressive literature, Liu et al. (arXiv:2407.16997) `p(Y|do(X))` distillation with
"changing the name back"; the projection is GEM. Per the Part 3 rubric the honest standing is
**category (e) at the outset**, with a route to (d) only through the full factorial of §F including
the interaction cell and a negative control.

## E.3 Why it should improve the forgetting–utility trade-off

The mechanism is first-order and quantitative, stated before measurement. Collateral damage to a
shared-answer twin is

> `Δz_{v_P}(s_B) ≈ −η Σ_{(s,i) ∈ supp μ} K_{θ0}((s_B,i_B,v_P),(s,i,v_P)) · e_{v_P}(s,i)`  (§8.3(ii))

a **sum over the support of the training distribution**. The candidate acts on all three factors:

- **the number of terms** falls from `|supp μ| ≈ B · |M(s)| ≈ B · 16` to `B · k = B · 1` — a 16×
  reduction in transfer terms per state at `L = 32`;
- **the magnitude of `e_{v_P}`** falls, because the ignorance target removes mass only down to
  `r̂*(v_P)` rather than to `ε`: `e^{ign}_{v_P} = p₀(v_P) − r̂*(v_P)` vs
  `e^{excl}_{v_P} = p₀(v_P) − ε`. Modest (a few per cent of `p₀(v_P)`), and I will not oversell it;
- **the inner product with the twin's own descent direction** is set to `≥ 0` exactly, by (B4),
  at a cost of `cos² = 6.8 %` of the forgetting rate.

Because disclosure has no headroom (§0.2(i)), the predicted improvement is **not** on disclosure. It
is on: shared-answer retain (C1 gate), ignorance-matching (`TV`, `H`, `MAR`), and the
degenerate-replacement count — the axes on which our 0.031/1.000/0 control is currently scoring a
*replacement* as a success.

## E.4 Conditions under which it should fail — and the observables that detect each

| # | failure condition | observable |
|---|---|---|
| F1 | **`P_ign` is itself sharply moded.** If the reference's behaviour on unknown fictional subjects is a confident single guess, the ignorance target's fixed point is *also* a replacement — just an honest one. Ignorance matching then does not avoid degeneracy (d); it redefines the bar as "no worse than the reference". | `H(P_ign)`, modal mass, `MAR(P_ign)` — job M2. **Kills H3's premise if it fires.** |
| F2 | **`P_ign` is a refusal/hedge.** If θ₀ hedges on unknown fictional names, the target *is* a refusal target, and `Î_item ≈ 0` would score as "matching ignorance". | `Î_item(P_ign)` on the unknown-subject panel, job M2. |
| F3 | **Gate A4 fails:** `TV(r̂*, p_{θ_ign^retrain})` large. The estimating closure is then no better founded than the assuming one. | job M4. |
| F4 | **`τ̄` large / `τ` diffuse.** Collection cost dominates (§C.4 table); candidate survives but loses its cost argument. | job M3. |
| F5 | **Pool staleness.** Lemma 5 displacement outruns `P`. | sawtooth in per-step disclosure between refreshes; held-out-config disclosure max (not mean). |
| F6 | **The projection is a no-op.** If `⟨g_f, v_ψ⟩ ≥ 0` at θ₀, (B4) never activates and the candidate collapses to its own IHP−P ablation. Finding 1's `+0.260` is a *relearning*-gradient cosine at θ_u, not the quantity (B4) acts on. | fraction of steps with an active constraint, logged; job M1b measures the right cosine at θ₀. |
| F7 | **The projection kills forgetting.** If the true `cos(g_f, v_ψ)` at θ₀ is ≥ 0.9, only `1−cos² = 19 %` of the descent rate survives. | same job M1b, before any training. |
| F8 | **Config transfer.** Training on a 3-config pool may still leave leakage on held-out schedules; the toy measured leakage 1.0 on other schedules from a single config, and max 0.011 over 30 held-out configs from a 3-config pool [M]. | held-out config panel with a **max**, plus a `k`-extrapolation cell. |
| F9 | **Conjunction floor.** At `k ≥ 2` parallel commits, `Pr[joint] = p(A)·p(B)` and killing the product means killing a factor (Thm 5.5). Indirect queries and P5 list answers are bounded below. | report the floor; or restrict Π to `k = 1` and say so. |
| F10 | **Degeneracy (e).** The frozen LM head forces the suppression to be realized as a learned subject readout. | `G_para` as a gate, interpolation width vs `w_ref`, detector-probe `ΔAUC`. |

## E.5 Observable predictions that separate this from simpler explanations

Three, in decreasing order of how sharply they separate. All are stated **at matched disclosure**
(steps tuned per arm to equal `D_any` on DEV), which is what excludes "it just trains harder" and
"it just trains less".

**P1 — the per-fact correlation, which is the test H1 failed to deliver.** For each forget fact,
record `ρ_φ = cos(g_f(φ), v_ψ(φ))` at θ₀ **before training** (job M1b). Prediction: the per-item
improvement in shared-answer twin accuracy, IHP vs IHP−P, **correlates with `ρ_φ`** — facts whose
conflict was projected out improve, facts with `ρ_φ ≈ 0` do not. "It just trains harder/less"
predicts a uniform shift with **no** correlation. This is the per-fact form that the ARC study got
wrong (aggregate gap 0.101 presented as confirming a per-fact prediction) and that H1 then failed;
I am stating it as a correlation with a pre-registered predictor, not as a group difference.

**P2 — the ignorance-shape prediction, which no suppression method can fake.** Over the 64 forget
facts, build the histogram of the modal generated answer. Our exclusion control puts **19/64 on a
single answer** ("Tbilisi"). Prediction: IHP's modal-answer histogram has entropy within `τ_H` of
`P_ign`'s, `MAR(IHP) ≈ MAR(P_ign)` rather than ≈ 1, and `TV(Ĝ_O, P_ign) ≤ τ_ign`. "It just
degrades the model" predicts empty/degenerate outputs (caught by `CF_strict`'s degeneracy term) or a
*more* concentrated attractor — **not** a matched spread. A distributional-shape match is the one
prediction that suppression, refusal, degeneration and replacement all fail in different
directions, which is why it is the primary separator.

**P3 — the edit-norm prediction for part C.** At matched disclosure, `‖Δθ‖_F` for IHP should be
**strictly smaller** than for IHP−H (uniform-`t`, all masked positions), and the ratio should track
the ratio of error terms (`k / E|M(s)| = 1/16`) rather than being a free parameter. "It just trains
less" is excluded by the matched disclosure; "it just uses fewer positions so each step is noisier"
predicts a *larger* norm, not a smaller one.

**And one prediction I am obliged to state even though it goes against my own candidate:** per
§B.4, IHP's same-answer coupling under the coordinator's screen is predicted to come back **high
(≈ 0.74–0.87), not ≈ 0.388.** If it comes back near 0.388, my §B derivation is wrong and the
entity-vs-token story is right — which would be good news for the method and bad news for this
document. I would rather be on record before the screen runs.

## E.6 Which of the 17 requirements the candidate does NOT satisfy

| R | status |
|---|---|
| R1 | **partial.** We can index by `(Π, 𝒜, ν̂)`, but `Π` must be restricted to `k = 1` or we accept R14's floor. |
| R3 | satisfied (two constraint families, hinge form). |
| R5 | **partial.** The target never references `F`; the *constraint* must. The hinge form removes Prop. 8.4's amplification, but a surface variant omitted from `F` is left at the ignorance rate, unsuppressed. |
| R6 | **partial, and this is the honest centre of §B.** The target is state-indexed and `F`-free; but Proposition B5 proves the *error direction* retains a shared `w_{v*}` leading term under **any** target. The edit is therefore not a function of `(S,R)` alone, and the projection removes only ≈ 7 % of the shared energy. **Not satisfied.** |
| R7 | **partial.** `ℜ_twin` and `ℜ_ind` are represented; `𝒮_neu` must be built; **ψ₂ (same-subject) pairs do not exist in the data at all**, so by Prop. 5.1 they are unconstrained and the damage would be invisible. **Not satisfied.** |
| R8 | **not satisfied.** We choose and declare where the edit lives (frozen LM head ⇒ degeneracy (e)), but R8 also requires *a term or constraint penalizing* it. We have detectors (`G_para`, probe `ΔAUC`, interpolation width), not a penalty. |
| R9 | satisfied — this is the A4 closure. |
| R10 | **prerequisite, not satisfied today.** `PLL_{θ0}`, `P_ign`, `Î_item`, canvas-level `eos_only`/post-`eos`/floor-free repetition, `SEL`, `MAR`, `G_para` — none exists; no evaluator in any of five stacks reads logits. |
| R11 | **not satisfied.** No endorsement contrast `EC` exists anywhere, and the candidate adds none. Stack B detects answer-supplying items and never scores them. |
| R12 | **not satisfied and not testable in our data** — all 128 answers are single-word. |
| R13 | **not satisfied.** SPEX-128 has no entailment structure, so `cl(φ)` is empty by construction and the claim must be restricted to the read-off adversary on `Q_φ`, explicitly. |
| R14 | **not satisfied without a restriction.** Either `Π` is restricted to `k = 1` or the config-dependent product floor is reported. We will report it. |
| R16 | **not satisfied, and this is the largest gap.** The candidate contains **no** mechanism against relearning. The forget–forget 0.916 channel is untouched, and Proposition B2 says it is supplied by the adversary's own objective rather than by our target — so the candidate is predicted to be **about as relearnable as the exclusion control** (0.023 → 0.605 at k=10 [M]). Any relearning number we produce under LoRA is a lower bound (three independent sources agree parameter-subset unlearning does not survive fine-tuning) and the gap cannot be attributed to the method. |
| R17 | satisfied in the reporting rule (§D.6), contingent on R10's instruments existing. |

R2 and R4 are the two the candidate is built from and I claim them satisfied *in construction*;
R4's fixed-point character is handled by iteration with a declared staleness and a logged sawtooth,
not by convergence.

---

# PART F — ABLATIONS

## F.1 The ablation that removes the distinguishing operation, everything else fixed

**IHP−H.** Identical target (one fresh length-matched stranger per state per step, soft-CE),
identical constraints with identical `γ, ε_eff`, identical projection against identical twins,
identical retain families and weights, identical optimizer, seeds, steps, checkpoints, LoRA config
and selection rule. **Only** the index set changes: states are teacher-forced
`random_tf_states_llada` at `t ~ U(eps,1)` and the objective is applied at **all** masked answer
positions. This isolates (C1)'s index set and nothing else. It is also the arm that re-creates the
1.04 GB soft-target tensor, so it doubles as the memory check.

Two further single-factor ablations, required for the factorial:

- **IHP−I** — hazard index set + projection, but the **exclusion target** `r_ε` in place of `r̂*`.
  Isolates the closure (A2+A3 vs A4). This is the arm that tests the coordinator's prediction
  directly, and §B.4 says it should differ from IHP by ≈ 0.15 in `cos(δ)`, not by 0.53.
- **IHP−P** — hazard index set + ignorance target, projection replaced by a **tuned scalar retain
  weight** (swept on DEV so the comparison is not a straw man). Isolates (B4).

Full `2×2×2` factorial over {index set} × {target} × {preservation}, 3 seeds × 3 memorizations per
cell, plus the interaction cell and a **negative control** (a third, unrelated mechanism — e.g.
SAM, which PRIOR_ART recommends over Fan et al.'s GP because GP's released implementation is a
no-op) to confirm any interaction is specific and not "any two things help". EMDU's Table 2 is the
template; EMDU's gap — no seed spread on the *ablation* cells — is the one we must not repeat.

## F.2 The strongest *simple* combination a reviewer will demand

Three, and all three must be run:

1. **Faithful EMDU with our selection protocol and our retain data.** Anchor-first KL + FAM at
   `τ = 1`, `t ~ U(0,1]`, LR 1e-5, plus a retain term with `ω` swept on DEV, at matched disclosure.
   This is the published SOTA objective, specified completely in PRIOR_ART (LR 1e-5, 40 steps,
   batch 4, accum 1, `α=0`, `τ=1`), and it is the obvious "you should have compared against this".
2. **The exclusion control with the twins added to `ℜ` and `ω` tuned.** Our own 0.031/1.000/0
   control is nearly saturated on disclosure, so the reviewer's real question is "what does your
   method add over the control with a tuned retain weight?" The answer must be on the composition,
   not on disclosure (§0.2(i)), and if it is not, the candidate has failed.
3. **SSD — the simple combination of my own ingredients.** Stranger-substituted teacher + soft-CE
   at teacher-forced `t ~ U(0,1)` + a scalar retain weight, no projection, no hazard set. This
   *already exists* as the control arm of the TD-SSD study (`FROZEN_ARMS.json`, `lam_T = 1`,
   adapters `llada_poc/adapters/ssd/ckpt_{50,150}`) and must be re-run at 128 facts under the
   current protocol. A reviewer who reads §A will say "this is SSD with extra steps"; the only
   defensible answer is the factorial of §F.1 with SSD as a cell.

**And the prior negative I must carry, unprompted.** TD-SSD — stranger teacher plus a
twin-difference surrogate `zmix = z_A − z_c + sg(z_c)`, whose gradient is `(J_A − J_c)ᵀe` — was run
on this project and **did not separate from its SSD control** (best point +0.0625 = 1 probe of 16,
worse at 150 steps) [M]. I am not re-proposing it, and the reason is mathematical, not rhetorical:
`(J_A − J_c)ᵀe` is **not a projection**. It subtracts the twin's Jacobian applied to the *same*
error vector, which neither guarantees nor even generally reduces the component of the update along
the twin's own gradient — it can leave it or amplify it. (B4) is the Euclidean projection onto
`{⟨v,u⟩ ≥ 0}` and guarantees `⟨v,u⟩ ≥ 0` by construction. That is a sharp operator-level difference
and it is the only reason I expect (B4) to behave differently. I also note the prior comparison was
underpowered (8 subjects, 16 probes, 1 seed) and **confounded** (the arms differed in two knobs:
the twin difference *and* the rotating anchor `lam_T`), so it does not settle the question — but it
does mean the burden is on us.

---

# 7. THE DECLARED FALLBACK

**IHP-cached.** Taken if job M3 returns `τ̄ ≥ 16` or a diffuse `τ`, or if on-policy collection
proves unstable (F5 sawtooth large).

Changes from the primary, and only these:

1. **States:** `fact_free_tf_states` (`mdmu/fxmix.py:130`) — teacher-forced at `t ~ U(eps,1)` with
   every answer position holding an `F` token **forced to mask**, which keeps μ on fact-free states
   (R4's truncation intent) without any trajectory collection.
2. **Index set:** all masked answer positions, weighted by the student's own confidence
   `conf_i^α` with `α = 1` — a cheap surrogate for the commit probability, and the point at which we
   are knowingly borrowing WGA's confidence weighting (PRIOR_ART: `w_i = p_θ^γ`), which makes this
   arm category (c) rather than (e).
3. **Target:** cached and `t`-independent, EMDU-FAM style — the reference is evaluated at
   `σ_S̃(c)` with the **answer span fully masked**, so the target depends only on `(fact, stranger)`
   and is exactly cacheable. Store top-`K = 512` log-probs plus a uniform smear of the residual
   mass: `64 × 32 × 512 × 8 B = 8.4 MB` per stranger draw. Refresh the stranger draw every `P`
   steps so §A.4's caching bias is bounded by the number of distinct draws rather than fixed.
4. Projection, constraints, retain families, optimizer, LoRA, selection rule: **unchanged.**

Cost: **3 forwards + 2 backwards per step**, no collection overhead — i.e. EMDU's forward cost.
Weaknesses, declared: it violates R4 (off-policy states); it reinstates §A.4's caching bias at the
level of the refresh period; and it gives up the 260× soft-target memory reduction, so the
chunked-KL mitigation from PRIOR_ART §2.2 becomes necessary again.

---

# 8. MEASUREMENTS REQUIRED, ORDERED, WITH KILL CRITERIA

All GPU jobs below are **requests for the coordinator to schedule**. Costs are estimates.

| # | job | cost | decides |
|---|---|---|---|
| **M1a** | Run the existing `grad_cosine.py` (adapter path only changed) on the **four already-trained baselines** `adapters_baselines/{emdu,mdu_t1,npo,sc_tf}`. Report same-answer vs different-answer coupling per arm, with each arm's disclosure alongside. | **≈12 GPU-min, zero training** | **Proposition B2 vs the `F`-indexing story.** If all four show ≈0.9, the causal account in Finding 1 §"completes a causal account" is not established by that measurement and the coordinator's screen is measuring suppression level. |
| **M1b** | At θ₀, per forget fact: `cos(δ)` (one **forward** per state, no backward — `δ = Wᵀe`) and `cos(a)` per layer, for **both** candidate targets (`r_ε` and `r̂*`); and `ρ_φ = cos(∇L_f(φ), ∇L_ret(ψ))`. | ≈15 GPU-min | **(B1)**: does `cos(δ)·cos(a)` reproduce the measured table? **(B3)**: is `cos(δ_ign) ≈ 0.82` vs `cos(δ_excl) ≈ 0.97`? **F6/F7**: is the projection a no-op or a forgetting-killer? **P1's predictor.** Answers the coordinator's question with **no training**. |
| **M2** | Build `P_ign`: `G^π_{θ0}(·|q[S←S̃])` over ≥25 strangers × 64 facts × 3 configs. Report `H`, modal mass, `MAR`, `Î_item`, `EC`. | ≈0.4 GPU-h | **F1/F2** — kills H3's premise and the whole target if `P_ign` is sharply moded or is a hedge. |
| **M3** | Commitment-distribution instrumentation (§C.3): `(τ, i_τ, conf_τ)` over 64 facts × 3 configs × 3 seeds. | ≈0.2 GPU-h | **F4** and the §C.4 cost table. `τ̄ ≥ 16` ⇒ take the fallback. |
| **M4** | Gate A4: one retain-only memorization SFT (`memorize_big.py` on the 64 retain facts) ⇒ `θ_ign^retrain`; then `TV(r̂*, p_{θ_ign^retrain})` and the entropy gap on the hazard index set. Also report `TV(r̂*, p_base)` and `w_min`'s distribution. | ≈0.5 GPU-h | **F3** — makes A4 a measured assumption instead of an **[A]**. |
| M5 | A3's own test, for the record: `TV(r_0|_{F^c}, P_ign)` — the formalization calls A3 "a testable assumption we should test before building on it". | folds into M2/M4 | whether the exclusion target was ever a valid counterfactual. |
| M6 | Non-GPU: build `𝒮_neu`; extend SPEX-128 with ψ₂ pairs for ≥20 entities and with ≥1 multi-token answer per relation (then a re-memorization run); `PLL_{θ0}` implementation. (`grad_cosine.py` already takes `--unlearn-adapter`, so M1a needs no code change.) | CPU + 1 re-memorization | R7, R10, R12. |

**Ordering rationale.** M1a and M1b together cost under half a GPU-hour, need no training, and can
falsify the mechanism story that motivates Part B *before* any objective is implemented. M2 can
kill the target. M3 decides primary vs fallback. Nothing should be trained before all four have
returned.

---

# 9. WHAT I COULD NOT MAKE WORK, AND WHY

Recorded as negatives because a well-argued infeasibility is worth more than a plausible objective.

1. **A target-only fix for the answer-token channel is impossible** (Proposition B5). The error
   direction's leading term is `Wᵀp_θ`, the model's own prediction, which equals `w_{v*}` while the
   model is memorized and is identical for every fact sharing the answer. No choice of `r` removes
   it; only a target with `r(v*) ≈ p_θ(v*)` would, and that target does not forget. This is why
   Part B is a constrained optimization and not a loss, and why I report the coordinator's
   prediction as one my own candidate fails.

2. **Cross-fact relearning transfer is probably irreducible by any objective of this family**
   (Proposition B2). The adversary's own one-hot target supplies the shared `−w_{v*}` direction at
   any suppressed model, so the 0.916 coupling and the 1.000-vs-0.514 split follow from
   *suppression plus a shared unembedding row*, not from how the suppression was achieved. The only
   remaining lever is `cos(a)` — the forward activations — which means deliberately pushing apart
   the representations of same-answer entities, i.e. **learning a gate** (degeneracy (e) by
   Corollary 2.4). I am not proposing that, and I do not have a construction that reduces
   relearning transfer without it. M1a decides whether this negative stands.

3. **Forget-forget orthogonalization is six times more expensive than retain-twin projection for a
   benefit that (2) says is unavailable.** The descent rate surviving a projection is
   `1 − cos²`: 93.2 % at the retain-twin cosine 0.260, **16 %** at the forget-forget cosine 0.916.
   Combined with (2), that closes the H1(a)/(c) direction on arithmetic rather than on taste.

4. **Mask-rate selection, the axis PRIOR_ART identifies as unoccupied everywhere, has little to
   offer here** (§0.2(ii)): under the frozen decoder the visited mask-rate marginal is already
   uniform on the grid, so `t ~ U(0,1)` matches it. What is mismatched is the conditional content at
   a given rate. I could not construct a mask-rate reweighting with a derivation behind it, and I
   declined to take the one-line change on the grounds that it is cheap and unoccupied.

5. **ψ₂ (same-subject) collateral cannot be measured in our data at all** — SPEX-128 gives each
   entity exactly one fact — so by Proposition 5.1 no objective built from our `(𝔉, ℜ)` constrains
   it and no evaluation would see it. §B.7's reverse-direction mechanism, which is the one genuinely
   MDLM-native route to an entity-indexed error vector, trades shared-answer coupling for exactly
   this unmeasurable damage. I recorded it and did not adopt it.

6. **No certificate is available at the resolution the design quantities are stated in.** `ε = 3.1e-4`
   and every fixed-point leak number are **design quantities**; a feasible battery certifies per-cell
   rates of order `1e-2` (Corollary 6.2). And since the control already sits at 0.031 inside the
   never-memorized band of 0.023–0.046, the honest headline is `ℛ`-inaccessibility against a
   declared recovery budget plus the composition vector — never a disclosure rate, and never
   "the fact is removed".

---

# 10. AMENDMENT 1 — the IGNORANT axis, Part B re-derived, R16 scoped

Written after the coordinator's θ₀ fp32 recomputation, the EMDU/MDU coupling control, and
`FINDING2_ignorant_axis.md`. Three CPU-only measurements of my own are reported here; they were
run on data already on disk and **no GPU job was run**.

## 10.0 Prop B2 confirmed; and the coupling screen measures the wrong thing

The coordinator's control settles it: EMDU (+0.773) and MDU-τ1 (+0.573), neither answer-token
indexed, both show the coupling, and in both forget-same-answer ≈ retain-same-answer. The fp32
recomputation at θ₀ gives forget 0.639 / retain 0.659 / forget-retain 0.620 — **all
indistinguishable**, which is exactly what §B.3 predicted (at θ₀ every fact is fit, so
`e_mem ≈ 0` for all of them and the residual's leading term is `∝ w_{v*}` with the same small
coefficient everywhere). Proposition B2 stands.

**One further derived consequence, which matters because it changes how the screen should be used.**
Write `δ_mem = −(1−p_θ(v*))·w_{v*} + m_θ`, `m_θ = Σ_{v≠v*} p_θ(v) w_v`. Both terms scale with
`(1−p_θ(v*))`, so the cosine is **independent of the suppression depth** and depends only on how
*spread* the residual prediction is: with `m_θ` a mixture over `j_eff` roughly isotropic rows,

> **(10.1)**  `cos(δ_mem, δ'_mem) ≈ j_eff / (j_eff + 2)`,  i.e.  `j_eff ≈ 2·cos/(1−cos)`.

Inverting the measurements: θ₀ (0.639) ⇒ `j_eff ≈ 3.5`; θ_u (0.916) ⇒ `j_eff ≈ 21.8`. So the
×1.43 amplification says only that **the model's residual prediction became more spread and more
generic** — which is precisely what an ignorance-matched model is supposed to do. And the
`selectivity` column is explained the same way: a *degenerate* arm's residual concentrates on a
single fact-independent token (`eos`), which is shared by **all** facts, so it raises the
different-answer coupling and collapses selectivity — measured 0.053 (ours, 0.000 degenerate) →
0.198 (EMDU, 1.000 degenerate) → 0.414 (MDU, 1.000 degenerate), selectivity 17.2× → 3.9× → 1.4×.

> **Prediction 10.A.** Same-answer coupling is **positively** associated with a generic, spread
> residual prediction, i.e. with the thing the IGNORANT cell rewards. On the matched re-run,
> selectivity should rank the arms by `(1 − DEGEN)`, and **IHP, if it succeeds on IGNORANT, is
> predicted to show same-answer coupling ≥ 0.90 and selectivity ≥ 15×** — i.e. to look *worse* than
> EMDU/MDU on the screen while being better on the outcome.
>
> **Therefore the coupling screen must not be used as a go/no-go on arms.** Used that way it
> selects for degeneracy. Its legitimate use is as the *selectivity ratio*, as a degeneracy
> diagnostic, reported next to DEGEN. This is a stronger form of the "Measure, Don't Optimize"
> caution: the correlate is not merely uninformative here, its sign is wrong.

## 10.1 Does the fall 0.620 → 0.261 undercut Part B? No — and I refuted my own first explanation

Three things, in the order I established them.

**(i) My first explanation was wrong, and I measured it wrong.** The single-shared-direction model
predicts `cos_fr ≈ 1/√((1+ρ_f)(1+ρ_r))` with `ρ = ‖private‖²/‖shared‖²`; from
`cos_ff = 0.916` and `cos_rr = 0.644` it predicts `cos_fr = √(0.916·0.644) = 0.768` against a
measured 0.261 — a deficit factor 0.340. I attributed the deficit to a forget-arm-specific shared
component, namely the replacement attractor (`Tbilisi ×19`), inferring `‖c_f‖ ≈ 2.8α`. **That is
refuted.** Partitioning the saved 128×128 cosine matrix (`outputs/grad_cosine_matrix_theta_u.npy`)
over different-answer, same-relation forget pairs by whether the two facts emit the *same* asserted
replacement:

| pair subset | n | mean cos | sd |
|---|---|---|---|
| **same** asserted replacement (includes the Tbilisi and seismoology sets) | 188 | **+0.0628** | 0.096 |
| **different** asserted replacement | 489 | **+0.0687** | 0.060 |
| same-answer control | 32 | +0.9159 | 0.067 |

Indistinguishable. The replacement direction is **not** a shared gradient component. CPU only, zero
GPU; reproducible via `scripts/repartition_ignorant.py` →
`outputs/repartition_ignorant.json`, from `grad_cosine_matrix_theta_u.npy` +
`spex/outputs/big_gens_uncached.json`.

**(ii) What the deficit then is, and why it is the opposite of protection.** With the δ-side
explanation closed, identity (B1) leaves only the activation term: the deficit must sit in
`cos(a_f, a_r) / cos(a_f, a_f') ≈ 0.34`. That is a **3× separation of forget-fact from retain-fact
activations, created by the edit** — the edit moved the forget facts' hidden states and left the
retain facts' alone. By Corollary 2.4 and Proposition 8.3 that is the signature of **degeneracy (e),
a learned subject-conditional readout**, which is exactly what a frozen output embedding forces.
So the fall `0.620 → 0.261` is not evidence that the exclusion target protects retain twins; on this
reading it is a **quantitative measurement of the gate** the target had to learn in order to look as
though it did. Stated as an inference, not a result: the alternative (private structure in `δ_r`
beyond the isotropic model) is not excluded, and **M1b decides it**, because M1b already measures
`cos(a)` per layer separately from `cos(δ)`.

**(iii) The statistic is the wrong one anyway, and the right one has the opposite sign.** Every
number in this family is a cosine between two *memorization* gradients. The damage statistic is
`cos(∇L_forget(φ), ∇L_ret(ψ))`, and the two objectives have **opposite sign structure** in the
answer-token coordinate: `δ_forget ≈ Wᵀ(onehot(v*) − r) ≈ +w_{v*}` while
`δ_ret ≈ Wᵀ(p_θ − onehot(v*)) ≈ −w_{v*}`. Hence

> **(10.2)**  `cos(∇L_forget(φ), ∇L_ret(ψ)) ≈ −cos(a_φ, a_ψ)`,  predicted `≈ −0.9` at θ₀.

`⟨v, u⟩ < 0` is the damage condition, so the predicted first-order damage is **large**, not small.
The reconciliation with the measured retain **1.000** is Corollary 2.2: retain accuracy is a *rank*
statistic and stays exactly 1.000 until the margin flips, so it cannot see margin erosion — and the
HISTORICAL measurement does see it (shared-object-token slice 0.832 vs 0.869 at matched leakage,
against retained means 0.905/0.919 [M]).

> **The Alice/Bob problem is therefore not smaller than assumed; it is invisible to a rank-valued
> metric.** The required instrument is the **margin**, `log p_θ(v_P|s_B) − log max_{v≠v_P} p_θ(v|s_B)`,
> i.e. P3's `λ` at logit level — which `EVAL_PROTOCOL` confirms no evaluator anywhere computes. If
> the matched re-run reports retain accuracy only, it will again report no damage whether or not
> there is any.

## 10.2 Part B, revised: a declared damage budget instead of a hard constraint

If (10.2) is right, the hard constraint `⟨v_j, u⟩ ≥ 0` removes `cos² ≈ 0.81` of the forget descent
rate and is unusable. §B.5's "93.2 % survives" was computed from the wrong cosine (0.260) and is
withdrawn. Replace the constraint with a **declared damage budget** `κ ∈ [0,1]`:

> **(10.3)**  `u_t = argmin_u ½‖u − g_t‖²`  s.t.  `⟨v_j, u⟩ ≥ −κ·‖v_j‖·‖g_t‖`,  `j = 1..J`.

`‖g_t‖` is a constant within the step, so the constraint stays **linear** in `u` and the same NNLS
solver applies. Closed form at `J = 1` with `cos(g,v) = −c`, `c > 0`: the constraint is active iff
`c > κ`, and then

> **(10.4)**  `u = g + ((c−κ)‖g‖/‖v‖)·v`,  **retained forget descent `⟨g,u⟩/‖g‖² = 1 − c(c−κ)`**,
> first-order twin damage `ΔL_ret ≤ η·κ·‖v‖·‖g‖`.

| `c` (to be measured, M1b) | `κ = 0` | `κ = 0.45` | `κ = c` |
|---|---|---|---|
| 0.26 | 0.932 | no-op | no-op |
| 0.60 | 0.640 | 0.910 | 1.000 |
| 0.90 | **0.190** | 0.595 | 1.000 |

**This is the honest restatement of Part B's claim, and it is a better one than the original.** The
projection's value is *not* free damage removal. It is that it converts the forgetting/preservation
exchange rate from an untunable scalar loss weight `ω` — which condition C2 says is "not derivable
a priori" — into a **declared budget `κ` with the closed-form price `1 − c(c−κ)`**, measured before
training rather than swept after it. M1b therefore does not merely gate Part B, it **sets `κ`**:
choose the smallest `κ` with `1 − c(c−κ) ≥ 0.5`, i.e. `κ ≥ c − 0.5/c`, which is `κ = 0` for
`c ≤ 0.707` and `κ ≈ 0.344` at `c = 0.9`.

## 10.3 (a) The primary claim, re-stated against IGNORANT — and a correction to that rate

### 10.3.1 The measured rate is wrong, and the headroom is larger than reported

`FINDING2` scores REPLACE by membership in the closed 32-answer pool. On the exclusion control's own
64 forget generations (`big_gens_uncached.json`, train question, frozen decoder) I parsed the
*asserted* value rather than pool membership. CPU only, zero GPU:

| cell | n | rate | `FINDING2` |
|---|---|---|---|
| DISCLOSE | 2 | 0.031 | 0.062 |
| asserts a type-correct alternative **in** the pool | 28 | 0.438 | counted REPLACE (0.500) |
| asserts a type-correct alternative **outside** the pool | 34 | **0.531** | **counted IGNORANT** |
| non-committal (asserts nothing) | **0** | **0.000** | — |

Reproducible: `scripts/repartition_ignorant.py` → `outputs/repartition_ignorant.json`. Parse
caveat, stated because it is a hand-written regex over three known templates: the type-gazetteer
version (§4(d)) is the real instrument and is job M8; this is an indicative lower bound on the
assertion rate, since anything the regex misses is counted non-committal and it currently misses
nothing.

Off-pool examples: `Montréal`, `Tashkent`, `Accra`, `tuba`, `oboes`, `bassoons`, and — all 16
`field` probes, which my first regex missed entirely — `seismoology ×8`, `mycological ×4`,
`hydrodynamics ×2`. **The exclusion control's true IGNORANT rate is ≈ 0.000, not 0.438**, and the
axis's headroom is ≈ **1.000, not 0.562**. The direction of `FINDING2`'s argument is strengthened;
only its ceiling-distance is wrong.

Two further defects this exposes, both required fixes before the matched re-run:

1. **REPLACE must be defined by type-correctness + mode stability, not pool membership** — the
   formalization already specifies it that way (§4(d): ignorance matching, `MAR`, type/arity
   well-formedness). Pool membership makes an off-pool confident falsehood score as the *desirable*
   cell. The coordinator's caveat anticipates this for open-pool RWKU; it is already biting on the
   closed-pool synthetic set.
2. **Several off-pool assertions are fluent non-words** — `seismoology` (×8), `harpa`, `clarion`,
   `tympani`, `Troms`, `Tinn`. These are §4(b)'s known hole: `malformed` fires only when there is
   *no* alphabetic token, so fluent real-word gibberish scores clean. They belong in DEGENERATE, not
   IGNORANT, and they are ~8–12 of 64 here.

### 10.3.2 What IHP predicts, cell by cell — and the result that makes it conditional

> **Proposition A1 [P].** At `T = 0` the sampler reads the denoiser only through
> `(argmax_v p_θ, max_v p_θ)` (Lemma 2). A per-position objective matched exactly to a target `r`
> therefore produces the object `argmax_v r(v|s,i)` — a deterministic functional of the target.
> Since `r̂*` is the reference's prediction at an **unknown-subject** state, it depends on the
> relation template and not on the subject, so `argmax_v r̂*` is **one value per relation**. Hence a
> perfectly-trained IHP at `T = 0` collapses onto one answer per relation, exactly as the exclusion
> target collapses onto `argmax_{v∉F} p₀`. **Ignorance matching does not escape degeneracy (d) at
> `T = 0`; it only changes which token the attractor is.** ∎

This is measured, not speculative: the control's attractors are **per-relation**, exactly as A1
requires — `home_city → Tbilisi` 19/32 (0.594 share, `H = 2.290` of 5 bits),
`field → seismoology` 8/16 (0.500, `H = 2.250`), `instrument → timpani` 3/14 (0.214, `H = 2.896`);
overall 26 distinct values, `H = 3.898` bits, top-1 share 0.297 of all 64 probes.

Worst case for IHP under A1 is therefore **worse than the control**: one mode per relation gives a
top-1 share of `32/64 = 0.500` (home_city alone) against the control's 0.297, and a modal histogram
entropy of `1.5` bits against 3.898. I state that prediction because it is what the derivation gives.

**Predictions per cell, conditioned on what M2 returns about `P_ign`:**

| regime (M2 decides) | DISCLOSE | DEGENERATE | REPLACE | IGNORANT |
|---|---|---|---|---|
| **(i) `P_ign` hedges with content** (my expectation, ~0.5 prior) | 0.02–0.06 | 0.00 *under a reference-relative refusal rule*; **up to 1.00 under the current 9-phrase list** | ≤0.10 | **0.85–0.95**, or **≈0.00** as currently scored |
| **(ii) `P_ign` is a confident off-pool guess** | 0.02–0.06 | 0.00 | ≈0.00 *as defined* (false negative) / **≈0.9 corrected** | ≈0.9 *as defined* / **≈0.0 corrected** |
| **(iii) `P_ign` is a confident in-pool guess (F1 fires)** | 0.02–0.06 | 0.00 | **≥0.50, worse than the control** | ≤0.40 |

- **DISCLOSE:** `P_ign`'s own rate, the never-memorized floor **0.023–0.046**. No improvement is
  claimed, and §0.2(i) says none is available.
- **DEGENERATE:** the target is a real, globally unmodified model's own output distribution, so it
  is in-support for fluency by construction — this is §7.1's remark that the ignorance form "already
  buys" (a) and (b). Predicted **0.00–0.05**, against MDU/EMDU's 1.000. **But** if `P_ign` hedges,
  the current fixed 9-phrase refusal list (`common_big.py:65-67`, which contains `"i do not know"`
  and `"no information"`) will score a perfectly ignorance-matched model as DEGENERATE at up to
  1.000. **Required fix, which the formalization already specifies (§4(a)):** a refusal is
  *item-invariant*, so score it by `Î_item` with the bar set by **θ₀'s own behaviour on the
  unknown-subject panel** — "including the case where the reference's own honest behaviour for
  unknowns *is* a hedge". Without this fix the matched re-run will discard IHP for the wrong reason.
- **REPLACE / IGNORANT:** the whole claim, and it is **entirely conditional on M2**. In regime (iii)
  the candidate is dead (§10.4).

### 10.3.3 The instrument has to change, and that is on me

Proposition A1 has a methodological consequence I did not state in §E.5: **my own P2 prediction —
modal-answer histogram entropy — is not testable at `T = 0` with one draw per fact**, because a
greedy decoder emits only `argmax r`, so the histogram measures the argmax map, not the
distribution. The prediction must be re-stated on instruments that can see a distribution:

1. **Logit-level, primary.** At the object position of each forget probe, read `Ĝ_O = p_θ(·|s,i)`
   and compare to `P_ign`: `TV(Ĝ_O, P_ign) ≤ τ_ign` and `|H(Ĝ_O) − H(P_ign)| ≤ τ_H`. One forward per
   probe. No evaluator in any of our five stacks reads logits (`EVAL_PROTOCOL`), so this must be
   built. Predicted separator: the control's spread is §8.2's closed form
   `H(r_0) = [H(p₀) − H_b(q) − q·H(p₀|_F)]/(1−q)`, a *different and computable* quantity from
   `H(P_ign)`; `TV(r_0|_{F^c}, P_ign)` is the formalization's own A3 test (job M5) and is the
   pre-registered discriminator.
2. **`MAR`, mode agreement across `(seed, π)` draws at `T = 1`.** Control predicted `MAR ≈ 1` with
   `mode ∉ Σ(O)` (the `𝒟_d` signature); IHP predicted `MAR ≈ MAR(P_ign)`. This is the only
   generation-level instrument that distinguishes "matched a spread distribution" from "memorized a
   replacement", and it requires `T > 0` and ≥2 draws per probe.
3. **Per-relation modal share** as the cheap proxy: control 0.594 / 0.500 / 0.214 by relation;
   IHP must beat those or Proposition A1 has bitten.

## 10.4 (b) F1 is the central risk — what I expect, and the ~4-GPU-minute decider

**What I expect `P_ign` to be, with reasoning.** For a *fictional* subject the reference has never
seen, an instruct-tuned model does one of three things, and our own data says which is most likely
*for this model on these prompts*: the control's outputs at suppressed forget states are **confident,
fluent, type-correct and attractor-collapsed** (`Tbilisi`, `seismoology`), and the *base* model's
behaviour on these same fictional entities is what produced the 0.023–0.046 floor — a rate low
enough that it is not naming the right city, but we have **never looked at what it names instead**.
My prior: **~0.5 on regime (i)** (hedge-with-content, because these are chat-templated questions
about a person with no referent), **~0.3 on regime (ii)** (confident off-pool confabulation),
**~0.2 on regime (iii)** (confident in-pool, F1 fires). Regime (ii) is the awkward one: it passes
the partition and fails the construct.

**The cheapest decider is a single-position logit probe, not generation.** Generation at 2 s/probe
over 64 facts × 8 strangers would be ~16 GPU-min and would only show the argmax (Prop. A1). Instead:

> **Job M2′ — F1 decider.** Build the canvas `[chat(q[S←S̃])] ‖ "⟨S̃⟩ lives in ⟨mask⟩."` and read
> `p_{θ0}` at the single mask position. One forward per `(fact, stranger)`; 64 facts × 8 strangers
> = 512 forwards, batched, at `L ≈ 48`. **≈ 3–4 GPU-minutes.** Report per relation:
> `H(P_ign)`, the top-5 with masses, the modal token's share, whether the mode is **in the
> 32-answer pool**, whether the mode is a **type-correct value at all** (vs a hedge opener), and
> `TV` between strangers (which also gives `σ²_priv` for (B3) and the variance figure of §A.4).
> **The same forward gives `r_0 = argmax_{v∉F} p₀` and `H(r_0)` for free**, so job M5 (the A3 test,
> `TV(r_0|_{F^c}, P_ign)`) folds in at zero extra cost. Add the 64 base-model generations at the
> original prompts (`P_ign^base`, free because the entities are fictional) for ≈ 2 more GPU-minutes,
> as the independent check that stranger substitution and base-model ignorance agree.
>
> **Kill rule, pre-registered:** if `mode(P_ign)` is a pool member for ≥2 of 3 relations **and**
> the modal share exceeds the control's per-relation share (0.594 / 0.500 / 0.214), then
> **F1 has fired, IHP's REPLACE cell is no better than the control's, and the candidate is dead** —
> for ~4 GPU-minutes, as asked. The fallback in §7 does not rescue it, because F1 is a property of
> the *target*, not of the state distribution: **if F1 fires, both IHP and IHP-cached die together**
> and the project needs a different target family (the §B.7 reverse-direction construction is the
> first candidate, with its declared data prerequisite).
>
> **Partial-failure branch, worth naming:** if `H(P_ign)` is adequate but its mode is a pool member,
> IHP survives *only* in a stochastic decoding family (`T > 0`), where the mass condition rather
> than the rank condition governs (Lemma 4) and the spread is actually sampled. That would be a
> restriction on `Π` and would have to be declared in the claim (R1, R14).

## 10.5 (c) R16 — confirmed scoped out, with one by-product prediction that costs no operation

**Confirmed.** The candidate is scoped to the composition trade-off only. The reason is now stronger
than "no mechanism": the coordinator's own EMDU/MDU control plus Proposition B2 establish that the
shared relearning direction is supplied by **the adversary's one-hot objective at any suppressed
model**, so it is not a quantity our target choice controls. Under LoRA any relearning number is a
lower bound anyway (three independent sources agree parameter-subset unlearning does not survive
fine-tuning), so the gap could not be attributed to the method even if it moved.

**What I will not add.** SAM is the only smoothing defense PRIOR_ART endorses (Fan et al.'s GP is a
released no-op: `.item()` detaches at `GA.py:258/340`), but (a) PRIOR_ART judges gradient-norm-style
defenses **category (a)/(b)** vs Fan et al. Eq. 10 — "this direction is closed" — (b) our own
gradient-flattening arm failed its pre-registered gate (Δ = −0.023 against ≤ −0.20 while `‖∇L_mem‖`
fell 38–40 %), and (c) it is a second distinguishing operation and would break the attribution rule.
Declined on all three grounds.

**The one thing I will claim, because it is a by-product and not an operation.**

> **Prediction 10.B.** Post-RTT recovery should be **monotone in the pre-attack REPLACE rate**
> across arms. Mechanism: a replacement is an installed competing binding `S → v'`; the adversary
> has only to flip an existing edge, and at a replaced model its error direction
> `Wᵀ(onehot(v') − onehot(v*)) = w_{v'} − w_{v*}` is a clean two-term direction in which *both*
> terms are shared across same-answer facts. At an ignorance-matched model the error is
> `m_π − w_{v*}` with `‖m_π‖ ≈ ‖w‖/√j_eff`, so there is no cooperating `+w_{v'}` term and no
> installed edge to flip.
>
> **Caveat, stated because I just refuted the adjacent claim:** §10.1(i) measured no shared
> gradient component along the replacement direction at θ_u (0.0611 vs 0.0639), which is evidence
> *against* the "shared `w_{v'}` term" half of this mechanism. Prediction 10.B therefore rests on
> the "installed edge" half only, and it is weaker than I would like. It is testable **for free** on
> arms we already have: the exclusion arm (REPLACE ≈ 0.969 corrected) and ARC-K512 (0.594 in-pool)
> have different replacement rates and `anvil` holds a recovery curve for one of them. If the
> correlation is absent, R16 stays unaddressed and I claim nothing on it.

## 10.6 Amendments to the measurement list

| # | change |
|---|---|
| **M1a** | **done by the coordinator. Prop B2 confirmed.** Add: run it on `sc_tf` and `npo` too, since those are the two non-degenerate non-`F`-indexed arms and are the discriminating cells. |
| **M1b** | **promoted to the single Part B gate, and its output is now `κ`, not a yes/no.** Must report `cos(δ)` and `cos(a)` *separately* (so (B1) is checked and §10.1(ii)'s gate inference is decided) and must measure `c = |cos(∇L_forget(φ), ∇L_ret(ψ))|` at θ₀ for both target families. Then `κ = max(0, c − 0.5/c)`. |
| **M2′** | **replaces M2 as written.** Single-position logit probe, ≈4 GPU-min, with the pre-registered kill rule of §10.4; absorbs M5. Run **first**. |
| **M3** | unchanged; still decides primary vs fallback on cost. |
| **M4** | unchanged (Gate A4), but note it is now *downstream* of M2′: if F1 fires there is no target to validate. |
| **M7 (new)** | **Margin instrument.** `λ(s_B) = log p_θ(v_P|s_B) − log max_{v≠v_P} p_θ(v|s_B)` on the shared-answer retain twins, plus the `𝒮_neu` selectivity contrast `SEL`. Without it, retain damage is invisible at 1.000 (§10.1(iii)). One forward per twin probe. |
| **M8 (new, CPU)** | Re-score every existing arm's generations with (i) type-correctness instead of pool membership, (ii) a fluent-non-word detector, (iii) `Î_item` against the unknown-subject panel. **Zero GPU, and it changes the primary metric of the matched re-run.** My partial version is in §10.3.1. |

Ordering: **M2′ → M1b → M8 (CPU, in parallel) → M3 → M4 → M7.** M2′ and M1b together are under
10 GPU-minutes and can kill the candidate or set its only free parameter before any arm is trained.
