# FORMALIZATION — machine unlearning for masked diffusion LMs (LLaDA-8B-Instruct)

Formalization lead. First-principles specification, written before technique selection. No method is
assumed; §7 derives the objective form and §8 audits the one object we have already used.

Status labels used throughout: **[P]** proved here; **[A]** assumption (stated, not proved); **[H]**
hypothesis (requires measurement); **[M]** measured (prior evidence, cited with its caveats).

Running example, used in every section:

> Forget φ = (Alice, lives-in, Paris). Preserve: ψ₁ = (Bob, lives-in, Paris) [shared object token
> **and** shared relation template]; ψ₂ = (Alice, plays, cello) [shared subject]; ψ₃ = ordinary use
> of the word "Paris"; ψ₄ = general capability.

---

## 0. Notation

| symbol | meaning |
|---|---|
| `V` | vocabulary, `|V| = 126 464`; `m = 126 336` (mask), `e = 126 081` (eos); `V̄ = V ∪ {m}` |
| `c ∈ V*` | prompt (chat-templated), `L` | answer-canvas length (`gen_length`) |
| `z ∈ V̄^L` | answer canvas; `s = (c, z) ∈ 𝕊` a **state**; `M(s) = {i : z_i = m}` |
| `p_θ(· | s, i) ∈ Δ(V)` | the **denoiser** (mask predictor) at masked position `i`; bidirectional — depends on all of `z`, not a prefix |
| `D_θ` | the denoising family `(s,i) ↦ p_θ(·|s,i)`. **This is the trainable object.** |
| `π` | a decoding configuration `(N steps, B block_length, remask ∈ {lc, rnd}, T, cfg)`; `Π` a declared family |
| `G^π_θ(· | c) ∈ Δ(V^L)` | the law of the **generated canvas**. **This is the observable object.** |
| `θ₀` | the frozen reference (LLaDA-8B-Instruct, possibly after a memorization SFT) |

**The sampler, stated exactly** (this is the object everything in §1 turns on). Blocks
`B_1,…,B_{L/B}` are filled left to right. Inside block `b`, at step `n` from state `s_n`:

1. evaluate `p_θ(·|s_n,i)` for every `i ∈ M(s_n)`;
2. draw `x̂_i ~ π_T(·|s_n,i)` by Gumbel-max at temperature `T` (`T = 0` ⇒ `x̂_i = argmax_v p_θ(v|s_n,i)`);
3. confidences: `lc` ⇒ `conf_i = p_θ(x̂_i|s_n,i)`; `rnd` ⇒ `conf_i ~ U(0,1)` i.i.d.;
4. eligible set `E_n = M(s_n) ∩ B_b`; commit set `C_n = top-k_n of E_n by conf_i`;
5. `z_{n+1,i} = x̂_i` for `i ∈ C_n`, unchanged elsewhere. **Committed tokens are never re-masked.**

The chain is absorbing; the terminal canvas `y ∈ V^L` has law `G^π_θ(·|c)`.

**There is no sequence likelihood.** `G^π_θ` is *not* `∏_i p_θ(y_i | …)` for any ordering: the commit
order is data-dependent (step 4 reads the denoiser), commits are made in parallel from a product of
marginals (step 5), and the denoiser is re-evaluated on a canvas it partly wrote. Everything below is
built to respect this; any construction that silently reintroduces `log p(y|c)` is invalid for LLaDA.

---

## 0.1 The denoiser–sampler gap (five lemmas; the whole document rests on these)

**Reachable set.** `R(θ,π,c) = {s : Pr[s_n = s for some n] > 0}` under `π` from the all-mask canvas.

**Lemma 1 (reachability sufficiency). [P]** If `p_θ = p_{θ'}` on `{(s,i) : s ∈ R(θ,π,c), i ∈ M(s)}`
then `R(θ',π,c) = R(θ,π,c)` and `G^π_θ(·|c) = G^π_{θ'}(·|c)`.
*Proof.* Induction on `n`. Steps 1–5 read `θ` only through `{p_θ(·|s_n,i)}_{i∈M(s_n)}`. Equal inputs
give an identical transition kernel at `s_n`, hence an identical law and identical support at `n+1`. ∎

**Corollary 1.1 (prompt-disjointness). [P]** States carry the *entire* prompt, so for `c ≠ c'`,
`R(θ,π,c) ∩ R(θ,π,c') = ∅`. Consequently, in an unrestricted (per-state tabular) model class, the
denoiser may be modified at one prompt's states with *exactly zero* effect at any other prompt. This
is the formal source of both the good news in §8.1 and the impossibility in §6.1.

**Lemma 2 (non-identifiability; no converse to Lemma 1). [P]** `G^π_θ = G^π_{θ'}` does not imply
`D_θ = D_{θ'}` on `R`. At `T = 0` with `lc` remasking, steps 2–4 read `p_θ(·|s,i)` only through the
two numbers `(argmax_v p_θ, max_v p_θ)`. Any `p'` matching those at every `(s,i)` gives the identical
sampler. The sampler is therefore a projection of a `(|V|−1)`-dimensional simplex onto 2 dimensions
per position: massively non-injective. ∎
*Consequence.* Output-level objectives under-determine the denoiser; denoiser-level objectives
over-determine it. Neither is the "right" object, and §7 must say which direction the slack goes.

**Lemma 3 (simulation bridge, stochastic configs). [P, `rnd` remasking, `T = 1`]**
`TV(G^π_θ, G^π_{θ'}) ≤ Σ_{n=1}^N 𝔼_{s_n}[ Σ_{i∈E_n} TV(p_θ(·|s_n,i), p_{θ'}(·|s_n,i)) ]`.
*Proof.* Couple the per-position draws and the `U(0,1)` confidences; under `rnd` the confidences are
θ-independent, so the step outputs agree whenever all draws agree; hybrid argument over `n`. ∎
(For `lc` the bound fails as stated, because `conf_i` itself depends on `θ`: an arbitrarily small
change in the denoiser can permute the commit order. See Lemma 5.)

**Corollary 3.1 (the bridge is quantitatively vacuous). [P]** With Pinsker,
`TV(G,G') ≤ Σ_n Σ_{i∈E_n} √(KL_i/2)`. At `gen_length = 128`, `block = 32`, `128` steps the eligible
sum is `Σ_n |E_n| = 2112`. Requiring `TV ≤ 0.05` forces a mean per-position `TV ≤ 2.37 × 10⁻⁵`, i.e.
**mean per-position `KL ≤ 1.12 × 10⁻⁹`**. No training run attains that.
*Therefore: a per-position divergence objective can never be justified as a bound on the output
distribution.* The only usable transfer is a union bound on the **specific disclosure event**, not on
the distribution. This is a hard requirement on objective design (R6).

**Lemma 4 (greedy decoding is a rank condition; stochastic decoding is a mass condition). [P]**
At `T = 0`, position `i` emits an `F`-token at step `n` iff `argmax_v p_θ(v|s_n,i) ∈ F` and `i ∈ C_n`.
Hence `Pr[disclosure] = 0` **iff** at every reachable `F`-free state and eligible `i`:
`max_{v∈F} p_θ(v|s,i) < max_{v∉F} p_θ(v|s,i)`. A sufficient mass condition is
`p_θ(F|s,i) < max_v p_θ(v|s,i)`, and since `max_v p ≥ 1/|V|`, `p_θ(F|s,i) < 1/|V| = 7.91 × 10⁻⁶`
suffices unconditionally. For `T > 0` no rank condition suffices and the floor is
`1 − (1−ε)^{n̂}` over `n̂` eligible commits at mass `ε`. ∎
*Consequence.* Greedy and stochastic configs demand **structurally different constraints** (margin vs
mass). A single KL term is a poor instrument for both: it buys mass reduction far past the rank
flip for greedy, and buys rank flips that do not control mass for `T = 1`.

**Lemma 5 (confidence displacement; MDLM-specific). [P for the ordering, [H] for the consequence]**
Under `lc` remasking at `T = 0`, `conf(s,i) = max_v p_θ(v|s,i)`. Any edit that moves mass off `F` at a
position where `argmax ∈ F` strictly lowers `conf(s,i)` unless it raises the off-`F` maximum above the
old maximum. For the exclusion target of §8 this is explicit: `conf` falls from `p₀(v*)` to
`(1−ε)·max_{v∉F} p₀(v)/(1−p₀(F))`, which is `< p₀(v*)` whenever
`max_{v∉F} p₀(v) < p₀(F)(1−p₀(F))/(1−ε)` — generically true when the fact is strongly known
(e.g. `p₀(F) = 0.9`, runner-up `0.03` ⇒ `conf: 0.90 → 0.30`). The fact position therefore **loses its
place in the commit order** and is committed later, conditioning on *more* model-written context.
**[H]** That later context can re-entail the fact (the model first commits "…lives in the capital of
France…", after which the object position is forced), so suppression can *displace* disclosure rather
than remove it; the hazard at the later decision state may be higher than at the original one.
*Consequence.* The edit changes the state distribution it is trained on. The constraint set is
self-referential (`μ` depends on `θ`), so matching a target on "the states the sampler visits" is a
**fixed-point condition**, not a supervised constraint. Any objective must either iterate (on-policy
re-collection) or declare the staleness it tolerates.

---

## 1. The object being removed

### 1.1 Elimination of the wrong types

- **(O1) A parameter-space region** `θ ∈ Θ_forget`. Rejected: not identifiable (Lemma 2, plus the
  usual permutation/scale gauge symmetries), and has no observable consequence on its own. A
  parameter change is the *mechanism*, never the *specification*.
- **(O2) A conditional distribution `p(a|q)`.** Rejected: **it does not exist** for an MDLM (§0). The
  nearest existing object is `G^π_θ(·|q)`, which is indexed by `π`. Consequence: *forgetting is not a
  property of `θ`; it is a property of the pair `(θ, Π)`.* Any claim must name `Π`.
- **(O3) A set of `(query, answer)` pairs.** Too coarse: it says what must stop, never what must
  take its place, and so admits every degeneracy in §4.
- **(O4) An association between two token sets `A ↔ B`.** This is the folk notion of "a fact", but it
  is *prompt-blind*, and prompt-blind suppression provably destroys ψ₁ (Theorem 2.1). Rejected.
- **(O5) A property of the generative sampler.** Correct type — but it must be a **conditional**
  property (a property of the *map* `q ↦ G^π_θ(·|q)`), not of any single distribution.

### 1.2 Definitions

**Definition 1.1 (fact, surfaces, query class).** A fact `φ = (S, R, O)`. Its object surface set
`Σ(O) ⊂ V*` is the set of token sequences realizing `O` (tokenization variants, aliases,
capitalizations, translations); `F = F(φ) ⊂ V` is the set of token ids occurring in `Σ(O)`.
`Q_φ ⊂ V*` is the set of prompts whose correct answer is `O` by relation `R` about `S`.
`𝒪` is the type-compatible candidate set for `R` (cities, for `lives-in`).

**Definition 1.2 (disclosure, as adversary advantage).** Token containment is the wrong primitive
(§5.2, §5.6). Fix an adversary class `𝒜`, a query measure `ν_φ` on `Q_φ`, and `π ∈ Π`. Define

> `Adv_𝒜(θ; φ, π) = sup_{A∈𝒜} [ Pr_{q∼ν_φ, y∼G^π_θ(·|q)}[A(q,y) = O] − Pr[A₀ = O] ]`

where `A₀` is the best guess from the type prior on `𝒪` alone (so an uninformative output scores 0).
Whole-token span containment, the metric implemented in every current workspace, is the special case
`𝒜 = {read-off}`. The graded, multi-token and endorsement cases are instances of Def. 1.2 with larger
`𝒜` (§5.2, §5.6); so is relearning (§6.3), with `𝒜` closed under fine-tuning.

**Definition 1.3 (the forget object).** The object removed is the **restriction of the conditional
disclosure map to `Q_φ`**, with everything outside a declared closure pinned. The specification is an
operator on behaviour maps:

> `Forget(θ₀; φ, Π, 𝒜, δ, η) = { θ :`
> **(F)** `sup_{π∈Π} Adv_𝒜(θ; φ, π) ≤ δ`,
> **(P)** `∀ q ∉ cl(φ), ∀ π ∈ Π : d(G^π_θ(·|q), G^π_{θ₀}(·|q)) ≤ η`,
> **(W)** `θ` satisfies the well-formedness and non-degeneracy constraints of §4 `}`.

Three consequences, each load-bearing:

1. **The object is a pair.** Neither (F) nor (P) alone is the thing being removed. "Remove the
   association" and "change nothing else" are one specification; separating them is what produces
   every degenerate solution in §4.
2. **It is a statement about a map, not a distribution.** No finite sample from a single query
   distribution can define it, and (§6.1) no finite sample can certify it.
3. **Trainable ≠ measurable.** (F) and (P) are conditions on `G^π_θ`; training touches `D_θ`. Lemma 1
   says agreement of `D_θ` on the *reachable set* suffices for (F)/(P); Corollary 3.1 says the
   quantitative version of that bridge is vacuous; Lemma 5 says the reachable set moves as we train.
   **The gap is therefore not an approximation error but a change of problem type: from a supervised
   constraint to a fixed-point constraint on a self-generated state distribution.**

**Counterexample to my own Definition 1.3.** Def. 1.3 is satisfied equally by (i) a model that never
learned `φ`, and (ii) a model that detects `Q_φ` and routes around it (degeneracy (e)) — provided `𝒜`
cannot probe off `ν_φ`. It is also satisfied by a model that confidently answers "Tbilisi"
(degeneracy (d)), since that output gives the read-off adversary no advantage about `O`. Def. 1.3 is
**necessary and not sufficient**; §4 supplies the missing constraints, and Theorem 4.5 shows the
strict form of (P) actually *forces* (ii).

---

## 2. The preservation requirement

### 2.1 Constraints

Let `ℜ` be the declared retain set, `𝒮_neu` a declared set of *neutral* states (contexts where
`F`-tokens are the legitimate continuation: "the capital of France is ___", "___ Hilton", "Treaty of
___"), and `Q_φ^rev` the reverse-direction queries ("who lives in Paris?").

- **P1 (retain disclosure).** `∀(q,a) ∈ ℜ, ∀π ∈ Π : Pr_{G^π_θ}[a disclosed] ≥ (1−κ)·Pr_{G^π_{θ₀}}[·]`.
- **P2 (behavioural locality).** `∀q` in a declared neutral query set: `d(G^π_θ, G^π_{θ₀}) ≤ η`,
  with `d` a computable surrogate (§4(b)) since `TV` on `Δ(V^L)` is not estimable.
- **P3 (vocabulary neutrality).** `λ(𝒮_neu) := 𝔼_{(s,i)∼𝒮_neu}[ log p_θ(F|s,i) − log p_{θ₀}(F|s,i) ] ≈ 0`.
  This is a **logit-level** constraint and requires white-box evaluation.
- **P4 (capability).** Benchmark deltas within tolerance (MMLU panel, copy/quotation accuracy).
- **P5 (reverse direction).** On `q ∈ Q_φ^rev`, `S = Alice` must drop out of the answer *set* while
  the other members (Bob) survive. Note this is a constraint on a set-valued output and inherits
  Theorem 5.5.

### 2.2 Why ψ₁ (Bob) is the hard case, proved

ψ₁ shares with φ **the object token(s) `F`** and **the relation template**. The two states
`s_A = ("Where does Alice live?", z)` and `s_B = ("Where does Bob live?", z)` differ *only* in the
subject span.

**Theorem 2.1 (prompt-blind suppression is incompatible with P1 under a shared object token). [P]**
Let `v* ∈ F` be required by ψ₁'s answer. Suppose the realized model satisfies
`p_θ(F|s,i) ≤ ε` for all `s ∈ R(θ,π,q_B)` and all eligible `i`. Then

- (i) **`T = 0`:** if `ε < 1/|V| = 7.91 × 10⁻⁶`, then `max_{v∈F} p_θ ≤ ε < 1/|V| ≤ max_v p_θ`, so
  `argmax ∉ F` at every eligible position, and by Lemma 4 ψ₁'s disclosure is **exactly 0**.
- (ii) **`T = 1`, `rnd`, one commit per step:** the per-step hazard is
  `𝔼[Σ_{i∈C_n} p_θ(F|s_n,i)] ≤ k_n ε`; a union bound over steps gives
  `Pr[v* ever committed] ≤ (Σ_n k_n) ε = L ε`.

*Proof.* (i) is Lemma 4. (ii) is the first-passage union bound; `Σ_n k_n = L` since every position is
committed exactly once. ∎

**Corollary 2.2 (the cliff). [P]** At `|V| = 126 464`, `L = 128`: `ε = 10⁻⁵` gives ψ₁ ≤ `1.28 × 10⁻³`
under `T = 1` and, because `10⁻⁵ > 1/|V|` only by a factor 1.26, puts the greedy outcome *on a knife
edge* — any transfer that pushes `p_θ(F|s_B,i)` below `7.91 × 10⁻⁶` zeroes ψ₁ discontinuously.
**Prediction [H]:** the shared-object-token retain slice should be **bimodal across items**, not
gradually degraded. An 8-item mean (the current `shared_answer_subset`) will therefore under-report
it; the slice must be reported per item with a count of zeroed items.

So: *any edit expressible purely as a bound on the object-token mass, applied to a state set that
reaches ψ₁, destroys ψ₁.* The prompt-blind notion (O4) is exactly such an edit. This is the formal
content of "an edit expressed purely in answer-token probability must damage Bob", and its scope is
precisely "applied to a state set that reaches ψ₁" — which §8.1 shows is avoidable in principle and
§8.3 shows is not avoided in practice.

### 2.3 The minimal sufficient conditioning information

Write `vis(s) = (c, {(j, z_j) : z_j ≠ m})` for the visible context at state `s` — note that in a
bidirectional MDLM this includes *already-committed answer tokens*, at every position, in both
directions. We need a statistic `σ : 𝕊 → [0,1]` such that the target at `(s,i)` depends on `s` only
through `σ` (and `vis`-measurable features), with `σ(s_A) = 1`, `σ(s_B) = 0`.

**Proposition 2.3 (what `σ` must read). [P]** No `σ` measurable with respect to (relation template,
object-token identity, answer-position index) separates `s_A` from `s_B`: the two states agree on all
three. Hence `σ` must depend on the **subject span in the visible context**, and the minimal
sufficient statistic is the pair *(subject identity, relation identity)*. The answer region's content
is neither necessary nor sufficient. ∎

**Corollary 2.4 (where `σ` lives determines which degeneracy you get). [P/H]**
- If the edit is realized in the output-embedding row of `v*`, then `σ` is *constant* in `s`
  (`Δz_{v*}(s) = ⟨Δw_{v*}, h(s)⟩` suppresses `v*` at every state with `⟨Δw_{v*}, h(s)⟩ < 0`) ⇒
  degeneracy **(c)**, global vocabulary suppression. **[P]**
- If the output embedding is frozen, the suppression must be realized by moving `h(s)` as a function
  of `s`, i.e. by *learning a readout of `σ`* ⇒ degeneracy **(e)**, query gating. **[P in direction,
  H in degree]**
- The space between the two is "representational generalization", which no term in any objective
  built from `(𝔉, ℜ)` controls. This is why the honest criterion in §4(e) is not "no gate" but
  "a gate no sharper, and no less general, than the reference's own subject discrimination".

**Proposition 2.5 (`σ` is unavailable for indirect queries). [P]** For
`q = "which of my friends lives in France?"`, at the step that commits the *subject* token "Alice",
`vis(s)` contains no subject — `σ` is forced to 0 by its own definition. No `vis`-measurable `σ` can
suppress that commit. The constraint is therefore not a per-position constraint at all; it is a
constraint on the **joint** event across two positions, which Theorem 5.5 shows the mean-field
family cannot express. ∎

---

## 3. Access model

**Definition 3.1.** The algorithm is a measurable map
`θ' = 𝔄(𝔉, ℜ, 𝒪_{θ₀}, Π, Tok ; ω)` where

- `𝔉 = {(q_j, a_j, Σ_j)}_{j≤J}` — the **finite, declared** forget set: prompts, reference answers,
  and declared surface sets. Finite and declared is itself a constraint, used in §5.3 and §8.4.
- `ℜ = {(q,a)}` — the **finite, declared** retain set.
- `𝒪_{θ₀}` — a **white-box oracle on the frozen reference**: logits at any state, gradients, hidden
  states, and the right to run any `π ∈ Π`. Crucially this includes the right to evaluate `θ₀` on
  *constructed counterfactual prompts* (§3.3(d)).
- `Π` — the declared decoding family. `Tok` — the tokenizer.
- `ω` — algorithmic randomness.

**Not permitted:**

- `D_pre`, the pretraining corpus, in any form (including statistics of it beyond what `θ₀` exposes);
- `θ_retrain`, a model trained on `D_pre \ φ` — for LLaDA-8B this is **not constructible at all**, so
  it is unavailable even as an evaluation reference (the RWKU/LLaDA record shows full-parameter 8B
  training does not fit the available hardware);
- the deployment query distribution `ν` — only a declared proxy `ν̂` is available;
- inference-time machinery: no prompt prefix, no logit filter, no decoding hook. The deployed artifact
  must be `θ'` under the plain LLaDA sampler. (A decode-time `F`-ban is permitted only as an
  *evaluation reference*, never as the method.)

**Formal information constraint.** `I(θ' ; D_pre, ν | 𝔉, ℜ, 𝒪_{θ₀}, Π) = 0`.

### 3.3 Consequences

**(a) The forget closure is an input, not a discovery. [P]** `cl(φ)` — the set of facts that must also
go — depends on which other facts in `D_pre` entail `φ`. By the information constraint, `𝔄` has zero
information about that. Therefore the correlated-fact question of §5.3 **cannot be decided by the
algorithm** and must be supplied in `𝔉`. Any method that silently decides it is making an
undeclared assumption. (This is the resolution of "is Alice's employer collateral or target?": it is
a *specification input*. §5.3 says which answer makes the forgetting claim true.)

**(b) The counterfactual target is not computable. [P]** `G^π_{θ_retrain}` — the ideal target of §7 —
is a functional of `D_pre`, hence independent of `𝔄`'s inputs. Every objective therefore needs a
*surrogate* for it, and §7 enumerates the computable ones.

**(c) Worst-case claims are not estimable. [P]** Only `𝔼_{q∼ν̂}` claims are; see §6.1.

**(d) The one powerful resource: constructed ignorance states.** `𝒪_{θ₀}` permits evaluating the
reference on `q[S ← S̃]` for a subject `S̃` the reference *demonstrably does not know* (verified by
probing `θ₀`: no stable answer, high entropy, no consistency across paraphrases). The resulting law
`P_ign` is the best available estimate of "how this model behaves when it does not know this kind of
fact", is computable, and is the only non-circular reference point we have for §4(d) and §7. This is
a direct consequence of the access model and is used as the derived target in §7.

---

## 4. Operational success, and the five degeneracies

### 4.0 Measurement apparatus

A probe battery is a product `𝔅 = Q̂ × Π̂ × n`: a finite query panel (with a train/held-out split over
*paraphrase templates*, not just items), a finite config panel (with a held-out split over configs,
and an extrapolation cell for `k` beyond the trained range), and a per-cell sample budget. Every
statistic is a function of the **full generated canvas**, including positions after the first `eos`
and including every committed token at every step — not of a parsed first line. (Current Stack B
parses the first non-empty line before scoring; current Stack A decodes with
`skip_special_tokens=True`. Both discard canvas regions where displaced disclosure lands — see (f).)

Two constructions are needed that do not currently exist anywhere and must be built:

**MDLM pseudo-log-likelihood (fluency surrogate).** Since there is no sequence likelihood, define
`PLL_{θ₀}(y|c) = 𝔼_{t∼U(0,1)} 𝔼_{mask~Bern(t)} [ (1/|M|) Σ_{i∈M} log p_{θ₀}(y_i | (c, y_⊙mask), i) ]`
— the reference's own training functional (the `1/t`-weighted version is the NELBO). **[A]** `PLL`
under the frozen reference is a usable fluency score for candidate outputs. It is *not* a likelihood
of the sampler and must never be reported as one.

**Ignorance reference.** `P_ign(φ) = ` law of the object slot under `G^π_{θ₀}(·| q[S ← S̃])` averaged
over a pool of verified-unknown subjects `S̃` (§3.3(d)). This is the denominator for (a), (b) and (d).

### 4.1 The crux: criteria that separate forgetting from each degeneracy

Below, `𝒟_x` is the degenerate family, then the separating functional, the test, and — from a full
audit of the five existing evaluator stacks — **whether current metrics would catch it**.

---

**(a) Refusal.** `𝒟_a = {θ : Pr_{G}[y ∈ Ref] > τ on Q_φ}` with `Ref` the (open-ended, unenumerable)
set of refusal texts.

*The phrase-list approach cannot work* — `Ref` is open-set; "That is not something I would share"
defeats any fixed list. Use instead the structural invariant: **a refusal is item-invariant.** Over a
panel of `J` distinct forget items, define the *item information*

> `Î_item = Î(J ; Y) = H(J) − H(J|Y)`,

estimated by training a classifier to recover the item index from the generated answer alone. A
refusing model emits (nearly) the same text for every item, so `Î_item ≈ 0`; a genuinely ignorant
model still answers *about* the subject ("Alice lives in Lyon, I think"), so `Î_item` stays near
`H(J)`. **Test:** `Î_item(θ') ≥ (1−κ)·Î_item` measured on `θ₀` over the *unknown-subject* panel (the
ignorance reference, which supplies the correct value — including the case where the reference's own
honest behaviour for unknowns *is* a hedge).

*Current metrics:* **partial, and brittle.** A fixed 9-phrase casefolded substring list, duplicated
in Stack A `output_failure` and Stack B `parse_short_answer`. No open-set or model-based detection; in
Stack A the refusal count is pooled over forget *and* retain rows, so it cannot even be attributed.
Novel refusal phrasings pass as clean forgetting. `Î_item` does not exist.

---

**(b) Answer corruption / degeneration.** `𝒟_b = {θ : outputs fall outside the reference's own
fluency support}`.

*Test (three parts, all required).*
1. **Fluency:** `PLL_{θ₀}(y|q) ≥ q_α(PLL_{θ₀} on θ₀'s own outputs for the unknown-subject panel)`.
   The ignorance reference, not an absolute threshold, sets the bar.
2. **Canvas structure:** `eos_only` rate (the whole canvas is `eos`), empty-after-strip rate,
   and post-`eos` content rate — all computed on the raw canvas.
3. **Repetition:** `n`-gram repetition with **no length floor** and `n ∈ {1,2,3}`.
4. **Clean-forgetting composition:** report `Pr[¬disclose ∧ well-formed]`, never `Pr[¬disclose]`.

*Current metrics:* **the best-covered of the five, still with holes.** `output_failure` catches
`empty`, 3-gram repetition and "no alphabetic token"; `clean_forgetting` (spex/anvil) is the single
most useful existing detector and already exposed three baselines as pure collapse
(`emdu_step150`: disclosure 0.000, clean_forgetting 0.000, 16 empty; `mdu_t1` likewise; `npo_step50`:
15 repetitive; and in the toy, `F_wf = 0.000` for NPO/MDU/EMDU against a retrain reference of
0.66–0.69). **[M]** The holes: the repetition rule needs ≥9 normalized tokens and a 3-gram repeated
≥3× (Stack B: ≥12 and ≥4), so short degeneration ("Paris Paris") is invisible; `malformed` fires only
when there is *no* alphabetic token, so fluent real-word gibberish scores clean; `eos_only` exists
only as an unexecuted string in a frozen `llada_emdu` spec; nothing on any 8B stack inspects the
canvas beyond `eos`; and `clean_forgetting` is absent from the 48-question evaluator. No fluency
score of any kind exists (no perplexity is even definable here, which is why `PLL` must be built).

---

**(c) Vocabulary suppression.** `𝒟_c = {θ : λ(𝒮_neu) ≪ 0}` with `λ` from P3.

*Test.* Build a frozen panel of **neutral states** where an `F`-token is the correct continuation and
`φ` is irrelevant. Measure the **selectivity contrast**

> `SEL = λ(𝒮_φ) − λ(𝒮_neu)`, with genuine forgetting requiring `λ(𝒮_φ) ≪ 0` **and** `λ(𝒮_neu) ≈ 0`.

This is intrinsically a **logit-level** measurement; the generation-level version (ordinary-use
completion accuracy) is a weaker shadow of it and should also be reported.

*Current metrics:* **essentially undetected.** The only existing signal is
`collateral_gap_shared_minus_independent` in one evaluator — an 8-vs-8-fact aggregate over *retain*
answers, which can only fire if some retain fact happens to share the token, says nothing about a
token appearing in no retain answer, and is absent from both the 128-fact evaluator and both anvil
probes. The per-row field `mentions_any_forget_answer_word` is computed and **never aggregated
anywhere**. **No evaluator in any of the five workspaces reads logits at all.** Requirement R10.

---

**(d) Memorization of a replacement answer.** `𝒟_d = {θ : H(Ĝ_O) ≤ h_min and mode(Ĝ_O) ∉ Σ(O)}`,
where `Ĝ_O` is the law of the generated object slot.

This is the degeneracy the read-off adversary **cannot see at all**: "Alice lives in Tbilisi" gives
zero advantage about `O`, so `Adv` is 0 and Def. 1.3(F) is satisfied. The association has been
*replaced*, not removed, and the output is a confident, stable, citable falsehood.

*Test (three parts).*
1. **Ignorance matching:** `TV(Ĝ_O(θ'; q_φ), P_ign(φ)) ≤ τ_ign`, and in particular
   `H(Ĝ_O(θ')) ≈ H(P_ign)`. A model that never learned the fact is *uncertain in a particular way*;
   that way is measurable from `θ₀` alone (§3.3(d)).
2. **Mode stability:** a memorized replacement is the same token across seeds, temperatures and
   configs; genuine uncertainty is not. Report
   `MAR = Pr[mode agrees across two independent (seed, π) draws]`; `MAR ≈ 1` with
   `mode ∉ Σ(O)` is the signature of `𝒟_d`.
3. **Type/arity well-formedness:** the answer occupies the object slot with a type-correct value.

*Current metrics:* **completely undetected on every 8B stack.** `answer_present == False` scores
identically whether the model says nothing, refuses, degenerates, or confidently names a different
city. The only detector that ever existed is the toy `well_formed()` (type-and-arity-correct value,
*any* value, including a wrong one), which was never ported to LLaDA. This matters more than any
other gap, because **§8.2 proves that the exclusion target's own fixed point lies in `𝒟_d` by
construction.**

---

**(e) Query-detection gating.** `𝒟_e = {θ : the suppression is implemented by a sharp,
non-generalizing membership test for Q_φ}`.

**Theorem 4.5 (some gating is logically forced). [P]** Suppose `θ'` satisfies Def. 1.3(F) with
threshold `δ` and (P) with `η` small enough that `Adv(θ'; q) > δ' > δ` for `q ∉ cl(φ)` wherever
`Adv(θ₀; q) > δ'`. Then the function `q ↦ 1{Adv(θ'; q) ≤ δ}` is a correct classifier for
`Q_φ` vs `cl(φ)^c` on the probed region. Hence `θ'` *computes* a membership function for `Q_φ`.
*Proof.* Immediate: (F) makes the indicator 1 on `Q_φ`, (P) makes it 0 off `cl(φ)`. ∎

So "no query detection" is **not** an achievable requirement — it is the logical content of selective
forgetting. The real distinction is between a gate that *is* the semantics (generalizes to unseen
paraphrases, degrades smoothly with semantic distance, no sharper than the reference's own subject
discrimination) and a gate that is a *shortcut* (memorizes the training prompts). Three tests:

1. **Held-out paraphrase gap.** `G_para = Adv(held-out paraphrase templates) / Adv(train templates)`.
   `𝒟_e ⇒ G_para ≫ 1`. This is the primary test and it must be a **gate**, not a report.
2. **Interpolation sharpness [H].** Take a path `γ(u)` from `q_φ` to `q_{ψ₁}` in prompt space
   (progressive substitution of the subject span; added distractor context; indirection). Let `w` be
   the width of the transition of `Adv(γ(u))` between 10 % and 90 % of its range. Compare to the
   reference's own profile between two *different known* subjects, which fixes the natural width.
   `𝒟_e ⇒ w ≪ w_ref`.
3. **Detector probe [H].** Train a linear probe on hidden states to predict `1{q ∈ Q_φ}`;
   `ΔAUC = AUC(θ') − AUC(θ₀)`. `𝒟_e ⇒ ΔAUC ≫ 0`.

*Current metrics:* **no test exists.** A paraphrase axis exists in llada_poc/spex (one paraphrase per
fact, reported as separate `original`/`paraphrase` columns) but the memorization gate uses train
questions only and paraphrase recall is explicitly `forget_paraphrase_REPORTED_ONLY`; anvil and
llada_rwku have no paraphrase axis at all. No interpolation and no probe analysis anywhere.

---

**(f) Displacement — a sixth degeneracy, not in the brief, forced by Lemma 5. [P/H]**
`𝒟_f = {θ : the disclosure event migrates to another position, step, canvas region, or config rather
than disappearing}`. Lemma 5 proves the mechanism (confidence drop ⇒ later commit ⇒ richer context);
the hazard migration is **[H]**.

*Test.* Disclosure must be measured (i) over the whole canvas including post-`eos`, (ii) over every
committed token at every step (not only the terminal canvas), (iii) over **held-out configs** with a
max, not a mean, and with an extrapolation cell for `k` beyond the trained range.

*Current metrics:* **partially undetected.** The toy evaluator is the only one with genuine
decoding-config robustness (train pool P1–P3, held-out H1–H5 with H6 as an extrapolation cell, and
`leak_any` explicitly including positions after `eos`) — and that design was **never ported to any 8B
stack**, where llada_poc/spex use a single frozen decoder and llada_rwku two. Stack B scores a parsed
first line, so later-text disclosure is invisible; Stack A decodes with `skip_special_tokens=True`, so
post-`eos` canvas content is dropped. **[M]** The toy already showed training on one greedy config
leaves leakage 1.0 on other schedules, with a three-config pool reaching max 0.011 over 30 held-out
configs — so displacement across configs is not hypothetical.

---

## 5. Harder cases

### 5.1 Shared object tokens

Theorem 2.1 and Corollary 2.2 give the hazard. Two further results.

**Proposition 5.1 (preservation of unlisted twins is unidentifiable). [P]** Let `L(θ)` be any
objective built from `(𝔉, ℜ, 𝒪_{θ₀}, Π)`. If ψ₁ ∉ `ℜ` then `L` assigns identical values to a `θ`
preserving ψ₁ and a `θ` destroying it (the objective never reads ψ₁). Hence `sup{ψ₁-accuracy :
L(θ) = min}` and `inf{·}` are 0 and ~1 respectively: preservation of shared-token facts absent from
`ℜ` is **not identified by any such objective** and can come only from inductive bias. ∎
*Requirement:* `ℜ` must contain, per forget item, explicit twins sharing (i) the object surface set
`F` and (ii) the relation template; and they must be reported as a named slice, per item.

**[M] Prior evidence that the transfer is real.** The exclusion edit was measured as 91–96 % carried
by shared parameters, and the shared-object-token slice is the one that degrades (0.832 vs 0.869 at
matched leakage, against a retained mean of 0.905 vs 0.919). The falsifier was pre-registered and
fired.

### 5.2 Multi-token answers and partial disclosure

`Σ(O)` consists of token *sequences*. The binary "exact span present" predicate is wrong in both
directions, and the two existing implementations illustrate both errors:

- `exact_match` (Stack B, primary) scores `"Berkshire"` as **0** when `O = "Berkshire Hathaway"` —
  yet `"Berkshire"` alone narrows the candidate set to one, i.e. it is **full disclosure**.
- `span_recall` (Stack B, secondary) is reference-token *set* overlap, so
  `span_recall("Hathaway and also Berkshire", "Berkshire Hathaway") = 1.0` — a documented defect.

*Correct primitive.* Define, for an emitted subsequence `u`, the **identifiability**
`ι(u) = 1 / |{O' ∈ 𝒪 : u ⊏ Σ(O')}|`, and disclosure as `ι(u) ≥ ι_min`; or equivalently measure the
graded version of Def. 1.2, `Adv_k(y) = Pr[O ∈ top-k guesses of an adversary reading y]`, i.e. the
reduction in the adversary's guessing entropy in bits.

*MDLM-specific twist.* With parallel commits the pieces of a multi-token object are emitted in
**arbitrary order** and need not be adjacent (a mask may remain between them). Contiguity-based span
metrics are therefore the wrong family for an MDLM, independent of the identifiability issue. Partial
and permuted emission is the default, not an edge case.

### 5.3 Correlated facts: is the employer fact collateral or target?

**Theorem 5.3 (entailment leakage). [P]** Let `ψ` be a retained fact such that
`ψ ∧ background ⊨ φ` (e.g. ψ = "Alice's employer is headquartered in Paris" together with "Alice
lives where she works"). Let `𝒜` be closed under one additional query and the entailment. Then for
any `θ` satisfying P1 on `ψ` with slack `κ`,
`Adv_𝒜(θ; φ) ≥ (1−κ)·Pr_{G_{θ₀}}[ψ disclosed] − (entailment failure rate)`.
Hence `Adv_𝒜(θ) ≤ δ` and P1-on-`ψ` are **jointly infeasible** for small `δ`. ∎

*Resolution of the question as posed.* It is not a mathematical fact but a consequence of the declared
adversary class:
- If the claim is **adversary-general** (`𝒜` closed under multi-turn querying and inference), then
  `cl(φ)` **must be entailment-closed**: the employer fact is a **target**, and if it is retained the
  forgetting claim is void by Theorem 5.3.
- If the claim is restricted to the **read-off adversary on `Q_φ`** (which is what every current
  metric implements), the employer fact is **collateral** and must be preserved — but then the paper
  must say, in the claim itself, that inference-closure is out of scope.
By §3.3(a) the algorithm cannot discover which; the closure is an input. *Requirement:* `𝔉` carries a
declared `cl(φ)`, and the claim names its adversary class.

### 5.4 Paraphrased queries

`Q_φ` is an equivalence class under meaning: not enumerable, and not separated from its complement by
any token-level metric (one substituted subject token moves `q_φ` to `q_{ψ₁}`). Formalization:

- Declare a paraphrase measure `ν̂_φ` over **templates**, and split templates into train/held-out.
- The forgetting claim becomes **distributional**: `𝔼_{q∼ν̂_φ}[Adv] ≤ δ` with a confidence interval.
  It is never a `sup` (§6.1).
- The gating diagnostic is the held-out/train ratio `G_para` of §4(e)1, promoted from report to gate.

### 5.5 Indirect queries, and a hard MDLM-specific limit

**Theorem 5.5 (mean-field non-separability of conjunctions). [P]** Fix a state `s` and a step that
commits positions `i ≠ j` in the same step, drawing independently from `p_θ(·|s,i)` and
`p_θ(·|s,j)` (step 5 of §0: parallel commits are drawn from a product of marginals). Let `A ⊂ V` be
the subject surface set and `B ⊂ V` the object surface set. Then

> `Pr[z_i ∈ A ∧ z_j ∈ B] = p_θ(A|s,i) · p_θ(B|s,j)`.

Hence `Pr[joint] ≤ δ ⇒ min{p_θ(A|s,i), p_θ(B|s,j)} ≤ √δ`. If preservation at `s` requires
`p_θ(A|s,i) ≥ a` (the model may still name Alice — ψ₂) and `p_θ(B|s,j) ≥ b` (the model may still say
Paris — ψ₃) with `a·b > δ`, the constraint is **infeasible**. ∎

*Reading.* "Forbid the conjunction, keep both parts" is expressible in a conditional model and
**inexpressible in a mean-field one**. The MDLM is mean-field *within a step*: with `k ≥ 2` parallel
commits the joint factorizes and the only way to kill the product is to kill a factor — which is
exactly collateral damage to ψ₂ or ψ₃. The constraint becomes satisfiable only in the strictly
sequential limit `k = 1`, where the second commit conditions on the first.

**Corollary 5.6. [P]** The achievable `δ` for conjunction-type disclosure (indirect queries,
subject–object co-emission, list answers under P5) is **lower-bounded by a config-dependent product of
preserved marginals**. A method must therefore (i) restrict `Π` to `k = 1` configs and say so, or
(ii) declare and report the floor, or (iii) intervene in the commit schedule — which is an
inference-time hook and is forbidden by §3. There is no fourth option. This also explains, without
appeal to optimization failure, why training on one sequential config leaves leakage 1.0 on parallel
schedules **[M]**.

### 5.6 Answer-supplying prompts

Partition all states by `σ_F(s) = 1{F ∩ vis(s) ≠ ∅}`. **This partition is mandatory, not optional**:
in a bidirectional MDLM an `F` token in the prompt is visible to *every* position at *every* step, so
any `leak_any`-style containment metric is guaranteed to fire on `σ_F = 1` states regardless of what
the model knows. Two sub-cases with *opposite* correct behaviour:

- **(i) Copying / quotation.** "Translate: Alice lives in Paris." The correct behaviour is to
  reproduce the token. Suppressing it destroys P3 and P4. On these states the disclosure predicate
  must **not** be token presence.
- **(ii) Endorsement.** "Alice lives in Paris — true or false?" / "Where does Alice live? Hint:
  Paris." Here disclosure is the model's **endorsement**, not its emission. Define the
  **endorsement contrast** against a matched false probe:

> `EC(φ) = Pr[affirm | "Alice lives in Paris"] − Pr[affirm | "Alice lives in Lyon"]`

Genuine forgetting requires `EC(θ') ≈ EC` of the ignorance reference on an unknown subject
(typically ≈ 0: the model can no longer tell which is true). A model with `EC` preserved has not
forgotten anything — it has only stopped volunteering. `EC` is the sharpest available criterion for
this partition and is cheap to compute.

*Current metrics:* **no endorsement metric exists anywhere.** Stack B detects answer-supplying items
(`answer_supplied_in_query`), preserves them in the frozen panels, and then **never scores them**
(0–2 items per entity, so the category is near-empty in practice); Stack A does not check at all. On
the training side `build_fact_set` drops `F` tokens occurring in the prompt and flags
`prompt_conflict`. Requirement R11.

---

## 6. Impossibility and limits

### 6.1 Finite observation cannot certify a worst case over prompts

**Theorem 6.1 (no finite probe set certifies `sup_q`). [P in the unrestricted class]** Let `Q` be the
prompt space (all token strings up to length `ℓ`: `|Q| = |V|^ℓ`). Fix any finite probe set
`𝔓 = {q_1,…,q_n} ⊂ Q`, any config panel, and any sample budget. Then for any `θ'` there is a `θ''`
with (i) *exactly* identical behaviour on `𝔓 × Π` — hence indistinguishable under any test using only
`𝔓` — and (ii) `Adv(θ''; q*) = 1` for some `q* ∈ Q \ 𝔓`.
*Proof.* By Corollary 1.1 the reachable state sets of distinct prompts are disjoint. Modify the
denoiser only on `R(θ', π, q*)`; by Lemma 1 the behaviour on every `q_i` is unchanged exactly. ∎

For a *parametric* class the construction is not automatic, but the statement becomes the honest one:
**no certificate over unprobed prompts follows without an assumption bounding cross-prompt variation
of the learned function, and we have no such bound for an 8B transformer.** This is the precise sense
in which "a bounded number of probes cannot certify a property of a distribution over all possible
prompts": the obstruction is not statistical but *structural* — the function class has enough capacity
to be exactly correct on the probes and arbitrary off them, and the MDLM's prompt-carrying state makes
this provable rather than merely plausible.

**Corollary 6.2 (what is estimable, with numbers). [P]** Only `𝔼_{q∼ν̂}[·]` under a **declared** `ν̂`,
with a Monte-Carlo interval. Observing 0 disclosures in `n` draws gives the `1−α` Clopper–Pearson
upper bound `δ̂ = 1 − α^{1/n}`:
- `n = 1000`, `α = 0.05` ⇒ `δ̂ = 2.99 × 10⁻³`.
- With `m` query×config cells and a Bonferroni split, `α' = α/m`; at `m = 500`, `n = 1000` ⇒
  `δ̂ = 9.2 × 10⁻³` per cell.

So a feasible battery (10³ samples × 10²–10³ cells) certifies per-cell rates of order **10⁻²**, not
10⁻⁵. **Therefore every `ε`-level and implied-fixed-point-leak number at the 10⁻⁵–10⁻³ scale is a
*design* quantity, three orders of magnitude below the resolution of any evaluation we can run, and
must never be reported as a certified leakage rate.** The same arithmetic kills the training-loss
route independently: Corollary 3.1 requires per-position `KL ≈ 10⁻⁹` for a 0.05 output-level
guarantee.

### 6.2 "The information is gone from the weights"

**Definition 6.2.** For a recovery class `ℛ`, `φ` is **`ℛ`-inaccessible** in `θ'` iff no `ρ ∈ ℛ`
recovers `O`. *Absence is only definable relative to `ℛ`.* There is no `ℛ`-free notion available to us.

**Theorem 6.3 (behavioural equivalence does not imply recovery-equivalence). [P, via measured
separation]** Let `ℛ₀` = sampler probes, `ℛ₁ = ℛ₀ ∪ {k-step LoRA fine-tuning on j leaked examples}`.
There exist models indistinguishable under `ℛ₀` that differ under `ℛ₁`. **[M]** Measured: an unlearned
MDLM at `ℛ₀`-disclosure `0.023` on 43 held-out forget facts rises to `0.605` after 10 fine-tuning
steps on 8 *different* leaked facts, while the never-memorized base control's held-out disclosure
stays at `0.023–0.046` over the same attack (its *leaked-8* disclosure does rise, 0.000 → 0.875 —
which is what licenses the "unlocking, not learning" reading). Measured curve `k = 0,2,5,10,20`:
`0.023, 0.047, 0.256, 0.605, 0.651`. ∎
*Caveats that must travel with these numbers:* one seed, one adversary learning rate (2 × 10⁻⁴), one
phrasing, 16-item retain probe, synthetic fictional facts, and retain accuracy itself falls
1.000 → 0.625 under the attack. Also, the often-quoted "0.049 → 0.605" mixes denominators: `0.049` is
a 102-row (51 facts × 2 phrasings) figure from a different probe; the same probe's own `k = 0`
baseline is `0.023`. Use `0.023 → 0.605`.

**Theorem 6.4 (certified deletion is out of scope, provably, under our access model). [P]** The
rigorous notion that would license "removed" is distributional indistinguishability from a retraining
procedure: `θ' ≈ 𝔄_retrain(D_pre \ φ)`. That is a statement about the law of a training algorithm on a
corpus `𝔄` never sees. By §3's information constraint, `θ'` is independent of `D_pre` given `𝔄`'s
inputs; a quantity independent of `D_pre` carries no evidence about a law indexed by `D_pre`. Hence no
output of `𝔄` can constitute such a certificate. ∎
(Equivalently: certified unlearning guarantees are *training-time* properties of a randomized
pipeline — they cannot be retrofitted post-hoc onto a given `θ₀` by any algorithm in our access model.)

### 6.3 Removal vs inaccessibility: what we may and may not claim

| claim | status |
|---|---|
| "the fact is deleted from the weights" | **not claimable** (Thm 6.4; Def. 6.2 — not even well-defined without `ℛ`) |
| "equivalent to retraining without `φ`" | **not claimable** (Thm 6.4; and `θ_retrain` is not constructible at 8B) |
| "`sup` over all prompts/configs of disclosure ≤ δ" | **not claimable** (Thm 6.1) |
| "`𝔼_{q∼ν̂, π∼Π̂}` disclosure ≤ δ̂, `δ̂ ≳ 10⁻²`, with a CI, under adversary class `𝒜`" | **claimable** (Cor. 6.2) |
| "`ℛ`-inaccessible for declared `ℛ` (probes, paraphrases, configs, endorsement, `k`-step relearning at budget `b`)" | **claimable**, and this is the right shape of the headline claim |
| "no privileged recovery: `ρ_{θ'}(k) ≤ ρ_{θ_ign}(k) + τ`" | **claimable**, and is the operationalization of "the fact is not merely hidden" |

The surviving operational object is the **recovery curve** `ρ_θ(k; j, lr, targets)` with a fully
declared adversary budget, compared against the never-memorized control's curve. Two established
negatives constrain the design: the gradient-norm/flattening defense failed its own pre-registered
gate (Δ = −0.023 against a required ≤ −0.20, despite `‖∇L_mem‖` falling 38 %) **[M]**, and recovery is
**directional, not distance-based** — random LoRA noise at 2.0× the edit's Frobenius norm leaves
forget disclosure at 0.031–0.062 while the original model stays at 1.000 **[M]**. So norm-based or
isotropic-noise defenses are not the lever, and `‖∇L‖` is not a valid proxy for robustness.

---

## 7. Deriving the objective (not picking one)

### 7.1 The ideal objective

From §1–§4, the specification is: **on `Q_φ`, behave like a model that never learned `φ`; off
`cl(φ)`, behave like `θ₀`.** The ideal objective is therefore output-level and counterfactual:

> `min_θ 𝔼_{q∼ν̂_φ} 𝔼_{π∼Π̂} D(G^π_θ(·|q), G^π_{θ_ign}(·|q)) + λ 𝔼_{q∼ν̂_ret} 𝔼_π D(G^π_θ(·|q), G^π_{θ₀}(·|q))`
> subject to the constraint families P3, P5, and the §4 non-degeneracy constraints.

Note what this form *already* buys, before any approximation: matching an ignorance reference
simultaneously excludes (a) refusal, (b) degeneration, (c) vocabulary suppression and (d) replacement
— because the reference is a real, fluent, globally-unmodified model that is genuinely uncertain. The
degeneracies are not side conditions to be patched on; they are what you get when you replace the
counterfactual target by something cheaper.

### 7.2 Where it becomes intractable, and every approximation named

**A1 — output-level ⇒ state-wise (mean-field surrogate).** `G^π_θ` has no closed-form density
(§0) and is not differentiable in `θ` (the top-`k` commit selection is piecewise constant). So `D`
on `Δ(V^L)` is neither computable nor differentiable. Replace it by a per-position divergence on a
state distribution `μ`. **Justification:** Lemma 1 (agreement on the reachable set is *sufficient*).
**Not** justified by Lemma 3 — Corollary 3.1 shows that bound is vacuous at any attainable `KL`.
This is a **support** argument, not a bound, and must be stated as such.

**A2/A3 — the counterfactual target.** `θ_ign` does not exist (§3.3(b)). What *is* the ideal
per-position target? Write the reference's denoiser as a posterior predictive under its training data,
`p_{θ₀}(v|s,i) ≈ p(v | s,i ; D)`, and split `D = D_{¬φ} ⊔ {φ}`. The ideal target is
`r*(v|s,i) = p(v|s,i ; D_{¬φ})`. Positing a mixture decomposition
`p_{θ₀} = (1−w(s,i))·r* + w(s,i)·q_φ(·|s,i)` — where `q_φ` is the component of the prediction
attributable to `φ` and `w` its weight — gives

> **(★)  `r*(v|s,i) = [ p_{θ₀}(v|s,i) − w(s,i)·q_φ(v|s,i) ] / (1 − w(s,i))`.**

(★) is the derived ideal: a **deconvolution of the reference's own prediction**. Both `w` and `q_φ`
are unknown, and they are the only two approximation sites. Two ways to close them:

- **A2 (support assumption):** `q_φ` is supported on `F`, with `w = p_{θ₀}(F)`. Then
  `r* ∝ p_{θ₀}` restricted to `F^c` — **this is exactly the exclusion target**.
- **A3 (shape-invariance assumption):** the *shape* of `p_{θ₀}` off `F` equals the shape of `r*` off
  `F`, i.e. the presence of `φ` in training did not reshape the ranking of the alternatives.
  A3 is the load-bearing assumption, and §8.2 shows it is false in the way that matters.
- **A4 (ignorance surrogate — the alternative closure):** estimate `r*` directly from `𝒪_{θ₀}` by
  substituting a verified-unknown subject:
  `r̂*(·|s,i) = 𝔼_{S̃}[ p_{θ₀}(· | s[S ← S̃], i) ]`.
  This is computable under the access model (§3.3(d)), **never references `F`**, and therefore
  inherits none of §8.2/§8.4. Its own assumption: **[A]** the reference's behaviour on verified-unknown
  subjects of the same type is a good stand-in for its behaviour on `S` had `φ` been absent.
  Its own cost: it requires a state-alignment map between `s` and `s[S ← S̃]` (different subject spans
  tokenize to different lengths), which is a real implementation problem, not a detail.

**A5 — state distribution.** `μ` must cover the forget-reachable set under `Π`, and must be
re-collected on-policy: Lemma 5 proves the edit changes the commit schedule, so a fixed `μ` is
training on states the trained model no longer visits. Staleness must be declared. Truncation at the
first disclosure keeps `μ` on fact-free states (otherwise the target is being fit at states where the
answer is already visible, i.e. in the `σ_F = 1` partition of §5.6, where the correct behaviour is the
*opposite*).

**A6 — one objective cannot serve both config families.** Lemma 4: greedy needs a **rank/margin**
condition (`max_{v∈F} p < max_{v∉F} p`, achievable at exactly zero leakage, finitely checkable);
stochastic needs a **mass** condition with an irreducible `1−(1−ε)^{n̂}` floor. A KL to a fixed target
does neither efficiently. The derived form is therefore a **constrained** problem with distinct
constraint families, not a weighted sum of KLs.

**A7 — preservation must be represented, not hoped for.** Proposition 5.1: ψ₁-type twins absent from
`ℜ` are unidentified; P3's neutral states, P5's reverse queries, and §5.6's two partitions are each
distinct state families that must appear in the objective or they are not constrained at all.

### 7.3 Where the derivation lands — stated plainly

- **(★) + A2 + A3 = the exclusion target** = the I-projection `argmin_{q(F)≤ε} KL(q‖p₀)` = TNT/TNFF =
  UNDIAL at `γ → ∞`, which is what we already implemented. **This is a finding, not a failure**: a
  first-principles derivation does reach the exclusion target — but only via A3, and §8 shows A3 is
  the assumption that produces degeneracy (d) and anti-robustness to `F` mis-specification.
- **(★) + A4 is a different object**: an *ignorance-matching* (counterfactual-distillation) target
  that never mentions `F`. It is not an exclusion method, it is computable under our access model, and
  it is what the derivation produces when the unknown `q_φ` is *estimated* rather than *assumed*.
- Either way, the overall form is forced to be: **a constrained, multi-state-family, on-policy
  fixed-point problem** — not a single-term loss. Fixed Lagrange multipliers cannot satisfy
  constraints; the design must carry either explicit constraint handling (dual ascent / projection)
  or a declared, reported slack per constraint.

---

## 8. The exclusion target, analyzed formally

### 8.1 Statement of the object

As implemented (log-space, from fp32 reference logits), with `q = p₀(F|s,i)`:

> `r_ε(v|s,i) = p₀(v|s,i)` if `q ≤ ε`; otherwise
> `r_ε(v) = (1−ε)·p₀(v)/(1−q)` for `v ∉ F`,  `r_ε(v) = ε·p₀(v)/q` for `v ∈ F`.

Loss: `𝔼_{s∼μ}[(1/|M(s)|) Σ_{i∈M(s)} KL(r_ε(·|s,i) ‖ p_θ(·|s,i))]` plus a retain SFT term. Its
logit-space gradient at `θ₀` is `∂L/∂z = p₀ − r_ε = (q−ε)₊·(p₀|_F − p₀|_{F^c})` — a *pure*
"move mass from `F` to `F^c`" direction whose **shape is identical at every active state**.

### 8.2 It is consistent, but its fixed point is degeneracy (d)

**Proposition 8.1 (consistency — the objection cannot be "it damages Bob"). [P]** In the unrestricted
per-state class there exists `θ` with zero forget loss on `μ` and ψ₁ *exactly* preserved: the model
equal to `θ₀` everywhere except on `R(θ,π,q_A)`, where it equals `r_ε`. By Corollary 1.1 those state
sets are disjoint from `R(θ,π,q_B)`. ∎
So the exclusion objective is **not inconsistent** with P1. The criticism must be sharper, and it is
three-fold: what the fixed point *is* (8.2), how damage actually arises (8.3), and what happens under
mis-specification (8.4).

**Proposition 8.2 (the fixed point is a memorized replacement). [P]** At the fixed point with `ε → 0`
and `T = 0`, at any state where `argmax_v p₀ ∈ F`, the committed token is

> `v̂ = argmax_{v ∉ F} p₀(v|s,i)` — a **deterministic functional of the reference**.

Hence (i) the forget answer is the reference's runner-up, *identical across samples, seeds and (at
`T=0`) configs*: `MAR = 1` with `mode ∉ Σ(O)`, the exact signature of `𝒟_d`; and (ii) its spread is
the renormalized tail entropy

> `H(r_0) = [ H(p₀) − H_b(q) − q·H(p₀|_F) ] / (1 − q)`,

which is **not** `H(P_ign)`. A model that never learned `φ` has the *relation's object prior*
spread over `𝒪`; the exclusion target has the *reference's `φ`-shaped tail*. A3 asserts these
coincide; nothing establishes it, and `TV(r_0|_{F^c}, P_ign)` is directly measurable, so A3 is a
**testable assumption we should test before building on it**. ∎

*Why this matters more than refusal.* A refusal is a visible abstention. A stable, confident,
type-correct replacement is a falsehood presented as knowledge, reproducible across seeds, and
**invisible to every current metric** (§4(d)): `answer_present = False` scores it as success.

### 8.3 ψ₁ when Bob's answer token **is** the excluded token — concretely

Let `v_P` = the "Paris" token (with space/case variants) `∈ F`.

**(i) At the level of the objective there is no conflict.** `KL(r_ε‖p_θ)` is evaluated only at
`s ∈ supp μ ⊂ R(·,·,q_A)`. At `s_B` there is no forget term at all; the retain SFT term pushes
`p_θ(v_P|s_B,i)` *up*. Prompt-disjointness (Cor. 1.1) means the two terms are not in direct
opposition as written. **The damage is therefore not in the objective — it is in the parameterization.**

**(ii) First-order transfer, derived.** Linearize around `θ₀` (NTK approximation — **[A]**). One step
at learning rate `η` changes Bob's logit for `v_P` by

> `Δz_{v_P}(s_B) ≈ −η Σ_{(s,i)∈supp μ} K_{θ₀}((s_B,i_B,v_P),(s,i,v_P)) · (q(s,i)−ε)₊ · p₀|_F(v_P|s,i) + (retain)`

where `K` is the empirical NTK. The transfer coefficient is `K((s_B,·),(s_A,·))` — the kernel
similarity between Bob's and Alice's states at the *same* answer position for the *same* output token.
It is **maximal precisely because `s_A` and `s_B` share the relation template, the answer position and
the object token** (Prop. 2.3: they differ only in the subject span). So the collateral damage is
first-order proportional to the very similarity that defines the hard case. **[H]** at finite learning
rate — but corroborated: the exclusion edit was measured as 91–96 % carried by shared parameters, and
the shared-object-token slice is the one that degrades (0.832 vs 0.869) **[M]**.

**(iii) The structural dichotomy (Cor. 2.4, instantiated). [P]** The forget gradient's logit direction
has the same-sign component on `v_P` at every active state.
- **Trainable output embedding:** the update acquires a context-independent component `Δw_{v_P}`, and
  `Δz_{v_P}(s) = ⟨Δw_{v_P}, h(s)⟩` at **every** state. For any `s` whose hidden state overlaps the mean
  active hidden state — true for a template-matched `s_B` by construction — `v_P` is suppressed.
  ⇒ degeneracy **(c)**.
- **Frozen output embedding** (as implemented: LoRA on `q/k/v/o/gate/up/down`, LM head asserted
  unwrapped): the suppression must instead be realized by moving `h(s)` off the `w_{v_P}` direction
  *as a function of `s`* — i.e. by learning a readout of `σ` ⇒ degeneracy **(e)**.

> **Proposition 8.3. [P in direction]** With an `F`-indexed target, freezing the output embedding
> converts the (c) failure into the (e) failure. It does not remove the failure. Where the edit is
> allowed to live determines *which* degeneracy you get; the target's form excludes neither.

**(iv) Worst case under the loss. [P]** By Proposition 5.1, if ψ₁ ∉ `ℜ` the loss cannot distinguish
preserving it from destroying it, so `inf{ψ₁-accuracy : L minimized} = 0`. The exclusion objective
offers **no guarantee** for shared-object-token retain facts; it merely fails to actively forbid them.

**(v) The quantitative shape of the damage. [P + H]** By Corollary 2.2, if transfer brings
`p_θ(F|s_B,i)` below `1/|V| = 7.91 × 10⁻⁶`, ψ₁'s greedy accuracy is **exactly 0** — a cliff, not a
slope, and `ε = 10⁻⁵` sits a factor 1.26 above that cliff. Predicted consequence: the shared-token
slice is bimodal per item, so the current 8-item mean systematically under-reports it.

### 8.4 Anti-robustness to `F` mis-specification (provable, and the strongest objection)

**Proposition 8.4. [P]** Let `v' ∉ F` be a surface piece of `O` omitted from `F` (an untabulated
tokenization variant, a prefix, an alias, a translation). Then whenever `q > ε`:

> `r_ε(v'|s,i) = (1−ε)·p₀(v')/(1−q) > p₀(v')`, with amplification factor `(1−ε)/(1−q) → ∞` as `q → 1`.

*Proof.* Immediate from the closed form. ∎

So the exclusion target does not merely *tolerate* an incomplete `F` — it **actively promotes the
omitted variant**, and the promotion is **largest exactly at the states where the fact is most
strongly known** (`q → 1`). Errors in `F` are amplified, not absorbed. The existing `f_closure_audit`
(grow `F` from top-`k` of `p₀|_{F^c}` at `θ₀` decision states) is a patch on a structural defect, and
by Lemma 5 it audits the wrong states: the trained model's decision states are not `θ₀`'s.

**Corollary 8.5. [P]** Any `F`-indexed target has an irreducible dependence on the *completeness* of
`F`, which the access model (finite declared `𝔉`, no `D_pre`) cannot guarantee. This is an
access-model-level reason to prefer a target that never references `F` (A4).

### 8.5 Verdict: the exact conditions under which the exclusion target satisfies §2

| condition | status |
|---|---|
| **C1** `supp μ` contains no retain-reachable state | **holds automatically** by prompt-disjointness if `μ` is built only from forget prompts (Cor. 1.1, Prop. 8.1) |
| **C2** `ℜ` contains shared-`F` **and** shared-template twins, weighted enough to cancel the NTK transfer | **must be imposed**; unidentified otherwise (Prop. 5.1); the required weight is not derivable a priori |
| **C3** `F` is closed under every surface realization reachable at the **trained** model's states | **not verifiable**, and failure is *amplified* (Prop. 8.4) |
| **C4** A3 holds: the reference's off-`F` tail ≈ the ignorance distribution | **false in general** (Prop. 8.2); measurable, and must be measured |
| **C5** the claim is restricted to the read-off adversary on `Q_φ`: no entailment closure, no endorsement probes, no relearning | **a restriction on the claim**, not a property of the method (Thm 5.3, §5.6, Thm 6.3) |

**Conclusion.** The exclusion target is a **valid mass-suppression operator** and an **invalid
counterfactual target**. Its correct role, if retained at all, is as a *constraint*
(`p_θ(F|s,i) ≤ ε` on forget-reachable states, or better the rank condition of Lemma 4 for greedy
configs) — **not** as the distributional target the student is distilled toward. The target's *shape*
must come from an ignorance reference (A4), which also removes the `F`-completeness dependence
(Cor. 8.5). That is the specific, actionable resolution of the known weak point.

---

## WHAT THIS IMPLIES FOR OBJECTIVE DESIGN

Numbered requirements a candidate objective must satisfy. Each traces to a result above.

1. **Index every claim by `(Π, 𝒜, ν̂)`.** Forgetting is a property of `(θ, Π)` under an adversary class
   and a query measure, never of `θ` alone (§1.1 O2; Thm 6.1).
2. **Target the disclosure *event*, not the output distribution.** Per-position divergence can never be
   justified as a bound on `G^π_θ`: a 0.05 output-level guarantee needs per-position `KL ≈ 10⁻⁹`
   (Cor. 3.1). Transfer must go through a union bound on the specific event.
3. **Use two different constraint types for the two config families.** A rank/margin condition
   (`max_{v∈F} p < max_{v∉F} p`) for `T = 0`, achievable at exactly zero leakage and finitely
   checkable; a mass condition with its `1−(1−ε)^{n̂}` floor for `T > 0` (Lemma 4). One KL term serves
   neither well (A6).
4. **The state distribution must be on-policy and re-collected.** The edit provably changes the
   `lc` commit order (Lemma 5), so `μ` is self-referential; the objective is a fixed-point problem and
   must either iterate or declare its staleness. Truncate at first disclosure to stay on fact-free
   states.
5. **Never make the target a function of `F` alone.** An `F`-indexed target amplifies `F`
   mis-specification by `1/(1−p₀(F))` (Prop. 8.4) and makes correctness depend on a completeness
   property the access model cannot supply (Cor. 8.5). Prefer a target whose *shape* comes from an
   ignorance reference (A4) and use `F` only to express a constraint.
6. **Condition the edit on the subject–relation pair, not on the answer tokens.** The minimal
   separating statistic is `(subject identity, relation identity)`; answer-token information cannot
   separate Alice from Bob (Prop. 2.3). An unconditional mass bound with `ε < 1/|V|` zeroes a
   shared-token retain fact exactly (Thm 2.1, Cor. 2.2).
7. **Represent every preservation constraint as its own state family in the objective.** Shared-`F`
   and shared-template twins (Prop. 5.1), neutral-use states for P3, reverse-direction states for P5,
   and both partitions of §5.6. What is not represented is not constrained — it is left to inductive
   bias.
8. **Choose, declare, and report where the edit may live.** Trainable output embedding ⇒ global
   vocabulary suppression; frozen output embedding ⇒ learned query gate (Prop. 8.3). The objective
   must include a term or constraint that penalizes whichever one it has made available.
9. **Target the ignorance reference, not the reference's runner-up.** The derived ideal is a
   deconvolution (★); closing it with A2+A3 yields the exclusion target whose fixed point is a
   memorized replacement (Prop. 8.2). Closing it with A4 (verified-unknown-subject substitution) is
   computable under the access model and is the only available non-circular counterfactual.
10. **Build the two missing measurement instruments** before trusting any objective: the MDLM
    pseudo-log-likelihood `PLL_{θ₀}` (no perplexity exists for an MDLM) and the ignorance reference
    `P_ign`. Then add the five currently-missing detectors: `Î_item` (open-set refusal),
    canvas-level `eos_only`/post-`eos` content and floor-free repetition, the logit-level selectivity
    contrast `SEL` (no evaluator anywhere reads logits), ignorance-matching + mode-agreement `MAR`
    (replacement is undetected on every 8B stack), and the held-out-paraphrase gate `G_para`.
11. **Partition all scoring by `σ_F(s)`.** On answer-supplying prompts, token containment is
    meaningless (a bidirectional denoiser sees the prompt token at every position); score copying by
    fidelity and score endorsement by the endorsement contrast `EC` (§5.6). Nothing currently
    implements `EC`.
12. **Score multi-token objects by identifiability, not completeness.** `exact_match` scores
    "Berkshire" as 0 (false negative) and `span_recall` scores permuted pieces as 1.0 (false
    positive). With parallel commits, pieces are emitted out of order and non-adjacently, so
    contiguity-based span metrics are structurally wrong for an MDLM (§5.2).
13. **Declare `cl(φ)` as an input, and accept its consequence.** The algorithm cannot discover which
    retained facts entail `φ` (§3.3(a)); if any does and is retained, the adversary-general forgetting
    claim is false by Theorem 5.3. Either close the forget set under entailment or restrict the claim
    to the read-off adversary, explicitly.
14. **Accept and report the conjunction floor.** For `k ≥ 2` parallel commits, "forbid the conjunction,
    keep both parts" is infeasible: `Pr[joint] = p(A|s,i)·p(B|s,j)` and killing the product means
    killing a factor (Thm 5.5). Either restrict `Π` to `k = 1`, or report the config-dependent floor.
    Schedule intervention is forbidden by the access model.
15. **State the forgetting claim at a resolution the evaluation can reach.** A feasible battery
    certifies per-cell rates of order `10⁻²` (Cor. 6.2). Never report an `ε`- or
    fixed-point-leak number at `10⁻⁵` as a certified leakage rate; label them design quantities.
16. **Claim `ℛ`-inaccessibility, never deletion.** Certified deletion is provably outside the access
    model (Thm 6.4); behavioural equivalence does not imply recovery-equivalence (Thm 6.3:
    `0.023 → 0.605` held-out in 10 steps against a `0.023–0.046` never-memorized control). Report the
    recovery curve against that control, and do not use `‖∇L‖` as a robustness proxy — the
    gradient-flattening defense failed its own gate while the proxy moved 38 %, and recovery is
    directional, not distance-based.
17. **Report composition, never a single rate.** `Pr[¬disclose]` is satisfied by refusal,
    degeneration, suppression, replacement and gating alike. The reportable quantity is the joint
    `(disclosure, well-formed, ignorance-matched, selective, paraphrase-general, non-relearnable)`
    with every component gated separately, and the ignorance reference as the denominator for each.
