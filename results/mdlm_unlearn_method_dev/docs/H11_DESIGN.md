# H11 — MINIMAL-INTERDICTION UNLEARNING

**Owner:** objective and architecture design lead
**Date:** 2026-10-20
**Status:** Tier 2 (active development). Crux derivation + design + prerequisite measurements.
**I have run no GPU jobs.** Two CPU-only measurements are reported here, reproducible from
`scripts/h11_kmin.py` → `outputs/h11_kmin.json`.

**Naming collision, flag for the register:** `H10_DESIGN.md` §8.1 already uses the label `H11` for a
*failure condition* of H10 ("staleness"). This document's H11 is the new hypothesis. Rename one of
them before the register is updated; I suggest H10's failure row become `H10-F11`.

Status labels: **[P]** proved here, **[A]** assumption, **[H]** hypothesis needing measurement,
**[M]** measured.

---

## 0. VERDICT FIRST

Three results, in the order they bear on the decision.

**(1) The crux is a partial failure, and it fails in the worst available direction.** Interdiction
pressure does *not* propagate uniformly — it is local in a precise, derivable sense (Prop. 1.2).
But the locality factorizes as **(within-state activation similarity) × (unembedding-row similarity
of the two tokens)**, and *both* factors are maximal exactly for the co-answer tokens we want to
keep. `Pakistan` is dragged down in proportion to `⟨w_Pakistan, w_Karachi⟩`, which is larger than
random precisely because the two tokens are semantically adjacent. So H11 is not "ordinary
suppression in disguise" — it is **selective suppression whose selectivity points the wrong way**.
Whether that is fatal is a single measurable inequality (§1.4) and it is measurable before training.

**(2) `k = 1` is arithmetically impossible, and I measured it.** ASCR-12s is a *fractional recall
threshold*, `ASCR = |salient ∩ gen| / n_sal` with `τ = 0.5`. Interdicting `k` of `n_sal` stems moves
ASCR by exactly `k/n_sal`, so crossing τ requires `k > n_sal·(ASCR₀ − τ)`. Reimplementing the frozen
salience spec (and reproducing its published numbers to within 1 %), at a memorized `ASCR₀ ≈ 1.0`:

> **mean `k_min` = 5.07 stems, median 5, = 60.6 % of the salient set; 74.7 % of items need `k ≥ 4`.**
> `k = 1` crosses τ for **< 1 %** of items at any `ASCR₀ ≥ 0.7`.

**"Minimal interdiction" means a *majority* of the salient set, not the weakest link.** The premise's
stated instance — interdict `Karachi`, leave `Pakistan` — does not clear the bar on this metric. The
premise survives only in an attenuated form, which is still worth having: 5.07 stems is **14.7 % of
the mean 34.4 answer tokens** and **5.3 % of the 96-position canvas**, against 100 % for every
existing method. A ~7× reduction in pressure footprint, not a ~20× one.

The construct ("identifiability needs only the weakest link broken") is *right*; the **instrument**
(a 0.5 recall threshold over ~8 equally-weighted stems) does not implement it. That mismatch is
ASCR's, not H11's, and it is worth recording as a finding about the metric.

**(3) The failure mode H11 is predicted to escape is not the one it claims, because the floor has
moved.** The pristine TOFU base (never saw the data, n = 400) is **DISCLOSE 0.030 / DEGENERATE 0.000
/ ABSTAIN 0.1125 / REPLACE 0.470 / IGNORANT 0.3875** [M]. **REPLACE at 0.47 *is* what ignorance looks
like on TOFU.** So (i) H11's predicted landing in REPLACE is not a failure here, and (ii) the
objective cannot be "maximise IGNORANT" — a method reaching IGNORANT 0.9 would be *over-abstaining
relative to ignorance*. The objective is **profile matching** (§5.3), and the only cell that must
move is DISCLOSE.

---

## 1. THE CRUX: is subset interdiction local?

### 1.1 Setup

State `s` = (prompt, partially masked answer canvas). Teacher-forced on the gold answer, so position
↔ gold-token correspondence is exact. Let `i ∈ I_int` be an interdicted position holding gold token
`v_i`, and `j ∈ I_keep` a kept position holding `v_j`. The interdiction term puts a logit-space error
at `i` **only**:

> `e_i = p_θ(·|s,i) − r_int(·|s,i)`,  and `∂L_int/∂z_θ(·|s,j) = 0` for `j ≠ i` identically.

So, as in `H10_DESIGN.md` Prop. 2.3, there is **no direct gradient** at `j`. The question is the size
of the shared-parameter (NTK) transfer.

### 1.2 The transfer, factorized [P]

One step of plain gradient descent at rate `η` with `g = (∂z_i/∂θ)^T e_i`:

> **(1.1)** `Δz_j(v) = −η Σ_{v'} K_θ((s,j,v),(s,i,v')) · e_i(v')`

with `K_θ` the empirical NTK **across positions of the same state**. Evaluate it in the LoRA-B
subspace, which is where the project's gradient measurements live. For a wrapped linear
`h ↦ h + (α/r)BAh`, write `a_p = A x_p ∈ R^r` for the low-rank input activation at position `p` and
`u_{p,v} = (α/r)·∂z_p(v)/∂(layer output at p)`. Then `∂z_p(v)/∂B = u_{p,v} a_p^T`, and because
`⟨uv^T, u'v'^T⟩ = ⟨u,u'⟩⟨v,v'⟩`:

> **(1.2)** `K_θ((s,j,v),(s,i,v')) = ⟨a_i, a_j⟩ · ⟨u_{j,v}, u_{i,v'}⟩`  (summed over wrapped modules)

With the LM head frozen, `z = W h`, so `u_{p,v} ≈ J_p^T w_v` with `J_p` the Jacobian from the layer
output at `p` to the final hidden at `p`; for the upper blocks `J_p ≈ I` and
`⟨u_{j,v}, u_{i,v'}⟩ ≈ ⟨w_v, w_{v'}⟩`. Substituting into (1.1) with `δ_i := W^T e_i`:

> **Proposition 1.2 (interdiction locality) [P].**
> `Δz_j(v) ≈ −η · ⟨a_i, a_j⟩ · ⟨w_v, δ_i⟩`
>
> At a memorized state the interdiction target has little mass on `v_i`, so `δ_i ≈ w_{v_i}` and
>
> **`Δz_j(v_j) ≈ −η · ⟨a_i, a_j⟩ · ⟨w_{v_j}, w_{v_i}⟩`**
>
> i.e. **interdiction is local in two senses simultaneously** — it bleeds onto position `j` in
> proportion to the *within-state activation similarity* `⟨a_i,a_j⟩`, and onto token `v` in
> proportion to the *unembedding-row similarity* `⟨w_v, w_{v_i}⟩`. ∎

**Approximation named:** (1.2) is the *parameter-sharing* path only. A second path exists — position
`j`'s activations depend on position `i`'s through **attention**, in both directions, at fixed θ.
That path is not a weight-change effect and does not appear in (1.1); it means the realized drag can
exceed (1.2). Cost of the approximation: (1.2) is a **lower bound** on the drag, which is the
unsafe direction, so the measurement in §1.5 must be the arbiter rather than the algebra.

### 1.3 Why this is the worst available answer, and not a reassuring one

Both factors are maximal for the thing we want to preserve:

- `⟨a_i, a_j⟩`: `i` and `j` are positions in the **same answer of the same prompt**; they share the
  entire prompt context and most of the canvas. High by construction — this is the same quantity
  that drove `cos(a) ≈ 0.9` in `OBJECTIVE_DESIGN.md` §10.1, there measured *across* facts, here
  *within* a state, where it should be higher still.
- `⟨w_{v_j}, w_{v_i}⟩`: `Karachi` and `Pakistan`, `Historical` and `Fiction` — the kept salient
  tokens are the ones **semantically adjacent to the interdicted ones**, because they are
  co-constituents of one fact. Random unembedding rows have cosine ≈ 0.01–0.05; semantically
  adjacent rows are plausibly 0.2–0.4 (unmeasured on this checkpoint).

> **So the honest crux verdict is:** interdiction is *not* uniform suppression — the drag carries a
> factor `⟨w_{v_j}, w_{v_i}⟩ < 1` that uniform pressure does not have — but the surviving drag is
> **concentrated on exactly the co-answer tokens H11 promises to preserve.** H11 therefore does not
> get preservation for free from locality; it has to *buy* it with an explicit term (§2.2), and
> whether it can is the inequality below.

### 1.4 The feasibility inequality, stated so it can be checked

What matters is not the logit drag but whether the kept token is still emitted. At `T = 0` that is a
rank condition, and the protocol already has the right instrument — the **salient-position
teacher-forced margin** with its declared ±0.125-nat dead band:

> `M_j(θ) = log p_θ(v_j|s,j) − log max_{v ≠ v_j} p_θ(v|s,j)`

> **Condition H11-F.** Over the whole training run, for every kept salient position `j`:
> `Σ_steps |Δz_j(v_j) − max_v Δz_j(v)| ≤ M_j(θ_ref) − 0.125`.
> Equivalently: the accumulated drag must stay inside the evaluation's own dead band.

This is the condition H11 lives or dies by, it is denominated in the *same functional the evaluation
will use*, and it is measurable in **one gradient step** (§1.5).

### 1.5 H11-M0b — the decisive pre-training experiment (≈3 GPU-min)

> Take 20 forget items. Form `L_int` with pressure at `I_int` **only** (no anchor, no C1 term).
> Compute `g = ∇_θ L_int`. Take **one** step at the production LR. Measure `M` before and after at
> (a) `I_int`, (b) `I_keep` (the non-interdicted salient positions), (c) structural (non-salient)
> answer positions, (d) the same author's **retain** probes' salient positions.
> Report the **leakage ratio** `ρ = mean ΔM(I_keep) / mean ΔM(I_int)`, and the per-item scatter of
> `ΔM_j` against the predicted `⟨a_i,a_j⟩·⟨w_{v_j},w_{v_i}⟩` — which is a direct test of Prop. 1.2,
> not just of the premise.
>
> **Kill rule, pre-registered: `ρ ≥ 0.5` ⇒ the pressure is not local, H11 collapses into uniform
> suppression, and I will say so.** `ρ ≤ 0.15` ⇒ locality holds and the anchor of §2.2 has a
> plausible job. In between ⇒ the anchor is load-bearing and `λ_keep` must be tuned against H11-F.
>
> This overlaps registered **H8** (antiparallel forget/retain gradients, predicted cos ≈ −0.9,
> ~10 GPU-min, kill rule cos ≈ 0) — same backbone, same machinery. **Schedule them together.**

> **H11-M0a — the token-side factor, CPU only, zero GPU.** Load the frozen LM head
> (`model.transformer.ff_out`, 126464 × 4096) and compute, for all 400 forget items, the
> distribution of `⟨ŵ_{v_j}, ŵ_{v_i}⟩` over (interdicted, kept-salient) pairs, against a
> random-token-pair baseline. If the ratio to random is ≈ 1, the token-side leak is random-level and
> only `⟨a_i,a_j⟩` matters; if it is ≳ 5×, Prop. 1.2's bad direction is confirmed before any GPU
> time. (I could not run this: no `torch` in the environment I have.)

---

## 2. THE OBJECTIVE

### 2.1 The interdiction set

> **(2.1)** `salient(q,a) = { t ∈ stems(content_tokens(a)) : t ∉ stems(toks(q)), adf(t) ≤ 12 }`
> — the frozen ASCR-12s set (`EVAL_PROTOCOL_TOFU.md` §1.3), computed from the **gold** answer.
> `n_sal = |salient(q,a)|`.
>
> **(2.2)** `k_i = min( n_sal , ⌊ n_sal · (ASCR₀ − τ) ⌋ + 1 )`, `τ = 0.5` — item-adaptive, derived
> in §3, **not** a hyperparameter.
>
> **(2.3)** `I_int(s) = ` the positions of the `k_i` members of `salient(q,a)` of **lowest `adf`**
> (rarest first); `I_keep(s) = ` the positions of the remaining `n_sal − k_i` salient stems;
> `I_struct(s) = ` the remaining masked answer positions.

Because `salient(q,a)` is gold-derived and `adf` is a frozen table, **`I_int` is θ-independent** on a
teacher-forced state. That matters for §4.

### 2.2 The two terms, and which one is the contribution

> **(2.4)** `L_H11(θ) = (1/|I_int|) Σ_{i∈I_int} CE( r(·|s,i), p_θ(·|s,i) )`
> `+ λ_keep · (1/|I_keep|) Σ_{j∈I_keep} [ M_j(θ_ref) − M_j(θ) − Δ ]₊`
> `+ λ_struct · (1/|I_struct|) Σ_{j∈I_struct} CE( p_{θ_ref}(·|s,j), p_θ(·|s,j) )`
> `+ λ_r · L_retain`
>
> with `Δ = 0.125` nats (the evaluation's own dead band), `θ_ref` the retain-trained reference, and
> `r(·|s,i) = p_{θ_ref}(·|s,i)` — i.e. **C1's target, applied at `I_int` only.**

**The distinguishing operation is the second term**, not the first and not the selection. The first
term is C1 restricted to a subset; subset selection by position is *occupied* prior art (§7). What is
unoccupied is **asymmetric treatment within a single forget answer: suppress `k` salient positions
while explicitly holding the remaining salient positions at their reference margin.** Per the
prior-art sweep, every published selector (WGA/TNPO/WTNPO, SU, SPAN, FUnDIAL, CUSE) weights the
*fact* tokens and none anchors part of the forget answer; SU does the **opposite**, dilating the
selected token to a 5-gram *"to ensure the removal of complete information related to the token… the
whole phrase surrounding the token"*. H11 contracts where SU expands, and adds a within-answer
anchor nobody has.

**Why a hinge and not a KL.** A KL to `θ_ref` at `I_keep` would *also* pull those positions toward
the reference's (ignorant) distribution, destroying the kept content — the opposite of the intent.
The hinge is on the **margin**, is inactive while the margin is intact, and switches on only when the
drag of §1.2 threatens the rank. It is the same hinge discipline as A6, and it is denominated in the
evaluation's own preservation functional.

### 2.3 Why the interdiction target is C1's and not an exclusion target

Three candidate `r` at `I_int`: (a) renormalized/exclusion → mode is the runner-up → `Lahore`;
(b) uniform/MaxEnt → degeneracy; (c) the retain reference's own distribution → C1's target. We take
(c). By Prop. A1 of `H10_DESIGN.md` (at `T = 0` the sampler reads only the argmax, so the student
emits `argmax r`), (a) **forces** substitution and (b) **forces** degeneration. Only (c) puts the
emitted token under the control of a real, globally unmodified model. And §0(3) says that lands in
REPLACE at roughly the pristine rate, which on TOFU is the correct target, not a failure.

---

## 3. `k`: the arithmetic, measured

`ASCR = |salient ∩ stems(gen)| / n_sal`, so perfect interdiction of `k` stems gives
`ASCR = ASCR₀ − k/n_sal`, and `DISCLOSE` (which also requires `n_sal ≥ 3`) is cleared iff

> **(3.1)** `k > n_sal · (ASCR₀ − τ)`.

**Measured** (`scripts/h11_kmin.py`, CPU only; independent reimplementation of the frozen salience
spec, validated against the protocol's published figures — mean `n_sal` **8.01** vs published 7.94,
`n_sal ≤ 2` rate **0.090** vs 0.095, and the full `adf` sweep tracks 3.31/5.08/6.71/8.01/9.34/11.77
against 3.48/5.10/6.75/7.94/9.29/11.77):

| `ASCR₀` | mean `k_min` | median | mean `k/n_sal` | P(`k≥3`) | P(`k≥4`) | footprint of 34.4 answer tokens |
|---|---|---|---|---|---|---|
| **1.00** | **5.07** | 5 | **0.606** | 0.940 | **0.747** | **0.147** |
| 0.90 | 4.08 | 4 | 0.492 | 0.846 | 0.591 | 0.119 |
| 0.80 | 3.14 | 3 | 0.374 | 0.659 | 0.387 | 0.091 |
| 0.70 | 2.09 | 2 | 0.249 | 0.288 | 0.044 | 0.061 |
| 0.60 | 1.30 | 1 | 0.166 | 0.008 | 0.000 | 0.038 |

Scorable stratum (`n_sal ≥ 3`): **364/400 = 0.910**, mean `n_sal` 8.64, median 9, p90 14.

**Four consequences, all answering the brief's item 4:**

1. **`k = 1` is not enough and never close.** At any `ASCR₀ ≥ 0.7` it clears τ for < 1 % of items.
   The memorized backbone's `ASCR₀` must be measured (it is not yet, on the TOFU backbone) but a
   checkpoint gated at distinctive recall ≥ 0.60 will have `ASCR₀` high, so `k_min ≈ 4–5`.
2. **"Minimal" means a majority of the salient set** (60.6 % at `ASCR₀ = 1.0`). The premise's
   framing is quantitatively wrong on this metric, and the "pressure on 1–2 positions, not the span"
   argument for no-collapse does not hold.
3. **But the footprint is still small**: 5.07 stems ≈ 14.7 % of answer tokens, 5.3 % of the
   96-position canvas, versus 100 % for every existing method. The no-collapse argument survives in
   the weaker form "pressure on a seventh of the answer".
4. **`adf` rank is irrelevant to crossing τ.** ASCR weights every salient stem equally — `adf` only
   gates *membership*. So which `k` stems are chosen changes ASCR not at all; only the **count**
   matters. The rarest-first ordering in (2.3) is justified only by the *semantic* notion of
   identifiability and by the non-targeted heads. This has a sharp consequence for §6: **random-`k`
   must be the control**, and if it ties with salient-`k` the ranking has no content.

**A correction to the brief's worry.** "Interdicting one token may move items into a stratum rather
than out of disclosure" — it cannot. `n_sal = |salient(q,a)|` is computed from the **gold** answer,
so `LOW_SPECIFICITY` membership (`n_sal ≤ 2`) is a fixed property of the item and interdiction cannot
change it. Interdiction moves `ASCR` only. What *is* true and worse: the 9.0–9.5 % `LOW_SPECIFICITY`
items are **never pooled**, and 4 of the 7 genuine pristine disclosures live there — so H11 (like
every method) has a declared blind spot on the stratum enriched in true disclosure.

---

## 4. The mathematical object, classified

**On teacher-forced states: ordinary descent on a fixed, non-smooth scalar loss.** All three index
sets in (2.3) are derived from the gold answer and the frozen `adf` table, so they are
**θ-independent**; `k_i` from (2.2) is θ-independent once `ASCR₀` is measured once and frozen; the
hinge `[·]₊` is non-smooth but is a fixed function of θ. There is no stop-gradient, no θ-dependent
index set, no sequence of surrogates.

> **H11 is the only one of the three designs (C2/H10/H11) that is an ordinary fixed loss.** That is
> a real advantage for attribution and for cost: it needs no trajectory collection, no on-policy
> pool, no staleness declaration.

Two caveats that would break it, with their costs:

| choice | consequence |
|---|---|
| train on **on-policy** decode states instead | the canvas is model-written, so the stem→position map is no longer exact; `I_int` becomes θ-dependent and we are back to a surrogate sequence. **Recommendation: do not.** H11 does not need it. |
| make `k_i` adaptive **during** training (recompute from the current `ASCR`) | `I_int` becomes θ-dependent. **Recommendation: freeze `k_i` from the θ₀ measurement.** |

Cost per step: forget forward (grad) + reference forward (no-grad, `disable_adapter()`) + retain
forward (grad) + retain reference forward = **4 forwards, 1 backward** — one backward, unlike the
projection design. The margin hinge needs `M_j(θ_ref)`, which comes free from the reference forward
already required by C1.

---

## 5. The signature prediction, quantified

### 5.1 The decomposition ASCR cannot see — this is the signature

ASCR decomposes **exactly**:

> **(5.1)** `ASCR = (k/n_sal)·R_int + ((n_sal−k)/n_sal)·R_keep`

with `R_int` = fraction of *interdicted* stems recalled, `R_keep` = fraction of *kept* salient stems
recalled. Define the **selectivity gap**

> **(5.2)** `Δ_sel = R_keep − R_int`.

At `n_sal = 8`, `k = 5`, matched ASCR = 0.375:

| arm | `R_int` | `R_keep` | `ASCR` | `Δ_sel` |
|---|---|---|---|---|
| **H11 as advertised** | **0.00** | **1.00** | 0.375 | **1.00** |
| uniform pressure (C1 at all positions) | 0.375 | 0.375 | 0.375 | **0.00** |

**Identical ASCR, opposite decomposition.** `Δ_sel` is the signature, it is a single per-item scalar
computable from generations the partition already scores, and **the headline metric is blind to it.**
Prediction: `Δ_sel(H11) ≥ 0.7`, `Δ_sel(C1) ≤ 0.15`.

**Why this is the right statistical form.** The project's standing kill rule is that *"a per-fact
prediction is confirmed only by a within-stratum continuous correlation, computed before the group
comparison is reported"* — recorded after ARC (0.101 aggregate gap) and after Finding 1 (whose
continuous predictor came back `r = 0.014, p = 0.936`). `Δ_sel` is a per-item continuous statistic
over 364 scorable items in 20 authors, so it is **not** subject to the 0.125 MDD that binds cell
rates. The pre-registered test is the within-author distribution of `Δ_sel`, not a cell difference.

**What refutes it:** `Δ_sel ≈ 0` means the pressure propagated and the crux failed (§1.5 should have
caught it first). `Δ_sel ∈ [0.3, 0.7]` means partial locality and `λ_keep` is doing the work.
`R_keep` falling below its θ₀ value is condition H11-F violated.

### 5.2 Edit norm

Pressure footprint is `k/|masked answer positions|`. Under uniform `t ~ U(0,1)` the comparison arm
puts pressure at `≈ 0.5 · 34.4 ≈ 17` positions, H11 at `5.07`, a ratio of **0.29**. Prediction at
matched ASCR: `‖Δθ‖_F(H11)/‖Δθ‖_F(C1) ∈ [0.30, 0.55]` if the norm grows as `√(#terms)`, `[0.15,
0.30]` if linearly. I declare the range rather than a point because the scaling is not derivable.

**Required caveat: no edit Frobenius norm has ever been measured in this project.** The only such
number on file (E4, random perturbation at 2× the edit norm) is **withdrawn** — the probe perturbed
only the 128-module attention subspace holding 28.7 % of the edit energy. So the norms for every arm
must be computed fresh, over the **full** 224-module adapter, and it costs nothing (adapter
arithmetic on CPU).

### 5.3 The partition prediction, re-based on the pristine floor

Against the measured pristine base (DISCLOSE 0.030 / DEGEN 0.000 / ABSTAIN 0.1125 / REPLACE 0.470 /
IGNORANT 0.3875, n = 400):

> **Prediction 5.3.** H11 moves **DISCLOSE** from the memorized ≈1.0 toward 0.030 and leaves the
> other four cells within ±0.10 of the pristine profile. The reportable quantity is the
> **profile distance** `TV(profile(arm), profile(pristine))` over the six cells, **not** any single
> cell rate.
>
> Corollaries that distinguish it: DEGENERATE stays ≈ 0.000 (MDU/EMDU reach 1.000 [M]); REPLACE does
> **not** exceed pristine's 0.470 by more than the MDD; and the **modal-replacement concentration**
> across items stays low, because each item's interdiction target is its own reference distribution
> rather than a single off-`F` attractor — contrast the synthetic control's `Tbilisi ×19` and
> `seismoology ×8`.

This is also the honest answer to the brief's claim that H11 escapes both failure modes: **it escapes
DEGENERATE (which C1 already escapes) and it does not escape REPLACE — but on TOFU, REPLACE at ≈0.47
is ignorance, so "escaping" it was never the right goal.**

---

## 6. Failure conditions, observables, and the cheapest decisive experiment

| # | condition | observable / kill rule |
|---|---|---|
| **F1** | **Crux fails**: drag propagates to `I_keep`. | **H11-M0b**, `ρ ≥ 0.5` ⇒ dead. Pre-training, ≈3 GPU-min. |
| **F2** | **Token-side leak is large**: `⟨w_keep, w_int⟩ ≫` random. | **H11-M0a**, CPU, zero GPU. |
| **F3** | **`k_min` too large** — already measured: mean 5.07 = 60.6 % of the salient set. Not fatal; it reframes the premise and kills the "1–2 positions ⇒ no collapse" argument. | measured; `outputs/h11_kmin.json`. Needs `ASCR₀` on the TOFU backbone to fix `k_i`. |
| **F4** | **The anchor and the interdiction deadlock** — `λ_keep` large enough to hold `M_j` prevents `R_int → 0`. | joint plot of `R_int` vs `c_keep` violation rate over the `λ_keep` sweep; if no `λ_keep` gives `R_int ≤ 0.2` **and** `H11-F` satisfied, dead. |
| **F5** | **Circularity manufactures the win** (§6.1). | the three non-targeted heads **and** the random-`k` control must all agree. |
| **F6** | **Effect below the resolution of the benchmark tier.** K = 20 authors, MDD **0.125** on paired cell-rate differences. | state `Δ_sel` (continuous, per item) as primary and cell rates as secondary; any cell-rate claim below 0.125 is **not reportable**. |
| **F7** | **`LOW_SPECIFICITY` blind spot** — 9.0–9.5 % of items never pooled, and 4 of 7 genuine pristine disclosures live there. | report the stratum separately, declared; H11 cannot help there. |
| **F8** | **Gold-anchoring does not transfer.** `I_int` comes from the gold answer; for a paraphrased question the gold stems are unchanged (fine), but for an indirect or reverse-direction query the gold answer differs and `I_int` is undefined. | held-out paraphrase templates pass; indirect queries are a **declared scope limit**. |
| **F9** | **C1 has no base.** The frozen Phase-A2 cell gate says: run the six-cell partition on the retain reference's own forget generations; **REPLACE ⇒ do not build C1**. H11 inherits that gate wholesale. | Phase A2, already queued; `adapters/retain_ref_tofu` exists with no report yet. |
| **F10** | **Interdicted positions emit the runner-up anyway**, because C1's reference target at a *memorized* position may still have the gold token as its argmax if the reference knows the fact from register alone. | the reference's own `ASCR` on forget (the ignorance-ceiling gate, ≤ 0.15) already bounds this; report per-item `r(v_i|s,i)`. |

**Cheapest decisive experiment, pre-training:** **H11-M0b** (§1.5), ≈3 GPU-min, 20 items, one
gradient step, and it measures Prop. 1.2 directly rather than only its consequence. Paired with
**H11-M0a** (CPU, free) it decides both factors of the locality product before any arm is trained.
That is the H11 analogue of H10-M0, and like it, I am stating the kill rule in advance.

### 6.1 The circularity, and what it costs

`I_int` is selected from `salient(q,a)` — **exactly the token set ASCR-12s scores**. Selecting
interdiction targets by the metric's own statistic and then reporting that metric is training on the
test. Three things make this tractable rather than disqualifying:

1. **Decompose what is borrowed.** ASCR has two ingredients: the *set* and the *threshold on
   fractional recall*. H11 borrows the set. §3(4) shows the set's **ranking** is irrelevant to
   crossing τ — only the count is — so what H11 actually borrows is "which tokens the metric
   counts", which is the definition of the construct. **Any targeted method borrows this relative to
   its own metric**; the five published AR selectors all do.
2. **Pre-register heads H11 did not target**, and require the effect on all of them:
   - **`adf` threshold sweep** — re-score at `adf ≤ 4` and `adf ≤ 20`, which are *different token
     sets* (mean `n_sal` 5.08 and 9.34 vs 8.01). H11 targeted the `adf ≤ 12` set; if the gain
     survives on both neighbours it is not an artifact of the exact set.
   - **TOFU's own ROUGE-L recall** (`rouge_score`, recall, stemmer ON) — all tokens, no
     question-subtraction, no `adf` gate. Not targeted.
   - **The six-cell profile distance** (§5.3) — a judgement, not a recall threshold.
3. **The random-`k` control is the non-circular baseline**: same `k`, positions drawn uniformly from
   `salient(q,a)`. It is *not* selected by the ranking, the metric still scores it, and §3(4)
   predicts it ties with salient-`k` on ASCR. **If salient-`k` beats random-`k` on the
   non-targeted heads, the ranking has content; if it does not, the ranking adds nothing** — and the
   circularity objection dissolves along with the selection criterion.

**What it costs, stated plainly:** the ASCR number for H11 **cannot be the headline**. It must be
labelled as the *targeted* head, and the claim must rest on the non-targeted heads and on `Δ_sel`.
Note the project already carries a structurally identical problem one level up: TOFU's Forget Quality
*is* a KS test against the retain model's truth-ratio distribution, so **C1 trains on a surrogate of
its own metric too** — which is one of the reasons `Forget Quality` is declared N/A here.

---

## 7. Composition, and the ablation that isolates H11

### 7.1 With C1 — one operation

H11 **is** C1 with its forget pressure restricted to `I_int` plus the margin anchor on `I_keep`. The
single distinguishing operation is the **partition of the answer into {interdict, anchor, structural}
with asymmetric treatment**. Arms:

| arm | forget pressure applied at | `I_keep` treatment | isolates |
|---|---|---|---|
| **A. C1** | all masked answer positions | none | the base (prior art; carries no claim) |
| **B. C1 \| random-`k`** | a uniform random `k`-subset of `salient` | none | **the salience ranking** (§6.1(3)) |
| **B′. C1 \| all-salient** | all `n_sal` salient positions | none | **sparsity vs the anchor** |
| **C. interdiction only** | `I_int` | none | the anchor's necessity |
| **D. H11** | `I_int` | margin hinge | the candidate |

**The claim is `D > B` and `D > B′`.** `D > A` is not a claim — it would be attributable to "weight
the salient positions", which is occupied prior art. `C` vs `D` isolates the one element that is
genuinely unoccupied.

Matched across arms, non-negotiable: total forget-gradient mass (so B′ and D are not separated by
effective learning rate), target answer length, states, reference, retain data, optimizer, steps,
seeds, LoRA config, selection rule.

### 7.2 With H10 — **do not compose**

H10's index set is "positions currently proposing a fact token"; H11's is "the minimal salient
subset". These are **substantially the same positions** with **opposite treatment** — H10 raises the
*complement's* confidence to delay them, H11 suppresses the subset and anchors the complement's
margin. Composing them is two operations on overlapping index sets and attribution fails. They are
**competing designs for the same resource** and should be run as alternatives.

Two asymmetries worth recording for the choice between them:

- **H11 is cheaper and cleaner.** It works on teacher-forced states, so it is an ordinary fixed loss
  (§4) with 4 forwards / 1 backward and no trajectory collection. H10 requires on-policy states, a
  θ-dependent index set, and a declared staleness.
- **H10's premise has a free oracle test; H11's crux has a 3-minute gradient test.** Both can be
  killed for under 10 GPU-minutes, and they should be screened in parallel before either is built.

### 7.3 Novelty position, per the rubric

- **Position-selective unlearning: occupied.** Five published AR selectors (WGA/TNPO/WTNPO's
  continuous `p_θ^α`; SU's two-assistant disagreement + 5-gram dilation; SPAN's EK-FAC saliency +
  self-consistency; FUnDIAL's `𝟙_e` NER indicator; CUSE's counterfactual log-prob gap), plus SEUL.
  All weight the *fact* tokens. The prior-art sweep's verdict is explicit: *"select which positions
  to forget is NOT novel."*
- **Selection by corpus-level `adf` salience:** a different statistic for a known operation →
  **category (b)/(c)**.
- **Minimality to destroy identifiability:** the primitive (`ι(u) = 1/|{O' : u ⊏ Σ(O')}|`, R12)
  appears only in our own formalization; SU does the **opposite** (dilation to guarantee complete
  removal). Genuinely unoccupied as a *criterion*, but §3 shows the chosen metric does not reward
  minimality, so the criterion is not what produces any gain.
- **The within-answer asymmetric anchor: unoccupied.** No published method anchors part of a forget
  answer while suppressing another part (MDU: *"no KL anchor on retain"*; nobody partitions within
  the forget answer).
- **Honest standing: category (c) at the outset**, (e) at best on the MDLM delta; route to (d) only
  through `D > B` **and** `D > B′` with the interaction demonstrated and ≥3 seeds on the ablation
  cells. **The burden is on novelty.**

---

## 8. Measurements needed, in order

| # | job | cost | decides |
|---|---|---|---|
| **H11-M0a** | unembedding-row cosines for (interdicted, kept) pairs vs random | **CPU, free** | F2; half the locality product, before any GPU time |
| **H11-M0b** | one-step leakage probe, 20 items, `ρ = ΔM(keep)/ΔM(int)`; scatter against Prop. 1.2's prediction | **≈3 GPU-min** | **F1 — the crux.** `ρ ≥ 0.5` ⇒ dead. Schedule with registered H8 |
| **H11-M1** | `ASCR₀` per item on the TOFU memorized backbone (one decode pass already planned for Phase A) | free (rides Phase A) | fixes `k_i` via (2.2); without it `k` is a guess |
| **H11-M2** | `λ_keep` × `k` sweep on DEV, 2×2, `R_int` vs `H11-F` violation | ≈1 GPU-h | F4 |
| **H11-M3** | the non-targeted heads: `adf ∈ {4, 20}` re-score, TOFU ROUGE-L recall, six-cell profile distance | free (re-scoring) | F5, the circularity |
| **H11-M4** | full-adapter `‖Δθ‖_F` for every arm, 224 modules | **CPU, free** | §5.2; **no such number exists in the project today** |
| — | **blocking, inherited:** Phase A2 cell gate on the retain reference's own forget generations | already queued | F9 — if it reads REPLACE, C1 is not built and H11 has no base |

**Ordering: H11-M0a (free) → H11-M0b (3 min) → Phase A2 gate → H11-M1 → M2.** The first two cost
under 5 GPU-minutes and can kill the crux before anything is built.

---

## 9. What I could not make work

1. **`k = 1` is dead on arithmetic, measured.** Crossing `τ = 0.5` needs `k > n_sal(ASCR₀ − τ)`;
   at a memorized `ASCR₀` that is a *majority* of the salient set (mean 5.07 of 8.64). The premise's
   own worked example — interdict `Karachi`, keep `Pakistan` — does not clear the metric. I could not
   find a formulation that escapes this, because it is the shape of the instrument: a fractional
   recall threshold over equally-weighted stems cannot be crossed by breaking one link, whatever
   "identifiability" means semantically. **This is a finding about ASCR-12s, and it is worth
   recording as such: the metric does not implement the construct the brief reasons from.**
2. **Locality does not give preservation for free, and its selectivity points the wrong way.**
   Prop. 1.2's drag factorizes into `⟨a_i,a_j⟩·⟨w_{v_j},w_{v_i}⟩`, and both factors are maximal for
   co-answer tokens. The explicit anchor (2.4) is therefore **required**, not an optional refinement,
   and I could not construct a version of H11 that preserves `I_keep` by locality alone.
3. **H11 cannot escape REPLACE, and should not try.** At `T = 0` the sampler emits `argmax r`
   (`H10_DESIGN.md` Prop. A1), so the interdicted position emits *something* type-correct. The brief's
   "no replacement target for the span" does not hold: a per-position target has a mode and the
   decoder takes it. The repair is that **REPLACE ≈ 0.470 is the pristine TOFU behaviour**, so the
   objective is profile matching, not cell maximisation.
4. **The circularity is real and only partly mitigable.** It is not removable, because the metric
   defines the construct; it is reducible to a labelled, bounded cost via the three non-targeted
   heads and the random-`k` control. I note the project carries the same defect one level up: C1
   trains on a surrogate of TOFU's own Forget Quality.
5. **Nothing here can be resolved below 0.125** on paired cell rates with K = 20 authors. That is why
   `Δ_sel` — continuous, per item, and invisible to ASCR — is the primary, and it is the only form of
   the prediction that the standing per-fact kill rule accepts.
