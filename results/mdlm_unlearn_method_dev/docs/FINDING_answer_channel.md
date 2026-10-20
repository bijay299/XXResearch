# Finding 1 — CORRECTED. The causal claim is retracted; the structural observation stands

**Status: the causal attribution in the first version of this document is WITHDRAWN.**
Published 2026-10-20 in commit `94b3a3d8a6fe` and retracted the same day after two
independent agents objected and a decisive experiment confirmed them. The original text is
preserved at the end of this file.

---

## What I claimed, and why it was wrong

**Claimed:** the exclusion target removes probability mass from the answer-token set `F`;
two facts sharing an answer share the same `F`; therefore *the objective's F-indexing drives
their gradients into near-collinearity* (0.388 → 0.916), which is the leak channel.

**Two independent refutations, arrived at separately:**

1. *Analytical.* At LoRA init the per-fact gradient in the `lora_B` subspace is a sum of
   outer products `(A x_t) ⊗ δ_t` with `δ = Jᵀ e`, `e = p_θ − onehot(v*)`. So
   `cos(g, g') = cos(δ, δ') · cos(Ax, Ax')` exactly. For a **suppressed** fact `p_θ(v*) ≈ 0`,
   so `δ ≈ −Jᵀ w_{v*}` — the unembedding row of the answer token, at near-maximal magnitude.
   Two facts with the same answer share that row, so `cos(δ) → 1` **whatever produced the
   suppression.** For a fitted fact `e ≈ 0` and the direction is small and unstructured.
   The whole observed ordering follows: 0.92 (both suppressed) > 0.64 (both fitted) > 0.26
   (mixed), collapsing to a common ≈0.62-0.66 at θ₀ where every fact is fitted (fp32
   values; see below).
2. *The control I already had and did not use.* At θ₀ the same-answer coupling is
   indistinguishable across all three groups, and just as strong on **retain facts the edit
   never touched.** I used exactly this control to kill H1 ("retain-retain is of the same
   order, so it is a property of the loss and parameterization") and then failed to apply it
   to the hypothesis that replaced H1.

### fp32 recomputation — and my bug was inflating my own effect

The published θ₀ figures came from a run that cast gradients to **fp16**; at θ₀'s mean
gradient norm of 4.3e-6, against fp16's smallest subnormal of 5.96e-8, roughly 99.98% of the
23.07 M coordinates were unrepresentable (evidenced by 116 exactly-zero off-diagonal
cosines at θ₀ and none at θ_u). Recomputed in fp32
(`outputs/grad_cosine_{theta_0_fp32,theta_u_fp32}.json`):

| cell | θ₀ (fp16, as published) | **θ₀ (fp32)** | **θ_u (fp32)** |
|---|---|---|---|
| forget-forget, same answer | +0.3876 | **+0.6391** | **+0.9159** |
| retain-retain, same answer | not computed | **+0.6590** | +0.6440 |
| forget-retain, same answer | not computed | **+0.6202** | **+0.2612** |
| forget-forget, same relation | +0.1263 | +0.1805 | +0.1041 |
| forget-forget, different answer | +0.0596 | +0.0994 | +0.0533 |
| forget-forget overall | +0.0648 | +0.1079 | +0.0670 |
| retain-retain overall | +0.0463 | +0.0817 | +0.0561 |
| forget-retain overall | +0.0412 | +0.0701 | +0.0186 |
| gradient-set effective rank | 33.46/64 | **21.86/64** | 29.48/64 |

**Three corrections, all of which cut against the original claim:**

- **The "×2.36 amplification" headline was an artifact of my own fp16 bug.** The correct
  figure is 0.639 → 0.916, i.e. **×1.43**. The numerical defect inflated the effect I was
  claiming.
- **At θ₀ the three same-answer groups are indistinguishable** — 0.639 (forget-forget),
  0.659 (retain-retain), 0.620 (forget-retain). This is the critic's F1 confirmed directly
  at full precision: the coupling is intrinsic, and strongest of all on retain-retain pairs
  the edit never touches.
- **A new observation that runs opposite to the retracted story.** Forget-retain same-answer
  coupling *falls* from **0.620 to 0.261** under unlearning. The objective **decoupled**
  forget facts from their retain twins rather than coupling them — which is consistent with
  measured retain accuracy holding at 1.000, and is the reverse of a collateral-damage
  mechanism. Effective rank also *rose* (21.9 → 29.5), not fell.

So at full precision: unlearning raises forget-forget coupling (×1.43), leaves retain-retain
flat (0.659 → 0.644), and more than halves forget-retain coupling. Only the first of those
three was reported, and it was the one the bug exaggerated.

## The decisive experiment

`grad_cosine.py` at θ_u for two published baselines that are **not** answer-token-indexed,
both driven to disclosure 0.000 (`outputs/grad_cosine_{emdu150,mdu150}.json`). If F-indexing
caused the coupling, these should not show it.

| arm | F-indexed | same-answer | retain same-answer | diff-answer | **selectivity** | eff. rank |
|---|---|---|---|---|---|---|
| exclusion target (ours) | **yes** | **+0.916** | +0.644 | +0.053 | **17.2×** | 29.5/64 |
| EMDU, 150 steps | no | +0.773 | +0.753 | +0.198 | 3.9× | 10.5/64 |
| MDU τ=1, 150 steps | no | +0.573 | +0.575 | +0.414 | 1.4× | 4.5/64 |

**Both non-F-indexed methods show substantial same-answer coupling.** The causal claim is
dead. Suppression produces the coupling; the target's indexing is not required.

The forget-vs-retain contrast is the clincher. For EMDU and MDU the two are equal
(0.773/0.753 and 0.573/0.575). Only in our arm do they diverge (0.916 vs 0.644) — and that
is explained by *differential suppression*: our arm drives forget disclosure to 0.031 while
holding retain at 1.000, so forget residuals are large and retain residuals small. EMDU and
MDU degrade retain too (0.906 and 0.969, with 17 output failures each), so both groups carry
large residuals and couple equally. No appeal to F-indexing is needed anywhere.

## What survives

1. **Same-answer coupling is the dominant structure in per-fact gradient geometry**, in every
   method measured, at 1.4× to 17.2× the different-answer baseline. It is a property of
   suppression and of the parameterization, not of any particular objective.
2. **Selectivity differs sharply between methods** and is a new, cheap descriptor:
   17.2× (ours) vs 3.9× (EMDU) vs 1.4× (MDU). MDU's near-uniform coupling (diff-answer
   0.414, effective rank 4.5/64) says its gradients are *globally* collapsed — consistent
   with its output collapse, and a mechanistic correlate of the degeneracy we measured
   behaviourally. This is a description, not yet a claim about anything causal.
3. **The θ₀ → θ_u measurements themselves**, with the caveat in the next section.

## What is withdrawn

- **The causal story** (F-indexing → coupling → leak). Retracted outright.
- **"Unlearning creates the channel."** It amplifies an intrinsic coupling by suppressing
  facts; so does every method tested.
- **The behavioural confirmation, as stated.** Non-sharing recovery at k=10 by relation is
  field 12/12, home_city 6/14, **instrument 0/9** — and 9 of 9 non-sharing instrument facts
  emit a *leaked* instrument rather than their own (`timpani` ×7, `clarinet` ×2); one
  "recovered" generation reads `"…plays the timpani, the timpani, the timpani…"`. The attack
  injects leaked answers across a whole relation, and the 3 "sharing" instrument facts score
  as recovered because their reference *is* one of the injected answers. Fisher one-sided:
  as published 8/8 vs 18/35 → p = 0.011; dropping the instrument stratum → 5/5 vs 18/26,
  **p = 0.198**; Bonferroni over the 5 reported `k` values puts the uncorrected figure at
  0.054. **Only home_city survives uncontaminated: 4/4 vs 6/14, p = 0.069.** Suggestive, not
  confirmed.
- **The continuous per-fact prediction.** Within the 35 non-sharing facts, the correlation
  between max gradient cosine to the leaked facts and recovery at k=10 is **r = +0.014,
  p = 0.936** (across all 43 it is r = +0.384, p = 0.011, but that is the binary same-answer
  indicator re-expressed). The cosine adds nothing over string equality of answers, so it
  cannot distinguish gradient geometry from the template-level answer-injection mechanism
  that is *directly observed* above.
- **A numerical defect in my own script.** Gradients were cast to **fp16** before the inner
  products. fp16's smallest subnormal is 5.96e-8, and at θ₀'s mean gradient norm of 4.3e-6
  only ~0.02% of the 23.07 M coordinates were representable — evidenced by **116 exactly-zero
  off-diagonal cosines at θ₀ and none at θ_u**. Fixed to fp32; the θ₀ figures above are from
  the fp16 run and are **not quotable** pending recomputation (the direction of the error is
  deflation, so the conclusion is robust but the numbers are not).

## H1's status, revised again

The critic argues H1 was over-refuted, and the arithmetic supports that:
`‖Σ of 8 leaked unit gradients‖ = 3.213`, with `cos(Σ, g_j)` = 0.402 for leaked facts versus
**0.119** for non-sharing held-out facts — so non-sharing facts receive ≈**29.7%** of the
first-order loss reduction a leaked fact gets. Effective rank 29.5/64 is also a large
deviation from isotropy (64 iid unit vectors give 63.98; top-10 energy 42.2% vs 15.6%
expected). And 29.5 ≈ 32 = the number of distinct answers — I used that number as evidence
*against* a shared direction and then as the mechanism.

**H1 is therefore recorded as REFINED, not refuted**: there is no single global direction,
but the gradient set is far from isotropic and transfer of ~30% per step is real.

## Method consequence

The design does **not** rest on the retracted causal story. Preservation is addressed by a
first-order-exact projection onto the retain-gradient cone, whose guarantee is an identity
about `⟨v_j, u⟩ ≥ 0` and is untouched by any objection here. The measured forget-retain
same-answer coupling of 0.260 enters only as the quantity that sets the projection's cost
(`1 − cos² ` = 93.2% of the forgetting rate retained). The same arithmetic closes the
orthogonalization variants of H1: at 0.916, forget-forget orthogonalization would retain
only 16%.

Same-answer coupling is retained as a **descriptive mechanism check**, not as a target —
and explicitly not as evidence of a causal mechanism.

---

## Appendix: the original text, preserved

The first version of this document claimed: *"The coupling is carried by the shared answer
token… unlearning CREATES the channel… every structural quantity is essentially unchanged
except the same-answer coupling, which more than doubles… This completes a causal account:
(1) the exclusion target removes probability mass from the answer-token set F; (2) two facts
with the same answer have the same F; (3) so the objective drives their gradients into
near-collinearity (0.388 → 0.916); (4) hence an adversary recovering one fact recovers all
its same-answer partners (1.000 vs 0.514 at k=10)."*

Steps (1)–(3) do not follow, for the reasons above. Step (4) is contaminated by answer
injection in one relation stratum. The document also asserted the finding was *"reported
with the same standard I failed to meet in the ARC study"* — it was not: ARC's error was
presenting an aggregate gap as a per-fact confirmation, and this document presented an
8-vs-35 group difference as a per-fact confirmation while its own continuous predictor had
r = 0.014.
