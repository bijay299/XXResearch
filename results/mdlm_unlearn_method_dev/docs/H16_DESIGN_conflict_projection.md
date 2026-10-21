# H16 (contingency) — localized anchor matching with retain-conflict projection

**Status:** designed, not run. Gated on the low-λ comparison (see §6 decision rule).
**Date:** 2026-10-21
**Rests on:** `docs/FINDING14_h15_result.md`

---

## 1. The limitation, measured not assumed

Finding 14 established two things that together pick out this candidate:

1. **Position localization works.** The divergence selector moves the positions it selects
   (`p_true` −0.302) and leaves the rest alone (−0.015 → +0.015). Its complement carries nothing
   forgettable: the anti-control's `p_true` *rises* 0.606 → 0.733 and its entropy *sharpens* to
   0.853, and at generation level it reaches DISCLOSE 0.725 at retain 0.719 — i.e. no unlearning
   at all. The selector is validated.
2. **It does not protect retain.** Retain ASCR 0.199 against θ₀'s 0.762, with retain
   DEGENERATE 0.263. Retain queries are different *inputs*, not different positions.

And the retain-weight sweep closed the obvious alternative: `λ_retain ∈ {4,16}` does not protect
retain, it **destroys the forgetting** — `p_true` ends *above* θ₀ (0.675, 0.701, 0.757 against
0.606). That is the "buy retain by not forgetting" failure R2 forbids.

> So the damage transfers **across inputs through the shared parameters**. Selecting where in a
> forget sequence to apply the loss bounds its forget-side footprint and says nothing about the
> function change induced on inputs the loss never sees. A cross-entropy term on 400 retain
> answers does not cover that function space, and raising its weight only cancels the update.

## 2. Operation

Act on the **update direction** rather than on the loss's support. At each step:

```
g_f = ∇θ L_forget      (localized anchor KL, exactly H15's loss — the validated part)
g_r = ∇θ L_retain      (cross-entropy on a retain minibatch; already computed by H15)

maintain R = orthonormal basis of the last k retain gradients   (rank-k retain subspace)
g_f⊥ = g_f − R Rᵀ g_f                                            (remove the retain-conflicting part)
step on  g_f⊥ + λ_r · g_r
```

At first order, a step along `g_f⊥` leaves the retain loss unchanged while still reducing the
forget loss by `⟨g_f, g_f⊥⟩ = ‖g_f⊥‖²`. The forgetting that survives projection is exactly the
forgetting that does **not** require moving retain — which is ATWU's retain-conflict criterion
(arXiv 2606.06320 Eq. 2) applied in **parameter space** rather than at the token level.

**Why rank-k and not rank-1.** Projecting one direction out of a 42 M-dimensional LoRA space
removes essentially nothing, and a single minibatch gradient is a noisy estimate of the retain
direction. `R` is therefore built from the last `k` retain gradients, re-orthonormalized
(`k ∈ {1, 8, 32}` to be swept), so `k=1` is included as the degenerate case and the sweep shows
whether rank matters. This is the design's main risk and it is instrumented, not assumed.

**Variants to compare, one flag apart:**

| arm | update |
|---|---|
| `h15` | `g_f + λ_r g_r` — Finding 14's arm, the baseline for this comparison |
| `h16_proj` | `g_f⊥ + λ_r g_r` — hard projection against the rank-k retain subspace |
| `h16_pcgrad` | project only when `⟨g_f, g_r⟩ < 0` (PCGrad rule), else leave `g_f` |
| `h16_anti` | project onto `R` instead of off it — **attribution control**; should forget little and preserve retain trivially |

## 3. Pre-registered predictions and kill rules

To be fixed in a `PREREG_H16` document before the first arm runs, with the same thresholds as
H15 so the two are directly comparable:

| # | prediction | kill condition |
|---|---|---|
| **Q1** | retain ASCR ≥ **0.40** | < 0.40 → parameter-space projection does not protect retain either |
| **Q2** | DISCLOSE ≤ **0.15** and DEGENERATE ≤ **0.25** | either exceeded → forgetting was bought away, same failure as the λ_retain sweep |
| **Q3** | measured retain-gradient alignment `⟨ĝ_f, ĝ_r⟩` falls toward 0 over training | if it does not, the projection is not doing what it claims and Q1/Q2 are uninterpretable |
| **Q4** | `h16_anti` fails Q1+Q2 together | if the anti-projection matches, the effect is not conflict removal |

Q1 and Q2 must hold **together**, and the comparison must be made **at matched forget quality
against MDU on the same λ grid**, as a curve — not at one λ.

Q3 is the mechanism instrument and is the reason this is worth running even if Q1 fails: it
measures directly whether forget and retain gradients are *geometrically separable* at all in
this model. If alignment cannot be driven toward zero, that is a strong negative result about
LoRA unlearning on MDLMs and it closes a whole class of methods, not just this arm.

## 4. Cost

One extra backward pass per step (the retain gradient is already computed for the loss; it just
has to be taken separately rather than summed), plus a `k × 42 M` basis held in fp32 — 1.3 GB at
`k=8`, which fits alongside the 19.5 GiB training peak. Re-orthonormalization is `O(k²·d)` and
negligible. ~0.9 s/step expected against H15's 0.59 s/step.

## 5. Novelty position, stated honestly

Gradient projection and conflict removal are **standard** in continual learning and multi-task
optimization (orthogonal gradient descent, GEM/A-GEM, PCGrad, O-LoRA). Nothing about the
projection machinery is new, and the write-up must say so plainly.

What is not already published, as far as the primary-source sweep found: pairing it with an
**anchor-matching** forget loss on a **masked diffusion LM**, localized by the
**conditional-vs-anchor divergence** selector validated in Finding 14 — and, more importantly,
the *reason* for doing so is a measurement (Finding 14 §4) rather than an analogy to continual
learning. ATWU addresses retain conflict at the token level with a learned scorer; this
addresses it in parameter space with a closed-form signal. The two are composable, not rival,
and ATWU remains a required baseline either way.

**If H16 works, the contribution is the diagnosis plus the combination, not the projection.**
That framing is weaker than "new objective" and is the correct one.

## 6. Decision rule

The low-λ sweep (`queue_lowlam.sh`) is testing H15 and MDU at `λ_forget ∈ {1,2,3}`, the region
where retain survives — MDU at `λ_forget=2` reaches DISCLOSE 0.225 / DEGENERATE 0.092 / retain
ASCR **0.403**, the best point any arm has reached, and every H15 arm so far sat at
`λ_forget ∈ {4,8,16}` where both arms destroy retain.

- **If H15 clears R1 and R2 together in that region**, and beats MDU at matched forget on the
  curve, H16 is not needed and effort goes to seeds, ATWU, and the money table.
- **If H15 clears neither**, H16 is registered and run, because the diagnosis says the remaining
  leak is in parameter space and Q3 makes the run informative whichever way it falls.
- **If H15 matches MDU but does not beat it**, the honest conclusion is that localization buys
  nothing at usable operating points, and that is recorded as a negative against the mechanism
  that Finding 14 validated — a result worth publishing on its own.
