# Results — relearning robustness for masked-diffusion unlearning

All numbers measured on this host, 2026-10-20, GPUs 2–3 only. Synthetic facts about
fictional entities; **not** a benchmark result. Read `../RELATED_WORK.md` first: **the
mechanism tested in Phase 2 is already published** (Fan et al., ICML 2025) and no novelty
is claimed for it.

The data is the 128-fact set (64 forget / 64 retain, 4 entities per answer) from the prior
sprint. Its memorization reference missed its own 0.80 gate at 0.797, so all forget metrics
are restricted to the **51 forget facts the reference actually recalled**; with 8 given to
the adversary, **43 held-out facts × 1 phrasing** are probed per checkpoint.

---

## Phase 0 — is masked-diffusion unlearning vulnerable to relearning? **Yes.**

Nobody had measured this: MDU and EMDU are the only two masked-diffusion unlearning papers
and neither tests robustness. Protocol is Deeb & Roger's **RTT** — the adversary gets 8 of
the 64 forget facts and runs plain masked-diffusion fine-tuning; disclosure is measured on
the **43 held-out facts it never saw**. Because the facts are independent fictional
bindings, the leaked 8 carry no information about the held-out 43, so recovery there is
**residual knowledge being unlocked, not learning**. The never-memorized model is the floor
that makes that inference valid.

| adversary steps `k` | θ_u (unlearned control) | base (never memorized) | gap |
|---|---|---|---|
| 0 | 0.023 | 0.000 | 0.023 |
| 2 | 0.046 | 0.023 | 0.023 |
| 5 | 0.256 | 0.023 | 0.233 |
| **10** | **0.605** | **0.023** | **0.582** |
| 20 | 0.651 | 0.046 | 0.605 |

**Pre-registered gate: AXIS REAL iff gap at k=10 ≥ 0.25. Measured 0.582 — cleared by
2.3×.**

The floor behaves exactly as required: `base` *does* learn the 8 facts it was given
(leaked-fact disclosure 0.000 → 0.625 → 0.875) while its held-out disclosure stays flat at
0.023–0.046. So the adversary demonstrably teaches what it is given and demonstrably does
not teach the held-out facts. θ_u's rise to 0.605 is therefore recovery.

**Finding: an unlearned masked-diffusion model at 0.049 single-shot disclosure returns to
0.605 on unseen forget facts after 10 fine-tuning steps on 8 leaked ones.** Low disclosure
is not evidence of removal in this setting.

Retain accuracy falls from 1.000 to 0.625 over the attack (and `base`'s rises 0.000 →
0.125), so the adversary also does collateral damage; the forget signal is far larger than
that drift.

---

## The vulnerability is about gradient direction, not parameter distance

Added after an accidental observation, then measured deliberately. A random LoRA
perturbation was applied with its effective update scaled to a fixed fraction of the
Frobenius norm of the **real unlearning edit** (‖ΔW‖_F = 11.586 over 224 modules), with no
optimiser and no training:

| perturbation ‖dW‖_F | relative to the unlearning edit | θ_u forget | θ_u retain | θ₀ forget |
|---|---|---|---|---|
| 0.00 | — | 0.031 | 1.000 | 1.000 |
| 2.90 | 0.25× | 0.031 | 1.000 | 1.000 |
| 5.79 | 0.5× | 0.031 | 1.000 | — |
| 11.59 | **1.0×** | 0.062 | 1.000 | — |
| 23.17 | **2.0×** | 0.031 | 1.000 | — |

**Undirected perturbation up to twice the norm of the entire unlearning edit does not bring
the fact back.** The θ₀ row is the necessary control: the same perturbation leaves a
*present* fact fully disclosed (1.000), so the probe is not merely insensitive.

This **refutes a hypothesis I had formed** — that θ_u is fragile, sitting a short hop from
the memorized solution — and it exposes a gap in my own Proposition 2: a robustness
*radius* is the wrong object, because the adversary's **direction** is what matters, not its
distance. Recovery requires gradient information about the forget set, not displacement.

---

## Phase 2 — does the gradient-norm mechanism work on the MDLM objective? **Yes, as a
mechanism.**

Stage-2 design forced by `theory/THEORY.md`:
- **P3a** — the penalty is satisfied at the memorized minimum (`‖∇L‖ = 0` there too), and
  from inside that basin it drives the fact all the way back (`L → 0.0000` for every
  moderate `λ_G`, verified). So it is applied **only** to an already-unlearned model, never
  from θ₀.
- **P3b** — raw ascent diverges in 40 dimensions, so the first term is an **anchor** to θ_u,
  not ascent.

Objective minimised: `λ_G·‖∇L_mem(θ)‖ + λ_A·KL(p_θ ‖ p_{θ_u})` on forget and retain states,
with the finite-difference surrogate of P4 (which penalises `‖g‖`, **not** `‖g‖²` — the two
are not interchangeable in the gradient and the code is labelled accordingly).

**Instrument.** `‖∇L_mem‖` is measured on **4 frozen batches with fixed masks**, because
the per-step value on a fresh random batch is dominated by the mask rate `t ~ U(0,1)` and
is not comparable across steps.

| step | `‖∇L_mem‖` (frozen probe) | `L_mem` (frozen probe) |
|---|---|---|
| θ_u baseline | **1.1547** | 0.9188 |
| 0 | 1.1174 | 0.9133 |
| 25 | 0.9628 | 0.9130 |
| 50 | 0.8753 | 0.9138 |
| 75 | 0.8206 | 0.9104 |
| 100 | 0.7852 | 0.9094 |

**The gradient norm falls ~32% by step 100 while the forget loss is held to within 0.005.**
That is the mechanism doing exactly what the theory says: flattening without moving the
loss value. Cost: 0.79 s/step vs 0.60 s/step for the anchor-only control (**1.32×**).

### The control is provably a no-op, and that matters

With `λ_G = 0` the objective is the anchor alone, which is identically zero at its own
initialisation — so under **SGD** its gradient is exactly zero and the model cannot move.
Verified: `‖∇L_mem‖_frozen = 1.154713` at steps 0, 5 **and** 9 — bit-identical. The
`λ_G = 0` arm therefore *is* θ_u, and Phase 0's θ_u curve is its relearning curve.

---

## Two bugs found and fixed, both of which had faked a result

Recorded because each produced a confident wrong signal before being caught.

1. **AdamW made the no-op control move.** The first stage-2 runs used AdamW; its
   per-coordinate normalisation turns a zero gradient into a step of ≈`lr` per coordinate,
   so the `λ_G = 0` control random-walked off θ_u. Switched to SGD, under which the control
   is exactly stationary (above).
2. **The guard compared incomparable quantities and fired on every arm.** `L_mem` was
   checked on each step's *random* batch against a baseline from batch 0. Since the mask
   rate is `t ~ U(0,1)`, that number swings 0.37 → 2.21 with no real drift, and the guard
   aborted all three arms at step 10 — including `λ_G = 0`, which cannot move at all. Both
   the baseline and the guard now read the frozen probe. **The early "guard fired"
   results were artefacts and are not evidence of P3a drift.**

A third, non-fatal misstep: the first stage-2 hyperparameters (grad clip 1.0, lr 5e-5) moved
the parameters at most 7.5e-3 over 150 steps against an unlearning edit of norm 11.6 —
0.07% of the relevant scale — so `‖∇L_mem‖` sat flat at ~1.13 and the mechanism looked
dead. It was a step-size failure, not a mechanism failure; lr 5e-3 was calibrated by a
30-step sweep before the main run.

---

## Phase 3 — does flattening actually slow the attack? *(in progress)*

The outcome that matters is the **downstream relearning curve**, not the gradient norm.
Song et al. ("Measure, Don't Optimize") report that optimising an audit proxy can teach a
model to hide from the audit while *increasing* post-attack recovery, so the gradient norm
is treated here as the mechanism check and never as the result.

Pre-registered verdict rules (from `../PLAN.md`):
- **SUPPORTED** — flattened held-out disclosure at k=10 is ≥ 0.20 below the control's,
  single-shot disclosure not worse by > 0.05, retain not worse by > 0.05.
- otherwise **NOT SUPPORTED**, reported with the numbers.

A win via output degeneracy does not count: clean forgetting is reported for every arm.

**Adversary-strength sweep is mandatory, not optional.** Deeb & Roger bypassed RepNoise —
a defense of this same family — with an extended learning-rate sweep, and Lemma 1 only
controls the *initial* descent rate; a step large enough to leave the `O(η²)` regime
escapes a first-order-flat point. So the flattened arm is run against adversaries at more
than one learning rate, and a robustness claim that holds at only one is reported as
holding at only one.
