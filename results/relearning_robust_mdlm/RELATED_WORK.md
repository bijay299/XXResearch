# Related work, and what this study does NOT claim

A literature check was run **in parallel with Phase 0, before the method was built**,
precisely to avoid re-deriving something standard. It found the core mechanism already
published. This file records that honestly and up front, because the finding changes what
the study is allowed to claim.

## The mechanism I proposed is already published

**Fan, Jia, Zhang, Ramakrishna, Hong, Liu. "Towards LLM Unlearning Resilient to Relearning
Attacks: A Sharpness-Aware Minimization Perspective and Beyond." ICML 2025.**
[arXiv 2502.05374](https://arxiv.org/abs/2502.05374) ·
[code](https://github.com/OPTML-Group/Unlearn-Smooth)

Their paper contains, as named variants:

| What I proposed | Where it already exists |
|---|---|
| Model the relearning adversary as a weight-space min-max over `‖δ‖ ≤ ρ` | their formulation of robust unlearning as SAM on the forget loss |
| Derive a gradient-norm penalty by Taylor expansion of that min-max | their eq. (6) |
| `ℓ_f(θ) + ρ‖∇_θ ℓ_f(θ)‖₂` as the objective | their eq. (10), variant **GP** |
| Exact Hessian-vector-product implementation | their eq. (7) |
| Finite-difference surrogate `[∇ℓ(θ+μv) − ∇ℓ(θ)]/μ` | their eq. (8), variant **CR** |
| "drives convergence to a stationary point, reducing curvature, which improves resilience" | their stated conclusion, in those words |

My `theory/THEORY.md` Lemma 1 and Proposition 2 are therefore **exposition of a derivation
already in print**, not a contribution. They are kept because they are what the sprint
actually reasoned from, and because P3a/P3b (below) are not in that paper — but the
*mechanism* is Fan et al.'s and is cited as such throughout.

Re-derivations of the same penalty appear in **PRISM** (Yan et al.,
[2509.23362](https://arxiv.org/abs/2509.23362)) and **SARE** (Fang et al.,
[2601.16527](https://arxiv.org/abs/2601.16527)); SARE also states explicitly that the
regulariser depends on gradient magnitude rather than loss magnitude, which is my Lemma 1.
**TAR** (Tamirisa et al., ICLR 2025, [2408.00761](https://arxiv.org/abs/2408.00761)) states
the conceptual target verbatim — *"we seek to flatten the adversary's loss at a high value,
rather than simply raise its y-intercept"* — reached by meta-learning rather than a
penalty.

**Consequence: this study does not propose a new unlearning method, and makes no novelty
claim for the gradient-norm mechanism.**

## The evaluation protocol I built independently is also the standard one

My Phase-0 probe — hand the adversary a subset `T` of the forget set, measure recovery on
the held-out complement `V`, with a never-memorized model as the floor — is **RTT**
(Retraining on T) from **Deeb & Roger, "Do Unlearning Methods Remove Information from
Language Model Weights?"** ([2410.08827](https://arxiv.org/abs/2410.08827)). The
never-memorized floor matches **"From Dormant to Deleted"** (Siddiqui et al.,
[2505.22310](https://arxiv.org/abs/2505.22310)), which uses retrain-from-scratch as the
floor and reports the same qualitative jump I measured.

Converging independently on the field's protocol is reassurance that the measurement is
sound. It is not a contribution either.

Related: **Hu et al., "Jogging the Memory of Unlearned LLMs Through Targeted Relearning
Attacks," ICLR 2025** ([2406.13356](https://arxiv.org/abs/2406.13356)); **Lynch et al.,
"Eight Methods to Evaluate Robust Unlearning in LLMs"**
([2402.16835](https://arxiv.org/abs/2402.16835)).

## Two objections from the literature that this study must answer

**1. Deeb & Roger: defenses of this family are bypassed by a hyperparameter sweep.** They
RTT'd RepNoise with an extended learning-rate/epoch sweep and recovered accuracy as high
as fine-tuning the original model, concluding that *"fine-tuning robustness techniques can
be bypassed when using an extensive hyperparameter search."* My Lemma 1 controls only the
*initial* descent rate; a step large enough to leave the `O(η²)` regime escapes a
first-order-flat point, and SGD noise alone escapes a saddle. **So a single-learning-rate
adversary is not a fair test**, and an adversary-strength sweep is required before any
robustness claim. This is now a pre-registered part of Phase 3 rather than an afterthought.

**2. Song et al., "Measure, Don't Optimize"** ([2608.11408](https://arxiv.org/abs/2608.11408))
report that optimising an audit score directly *"does not promote genuine deletion.
Instead, the model learns to hide knowledge from the audit, producing lower audit scores
but greater post-attack recovery."* That is a direct Goodhart objection to minimising
`‖∇L_mem‖` as a robustness surrogate. It is the reason this study reports the **downstream
relearning curve**, not the gradient norm, as the outcome — the gradient norm is only the
mechanism check.

**3. Opposite-sign prescription.** **Sharp MinMax** (Tang & Khanna, ICLR 2026,
[2506.13715](https://arxiv.org/abs/2506.13715)) applies sharpness *maximisation* on the
forget side, targeting the utility/forgetting trade-off rather than relearning robustness.
Worth distinguishing, not reconciling — different objective.

## What is actually unoccupied

Masked-diffusion / diffusion-LLM unlearning has, by targeted search, **exactly two
papers** — **MDU** (Lee et al., [2605.18253](https://arxiv.org/abs/2605.18253)) and
**EMDU** (Garber et al., [2610.25447](https://arxiv.org/abs/2610.25447)) — plus knowledge
*editing* work (**TimeROME-DLM**, [2606.12841](https://arxiv.org/abs/2606.12841)).
**Neither MDU nor EMDU evaluates robustness to relearning or fine-tuning attacks.**

So the questions this study can legitimately ask are:

1. **Is masked-diffusion unlearning vulnerable to relearning at all?** Nobody has measured
   it. The forget objective here is a KL to a masked anchor sampled over timesteps, and
   generation is bidirectional iterative unmasking, so the autoregressive relearning
   picture does not transfer by assumption.
2. **Does Fan et al.'s smoothness family transfer to the MDLM denoising objective?** A
   transfer-and-adaptation study, framed and cited as such.
3. **Is the vulnerability about parameter distance or gradient direction?** The noise probe
   in this study (undirected perturbation up to 2× the norm of the whole unlearning edit)
   was added to separate those, and is not something the cited papers run.

Two findings in `theory/THEORY.md` also appear to be specific to this setting rather than
drawn from the cited work, and are reported as observations, not claims of novelty:
**P3a** (the penalty's spurious attractor at the memorized solution makes it unsafe as a
one-stage objective from `θ₀`, and forces a second-stage design) and **P3b** (high loss and
zero gradient coincide only in one dimension, so the first term must be an anchored loss
rather than raw ascent).

## Not verified

Łucki et al., "An Adversarial Perspective on Machine Unlearning for AI Safety" — cited
widely as a canonical "unlearning is shallow" result; not fetched directly. Tran & Woo,
[2604.09391](https://arxiv.org/abs/2604.09391) (relearning-convergence-delay) — abstract
only. The broader cluster (RepSelect, CIR, MUDMAN, OPC, WARP, SAUL, StableUN) was listed
but not read in depth. Keyword search over abstracts will have missed papers that discuss
diffusion LMs without those phrases in the abstract.
