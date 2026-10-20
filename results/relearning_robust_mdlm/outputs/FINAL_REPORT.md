# Relearning robustness for masked-diffusion unlearning

**Fresh sprint, 2026-10-20, started 10:05 ET. GPUs 2–3 only. All synthetic data.**

> **Headline.** Three results, one of them negative about the thing I set out to build.
>
> 1. **Masked-diffusion unlearning is vulnerable to relearning, and nobody had measured
>    it.** A model at 0.049 single-shot forget disclosure returns to **0.605** on 43
>    forget facts the adversary never saw, after **10 fine-tuning steps on 8 leaked
>    facts** — against a never-memorized floor of 0.023.
> 2. **The vulnerability is directional, not positional.** Random perturbation of up to
>    **2× the Frobenius norm of the entire unlearning edit** leaves disclosure at 0.031,
>    while the same perturbation leaves a *present* fact at 1.000. Recovery needs gradient
>    information about the forget set, not parameter displacement.
> 3. **The published gradient-norm defense transfers as a mechanism but not as a
>    defense.** It reduced `‖∇L_mem‖` by **39.9%** at unchanged forget loss and bought
>    **0.024** — one probe in 43 — against the attack, where the pre-registered bar was
>    0.20. **NOT SUPPORTED.**
>
> **No novelty claim.** The mechanism in (3) is Fan et al., ICML 2025 (variant GP/CR). The
> protocol in (1) is Deeb & Roger's RTT. See `../RELATED_WORK.md`, written before the
> method was built.

---

## 1. Why this axis

The previous sprint's own numbers closed the obvious one. On the same 128-fact data the
existing anchor-KL control already reached **0.049 disclosure, 1.000 retain, 0 degenerate
outputs** — near-saturated. There was no headroom for "a method that forgets more," so a
new method needed an axis where the strong control *measurably fails*.

Per the standing rule from the last three failures — *every candidate looked good only on a
probe too small to break it* — **the probe was built and the axis proved real before any
method was written.**

## 2. Phase 0 — the vulnerability (new measurement)

Masked-diffusion unlearning has exactly two papers (MDU, EMDU) and **neither evaluates
robustness to relearning**. Protocol: the adversary receives **8 of the 64 forget facts**
and runs plain masked-diffusion fine-tuning; disclosure is measured on the **43 held-out
recalled forget facts it never saw**. Because the facts are independent fictional
entity→answer bindings, the leaked 8 carry **no information** about the held-out 43, so
recovery there is residual knowledge unlocked, not learned.

| adversary steps `k` | θ_u (unlearned) | base (never memorized) | gap |
|---|---|---|---|
| 0 | 0.023 | 0.000 | 0.023 |
| 2 | 0.046 | 0.023 | 0.023 |
| 5 | 0.256 | 0.023 | 0.233 |
| **10** | **0.605** | **0.023** | **0.582** |
| 20 | 0.651 | 0.046 | 0.605 |

**Pre-registered gate: AXIS REAL iff gap at k=10 ≥ 0.25. Measured 0.582, cleared by 2.3×.**

The floor validates the inference: `base` *does* learn its 8 given facts (leaked disclosure
0.000 → 0.625 → 0.875) while its held-out disclosure stays flat at 0.023–0.046. The
adversary provably teaches what it is given and provably does not teach the held-out facts.

Collateral: retain falls 1.000 → 0.625 over the attack (base's rises 0.000 → 0.125), so the
adversary damages the model generally; the forget signal is much larger than that drift.

## 3. The vulnerability is about direction, not distance

A random LoRA was applied with its effective update scaled to a fraction of the real
unlearning edit's norm (‖ΔW‖_F = 11.586 over 224 modules) — no optimiser, no training:

| ‖dW‖_F | relative to the unlearning edit | θ_u forget | θ_u retain | θ₀ forget |
|---|---|---|---|---|
| 0.00 | — | 0.031 | 1.000 | 1.000 |
| 2.90 | 0.25× | 0.031 | 1.000 | 1.000 |
| 5.79 | 0.5× | 0.031 | 1.000 | 1.000 |
| 11.59 | **1.0×** | 0.062 | 1.000 | 1.000 |
| 23.17 | **2.0×** | 0.031 | 1.000 | 1.000 |

The θ₀ column is the control that makes this readable: the same perturbation leaves a fully
present fact fully disclosed, so the probe is sensitive, not inert.

**This refuted a hypothesis I had formed** — that θ_u sits a short hop from the memorized
solution and is therefore fragile. It is not. And it **invalidates the framing of my own
Proposition 2**: a robustness *radius* is the wrong object, because the adversary's
direction carries the information, not its distance travelled.

## 4. Phase 2 — the mechanism transfers

Design forced by the theory before any run (`theory/THEORY.md`):

- **P3a** — the gradient-norm penalty is satisfied at the memorized minimum too
  (`‖∇L‖ = 0` there), and started inside that basin it drives the fact all the way back
  (`L → 0.0000` for every moderate `λ_G`, verified symbolically and numerically). So the
  penalty is applied **only as a second stage** to an already-unlearned model, never from
  θ₀.
- **P3b** — raw ascent diverges in 40 dimensions (high loss and zero gradient coincide only
  in 1-D), so the first term is an **anchor** to θ_u, not ascent.

Objective: `λ_G·‖∇L_mem(θ)‖ + λ_A·KL(p_θ ‖ p_{θ_u})` on forget and retain states, using
the P4 finite-difference surrogate — which penalises `‖g‖`, **not** `‖g‖²`; the two are not
interchangeable in the gradient and the code is labelled accordingly.

`‖∇L_mem‖` is measured on **4 frozen batches with fixed masks**, because the per-step value
on a fresh batch is dominated by the mask rate `t ~ U(0,1)` and is not comparable across
steps.

| | `‖∇L_mem‖` (frozen) | `L_mem` (frozen) |
|---|---|---|
| θ_u baseline | 1.1547 | 0.9188 |
| after 150 steps | 0.7352 (−36.3%) | 0.9128 |
| **after 300 steps** | **0.6941 (−39.9%)** | **0.9195 (+0.0007)** |

The guard never fired. Cost 0.744 s/step vs 0.602 for the control (**1.24×**), 6 fwd + 4
bwd per step, 17.4 GiB peak.

### The control is provably a no-op

With `λ_G = 0` the objective is the anchor alone, identically zero at its own
initialisation, so under **SGD** its gradient is exactly zero. Verified:
`‖∇L_mem‖_frozen = 1.154713` at steps 0, 5 **and** 9 — bit-identical. The `λ_G = 0` arm
therefore *is* θ_u, and Phase 0's θ_u curve is its relearning curve.

## 5. Phase 3 — and it buys nothing

| adversary steps | base (floor) | θ_u (control) | flattened 150 | flattened 300 |
|---|---|---|---|---|
| 0 | 0.000 | 0.023 | 0.046 | 0.046 |
| 2 | 0.023 | 0.046 | 0.000 | 0.046 |
| 5 | 0.023 | 0.256 | 0.279 | 0.279 |
| **10** | **0.023** | **0.605** | **0.581** | **0.581** |
| 20 | 0.046 | 0.651 | 0.651 | 0.651 |

Retain is identical across all three arms at every `k` (1.000 → 0.625); degenerate forget
outputs are 1 for the control and 2 for both flattened arms, so there is no hidden
collapse.

**Pre-registered verdict:**

| arm | Δ(k=10) | needs | Δ single-shot | needs | Δ retain | needs | verdict |
|---|---|---|---|---|---|---|---|
| flattened 150 | −0.023 | ≤ −0.20 | +0.023 | ≤ +0.05 | +0.000 | ≥ −0.05 | **NOT SUPPORTED** |
| flattened 300 | −0.023 | ≤ −0.20 | +0.023 | ≤ +0.05 | +0.000 | ≥ −0.05 | **NOT SUPPORTED** |

−0.023 is **one probe out of 43** — the quantisation floor of the measurement. And the
**dose-response is flat**: 300 steps (−39.9% gradient norm) gives *identically* 0.581, the
same as 150 steps (−36.3%), and both converge to the control exactly at k=20 (0.651).

**A 39.9% reduction in the penalised quantity, at unchanged forget loss, produced no
measurable robustness gain.**

### The `λ_G` sweep is confounded, and I am not reporting it as a sweep

I intended `λ_G ∈ {1, 10}` as the dose variable. It is not one. Measured end-state of the
penalised quantity:

| arm | `λ_G` | steps | `‖∇L_mem‖` reached | reduction |
|---|---|---|---|---|
| control | 0 | — | 1.1547 | 0% |
| anvil_main | 1 | 150 | 0.7352 | −36.3% |
| anvil_main | 1 | 300 | 0.6941 | −39.9% |
| anvil_lamG10 | **10** | 300 | **0.6902** | **−40.2%** |

**A 10× increase in `λ_G` changed the achieved reduction by 0.6%.** The cause is the
gradient clip at norm 1.0, which is applied after the penalty gradient is accumulated: the
clip normalises `λ_G`'s scale away, so `λ_G` alters only the direction mix, not the step
magnitude. The real dose variable in this study was therefore **steps**, which gave
−36.3% vs −39.9% and *identical* relearning curves (0.581 at k=10 for both).

Because `λ_G = 10` lands in the same mechanism state as `λ_G = 1` (0.690 vs 0.694), its
relearning probe would be uninformative and was not run — stated rather than quietly
omitted. The honest scope of the negative result is: **over the range of reduction actually
achieved, 0% → 40.2%, there is no robustness gain.** Whether a much deeper reduction (say
−90%, which would need the clip raised or removed) would help is **untested**, and that is
the first thing a follow-up should do.

This is a direct empirical instance of the objection Song et al. raise in *"Measure, Don't
Optimize"* — that a quantity which *correlates* with recovery need not *control* it when
optimised — now measured on the masked-diffusion objective, which is exactly the transfer
question the literature left open. It is also consistent with Deeb & Roger's finding that
defenses of this family fall to stronger adversaries: here the defense does not even hold
against the *baseline* adversary.

## 6. What this does and does not license

**Does:** in this setup, reducing `‖∇L_mem‖` by 40% does not slow a relearning attack on a
masked-diffusion LM. The gradient norm is a mechanism check, not a robustness metric, and
should not be reported as evidence of robustness without the downstream curve.

**Does not:** this is **not** a refutation of Fan et al. They report GP working for
autoregressive LLMs on WMDP and MUSE; I tested a transfer to a different objective, on
synthetic data, at one `λ_G` family, one seed, one model. Specifically unexplored:

- `λ_G` beyond the range probed, and `ρ` (the finite-difference radius) not swept at all;
- the exact HVP route rather than the finite-difference surrogate;
- their other four variants (SAM, RS, CR, WA) — only the GP-style penalty was tested;
- applying the penalty *jointly* with unlearning rather than as a second stage (which P3a
  forbids for this penalty, but their formulation differs);
- an adversary-strength sweep, which was planned and became moot once the defense showed
  no effect at the baseline adversary.

**Also carried over:** the reference model missed its own 0.80 memorization gate at 0.797,
so all forget metrics are restricted to the 51 recalled facts, and absolute numbers are
indicative. One seed. Synthetic fictional facts. Single phrasing per probe in the k-sweep.

## 7. Three bugs that produced confident wrong signals

Recorded because each one looked like a result first.

1. **AdamW made the provable no-op move.** Its per-coordinate normalisation turns a zero
   gradient into a step of ≈`lr`, so the `λ_G = 0` control random-walked off θ_u. Fixed by
   SGD, under which it is bit-identically stationary.
2. **The guard compared incomparable quantities and fired on every arm.** `L_mem` was
   checked on each step's random batch against a baseline from batch 0; with `t ~ U(0,1)`
   that value swings 0.37 → 2.21 with no drift. It aborted all three arms at step 10 —
   including the one that cannot move. **Those early "guard fired" results were artefacts
   and are not evidence of the P3a drift they appeared to confirm.** Both baseline and
   guard now read the frozen probe.
3. **A step-size failure masqueraded as a dead mechanism.** The first hyperparameters (clip
   1.0, lr 5e-5) moved parameters at most 7.5e-3 over 150 steps against an unlearning edit
   of norm 11.6 — 0.07% of the relevant scale — so `‖∇L_mem‖` sat flat at ~1.13 and the
   mechanism looked inert. lr 5e-3 was then calibrated by a 30-step sweep before the main
   run.

A fourth, caught by design rather than by luck: my **Proposition 3 drift test was wrong**
(it asserted large `λ_G` causes drift, and failed at `λ_G = 50`). The true dependence is on
the starting *basin*, not on `λ_G`. The wrong version is preserved in `theory/THEORY.md`
beside the correction, as is the radius framing that §3 invalidated.

## 8. The one next step worth taking

The open question this sprint actually sharpened is **§3**, not the method: recovery
requires gradient information about the forget set, and is indifferent to parameter
displacement of twice the edit's norm. That says the thing to control is not the *location*
or *flatness* of θ_u but **what the forget-set gradient at θ_u still points at**. A
measurement worth making next: the cosine similarity between the adversary's recovery
direction and the original unlearning edit, as a function of `k`. If recovery runs
anti-parallel to the unlearning edit, then the edit's own direction is the leak, and that
is a mechanism no flattening penalty can address.
