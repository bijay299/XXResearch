# H12 — ESD: Entity-Masked Self-Distillation

**Status:** gating test running (`p099`, `scripts/entity_mask_target.py`); novelty check in progress
**Date opened:** 2026-10-21

---

## 1. The precise limitation this addresses

Not "unlearning is hard". The specific, measured failure of every baseline in
`docs/BASELINE_MATCHED.md` and `docs/SCREEN_RESULTS.md`:

> **Failure is strictly bimodal.** 9 of 15 grid points across MDU(τ), EMDU, NPO(β), SC-TF and
> the exclusion target reach DISCLOSE 0.000 **at DEGENERATE 1.000** — they emit `''` or
> `'........'`. The two arms that stay fluent disclose ≈0.45. The desirable cell — fluent,
> on-topic, not disclosing — holds **4 of 960** generations.

The pristine reference (a model that never saw the data) sits at DISCLOSE 0.030,
DEGENERATE 0.000, ABSTAIN 0.113, REPLACE 0.470, IGNORANT 0.388. So the target cell is not
empty in principle; it is empty under these objectives.

**Why.** GA, NPO, MDU and EMDU are all *suppression* objectives: they specify where
probability mass must leave (the true token) and say nothing about where it must go. The
cheapest descent direction at a half-filled answer canvas is the model's unconditional
preference, which at an answer position is punctuation and EOS. The objectives get what they
asked for. **The problem is mass redistribution, not suppression** — and no amount of
hyperparameter search fixes an objective whose minimum is in the wrong cell.

## 2. Operation

The Alice/Bob example states the target directly: at the answer position the model should
behave as it does for a person it was never told about — still fluent, still a city, just not
Alice's city. LLaDA can be asked for that distribution with no second model, because it is a
masked denoiser **trained to condition on mask tokens at arbitrary sequence positions,
including the prompt**:

```
target(i)  =  p_θ₀( · | canvas, prompt with the entity-mention tokens → [MASK] )     [frozen]
train      =  minimise  KL( p_θ( · | canvas, full prompt )  ‖  sg target(i) )  over answer
                        positions i of forget facts
             +  λ_r · CE( p_θ, true answer )  over retain facts
```

Three design choices, each for a stated reason:

| choice | why |
|---|---|
| target from **frozen θ₀**, not current θ | a target computed from the live model is a moving target and both sides can drift together to anything; freezing makes this an ordinary fixed scalar objective, the property that made the ordering candidate preferable to earlier ones |
| **stop-gradient** on the target | the loss must not be satisfiable by moving the target |
| entity span taken from the **prompt**, not the answer | masking the answer would be teaching abstention; masking the prompt asks a different *question* |

**The architectural claim, stated so it can be refuted:** an autoregressive model has no
calibrated response to a masked prompt — it never saw masked input in training — so this
target is unavailable to it. If an AR model with mask tokens inserted into the prompt produces
a comparably non-disclosing, non-degenerate distribution, this claim is false and ESD is not
MDLM-specific. That is testable and should be tested before the claim is published.

## 3. Failure conditions, each with its observable

| # | failure condition | observable | where measured |
|---|---|---|---|
| F1 | the question identifies the entity by other means ("the author born in Taipei on 05/11/1991 who writes leadership"), so masking the name does not produce ignorance and **the target leaks the fact** | `entity_masked` DISCLOSE not materially below `natural` | gate K1 |
| F2 | the entity-masked distribution is **itself degenerate** — mass on EOS — so ESD inherits the exact failure it exists to avoid | `entity_masked` DEGENERATE > 0.15 | gate K2 |
| F3 | the effect is **generic prompt corruption**, not entity ignorance — masking any span would do it, so the mechanism claim is void | `control_masked` DISCLOSE within 0.10 of `entity_masked` | gate K3 |
| F4 | **canvas leak**: the target is computed on the same canvas, so once fact tokens are committed into the answer region the model reads the entity off the *answer* and the target re-discloses even with the prompt masked | per-mask-rate disclosure of the target; target disclosure as a function of committed answer tokens | post-gate, before training |
| F5 | parameters are shared, so matching the entity-masked distribution on forget facts drags retain facts of the same shape — the Alice/Bob separation fails | retain ASCR; the matched retain profile | training arms |
| F6 | the target is fluent but *wrong in a way that is still identifying* (names a different fact of the same author) | ASCR against author-salient stems, which is what ASCR-12s already measures | training arms |

F1–F3 are pre-registered kill rules and run before any training is funded. F4 is the one I
expect to bite and it has a prepared mitigation (compute the target on a canvas whose answer
region is fully masked, or mask entity mentions in the answer canvas too) — but the mitigation
is not adopted until F4 is measured, because adopting untested mitigations is how the ordering
line accumulated two inert objectives.

## 4. Observable separating predictions

These distinguish ESD from the baselines as *mechanisms*, not merely as better points. Each is
falsifiable and none is a restatement of the objective.

**P1 — the trajectory in the (DISCLOSE, DEGENERATE) plane differs in shape.** Sweeping the
strength parameter, MDU(τ), NPO(β) and EMDU trace a path from (0.45, ~0) to (0.00, 1.00):
DEGENERATE rises monotonically as DISCLOSE falls. ESD predicts DISCLOSE falls while
**DEGENERATE stays below 0.15 throughout**, because its minimum is a fluent distribution.
Refuted if ESD's sweep traces the same curve.

**P2 — the mass that leaves the true token lands in REPLACE, not ABSTAIN or DEGENERATE.**
Prediction: ΔREPLACE ≥ 0.5 · (−ΔDISCLOSE). The suppression baselines put it in DEGENERATE.
This is a prediction about *which cell*, which the six-cell partition reads directly and which
a single forget-quality scalar would hide.

**P3 — Alice/Bob separation survives at matched forget quality.** At equal DISCLOSE on the
forget split, ESD's retain ASCR exceeds every baseline's, because the target is on-topic for
the question rather than a global suppression direction. Refuted if retain ASCR is at or below
the baselines at matched DISCLOSE.

**P4 — ESD degrades gracefully where suppression does not, under RTT relearning.** Weaker
prediction, stated as such: a model whose answer distribution has been *moved* rather than
*flattened* has no sharp low-confidence basin for relearning to climb out of, so the RTT
recovery slope should be shallower than NPO's. The relearning study (`a67d610044d0`) found
MDLM unlearning recovers 0.049 → 0.605 in 10 steps and that recovery is **directional, not
positional** — which is why this is a prediction worth making and also why it may fail.

**P5 — the effect is entity-specific, not corruption-specific.** The length-matched control
mask does not reproduce it. Tested in the gate as K3, and retested as a training control.

## 5. What is NOT claimed yet

- That ESD beats SOTA. No training has been run.
- That the mechanism is novel. The novelty review is in progress; the closest known prior art
  is `Who's Harry Potter` (arXiv 2310.02238), whose generic replacement predictions come from a
  separately fine-tuned *reinforced model*, and oracle matching (arXiv 2506.06278), which
  requires a *retain-only retrained reference*. ESD's claim is that neither second model is
  needed because the architecture supplies the ignorance distribution. **If the review finds
  this anticipated, the candidate closes.**
- That the target is usable at all. That is what `p099` is deciding.

## 6. Cost

Gate: ~12 GPU-minutes, three decodes × 120 facts, no training. Training arms, only if the gate
passes: 2 arms × 400 steps ≈ 0.4 GPU-h, plus partition evaluation at 2 checkpoints.
