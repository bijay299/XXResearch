# Pre-registration H37 — does FINDING28 survive a sound knowledge axis? A designed replication

**Date:** 2026-10-22 · **Status:** pre-registered before any scored run · **Backbone:** H18 124M MDLM
**This is an audit of my own strongest claim, not a new method.**

---

## 1. Why this is necessary

`FINDING28` reports that static metrics do not predict attack robustness: Spearman(residual forget
knowledge, max recovery) = −0.148 over 125 arms, with 18 pairs of arms whose probe readings are
identical to four decimals differing 1.9×–37× in recovery.

Its x-axis is the **closed-pool ranking probe**, and that probe has just been shown to be invalid
(`FINDING29`). It scores candidates by summing log-probs over an EOS-padded length-32 sequence, so
padding enters every score. Measured consequences:

| model | ranking probe (retain) | marginal probe (retain) |
|---|---|---|
| `theta_ref` | 0.6725 | **0.9977** (exact-span 1.0000) |
| PUA λ=1 | **0.9000** | **0.9737** |

The ranking probe reads PUA as far better than `theta_ref` on retain when it is in fact slightly
**worse**, and reads `theta_ref` — which recalls retain perfectly — at 0.67.

**This threatens FINDING28 in a specific, statistical way.** A noisy x-axis attenuates correlation
toward zero (regression dilution). So ρ = −0.148 may reflect *measurement error in the probe*
rather than a true absence of relationship, and "identical probe readings, 37× different recovery"
may be "identical noisy readings of genuinely different knowledge." FINDING28 §7 already flagged
the convenience sample and family imbalance; this is the deeper problem, and it cannot be fixed by
reanalysis: only 2 of the 125 checkpoints still exist, so the arms must be rebuilt.

## 2. The hypothesis

> **H37.** FINDING28's conclusion is a property of unlearning, not of a broken probe. With a
> validated knowledge axis and a balanced design, a validated relearning attack still separates
> suppression-only from damage+distil arms that the knowledge axis cannot separate.

## 3. The knowledge axis, and why it has real endpoints

`marginal_probe_small.py`: mask the value span in place, keep true left and right context, read the
marginal probability (and exact-span argmax accuracy) of the **true** value tokens. No padding
enters the measurement.

It is anchored by two **measured** models rather than an analytic chance level — which the ranking
probe's "chance 0.1135" was, and which does not transfer between probes:

| anchor | forget exact-span | role |
|---|---|---|
| `theta_ref` (never unlearned) | **1.0000** | ceiling |
| exact retrain oracle (`FINDING21`) | **0.3375** | floor — what "the facts were never seen" actually looks like under this measurement |

Normalised knowledge is reported as `(x − oracle)/(ref − oracle)`. **Note this reframes a prior
claim:** `FINDING21` called the oracle "at chance (0.1100 vs 0.1135)" under the ranking probe; under
a context-rich measurement the same oracle sits at 0.3375, because with a closed answer domain and
true surrounding context a model that never saw a fact can still often infer it. The oracle is the
floor either way, but "at chance" is probe-specific language and will not be reused without naming
the probe.

**Both probes are run on every arm**, so the comparison between them is within-arm and the
disagreement is measurable rather than argued.

## 4. The design (balanced, unlike FINDING28's convenience sample)

8 suppression-only arms and 8 damage+distil arms, seed 0, chosen to **span the knowledge axis**
rather than to flatter either family:

| family | arms |
|---|---|
| suppression-only | `mdu` λ ∈ {0.5, 4}, `afc` λ ∈ {1, 8}, `emdu` λ ∈ {1, 8}, `ga` λ ∈ {0.5, 4} |
| damage+distil | AFC teacher at α ∈ {0, 0.1, 0.3, 0.5, 0.6}, MDU teacher at α ∈ {0, 0.3, 0.6} |

The two **α=0 arms are the controls**, and they are pre-declared as belonging to the
suppression side on the robustness axis: they receive no damage, so if the separator is damage they
must land with the suppression arms. In `FINDING28` they did, and that was the strongest internal
check it had.

Every arm: both knowledge probes, behavioural infill six-cell + TV→oracle, and RTT at
lr ∈ {1e-5, 3e-5, 1e-4} with `FINDING20`'s validity gate, **t_frac fixed at 0.5 for all 16 arms**,
recovery as max-over-valid-attacks.

## 5. Separating predictions, declared now

| outcome | reading |
|---|---|
| recovery bands still disjoint **and** marginal-probe Spearman \|ρ\| < 0.4 | **H37 true.** FINDING28 survives on a sound axis and gets stronger: balanced design, validated probe, pre-registered. |
| bands disjoint but marginal-probe \|ρ\| ≥ 0.7 | FINDING28's *headline* is wrong: knowledge **does** predict robustness once measured properly, and the old ρ was dilution. FINDING28 is rewritten — the band separation survives but "no static metric predicts robustness" does not. |
| bands overlap | the band separation was an artifact of the unbalanced convenience sample. **FINDING28 is retracted.** |
| the two probes agree closely (within-arm rank correlation > 0.9) | FINDING29's artifact is specific to PUA's pool-shaping and does not contaminate FINDING28's arms; the old ρ stands as reported |

## 6. Kill rules

**K1.** If the α=0 controls do **not** land in the suppression band, the mechanism attributed to
damage in `FINDING23`/`FINDING28` is wrong. Stop and re-examine before reporting anything else.

**K2.** No arm is dropped after seeing its recovery. The 16 arms are fixed by this document. If an
arm's attack fails the validity gate at all three lrs, it is reported as **ungated**, not silently
excluded — the gate's purpose is to stop a weak attack reading as robustness, and dropping gate
failures would quietly reintroduce exactly that.

## 7. What this cannot become

This is an audit. A positive result does not improve the forgetting–utility trade-off and will not
be presented as progress toward the mandate's bar — it only establishes whether my strongest
existing claim is sound. A negative result costs me that claim, which is the point of running it.
