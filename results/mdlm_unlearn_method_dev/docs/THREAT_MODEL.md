# The REPLACE question is a threat-model question, and our metric already answers it

**Date:** 2026-10-21 · **Resolves:** the open item in `REPORT.md` §3
**Status:** argument + existing measurements; no new experiment required

---

## 1. The disagreement

Our evaluation frame scores **REPLACE** — a fluent, confident, *wrong* answer — as acceptable,
because the retrain-only pristine reference does it at 0.470. EMDU's authors, and
[arXiv 2407.16997](https://arxiv.org/abs/2407.16997), argue the opposite: a confident wrong
answer is *worse* than abstention, so an empty or refusing output is the safer outcome.

We previously recorded this as an unresolved weakness in our frame. It is not a weakness; it is a
consequence of which threat model you adopt, and the two threat models are in genuine conflict.

## 2. Our objective is exactly a distinguishing advantage

For two distributions `P`, `Q` over the same finite set,

```
TV(P, Q) = sup_A |P(A) − Q(A)|
```

which is the **optimal advantage of a single-observation distinguisher** between `P` and `Q`.
Our objective is `TV(forget profile → pristine profile)` over the six-cell partition. So the
number we minimise is, verbatim, *the best an adversary can do from one query in deciding whether
a given fact was unlearned out of this model or was never in it.*

That is the **indistinguishability / privacy** threat model, and it is TOFU's own framing:
approximate the model that was retrained without the forget set. Under it, matching pristine's
REPLACE rate of 0.470 is not a concession — it is the target. A model that abstains where a
pristine model would confabulate is **detectably different**, and the detection is itself the
leak: it reveals which facts were removed, hence that they were in the training set.

> **The REPLACE question therefore has a definite answer under our threat model.** Confabulating
> at the pristine rate is correct, because the pristine model confabulates. Abstaining instead is
> a privacy failure, not a safety improvement.

**Honest bound on the claim.** TV over the six cells is the advantage of an adversary who observes
only the *cell label*. The cell label is a coarsening of the raw generation, so a stronger
adversary reading the text could do strictly better. Our TV is a **lower bound** on the true
distinguishing advantage, and should be cited as one.

## 3. The consequence, in our own measured numbers

Reading each arm's TV as a membership-inference advantage:

| arm | TV→pristine = single-query advantage | note |
|---|---|---|
| **AFC λ=1** | **0.299** | best in project |
| MDU λ=3 | 0.352 | |
| ATWU best of 11 | 0.466 | |
| θ₀ — *no unlearning at all* | 0.770 | the do-nothing baseline |
| RMU (calibrated) | **0.846** | **more detectable than doing nothing** |
| EMDU λ=8 | **1.000** | **perfectly detectable from one query** |

Two results fall straight out:

1. **EMDU's "safe" abstention is the worst possible outcome under this threat model.** DEGENERATE
   1.000 gives TV 1.000: an adversary identifies every unlearned fact with certainty from a single
   query. The output that is safest in the harm-avoidance view is maximally leaky in the privacy
   view.
2. **RMU is more detectable than not unlearning at all** (0.846 vs θ₀'s 0.770). It forgets well
   (DISCLOSE 0.050) and retains well (0.704), but it does so by emitting gibberish 82.5% of the
   time, which a pristine model never does. Strong forgetting plus a distinctive failure signature
   is worse, for privacy, than leaving the fact in.

Neither of these is visible to a DISCLOSE-only or ROUGE-L-based evaluation, which is the
`FINDING15` metric trap from a different direction.

## 4. The two threat models cannot both be satisfied

| threat model | goal | REPLACE | ABSTAIN | right metric |
|---|---|---|---|---|
| **indistinguishability / privacy** | be indistinguishable from a retrained model | **required**, at pristine's rate | a leak | TV→pristine |
| **harm avoidance / safety** | never assert false or harmful content | **forbidden** — misinformation | **required** | refusal rate, factuality |

These are not two measurements of one goal; they are opposed. Matching a pristine model *requires*
confabulating at its rate, which the safety view forbids outright. No single arm can be optimal
for both, and an arm that is excellent under one can be the worst available under the other —
EMDU λ=8 demonstrates exactly that.

**What this licenses us to say.** Our headline is defended on indistinguishability grounds, which
we now declare explicitly rather than leaving implicit, and we do **not** claim our arms are
preferable under a harm-avoidance threat model. EMDU is not wrong; it optimises a different
objective, and we mis-framed it once already
(`RESULTS.md` retraction, `NOVELTY_FINAL.md`).

**What it does not license.** It does not make REPLACE harmless in deployment. A system whose
threat model is misinformation should not use TV→pristine as its objective, and we should not
present our frontier to such a reader as the relevant one.

## 5. Open

Whether an adversary reading raw generations (not cell labels) gets materially more than TV is
untested, and would tighten §2's lower bound into a real attack. That is a concrete, cheap
follow-up: train a classifier on generations from forget vs. never-seen facts and compare its
advantage to the TV figures above.
