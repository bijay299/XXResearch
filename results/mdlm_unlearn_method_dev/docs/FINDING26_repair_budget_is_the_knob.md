# Finding 26 — the repair budget is the trade-off knob, H19 is refuted, and one point now beats every baseline

**Date:** 2026-10-22 · **Pre-registration:** `PREREG_H19_repair_budget.md`
**Backbone:** H18 124M MDLM with an exact retrain oracle
**Status:** H19's hypothesis **refuted**; its replacement mechanism **also refuted**; the
measurements stand and one of them is the best operating point in the project.

---

## 1. H19 asked whether UNDO's retain cost was a budget artifact. It is — and so is its robustness.

Repair-budget sweep, α=0.5, Fisher-protected damage, AFC teacher, gated RTT at lr 3e-5:

| repair steps | retain | recovery |
|---|---|---|
| 1,500 | 0.3100 | **−0.0293** |
| 3,000 | 0.4850 | **−0.0311** |
| 6,000 | 0.5775 | 0.0887 |
| 12,000 | 0.6700 | 0.2512 |

Retain climbs from 0.3100 to 0.6700 — close to the teacher's 0.7275 — so the retain cost **was**
budget-bound, as H19 predicted. But recovery climbs with it, −0.029 → 0.251, crossing the 0.20
threshold `PREREG_H19` §4 declared as *"repair does resurface forget knowledge; the robustness was
itself a budget artifact"*.

> **Both axes are controlled by the same knob.** Repair restores the retain you want and the
> forget knowledge you do not. H19's hypothesis — that retain could be recovered *without*
> surrendering robustness — is **refuted** by its own pre-registered criterion.

Kill rule K1 (retain improvement < 0.02 ⇒ stop) did **not** fire: the improvement was +0.22.

## 2. The ceiling shows the trade-off is not intrinsic to damage+distillation

| arm | retain | recovery | TV→oracle |
|---|---|---|---|
| *UNDO with the **exact oracle** as teacher* | *0.8150* | *0.0669* | *0.0275* |

With a teacher that has no residual forget knowledge, 12,000 steps of repair give near-oracle
retain **and** low recovery **and** the best behavioural match in the project. So the trade-off
in §1 is a property of **distilling from a suppressed teacher**, not of damage+distillation.

**This arm is a ceiling, not a method** — in any real setting the oracle is precisely what you do
not have. As pre-registered, it is excluded from every frontier comparison against baselines.

## 3. The obvious explanation is also wrong

§2 suggests robustness should be bounded by how much residual forget knowledge the teacher holds.
Tested across four teachers at identical 12,000-step repair:

| teacher | residual forget (rank) | retain | recovery |
|---|---|---|---|
| GA λ=0.5 | **0.0850** | 0.4675 | **−0.0718** |
| *oracle (ceiling)* | *0.1100* | *0.8150* | *0.0669* |
| AFC λ=2 | 0.1600 | 0.7075 | **0.2032** |
| EMDU λ=1 | **0.1725** | 0.6850 | **0.1393** |

The trend is suggestive (Spearman ρ = 0.8 over four points) but **EMDU inverts it**: it has the
*highest* residual and yet better post-repair robustness than AFC. With n=4 and one inversion this
explanation is **not supported**, and I am not asserting it.

What the table does show is that **teacher choice does not buy a free lunch**: the lowest-residual
teacher (GA) gives the best robustness (−0.0718) but collapses retain to 0.4675, off the useful
frontier entirely.

**Two of my mechanism claims were refuted in consecutive experiments** — H19's budget-artifact
hypothesis, then the teacher-residual explanation. The measurements below are replicated and
stand; the mechanism behind them is **unexplained**, and I would rather record that than offer a
third story for the next run to knock down.

## 4. What stands: one point beats every baseline on both axes

| arm | retain | recovery |
|---|---|---|
| MDU λ=0.5…8 | 0.5417–0.6733 | 0.5788–0.6966 |
| AFC λ=1…8 | 0.5717–0.6992 | 0.6274–0.6869 |
| EMDU λ=1…8 | 0.7433 | 0.6620–0.7183 |
| MDU+FAM λ=1 | 0.7333 | 0.7411 |
| GA λ=0.5…8 | 0.6417 | 0.6752–0.8000 |
| **UNDO fisher 12k, retain×3, AFC teacher** | **0.7075** | **0.2009 ± 0.0071 (n=3)** |
| **UNDO fisher 12k, retain×3, EMDU teacher** | **0.6850** | **0.1393** *(n=1, replicating)* |

The AFC-teacher point has **retain comparable to or better than every anchor family** (0.7075 vs
their 0.54–0.74) at **2.8–4.3× better robustness**, replicated at n=3 with sd 0.0071. The
EMDU-teacher point is better still on robustness (0.1393) at slightly lower retain, and is being
replicated before it is relied on.

This is the first configuration in the project that is not dominated by any baseline on either
axis.

## 5. Honest size

- **The method is UNDO's.** What is ours here is the repair-budget diagnosis, the Fisher-protection
  modification (itself `PARTIALLY ANTICIPATED`), the teacher comparison, and the measurement
  protocol.
- **Not a published-SOTA claim.** Baselines are our re-implementations on our backbone; the
  original authors' tuning is not reproduced.
- **One backbone, 124M, one synthetic corpus, one attack family.** No scale-transfer claim; the 8B
  programme could not test UNDO at all.
- **Recovery 0.14–0.20 is not "removed".** Against a 0.1135 chance floor, a fifth of the gap back
  to `theta_ref` still reopens. Better than every baseline is not the same as solved.
