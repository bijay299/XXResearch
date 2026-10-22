# Pre-registration H19 — is UNDO's retain cost a repair-budget artifact?

**Date:** 2026-10-22 · **Status:** pre-registered before any run · **Backbone:** H18 124M MDLM

---

## 1. Where the evidence leaves us

| arm | retain | recovery (max-over-lr, gated) |
|---|---|---|
| teacher (AFC λ=2) | 0.7275 | 0.6573 |
| every anchor/GA family, λ ∈ {0.5…8} | 0.54–0.74 | **0.5608–0.7568** (27 runs) |
| UNDO plain α=0.5 | 0.3650 | 0.0513 |
| **UNDO retain-Fisher α=0.5** | **0.4567** | **0.0498** |
| oracle | 0.8875 | — |

`FINDING25`: no anchor family is robust, and higher λ makes robustness *worse*. `FINDING23`: only
damage+distillation resists, and the damage is the active ingredient (α=0 control gives none).
`FINDING24`: Fisher-protected damage buys +0.09 retain at equal damage and equal recovery.

**The remaining gap is one number.** Robustness is solved (0.05 vs 0.56–0.76). Retain is not:
0.4567 against the teacher's 0.7275. No arm yet has both.

## 2. The hypothesis

Repair runs for only 3000 steps on ~1400 prompts. The student starts from 50%-damaged weights, so
0.4567 may simply be **under-repaired**, not a floor.

> **H19.** UNDO's retain cost is largely a repair-budget artifact. Extending distillation should
> move retain toward the teacher's 0.7275 **without** restoring recovery, because the damaged
> student has no latent forget knowledge for repair to resurface — the teacher it distils from is
> itself suppressed.

This is what UNDO's authors claim for their strongest setting ("matches the robustness of a model
retrained from scratch with perfect data filtering"), and it is the one claim of theirs we have
not tested, because our budget was set for feasibility rather than convergence.

## 3. Arms

| arm | change | purpose |
|---|---|---|
| **A** `undo_h19_long` | α=0.5, Fisher, **12000 steps** (4×) | the test |
| **B** `undo_h19_ret` | α=0.5, Fisher, 12000 steps, retain-oversampled pool | faster retain repair, if pool composition matters |
| **C** `undo_h19_ceil` | α=0.5, Fisher, 12000 steps, **teacher = the exact ORACLE** | **CEILING, not a method** |

Arm C is explicitly **not an unlearning method** — in any real setting the oracle is what you
cannot have. It measures what damage+distillation could achieve if the teacher were perfect, which
separates "repair is incomplete" from "the teacher is the limit".

Every arm is evaluated identically: ranking probe (knowledge), infill six-cell + TV→oracle
(behaviour), and RTT at the adversarially-optimal lr 3e-5 under `FINDING20`'s validity gate.

## 4. Separating predictions, declared now

| outcome | reading |
|---|---|
| **A reaches retain ≥ 0.65 at recovery ≤ 0.10** | the retain cost was a budget artifact. This point **dominates every anchor family on both axes** — retain comparable to theirs (0.54–0.74) at 10× better robustness. The first frontier-dominating result in the project. |
| A reaches retain 0.50–0.65 at recovery ≤ 0.10 | partial; the trade-off is real but milder than measured, and the frontier shifts |
| A's retain plateaus < 0.50 | the cost is **intrinsic** to damage+distillation, H19 is false, and UNDO's trade-off stands as measured |
| A's recovery rises above 0.20 | repair **does** resurface forget knowledge; the robustness was itself a budget artifact, which would be a negative about UNDO and more important than the retain question |

## 5. Kill rule

**K1.** If 4× the steps improves retain by < 0.02 over the 3000-step arm, repair budget is not the
bottleneck. Stop this line and report the trade-off as intrinsic. No further budget escalation.

## 6. What this cannot become

- Arm C is a ceiling and will never be reported as a method or in a frontier comparison against
  published baselines.
- A positive result is a statement about **UNDO's own operating point** on this backbone. The
  method remains UNDO's; what would be ours is the diagnosis that its published retain cost is
  budget-bound, plus the Fisher-protection modification (itself partially anticipated).
- 124M, one synthetic corpus, one attack family. No scale-transfer claim.
