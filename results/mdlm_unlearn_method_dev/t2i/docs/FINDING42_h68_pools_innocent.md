# Finding 42 — H68: the pool repair is INNOCENT. The `frag` allocation is implicated, and FINDING38's in-sample bias is now quantified at ~6.7 points

**Date:** 2026-10-23 · **Rule applied:** `FINDING39` §4, declared before H68 ran
**Evidence:** `outputs/h68_analysis.json`, `outputs/gen_h68v2_rf_a0.3_*.json`,
`outputs/maskaudit_h68v2_rf_a0.3_*.json`

---

## 1. The result

A matched v2 rebuild of `retain_fisher` — identical to `H64` in every parameter except `--protect`:

| arm | allocation | pools | target | **retain8** | utility (n=11) |
|---|---|---|---|---|---|
| `h47_adv_a0.3` (historical) | `retain_fisher` | **v1, partly in-sample** | 0.000 | **0.8667** | — |
| **`h68v2_rf_a0.3_church`** | `retain_fisher` | **v2, out-of-sample** | 0.333 | **0.8000** | 0.8485 |
| **`h68v2_rf_a0.3_parachute`** | `retain_fisher` | **v2, out-of-sample** | 0.933 | **0.9750** | 0.9636 |
| `h64v2_frag_a0.3_church` | **`frag`** | v2 | 0.000 | **0.0000** | 0.0909 |
| `h64v2_frag_a0.3_parachute` | **`frag`** | v2 | 0.067 | **0.0833** | 0.1455 |

**`FINDING39` §4 branch 1 fires on both concepts.** `retain_fisher` on the repaired pools holds
retain at **0.8000** and **0.9750**. The pool repair did **not** cause `H64`'s collapse, so the
`frag` allocation is implicated and `FINDING39`'s pre-declared reading stands.

**The costlier branch did not materialise.** Branch 2 — that the pool repair was the cause and every
published damage+distil retain number in this phase was an in-sample artifact — is **refuted**. That
was the outcome that would have cost me the most, and it is reported as refuted on its own evidence.

## 2. ✅ FINDING38's bias is now measured, not just signed

`FINDING38` established that v1 pools contained the evaluation strings and argued the bias favoured
our own arms. It could not say *how much*. The matched pair now does:

> **church `retain_fisher` α=0.3: retain8 0.8667 (v1, in-sample) → 0.8000 (v2, out-of-sample).
> The in-sample inflation was ≈ 6.7 percentage points.**

So the contamination was **real and modest**, not load-bearing. Consequences, stated precisely:

- `FINDING38` §3's withdrawal of **cross-family utility rankings** stands — a 6.7-point inflation on
  our arms against zero on the baselines is still enough to corrupt a ranking against AdvUnlearn's
  0.992 or UCE's 0.850.
- But the **within-family α-ladder is NOT invalidated**. `FINDING39` §4 warned that branch 2 would
  force extending the withdrawal to it; branch 1 means that extension is **not** required. The
  ladder's *shape* survives with every rung shifted down by roughly this amount.
- `ERRATUM_r2` C10 should be read with this magnitude attached rather than as an unbounded doubt.

One asymmetry worth recording: at α=0.3 on v2 pools, **parachute's `retain_fisher` arm barely erases
at all** (target 0.933 against a base of 1.000) while church reaches 0.333. The same per-concept
weakness on parachute appears in `H47` §7 and `H65`'s erasure depths. The parachute arm's high
retain (0.9750) is therefore *not* a better operating point — it is an arm that mostly did not do
anything.

## 3. ⚠ What this does NOT establish, and the control that would

The two arms are matched on α, β, teacher, distillation steps, learning rate, pool recipe, seed, and
**realised mask mean (0.3000 in both, exactly)**. They are **not** matched on the quantity that
actually perturbs the weights:

| arm | realised perturbation norm |
|---|---|
| `h64v2_frag_a0.3_church` | **20.827** |
| `h68v2_rf_a0.3_church` | **15.091** |
| | **ratio 1.380×** |

So "the `frag` allocation destroys utility" still confounds **where** the damage goes with **how
much** lands. `PREREG_H69` pre-registers the norm-matched arm (β scaled to 0.07245) and was written
**before** H68's number existed, precisely so it could not become a post-hoc rescue. It is queued
behind `H66`'s FRP work, which has higher priority because the competitive-baseline gap binds every
claim in this phase.

**A correction to my own reporting:** in a status update I read this ratio off a truncated log line
and reported it as "~14×". It is **1.38×**. The error mattered — a 14× mismatch would have made the
H64/H68 comparison nearly worthless, whereas 1.38× is a modest confound that leaves branch 1
readable. The matching audit is now **computed** by `h68_analyse.py` rather than eyeballed, which is
why the real number surfaced.

## 4. The best out-of-sample damage+distil point measured, and it still misses the bar

`h68v2_rf_a0.3_church`: **target 0.333 at retain8 0.8000**, measured fully out-of-sample. That is
the best honest erasure/utility operating point this phase has produced for the damage+distil
family.

It still **does not meet the mandate's bar**, which requires retain ≥ 0.85 *and* max-over-valid
post-attack H ≤ 0.40. Retain is 0.800, below the floor, and **no attack has been run on this arm** —
every arm in this phase has looked acceptable before the attack and been reversed by it. No
robustness claim is made or implied.

## 5. Scope

Two concepts, one α, one teacher, one seed per arm. `retain_fisher`'s v1 figure (0.8667) is quoted
from the historical record and was not recomputed; the 6.7-point estimate is therefore a comparison
across two builds that differ only in pool recipe **to the extent the historical build is
reproducible**, which construction nondeterminism before `08742c1` limits. The v2 arms are
deterministic.
