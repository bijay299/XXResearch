# Pre-registration H69 — norm-matched `frag`: is it the allocation's SHAPE or its MAGNITUDE?

**Date:** 2026-10-23 · **Status:** pre-registered. **H68's utility number does not exist yet**, and
this document is written before it does, precisely so it cannot become a post-hoc rescue of H64.
**Conditional:** runs only if H68 fires FINDING39 §4 **branch 1** (the pools are innocent and the
`frag` allocation is implicated). If branch 2 fires, H69 is moot and will not be run.

---

## 1. The gap this closes

`h68_analyse.py` computes what the H64/H68 comparison is matched on. It is matched on α, β, teacher,
distillation steps, learning rate, pool recipe, seed, **and realised mask mean (0.3000 in both,
exactly)**. It is **not** matched on the quantity that actually perturbs the weights:

| arm | allocation | realised mask mean | **realised perturbation norm** |
|---|---|---|---|
| `h64v2_frag_a0.3_church` | `frag` | 0.3000 | **20.827** |
| `h68v2_rf_a0.3_church` | `retain_fisher` | 0.3000 | **15.091** |
| | | | **ratio 1.380×** |

Same at α=0.5 on the integration-test arms: 31.91 vs 25.15, ratio 1.269×.

So even if H68 shows the pools are innocent, "the `frag` allocation destroys utility" would still
confound **where** the damage goes with **how much** of it lands. A 1.38× larger perturbation is a
modest mismatch — not the order-of-magnitude one I briefly misreported from a truncated log line —
but it is the whole of the quantity under test, and the audit's issue 10 asked exactly for controls
with a *matched realised* distribution.

## 2. The arm

`frag` with β scaled to equalise the realised perturbation norm. Because the perturbation is
`s · β · N`, its norm is linear in β:

> **β_matched = 0.1 × (15.091 / 20.827) = 0.07245**

Everything else identical to H64: α=0.3, `--protect frag`, AdvUnlearn teacher, 6000 steps, lr 1e-5,
pools v2, seed 0. Label `h69v2_fragnorm_a0.3_<concept>`. The realised norm is **verified from the
mask audit after the build** and reported; if it does not land within 5 % of 15.091, β is corrected
once and that correction is disclosed.

## 3. Separating predictions, declared now

| outcome | reading |
|---|---|
| retain8 still collapses (≤ 0.30) at matched norm | the **SHAPE** of the allocation is what destroys utility — concentrating damage on forget-critical weights is harmful *per unit of damage*. H64's negative is about the allocation, as `FINDING39` read it |
| retain8 recovers to ≥ 0.80 at matched norm | H64's collapse was substantially a **MAGNITUDE** effect, and `FINDING39`'s attribution to the allocation is **wrong and must be corrected**. The frag allocation would then be untested at matched damage |
| retain8 lands between | partial; both contribute and this control does not separate them cleanly. Reported as such |

**I expect the first outcome**, because `H51`'s `forget_fisher` arm collapsed to 0.0917 at the
*same* β as `h47`'s `retain_fisher` 0.8667 — a comparison in which magnitude was not obviously
inflated either. But that pair was built on v1 pools and is not clean, which is the whole reason
this document exists. Recording the expectation so the second outcome cannot be quietly absorbed.

## 4. Cost and priority

~35 GPU-minutes per concept, two concepts, so ~1.2 GPU-h. **It is queued BEHIND H66's FRP ladder and
attack grid**, which is the higher priority: FRP is a competing published defence and the
competitive-baseline gap binds every claim in this phase, whereas H69 refines the attribution of an
already-dead arm of ours. If budget forces a choice, FRP wins and H69 is reported as unrun.

## 5. Honesty constraint

If H69 fires the second outcome, `FINDING39` §2's mechanism sentence — "the forget-critical factor
dominates and multiplying by (1 − F_retain) does not rescue it" — is **withdrawn**, not softened,
and `FINDING39` gets a §0 correction notice under a new identifier in the manner of `FINDING35` §0.
