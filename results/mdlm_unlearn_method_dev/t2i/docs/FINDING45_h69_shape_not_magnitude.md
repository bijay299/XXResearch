# Finding 45 — ✅ H69: it is the allocation's SHAPE, not its magnitude. FINDING39's attribution is confirmed clean.

**Date:** 2026-10-23 · **Pre-registration:** `PREREG_H69_norm_matched_frag.md`, written before H68's
utility number existed · **Evidence:** `outputs/gen_h69v2_fragnorm_a0.3_*.json`,
`outputs/maskaudit_h69v2_fragnorm_a0.3_*.json`

---

## 1. The result

β was scaled to **0.07245** so the realised perturbation norm would match `retain_fisher`'s. It
landed at **15.089** (church) and **15.031** (parachute) against the 15.091 target — inside the 5 %
tolerance declared in `PREREG_H69` §2, so no correction was needed.

| arm | allocation | β | **realised norm** | church retain8 | parachute retain8 |
|---|---|---|---|---|---|
| `h64v2_frag_a0.3` | `frag` | 0.1 | 20.83 | **0.0000** | 0.0833 |
| **`h69v2_fragnorm_a0.3`** | **`frag`** | **0.07245** | **15.09** | **0.0000** | **0.0583** |
| `h68v2_rf_a0.3` | `retain_fisher` | 0.1 | **15.09** | **0.8000** | **0.9750** |

**Two clean comparisons, each varying one thing:**

- **matched norm, different shape** (H69 vs H68): `frag` **0.0000** / `retain_fisher` **0.8000**
- **matched shape, different norm** (H69 vs H64): **0.0000** / **0.0000** — a 28 % reduction in β
  changed nothing

> **It is the SHAPE of the allocation that destroys utility, not the magnitude.** `PREREG_H69` §3's
> first declared outcome, which was also the expectation recorded there.

## 2. What this closes

`FINDING39` attributed `H64`'s collapse to the `frag` allocation, and `FINDING42` flagged that the
attribution still confounded *where* the damage lands with *how much* of it lands (the 1.38× norm
mismatch). **That confound is now removed.** The attribution in `FINDING39` §2 — *"the
forget-critical factor dominates and multiplying by (1 − F_retain) does not rescue it"* — stands as
stated, on a clean control.

Combined with `FINDING42` (the pool repair is innocent) the chain is complete: neither the pool
recipe nor the damage magnitude explains `H64`. The allocation does.

**Selectivity is now exhausted in all three forms, each on a clean comparison:**

| allocation | targets forget-critical | spares retain-critical | retain8 (v2 pools) | matched norm? |
|---|---|---|---|---|
| `retain_fisher` | ✗ | ✓ | **0.8000** | 15.09 |
| `forget_fisher` | ✓ | ✗ | 0.0917 *(v1 pools)* | not matched |
| `frag` | ✓ | ✓ | **0.0000** | **15.09** |

## 3. Why the intersection is worse than either half, read against FINDING44

`FINDING44` found that an attacked FRP model recovers its concept through the weights FRP **did not**
prune — the concept is re-encodable outside the forget-critical set a saliency method identifies.

H69 is the same lesson from the defence side. Damaging exactly the forget-critical-and-retain-sparing
weights destroys the model **per unit of damage**, which means those weights are not a separable
"concept route" whose removal is cheap: they are load-bearing for general generation. Two
independent measurements, one from the attack side and one from the defence side, now say the same
thing — **saliency-identified forget-critical weights are neither sufficient to remove the concept
nor safe to damage.**

That is a sharper and better-supported version of `PREREG_H51` §7's claim, which rested on utility
collapse alone and was downgraded to provisional by `ERRATUM_r2` C18. It is **still not a proof of
inseparability**: no direct overlap measurement between the forget-critical and retain-critical sets
has been computed, and C18's downgrade stands. But the provisional claim now has two independent
lines of evidence instead of one.

## 4. What was deliberately not run, and why

`H69` fails **K2** on both concepts (retain 0.0000 / 0.0583), so it is not a candidate and no
robustness number could be admissible. **Its attack grid was deliberately skipped.** `H64` — the
same allocation at 1.38× the damage — already produced **0 of 9 gate-passing attacks on church**
(`FINDING39` §5), establishing "too broken to attack" for this family. Twelve more attacks on an
identically-broken arm would consume ~2 GPU-hours and add nothing.

This is a deviation from the issue-6 requirement that every arm carry a gated+ungated record, taken
knowingly and recorded here rather than left as a silent gap.

## 5. Scope

One α, one teacher, one seed per arm, two concepts. The norm match is on the **aggregate** realised
perturbation norm; the per-tensor distributions still differ by construction, which is exactly what
"shape" means and is not a confound to be removed. `forget_fisher`'s 0.0917 is a v1-pools figure
quoted from the historical record and is **not** norm-matched — the `forget_fisher` row above is
therefore the one leg of the three-way comparison that is not clean, and it is not load-bearing for
the conclusion.
