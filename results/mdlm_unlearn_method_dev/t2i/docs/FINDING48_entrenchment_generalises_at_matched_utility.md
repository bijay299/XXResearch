# Finding 48 — at UTILITY-MATCHED removal, entrenchment generalises across all three attacked arms. FINDING47's withdrawal was based on a confounded comparison — which FINDING47 itself identified.

**Date:** 2026-10-23 · **Supersedes:** `FINDING47` §1's conclusion · **Reinstates, on better
evidence:** `FINDING46`'s claim
**Evidence:** `outputs/redundancy_{base,attacked,attacked_advunlearn,attacked_h68rf,um_*}_church.json`

---

## 1. The comparison FINDING47 §2 named and did not run

`FINDING47` compared arms at **fixed removal fractions** and concluded entrenchment does not
generalise. Its own §2 flagged why that was unsafe: the arms are not matched on overall fragility, so
"whose concept dies first" partly measures "whose model was already more fragile". It named the fix
— compare each arm at the fraction where its **retain8 reaches a common level** — and recorded it as
not done.

It is now done. Each arm swept until its retain8 crosses **0.45**, reading the target there:

| arm | removal fraction | retain8 there | **target accuracy there** |
|---|---|---|---|
| **base model** | 0.250 | **0.450** | **0.333** |
| FRP-attacked | 0.260 / 0.270 | 0.483 / 0.425 | **1.000 / 0.867** |
| official AdvUnlearn-attacked | 0.270 / 0.280 | 0.483 / 0.400 | **0.867 / 0.933** |
| damage+distil-attacked | 0.170 | 0.425 | **0.867** |

These are **direct measurements** at the bracketing fractions, not interpolations. An earlier
interpolated version of this table gave FRP-attacked as **0.995**; measured directly it brackets
**0.867–1.000** around retain8 = 0.45, so that figure was slightly optimistic and is corrected here.
The conclusion is unchanged: every attacked arm sits at **0.867–1.000** where the base model sits at
**0.333**.

> **At matched utility, all three attacked arms retain the concept at 0.80–1.00 while the base model
> retains it at 0.333.** Entrenchment generalises.

The fixed-fraction picture inverted because the arms reach retain8 = 0.45 at very different removal
fractions — the damage+distil arm at **17 %** against the base model's **25 %**. Comparing it to base
at 25 % was comparing a model already past its collapse point to one just reaching it.


## 1a. ✅ SECOND CONCEPT: replicated on parachute, 3 of 3 arms

`§4` listed "a second concept" as the largest gap. It is now measured. Parachute, random removal,
read at the rung where each arm's retain8 is nearest 0.45:

| arm | fraction | retain8 there | **target there** |
|---|---|---|---|
| **base model** | 0.250 | 0.450 | **0.533** |
| FRP-attacked | 0.180 | 0.442 | **1.000** |
| official AdvUnlearn-attacked | 0.230 | 0.475 | **0.733** |
| damage+distil-attacked | ~0.140 | ~0.45 | **1.000** |

**All three attacked arms exceed base on parachute as well** (1.000 / 0.733 / 1.000 against 0.533),
so entrenchment replicates across both concepts and all three defence families.

Two honest notes:

- **Base parachute starts higher** (0.533 at the matched point against church's 0.333), so the
  headroom to demonstrate entrenchment is smaller on this concept and the AdvUnlearn margin
  (0.733 vs 0.533) is correspondingly narrower than church's (0.867–0.933 vs 0.333).
- ⚠ **I reported this as a partial replication one interval earlier, and that was wrong.** With the
  AdvUnlearn parachute curve bracketed only between retain8 0.958 and 0.408, I *extrapolated* its
  matched point to ≈ 0.60 and called the arm marginal. Measuring the bracket directly (r0.18, r0.21,
  r0.23) puts it at **0.733**. The extrapolation was the error; the fill runs were already queued
  for exactly that reason and they corrected it. The damage+distil arm's matched point remains
  interpolated, but between two rungs that both read **1.000**, so no extrapolation is involved.

## 2. Where this leaves today's three findings

| finding | claim | status |
|---|---|---|
| `FINDING46` | the attack entrenches the concept (general) | **reinstated**, on 3 arms at matched utility rather than 1 arm at a fixed fraction |
| `FINDING47` §1 | entrenchment does not generalise | **superseded** — its fixed-fraction comparison was confounded |
| `FINDING47` §2 | the comparison is confounded by unmatched fragility | **confirmed, and it was the explanation** |
| `FINDING47` §5 | is the FRP result stable within-arm? | **yes** — 3 attack seeds, §3 below |

**I am not going to present this as vindication.** `FINDING46` was published on one arm with no
cross-arm and no matched comparison; it was **underdetermined when published**, and `FINDING47` was
right to withdraw it on those grounds. What has changed is the evidence, not the reasoning. The
sequence — overreach, withdrawal, reinstatement on a controlled comparison — is three turns in one
day, and the lesson is the one `FINDING47` §4 already drew: the queued control should gate
publication, not follow it.

## 3. The FRP arm is stable across attack seeds

Clean re-runs after the label-collision fix (`6ffd0aa`), church, target accuracy:

| removal | sd0 | sd1 | sd6 | base |
|---|---|---|---|---|
| random 25 % | 1.000 | 1.000 | 1.000 | 0.333 |
| random 50 % | 0.933 | 0.733 | 0.867 | 0.067 |
| magnitude 10 % | 1.000 | 1.000 | 0.867 | 0.000 |
| magnitude 25 % | 0.867 | 0.267 | 0.533 | 0.000 |

Direction is stable; **magnitude is strongly seed-dependent** at the harsher rungs (magnitude-25 %:
0.867 / 0.267 / 0.533). Quoting sd0 alone would overstate it, so the spread is reported. The seeds
also differ from each other — the positive check that the collision is gone, not merely patched.

## 4. What this supports, stated at its actual strength

**Supported:** on SD v1.4, church, three different defence families, a 300-step/16-image relearning
attack leaves the concept markedly more resistant to feed-forward weight removal than the base
model's own representation of that concept, when compared at matched retain utility.

**Not established:**
- ~~**a second concept.**~~ **Now established** — see §1a: parachute replicates on all three arms.
- **attack-seed replication for AdvUnlearn and damage+distil.** Those are single runs (seed 0); only
  the FRP arm has three seeds.
- **the mechanism.** The likely cause remains the one `FINDING46` §3 gave — 300 steps on 16 images of
  one concept overfits it, writing it in more strongly than pretraining, which saw it as a vanishing
  fraction of data. That is an explanation, not a measurement.
- **that removal probes representation.** Zeroing weights measures where the function can be
  disrupted, not where the concept is stored.
- **matching at one retain level.** 0.45 was chosen because base and FRP-attacked already sat near
  it. The curves cross; a different matching level could order the arms differently, and the full
  curves are published so that can be checked.

Direct measurements at the matched fractions (rather than interpolated ones) were running as this was
written; §1's AdvUnlearn and damage+distil rows are interpolated between bracketing rungs.

## 5. Why it matters for the mandate

It does not change the headline — **seven methods reversed, mandate unmet.** It sharpens the threat
model: a defender who re-erases a recovered model is not returning to the starting position, and
this now holds across every defence family tested rather than one. Any sequential-unlearning or
re-erasure protocol inherits that.
