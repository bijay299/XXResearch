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

| arm | removal fraction at retain8 = 0.45 | **target accuracy there** |
|---|---|---|
| **base model** | 0.250 | **0.333** |
| FRP-attacked | 0.270 | **0.995** |
| official AdvUnlearn-attacked | 0.283 | **0.867** |
| damage+distil-attacked | 0.170 | **0.800** |

> **At matched utility, all three attacked arms retain the concept at 0.80–1.00 while the base model
> retains it at 0.333.** Entrenchment generalises.

The fixed-fraction picture inverted because the arms reach retain8 = 0.45 at very different removal
fractions — the damage+distil arm at **17 %** against the base model's **25 %**. Comparing it to base
at 25 % was comparing a model already past its collapse point to one just reaching it.

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
- **a second concept.** All of §1 is church. Parachute has base and FRP-attacked curves but no
  matched-utility points for the other two arms.
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
