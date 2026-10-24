# Finding 51 — FINDING48's headline confirmed on BISECTED matched points; the entrenchment gradient SATURATES

**Date:** 2026-10-24 · **Closes:** `FINDING50` §4's open question and the instrument flaw it named
**Evidence:** `outputs/matched_{base,frp,advu,h68rf}_church.json`,
`outputs/redundancy_dose*_s600_church.json`

---

## 1. The instrument flaw is fixed at the root

`FINDING50` §3 identified the cause of two of the previous day's corrections: `redundancy_sweep.py`
takes **removal fractions**, so every utility-matched reading depended on guessing a bracket in
advance, and badly-placed brackets inverted a conclusion twice.

`scripts/matched_removal.py` replaces guessing with **bisection on the removal fraction to hit a
target `retain8`**. Seven iterations, tolerance 0.04. Every arm below lands in **retain8
0.425–0.458** — a genuinely common operating point rather than a nearest-available bracket.

## 2. FINDING48's headline, re-measured

Church, random removal, bisected to retain8 ≈ 0.45:

| arm | fraction removed | **retain8 (matched)** | **target accuracy** |
|---|---|---|---|
| **base model** | 0.2531 | **0.4583** | **0.2667** |
| FRP-attacked | 0.2625 | 0.4333 | **1.0000** |
| official AdvUnlearn-attacked | 0.2719 | 0.4250 | **1.0000** |
| damage+distil-attacked | 0.1688 | 0.4333 | **0.8667** |

> **Confirmed, and the gap is wider than the bracket-based table showed: base 0.267 against
> 0.867–1.000 for all three attacked arms, at matched retain utility within 0.033.**

The base reading moves from 0.333 (nearest bracket, retain8 0.450) to 0.267 (bisected, retain8
0.4583) — one image on a 15-image metric. `FINDING48` §1's conclusion is unchanged and now rests on
matched measurements rather than approximations.

## 3. The gradient saturates

`FINDING50` §4 left open whether entrenchment keeps rising past 300 steps. 600-step points on both
arms:

| steps | FRP | damage+distil |
|---|---|---|
| 30 / 100 | 0.733 | 0.467 |
| 75 / 150 | **1.000** | 0.733 |
| 150 / 200 | **1.000** | 0.733 |
| 300 | 0.867 | 0.867 |
| **600** | **1.000** | **0.867** |

**Entrenchment rises over the low-to-mid budget range and then plateaus at the metric ceiling.** It
is not unbounded in budget. The 0.867-vs-1.000 wobble along the plateau is **two images on a
15-image metric** and is reported as quantisation noise, not structure.

So the final shape of this line, after four corrections: entrenchment is **a saturating function of
attack budget**, present on all three defence families at matched utility, invisible to `H_after`
because that metric ceilings out well before entrenchment does.

## 4. What this line cost, and what it is worth

Five findings (46–51), four corrections, roughly 12 GPU-hours. **It does not advance the mandate.**
The headline is unchanged — **seven methods reversed, no arm at retain ≥ 0.85 with max-valid
H ≤ 0.40** — and entrenchment was never load-bearing for it.

What it is worth: a threat-model refinement (a defender re-erasing a recovered model faces a harder
problem than the original, and the effect grows with the attacker's budget), a measured attacker
cost an order of magnitude below this phase's standard (`FINDING49` §3: 20–30 steps), and a repaired
instrument. What it cost: time not spent on the binding constraint, which remains that **four of the
five competing defences are unrun**.

Parachute bisection is **running** as a second-concept check on the fixed instrument.
