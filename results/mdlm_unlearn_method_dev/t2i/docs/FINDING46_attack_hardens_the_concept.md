# Finding 46 — the attack does not merely restore the concept; it writes it in **more robustly than pretraining did**

> 🔄 **STATUS: REINSTATED by `FINDING48`, on better evidence than this finding had.** The
> withdrawal below stands as a record of the correct intermediate judgement: this finding was
> underdetermined when published (one arm, no cross-arm, no utility-matched comparison). At
> UTILITY-MATCHED removal, all three attacked arms retain the concept at 0.80–1.00 against the
> base model's 0.333, so the claim holds — see `FINDING48` §1 and its §4 scope limits.
>
> ⚠ **(superseded) GENERAL CLAIM WITHDRAWN — see `FINDING47`.** This finding generalised from a SINGLE
> attacked arm. Two further arm families, attacked and swept identically, do not reproduce it:
> the damage+distil arm is **worse than base** (church random-25 %: 0.000 vs base 0.333) and
> the official AdvUnlearn arm matches base at random-50 % and magnitude-10 %. The cross-arm
> comparison is additionally confounded by unmatched overall fragility. §2–3's claims that the
> attack entrenches the concept, and that "an adversary does not merely restore, they
> entrench", are **withdrawn**; what stands is listed in `FINDING47` §3. The title above is
> left as published.

**Date:** 2026-10-23 · **Cost:** ~1.5 GPU-h, all from already-attacked checkpoints
**Evidence:** `outputs/redundancy_{base,attacked}_{church,parachute}.json`,
`outputs/frp_refill_*.json`, `outputs/rezero_*.json`, `outputs/rezeroctl_church_r3.json`
**Status:** random-removal arm complete in all three states; magnitude arm complete for base and
attacked, **erased state still running**.

---

## 1. The measurement

Take a model, zero a fraction of its **feed-forward** weights, and re-measure the target concept and
retain utility. Run it on three states of the same pipeline: the **base** model, the **erased**
model (FRP), and the **attacked** model.

**Random selection** (target accuracy; retain8 in parentheses):

| removed | base church | attacked church | base parachute | attacked parachute |
|---|---|---|---|---|
| 0 % | 1.000 (1.000) | 1.000 | 1.000 (1.000) | 1.000 |
| 3 % | 1.000 (1.000) | **1.000** (0.967) | 1.000 (1.000) | **1.000** (0.933) |
| 10 % | 1.000 (1.000) | **1.000** (0.975) | 1.000 (1.000) | **1.000** (0.817) |
| 25 % | **0.333** (0.450) | **1.000** (0.475) | **0.533** (0.450) | **1.000** (0.267) |
| 50 % | **0.067** (0.050) | **0.933** (0.167) | **0.133** (0.050) | **0.800** (0.092) |

**Largest-magnitude selection:**

| removed | base church | attacked church | base parachute | attacked parachute |
|---|---|---|---|---|
| 3 % | **0.600** (0.833) | **1.000** (0.683) | 0.933 (0.833) | **1.000** (0.467) |
| 10 % | **0.000** (0.050) | **1.000** (0.258) | 0.733 (0.050) | **0.933** (0.125) |
| 25 % | 0.000 (0.017) | 0.867 (0.050) | 0.067 (0.017) | 0.800 (0.050) |
| 50 % | 0.000 (0.008) | 0.467 (0.042) | 0.133 (0.008) | 0.733 (0.058) |

**Erased (FRP), random selection** — included to show the instrument tracks a suppressed concept:

| removed | church target | parachute target |
|---|---|---|
| 3 % | 0.200 | 0.333 |
| 10 % | 0.133 | 0.333 |
| 25 % | 0.000 | 0.400 |
| 50 % | 0.000 | 0.200 |

## 2. The result

> **The post-attack concept survives weight removals that completely destroy the base model's own
> representation of the same concept.** Magnitude-10 % removal takes base church from 1.000 to
> **0.000** and leaves attacked church at **1.000**. Random-50 % takes base church to **0.067** and
> leaves attacked church at **0.933**.

**The instrument works.** It is not saturated or blind: it drives the base model's concept to 0.000
and the erased model's to 0.000, so the attacked model's 1.000 is a measurement, not a ceiling
artifact. That control is what makes this readable, and it is the reason this finding was not
published two updates earlier.

## 3. The most likely explanation, stated plainly rather than dressed up

The attacked model has received **300 gradient steps on 16 images of one concept**. It is *overfit*
to that concept, and an overfit concept is written in with larger and more distributed weights than
pretraining — which saw that concept as a vanishing fraction of its data — ever produced. **There is
nothing mysterious here.** The redundancy is a consequence of concentrated single-concept
fine-tuning.

But the consequence for the threat model is worse than "the concept cannot be removed":

> **After an attack, the concept is harder to remove than it was before any erasure was attempted.
> An adversary does not merely restore the capability — they entrench it.**

Any defence applied *after* an attack therefore faces a harder problem than the original one. And a
defender who re-erases a recovered model is not returning to the starting position.

## 4. What this explains

- **Why every method in this phase fails.** The region in which the concept could be subtracted out
  lies far beyond the region in which the model still functions. At random-25 % the concept is at
  1.000 while utility has fallen to 0.475/0.267.
- **FRP's narrow usable window and non-monotone ρ ladder** (`FINDING43` §3): past ρ ≈ 5 % the model
  breaks before the concept does.
- **Why `frag` destroyed utility** (`FINDING45`): there is no small forget-critical subset whose
  removal is cheap, because the concept is not in any small subset.

## 5. Bounded correctly: what the supporting checks do and do not show

- **The re-zero test** — re-apply FRP's pruning mask to the attacked checkpoint — leaves the target
  at **1.000** on church (4 415 427 positions) and **1.000** on parachute (7 434 307 positions).
- **Its null control** — a *random* matched-count set — also leaves it at **1.000** with identical
  utility (0.9758). So the re-zero test does **not** show FRP's chosen positions are specially
  irrelevant; it shows **no 3 % subset is where the recovered concept lives.** This correction was
  made before publication, not after.
- **The concentration ratio** (`FINDING44`) is the statistic that implicates FRP's mask specifically:
  **1.0169** — the attack's update was uniform across pruned and surviving positions, so the
  adversary did not preferentially rebuild the pruned route.

## 6. ⚠ A provenance error of mine, caught and repaired

`redundancy_sweep.py` keyed its output filename on **concept only**, so the base-model run silently
overwrote the attacked run's JSON, and my first rename then mislabelled the base numbers as
"attacked". The JSON's own `arm` field (`_base_ref`) exposed it. The attacked results were rebuilt
from their run logs with a provenance note, the per-arm `gen_*.json` primary records were never
affected, and the script now keys on **state + concept** so it cannot recur.

## 7. Scope

Two concepts, one attack configuration per concept (the highest-recovery one), feed-forward weights
only, one removal granularity per rule. Removal is a crude intervention: it measures where the
function *can* be disrupted, not where the concept is *represented*. The erased-state magnitude arm
was still running at this writing. No claim is made that this generalises beyond SD v1.4 or beyond
the relearning attack family.
