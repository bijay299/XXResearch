# Finding 47 — ❌ CORRECTION to FINDING46: entrenchment does NOT generalise. It appeared on one arm family and failed on two others.

**Date:** 2026-10-23 · **Corrects:** `FINDING46`, published the same day, commit `62a4cea`
**Evidence:** `outputs/redundancy_attacked_{,h68rf_,advunlearn_}church.json`
**Status:** `FINDING46`'s general headline is **withdrawn**. The measurement it rests on stands; the
generalisation does not.

---

## 1. What FINDING46 claimed, and what it should have claimed

`FINDING46` stated, as a general property of the attack:

> *"After an attack, the concept is harder to remove than it was before any erasure was attempted.
> An adversary does not merely restore the capability — they entrench it."*

**That was measured on exactly one attacked arm** — the FRP arm — and generalised. Two further arm
families, both attacked identically (300 steps, 16 images, seed 0, lr 1e-5) and swept identically,
do not reproduce it.

Church, target accuracy after removing a fraction of feed-forward weights:

| state | random 3 % | random 10 % | **random 25 %** | random 50 % | magnitude 10 % |
|---|---|---|---|---|---|
| **base** | 1.000 | 1.000 | **0.333** | 0.067 | 0.000 |
| attacked — **FRP** arm | 1.000 | 1.000 | **1.000** | **0.933** | **1.000** |
| attacked — **damage+distil** arm | 1.000 | 1.000 | **0.000** | 0.000 | 0.000 |
| attacked — **official AdvUnlearn** | 1.000 | 1.000 | **0.867** | 0.067 | 0.000 |

- the **damage+distil** attacked arm is **worse than base** (0.000 vs 0.333 at random-25 %);
- the **AdvUnlearn** attacked arm is above base at 25 % (0.867 vs 0.333) but **identical to base** at
  random-50 % (0.067) and magnitude-10 % (0.000);
- only the **FRP** arm shows persistence far above base at every fraction.

**Correct statement:** *on the FRP-attacked arm the recovered concept was markedly more robust to
weight removal than the base model's own concept; this did not replicate on two other attacked arms,
one of which was less robust than base. Entrenchment is not established as a property of the
attack.*

## 2. ⚠ And the cross-arm comparison is confounded anyway

The three attacked arms are **not matched on how fragile they are overall**, which is exactly the
variable concept-persistence must be read against. At random-25 % removal:

| attacked arm | target | **retain8** |
|---|---|---|
| FRP | 1.000 | **0.4750** |
| AdvUnlearn | 0.867 | **0.6167** |
| damage+distil | 0.000 | **0.0917** |

The damage+distil arm had already been through noise-plus-distillation and entered the sweep far
more fragile (retain8 0.0917 where FRP keeps 0.4750). Its concept dying first is **consistent with
global fragility rather than with any statement about the concept**. So this comparison cannot
cleanly attribute the difference to the defence family, and I am not going to claim it does.

**What a clean test needs:** sweep each arm to its own *utility-matched* removal level — compare
concept accuracy at the fraction where each arm's retain8 reaches, say, 0.45 — rather than at a
fixed fraction. That is a straightforward reanalysis of the existing curves plus a few interpolating
points, and it is **not done**.

## 3. What survives from FINDING46

| claim | status |
|---|---|
| the instrument detects concept loss (base → 0.000 under magnitude-10 %) | **stands** |
| no 3 % subset of FF weights holds the recovered concept (re-zero + random null control both 1.000) | **stands** |
| the attack's update is uniform across pruned and surviving positions (concentration 1.0169) | **stands** (`FINDING44`) |
| on the FRP arm, the attacked concept outlives the base model's concept | **stands**, single arm |
| removing the concept requires removals that destroy the model *on the arms measured* | **stands** |
| **"the attack entrenches the concept" as a general property** | **WITHDRAWN** |
| **"an adversary does not merely restore, they entrench"** | **WITHDRAWN** |

The phase's headline is untouched: seven methods reversed, mandate unmet. Entrenchment was never
load-bearing for that.

## 4. Why I got this wrong, and the specific lapse

I ran the base-model control — which is what made `FINDING46` publishable — and then generalised
across *defence families* without running a second family. The base control tested "is this a metric
artifact?" It did not test "is this specific to this arm?", and those are different questions. I
published the stronger claim on the strength of the weaker control, 45 minutes before the
counter-evidence came in from a check I had already queued.

That is the same failure mode this project has recorded four times as n-too-small, and once in
`FINDING35` §0 as generalising from a convenience set. The honest reading is that the queued check
should have gated publication, not followed it.

## 5. In flight

Two further FRP-arm attacks (seeds 1 and 6) are being swept to test whether the FRP result is even
stable **within** that arm across attack seeds. If it is not, §3's single-arm claim narrows further
to a single attack run.
