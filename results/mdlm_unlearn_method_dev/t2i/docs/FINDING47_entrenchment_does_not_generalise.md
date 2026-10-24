# Finding 47 — ❌ CORRECTION to FINDING46: entrenchment does NOT generalise. It appeared on one arm family and failed on two others.

**Date:** 2026-10-23 · **Corrects:** `FINDING46`, published the same day, commit `62a4cea`
**Evidence:** `outputs/redundancy_attacked_{,h68rf_,advunlearn_}church.json`
**Status:** `FINDING46`'s general headline is **withdrawn**. The measurement it rests on stands; the
generalisation does not.

---

> 🔄 **§1's CONCLUSION IS SUPERSEDED by `FINDING48`.** §1 compared arms at FIXED removal
> fractions. §2 of this document flagged that as confounded by unmatched fragility and named the
> fix; the fix has now been run, and at UTILITY-MATCHED removal all three attacked arms retain the
> concept at 0.80–1.00 against base 0.333. **Entrenchment does generalise.** §2's confound was the
> explanation, and §5's within-arm stability question is answered yes. The measurements in §1 are
> correct as measurements; the inference from them was not.
>
> ✅ **HOLD LIFTED — §1's cross-arm table is CONFIRMED by clean re-measurement.**
>
> A bug in my sweep script put this table on hold: `redundancy_sweep.py` built its per-rung label
> from **concept + mode + fraction, omitting the state**, so two concurrent same-concept sweeps
> collided on both `adapters/_redun_<label>/unet.pt` and `outputs/gen_<label>.json`. It surfaced
> when an unrelated FRP seed-replication returned **bit-identical rows for two different attacked
> models**.
>
> Both affected legs were re-run with distinct state-keyed labels, sequentially. **They reproduce
> the original numbers exactly:**
>
> | church, random removal | 3 % | 10 % | 25 % | 50 % |
> |---|---|---|---|---|
> | damage+distil — original | 1.000 | 1.000 | 0.000 | 0.000 |
> | damage+distil — **clean re-run** | 1.000 | 1.000 | **0.000** | **0.000** |
> | AdvUnlearn — original | 1.000 | 1.000 | 0.867 | 0.067 |
> | AdvUnlearn — **clean re-run** | 1.000 | 1.000 | **0.867** | **0.067** |
>
> So the two legs happened to stagger in time and never actually overwrote each other. **§1 stands
> as published**, now on verified measurements rather than lucky ones. The contaminated originals
> are kept at `outputs/CONTAMINATED_label_collision/` for comparison.
>
> Still pending: the **FRP seed replication** (attack seeds 1 and 6), which the collision *did*
> corrupt — that is where the identical rows came from. §5's question about within-arm stability is
> unanswered until those clean sweeps land.

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
