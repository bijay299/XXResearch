# Pre-registration H80 — Meta-Unlearning (2410.12777) ported to our testbed

**Date:** 2026-10-24 · **Status:** pre-registered; **no H80 number exists.**
**Source:** arXiv 2410.12777, *Meta-Unlearning on Diffusion Models: Preventing Relearning
Unlearned Concepts* · **Code:** not usable here (as with all five competing defences)

---

## 1. Why this is the one remaining idea worth testing

Every defence this phase has tested tries to **remove or damage** the concept: closed-form edits
(UCE), negative guidance (ESD), prompt-space adversarial training (AdvUnlearn), pruning (FRP),
noise-plus-distillation (damage+distil), and three Fisher allocations of ours. All are reversed.

Meta-Unlearning attacks a different variable. Its mechanism, quoted:

> *"if the meta-unlearned DM undergoes malicious finetuning on unlearned concepts, the related
> benign concepts retained within it will be triggered to self-destruct, hindering the relearning"*

It does not try to make the concept unrecoverable. It makes **recovery destroy the model**, which is
the one lever the mechanism results in `FINDING46`/`FINDING51` leave open: the concept is
redundantly encoded and cannot be subtracted out, but nothing we have tried makes *the act of
relearning* costly.

Its diagnosis also matches our own measurement. The paper's premise is that benign concepts related
to the target (its example: "skin" ↔ "nudity") facilitate relearning. `T2I_BASELINES` §2 measured
exactly that structure here — the cross-attention positions the UNet reads are **half-collinear
across concepts** (cos 0.52–0.59).

## 2. The port, transcribed before implementing

The paper adds a **meta objective** to an existing unlearning loss. Our instantiation:

```
θ' = θ − α ∇θ L_forget(θ)                  # ONE simulated malicious-finetuning step
L  = L_erase(θ) + L_retain(θ) + λ_meta · ( − L_benign(θ') )
```

- `L_erase` — keep the target erased, as a distillation-to-erased-teacher term
- `L_retain` — benign concepts must stay **good at θ** (used as-is)
- `−L_benign(θ')` — benign concepts must become **bad at θ'**, i.e. after a simulated attack step.
  Minimising this maximises the benign loss *under attack*: the self-destruct trigger.

**First-order approximation.** The exact gradient of `L_benign(θ')` through `θ'` needs a
second-order term. We use the first-order approximation standard in MAML variants, which is a
**stated deviation** from an exact implementation and could weaken the effect. If H80 fails, the
approximation is a live alternative explanation and will be named as such rather than ignored.

**Declared hyperparameters:** α (inner step) 1e-5, matching the attack's own learning rate;
λ_meta ∈ {0.5, 1.0, 2.0}; 1500 outer steps; the erased teacher is `frp_church_r3_l05`
(target 0.1333 at retain 0.9667, the phase's best operating point).

## 3. Separating predictions, declared now

| outcome | reading |
|---|---|
| retain ≥ 0.85 **and** max-over-valid H ≤ 0.40 under the frozen grid | **the mandate's bar met — by a published method.** Reported as theirs, and `PREREG_H40` refuted by someone else's work |
| retain ≥ 0.85, H ≤ 0.50 but > 0.40 | real frontier improvement, short of the bar |
| H → 1.000 like everything else | the meta objective does not transfer, or our first-order port is too weak — **both stated, not one** |
| retain < 0.60 | the self-destruct fires on normal use, not just under attack: the defence destroys the model it is protecting |
| utility collapses **only after attack**, with H low | **the intended behaviour.** Must be checked against the issue-7 utility probe, not inferred from H alone |

**The fourth and fifth rows are why the utility probe matters here more than anywhere else in this
phase.** A defence whose *design* is to self-destruct under attack will look like a robustness win
on H while having destroyed the model. `FINDING20`'s gate plus the post-attack utility panel are
what separate "resisted" from "immolated", and both are mandatory on every H80 arm.

## 4. Kill rules

**K1.** Frozen six-point grid, identical to every other arm in the phase.
**K2.** retain8 < 0.60 **as-is** ⇒ not a candidate, regardless of robustness.
**K3.** A low H with post-attack utility < 0.50 is **reported as self-destruction, not robustness.**
**K4.** Both concepts before any claim.

## 5. Honesty constraints

Meta-Unlearning is **their** method. A positive result is a result about their work, reported as
such. This is **our port from the paper**, first-order and on our testbed, so any disagreement with
their numbers is attributable to the port. My expectation, recorded: given eight sub-ceiling numbers
of mine have dissolved this week, I expect either full reversal or self-destruction — but this is
the first mechanism tested here that is *designed* for the failure mode we keep measuring, so it has
the best prior of anything in the phase.
