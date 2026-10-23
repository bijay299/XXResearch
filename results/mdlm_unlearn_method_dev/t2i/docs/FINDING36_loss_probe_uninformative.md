# Finding 36 — the loss-based trainability probe measures nothing about recovery in T2I; the base model scores the same as an erased one

**Date:** 2026-10-23 · **Model:** SD v1.4 (pinned) · **Concept:** church
**Status:** resolves the audit's open item 12 ("cannot verify whether the phases actually
disagree") and **invalidates the instrument behind part of `FINDING32`.**

---

## 1. The measurement

A frozen fixed-noise probe records how much of the available MSE loss the attacker removes on its
own training data — the same normalised statistic the MDLM phase used in `FINDING33`. Run across a
calibrated set of arms, where the key row is a model with **nothing erased at all**:

| arm | erasure state | **loss removed** | concept after attack |
|---|---|---|---|
| **base SD v1.4** | **nothing erased** | **6.94%, 6.52%** | 1.000 → 1.000 *(nothing to recover)* |
| official AdvUnlearn | erased, retain 0.992 | 6.51%, 7.10% | → **1.000** |
| composite α=0.3 | erased, retain 0.867 | 9.25%, 9.86% | → 0.933 |
| UCE (different family) | erased, retain 0.850 | 10.24% | → **1.000** |
| *MDLM arms (`FINDING33`)* | *erased* | ***93.6–98.9%*** | *recovery → 0.000* |

> **The base model, which has no erased concept and therefore nothing to restore, removes 6.5–6.9%
> of its available loss — statistically indistinguishable from the official erased arm's
> 6.5–7.1%.**

## 2. What that means

The statistic measures **how far 300 steps of fine-tuning move the diffusion MSE objective**, and
the answer is ~7–10% whether or not anything was ever erased. It carries **no information** about
whether the attacker recovered a concept. On the same arms the *content* measure separates
perfectly: base 1.000 → 1.000 (no change possible), erased arms → 1.000 (full restoration).

Two things follow, both against my own earlier reasoning:

**(a) `FINDING32`'s T2I trainability argument loses its instrument.** Its late-attack-loss measure
is the unnormalised version of this same quantity. The monotone rise it reported with α
(0.3393 → 0.4864) is real as a *number*, but this calibration shows that family of measure cannot
distinguish "hard to fine-tune" from "nothing to recover" — so it cannot support conclusions about
whether an attack succeeded. `FINDING32` §3's numbers stand as measurements; §5's inference from
them to a confound does not.

**(b) The two phases never disagreed.** `FINDING33` §3 tabulated "attacker can fit the arm? yes,
93.6–98.9%" for MDLM against T2I gate-failure rates as though they were one axis. They are not
comparable: in a masked-LM the training loss and the recoverable knowledge are tightly coupled, so
loss removal tracked recovery; in a diffusion model a ~7% MSE move is the *baseline* behaviour of
fine-tuning and is already enough to restore a concept completely. That row is **struck**, not
reinterpreted.

## 3. Why the gate was defined on content, and why that was right

`FINDING20`'s validity gate requires the attack to raise **recovered content** on its own training
split, not to reduce a loss. Nothing in this project anticipated this finding, but it is the reason
the protocol survived it: had the gate been loss-based, every attack in the T2I phase would have
been certified at ~7% regardless of whether anything was erased, and the base-model control would
have looked identical to a defended arm.

It also sharpens a limit already on record. The gate's three known failure modes (ceiling artifact,
T-harder-than-H false negatives, seed instability) are all about *content* measurement being
coarse. This finding adds that the obvious "more principled" alternative — a loss-based criterion —
would have been strictly worse.

## 4. Scope

- One concept (church), one model, one attack family, 2 seeds per arm, 300-step attacks.
- The probe is 8 latents at fixed noise and fixed timesteps; a larger probe would tighten the
  percentages but cannot change the base-vs-erased comparison, which is the load-bearing row.
- This says the **loss-removal** statistic is uninformative about recovery *in this setting*. It
  does not show the MDLM phase's use of it was wrong there — in that setting it tracked recovery,
  and `FINDING33`'s conclusion rests on 93.6–98.9% against a recovery collapse, a far larger
  separation than anything here.
- A broken-arm point (α=0.7, retain ~0.05) is still running and will extend the scale, but cannot
  overturn §1: the base-model row alone establishes the statistic has no discriminating power.
