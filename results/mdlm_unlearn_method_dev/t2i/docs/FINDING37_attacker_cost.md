# Finding 37 — ONE image and 50 gradient steps fully reverses the official adversarially-trained SOTA

**Date:** 2026-10-23 · **Model:** SD v1.4 (pinned) · **Defence:** official `OPTML-Group/AdvUnlearn`
**Concepts:** church (object) and Van Gogh (style) · **Protocol:** prompt-disjoint T/H,
`FINDING20` gate
**Status:** the practically meaningful form of this phase's result.

---

## 1. Why budget was the wrong thing to hold fixed

Every attack in this phase used **300 steps on 16 images**. That establishes erasure is reversible
but says nothing about *how cheaply* — and the cost is what decides whether a defence is worth
anything. A defence undone by 16 images is weak; one undone by a single image is not a defence.

**Direction matters here:** this sweep *weakens* the adversary, so it can only make the defence
look **better**. It is the opposite of the ratchet the audit identified (`FINDING35` §4), where
strengthening an adversary against hoped-for arms made `PREREG_H40` unfalsifiable in practice.

## 2. The measurement

Official AdvUnlearn weights, attacker budget swept **downward**, lr 1e-5, seed 0:

| attacker budget | church: H after | Van Gogh: H after |
|---|---|---|
| **1 image, 50 steps** | **1.000** ✓gate | **1.000** ✓gate |
| 1 image, 150 steps | **1.000** ✓ | **1.000** ✓ |
| 2 images, 150 steps | **1.000** ✓ | **1.000** ✓ |
| 4 images, 50 steps | **1.000** ✓ | **1.000** ✓ |
| 4 images, 150 steps | **1.000** ✓ | **1.000** ✓ |
| *(16 images, 300 steps — the original)* | *1.000* | *1.000* |

Pre-attack: church 0.400, Van Gogh 0.467. Every point passes the validity gate (T rises +0.542 to
+0.667), so none of these is a weak attack being credited.

> **One image and fifty gradient steps restores the concept completely, on both an object and a
> style. The curve is flat — the smallest budget tested already saturates, so the true threshold is
> at or below it.**

The attacker's single image is itself **generated from the public base model**. No scraped data, no
privileged access beyond the weights.

## 3. What this changes about the phase's conclusion

The phase's result was "every usable erasure is fully reversible". That is now quantified: the
reversal costs **one self-generated image and fifty optimiser steps** — seconds of compute. Against
a weight-access adversary, the five released AdvUnlearn checkpoints provide no measurable
protection at any budget tested.

This sharpens, and does not alter, the fair framing from `FINDING34`: AdvUnlearn claims robustness
against adversarial **prompt** attacks against fixed weights, and demonstrably achieves it. This is
a strictly stronger threat model their paper does not claim to defend. The finding is about the
**scope gap** — prompt-space hardening does not survive weight access — and about how *small* the
margin is once weights are exposed.

It also reframes the practitioner question. The relevant number is not "does the erasure metric
look good" but "what does it cost to undo", and here that cost is indistinguishable from zero.

## 4. Scope and limits

- Two concepts, one model, one erasure family (official AdvUnlearn), seed 0, lr 1e-5.
- The floor is **not located**: n=1/50 steps is the smallest budget run, and it already saturates.
  A finer sweep (fewer steps, partial-image) would find where recovery begins; it would only make
  the number smaller.
- H is a 15-sample closed-set CLIP measure at a 1.000 ceiling, so "1.000" means indistinguishable
  from base at this resolution, not bit-identical (`T2I_BASELINES.md` §1).
- The attacker's images come from the base model. A real adversary with photographs would be at
  least as strong, so this is not an optimistic assumption about attacker capability.
- Prompt-space attacks were not run — the literature already reports those, and they are the regime
  AdvUnlearn was built for.
