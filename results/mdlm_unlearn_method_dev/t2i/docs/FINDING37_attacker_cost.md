# Finding 37 (REWRITTEN) — the attack needs ~300 steps for full recovery; the original "1 image / 50 steps" claim was a measurement artifact

**Date:** 2026-10-23 · **Model:** SD v1.4 (pinned) · **Defence:** official `OPTML-Group/AdvUnlearn`
**Concepts:** church (object), Van Gogh (style) · **Protocol:** prompt-disjoint T/H, `FINDING20` gate
**Status:** ⚠ **This finding replaces a retracted version.** The first version claimed a flat curve
— that one image and fifty gradient steps sufficed — and that claim was **void**, produced by the
evaluation bug in `ERRATUM_text_encoder_eval_bug.md`. Everything below is re-measured with the
fixed harness.

---

## 0. What the first version got wrong, and why

The post-attack evaluation loaded the **base** text encoder instead of the erased one, so for any
text-encoder defence it reported the base model's score — 1.000 — regardless of what the attack did.
That produced a perfectly flat curve across every budget, which I wrote up as a finding ("the
smallest budget tested already saturates") rather than treating as the symptom it was.

I had even run a verification (§2a of the old version) and it passed: the *budget* was demonstrably
wired through, varying actual work over a 55× range. But I verified the **input** and never checked
the **output**. The flat curve with zero variance was the tell, and I noted it twice as evidence.

## 1. The corrected measurement

Official AdvUnlearn, 16 images, lr 1e-5, seed 0, budget swept downward:

| attacker budget | church: H after | gate | Van Gogh: H after | gate |
|---|---|---|---|---|
| **2 steps** | 0.400 → **0.333** | **FAIL** | 0.467 → **0.467** | **FAIL** |
| **50 steps** | 0.400 → **0.533** | PASS | 0.467 → **0.800** | PASS |
| **300 steps** | 0.400 → **1.000** | PASS | 0.467 → **1.000** | PASS |

Seed replication at 300 steps (3 seeds each, all gate-passing):

| concept | sd0 | sd1 | sd6 | max-over-valid |
|---|---|---|---|---|
| church | 1.000 | 0.800 | 1.000 | **1.000** |
| Van Gogh | 1.000 | 0.933 | 1.000 | **1.000** |

> **There is a real, monotone cost curve.** Two steps recovers nothing and legitimately fails the
> validity gate. Fifty steps recovers partially (0.533 on church, 0.800 on Van Gogh). Three hundred
> steps recovers completely on both. The post-attack values now vary across seeds and budgets,
> which is what a working measurement looks like.

## 2. What this does and does not change

**Unchanged:** the headline. At 300 steps on 16 self-generated images, the official
adversarially-trained SOTA — retain 0.992, pre-attack church 0.400 / Van Gogh 0.467 — is restored to
**1.000** on both an object and a style, 3/3 gate-passing each. `FINDING34`'s central claim survives
correct measurement.

**Changed, and materially:** the *cost*. The retracted version said one image and fifty steps
sufficed, implying the defence was worth nothing at all. The truth is less dramatic and more
useful: full reversal takes a few hundred optimiser steps, and a genuinely minimal adversary
(2 steps) achieves nothing. The defence is defeated cheaply, not freely.

**Also changed:** the gate now does visible work at the bottom of the range. The 2-step attacks
fail validity and are reported as **no claim possible** — they are not evidence the defence held.
Under the broken harness every budget passed, including ones doing almost no work, because validity
was computed on the same base-encoder pipeline. **That is a limit of the gate worth recording: it
cannot detect a defective evaluation, only a weak attack.**

## 3. Scope

- Two concepts, one erasure family (official AdvUnlearn), lr 1e-5, 16 images, seeds {0,1,6} at the
  300-step point and seed 0 only at 2 and 50 steps.
- The curve is located between 2 and 300 steps but not finely resolved; the 50-step point shows
  partial recovery differs by concept (0.533 vs 0.800), so the threshold is concept-dependent.
- H is a 15-sample closed-set CLIP measure at a 1.000 ceiling (`T2I_BASELINES.md` §1).
- Attacker images are base-model generations; a real adversary with photographs would be at least
  as strong.
- Prompt-space attacks were not run — that is the regime AdvUnlearn was built for and demonstrably
  handles.
