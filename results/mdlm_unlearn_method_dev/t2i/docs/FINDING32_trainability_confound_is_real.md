# Finding 32 — the trainability confound is real, measurable, and inseparable from "robustness" on this ladder

**Date:** 2026-10-22 · **Model:** Stable Diffusion v1.4 (pinned) · **Concept:** church
**Pre-specified in:** `FINDING30` §4 (MDLM phase), implemented here
**Status:** the measurement `FINDING30` said was the most important open item in the project. It
came back positive for the confound, and it weakens damage+distil robustness claims in **both**
phases — including my own.

---

## 1. The question

Damage+distil arms resist relearning attacks. Two explanations were pre-registered as
indistinguishable without this test:

| explanation | implication |
|---|---|
| **(a) fact-specific** — the erased concept is genuinely harder to resurface | robustness, as claimed |
| **(b) general trainability loss** — the damaged model is harder to fine-tune on *anything* | the "robustness" is a degraded optimisation landscape, and every robustness number is inflated |

## 2. Why the obvious control does not work, and what replaces it

`FINDING30` §4 specified: attack the arm on a **retained** concept and compare learning speed.
Implemented with `tench` (never an erasure target on any arm) — and the reference arms **fail the
validity gate**:

| run | T before → after | gate |
|---|---|---|
| base SD v1.4, tench | 1.000 → 1.000 | **FAIL** (rise +0.000) |
| UCE teacher, tench | 0.875 → 1.000 | **FAIL** (rise +0.125) |

They cannot rise by 0.20 because they already know tench. **A rise-based gate cannot measure
trainability on a concept the model already has** — the same ceiling artifact already flagged for
the base-model control. This is a limitation of the gate as formulated in `FINDING20`, found by
trying to use it outside its design range.

**Replacement measure, ceiling-free: fit speed on the attacker's own data.** The attack's training
loss has no ceiling and is defined on every arm. Reported as the mean of the last 3 logged losses,
averaged over the 3 attack budgets.

This is a **controlled** comparison: identical attacker images (same base-model generations, same
seeds), identical budget, identical optimiser. Only the arm differs.

## 3. The measurement

| arm | α | retain8 | **late attack loss** | max gated H after attack |
|---|---|---|---|---|
| UCE teacher | 0.0 | 0.850 | **0.3394** | 1.000 (3/3 gated) |
| UNDO α=0 control | 0.0 | 0.850 | **0.3393** | 1.000 (3/3 gated) |
| UNDO α=0.3 | 0.3 | 0.458 | **0.3823** | 1.000 (3/3 gated) |
| UNDO α=0.5 | 0.5 | 0.092 | **0.4397** | 0.400 (1/3 gated) |
| UNDO α=0.7 | 0.7 | 0.000 | **0.4864** | no valid attack (0/3) |

> **Damage makes the model monotonically harder to fit, with no exceptions.** Late loss rises
> 0.3393 → 0.4864 across the ladder; slope **+0.2076 per unit α**.

Two internal checks that the measure is sound:

- The two independent α=0 arms — the UCE teacher and the damage-free distilled copy of it — agree
  to **four decimal places** (0.3394 vs 0.3393). The measure is precise at the scale of the effect.
- Gate failures track it exactly: 0/3 attacks fail at α=0, 0/3 at α=0.3, **2/3 at α=0.5**, **3/3 at
  α=0.7**. Attacks fail the gate precisely where the arm has become hard to fit.

## 4. The decisive detail: degradation arrives *before* any benefit

**α=0.3 is already measurably harder to fine-tune (+0.043 late loss, a 13% rise) and buys exactly
zero robustness — still restored to 1.000 by all three attacks.** It has also already lost half the
model's utility (retain 0.850 → 0.458).

So on this ladder the ordering is: utility loss and trainability loss come first; robustness
appears only at α=0.5, by which point retain is 0.092 and two thirds of attacks can no longer
establish validity.

> **Explanation (b) is not merely possible here — it is the better-supported reading.** The only
> arm showing resistance is also the arm that is nearly untrainable and generates almost nothing
> correctly. On this ladder, "robust" and "broken" are not separable.

## 4a. The non-targeted-concept control, and a sharper result than expected

`tench` was never an erasure target on any arm, so attacking it isolates general fine-tunability
from anything specific to the erased concept.

| arm | α | fit speed on **tench** | tench accuracy before → after attack |
|---|---|---|---|
| base SD v1.4 | 0.0 | **0.3111** | 1.000 → 1.000 |
| UCE teacher | 0.0 | **0.3115** | 0.933 → 1.000 |
| UNDO α=0.3 | 0.3 | **0.3318** (+6.6%) | **0.133 → 1.000** (gate PASS, T rise +1.000) |

Two things fall out, and the second is the more important.

**(i) General trainability loss is real but MILD at α=0.3.** +6.6% on a concept the erasure never
targeted — the same direction as §3's church measurement (+13%) and about half the size. Damage
does degrade fine-tunability globally, not only on the erased concept, which is the signature of
explanation (b). But at α=0.3 the degradation is nowhere near enough to stop an attacker.

**(ii) The damage's own collateral destruction is also trivially reversible.** Damage at α=0.3
destroyed tench as a side effect (1.000 → 0.133, consistent with retain 0.458). **The attacker
restored it to 1.000 in 300 steps**, on a concept it was never asked to erase.

> So at α=0.3 **everything the damage destroyed comes back under attack** — the targeted concept
> (church 0.000 → 1.000) and the collateral casualties (tench 0.133 → 1.000) alike. The damage
> buys a large, durable utility loss and no durable forgetting in either direction.

This is the asymmetry `PREREG_H40` predicted, now measured on both sides: re-associating a text
direction with intact visual features is cheap for *any* concept, targeted or not, while the
utility cost of the damage is immediate and permanent unless repaired.

## 5. What this does to my own claims

**T2I (H38):** the α=0.5 resistance (H 1.000 → 0.400) cannot be credited to erasure. Combined with
retain 0.092, there is no usable operating point, and the apparent robustness is confounded.

**MDLM phase — and this is the uncomfortable part.** `FINDING31`'s headline result stands on a
disjoint band separation between suppression-only and damage+distil arms. `FINDING30` already
observed the same signature there: the treated arm admitted 1 of 6 attacks versus the baseline's
5 of 6, with T_rise peaking at 0.234 vs 0.428. This finding shows that in a different architecture,
modality and erasure method, that signature **is** trainability loss, scaling smoothly with damage.

That **raises the prior** that the MDLM band separation is partly the same artifact. It does not
settle it: the MDLM arms were repaired for 3,000–12,000 steps on ~1,400 prompts and retained
0.63–0.71 utility, so they were *not* broken models the way α=0.5 here is — and the mechanism may
genuinely differ at 124M with adequate repair.

**⚠ RESOLVED — see `FINDING33`.** The MDLM control has now been run and it **clears** `FINDING31`.
On the MDLM backbone the attacker removes **93.6–98.9%** of its available loss on every arm and
every attack passes the gate, while held-out recovery still collapses +0.5608 → +0.0000. A
5-point fit difference cannot explain that, so there the robustness **is** fact-specific. The
qualification I recorded against my own strongest result is lifted; the T2I findings in this
document are unaffected.

The two phases disagree because of **repair capacity**: MDLM repair restored utility to 0.63–0.71
and the arms stayed trainable, while T2I repair left retain at 0.09–0.32 and the arms became hard
to fit. `FINDING33` §3 sets that out as a testable prediction rather than a story.

## 6. Scope and limits

- One concept (church), one erasure method as teacher (UCE), one backbone, seed 0.
- Late attack loss is a **proxy** for trainability. It is measured on the attacker's objective, so
  it is the right proxy for *this* attack, but a different adversary could have a different
  landscape. It is not a general claim about the arm's fine-tunability on all tasks.
- The losses are single-sample per logged step (one latent, one timestep), averaged over only 3
  logged points × 3 budgets. The **monotonicity across five arms** and the 4-decimal agreement of
  the two α=0 arms are what carry the result, not any individual value.
- The non-targeted-concept comparison is now measured and reported in §4a; §3 never depended on it.
- This finding does **not** show damage+distil is worthless in general. It shows that on this
  ladder, in T2I, robustness and brokenness could not be separated — and that the control which
  would separate them is now available and cheap.
