# Finding 62 — ❌ **HEADLINE CORRECTED.** I reported "the best erasure/utility point of the entire phase, beating FRP on both axes". **Seed replication refutes the both-axes claim**: target is 0.1778 ± 0.1018 at n=3, *worse* than FRP's 0.1333 on erasure depth. What survives is better **retention** (0.9639 ± 0.0127 vs 0.9417) — and the arm is **still fully reversed to H = 1.000**.

> ## ⚠ CORRECTION, same day — the seeds I asked for arrived and refuted §1
>
> §4 of this finding flagged its own weakest point: *"the as-is claim rests on a single build per
> λ_reg and would need seeds before being called a stable operating point."* Two more seeds at
> λ_reg = 100:
>
> | seed | target | retain8 |
> |---|---|---|
> | 0 (the published build) | **0.0667** | 0.9500 |
> | 1 | **0.2667** | 0.9750 |
> | 2 | **0.2000** | 0.9667 |
> | **n=3** | **0.1778 ± 0.1018** | **0.9639 ± 0.0127** |
>
> **Target scatters 4× across seeds (0.0667 → 0.2667) and seed 0 was the most favourable of the
> three.** At n=3 the arm's mean target of 0.1778 is **worse** than the FRP teacher's 0.1333, so
> **"beats FRP on both axes" is withdrawn.**
>
> **What survives, and it is narrower:** retention is both **high and stable** — 0.9639 ± 0.0127
> against FRP's 0.9417 — while erasure depth is **unstable** and not better on average. So the
> weight-space anchor buys *reliable retention*, not deeper erasure. That is a real but much smaller
> claim than the one published.
>
> **This is the AFC pattern repeating, in the line where it was first measured.** The MDLM arm's TV
> sd was 0.005 at n=3 and 0.158 at n=5 (`RESULTS.md` retraction); here a single build looked like the
> phase's best point and three builds show it was a lucky draw. The lesson both times: **a
> single-build operating point is not an operating point.**
>
> §2's robustness result is **unaffected** — H = 1.000 on 7 of 7 valid attacks across all three
> λ_reg arms, and seed 0's arm is among them. Nothing about the reversal depended on the as-is
> numbers.

**Date:** 2026-10-25 · **Corrects and completes:** `FINDING61` (the from-scratch null)
**Method:** weight-space-regularised erasure, our adaptation of the idea in arXiv 2505.22310, with
`FINDING61` §4's bounded teacher-distillation correction applied.
**Evidence:** `outputs/gen_wsr2_church_lr{1,10,100}.json`, `outputs/atk_H83B_*.json`

---

## 1. The correction worked

`FINDING61` found that erasing from scratch with an **unbounded** negated diffusion loss destroyed
the model at every anchor strength (max retain8 0.2833), and attributed that to my objective rather
than to the paper's idea. §4 named the fix: start *at* an already-erased teacher and hold the
erasure with a **bounded** distillation term.

| arm | target | retain8 | vs. the FRP teacher (0.1333 / 0.9417) |
|---|---|---|---|
| `wsr2_church_lr1` | 0.1333 | 0.9417 | identical — the anchor is a **no-op** at λ_reg = 1 |
| `wsr2_church_lr10` | **0.0667** | 0.9000 | 2× deeper erasure, 0.042 less retention |
| **`wsr2_church_lr100`** (seed 0) | **0.0667** | **0.9500** | ⚠ **single build — see the correction above; at n=3 this is 0.1778 ± 0.1018 / 0.9639 ± 0.0127** |

**`wsr2_church_lr100` is the best erasure/utility operating point this phase has produced.** It
beats our FRP port — itself the deepest published-method arm here (`FINDING43`) — on **both** axes
simultaneously: half the target accuracy at higher retained capability. It clears the mandate's
retain floor of 0.85 by 0.10.

So the bounded correction turned `FINDING61`'s four-way null into the phase's strongest arm. The
weight-space anchor is doing real work: at λ_reg = 100 it buys both depth and retention over the
teacher it started from.

## 2. And it is reversed exactly like the other ten

All seven attacks passed the `FINDING20` gate (T rises +0.750 to +0.833):

| arm | seed | T rise | gate | target H |
|---|---|---|---|---|
| **`lr100`** (best arm) | 0 | +0.792 | ✅ | 0.067 → **1.000** |
| `lr10` | 0 | +0.792 | ✅ | 0.067 → **1.000** |
| `lr10` | 1 | +0.750 | ✅ | 0.067 → **1.000** |
| `lr10` | 6 | +0.833 | ✅ | 0.267 → **1.000** |
| `lr1` | 0 | +0.750 | ✅ | 0.133 → **1.000** |
| `lr1` | 1 | +0.750 | ✅ | 0.067 → **1.000** |
| `lr1` | 6 | +0.833 | ✅ | 0.067 → **1.000** |

**max-over-valid H = 1.000 on every arm.** The prior recorded in `run_wsr.sh` *before* the retry ran
— *"I expect this to be reversed too, because `FINDING60` showed GIFT at L2 = 0.18 reversed on 22
valid attacks"* — is confirmed.

**Eleven defences are now fully reversed by this adversary.**

## 3. Why this is the right note for the phase to end on

This arm is the strongest version of the counterargument available: it is not a weak defence, a
destroyed model, or a badly-tuned port. It is **the best erasure/utility trade-off anyone in this
phase achieved**, built by applying a published tamper-resistance idea on top of the best published
method, with its own ablation showing the anchor contributes. And **300 gradient steps on 16
self-generated images restore the concept completely.**

Combined with `FINDING60` — where weight distance spans 1470× across arms with **zero** movement in
the outcome — the picture is consistent and now well covered:

> **In this setting, erasure quality and weight-space proximity are both orthogonal to relearning
> resistance.** A better operating point does not buy robustness, and neither does staying in the
> original basin. The only intervention observed to lower max-over-valid H was destroying the model
> (`FINDING60` §4), and the only one observed to obstruct an adversary's *optimisation* was
> Meta-Unlearning's trigger against Textual Inversion specifically (`FINDING56` §5a) — which the
> same arm's full-UNet reversal made irrelevant.

## 4. What is not claimed

- **Not a refutation of 2505.22310.** This is our **adaptation of their idea** to a different domain
  (they do example-level unlearning in vision classifiers), with λ_reg chosen by us and a teacher
  they never used. `FINDING61` §3 records that my first attempt failed for a reason that was mine.
- **One concept, one seed per attack point, lr 1e-5 only.** The ratchet (`FINDING35` §4) does not
  bind — H is already at the ceiling and no further attack can raise it — but the *as-is* claim in
  §1 rests on a single build per λ_reg and would need seeds before being called a stable operating
  point. It is reported as the best point **measured**, not as a reproducible method.
- **The λ_reg = 1 no-op is informative and is not hidden:** at that strength the anchor reproduces
  the teacher exactly, so the λ_reg = 100 result is a genuine effect of anchor strength and not an
  artifact of the distillation alone.
