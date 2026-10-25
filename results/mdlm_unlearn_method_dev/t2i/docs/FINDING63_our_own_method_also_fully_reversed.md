# Finding 63 — **UTRAP, the method we designed from all eleven prior failure modes, is also fully reversed: 6/6 valid attacks, every H = 1.000.** Prediction 1 is falsified by its own declared falsifier. Twelve of twelve.

**Date:** 2026-10-25 · **Pre-registration:** `PREREG_H84_unrolled_trap.md`, predictions recorded
before any arm existed
**Method:** UTRAP — **ours**, not a port. Unrolled multi-step adversarial immunization, full-UNet
scope.
**Protocol:** `PROTOCOL_FROZEN_v1`, the identical six-point grid all eleven previous defences were
measured under. No scoring, alias or success-definition change.
**Evidence:** `outputs/gen_utrap_church_l*.json`, `outputs/atk_H84_*.json`,
`outputs/utrap_utrap_church_l0.3k8.json`, `scripts/erase_utrap.py`

---

## 1. The result

`utrap_church_l0.3k8` — as-is **target 0.0000 at retain8 0.7333**, i.e. the concept fully
suppressed on a usable model. Six attacks, **all six gate-passing** with T rising from **0.000 to
0.958–1.000**:

| seed | lr | T before → after | rise | gate | target H | utility |
|---|---|---|---|---|---|---|
| 0 | 1e-5 | 0.000 → 1.000 | **+1.000** | ✅ | 0.000 → **1.000** | 0.688 → 0.938 |
| 1 | 1e-5 | 0.000 → 1.000 | **+1.000** | ✅ | 0.000 → **1.000** | 0.792 → 0.938 |
| 6 | 1e-5 | 0.000 → 0.958 | +0.958 | ✅ | 0.000 → **1.000** | 0.833 → 0.917 |
| 0 | 5e-5 | 0.000 → 1.000 | **+1.000** | ✅ | 0.000 → **1.000** | 0.688 → 0.875 |
| 1 | 5e-5 | 0.000 → 1.000 | **+1.000** | ✅ | 0.000 → **1.000** | 0.792 → 0.896 |
| 0 | 1e-4 | 0.000 → 1.000 | **+1.000** | ✅ | 0.000 → **1.000** | 0.688 → 0.167 |

**max-over-valid H = 1.000.** `PREREG_H84` prediction 1 — *"UTRAP reaches max-over-valid H < 1.000
… **Falsified by H = 1.000**"* — is **falsified by exactly the criterion it named.** Prediction 2
(*"the one I actually doubt: it does not meet the mandate's bar"*) is confirmed.

**The attack restores utility too**, at every lr except the most aggressive: 0.688–0.833 → 0.875–0.938.
Only lr 1e-4 damaged the model (0.167) and it still reached H = 1.000, so the attacker never needed
to be destructive.

## 2. This is the strongest form of the phase's negative result

UTRAP was not a guess. Every design choice was derived from a measured failure (`PREREG_H84` §1):

| the failure | UTRAP's response | did it help? |
|---|---|---|
| IMMA's inner loop is **one step**; 3× budget walks past it (`FINDING59`) | **K=8 unrolled steps of the real attack** | **no** |
| GIFT/IMMA constrain **5.11%** of the UNet; a **0.046%** LoRA reverses both (`FINDING58`) | **full 859.5M-parameter scope** | **no** |
| proximity and erasure depth are orthogonal to resistance (`FINDING60`, `FINDING62`) | no anchor, no depth term | n/a |
| obstructing the adversary's **optimisation** was the one positive signal (`FINDING56` §5a) | that mechanism, scaled up | **no** |

Both structural causes of the previous failures were removed, and the outcome did not move by a
single quantisation step. The prereg said what that would mean, and it says it unchanged:

> *"the phase's conclusion hardens from 'every published defence fails' to **'every defence we could
> design from the failure modes also fails'**. That is a weaker outcome for the mandate but a
> stronger scientific statement, and it will be reported as such without softening."*

**Twelve defences, eleven published or derived from publications and one of our own, all reversed to
the base model's ceiling by 300 gradient steps on 16 self-generated images.**

## 3. Where the mandate stands, stated plainly

The mandate asks for a method that **outperforms the strongest relevant existing methods on a
clearly defined forgetting–utility trade-off**, supported by real-model results.

- **On forgetting-under-attack: we have no method, and neither does the literature.** No arm —
  eleven published-or-derived, one ours — reached retain8 ≥ 0.85 with max-over-valid H ≤ 0.40. The
  bar is unmet and I am not going to present anything as meeting it.
- **On the as-is trade-off: there is a real result, and it is modest.** `wsr2_church_lr100`
  (`FINDING62`, as corrected) holds retention **0.9639 ± 0.0127** against our FRP port's 0.9417
  while erasure depth is statistically indistinguishable — a **reliable-retention** gain, not deeper
  erasure, and explicitly narrowed after seed replication refuted the stronger claim.
- **The defensible contribution of this phase is the negative result and its mechanism**, not a
  method: that concept erasure in SD-1.4 suppresses expression rather than removing capability, that
  this holds across four adversary families down to **768 trainable scalars**, that neither erasure
  depth nor weight-space proximity predicts resistance, and that even a defence trained against the
  real multi-step attacker over the full parameter set does not change the outcome.

## 4. What UTRAP does not establish, and two builds that died

- **Not that the idea is unworkable in general.** `λ` is sharply bounded — 0.3 is usable, 1.0
  **destroys** the model (retain8 0.0000) — so the usable window is narrow and was explored with one
  arm, not a frontier. K was tested at **8 only**: the λ=1.0/**K=4** arm and the λ=3.0/K=8 arm both
  **died mid-build when `/data` hit 100%**, and are logged as `BUILD FAILED — NOT a kill-rule
  result` so they cannot be mistaken for utility failures. **K's effect is therefore unmeasured.**
- **The retain floor was already missed before the attacks.** At retain8 0.7333 this arm could not
  have met the mandate even with perfect resistance; a higher-retain variant is the obvious follow-up
  and is **blocked on disk**, not on science.
- **The approximation stands as disclosed.** `PREREG_H84` §2 named the detached trajectory in advance
  as the limitation most likely to explain a failure. It remains a live alternative explanation: a
  fully-differentiated unroll might behave differently, and that is not what was run.
- **K4 did not bind** — H is at the ceiling, so no escalation can raise it and none was needed. That
  is the one small mercy: unlike `FINDING59` and `FINDING62`, there is no sub-ceiling number here to
  be provisional about.

## 5. Scope

One concept (church), one teacher (`frp_church_r3_l05`, shared with the H80/H82/H83 arms for
comparability), one seed per build, 400 outer steps, K=8, λ ∈ {0.3, 1.0} measured. Six attacks at
the frozen grid's three learning rates. Our own method, our own hyperparameters, and a null result
reported against our own pre-registered prediction.
