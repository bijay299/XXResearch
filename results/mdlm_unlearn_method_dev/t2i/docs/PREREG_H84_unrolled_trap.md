# Pre-registration H84 — **UTRAP**: unrolled multi-step adversarial immunization, full-UNet scope

**Written:** 2026-10-25, **before any UTRAP arm exists.**
**This is OURS**, not a port. It is a deliberate, non-obvious combination of existing techniques,
which the mandate explicitly permits.
**Target:** the mandate's bar — retain8 ≥ 0.85 with **max-over-valid H ≤ 0.40** on
`PROTOCOL_FROZEN_v1`. Eleven defences have been measured against this bar and **none has met it**.

---

## 1. Designed from the failure modes, not from a paper

Every one of the eleven reversals in this phase has a diagnosed cause, and three of those diagnoses
point the same way:

| finding | what it established | what it implies for a design |
|---|---|---|
| `FINDING59` §0 | IMMA's inner loop approximates the attacker with **one gradient step**; a 3×-longer attacker walks past it | the defence must be trained against a **multi-step** attacker, not a single step |
| `FINDING58` §3a, `FINDING59` §3 | GIFT and IMMA constrain **ψ = 5.11% of the UNet**; an 859.5M-parameter adversary ignores it, and a **0.046%** LoRA on 16 output projections suffices to reverse both | the defence must act on the **full parameter set**, not a chosen subspace |
| `FINDING57` §4 | the concept returns through the **surviving** weights; any 3% of feed-forward is redundant | subspace-restricted interventions cannot work in principle here |
| `FINDING60`, `FINDING62` | weight-space proximity (1470× range) and erasure depth are **both orthogonal** to resistance | do **not** optimise for proximity or for deeper erasure |
| `FINDING56` §5a | Meta-Unlearning's trigger **did** obstruct Textual Inversion's *optimisation* at matched utility — the only positive signal in the phase | obstructing the attacker's **optimisation** is the one mechanism with evidence behind it |

**The design follows directly:** take the one mechanism that demonstrably obstructed an adversary
(optimisation obstruction), fix the two structural reasons it failed elsewhere (one-step inner
approximation, 5% scope), and optimise against the **strongest** adversary in the phase rather than
the weakest.

## 2. The objective

Let `A_K(θ)` be **K steps of the actual full-UNet relearning attack** — the same attack
`attack_relearn.py` runs, on 16 self-generated target images, from θ.

```
min_θ   L_retain(θ)  −  λ · L_target( A_K(θ) )
```

- `L_retain(θ)` — ordinary diffusion loss on the `pools_v2` retain set. Holds utility.
- `−λ · L_target(A_K(θ))` — **maximise the attacker's target loss at the END of its K-step
  trajectory.** Not at θ. Not after one step. At the point the attacker actually reaches.
- **Scope: all 859.5M UNet parameters.** No ψ subset.

The gradient through `A_K` uses the first-order (detached-trajectory) approximation: run K attacker
steps, evaluate `∇L_target` at `θ_K`, apply it at θ. **This is the same approximation class my
Meta-Unlearning port used and `FINDING56` §4 disclosed — but applied over K steps instead of one,
which `FINDING59` identified as the operative difference.** It is stated here, in advance, as the
port-level limitation most likely to explain a failure.

**Pre-declared:** λ = 1.0 primary (sweep 0.3 / 1 / 3), **K = 8** inner steps primary (sweep 4 / 8),
inner lr 1e-5 matching the real attack's primary rung, 400 outer steps, teacher
`frp_church_r3_l05` (so this is directly comparable to the H80/H82/H83 arms built on the same
teacher).

## 3. Why this is not just "Meta-Unlearning with more steps"

Three differences, each tied to a measured failure:

1. **The inner loop is the real attack at its real budget**, not a simulated step — IMMA did this
   but on 5% of the parameters; Meta-Unlearning did neither.
2. **The outer objective targets the attacker's endpoint**, so what is being made difficult is the
   attack's *outcome*, not its first step.
3. **No weight-space anchor and no deeper-erasure term**, because `FINDING60` and `FINDING62`
   measured both to be orthogonal to resistance. Adding them would be optimising a variable already
   shown not to matter.

## 4. Protocol — unchanged, deliberately

`PROTOCOL_FROZEN_v1`: the identical six-point grid (seeds {0,1,6}@1e-5, {0,1}@5e-5, {0}@1e-4; 300
steps, 16 images) every one of the eleven defences was measured under. `FINDING20` validity gate.
`max-over-valid` H. **No scoring, alias or success-definition change** — that constraint is explicit
and this arm is measured exactly as its predecessors were.

## 5. Kill rules, declared now

- **K1** retain8 < 0.60 as-is ⇒ reported as destroying the model, not attacked.
- **K2** a low H with post-attack utility < 0.50 ⇒ **self-destruction, not robustness** (the
  `PREREG_H80` K3 rule; `FINDING56` §3a showed attacks can even *repair* a trap's damage).
- **K3** gate failures contribute nothing and are reported as attacks that did not work.
- **K4 — the one that caught me twice today.** If max-over-valid H lands **below** the ceiling, the
  result is **provisional and will NOT be published as a headline** until the full escalation has
  run: LoRA, Textual Inversion, DreamBooth, **and 3× the attack budget on whichever family scores
  highest.** `FINDING59`'s headline died to a 3× LoRA budget and `FINDING62`'s died to two extra
  seeds, both within hours of publication. **The escalation runs before the claim, not after.**
- **K5** any as-is operating point is reported at **n ≥ 3 seeds** or explicitly as a single build.

## 6. Predictions, recorded before any arm is built

1. **Primary:** UTRAP reaches **max-over-valid H < 1.000** — i.e. it is not fully reversed — at
   retain8 ≥ 0.60. Reasoning: it is the first defence here trained against a multi-step attacker
   over the full parameter set, and the two structural causes of the previous failures are both
   removed. **Falsified by H = 1.000.**
2. **Secondary, and the one I actually doubt:** it does **not** meet the mandate's bar. I expect
   partial resistance that the K4 escalation then erodes, exactly as IMMA's did. Predicting
   otherwise would be ignoring eleven consecutive reversals.
3. **Cost:** K=8 inner steps make each outer step ~9× a plain step, so 400 outer steps ≈ 3600
   forward/backward passes — comparable to the 1500-step budgets used elsewhere in the phase.

**If prediction 1 fails and UTRAP is also reversed to 1.000, that is the twelfth reversal and the
phase's conclusion hardens from "every published defence fails" to "every defence we could design
from the failure modes also fails."** That is a weaker outcome for the mandate but a stronger
scientific statement, and it will be reported as such without softening.
