# Finding 61 — my adaptation of *From Dormant to Deleted*'s **method** found **no usable operating point**. The failure is attributable to my erasure objective, not to their idea, and the corrected retry is running.

**Date:** 2026-10-25 · **Instrument:** `scripts/erase_wsr.py`, `run_wsr.sh`
**Closes the gap named in** `FINDING60` §5: *"Does not test their method, only their predictor."*
**Evidence:** `outputs/gen_wsr_church_lr{0,1,10,100}.json`, `outputs/wsr_wsr_church_lr*.json`

---

## 1. The result: every arm fails the utility floor

`L = L_erase + L_retain + λ_reg · ‖θ − θ_base‖² / ‖θ_base‖²`, with `L_erase` the **negated**
diffusion loss on 16 self-generated church images, 1500 steps at lr 1e-5.

| λ_reg | target | retain8 | K1 (≥ 0.60) |
|---|---|---|---|
| **0** (ablation, no anchor) | 0.0000 | **0.0000** | ❌ destroyed |
| **1** | 0.0000 | **0.2833** | ❌ |
| **10** | 0.0000 | **0.0667** | ❌ |
| **100** | 0.0000 | **0.2583** | ❌ |

**No arm reaches the 0.60 floor, so none was attacked** and this finding makes **no robustness claim
either way**. The paper's mechanism is untested on the axis that matters, and saying so is the result.

## 2. What the ablation does establish

The λ_reg = 0 ablation was required precisely so the anchor's effect would be attributable, and it
pays off:

- **Without the anchor the model is destroyed outright** (retain8 **0.0000**).
- **With it, retention is partially protected** — 0.2833 at λ_reg = 1.

So the weight-space anchor demonstrably does something real on the **utility** axis. That is a
measured dose-response, not an inference. It just never reaches usability.

## 3. ⚠ Why this is my failure, not theirs

Retention is **non-monotone** in λ_reg: 0.0000 → 0.2833 → 0.0667 → 0.2583. A monotone anchor
strength producing non-monotone utility is the signature of a **different term dominating**, and the
dominant term is mine:

**`L_erase` is an unbounded negated diffusion loss.** Maximising an MSE has no finite optimum, so
1500 Adam steps drive the weights wherever they like and the anchor is fighting a divergent
objective rather than shaping a convergent one. This is the **same failure mode that destroyed my
IMMA arms at α=1e-5** (`FINDING59` §0) — I repeated it.

2505.22310 does not use a negated diffusion loss. It unlearns **examples in a vision classifier**,
where the unlearning objective is bounded by construction. **So the right reading is: my adaptation
of their idea used a badly-chosen erasure term, and the anchor could not rescue it.** Nothing here
licenses a claim about weight-space-regularised unlearning as a method class.

## 4. The corrected retry, and why it is the better experiment anyway

Running now: **anchor an already-erased model instead of erasing from scratch.** Start from
`frp_church_r3_l05` (target 0.1333 at retain **0.9417** — a usable arm), hold the erasure with a
distillation term to that teacher, and apply the anchor toward the **base** weights:

`L = ‖ε_θ − ε_teacher‖²(target) + ‖ε_θ − ε_teacher‖²(retain) + λ_reg · ‖θ − θ_base‖²/‖θ_base‖²`

Every term is **bounded**. This tests the paper's actual proposition — *stay inside the original
basin while holding the unlearning* — starting from a model that works, rather than asking an
unbounded ascent to find its own way there. It is also the construction `H80`/`H82` used, so the arm
will be directly comparable to the Meta-Unlearning and IMMA arms built on the same teacher.

**Prior, recorded before it runs:** I expect it to be **reversed to H = 1.000** like the other ten.
`FINDING60` already showed GIFT at L2 = 0.18 — all but identical to base — reversed on 22 valid
attacks, so proximity to the original basin is already known not to confer resistance in this
setting. The retry's value is that it tests the *method* at a usable operating point, which this
finding could not.

## 5. Scope

One concept (church), one seed per arm, four λ_reg values, 1500 steps. `L_erase`'s unboundedness is
my design choice and is the stated cause of the null. The anchor is normalised by ‖θ_base‖² so
λ_reg is scale-free; λ_reg = 100 therefore represents a genuinely strong anchor, and it still did
not produce a usable model — which is itself evidence that the problem is the other term.
