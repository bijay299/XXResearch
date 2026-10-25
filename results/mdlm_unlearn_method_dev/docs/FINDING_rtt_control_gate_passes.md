# The RTT positive control's gate **passes** — measured for the first time. The budget is valid, so the low recovery at it is a real measurement. But it is the *weakest* valid budget, and under max-over-valid the stronger ones govern: **AFC is not relearning-robust.**

**Date:** 2026-10-24
**Evidence:** `outputs/rtt_mdu_l8_lr1e5_gated.json`, `outputs/arm_mdu_l8_s0_rebuild.json`
**Supersedes:** my own repeated statement that "the control does not recover, so the attack is
invalid". With T now measured, that statement is **wrong in its premise** and is corrected here.

---

## 1. What was unknown, and why it mattered

The named RTT positive control, `mdu_l8_lr1e5`, recovered only **8.99%** of the gap back to θ₀ on
held-out facts. It carried **no `attack_validity` block**: it was produced before the gate existed,
so **T** — recovery on the facts the attacker is itself fine-tuned on — was never measured for it.

With T unmeasured, 8.99% is ambiguous between two completely different readings:

- the unlearning genuinely resisted a working attack, or
- the attack never worked at all, in which case AFC's **0%** at the same budget is not robustness but
  an artifact.

I have reported the second reading as the live one in every status update. That was the right default
with T unknown — but it was a statement about missing evidence, not a measured fact, and it is now
settled.

## 2. The measurement

`scripts/train_rtt.py` now computes the gate (threshold **+0.20** absolute rise on T). Re-run at the
control's **own recorded hyperparameters**, read straight out of `rtt_mdu_l8_lr1e5.json` — t_frac
0.2, probe steps 0/10/25/50/100/200, lr 1e-5, rank 16, batch 2, gen 64, n_eval 60, seed 0:

| | value |
|---|---|
| T DISCLOSE before → after | 0.0500 → **0.2833** |
| **T rise** | **+0.2333** |
| gate (≥ 0.20) | ✅ **PASS — valid = true** |
| T as fraction of θ₀ | **0.3144** |
| H (held-out) before → after | 0.0500 → 0.1167 |
| **recovery on H** | **8.99% of the gap to θ₀** |

Per-step: T stays flat through step 25, reaches 0.1667 by step 50, and 0.2833 by step 200 — a real
learning curve, not noise.

**So the attack works, and the 8.99% is a legitimate measurement rather than a broken instrument.**

This also reproduces the original exactly: H 0.0500 → 0.1167, absolute rise 0.0667, 8.99% — identical
to the ungated `rtt_mdu_l8_lr1e5.json` to four decimals, on a **rebuilt** arm
(`mdu_l8_s0_rebuild`, itself an exact reproduction of `mdu_pub`). That is the fourth independent
reproduction in this line today.

## 3. ⚠ But it is the weakest valid budget, and that is what decides the robustness question

Passing at **+0.2333** clears a 0.20 threshold by 0.033, and gets T to only **31% of θ₀**. Compare
the other gated budgets:

| budget | T rise | T as fraction of θ₀ | MDU/EMDU recovery on H | **AFC recovery on H** |
|---|---|---|---|---|
| **lr 1e-5, 200 steps (this one)** | +0.2333 | **0.314** | 8.99% | **−9.52%** (see §4) |
| lr 3e-5, 300 steps (`*_v3`) | +0.475 | 0.619 | 45.6% | **35.3%** |
| 600 steps (`h37_sup_*`) | +0.66 … +0.71 | — | 51–68% | **47–62.7%** |

Under `max-over-valid` — the rule this project and the t2i line both use, because the maximum over
valid attacks only ever rises as more attacks are run — **the strongest valid budget governs, not the
weakest.** At those budgets both arms relearn substantially, and AFC's maximum is **62.7%**, inside
the baseline's 51–68% range.

**Therefore: AFC is not relearning-robust, and the low numbers at this weak budget do not make it
so.** What the gate result changes is the *reason* the claim fails. It is no longer "the control is
invalid so nothing can be said"; it is "the control is valid at this budget, the budget is weak, and
at every stronger valid budget AFC recovers as much as the baseline."

## 4. AFC at the same budget, now gated — and why it is **not** a robustness result

`afc_l1_lr1e5_gated` has landed, on the `afc_l1_s0_bridge` arm (necessary because AFC's original
checkpoint is among those deleted, legitimate because that bridge reproduces `h16_none_f1` exactly):

| | MDU λ=8 (control) | **AFC λ=1** |
|---|---|---|
| T before → after | 0.0500 → 0.2833 | 0.0833 → 0.2833 |
| T rise | +0.2333 | **+0.2000** |
| gate (≥ 0.20) | ✅ pass | ⚠ **pass, exactly at the threshold** |
| T as fraction of θ₀ | 0.314 | 0.282 |
| H before → after | 0.0500 → 0.1167 | 0.2667 → **0.2167** |
| recovery on H | **+8.99%** | **−9.52%** |

So with a valid attack at this budget, AFC's held-out disclosure went **down**, while the control's
went up. Three reasons that is **not** a robustness claim:

1. **The gate passes at exactly +0.2000**, the threshold value. A single probe either way flips
   validity. This is the most marginal valid attack in the project.
2. **The effects are near measurement granularity.** `n_eval = 60`, so one fact is 0.0167; AFC's
   −0.05 is **three facts**, and the control's +0.0667 is four. Neither is distinguishable from
   noise at this n, and a negative "recovery" is on its face a noise reading, not evidence of
   hardening.
3. **⚠ The two arms are not at matched disclosure, so the comparison is apples-to-oranges.** The
   control is MDU **λ=8** — a far deeper erasure, sitting at H 0.0500 before the attack — while AFC
   λ=1 sits at **0.2667**, five times higher. AFC has far less headroom to "recover" toward θ₀'s
   0.792 because it is already disclosing a quarter of the held-out facts at rest. A recovery
   *fraction* computed against a common θ₀ reference is not comparable between arms whose starting
   points differ by 0.22. **The control's job is to validate the attack, which it does; it is not a
   matched baseline for AFC.**

§3's conclusion is unchanged and does not rest on any of this: at the stronger budgets, where both
arms are gated and the attack reaches 62% of θ₀ on T rather than 28%, **AFC recovers 35.3% and up to
62.7%** — inside the baseline's 51–68%. **AFC is not relearning-robust.**

Two things this does **not** establish: that 8.99% is a meaningful level of residual memory (no
never-trained oracle exists here, the same limitation the t2i line records as `ERRATUM_r2` C17), and
that any budget between 1e-5/200 and 3e-5/300 behaves monotonically — the recovery jump from 9% to
46% across those two budgets is large and unexplained.

## 5. Scope

One arm per budget, one seed (0), one dataset. The rebuild substitution is justified by the three
bridges and five exact seed reproductions recorded in
`FINDING_afc_tv_advantage_does_not_survive.md` §5a, not assumed. The gate threshold of +0.20 is the
project's pre-existing convention and was not chosen with these numbers in view.
