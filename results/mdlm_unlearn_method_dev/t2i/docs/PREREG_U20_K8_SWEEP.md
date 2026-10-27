# PREREG U20 — complete the attacker sweep on the three K=8 builds

**Declared 2026-10-27, BEFORE execution.** Development data; no confirmatory claim. Frozen on first
result; corrections in a dated amendment.

## 1. Why this and not something else

`MANDATE_v2` §3.1 item 3, as strengthened today, claims: *every arm that retains usable generative
capability is driven to H = 1.000 by a relearning attack of at most 25 steps; the only arms that
resist are ones whose capability has been destroyed.*

Three arms are the sole remaining ambiguity in that claim. `u4_k8_s400_b{0,1,2}` retain usable
capability (retain8 0.9583, 0.9583, 0.9792) yet their max-over-valid `H_after` is **0.9333, 0.9333
and 0.8000** — below 1.000. All nine of their cells sit at attacker lr **1e-5**, and only 1e-5:

| arm | lr | seeds | `H_after` | max |
|---|---|---|---|---:|
| `u4_k8_s400_b0` | 1e-5 | 0, 1, 6 | 0.8667, 0.7333, 0.9333 | 0.9333 |
| `u4_k8_s400_b1` | 1e-5 | 0, 1, 6 | 0.8000, 0.8000, 0.9333 | 0.9333 |
| `u4_k8_s400_b2` | 1e-5 | 0, 1, 6 | 0.6667, 0.7333, 0.8000 | 0.8000 |

U18's church curve shows 1e-5 is **below the peak**: on UTRAP K=0, mean H is 0.267 at 1e-5 and
0.933 at 5e-5 on an identical budget. MANDATE clause 1c states that a single-rate result is a lower
bound, not a robustness estimate. These three arms are therefore very likely under-measured, and
the central claim currently rests on calling them "under-swept" rather than on having swept them.

**This is not an attempt to rescue UTRAP.** K=8 was retired on cost grounds (§3.2) and that decision
does not depend on this experiment. The reason to spend the GPU is that these are the only cells
standing between the central negative result and an unqualified statement.

## 2. Arms and the single variable

Checkpoints already exist; **no defence training is run.** Attack only.

- arms: `u4_k8_s400_b0`, `b1`, `b2` (3 independent defence builds of the same configuration)
- attacker lr: **5e-5 and 1e-4** (1e-5 already done; 2e-4 excluded — U18 and U19 show it is
  outside the useful range for every arm tested, 0/3 valid on each)
- seeds: 0, 1, 6 — the same three used at 1e-5, so the new cells are paired with the old
- fixed: concept church, 25 steps, `attack_relearn_v2.py`, `--validate-reload`

18 cells. Everything except attacker lr is held at the values already used for these arms.

## 3. Pre-declared outcomes

| outcome | reading |
|---|---|
| all three arms reach max-over-valid `H_after` = 1.000 | the central claim becomes unqualified for these arms: **52 of 55** concept-matched arms reach 1.000 and every remaining exception is a destroyed model. The "under-swept" caveat is removed. |
| max-over-valid `H_after` stays **< 1.000 on all three builds**, with cells gate-valid and non-destroyed and retain8 still >= 0.85 | **the project's first genuine resistance.** This would be a real finding and would require revisiting §3.2's retirement of K=8 — not on cost grounds, which are unchanged, but because "no measurable benefit" would no longer be accurate. Would go to n>=3 attack seeds per rung plus a 50- and 100-step budget before any claim. |
| mixed across builds | build-to-build variance dominates, consistent with the 0.5333 spread already measured on these arms; report as inconclusive and leave the caveat in place. |
| cells read destroyed (`T_rise < 0` with `T_before < 0.90`) | excluded per refined clause 1b; if that removes a rung entirely, report the rung as out of range rather than as defence. |

**Primary quantity: max-over-valid `H_after`** per clause 7 — not `delta_H`, which that clause
shows is noise-dominated.

## 4. What U20 cannot establish

- It cannot revive UTRAP as a method. The cost finding (3.3x K=0's wall-clock for a 0.06 difference
  in H against the untreated teacher) is untouched by any outcome here.
- One concept (church), one budget (25 steps), one attack family.
- `H_before` for these arms carries evaluation-seed noise of the size documented in clause 7, so
  per-cell deltas are not interpretable.

## 5. Budget and constraints

18 cells x ~3 min ~= **1.0 GPU-hour**, cap **1.5**. Runs on the single free GPU alongside S1, for
**two concurrent GPU jobs total**, within the standing limit. GPUs 2-3 belong to another user and
are untouched. Reservation via `scripts/gpu_reserve.sh`; space guard at the 500 GiB authorisation
and 25 GiB free on the shared volume, as in U19.
