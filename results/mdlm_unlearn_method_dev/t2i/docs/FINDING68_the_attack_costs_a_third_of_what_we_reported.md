# FINDING 68 — The attacker's real cost is **≤ 100 steps, not 300**. At **25 steps** — one twelfth of the reported budget — the mandate-qualifying arm is already at **13/15**. The project's headline overstates the attack cost by at least 3×.

**Date:** 2026-10-26 · **Experiment:** E11 · **Closes a named audit gap** (audit §8: *"Use
time-to-threshold, recovery-versus-budget curves, area under a fixed-budget curve… These are
different estimands from final maximum recovery"*)
**Adversary:** `attack_relearn.py` **unmodified**, only `--steps` varied. Seed 0, lr 1e-5 —
identical to the already-measured 300-step point, so the curve is comparable to it by construction.
**Evidence:** `outputs/atk_E11_*.json`, `outputs/e11_budget_curve.json`, `logs/e11_budget.log`

---

## 1. The curve

`utrap_church_l0.15k8` — target 0.0000 / retain8 0.8667, the only SD-1.4 arm that cleared the
mandate's retain floor:

| steps | T before → after | gate | **H after** | as fraction of ceiling |
|---:|---|---|---:|---:|
| **25** | 0.000 → 0.875 | **PASS** | **0.8667** (13/15) | **87%** |
| **50** | 0.000 → 0.958 | **PASS** | **0.9333** (14/15) | **93%** |
| **100** | 0.000 → 1.000 | **PASS** | **1.0000** (15/15) | **100%** |
| 200 | 0.000 → 1.000 | PASS | 1.0000 | 100% |
| 300 *(the reported point)* | 0.000 → 1.000 | PASS | 1.0000 | 100% |

**The ceiling is reached by 100 steps and 25 steps already recovers 13 of 15 images.** Every budget
is gate-valid — even 25 steps raises T by +0.875, far above the 0.20 gate.

## 2. What this corrects in our own reporting

The project's headline has been *"reversed by **300 gradient steps** on 16 self-generated images"*.
That number is now known to be **at least 3× more generous to the defence than the truth**: the
minimum sufficient budget for this arm is ≤ 100 steps, and the quarter-budget attack at 25 steps
reaches 87% of the ceiling.

This is a correction **against** our own defences, not for them. Every statement of the form "300
steps suffices" should be read as "≤ 100 steps suffices, and we never looked below 300."

It also disposes of a cost claim I had already withdrawn on other grounds. `FINDING59` inferred a
**3× cost multiplier** from LoRA succeeding at 900 rather than 300 steps;
`ERRATUM_interpretations_r5.md` §5 withdrew that for want of intermediate budgets. Now that the
interior has been measured for one arm, the direction is the opposite of a defence advantage: the
full-UNet attacker needs **a third** of the budget we were quoting, so the cost axis was not merely
unmeasured, it was mis-signed.

## 3. This is a genuine one-factor dose-response, unlike the one I retracted

`ERRATUM_interpretations_r5.md` §5 withdrew the TI "monotone dose-response" because it varied
**seeds, learning rates and step counts simultaneously**. This curve varies **only `--steps`** —
same arm, same checkpoint, same seed, same lr, same adversary code, same evaluation prompts and
seeds. The monotone rise 0.8667 → 0.9333 → 1.0000 is therefore a controlled curve in the sense the
earlier one was not.

## 4. Utility rises at every budget too

| steps | utility panel before → after |
|---:|---|
| 50 | 0.8333 → 0.9167 (+0.0833) |
| 100 | 0.8333 → 0.9167 (+0.0833) |
| 200 | 0.8333 → 0.9583 (+0.1250) |

So the "recovery without collateral damage" pattern holds across the interior of the budget range,
not only at 300 (where it was 0.8333 → 0.9792). A **25-step** attack recovers most of the concept
*and* improves the measured panel.

## 5. Limits

- **One arm, one seed, one learning rate, one concept.** This locates the threshold for
  `utrap_church_l0.15k8` at lr 1e-5; it does not establish ≤ 100 steps as a general figure. E11b is
  running the same curve on `frp_church_r3_l05` to see whether the threshold is method-specific.
- **H is quantised to 1/15.** 0.8667 is 13/15 and 0.9333 is 14/15, so the 25→50 step difference is
  **one image**. The 25→100 difference (13/15 → 15/15) is two. Do not read the curve as having
  resolution it lacks.
- **The grid is coarse below 100.** The true threshold lies somewhere in (50, 100]; no finer
  bisection was run.
- **Nothing here is a FINAL-prompt measurement.** These are DEV prompts
  (`ERRATUM_split_leak.md`), as every H in the corpus is.
- **It does not change any robustness verdict.** The ceiling was already reached at 300; this only
  shows it was reached sooner. No arm moves toward the mandate's bar, which stays unmet.

## 6. Addendum (E11b) — the same curve on a second method. It saturates **faster**, but the two arms do not start from the same place, so this is not yet a clean method comparison.

`frp_church_r3_l05` (target 0.1333 / retain8 0.9417), same adversary, same seed, same lr, only
`--steps` varied:

| steps | T after | gate | H before | **H after** |
|---:|---|---|---:|---:|
| **25** | 0.958 | PASS | 0.1333 | **0.9333** (14/15) |
| **50** | 0.958 | PASS | 0.1333 | **1.0000** (15/15) |
| 100 | 1.000 | PASS | 0.1333 | 1.0000 |
| 200 | 1.000 | PASS | 0.1333 | 1.0000 |
| 300 | 1.000 | PASS | 0.1333 | 1.0000 |

**Both arms, side by side:**

| arm | H before | 25 steps | 50 steps | 100 steps | first at ceiling |
|---|---:|---:|---:|---:|---:|
| `utrap_church_l0.15k8` | 0.0000 | 0.8667 | 0.9333 | 1.0000 | **≤100** |
| `frp_church_r3_l05` | 0.1333 | 0.9333 | **1.0000** | 1.0000 | **≤50** |

**§2's correction holds across two methods and tightens.** Neither arm needs anything close to 300
steps; FRP is at the ceiling by 50. So "reversed by 300 gradient steps" overstates the attacker's
cost by **3× for UTRAP and 6× for FRP**.

**What this does NOT establish, and the reason is the audit's own point.** The two arms start from
**different H_before** — 0.0000 vs 0.1333, i.e. FRP already produces the target in 2 of 15 images
before any attack. FRP therefore has 13 images to gain and UTRAP has 15, so its faster saturation is
partly or wholly a shorter climb, not a weaker defence. This is exactly the *"comparisons at matched
initial utility"* requirement from audit §8, and it is **not met here**. So:

- **Supported:** both methods are reversed far below the reported budget, and the budget axis was
  being reported at least 3× too generously.
- **Not supported:** that cost discriminates between these two methods. 50 vs 100 is one grid step
  on a coarse grid, between arms with different starting points.

Cost remains the only axis in this project that has shown *any* variation between arms — the H
ceiling discriminates nothing — so it is worth measuring properly. That would need arms matched on
H_before, a finer grid, and more than one seed, none of which this has.

**Still unmeasured: the low end.** Both curves are already ≥ 0.87 at the smallest budget tried.
Nothing below 25 steps has ever been run, so the true floor is unknown and could be far lower. E11c
is running `--steps` ∈ {5, 10, 15} on both arms to find it.
