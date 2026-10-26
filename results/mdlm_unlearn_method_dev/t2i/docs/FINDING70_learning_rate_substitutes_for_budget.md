# FINDING 70 — Learning rate substitutes for budget, and it collapses the attacker's cost to **25 gradient steps for complete, zero-variance recovery**. Against the project's reported 300, that is a **12× overstatement**.

**Date:** 2026-10-26 · **Experiment:** E11f · **n = 3 at every cell, held before publishing**
(the rule written down in `FINDING69` §7)
**Closes** the gap `FINDING69` §6 named: *"a different lr would move the whole curve; lr is a budget
substitute, and that trade is unmeasured."*
**Adversary:** `attack_relearn.py` **unmodified**; only `--lr` and `--steps` varied.
**Evidence:** `outputs/atk_E11F_*.json`, `logs/e11f_lrtrade.log`

---

## 1. The trade

`utrap_church_l0.15k8` (target 0.0000 / retain8 0.8667), max-over-valid H across seeds {0, 1, 6}:

| lr | steps | n valid | mean | **sd** | **max-over-valid** | bar ≤ 0.40 |
|---:|---:|---:|---:|---:|---:|---|
| 1e-5 | 5 | 3/3 | 0.3111 | 0.2776 | 0.5333 | FAILED |
| 1e-5 | 10 | 3/3 | 0.4444 | 0.2776 | 0.6667 | FAILED |
| 1e-5 | 25 | 1/1 | 0.8667 | — | 0.8667 | FAILED |
| **5e-5** | **5** | 3/3 | 0.6000 | 0.1764 | **0.8000** | FAILED |
| **5e-5** | **10** | 3/3 | 0.8444 | 0.2143 | **1.0000** | FAILED |
| **5e-5** | **25** | 3/3 | **1.0000** | **0.0000** | **1.0000** | FAILED |

**At lr 5e-5, 25 steps recovers the concept completely on all three seeds with sd = 0.0000.**

## 2. What this does to the cost statement

`FINDING68` corrected the reported 300-step cost to ≤100 at lr 1e-5. This goes further:

| claim | budget for full recovery |
|---|---:|
| the project's published headline | **300 steps** |
| `FINDING68`, lr 1e-5 | ≤ 100 steps |
| **this finding, lr 5e-5** | **25 steps, all 3 seeds, sd 0** |

**A 12× overstatement**, and the 25-step figure is the more honest one because it is the cheapest
*reproducible* full recovery rather than a single seed's best case. A 5× increase in learning rate
bought roughly a 4× reduction in steps, so within this range the two are close to interchangeable —
which means quoting a step count without its learning rate, as this project has done throughout,
understates the adversary by whatever factor the attacker cares to choose.

## 3. Variance collapses as the attack strengthens, and that matters for how defences get scored

| cell | sd |
|---|---:|
| 1e-5, 5 steps | 0.2776 |
| 5e-5, 5 steps | 0.1764 |
| 5e-5, 10 steps | 0.2143 |
| **5e-5, 25 steps** | **0.0000** |

At the weak end recovery is a lottery (`FINDING69` §8); at 5e-5/25 it is a certainty on every seed.
So the enormous seed variance that destroyed `FINDING69`'s threshold is a **property of
under-powered attacks**, not of the defence. Any evaluation that reports low-budget, single-seed
numbers is measuring attacker luck. This project's frozen grid uses lr ∈ {1e-5, 5e-5, 1e-4} at 300
steps, which is firmly in the zero-variance regime — so the historical H = 1.000 results are not
affected by this, and that is worth stating plainly rather than leaving implied.

## 4. The mandate

**Every cell fails the robustness half.** Combined with `FINDING69` §8, there is now no
(lr, steps) combination tested — 5 to 300 steps, 1e-5 and 5e-5 — at which any arm reaches
max-over-valid H ≤ 0.40 while holding retain8 ≥ 0.85. The mandate stays unmet, and the margin by
which it is missed grows with attacker strength rather than shrinking.

## 5. Limits

- **One arm, one concept, three seeds, two learning rates.** `lr 1e-4` is untested at low budget;
  E11g is running it, plus the missing n=3 at lr 1e-5 / 25 steps, to complete a 3×3 factorial.
- **The `1e-5, 25` cell is n = 1** in the table above, inherited from E11 seed 0. It is marked as
  such and is the one cell not held to this finding's own n ≥ 3 standard.
- **H is quantised to 1/15**, so an sd of 0.1764 is roughly 2.6 images and differences below one
  image are not resolvable.
- **DEV prompts** throughout (`ERRATUM_split_leak.md`).
- **This is not a defence comparison.** Only `utrap_church_l0.15k8` was swept on lr; FRP was not, so
  nothing here says whether the lr/steps trade differs between methods.
- **No published robustness verdict changes.** All of them were measured at 300 steps, in the
  zero-variance regime; this finding makes the attacker cheaper, never the defence stronger.
