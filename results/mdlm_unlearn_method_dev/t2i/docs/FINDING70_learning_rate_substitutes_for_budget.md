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

## 6. Completion (E11g) — the full 3×3 factorial. **Five gradient steps suffice for complete recovery**, the budget axis is **non-monotone** at high lr, and the mechanism is damage.

All 13 cells, `utrap_church_l0.15k8`, n = 3 wherever a cell was swept (the three `1e-5` cells above
25 steps remain n = 1 and are marked):

| lr | steps | valid | mean H | sd | **max-over-valid** | **post-attack utility (mean)** |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-5 | 5 | 3/3 | 0.3111 | 0.2775 | 0.5333 | 0.9514 |
| 1e-5 | 10 | 3/3 | 0.4444 | 0.2776 | 0.6667 | 0.9375 |
| 1e-5 | 15 | 3/3 | 0.7111 | 0.1924 | 0.9333 | 0.9583 |
| 1e-5 | 25 | 3/3 | 0.8889 | 0.0385 | 0.9333 | 0.9236 |
| 1e-5 | 50 | 1/1 | 0.9333 | — | 0.9333 | 0.9167 |
| 1e-5 | 100 | 1/1 | 1.0000 | — | 1.0000 | 0.9167 |
| 1e-5 | 200 | 1/1 | 1.0000 | — | 1.0000 | 0.9583 |
| 5e-5 | 5 | 3/3 | 0.6000 | 0.1764 | 0.8000 | 0.8750 |
| 5e-5 | 10 | 3/3 | 0.8444 | 0.2143 | 1.0000 | 0.8681 |
| **5e-5** | **25** | 3/3 | **1.0000** | **0.0000** | **1.0000** | **0.9236** |
| **1e-4** | **5** | 3/3 | 0.7111 | 0.2524 | **1.0000** | 0.8819 |
| 1e-4 | 10 | 3/3 | 0.9555 | 0.0385 | **1.0000** | 0.8472 |
| **1e-4** | **25** | 3/3 | **0.6222** | 0.2341 | 0.8667 | **0.5278** |

**Every one of the 13 cells fails the mandate's robustness half.**

### 6a. The cheapest complete recovery is **five** gradient steps

At **lr 1e-4, 5 steps**, seed 6 reaches **H = 1.0000** — the full ceiling, 15/15, on a
gate-valid attack. Against the published headline of **300 steps** that is a **60× overstatement**
of the attacker's cost in the best case.

The more defensible figure remains **lr 5e-5, 25 steps**: H = 1.0000 on **all three seeds**,
sd = 0.0000, and post-attack utility **0.9236** — i.e. the adversary gets the concept back
completely, reproducibly, and leaves the model essentially intact. That is the operating point an
actual attacker would choose, and it is **12× cheaper than what we reported**.

### 6b. The budget axis is **not monotone**, which corrects `FINDING69` §2

`FINDING69` §2 described recovery as a monotone dose-response. **That was true only of lr 1e-5**,
the single rate it swept. At **lr 1e-4** recovery **falls** from 0.9555 (10 steps) to **0.6222**
(25 steps) — more attacker budget produces *less* target generation.

**The mechanism is in the records already and needs no new run: post-attack utility collapses.**

| lr 1e-4 | 5 steps | 10 steps | **25 steps** |
|---|---:|---:|---:|
| mean H | 0.7111 | 0.9555 | 0.6222 |
| mean post-attack utility | 0.8819 | 0.8472 | **0.5278** |
| per-seed utility | — | — | 0.8542 / **0.2917** / **0.4375** |

So past ~10 steps at 1e-4 the attack stops relearning and starts **destroying the model**, and a
destroyed model generates fewer recognisable churches too. By contrast **lr 1e-5 never damages
anything** — utility stays 0.92–0.96 at every budget from 5 to 200 steps.

This matters beyond bookkeeping: it means **H alone cannot order attacks**. A low H can indicate
either a weak attack or an overpowered one, and only the paired utility distinguishes them. Every
H-only comparison in this project is exposed to that ambiguity at high lr — including the frozen
grid's own lr 1e-4 column, which `FINDING63` had already noted left utility at 0.167.

### 6c. Variance is a property of attack power, in both directions

sd runs 0.2775 (1e-5/5, under-powered) → 0.0385 (1e-5/25) → **0.0000** (5e-5/25, saturated) →
0.2341 (1e-4/25, **over-powered and damaging**). Low variance marks a *well-matched* attack; both
extremes are noisy. So the `FINDING69` §8 lesson generalises: single-seed numbers are
uninterpretable at either end of the power range, not just the weak end.

### 6d. Limits

- **One arm, one concept, three seeds.** No method comparison; FRP was never swept on lr.
- Three `1e-5` cells (50, 100, 200 steps) are still **n = 1**, inherited from E11. They sit at or
  near the ceiling where variance is small, but they are not held to the n ≥ 3 standard.
- **H quantised to 1/15**; sd 0.2341 is ≈ 3.5 images.
- **DEV prompts** throughout.
- The damage mechanism in §6b is inferred from **paired utility on the same 48-image panel**, which
  is a closed-set CLIP measure, not an image-quality assessment. FID remains absent
  (`ERRATUM_interpretations_r5.md` §10), so "destroying the model" means "losing the measured
  non-target panel", nothing stronger.
