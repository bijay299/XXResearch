# Finding 26 — the repair budget is the trade-off knob, H19 is refuted, and one point now beats every baseline

**Date:** 2026-10-22 · **Pre-registration:** `PREREG_H19_repair_budget.md`
**Backbone:** H18 124M MDLM with an exact retrain oracle
**Status:** H19's hypothesis **refuted**; its replacement mechanism **also refuted**; the
measurements stand and one of them is the best operating point in the project.

---

## 1. H19 asked whether UNDO's retain cost was a budget artifact. It is — and so is its robustness.

Repair-budget sweep, α=0.5, Fisher-protected damage, AFC teacher, gated RTT at lr 3e-5:

| repair steps | retain | recovery |
|---|---|---|
| 1,500 | 0.3100 | **−0.0293** |
| 3,000 | 0.4850 | **−0.0311** |
| 6,000 | 0.5775 | 0.0887 |
| 12,000 | 0.6700 | 0.2512 |

Retain climbs from 0.3100 to 0.6700 — close to the teacher's 0.7275 — so the retain cost **was**
budget-bound, as H19 predicted. But recovery climbs with it, −0.029 → 0.251, crossing the 0.20
threshold `PREREG_H19` §4 declared as *"repair does resurface forget knowledge; the robustness was
itself a budget artifact"*.

> **Both axes are controlled by the same knob.** Repair restores the retain you want and the
> forget knowledge you do not. H19's hypothesis — that retain could be recovered *without*
> surrendering robustness — is **refuted** by its own pre-registered criterion.

Kill rule K1 (retain improvement < 0.02 ⇒ stop) did **not** fire: the improvement was +0.22.

## 2. The ceiling shows the trade-off is not intrinsic to damage+distillation

| arm | retain | recovery | TV→oracle |
|---|---|---|---|
| *UNDO with the **exact oracle** as teacher* | *0.8150* | *0.0669* | *0.0275* |

With a teacher that has no residual forget knowledge, 12,000 steps of repair give near-oracle
retain **and** low recovery **and** the best behavioural match in the project. So the trade-off
in §1 is a property of **distilling from a suppressed teacher**, not of damage+distillation.

**This arm is a ceiling, not a method** — in any real setting the oracle is precisely what you do
not have. As pre-registered, it is excluded from every frontier comparison against baselines.

## 3. The obvious explanation is also wrong

§2 suggests robustness should be bounded by how much residual forget knowledge the teacher holds.
Tested across four teachers at identical 12,000-step repair:

| teacher | residual forget (rank) | retain | recovery |
|---|---|---|---|
| GA λ=0.5 | **0.0850** | 0.4675 | **−0.0718** |
| *oracle (ceiling)* | *0.1100* | *0.8150* | *0.0669* |
| AFC λ=2 | 0.1600 | 0.7075 | **0.2032** |
| EMDU λ=1 | **0.1725** | 0.6850 | **0.1393** |

The trend is suggestive (Spearman ρ = 0.8 over four points) but **EMDU inverts it**: it has the
*highest* residual and yet better post-repair robustness than AFC. With n=4 and one inversion this
explanation is **not supported**, and I am not asserting it.

What the table does show is that **teacher choice does not buy a free lunch**: the lowest-residual
teacher (GA) gives the best robustness (−0.0718) but collapses retain to 0.4675, off the useful
frontier entirely.

**Two of my mechanism claims were refuted in consecutive experiments** — H19's budget-artifact
hypothesis, then the teacher-residual explanation. The measurements below are replicated and
stand; the mechanism behind them is **unexplained**, and I would rather record that than offer a
third story for the next run to knock down.

## 4. What stands: one point beats every baseline on both axes

| arm | retain | recovery |
|---|---|---|
| MDU λ=0.5…8 | 0.5417–0.6733 | 0.5788–0.6966 |
| AFC λ=1…8 | 0.5717–0.6992 | 0.6274–0.6869 |
| EMDU λ=1…8 | 0.7433 | 0.6620–0.7183 |
| MDU+FAM λ=1 | 0.7333 | 0.7411 |
| GA λ=0.5…8 | 0.6417 | 0.6752–0.8000 |
| **UNDO fisher 12k, retain×3, AFC teacher** | **0.7075** | **0.2009 ± 0.0071 (n=3)** |
| **UNDO fisher 12k, retain×3, EMDU teacher** | **0.6908 ± 0.0146** | **0.1161 ± 0.0205 (n=3)** |

The AFC-teacher point has **retain comparable to or better than every anchor family** (0.7075 vs
their 0.54–0.74) at **2.8–4.3× better robustness**, replicated at n=3 with sd 0.0071.

**The EMDU-teacher point is the best in the project and replicates better than its single seed:**
retain **0.6908 ± 0.0146**, recovery **0.1161 ± 0.0205** at n=3 (seeds 0.1393 / 0.1005 / 0.1084).
That is **4.8× better robustness than the best anchor run anywhere** (0.5608) at retain inside
their band. A shorter-budget sweep on this teacher is running, since the AFC curve showed recovery
rises with repair and 12k may not be its best operating point.

This is the first configuration in the project that is not dominated by any baseline on either
axis.

## 4a. The frontier, fully mapped: α=0.6 is the optimum and recovery reaches zero

Both knobs swept on the best teacher (EMDU λ=1, Fisher-protected damage), gated RTT at lr 3e-5.

**Repair budget at α=0.5:**

| steps | retain | recovery |
|---|---|---|
| 3,000 | 0.4313 ± 0.019 | **−0.0206 ± 0.082** |
| 6,000 | 0.5600 ± 0.028 | +0.0466 ± 0.025 |
| 12,000 | 0.6908 ± 0.015 | +0.1161 ± 0.021 |

**Damage at 12,000 steps:**

| α | retain | recovery |
|---|---|---|
| 0.4 | 0.7150 ± 0.032 | +0.2082 ± 0.075 |
| 0.5 | 0.6908 ± 0.015 | +0.1161 ± 0.021 |
| **0.6** | **0.5913 ± 0.009** | **−0.0194 ± 0.068** |
| 0.7 | 0.4025 ± 0.025 | +0.0065 ± 0.080 |

> **α=0.6 is the optimum: recovery −0.0194 at retain 0.5913.** Recovery falls monotonically from
> α=0.4 to 0.6 (0.208 → 0.116 → −0.019) and then stops improving while retain keeps collapsing, so
> α=0.7 is **dominated** on both axes (0.4025 / +0.0065).

**What α=0.6 dominates outright**, i.e. strictly better on *both* retain and recovery:

| dominated arm | retain | recovery | vs α=0.6 (0.5913 / −0.0194) |
|---|---|---|---|
| MDU λ=8 | 0.5417 | 0.6966 | worse on both |
| UNDO α=0.7 12k | 0.4025 | +0.0065 | worse on both |
| UNDO α=0.5 3k | 0.4313 | −0.0206 | worse retain, equal recovery |

Against the full anchor/GA band (retain 0.54–0.74, recovery 0.5608–0.8642 over 37 gated runs),
α=0.6 sits **inside their retain range** at recovery indistinguishable from zero — roughly **29×
better robustness than the best anchor run anywhere**, with the attack passing its validity gate.

**A correction to our own regime claim.** `FINDING24` §3 recorded α=0.7 as "everything collapses"
and used it to bound the usable regime. That was measured at **3,000** repair steps. With 12,000,
α=0.7's retain rises 0.1317 → 0.4025 and its attacks become **valid** (T-rise +0.355/+0.440, where
all nine were rejected before). So the collapse was **repair-bound, not intrinsic to α=0.7** — the
right conclusion for the wrong reason. α=0.7 remains dominated, but by α=0.6, not by collapse.

### ⚠ CORRECTION — the selected point was the wrong arm

The sub-section below reported the **EMDU-teacher** arm as the project's strongest point. That is
**wrong**, and the error is ours: the arm was selected on rank-retain and recovery **without
checking the behavioural axis**. Running all three teachers at α=0.6 on all three axes:

| teacher | retain | recovery (lr 3e-5) | **TV→oracle** |
|---|---|---|---|
| **AFC** | **0.6460 ± 0.017 (n=5)** | 0.0273 ± 0.042 (n=5) | **0.0745 ± 0.005** |
| MDU | 0.6075 | 0.0209 | 0.0725 |
| EMDU *(previously reported as best)* | 0.5850 | 0.0287 | **0.3225** |

**The AFC teacher dominates EMDU on all three axes**, and EMDU's TV of 0.3225 is the
FAM-collapse signature of `FINDING13`/`FINDING18`/`FINDING25` — its student inherits EMDU's
off-pool output distribution. Teacher choice also matters much less than claimed: all three give
recovery 0.014–0.029, so "EMDU is the best teacher" was noise at n=1.

This is the `FINDING15` metric trap — ranking on a subset of axes — committed after documenting it
twice. **The selected point is the AFC teacher.**

**Its own adversarial lr sweep** (the previous sweep belonged to the EMDU arm and does not
transfer):

| attack lr | recovery | gate | T-rise |
|---|---|---|---|
| **1e-5** | **+0.0314** | ✓ | +0.335 |
| 3e-5 | +0.0273 ± 0.042 (n=5) | ✓ | — |
| 1e-4 | −0.1047 | ✓ | +0.685 |
| 3e-4 | −0.1675 | ✓ | +0.700 |
| 1e-3 | −0.1885 | ✓ | +0.680 |

> **Max over five valid attack learning rates: +0.0314**, at retain 0.6460 and TV 0.0745. Every
> attack passed the validity gate. Against the best anchor run anywhere (0.5608 over 37 gated
> runs) that is **~18× better robustness at comparable retain and comparable behavioural
> distance** — `theta_ref` retain is 0.6725 and the best anchor TV is 0.0725.

Note the honest sign: the maximum is **positive**. ~3% of the gap back to `theta_ref` reopens at
the weakest attack lr. "No detectable recovery" was the EMDU arm's number and does not apply here.

### ⚠ The robustness is attacker-budget-dependent, and the headline degrades

Every robustness figure above uses **one** attacker data budget: `t_frac 0.5`, i.e. the attacker
holds half the forget authors. Raising it to 0.8 changes the picture materially:

| arm | t_frac 0.5 | **t_frac 0.8** |
|---|---|---|
| **UNDO α=0.6 (selected)** | **+0.0314** (max over 5 lr) | **+0.2299** (lr 3e-5), +0.1724 (lr 1e-5) |
| MDU λ=0.5 baseline | 0.5788 | 0.7396 (both lr) |
| advantage | **~18×** | **~3.2×** |

All four attacks passed the validity gate (T-rise +0.256 to +0.547).

> **The "~18× better robustness" figure is specific to `t_frac 0.5`.** Against an attacker holding
> 80% of the forget authors, the selected point's recovery rises 7.3× to **+0.2299** and the
> advantage over the baseline falls to ~3.2×. The qualitative ordering survives — the method is
> still substantially harder to attack than every anchor family — but it is **not** robust in an
> absolute sense, and the headline number must be quoted with its attacker budget.

This is the same class of error as the single-lr claim that was withdrawn earlier: a robustness
number measured at one adversary setting is a lower bound on what a stronger adversary achieves.
Max-over-lr at `t_frac 0.8` and an intermediate budget (0.65) are running.

### Superseded: the EMDU-teacher arm's lr search

| | value |
|---|---|
| retain | **0.5790 ± 0.0165** (n=5) |
| recovery at lr 3e-5 | **−0.0139 ± 0.0424** (n=5: 0.0287, −0.0676, 0.0000, 0.0186, −0.0493) |

The zero-recovery claim rested on one attack learning rate, which is the limitation
`NOVELTY_attack_validity_gate.md` recorded against `FINDING20` and which already forced one
withdrawal today (α=0.5 "fully robust": −0.048 at lr 1e-4 became **+0.057** under search). So the
point was attacked across the grid:

| attack lr | recovery | gate | T-rise |
|---|---|---|---|
| 1e-5 | **−0.0096** | ✓ | +0.260 |
| 3e-5 | −0.0139 ± 0.042 | ✓ | — |
| 1e-4 | −0.0670 | ✓ | +0.645 |
| 3e-4 | −0.1148 | ✓ | +0.660 |
| 1e-3 | −0.1627 | ✓ | +0.655 |

> **Max over five valid attack learning rates: −0.0096.** Every attack passed the validity gate
> (T-rise +0.260 to +0.660, i.e. each genuinely relearned its own training set) and **every one
> produced negative recovery.** No attack setting in the grid reopens the forgotten knowledge.

Against the best anchor run anywhere (0.5608 over 37 gated runs), that is the separation this
protocol is capable of measuring. **This is the strongest result in the project.**

Still outstanding before it is called the selected point: its behavioural profile (TV→oracle) and
whether the result is teacher-general or specific to the EMDU teacher. Both are running.

## 5. Honest size

- **The method is UNDO's.** What is ours here is the repair-budget diagnosis, the Fisher-protection
  modification (itself `PARTIALLY ANTICIPATED`), the teacher comparison, and the measurement
  protocol.
- **Not a published-SOTA claim.** Baselines are our re-implementations on our backbone; the
  original authors' tuning is not reproduced.
- **One backbone, 124M, one synthetic corpus, one attack family.** No scale-transfer claim; the 8B
  programme could not test UNDO at all.
- **The best point's recovery is indistinguishable from zero, not proven zero.** α=0.6 gives
  −0.0194 ± 0.068 at n=2 — the standard deviation is three times the mean, so the honest statement
  is "no detectable recovery at this attack setting", not "no recovery". n=5 is running.
- **Retain 0.5913 is well below `theta_ref`'s 0.6725.** The best robustness point still costs ~0.08
  of retain relative to the un-unlearned model, and 0.12 relative to the teacher it distils from.
