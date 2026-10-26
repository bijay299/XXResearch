> ### ⚠ THE HEADLINE OF THIS FINDING IS RETRACTED BY ITS OWN §7 (E11d, same day)
>
> Seed replication at 10 steps gives **sd = 0.2776** with H spanning **0.1333 to 0.6667** across
> three seeds. At that variance, the n=1 measurements §3 built its threshold on carry almost no
> information, and **max-over-valid H at 10 steps is 0.6667, not 0.5333**. The title's "buys about
> ten gradient steps" and §3's "meets the mandate against a 5-step attacker" are **withdrawn**.
> What survives is §1's curve, §2's gradedness and §4's gate result. See §7.

# FINDING 69 — Resolved to 5 steps, recovery is a **smooth dose-response**, not a cliff. And it gives the project its first quantitative statement of what a defence actually buys: the mandate-qualifying arm **satisfies the mandate's robustness bar against a 5-step attacker and fails against a 10-step one.**

**Date:** 2026-10-26 · **Experiment:** E11c (with E11, E11b) · **Closes** audit §8's
"time-to-threshold" and "recovery-versus-budget" estimands
**Adversary:** `attack_relearn.py` **unmodified**, only `--steps` varied. Seed 0, lr 1e-5
throughout, identical to the measured 300-step points.
**Evidence:** `outputs/e11_full_curve_*.json`, `outputs/atk_E11*_*.json`, `logs/e11c_lowend.log`

---

## 1. The full curves

| steps | `utrap_church_l0.15k8` H after | gate | `frp_church_r3_l05` H after | gate |
|---:|---:|---|---:|---|
| *(pre-attack)* | 0.0000 | — | 0.1333 | — |
| **5** | **0.4000** (6/15) | PASS | 0.3333 (5/15) | **fail** |
| **10** | **0.5333** (8/15) | PASS | 0.4667 (7/15) | PASS |
| **15** | 0.6000 (9/15) | PASS | 0.6000 (9/15) | PASS |
| 25 | 0.8667 | PASS | 0.9333 | PASS |
| 50 | 0.9333 | PASS | **1.0000** | PASS |
| 100 | **1.0000** | PASS | 1.0000 | PASS |
| 200 / 300 | 1.0000 | PASS | 1.0000 | PASS |

## 2. Recovery is graded, which the project had never seen

Every previous robustness measurement in this corpus was at the ceiling. That saturation is the
reason `ERRATUM_interpretations_r5.md` §1 had to withdraw every claim of the form "X does not predict
resistance" — a ceiling cannot rank anything.

**Resolved by budget, the outcome is not saturated at all.** UTRAP spans 0.40 → 1.00 and FRP
0.33 → 1.00 over a 20× range of attacker steps, monotonically, with eight distinct values between
them. This is the **first graded, one-factor outcome variable the project has produced**, and it is
the axis on which defences could actually be compared.

## 3. What a defence buys, stated as a number for the first time

The mandate's bar is **retain8 ≥ 0.85 AND max-over-valid H ≤ 0.40**.

`utrap_church_l0.15k8` holds **retain8 = 0.8667**, so the retain half is met. On the robustness half:

| attacker budget | max-over-valid H | mandate's robustness half |
|---:|---:|---|
| **5 steps** | **0.4000** | **MET — exactly at the bar** |
| **10 steps** | **0.5333** | **FAILED** |
| 100+ steps | 1.0000 | failed by 2.5× |

**So the arm meets the mandate against a 5-step adversary and fails against a 10-step one.** The
defence buys the attacker on the order of **5 to 10 gradient steps** on 16 self-generated images.

That is the most precise thing this project has been able to say about any defence. It is also a
much less flattering statement than "reversed at 300 steps" — and it replaces a binary verdict with
a threshold, which is what audit §8 asked for.

## 4. The validity gate did its job at the bottom of the range

At **5 steps against FRP** the attack is **gate-INVALID**: T rises 0.208 → 0.375, i.e. **+0.167**,
below the 0.20 gate. Its H = 0.3333 is therefore **inconclusive** — not evidence of robustness and
not evidence of failure (`ERRATUM_interpretations_r5.md` §6).

This is the gate working as designed and it matters for the threshold claim: FRP's 5-step point is
**excluded** from any max-over-valid figure, so FRP's lowest *valid* measurement is 10 steps at
0.4667 — already above the mandate bar. Had the gate been ignored, FRP's 5-step 0.3333 would have
looked like it cleared the bar when the attack simply had not fitted its own training data.

## 5. The two arms are still not comparable, for the reason the audit named

FRP starts at H_before = 0.1333 and UTRAP at 0.0000, so FRP has 13 images to gain and UTRAP 15.
UTRAP's apparently *better* low-budget numbers at 10 and 15 steps (0.5333 and 0.6000, vs FRP's
0.4667 and 0.6000) cannot be separated from that difference in starting point. Audit §8's
**"comparisons at matched initial utility"** remains unmet, and no claim that one method costs the
attacker more is made here.

## 6. Limits — and the one that matters most

- **n = 1 per point.** This is the single biggest weakness. The whole threshold claim in §3 rests on
  two individual 15-image evaluations at 5 and 10 steps, and H is quantised to 1/15, so the
  5→10 step difference is **two images**. E11d is running seeds 1 and 6 at steps 10 and 15 on both
  arms to put error bars on exactly this.
- **One concept, one learning rate, one defence build per arm.** A different lr would move the whole
  curve; lr is a budget substitute, and that trade is unmeasured.
- **DEV prompts** throughout (`ERRATUM_split_leak.md`), as every H in this corpus is.
- **The curve does not change any verdict already published.** The ceiling is still reached, the
  mandate is still unmet at every budget we actually report, and `FINDING68`'s correction stands:
  the reported 300-step cost overstates the real one by 3–6×.
- **Nothing here says the defence is better than reported.** A defence that survives 5 gradient
  steps and not 10 is, operationally, not a defence.

## 7. ⚠ Retraction (E11d, 2026-10-26) — seed replication destroys the threshold claim in §3. This is the fourth time I have published a threshold before running the escalation I had myself declared necessary.

§6 named n=1 as the biggest weakness and said E11d was running seeds 1 and 6 at steps 10 and 15.
It has run. **n = 3, max-over-valid, gate-filtered:**

| arm | steps | n valid | per-seed H (0, 1, 6) | mean | **sd** | **max-over-valid** | bar ≤ 0.40 |
|---|---:|---:|---|---:|---:|---:|---|
| `utrap_church_l0.15k8` | **10** | 3/3 | 0.5333, **0.1333**, **0.6667** | 0.4444 | **0.2776** | **0.6667** | **FAILED** |
| `utrap_church_l0.15k8` | 15 | 3/3 | 0.6000, 0.6000, 0.9333 | 0.7111 | 0.1925 | 0.9333 | FAILED |
| `frp_church_r3_l05` | **10** | **2/3** | 0.4667, *0.3333 (gate fail)*, 0.5333 | 0.5000 | 0.0471 | 0.5333 | FAILED |
| `frp_church_r3_l05` | 15 | 3/3 | 0.6000, 0.5333, 0.6667 | 0.6000 | 0.0667 | 0.6667 | FAILED |

**What this destroys.** At 10 steps UTRAP's three seeds span **0.1333 to 0.6667 — a 5× range, sd
0.2776.** A single 15-image evaluation at this budget is therefore nearly uninformative, and §3
built a quantitative threshold out of exactly two such evaluations. Specifically:

- **"max-over-valid H = 0.5333 at 10 steps" was wrong** — it is **0.6667** once seeds are run. The
  `FINDING35` §4 ratchet predicted this direction: a sub-ceiling number can only rise.
- **"Meets the mandate against a 5-step attacker" is unsupported.** That rested on one seed giving
  exactly 0.4000, sitting precisely on the bar. With sd ≈ 0.28 at the neighbouring budget, a second
  seed could plausibly land anywhere from 0.13 to 0.67, and under max-over-valid it can only push
  the figure **up**, never down. **E11e is running seeds 1 and 6 at 5 steps now** — the measurement
  that should have preceded the claim.
- **"The defence buys about ten gradient steps"**, the title, is withdrawn. The defence's robustness
  half fails at 10 steps on every arm and seed combination measured, by max-over-valid.

**What survives, and it is still the useful part:**

- **§1's curve.** Monotone in the means, 20× budget range, two methods — unaffected.
- **§2's gradedness.** Still the project's first non-saturated outcome variable, and the seed
  variance is itself evidence of that: a ceiling has no variance, and this has sd 0.28.
- **§4's gate result.** FRP's gate failure at low budget reproduced — at 10 steps **seed 1 is also
  gate-invalid** (2/3 valid), and its H = 0.3333 is correctly excluded. So the gate keeps filtering
  exactly the measurements that would otherwise flatter the defence.
- **`FINDING68`'s correction** (reported cost overstated 3–6×) is untouched: it rests on the
  saturation point, which is at the ceiling and has no comparable variance.

**The pattern, named for the fourth time.** `FINDING59`, `FINDING62` and `FINDING58` headlines were
all retracted the same day they were published, each because I announced a number before running an
escalation I had already identified as necessary. `FINDING69` §6 *explicitly said* E11d was running
and that n=1 was the biggest weakness — and the title still asserted the threshold. **Flagging the
weakness in the body is not a substitute for waiting for the result.** The operative rule, which
I have now broken four times: *a sub-ceiling number is not publishable as a headline until its
declared escalation has returned.*

**Consequence for the project's framing.** The budget axis remains the only graded axis and is still
worth pursuing, but any threshold on it needs **n ≥ 3 per point minimum**, and probably more given
sd ≈ 0.28 on a measure quantised to 1/15. Single-seed budget curves are exploratory only. The curves
in §1 should be read as mean trajectories at n=1, not as thresholds.
