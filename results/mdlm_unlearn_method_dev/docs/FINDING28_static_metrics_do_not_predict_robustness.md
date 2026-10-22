# Finding 28 — no static metric predicts attack robustness: 125 arms, 18 matched pairs, Spearman −0.15

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM with an exact retrain oracle (`FINDING21`)
**Protocol:** all arms at a **matched** attacker budget `t_frac = 0.5`, every attack passing
`FINDING20`'s validity gate, recovery as **max over valid attacks**
**Status:** the broadest result in the project — it rests on every gated run accumulated, not on a
new experiment. It also subsumes and explains a negative that killed a planned line (§5).

---

## 1. The measurement

Across **125 arms** that have both a closed-pool knowledge probe and at least one valid attack at
the same attacker budget:

> **Spearman(residual forget knowledge, max recovery) = −0.148.**

Not weakly positive. Essentially zero, with the sign pointing the *wrong* way. Residual knowledge
spans 0.1625 on a 0.5840-wide scale (chance 0.1135 → `theta_ref` 0.6975); recovery spans 0.9972.
The axis the field reports and the axis that matters are unrelated.

## 2. The attack separates the arms into two bands; no static metric does

| group | n | **max recovery** | residual forget knowledge |
|---|---|---|---|
| suppression-only (MDU, EMDU, AFC, MDU+FAM, GA; λ ∈ {0.5…8}) | 43 | **+0.5608 … +0.9143** | 0.0825–0.2450 |
| *α=0 controls (distil, **no** damage)* | 2 | *+0.5622 … +0.5742* | *0.1450–0.1725* |
| damage+distil, α > 0 | 80 | **−0.0829 … +0.4424** | 0.1025–0.2225 |

- On **recovery** the two real groups are **disjoint**: the worst damaged arm (+0.4424) is still
  +0.1184 better than the best suppression-only arm (+0.5608). No overlap in 123 arms.
- On **residual knowledge** the same two groups **overlap over 0.1025–0.2225** — 74% of the
  suppression group's entire range.
- The **α=0 controls land inside the suppression band** (+0.5622, +0.5742), which is exactly where
  construction says they belong: they receive no damage. They are the only damage+distil arms
  above +0.45, and excluding them is not cherry-picking — it is the control behaving correctly and
  confirming that *damage*, not distillation, is the separator (`FINDING23`).

## 3. Eighteen exactly-matched pairs

Pairs of arms whose static residual knowledge is **identical to four decimal places**, one from
each family:

| residual | suppression-only arm | recovery | damage+distil arm | recovery | ratio |
|---|---|---|---|---|---|
| 0.1450 | `base_afc_f8p0_s1` | +0.6996 | `undo_temd_6k_s0` | +0.0645 | 10.8× |
| 0.1475 | `base_afc_f8p0_s0` | +0.6977 | `undo_temd_a06_12k_s3` | +0.0186 | 37.5× |
| 0.1550 | `base_mdu_f8p0_s1` | +0.6941 | `undo_s_a0p5` | +0.0574 | 12.1× |
| 0.1600 | `base_mdu_f8p0_s0` | +0.7109 | `undo_b1500` | −0.0293 | >20× |
| 0.1650 | `base_mdu_fam_f8p0_s2` | +0.7291 | `undo_temd_a06_12k_s4` | −0.0493 | >20× |
| 0.1725 | `base_emdu_f8p0_s1` | +0.6633 | `undo_tga_12k` | −0.0718 | >20× |
| 0.1800 | `base_emdu_f4p0_s1` | +0.7538 | `undo_tmdu_a06_s0` | +0.0209 | 36.1× |
| 0.1950 | `base_emdu_f1p0_s2` | +0.7538 | `undo_p_a0p5` | −0.0311 | >20× |
| 0.2125 | `base_emdu_f1p0` | +0.6738 | `h31b_comp_s0` | −0.0106 | >20× |

(9 of 18 shown; the full 18 are in §7 of the generating script's output. The smallest ratio among
all 18 is 1.9×, the largest >20×.)

**A practitioner handed the probe output alone cannot tell these pairs apart.** The probe reports
the same number; the attacker recovers 2–37× more from one than the other.

## 4. And the inversion is worse at the extreme

**Gradient ascent has the lowest residual knowledge of any family measured — below chance —
and among the highest recovery.**

| arm | residual (chance 0.1135) | max recovery |
|---|---|---|
| `base_ga_f0p5` | **0.0825** | +0.7194 |
| `base_ga_f8p0_s2` | 0.0925 | **+0.9143** |
| `undo_h19_ceil` *(oracle teacher, ceiling)* | 0.1025 | **+0.0669** |
| selected point (`undo_tafc_a06`) | **0.1755** | **+0.0788** |

GA arms sit at 0.0825–0.1000 — *below* the 0.1135 chance floor, which on a static reading is
"better than the exact retrain oracle" — while an attacker recovers 0.61–0.91 of the knowledge.
The selected point has **more than twice GA's residual** and recovers **9–12× less**.

Selecting on the static probe would pick GA and be maximally wrong.

## 5. What this explains, and the line it killed

I checked a planned **iterated-UNDO** line before spending GPU time on it: use the selected point
as the teacher for a second damage-and-distil round, reasoning that each round should drive
residual knowledge toward the oracle's 0.1100.

The premise is false. The selected point's residual is **0.1755**, *higher* than its own teacher
class AFC λ=8 at **0.1533** — repair transfers knowledge back. Recursive rounds would start from a
progressively *worse* teacher, so the line is dead as conceived and was not run.

This finding is the general version of that. Along the damage ladder, holding teacher and repair
budget fixed and varying only α:

| α | residual | max recovery | TV→oracle |
|---|---|---|---|
| 0 (no damage) | 0.1450 | +0.5742 | 0.1000 |
| 0.5 | 0.1658 | +0.1221 | 0.0850 |
| 0.6 (selected) | **0.1755** | **+0.0788** | 0.0745 |
| teacher AFC λ=8 | 0.1533 | +0.6996 | 0.0758 |

Residual moves by 0.0305 — about 5% of the chance-to-reference scale, and *upward* — while
recovery falls **7.3×**. Over the two full α ladders the residual-vs-α slopes have **opposite
signs** (+0.0352 and −0.0993), i.e. noise around flat.

> **Damage does not remove knowledge. It removes the attacker's route to it.** This is a
> mechanistic claim about what UNDO buys, and it is why "forget harder" and "distil longer" both
> fail to help.

## 6. Why this matters beyond this project

`FINDING25` §3 showed **behavioural** metrics (TV→oracle, six-cell, DISCLOSE) cannot separate these
methods: TV sits at 0.0725–0.0783 across a λ range that moves recovery by 0.12, and EMDU scores
**DISCLOSE 0.000** while leaving the knowledge fully recoverable.

This finding closes the remaining escape route. The natural response to `FINDING25` is *"use a
knowledge probe instead of behavioural metrics"* — and I had been treating the closed-pool ranking
probe as the trustworthy axis for most of this project. It is not:

| metric class | separates the two robustness bands? |
|---|---|
| behavioural (TV→oracle, six-cell, DISCLOSE) | **no** (`FINDING25` §3) |
| static knowledge probe (closed-pool ranking) | **no** (this finding: ρ = −0.15, 74% band overlap) |
| validated relearning attack | **yes** (disjoint bands, +0.1184 gap, 123 arms) |

**Reporting a validated attack is not an optional robustness appendix. It is the only measurement
in this study that tracks what unlearning is supposed to deliver.** Everything cheaper than an
attack was measured here and found uninformative.

## 7. Scope and honest limits

- **One backbone** (124M), one synthetic corpus, four attributes with closed answer pools.
- **One attack family for this claim.** All 125 arms are attacked by span-SFT relearning. A second
  family (whole-passage diffusion) is being measured separately; it is *not* folded in here.
- **ρ = −0.148 is a descriptive statistic over a convenience sample**, not a designed experiment.
  The 125 arms are every gated run this project happens to have produced, so the sampling of
  (method, λ, α, seed) is uneven and the families are unbalanced (43 vs 82). The matched pairs in
  §3 and the α ladder in §5 are the load-bearing evidence; the correlation is a summary, and a
  deliberately balanced design would be the right way to pin it down.
- Most cells are **1 attack × 1 seed**; recovery is max-over-valid-attacks, which is the right
  direction for an adversarial quantity but makes cells with more attacks look worse. The band
  separation survives this (the suppression band has mostly n=1 cells, so if anything it is
  *understated*).
- `undo_b1500` and a few others have **negative** recovery — the attack left the arm slightly
  *worse* at the forget facts than it started. Treated as ≈ 0; no claim rests on the sign.
- The two groups differ by construction in more than damage (training length, teacher, pool
  composition), so §2's band separation is a **family-level** contrast, not a controlled one. The
  controlled version is the α ladder in §5, where only α varies — and it shows the same thing.

## 8. The honest size of it

This is a **methodological** result, not the method the mandate asks for. It does not improve the
forgetting–utility trade-off; it shows that two of the three ways this field measures that
trade-off cannot see the axis they claim to measure, demonstrated against a true retrain reference
rather than argued. It also cost one planned line (§5) its premise, which is the cheapest way that
line could have ended.
