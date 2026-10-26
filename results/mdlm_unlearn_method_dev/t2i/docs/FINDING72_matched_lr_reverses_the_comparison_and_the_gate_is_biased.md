# FINDING 72 — At matched, non-destructive lr, UTRAP is recovered **as fast or faster** than FRP, reversing `FINDING68` §6's coarse-grid impression. And the validity gate's exclusion rate is **non-monotone in T_before** (n = 520), so "max-over-valid" is computed over a sample whose filtering correlates with how well-erased the arm is.

**Date:** 2026-10-26 · **Experiments:** E11i (matched-lr completion) + a zero-GPU corpus analysis
**Evidence:** `outputs/atk_E11I_*.json`, `outputs/gate_bias_by_T_before.json`,
`logs/e11i_frp_matched.log`

---

## 1. At matched lr and budget, the two arms' order **flips**

`FINDING68` §6 reported FRP reaching the ceiling by 50 steps and UTRAP by 100, and read that as FRP
being easier to reverse — while flagging that the arms' different `H_before` made it not a clean
comparison. Completing both to **n = 3** at lr 1e-5 (the non-destructive rate, per `FINDING71`):

| steps | UTRAP mean H | UTRAP max | **FRP mean H** | **FRP max** | FRP gate-valid |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.3111 | 0.5333 | — | 0.5333 | **1/3** |
| 10 | 0.4444 | 0.6667 | 0.5000 | 0.5333 | **2/3** |
| **15** | **0.7111** | **0.9333** | 0.6000 | 0.6667 | 3/3 |
| **25** | **0.8889** | 0.9333 | 0.8667 | 0.9333 | 3/3 |

**At 15 steps UTRAP's mean H is 0.7111 against FRP's 0.6000, and its max is 0.9333 against 0.6667.**
At 25 steps they are level. So in the resolved, n = 3, matched-lr region **UTRAP is recovered as
fast or faster than FRP** — the opposite of `FINDING68` §6's impression.

**Why that impression was wrong:** the 50- and 100-step ceiling points it rested on were **n = 1
each**. With sd ≈ 0.19–0.28 in this region (`FINDING69` §7), a single-seed ordering carries no
weight. `FINDING68` §6 is **corrected**: it should never have ordered the two methods, and it did
say so in its own text — the error was letting the ordering appear in the comparison table at all.

**No method ranking is asserted here either.** UTRAP starting at `H_before` 0.0000 vs FRP's 0.1333
still confounds it in the other direction, and §2 adds a second confound.

**One reproducibility check passed incidentally:** FRP seed 0 at 25 steps was run independently in
E11b and again in E11i and returned **H = 0.9333 both times** — identical. The attacker is
deterministic given its seed, as intended.

## 2. The validity gate excludes attacks non-randomly, and both tails suffer

Zero-GPU, over **every** full-UNet attack record with a recorded gate outcome (n = 520):

| `T_before` | n | valid | **validity rate** | mean T_rise |
|---|---:|---:|---:|---:|
| ≈ 0.000 | 122 | 86 | **0.705** | 0.5847 |
| 0.001–0.100 | 79 | 68 | 0.861 | 0.6709 |
| 0.100–0.250 | 119 | 113 | **0.950** | 0.6975 |
| 0.250–0.500 | 133 | 117 | 0.880 | 0.5191 |
| **> 0.500** | 67 | 35 | **0.522** | **0.1866** |
| all | 520 | 419 | 0.806 | — |

**The rate is non-monotone: 0.705 at the bottom, 0.950 in the middle, 0.522 at the top.** The two
tails fail for **opposite** reasons:

- **High `T_before` — a ceiling effect, and it is mechanical.** The gate requires an *absolute* rise
  of ≥ 0.20, so an arm already at T = 0.6 must reach 0.8. Mean T_rise in that bin is **0.1866** —
  sitting just under the threshold. This was the asymmetry I went looking for and it is confirmed.
- **`T_before` ≈ 0 — the opposite, and I did not predict it.** Despite the *largest* headroom, these
  arms show only 70.5% validity. The plain reading is that thoroughly-erased arms are genuinely
  harder for a weak attacker to move at all, so the attack fails to fit its own training prompts.

**The consequence for every result in this project.** "Inconclusive" is **not** randomly
distributed: it concentrates on the most-erased arms — precisely where a successful defence would
sit — and on the least-erased. So **max-over-valid is a maximum over a filtered subset whose
filtering correlates with the quantity being measured.** For a heavily-erased arm roughly **30% of
attempted attacks are discarded**, and the discarded ones are the weak attacks. Max-over-valid is
therefore conservative in the attacker's favour by construction — which is the right direction for a
robustness claim, but it should be stated rather than left implicit.

This does **not** undermine the gate. `ERRATUM_interpretations_r5.md` §6 is reaffirmed: an
ineffective attack is inconclusive, and counting it as robustness would be worse. What it adds is
that the *rate* of inconclusiveness is itself informative, and that it has never been reported.

## 3. Limits

- §1 is **two arms, one concept, one lr, three seeds**, and the `H_before` mismatch is unresolved —
  audit §8's "matched initial utility" still requires builds matched on pre-attack suppression,
  which needs disk.
- §2 is **observational and confounded.** `T_before` correlates with arm, defence family, budget and
  learning rate, all pooled within each bin. The high-`T_before` mechanism is mechanically forced by
  an absolute threshold; the low-`T_before` depression is **a plain reading, not an isolated cause**
  — it could equally reflect which arms and budgets happen to populate that bin.
- The bins are unequal in n (67 to 133) and the record set spans the whole project, including the
  136 VOID-utility runs (their T/H halves remain admissible, so they are included here).
- A **relative** gate — fraction of available T rise rather than an absolute 0.20 — would remove the
  ceiling asymmetry. That is a protocol change, it would reclassify historical records, and it is
  **not** being made here: `PROTOCOL_FROZEN_v1`'s gate stays as measured, with this bias documented
  instead.
