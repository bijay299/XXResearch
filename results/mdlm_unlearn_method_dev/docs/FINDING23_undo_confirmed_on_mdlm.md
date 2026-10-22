# Finding 23 — UNDO's mechanism holds on a masked diffusion LM, and static metrics cannot see it

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM (`FINDING21`), exact retrain oracle
**Scripts:** `train_small_unlearn.py`, `train_small_undo.py`, `train_small_rtt.py`
**Status:** the first positive mechanism result in this project. Not yet a SOTA claim — see §6.

---

## 1. What the exact oracle made possible

`theta_oracle` was pretrained and fine-tuned with the forget authors entirely absent and sits
**at chance** on them (0.1100 vs 0.1135). So for the first time a method can be scored against
ground truth rather than an approximated `pristine`, and "did the knowledge go away" is separable
from "did the behaviour change".

## 2. Two things every anchor arm does, measured against that oracle

Fifteen arms (MDU / EMDU / AFC / MDU+FAM / gradient-ascent × λ_f ∈ {0.5, 1, 2}):

| arm | rank forget | rank retain | DEGENERATE | TV→oracle |
|---|---|---|---|---|
| `theta_ref` | 0.6975 | 0.6725 | 0.0050 | — |
| **oracle (target)** | **0.1100** | 0.8875 | 0.0000 | **0** |
| mdu λ=0.5 | 0.2050 | 0.6700 | 0.0000 | **0.0725** |
| afc λ=2.0 | 0.1600 | 0.7275 | 0.0000 | 0.0750 |
| FAM arms (emdu, mdu_fam; all λ) | 0.17–0.25 | 0.72–0.78 | 0.0325–0.1200 | **0.3225** |
| gradient ascent | 0.0825–0.0850 | 0.635–0.875 | 0.0000 | 0.9575+ |

**(a) The suppression–removal gap.** Best TV is 0.0725 — behaviourally near-indistinguishable
from the oracle — while every arm still ranks forget facts at 0.16–0.25 against a 0.1135 chance
floor (z ≈ 2.5–4.5 at n=400). **Behavioural metrics are satisfied while the knowledge is still
there.** Every MDLM unlearning paper we know of reports only behavioural measures.

**(b) FAM collapses, the prompt-masked anchor does not.** All five FAM arms land at TV exactly
0.3225 with infill DISCLOSE 0.0000 — fluent off-pool text — against DEG 0.0000 and TV 0.0725–0.09
for the prompt-masked anchors. `FINDING13`/`FINDING18`'s "FAM is degenerate" reproduced on an
independent backbone.

**(c) AFC's retain advantage replicates.** At n=5 seeds: AFC λ=1 rank forget 0.1763 (sd 0.0148),
MDU λ=0.5 0.2219 (sd 0.0166) at matched retain and TV. AFC forgets *more* at equal cost, with
**comparable** variance — not the 3× instability that forced the 8B retraction (`RESULTS.md`).

## 3. UNDO, with its own control, under a gated and lr-swept attack

`theta_perturbed = (1−α)·theta_suppressed + α·β·N`, then distil the suppressed teacher's outputs
back in. Distillation pool covers the forget distribution (teacher outputs as targets, never gold
answers) — the thing our 8B attempt got wrong.

RTT: author-disjoint T/H split, attack trained on T only, recovery measured on held-out H.
**Every attack passed `FINDING20`'s validity gate** (T-rise +0.645 to +0.735 against a 0.20
threshold). Figures are **max over the adversarial lr grid** {3e-5, 1e-4, 3e-4, 1e-3}; lr 3e-5 was
adversarially optimal for every arm.

| teacher | arm | max-over-lr recovery | rank retain |
|---|---|---|---|
| AFC λ=2 | teacher | **0.6573** | 0.7275 |
| | UNDO α=0.3 | **0.3256** | 0.5875 |
| | UNDO α=0.5 | **0.0574** | 0.3550 |
| MDU λ=0.5 | teacher | **0.5608** | 0.6700 |
| | UNDO α=0 *(control)* | **0.5742** | — |
| | UNDO α=0.3 | **0.3415** | — |

> **The damage is the active ingredient, and the α=0 control proves it.** On the MDU teacher,
> distilling into an *undamaged* copy gives 0.5742 — **no better than the teacher's 0.5608**.
> Adding damage cuts recovery to 0.3415 (α=0.3) and 0.0574 (α=0.5, AFC line, an 11.5×
> reduction). Distillation alone confers no robustness; the perturbation does all the work.

This is UNDO's central claim tested on a masked diffusion LM for the first time, with an exact
oracle, a validated attack, and the control that discriminates the mechanism.

## 4. The result is invisible to static metrics

| α (AFC teacher) | rank forget | rank retain | TV→oracle | max-over-lr recovery |
|---|---|---|---|---|
| teacher | 0.1600 | 0.7275 | 0.0750 | 0.6573 |
| 0.0 | 0.1450 | 0.6625 | 0.1000 | — |
| 0.1 | 0.1475 | 0.6225 | 0.0875 | — |
| 0.3 | 0.1500 | 0.5875 | 0.0750 | 0.3256 |
| 0.5 | 0.1550 | 0.3550 | 0.0850 | 0.0574 |

Static forget rank barely moves across α (0.1450 → 0.1550) and TV stays 0.075–0.10, while retain
falls monotonically. **Judged statically the damage looks useless-to-harmful; judged under attack
it is decisive.** A study reporting only static trade-offs would conclude the opposite of the
truth.

The cost is retain, with a usable band and a cliff: α=0.3 buys a 2.0× robustness gain for −0.14
retain; α=0.5 buys 11.5× but collapses retain to 0.3550.

## 5. Bearing on the two walls

`FINDING18` and `FINDING19` predicted that a method escaping both walls must source its forget
target from outside the model's own distributions. Damage + distillation is exactly that route,
and it is the only family here that reduces recovery at all: every anchor arm (which takes its
target from the model's own distribution on a modified input) sits at 0.56–0.66 recovery. **This
is the first positive evidence for that joint prediction.**

## 6. What this is NOT

- **Not a SOTA claim.** The comparison is UNDO against its own teachers on one 124M backbone and
  one synthetic corpus. No published method was re-implemented and tuned here at parity.
- **Not scale-transferable.** 124M with a corpus we built. The 8B attempt failed for feasibility
  reasons (`PREREG_H18` §1), so we cannot say this holds at 8B.
- **Not "fully robust".** A claim of ours from one turn earlier — that α=0.5 showed *negative*
  recovery and was therefore fully robust — is **withdrawn**: that was the single-lr (1e-4) figure
  of −0.048, and under the adversarial lr search it is **+0.057**. Small, but positive. This is
  precisely the limitation `NOVELTY_attack_validity_gate.md` recorded against `FINDING20`, and it
  caught our own claim one turn later. Single-lr robustness numbers are lower bounds, full stop.
- **α=0 and α=0.1 on the AFC line are lower bounds** (lr 1e-4 only); their sweeps are running.

## 8. AMENDMENT: seed replication and the completed lr sweeps

The figures in §3 were **single-seed**, which is the failure mode that produced a retraction on
8B. Replicated at n=3 seeds, RTT at the adversarially-optimal lr 3e-5:

| α (AFC teacher) | recovery, n=3 | retain, n=3 |
|---|---|---|
| 0.3 | **0.2719 ± 0.050** | 0.6017 ± 0.034 |
| 0.5 | **0.0513 ± 0.020** | 0.3650 ± 0.011 |

Against the teacher's 0.6573 that is a 2.4x and 12.8x reduction, and no seed approaches the
teacher. **Replication confirms the result** — the opposite outcome to the 8B AFC stability claim,
which replication refuted.

Completing the lr sweeps also corrected our own lower bounds UPWARD: α=0 went 0.4424 -> **0.5622**
and α=0.1 went 0.3594 -> **0.4424**. Single-lr figures understated recovery by 0.08-0.12
throughout, consistent with the α=0.5 withdrawal in §6. The full frontier:

| α | rank retain | recovery (max over lr) |
|---|---|---|
| teacher | 0.7275 | 0.6573 |
| 0.0 *(control)* | 0.6625 | 0.5622 |
| 0.1 | 0.6225 | 0.4424 |
| 0.2 | 0.6375 | 0.4265 |
| 0.3 | 0.5875 | 0.2719 ± 0.050 |
| 0.4 | 0.5300 | 0.1878 |
| 0.5 | 0.3550 | 0.0513 ± 0.020 |

Robustness gained per unit retain lost falls 2.75 -> 2.37 -> 1.63 across α = 0.3/0.4/0.5, so
α=0.3 is the knee. `FINDING24` acts on that inefficiency.

## 7. A limit of the validity gate itself, recorded

At lr 1e-3 every arm shows **negative** recovery while still passing the T-rise gate: the attack
relearns its own training set yet damages held-out performance. So the gate catches *broken*
attacks, not *over-strong* ones. Reporting max-over-lr handles this in practice, but the gate
alone is not sufficient and we should not present it as if it were.
