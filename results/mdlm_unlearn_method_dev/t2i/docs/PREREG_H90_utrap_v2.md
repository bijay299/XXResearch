# PREREG H90 — does unroll depth matter when the inner attacker **is** the evaluation attacker?

**Written:** 2026-10-25, during the `0b435427` audit remediation.
**Committed before any v2 arm is built or attacked.** No `erase_utrap_v2.py` checkpoint exists at
the time of writing; the only runs of that script are the labelled `SMOKE_*` integration tests in
`logs/smoke_v2.log`, whose checkpoints were deleted and from which no result is claimed.
**Precedence:** publicly verifiable — this commit contains no results (`PREREG_PRECEDENCE.md` §4).

---

## 0. Why this needs running at all

`FINDING63` §3a(a) reported the λ×K sweep as falsifying `PREREG_H84`'s hypothesis that one-step
inner approximation was the problem:

> *"doubling K from 8 to 16 left max-over-valid H at 1.000 … Depth was the thing I believed was
> missing, and it is not."*

That inference is **withdrawn** (`ERRATUM_utrap_inner_attacker.md`). What the sweep varied was the
horizon of a surrogate that differed from the evaluation attacker on **six other axes at once**:
SGD instead of Adam, rebuilt every outer step instead of persistent, batch 2 instead of 1, no
gradient clipping instead of norm 1.0, target images from the already-erased teacher instead of the
base model, and one repeated **evaluation** prompt instead of eight disjoint `T` prompts.

So the hypothesis `PREREG_H84` set out to test has **not been tested**. This plan tests it.

A quantitative reason to expect the surrogate was weak, from the smoke run's new probe: across 8
inner steps the fixed-noise target probe fell only **0.1404 → 0.1363**, about **3%** of the probe
loss. A surrogate that removes 3% of the loss the real attacker drives to near zero over 300 steps
is not obviously approximating it at all. That number is an integration-test observation on 6 outer
steps, not a result, and it is the thing H90 measures properly.

## 1. Hypothesis

**H90.** With the inner attacker reconciled to the evaluation attacker on every axis except
horizon, max-over-valid H is **non-increasing in K**, and at least one K reaches
**max-over-valid H < 1.000** on an arm that clears the utility floor.

**Falsifier, stated before any run:** if every arm at every K reaches max-over-valid H = 1.000 while
clearing K1, H90 is **falsified**, and the conclusion is that unroll depth does not buy resistance
**even against a faithful simulation of the evaluation attacker** — which is the claim `FINDING63`
made without the evidence for it.

**Second falsifier, for the mechanism rather than the outcome:** if the inner-attacker probe delta
does not grow with K (i.e. K=32 adapts no more than K=1), the unroll is not doing what the design
assumes and **no conclusion about depth may be drawn from the run at all**. This is checked
**first**, and reported even if it kills the study.

## 2. Arms — fixed now

Four arms, all `erase_utrap_v2.py`, λ = 0.15 (the SD-1.4 setting that cleared the retain floor),
concept **church**, teacher `adapters/frp_church_r3_l05/unet.pt`, inner attacker at its **defaults**
(Adam, lr 1e-5, batch 1, clip 1.0, persistent state, `--target-source base`,
`--target-prompts def_train`):

| arm | K | `horizon_role` | outer steps | Σ inner+outer forward-backwards |
|---|---:|---|---:|---:|
| `utrap2_church_k0` | **0** | `ABLATION_K0` | 400 | 400 |
| `utrap2_church_k1` | **1** | `SHORT_HORIZON_SURROGATE` | 200 | 400 |
| `utrap2_church_k8` | **8** | `SHORT_HORIZON_SURROGATE` | 45 | 405 |
| `utrap2_church_k32` | **32** | `SHORT_HORIZON_SURROGATE` | 13 | 429 |

### 2a. Matched compute — defined now, not after the fact

Outer steps are chosen so **total UNet forward-backward passes** are held near-constant
(400–429, within 7.3%), because each outer step costs 1 retain pass + K inner passes + 1 endpoint
pass. The audit's requirement was a matched-compute control; this is its definition, fixed before
any arm exists so it cannot be rebalanced after seeing outcomes.

**K=0 is the control that was missing.** With K=0 the objective degenerates to
`L_retain(θ) − λ·L_target(θ)` — erasure with no adversarial anticipation at all. If K=0 performs as
well as K=8, the unrolling contributes nothing and every K-sweep conclusion in `FINDING63` and this
plan is about something else.

**Confound retained and declared:** matching total compute means the four arms take **different
numbers of outer optimiser steps** (400 vs 13). An effect could be depth or outer-step count. A
second, compute-unmatched series at fixed 400 outer steps would separate them, costs ~4× more, and
is **not** in this plan. So this plan can show "depth at matched compute does/does not help"; it
cannot attribute a difference to depth alone. Stated now rather than discovered later.

## 3. Evaluation — the frozen grid, no exceptions

Each arm that clears **K1 (retain8 ≥ 0.60)** gets **three** full-UNet attacks at the primary
lr 1e-5, seeds {0, 1, 6}, 300 steps — `attack_relearn.py` unchanged, so the adversary is identical
to every historical record.

- **Stopping rule:** three attacks per arm, run to completion. **No early stop on reaching the
  ceiling**, and no escalation beyond three in this study. Any further attack is reported
  separately as adaptive escalation (`ERRATUM_interpretations_r5.md` §9).
- **Arms failing K1 are not attacked** and are reported as utility failures, not robustness results.
- **Vacuity check before attacking:** any arm whose as-is target suppression is ≥ 0.95 is **not
  attacked**, because H would measure the base model rather than a defence
  (`ERRATUM_uce_lambda_scale.md` §6, and the six `gift_church_b0.1` records this rule retroactively
  disqualifies).
- **Evaluation seeds come from `split_registry.seed_for("dev_eval", ·)`**, common across all four
  arms, while defence-build and data-generation streams stay independent.
- Reporting is on **DEV** prompts and labelled DEV. These are the historical H prompts and they are
  a development set (`ERRATUM_split_leak.md`). **No FINAL evaluation in this study**; FINAL stays
  frozen and unexamined.

## 4. Primary and secondary outcomes, declared

**Primary:** max-over-valid H per arm, at the fixed three-point grid.

**Secondary, recorded regardless of the primary:**
1. **inner-attacker probe delta vs K** — the mechanism check of §1's second falsifier;
2. retain8 and all-non-target utility as-is, **both denominators reported**;
3. paired post-attack utility with explicit denominators;
4. gate outcomes for all twelve attacks, **gated and ungated H both reported**;
5. inner update norms and grad norms from `attack_core`'s telemetry.

## 5. What this study cannot settle

- **One concept, one teacher, one defence build per K.** Defence-build replication is not attack-seed
  replication, and attack seeds do not substitute for it.
- **A saturated ceiling still cannot rank depths.** If all four arms land at H = 1.000, the result is
  "depth did not prevent reaching the ceiling at this budget" — not "depth has no value"
  (`ERRATUM_interpretations_r5.md` §1). Recovery-versus-budget curves would be needed and are not
  in this plan.
- **The trajectory remains detached.** A fully differentiated unroll is still untested and is not
  claimed either way.
- **The compute/outer-step confound** of §2a.

## 6. Cost and authorisation

4 builds × ~1.0 h + 12 attacks × ~0.9 h ≈ **14 GPU-h**, two GPUs at a time, within the standing
authorisation. Storage: 4 checkpoints × 1.7 GB ≈ **6.8 GB**.

**Blocker at the time of writing: `/data` is at 100% capacity with ~14 GB free.** 6.8 GB of
checkpoints plus attack intermediates is too tight to start safely on a filesystem shared with
another project. This plan is therefore **ready but not launched**, and the blocker is reported in
`HANDOFF_r5.md` rather than worked around by deleting anything.

---

## 7. Amendment, 2026-10-26, **before any H90 arm has been built** — add a λ≈1.2 arm

`scripts/diag_coeff_scale.py` measured, on the real model: ‖∇L_retain‖ = 0.1856 and
‖∇L_target‖ = 0.1503 over the full scope, so **λ for parity is 1.23** while this plan specifies
**λ = 0.15** throughout. λ=0.15 was inherited from the historical UTRAP sweep, whose inner attacker
was a different and much weaker process (`ERRATUM_utrap_inner_attacker.md`), so it carries no
authority for the corrected attacker.

**Amendment:** the K ∈ {0, 1, 8, 32} series stays at λ = 0.15 for continuity with the historical
arms, and **one additional arm at λ = 1.2, K = 8** is added, so the effect of λ is measured rather
than assumed. Cost: +1 build +3 attacks ≈ 3.5 GPU-h on top of the 14 already priced.

> **CORRECTION 2026-10-26 (audit defect D-017).** This amendment originally read "so the λ
> **under-weighting** is measured". That wording presumed λ=0.15 is too low, by treating
> gradient-norm parity (λ≈1.2) as the correct target. It is not a target: UTRAP-v2 sums both
> gradients into one `p.grad` before a single Adam step (`scripts/erase_utrap_v2.py:265-274`), so
> the ratio sets the summed *direction* and parity is at most a directional heuristic — Adam
> normalises the overall scale away, and nothing measured here shows ≈45° is optimal. The λ=1.2 arm
> is therefore a **two-point comparison**, not a correction toward a known-better value, and the
> prediction attached to it must not assume λ=1.2 wins. See `FINDING67` §7.

This amendment is made **before any arm exists** and is recorded here rather than in the plan body,
per `PREREG_PRECEDENCE.md` §4. It does not alter any prediction, falsifier or stopping rule above.

---

## 8. Amendment, 2026-10-26 — **§2a's matched-compute arithmetic is wrong**, and the built arms are not matched

**Raised in review.** §2a stated each outer step costs "1 retain pass + K inner passes + 1 endpoint
pass" — which is **K+2** — and then computed totals as if it were **K+1**. The error is in the
arithmetic, not the description.

`erase_utrap_v2.py` per outer step actually runs: **1 retain backward (batch 2) + K inner attack
passes (batch 1) + 1 endpoint backward (batch 2)**, plus a fixed-noise probe every `--probe-every`
steps.

| arm | K | outer | §2a claimed (K+1) | **actual passes (K+2)** | **actual samples (K+4)** |
|---|---:|---:|---:|---:|---:|
| K=0 | 0 | 400 | 400 | **800** | **1600** |
| K=1 | 1 | 200 | 400 | **600** | **1000** |
| K=8 | 8 | 45 | 405 | **450** | **540** |
| K=32 | 32 | 13 | 429 | **442** | **468** |

**So the five arms built on 2026-10-26 are not compute-matched.** Worse, compute falls monotonically
with K — 800 → 442 passes, 1600 → 468 samples — so the higher-K arms received **systematically
less** training than K=0. That biases the comparison **against** unroll depth, which is the exact
hypothesis this plan tests. Any K-ordering read off those arms would have been an artifact of the
budget, not of depth.

**Corrected outer steps.** Matching on **samples** is the stricter criterion and is adopted:

| arm | K | outer for matched **passes** (800) | **outer for matched samples (1600) — ADOPTED** |
|---|---:|---:|---:|
| K=0 | 0 | 400 | **400** |
| K=1 | 1 | 266 | **320** |
| K=8 | 8 | 80 | **133** |
| K=32 | 32 | 23 | **44** |

**What happens to the arms already built.** They are **kept, not deleted**, and relabelled
`utrap2_church_l0.15k{K}` → the **compute-UNMATCHED series**. They remain valid individual defence
builds and their as-is measurements stand:

| arm | as-is target | retain8 |
|---|---:|---:|
| `utrap2_church_l0.15k0` | 0.0000 | 0.8667 |
| `utrap2_church_l0.15k32` | 0.1333 | 0.9583 |
| `utrap2_church_l1.2k8` | 0.0667 | 0.9583 |

A **second series** at the corrected outer steps is built under the label
`utrap2m_church_l0.15k{K}` (`m` = matched). K=0 at 400 outer is already correct, so that arm is
**shared** between the two series rather than rebuilt.

**Probe overhead, declared:** the fixed-noise probe is 2 forward-only evaluations of batch 8 every
`--probe-every` outer steps. At the corrected settings this is a different fraction of each arm's
total (K=0: 8 probes over 400 outer; K=32: 1 probe over 44). It is forward-only and excluded from
the pass/sample counts above; it is **not** matched across arms, and that residual is recorded here
rather than claimed away.

**Nothing is published from the unmatched series as a K-comparison.** §3's grid runs against the
matched series.
