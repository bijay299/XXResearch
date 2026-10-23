# Finding 44 — ❌ my pre-registered mechanism for FRP's reversal is WRONG. The pruned weights were never re-learned; the concept came back through the weights FRP left alone.

**Date:** 2026-10-23 · **Diagnostic declared in:** `PREREG_H66` §5a, before any FRP attack ran
**Evidence:** `outputs/frp_refill_church_r3.json`, from a `--keep-checkpoint` re-run of the
highest-recovery FRP attack
**Status:** the **prediction** in `PREREG_H66` §5a was correct; the **mechanism** it gave was not.

---

## 1. What I predicted, and what the measurement says

`PREREG_H66` §5a, written before a single FRP attack existed:

> *"FRP zeroes weights and nothing holds them at zero... a relearning adversary fine-tunes every
> parameter, so the pruned entries are free to become nonzero again on the first optimiser step...
> the mask is an initial condition, not a constraint."*

The outcome prediction — full reversal, H → 1.000 — was right (`FINDING43`). **The mechanism is
refuted.**

| statistic | value | reading |
|---|---|---|
| pruned entries nonzero after attack | 4 414 887 / 4 415 427 = **99.99 %** | **near-trivial** — see §2 |
| share of the attack's total update **norm** on pruned positions | **3.020 %** | — |
| pruned share of **entries** | **2.970 %** | — |
| **concentration ratio** | **1.0169** | update is **UNIFORM**, not concentrated |
| per-entry RMS update: pruned vs surviving | 1.987e-04 vs 1.970e-04 | **identical** |
| post-attack weight **mass** in formerly-pruned positions | **0.00005 %** | **negligible** |
| ‖Δw‖ pruned vs surviving | 0.418 vs **2.366** | the work happened elsewhere |

**The adversary did not refill the mask.** It spread its update essentially uniformly and recovered
the concept through the **surviving 97 % of weights** — the ones FRP's own forget-critical,
retain-sparing scoring declined to prune.

## 2. ⚠ And my own diagnostic's first verdict was misleading

The script I wrote 30 minutes earlier thresholded on *fraction of pruned entries that became
nonzero* and reported, at 99.99 %:

> *"the pruning mask behaved as an INITIAL CONDITION: the adversary re-learned the zeroed weights"*

**That verdict was wrong, and it happened to agree with my prediction**, which is the dangerous
combination. Under Adam essentially *every* parameter receives a nonzero update, so ~100 % is the
expected value whatever the mechanism — the statistic cannot discriminate. The refilled weights are
**240× smaller** than their surviving neighbours (1.68e-04 vs 4.02e-02).

The script now computes the statistic that does discriminate — update norm on pruned positions
against their share of entries — and carries an explicit caveat that `frac_refilled` is reported but
is **not** the verdict. Had I accepted the first output, I would have published a confirmation of my
own hypothesis built on a statistic with no power to test it. The only reason I did not is that the
magnitude ratio looked implausible and I checked.

## 3. What the corrected mechanism implies, and it is stronger than mine

FRP prunes the 3 % of feed-forward weights its scoring ranks most forget-critical and least
retain-critical. The adversary restores the concept **without using those positions at all.**

So the concept is not merely *non-separable* from retain capability in weight space
(`PREREG_H51` §7, `FINDING42`) — it is **re-encodable in weights that the forget-critical scoring
did not select.** Removing the route a saliency method identifies does not remove the capacity to
rebuild the function elsewhere. That is a sharper statement than "forget and retain weights
overlap", and it is the first mechanistic evidence in this phase rather than an inference from
utility collapse.

**It also bears on `ERRATUM_r2` C17**, the project's deepest open limitation — whether recovery is
*reactivation of residual memory* or *relearning from the attacker's data*. The concept returning
through weights the defence never touched, with **no preferential update to the removed route**,
looks more like relearning than like reactivation of what was removed. **This is evidence, not a
resolution**: a redundant encoding already present in the surviving weights would also produce this
signature, and without a never-trained oracle the two remain distinguishable only in principle. C17
stands, now with its first piece of mechanistic evidence attached.

## 4. Scope, and what would strengthen this

One arm (`frp_church_r3_l05`), one attack (seed 0, lr 1e-5, the highest-recovery configuration), one
concept. The diagnostic is a weight-space accounting identity, not a causal test: it shows the
update was not concentrated on pruned positions, not that those positions are *irrelevant*.

The decisive follow-up, not run: **re-zero the pruned positions in the attacked checkpoint and
re-measure the target.** If the concept survives that, the surviving weights alone carry it and the
conclusion is direct. It costs one weight mask and one evaluation — roughly 5 GPU-minutes — and is
the obvious next measurement. It is recorded here as **planned, not done**, rather than asserted.

## 5. Corrections this forces

- **`FINDING43` §2** says *"the mask is an initial condition, not a constraint"* and calls the
  refill inference pending. That sentence is now **withdrawn as a mechanism** — the outcome it
  explained is real, the explanation is not. `FINDING43` §2 should be read against this finding.
- **`PREREG_H66` §5a**'s mechanism paragraph is **falsified** and stays in place as a falsified
  pre-registration, which is what pre-registrations are for.
