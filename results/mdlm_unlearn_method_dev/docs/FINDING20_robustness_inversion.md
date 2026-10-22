# Finding 20 — the static Pareto winner is the robustness loser, and our selected method is the casualty

**Date:** 2026-10-21 · **Script:** `scripts/train_rtt.py` · **Attempts:** `rtt_*_v3`
**Outputs:** `outputs/rtt_afc_l1_v3.json`, `rtt_mdu_l3_v3.json`, `rtt_mdu_l8_v3.json`
**Status:** disqualifies AFC as a SOTA claim; establishes the validity gate as a protocol
requirement

---

## 1. The measurement

RTT relearning attack (Deeb & Roger): fine-tune the unlearned model with ordinary SFT on an
author-disjoint set `T` (n=69), then score the six-cell profile on held-out facts `H` (n=40) that
the attacker **never supplied**. Recovery on `H` separates removal from thin suppression.
lr 3e-5, rank 16, 300 steps, probes at {0, 25, 100, 300}.

| arm | H pre | H post | absolute rise | frac of θ₀ reopened | T-rise (validity) | valid |
|---|---|---|---|---|---|---|
| **AFC λ=1** | 0.225 | 0.425 | **0.200** | **0.353** | 0.375 | ✓ |
| **MDU λ=3** | 0.225 | 0.300 | **0.075** | **0.132** | 0.275 | ✓ |
| MDU λ=8 | 0.025 | 0.375 | 0.350 | 0.456 | 0.475 | ✓ |

θ₀ reference DISCLOSE = 0.792.

## 2. Why this is decisive and not a confound

AFC and MDU λ=3 are at the **identical** pre-attack held-out disclosure, 0.225. There is no
starting-point advantage to correct for, no matching argument to make — the two arms begin at the
same place on the axis the attack moves, and under the same attack AFC reopens **2.7×** as much.

MDU λ=8 is the control that confirms the attack bites: it starts lowest (0.025) and reopens the
most in absolute terms (0.350), i.e. the deepest-looking suppression was the most superficial.

> **The inversion.** Static trade-off quality and robustness to relearning are not merely
> imperfectly correlated here — on these arms they are **anti**-correlated. AFC beats MDU λ=3 on
> TV→pristine (0.299 vs 0.352) and retain (0.534 vs 0.445), and loses to it on recovery by 2.7×.

### Consequence for this project

AFC was the selected point. It is now disqualified as a SOTA claim: it buys its better static
frontier with shallower suppression. The honest summary of AFC is **better average static
trade-off, a catastrophic-failure mode the baseline lacks (seed 3 at DEG 0.508), and worse
robustness at matched disclosure.** The ablation conclusion it was built to deliver — that
EMDU's KL *direction* carries the benefit and FAM is what hurts — is unaffected and survives.

### Consequence for the earlier entropy hypothesis

Already refuted, now explained by the same data. We predicted that AFC's higher output entropy
(2.389 vs MDU's 1.630 nats) would make it *harder* to relearn. It is **easier** to relearn. High
entropy at the output is not depth of removal; it is compatible with the fact being intact
underneath and merely flattened.

## 3. The protocol requirement this run established

**Three earlier RTT runs were invalid and looked like success.** At lr 1e-4 and lr 1e-5 recovery
was ~0%, which reads as robustness. It was the attack destroying the model — DEGENERATE peaked
0.93–0.98. Root cause: **we never scored the attack's own training set.**

The fix is five lines and a gate:

```python
R["attack_validity"] = {..., "valid": bool(tT_d - t0_d >= 0.20),
    "meaning": ("the attack must relearn the facts it is TRAINED ON...")}
```

> A relearning attack must demonstrate that it relearns `T` before a flat curve on `H` counts as
> evidence of anything. Without the gate, a broken attack and a robust method are
> indistinguishable — and three of our own runs sat on the wrong side of that line.

The novelty check is now done (`NOVELTY_attack_validity_gate.md`): **partially anticipated.** The
principle is Athalye et al.'s ([1802.00420](https://arxiv.org/abs/1802.00420) §3) — a broken
attack impugns the attack, not the defence — and ours is a domain-specific instance of it. None
of seven unlearning papers checked states or reports the control. It is a methodological note
with a released instrument, not a contribution that carries a paper.

## 3a. A limitation of this measurement, found by that same check

UNDO and Lynch et al. both run an **adversarial learning-rate search** and report the
best-performing attack ("always mark the adversarially optimal performance", Lee et al.;
a ten-value lr grid, Lynch et al.). **Our numbers above come from a single learning rate.**

The paired comparison stands — AFC and MDU λ=3 were attacked at the identical setting from the
identical pre-attack disclosure of 0.225 — but the absolute recoveries are **lower bounds**, and
the ordering has not been shown stable under an adversarial lr search. Until that sweep runs,
read this finding as **"AFC is less robust than MDU λ=3 at lr 3e-5"**, not "under the strongest
available attack". The sweep is queued, and if it reverses the ordering this finding is wrong and
will be corrected.

## 4. Status

Nineteen candidates closed. AFC remains the best *static* arm and is no longer a candidate for
the headline. The remaining open route is UNDO (`FINDING19` §4.3's prediction), whose arms will be
run through **this same validated protocol** — which is now the bar any method in this project has
to clear, and which AFC does not.
