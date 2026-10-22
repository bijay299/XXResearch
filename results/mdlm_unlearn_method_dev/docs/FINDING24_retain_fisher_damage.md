# Finding 24 — retain-Fisher-weighted damage buys retain at equal damage and equal robustness

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM, exact retrain oracle
**Script:** `scripts/train_small_undo.py --protect retain_fisher`
**Novelty:** **PARTIALLY ANTICIPATED** — see §5 before reading §2 as a contribution.

---

## 1. The gap this addresses

`FINDING23` measured UNDO's robustness–retain frontier on this backbone and found it accelerating
against us:

| α | rank retain | recovery | robustness gained per unit retain lost |
|---|---|---|---|
| teacher (AFC λ=2) | 0.7275 | 0.6573 | — |
| 0.3 | 0.6017 | 0.2719 | **2.75** |
| 0.4 | 0.5300 | 0.1878 | 2.37 |
| 0.5 | 0.3650 | 0.0513 | 1.63 |

UNDO perturbs **every weight uniformly**, yet `FINDING19` established that forget knowledge is
localisable. So the uniformity is a plausible inefficiency: weights that matter for retain are
damaged as hard as everything else.

**The modification.** Compute a per-parameter retain-importance diagonal,
`imp = E[(∂L_retain/∂θ)²]` over retain value spans, normalise per tensor by its own 99th
percentile, and scale each parameter's perturbation by `(1 − imp)`. Then **renormalise the weights
so mean damage equals plain UNDO at the same α** — the comparison must be equal-damage, not
less-damage. Measured: 0.2989 vs 0.2998 at α=0.3, 0.4945 vs 0.4991 at α=0.5.

## 2. Result, n=3 seeds, RTT at the adversarially-optimal lr 3e-5

| arm | rank retain | recovery |
|---|---|---|
| teacher | 0.7275 | 0.6573 |
| plain UNDO α=0.3 | 0.6017 ± 0.034 | 0.2719 ± 0.050 |
| **retain-Fisher α=0.3** | **0.6550 ± 0.003** | 0.2842 ± 0.044 |
| plain UNDO α=0.5 | 0.3650 ± 0.011 | 0.0513 ± 0.020 |
| **retain-Fisher α=0.5** | **0.4567 ± 0.029** | 0.0498 ± 0.077 |

**Retain +0.053 (α=0.3) and +0.092 (α=0.5) at matched mean damage, with recovery statistically
indistinguishable.** Every attack passed `FINDING20`'s validity gate.

Only the retain gain is claimed. The apparent robustness differences sit well inside the seed
spread and are not claimed in either direction.

## 3. The discriminating control: it is the Fisher signal, not non-uniformity

The obvious alternative explanation is that *any* non-uniform damage helps, and Fisher is
incidental. So the control draws the **same per-parameter weight distribution and shuffles it
within each tensor** — equally non-uniform, equal in mean (damage 0.4931 vs 0.4945, mean
importance 0.1394 identical by construction), carrying **no retain information**:

| arm (α=0.5) | rank retain | recovery |
|---|---|---|
| plain UNDO (uniform) | 0.3650 ± 0.011 | 0.0513 ± 0.020 |
| **retain-Fisher protected** | **0.4567 ± 0.029** | 0.0498 ± 0.077 |
| **shuffled-Fisher control** | **0.3667 ± 0.031** | −0.0037 ± 0.047 |

> The shuffled control recovers plain UNDO's retain **exactly** (0.3667 vs 0.3650) and gets none
> of the +0.092. The gain is attributable to the retain-Fisher signal, not to the damage being
> non-uniform.

This is the same logic as UNDO's own α=0 control, and it is the measurement that makes the result
worth reporting at all.

## 4. The cost, stated

Protected arms retain **more forget knowledge statically**: rank forget 0.1850 (α=0.3) and 0.1950
(α=0.5) against plain UNDO's 0.1500/0.1550, on a 0.1135 chance floor. Protecting retain-critical
weights evidently protects some shared forget structure too. So this is not a free win — it moves
along a different trade-off: **better retain, more residual static knowledge, equal robustness.**

## 5. Novelty: partially anticipated, and the conceptual content is not ours

A primary-source check returned **PARTIALLY ANTICIPATED**, with the blunt assessment that very
little of the idea is conceptually new. Every ingredient is published:

| prior art | what it already does | distance |
|---|---|---|
| **DP-KFC** ([2605.13418](https://arxiv.org/abs/2605.13418)) | *"DP-SGD injects isotropic noise"* against an anisotropic Fisher landscape, so shape the noise by Fisher | the exact mechanism argument, for DP training |
| **WAGLE** ([2410.17509](https://arxiv.org/abs/2410.17509)) | attribution score **subtracts** a retain-gradient term to avoid retain-critical weights | same protective intent, as a binary gradient mask |
| **SSD** ([2308.07707](https://arxiv.org/abs/2308.07707)) | dampens weights proportional to forget-importance via Fisher | importance → modification magnitude, inverted target |
| **EWC** ([1612.00796](https://arxiv.org/abs/1612.00796)) | *"selectively slowing down learning on the weights important for those tasks"* | the importance signal, as a regulariser not a noise scale |
| **UNDO** ([2506.06278](https://arxiv.org/abs/2506.06278)) | confirmed uniform: *"We sample N using Xavier initialization"*, no selective-noise discussion | the pipeline we modify |

Nobody found combines retain-Fisher-weighted scaling of an *injected perturbation* inside
perturb-then-distill. But that is a predictable synthesis of EWC's signal with UNDO's recipe, and
DP-KFC shows the field already converged on "shape the noise by Fisher" independently.

**What is defensible:** the empirical result in §2–§3 — this specific application, with the
equal-mean-damage design and the shuffled control — not a new mechanism. A writeup must cite
EWC, SSD, WAGLE and DP-KFC as required related work.

**Caveat on the check itself:** WebSearch was unavailable, so coverage is arXiv abstracts/API
only. OpenReview and informal venues were not searched, and absence of a hit is not a clean bill.

## 6. What this is NOT

- **Not a SOTA claim.** It improves *UNDO against itself* on one 124M backbone and one synthetic
  corpus. No published method was re-implemented and tuned at parity.
- **Not scale-transferable.** 124M, our own corpus. The 8B attempt failed on feasibility.
- **Not a robustness improvement.** Recovery is indistinguishable; only retain moves.
- **Single teacher so far.** The α=0.3 shuffled control and an MDU-teacher replication are
  running; until they land, both the control and the effect rest on one teacher at one α.
