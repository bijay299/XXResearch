# Finding 27 — for two anchor families, a strong attacker restores the forgotten facts to the un-unlearned model's own level

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM with an exact retrain oracle
**Protocol:** RTT, author-disjoint T/H, attacker budget `t_frac 0.8`, `FINDING20` validity gate,
recovery reported as **max over all attacks**
**Status:** the sharpest negative in the project. Verified against raw ranks, with the fragile part
of the arithmetic identified and discarded.

---

## 1. The measurement

All families attacked at a **matched** attacker budget (`t_frac 0.8`), every attack passing the
validity gate:

| arm | n attacks | max recovery fraction | max post-attack rank `H_after` | reaches `theta_ref`'s 0.6975? |
|---|---|---|---|---|
| MDU+FAM λ=1 | 4 | **+1.1039** | **0.7375** | **yes** |
| EMDU λ=1 | 6 | **+1.0065** | **0.7000** | **yes** |
| GA λ=0.5 | 4 | +0.8661 | 0.6125 | no |
| AFC λ=1 | 6 | +0.7759 | 0.6000 | no |
| MDU λ=0.5 | 2 | +0.7396 | 0.5875 | no |
| **UNDO α=0.6, AFC teacher (selected point)** | 18 | **+0.2299** | — | no |

For EMDU the raw trace is `H` 0.3125 → **0.7000**; for MDU+FAM, 0.3125 → **0.7375**. Both exceed
`theta_ref`'s forget rank of 0.6975.

> **For these two families, unlearning achieved nothing durable.** After the attack the model
> ranks the forgotten facts at least as well as the model that was never unlearned at all. The
> chance floor is 0.1135, so this is not a marginal effect on the forgetting axis — it is the
> complete undoing of the intervention.

## 2. What was checked, and what is being discarded

The recovery fractions above 1.0 were initially suspect as a normalisation artifact, so the
arithmetic was recomputed from raw pre/post ranks. It reproduces exactly:
`frac = (H_after − H_before) / (0.6975 − H_before)`.

**Two qualifications, both of which weaken the headline and are kept:**

1. **The margin past 1.0 is tiny.** 0.7000 and 0.7375 against 0.6975 is 1–4 probe items out of
   400. "Slightly exceeds" is accurate; "substantially exceeds" is not.
2. **`theta_ref`'s 0.6975 is a point estimate** hard-coded in `train_small_rtt.py`, with no error
   bar. Comparing against a bare constant is what makes ">1.0" look dramatic, and that comparison
   is the fragile part.

**So the claim is stated as "restores to at least `theta_ref`'s level", not "beyond it."** That
form does not depend on the fragile arithmetic and is what the raw ranks support.

## 3. Why this is worse than the recovery fraction suggests

Both families that fully revert are **FAM-anchored** (EMDU, and MDU with full answer masking) —
the same construction `FINDING13`/`FINDING18` showed to be degenerate, and the one with the worst
behavioural profile (TV 0.3225). So these arms are simultaneously:

- the furthest from the oracle behaviourally (TV 0.3225 vs the prompt-masked anchors' 0.0725),
- and the least durable under attack (full reversion to reference level).

A practitioner selecting on behavioural metrics alone would see TV 0.3225 and reject them for the
wrong reason, or — worse — see EMDU's **DISCLOSE 0.000** and select them as the *best* forgetters.
`FINDING23` showed EMDU scores perfectly on disclosure while leaving the knowledge intact; this
finding shows an attacker then restores it completely.

## 4. Scope

- One backbone (124M), one synthetic corpus, one attack family (SFT-style relearning), one
  attacker budget for this claim (`t_frac 0.8`; at 0.5 these families recover 0.5608–0.7538, which
  does **not** reach reference level).
- Baselines are our re-implementations; the original authors' tuning is not reproduced.
- `n` per cell is 2–6 attacks, not 2–6 independent unlearning runs — the training seed is fixed
  per family and only the attack varies in most cells.
