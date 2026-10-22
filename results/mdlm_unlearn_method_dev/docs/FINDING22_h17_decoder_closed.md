# Finding 22 — the decoder attack closes: AFC's spread is backbone decoder variance, not unlearning fragility

**Date:** 2026-10-22 · **Pre-registration:** `PREREG_H17_decoder_attack.md`
**Script:** `scripts/decode_attack.py` · **Outputs:** `outputs/decatk_theta0.json`,
`outputs/decatk_afc_l1.json`
**Status:** closes H17. The reference arm it was waiting on has now been run.

---

## 1. The owed arm

H17 asked whether an attacker who merely changes the **decoder** — no weights, no data, no
gradients — can reopen facts an unlearning method removed. Every MDLM unlearning result we know
of (MDU [2605.18253](https://arxiv.org/abs/2605.18253), EMDU
[2610.25447](https://arxiv.org/abs/2610.25447), and every arm in this project) is evaluated at a
single decoder: `steps = gen_length = block_length = 64`, `low_confidence`.

The pre-registration named a mandatory fourth arm: **θ₀ itself**, as the no-unlearning reference
for the grid's own variability. Without it, an arm's spread across decoders cannot be attributed
to unlearning fragility rather than to decoder variance that was always there. That arm was owed
for a long time and is now run: 24 configurations, 60 forget facts, `steps ∈ {16,32,64,128}` ×
`block ∈ {16,32,64}` × `remasking ∈ {low_confidence, random}`.

## 2. The measurement

| arm | DISCLOSE at the training config | max over grid | min over grid | **rise over training config** | spread |
|---|---|---|---|---|---|
| **θ₀ — no unlearning** | 0.8500 | 0.8667 | 0.6833 | **+0.0167** | 0.1834 |
| **AFC λ=1** | 0.2250 | 0.2500 | 0.0250 | **+0.0250** | 0.2250 |

Pre-registered kill rules, applied as written:

| rule | threshold | measured | fires? |
|---|---|---|---|
| decoder-FRAGILE | any arm's max DISCLOSE exceeds its training-config DISCLOSE by ≥ 0.15 | AFC **+0.0250**, θ₀ +0.0167 | **no** |
| decoder-ROBUST, line closes | spread < 0.05 for all arms | AFC 0.2250, θ₀ 0.1834 | **no** |

Neither threshold fires, so the verdict stays `PARTIAL_report_spread_only` — which is why the
reference arm matters.

## 3. What the reference arm settles

> **θ₀, with no unlearning applied at all, varies by 0.1834 across the decoder grid.** AFC's
> spread of 0.2250 is therefore not evidence that unlearning introduced decoder fragility: the
> backbone already swings by nearly that much before any method touches it.

And the quantity an attacker actually gains is small in absolute terms. Switching from the
training decoder to the *best* adversarial decoder moves AFC's disclosure from 0.2250 to 0.2500
— **+0.025**, against θ₀'s own +0.0167 of headroom at the top end. The attacker ends at 0.25
disclosure where θ₀ sits at 0.85.

**So the decoder attack is not a threat for these arms, and H17 closes as a negative.** A reader
should still report spread rather than a single decoder — that part of the gap statement stands,
and it costs nothing — but "choose a better decoder" does not recover removed facts here.

## 4. Why this was worth running even though it is a negative

Without the θ₀ arm, AFC's 0.2250 spread is readable two ways: as unlearning being decoder-fragile
(a real vulnerability), or as the backbone's own decoder variance (no vulnerability at all). The
reference arm discriminates them, and it chose the second. This is the same discipline as
`FINDING20`'s attack-validity gate: an attack result means nothing until you have shown what the
attack does to a control.

The line was also already closed on novelty grounds — the adversarial-decoder idea is anticipated
by Zhang et al.'s adversarial-RL work (`NOVELTY_CHASE.md`) — so this finding adds the measurement,
not a claim.

## 5. Status

Twenty candidates closed. H17 is now closed on **measured** evidence as well as novelty, and the
pre-registered arm the protocol demanded is no longer outstanding.
