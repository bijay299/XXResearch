# Finding 16 — AFC: separating EMDU's two changes, and the half that helps

**Date:** 2026-10-21
**Status:** strongest result so far. One seed, one backbone — **not yet a claim against SOTA**
(ATWU unrun, no seeds, no relearning).
**Script:** `scripts/train_h16.py --mode none` · **Arms:** `h16_none_f1`, `afc_*`
**Outputs:** `outputs/arm_h16_none_f1.json` and the `arm_*` family

---

## 1. The factorial cross neither paper evaluated

MDU ([2605.18253](https://arxiv.org/abs/2605.18253)) and EMDU
([2610.25447](https://arxiv.org/abs/2610.25447)) differ in **two** things at once, and EMDU
changes both together:

| | KL direction | anchor's answer span | position weights |
|---|---|---|---|
| MDU | anchor-**last** `KL(p_θ‖p_a)` | random (`y_t`, same noise as the student) | uniform `1/|M_t|` |
| EMDU | anchor-**first** `KL(p_a‖p_θ)` | **full answer-span masked (FAM)** | uniform `1/|M_t|` |
| **AFC** (this) | anchor-**first** | random | uniform |

EMDU motivates both changes well — Prop. 1 gives `∇_z L_AF = q − p` against an anchor-last
gradient weighted by `q` itself, and Thm 2 bounds the latter by `C(p)·ε·log(V/ε)`, so the
anchor-last gradient vanishes at a memorized near-one-hot prediction; FAM is motivated by the
observation that surviving answer tokens cue the fact in the anchor. Both motivations are sound.
**But they are never separated**, and the cross is never run.

## 2. Result on the declared objective

The project's objective is to minimise `TV(profile(arm), profile(pristine))` on the forget split
while preserving retain. Generation-level, 120 forget / 80 retain, θ₀ backbone, `λ_retain = 1`:

| arm | TV→pristine | retain ASCR | DISCLOSE | DEGENERATE |
|---|---|---|---|---|
| MDU λ=8 | 0.191 | **0.091** | 0.067 | 0.150 |
| **AFC λ=1** | **0.229** | **0.533** | 0.208 | **0.050** |
| proj k1 λ=1 | 0.245 | 0.524 | 0.142 | 0.133 |
| MDU λ=3 | 0.282 | 0.404 | 0.233 | 0.050 |
| proj k8 λ=1 | 0.337 | 0.608 | 0.175 | 0.192 |
| MDU λ=2 | 0.349 | 0.403 | 0.225 | 0.092 |
| MDU λ=1 | 0.407 | 0.524 | 0.317 | 0.075 |
| EMDU λ=8 | 1.000 | 0.580 | 0.000 | 1.000 |
| *pristine target* | *0* | — | *0.030* | *0.000* |
| *θ₀ start* | *0.770* | *0.762* | *0.792* | *0.008* |

**At matched retain — 0.533 against MDU's 0.524 — AFC's TV is 0.229 against 0.407, a 44%
reduction.** AFC dominates MDU at λ ∈ {1,2,3} on both axes simultaneously, and also dominates
the rank-1 projection arm. Only MDU λ=8 reaches a lower TV, and it does so with retain
**destroyed** (0.091 against θ₀'s 0.762), which is not a usable model.

**So the KL direction is the useful half of EMDU's change and FAM is the harmful half.** Keeping
the direction and discarding FAM beats both published configurations. EMDU's own Theorem 2
supplies the reason the direction helps; our Finding 13 measurement supplies the reason FAM
hurts — FAM removes the fact leak *and* the form scaffold at every noise level, and the
all-masked, all-answer-masked anchor is 97.9% degenerate as a generator.

## 3. Three single-axis views, three different winners — and why that matters

The same eight arms rank differently depending on which axis is used:

| ranking axis | winner | why it is wrong alone |
|---|---|---|
| DISCLOSE only | EMDU (0.000) | emits pure garbage, DEGENERATE 1.000 |
| TV(forget) only | MDU λ=8 (0.191) | retain destroyed, 0.091 |
| DISCLOSE + retain only | proj k8 | DEGENERATE 0.192 vs AFC's 0.050 |

I walked into the first of these myself (Finding 15 §3) and into the third one step later, when
my own projection arm looked best on disclosure-and-retain while being worse on the full
profile. The honest reading requires **both** aggregates: TV over the full six-cell forget
profile *and* retain ASCR. That pairing is what the table in §2 uses, and AFC is Pareto-optimal
under it at usable retain.

This is a reviewable claim about the literature's standard metrics: MDU and EMDU report TOFU
forget quality as ROUGE-L and token probability, and on those axes a degenerate model scores
perfectly.

## 4. What is NOT claimed

- **Not SOTA.** ATWU ([2606.06320](https://arxiv.org/abs/2606.06320)) claims state-of-the-art
  forget–retain trade-offs on TOFU and RWKU and is implemented here but **has not been run**.
  No comparison against the actual state of the art exists yet.
- **Not a new algorithm.** AFC is a factorial cell of two published methods. The contribution is
  identifying that EMDU's two changes must be separated, measuring which half carries the
  benefit, and the evaluation argument in §3 — not a new objective. It will be written that way.
- **One seed, one backbone, one dataset**, `λ_retain` fixed at 1, 300 steps, LoRA r=16 on 224
  modules. H15's retain was non-monotonic in λ on exactly this setup, so single points are not
  trustworthy and the AFC curve (λ ∈ {0.5, 1, 2, 4, 8}) is being mapped before anything is
  concluded.
- **No robustness claim.** Anchor-matching families recover under finetuning attack
  ([2506.06278](https://arxiv.org/abs/2506.06278)) and our own study found MDLM unlearning
  reversible 0.049 → 0.605 in 10 steps. RTT is unrun.
- AFC is still far from the target: DISCLOSE 0.208 against a pristine 0.030, retain 0.533
  against θ₀'s 0.762.

## 5. Next, in order

1. **ATWU**, published hyperparameters, ρ swept — the gate on any SOTA language.
2. **AFC curve** at λ ∈ {0.5, 2, 4, 8} and **AFC + the validated position selector**, to test
   whether localization helps *this* loss where it failed on MDU's.
3. **Three seeds** on the selected point, with the frontier redrawn as curves.
4. **RTT relearning** on the selected point.
