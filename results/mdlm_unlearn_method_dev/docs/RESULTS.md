# Results — AFC against MDU, EMDU and ATWU on TOFU/LLaDA

> **RETRACTION, 2026-10-21 (n=5 seeds).** The stability claim in §2 below is **REFUTED by our own
> data** and is retained only as a record of the error. At n=3 AFC's TV sd was 0.005; at n=5 it is
> **0.158 against MDU's 0.033** — AFC is roughly **5x MORE variable**, not 9-16x less. Seed 3
> produced TV 0.587 with DEGENERATE **0.508**. AFC's mean DEGENERATE at n=5 is **0.217 vs MDU's
> 0.122**, so the lower-degeneracy claim also fails at lam=1, and AFC breaches DEG <= 0.25 on 1 of
> 5 seeds.
>
> **What survives:** AFC's mean TV is still better (0.314 vs 0.424) and it wins 4 of 5 paired
> seeds, losing seed 3 by -0.171. The honest claim is therefore **"better on average, with a
> catastrophic-failure mode the baseline does not have"** -- not "better and more stable". The
> n=3 figures were three lucky seeds. n=8 is running to characterise the failure mode.


**Date:** 2026-10-21
**Backbone:** LLaDA-8B-Instruct rev `08b83a6feb34df1a6011b80c3c00c7563e963b07`, memorization
adapter `theta0_tofu` (qualified 0.791/0.787 recall, bitwise save/reload verified)
**Protocol:** `eval_arm_partition.py`, 120 forget / 80 retain probes, pinned sampler, greedy
(T=0), `gen_length=64`, `steps=64`, six-cell partition + ASCR-12s
**Objective:** minimise `TV(forget profile → pristine)` while preserving retain ASCR

---

## 1. Headline, at matched disclosure

Three seeds per arm (0, 1, 2), paired, mean ± sample sd:

| arm | n | DISCLOSE | TV→pristine | retain ASCR | DEGENERATE |
|---|---|---|---|---|---|
| **AFC λ_f=1 (selected)** | 3 | 0.136 ± 0.063 | **0.223 ± 0.005** | **0.576 ± 0.038** | **0.117 ± 0.058** |
| MDU λ_f=3 | 3 | 0.133 ± 0.090 | 0.352 ± 0.081 | 0.445 ± 0.053 | 0.181 ± 0.148 |
| MDU λ_f=1 | 3 | 0.233 ± 0.104 | 0.423 ± 0.046 | 0.557 ± 0.052 | 0.194 ± 0.147 |

MDU λ_f=3 is the matched-disclosure comparison: **0.133 against AFC's 0.136**, statistically
indistinguishable. At that matched point AFC gives

- **TV→pristine 0.223 vs 0.352** — 37% lower
- **retain ASCR 0.576 vs 0.445** — 29% higher
- **DEGENERATE 0.117 vs 0.181**

AFC also wins all four coordinates against MDU λ_f=1, and its TV advantage holds on **every
individual seed** (+0.179, +0.254, +0.167).

## 2. Stability — RETRACTED (see the note at the top; kept as a record of the error)

| arm | TV sd | DEGENERATE range across seeds | seeds breaching DEG ≤ 0.25 |
|---|---|---|---|
| **AFC λ_f=1** | **0.005** | 0.050 – 0.150 | **0 / 3** |
| AFC λ_f=2 | 0.053 | 0.042 – 0.292 | 1 / 3 |
| MDU λ_f=1 | 0.046 | 0.075 – 0.358 | 1 / 3 |
| MDU λ_f=3 | 0.081 | 0.050 – 0.342 | 1 / 3 |

AFC λ_f=1 has **9–16× lower TV variance** than any MDU setting and is the only arm that never
breaches the degeneracy threshold. MDU's degeneracy swings from 0.075 to 0.358 on identical
settings with only the seed changed.

**Consequence for the literature:** a single-seed comparison between these methods is close to
uninformative, and both parent papers report single configurations. This is part of the
contribution, not a caveat.

It also forced a correction of our own: λ_f=2 was selected on seed 0 alone (DEG 0.042) and
**replication reversed it** — λ_f=2 has 10× the TV variance and breaches the degeneracy
threshold on 1 of 3 seeds. The selected point is λ_f=1.

## 3. ATWU — 11 configurations, and the honest framing

ATWU ([2606.06320](https://arxiv.org/abs/2606.06320)) claims SOTA forget–retain trade-offs on
TOFU and RWKU, so it cannot be skipped. Implemented faithfully from the PDF: SatGA+
`g·p^{βg}·log p`, `g = σ(⟨w,h⟩)` with `w` initialised to `0_d`, binary-entropy and budget
penalties, alternating scorer/model updates, hidden states taken at the lm_head input.

Given its published 8B/TOFU settings (α=0.50, γ=3.00, β=7.00, scorer-lr 0.050, lr 1.5e-5,
λ_ent=1, λ_pop=15) **plus** sweeps over everything we could not recover or that could plausibly
disadvantage it:

| swept | values |
|---|---|
| ρ (budget — supplementary-only in the paper) | 0.10, 0.25, 0.50 |
| β (saturation exponent) | 1.0, 1.5, 2.0, 7.0 |
| model lr | 1.5e-5 (published), 1e-4 (matched to AFC) |
| steps | 300, 900 |
| selector | learned, our divergence selector |
| forget loss | SatGA+, anchor KL |

**Best of 11 configurations by TV: 0.466 (retain 0.552, DISCLOSE 0.433, DEG 0.025).** AFC's TV
is **less than half** that, at comparable retain.

**What we will and will not say.** ATWU is specified for autoregressive models. The MDLM
adaptation — applying SatGA+ at masked answer positions of a teacher-forced state, with hidden
states from the lm_head input — is **ours, not theirs**, and may not do the method justice. So
the claim is *"our MDLM adaptation of ATWU underperforms AFC on this protocol"*, **not** *"AFC
beats ATWU"*. ATWU is also explicitly composable with any forget loss, so ATWU's selector on
AFC's anchor is a natural stronger configuration that we have only probed at one setting.

Two measured notes on it: its learned selector agrees with our free divergence selector only
**33%** of the time on the anchor loss, so the two criteria find substantially different
positions; and its consistently low degeneracy (0.017–0.042 across all configurations) is a
genuine strength of the saturated loss that AFC does not match.

## 4. Full frontier (single seed unless noted)

Under the project's standard (DEGENERATE ≤ 0.25):

| arm | DISCLOSE | DEGENERATE | retain | TV→pristine |
|---|---|---|---|---|
| θ₀ (start) | 0.792 | 0.008 | 0.762 | 0.770 |
| **AFC λ=1 (n=3)** | **0.136** | **0.117** | **0.576** | **0.223** |
| AFC λ=2 (n=3) | 0.100 | 0.144 | 0.534 | 0.246 |
| AFC λ=4 | 0.150 | 0.117 | 0.541 | 0.259 |
| AFC λ=0.5 | 0.258 | 0.117 | 0.582 | 0.345 |
| MDU λ=3 (n=3) | 0.133 | 0.181 | 0.445 | 0.352 |
| MDU λ=1 (n=3) | 0.233 | 0.194 | 0.557 | 0.423 |
| MDU λ=8 | 0.067 | 0.150 | 0.091 | 0.191 |
| ATWU best of 11 | 0.433 | 0.025 | 0.552 | 0.466 |
| EMDU λ=8 | 0.000 | **1.000** | 0.580 | 1.000 |
| *pristine target* | *0.030* | *0.000* | — | *0* |

Excluded for degeneracy: EMDU (1.000), AFC λ=8 (0.908), H16 projection k=32 (0.275).

## 5. Limitations

- **One backbone, one dataset.** TOFU-style split on a synthetic-memorization LLaDA backbone.
  No RWKU, no Dream-7B. Transfer is untested.
- **No robustness claim.** RTT relearning is unrun. Anchor-matching families recover under
  finetuning attack ([2506.06278](https://arxiv.org/abs/2506.06278)), and our own closed study
  found MDLM unlearning reversible 0.049 → 0.605 in 10 steps. This applies to AFC exactly as to
  MDU.
- **Narrow usable band.** AFC collapses at λ_f=8 (DEGENERATE 0.908, retain 0.011). Usable range
  λ_f ≈ 0.5–4.
- **Still far from the target.** DISCLOSE 0.136 against pristine's 0.030; retain 0.576 against
  θ₀'s 0.762. AFC moves the frontier; it does not solve the problem.
- **ATWU adaptation is ours** (§3).
- **AFC is not a new algorithm** — it is a factorial cell of two published methods, every term
  theirs. The contribution is separating EMDU's two simultaneous changes, measuring which half
  carries the benefit and why the other hurts, the stability finding in §2, and the evaluation
  argument below.

## 6. The evaluation argument

Three single-axis views of the same arms give three different winners:

| ranking axis | winner | why it is wrong alone |
|---|---|---|
| DISCLOSE only | EMDU 0.000 | DEGENERATE 1.000 — emits garbage |
| TV(forget) only | MDU λ=8, 0.191 | retain 0.091 — model destroyed |
| DISCLOSE + retain | H16 projection | DEGENERATE 0.192 vs AFC's 0.042 |

We fell into the first and the third ourselves. MDU and EMDU report TOFU forget quality as
ROUGE-L and token probability, on which **a degenerate model scores perfectly**. An honest
comparison needs the full six-cell profile *and* retain utility together — the pairing used
throughout this document.

## 7. Negative results, recorded as such

Eight candidates closed on measured evidence: H1, H6, H11, C1/SIO, the ordering line
(H10a/b/c — `FINDING11`, `FINDING12`), ESD (anticipated by MDU), the form-preserving anchor
(H14), localized anchor matching (H15 — `FINDING15`), and retain-conflict projection
(H16 — rank sweep 3/3 worse than no projection). The divergence position selector is validated
as a diagnostic — its complement carries nothing forgettable — but has now failed to improve
three different anchor losses, which is recorded as a repeated negative rather than buried.
