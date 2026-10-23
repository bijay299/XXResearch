# Pre-registration H67 — does the FRAG predictor transfer from LLMs to T2I diffusion?

**Date:** 2026-10-23 · **Status:** pre-registered. **No FRAG score has been computed for any arm.**
**Source:** arXiv 2608.25429 Appendix A.1 (Eq. 7–9), read directly from the PDF.
**Cost:** weights and calibration forward passes only — **no image generation, no attack, no
training**. Runs on CPU so it does not touch the two-GPU budget.

---

## 1. The claim being tested, in its authors' terms

FRAG is a **training-free predictor of relearning robustness**: it scores whether an unlearning
update landed on forget-critical weights while sparing retain-critical ones, without running an
attack. Their reported result, on LLMs:

| predictor | Spearman ρ vs ΔES (pooled, healthy checkpoints) | without FRP checkpoints |
|---|---|---|
| global ℓ2 distance | −0.36 | −0.10 |
| **FRAG** | **−0.78** | **−0.74** |

More negative is better. **H67 asks whether that transfers to a text-to-image latent diffusion
model**, which the paper does not test and does not claim.

This project is unusually well placed to ask, because it has **34 arms with measured post-attack
recovery under a documented protocol** — the expensive half of such a study already paid for.

## 2. The computation, transcribed before implementing it

Per layer ℓ, with `W₀` the **base** model's weight and `W_u` the arm's:

- `x_f,j`, `x_r,j` — per-input-channel ℓ2 activation norms (Wanda-style), **both computed on `W₀`**;
  the paper is explicit that `W_u` *"is read but never executed"*
- `F_ij = |W₀_ij| · x_f,j / (x_r,j + ε)` and `R_ij = |W₀_ij| · x_r,j / (x_f,j + ε)`
- `D_ℓ = (W_u − W₀)²` — **squared**, per Eq. (8)
- one **global** cosine across all layers, not per-layer cosines averaged (Eq. 9):
  `A_f = Σ_ℓ⟨F_ℓ,D_ℓ⟩ / (√Σ_ℓ‖F_ℓ‖² · √Σ_ℓ‖D_ℓ‖²)`, likewise `A_r`
- `FRAG = A_f − γ·A_r`, with **γ = 1**, **ε = 1e-6** (their defaults)

## 3. Port decisions, declared now

**(a) Module scope.** The paper scores every linear projection inside each transformer block, and
says that for other architectures the predictor *"falls back to all `nn.Linear` modules"*. For the
SD UNet that is the **160 linears inside `BasicTransformerBlock`s**: 128 attention projections
(self- and cross-attention `to_q/to_k/to_v/to_out`) plus the 32 feed-forward linears. This is
**wider than H66's FRP scope**, which is MLP-only per FRP's own Table 4 — the two scopes are
different in the paper too, and are kept different here.

**(b) Calibration.** 16 forget prompts and 16 retain prompts (recipe v2 pools, `verify_splits`
CLEAN), each over 8 timesteps on a fixed grid, seed 42. One 64×64 UNet forward presents 4 096
positions per block, so this gives ≳ 10⁵ positions per side against the paper's 128 × 256 = 32 768.
Computed **once on the base model** and reused for every arm, which is what makes this cheap.

**(c) ⚠ Eligible arms — and the exclusion that matters.** `D_ℓ = (W_u − W₀)²` is a **UNet** weight
difference. **Every official AdvUnlearn arm erases the TEXT ENCODER and leaves the UNet
bit-identical**, so `D ≡ 0` and FRAG is undefined for them — not low, undefined. They are
**excluded**, and that exclusion removes the best-utility arms in the whole phase. Eligible arms are
those with a genuinely modified UNet: UCE, ESD, damage+distil, composites, and H66's FRP.

This is a real limitation of the predictor in this setting and is reported as a finding in its own
right, not buried: **FRAG as specified cannot score a text-encoder erasure at all.**

## 4. The outcome variable and the two populations, fixed now

- **Outcome:** `max-over-valid H` after attack, from `run_inventory.json`, post-dating the critical
  evaluation fix. Arms with no gate-passing attack are excluded (no valid recovery number exists).
- **Population A — all eligible arms.**
- **Population B — "healthy" arms only**, `legacy_retain8 ≥ 0.60`. This mirrors the paper's own
  protocol, which computes its correlations *"over healthy checkpoints only; collapsed and noise
  controls are excluded"*. Our α=0.5/0.7 damage arms are exactly such collapsed controls.

Both populations are reported. **Population B is the pre-declared primary**, because it is the
paper's own protocol; reporting only A would test a different claim than theirs.

Comparators computed on the same arms: **global ℓ2** `‖W_u − W₀‖`, plus `A_f` and `A_r` separately,
so a result can be attributed to the forget term, the retain term, or the gap.

## 5. ⚠ The statistical limit, declared before seeing n

Eligible-and-valid arms will number roughly **10–20**. On n = 15, a Spearman's 95 % CI spans
roughly ±0.5: this study **cannot** distinguish ρ = −0.78 from ρ = −0.3, and **cannot** establish
that FRAG transfers. It can detect a **sign reversal or a flat relationship**, which is the
falsifiable part.

This project has recorded **four** false signals from n-too-small (`T2I_REPORT` §4), so the
pre-committed reporting rule is: **report ρ with its n and bootstrap CI, and make no transfer claim
in either direction unless the CI excludes zero.** A point estimate near their −0.78 is *consistent
with* transfer and is not evidence of it.

## 6. Separating predictions

| outcome | reading |
|---|---|
| FRAG ρ clearly negative, CI excludes 0, and beats ℓ2 on the same arms | the predictor transfers to T2I diffusion — a **positive result about a published method**, reportable as such and not as ours |
| FRAG ρ ≈ 0 or positive with CI excluding 0 | it does **not** transfer; a bounded negative about a published predictor in a modality its authors never claimed |
| CI includes 0 (the likely case at this n) | **inconclusive**, reported as inconclusive. No transfer claim, no "trending" language |
| ℓ2 matches or beats FRAG | the paper's central contrast does not reproduce here; worth reporting given they argue ℓ2 is insufficient |

**I expect inconclusive**, on n alone. Recording that so an inconclusive outcome is not later
presented as a finding, and a clean one is not claimed as predicted.

## 6a. Calibration-diversity limitation, noted after launch and before any score was read

The paper calibrates on **128 unique sequences per side**. Our recipe-v2 `FORGET_POOL` for a concept
holds only **3–4 unique prompt strings**, so the forget side has 16 prompt-slots drawn from 4 unique
texts, with diversity coming from 8 timesteps × a distinct latent per forward (128 forwards, 128
distinct latents) rather than from text. The retain side has 16 unique strings.

**Position count exceeds the paper's budget; unique-text diversity on the forget side does not.**
This is recorded as a limitation of the port, not corrected mid-run: enlarging the forget pool would
change what `D_f` means for every arm and would have to be a new recipe. Flagged here, dated, and
before any FRAG score was read — the run had started but no number had been produced.

## 7. Honesty constraints

- FRAG is **their** predictor. A positive result is a result about their work.
- No arm is added, dropped, or rebuilt after a FRAG score is seen. The eligible set is fixed by the
  rule in §3(c) and §4 **before** any score is computed.
- The text-encoder exclusion is reported as a scope limitation of the predictor, with the count of
  arms it removes stated explicitly.
