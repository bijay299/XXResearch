# Pre-registration H66 — a faithful port of Forget-Retain Pruning (FRP) to Stable Diffusion, as a competing baseline

**Date:** 2026-10-23 · **Status:** pre-registered. **No FRP number exists at this writing.**
**Source:** arXiv **2608.25429** (Chen et al., EMNLP 2026), Algorithm 1, Eq. (3)–(4) and (6),
Table 4 — all read directly from the PDF.
**What this is:** the first of the five competing defences for this threat model to be **run at
all**. It is **our port**, labelled exactly as our UCE and ESD re-implementations are.

---

## 1. Why this, and why it is not a run of the authors' code

`AUDIT_CLOSURE_r2` §7's recommendation was to run FRAG/FRP because it was the only one of the five
with released code. **That was wrong and is corrected in `ERRATUM_r2` C19:**
`github.com/Yi1-Chen/FRAG` holds only a README and two images, and says verbatim *"Code release is
in preparation."* **Zero of the five competing defences have usable released code.**

The recommendation survives on different grounds. FRP is the cheapest of the five to port because:

| property | FRP |
|---|---|
| Algorithm 1 | fully specified in the paper |
| training | **none** — calibration forward passes and a pruning mask |
| cost here | minutes, not GPU-hours |
| declared settings | ρ = 3 % (MLP only), β = 0.05, zero-out, 400 seq × 256 tok, seed 42 |

So this is a **port of a published algorithm from its paper**, not a reproduction of the authors'
numbers and not a run of their implementation. Any disagreement with their reported results is
attributable to the port or to the domain change, and will not be presented as a failure of their
method.

## 2. The algorithm, transcribed before implementing it

For each target module with weight `W ∈ R^{d_o × d_i}`:

1. `x_f, x_r ←` input-channel norms on `D_f`, `D_r` (Wanda-style activation calibration)
2. `F_ij ← |W_ij| · x_f,j / (x_r,j + ε)` and `R_ij ← |W_ij| · x_r,j / (x_f,j + ε)`
3. per **output row** `i`: `S_ij ← rank_j(F_ij) − β·rank_j(R_ij) + λ·rank_j(|W_ij|)`
4. `P_i ← TopK(S_i,:, ⌊ρ·d_i⌋)`
5. `W_{i,P_i} ← 0`

`rank_j` ranks within the output row, larger value → larger rank. Pruning is **zero-out**, and the
paper's scope is **MLP only**.

## 3. The three port decisions, declared now

**(a) Module scope.** The paper targets the three MLP projections per transformer block. The SD
UNet analogue is the **feed-forward linears inside each `BasicTransformerBlock`**: `ff.net.0.proj`
(the GEGLU input projection) and `ff.net.2` (the output projection). Verified on the pinned model:
**32 layers, 148.7 M parameters = 17.3 % of the 859.5 M UNet.** Attention projections (128 linears)
are **excluded**, matching the paper's MLP-only setting — and deliberately so, since UCE's edit
surface is exactly the cross-attention K/V, and keeping FRP off that surface keeps the two methods
mechanistically distinct rather than overlapping.

**(b) Calibration.** The paper uses 400 sequences × 256 tokens = ~102 400 token positions per side.
In the SD UNet a single forward at 64×64 latent presents **4 096 spatial positions per block**, so
token count is not the binding constraint — prompt diversity is. Declared: the concept's
`FORGET_POOL` prompts (recipe v2) as `D_f` and 32 `RETAIN_POOL` prompts as `D_r`, each over **8
timesteps** sampled on a fixed grid, seed 42. That yields ≳ 10⁵ positions per side, exceeding the
paper's budget, and uses **only prompts a practitioner has** — no concept images, no oracle.

**(c) λ is never given a value anywhere in the paper.** Table 4 lists ρ, β, damage mode, calibration
and seed; λ appears in Algorithm 1 and Eq. (6) and is never assigned. This is a reporting gap in the
paper. I will **not** resolve it by choosing whatever flatters the result:

> **Pre-declared primary: λ = 0.05**, matching β's magnitude, on the stated reasoning that the paper
> describes both as modifiers to a dominant forget term. The **full λ ∈ {0, 0.05, 1.0} sweep is
> reported in all cases** as a sensitivity analysis, because the method is training-free and the
> sweep costs minutes. The headline FRP row is λ = 0.05 regardless of which λ performs best.

## 4. The operating point, selected by a rule fixed before any number is seen

ρ = 3 % is the paper's LLM setting and may erase nothing or everything in a diffusion model. The
full ladder **ρ ∈ {1, 3, 5, 10, 20} %** is reported. The arm carried forward to the attack grid is
chosen by this rule, declared now:

> **The smallest ρ whose target suppression reaches ≤ 0.400** — the erasure depth of the official
> AdvUnlearn church encoder, so FRP is compared to the robust SOTA at matched erasure rather than at
> whatever depth suits it. **Utility is not consulted in this selection.** If no ρ ≤ 20 % reaches
> 0.400, that is reported as the port failing to erase, and no robustness number is claimed.

The paper's own ρ = 3 % point is reported alongside, whether or not it is the selected one.

## 5. Separating predictions

| outcome | reading |
|---|---|
| erasure-matched FRP at retain ≥ 0.85 with max-over-valid H ≤ 0.40 | **the mandate's bar met by a PUBLISHED method, not ours.** The correct report is that FRP transfers to T2I and beats everything we built — and `PREREG_H40` is refuted by someone else's method |
| FRP at retain ≥ 0.85 with H ≤ 0.50 but > 0.40 | a real frontier point short of H40's threshold; H40 weakened, FRP the best arm measured |
| FRP fully reversed (H → 1.000) like every other arm | the relearning adversary defeats FRP too; the phase's negative result extends to a method designed for relearning robustness, which is a **stronger** version of it |
| no ρ ≤ 20 % erases to 0.400 | the port does not transfer; reported as a port/domain limitation, **not** as evidence against the paper |
| utility collapses at the ρ needed to erase | the forget/retain inseparability of `PREREG_H51` §7 reappears in pruning as well as in noise — a second independent route to the same provisional mechanism claim |

**I expect the third outcome.** Every arm in this phase has been fully reversed, including the
adversarially-trained robust SOTA, and FRP's robustness evidence is on LLM text benchmarks against a
one-epoch attack — a weaker adversary than 300 steps on 16 images. Recording that expectation now so
a negative cannot be dressed up as a prediction and a positive cannot be claimed as one.

## 6. Protocol and honesty constraints

- Attacks: `PROTOCOL_FROZEN_v1`'s six-point grid, **identical** to the grid `H65` measured the
  AdvUnlearn baselines under, so the comparison is budget-matched by construction.
- Utility and erasure: `gen_eval.py --target`, pools recipe v2, `verify_splits.py` CLEAN.
- Post-attack utility (issue 7) measured on every attack.
- Both concepts (church, parachute) before any claim, per `PREREG_H64` K3.
- **FRP is a published method and will never be described as ours**, whatever it scores. If it wins,
  it wins; the mandate is not served by hiding that.
- The port's fidelity limits are stated with the result: MLP-only scope mapped to SD's FF layers,
  our calibration construction, and λ chosen by us because the paper omits it.
