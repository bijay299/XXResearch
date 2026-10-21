# AFC — Anchor-First matching on the Conditional-answer anchor

**Status:** selected candidate. Evidence is one seed on one backbone; ATWU's fair comparison and
the seed replicates are still running. **Not yet a SOTA claim.**
**Implementation:** `scripts/train_h16.py --mode none` (the `none` mode is AFC; the projection
modes are the closed H16 study)
**Findings:** `FINDING16_AFC.md`, with the negatives in `FINDING12`–`FINDING15`

---

## 1. What AFC is

For each forget pair `(x, y)`, sample a mask rate `t ~ U[0,1]` and a teacher-forced response
state `y_t`. Let `M_t` be the masked answer positions. Let `m` be the prompt with **every prompt
token replaced by the mask token**, and `θ₀` the frozen parameters at the start of unlearning.

```
L_forget(θ) = E_{(x,y)~D_f, t, y_t} [ (1/|M_t|) Σ_{i∈M_t}  KL( p_{θ₀}(·|m, y_t)_i  ‖  p_θ(·|x, y_t)_i ) ]
L(θ)        = λ_f · L_forget(θ)  +  λ_r · E_{(x,y)~D_r, t}[ CE(p_θ, y) ]
```

Selected operating point: `λ_f = 2`, `λ_r = 1`, 300 steps, LoRA r=16 on 224 modules
(`.*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$`), lr 1e-4, AdamW,
grad-clip 1.0, anchor temperature τ=1.

**Everything in that objective is published.** The anchor is MDU's
([2605.18253](https://arxiv.org/abs/2605.18253)) prompt-masked frozen-θ₀ anchor with its random
teacher-forced answer state. The KL direction is EMDU's
([2610.25447](https://arxiv.org/abs/2610.25447)) anchor-first direction. The retain term is the
same cross-entropy both use. **AFC is the cell of the 2×2 that neither paper ran:**

| | KL direction | anchor's answer span |
|---|---|---|
| MDU | anchor-last `KL(p_θ‖p_a)` | random `y_t` |
| EMDU | anchor-first `KL(p_a‖p_θ)` | **full answer-span masked (FAM)** |
| **AFC** | **anchor-first** | **random `y_t`** |

EMDU changes both coordinates at once. Separating them is the whole contribution.

## 2. Why each half behaves as it does — measured, not argued

**The direction helps, for EMDU's own stated reason.** Their Prop. 1 gives
`∇_z L_AF = q − p` against an anchor-last gradient weighted by `q` itself, and Thm 2 bounds the
latter by `C(p)·ε·log(V/ε)`: at a memorized near-one-hot prediction the anchor-last gradient
vanishes while the anchor-first one does not. We take this on their authority and hold the
direction fixed at anchor-first in every AFC arm.

**FAM hurts, for a reason we measured.** `FINDING13`: the all-masked anchor is two different
targets at the two ends of the noise schedule — fact-free but 42.2% punctuation and 4.41 nats
early (`p_true` 0.064), well-formed but fact-knowing late (`p_true` 0.651). FAM removes the fact
leak at every noise level **and the form scaffold with it**. Decoded without a scaffold the
all-masked, answer-masked anchor is **97.91% DEGENERATE** across 335 facts. Keeping the random
answer state keeps the scaffold, and the trained model keeps its form: AFC λ=2 has DEGENERATE
**0.042** where EMDU has **1.000**.

## 3. Results

Generation-level six-cell partition, 120 forget / 80 retain, θ₀ TOFU backbone, one seed.
Objective: minimise `TV(forget profile → pristine)` while preserving retain ASCR.

| arm | DISCLOSE | DEGENERATE | REPLACE | IGNORANT | retain ASCR | TV→pristine |
|---|---|---|---|---|---|---|
| θ₀ (start) | 0.792 | 0.008 | 0.092 | 0.108 | 0.762 | 0.770 |
| **AFC λ=2 (selected)** | **0.158** | **0.042** | 0.533 | 0.267 | **0.550** | **0.234** |
| AFC λ=1 | 0.208 | 0.050 | 0.417 | 0.325 | 0.533 | 0.229 |
| AFC λ=4 | 0.150 | 0.117 | 0.492 | 0.242 | 0.541 | 0.259 |
| AFC λ=0.5 | 0.258 | 0.117 | 0.367 | 0.258 | 0.582 | 0.345 |
| MDU λ=1 | 0.317 | 0.075 | 0.175 | 0.433 | 0.524 | 0.407 |
| MDU λ=2 | 0.225 | 0.092 | 0.233 | 0.450 | 0.403 | 0.349 |
| MDU λ=3 | 0.233 | 0.050 | 0.300 | 0.417 | 0.404 | 0.282 |
| MDU λ=8 | 0.067 | 0.150 | 0.392 | 0.392 | 0.091 | 0.191 |
| EMDU λ=8 | 0.000 | 1.000 | 0.000 | 0.000 | 0.580 | 1.000 |
| ATWU ρ=0.25, β=7 (published) | 0.433 | 0.025 | — | — | 0.552 | 0.466 |
| *pristine target* | *0.030* | *0.000* | *0.470* | *0.388* | — | *0* |

**AFC λ=2 dominates MDU λ=1 on all four coordinates** — lower DISCLOSE (0.158 vs 0.317), lower
DEGENERATE (0.042 vs 0.075), higher retain (0.550 vs 0.524), lower TV (0.234 vs 0.407) — and
dominates MDU λ=2 and λ=3 as well. The cleanest single statement: among MDU points that keep
retain ≥ 0.40, the best disclosure is **0.225 at retain 0.403**; AFC λ=2 reaches **0.158 at
retain 0.550**.

## 4. Limitations, each measured

- **Narrow usable band.** AFC collapses at λ_f = 8: DEGENERATE **0.908**, retain **0.011**. The
  usable range is λ_f ≈ 0.5–4 and the method fails catastrophically past it, as the baselines do.
- **Localization does not help it.** AFC + the divergence position selector is *worse*
  (TV 0.312 vs 0.229). That selector is validated as a *diagnostic* — its complement carries
  nothing forgettable — but it has now failed to improve three different anchor losses
  (MDU's, the entity anchor's, AFC's). Recorded as a repeated negative.
- **Still far from target.** DISCLOSE 0.158 against pristine's 0.030; retain 0.550 against θ₀'s
  0.762. AFC moves the frontier; it does not solve the problem.
- **One seed, one backbone, one dataset**, `λ_r` fixed at 1, 300 steps. Seed replicates for AFC
  and MDU (paired, seeds 1 and 2) are running.
- **No robustness claim.** RTT relearning is unrun. Anchor-matching families recover under
  finetuning attack ([2506.06278](https://arxiv.org/abs/2506.06278)) and our own closed study
  found MDLM unlearning reversible 0.049 → 0.605 in 10 steps.
- **ATWU comparison incomplete.** At its published β=7 the SatGA target is `e^{−1/7} = 0.867`,
  which is *above* this backbone's mean answer-token probability (0.606), so the published
  setting cannot forget here and in fact raises `p_true` to 0.876. That is a regime mismatch,
  not a defect in ATWU, and β ∈ {1, 1.5, 2} (targets 0.368 / 0.513 / 0.607) is being swept so
  the baseline is judged at a setting that can work on this model.

## 5. The evaluation argument, which is half the contribution

Three single-axis views of the same arms give three different winners:

| ranking axis | winner | why it is wrong alone |
|---|---|---|
| DISCLOSE only | EMDU, 0.000 | DEGENERATE 1.000 — emits garbage |
| TV(forget) only | MDU λ=8, 0.191 | retain 0.091 — model destroyed |
| DISCLOSE + retain | H16 projection | DEGENERATE 0.192 vs AFC's 0.042 |

We walked into the first and the third ourselves (`FINDING15` §3). MDU and EMDU report TOFU
forget quality as ROUGE-L and token probability, on which **a degenerate model scores
perfectly**. Any honest comparison needs the full six-cell profile *and* retain utility
together, which is the pairing used in §3.

## 6. Honest framing

AFC is **not a new algorithm**. It is a factorial cell of two published methods, and every term
in its objective is theirs. The contribution is: separating EMDU's two simultaneous changes,
measuring which half carries the benefit and why the other half hurts, demonstrating that the
resulting configuration dominates both parents on a protocol that cannot be gamed by
degeneracy, and the evaluation argument in §5. That is an ablation-and-evaluation contribution,
and the write-up will say so rather than dress it as a new objective.
