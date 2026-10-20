# C — Mask-State Discriminant Adapter (MSDA)

Agent C (parameterisation track), wave 1. Evidence here is exact linear algebra and a tiny bidirectional
toy. **Nothing here is evidence about LLaDA-8B.** `$R = research/llada_method_design_v1`.

**In one paragraph.** MSDA is a LoRA-shaped adapter on the MLP down-projections. Its input factor `A` is
**frozen**. `A`'s rows are the top generalised eigenvectors of two second moments of that layer's input
("keys"), both measured at `[MASK]` answer positions:
- the forget-state moment, taken over the *mask-state distribution* of the forget set: teacher-forced masks
  plus native-rollout states under ≥2 decoding configurations;
- the retain-state moment, taken over retain states produced by the same masking process.

Only `B` (zero-init) is trained, with an MDU-style objective plus retain-KL; at the end `BA` is merged into `W`.
The adapter's row space is therefore the set of key directions with the highest forget-to-retain energy
ratio, so the selectivity is explicit rather than coming only from having fewer parameters. Rank rule:
`r = #{μ_j ≥ τ}`. Proposition 1 gives the layer-local cost of any edit and the exact point where selectivity
becomes impossible (μ ≲ 1).

What the toy supports and does not support:
- **Supported:** the GEP subspace beats a random frozen subspace by a wide margin; the state set must cover
  partially-visible / rollout states; in-context copying is protected.
- **Not established:** a retention advantage over tuned LoRA at matched forgetting. MSDA wins 2/3 seeds at
  strict forgetting, loses 1/3 there, and loses at loose forgetting (§7).

---------------------------------------------------------------------------------------------------------
## 1. Failure / inefficiency of current practice

**(a) Full fine-tuning (MDU, and the GA/NPO/… baselines transplanted into it) updates `W` only along the
forget keys, in the Euclidean metric.**
- Source: `/tmp/claude-0/src/MDU/src/unlearn_mdu_llada.py` `train()` (l.1629–1744) loads the full model and
  hands it to a HF `Trainer`. `grep -rin "lora\|peft"` over the MDU repo returns nothing, so it is full FT.
- For any linear map `W_ℓ` with input key `k` and back-propagated output gradient `g`, every step is
  `ΔW = −η Σ g(s,i) k(s,i)ᵀ`. AdamW rescales this per element.
- So, before Adam's diagonal rescaling, the update's row space is span{forget keys}, with no reference to
  how retain keys are distributed.
- For one key, compare the *least-damage* edit that achieves `ΔW k = r` with the Euclidean edit
  `r kᵀ/‖k‖²`. Their retain damage `E_R‖ΔW k_R‖²` differs by the factor
  `(kᵀC_R k)(kᵀC_R⁻¹k)/‖k‖⁴ ∈ [1, (κ+1)²/4κ]` (Kantorovich), where `κ = cond(C_R)`.
- Exact check, `out_linear_prop.json`: gains 1.01 / 1.55 / 6.2 / 16.1 at κ = 1.5 / 9.6 / 106 / 1018. All
  are ≤ the bound (1.04 / 2.9 / 27 / 255).

**(b) MDM-specific reason the anisotropy is large.** Predictions are read only at `[MASK]` positions, and
every masked position starts from the same `[MASK]` embedding. So keys at masked positions share a large
common component.

Toy (`out_diag_keys.json`, 3 seeds):
- The top principal direction of the retain covariance at masked positions *is* the retain mean (|cos| = 1.00).
- That direction carries 74–80% of retain-key variance and 74–82% of forget-key energy.
- It carries only 3–9% of the energy of the fact-carrying difference `Δk = k(s) − k(anchor(s))`.

A Euclidean update driven by `k` therefore puts most of its row mass on the direction every masked readout
uses.

**(c) LoRA (zero-init B) has no retain-aware selectivity at initialisation.**
- With `B = 0`, `∂L/∂A = Bᵀ(…) = 0` and `∂L/∂B = G(AK)ᵀ`.
- So the first update is `ΔW = −η G Kᵀ AᵀA`: the Euclidean full-FT gradient filtered through a random
  r-dimensional subspace.
- A random rotation leaves the expected damage-to-forget ratio unchanged. Whatever selectivity LoRA has
  beyond "fewer parameters" must come from how `A` evolves under the same loss, which is again
  forget-key-driven.
- Toy: a random *frozen* `A` was much worse than the GEP `A`. At r = 16 it reached matched forgetting in
  only 0–2 of 3 seeds, depending on the threshold. LoRA4 failed at the strict threshold, and LoRA16/48 were
  the strongest baselines (§7).

**(d) A ROME/MEMIT-style single-state key edit does not cover the MDM state distribution.**
- In a bidirectional mask predictor the key at the edited layer depends on the state: which answer tokens
  are visible, and whether they are teacher-forced or the model's own rollouts.
- Toy: the whitened cosine between fact-carrying keys of the all-masked state and the partially-visible
  states is 0.53–0.91 (seed-dependent).
- A closed-form edit fitted only on the all-masked state (`CF-single`) could **not** reach matched
  forgetting on teacher-forced states at any regularisation (§7).
- TimeROME-DLM (arXiv 2606.12841, **SNIPPET-ONLY**) addresses the same issue differently, with a
  per-denoising-step *inference-time* edit memory and frozen weights. That is an inference-time component,
  which this proposal avoids.

---------------------------------------------------------------------------------------------------------
## 2. Mathematical formulation

**Notation**
- Prompt `c`; answer `y ∈ V^L` (SFT answer region including EOS padding).
- A state `s = (c, y_s)` has answer mask set `M(s) ⊆ {1..L}`. The mask predictor `p_θ(·|s)_i` is read at
  `i ∈ M(s)`.
- Anchor `a(s)`: `s` with the prompt replaced by `[MASK]` (MDU's prompt-masked anchor).
- Edited layer ℓ: the MLP down-projection `W_ℓ ∈ R^{d×m}` (`m` = MLP width). Its input at position `i` in
  state `s` is the key `k_ℓ(s,i) ∈ R^m`.
- `Q_F`, the forget-state distribution, is a mixture over forget pairs of:
  - (i) the SFT forward process, `t ~ U(0,1)`, each answer token masked i.i.d. w.p. `t`;
  - (ii) every intermediate state visited by native `generate` under a set `G` of ≥2 decoding configs
    (e.g. `low_confidence` vs `random` remasking × two `block_length`s) with the current model.
- Only masked answer positions count.
- `Q_R`: the same process on retain pairs plus generic instruction data, masked positions only.

**Second moments and the adapter**
- Fact-carrying key: `Δk(s,i) = k(s,i) − k(a(s),i)`.
- Forget moments: `C_Δ = E_{Q_F}[ΔkΔkᵀ]` and `C_F = E_{Q_F}[kkᵀ]`.
- Retain moment: `C_R = E_{Q_R}[kkᵀ] + ρ·(tr C_R/m)·I`, with shrinkage `ρ` because it is a finite-sample estimate.
- Erasure-request representation: the generalised eigenproblem `C_X u = μ C_R u` with `X ∈ {Δ, F}`,
  `μ_1 ≥ … ≥ μ_r`.
- `A_ℓ ∈ R^{r×m}` holds rows `u_1..u_r`. In the proposition they are `C_R`-orthonormal; in the
  implementation they are unit-normalised, which only rescales B.
- Adapter: `W_ℓ ← W_ℓ + B_ℓ A_ℓ`, `B_ℓ ∈ R^{d×r}` zero-initialised and trained, `A_ℓ` frozen.

**Proposition 1** (layer-local; exact for one linear map; does *not* assert output-level forgetting or
global utility). Define:
- `D_R(ΔW) = E_{Q_R}‖ΔW k‖²`, the change in the layer-ℓ output energy on retain states;
- `E_F(ΔW) = E_{Q_F}‖ΔW Δk‖²`, the change in the layer's response to the fact-carrying component.

Then:
- **(a) Universal lower bound.** Every ΔW has `D_R ≥ E_F / μ_1(C_Δ, C_R)`. Proof: for each row `w`,
  `wᵀC_R w ≥ wᵀC_Δ w/μ_1` (generalised Rayleigh quotient); sum over rows.
- **(b) MSDA attains it up to the spectral spread.** If `A` is `C_R`-orthonormal, then for `ΔW = BA`:
  `D_R = ‖B‖_F²` and `E_F = Σ_j μ_j‖b_j‖²`. Hence `E_F/μ_1 ≤ D_R ≤ E_F/μ_r`, so MSDA is within a factor
  `μ_1/μ_r` of the best damage-per-fact-effect achievable at layer ℓ.
- **(c) Cost of fact-blindness.** Making layer ℓ blind to the fact-carrying component on `Q_F`
  (`(W+ΔW)Δk = 0`) requires `E_F = E_{Q_F}‖W Δk‖² =: ε_ℓ`. So *any* linear edit of `W_ℓ` costs at least
  `D_R ≥ ε_ℓ/μ_1`.
- (a)–(b) hold verbatim with `(C_F, k)` in place of `(C_Δ, Δk)`. The default implementation uses `X = F`,
  which beat `X = Δ` in the toy (§7).
- **(d) Single-key exact optimum.**
  `min{E_R‖ΔWk_R‖² : ΔWk=r} = ‖r‖²/(kᵀC_R⁻¹k)`. The Euclidean edit pays the Kantorovich factor of §1a.
  Verified numerically: formula = numeric to 4 d.p.; random edits never beat the bound in (a).

**Failure boundary (feasibility).**
- Selective intervention at layer ℓ is feasible only if `μ_1(C_X, C_R) ≫ 1`. At `μ_1 ≲ 1` every edit of
  `W_ℓ` costs at least as much retain-output energy as the fact-effect energy it creates. MSDA, and every
  other adapter on that layer, is then non-selective.
- Special case: if a forget key is a typical draw from `Q_R`, then `kᵀC_R⁻¹k ≈ m` and `D* ≈ ‖r‖²/m`
  whatever κ is, so whitening gains nothing. Exact check: `D*_in-dist` = 0.19–0.32 vs 0.002 for a generic
  key at κ = 1018, with ‖r‖² ≈ 11–15 and m = 40.
- State-coverage corollary: MSDA acts on state `s` only through `AΔk(s)`. States whose fact-carrying key
  has a small captured fraction `‖AΔk(s)‖²/‖Δk(s)‖²_{C_R⁻¹}` are untouched. This is why `Q_F` must include
  rollout states from more than one decoding config.

**Not claimed:**
- the edit removes training influence (it is approximate removal at best: Patil et al. 2024, SNIPPET-ONLY,
  found deleted facts recoverable from hidden states after ROME-type edits);
- `D_R` equals KL damage;
- protection extends to retain behaviour absent from `Q_R`, because `Ĉ_R` is a sample estimate.

---------------------------------------------------------------------------------------------------------
## 3. Executable update

```
inputs: θ0 (LLaDA-8B-Instruct, TOFU-SFT), forget pairs D_F, retain pairs D_R, generic prompts D_G
        layers L_edit (default: MLP down_proj of 8 contiguous middle layers; module names UNVERIFIED, confirm
        on Hyperplane), rank rule r_ℓ = #{μ_j ≥ τ} (τ=4 in the toy; cap 256), shrinkage ρ=1e-2, configs G (≥2), λ_R
Stage 1  forward-only statistics with θ0, hooks on down_proj inputs (fp32 accumulators per layer)
  for (c,y) in D_F:
     S_tf  = {mask y iid w.p. t_j, t_j on a stratified grid of n_tf=16 levels}
     S_rol = all intermediate states of generate(c; g) for g in G        # states the sampler visits
     for s in S_tf ∪ S_rol:  k  = keys(s)[answer∩M(s)];  ka = keys(a(s))[same positions]
                             C_F += k kᵀ ; C_Δ += (k−ka)(k−ka)ᵀ
  for (c,y) in D_R ∪ D_G: same states (TF + one rollout config), C_R += k kᵀ at masked answer positions
  normalise by counts;  C_R += ρ (tr C_R / m) I
Stage 2  per layer: L = chol(C_R); M = L⁻¹ C_X L⁻ᵀ (X = F default, Δ ablation); U_r = top-r eigvecs(M)
         A = (L⁻ᵀ U_r)ᵀ ; normalise rows ; freeze.     report μ spectrum per layer; r_ℓ = #{μ_j ≥ τ}; r_ℓ = 0 ⇒ skip layer
Stage 3  insert ΔW_ℓ = B_ℓ A_ℓ (B_ℓ = 0). Train B only, AdamW, cosine LR, clip 1.0:
         L = MDU forget loss  (1/|M_t|)Σ KL(p_θ(·|x,y_t) ‖ p_θ0(·|m,y_t)^τ/Z)  on states from S_tf ∪ S_rol
           + λ_R · KL(p_θ0 ‖ p_θ) on retain states   (θ0 = same network with adapters disabled; no 2nd copy)
         every N steps optionally refresh S_rol with the *current* model (rollouts under G) — forget leakage of
         the edited sampler is exactly what must be covered
         early-stop at the matched forgetting level (operating-point protocol of the coordinator)
Stage 4  merge W_ℓ += B_ℓ A_ℓ. Deployed model = plain LLaDA weights; no prompt, no filter, no inference hook.
```

**Approximations, stated:**
- `A` comes from second moments of θ0's keys and is not updated as the edit moves the keys (one optional
  re-estimation after training).
- The rows are the discriminant directions of the layer's *input*. Downstream nonlinearity is ignored when
  choosing them; the loss in Stage 3 is the actual output-level objective.
- Rollout states are from θ0 unless refreshed.

**Closed-form variant (ablation `CF-ours`).** `ΔW = R K_Fᵀ(K_F K_Fᵀ + μ C_R)⁻¹`, MEMIT form, with residuals
`R = W(k_anchor − k)`. It is forward-only, but in the toy it was dominated by the trained variants (§7).
It is kept only as an ablation.

**`[MASK]`-gating (lead 3).** The option is to apply ΔW only where the input token is `[MASK]`. It is
**not** the default: in the toy it removed the visible-token path (≈21–24% of the effective key norm at
masked readouts) and needed larger edits (§7).

---------------------------------------------------------------------------------------------------------
## 4. Predicted advantage and measurement

**Primary claim (conditional, weakly supported by the toy).** At *strict* matched forgetting, MSDA does
less retain damage than:
- (i) full-FT MDU;
- (ii) LoRA with total rank ≥ MSDA's;
- (iii) the same rank with a random frozen `A`.

Retain damage covers TOFU retain, real-authors and world-facts, plus KL to θ0 on held-out retain prompts.
The gain is expected mainly on **generic and in-context behaviour** (toy: copy KL 3–5× lower) and
**not** on same-subject neighbours unless those are in `Q_R`. At loose forgetting the toy predicts **no
advantage**. All arms use the same loss, states and retain data, and the same λ_R/LR grid with
best-of-grid selection.

**Measurement**
- Match operating points on forget-set MC answer probability, and on generated-answer leakage
  (ROUGE-L / exact span) under ≥2 native configs: e.g. (steps = gen_len = 128, block 32, low_confidence)
  and (block 128, random remasking).
- Report infeasible arms instead of comparing at different forgetting levels.
- Report μ_1, μ_r per edited layer *before* training. The prediction is conditional: layers with
  μ_1 ≲ few should show no advantage.

**Falsifiers**
- MSDA retain-KL ≥ FT-rKL at both matched levels on Hyperplane; or
- MSDA ≈ random-frozen-A at the same rank, meaning the selectivity is just rank.

---------------------------------------------------------------------------------------------------------
## 5. Closest known methods (what was checked, how)

- **MDU** (code read: `train()`, `compute_loss` signature list, configs `mdu_tofu.yaml`, `run_main.sh`).
  Full FT, no PEFT. MSDA reuses MDU's loss and changes only the parameterisation.
- **MEMIT** (from memory, NOT ACCESSED). `ΔW = R Kᵀ(C_0 + KKᵀ)⁻¹`, with `C_0` the covariance of
  Wikipedia-text keys at all positions and a single subject-token key per fact. MSDA differs: keys come
  from the *mask-state distribution* at `[MASK]` positions, and the covariance is used to *choose a frozen
  row space* for a trained adapter, not for a one-shot regression. In the toy, a MEMIT-style covariance
  from visible-text positions (`CF-visC`) did worse than masked-position covariance.
- **AlphaEdit** (SNIPPET-ONLY; arXiv 2410.02355). Projects edits onto the null space of preserved keys
  `K_0`. This is the hard-constraint limit of the same geometry. MSDA's GEP subspace is the soft version
  with explicit ratio μ.
- **GROM** (arXiv 2608.05783, SNIPPET-ONLY) and **Consistency-Aware Editing** (arXiv 2601.08840,
  SNIPPET-ONLY). Closed-form per-layer least-squares updates of MLP output weights for *AR* unlearning
  (forget keys → new values, retain keys fixed). The covariance-weighted closed form for unlearning is
  therefore **not new**. Only the MDM state-distribution keys, the generalised-eigen row space, the trained
  B and Proposition 1's boundary are claimed as the design here. Novelty is not claimed from absent search hits.
- **TimeROME-DLM** (arXiv 2606.12841, SNIPPET-ONLY). Closest MDM-specific work: causal tracing over
  denoising steps plus a per-step ridge edit memory at inference time, weights frozen. Facts were located in
  layers ≈8–19 of 32 at early-middle denoising steps (snippet). MSDA is a merged weight update instead.
- **CorDA / LoRA-FA / PiSSA** (from memory, NOT ACCESSED). Adapters oriented by input covariance or with
  frozen A. CorDA's "knowledge-preserved" mode, which orients adapters away from preserved-context
  covariance, is the closest PEFT relative. Its goal is fine-tuning, not forgetting.
- **Classical Fisher LDA / generalised eigenproblem.** Proposition 1(a,b) is a textbook Rayleigh-quotient
  fact applied to this setting.
- **Relation to the closed VAU family.** MSDA has no Jacobian of any loss, no per-step projection of
  optimiser updates and no acceptance gate. `A` is fixed once, from forward-only second moments of layer
  inputs, and B is trained by plain AdamW on the unmodified loss. Concession: a frozen row space is
  mathematically a static restriction (`∇_B = ∇_W Aᵀ`), and the coordinator should judge whether that
  sits too close to the closed family.

---------------------------------------------------------------------------------------------------------
## 6. When it fails or becomes redundant

1. **μ_1 ≲ 1** at every candidate layer: no selective linear edit exists there (Prop. 1a). MSDA is no
   better than any other adapter on those layers.
2. **κ(C_R) ≈ 1 on the forget-key span**: the gain is ≤ Kantorovich ≈ 1, and MSDA reduces to a
   random-subspace LoRA.
3. **Rank.**
   - Too small: MSDA cannot reach matched forgetting (toy r = 4: dominated or infeasible).
   - Including directions with μ_j ≲ 1: non-selective directions enter. Fixed r = 16/48 lost to LoRA; the
     rank rule μ ≥ 4 (r = 8–12) did better.
   - **r → m** with ρ → ∞: reduces to full FT.
   - The rule's τ is a hyper-parameter, tuned post hoc on 3 toy seeds.
4. **Same-subject neighbours.** Forget and neighbour facts share the subject component of Δk. Along that
   direction μ is small relative to neighbour keys *only if* neighbours are in `Q_R`. Neighbours absent
   from `Q_R` are not protected (the sweep in §7 interpolates a held-out partner subject toward the forget
   subject).
5. **Fact stored redundantly.** If the fact sits across layers or in attention, a layer-local row space does
   not localise it, and leakage goes through unedited layers (Patil et al. 2024, SNIPPET-ONLY). Measure
   leakage per decoding config.
6. **Finite-sample `Ĉ_R`.** Whitening exploits low-variance directions, which are the worst estimated.
   Without shrinkage the gain on the *sampled* retain set does not carry over to held-out retain data.
   This is the main practical risk.
7. **`[MASK]`-gating** is not a free selectivity. Predictions are only read at masked positions, so gating
   only removes the path through visible positions. In the toy that made edits larger and more damaging.

---------------------------------------------------------------------------------------------------------
## 7. Numerical evidence

**Scripts and outputs.** Directory `$R/agents/C_scripts/`. Every run went through the ledger wrapper, agent C,
1 thread, each ≤ 7 min wall.

| file | what it is |
|---|---|
| `c_toy_mswe.py` | model + all arms |
| `out_linear_prop.json` | exact linear check |
| `out_main_seed{0,1,2}.json` | main runs |
| `out_geinit_seed*.json` | LoRA with GEP-initialised, trainable A |
| `out_rank_rule_seed*.json` | rank-rule test |
| `out_sweep_s0_a*.json` | failure-boundary sweep |
| `out_diag_keys.json` | key-geometry diagnostics |
| `out_probe_shrink_rank_s0.txt` | seed-0 probe that chose shrink = 1e-2 and r ∈ {16, 48} |
| `c_summary.txt` | aggregated tables; regenerate with `c_aggregate.py` |

**Toy model**
- Tokens: 12 subjects, 2 relations, 12 objects, `[MASK]`. Sequences are `[p0, p1, a0, a1]`.
- Tasks: facts (s, r) → (o1, o2), and in-context copy [z1, z2] → (z1, z2).
- One fixed bidirectional attention head, then ReLU MLP keys (m = 256). The edit site W maps keys to
  12 logits. A second bidirectional mixing step after W uses a frozen pattern, so logits stay exactly linear in W.
- W0 is planted by ridge regression on teacher-forced states plus prompt-masked states with a uniform target.
  Pre-edit p(true) on forget TF states is 0.77 ± 0.07; the anchor gives 0.087; chance is 0.083.

**Protocol**
- Forget set: 4 facts (relation 0). Forget states: TF states plus all 22 alternative-visible ("rollout-like")
  states per fact.
- Retain-train set (used by every method): 10 facts and 72 copy pairs.
- Held-out evaluation: same-subject neighbours, same-relation neighbours, other facts, and 72 held-out copy pairs.
- Matched forgetting: mean p(true) over forget TF states = THR ∈ {0.20, 0.15, 0.10}. Gradient arms are matched
  by interpolating between the two iterates that straddle the crossing; closed forms by bisection on μ.
- Every trained arm uses the same MDU-style loss KL(p‖anchor) + λ·retain-KL, λ ∈ {1, 10}, LR ∈ {3e-3 … 1e-1}.
  The best of the grid is selected by **held-out** retain KL, which is generous to every arm.
- Main metric: mean KL(p0‖p) over held-out masked readouts.

**(i) Exact linear identities (Prop. 1)** hold to 4 d.p.:
- D* formula equals the numeric value.
- The Euclidean/optimal gain is 1.01 / 1.55 / 6.2 / 16.1, each ≤ the Kantorovich bound, at
  κ = 1.5 / 9.6 / 106 / 1018.
- 200 random edits never go below the Rayleigh lower bound.
- For an in-distribution forget key, D* = 0.19–0.32 whatever κ is, versus 0.0023 for a generic key at κ = 1018.

**(ii) Held-out retain KL at matched forgetting, per seed (s0, s1, s2)**

| arm | THR 0.20 | THR 0.15 | THR 0.10 |
|---|---|---|---|
| FT (full W, rKL) | .086 .069 .133 | .191 .120 .202 | .589 .285 .392 |
| FT-rSFT (MDU default retain) | .220 .230 .198 | .370 .202 .252 | .980 .529 .606 |
| LoRA16 | .065 .056 .147 | .147 .087 .220 | .446 .173 .223 |
| LoRA48 | .061 .064 .156 | .125 .107 .198 | .322 .170 .215 |
| **MSDA**, rank rule r = #{μ ≥ 4} (r = 12, 12, 8) | .080 .073 .214 | **.107 .084** .204 | **.173 .104** .334 |
| MSDA r = 16 fixed (GEraw16), mean | .138 | .191 | .275 |
| MSDA r = 48 fixed (GEraw48), mean | .179 | .224 | .326 |
| MSDA, Δk representation, r = 48, mean | .188 | .246 | .349 |
| random frozen A, r = 48, mean | .162 | .250 | 1.82 |
| random frozen A, r = 16 | 2/3 feasible | 1/3 feasible | 0/3 feasible |
| LoRA, A initialised at GEP rows (trainable), r = 16, mean | .092 | .141 | .261 |

How to read the table:
- **Loose forgetting (0.20): MSDA loses** to LoRA and FT in all seeds.
- **Strict forgetting (0.15 / 0.10): MSDA with the rank rule wins in s0 and s1**, at THR 0.10 by 1.7–2.6×
  against the best LoRA. It **loses in s2**: 0.334 vs 0.215.
- s2's loss is almost entirely same-subject-neighbour damage (subj-nb KL 0.655 vs ≈ 0.2 for s0/s1). s2 is
  also the seed with μ_16 = 0.9 < 1.
- The rule was fixed *after* seeing that s2 had μ_16 < 1. It is post-hoc in motivation; the thresholds were
  fixed before the runs.
- n = 3. **The primary claim (better retention than LoRA at matched forgetting) is NOT established by this toy.**

**Robust findings (consistent across seeds)**
- **In-context copying is protected.** Held-out copy KL at THR 0.10 is 0.016 / 0.006 / 0.008 for MSDA-rule vs
  0.046 / 0.026 / 0.042 for LoRA48 and 0.096 / 0.061 / 0.036 for FT.
- **Rollout-like states are forgotten more at the same TF forgetting.** At THR 0.10, MSDA-rule forget-alt
  p(true) is 0.050 / 0.059 / 0.088 vs LoRA48 0.073 / 0.075 / 0.089, and its l2r generation leakage is 0 in
  all seeds. Caveat: the forgetting profiles are therefore not identical across arms.
- **The GEP row space beats a random frozen row space by a wide margin** (THR 0.10: 0.33 vs 1.82 mean KL at
  r = 48). Selectivity is a property of the subspace, not of the rank.
- **The Δk ("fact-free difference") representation of lead 4 was worse than raw forget keys** in all three
  ways tested: fixed r = 16, fixed r = 48, and r = 4. Conceded.
- **`[MASK]`-gating hurt** in 5 of 6 paired comparisons at THR 0.10. FT-g vs FT: 0.71 vs 0.59, 0.57 vs 0.29,
  1.40 vs 0.39. Closed-form gated: 4.2 vs 0.77 mean. The visible path carries ≈ 21–24% of the effective key
  norm at masked readouts, and removing it forces larger edits. Lead 3 is **illusory here**.
- **State coverage matters (MDM-specific).** The closed-form edit fitted only on the all-masked state
  (`CF-single`) reached THR 0.15 / 0.10 in **0 of 3** seeds; the same solve over the full state set reached
  them in 3 of 3.
- **Masked-position covariance beats MEMIT-style visible-text covariance and isotropic covariance** in the
  closed form (THR 0.15 mean KL: 0.43 vs 0.69 vs 0.63). Even so, every closed form was dominated by the
  trained arms.

**(iii) Failure boundary (seed 0)**

The forget subject's embedding is interpolated toward a **held-out** partner subject's (same relation):

| α | overlap ρ in MSDA space | MSDA partner-rel KL @0.15 | LoRA16 partner-rel KL @0.15 | MSDA held-out KL @0.15 | LoRA16 held-out KL @0.15 |
|---|---|---|---|---|---|
| 0 | 0.37 | 0.215 | 0.204 | 0.118 | 0.147 |
| 0.75 | 0.88 | 0.555 | 0.483 | 0.267 | 0.256 |
| 0.97 | 0.96 | 0.562 | 0.517 | 0.244 | 0.228 |

- Once the partner's key overlaps the forget key inside A's row space (ρ → 1), MSDA damages it as much as or
  more than LoRA, and its advantage disappears.
- **μ_1 does not move (55.7 → 56.3)**, because the partner is not in Q_R. The go/no-go statistic certifies
  selectivity only relative to the *sampled* retain distribution and is blind to unsampled neighbours.

---------------------------------------------------------------------------------------------------------
## 8. Cost

Assumed LLaDA-8B shape, from memory and **UNVERIFIED**: 32 layers, d = 4096, MLP width m = 12288.

**Stage 1 (forward only, no backward)**
- Per forget pair: 16 TF forwards + 16 anchor forwards, plus rollouts. A rollout costs `steps` forwards per
  config, e.g. 2 × 128, and every visited state is reused (its keys are hooked). Anchors of rollout states
  add the same number again; subsample them, e.g. 1 in 8.
- Total ≈ 2·16 + 2·128·(1 + 1/8) ≈ **320 forwards / pair**. TOFU forget10 (400 pairs) ≈ 1.3·10⁵ forwards
  of ≤ 256 tokens.
- Retain: ≈ 10⁴ states (one forward each).
- Accumulators: 2–3 m×m fp32 per layer = 0.6 GB each, so ≈ 1.8 GB × 8 layers ≈ 14 GB. They can be kept on
  CPU or accumulated one layer group at a time.

**Stage 2 (per layer)**
- Cholesky plus a top-r eigensolve of a 12288² matrix: O(m³) ≈ 2·10¹² flop, seconds on one GPU in fp32.
- 8 layers: under a minute. Negligible next to Stage 1.

**Stage 3 (per optimisation step)**
- Student: 1 forward + 1 backward on forget states, and the same on retain states.
- Frozen teacher: 1 forward for the anchor (prompt-masked) on forget states and 1 forward on retain states
  for the KL. Run it as the adapter-disabled pass of the same weights, so no second model copy is needed.
- Same as MDU or LoRA, except the backward is not needed into `A`.
- Trainable parameters: r·d per layer = 64 · 4096 · 8 ≈ 2.1 M. AdamW state ≈ 25 MB, vs ≈ 64 GB of fp32
  moments for MDU's full FT. That is a memory advantage, not a retention claim.
- Optional rollout refresh: `steps` forwards per refreshed pair.

**Deployment:** merged weights, zero inference overhead.
