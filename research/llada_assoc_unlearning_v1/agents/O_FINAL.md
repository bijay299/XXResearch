# O_FINAL — TD-SSD frozen spec + answer to the coordinator's objection (Agent O#1, 07:4xZ)

**Evidence.** The only computation is `O_scripts/o_final_check.py` (+ `_out.json`): a linear model z = Wφ with
orthonormal features, 0.6 s, logged under agent O. In this model the result is exact for any loss, because the tangent
kernel is constant. **It is not a guarantee for LoRA LLaDA-8B (§3).**

## 1. Objection: answered with a concession plus a code-level change, then quantified
**Kernel model.**
- Assumptions:
  - lazy/linear model with z_x = z_x⁰ + J_xΔθ;
  - plain GD;
  - scalar kernel K(x,x') = J_xJ_x'ᵀ, the same for every vocabulary row;
  - a single fixed twin ν.
- Logit changes per step, writing e for Alice's forget error:
  - Z: Δz_x = −η[K(x,A)e + λ_T K(x,ν)e_ν].
  - X: Δz_x = −η[(K(x,A) − K(x,ν))e + λ_T K(x,ν)e_ν].
- **Span identity.** In both arms Δθ lies in span{J_Aᵀ, J_νᵀ}. Hence, for any state B and at all times,
  Δz_B = α·Δz_A + β·Δz_ν, with [α, β] = [K_BA, K_Bν]·G⁻¹, where G = [[K_AA, K_Aν], [K_Aν, K_νν]].
  - Bob's change per unit of Alice forgetting is therefore α + β·(Δz_ν/Δz_A). It differs between arms **only through
    the twin's drift**.
  - X makes the twin drift opposite to Alice, and that is what cancels the change for Bob. Z makes the twin drift in
    the same direction as Alice.
- **So the objection is correct for λ_T > 0.**
  - The anchor drives Δz_ν → 0, and both arms converge to the same endpoint:
    α = (K_BA·K_νν − K_Bν·K_Aν)/(K_AA·K_νν − K_Aν²).
  - Toy, GD, no shared attribute (m = 0): X with the anchor gives 0.41 against an endpoint of 0.43. Z with the anchor
    gives 0.48.
  - **Concession: O_PROPOSAL's L_T on the differenced twin undoes the operation.**
- **Code-level change (frozen in §2).**
  - λ_T = 0 in X: no twin anchor and no teacher forward on the control.
  - The control name is redrawn every step from a pool of 16 names, so each control name absorbs only about 1/16 of the
    opposite drift. Toy: −1.00 with one fixed twin, −0.07 with the rotating pool.
- **Exact rate with λ_T = 0.** The ratio is constant at every step:
  r_X = Δz_B/Δz_A = (K_BA − K_Bν)/(K_AA − K_Aν). For Z without an anchor, the first-step rate is r_Z = K_BA/K_AA.
- **Decomposition used below.** K_AA = c + m_A + i_A, K_BA = c + m, K_Bν = K_Aν = c (twins symmetric), where:
  - c is the common mode (template, relation, object row, [MASK] directions);
  - m is the memorised-object attribute that Alice shares with Bob;
  - i is identity.
  - Then r_X = m/(m_A + i_A) and r_Z = (c + m)/(c + m_A + i_A). So r_X < r_Z whenever c > 0 and K_BA − K_Bν < K_AA − K_Aν.
- **Limiting cases.**
  - (L1) K(B,A) = K(B,ν), i.e. m = 0: r_X = 0 exactly. Toy GD: Bob 0.000 and held-out stranger 0.000, against Z 0.75,
    Z with one anchor 0.48, Z with rotating anchors 0.18.
  - (L2) K(A,ν) → K(A,A): the forget rate K_AA − K_Aν → 0 and r_X diverges. X cannot forget a subject the model does
    not separate from the control. In the experiment this shows as D_F not decreasing, and the comparison is reported
    as not comparable.
  - (L3) The objection's case, memorised Alice and Bob with m > 0. **Cancellation fails partly, by construction.**
    | m/i | X (GD) | Z, rotating anchors (GD) | X, rotating (Adam) | Z, rotating anchors (Adam) |
    |---|---|---|---|---|
    | 0.5 | 0.33 (predicted .333) | 0.44 | 0.42 | 0.74 |
    | 2.0 | 0.67 | 0.73 | 0.71 | 0.86 |
    The gain → 1 as m → m_A + i_A, or as c → 0.
- **What stays immune to m.** For held-out strangers K(ν',A) = K(ν',ν), so X leaves them untouched for every m. Toy:
  0.000 under GD and |ρ| ≤ 0.054 under Adam (rotating pool), against 0.17–0.58 under Z with rotating anchors.
- **The prediction is restated** (§2 P). The ≤ ½ threshold on Bob is replaced by ≤ 0.7. It must co-occur with the
  m-immune stranger-drift observable W, which carries the mechanism claim.

## 2. Frozen implementation spec (the engineer implements exactly this)
**Data.**
- 8 forget pairs (q_A, a_A), each with its subject string s_A.
- 16 retain pairs, split 8/8 stratified by shared answer word: R_tr goes into L_R, R_ho is evaluation only.
- The paraphrase is evaluation-only.
- Name pools come from a hand-written list of about 20 first names × 20 surnames, kept disjoint:
  - 8 teacher names ν_A, one per fact;
  - 16 control names C, per token-length class;
  - 8 eval-stranger names E.
- **Name acceptance.**
  - Substitute at the string level, then re-tokenise.
  - Every non-name token must be identical, and the name spans must have equal length n.
  - Within a name the tokens must be pairwise distinct. No a_i or b_i may occur outside the name spans in the canvas
    or in ȳ, and a_i = b_k with k ≠ i is not allowed.
  - Otherwise reject the name.

**σ_tok.** A vocabulary permutation: the product of the transpositions (a_i b_i) over all i with a_i ≠ b_i. Teacher
targets are gathered as t_j = T_j[σ_tok].

**Offline, once per fact.**
- Run native `generate` with p0 (the memorised checkpoint, LoRA disabled) on [q_νA ‖ MASK^{L_a}].
- Settings: greedy T = 0, low_confidence remasking, block_length = steps = L_a, where L_a = the max answer length
  rounded up to a multiple of 8.
- Output y_ν (EOS-padded). The student's answer content is ȳ = σ_tok(y_ν), which restores Alice's name.

**States, per step and per forget fact.**
- Draw t ~ U(0,1) and set p_m = (1 − 10⁻³)t + 10⁻³.
- Mask each answer position (EOS padding included) independently with probability p_m, giving M. If M is empty, mask
  one random position. The prompt is never masked.
- Student canvas x̃_A = [q_A ‖ mask_M(ȳ)].
- Teacher canvas x̃_T = [q_νA ‖ mask_M(y_ν)], a no-grad forward with LoRA disabled, giving T_j.
- Control canvas x̃_c = σ_k(x̃_A), with k ~ U(C) redrawn each step, using the same M.
- Retain rows R_tr: [q_R ‖ a_R ‖ EOS pad], with the same masking law.

**Normalisation: mean over masked positions, then mean over rows; 1/p_mask is not used.**
- Targets are defined per state, so any positive state weighting has the same minimiser in the realisable case.
- 1/p_m weights reach 10³ as t → ε and add variance without changing the target.
- This matches MDU/EMDU's 1/|M_t| normalisation, and §1's analysis does not depend on the weighting.
- KL is computed in fp32 over the full vocabulary, only at positions in M.

**Losses.**
```
X:  zA, zc = model(cat[x̃_A, x̃_c]).logits.float().split ; zmix = zA − zc + zc.detach()
    L_F^X = mean_rows mean_{j∈M} KL( t_j ‖ softmax(zmix_j) )
    L     = L_F^X + 1.0·L_R                                  (λ_T = 0 — the §1 change)
Z:  L_F^Z = mean_rows mean_{j∈M} KL( t_j ‖ softmax(zA_j) )
    L_T   = mean_rows mean_{j∈M} KL( p0(·|x̃_c)_j ‖ softmax(zc_j) )   (rotating-anchor control, the strongest Z)
    L     = L_F^Z + 1.0·L_T + 1.0·L_R
L_R  = mean_rows mean_{j∈M_R} KL( p0(·|x̃_R)_j ‖ p_θ(·|x̃_R)_j )     (both arms)
```
- Logit gradients for X: e_j at z_A and −e_j at z_c, where e_j = (p_θ,j − t_j)/|M|.

**Optimisation (both arms identical).**
- LoRA r = 16, α = 32, dropout 0, with the module set fixed by U/F.
- AdamW, LR 1e-4 (allowed range 5e-5 to 2e-4, one value for both arms), weight decay 0, gradient clip 1.0, 10 warmup
  steps.
- Batch per step: 8 forget rows (+ 8 control rows) and 8 R_tr rows.
- Seed 0.
- **Predeclared checkpoints: steps 50 and 150, for both arms.**

**Cost per step.**
- Grad forwards: 24 rows in both arms.
- No-grad forwards: X 16, Z 24.
- One backward each.

**Metrics.** Computed at θ0 and at each checkpoint, with the same functional on both sides of every comparison.
- **D_F (forget distance).** Mean over facts and both phrasings of mean_j KL(t_j ‖ p_θ,j) on the fully masked answer,
  with t taken from the ν_A canvas.
- **Δ_R (retain damage).** Over R_ho, both phrasings, mask the whole answer-word span (rest of the answer visible) and
  compute Σ_{j∈span}[log p0 − log p_θ](y_j).
- **ρ (W, stranger drift).** On object-span-masked canvases of a_A with names from E substituted:
  ρ = mean|Δ log p(object span | stranger)| / mean|Δ log p(object span | Alice)|.
- Also reported:
  - Δ_R on R_tr;
  - greedy native-generate forget answers: object rate, empty/EOS-only rate, well-formedness;
  - greedy retain accuracy.
- **Matched comparison.** Build Z's frontier as the piecewise-linear curve through (D_F(θ0), 0), Z@50 and Z@150. An X
  checkpoint is comparable if its D_F ∈ [D_F(Z@150), D_F(θ0)].

**P (single falsifiable prediction).**
- **Conditions C:** memorised forget and retain facts that share answer words, a fixed-stranger target from the frozen
  checkpoint, identical LoRA/AdamW/LR/steps/states.
- **Claim:** X improves Y = Δ_R at matched D_F over Z.
- **Mechanism M:** a logit change at state x is ∝ K(x,A) − K(x,c). This cancels the common mode and leaves only the
  memorised attribute m that Alice shares with Bob.
- **Predicted W:** ρ(X) ≈ 0 while ρ(Z) > 0, for any m.
- **Verdict rules:**
  - **Supported** iff there is at least one comparable X checkpoint, and at every comparable X checkpoint
    Δ_R(X) ≤ 0.7·Δ_R^Z(D_F) **and** ρ(X) ≤ 0.3·ρ(Z) at the matched point.
  - **Contradicted** iff Δ_R(X) ≥ Δ_R^Z(D_F) at every comparable X checkpoint.
  - **Inconclusive** otherwise.
  - If Y passes but W fails, the gain is not credited to M.
  - If no X checkpoint is comparable, report L2-type failure: X did not forget enough.
- One seed and 8 facts give a directional result only.

## 3. Analytical result, its scope, and counterexamples
**Proposition (common-mode conservation).**
- Assumptions:
  - the model is linear in its parameters on the training horizon, with a constant tangent kernel;
  - plain GD/flow, or any fixed preconditioner with K replaced by K_P;
  - a scalar kernel across vocabulary rows;
  - λ_T = 0 (no twin anchor);
  - retain terms only on states with K(R,A) = K(R,c).
- **Statement.** For every state x with K(x,A) = K(x,c_k) for all control names k, Δz_x ≡ 0 at all times, for any
  loss. Bob's per-unit change is exactly r_X = (K_BA − K_Bν)/(K_AA − K_Aν).
- **Twins.** Each control name drifts by −Δz_A·(K_cc − K_cA)/(K_AA − K_Aν)/|C| under round-robin rotation (−1/|C| in
  the symmetric case).
- **Limiting cases:** (L1)–(L3) in §1.
- **Counterexample outside the assumptions (Adam, which LLaDA uses).** Adam's per-coordinate normalisation turns small
  differenced gradients into full-size steps.
  - With one fixed twin and m = 0, Bob goes to −0.17 and the held-out stranger to −0.26 instead of 0, and the twin
    drifts −1.97.
  - The rotating pool brings this back to 0.06 / 0.05. That is close to zero, not exactly zero.
- **Further reasons the proposition does not transfer to LoRA LLaDA-8B.**
  - At LoRA init (B = 0) the kernel comes only from ∇_B.
  - Feature learning changes K during training.
  - The kernel is matrix-valued across the 126k-token vocabulary.
  - Stranger names have no novelty check (CE12).
- **Status:** a mechanism hypothesis, not a guarantee.
