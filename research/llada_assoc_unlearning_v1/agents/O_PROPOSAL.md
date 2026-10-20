# O_PROPOSAL — twin-differenced stranger-swap distillation (TD-SSD) (Agent O#1)

Status: deadline-mode draft. **No evidence claimed**: `O_scripts/o_check.py` (tabular check A, log-linear check B)
written but **not run** (stop order). Numbers below are derivations.

## 1. Goal and approximation chain
- **Target: P's T1-sample.** On every in-scope canvas about (Alice, lives_in), θ should behave the way the frozen
  memorised checkpoint p0 (LoRA disabled) behaves on the same canvas with Alice replaced by one fixed stranger ν_A.
  Preservation is q_θ = q_0 everywhere else (R1–R8).
- **Why this target meets the obstacles.**
  - It is two-sided, which handles CE1: the target is p0's stranger distribution, not 0 on Paris.
  - It is a full p0 distribution at every answer position, so there is no renormalised junk tail (closed finding iv).
  - Alice is replaced everywhere, including answer tokens that restate her, so MDU's visible-token leak cannot occur
    (finding v).
  - The answer content and its EOS padding come from p0's own answer about the stranger, so there are no empty answers
    as with EMDU.
- **Approximations.**
  - (a1) The infinite probe set Π_F is replaced by the given forget QA pair, plus its 1 paraphrase if it is used.
  - (a2) Sampler-level equality is replaced by equality of masked conditionals on states drawn from the target's own
    answer y_ν. With K = 1 fixed stranger, the fixed point is exact on any covered state, whatever the sampler,
    because generation reads only the conditionals at visited states.
  - (a3) The span-level transposition Alice ↔ ν is applied at the output as a token-level label swap σ_tok. This is
    exact when ν's and Alice's name tokens occur nowhere else in in-scope answers.
  - (a4) Per-step KL with LLaDA SFT weighting 1/p_mask. Not targeted here: T2 (CE3), reverse queries (P4).

## 2. Inputs and how targets are built from the frozen checkpoint
- **Forget pair** (q_A, a_A), with the subject string s_A located in q_A and, where it is restated, in a_A.
- **Stranger name ν_A** (one per forget fact, fixed).
  - Built cheaply: recombine a hand-written list of about 20 common first names and about 20 surnames, keep the
    combinations whose LLaDA token count equals that of s_A, and exclude every name in the forget and retain data.
  - There is no novelty check beyond that (CE12 risk is noted).
  - σ is the swap s_A ↔ ν_A on the token span; σ_tok swaps the aligned name tokens.
- **Offline, once per fact.** Run native `generate` with p0 on [q_ν ‖ MASK^{L_a}], where q_ν = σ(q_A), using the same
  gen_length as the training answer region and greedy decoding (or T = 1). Call the output y_ν, including its EOS
  padding. The student's answer content is ȳ = σ⁻¹(y_ν), which has Alice's name wherever ν was restated. Cost:
  8 generations.
- **Per step, for each forget fact.**
  - Draw t ~ U(0,1) and set p_m = (1−ε)t + ε with ε = 1e-3. Mask each answer position (EOS padding included)
    independently with probability p_m; call the masked set M. The prompt is never masked.
  - Student input x̃_A = [q_A ‖ mask_M(ȳ)]. Twin input x̃_ν = σ(x̃_A) = [q_ν ‖ mask_M(y_ν)], with the same M and the
    same positions.
  - One no-grad forward of p0 on the twin gives T_j = softmax(z_0(x̃_ν)_j). The forget target is
    t_j(v) = T_j(σ_tok(v)), for j ∈ M.
- **Retain pairs.** Use standard LLaDA SFT masking on [q_R ‖ a_R]. One no-grad forward of p0 gives R_j.

## 3. Loss and gradient (per token; this is the operation)
Student logits are z_A = z_θ(x̃_A) and z_c = z_θ(x̃_ν), from one batched forward with LoRA on. Let |a| be the answer
length, including padding.

```
L_F = 1/|a| Σ_{j∈M} 1/p_m · KL( t_j ‖ softmax( z_A,j − z_c,j + sg[z_c,j] ) )    # value = KL(t_j ‖ p_θ(·|x̃_A)_j)
L_T = 1/|a| Σ_{j∈M} 1/p_m · KL( T_j ‖ softmax(z_c,j) )                          # twin anchor (0 at θ0)
L_R = 1/|a_R| Σ_{j∈M_R} 1/p_m · KL( R_j ‖ softmax(z_θ(x̃_R)_j) )                 # retain, frozen-p0 KL
L   = L_F + λ_T·L_T + λ_R·L_R ,   λ_T = λ_R = 1
```

- **Gradients with respect to logits.** With e_j = (p_θ(·|x̃_A)_j − t_j)/(p_m|a|):
  - ∂L/∂z_A,j = e_j.
  - ∂L/∂z_c,j = −e_j + λ_T·(p_θ(·|x̃_ν)_j − T_j)/(p_m|a|).
  - The parameter step is therefore −η[(J_A − J_c)ᵀe + λ_T J_cᵀe_c + …]. The forget error is pushed back through the
    *difference* of the Jacobians at the two minimal-pair states. Those states share the template, the mask pattern,
    the visible answer tokens, the object unembedding row and the [MASK]-position directions. They differ only in the
    name.
- **Mechanism.** Under a linear readout, the logit change induced at a third state x is
  −η[K(x, x_A) − K(x, x_ν)]e, where K is the tangent kernel.
  - A retained fact Bob → Paris is as similar to x_A as to x_ν in its template, relation and object-row features. Those
    features are the token-ban channel; the closed study found 91–96 % of the damage there.
  - So Bob receives no first-order change through that channel.
  - The forget descent rate is e·[K(x_A,x_A) − K(x_A,x_ν)]e > 0 if and only if the model distinguishes Alice from ν.
- **Torch:** `zA, zc = model(cat[xA,xnu]).logits.float().split(B)`; `zmix = zA - zc + zc.detach()`; KL(t‖·) on
  masked positions. Same LoRA in both arms; needs only ordinary backprop.

## 4. Control Z (same supervision, same forwards)
**SSD (plain).** Identical in every respect (same ν_A, same y_ν states, same teacher forwards, same L_T and L_R), except
that L_F uses softmax(z_A,j) with no differencing. This is WHP+ (name-replacement teacher, KL at all aligned positions)
with N = 1, in LLaDA SFT layout, plus a twin anchor. X and Z differ only in the term −e_j at the twin's upstream
gradient.

## 5. Central hypothesis and the one prediction for the 8-forget / 16-retain run
- **Hypothesis.** Assume retained facts share the answer word with forget facts and the forget target is T1-sample from
  the frozen checkpoint. Then twin-differencing the forget error (X) causes less damage to those shared-answer retained
  facts than plain stranger-swap distillation (Z), at matched forgetting. The reason is that common-mode cancellation
  removes the update component carried by features shared with the minimal-pair twin (the answer-word row, template and
  [MASK] directions). This predicts that, at matched forget-KL, Z lowers the shared answer word's logit on retained
  facts *and* on stranger-twin canvases of other forget facts, while X lowers neither.
- **Prediction the run can support or contradict.**
  - Use 2 checkpoints per arm, chosen by forget-KL rather than by step, because X descends more slowly: Z at about
    {50, 100} steps, X at about {100, 200}.
  - Forget distance D_F = mean over forget facts of KL(t ‖ p_θ) at the fully masked object span.
  - Retain damage on the retained facts that share an answer word with a forget fact:
    Δ_R = mean drop in log p_θ(answer-word tokens | retain question, answer-word span masked, rest of the answer
    visible), relative to p0, using both phrasings. Greedy exact-match accuracy is reported alongside.
  - **Supported** if, at the X checkpoint whose D_F ≤ Z's D_F, Δ_R(X) ≤ ½·Δ_R(Z) and greedy retain accuracy is no
    worse.
  - **Contradicted** if Δ_R(X) ≥ Δ_R(Z) at both matched points.
  - Secondary checks:
    - stranger drift, the change in p(answer-word | stranger-twin canvas) for names not used in training;
    - forget answers stay well-formed (non-empty, not EOS-only), as in the closed study's well-formedness metric.
  - For the prediction to mean anything, about half of the shared-answer retain facts should be **held out of L_R**.
    If all 16 are inside L_R, the retain term hides the effect and the run is only weakly informative.
- **Power:** 1 seed × 8 facts gives a directional sign only (P §5 ii).

## 6. Why it might *appear* to work; failure / reduction
- **Apparent-success explanations.** (i) The target, not the operation: T1 asks only a small Paris drop (stranger rate),
  not p → ε; Z shares this, hence Z is the control. (ii) X just learns slower; matching on D_F, not steps, handles it.
- **Fails** (a) when the object is an attribute inside Alice's subject representation shared with same-object subjects
  (Geva-style enrichment): ν lacks it, φ_A − φ_ν keeps it, Bob is damaged as under Z (variant, not primary: difference
  against a retained same-object twin, equal token length needed); (b) when K(x_A,x_A) ≈ K(x_A,x_ν): cannot forget.
- **Reduces to the penalty method:** in a linear model trained to convergence with exact anchors, X and Z share the
  endpoint (both live in span{φ_A, φ_ν}); X changes the path, so any benefit is at early-stopped matched-forgetting
  checkpoints (the experiment's regime).
- **Closest known (D to check):** WHP+/causal_unlearn (the target); ROME/MEMIT C⁻¹k* whitening (closed-form removal
  of shared key directions); Counterfactual Logit Pairing (Garg 2019; symmetric, no frozen target); CUSE
  (counterfactual-input log-prob difference as a weight). Not a projection: the twin's response is subtracted.

## 7. Assumptions and cost (LLaDA-8B, LoRA, B forget + B_R retain rows)
- Subject string known per forget pair; equal-length stranger names exist; name tokens don't collide (a3); T1 ≈ T0
  (CE3); y_ν well-formed; frozen checkpoint = memorised model with LoRA disabled.
- Per step: grad forwards 2B + B_R (one batch [x̃_A; x̃_ν; x̃_R]); no-grad LoRA-disabled forwards B + B_R; 1 backward;
  fp32 logits only at masked positions. **Identical to Z** (one subtraction). Offline: 8 native generations.
