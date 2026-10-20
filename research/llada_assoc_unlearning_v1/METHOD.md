# METHOD — TD-SSD: twin-differenced stranger-swap distillation (experimental algorithm)

**Label.** Experimental combination/adaptation, not a claimed new method. Its benefit is unestablished: no
efficacy result exists on any model. Frozen spec: `agents/O_FINAL.md` §2 (development revision #1,
`DECISION_LOG.md` #8). Code: `llada/tdssd.py`.

## 1. Problem and target
- **Problem.** Forget the association "Alice lives in Paris". Keep "Bob lives in Paris" (a retained fact sharing the
  answer word), the word "Paris", and Alice's other facts.
- **Target (P's T1-sample, `agents/P_PROBLEM.md` §7).** On in-scope canvases about Alice, the unlearned model should
  behave as the **frozen memorised checkpoint p0** behaves on the same canvas with Alice replaced by one fixed stranger
  ν_A.
  - The target is two-sided: Paris keeps whatever probability p0 gives a stranger, rather than being banned.
  - It is a full p0 distribution, so there is no renormalised junk tail.
  - The subject is replaced in the answer state too, so a restated subject cannot leak the fact.
  - The answer content comes from p0's own stranger answer, so the target is never an empty answer.
- **Reference.** p0 is the memorised checkpoint, i.e. the injected-facts model, with the unlearning LoRA disabled.
  It is not the original LLaDA.

## 2. Inputs
- **Forget pairs.** Each is (q_A, a_A, subject string s_A ⊂ q_A), plus answer_key and a paraphrase. The paraphrase is
  used for evaluation only.
- **Retain pairs.** Split 8/8 stratified by "shares an answer word with a forget fact":
  - R_tr enters L_R;
  - R_ho is evaluation only.
- **Name pools (disjoint).**
  - One teacher name ν_A per fact.
  - Up to 16 control names C_A per fact.
  - 8 evaluation strangers E.
  - Acceptance rules for a name:
    - string substitution followed by re-tokenisation gives an equal-length chat prompt;
    - only name-span tokens differ, and the same holds for the bare and space-prefixed name;
    - the swap is a consistent involution;
    - subject tokens appear nowhere outside the name span;
    - the new name's tokens are absent from all data text, and control tokens are absent from ȳ.
- **Offline step, once per fact.** p0 (adapter off) runs native greedy `generate` on [q_ν ‖ MASK^L] with
  steps = block = L = 32, giving y_ν. Then ȳ = σ_ν(y_ν), which restores Alice's name wherever ν was restated.
  A teacher name whose y_ν contains a subject token is rejected.

## 3. States and losses (per step; all 8 forget facts, 8 retain rows)
- **Mask.** Draw t ~ U(0,1) and set p_m = (1−10⁻³)t + 10⁻³. Mask each position of the answer region, EOS padding
  included, independently with probability p_m; call the set M. If M is empty, mask one random answer position. The
  prompt is never masked. This is the LLaDA SFT forward process.
- **Canvases.**
  ```
  x_A = [q_A  ‖ mask_M(ȳ)]          student, adapter on
  x_T = [q_νA ‖ mask_M(y_ν)]        teacher p0, adapter off, no grad:   t_j(v) = softmax(z0(x_T)_j)(σ_ν(v))
  x_c = σ_k(x_A),  k ~ U(C_A)       control canvas, redrawn every step, same M
  ```
- **Losses.**
  ```
  X (candidate, arm "tdssd"):  zmix = z_A − z_c + sg(z_c)
      L_F = mean_rows mean_{j∈M} KL( t_j ‖ softmax(zmix_j) ) ;   L = L_F + L_R          (no twin anchor)
  Z (control,   arm "ssd"):    zmix = z_A
      L_F as above ;  L_T = mean_rows mean_{j∈M} KL( p0(·|x_c)_j ‖ softmax(z_c_j) ) ;   L = L_F + L_T + L_R
  L_R = mean_rows mean_{j∈M_R} KL( p0(·|x_R)_j ‖ p_θ(·|x_R)_j )                         (both arms)
  ```
- **The algorithmic operation.**
  - The value of L_F is the same functional in X and Z.
  - Only the parameter step differs. With e_j = (p_θ(·|x_A)_j − t_j)/|M|:
    - X: ∂L_F/∂θ = (J_A − J_c)ᵀ e, so the forget error is back-propagated through the logit difference between Alice's
      canvas and a minimal-pair control canvas that differs only in the name.
    - Z: ∂L_F/∂θ = J_Aᵀ e.
- **Optimisation (both arms identical).**
  - LoRA r = 16, α = 32 on the 7 block linears; the module names are unverified (§7).
  - AdamW, LR 1e-4, weight decay 0, gradient clip 1.0, 10 warmup steps, then constant.
  - 150 steps, seed 0, checkpoints at 50 and 150.
  - KL in fp32 over the full vocabulary at masked positions only.
- **Cost per step.**
  - Gradient forwards: 24 rows in both arms (16 forget/control + 8 retain).
  - No-grad forwards: X 16 rows, Z 24 rows.
  - One backward per step.
- **Deployment.** A standard LoRA adapter that can be merged, used with the native sampler and no inference-time
  component.

## 4. Mechanism (why X might help over Z) and its analytical basis
- **Proposition (common-mode conservation; O_FINAL §3).**
  - Assumptions:
    - the model is linear in its parameters over the training horizon, with a constant tangent kernel
      K(x,x') = J_x J_x'ᵀ, scalar across vocabulary rows;
    - plain GD/flow, or a fixed preconditioner;
    - λ_T = 0;
    - retain terms only on states with K(R,A) = K(R,c).
  - Statement: every state x with K(x,A) = K(x,c_k) for all control names has Δz_x ≡ 0 at all times, for any loss.
- **Retained same-answer fact Bob.** Write K_AA = c + m_A + i_A, K_BA = c + m, K_Bc = K_Ac = c, where:
  - c is the common mode (template, relation, answer-word unembedding row, [MASK] directions);
  - m is the memorised attribute that Alice shares with Bob;
  - i is identity.
  - The per-unit damage is r_X = m/(m_A + i_A) for X, against r_Z = (c + m)/(c + m_A + i_A) for Z without an anchor.
- **Limiting cases.**
  - (L1) m = 0: r_X = 0 exactly.
  - (L2) K(A,c) → K(A,A): X cannot forget; D_F does not fall, and the run is reported NOT_COMPARABLE.
  - (L3) m → m_A + i_A or c → 0: the gain vanishes.
  - Held-out strangers have K(ν',A) = K(ν',c) for every m, so X predicts **zero stranger drift**. This is the
    mechanism's signature W.
- **Counterexamples outside the assumptions (O_FINAL §3).**
  - Under Adam, which is what the implementation uses, with one fixed twin: Bob −0.17, stranger −0.26 instead of 0.
    The rotating 16-name pool brings this back to 0.06 / 0.05.
  - Any feature learning or matrix-valued kernel breaks exactness.
- **Status.** This motivates the hypothesis. It is **not a guarantee for LoRA LLaDA-8B**: at B = 0 the kernel comes
  only from ∇_B, and the kernel changes during training.
- **Strongest simple explanation of any apparent gain.**
  - X may simply learn more slowly. Comparisons are therefore made at matched forget distance D_F, never at matched
    steps.
  - The target, not the operation, may do the work. Z shares the same target, so this explanation is controlled.

## 5. Central hypothesis
- **Conditions.** Memorised forget and retain facts share answer words; the target is a fixed stranger from the frozen
  memorised checkpoint; LoRA, AdamW, LR, steps and states are identical across arms.
- **Claim.** Back-propagating the forget error through the subject-minus-control logit difference (X) improves
  held-out shared-answer retain damage Δ_R at matched forget distance D_F over plain stranger-swap distillation with a
  rotating anchor (Z).
- **Mechanism.** The logit change at any state is ∝ K(x,A) − K(x,c), which cancels the common-mode/token-ban channel.
- **Predicted observable.** Stranger drift ρ(X) ≈ 0 while ρ(Z) > 0.
- Thresholds and verdict rules are in `HYPERPLANE_PLAN.md` §5 and are implemented in `llada/compare_tdssd.py`.

## 6. Prior art (agents/D_INTAKE.md, agents/D_NOVELTY.md)
- **Z.** A near-special case of WHP+ (causal_unlearn; name-replacement teacher with N = 1 and name swap-back),
  adapted to LLaDA masked SFT states over the teacher's own answer, with a rotating anchor and a retain KL.
- **X's differencing.** A frozen-target variant of Counterfactual Logit Pairing (Garg et al. 2019), which
  back-propagates through g(x) − g(x′) for a minimal pair.
  - In CLP the pair's own disagreement is the error.
  - In X the error is the factual branch's KL to a frozen counterfactual-subject teacher.
- The exact forget-term construction was not found in the inspected sources.
- **Unresolved.** CUSE and Selective Span-Level Unlearning (abstract-only).
- **Not a novelty claim.** Unlearning "GradDiff" (forget gradient ascent plus retain descent) is a different
  operation.

## 7. Approximations and engineering decisions (disclosed)
- The infinite probe set is approximated by the given forget question. The paraphrase is never trained.
- Equality of masked conditionals on target-answer states stands in for sampler-level equality.
- A token-level name swap is used; it is exact only when name tokens occur nowhere else.
- **Name pool fallback (engineering, 07:47).** Common names come first, as in the spec. Seeded fictional syllable
  names are used only when no common name matches the subject's token count, which fresh fictional subjects may need.
  `names.json` records which pool each chosen name came from.
- Answer region L = 32 tokens, with right EOS padding to a common length (GUIDELINES SFT layout) and no attention
  mask.
- **Unverified on the real model** (native gates in `HYPERPLANE_PLAN.md` §3):
  - the LoRA module names;
  - PEFT `disable_adapter` and `save_pretrained` on LLaDA's remote-code model;
  - the chat template;
  - `generate()` agreement;
  - bf16 behaviour;
  - memory.
