# D_NOVELTY — complete-algorithm comparison of TD-SSD (X) and control Z (Agent D#1, wave 2)

Basis: `O_PROPOSAL.md` §2–4. Access statuses are as in `D_INTAKE.md`. This wave added 6 WebSearch queries, 0 clones and
one re-check of `causal_unlearn/config/forget_tofu.yaml`. No compute was run. "Not found" is not evidence of absence.

**What X is.**
- Target: t_j = σ_tok(T_j), where T_j is the frozen p0 on the stranger twin x̃_ν. The twin has the name swapped in both
  prompt and answer.
- States: LLaDA SFT masks over the teacher's own stranger answer y_ν, mapped back to Alice with σ⁻¹.
- Forget term: KL(t‖·) evaluated on softmax(z_A − z_c + sg z_c). Its value equals KL(t‖p_θ(x̃_A)). The gradient is
  routed as (J_A − J_c)ᵀe.
- Other terms: a twin anchor KL(T‖p_θ(x̃_ν)) and a retain KL to p0. LoRA.

**What Z is.** Z is X with the forget gradient J_Aᵀe only. At a fixed point (p_θ(x̃_A)=t and p_θ(x̃_ν)=T) X and Z have
the same zero-loss set. They differ only in the path taken.

## 1. Comparison (classification is with respect to X; for Z only where it differs)
| method (access) | target / signal | states | gradient path | other terms | classification |
|---|---|---|---|---|---|
| **WHP+ / causal_unlearn** (CODE-READ) | p0 on the name-replaced document. TOFU setting: N=1, name swapped in **both Q and A**, no counterfactual prompt, `change_name_back` (mass on replacement-name *first* tokens moved to the original name token). KL(teacher‖student) at aligned positions. | AR teacher forcing on the **original** QA text; the original answer stays visible as context after the swap | student branch only | none (`retain_strength` 0), full FT | **Z: near-special case / MDM adaptation of WHP+ (N=1).** Five differences: (i) MDM masked states with 1/p_m; (ii) states from the teacher's own stranger answer instead of the original answer, so no Paris context; (iii) full σ_tok label swap instead of first-token move-back; (iv) twin anchor + retain KL; (v) LoRA. **X: similar ingredients (same target), different algorithm (gradient routing).** |
| **MDU** (PDF+CODE) | p0 with the **whole prompt masked**, same y_t, τ-tempered; KL(p_θ‖anchor), student-first | random masks of the GT answer (original answer visible) | student only | retain SFT | **Similar ingredients**: frozen counterfactual-conditioned teacher on MDM states. Different counterfactual (all prompt vs name only), states, direction and gradient. Z ≈ "MDU with a subject-swap anchor in place of the prompt-mask anchor, on teacher-answer states". Not a limiting case: masking the subject span alone is not MDU either. |
| **EMDU** (PDF) | p0 with prompt **and** answer masked (state-independent); KL(anchor‖p_θ) (same direction as X) | random masks of the GT answer | student only | α/t retain CE | **Similar ingredients, different algorithm.** |
| **TNT/TNFF** (CODE) | same-context p0 with the negative token zeroed; positives self-distilled to p0 | fixed own generations, teacher-forced | student only | none | **Different algorithm**: exclusion target, no counterfactual input. Shares only the same-position frozen-teacher KL(target‖student). |
| **UNDIAL** (CODE) | same-context softmax(z0 − γ·onehot(label)) | teacher-forced forget sequence | student only | none (GradDiff in open-unlearning) | **Different algorithm** (target family = TNFF as γ→∞). |
| **D2F** (CODE+MIRROR) | LLM-written counterfactual **object** answer A⁺ vs original A⁻; reference-DPO on span-weighted log-probs | AR teacher forcing on A⁺ and A⁻, same prompt | through a **difference of log-probs of two outputs given the same input** | retain KL+NLL, LoRA | **Different algorithm.** D2F differences over *outputs*; X differences over *inputs*. Both use counterfactual supervision. |
| **CUSE** (SNIPPET) | counterfactual inputs that replace target evidence; the change in token log-prob between original and counterfactual input **weights** forgetting; on-policy samples; frozen teacher with privileged contexts | on-policy | **unknown** (a weight suggests no-grad, but unverified) | retention targets | **UNRESOLVED (lack of access).** Closest ingredient overlap: factual-vs-counterfactual input log-prob difference. |
| **Selective Span-Level** (SNIPPET) | contrasts gradients from forget vs retain sets to select tokens; spans via self-consistency | ? | ? | ? | **UNRESOLVED.** The abstract describes *selection*, not differencing in the loss. |
| **Counterfactual Logit Pairing**, Garg et al. 2019 (SNIPPET + memory) | task loss + λ·E\|g(x) − g(x′)\| on classifier logits, x′ = identity-term substitution | supervised classification data | **through g(x) − g(x′)**: λ·sign(Δ)·(J_x − J_x′) | none | **Similar ingredients, different algorithm, and the closest operation.** CLP's gradient has X's Jacobian-difference structure. In CLP the pair's own disagreement is the error, aimed at invariance (Δ→0) without a frozen target. In X the error is the factual branch's KL to a frozen counterfactual teacher. X's surrogate is a function of Δ = z_A − z_c with offset sg z_c. With the anchor near T, its minimiser is Δ ≈ 0 on non-name tokens: a *distributional, frozen-target-anchored logit pairing*. Adversarial Logit Pairing (Kannan 2018; memory) has the same structure. |
| **IIT** (Geiger 2022; SNIPPET) | KL to the causal model's counterfactual output after swapping internal activations between base and source inputs | pairs | through the intervened pass (both-pass gradient implementation-dependent) | task loss | **Different algorithm** (aligns internal variables). |
| **ULD** (Ji et al. 2024; SNIPPET) | unlearned logits = base − α·assistant, assistant trained with reversed objectives | — | logit difference **between models**, applied at inference | — | **Different algorithm.** |
| **Frozen-difference targets**: ESD/ELM/CFG, MDU Eq. 6, WHP's v_base − α·ReLU(v_reinf − v_base), RepE/LoRRA (memory) | target built from a difference of **frozen** logits or representations | various | student only | various | **Different**: none back-propagates through a *student* difference. |
| **ROME/MEMIT/AlphaEdit** (CODE) | closed-form key→value edit; C⁻¹ whitening or a null-space projector removes shared key directions | — | no backprop to weights | essence KL | **Different algorithm** with a similar intent (protect shared directions). |
| **CAE**, entity-level editing (2601.08840; SNIPPET) | joint low-rank edit over many prompts of one entity + a consistency regulariser aligning edit directions | prompts | student | — | **Different** (entity-level, no counterfactual twin). |
| **"GradDiff" in unlearning** (Maini et al.) | GA on forget + descent on retain | — | — | — | **Name clash only**: it is a forget-minus-retain loss, not a gradient through a minimal-pair difference. |

## 2. Notes the designers should hear
- X has the Jacobian-difference gradient (J_A − J_c)ᵀe. This is the gradient of *any* loss written as a function of the
  pair difference Δ, which is exactly what logit-pairing penalties produce. Therefore "the differencing operation is
  new" is **not** supportable. What I did not find is narrower: a forget term whose *value* is the KL to a frozen
  counterfactual-subject teacher at the factual branch, with the twin branch receiving the negated error and a
  separate frozen-target anchor.
- Z is WHP+ (TOFU, N=1) carried into LLaDA SFT layout. Its distinctive state choice is masks over the teacher's own
  stranger answer rather than over the original answer. That choice belongs to the target and states, not to the
  operation X tests, and it should be credited to WHP+ in the handoff.
- The most likely reviewer reduction is: "X = Z + a CLP-style pairing penalty between Alice and stranger canvases."
  These are not exactly equal: X has no separate λ, and the pairing error is t-driven. The handoff should include an
  arm Z+CLP, i.e. Z + λ·KL(σ_tok p_θ(·|x̃_ν) ‖ p_θ(·|x̃_A)), with gradient into both branches, or an L1 logit-pair term, if the budget allows.
  Otherwise the reduction should be stated as unresolved.

## 3. Novelty statement for the handoff (3 lines)
Experimental combination (not a claimed new method). The WHP+ name-swap (stranger) teacher, N=1 with label back-swap, is
adapted to LLaDA SFT masked states over the teacher's own answer, plus a twin anchor (= Z). X re-routes the forget error
through the factual-minus-twin logit difference: a frozen-target variant of counterfactual logit pairing (Garg 2019).
This exact forget-term construction was not found in the inspected sources. CUSE and Span-Level (abstract-only) are
unresolved.
