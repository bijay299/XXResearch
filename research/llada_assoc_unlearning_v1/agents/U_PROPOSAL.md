# U_PROPOSAL — selective-neuron down-projection LoRA ("SN-LoRA") (Agent U#1)

Written under the coordinator's priority update: it must be PEFT-LoRA compatible, ≤ 120 lines, with no synthetic
runs. `U_scripts/u_check.py` was written before the update and **never run**. No result below comes from it.

## 0. Why the update site, and not the loss, is the lever (first principles)
- Under a linear-in-parameters view (small update, tangent model), a parameter block with input feature φ(s)
  changes a retained state's logits in proportion to the kernel overlap ⟨φ(s_F), φ(s_R)⟩. This is C's closed-study
  derivation: the 91–96 % shared-parameter carriage.
- For LoRA on q/k/v/o/gate/up (and on the head), φ is the normed residual h̃, which is **linear** in the state.
  A forget state (Alice, lives_in, [M]) and a retained state (Bob, lives_in, [M]) share the relation component, the
  extracted "Paris" attribute and the mean direction. So they overlap at **first order**, and the edit acts like a
  token or relation ban.
- In LLaDA's SwiGLU, the input of `ff_out` is a(h̃) = silu(W_g h̃) ⊙ (W_u h̃). Each coordinate a_n is a **product**
  of two linear read-outs of the state. These are the network's own conjunctive features. If the update reads only
  neurons n with a_n(s_F) large and a_n(s_R) ≈ 0 on one-sided neighbours (same object/other subject, same
  subject/other relation, the object as a word, strangers), the overlap Σ_{n∈M} a_n(s_F) a_n(s_R) is small *by
  selection*. Its size can be measured forward-only, before any training.

## 1. Update realisation (what is trained and frozen; equations)
- **Frozen:** every base weight of LLaDA-8B-Instruct. The **reference** p_ref is the same model with the adapter
  disabled (`with model.disable_adapter():`), i.e. the memorised checkpoint.
- **Trained:** PEFT LoRA on `ff_out` only (the SwiGLU down-projection, d_ff = 12288 → d = 4096). It covers a band
  of mid layers L_e (default 8–23; to be chosen by the forward-only score below), with rank r = 8, α = 16, dropout 0.
  Update: Δh_l = (α/r) B_l (A_l ⊙ 𝟙 m_lᵀ) a_l(h̃), where m_l ∈ {0,1}^{12288} is a **fixed neuron mask**.
  - Implementation: multiply `lora_A.weight` by m_l at init, then register a gradient hook g ↦ g·m_l.
  - Trainable entries: B_l plus K masked columns of A_l per layer.
- **Neuron selection (forward-only, closed form).** Hook `ff_out` inputs a_l(h̃) at masked answer positions of
  training states (§3), under p_ref. For each forget fact f, layer l and neuron n:
  - S_f(n) = |mean_{s∈F_f} a_n(s)| / ( sqrt(max_g mean_{s∈R_g} a_n(s)²) + ε ).
  - The retain groups R_g are: (i) the retain facts sharing f's answer word; (ii) the other retain facts;
    (iii) f's subject with other relations, if available; (iv) f's question with the subject replaced by novel names;
    (v) generic text that contains the answer word.
  - The mask m_l is the union over f of the top-k_f = 8 neurons per fact per layer, so K ≤ 64 per layer.
  - Diagnostic logged before training: the predicted damage ratio Ô(f, s_R) = Σ_{n∈M} a_n(s_F)a_n(s_R) / Σ_{n∈M} a_n(s_F)².
- **Structural property (approximate, stated precisely).** At first order in the update, the logit change at a
  retained state is ∝ the overlap of the masked activations, Ô. The bound is exact only in the tangent regime. After
  training, B and A change and second-order terms appear.
  - Bidirectionality: Δh is added at *every* position whose selected neurons fire and later layers spread it over
    the canvas, so invariance is per canvas, not per position; a mixed canvas ("Alice lives in [M]; she was born in
    [M]") is **not** protected. No exact zero: silu never returns exactly 0.

## 2. Learning signal and data
- **Signal:** a generic per-state target q*(·|s,i) at masked answer positions, supplied by O. Default if O does not
  supply one: the subject-substituted reference q*(·|s,i) = p_ref(·|s[subject→ν], i), averaged over 4 novel names ν,
  with name-token mass mapped back to the true subject's tokens (WHP+-style; D §3.2). It is two-sided, so absence is
  not forced.
- **Loss (one function):**
  L = mean_{f, s∈F_f, i∈M(s)} KL(q*(·|s,i) ‖ p_θ(·|s,i)) + λ_r · mean_{s∈R, i∈M(s)} KL(p_ref(·|s,i) ‖ p_θ(·|s,i)),
  with λ_r = 1 and no 1/t weight. Both KLs are computed from fp32 logits, and the reference passes run under
  `no_grad`. The loss is standard; the operation under test is the update site and the mask.
- **Data:** forget QA (8 facts) plus 1 paraphrase each, with the answer-word span marked; the retain QA (16 facts,
  some sharing answer words) plus paraphrases; ≥ 8 novel names (for q* and selection group iv); 32 generic sentences
  that contain the shared answer words (group v).

## 3. Training-state construction (LLaDA-SFT-faithful)
- **Canvas/masking:** chat-templated prompt (never masked) + answer + EOS padding to gen_length; t ~ U(0,1), answer
  positions masked iid w.p. t; w.p. ½ the answer-word span is masked in full (decision state always covered).
- **Batches:** each step uses 4 forget states and 4 retain states. Retain states use the same template distribution.
  - For each forget state, add its *minimal-pair* retain state when one exists: the same template with a
    shared-answer retain subject. This sits inside the λ_r term and costs nothing extra.

## 4. Inference / deployment
- `merge_and_unload()` gives a standard LLaDA checkpoint (no config change); native `generate` unchanged. Removing
  the adapter restores p_ref exactly; before merging, the edit is a separable delta, as for any LoRA.

## 5. Control (simplest, same supervision, same loss)
- **Z:** identical PEFT LoRA on `ff_out`, with the same layers, rank, loss, data, states and steps, but **no neuron
  mask** (m_l = 𝟙). Z isolates the selection operation.
- If budget permits: Z′ = default LoRA on q/k/v/o/ff_proj/up_proj/ff_out, all layers, shown as the standard baseline.

## 6. Central hypothesis
- **Statement.** Suppose the forget and retain facts share answer words, and LLaDA stores some (subject, relation)
  conjunctions in mid-layer SwiGLU neurons that are selective (measured by S_f). Then masking the ff_out-LoRA input to
  those neurons (X) lowers retain damage per unit of forgetting on the shared-answer retain facts, relative to the
  unmasked ff_out LoRA (Z). The mechanism is that the first-order logit change at a retained state is proportional to
  the masked-activation overlap Ô. This overlap is small by selection under X and dominated by shared neurons under Z.
- **Observable W.** The per-retain-fact damage under X ranks with Ô, which is computed before training.

## 7. ONE prediction for the 8-forget / 16-retain / 1-paraphrase / 1-seed / 2-checkpoint experiment
- **Measure.** At each checkpoint compute D_shared = mean Δlog p_θ(gold answer) on the retain facts that share an
  answer word with a forget fact (paraphrase prompts included; same ELBO estimator and seed on both arms). Compute
  D_forget = mean Δlog p_θ(gold answer) on the forget facts. The damage ratio is ρ = D_shared / D_forget.
- **Prediction:** ρ_X ≤ ½ ρ_Z at both checkpoints, while D_forget(X) is at least ½ D_forget(Z) at the second
  checkpoint.
- **Contradicted** if ρ_X ≥ ρ_Z at either checkpoint. A "capacity-only" reading is also possible: X forgets much less
  (D_forget(X) < ½ D_forget(Z)). In that case the result is uninformative, and the record should say so rather than
  claim success.
- With n = 16 and 1 seed, this is a direction-and-size check, not a significance test.

## 8. Assumptions and what must be checked on LLaDA-8B
- **(A1) Module names.** `ff_out`, `ff_proj`, `up_proj` and `act` come from the vendored dllm `LLaDALlamaBlock`
  (input of ff_out = act(ff_proj(x)) * up_proj(x)). The checkpoint's names are unverified: check
  `model.named_modules()` on the HF checkpoint before setting `target_modules=["ff_out"]` and `layers_to_transform`.
- **(A2) Selective neurons exist.** If max S_f is about 1 at every layer (the fact is fully in superposition), X
  reduces to a low-capacity Z. Report the S_f distribution *before* training. This is a go/no-go readout.
- **(A3) Paraphrase generalisation.** The mask is selected on the training paraphrase only, so held-out phrasings may
  not activate the selected neurons (detector-evasion risk, Gap_HO).
- **(A4)** The target q* is O's responsibility. The control shares it, so the comparison does not depend on it.

## 9. Strongest simple explanation of an apparent win, failure conditions, closest methods
- **Simple explanations.** (i) X has fewer effective parameters (a weaker edit): the ρ normalisation and the
  D_forget floor in §7 guard against this. (ii) ff_out-only placement helps: Z shares it, so it cannot favour X.
- **Failure conditions:** name-token-sharing subjects (neurons partly active); indirect mention ("Bob's wife");
  mixed born_in/lives_in canvases (§1); born_in with the same object when the selected neurons encode the subject's
  "Paris-ness" rather than the conjunction.
- **Closest known methods:** Knowledge Neurons, DEPN, selective pruning (Pochinkov & Schoots); saliency-masked FT
  (SalUn, SSD); T-Patcher (inserted neuron); ROME/MEMIT (same down-projection, linear key C⁻¹k). X differs from these in the selection contrast (one-sided neighbours, including shared-answer retain facts) and
    in using a LoRA restricted to the selected input columns, trained on a two-sided per-state target. D should check
    SalUn/SSD equivalence.
- **Original non-LoRA design (shelved by the deadline).** Insert K new SwiGLU neurons. Each has a closed-form
  subject-detector gate and a relation-detector up-factor, with only the value trained. With a hard gate it is exactly
  invariant on canvases where the gate is off. It leaves F latent in the base weights.

## 10. Cost (LLaDA-8B)
- Selection: ≈ 200 hooked forwards accumulating per-neuron moments (fp32, 12288 × |L_e| × groups); negligible memory.
- Training: ≈ 1 M LoRA parameters (r = 8, 16 layers; ≤ 64 effective A columns per layer); per step ≈ 8 student
  forwards + 8 backwards + ≤ 20 no-grad reference forwards; memory = bf16 model (16 GB) + band activations. Same cost
  as Z plus the one-off selection pass.
