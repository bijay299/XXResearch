# E — Mathematical critique of A (FFX-OP), B (FCH-Mix), C (MSDA) and SPEC v0.1 (FX-Mix)

Evidence: `agents/E_scripts/counterexamples.py` → `counterexamples_out.json`, `counterexamples_stdout.txt`
(ledger agent E, 70 s, 1 thread). Blocks are cited below as [E1]..[E15]. Sampler semantics were checked against
`/tmp/claude-0/src/LLaDA/generate.py` @9182493 (Gumbel-max ⇒ x0 ~ softmax(z/T); `random` confidences are iid
uniforms independent of x0; `low_confidence` confidence = untempered softmax(z)[x0] **in the model dtype**; top-k is
taken over masked positions of the current block; no re-masking; `main()` decodes with `skip_special_tokens=True`, so
tokens after an EOS stay in the text).

## 1. Verdict table

| # | claim (source) | verdict | reason / counterexample |
|---|---|---|---|
| A1 | argmin_{q(A)=0} KL(q‖p0) = p0(·\|A^c), value −log p0(A^c) (A §2) | CORRECT (standard) | needs p0(A^c)>0; numeric = closed form to 8e-6 [E1]. Csiszár I-projection; TNT/TNFF target. Not novel. |
| A2 | conditionals q*(x_j\|s)=p0(x_j,A^c\|s)/p0(A^c\|s) (A §2) | CORRECT-WITH-CONDITIONS | only for a *consistent* joint. LLaDA's conditionals define no joint: sampled output depends on the order (l2r vs r2l: TV 0.40 for one conditional family [E2]). So "q*" depends on the config; only the per-position target is well defined. |
| A3 | fact-visible states need no training (A §2, A2) | CORRECT-WITH-CONDITIONS | true for the token event Λ_F: the first F commit is always made from a fact-free state (B 1a). False for string-level metrics when an identifying piece is outside F, and silent when the *prompt* shows a fact token (no state is fact-free). |
| A4 | mean-field q̂ = q* iff the other positions' fact mass doesn't depend on x_j (A §2) | CORRECT | exact condition: p0(A^c_{−j}\|s,x_j=v) constant over v∉F. Counterexample to q̂=q*: joint (f,a).5,(b,f).3,(b,c).2 ⇒ q*=δ(b,c), q̂₂=(a .714, c .286); product of q̂ puts **0.714** on (b,a), which has p0-mass 0 [E2]. |
| A5 | F1: forget weight on (fact pos, fact-free state) = 1/L of the total (A §1) | arithmetic CORRECT; interpretation INCORRECT | exact fraction (1−ε^k)/((1−ε)L) (MC agrees [E13]); 2nd moment ln(1/ε)/(1−ε)=6.91 ✓. But "the only terms that can change leakage" is false: GA at a *non-fact* position raises p(F) there (0.20→0.28 for one unit step [E13]), and P2 itself sums over all eligible positions. Demote to a remark. |
| A6 | P1: p_θ=q̂ (or q*) on fact-free reachable states ⇒ Leak=0, any config T≥0 (A §2) | CORRECT-WITH-CONDITIONS | needs cfg=0 and no logit change before the draw. `logits_eos_inf=True` breaks it [E6]. Unattainable with finite logits; it is a corollary of B 1(a), not a separate result. |
| A7 | P2 union bound Leak ≤ E Σ_{t<t*} Σ_{E_t} π_T(F) (A §2) | INCORRECT as printed | needs t ≤ t* (the decision state). L=1, one step: leak 0.3, printed bound 0 [E3]. The appendix (t ≤ t*) is right. |
| A8 | P2 "random: E_t may be replaced by the committed set" | CORRECT (random only) | fails for low_confidence: leak 0.30 vs committed-set bound 0.09 [E4]. |
| A9 | KL(q̂‖p) ≥ −log(1−p(F)) ⇒ π_1(F) ≤ 1−e^{−KL} | CORRECT (T=1 only) | false for 0<T<1: π_0.5(F)=0.47 > 0.40 for p=(.4f,.3,.3); B's |V|=126k case gives 0.44 > 0.30 [E5]. |
| A10 | T=0: no fact commit if TV(q̂,p) < ½max q̂ | CORRECT-WITH-CONDITIONS | strict inequality, so ties are excluded. Breaks under `logits_eos_inf` [E6]: TV 0.30 < 0.43, yet greedy commits f. |
| A11 | "on-policy forget loss upper-bounds T=1 leakage for configs in 𝒟" (A §2) | UNSUPPORTED | the loss is a mean over positions on a ½TF+½stale mixture whose pool has T∈{0,0.7,1}. The bound needs the **sum** over eligible position-steps of the *current* model's own rollouts, per config. At LLaDA default (gen 128, block 32, 128 steps) Σ_n\|E_n\| = 2112, so certifying 5% needs mean per-position KL ≤ 2.4e-5 [E9c]. |
| A12 | at θ0 the loss = Σ −log(1−p0(F)) and the logit gradient = p0(F)(p0\|_F − q̂) | CORRECT | general ε form in §5 Lemma T(iii). Selectivity holds at θ0 only; for θ≠θ0 the term distils toward p0 at every position. |
| B1 | Prop 1(a) exact first-commit decomposition | CORRECT | disjoint first-commit events; fresh noise each step, so the process is Markov in (n, s_n). |
| B2 | 1(b) h_n ≤ Σ_{E_n} q_T(F) for lc despite draw-dependent selection; rnd factor k/\|E\| | CORRECT | uses only S_n ⊆ E_n. For rnd, P(i∈S_n)=k_n/\|E_n\| independently of the draws. Loose for lc with T>0 (B: ×9). |
| B3 | 1(c) q_T(F) ≤ ρ=p(F)/p_max for T∈[0,1], F a set | CORRECT | Σ_f p_f^β/Σ_v p_v^β ≤ Σ_f (p_f/p_max)^β ≤ ρ for β≥1. Max violation over 20k random cases 2e-16 [E5]. Fails for T>1 (T=2: 0.25 > 0.11) and is loose by 1/p_max at T=1. |
| B4 | 1(d) chain P ≤ E_{d^ff}[Hρ] ≤ (Σ_n\|E_n\|)·sup ρ | CORRECT-WITH-CONDITIONS | outer inequalities hold. The middle term (Hρ sums over M(s) ⊇ E_n) can exceed the last (block 1, L=2: 3ρ vs 2ρ [E9b]). Write both forms from the E_n sum. d^ff is a mass (≤N), so "E_{d^ff}" is an unnormalised sum. |
| B5 | greedy: ρ<1 on supp d^ff ⇒ leak exactly 0 | CORRECT-WITH-CONDITIONS | ρ<1 ⇒ p_f<p_max strictly, so the result holds under any argmax/top-k tie-break. ρ=1 fails: torch.argmax returns the lowest id [E14]. It is path-specific: bf16 confidences tie (0.9961 and 0.9980 → 0.99609 [E14]), so a CPU-fp64 certificate can follow a different greedy path than a GPU-bf16 deployment. Breaks under `logits_eos_inf` [E6]. |
| B6 | 1(e) P ≤ C·E_μ[Hρ], C=sup d^ff/μ | CORRECT | for any measure μ (requires μ>0 on supp d^ff). |
| B7 | 1(e) "μ=Σw_k d^ff_k ⇒ C ≤ 1/w_k" | INCORRECT for the mechanism's μ | the mechanism uses *normalised* d̄; then C ≤ \|d^ff_k\|/(w_k(1−α)), with \|d^ff_k\| ≤ N_k. Semi-AR example (a=.05, N=32): leak 0.806, bound with C=1/w is 0.053, correct C=16.1 gives 0.849 [E9]. |
| B8 | γ=log[p0F(1−ε)/(ε(1−p0F))]_+ is the I-projection onto {q(F)≤ε} | CORRECT | binary-KL convexity, plus within-group conditionals kept. Matches the numeric SLSQP solution to 2e-8 in both cases; p0F ≤ ε ⇒ r=p0 [E1]. **Numerics:** the literal formula and B's `γ[:,None]·onehot(F)` give **NaN** when p0F rounds to 1.0 (fp32 gap 20, bf16 gap 8 [E8]), which is likely on a 1000-epoch TOFU SFT model. F's `fxmix.exclusion_target` is log-space and correct. |
| B9 | Pinsker: KL ≤ κ ⇒ p_θ(F) ≤ ε+√(κ/2) | CORRECT, weak | data processing gives the sharper kl(ε‖p_θ(F)) ≤ κ (Lemma T(iv)). |
| B10 | "span-level leak ⇒ Λ_F, so P(span) ≤ P(Λ)" (B §2) | CORRECT-WITH-CONDITIONS | holds only if *every* tokenisation of the span contains an F id. F={" 1951"} misses "19"+"51". |
| B11 | "truncation at first F-commit biases d" (B §3) | INCORRECT attribution | truncation reproduces d^ff exactly. The bias comes from taking J states uniformly *per rollout* and always keeping the decision state: d̄(s0)=0.40 vs sampled 0.625 [E10]. This changes the constant, not validity. |
| C1 | Prop 1(a) D_R ≥ E_F/μ₁ (generalised Rayleigh) | CORRECT-WITH-CONDITIONS | holds only in the metric of the *shrunk* C_R used in the GEP. Unshrunk D_R can be 0 < E_F/μ₁=0.005 [E11]. Textbook. |
| C2 | 1(b) C_R-orthonormal A ⇒ D_R=‖B‖², E_F=Σμ_j‖b_j‖², ratio within μ₁/μ_r | CORRECT | A C_Δ Aᵀ = diag(μ). The ratio bounds hold for any ΔW with rows in span(u₁..u_r), so row normalisation is harmless (Adam is not invariant to it). |
| C3 | 1(c) blindness costs ≥ ε_ℓ/μ₁ | CORRECT (layer-local) | not relevant to leakage: Δk is measured against MDU's anchor, which already encodes the fact when the subject is visible (A F2), and residual/attention paths bypass the layer. |
| C4 | 1(d) single-key optimum ‖r‖²/(kᵀC_R⁻¹k); Kantorovich factor ≤ (κ+1)²/4κ | CORRECT (textbook) | numbers in C §7 match (255 at κ=1018). |
| C5 | layer-input/-output energy as a selectivity criterion for output leakage | UNSUPPORTED | energy is neither necessary nor sufficient: an edit of energy 25 in the readout null space changes no logit, while energy 0.0036 flips the argmax [E12]. C concedes D_R≠KL. |
| C6 | §1(c) first LoRA step ΔW = −ηGKᵀAᵀA; "random rotation keeps expected ratio" | CORRECT for SGD; UNSUPPORTED for Adam/ratio | Adam's first step is ≈ −η sign(∇_B)·A. A ratio of expectations is not an expected ratio. |
| S1 | SPEC v0.1: "ε>0 restores the fixed point p_θ(F)=ε at trained states" | CORRECT-WITH-CONDITIONS | true for the forget term alone in a realisable parameterisation (1-D AdamW: converges to 1.0003e-3 [E15]). With the retain term and shared LoRA it is a stationary point of the sum only. **That fixed point leaks** (S2). |
| S2 | SPEC v0.1 default ε=1e-3 | INCORRECT as a default for LLaDA | under T=1 random remasking the exact target leaks 1−(1−ε)^n, where n = #committed positions with p0(F)>ε: n=8 → 0.8% (toy), 32 → 3.2%, 128 → **12.0%** [E8]. |
| S3 | SPEC: "ε=0 + forward KL + AdamW has no finite stationary point" (OBJ_B_to_A §3) | CORRECT; "O(lr) steps" overstated | the logit offset keeps drifting but sub-linearly, because Adam's v lags: −4.8, −8.2, −15.8, −26.5 at 1k/5k/20k/80k steps; wd=0.01 slows it (−21.7) without stopping it within 80k steps [E15]. The operating point is set by the step budget. |
| S4 | SPEC P-C "TF-only measures cannot control leakage" | CORRECT-WITH-CONDITIONS | only as non-identifiability in the unrestricted (tabular) class, i.e. zero μ-loss is compatible with leakage off supp μ. With parameter sharing, nothing is proved either way. |

## 2. Cross-cutting checks (assignment §4)
- **Shapes/normalisation.** SPEC uses (1/|M(s)|)Σ_{i∈M(s)}, but the hazard bound weights every *eligible
  position-step* equally (Σ, no 1/|M|). The loss is therefore not the bound's integrand. Either is a legitimate design;
  do not call the loss a bound. EOS-pad positions (≈75% at gen 128 for 30-token answers) dilute the mean. They carry
  ~0 gradient at θ0 and act as distillation-to-θ0 afterwards.
- **TF vs generated states.** There is no theorem either way under parameter sharing. All "TF fails" evidence is
  tabular (zero sharing) or toy. P-C must say so.
- **Sampling/masking biases.** Truncation is exact for d^ff, while per-rollout J-sampling is biased [E10]. The pool
  is stale by ≤R steps, so any coverage/certificate statement holds only for the θ whose *fresh* rollouts are used.
  For a finite pool, C=∞ for T>0 (B's own sup form B3 = 1).
- **Random vs low-confidence.** The k/|E| factor and committed-set forms are valid only for `random` [E4]. The lc
  bound uses eligible positions and is loose.
- **Conditioning already supplies the answer.** (i) F(c,y*) excludes prompt tokens. A forget prompt that shows a fact
  token loses that token from F, and no state is fact-free, so every proposition is silent. Define F per *fact*, and
  report prompts containing F tokens separately (copying from context is not recall). (ii) Restated subject: harmless
  for r_ε, which is evaluated at the same state with the prompt visible. It is fatal for MDU's anchor (A F2, A→C).
  (iii) Multi-token facts: a frequency/stop-list filter can drop the identifying piece. If LLaDA's tokenizer splits
  digits (UNVERIFIED, HF blocked), every digit is top-frequency, so rule A2(c) empties F for dates; without the filter,
  F={1,9,5} forbids those digits everywhere.
- **Token-level union event is stricter than retrain.** Toy city/year values share CA/CB/YD/YU pieces across
  entities, so the retrain model has leak_any 0.08–0.13 (SHARED §7). A target with r(F)=0 drives this to 0, which is
  over-forgetting and a detectable signature. Thresholds must be stated relative to the retrain level.
- **Post-EOS tokens.** Upstream decoding keeps them, so Λ_F must range over all gen_length positions. A metric that
  truncates at the first EOS under-counts.
- **CFG.** p_cond(F)=0.05 (ρ=0.053) with p_uncond(F)=0.001 gives p_cfg(F) = 0.28/0.73/0.99 at s=0.5/1/2 [E7]. Out of
  scope unless ρ is computed on the combined logits and the unconditional state is trained.
- **Other logit edits.** `logits_eos_inf` and the end-think boost act before the draw and invalidate ρ/TV
  certificates [E6]. `confidence_eos_eot_inf` affects only selection, so the bounds survive it.
- **Optimiser.** Clipping does not change stationary points. AdamW's wd pulls the LoRA toward 0, i.e. toward θ0
  (toward leakage). Adam has no fixed point under minibatch noise. Fixed-point statements are about the loss, not the
  optimiser trajectory.
- **LoRA.** With B=0, ∂L/∂A=0 at step 0 (only B moves first) and the student equals θ0, so the first pool equals
  reference rollouts. No proposition needs full rank. "Student = target on trained states" needs realisability, which
  rank-r LoRA plus the retain term generally lack; the infimum is not attained for ε=0.
- **Forward vs reverse KL.** Forward KL(r‖p_θ) is finite for ε=0, and its F-gradient p_θ(f)−r(f) vanishes as
  p_θ(F)→ε. Reverse KL is infinite for ε=0 and usable only with ε>0. No 1/t weight is needed: no claim is an ELBO.
- **Mean-field inconsistency.** It is irrelevant to the support/leak statements. For fluency it matters under
  parallel commits [E2].

## 3. Most serious problem
No stated result connects the **optimised loss** to leakage at LLaDA scale:
- the bound needs per-config sums over ~2×10³ eligible position-steps of the current model's own rollouts;
- B's coverage constant is wrong by the visitation mass (up to N);
- the KL→leak step fails for 0<T<1;
- SPEC v0.1's ε=1e-3 per-position floor makes the target's own fixed point leak up to 12% at gen 128.

What survives is a **post-hoc sampler-level certificate** for the final model, plus an exact support statement.

## 4. Required corrections to SPEC v0.1 (FX-Mix)
1. **ε.** Report the implied fixed-point leak 1−(1−ε)^{n̂}, where n̂ = mean number of committed positions with
   p0(F)>ε on θ0 rollouts (measure it on Hyperplane). Choose ε ≤ δ/n̂ for the evaluation threshold δ (worst case
   n̂=gen_length: ε ≈ 1e-4 for δ≈1%). Keep ε=1e-3 only for the toy (gen 8: ≤0.8%). Alternative: ε=0 with the step
   budget as the explicit knob [E15].
2. **Target computation.** State r_ε in log-space (as `fxmix.exclusion_target` does). Delete the literal
   γ·onehot form [E8].
3. **Theory section.**
   - P-H in the ρ form with min(1,ρ), the E_n sum, t ≤ τ, and the exact assumption list of §5.
   - P-C with C ≤ |d^ff_k|/(w_k(1−α)), as a remark (population measure, same θ).
   - Drop every sentence saying the loss bounds leakage.
4. **Scope list (§1).** Add logits_eos_inf=False, end_think_logit_boost=0, the same gen_length, deployment dtype
   (bf16 ties), and Λ over all positions incl. post-EOS. The prompt must contain no F token, else report separately.
5. **Certificate.** Compute B̂_π = (1/R)Σ_r Σ_{n≤τ_r} Σ_{i∈E_n} min(1,ρ) on fresh rollouts of the *final* model, in
   deployment precision, per eval config. Report it next to the observed leak. It is an unbiased estimate of an upper
   bound for that θ and π only.
6. **State sampler.** Either sample the step index uniformly over 0..N−1 and reject steps at or after τ (gives d̄^ff),
   or describe μ as "uniform per rollout + decision state" [E10]. Either way, μ>0 on visited states is all the
   remark needs.
7. **Metrics.** Primary forget metric: Λ_F (any F id, any position) plus span exact-match over all tokenisation
   variants of F. ROUGE-L cannot reach 0 when answers restate the subject, so it is secondary. Thresholds are relative
   to the retrain reference's level.
8. **F definition.** Define F per fact (not per prompt), and log, per forget item, |F| and the fraction of F ids
   among the top-N retain tokens. Make the tokenizer's digit handling a Hyperplane check (UNVERIFIED here).
9. **Wording.**
   - "only F-logits move" is a property of the *target*, not of the update.
   - "fixed point = target" requires realisability.
   - Rollouts at step 0 equal θ0's (LoRA B=0).

## 5. What may appear in THEORY.tex

**Assumptions (H).**
- Fixed prompt c with no token of F; gen_length L; F⊂V nonempty.
- Sampler π=(N, block b, rule∈{lc,rnd}, T∈[0,1]) with the pinned `generate` semantics: cfg_scale=0,
  logits_eos_inf=False, end-think boost 0; draws x0_i ~ π_T(·|s,i)=softmax(z(s,i)/T) independent over i (T=0:
  argmax with the implementation's fixed tie-break); selection S_n ⊆ E_n(s) (masked positions of the current
  block), |S_n|=k_n (deterministic); no re-masking.
- Model: any deterministic map s ↦ z(s,i).
- s_n = state before step n; τ = first n whose committed set contains an F id (τ=∞ if none);
  Λ_F={τ<∞} (all L positions); ρ(s,i)=p(F|s,i)/max_v p(v|s,i).

**Proposition 1 (first-commit decomposition and sampler-uniform hazard bound).** Under (H):
- (a) P(Λ_F)=Σ_n E[1{τ≥n} h_n(s_n)], with s_n fact-free on {τ≥n}, where h_n(s)=P(∃i∈S_n: x0_i∈F | s_n=s).
- (b) h_n(s) ≤ Σ_{i∈E_n(s)} π_T(F|s,i); for rule=rnd, h_n(s) ≤ (k_n/|E_n(s)|) Σ_{i∈E_n(s)} π_T(F|s,i).
- (c) π_T(F|s,i) ≤ min{1, ρ(s,i)} for every T∈[0,1], and π_1(F|s,i)=p(F|s,i).
- (d) Hence P(Λ_F) ≤ B_π := E[Σ_{n≤τ∧(N−1)} Σ_{i∈E_n(s_n)} min{1,ρ(s_n,i)}] (with the k_n/|E_n| factor for rnd).
- (e) If T=0 and ρ(s,i)<1 for every reachable fact-free s and every i∈E_n(s), then P(Λ_F)=0, for any tie-break.

*Proof sketch (checked).*
- (a) The events {τ=n} are disjoint. Given the past, step n uses fresh Gumbel/uniform noise and a deterministic
  model, so P(τ=n | past) = 1{τ≥n} h_n(s_n).
- (b) {∃i∈S_n: x0_i∈F} ⊆ ∪_{i∈E_n}{x0_i∈F}, then the union bound. For rnd, S_n is a uniform k_n-subset independent
  of x0, so P(i∈S_n)=k_n/|E_n|.
- (c) For T=0: argmax∈F ⇒ p(F) ≥ p_max. For β=1/T≥1: Σ_f p_f^β/Σ_v p_v^β ≤ Σ_f (p_f/p_max)^β ≤ ρ.
- (d) Combine (a)–(c).
- (e) ρ<1 ⇒ p_f<p_max for every f∈F, strictly, so no F id is ever the argmax.

*Status.* Elementary first-passage + union bound applied to the pinned sampler. Present it as such, not as the
method's novelty. Its uses are (i) a post-hoc certificate B̂_π (unbiased MC estimate of B_π for the evaluated θ),
and (ii) the observation that the leak-relevant measure is the model's own pre-τ visitation for each π.
*Boundary cases to print with it:* T>1 [E5], cfg>0 [E7], logits_eos_inf [E6], ρ=1 ties [E14], dtype-dependent
greedy paths [E14].

**Lemma T (exclusion target; standard I-projection, cf. Csiszár; TNT/TNFF; UNDIAL).** For p0=softmax(z0) with
p0(F)<1 and ε∈[0,1), let r_ε = argmin_{q(F)≤ε} KL(q‖p0). Then:
- (i) r_ε=p0 if p0(F)≤ε. Otherwise r_ε = ε·p0|_F + (1−ε)·p0|_{F^c}, i.e. softmax(z0−γ1_F) with B's γ.
- (ii) KL(r_ε‖p0) = kl(ε‖p0(F))₊; for ε=0 this is −log(1−p0(F)).
- (iii) ∇_z KL(r_ε‖softmax z)|_{z0} = (p0(F)−ε)₊·(p0|_F − p0|_{F^c}), which is zero iff p0(F)≤ε.
- (iv) For any p, KL(r_ε‖p) ≤ κ ⇒ kl(r_ε(F)‖p(F)) ≤ κ (data processing to {F,F^c}). Hence
  p(F) ≤ x_ε(κ) := max{x: kl(ε‖x)≤κ}, and x_0(κ) = 1−e^{−κ}. Also max p ≥ max r_ε − √(κ/2) (Pinsker).

*Proof sketch (checked).*
- (i) Chain rule: KL = kl(q(F)‖p0(F)) + within-group terms. The within-group terms vanish at the group conditionals,
  and kl(·‖p0F) decreases on [0,p0F].
- (iii) p0−r: for v∈F, p0_v(1−ε/p0F) = (p0F−ε)·(p0|_F)_v; for v∉F, −(p0F−ε)·(p0|_{F^c})_v.
- (iv) Coarse-graining cannot increase KL.

*Status.* Standard; cite it, and keep it as a Lemma, not a contribution.

**Remarks only (do not state as propositions):**
- R1. The global I-projection q* and its conditionals assume a consistent joint, which LLaDA lacks [E2]. Mean-field
  q̂ ≠ q* [E2]. P1 is Prop 1 with π_T(F)=0.
- R2. Concentrability, coverage bound, non-identifiability of TF-only training (zero-sharing class), with
  C ≤ |d^ff_k|/(w_k(1−α)) [E9]. Standard (Kakade–Langford, Munos, DAgger).
- R3. Scale of the union bound. For T=1, Prop 1(d) + Lemma T(iv) give P ≤ E[Σ_{n≤τ} Σ_{E_n} x_ε(κ_{n,i})],
  which is vacuous unless κ ≲ δ/Σ|E_n| (2.4e-5 at LLaDA default for δ=5%) [E9c].
- R4. ε-floor fixed-point leak 1−(1−ε)^n [E8]. ε=0 drift under Adam [E15]. AdamW wd pulls toward θ0.
- R5. F1 weight count, re-worded as "share of SFT-weighted forget mass on (fact position, fact-free state) pairs"
  [E13].
- R6. C's Prop 1(a)–(d): textbook generalised-Rayleigh/Kantorovich facts. They are layer-local, hold in the shrunk
  metric [E11], are not output-level [E12], and are not part of FX-Mix.
