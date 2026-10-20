# SPEC — authoritative specification (coordinator-owned)

Version **v1.3.1 (05:17Z, wording: SC definition; v1.3 04:52Z; v1.2 04:52Z; v1.1 04:44Z; v1 04:40Z)**. History: v0 04:24Z (A+B merge) → v0.1 04:27Z (B→A objection: pool, ε>0, held-out configs,
ρ-form bound) → v1 (D reduction: controls, primary advantage; E critique: target numerics, ε for LLaDA, theory
scope, certificate, metrics, F per fact) → v1.1 (RESP_A: both matching cuts, extrapolation cell, fp32 target, certificate SE) → v1.2 (E theory check:
Prop 2 ε-qualifier and item (d), wording) → **v1.3** (RESP_B: ε by eligible slots, exact-π_T certificate, F-closure
audit, shared-value-token slice). Every equation, code path and claim in the package must agree with this file;
G reconciles before publication. Changes after the validation lock are forbidden except one documented correctness repair.

## 0. Name and one-line description
**FX-Mix** — *fact-free eXclusion distillation on a native-sampler state Mixture.* Distil, into LoRA weights, the frozen
reference's own per-position prediction with the fact tokens excluded, at every fact-free state that the *current*
model's native LLaDA sampler visits under a pool of decoding configurations (plus teacher-forced states with the fact
positions masked), stopping each rollout at its first fact commit.

## 1. Forgetting notion and scope
Forget request R = {(c, y*, F)}: prompt c, reference answer y*, **fact-token set F defined per fact** (token ids of the
fact's distinctive surface pieces incl. tokenisation variants; prompt must contain no F token — otherwise the item is
reported separately as "conditioning supplies the answer"). **F-closure audit (mandatory for LLaDA, v1.3):** before
training, take the top-5 tokens of p0|_{F^c} at θ0 rollout decision states and add every token that detokenises to a
prefix/substring of a surface form of the fact (the exclusion target multiplies non-F tokens by 1/(1−p0(F)), so it would
otherwise promote unlisted pieces of the fact); log the added ids; report string-level leakage next to Λ_F.
Target property = **reduced recall in natively generated answers** (type i): for each native decoding configuration π
of a declared grid (incl. configs never used in training), the event Λ_F = {some generated position — any of the
gen_length positions, incl. after EOS — holds a token of F} is rare; answers remain well-formed (no refusal template,
no degeneration); retained knowledge and general capability stay near the original model.
**Not claimed:** removal of training influence, retraining equivalence, latent-knowledge removal (MC probability,
probing, relearning), robustness to re-masking samplers, prefix/priming or context injection, cfg_scale>0, T>1,
logits_eos_inf=True / end-think boost, other gen_length, paraphrases expressed outside F, dtype-dependent greedy paths.

## 2. Objective and update
L(θ) = λ_f · E_{(c,y*,F)∼R} E_{s∼μ_θ̄(c)} [ (1/|M(s)|) Σ_{i∈M(s)} KL( r_ε(·|s,i) ‖ p_θ(·|s,i) ) ] + λ_r · L_ret(θ)

- **Target (Lemma T)** r_ε(·|s,i) = argmin_{q: q(F)≤ε} KL(q ‖ p_θ0(·|s,i)), reference at the **same** state (prompt
  visible): r_ε = p0 if p0(F) ≤ ε, else ε·p0|_F + (1−ε)·p0|_{F^c}. **Computed in log-space** (log p0|_{F^c} =
  z0 − logsumexp_{F^c} z0, etc.) **from fp32 reference logits** (bf16 can misplace the p0(F) ≤ ε switch); the literal
  γ·onehot formula is forbidden (NaN when p0(F) rounds to 1).
  ε: toy 1e-3 (fixed-point leak ≤ 1−(1−ε)^8 ≈ 0.8 %; eligible-slot certificate floor ≤ 36ε = 3.6 %); LLaDA default
  **ε = 1e-5** (v1.3; chosen ≤ δ/n̂_elig with n̂_elig ≤ Σ_n|E_n| = 2112 at gen 128/block 32/128 steps, so the target's own
  certificate floor is ≤ 2.1 % for low-confidence configs); report the implied fixed-point leak ≥ 1−(1−ε)^{n̂} and the
  certificate floor ε·n̂_elig, with n̂, n̂_elig measured on θ0 rollouts; ε = 0 is allowed
  only with the step budget declared as the strength knob (no finite minimiser; drift under Adam).
- **States μ_θ̄(c)** (fact-free only = no visible F token):
  - fraction α = 0.25: TF_ff — t ~ U(0,1); mask each non-fact answer position iid w.p. t; always mask every position j
    with y*_j ∈ F (answer region = gen_length incl. EOS padding).
  - fraction 1−α: rollout states — for config π_k ~ w (uniform over the pool Π_train), run native `generate` with the
    current student θ̄ (no grad; pool refreshed every R steps, so stale ≤ R steps); record the state before each step;
    stop at the first step whose committed set contains an F token (τ); keep J states sampled uniformly from the
    recorded pre-τ states **plus the decision state s_τ when it exists** (this per-rollout sampling is a design choice,
    not an unbiased sample of the visitation measure).
- Positions: all masked answer positions M(s) (mean). No 1/t weight. **The loss is not a leakage bound** (see §5).
- Retain: LLaDA SFT loss on the retain set, λ_r = 1 (as MDU/GD).
- Parameterisation: LoRA (toy r=8; LLaDA r=16) on q/k/v/o/gate/up/down of all blocks, A Gaussian, B=0 (so the first
  pool equals θ0 rollouts and ∂A=0 at step 0); reference = adapter-disabled pass. Merged for deployment.
- Optimiser: AdamW (wd 0), grad-clip 1.0; toy constant LR; LLaDA cosine.

## 3. Decoding configurations
Toy (gen_length 8), (steps, block_length, remasking, T):
- Training pool Π_train: P1 (8,8,lc,0); P2 (8,8,rnd,1); P3 (4,8,rnd,1).
- Held-out: H1 (8,2,lc,0); H2 (4,8,lc,0.5); H3 (8,8,rnd,0); H4 (8,8,lc,1); H5 (4,4,rnd,0.7); H6 (2,8,rnd,1).
- **Primary forgetting statistic = max over held-out configs H1–H5** (all within the pool's maximum tokens-per-step k≤2);
  H6 (k=4 > pool max) is a pre-registered **extrapolation cell**, reported separately and as max over H1–H6 (v1.1).
LLaDA (gen_length 128): pool (128,128,lc,0), (128,128,rnd,1), (64,128,rnd,1), (32,128,rnd,1) [k up to 4; v1.1];
held-out grid and extrapolation cells in HYPERPLANE_PLAN.

## 4. Arms (matched data access, teacher, tuning grid, step budget; compute reported)
| arm | target | states | isolates |
|---|---|---|---|
| NPO+RT, MDU τ=1 +RT (GD optional) | NPO β=0.2 / prompt-masked anchor | random TF states | literature context |
| SC-TF | r_ε | random TF states (fact-visible allowed) | E1: target alone |
| SC | r_ε | MDU loop with its rollout flag on: the flag *replaces* the random TF state by one random-stop greedy-lc rollout state of the current model per batch (no truncation, refreshed every step), as in MDU code L1365–1371 | straightforward combination |
| SC-TF-cm | r_ε | SC-TF with extra TF states to match FX-Mix forwards | E2: more forget pressure |
| FX-Mix-greedy | r_ε | TF_ff + ROLL(P1 only), truncation | value of the pool |
| **FX-Mix** | r_ε | TF_ff + ROLL(P1,P2,P3), truncation | candidate |
| optional: no-truncation, random-masked TF, ε=0 | | | ops #2, #3, ε |
References (not arms): retrain-without-forget model; decode-time F-ban on the base model (inference-time, FX-Mix's
fixed point, never deployed).

## 5. Theory (THEORY.tex; standard tools, stated as such)
- **Lemma T** (I-projection exclusion target: closed form, KL value, gradient at θ0 = (p0(F)−ε)₊(p0|_F − p0|_{F^c}),
  coarse-graining bound kl(r_ε(F)‖p(F)) ≤ KL(r_ε‖p)).
- **Proposition 1** (first-commit decomposition and sampler-uniform hazard bound, assumptions (H) of E §5): exact
  decomposition; h_n ≤ Σ_{E_n} π_T(F) (rnd: ×k_n/|E_n|); π_T(F) ≤ min{1, ρ} for T∈[0,1]; P(Λ_F) ≤ B_π; greedy exact-zero
  when ρ<1 on reachable fact-free states. Use: **post-hoc certificate B̂_π** on fresh rollouts of the final model, in
  deployment dtype and tie-break, with its Monte-Carlo standard error, computed in the **eligible-sum form with the exact
  per-position π_T(F|s,i)** of the evaluated config (T=0: 1{argmax ∈ F}; ×k_n/|E_n| for random remasking); the ρ form is
  only the T-uniform statement (up to |V|× loose at T=1) (v1.3).
- **Proposition 2** (escape characterisation, unrestricted zero-sharing class): sup over models matching r_0 on a state
  set S of the leak under π equals the target sampler's escape probability from S; = 0 if S ⊇ the target sampler's own
  visitation; ≥ 1−1/L (m=1) or = 1 (m≥2 fact positions) for S = TF fact-free states under one-token random decoding.
  Stated for ε=0 (r_0 = decode-time ban); for the default ε>0 use item (d) (sup = P^{r̄_ε}(leak before escape or
  escape)). Says nothing about networks with parameter sharing (that is the empirical SC-TF vs FX-Mix question).
- Remarks only: consistent-joint q* vs mean-field (R1); coverage constant C ≤ |d^ff_k|/(w_k(1−α)) (R2); scale of the union bound (R3); ε fixed-point leak and ε=0 drift (R4);
  SFT-weight share on leak-relevant terms (R5).
- Never claim: that the loss bounds leakage; global forgetting or utility from a local constraint; anything for T>1, CFG.

## 6. Metrics (toy)
Per config: leak_any = Λ_F rate (any value token of the fact at any answer position), leak_full = exact value span,
well-formed rate of forget answers; retained-fact accuracy (neighbour / other / shared-token facts), copy accuracy;
off-manifold fraction of first-commit decision states; certificate B̂_π; compute counters (student fwd/bwd, reference
fwd, rollout fwd) and wall time. Thresholds are stated **relative to the retrain reference** (its leak_any 0.08–0.13
is chance from shared value pieces; driving leak_any to 0 is an over-forgetting signature, reported as such).

## 7. Primary advantage and decisive comparison
Primary: **config-robust forgetting** — lower held-out max leakage than SC-TF and SC at matched retention, with
well-formed forget answers. To prevent the matching axis from choosing the winner, the full (held-out max leak,
retention) curve over each arm's strength grid is reported, together with **both cuts**: leak at matched retention and
retention at matched leak (v1.1); pre-registered slice in both cuts: retained facts that **share a value token** with a
forgotten fact (C's falsifier: the state design is predicted to give no retention gain there) (v1.3). FX-Mix is a distinct method only if it beats SC-TF and SC there (D §3.6); otherwise the
outcome is reported as "TNFF-in-MDU" (target credited to TNT/UNDIAL).
