# Agent A proposal — FFX-OP: Fact-Free eXclusion distillation on On-Policy decode states

Track: masked-token objectives vs generated-answer leakage in LLaDA. Target property: **(i) reduced recall in
generated answers**. It is not refusal (no IDK target), not inference-time suppression (parameter/LoRA update, plain
`generate`), and **not** a claim of removing training influence or of retraining equivalence: latent knowledge (MC
probability, relearning, probing) can survive and must be measured separately.

One-line mechanism: on states of the forget prompt in which **no fact token is visible**, collected both by
teacher-forced masking and **by running the student's own native samplers under several decoding configs**,
distil the student at every eligible masked position toward the **frozen reference renormalised without the fact
tokens** (forward KL); retain loss unchanged. The target is TNT's token-level I-projection (Zhang et al. 2024, see §5)
transplanted to MDM states; what is new for MDMs is the state set and the leakage bound that justifies it (§2, P2).

---------------------------------------------------------------------------------------------------------------
## 1. Failure / inefficiency of current practice

Notation as in §2. All numbers marked [tab] are exact from `A_scripts/tabular_check.py` (zero parameter sharing),
[share] from `A_scripts/sharing_check.py` (log-linear shared-parameter mask predictor, convex fits). Toy: V=8, L=4,
forget prompt "When was X born?" → y* = `X born 1951 EOS`, fact token 1951; retain prompts share the template (R1),
the subject X (R2, "Where was X born?") and the answer token 1951 (R3, another person born 1951).
Six native configs C1–C6 (greedy/T=0.5/T=1 × low_confidence/random × 1 or 2 tokens/step × block 4 or 2).

**F1 — GA/NPO/SFT-weighted forget objectives spend ~1/L of their weight on leakage-relevant terms (derivation).**
With the LLaDA SFT weighting (CE/p_mask, iid masking at p_mask≈t, t~U), every answer position j gets expected
weight E[1{j∈M}/t]=1. The only terms that can change generated-answer leakage of a fact-token set occupying
positions P_F (|P_F|=k) are fact positions at states with all of P_F masked (P2 below): weight
Σ_{i∈P_F} E[t^{k}/t] = k·(1/k) = 1, i.e. a fraction **1/L of the total for any k** (L = answer length incl. EOS
padding; 0.8% at L=128). The remaining 99% (template, subject restatement, EOS padding, states where the fact is
already visible) is pure collateral pressure: GA's logit gradient at θ0 is (p−e_{y*_j}) at *every* masked position,
so EOS/template positions receive the full push (in [share] GA collapses: λ≥1 gives junk=1.00, X-kept 0.00, and the
collapsed model emits 1951 at other positions → max-leak 1.00 at every λ). The 1/t weight also makes the forget
gradient heavy-tailed: per-position second moment ∫_ε^1 dt/t = ln(1/ε) ≈ 6.9 (ε=1e-3), half of which comes from draws
with t<√ε≈0.032 (3% of samples, ≥97% of the answer visible).

**F2 — MDU's prompt-masked anchor still predicts the fact when visible answer tokens identify the subject.**
The anchor p_θ0(·|m, y_t) sees y_t. [tab] With the prompt-masked conditional = Bayes mixture over prompts, the anchor's
p(1951) at the fact position is 0.21 on the all-mask state but **0.44–0.46 once the subject token X is visible**
(reference: 0.88–0.92). MDU τ=1 target model leaks 1.00 under C3 (greedy, semi-AR block 2) and 0.41–0.57 under the
T>0 configs; τ=0.5 likewise (C3 1.00). [share] MDU τ=1 never gets max-leak below 1.00 for λ∈[0.03,100] (C3=1.00,
C2≈0.46), with X-kept falling 0.97→0.76 and junk 0.01→0.12. The paper's own Table 6a (PDF text line ~975–977) shows
τ=1 → "…born on October 26, 1951." (GT year kept) and τ=0.5 → "R theev Majumdar was born,19" (partial year prefix).
Restricting MDU to fact-free states does not help ([share] MDU_tau1_ffTF/OP: max-leak 1.00 at all λ): **the anchor
target, not the state set, is the binding problem**.

**F3 — Teacher-forced states cover only y*'s state manifold, and a target that succeeds leaves it.**
[tab] Even the *exact* I-projection target q* = p0(·|no 1951), imposed exactly on all 8 fact-free teacher-forced (TF)
states, leaks 0.25 (C2, T=1 random) / 0.12 (C5) / 0.06 (C4): q* correctly redirects to alternatives
("X born 1952 Paris"), committing tokens not in y* (Paris at position 4), after which the model is at an untrained
state where the reference still says 1951. The same target on *all* fact-free states leaks 0 under all six configs.
Q1 diagnostics confirm the decode-time state at which the reference commits the fact is sampler-dependent:
C1 (greedy low-conf) commits 1951 **last** (3 tokens visible) w.p. 1; C2 spreads it uniformly over 0–3 visible;
C4 commits it with 2 visible w.p. 0.87; under C2, 0.10 of the 0.87 reference leakage (12%) is already committed at
non-TF states (C4/C5: <2%).

**F4 — Flattening targets (ME, τ<1 tilts) do not flip ranks, so greedy decoding still leaks.** [share] ME reaches
X-kept 0.03 and junk 0.42 yet greedy configs C1/C3/C6 leak 1.00 at every λ (the fact remains the arg-max of a
near-uniform distribution). Greedy leakage is governed by rank, sampling leakage by mass; a target with
q(fact)=0 controls both.

---------------------------------------------------------------------------------------------------------------
## 2. Mathematical formulation

- Prompt c (never masked); answer region z ∈ (V∪{m})^L of fixed length L (gen_length; EOS padding is ordinary
  vocabulary). State s=(c,z); M(s)={j: z_j=m}. Mask predictor: p_θ(·|s,j), j∈M(s). Reference p_0 := p_θ0 (frozen).
- Native sampler D=(steps, block_length, remasking∈{low_confidence,random}, T, cfg_scale=0): per step x0_j ~ π_T(·|s,j)
  ∝ p_θ(·|s,j)^{1/T} (arg-max if T=0) independently for j∈M(s); commit k_t of the eligible set E_t (masked positions
  of the current block) by confidence or at random; never re-mask. ρ_D^θ = distribution of (s_t, E_t) along it.
- Fact-token set F=F(c,y*)⊂V (operational definition §3); fact event A_F={y: ∃j y_j∈F}; leakage
  Leak_D(θ)=P_D^θ(A_F) (token-level; string-level metrics are reported separately, §6).
- **Fact-free state**: no visible z_j∈F. Write S_ff for the set.
- **Exact I-projection** (for a consistent p_0 with joint p_0(y|c)): q*(y|c)=argmin_{q:q(A_F)=0} KL(q‖p_0)
  = p_0(y|c)1[y∉A_F]/p_0(A_F^c|c). Its conditionals q*(x_j|s)=p_0(x_j,A_F^c|s)/p_0(A_F^c|s) exist iff
  p_0(A_F^c|s)>0 and are **undefined on states with a visible fact token (q*(s)=0)** — [tab] q*-mass of the three
  fact-visible TF states checked = 0.0. Hence the forget term is defined only on S_ff (Q3).
- **Mean-field exclusion target (used):** for s∈S_ff, j∈M(s):
  q̂_j(v|s) = p_0(v|s,j)·1[v∉F] / (1−p_0(F|s,j)).
  q̂_j=q*_j when, given s, the other masked positions' fact mass does not depend on x_j (e.g. it is ≈0); it is the
  TNFF target of Zhang et al. applied to MDM conditionals. q̂ is one reference pass, always a valid distribution,
  and needs no joint. The family {q̂_j} is generally **not** consistent ([tab] pairwise-violation 1.06–1.28 vs 0.00
  for q*); P2 does not need consistency.
- **Objective** (λ>0, 𝒟 a finite pool of decoding configs, ρ_TF = TF fact-free states):
  L(θ) = λ·E_{s∼½ρ_TF+½ρ_OP(θ̄)} [ (1/|E(s)|) Σ_{j∈E(s)} KL(q̂_j(·|s) ‖ p_θ(·|s,j)) ] + L_retain(θ),
  ρ_OP(θ̄)=mixture over D∈𝒟 of ρ_D^{θ̄} restricted to S_ff, θ̄ = stop-gradient copy (DAgger-style; refreshed every K
  steps); E(s)=M(s) for TF states, E_t for on-policy states. ρ_TF: t~U(0,1), mask every non-fact answer position
  iid w.p. t and **every position with y*_j∈F always**. No 1/t weight on the forget term.
- At θ=θ0 the forget loss equals Σ −log(1−p_0(F|s,j)) (a calibrated leakage proxy) and its logit gradient is
  p_0−q̂ = p_0(F|s,j)·(e_F-part − q̂): **zero wherever the reference puts no mass on fact tokens** (EOS padding,
  template), so token selectivity is automatic; contrast GA (F1).

**P1 (no-leak support, exact).** If p_θ's conditionals equal q̂ (or q*) on every fact-free state reachable by D, then
Leak_D=0 for every D with T≥0, any steps/block/remasking: the first fact commit would have to occur at a fact-free
state where the committed token has probability 0 (T>0) or is not the arg-max (T=0, ties aside).
[tab] FFX on all fact-free states and q* on all fact-free states: leakage 0.000 under all six configs.

**P2 (leakage bound for the student; union bound, proof in Appendix A1).** For any D∈𝒟 with T>0, cfg_scale=0:
Leak_D(θ) ≤ E_{τ∼D,p_θ} Σ_{t<t*} Σ_{j∈E_t} π_T(F|s_t,j), t* = first fact commit; for remasking='random' E_t may be
replaced by the committed set. For T=1, π(F|s,j) ≤ 1−exp(−KL(q̂_j‖p_θ(·|s,j))) ≤ KL(q̂_j‖p_θ) because
KL(q‖p) ≥ −log p(supp q). For T=0, no fact commit at (s,j) if TV(q̂_j,p_θ(·|s,j)) < ½·max_v q̂_j(v) (Pinsker:
TV ≤ √(KL/2)). So the on-policy forget loss (with E(s)=E_t and a sum over the trajectory) *upper-bounds* T=1 leakage
for the configs in 𝒟. Scope (not overclaimed): only configs in 𝒟, only prompts whose states were trained, no CFG
(CFG extrapolation (s+1)(cond−uncond) can amplify residual fact mass), token-level event only. It says nothing
about latent knowledge or other prompts/paraphrases; a local constraint does not imply global forgetting/utility.

---------------------------------------------------------------------------------------------------------------
## 3. Executable update (per optimisation step)

```
given: forget batch {(c, y*)}, retain batch, frozen reference θ0 (LoRA: reference = adapter disabled),
       student θ, fact sets F(c,y*), config pool 𝒟, state buffer B (per forget example), λ, K, n_op
# F(c,y*) (approximation A2): token ids in y* that are (a) not in the prompt, (b) not EOS/EoT/special/punctuation,
#   (c) not among the top-N most frequent tokens of the retain answers; plus the ids of the same strings with/without
#   leading space and their digit/sub-word re-tokenisations (tokenizer.encode of variants; first token of each).
every K steps, for each forget example (no grad, student θ̄=θ):
    D ~ Uniform(𝒟)                                   # e.g. {(128,32,low_conf,0), (64,128,low_conf,0),
                                                     #       (128,32,random,0.7), (128,128,low_conf,1.0)}
    run LLaDA generate(D) with θ̄ from the all-mask answer; at every step t record (s_t, E_t) if s_t ∈ S_ff
    stop at the first step whose committed x0 contains a token in F (keep that pre-commit state, flag=leak)
    B[(c,y*)] ← sample n_op recorded states (always keep leak-flagged ones)
for each forget example:
    # teacher-forced fact-free state
    t ~ U(0,1); M = {j: y*_j ∈ F} ∪ {j: u_j < t};  s_TF = (c, y* with M masked)
    S = {s_TF} ∪ B[(c,y*)]                             # each with its E(s)
    for s in S (batched):
        with no_grad: l0 = θ0(s).logits;  l0[..., F] = -inf;  q̂ = softmax(l0)            # 1 frozen fwd
        l = θ(s).logits                                                                    # 1 student fwd
        L_f += mean_{j∈E(s)} [ -Σ_v q̂_j(v) log_softmax(l_j)(v) ]      # CE to soft target = KL + const
L = λ·L_f/|S_total| + L_retain   (L_retain = LLaDA SFT loss on retain batch, as in MDU, λ_retain=1)
AdamW step (clip 1.0); log L_f and the per-config leak-flag rate of the rollouts (free diagnostic).
```
Approximations, all explicit: (A1) mean-field q̂ instead of exact q* conditionals (exact needs joint fact mass;
rejected variants in App. A3); (A2) heuristic token-level F, so string-level leakage through unlisted
tokenisations is possible; (A3) on-policy states are constants (no gradient through the sampler), stale ≤K steps;
(A4) bound P2 only for 𝒟 and cfg_scale=0; (A5) forward KL chosen (mass-covering, finite with q̂(F)=0); reverse KL
(TNRR) is a variant; (A6) whether the HF `modeling_llada.py` exposes logits over the full vocab incl. special ids
is UNVERIFIED here (HF blocked) — F excludes specials, so only the fact ids are set to −inf.

---------------------------------------------------------------------------------------------------------------
## 4. Predicted advantage (one) and its measurement

**Primary: higher retain-set knowledge / general utility at matched generated-answer forgetting, where forgetting
is the max over ≥3 native decoding configs.** Mechanistic reasons: (a) gradient vanishes where the reference has no
fact mass (no EOS/template pressure, F1); (b) bounded fixed point equal to the reference's own alternatives (minimal
conditional change); (c) rank of the fact is flipped, so greedy configs are covered (F4); (d) on-policy states
cover the alternative continuations the unlearned model actually takes (F3).

Toy evidence [share] (6 configs, matched on max-leak): only FFX reaches max-leak ≤0.10 — FFX-TF λ=30: 0.094,
retain decode-TV 0.028; FFX-OP λ=10: 0.079, TV 0.030 — and only FFX-OP reaches ≤0.05 (λ=30: 0.038, TV 0.040,
X-kept 0.93, junk 0.05). GA, ME, MDU τ=1 (TF, ff-TF, or on-policy states) are **infeasible even at ≤0.30**
(greedy configs stay at 1.00). Single-config matching (C2, T=1 random, leak≈0.42–0.46): FFX λ=0.3 retain-TV
0.010–0.011, X-kept 0.93 vs MDU τ=1 λ=100 TV 0.043, X-kept 0.76; ME at C2≈0.39: TV 0.21, X-kept 0.03.
Cost exposed honestly: R3 (retain fact = same token 1951) recall drops 0.95→0.87 at max-leak 0.038; on-policy
states give only a modest gain over TF in the shared model (0.079 vs 0.094 at comparable λ; feasibility at 0.05).

Hyperplane measurement (TOFU forget10 on the authors' LLaDA-8B-Instruct SFT checkpoint if obtainable, else our own
SFT): forget metric = fact-token recall + ROUGE-L recall + exact substring match of the GT answer span, max over
G1=(128 steps, gen 128, block 32, low_conf, T=0), G2=(64 steps, block 128, low_conf, T=0), G3=(128, block 32, random,
T=0.7), plus one held-out config not in 𝒟 (e.g. 32 steps, block 8). Sweep λ (or steps) per method; pick for each
method the earliest checkpoint with max-config forget ROUGE-L ≤ a common threshold; compare retain ROUGE-L / MC-prob,
real-authors, world-facts, and a general benchmark. Arms: GD, NPO, MDU τ∈{0,0.5,1}, FFX-TF, FFX-OP; report
infeasible arms rather than unmatched comparisons. Cheap zero-training diagnostic first: anchor p_θ0(fact|m,y_t) vs
p_θ0(fact|c,y_t) on fact-free TF states of forget10 (predicts MDU τ=1 under-forgetting where the ratio is high).

---------------------------------------------------------------------------------------------------------------
## 5. Closest known method (what was checked, how)

- **TNT / TNFF (Zhang, Ranganath, Tafvizi, TMLR 2024, arXiv 2406.13660)** — closest for the *target*. Paper:
  SNIPPET-ONLY (I-projection "closest in KL with zero probability on unwanted outputs"; negatives from the model's
  own generations). Code **read**: `github.com/google/t5patches` @ 09285b3, `t5patches/models.py` L565–641
  (`TNFFModel._get_desired_dist`: original softmax with the negative token's probability zeroed and renormalised;
  CE to it, weight α on negative positions; L644+ TNRR = reverse-KL variant). AR next-token conditionals at
  annotated negative token positions. **FFX's target is TNFF's target**; FFX differs only in MDM state set
  (fact-free arbitrary-visible-subset states, exclusion at every eligible masked position, on-policy MDM rollouts
  under a config pool) and the MDM leakage bound P2.
- **MDU (Lee et al., arXiv 2605.18253)** — PDF read (§4 Eqs. 4–6, §6.3 convergence, Table 5, Table 6a); code read
  `/tmp/claude-0/src/MDU/src/unlearn_mdu_llada.py` L947–1003 (`_null_anchor_kl`, τ·logits, reverse-in-my-notation
  KL(p_c‖target) default), L1102–1147 (random TF states, all masked answer positions incl. EOS pad, anchor = ref with
  prompt masked), **L1005–1100 `_traj_rollout_na_loss`: on-policy rollout states already exist in MDU's code**
  (current model, greedy low-confidence top-k only, random stop step, anchor target, no fact-free restriction, not
  in the paper's main method). So on-policy MDM states are not new; the fact-free restriction, exclusion target,
  and multi-config pool are the differences.
- UNDIAL (arXiv 2402.10052; SNIPPET-ONLY): self-distillation with the target token's logit reduced by γ at every
  forget-sequence position; FFX ≈ UNDIAL with γ=∞ restricted to fact tokens on fact-free MDM states.
  Unilogit (Findings ACL 2025; SNIPPET-ONLY): target token set to uniform probability, reverse KL.
- ME / MDU τ=0 (uniform), WHP (generic predictions from a reinforced model; NOT ACCESSED), AltPO (alternative
  answers; NOT ACCESSED — my rejected counterfactual-fill variant is its one-token MDM analogue).
- Adversarial-RL dLLM unlearning (ICML 2026 poster; SNIPPET-ONLY): reports sensitivity of dLLM unlearning to
  generation hyper-parameters and context-embedded elicitation — consistent with F3/F4; method differs (context
  generator). TILDE (2607.06432), Temper-Then-Tilt (2602.10217), unified-KL diffusion unlearning (2605.30825):
  SNIPPET-ONLY, image-diffusion KL-projection/tilt targets.
- Searches used: 5 WebSearch queries by operation (renormalised self-distillation target; MDM unlearning on-policy
  trajectory; minimal-KL zero-mass projection; TNT token-level; dLLM unlearning decoding robustness). Absence of
  hits is **not** a novelty claim; D should verify TNT's paper text and any MDM follow-ups.

---------------------------------------------------------------------------------------------------------------
## 6. Simple conditions under which it fails or becomes redundant

1. **Redundant with TNFF-at-TF-states** if the network generalises from teacher-forced fact-free states to its own
   decode states (then FFX-OP ≈ FFX-TF; [share] gap was small: 0.079 vs 0.094). Falsifier: FFX-OP vs FFX-TF at
   matched retain shows no forget difference on held-out configs.
2. **Redundant with MDU** if, on real data, the anchor rarely predicts the fact (anchor/conditional fact-prob ratio
   ≈0 on forget10) — then MDU τ≈1 already has q(fact)≈0 and the difference reduces to EOS/template handling.
3. **Deep memorisation (π_f=p_0(F|s,j)→1 at all fact-free states):** q̂ is the reference's renormalised *tail*,
   poorly estimated and possibly EOS, a sub-piece of the fact ("19"), or junk → truncation-like behaviour or
   string-level leakage via unlisted tokenisations (MDU Table 6a τ=0.5 "born,19" is exactly this pattern). The
   result is "the reference's second guess" (a confabulated alternative), not removal.
4. **Fact tokens shared with retain answers** (dates, months, common names): exclusion is local to the forget prompt
   but shared parameters move retain prompts ([share] R3 recall 0.95→0.87). Fails badly if F must contain
   high-frequency tokens.
5. **Paraphrased prompts / other surface forms / CFG decoding / configs ∉ 𝒟**: outside P2's scope; may still leak.
6. Multi-token facts with shared prefixes: the union event A_F is stronger than needed (forbids each piece
   anywhere); the exact joint event would need joint fact mass (App. A3), not computed.
7. Rollout cost dominates if K is small; if rollouts are skipped, FFX-OP reduces to FFX-TF (= TNFF on MDM TF states).

---------------------------------------------------------------------------------------------------------------
## 7. Numerical evidence

Scripts/outputs (run via ledger, agent A, 1 thread): `$R/agents/A_scripts/tabular_check.py` → `tabular_check_out.json`
(6 s); `$R/agents/A_scripts/sharing_check.py` → `sharing_check_out.json`, `sharing_check_stdout.txt` (206 s).
Exact enumeration of `generate` semantics (Gumbel arg-max = tempered sampling, untempered confidence, top-k inside
block, uniform k schedule, no re-mask). Tie-break lower index (torch.topk order unspecified — stated).

**(a) Leakage of the target model** [tab] (target on its training states, reference elsewhere), configs C1..C6:

| target (states) | C1 | C2 | C3 | C4 | C5 | C6 | non-fact rowTV vs ref | consistent? |
|---|---|---|---|---|---|---|---|---|
| reference | 1.00 | .87 | 1.00 | .91 | .99 | 1.00 | 0 | yes |
| GA-like β=3 (all TF) | 1.00 | .63 | 1.00 | .56 | .90 | .67 | .42 | no (.94) |
| ME = MDU τ=0 (all TF) | 0* | .42 | 0* | .41 | .43 | .17 | .79 | no |
| MDU τ=1 (all TF) | 0 | .53 | 1.00 | .41 | .57 | .33 | .18 | no (.88) |
| FFX mean-field (ff-TF) | 0 | .22 | 0 | .003 | .015 | .17 | .019 | no (1.28) |
| counterfactual-fill = Bayes-excl. (ff-TF) | 1.00 | .37 | 0 | .14 | .36 | .50 | .20 | no |
| exact q* (ff-TF) | 0 | .25 | 0 | .06 | .12 | .17 | .15 | no (.82) |
| FFX mean-field (all fact-free) | 0 | 0 | 0 | 0 | 0 | 0 | .019 | no (1.06) |
| exact q* (all fact-free) | 0 | 0 | 0 | 0 | 0 | 0 | .15 | **yes (0.000)** |

*ME greedy: uniform ties → EOS arg-max → junk 1.00. ME's T>0 leak includes the V=8 chance level 1−(7/8)^4=0.41.
(b) Distortion: junk (outside answer grammar) under C2: FFX-MF .036, q* .089, MDU τ=1 .30, ME .77; P(subject X
kept) C2: FFX .75–.79, q* .72–.77, MDU .47, ME .20. Retain distortion needs sharing → [share] numbers in §4.
(c) q* is the only consistent family; CF-fill and Bayes-exclusion coincide exactly on the consistent reference
(sanity check of the identity) but redirect continuations off the TF manifold and leak more than mean-field.
**Inconsistent reference** (per-(state,pos) log-normal noise σ, TF pairwise violation .35 / 1.47):
Bayes-exclusion negative mass before clipping mean/max = .037/.30 (σ=.3), **.47/3.42 (σ=1)**; non-fact rowTV to
the clean q*: MF .14/.17, CF .09/.12, Bayes .16/.29 (σ=.3/1) vs perturbation size .03/.13. Bayes-exclusion is
ill-posed on real (inconsistent) conditionals; CF is robust but costs a second reference pass and does not
reduce leakage in the toy. Leakage with perturbed references stays driven by off-manifold states (F3).
**Q1 diagnostics** (reference): first fact commit at (TF?, #visible): C1 (TF,3) 1.00; C2 ≈uniform over 0–3 visible, 0.10
(abs.) at non-TF; C4 (TF,2) .87.

---------------------------------------------------------------------------------------------------------------
## 8. Cost per optimisation step (LLaDA-8B, per forget example; retain branch identical to MDU/GD)

| method | frozen-teacher fwd | student fwd | student bwd | rollout fwd (no grad) |
|---|---|---|---|---|
| GA / GD | 0 | 1 | 1 | 0 |
| NPO, MDU | 1 | 1 | 1 | 0 |
| FFX-TF | 1 | 1 | 1 | 0 |
| FFX-OP (n_op states, refresh every K) | 1+n_op | 1+n_op | 1+n_op | ≈ steps_D/K per step (e.g. 128/8 = 16 amortised) |

Extra memory: reference copy (16 GB bf16) unless LoRA (reference = adapter disabled, no copy); state buffer
negligible; target needs only an index mask over |F| ids. FFX-TF is MDU-cost; FFX-OP with n_op=1, K=8, 128-step
rollouts ≈ 2 ref fwd + 2 student fwd/bwd + 16 no-grad fwd per example ≈ 3–4× MDU wall-clock (rollouts dominate;
fewer rollout steps, e.g. 32, cut this to ≈1.5–2×). Retain: +1 fwd/bwd (same for all arms).

---------------------------------------------------------------------------------------------------------------
## Appendix — reasoning notes (for later rounds)

**A1 (proof of P2).** Let t* be the first step at which some committed token lies in F. For t<t*, s_t∈S_ff.
A_F = {t*<∞} ⊆ ∪_t ∪_{j∈E_t} {x0_j(t)∈F, t≤t*}. Given s_t, x0_j(t)~π_T(·|s_t,j) independently of the past;
P(A_F) ≤ Σ_t E[1{t≤t*} Σ_{j∈E_t} π_T(F|s_t,j)]. For random remasking the committed set is independent of x0, so
only committed j count. KL(q‖p) = −Σ_{supp q} q log(p/q) ≥ −log Σ_{supp q} p = −log(1−p(F)). T=0 claim:
p(f) ≤ q̂(f)+TV = TV and p(v*) ≥ q̂(v*)−TV, so arg-max ∉F if TV < q̂(v*)/2. ∎
**A2 (why states with visible fact are excluded).** q* has zero mass there (conditionals undefined), and by A1 such
states never host a first fact commit; training them only moves non-fact content (pure distortion risk).
**A3 (rejected targets).** Bayes-exclusion q(x_j|s)=[p(x_j|s)−π_f p(x_j|s,x_i=f)]/(1−π_f): exact for a consistent
p_0 and the single-position event, 2 ref passes, but ill-conditioned (π_f≈0.9 → 10× amplification) and up to 3.4
negative mass on inconsistent conditionals. Counterfactual-fill E_{v~q̂_i}p(x_j|s,x_i=v): always valid; with forward
KL the CE is linear in the target, so one sampled fill gives an unbiased gradient for +1 ref pass — but it moves
mass to alternative continuations, which (F3) leaves the trained manifold; it is the one-token MDM analogue of
AltPO. Exact q* for multi-token A_F needs p_0(A_F|s) by chain rule (|P_F| passes). None beat mean-field on leakage
in the toy; mean-field kept as default. MDU anchor with visible answer tokens: rejected (F2).
**A4 (what would change my mind).** (i) Real forget10 anchor diagnostic shows anchor fact-prob ≪ conditional →
F2 weak, MDU τ=1 competitive. (ii) FFX-OP ≈ FFX-TF on held-out configs → drop rollouts (cost ×1). (iii) String-level
recall stays high while token recall →0 → F definition inadequate; switch to span-level event with chain-rule mass.
**A5 (limits of the toy).** V=8, L=4 inflates chance leakage of uniform targets; the log-linear predictor is far
less expressive than an 8B transformer, so "infeasible" for GA/ME/MDU may not transfer; zero-sharing table is the
pessimistic no-generalisation case. Both are evidence about targets/states, not about LLaDA-scale outcomes.
**A6 (expected objections).** "This is TNT" — yes for the target; the MDM contribution is the state set + bound,
which must earn its keep empirically (§6.1). "On-policy states are in MDU code" — yes (L1005–1100); differences are
fact-free restriction, exclusion target, config pool. "Token-level F is brittle" — conceded (§6.3–6.4).
