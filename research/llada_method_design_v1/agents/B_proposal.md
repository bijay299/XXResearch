# B — First-Commit-Hazard unlearning on a native-sampler mixture (FCH-Mix)

Agent B (denoising-process designer), wave 1. Scripts/outputs: `$R/agents/B_scripts/`. All numbers below are from an
**exact tabular model** (no neural training); they establish a mechanism and a failure mode, not an LLaDA-8B result.

## 1. Failure / inefficiency of current practice

**Claim.** For a masked diffusion LM decoded natively (commit-without-remask), leakage of a forgotten fact is a
*first-passage event*: it happens iff some fact token is committed, and the first such commit is always made from a
**fact-free decision state** that the *deployed sampler* reached with the *current (unlearned) model's own commits*.
Teacher-forced (TF) training states — MDU's y_t ~ q(·|y,t), i.e. the ground-truth answer y with a random subset masked
(MDU code `_random_sft_null_anchor_loss`, L1102–1147) — only contain fact-free states whose visible tokens equal y.
After suppression the model commits *alternatives* (another surface form, another position for the fact, or
incoherent pairs from parallel commits), which moves it off the GT-masked manifold, where nothing was trained.

Exact counterexample (§7, unigram-backoff world, 33 native configs, budget 400 update steps):
- TF states + minimal suppression target: leak = **0** for greedy low-confidence single block, but **0.50** for
  `random` remasking T=0.5 and **0.33** for `random` greedy; plateau, i.e. *infeasible at any budget* (support issue):
  **99.5 %** of its residual leak (summed over configs) is committed from states outside supp(TF).
- TF + all fact-bearing paraphrase templates of the data distribution ("augment with paraphrases"): plateau at
  leak_max **0.23** (semi-AR block 1 with T=0.5; 2-token-per-step schedules).
- On-policy rollouts of the current model under ONE config (greedy, low-confidence, 1 block — the analogue of MDU's
  `_traj_rollout_na_loss`, L1005–1100): its own config → 0, but leak_max = **1.0** (greedy with block_length 1 or with
  2 tokens/step).
- On-policy rollouts with random order and T=1 but one token per step: residual leak concentrates on
  **2-tokens-per-step** schedules (leak 0.06 at steps=2, block=2): parallel commits draw tokens independently from
  per-position marginals and create product-of-marginals states that sequential rollouts rarely visit.
External, SNIPPET-ONLY: the Adversarial-RL dLLM-unlearning abstract (ICML 2026 poster 65904) reports that DLM
unlearning performance is "highly sensitive to generation hyperparameters". MDU's own limitation statement (paper §
limitations, per SHARED_CONTEXT §4) is the same mismatch.

## 2. Mathematical formulation

Prompt c (never masked); answer region of length L (= gen_length, EOS-padded); vocabulary V; mask m. A state is
s ∈ (V∪{m})^L, M(s) its masked positions. Student p_θ(·|c,s,i) = softmax z_θ(c,s,i), i∈M(s); frozen initial model θ0.
Fact set F = F(c,y) ⊂ V (content tokens of the forget answer; approximation, §3). s is *fact-free* if no visible
token is in F. Leak event Λ = {some committed token ∈ F} (a span-level leak implies Λ, so P(span leak) ≤ P(Λ)).

**Native sampler class.** π = (N steps, block b, rule ∈ {lc, rnd}, temperature T), cfg_scale 0 (generate.py
@9182493). Step n acts on block β(n); eligible set E_n(s) = M(s) ∩ β(n); k_n from the uniform schedule. Draws
x0_i ~ q_T(·|s,i) = softmax(z/T) independently over i (Gumbel-max; q_0 = argmax), selection S_n ⊂ E_n of size k_n =
top-k of p_θ(x0_i|s,i) (lc) or uniform and independent of the draws (rnd); commit, never remask.
Define ρ_θ(s,i) = p_θ(F|s,i) / max_v p_θ(v|s,i), Hρ(s) = Σ_{i∈M(s)} ρ_θ(s,i), and the fact-free visitation
d^ff_π(s) = Σ_n P_π(s_n = s, no leak before n) (mass ≤ N).

**Proposition 1 (first-commit decomposition + sampler-uniform hazard bound).**
(a) P_π(Λ) = Σ_n E_π[1{no leak before n} h_n(s_n)], h_n(s) = P(∃ i∈S_n: x0_i∈F | s) (exact; disjoint first leaks).
(b) h_n(s) ≤ Σ_{i∈E_n(s)} q_T(F|s,i) for both rules (a selected F draw requires an eligible F draw — valid although
lc selection depends on the draws); for rnd, h_n ≤ (k_n/|E_n|) Σ_{i∈E_n} q_T(F|s,i).
(c) Lemma: for 0 ≤ T ≤ 1, q_T(F|s,i) ≤ ρ_θ(s,i). (T=0: argmax∈F ⇒ ρ≥1. T∈(0,1], β=1/T≥1:
q_T(f) = p_f^β/Σ_v p_v^β ≤ (p_f/p_max)^β ≤ p_f/p_max.)
(d) Hence for **every** π ∈ Π_{≤1} = {any N, b, lc or rnd, T∈[0,1], cfg 0} with this gen_length:
  P_π(Λ) ≤ E_{s∼d^ff_π}[Hρ(s)] ≤ (Σ_n |E_n|) · sup_{s∈supp d^ff_π, i∈M(s)} ρ_θ(s,i);
  and if ρ_θ < 1 (fact never the argmax) on supp d^ff_π, greedy decoding leaks with probability exactly 0.
(e) Coverage: for any training measure μ on states, P_π(Λ) ≤ C_{π|μ} E_μ[Hρ], C_{π|μ} = sup_s d^ff_π(s)/μ(s).
  For μ = Σ_k w_k d^ff_{π_k} (on-policy mixture), C_{π_k|μ} ≤ 1/w_k. For μ = TF states, C = ∞ as soon as π reaches a
  fact-free state off the GT-masked manifold (the §1 counterexample).
**Honesty.** d^ff_π depends on θ: the bound certifies the θ at which d^ff is measured (fresh rollouts give a Monte
Carlo certificate), it is not a property preserved by later updates. It says nothing about states no π∈Π reaches with
θ: prefix/partial-answer injection, re-masking samplers or attacks (TrajHijack, SNIPPET-ONLY), T>1, cfg>0 (where ρ must
be computed from the CFG-combined logits), other gen_lengths, or paraphrases outside F. It is a reduced-recall
(type i) statement only — no claim about removal of training influence.

**Mechanism.** Minimise the bound's integrand on the measure that makes (e) finite for the deployed configs:
  L_forget(θ) = E_{s∼μ_θ} [ (1/|M(s)|) Σ_{i∈M(s)} KL( r_ε(·|c,s,i) ‖ p_θ(·|c,s,i) ) ],
  μ_θ = (1−α) Σ_k w_k d̄^ff_{π_k,θ} + α · TF_F (GT states with all F positions force-masked),
with d̄ = normalised visitation of the CURRENT student, Π_train = {deployment default (greedy lc); wide-sequential
(rnd, T=1, 1 token/step); wide-parallel (rnd, T=1, ≥2 tokens/step)}, and the **minimal-edit target**
  r_ε = argmin_{q: q(F)≤ε} KL(q‖p_θ0(·|c,s,i)) = softmax(z_θ0 − γ·1_F), γ = log[p0(F)(1−ε)/(ε(1−p0(F)))]_+
(closed-form I-projection of the frozen model **at the same state**). Pinsker: KL(r_ε‖p_θ) ≤ κ ⇒ p_θ(F) ≤ ε+√(κ/2),
so small loss ⇒ small ρ (surrogate, not equality). Total loss L = L_forget + λ·L_SFT(retain) (LLaDA SFT loss, as MDU).
Retention lever: at every visited state only the F-logits move relative to θ0 (non-F ranking and EOS/template
predictions untouched); states with p0(F) ≤ ε get target = θ0, i.e. the on-policy loss doubles as a
retain-on-own-trajectory regulariser. States visited only after a leak are never trained (irrelevant by (a)).

## 3. Executable update (pseudo-code)

```
inputs: student θ (LoRA or full FT), frozen θ0 (= adapter-off pass for LoRA), forget set {(c,y)}, retain set,
        Π_train = [(N0,b0,'low_confidence',0.0), (L,L,'random',1.0), (L/4,L/4,'random',1.0)], w = (1/3,1/3,1/3),
        eps=1e-3, alpha=0.2, J=8 states/rollout, R=4 (pool refresh), lam=1, AdamW, clip 1.0
F(c,y) := token ids of y that are not in c, not special/EOS, not in a stop-list        # APPROXIMATION (§6)
every R steps:                                                                           # rollout pool, no grad
  for (c,y) in forget batch: k ~ w; x = all-mask answer
     run pinned generate() semantics with θ under Π_train[k]; before each step record s_n;
     at the first step whose committed set contains a token of F: record s_n (decision state) and STOP
     pool += J states drawn uniformly from the recorded ones (always keep the decision state if any)
step:
  S = (1-alpha)·B states from pool  ∪  alpha·B TF states (t~U(0,1); mask y iid w.p. t; then mask every F position)
  for (c,s) in S:
     with no_grad: z0 = θ0(c,s)[M(s)]; p0F = softmax(z0)[:,F].sum(-1)
                   γ = clamp(log(p0F(1-eps)) - log(eps(1-p0F)), min=0); r = softmax(z0 - γ[:,None]·onehot(F))
     z = θ(c,s)[M(s)];  Lf += mean_i  Σ_v r_iv (log r_iv - log_softmax(z_i)_v)
  L = Lf/|S| + lam · LLaDA_SFT(retain batch)  ;  backward ; AdamW
eval/certificate: for each config π in the eval grid, fresh rollouts of θ: report leak rate AND
  Ĉ_π = mean over rollouts of Σ_n Σ_{i∈E_n} ρ_θ(s_n,i) up to the first F-commit (upper bound on P_π(Λ) by Prop 1d).
```
Approximations: token-set F (misses paraphrases in other tokens); loss over all masked positions (superset of
eligible ones, covers every block layout); pool states are slightly stale (R steps) — off-policy by ≤R updates;
truncation at first F-commit biases d toward early states when leakage is frequent (variant: ban F at commit time
and re-weight by survival — exact for greedy and for rnd, approximate for lc with T>0; not used in §7).
Optional (tested in §7, *not* default): hazard prioritisation μ ∝ d^ff·(Hρ+0.01) — faster forgetting, larger footprint.

## 4. Predicted advantage (primary: retention at matched *config-robust* forgetting)

Forgetting is defined as **max over a native-config grid G** (not one config). Prediction: arms whose state
distribution is TF-only (MDU, MDU+I-projection target, TF+paraphrases) or single-config on-policy (MDU rollout flag)
plateau at a higher max-over-G leakage (worst cells: random remasking, T>0, few steps/many tokens per step); reaching
the FCH-Mix level requires over-training (or is infeasible), which costs retention. At matched max-over-G forgetting,
FCH-Mix keeps more retain-set and general utility than (i) over-trained MDU and (ii) a brute-force broad-state arm.
Measurement (Hyperplane, LLaDA-8B-Instruct TOFU-SFT, forget10): G = steps {32,128} × block {32,128} × {lc, rnd} ×
T {0, 0.7}, gen_length 128 (16 cells; Π_train covers 3, the rest held out). Forget metric: fact-token hit rate
(any F token) and ROUGE-L recall vs GT, max over G. Sweep each arm's steps/lr, pick the first checkpoint with
max-over-G hit-rate ≤ target (report infeasible arms as such), then compare retain ROUGE-L, truth ratio, real-authors /
world-facts ROUGE, and a held-out perplexity (get_log_likelihood MC). Secondary: certificate Ĉ_π vs observed leak.

## 5. Closest known method (what was checked)

- **MDU** (code `/tmp/claude-0/src/MDU/src/unlearn_mdu_llada.py`, read L1–60, L316–420, L940–1150, L1334–1400,
  L1455–1490; paper per SHARED_CONTEXT). Default: TF states + KL to prompt-masked anchor (τ). **`_traj_rollout_na_loss`
  (L1005)** is the closest mechanism: on-policy no-grad rollout of the current model, **greedy argmax, low-confidence,
  one block, T=16 steps with ceil schedule, single random stop step, model predictions placed (its help string says
  "option B, GT tokens placed" — the code does option C)**, then null-anchor KL at that state. Differences here: a
  mixture of native configs incl. random order, T=1 and parallel-commit schedules (the §7 numbers show its single
  config does not transfer), truncation at the first fact commit (decision state kept), a same-state minimal-edit
  target instead of the prompt-masked anchor, and Prop. 1 as the criterion. Honest assessment: the *update* is a
  modest generalisation of MDU's own rollout variant; the contribution is the first-commit analysis that says which
  states matter and why a single config is insufficient. MDU's `_denoising_trajectory_weights` (L316) uses a greedy
  rollout to pick *positions* (late-unmasked tokens) for GA, not states.
- UNDIAL (logit-adjusted self-distillation target; memory, NOT re-read): our r_ε is of this type (adaptive γ, F set).
- On-policy forget samples in AR unlearning: "LLM Unlearning with LLM Beliefs" (arXiv 2510.19422, on-policy BS-S
  variant) — SNIPPET-ONLY; leak@k / RULE (arXiv 2511.04934) — SNIPPET-ONLY. DAgger (Ross et al. 2011) and
  concentrability coefficients (Munos; Kakade & Langford) are the standard source of (e) — from memory.
- Adversarial RL for robust dLLM unlearning (ICML 2026 poster 65904): adversarial *context* generator; SNIPPET-ONLY.
- DEMASK / parallel-decoding dependency (product-of-marginals error) — SNIPPET-ONLY; supports the parallel-commit
  finding. TrajHijack ("Re-Mask and Redirect", ICML 2026) — SNIPPET-ONLY; breaks the no-remask assumption.
- A2D (arXiv 2509.23286), TimeROME-DLM, LRCU: NOT ACCESSED. Not a VAU-family method (no Jacobian projection, no gate).

## 6. When it fails or becomes redundant

- **Redundant** if parameter sharing already transfers TF suppression to generated states (C_{π|TF} effectively
  finite): then FCH-Mix ≈ "TF + I-projection target" (arm TF) at higher cost. Cheap control on GPU: MDU vs
  MDU+I-proj vs FCH-Mix at matched max-over-G; if MDU's max-over-G already matches, the coverage argument is moot.
- Redundant for greedy lc single-block deployment only (TF arm already 0 there in §7).
- **Fails** when the fact is expressible outside F (paraphrase tokens, numbers spelled out, sub-token splits), for
  samplers outside Π_{≤1} (T>1, cfg>0 unless included, re-masking samplers, prefix injection, other gen_length), and
  when the wide samplers give a deployed config's decision states tiny mass (C large; slow, bound vacuous).
- Retention is *not* guaranteed and the toy does **not** show a retention win: OP_mix's footprint exceeds TF's (TF is
  cheaper because it does not achieve the forgetting), and on the κ-sharing retain proxy OP_mix is 0.01–0.02 *worse*
  than the cover-all oracle at matched forgetting (on-policy weight concentrates full edits on generic early states
  such as the all-masked answer, which other prompts share). If that transfer is real at scale, FCH-Mix needs the
  retain loss (not modelled in the toy) or a down-weighting of early generic states — open, falsifiable on GPU.
- Reduction: Π_train = {greedy lc, 1 block}, null-anchor target ⇒ MDU's rollout variant; Π_train = ∅, α=1 ⇒ TF+I-proj.

## 7. Numerical evidence (exact tabular)

Scripts: `B_scripts/tabular_commit_hazard.py` (main, 399 s, ledger agent B), `B_scripts/mc_validate_generate.py`
(validation), `B_scripts/summarize.py`. Outputs: `tabular_commit_hazard_out.json`, `_log.txt`, `_summary.txt`,
`mc_validate_out.json`.
**World.** L=4, V={B,I,F,Y,W,U,E} (F = fact, E = EOS) + mask; all 8^4 = 4096 states enumerated. Base model = exact
Bayes posterior of π = 0.98·{BIFE .55 (GT), FWYE .25, YWFE .10, UEEE .10} + 0.02·product backoff (unigram of the
templates, or uniform — sensitivity variant), + fixed 0.02 logit noise. Native sampler reproduced exactly; leak
computed by exact DP with absorption at the first F commit. **Validation:** the DP matches Monte Carlo of the
*pinned* `generate()` (imported read-only, fake model returning the tables; 20k samples × 6 configs): |z| ≤ 1.96.
33 eval configs = (steps,block) ∈ {(4,4),(2,4),(1,4),(4,2),(2,2),(4,1)} × {lc, rnd} × T∈{0, .5, 1}.
**Training caricature.** Per (state, masked pos): p ← p^{1−λ} r^λ, λ = min(100·μ(s)/|M(s)|, 1), 10 steps/round,
μ refreshed each round (on-policy arms), 40 rounds. Target r = I-projection (ε=1e-3) except TF_uniform (MDU τ=0).
Retention proxies: footprint Σ_{s,i} KL(p0‖p_θ) over all states, #edited (KL>1e-3); retainF = recall of F for a
retain prompt (IFEE .7, FEEE .2, UEEE .1) receiving κ=0.5 of the forget edit (prompt-independent sharing model).
Design history (not hidden): v1 mixture of 4 narrow configs left held-out rnd+T configs at 0.09–0.21; v3 (greedy lc +
rnd/T=1 sequential) left 2-tokens/step schedules at 0.06; v4 (reported) adds one parallel-commit config (2,2,rnd,1).
OP_mix trains on 3 configs, 30 held out; the parallel-commit addition was chosen after seeing v3.

(a)+(b) Unigram backoff (uniform-backoff variant in brackets):

| arm (state distribution) | first ckpt leak_max ≤ .05 | final leak mean / max (all 33) | final footprint KL / #edit |
|---|---|---|---|
| TF (MDU states, I-proj target) | infeasible | .148 / .496 [.131/.452] | 51 / 12 |
| TF_uniform (MDU τ=0 analogue) | infeasible | .374 / .620 [.41/1.0] | 54 / 32 |
| TF_para (+ all fact templates) | infeasible | .061 / .232 [.044/.167] | 118 / 29 |
| OP_single (MDU rollout analogue) | infeasible | .417 / 1.000 | 24 / 8 |
| OP_wide (rnd, T=1, 1 tok/step) | infeasible [r22] | .0055 / .060 [.0025/.023] | 553 / 1309 |
| **OP_mix (FCH-Mix)** | **r2: max .048, KL 145, 325 edits** [r2: .038, 132] | **.0015 / .011** (held-out max .011) [.0015/.0099] | 385 / 1116 |
| OP_mix_hz (hazard-prioritised) | r2: max .015, KL 265 | .0011 / .004 | 782 / 1344 |
| cover_all (uniform over all fact-free states; oracle, impossible at scale) | r4: max .033, KL 1238 | .0010 / .004 | 1392 / 1344 |

At (roughly) matched forgetting, round 4: OP_mix leak mean/max .0033/.0265 with footprint **183** (472 edits) vs
cover_all .0061/.0333 with footprint **1238** (1344 edits) — **6.8× smaller edit** [uniform variant: 159 vs 766, 4.8×].
**But the κ-sharing retain proxy does not favour FCH-Mix**: retainF .835 vs .857 (cover_all) [.874 vs .885]; base .927.
On-policy weighting concentrates full-strength edits on high-traffic generic states (all-masked, `___E`), which in this
prompt-independent sharing model are exactly the states the retain prompt shares. So the toy supports *feasibility of
config-robust forgetting* and *smaller edit footprint*, not a retention win; retention must be tested on GPU.
TF_uniform's leak is a 7-token-vocab artefact (uniform puts 1/7 on F); not transferable to |V|=126k.

(c) Bound tightness (exact leak vs B1 = Σ_n E[Σ_{E_n} q_T(F)], B2 = ρ-form valid for all T ≤ 1, B3 = sup form):
- **Tight** when every eligible position is committed (one-shot N=1, or 2 tokens/step in a 2-block): B1/leak = 1.00–1.03
  (base and trained models).
- **Loose ×2** for `random` remasking (k/|E| factor dropped) — e.g. TF model N4/b4/rnd/T.5: .496 vs B1 .867.
- **Loose ×9** for low-confidence with T=1 after training: OP_mix N4/b4/lc/T1: leak .0010, B1 .0094, B2 .016 —
  low-confidence F draws are rarely selected; the union bound over eligible positions ignores that.
- **Sup form B3 is vacuous (=1) for every T>0 config** (T>0 reaches untrained off-manifold states with tiny probability);
  it is non-vacuous only for greedy: B3 = .028–.14 with ρ<1 on the reachable set ⇒ certified **exactly 0** greedy leak
  for all 6 greedy schedules × both remasking rules (OP_mix), matching the exact DP. The expectation form (B2) is the
  usable certificate; it is within 1–16× of the exact leak for the trained models.
- Base model: B1 = 2.0–2.5 > 1 (vacuous) because several positions carry F at once.

## 8. Cost per optimisation step

Unit = one LLaDA-8B forward on (|c|+gen_length) tokens; backward ≈ 2 units. Per forget *training state*:
- MDU default: student fwd+bwd (3) + frozen anchor fwd (1) = **4**. MDU rollout flag: + ≈T/2 = 8 no-grad fwds ⇒ ≈12.
- FCH-Mix: student fwd+bwd (3) + frozen θ0 fwd at the same state for r_ε (1) = 4, plus amortised rollouts: one rollout
  of config k costs N_k no-grad fwds (Π_train with gen 128: N = 128, 128, 32 ⇒ mean ≈ 96; truncation at the first
  F-commit makes early rollouts shorter); J = 8 states/rollout, pool reused R = 4 steps ⇒ 96/32 = **3** ⇒ total ≈ **7**
  (≈1.75× MDU, ≈0.6× MDU-rollout). J=16 or R=8 ⇒ ≈5.5; using steps 64 for the sequential configs halves rollout cost
  (but is itself a 2-token/step schedule). The α = 0.2 TF states cost 4 each. Retain SFT batch as MDU (3/state).
- Certificate Ĉ_π costs nothing extra (ρ from the rollout's own logits); eval leak grids cost N fwds per sample.
- Memory: frozen θ0 as MDU's `ref_model` (or LoRA adapter-off pass, no copy); pool stores token ids only
  (B·J·(|c|+L) ints); no per-state logit caching (|V| = 126k).
