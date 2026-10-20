# HYPERPLANE_PLAN — integration instructions, dependencies, and one bounded falsification experiment

Status: **real-model execution UNTESTED.** Nothing here has run on LLaDA-8B. The integration code was dry-run only
against a tiny random model on CPU (`tests/test_integration_dryrun.py`). Do not run this plan until the separate
Hyperplane feasibility report has confirmed the dependencies in §1. This plan does not substitute benchmarks or
checkpoints: where an asset is missing, the corresponding stage stops.

## 1. Dependencies on the separate Hyperplane feasibility report
| # | dependency | needed for | if missing |
|---|---|---|---|
| D1 | `GSAI-ML/LLaDA-8B-Instruct` loads with `trust_remote_code=True`; `model(input_ids).logits` shape (b, l, 126464); mask id 126336, EOS 126081, EoT 126348 match `model.config`/tokenizer | everything | stop |
| D2 | **LoRA target module names** of `modeling_llada.py` (UNVERIFIED here; our guess: `q_proj,k_proj,v_proj,attn_out,ff_proj,up_proj,ff_out` under `blocks.N`); `check_lora_targets(model)` must report 7 × n_layers matches | adapter training | fix `lora_target_regex` in `configs/hyperplane_fxmix.yaml`, re-run check |
| D3 | one LoRA training step runs in bf16 on one A100-40GB (grad checkpointing if needed), and `disable_adapter()` returns base logits | reference pass without a second model copy | fall back to a frozen second copy (+16 GB) |
| D4 | native `generate` from the pinned repo runs and our `generate_traced` reproduces it token-for-token | all leakage numbers | stop until fixed |
| D5 | TOFU data (forget01/forget05/retain, paraphrased questions, real_authors, world_facts) | data | stop |
| D6 | **Benchmark starting checkpoint.** MDU's TOFU-SFT LLaDA is not released (MDU README; EMDU App. E.1); EMDU's SFT models are promised "upon acceptance"; `Jameszhiwei/llada-tofu-forget10-SFT` and LRCU's checkpoints are named in LRCU's records but **not accessed** (HF blocked here). | comparability with published numbers | use our own SFT (§3 S0), which is the plan's default; published-number comparability is then explicitly *not* claimed |
| D7 | tokenizer behaviour for digits/dates and leading spaces (affects F) | F construction | run `f_closure_audit` and log; items whose F would be empty are excluded and counted |
| D8 | GPU availability (shared host; `scripts/gpu_select.sh` etiquette) | scheduling | wait |

## 2. Integration (exact entry points; `integration/llada_fxmix.py`, config `configs/hyperplane_fxmix.yaml`)
- `load_config(path) -> FXConfig`; `load_llada(cfg, device)`; `apply_lora(model, cfg)`; `check_lora_targets(model)`.
- Data: `chat_prompt_ids(tokenizer, question)`, `answer_ids(tokenizer, answer, gen_length)`, `sft_batch(...)` (GUIDELINES SFT
  layout: prompt unmasked, answer + EOS padding to `gen_length`=128).
- Fact sets: `build_fact_set(tokenizer, prompt_ids, answer, fact_strings)` then **mandatory** `f_closure_audit(...)` at θ0
  rollout decision states (SPEC v1.3 §1); log added ids per item.
- States: `tf_ff_states(full_ids, ans_start, fact_mask)`; `collect_pool(model, prompts, prompt_am, fact_mask, cfg)` (uses
  `generate_traced` with first-fact-commit truncation; current student, no grad); pool entries (steps, block, remasking, T):
  (128,128,lc,0), (128,128,rnd,1), (64,128,rnd,1), (32,128,rnd,1).
- Losses: `fx_forget_loss_llada(...)` (exclusion target from fp32 reference logits in log-space, ε = 1e-5 default) and
  `sft_loss_llada(...)` (retain); `train_step(...)`; `make_optimizer(model, cfg)` (AdamW, cosine, clip 1.0).
- Diagnostics: `fixed_point_report(...)` (implied fixed-point leak, certificate floor, n̂, n̂_elig), `certificate(...)`
  (B̂_π, eligible-sum form with exact π_T(F), MC s.e.), `tempered_fact_prob(...)`.
- Controls must use the *same* harness, data, LoRA targets, LR grid and evaluation: SC-TF (exclusion target on random TF
  states), EMDU (anchor-first KL to prompt+answer-masked anchor), MDU τ=1 (student-first KL to prompt-masked anchor) and
  NPO+RT. If MDU's author code is used as a cross-check, pass `--loss_type null_anchor --match_mode random` explicitly
  and log resolved args (the released `run_main.sh` omits them; D static finding A).

## 3. The one bounded experiment
**Hypothesis that can be falsified (H-real).** On LLaDA-8B-Instruct with injected fictitious facts, FX-Mix's state
design (fact-free TF + multi-config native rollouts truncated at the first fact commit) lowers fact leakage on
*held-out* native decoding configurations below SC-TF (same target on random teacher-forced states) at matched retention.
The CPU toy did **not** support this (see RESULTS.md); the experiment tests whether that negative transfers.

**S0 — assets (≈1.5 GPU-h).** LoRA-SFT (r=64, all 7 linear types, lr 1e-4, until forget ROUGE-L recall ≥ 0.8 under H1)
LLaDA-8B-Instruct on TOFU forget05 ∪ 400 retain QAs → *base*; same recipe on (forget05 \ forget01) ∪ the same 400 retain
QAs → *retrain reference*. Merge LoRA. Gate: base Λ_F on forget01 ≥ 0.8 under H1 and P1; retrain Λ_F within the
chance band measured on retain-shuffled facts. Fail ⇒ stop (facts not memorised / reference unusable).

**S1 — zero-training diagnostics (≈1 GPU-h), decisive gate.** On *base*, forget01 items:
(a) decision-state census: for every config in §4, the fraction of first fact-commit states that are **off the
ground-truth-masked manifold** (some visible answer token ≠ y*); (b) decode-time F-ban (FX-Mix's ε=0 fixed point):
leakage, EOS-only rate, fluency (§5); (c) MDU-anchor fact mass on fact-free TF states (EMDU's leakage mechanism);
(d) `fixed_point_report` for ε. **Stop rule:** if (a) < 20 % for every held-out config, the coverage argument has no
purchase at scale ⇒ record "state design redundant at LLaDA scale" and skip S3's FX-Mix arm.

**S2 — development (dev authors = 2 authors of forget05 \ forget01; ≈2 GPU-h).** Choose each arm's LR from {5e-5, 1e-4,
2e-4} (300 steps, cosine, LoRA r=16, forget states 8/step, retain batch 8, λ_r=1) by Cut A (§6) on dev items only.
**No final-set item, config or seed is used in S2.**

**S3 — final (forget01, 40 QAs; ≈8 GPU-h).** Arms with the LR fixed in S2: **FX-Mix** and **SC-TF** (2 unlearning seeds
each; the decisive pair), **FX-Mix-greedy** (pool = P1 only; 1 seed; isolates the pool), **EMDU**, **MDU τ=1**,
**NPO+RT** (1 seed each; literature context). Evaluate the final checkpoint and checkpoints at 100/200 steps.

## 4. Decoding grid (gen_length 128, cfg 0, no logit edits; bf16; Λ_F over all 128 positions incl. post-EOS)
- Pool (training, also reported): P1 (128,128,lc,0), P2 (128,128,rnd,1), P3 (64,128,rnd,1), P4 (32,128,rnd,1).
- Held-out: H1 (128,32,lc,0) [README default block 32], H2 (64,128,lc,0.5), H3 (128,128,rnd,0), H4 (128,128,lc,1),
  H5 (64,64,rnd,0.7), H6 (64,32,lc,0) [2 tok/step semi-AR].
- Extrapolation (reported separately): X1 (16,128,rnd,1) [8 tok/step]; ARL/EMDU length/steps grid
  (L,S) ∈ {(32,32),(64,32),(128,32),(64,64),(128,64)} with lc greedy.
- Stochastic configs: 2 samples per prompt; fixed evaluation seeds shared by all arms.

## 5. Measures (success cannot come from refusal, emptiness, disfluency or a likelihood proxy)
- **Forgetting:** Λ_F rate (any F id at any position) and exact surface-span match over tokenisation variants, per config;
  primary = max over H1–H6. ROUGE-L recall vs GT (secondary; it cannot reach 0 when answers restate the subject).
- **Abstention/degeneration guards** on forget queries: EOS-only rate, refusal-template rate ("I don't know", "no
  information"), repetition rate (max token frequency), base-model ELBO perplexity of the generated answer (MC, 32
  masks). An arm whose forget answers are > 50 % EOS-only or refusal is reported as **abstention (type ii)**, not as
  reduced recall, and cannot "win" H-real.
- **Retention:** retain-QA fact-token recall and ROUGE-L under H1, P1, P2 (mean); the pre-registered **shared-value-token
  slice** (retain QAs whose answers contain any forget F id); real_authors / world_facts ROUGE-L under H1; 200 MMLU
  questions by single-token likelihood (`get_log_likelihood`, mc_num=1).
- **Conditioning supplies the answer:** items whose prompt contains an F token are excluded from Λ_F and reported apart;
  paraphrased TOFU questions reported separately (outside the method's scope).
- **Mechanism:** off-manifold first-commit fraction after training (SC-TF vs FX-Mix); certificate B̂_π vs observed leak.
- **Latent knowledge (type iv, reported, not claimed):** MC answer probability on forget01 (TOFU p), and a 20-step
  relearning probe on 50 % of forget01 measured on the other 50 %.

## 6. Matched operating points and decision rules (pre-registered before S3)
- Retention constraint: mean retain fact-token recall over H1/P1/P2 ≥ base − 0.05 and MMLU ≥ base − 2 points.
- Cut A: lowest held-out max Λ_F among an arm's feasible checkpoints. Cut B: best retention among checkpoints with
  held-out max Λ_F ≤ the retrain reference's level + 0.05. Report the full curves and both cuts.
- **H-real supported** iff in both seeds Cut A(SC-TF) − Cut A(FX-Mix) ≥ 0.05, FX-Mix is not abstention-flagged, and
  FX-Mix's Cut B retention is not lower than SC-TF's by > 0.02.
- **H-real falsified** iff |Cut A(SC-TF) − Cut A(FX-Mix)| < 0.05 in both seeds (state design redundant ⇒ the method is
  SC-TF = "TNFF-in-MDU", target credited to TNT/UNDIAL); stop.
- Anything else: inconclusive; do not expand the study under this plan.
- Native-fidelity gate (D4), S0 gate and S1 stop rule apply before any comparison is read.

## 7. Compute estimate (projection, not measured on GPU; A100-40GB, bf16)
Assumptions: one LLaDA-8B forward on a batch of 16 × ~190 tokens ≈ 0.2 s; backward ≈ 2× forward.
- Training step (8 forget states + 8 retain): SC-TF ≈ 3 fwd-eq. (student) + 0.5 (reference) ≈ 0.7 s; FX-Mix adds
  rollouts: 8 prompts × mean 88 steps per refresh, R=4 ⇒ ≈ 22 batched forwards/step ≈ 2.2 s ⇒ ≈ 3–4× SC-TF wall per step
  (CPU-measured ratio in RESULTS.md). 300 steps: SC-TF ≈ 4 min, FX-Mix ≈ 15 min.
- Evaluation per checkpoint: 80 prompts (40 forget + 40 retain) × 11 configs × ≤ 2 samples × ≤ 128 steps ≈ 25 min;
  + MMLU likelihood ≈ 2 min.
- Totals: S0 1.5 GPU-h; S1 1 GPU-h; S2 ≈ 2 GPU-h; S3 ≈ 6 arms-seeds × 3 checkpoints × 27 min + training ≈ 8 GPU-h.
  **≈ 12–13 GPU-h total (≈ 3.5 h wall on 4 GPUs).** Stopping: abort if any stage exceeds 2× its estimate.
- Memory: 16 GB bf16 weights + LoRA r=16 optimiser state (≈ 0.2 GB) + activations (grad checkpointing); rollout pool stores
  token ids only; the |V|=126,464 exclusion target is one fp32 log-softmax per masked position (no caching of logits).

## 8. What a positive outcome would and would not mean
Positive: the state design matters for config-robust token-level recall at 8B scale on injected fictitious facts.
It would not show removal of training influence, robustness to re-masking/prefix/context attacks or CFG, transfer to
real-world knowledge (RWKU), or superiority over methods not run here.
