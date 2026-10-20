# HYPERPLANE_PLAN — integration instructions, dependencies, and one bounded falsification experiment

Status: **real-model execution UNTESTED.** Nothing here has run on LLaDA-8B. The integration code was dry-run only
against a tiny random model on CPU (`tests/test_integration_dryrun.py`, `tests/test_integration_comparators.py`). Do not run this plan until the separate
Hyperplane feasibility report has confirmed the dependencies in §1. This plan does not substitute benchmarks or
checkpoints: where an asset is missing, the corresponding stage stops.

## 0. Status after the locked CPU validation — is one Hyperplane test justified?
The locked CPU validation returned **NO_DISTINCT_METHOD_AFTER_CONSTRUCTION** (RESULTS.md §3): FX-Mix's state design gave
no held-out-leakage gain over SC-TF (+0.007, 4/6 units) or compute-matched SC-TF (0.027 vs 0.027) at lower retention.
In the toy, the CPU analogue of the S1 stop rule below would fire (SC-TF's Cut-A held-out leakage 0.021–0.052 vs the
retrain reference's 0.167–0.250; G_REPORT §7).
- **Not justified:** a full S2–S3 comparison as a test of FX-Mix as a new method.
- **Justified at most:** S0 + S1 (≈ 3–3.5 GPU-h), framed as the bounded falsification test of whether the CPU negative
  transfers to LLaDA-8B — the one open question the toy cannot answer is whether an 8B mask predictor generalises
  teacher-forced suppression to its off-manifold decision states as the 2-layer toy did (THEORY Prop. 2 shows this
  generalisation is what decides the outcome). S2–S3 run only if both S1 stop rules fail. If Hyperplane time is
  scarce, skip this plan entirely; nothing else in the package depends on it.

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
- Controls use the *same* harness, data, LoRA targets, LR grid and evaluation, selected by `arm:` in the YAML `method:`
  section (`fx_mix | fx_mix_greedy | sc_tf | emdu | mdu_t1 | npo`): `random_tf_states_llada` + `sc_tf_loss_llada` (SC-TF),
  `emdu_loss_llada` (anchor-first KL to the adapter-off reference on the fully masked prompt+answer), `mdu_loss_llada`
  (student-first KL to the prompt-masked anchor, τ), `npo_loss_llada` (β=0.2). Added after the locked CPU validation
  (G finding A17); dry-run tested on a tiny random model (`tests/test_integration_comparators.py`, 10/10); real model UNTESTED. If MDU's author code is used as a cross-check, pass `--loss_type null_anchor --match_mode random` explicitly
  and log resolved args (the released `run_main.sh` omits them; D static finding A).

## 3. The one bounded experiment
**Hypothesis that can be falsified (H-real).** On LLaDA-8B-Instruct with injected fictitious facts, FX-Mix's state
design (fact-free TF + multi-config native rollouts truncated at the first fact commit) lowers fact leakage on
*held-out* native decoding configurations below SC-TF (same target on random teacher-forced states) at matched retention.
CPU development data (dev worlds 0–1) did **not** support this; the locked CPU result is in RESULTS.md §3. The
experiment tests whether the CPU outcome transfers to LLaDA scale.

**S0 — assets (≈1.5 GPU-h).** LoRA-SFT (r=64, all 7 linear types, lr 1e-4, until forget ROUGE-L recall ≥ 0.8 under G1)
LLaDA-8B-Instruct on TOFU forget05 ∪ 400 retain QAs → *base*; same recipe on (forget05 \ forget01) ∪ the same 400 retain
QAs → *retrain reference*. Merge LoRA. Gate: base Λ_F on forget01 ≥ 0.8 under G1 and P1; retrain Λ_F within the
chance band measured on retain-shuffled facts. Fail ⇒ stop (facts not memorised / reference unusable).

**S1 — cheap decisive gate (≈ 1.5–2 GPU-h).** FX-Mix can only beat SC-TF where SC-TF leaves residual leakage, so
test that first. On *base*, dev items (2 authors of forget05 \ forget01):
(a) zero-training: decode-time F-ban (FX-Mix's ε=0 fixed point) leakage, EOS-only rate and fluency (§5); MDU-anchor
fact mass on fact-free TF states (EMDU's leakage mechanism); `fixed_point_report` (n̂, n̂_elig) for ε; `f_closure_audit`;
(b) one SC-TF run (LR 1e-4, 300 steps, 1 seed) evaluated on the held-out grid, plus the census of its residual
first-commit decision states (fraction off the ground-truth-masked manifold).
**Stop rule (pre-registered):** if SC-TF's held-out max Λ_F at the retention constraint is ≤ the retrain reference's
level + 0.02, there is nothing for the state design to remove ⇒ record "H-real cannot be supported at LLaDA scale" and
stop; likewise stop if < 20 % of SC-TF's residual decision states are off-manifold (the coverage argument has no
purchase). Only otherwise proceed to S2–S3.

**S2 — development (dev authors = 2 authors of forget05 \ forget01; ≈2 GPU-h).** Choose each arm's LR from {5e-5, 1e-4,
2e-4} (300 steps, cosine, LoRA r=16, forget states 8/step, retain batch 8, λ_r=1) by Cut A (§6) on dev items only.
**No final-set item, config or seed is used in S2.**

**S3 — final (forget01, 40 QAs; ≈8 GPU-h).** Arms with the LR fixed in S2: **FX-Mix** and **SC-TF** (2 unlearning seeds
each; the decisive pair), **FX-Mix-greedy** (pool = P1 only; 1 seed; isolates the pool), **EMDU**, **MDU τ=1**,
**NPO+RT** (1 seed each; literature context). Evaluate the final checkpoint and checkpoints at 100/200 steps.

## 4. Decoding grid (gen_length 128, cfg 0, no logit edits; bf16; Λ_F over all 128 positions incl. post-EOS)
- Pool (training, also reported): P1 (128,128,lc,0), P2 (128,128,rnd,1), P3 (64,128,rnd,1), P4 (32,128,rnd,1).
- Held-out (named G1–G6 to avoid a clash with the toy's H1–H6): G1 (128,32,lc,0) [README default block 32], G2 (64,128,lc,0.5), G3 (128,128,rnd,0), G4 (128,128,lc,1),
  G5 (64,64,rnd,0.7), G6 (64,32,lc,0) [2 tok/step semi-AR].
- Extrapolation (reported separately): X1 (16,128,rnd,1) [8 tok/step]; ARL/EMDU length/steps grid
  (L,S) ∈ {(32,32),(64,32),(128,32),(64,64),(128,64)} with lc greedy.
- Stochastic configs: 2 samples per prompt; fixed evaluation seeds shared by all arms.

## 5. Measures (success cannot come from refusal, emptiness, disfluency or a likelihood proxy)
- **Forgetting:** Λ_F rate (any F id at any position) and exact surface-span match over tokenisation variants, per config;
  primary = max over G1–G6. ROUGE-L recall vs GT (secondary; it cannot reach 0 when answers restate the subject).
- **Abstention/degeneration guards** on forget queries: EOS-only rate, refusal-template rate ("I don't know", "no
  information"), repetition rate (max token frequency), base-model ELBO perplexity of the generated answer (MC, 32
  masks). An arm whose forget answers are > 50 % EOS-only or refusal is reported as **abstention (type ii)**, not as
  reduced recall, and cannot "win" H-real.
- **Retention:** retain-QA fact-token recall and ROUGE-L under G1, P1, P2 (mean); the pre-registered **shared-value-token
  slice** (retain QAs whose answers contain any forget F id); real_authors / world_facts ROUGE-L under G1; 200 MMLU
  questions by single-token likelihood (`get_log_likelihood`, mc_num=1).
- **Conditioning supplies the answer:** items whose prompt contains an F token are excluded from Λ_F and reported apart;
  paraphrased TOFU questions reported separately (outside the method's scope).
- **Mechanism:** off-manifold first-commit fraction after training (SC-TF vs FX-Mix); certificate B̂_π vs observed leak.
- **Latent knowledge (type iv, reported, not claimed):** MC answer probability on forget01 (TOFU p), and a 20-step
  relearning probe on 50 % of forget01 measured on the other 50 %.

## 6. Matched operating points and decision rules (pre-registered before S3)
- Retention constraint: mean retain fact-token recall over G1/P1/P2 ≥ base − 0.05 and MMLU ≥ base − 2 points.
- Cut A: lowest held-out max Λ_F among an arm's feasible checkpoints. Cut B: best retention among checkpoints with
  held-out max Λ_F ≤ the retrain reference's level + 0.05. Report the full curves and both cuts.
- **H-real supported** iff in both seeds Cut A(SC-TF) − Cut A(FX-Mix) ≥ 0.05, FX-Mix is not abstention-flagged, and
  FX-Mix's Cut B retention is not lower than SC-TF's by > 0.02.
- **H-real falsified** iff |Cut A(SC-TF) − Cut A(FX-Mix)| < 0.05 in both seeds (state design redundant ⇒ the method is
  SC-TF = "TNFF-in-MDU", target credited to TNT/UNDIAL); stop.
- Anything else: inconclusive; do not expand the study under this plan.
- Native-fidelity gate (D4), S0 gate and S1 stop rules apply before any comparison is read; S1 uses dev items only.

## 7. Compute estimate (projection, not measured on GPU)
Basis: F's FLOP arithmetic (`results/dev/COST.md` §2; LLaDA-8B ≈ 8.0 B params, ≈ 200-token sequences, ≈ 3.2e12 FLOPs per
sequence-forward) converted to the Hyperplane A100-40GB (312 TF/s bf16 peak) at an assumed 35 % MFU (≈ 1.1e14 FLOP/s).
- Training step (8 forget states + 8 retain; LoRA backward ≈ 2× forward): SC-TF ≈ 1.8e14 FLOPs ≈ 1.7 s; FX-Mix adds
  ≤ 176 no-grad rollout forwards/step (pool mean 88 steps, 8 prompts per refresh, R=4) ⇒ ≈ 7.4e14 ≈ 6.8 s (≈ 4.1×
  SC-TF; truncation and all-leaked early stop reduce it). 300 steps: SC-TF ≈ 9 min, FX-Mix ≈ 34 min.
- Evaluation dominates: one checkpoint over forget 40 prompts × 7 configs (P1, P2, G1, G2, G3, G5, X1; stochastic incl.
  random-order G3 ×2 samples) + retain 40 prompts × 3 configs ≈ 6.3e4 sequence-forwards ≈ 2.0e17 FLOPs ≈ 31 min. Evaluate only the final checkpoint
  of each LR (2 strength points per arm-seed) in S3; the remaining held-out cells (G4, G6, ARL (L,S) grid) only for the
  Cut-A checkpoint of FX-Mix and SC-TF.
- Totals: S0 ≈ 1.5 GPU-h; **S1 ≈ 1.5–2 GPU-h (decisive gate)**; S2 ≈ 3 GPU-h; S3 ≈ 8 arm-seeds × 2 evals × 31 min (≈ 8.3 h) +
  training (FX-Mix 4 × 34 min, others ≈ 9–15 min each; ≈ 3 h) ≈ 11–11.5 GPU-h. **≈ 16–18 GPU-h total if every stage runs
  (≈ 4.5 h wall on 4 GPUs); ≈ 3 GPU-h if the S1 gate stops the study.** Abort any stage exceeding 2× its estimate.
- Memory: 16 GB bf16 weights + LoRA r=16 (41.9 M params; fp32 params/grads/AdamW ≈ 0.67 GB) + activations (gradient
  checkpointing on 40 GB); reference via adapter-disable (no second copy; MDU's code keeps a separate 16 GB ref model);
  rollout pool stores token ids only (≈ 74 KB/refresh); upstream `add_gumbel_noise` casts full logits to float64
  (≈ 1.6 GB at batch 8 × 200 tokens); exclusion target ≈ 2–3 GB transient fp32 over (masked positions × 126,464).
- ε and certificate: at ε = 1e-5 the target's implied fixed-point leak ≤ 1.3e-3 at n̂ ≤ 128; the certificate floor
  ε·Σ|E_n| is ≈ 0.08 for single-block 128-step low-confidence decoding (Σ|E_n| = 8256) but ≈ 1.3e-3 for random
  remasking — report the certificate as informative only for random-remasking configs and the greedy exact-zero check.

## 7b. Real-model risks identified by the implementation engineer (F)
1. LLaDA's LM head may also be named `ff_out` (dllm's vendored `modeling_llada.py`); a suffix match would wrap the head.
   The guard regex requires `blocks.N.` — verify with `check_lora_targets` (D2).
2. Rollouts ≈ 4× SC-TF FLOPs per step (above).
3. SFT layout (EoT before EOS padding) unverified for whichever TOFU-SFT checkpoint is used (`append_eot` switch).
4. `build_fact_set` is heuristic; run `f_closure_audit` and log additions before training.
5. CUDA RNG handling (generator device) was fixed in the port but never executed on a GPU.
6. Native-generation evaluation cost (≈ 30 min per config for 400 prompts × 4 samples at 128 steps) — keep the grid
   and prompt subsets as specified.

## 8. What a positive outcome would and would not mean
Positive: the state design matters for config-robust token-level recall at 8B scale on injected fictitious facts.
It would not show removal of training influence, robustness to re-masking/prefix/context attacks or CFG, transfer to
real-world knowledge (RWKU), or superiority over methods not run here.
