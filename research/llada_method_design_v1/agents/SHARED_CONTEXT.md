# SHARED CONTEXT — llada_method_design_v1 (read fully before working)

Coordinator-maintained. Sprint start ≈ 2026-10-20T03:30Z; hard stop 11:30Z (8 h incl. publication).
Repo: `/home/user/XXResearch`, branch `claude/sleepy-feynman-a8dzok`, sprint dir `research/llada_method_design_v1/` (= `$R`).
Do **not** git commit/push — the coordinator does that. Write only inside your assigned paths.

## 1. Assignment (condensed)
Construct ONE mathematically motivated unlearning method for **masked diffusion LMs (LLaDA, text generation)**,
implement its core on CPU, deliver a Hyperplane (GPU host) test package. Primary advantage (default):
**retained knowledge / general capability at matched forgetting**; constraints: forgetting must hold under **more than one
native decoding configuration**, compute practical. Prefer parameter/adapter updates (no refusal prompt, no output filter).
No overclaiming: do not claim novelty from absent search hits, do not claim retraining equivalence, do not claim a
local constraint guarantees global forgetting/utility. Distinguish: (i) reduced recall (generated-answer leakage),
(ii) refusal, (iii) inference-time suppression, (iv) approximate removal of training influence.

## 2. Environment / budget (hard)
- CPU only: 4 cores, 15 GiB RAM. Python with torch: `/tmp/claude-0/env-llada/bin/python` (torch 2.9.1 CPU, numpy, scipy,
  sympy, pytest). System python3 has numpy only.
- **Always** run compute through the ledger wrapper, with ≤1 thread unless told otherwise:
  `/tmp/claude-0/env-llada/bin/python $R/tools/ledger_run.py --agent <A..G> --label "<what>" --threads 1 -- /tmp/claude-0/env-llada/bin/python <script>`
  Keep any single run ≤ 10 min wall. Wave-1 designers: analytic / exact tabular checks only (no neural training).
- Network: **github.com via `git clone` works; api.github.com works; conda-forge works. arxiv.org, huggingface.co,
  proceedings.mlr.press, openreview, semantic scholar are BLOCKED** (both curl and WebFetch). WebSearch may return
  snippets — label anything learned only from snippets as `SNIPPET-ONLY`. No model-weight downloads, no GPUs.
- Pinned local sources (read-only; do not modify):
  - LLaDA: `/tmp/claude-0/src/LLaDA` @ `9182493720ed723ef8031210d85959364e51cbe0` (GUIDELINES.md, generate.py, get_log_likelihood.py, chat.py)
  - MDU code: `/tmp/claude-0/src/MDU` @ `e5e8e8274ff8f56f16e978803d9e902359eb5272` (`src/unlearn_mdu_llada.py` 1744 lines)
  - MDU paper PDF (supplied by user): `/tmp/claude-0/src/MDU_2605.18253v1.pdf` (sha256 82e1d2ad…cf3e0), 16 pp.

## 3. LLaDA mechanics (verified from pinned source unless marked)
- Mask predictor = bidirectional Transformer encoder (AR backbone with causal mask removed). `[MASK]` id 126336.
  EOS id 126081, EoT id 126348 (generate.py). Model class/module names live in `modeling_llada.py` on the HF hub —
  **NOT inspectable here (HF blocked) → UNVERIFIED; Hyperplane task must confirm.**
- Pretraining forward process: `t~U(0,1)`, `p_mask=(1-eps)t+eps`, eps=1e-3, every token masked iid w.p. p_mask;
  loss = Σ_masked CE / p_mask, divided by (b·l).
- **SFT (the relevant regime):** prompt is never masked; the response *including padded EOS tokens* is masked iid
  with p_mask; loss = Σ_masked CE/p_mask/answer_len(incl. EOS pad), averaged over batch. So the model is trained on
  teacher-forced states y_t = (prompt visible, random subset of the ground-truth answer visible), weight 1/t.
  EOS padding ⇒ the model learns to emit EOS to fill a fixed generation length.
- **Native sampling (`generate`)**: answer region of fixed `gen_length` initialised to [MASK]; split into
  `gen_length/block_length` semi-autoregressive blocks; `steps/num_blocks` steps per block; at each step the model
  predicts all positions; candidate x0 = argmax(logits + Gumbel(temperature)) (temperature 0 = greedy);
  confidence = softmax prob of x0 (`remasking='low_confidence'`) or uniform random (`'random'`); only positions in the
  current block are eligible; the top-k by confidence are committed, k = uniform schedule; **committed tokens are never
  re-masked**. Optional CFG: unconditional pass replaces prompt tokens by [MASK]; logits = un + (s+1)(cond−un).
  Options `logits_eos_inf`, `confidence_eos_eot_inf` (note: the latter line also writes −inf into `logits[...,126348]`).
- Likelihood eval (`get_log_likelihood`): MC over masked count k~U{1..L} (stratified across the batch), CE/p_mask, mc_num=128.
- Hence "native decoding configurations" = (steps, gen_length, block_length, remasking ∈ {low_confidence, random},
  temperature, cfg_scale). Generated-state distribution ≠ teacher-forced training-state distribution.

## 4. Closest work known so far (to be verified/extended by D)
**MDU (Lee et al., arXiv 2605.18253, May 2026; code above).** Per forget pair, sample t~U(0,1), y_t~q(·|y,t)
(teacher-forced random masking of the ground-truth answer); loss
`(1/|M_t|) Σ_{i∈M_t} KL( p_θ(·|x,y_t) ‖ (1/Z) p_θ0(·|m,y_t)^τ )` where `m` = prompt replaced by [MASK] (prompt-masked
"anchor", from frozen initial θ0), τ∈[0,1] (τ=0 ⇒ uniform = max-entropy; τ=1 ⇒ anchor) + λ·retain SFT loss (λ=1),
AdamW, cosine LR, clip 1.0, full fine-tuning (check code for LoRA). Interpretable as tilt p^{1−τ}·anchor^τ (CFG-like).
Evaluated on TOFU forget10 (LLaDA-8B-Instruct SFT'd 1000 epochs on TOFU; Dream-7B) and RWKU; metrics ROUGE-L and
MC answer prob; generation with 128 denoising steps / 128 tokens (their config); baselines GA, GD, NPO, SimNPO, WGA, DPO
transplanted by substituting the MDM SFT loss for AR NLL. Author-reported only. Their stated limitation: training uses
random masked states while generation follows a full trajectory with specific sampling/remasking.
Code has extra variants (not in paper's main method): `_traj_rollout_na_loss` (states from model rollouts),
token-category weighting (`novel_percentile`, `_cat_oracle_weights`, `conf_threshold`) — D must characterise them.
Coordinator observation to examine (not established): the anchor `p_θ0(·|m,y_t)` still sees the *visible answer tokens*
(TOFU answers often restate the subject; partial facts can be visible), so the anchor itself may predict the fact
(e.g. paper Table 6a, τ=1 output keeps the GT year "1951").
**Others (not yet accessible here; D to search/verify):** Adversarial RL for robust dLLM unlearning
(PMLR v306 zhang26di); TimeROME-DLM (arXiv 2606.12841); LRCU public disclosures/checkpoints (HF collection
guanmingchiu/lrcu-dlm-weight-unlearning); A2D (arXiv 2509.23286, any-order any-step safety alignment for dLLMs).
AR unlearning: GA, GD, NPO, SimNPO, WGA, DPO/IdkPO, RMU, WHP (generic-prediction targets), UNDIAL (self-distillation
with logit-adjusted teacher), ME (max-entropy), AltPO (alternative answers), token-selective unlearning, etc.

## 5. Closed prior XXResearch work — constraints only (do not reopen, rerun, modify or rename)
- VAU / flow_method_design_v1 (FLUX, branch `claude/epic-fermat-xrwssq`): Adam-metric correction of LoRA steps against a
  sketched retained-Jacobian probe + non-progress rejection + Hoeffding acceptance gate. Verdict NO_SUPPORTED_CANDIDATE
  (no qualifying advantage at matched erasure; 20–64× compute). **Gradient-projection-against-retained-Jacobian
  families and acceptance gates are therefore not to be re-proposed under a new name.**
- CPR/FLUX studies: closed. Lesson carried forward: avoid big synthetic factorial studies; match operating points;
  report infeasible arms instead of silently comparing different forgetting levels.

## 6. Deliverable format for each designer (wave 1)
Write `$R/agents/<X>_proposal.md` (≤ ~350 lines) containing exactly:
1. **Failure/inefficiency** of current practice, supported by a source, derivation or explicit counterexample.
2. **Mathematical formulation** (define every symbol: prompt c, answer y∈V^L, mask set M, state s, model p_θ, etc.).
3. **Executable update** (pseudo-code precise enough to implement; state every approximation).
4. **Predicted advantage** (one primary) and how it would be measured at matched forgetting.
5. **Closest known method** (what you checked and how — files/lines/sections read; mark SNIPPET-ONLY / NOT ACCESSED).
6. **Simple condition under which it fails or becomes redundant** (incl. reduction to a simple control).
7. **Numerical evidence**: one small exact/tabular check script in `$R/agents/<X>_scripts/` + its output file.
8. **Cost** per optimisation step in forward/backward passes (frozen-teacher and student), extra memory.
Keep it falsifiable; prefer concessions to rhetoric.

## 7. Substrate and coordinator probe (added 03:54Z; available to waves 2+)
- CPU substrate in `$R/src/mdmu/` (see `$R/src/README_SUBSTRATE.md`): toy fact world (2 question templates × 2 answer
  phrasings; answers restate the subject), 2-layer bidirectional mask predictor, LLaDA SFT loss, port of upstream
  `generate`, generation-based leakage metrics under configs D1 (low_conf greedy, 1 tok/step), D2 (random remasking, T=1),
  D3 (low_conf greedy, block 2), D4 (low_conf, 2 tok/step, T=0.5). Dev checkpoints `results/dev/substrate/w0_i0/{base,retrain}.pt`.
- Coordinator de-risking probe (`results/dev/coord_probe/`, LoRA r=4, lr 2e-3, +retain SFT): GD, NPO, MDU τ∈{0,1} all reach
  exact-value leakage 0 within 50 steps, but **forget-prompt well-formedness is 0 for every arm** (retrain reference: ~1.0
  under D1–D3, i.e. fluent hallucinated values), and **MDU τ=1 still emits forgotten value tokens (leak_any 0.10–0.54 vs
  retrain chance 0.08–0.13)** — consistent with its anchor seeing the restated subject in visible answer tokens. NPO keeps
  retained facts best among these (D1 1.0 / D2 0.87–0.94 at 200–300 steps). Forgetting in the toy is easy; matched-forgetting
  comparisons need a finer strength knob. Base model under D4 (2 tok/step) is itself only 71 % well-formed.
