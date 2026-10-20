# D — Prior art, reduction to known methods, and controls (Agent D#1, wave 2)

Scope: **FX-Mix** (A's FFX-OP merged with B's FCH-Mix) and C's **MSDA**. No training was run; every external claim
carries its access status; "not found" is never a novelty argument. Numbers in §1–2 are **author-reported**; code
observations are **static reading** (MDU/dllm not executed; dllm not installed here).

## 0. Bottom line
- **Target is known.** A's ε=0 target is TNFF (code read) = UNDIAL with γ=∞, applied to a token *set*; B's ε version is
  an adaptive-γ logit adjustment (Unilogit-like, snippet). At its fixed point FX-Mix puts **"decode the frozen reference
  with F banned"** into the weights on the forget prompts (if p_θ=r at every fact-free state, the native sampler *is*
  the banned-reference sampler).
- **Loop is known.** MDU's code has an on-policy MDM rollout variant (`_traj_rollout_na_loss`); the AR analogue is
  on-policy distillation from a logit-masked teacher (GKD-style; from memory).
- **Direction/anchor fix is known.** EMDU (Garber & Tirer, arXiv 2610.25447, 18 Oct 2026; SNIPPET-ONLY) reports MDU's
  visible-forget-token anchor leak and switches to an anchor-first KL — pre-empting A's F2 and the direction choice.
- **Specific to FX-Mix: the state distribution** — (i) a pool of native configs incl. random remasking T=1 with 1 and
  ≥2 tokens/step, (ii) truncation at the first fact commit (decision state kept), (iii) fact-free TF states (F
  force-masked), justified by the first-commit hazard bound. A distinct method survives **only if** pool/truncation
  beat the straightforward combination SC (§3.4) on held-out-config leakage at matched retention.

## 1. Inspected sources
Statuses: CODE-READ, PDF-READ, SECONDARY (third-party summary read in full), SNIPPET-ONLY, NOT ACCESSED.

| source | identity / pin | status | what was read |
|---|---|---|---|
| MDU paper | arXiv 2605.18253v1, local PDF sha256 82e1d2ad…cf3e0 | PDF-READ (pdftotext) | §4 Eqs. 3–6, §5 Table 1, Tables 2/3/5/6/7, Limitations, App. A Alg. 1, App. B.1–B.4 |
| MDU code | leegeoru/MDU @ e5e8e82 (pinned) | CODE-READ | `unlearn_mdu_llada.py` L1–1744 (except diagnostic/logging blocks); `run_main.sh`; `configs/*.yaml`; `scripts/eval_tofu_llada.py` L20–80; `unlearn_mdu_dream.py` L1–30 |
| dllm (MDU's library) | ZHZisZZ/dllm @ ca17675 (HEAD 2026-04-14; MDU's exact version unknown) | CODE-READ (partial) | `core/samplers/mdlm.py` L15–35, `core/trainers/mdlm.py` L27–100, `schedulers/alpha.py` L86–88, `utils/data.py` L87–117 |
| Adversarial-RL dLLM unlearning | ICML 2026 poster 65904; PMLR v306 "zhang26di" not verified | SNIPPET-ONLY | abstract via snippets; code = anonymous repo, not found |
| TimeROME-DLM | arXiv 2606.12841 | SNIPPET-ONLY | abstract-level snippets |
| LRCU / Open-DLU | guan404ming/open-dlu @ fb98433 (2026-06-16); HF collection guanmingchiu/lrcu-dlm-weight-unlearning | CODE-READ; HF checkpoints NOT ACCESSED | README; `component/forget/{capped,dlm_cap,anchor}.py`; `mask/{bernoulli,t_conditional}.py`; `variant/{__init__,_common,tau}.py`; `config.py`; `main.py`; `eval.py` (grep); `bench/results` TOFU JSONs |
| A2D | arXiv 2509.23286 (ICLR 2026) | SECONDARY + SNIPPET | zhaoyang97/Paper-Notes-en note (raw.githubusercontent); paper and code NOT ACCESSED |
| Recovery Alignment | arXiv 2510.00565 (ICLR 2026); mdl-lab/dlm-priming-vulnerability @ b2c9769 | CODE-READ | README; `train.py` L296–520 (rollout, GRPO loss); `alignment/arguments.py` defaults |
| TNT / TNFF | arXiv 2406.13660 (TMLR 2024); google/t5patches @ 09285b3 | CODE-READ (paper SNIPPET-ONLY) | `t5patches/models.py` L398–470, L565–641; README |
| UNDIAL | arXiv 2402.10052 (NAACL 2025); open-unlearning @ b299290 | CODE-READ (re-implementation); paper SNIPPET-ONLY | `trainer/unlearn/undial.py`; `trainer/utils.py::compute_undial_loss` L70–99 |
| Unilogit; RKLU; DUET; ASU | 2505.06027; NAACL 2025; ICLR 2026; 2603.01285 | SNIPPET-ONLY | abstracts |
| EMDU | arXiv 2610.25447 (2026-10-18) | SNIPPET-ONLY | abstract (two queries) |
| DiffuGuard; RAEC; CURE; MOSA; DIJA | 2509.24296; 2609.00495; NeurIPS'26 oral 151877; 2508.12398; 2507.11097 | SNIPPET-ONLY / radar | abstracts; GusSand/research-radar backfill file |
| ReGLU/RILA; CorDA; GROM; null-space LoRA (MLLM); Fisher-LoRA | 2604.17396 (Findings ACL'26); 2406.05223; 2608.05783; 2605.05909; 2408.06621 | SNIPPET-ONLY | abstracts |
| WHP; GKD; LoRA-FA; AlphaEdit; MEMIT; DAgger | — | NOT ACCESSED (memory; marked where used) | — |
| "dLLM knows the answer before it decodes" | ICLR 2026, MDU ref. [38] | NOT ACCESSED | bibliography entry only |

Effort: 23 WebSearch, ~12 GitHub MCP searches, 4 clones. api.github.com search, icml.cc, openreview, alphaxiv,
openalex, aclanthology and arxiv were refused; GitHub MCP search and raw.githubusercontent.com worked.

## 2. Findings per source

### 2.1 MDU: paper (PDF-READ) vs code (CODE-READ)
**Paper.**
- Eq. 5: L = E[(1/|M_t|) Σ_{i∈M_t} KL(p_θ(·|x,y_t) ‖ p_θ0(·|m,y_t)^τ/Z)]; the anchor masks the prompt but keeps visible
  answer tokens. Alg. 1: t~U, y_t~q(·|y,t), plus retain SFT (1/t-weighted), λ=1.
- AdamW, cosine LR, clip 1.0, batch 4×4. Base model: LLaDA TOFU-SFT, 1000 epochs at lr 2e-5 (README: SFT code and
  checkpoint not released).
- The TOFU decoding config is not stated in the PDF text read. Configs say 128 denoise steps / 128 tokens;
  `eval_tofu_llada.py` defaults to steps=256, T=0, low_confidence; dllm's sampler default is block_size=128 (single
  block; at dllm HEAD).
- Author-reported Table 2 (LLaDA), forget rL / retain rL: Base .884/.870; NPO .372/.726; τ=0 .069/.868;
  τ=0.5 .098/.853; τ=1 .034/.511. Table 6a: τ=1 still emits "…1951", τ=0.5 emits "born,19".
- Limitation, verbatim at PDF lines 621–624: training uses randomly sampled masked states, while generation follows a
  full trajectory with specific sampling and remasking strategies.

**Code** (`src/unlearn_mdu_llada.py`):
- **Eq. 5 path:** `_random_sft_null_anchor_loss` (L1102–1147), reached only with `--loss_type null_anchor
  --match_mode random` (L1363–1371).
- **`_null_anchor_kl`** (L947–1003): target logits τ·z_u (optionally ESD η). Default `forward` = **KL(p_θ‖target),
  student-first**; `reverse` = target-first; option to exclude specials. No 1/t weight (mean over masked positions).
- **`_traj_rollout_na_loss`** (L1005–1100, flag `null_anchor_traj_rollout`): no-grad rollout of the **current** model
  from the all-mask **GT-length** answer region; greedy argmax, low-confidence top-k, ceil(remaining/steps_left)
  schedule, one block, T=`null_anchor_traj_steps`=16. Stops at **one random step ∈[1,T−1] per batch** and trains that
  single state (all still-masked positions) against the frozen prompt-masked anchor. No fact filter, so post-leak
  states are trained; refreshed every step. The help text says "option B, GT tokens placed"; the code places model
  predictions (option C), as B also noted.
- **`_denoising_trajectory_weights`** (L316–722):
  - Greedy low-confidence rollout (`denoise_steps`) of the passed model, recording each position's unmask step and
    confidence.
  - Selects "late" content positions: top `novel_percentile`% by step (code default 30), adaptive mean+kσ, or, if
    `conf_threshold`>0, positions committed with confidence < threshold.
  - Match modes (`token_id`/`position`/`factual_filter`/`tid_prob`/`gap`…) map these onto GT positions. Weights are
    binary (`weight_beta` deprecated) or a stochastic sigmoid of step percentage.
  - Feeds `_denoise_novel_ga_loss` (L769–945), which masks **only the selected GT positions** and applies GA / bounded
    GA / NPO / null-anchor. Here the null-anchor uses the *current* model (no-grad), not `ref_model`.
- **`_cat_oracle_weights`** (L271–314): weight 1 on GT positions whose category in an externally supplied cache is in
  `cat_target_cats` (default "3" = factual span), excluding punctuation/specials; used for GA. This is MDU-code's own
  *fact-position-restricted GA*; the cache is not released.
- **Static finding A — `run_main.sh` omits the objective flags.** It passes **no `--loss_type` and no `--match_mode`**,
  so the defaults `loss_type="ga"` (L1432) and `match_mode="token_id"` apply. The LLaDA/Dream presets therefore run
  *trajectory-matched GA* (all regenerated content tokens at `novel_percentile 100`); `--null_anchor_tau` is ignored
  and `ref_model=None`. The Dream file's docstring (L16) does pass `--loss_type null_anchor --match_mode random`, so the
  paper's runs probably used them and the script is incomplete. **Hyperplane MDU arms must set these flags explicitly
  and log the resolved args.**
- **Static finding B — training states never cover the deployed answer length.** Labels are the chat-template GT
  answer, collator padding is −100, and `MDUCollator.after` drops `attention_mask`. Training states cover **only the
  GT-length answer region (no EOS padding to gen_length)**, while deployment generates 128 tokens with EOS fill
  (version caveat: dllm HEAD). No PEFT/LoRA in the code (C agrees).

### 2.2 The other starting points
- **ARL** (SNIPPET-ONLY): DLM unlearning is highly sensitive to generation hyperparameters; facts resurface when the
  target is embedded in informative prefix/suffix context. Method: an RL context-generator adversary vs the DLM, with
  credit-assignment and stability components. **Unknown:** whether the DLM side samples decoding configs in training
  — the key unresolved overlap with FX-Mix's pool.
- **TimeROME-DLM** (SNIPPET-ONLY): training-free causal tracing over denoising steps plus a low-rank edit at the traced
  coordinate, re-applied at every step, weights frozen. Author-reported −83 nats on TOFU forget01 (LLaDA-8B-Base),
  transferring to other DLMs. Inference-time editing (category iii), not a weight-update competitor; relevant to C.
- **LRCU** (CODE-READ):
  - Recipe: capped GA, L=−mean clamp(CE, cap=1); min-SNR t-weighting; 3 causally localised blocks (e.g. 5–7);
    Bernoulli(t) TF masks, t∈[0.1,0.9]; CE retain; 500 steps at lr 5e-5. Also an MDU-style `AnchorForget` with an
    **all-mask frozen anchor** (student-first KL) and reliance-gated caps from all-mask CE.
  - TF states only, no rollouts; evaluation on one greedy-lc config (gen 64 / steps 64 / block 32). Author records
    (n=100) show low base forget ROUGE (0.21–0.32).
  - Records name a public base `Jameszhiwei/llada-tofu-forget10-SFT`; README lists `llada-8b-tofu-sft`. Both NOT
    ACCESSED, both candidate Hyperplane bases.
- **A2D** (SECONDARY): EOS targets at **all** masked positions of harmful responses, uniform-t TF masking, plus a
  retain set (refusal, type ii). Authors claim robustness across L2R, confidence and random orders **with TF states
  only** — the strongest published counter-hypothesis to FX-Mix's rollout argument.

### 2.3 Search by operation
- **(a) Same-state teacher with target tokens removed or logit-penalised.**
  - **TNFF** (`models.py` L584–641): at negative-weight positions, the original softmax with *that position's output
    token* zeroed and renormalised; at positive positions, the original softmax. CE (= forward KL + const), weight
    α=1e4 on negatives; TNRR is the reverse-KL variant. TNT abstract (snippet): negatives drawn from **the model's own
    generations**.
  - **UNDIAL** (open-unlearning `compute_undial_loss`): teacher softmax(z_ref − β·onehot(label)) at every position, CE,
    GradDiff retain. The re-implementation averages over all positions with no label mask inside the function.
  - Unilogit (snippet): adaptive target logit giving the forget token uniform probability, from the current model.
    RKLU (snippet): reference minus forget-finetune logit increment, reverse KL. WHP (memory): reinforced-model
    generic logits.
  - **Mapping:** A's r (ε=0) = TNFF with "negative token" → set F at every masked position. B's r_ε = UNDIAL/Unilogit
    with γ chosen so r(F)=ε. New in the target: only set-valued F at every masked MDM position (rests on TNFF/UNDIAL
    code, both single-token per-position).
- **(b) On-policy / self-generated states.**
  - TNT: θ0's generations (fixed dataset). MDU `_traj_rollout_na_loss`: current-model MDM rollouts (code).
  - Recovery Alignment (code): GRPO on the current LoRA model's rollouts (random remasking, T=0.7, 128 steps),
    started from **contaminated** TF states (a few GT harmful tokens visible, initial step 2–16), reward-model score,
    k3-KL to the adapter-off reference.
  - Reveal&Release and DareU (snippets): self-generated forget data / RL. No DAgger/concentrability analysis for
    unlearning found in 2 queries — **not** evidence of absence. GKD analogy from memory.
- **(c) dLLM unlearning/safety on trajectory states or multiple orders.** MDU; EMDU (all-answer-masked anchor =
  state-*independent*, whereas FX-Mix's r is same-state and prompt-conditional); LRCU (TF only); A2D (TF only,
  any-order claim); Recovery Alignment (rollouts *from* fact-visible states — the opposite of FX-Mix's restriction);
  ARL (adversarial contexts). DiffuGuard, RAEC and CURE are inference-time (category iii).
- **(d) Banned-token decoding distilled into weights.** No direct hit in 1 query; TNT is closest. That FX-Mix's fixed
  point *is* ban-decoding of the reference on fact-free states is D's derivation (§3.2).
- **(e) For MSDA.** **ReGLU/RILA** (Findings ACL 2026, snippet) — closest found: LoRA initialised from eigenvectors of
  a *balanced covariance contrasting forget and retain* activations, plus a retain-orthogonal output regulariser
  (TOFU/WMDP). CorDA (snippet): covariance-oriented SVD init; knowledge-preserved mode trains the smallest components;
  in PEFT ≥0.15. Null-space frozen-A LoRA for MLLM unlearning (2605.05909, snippet). GROM (snippet): ridge
  least-squares closed form, forward-only. AlphaEdit, MEMIT, LoRA-FA: memory only.

## 3. Reduction (task 3)

### 3.1 FX-Mix ingredients and their provenance
| ingredient | FX-Mix choice | provenance (access) |
|---|---|---|
| per-position target | p_θ0(·\|c,s,i) with F removed (ε=0) or softmax(z0−γ1_F) (ε) | TNFF (code); UNDIAL (code), γ=∞; Unilogit adaptive γ (snippet) |
| divergence direction | KL(r ‖ p_θ), target-first | TNFF CE (code); EMDU anchor-first (snippet); MDU `reverse` flag (code) |
| teacher state | **same** state, prompt visible | TNFF/UNDIAL same-context teacher (code); MDU masks the prompt; EMDU masks the whole answer |
| on-policy MDM states | current-student native sampler, refreshed | MDU `_traj_rollout_na_loss` (code), single greedy config |
| config pool | lc-greedy + rnd T=1 (1 tok/step) + rnd T=1 (≥2 tok/step) | none found (ARL unresolved) |
| first-commit truncation, decision state kept | yes | none found; DAgger/first-passage arguments are standard (memory) |
| fact-free TF states (F force-masked) | fraction α | none found; MDU `cat_oracle` labels fact positions but uses them for GA |
| retain | LLaDA SFT retain, λ=1 | MDU/GD (paper, code) |
| analysis | I-projection; first-commit decomposition; hazard union bound for T∈[0,1] | I-projection: TNT (snippet). First-passage/union bound: standard. MDM-sampler instantiation: A/B |

### 3.2 Closest known algorithm, and the simplest alternative explanations
- **Closest known algorithm, full-algorithm level.** MDM: MDU with `--loss_type null_anchor --match_mode random
  --null_anchor_traj_rollout True` — same loop, retain term, optimiser and on-policy rollout states; differs in target
  (anchor), KL direction, state filter and single config. AR: TNT/TNFF on the model's own generations (FX-Mix's target
  on θ0-sampled, unrefreshed states); with refresh, on-policy distillation from a ban-masked teacher (GKD-style, memory).
- **Fixed-point identity (D's derivation).** If p_θ = r (ε=0) on every fact-free state the student can reach, its
  native sampler is *the reference sampler with F banned* (logits[F]=−inf), because every visited state is fact-free.
  FX-Mix therefore distils the decode-time F-ban of θ0 into the weights on the states ban-decoding visits; at
  convergence the on-policy states equal the ban-sampler's (the DAgger fixed point). This yields the
  **known-objective control** of §4: decode-time ban on the base model (an inference-time reference, not a
  deliverable).
- **Simplest alternative explanations of any FX-Mix advantage over MDU/NPO:**
  - **(E1) Target alone.** The exclusion target flips the fact's rank at every trained state, with zero gradient where
    p0(F)=0, and shared parameters may carry TF-trained suppression onto decode states. Supporting: A2D's TF-only
    multi-order claim; A's toy FFX-TF 0.094 vs FFX-OP 0.079.
  - **(E2) More forget pressure per step.** Extra rollout states add forget gradient, so comparisons must be strength-
    and compute-matched.
  - **(E3) Metric-objective coincidence.** Token-level leakage of F is exactly what the loss bans everywhere, so any
    global F-ban scores well. Check string-level, paraphrase and shared-token retain damage.
  - **(E4) KL direction/anchor.** EMDU attributes MDU's under-forgetting to visible-token anchors and to student-first
    KL stalling at near one-hot predictions; part of the gain may be just that.

### 3.3 Comparison table
| method | objective & target | training-state distribution | parameterisation | optimiser / update order | teacher / reference | inference-time behaviour | selection / stopping | compute & memory per forget state (fwd=1, bwd≈2) |
|---|---|---|---|---|---|---|---|---|
| MDU (paper) | KL(p_θ‖anchor^τ/Z), anchor = θ0, prompt masked, same y_t | TF: t~U, iid random masks of the GT-length region | full FT | AdamW cosine, forget+λ·retain per step | frozen θ0 copy | plain sampler | τ sweep, fixed epochs | 3+1 = 4; 2 model copies |
| MDU rollout flag (code) | same | 1 state/step: current-model greedy-lc rollout, T=16, random stop; no filter | full FT | same | frozen θ0 | plain | as MDU | 4 + ~8 no-grad ≈ 12 |
| TNFF (AR, code) | CE to θ0 with negative token zeroed (negatives); θ0 itself (positives) | θ0's own generations, fixed set | full FT (T5X) | Adafactor/T5X (memory) | frozen θ0 | plain | α=1e4 weight | 1 teacher fwd + 1 fwd/bwd |
| UNDIAL (AR, code) | CE to softmax(z0 − β·onehot(label)), all positions | GT forget sequences (TF) | full FT / LoRA | AdamW + GradDiff retain | frozen θ0 | plain | β | as TNFF |
| EMDU (snippet) | KL(anchor‖p_θ), anchor = θ0 with the answer span fully masked | TF (presumed MDU-like; unverified) | unknown | unknown | frozen θ0 | plain | unknown | ≈ MDU |
| **SC** (§3.4) | KL(r‖p_θ), r = θ0 same state minus F | MDU rollout flag (+ MDU TF variant) | as MDU (or LoRA) | as MDU | frozen θ0 / adapter-off | plain | as MDU | ≈ 12 |
| **FX-Mix** | KL(r‖p_θ), r = θ0 same state minus F (ε=0 or ε) | α fact-free TF (F force-masked) + (1−α) states from a pool of 3 native configs at gen_length, truncated at first F commit, J states/rollout, refresh every R steps | LoRA or full FT | AdamW, forget+retain; rollouts no-grad | frozen θ0 / adapter-off | plain sampler (no filter) | matched max-over-config forgetting; certificate Ĉ_π as diagnostic | ≈ 4 + amortised rollouts ≈ 7 (B) to 3–4× MDU (A, n_op=1, K=8) |
| MSDA (C) | MDU loss + retain-KL | TF + rollout keys for statistics; TF+rollout training states | frozen-A LoRA, A = top generalised eigvecs of (C_F, C_R) at [MASK] keys; B trained; merged | AdamW on B only | adapter-off θ0 | plain (merged) | rank rule μ≥τ | +Stage-1 ≈ 320 fwd/pair; m×m accumulators (≈14 GB for 8 layers) |

### 3.4 Strongest straightforward combination (SC) and what FX-Mix adds
**SC definition.** MDU's training loop *as released*, with `--null_anchor_traj_rollout True` and target-first KL
(`--null_anchor_kl_dir reverse`). Replace `_null_anchor_kl`'s target with TNFF/UNDIAL-γ=∞ over the token set F at the
**same** state, prompt visible (one line: `logits_ref[..., F] = -inf`). Keep the retain SFT term. **SC-TF** = SC with
the rollout flag off (MDU's random TF states). Both need about 10 lines of change to MDU's code and no new concept.

**FX-Mix − SC**, with the observation that would show each operation matters (at matched retention, or matched mean
forgetting):

| # | added operation | observation |
|---|---|---|
| 1 | **Config pool**: random remasking T=1 with 1 and ≥2 tokens/step, alongside greedy-lc | Max leakage over **held-out** configs: FX-Mix < SC (single greedy config), with SC's leakage concentrated in rnd/T>0/parallel cells. B's tabular toy predicts SC-like leak_max≈1.0 vs 0.011. Null ⇒ #1 redundant. |
| 2 | **First-commit truncation**, decision state kept, no post-leak states | Ablation "pool without truncation" at equal state budget; predicted same or worse forgetting with more retain damage. Equal ⇒ only a compute saving. |
| 3 | **Fact-free TF states** (F force-masked) replacing random masks that can leave F visible | Ablation: α-states random-masked vs force-masked. By A's A2, fact-visible states cannot host a first commit; expected small. None ⇒ drop it. |
| 4 | Answer region at gen_length with EOS padding (MDU uses GT length; static finding B) | SC at gen_length vs GT length. If gen_length SC closes the gap, the "advantage" was a length/format confound. |
| 5 | J states per rollout + buffer refreshed every R steps (MDU: 1 random-stop state per step, refreshed each step) | Cost/variance only; check wall-clock-matched curves. |
| 6 | Hazard bound and certificate Ĉ_π | A diagnostic, not an update. "Matters" only if Ĉ_π tracks held-out leakage (B's toy: within ×1–16 for the expectation form). |

**Mechanism observation (cheap, decisive).** For each trained model and config, take first-commit decision states from
fresh rollouts and record the **fraction off the GT-masked manifold**. FX-Mix's claim needs that fraction to be
substantial for SC-TF/SC at the deployed configs, and FX-Mix to lower hazard there. If decision states stay
on-manifold, E1 explains the result and the method is SC-TF.

### 3.5 MSDA reduction
- **Closest:** ReGLU/RILA (SNIPPET-ONLY) — LoRA from forget-vs-retain contrastive-covariance eigenvectors plus a
  retain-orthogonal penalty. Others: CorDA-KPM (covariance-oriented init), LoRA-FA (frozen A). Proposition 1(a,b) is a
  textbook generalised Rayleigh-quotient bound.
- **Straightforward combination:** LoRA-FA with A frozen to RILA/GEP directions computed on **[MASK]-position keys over
  TF and rollout states**, trained with the chosen forget loss. That *is* MSDA.
- **What MSDA adds:** the key distribution (masked readout positions, rollout states), A frozen (RILA trains A), and
  the μ≥τ rank rule. C's toy already has a RILA-like arm ("GEP-init, trainable A": THR 0.10 mean .261 vs MSDA-rule
  per-seed .173/.104/.334) — mixed at n=3.
- **Verdict:** orthogonal to FX-Mix (a parameterisation that can host the FX target); no distinct algorithm beyond
  "RILA on MDM mask-state keys with frozen A". Whether a frozen row space sits too close to the closed VAU projection
  family is the coordinator's call (C concedes ∇_B = ∇_W·Aᵀ).

### 3.6 Verdict on distinctness
- **Known:** the target (TNFF/UNDIAL family); the divergence direction and anchor diagnosis (EMDU, snippet); the
  on-policy MDM loop (MDU code); the retain term.
- **Not found in anything inspected:** a **multi-config native-sampler state pool with first-commit truncation**,
  justified by a sampler-uniform first-commit bound, for MDM unlearning. This rests on code read of MDU, LRCU, Recovery
  Alignment and t5patches, plus snippets of ARL, EMDU, A2D and others; ARL's training details are the main unknown.
- **Assessment:** FX-Mix is a *recombination of known pieces plus one state-distribution design*. It is a distinct
  method only if #1 (and possibly #2) beats SC and SC-TF under the controls below; otherwise report it as
  "TNFF-in-MDU" (= SC-TF), crediting the target to TNT/UNDIAL and the direction to EMDU.

## 4. Controls (task 4)

### 4.1 CPU toy
Substrate D1–D4 (SHARED_CONTEXT §7). Designate one config (D3 or D4) as **held out of the pool** before running.

| id | arm / check | isolates |
|---|---|---|
| T0a | Forget loss at θ0 = Σ −log(1−p0(F\|s,i)); gradient 0 where p0(F)=0 (unit test) | implementation of r and the KL direction |
| T0b | Recorded rollout states match the substrate's `generate` port (same RNG, same commits) | the state recorder |
| T1 | **Decode-time F-ban on base** (logits[F]=−inf when sampling forget prompts; inference-time reference only) | FX-Mix's known fixed point: best forget-prompt fluency/leakage for this target. FX-Mix − T1 = distillation error; FX-Mix retain damage vs 0 = cost of moving it into weights |
| T2 | Retrain checkpoint (`retrain.pt`) | behavioural reference: chance leakage, well-formedness ≈1 |
| T3 | **SC-TF**: r target, MDU random TF states, target-first KL, retain SFT | E1 (target only) |
| T4 | **SC**: T3 + MDU rollout flag (greedy-lc, T=16, random stop, no truncation, GT-length region) | the straightforward combination |
| T5 | FX-Mix (A/B merge) | the candidate |
| T5a | T5 minus pool (greedy-lc only, with truncation) | operation #1 |
| T5b | T5 minus truncation (all rollout states kept) | #2 |
| T5c | T5 with random-masked TF states instead of F force-masked | #3 |
| T5d | ε=0 vs ε=1e-3; student-first KL (finite only with ε>0) | A vs B target; direction |
| T6 | Compute/strength-matched T3 (extra TF states = T5's forward budget), plus "noisy-TF" (visible non-F tokens resampled from p0) | E2; whether cheap off-manifold coverage replaces rollouts |
| T7 | MDU τ=1, GD, NPO | literature baselines (coordinator probe exists) |

Metrics per config (incl. held-out): leak_any, exact-value leak, well-formedness, retain recall/decode-TV, steps to
threshold, off-manifold first-commit fraction. Match via a strength knob (λ or LR × checkpoint); report infeasible arms.

### 4.2 Hyperplane (LLaDA-8B-Instruct, TOFU-SFT)
- **H0 — zero-training diagnostics on base:**
  - (a) MDU-anchor vs same-state p(F) on fact-free TF states — predicts MDU τ=1 under-forgetting (EMDU's mechanism);
  - (b) per config, the fraction of first fact commits at off-manifold states, for base and later SC-TF — ≈0 at scale
    ⇒ rollout argument weak;
  - (c) decode-time F-ban (T1 analogue) over the grid — string-level residue measures F-definition gaps.
- **H1 — literature baselines:** GD; NPO; MDU τ∈{0.5,1} with **explicit** `--loss_type null_anchor --match_mode
  random` (static finding A); MDU rollout flag; an EMDU-style arm (anchor-first KL to an all-answer-masked frozen
  anchor), reconstructed from the abstract and labelled as such.
- **H2 — reduction arms:** SC-TF, SC, FX-Mix, T5a/T5b, compute-matched SC-TF. All arms use one answer-region
  convention (gen_length with EOS padding) or report both (static finding B), and share LoRA/FT choice, retain data
  and LR grid.
- **H3 — outside the bound's scope (reported, not claimed):** held-out configs (steps/block not in the pool), CFG>0,
  T>1, other gen_length; paraphrased prompts; prefill/priming with a partial answer (Recovery Alignment threat — FX-Mix
  trains no fact-visible states); context embedding (ARL threat); MC answer probability and relearning (separates type
  (i) from (iv)).
- **Protocol:** forgetting = max over the pre-registered grid at matched levels. The primary comparison is FX-Mix vs
  SC and SC-TF; MDU/NPO are context. A win over MDU alone does **not** establish FX-Mix (E1, E4).

## 5. Unresolved comparisons
What each novelty-relevant claim rests on, and what would settle it:
1. **ARL full text/code.** Does the DLM side randomise steps/block/remasking/temperature in training? If yes, the
   config pool (#1) is prior art. (SNIPPET-ONLY.)
2. **EMDU full text.** State distribution (any rollouts?), exact anchor, direction, results; it may already contain
   SC-TF-like arms. (SNIPPET-ONLY, posted 2 days before this sprint.)
3. **TNT paper.** Whether negatives are re-sampled from the updated model (on-policy) and whether token *sets* are
   handled; A's I-projection attribution is SNIPPET-ONLY.
4. **On-policy distillation in diffusion LMs with a banned teacher** not searched (budget; GKD from memory). **A2D**
   state details/code (secondary only); was its any-order claim tested under T>0 / parallel commits?
6. **MDU's TOFU decoding config** (block length; 128 vs 256 steps) and dllm version. LRCU's HF checkpoints and the
   public `Jameszhiwei/llada-tofu-forget10-SFT` (unverified; HF blocked).
7. **ReGLU/RILA** details (generalised eigenproblem? trainable A?) for MSDA; CorDA-KPM as a control. **PMLR v306
   "zhang26di"** identity not verified; "dLLM knows the answer before it decodes" (MDU ref [38]) not read.

## 6. Verification of snippet-only sources (added on coordinator request, ~04:45Z)
**Caveat on the tool.** WebSearch returns a model-written summary plus a link list. It does **not** return verbatim
page text. Quotes below are verbatim *tool output*, not verbatim paper text. None of the pages could be fetched
(arxiv and icml.cc are refused by the proxy).

**EMDU — reproduced (link + summary) in 3/3 re-runs:**
- Queries (identical to the originals):
  - Q1 `"Enhanced Masked Diffusion Unlearning" EMDU anchor-first KL answer span masked` (extended);
  - Q2 `masked diffusion language model unlearning 2026 LLaDA forget set rollout states sampler new method arXiv`
    (extended);
  - Q3 `EMDU masked diffusion unlearning near one-hot gradient vanishes reversed KL anchor visible forget tokens TOFU
    RWKU LLaDA Dream 2610.25447` (extended).
- Link returned by all three: title "[2610.25447] Enhanced Masked Diffusion Unlearning", URL
  https://arxiv.org/abs/2610.25447.
- Q3 tool text: "The paper is arXiv 2610.25447, "Enhanced Masked Diffusion Unlearning," by Tomer Garber and Tom Tirer,
  submitted October 18, 2026 … random answer masking can leave key forget tokens visible to the anchor even when the
  question is fully masked … at a memorized (near-one-hot) prediction, the anchor-last logit gradient vanishes while
  the anchor-first divergence logit gradient does not … EMDU, which uses anchor-first KL divergence toward a frozen
  anchor whose entire answer span is masked."
- Q1 tool text: "on TOFU and RWKU using LLaDA-8B and Dream-7B, EMDU matches or exceeds MDU and adapted autoregressive
  baselines on forgetting, while better preserving retain utility". Q3 instead says "The results don't show TOFU or
  RWKU numbers … so I can't confirm the experimental claims".
- Q2 contains a garbled sentence attributing MDU's anchor-last objective to EMDU.
- **Status: existence, title, URL, authors, date and the two failure modes are consistently reproduced. The results
  claim is unconfirmed. Not verified against the page itself.**

**ARL — reproduced (link + summary):**
- Query: `adversarial reinforcement learning robust unlearning diffusion large language models ICML 2026`. Link: "Adversarial
  Reinforcement Learning for Robust Diffusion Large Language Model Unlearning", https://icml.cc/virtual/2026/poster/65904.
- Tool text: "unlearning results shift with generation hyperparameters … DLMs can reproduce supposedly erased content
  when it sits inside informative context … A context generator adversarially produces informative contexts to elicit
  unlearned knowledge, while the DLM is optimized to suppress undesired recall … the code is anonymized".
- No tool output says whether training randomises decoding configs (unresolved item 1 stands). Authors/PMLR ID are not
  reproduced.

**A2D — the "TF-only multi-order" claim is D's inference, not a quoted claim:**
- Source file: the third-party note https://raw.githubusercontent.com/zhaoyang97/Paper-Notes-en/main/docs/ICLR2026/llm_alignment/a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md
  (local copy sha256 df57dbe1…7cca).
- Verbatim from the note: "For the Harmful set, all masked positions are supervised with `[EOS]`" (L66); "A2D samples
  the timestep $t \sim U(0,1)$ for each sample during training and sets the mask ratio as $\lambda = (1-\epsilon)t +
  \epsilon$" (L70); "Effective across three decoding strategies (left-to-right, confidence-based, random)—true
  any-order defense." (L103).
- The note mentions no rollout or self-generated states. "TF only" in §2.2/§3.2 is therefore D's reading of a
  secondary summary and should be treated as **UNVERIFIED** until the paper is read.
