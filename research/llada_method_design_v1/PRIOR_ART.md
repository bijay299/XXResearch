# PRIOR_ART — verified closest comparisons and full-algorithm novelty assessment

Source of record: [`agents/D_prior_art.md`](agents/D_prior_art.md) (Agent D; 23 WebSearch queries, ~12 GitHub searches,
4 repos cloned read-only) plus targeted checks by A, B, C. Network policy blocked arxiv.org, huggingface.co,
proceedings.mlr.press, openreview.net and Semantic Scholar from this host, so several papers are known only from
search snippets; the user supplied the MDU and EMDU PDFs. **Nothing here is a novelty claim from an absent search hit.**
External numbers are **author-reported**; nothing external was reproduced.

## 1. What was actually inspected
| source | identity / pin | status | what was read |
|---|---|---|---|
| LLaDA (official) | github ML-GSAI/LLaDA @ `9182493` | CODE-READ | GUIDELINES.md (pretrain/SFT loss), `generate.py`, `get_log_likelihood.py`, `chat.py` |
| LLaDA modelling code | HF GSAI-ML/LLaDA-8B-* `modeling_llada.py` | NOT ACCESSED (HF blocked) | — module names UNVERIFIED |
| MDU paper | arXiv 2605.18253v1 (user-supplied PDF, sha256 `82e1d2ad…cf3e0`) | PDF-READ | §3–7, App. A–C |
| MDU code | github leegeoru/MDU @ `e5e8e82` | CODE-READ | `unlearn_mdu_llada.py` (all objective paths), `run_main.sh`, configs, eval script |
| dllm (MDU's library) | github ZHZisZZ/dllm @ `ca17675` (HEAD; MDU's version unknown) | CODE-READ (partial) | sampler defaults, MDLM trainer, scheduler, data utils |
| TNT / TNFF | arXiv 2406.13660; github google/t5patches @ `09285b3` | CODE-READ (paper SNIPPET-ONLY) | `models.py` L398–470, L565–641 |
| UNDIAL | arXiv 2402.10052; open-unlearning @ `b299290` re-implementation | CODE-READ (re-impl.) | `undial.py`, `compute_undial_loss` |
| LRCU / Open-DLU | github guan404ming/open-dlu @ `fb98433`; HF collection guanmingchiu/lrcu-dlm-weight-unlearning | CODE-READ; checkpoints NOT ACCESSED | capped-GA forget, t-conditional masks, anchor forget, configs, result JSONs |
| Recovery Alignment (priming vulnerability) | arXiv 2510.00565; github mdl-lab/dlm-priming-vulnerability @ `b2c9769` | CODE-READ | rollout + GRPO training loop |
| Adversarial-RL dLLM unlearning | ICML 2026 poster 65904 (PMLR v306 "zhang26di" not verified) | SNIPPET-ONLY | abstract |
| TimeROME-DLM | arXiv 2606.12841 | SNIPPET-ONLY | abstract |
| A2D | arXiv 2509.23286 | SECONDARY (third-party note) + SNIPPET | method summary |
| EMDU | arXiv 2610.25447v1 (Garber & Tirer, 2026-10-18; user-supplied PDF, sha256 `d8059beb…915a`) | PDF-READ | §1–5, App. C (Alg. 1), D (hyper-parameters), E.1 (MDU reproduction), F (Table 7 robustness, paraphrases, seeds) |
| Unilogit, RKLU, DUET, ASU, DiffuGuard, RILA/ReGLU, CorDA, GROM | various | SNIPPET-ONLY | abstracts |
| WHP, GKD, LoRA-FA, AlphaEdit, MEMIT, DAgger | — | NOT ACCESSED (memory only) | — |

## 2. Closest methods, by component
| component of FX-Mix | closest known | relation |
|---|---|---|
| per-position target r_ε | **TNT/TNFF** (renormalise without the negative token; CE to it) — code read; **UNDIAL** (logit − β at the label; β→∞) — re-impl. read; Unilogit (adaptive, snippet) | **same target** (ε=0) / an ε-floored variant; credited, not claimed |
| target-first KL; anchor problem of MDU | **EMDU** (PDF): anchor-first KL(p_anchor‖p_θ) to a frozen anchor with the prompt **and the full answer span** masked (FAM), on MDU's random TF states, τ=1, + retain CE; proves the anchor-last gradient vanishes at near-one-hot memorised predictions; reports MDU's anchor leaking through visible answer tokens (its Fig. 3) | pre-empts our observation about MDU's anchor (A's F2) and the KL direction; its forgetting mode is mostly empty EOS-only completions (71.8 % / 99.0 % of TOFU forget queries, LLaDA / Dream) |
| on-policy MDM states | **MDU code `_traj_rollout_na_loss`**: current-model greedy low-confidence rollout, 1 block, random stop, 1 state per step, anchor target, no fact filter (not in the paper's main method) | same loop idea, single config |
| decoding-robust DLM unlearning | **ARL** (snippet; also described in EMDU §2.1 as: unlearning performance sensitive to generation length and denoising steps; a separate LLM generates eliciting contexts, MDLM trained with GRPO to suppress) | unresolved whether its DLM side randomises decoding configs; EMDU evaluates (not trains) over ARL's 6 (length, steps) combinations, low-confidence decoding only |
| any-order robustness from TF states | **A2D** (secondary): EOS targets at all masked positions, uniform-t TF masks, claims robustness to several decoding orders | strongest published *counter-hypothesis* to our coverage argument |
| retain term, optimiser | MDU / GD | identical |
| analysis | I-projection (Csiszár; TNT); first-passage + union bound; concentrability/DAgger | standard tools, instantiated for the pinned sampler |

## 3. Strongest straightforward combination (SC) and what FX-Mix adds
**SC** = MDU's released loop with `--loss_type null_anchor --match_mode random --null_anchor_traj_rollout True`,
target-first KL, and the anchor replaced by the same-state reference with `logits[F] = −inf` (≈10 lines). **SC-TF** =
SC without the rollout flag (= "TNFF dropped into MDU's random teacher-forced loop").

FX-Mix − SC (each with the observation that would show it matters):
1. **Config pool** (greedy low-confidence + random-order T=1 one token/step + random-order T=1 multi-token/step) —
   shows if held-out-config leakage at matched retention is lower than SC's single greedy config.
2. **Truncation at the first fact commit, keeping the decision state** (no post-leak states) — ablation.
3. **Fact-free TF states** (fact positions always masked) instead of random masks — ablation, expected small.
4. **gen_length answer region with EOS padding** (MDU's code trains on the GT-length region, D static finding B) —
   held constant across all our arms, so not a source of our toy differences.
5. Hazard bound / certificate — analysis, not an update.

## 4. Full-algorithm comparison
| | MDU (paper) | MDU rollout flag (code) | EMDU (paper) | TNFF (AR) | SC-TF / SC | **FX-Mix** |
|---|---|---|---|---|---|---|
| objective & target | KL(p_θ‖anchor^τ/Z), prompt-masked anchor | same |KL(p_anchor‖p_θ), anchor: prompt + whole answer masked (state-independent per-position marginal)| CE to θ0 minus negative token | KL(r_0‖p_θ), same state | KL(r_ε‖p_θ), same state |
| training states | TF random masks, GT-length region | 1 random-stop greedy rollout state/step |TF random masks (MDU's)| θ0's own generations (fixed) | TF random (SC: + MDU rollout flag) | fact-free TF + pool of 3–4 native configs, truncated at first F commit |
| parameterisation | full FT | full FT |not stated (no LoRA/adapter mentioned)| full FT (T5X) | as chosen (our toy: LoRA) | LoRA (both factors trained; B=0 init) |
| optimiser / update order | AdamW cosine, forget+λ·retain | same |AdamW, forget + α·retain per step| Adafactor (memory) | AdamW | AdamW; rollouts no-grad every R steps |
| teacher / reference | frozen θ0 copy | frozen θ0 |frozen θ0 copy| frozen θ0 | frozen θ0 / adapter-off | adapter-off pass (no copy) |
| inference-time | plain sampler | plain |plain (EOS-only answers on most forget queries)| plain | plain | plain (merged LoRA) |
| selection / stopping | τ sweep, fixed epochs | — |LR × epochs sweep, largest forget/retain gap| α weight | strength grid | strength grid; post-hoc certificate as diagnostic |
| compute per forget state | 4 fwd-eq. | ≈12 |4 fwd-eq.| ≈4 | 4 / ≈12 | ≈7 (LLaDA projection) |

## 5. Novelty assessment (full algorithm)
- **Known:** the target (TNT/TNFF, UNDIAL γ→∞), the divergence direction and the MDU-anchor diagnosis (EMDU, PDF-read),
  the on-policy MDM rollout loop (MDU code), the retain term, and all mathematical tools.
- **Not found in anything inspected:** a *multi-configuration native-sampler state pool with first-commit truncation*,
  justified by a sampler-level first-commit analysis, for MDM unlearning. This rests on code reading of MDU, LRCU,
  Recovery Alignment and t5patches, and the MDU and EMDU papers; ARL and A2D are snippet/secondary only, and **ARL's
  training details are the main unresolved overlap** (EMDU's description of ARL mentions an adversarial context generator
  and GRPO, not decoding-config randomisation).
- **Contribution type, if any:** a training-state-distribution design (update procedure) — a recombination of known
  pieces plus one state-distribution operation. It is eligible as a method only if that operation is shown to matter
  against SC and SC-TF; otherwise the honest name is "TNFF-in-MDU" with the target credited to TNT/UNDIAL.
- **Empirical status of the added operation:** see RESULTS.md (locked CPU validation) — filled in from the locked run.

## 6. Unresolved comparisons
ARL full text/code (does it randomise decoding configs?); EMDU code (not yet released; paper says upon acceptance); TNT paper
(on-policy negatives? token sets?); A2D code (tested under T>0 / parallel commits?); on-policy distillation with
constrained teachers in diffusion LMs (not searched); MDU's exact TOFU decoding config and dllm version; LRCU's and
`Jameszhiwei/llada-tofu-forget10-SFT` checkpoints (HF blocked); RILA details (for C's MSDA).

## 7. MSDA (C's parameterisation track) — not selected
Reduces to RILA-style contrastive-covariance adapter directions on [MASK]-position keys with frozen A (D §3.5); as
specified it trains MDU's anchor loss, whose fixed point keeps the anchor's fact mass (A's objection: max-over-config
leak 1.00); its frozen retained-statistics row space is close to the closed VAU family. Possible future use only as
a parameterisation hosting the FX target (A's check: max-leak 0.034 at retain TV 1e-4 in A's log-linear toy).
