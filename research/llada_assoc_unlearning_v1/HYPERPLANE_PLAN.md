# HYPERPLANE_PLAN — one bounded real-model comparison: TD-SSD (X) vs SSD (Z) on LLaDA-8B + LoRA

**Status.** The code is ready to run. **Real-model execution is untested**: the authoring host has no GPU and did not
load LLaDA weights. Every GPU time and memory figure below is **PROJECTED** until measured. Hyperplane execution
requires a separate instruction. This plan neither inherits nor spends any earlier budget.

## 1. Inputs and dependencies
- **Code.** This repository at the commit named in `START_HERE.md`, path `research/llada_assoc_unlearning_v1/llada/`:

  | file | purpose |
  |---|---|
  | `tdssd.py` | losses, data prep |
  | `train_tdssd.py` | entry point |
  | `eval_tdssd.py` | raw-generation evaluation |
  | `compare_tdssd.py` | predeclared verdict |
  | `adapt_data.py` | data adapter |
  | `configs/{tdssd,ssd}.json` | arm configurations |

- **Upstream sampler.** `git clone https://github.com/ML-GSAI/LLaDA && git -C LLaDA checkout 9182493720ed723ef8031210d85959364e51cbe0`.
  `generate.py` is imported from it through `--llada-repo LLaDA`.
- **Model.** The **memorised** checkpoint qualified by Hyperplane. It is either
  - a merged model directory, passed as `--model <dir>`, or
  - base `GSAI-ML/LLaDA-8B-Instruct` at the revision your PoC brief pins (08b83a6feb34df1a6011b80c3c00c7563e963b07)
    plus the memorisation adapter, passed as `--mem-adapter <dir>`, which is merged before the unlearning LoRA is
    attached.

  The frozen reference p0 is this memorised model with the unlearning adapter disabled.
- **Python.** torch with CUDA, `transformers` compatible with LLaDA's remote code (LLaDA's README uses 4.38.2), and
  `peft` (any version with `get_peft_model`, `disable_adapter`, regex `target_modules` and `save_pretrained`). The code
  has no other heavy dependencies. **No additional model and no paid teacher.**
- **Data.** 8 forget facts and 16 retain facts, each with an original question, one prewritten paraphrase, an answer
  and a short `answer_key`. Each forget fact also needs its **`subject` string, appearing verbatim in its question**.
  Convert with:
  ```
  python llada/adapt_data.py --in facts.jsonl --out-dir data_tdssd --q-field question --para-field paraphrase \
      --a-field answer --key-field answer_key --subject-field subject --split-field split \
      --forget-value forget --retain-value retain
  ```
  - The adapter prints the retain split.
  - Unless the input sets `in_retain_loss`, every second retain fact within each stratum (shares an answer word with a
    forget fact / does not) is **held out of L_R**: 8 train, 8 held out. Δ_R is measured on the held-out
    shared-answer facts.
  - If fewer than about 4 held-out retain facts share an answer word, the prediction test is weak; report it as such.

## 2. Commands (2 GPUs; X and Z run concurrently)
```
export R=research/llada_assoc_unlearning_v1; M="--model <memorised dir>"   # or: --model GSAI-ML/LLaDA-8B-Instruct --mem-adapter <dir>
D="--forget data_tdssd/forget.jsonl --retain data_tdssd/retain.jsonl --llada-repo LLaDA"
# 0. prepare + native gates + names for the p0 evaluation (GPU0, ~5 min PROJECTED)
CUDA_VISIBLE_DEVICES=0 python $R/llada/train_tdssd.py --config $R/llada/configs/tdssd.json $M $D --out runs/prep --prepare-only
CUDA_VISIBLE_DEVICES=0 python $R/llada/eval_tdssd.py $M $D --train-out runs/prep --out runs/eval_p0.json
# 1. train both arms (one GPU each, in parallel; checkpoints at steps 50 and 150)
CUDA_VISIBLE_DEVICES=0 python $R/llada/train_tdssd.py --config $R/llada/configs/tdssd.json $M $D --out runs/X &
CUDA_VISIBLE_DEVICES=1 python $R/llada/train_tdssd.py --config $R/llada/configs/ssd.json   $M $D --out runs/Z &
wait
# 2. evaluate the four predeclared checkpoints
for A in X Z; do for S in 50 150; do
  python $R/llada/eval_tdssd.py $M $D --train-out runs/$A --adapter runs/$A/ckpt_$S --out runs/$A/eval_$S.json
done; done
# 3. predeclared comparison
python $R/llada/compare_tdssd.py --theta0 runs/eval_p0.json --x50 runs/X/eval_50.json --x150 runs/X/eval_150.json \
    --z50 runs/Z/eval_50.json --z150 runs/Z/eval_150.json --out runs/compare.json
```
`diff runs/prep/names.json runs/X/names.json` and the same `diff` against `runs/Z/names.json` must both be empty.
Name choice depends only on the data, the seed and p0's greedy answers; a mismatch indicates nondeterministic
generation and must be reported.

## 3. Native gates (abort or report before reading any unlearning output)
| # | gate | where | pass condition |
|---|---|---|---|
| N1 | LoRA module names | `native_gates.json: linear_leaf_names, lora_wrapped_per_block` | 7 wrapped linears in every block, LM head unwrapped (asserted). On failure, set `lora_target_regex` to LLaDA's real names, record the change, and keep it identical for both arms |
| N2 | adapter is a no-op at init; reference stays frozen; adapter active later | `init_adapter_on_minus_off_maxabs`; `ckpt_*/ref_invariance.json` | 0 at init (asserted); `ref_invariance_maxabs` = 0 and `adapter_on_minus_off_maxabs` > 0 at both checkpoints |
| N3 | sampler port vs upstream `generate` | `sampler_port_agrees_with_official` | all true (the port is used only as a fallback) |
| N4 | save/reload | eval stdout and json `save_reload_logit_maxabs` | ≤ 1e-2 (bf16) |
| N5 | special tokens / template | `mask_eos_eot` | `<|mdm_mask|>`, `<|endoftext|>`, `<|eot_id|>` for ids 126336/126081/126348. If the memorisation SFT put `<|eot_id|>` before the EOS padding, report it (answer layout here: tokens + EOS) |
| N6 | name acceptance on the real tokenizer | `names.json` | a teacher name for every fact and ≥ 8 controls per fact. Report how many names came from the fictional fallback pool |
| N7 | memorisation qualification | `eval_p0.json` summary | forget disclosure and retain accuracy high on both phrasings (Hyperplane's own qualification threshold); `copy_success` high |
| N8 | compute | `compute.json`, history `peak_mem_gib` | record. If memory overflows, use `retain_batch 4`; this is an engineering change, applied to both arms and disclosed |

`attention_mask` is never passed: rows are right-padded with EOS (GUIDELINES SFT layout), and generation uses batch 1
without padding.

## 4. Resources (all PROJECTED, A100/H100 80 GB, bf16)
- **Memory.** About 16 GB of weights. The LoRA (r 16 on 7 linears × 32 layers) is about 40 M parameters plus Adam
  state. Activations cover 24 gradient rows × ~70 tokens, with fp32 logits only at masked positions. Peak is projected
  at roughly 25–40 GB per arm.
- **Time.**
  - Training: about 1–2 s/step, so 150 steps take ≈ 3–5 min per arm, plus model load and y_ν generation (8 + rejected
    candidates) of ≈ 2–4 min.
  - Evaluation: each checkpoint needs 72 greedy generations × 32 forwards plus ≈ 100 likelihood forwards, ≈ 3–6 min
    including load. There are 5 evaluations.
- **Total.** ≈ 1–1.5 GPU-hours (≤ 2 GPUs concurrently), well inside the 6 GPU-hour allocation. CPU and storage stay
  small: adapters are ≈ 160 MB each × 4.

## 5. Predeclared comparison (frozen before any Hyperplane output; implemented in `compare_tdssd.py`)
- **Arms.**
  - X = TD-SSD (`configs/tdssd.json`).
  - Z = SSD with a rotating anchor (`configs/ssd.json`).
  - Same LoRA, AdamW, LR 1e-4, 150 steps, seed 0, states and data.
  - Checkpoints at steps 50 and 150 for each arm.
- **Matching variable: D_F.** The forget distance to the stranger target, on the fully masked answer, over both
  phrasings.
- **Y = Δ_R.** The log-probability drop of the answer-word span on **held-out** shared-answer retain facts, over both
  phrasings, relative to p0.
- **W = ρ.** Stranger drift: the mean |Δ log p(object span)| on 8 evaluation-stranger canvases, divided by the same
  quantity on the subject canvas.
- **Verdict.**

  | verdict | condition |
  |---|---|
  | **SUPPORTED** | at every comparable X checkpoint (D_F within Z's range): Δ_R(X) ≤ 0.7·Δ_R^Z(matched D_F) **and** ρ(X) ≤ 0.3·ρ^Z |
  | **CONTRADICTED** | Δ_R(X) ≥ Δ_R^Z at every comparable checkpoint |
  | **INCONCLUSIVE** | otherwise |
  | **NOT_COMPARABLE** | X did not forget enough (the L2 limiting case) |
  | **SUPPORTED_Y_ONLY** | Y passes but W fails; the gain is not credited to the mechanism |

  One seed and 8 facts give a **directional** result only.
- **Telling useful forgetting from failure.** Report these next to the verdict, from the raw greedy generations in
  each eval json; all are needed to interpret it:
  - **Forget disclosure** (original/paraphrase), compared with the **stranger disclosure** base rate. Success means
    the forget-key rate falls toward the stranger rate, not to 0.
  - **Forget failure rates**: empty / refusal / degenerate / other well-formed. Useful forgetting means mostly
    well-formed answers. A rise in empty or degenerate answers marks corruption.
  - **Retain accuracy**, split by shared answer word × in-loss/held-out. A drop on shared-answer facts only signals
    answer-word suppression.
  - **Copy success** ("Repeat exactly: <answer_key>"). A drop signals vocabulary suppression.
  - **Retain failure rates.**
  - **Compute**: `compute.json`, per arm.
- **Not tested here.** Relearning/recovery attacks, many-paraphrase extraction, or more than one paraphrase per fact.
  A relearning probe can be added later only by a separate instruction.

## 6. Frozen vs. still open
- **Frozen.** The mechanism, both arms, hyperparameters, checkpoints, metrics and verdict rules (`DECISION_LOG.md`
  #8; the engineering fixes in #9 are disclosed). Any engineering fix needed on Hyperplane, such as module names or
  batch size for memory, must be applied to both arms, logged, and kept free of outcome-guided changes.
- **Unverified interfaces.**
  - PEFT regex matching on LLaDA's remote-code module names (N1).
  - `PeftModel.disable_adapter()` on LLaDA (N2).
  - `model(ids).logits` without an attention mask.
  - The chat template and special tokens (N5).
  - Upstream `generate` on a PeftModel (N3).
  - Fast-tokenizer behaviour for name acceptance (N6).
  - bf16 throughput and memory (N8).
