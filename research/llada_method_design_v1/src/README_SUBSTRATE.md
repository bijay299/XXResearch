# CPU substrate (coordinator-authored, method-agnostic)

Written by the coordinator during wave 1 so the implementation engineer (F) could start on the method at once.
F owns it from wave 3 on (may modify; must keep tests passing). G did not author it.

| module | content |
|---|---|
| `mdmu/toyworld.py` | fresh synthetic facts: 24 entities × {city(2 tok), field, year(2 tok), pet}; 2 question templates × 2 answer phrasings (value at different positions); copy task (general in-context capability); split = forget {city, year} of 6 entities (12 facts), same-entity neighbours (12), other entities (72); 26 retained facts share a value token with some forgotten value |
| `mdmu/model.py` | 2-layer bidirectional mask predictor (d=64, 4 heads, SwiGLU, RMSNorm; q/k/v/o/gate/up/down names like LLaDA/Llama), LoRA (A Gaussian, B=0) |
| `mdmu/diffusion.py` | LLaDA SFT forward process + loss (GUIDELINES.md), port of upstream `generate` (blocks, steps, low_confidence/random remasking, Gumbel temperature, CFG), MC ELBO NLL |
| `mdmu/evaluate.py` | generation-based leakage (`leak_full`, `leak_any`), well-formedness, copy accuracy under 4 native decoding configs D1–D4 |
| `mdmu/base_train.py` | trains `base.pt` (all facts) and `retrain.pt` (never sees forget facts) from the same init |

Dev substrate (world 0, init 0, 3000 steps, `results/dev/substrate/w0_i0/`): base recalls 100 % of forget/neighbour/other
facts and 100 % copy under D1 (D2: other 99.6 %); retrain recalls 0 % of forget facts, 100 % neighbour/other/copy.
Tests: `tests/test_substrate.py` (10 tests: loss equals hand computation, prompt never masked, sampler commits are
monotone/blockwise with the upstream transfer schedule, greedy determinism, Gumbel sampling matches softmax, LoRA zero-init).

## Changes by F (wave 3) — semantics of existing entries preserved; all listed in results/dev/DEV_LOG.md
- `diffusion.generate`: batched top-k fast path when every row commits the same k (upstream per-row loop kept as the
  fallback, `_BATCHED_TOPK=False` restores it; equivalence tested in `tests/test_fxmix.py`).
- `model.LoRALinear.enabled` + `lora_disabled(model)` (adapter-off reference pass, PEFT `disable_adapter` analogue);
  `merged_copy(model)` (LoRA folded into plain linears; used for evaluation).
- `evaluate.DECODING_CONFIGS`: SPEC §3 grid added (P1–P3 pool, H1–H6 held out; `TRAIN_POOL`, `HELDOUT`,
  `HELDOUT_PRIMARY` = H1–H5, `EXTRAPOLATION` = H6); D1–D4 unchanged. `STOCHASTIC` also contains random-remasking
  configs (only new entry H3 is affected). `fact_eval(..., ban_values=True)` = decode-time F-ban reference;
  `return_per_fact=True` for slice accuracies.
- New: `fxmix.py` (FX-Mix target/states/loss, controls NPO, MDU, EMDU, SC-TF, SC, certificate) and `unlearn.py`
  (single training loop for all arms, compute counters, evaluation suite, mechanism statistics).
