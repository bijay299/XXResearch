# Integration audit of the BLOCK release (source pin `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5`)

Audit performed by reading every file in the release before running anything.
Release contains 39 files; no `requirements.txt`, no `src/` contents, no released
transcoder checkpoints, no model revision pins.

Three categories are kept separate, as required:
**(A) integration gaps** = the released code cannot run as published (crash/undefined name/missing file);
**(B) code-vs-paper discrepancies** = the code runs but does something the paper describes differently;
**(C) code-supported behaviour** = what the code actually does, which I adopt as "native".

---

## A. Integration gaps (blocking; minimal recorded fixes allowed by protocol)

| # | File:line | Gap | Minimal fix |
|---|---|---|---|
| A1 | `scripts/transcoder/get_activation_diffusion.py:129,136` | `ctx_out` is used but **never computed**. `context_embedder` is bound at line 99 and never called. `NameError` on the first batch. | `ctx_out = context_embedder(prompt_embeds)` under `no_grad`. |
| A2 | `scripts/transcoder/get_activation_diffusion.py:161`, `..._mlp.py:121` | `parser.add_argument(...)` inside a block that defines `parse`. `NameError` at startup. | `parse.add_argument` |
| A3 | `scripts/transcoder/get_activation_diffusion.py:240`, `..._mlp.py:193` | `output_path` is an unbound global; the arg is `args.output_path`. `NameError` at save time. | `args.output_path` |
| A4 | `scripts/transcoder/get_activation_diffusion_mlp.py:161` | Reads `prompts/coco_train2014_30k_prompts.txt`, which **is not in the release**. This is the pooled-text transcoder's entire training corpus. | Reconstructed from MS-COCO `captions_train2014.json` by seeded sampling; see `COCO_CORPUS.md`. Deviation recorded. |
| A5 | `scripts/transcoder/train.py:9`, `scripts/unlearn_canvas/diffusion.py:24`, `diffusion_baseline.py:15` | `from scripts.utils import setup_logger, get_module_by_name` — **`scripts/utils.py` does not exist** in the release. | Provided a 2-function `scripts/utils.py` shim (logger + dotted-path module getter). |
| A6 | `scripts/unlearn_canvas/diffusion_baseline.py:17` | Imports `scripts.transcoder.model_inference_ca.TranscoderCA` — **module absent** from the release. | Import removed; it is unused in that file. |
| A7 | `scripts/unlearn_canvas/diffusion_baseline.py:25-44` | `load_pipeline` builds the pipeline but has **no `return`** and never moves it to a device → `main` gets `None`, then `pipe(prompt)` raises `TypeError`. | `return pipe.to(device)`. |
| A8 | `scripts/transcoder/train.py:68` → `trainer.py:24-43` | `train.py` passes a **dict** `{layer: Dataset}`; `TranscoderTrainer` treats it as a `Dataset` (`len(dataset)` → number of layers = 1; `dataset[0]` → `KeyError`; `random_split` on a dict). | Pass the single `Dataset` for the requested layer (our runs train one layer at a time). |
| A9 | `scripts/transcoder/train.py:15-16` | Path template `layer{layer}_in` does not match what the activation scripts write (`context_embedder_in`, `text_embedder_in`). | Template changed to `{layer}_in` / `{layer}_out`. |
| A10 | `scripts/transcoder/trainer.py:215` | `wandb.log(info_eval, step=steps)` is **outside** the `if cfg.log_to_wandb` guard → fails when W&B is disabled. W&B is also mandatory via `os.environ["WANDB_TOKEN"]` at line 67. | Guarded; W&B disabled (no external services this sprint). |
| A11 | `scripts/transcoder/trainer.py:72` | `wandb.Histogram`/`asdict` path requires W&B even for local runs. | Covered by A10. |
| A12 | `scripts/transcoder/trainer.py:246-263` | `save()` is called as `trainer.save()` from `train.py`, so `typ=None` → the checkpoint is written to **`None.pt`**, but `diffusion.py:55-56` loads `context_embedder.pt` / `text_embedder.pt`. | `save(typ=<layer>, name=<run>)` passed explicitly. |
| A13 | `scripts/transcoder/trainer.py:62,253` | `make_run_name()` runs in `__init__` **and** again in `save()`, so the save directory is not the run directory reported at init (run number increments). | Explicit `name=` at save. |
| A14 | `scripts/unlearn_canvas/diffusion.py:38` | `FluxPipeline.from_pretrained(path, device_map="balanced")` with **no `torch_dtype`** → loads FLUX in **fp32** (~70 GB across text encoders + transformer). Cannot fit; and it silently disagrees with the baseline script, which uses `bfloat16` (`diffusion_baseline.py:37`). | Pinned `torch_dtype=bfloat16` for **all** arms; see `PRECISION.md`. |
| A15 | `scripts/transcoder/model_inference.py:27-39` | The transcoder is constructed in **fp32** and `replace_ffn_with_transcoder` only does `.to(device=device)` (no dtype). Feeding the bf16 pipeline's activations into an fp32 `nn.Linear` raises a dtype `RuntimeError`. | `forward()` upcasts `x` to the encoder dtype and casts the result back to `orig_dtype` (the file already intends this: it captures `orig_dtype` and casts the output). |
| A16 | `scripts/transcoder/model_inference.py:103` | `from_state_dict` does `state_dict.pop("k")` — **mutates the caller's dict**, so loading the same state dict twice fails. | Shallow-copy before popping. |
| A17 | repo root | `from constants.const import ...` and the relative path `prompts/unlearn_canvas/...` require the repo root both on `sys.path` and as cwd. No packaging/`__init__.py`. | Runner sets cwd + `PYTHONPATH` to the repo root. |

## B. Code-vs-paper discrepancies (recorded, not silently resolved)

| # | Paper says | Code does | Resolution for this sprint |
|---|---|---|---|
| B1 | E.1: "We apply an exponential moving average (EMA) of parameters with decay 0.999." | `TrainingConfig.ema_multiplier = 0.999` exists but **EMA is never applied anywhere** in `trainer.py`. | Follow the **code** (no EMA). Writing an EMA myself would be inventing absent algorithmic behaviour, which the protocol forbids. Recorded as an unresolved discrepancy. |
| B2 | E.1: "learning rate eta = 1e-4". | `trainer.py:56`: `lr = 2e-4 / (n_latents / 2**14)**0.5` (the Gao et al. scaling rule). For the **context** transcoder (`n_latents=65536`) this equals **1.0e-4** — consistent. For the **pooled** transcoder (`n_latents=12288`) it equals **2.31e-4** — inconsistent with the stated 1e-4. | Follow the **code** (the native implementation). Both values recorded. |
| B3 | Fig. 8: object prompt is a **numbered** list, "Respond with only the number." | `eval/llava.py:170-186` builds a **lettered** A)…T) list, "reply with the letter only", and asks "What is the object class of this image." | Follow the **code** (`eval/llava.py` verbatim) for native UA/IRA. Recorded. |
| B4 | §4.2: "Since the TopK operator is non-differentiable, we employ a straight-through estimator." | `model_training.py:83` uses `torch.topk` and decodes the gathered values directly. Gradients flow to the selected entries only; there is **no** STE onto unselected entries. | Follow the **code**. (The code's behaviour is the standard TopK-SAE gradient; no STE is needed for the selected values.) |
| B5 | §4.3: "redirected to reproduce the output of the **empty token** ∅". | `diffusion.py:218` uses the literal prompt **`"_"`**, not the empty string. | Follow the **code** (`"_"`). Recorded; the empty-string variant is *not* tested (out of scope). |
| B6 | E.1: batch size 16384 (projection surrogate) / 1024 (MLP surrogate). | `TrainingConfig.batch_size = 4096`, and `train.py` exposes **no** batch-size flag. | Use the **paper's** per-module values (16384 context / 1024 pooled) and expose the flag, since the config default matches neither and the paper is specific. Recorded. |
| B7 | — | `train.py` requires `--dead_threshold`; neither paper nor code states the value used. `TrainingConfig` default is `10_000_000` (forward passes). | Use the config default 10,000,000 and disclose that the paper does not pin it. |

## C. Code-supported native behaviour adopted as the baseline

- **Two intervention sites in FLUX** (confirmed by D.1 + code):
  - `transformer.context_embedder` — `Linear(4096 -> 3072)` on per-token T5 embeddings ("context"/projection transcoder).
  - `transformer.time_text_embed.text_embedder` — the modulation MLP on the **pooled CLIP** vector ("pooled-text" transcoder).
  Both are replaced; D.1: "we found it necessary to intervene at this layer as well".
- **Training pairs**: input = module input, target = the *original module's output* on that input (`get_activation_*` applies the real module). Context pairs are masked to **valid T5 tokens only**; pooled pairs are one per prompt.
- **Two different corpora** (this is the trap the protocol flagged, and it is real):
  - context transcoder: 80 prompts x 20 objects x (10 styles + no-style) = **17,600 prompts**, which is `prompts/unlearn_canvas/*.txt` in the release (verified: 80 lines per file x 20 files).
  - pooled transcoder: **30,000 MS-COCO prompts** x 11 variants = **330,000 pairs** (E.1). The release's 17,600-prompt corpus does **not** cover it.
- **Erasure update** (`diffusion.py:308-324`), per transcoder:
  `W_dec[A_t[j]] <- W_dec[A_0[j]] * ( z_0[A_0[j]] / clamp(z_t[A_t[j]], min=1e-6) )`
  so the decoded contribution of target latent `A_t[j]` becomes exactly the empty token's contribution
  `z_0[A_0[j]] * W_dec[A_0[j]]`. Pairing is index-order (the paper's "simplest strategy"); only `W_dec`
  rows are touched — encoder, biases and all other rows are untouched.
- **Target features** come from encoding the bare concept string (e.g. `"Bears"`), union over its valid T5 tokens.
- **Erasure is applied to a pristine transcoder per target** (the script loads checkpoints fresh each run), so edits never accumulate — matches the protocol's N-arm requirement.

### Known native hazard, not a bug to fix
`W_dec.data[latents_targets] = src_rows * scale` is a scatter with **possibly duplicated** target indices
(a latent active for two different target tokens). PyTorch leaves the winner unspecified. `src_rows` is
gathered before the write, so there is no read-after-write aliasing. Duplicate counts are measured and
reported rather than "fixed", since de-duplicating would change the native update.
