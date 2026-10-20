# RUN_MANIFEST — LLaDA Access & Engineering Feasibility Smoke Test

**Frozen at:** 2026-10-19T23:31:00-04:00 (before any model execution)
**Task start:** 2026-10-19T23:22:10-04:00
**Host:** hyperplane (Linux 5.15.0-136-generic, AMD EPYC 7742 ×2, 255 usable cores)
**Status at freeze:** model downloaded + hash-verified; zero GPU seconds charged so far.

This manifest is frozen *before* the model is executed. It states what will be run,
the settings, the limits, and the retry policy. It is not edited afterwards; deviations
are recorded in `FINAL_REPORT.md` instead.

---

## 1. Scope of this run

This is an **access + engineering feasibility smoke test**. It is explicitly NOT:

- a benchmark evaluation,
- a paper reproduction,
- an unlearning experiment,
- evidence that full-parameter MDU training fits this hardware.

Four checks are attempted: model access, native masked-diffusion inference,
one LoRA optimizer step, benchmark asset audit.
**Full baseline reproduction: NOT ATTEMPTED (by instruction).**

---

## 2. Pinned sources

| Asset | Source | Pinned revision |
|---|---|---|
| LLaDA official code | github.com/ML-GSAI/LLaDA | `9182493720ed723ef8031210d85959364e51cbe0` (2026-07-15) |
| MDU baseline code | github.com/leegeoru/MDU | `e5e8e8274ff8f56f16e978803d9e902359eb5272` (2026-05-25) |
| LLaDA-8B-Instruct (used) | huggingface.co/GSAI-ML/LLaDA-8B-Instruct | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| LLaDA-8B-Base (metadata only) | huggingface.co/GSAI-ML/LLaDA-8B-Base | `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` |

Both checkpoints: `gated=False`, `private=False`, **license MIT**. Access verified from
hyperplane by anonymous HTTP (no token present on host, none requested).

Only **LLaDA-8B-Instruct** is downloaded (14.94 GiB, 6 safetensors shards).
All 6 shard SHA-256 digests verified equal to the pinned HF LFS digests.

### Custom code review (pre-`trust_remote_code`)
`configuration_llada.py` (sha256 `c947a64d…`) and `modeling_llada.py` (sha256 `98bac7e5…`)
were read and grep-audited before enabling `trust_remote_code`. Imports are limited to
torch / transformers / Python stdlib. No `subprocess`, `os.system`, `eval(`, `exec(`,
`pickle`, `requests`, `urllib`, `socket`, no file writes, no network downloads.
`trust_remote_code=True` is enabled against these **local pinned files only**.

---

## 3. Environment

Isolated venv at `/data/bijaypandey/llada_smoke/venv_llada` (Python 3.10.12).

| Package | Version | Rationale |
|---|---|---|
| torch | 2.4.1+cu124 | matches host driver 550.144.03 / CUDA 12.4 |
| transformers | 4.49.0 | checkpoint was authored under 4.46.3; 4.49.0 is the nearest in-family version available here |
| peft | 0.14.0 | LoRA adapter |
| accelerate | 1.4.0 | — |
| numpy | 1.26.4 | pinned <2 for torch 2.4.1 compatibility |

**Deliberate divergence from MDU:** MDU's `requirements.txt` pins torch 2.7.0 /
transformers 4.57.0 and was tested on an H200. This run uses the stack above on A100-40GB.
This smoke test therefore does **not** validate MDU's own pinned environment.

All caches under `/data` (`HF_HUB_CACHE`, `PIP_CACHE_DIR`, `TMPDIR`); root filesystem is 100% full.

---

## 4. Model facts (from pinned config.json)

`mask_token_id=126336`, `eos_token_id=126081`, `pad_token_id=126081`, `bos=126080`,
32 layers, `d_model=4096`, `mlp_hidden_size=12288`, `n_heads=32`, `vocab=126464`,
`max_sequence_length=4096`, bf16. `pad_token_id != mask_token_id` — satisfies the
assertion in the author's `generate.py`.

Real module leaf names (from `model.safetensors.index.json`, 291 tensors):
`q_proj`, `k_proj`, `v_proj`, `attn_out`, `ff_proj`, `up_proj`, `ff_out`,
`attn_norm`, `ff_norm` (×32 blocks) plus `wte`, `ln_f`, `ff_out` at top level.

> **Caution recorded at freeze time:** the name `ff_out` is ambiguous — it occurs both as
> the 32 per-block MLP output projections *and* as the top-level LM head
> (`model.transformer.ff_out.weight`). LoRA targeting therefore **excludes `ff_out`**.

---

## 5. Planned inference (§4)

Author's masked-diffusion `generate()` from the pinned LLaDA repo, imported unmodified.
**Not** an autoregressive `.generate()` pipeline.

- 4 fixed benign prompts (frozen in `scripts/run_inference.py`), batch size **1** each.
- `gen_length=64` (≤64 generated tokens per prompt), `steps=64`, `block_length=32`
  (semi-autoregressive remasking, 2 blocks).
- `temperature=0.0` (deterministic argmax), `cfg_scale=0.0`, `remasking='low_confidence'`.
- `torch.manual_seed(42)` + `numpy.random.seed(42)` set before each prompt.
- Instruct chat template applied (`add_generation_prompt=True`).
- Recorded per prompt: prompt text, prompt token IDs, output token IDs, decoded text,
  seed, all decoding params, wall-clock runtime, `torch.cuda.max_memory_allocated`.
- Schedule check: assert the number of still-masked positions is strictly
  non-increasing across steps and reaches 0; assert output contains no `126336`.

**No prompt replacement and no repeated sampling to obtain better-looking outputs.**
The 4 prompts and seed are fixed here, before execution. Whatever they produce is reported.

## 6. Planned training fixture (§5)

One **successful optimizer step** on a disposable LoRA adapter, from the pristine
downloaded checkpoint.

- Fixture: one benign prompt→answer pair, frozen in `scripts/run_lora_step.py`.
- LoRA: `r=8`, `lora_alpha=16`, `lora_dropout=0.0`, `init_lora_weights=True`
  (PEFT default: A Kaiming-uniform, **B zero**), targets `["q_proj","k_proj","v_proj","attn_out"]`.
  `ff_out` excluded (LM-head ambiguity above).
- Optimizer AdamW, `lr=1e-4`, `betas=(0.9,0.999)`, `weight_decay=0.0`.
- Precision: base weights bf16; LoRA params cast to **fp32** for a stable optimizer step.
- Batch size **1**, total sequence length **128** tokens (≤128).
- Loss: the author's SFT recipe from pinned `GUIDELINES.md` §SFT —
  `forward_process` with `eps=1e-3`, `t~U(0,1)`, `p_mask=(1-eps)t+eps`;
  prompt positions restored after noising (prompt kept uncorrupted);
  `masked_indices = (noisy_batch == 126336)`;
  `token_loss = CE/p_mask`; `ce_loss = sum(token_loss/answer_lengths)/batch`;
  `answer_lengths` = count of non-prompt positions including padded EOS.
  Answer padded with EOS (126081) to the sequence length.
- Supervised mask asserted **nonempty** before the backward pass.

### Verification gates (all must pass)
1. Loss finite; gradient norm finite.
2. ≥1 intended trainable parameter changes by a finite, nonzero amount.
   (Zero-init LoRA `B` may legitimately give its partner `A` zero gradient on step 1 —
   not treated as a failure.)
3. Base-model parameters frozen and unchanged — verified over **all 291 base tensors**
   by pre/post SHA-256 of raw parameter bytes, plus `requires_grad` audit.
4. Optimizer `param_groups` membership set-equals the intended trainable parameter set.
5. Exactly **one** successful optimizer update (`step` counter == 1).

A forward pass or a backward pass alone does **not** pass this gate.
No additional training will be performed to improve the result.
Adapter kept locally with its hash; base weights are **not** uploaded.

---

## 7. Resource limits (measured from task start 23:22:10)

| Budget | Cap | Enforcement actually implemented |
|---|---|---|
| Elapsed | 3 h | `ledger.sh` computes from frozen `START_EPOCH`; checked between phases. **Advisory — not a hard kill.** |
| Aggregate GPU | 2 GPU-hours | `bounded.sh` wraps each GPU job in coreutils `timeout` (SIGTERM, SIGKILL after 30 s) and adds the attempt's wall-clock to `GPU_SECONDS` **including failures**. Per-job ceilings: inference 1800 s, LoRA 1800 s. |
| CPU | 8 core-hours | Measured post-hoc from `/usr/bin/time` + `getrusage` on launched jobs. **Measured, not capped.** |
| Storage | 60 GiB additional | `du -sb` of workspace in `ledger.sh`. Current: 14.94 GiB model + env. **Advisory.** |
| GPUs | 1 by default, 2 absolute max | Single device pinned by `CUDA_VISIBLE_DEVICES=0`; idleness re-checked immediately before allocation. |

**Honest statement of coverage:** the GPU-hour and elapsed budgets are enforced by
*per-job wall-clock timeouts plus a cumulative ledger that charges failed attempts*.
They are not enforced by a scheduler quota or cgroup. A single job cannot exceed its
timeout, and the cumulative total is checked between phases, but nothing kills the
session mid-phase if the aggregate is crossed. CPU-core-hours and storage are
**measured and reported, not hard-capped**.

### Retry policy
**At most one technical retry before a successful training update.** A retry is permitted
only for a technical failure (OOM, API/version incompatibility, device error) — never to
obtain a nicer loss or output. Both the failed and the retried configuration are preserved
and the report states precisely which one passed. Failed attempts are charged to the
GPU ledger. Budgets are never extended or reset.

### GPU availability
If no suitable idle GPU becomes available within **10 minutes**, the asset audit is
completed and a **blocked** report is published. No indefinite poller is left running.
Other users' jobs (currently GPUs 2 and 3) are never killed or interfered with.
