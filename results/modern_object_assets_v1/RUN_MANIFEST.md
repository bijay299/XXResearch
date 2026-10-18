> # ⚠️ SUPERSEDED IN PART — read [`AMENDMENT_01_ACCOUNTING_AND_PROVENANCE.md`](AMENDMENT_01_ACCOUNTING_AND_PROVENANCE.md) first
>
> **Dated 2026-10-18.** The amendment is **authoritative** over this document on
> the following points. The historical body below is **preserved unchanged** as
> originally published at `c8502fa`, including the wording now corrected.
>
> 1. **Retries.** This document's accounting underpinned a reported *0 of 1*
>    technical retries. The correct count is **1 of 1**: `smoke_train_01` failed
>    at import and `smoke_train_02` reran the same command after `PYTHONPATH`
>    wiring. A technical retry, not an outcome-driven rerun. All resource
>    figures are unchanged — the failed attempt's 2.15 device-seconds were
>    already charged and are not double-counted.
>
> 2. **"Frozen before any GPU work" (first line below) is WITHDRAWN as a
>    categorical assurance.** The records disagree: both frozen files embed
>    `frozen_utc: 2026-10-18T06:55:00+00:00`, which *postdates* the first GPU
>    allocation (06:50:37 UTC), while the `06:46:13`/`06:46:38 UTC` times quoted
>    below are **host-reported filesystem mtimes**, not an independently
>    timestamped prospective record. This chronology is **uncertain**. What the
>    records *do* support: the freeze preceded **production training**
>    (07:04:55 UTC) and **all image generation** (07:26:25 UTC), and no image had
>    been decoded before then. Checkpoint reload and pairing integrity are
>    unaffected.
>
> 3. **The `ledger/ledger.py` link below is broken.** The helper is not in
>    ordinary Git. It is published as the member `evidence/ledger/ledger.py`
>    inside
>    [`gem-flux-teapot-v1-evidence.tar.gz`](https://github.com/bijay299/XXResearch/releases/download/modern-object-assets-v1/gem-flux-teapot-v1-evidence.tar.gz)
>    (12,445,898 B, sha256 `49fdea2338028515e2ecf694d86ac0225684899a90759da3264886400bea7488`).
>
> 4. **"Enforced before every launch" overstates what exists.** The launcher has
>    pre-launch checks against an *operator-supplied estimate* plus end-of-job
>    accounting. **No runtime watchdog was implemented**; no automatic runtime
>    hard-stop is established. Recorded use stayed below every cap, but that is
>    an outcome, not enforcement.
>
> Status **`ASSETS_DELIVERED`** is retained as a technical status.

---

# RUN_MANIFEST — modern_object_assets_v1

**Frozen before any GPU work.** The machine-readable freeze is
[`configs/production_config.json`](configs/production_config.json) (written
`2026-10-18 06:46:13 UTC`) and [`manifests/CASES.json`](manifests/CASES.json)
(written `2026-10-18 06:46:38 UTC`). The first GPU allocation in the ledger is
`2026-10-18T06:50:37 UTC` (`smoke_train_01`). This document is the
human-readable rendering of that freeze; where the two differ, the JSON is
authoritative.

## What this is

An **infrastructure milestone**: train one existing baseline on one
non-sensitive object, then publish the actual trained checkpoint artifacts and
20 matched original/edited image pairs.

This is a **source-derived GEM configuration for an ordinary object**. It is
**not** an author-validated object benchmark and **not** a reproduction of paper
performance. No erasure-efficacy, suppression, retention, SOTA, novelty,
evaluator-bias, human-consensus or paper-readiness claim is made anywhere in
this release.

## Source and backbone

| | |
|---|---|
| Method | GEM (`gem`), official source |
| Source repo | `multimodal-ai-lab/GEM` |
| Source commit | `c328fc6bec687efe60223498d1b6bed2e5fbcffd` |
| Source license | MIT |
| Backbone | `black-forest-labs/FLUX.1-dev` |
| Backbone revision | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| Backbone license | FLUX.1 [dev] Non-Commercial License v1.1.1 (gated) |
| Backbone cache | pre-existing verified-pristine local snapshot, reused **read-only** |

Backbone access was verified against the gated repository for the pinned
revision before any training (HF API reported `gated: auto`,
`sha: 3de623fc…`, and an authenticated fetch of `LICENSE.md` at that revision
succeeded).

## Target and anchor

| | |
|---|---|
| Target | `teapot` |
| Anchor | `""` — the pinned source's default `EMPTY_CONCEPT` |
| Anchor provenance | `gem/operators/utils.py:3` → `EMPTY_CONCEPT = ""`; `GEMConfig.anchors` defaults to `[EMPTY_CONCEPT]` |

The anchor's **exact value is the empty string**. It was taken from the source
default, not chosen.

## Native GEM adaptation

| | |
|---|---|
| Adapter mode | `lora` (PEFT), from `GEMConfig.get_default_adaptation_config()` |
| Trainable subset | `qk_dual` |
| Subset rule | `startswith=['transformer']`, `exclude=['single_transformer_blocks']`, `endswith=['add_q_proj','add_k_proj','to_q','to_k']` |
| Target modules | **76** (19 dual blocks × 4 projections) |
| LoRA rank / alpha / dropout / bias | `16 / 16 / 0.1 / none` (source defaults) |
| Trainable parameters | **7,471,104** |
| Base transformer parameters | 11,901,408,320 |
| Trainable fraction | 0.0628 % of the transformer |
| Optimizer | `torch.optim.AdamW`, lr `1e-3` (source default, hardcoded in `erasure_gem.py`) |
| Loss | `relu(‖pred−anchor‖₂ − η·‖pred−target‖₂).sum()`, scaled `γ/n_latents`; `γ=100.0`, `η=1.0` |
| Trajectory | `use_inference_timesteps=True`, `partial_denoising_mode='student'`, guidance `3.0`, `end_at_timestep=10`, `ignore_until=0`, internal steps resolved to `28` |

The full per-module inventory is
[`manifests/trainable_parameter_inventory.json`](manifests/trainable_parameter_inventory.json).

## Step budget semantics

`max_step_budget = 250`, **in the source's own counter**. GEM decrements
`step_budget` by the number of **processed timesteps** per inner batch
(`num_parallel_steps`), not by optimizer updates. 250 therefore counts processed
timesteps, not 250 optimizer updates.

Predicted before the run: `expected_n_iterations = 1 + 250//10 = 26`,
11 latents per trajectory, ≈23 optimizer updates.
**Actual** (see *Results*): 250 processed timesteps, 23 optimizer updates,
native termination at `global_step 22`. The schedule was not extended.

## Precision, batching, resolution

| | source default | used | status |
|---|---|---|---|
| `model_dtype` | `float16` | **`bfloat16`** | disclosed deviation (mandated) |
| `max_parallel_timesteps` | `6` | **`1`** | disclosed deviation (mandated) |
| `image_size` | `512` | `512` | unchanged |
| `max_step_budget` | `250` | `250` | unchanged (already the default) |

No precision search and no batch search was performed.

## Inference (both conditions, identical)

| | |
|---|---|
| Steps | 28 (flux `MODEL_CONFIGS` default) |
| Guidance scale | 3.5 (flux `MODEL_CONFIGS` default) |
| Resolution | 512 × 512 |
| dtype | bfloat16 |
| Scheduler | `FlowMatchEulerDiscreteScheduler` from the pinned base snapshot; sigmas `linspace(1.0, 1/28, 28)` with dynamic `mu` shift via the pinned `FluxPipelineWrapper.set_timesteps` / `calculate_shift` |
| Image sequence length | 1024 patches |
| Tokenization | CLIPTokenizer + T5TokenizerFast, `prompt` and `prompt_2` both set to the case prompt, per the pinned `FluxPipelineWrapper.encode_prompt` |
| Batch size | 1 |

### Initial-noise handling

For each case the packed initial latents are built **once** from
`torch.Generator(device='cpu').manual_seed(eval_seed)` exactly as
`FluxPipeline.prepare_latents` does (shape `(1,16,64,64)` → packed `[1,1024,64]`,
bfloat16), then sha256-hashed, persisted to disk, and passed explicitly as
`latents=` to **both** the M0 and the edited run.

Matching seeds alone are **not** treated as proof of pairing: the stored tensor
and its hash are the proof. Each generation run additionally asserts that the
native `pipe.prepare_latents` call with the same seed reproduces the identical
tensor (logged as `MATCH`). Matching inputs do not guarantee matching image
layouts, and no such claim is made.

## Seeds

| role | value |
|---|---|
| Smoke training seed | 101 |
| **Production training seed** | **20261018** |
| Validation seed | 202 (unused — W&B disabled) |
| Evaluation seeds | 7001 – 7020 (20 distinct) |

The production training seed is distinct from the smoke seed and from every
evaluation seed.

## Retry policy

At most **one** technical retry for the whole project, within an unchanged
scientific configuration and the same ceilings. Never a retry because erasure or
appearance is disappointing. The predetermined native terminal endpoint is used;
an interrupted endpoint would be labelled *partial*. Resume only with
trustworthy state — a fresh restart is not an exact continuation.

**Retries actually used: 0.**

## Ceilings

| ceiling | limit |
|---|---|
| Deadline | 7 elapsed days from `2026-10-18T02:28:57−04:00` → `2026-10-25T02:28:57−04:00` |
| GPU | 24 cumulative allocated device-hours |
| Bring-up | ≤ 4 of those hours |
| Reserve | ≥ 4 hours retained for final generation / remote reload / recovery |
| Decoded images | 64 total = 40 primary + ≤ 24 technical |
| Storage | ≤ 200 GiB project-owned, ≥ 50 GiB free on `/data` |
| Hardware | ≤ 4 × 40 GB GPUs, only devices the native implementation uses |
| Services | no paid APIs, cloud GPUs, subscriptions or external messages; W&B/telemetry disabled |

Enforced before every launch by
[`ledger/ledger.py`](ledger/ledger.py) (append-only JSONL, `flock`-serialised),
which records launches, devices, timestamps, actual allocated GPU-seconds,
decoded-image counts, outcomes, retries and failures. **All** attempts are
counted, including the one that failed at import.

## Environment lock

Python 3.11.17 (uv-managed), built from the author's own `uv.lock`
(`uv sync --frozen`): torch 2.9.1+cu128, diffusers 0.37.1, transformers 5.8.1,
peft 0.19.1, accelerate 1.13.0, safetensors 0.7.0.
Full lock: [`manifests/uv.lock.frozen`](manifests/uv.lock.frozen) and
[`manifests/pip_freeze.txt`](manifests/pip_freeze.txt) (138 pinned packages).

## Publication format

* Documentation, small code, configs and manifests → this results directory in
  ordinary Git.
* Actual native trained checkpoint components and original-resolution paired
  images → **GitHub Release assets** under a new tag
  (`modern-object-assets-v1`). No large weights in ordinary Git; no Git LFS.
* GitHub caps each release asset at 2 GiB, so the ~22 GiB native merged
  transformer is published as an uncompressed tar split into **lossless 1 GiB
  parts**, with per-part and reconstructed-file sha256 and reconstruction
  instructions. **Nothing is quantized or cast to fit hosting.**
* Pristine base weights are **not** republished. Credentials are never published.
