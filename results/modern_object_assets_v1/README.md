# modern_object_assets_v1 — GEM / FLUX.1-dev `teapot`: trained checkpoint + 20 matched pairs

**Status: `ASSETS_DELIVERED`** — remote download, hash, lossless reconstruction
and reload verification all passed
(`REMOTE_VERIFICATION_PASSED`; [REMOTE_VERIFICATION.json](REMOTE_VERIFICATION.json),
[EVIDENCE_INDEX.md](EVIDENCE_INDEX.md)).

**Release:** [`modern-object-assets-v1`](https://github.com/bijay299/XXResearch/releases/tag/modern-object-assets-v1)
— 31 assets, 23,861,453,027 bytes.

An **infrastructure milestone**, not a method contribution and not a paper
result: one existing baseline (GEM) was trained once on one ordinary,
non-sensitive object (`teapot`) on FLUX.1-dev, and the **actual trained
checkpoint artifacts** plus **20 matched original/edited image pairs** are
published as downloadable release assets.

> ### What is and is not claimed
>
> **Delivered (technical):** a fresh, technically valid trained endpoint; a
> native checkpoint that saves and reloads bitwise-identically; 20 complete
> pairs generated under provably identical initial noise; full provenance.
>
> **Not assessed:** erasure efficacy. No detector-recognition, suppression,
> retention or visual-quality threshold was applied. No automated judge was run
> and no scientific human label was assigned. **A technically valid but
> ineffective edit would look exactly like this release**, and nothing here
> distinguishes the two.
>
> This is a **source-derived GEM configuration for an ordinary object** — *not*
> an author-validated object benchmark and *not* a reproduction of paper
> performance. No SOTA, novelty, evaluator-bias, retention, human-consensus or
> paper-readiness claim is made. This is an open asset release, **not** a
> blinded review package.

---

## 1. Scope

| | |
|---|---|
| Method | GEM — `multimodal-ai-lab/GEM` @ `c328fc6bec687efe60223498d1b6bed2e5fbcffd` (MIT) |
| Backbone | `black-forest-labs/FLUX.1-dev` @ `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` (FLUX.1 [dev] Non-Commercial License v1.1.1, gated) |
| Target | `teapot` |
| Anchor | `""` — the pinned source's default `EMPTY_CONCEPT` (exact value: the empty string) |
| Precision | `bfloat16` |
| Trainable subset | native GEM LoRA, `qk_dual`, r=16 α=16 dropout=0.1 bias=none |
| Trainable parameters | **7,471,104** across **76** modules (0.0628 % of the 11,901,408,320-parameter transformer) |
| Training from | pristine M0 (no reuse of any earlier edited checkpoint, image, label or prediction) |

*(PEFT's own log line reports `trainable%: 0.0443` because its denominator is the
whole wrapped pipeline — 16,878,070,243 params including the T5/CLIP text
encoders and VAE. Against the transformer alone it is 0.0628 %. Same 7,471,104
trainable parameters either way.)*

Full frozen configuration: [RUN_MANIFEST.md](RUN_MANIFEST.md) ·
[configs/production_config.json](configs/production_config.json).

## 2. Actual training counters

The source counts `max_step_budget` in **processed timesteps**, not optimizer
updates. Both are reported.

| counter | value |
|---|---|
| `max_step_budget` (source counter) | 250 |
| **Processed timesteps actually consumed** | **250** |
| **Optimizer updates actually performed** | **23** |
| Native termination | `Step budget (250) exhausted. Stopping at global_step 22.` |
| Endpoint | **predetermined native terminal endpoint** — not interrupted, not extended |
| Latents per trajectory | 11 (`partial_denoising_end_at_timestep=10`) |
| Training wall time | 5 m 37 s |
| Peak GPU memory (native log) | 38.33 GiB on one 40 GB A100 |
| Losses | 273 samples, **all finite**, range 0.0000 – 3263.0659 |
| Gradients | 23 updates, **all finite**, 152 tensors with grad (76 modules × lora_A/lora_B) |
| Updates with non-zero gradient | **18 / 23** |
| Updates with zero gradient | 5 (updates 17, 18, 19, 22, 23) |

**On the zero-gradient updates.** GEM's objective is a hinge:
`relu(‖pred−anchor‖₂ − η·‖pred−target‖₂).sum()`. A zero loss and zero gradient
on an update means the margin was already satisfied for those timesteps. This is
a property of the objective, reported as observed; it is not a failure and it is
not evidence about erasure efficacy either way. See
[evidence/loss_grad_report.json](evidence/loss_grad_report.json).

## 3. Technical-integrity gate — all checks passed

| check | result | evidence |
|---|---|---|
| Intended weights changed from M0 | **exactly 76 tensors** changed — the qk_dual targets, and nothing else (1160 keys compared) | [compare_M0_vs_postmerge.json](evidence/compare_M0_vs_postmerge.json) |
| No-op checkpoint? | **No** — a no-op would be a technical failure | same |
| Post-merge tensors finite | **all finite** | [finite_postmerge.json](evidence/finite_postmerge.json) |
| Save → reload | **`SAVE_RELOAD_BITWISE_IDENTICAL`** — keys, shapes, dtypes and tensor values all match | [reload_check_local.json](evidence/reload_check_local.json) |
| M0 reproducibility | pristine M0 digest `ddfc3688…` identical across the smoke and production runs | run + smoke evidence |

Canonical digests (sha256 over the per-key `{shape,dtype,value}` digest set of
the 1160-tensor transformer state):

* pristine M0 — `ddfc36881ade66941595d4920b1da9d84bad086b95aba7ce0f9befc50d6d3843`
* edited, post-merge immediately before the native save — `71c8f08ec5d9bc63ffc3ce3d3cceacde…`
* edited, reloaded from disk — **identical to the post-merge digest**

The comparison is between the **canonical post-merge state captured immediately
before saving** and the **state reloaded from disk**. It is *not* a claim that an
unmerged LoRA adapter and the merged weights are bit-identical — that separate
question was measured independently (§5).

## 4. Checkpoint representation

GEM's native save path (`save_pipeline`) infers components from the `qk_dual`
target-module prefixes and writes the **merged post-LoRA `transformer/`
component** — a full FLUX-derived transformer, which is what the author saves.
That is what is published.

| component | files | bytes |
|---|---|---|
| `transformer/config.json` | 1 | 553 |
| `transformer/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors` | 3 | 9,983,040,304 + 9,949,328,904 + 3,870,584,832 |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | 1 | 121,262 |
| **total merged transformer** | 5 | **23,803,075,855** |
| `lora_adapter/` (pre-merge PEFT adapter) | 3 | 29,914,559 |

Per-file sha256: [manifests/edited_transformer_sha256.txt](manifests/edited_transformer_sha256.txt)
(base, for comparison: [manifests/base_transformer_sha256.txt](manifests/base_transformer_sha256.txt)).
Archive-level sha256 and per-part hashes:
[manifests/ASSET_MANIFEST.json](manifests/ASSET_MANIFEST.json).

**Where the 76 changed tensors live.** Mapping them through the shard index:
**60** are in shard 1, **16** are in shard 2, and **0** are in shard 3 — so
**shard 3 of 3 is byte-identical to the corresponding pristine base shard**
(`0d9c7c66…`), as is `…index.json` (`780ee2ab…`). This is fully explained by the
subset rule: `qk_dual` **excludes** `single_transformer_blocks`, and shard 3
contains only those 192 single-block tensors plus `norm_out`/`proj_out` — it has
no dual-block tensor at all, so no changed tensor can land there.

`config.json` differs from the base's only in re-serialisation by the newer
diffusers (`_diffusers_version` `0.30.0.dev0`→`0.37.1`, a local `_name_or_path`,
and explicit defaults such as `axes_dims_rope` and `out_channels: null`). No
architectural field changed.

## 5. Does the small adapter reproduce the endpoint? — measured

The pre-merge LoRA adapter (27.6 MiB compressed) was re-applied to the **pinned
pristine base** and merged through GEM's own `merge_and_unload_all_adapters`,
then compared against the published merged checkpoint:

> **Measured result: `BITWISE_IDENTICAL` — 0 of 1160 tensors differ.**

So the adapter *is* a sufficient substitute for the 22 GiB merged component when
combined with the pinned gated base. This is a **measurement**, not an
assumption ([evidence/adapter_equivalence.json](evidence/adapter_equivalence.json)).
The full native merged component is published regardless, as the primary
artifact.

## 6. The 20 matched pairs

20 cases were authored and **frozen before any output was generated or viewed**
(`manifests/CASES.json`, written 06:46:38 UTC; first GPU allocation 06:50:37
UTC). Ten ordinary prompts and ten challenging prompts covering **composition**
(4), **occlusion** (3) and **style** (3), categorised from prompt metadata — not
from any image. Ordinary, non-sensitive content; no named artists, no protected
characters, no people, no brands.

Cases were **not** selected using recognition scores, rendering quality or
erasure outcomes.

| check | result |
|---|---|
| Pairs complete | **20 / 20** |
| Primary PNGs | **40** (20 M0 + 20 edited), all 512 × 512, original resolution |
| Initial noise byte-identical across conditions | **yes, all 20** (sha256 match) |
| Distinct initial-noise tensors | **20 / 20** |
| Prompts / seeds identical across conditions | yes / yes |
| Inference settings identical across conditions | yes |
| Missing or corrupt outputs | **none** |
| Degenerate (black / saturated / flat) outputs | **none detected** — all would have been preserved and disclosed |
| Edited output differs from its paired M0 output at file level | 20 / 20 |

That last row is a **file-level** observation only. It says the edited weights
changed the sampled output; it says nothing about whether the target concept was
erased, substituted, degraded or retained. **That question is unassessed here.**

Pairing proof: [manifests/PAIR_VERIFICATION.json](manifests/PAIR_VERIFICATION.json),
[manifests/initial_noise_index.json](manifests/initial_noise_index.json).
Matching seeds alone were not accepted as proof — the stored noise tensors and
their hashes are. Each generation run also asserted that the native
`prepare_latents` call reproduces the stored tensor exactly (`MATCH`).
Matching inputs do not guarantee matching image layouts, and no such claim is made.

Contact sheet: published as the standalone release asset
`gem-flux-teapot-v1-contact-sheet.png` (and also inside the images archive). It
is composited from the already-decoded PNGs — it performs no new decode and adds
nothing to the decoded-image count. This repository gitignores `*.png` by
standing policy, so no PNG is committed to ordinary Git; **a contact sheet and
hashes are in any case not a substitute for the images** — all 40 originals are
downloadable from the release.

## 7. Patches to the pinned source

Every patch is published and explained. **No patch changes training
mathematics, the optimizer, target/anchor construction, the trainable subset or
the sampling method. No memory-management port was implemented.**

### P0 — the reported `neutral_embeddings` deletion: **investigated, not a defect**

The assignment flagged an *apparent* undefined `neutral_embeddings` deletion in
`erasure_gem.py`'s final cleanup (line 309,
`del optimizer, neutral_embeddings, …`). **This was checked and is not a bug:**
`neutral_embeddings` is assigned unconditionally at line 106
(`neutral_embeddings = wrapper.encode_prompt(EMPTY_CONCEPT)`) before the training
loop, so the name is always bound when the `del` executes. **No repair was made**
— a repair would have been a change with no defect behind it.

(Separately and not repaired: the value is computed and reassigned but never
read by the loss. That is dead code in the upstream, not an error, and touching
it would risk the training path for no benefit.)

### P1 — skip the discarded training-preview decode *(the only code change)*

[`patches/P1_skip_discarded_training_preview_decode.diff`](patches/P1_skip_discarded_training_preview_decode.diff)

`BasePipelineWrapper.partially_denoise_until_timestep` always VAE-decodes a
preview image and returns it. GEM's training loop is its only caller with
`return_all_latents=True` and **discards that image** (`_, all_latents = …`).
Decoding it anyway costs a full VAE decode per training iteration (~23 per
production run) and produces exactly the *automatic decoded training previews*
this run's image ceiling forbids.

The patch returns early when `return_all_latents=True`. It is numerically inert:
`vae.decode` and `image_processor.postprocess` draw no random numbers and never
touch `latents_per_step`, the optimizer, the loss or the sampling trajectory.

**Effect: 0 decoded training previews. The entire 24-image technical allowance
went unused.**

### P2 — pin the backbone path *(no code change; runtime path wiring)*

GEM resolves the backbone through
`MODEL_CONFIGS['flux']['model_name_or_path'] = 'black-forest-labs/FLUX.1-dev'`,
which would resolve against the Hub's *current* `main`. The driver repoints it at
the already-cached, verified-pristine snapshot directory **for the pinned
revision**, making the run revision-exact and offline-safe while preserving
`create_pipeline`'s `base == config` identity (so the M0 load overwrites nothing
and an edited-checkpoint load still takes the overwrite path). Verified in both
directions in the logs.

### Harness notes (driver scripts, not GEM changes)

* `PYTHONPATH` wiring — GEM is a uv *virtual* project (no `[build-system]`), so
  it is never installed into the venv; the README relies on `python -m gem.train`
  from the repo root.
* Generation uses the pinned wrapper and its exact settings, but calls one prompt
  per invocation with explicit `latents=`, and does **not** call
  `enable_attention_slicing()` / `enable_model_cpu_offload()` (both are memory
  helpers used by GEM's `generate_and_save_images`; the 23 GiB transformer fits
  on one 40 GB device). Disclosed in each `index_*.json`.
* Observation-only hooks (digests, counters, gradient norms, adapter
  serialisation) wrap the native calls without altering them.

## 8. Resource ledger — actuals

Append-only, `flock`-serialised JSONL: [ledger/ledger.jsonl](ledger/ledger.jsonl).
Every attempt is counted, including one that failed at import.

| | used | ceiling |
|---|---|---|
| Elapsed | ~1.3 h at asset-packaging time | 7 days (deadline `2026-10-25T02:28:57−04:00`) |
| **Allocated GPU device-hours** | **0.253** | 24 |
| — of which bring-up | 0.048 | ≤ 4 |
| — reserve retained | ≫ 4 h | ≥ 4 |
| **Decoded images** | **40** | 64 |
| — primary | 40 | 40 |
| — technical / recovery | **0** | ≤ 24 |
| Project storage | 52.5 GiB | ≤ 200 GiB |
| `/data` free | ~520 GiB | ≥ 50 GiB |
| GPUs used | **1** (device 1; the native implementation uses one device) | ≤ 4 |
| **Technical retries** | **0** | ≤ 1 |
| Failures recorded | 1 (import-path error, 2.2 s, no GPU work) | — |

No paid APIs, cloud GPUs or subscriptions were used; W&B and telemetry were
disabled (`WANDB_DISABLED`, `WANDB_MODE=disabled`, `HF_HUB_DISABLE_TELEMETRY`).
No external messages were sent. CPU/BLAS workers were capped at 8 threads on a
busy shared host, and only one GPU was held (device 0 was left to another user).

## 9. Licensing, attribution and redistribution

### GEM source — MIT
`manifests/GEM_LICENSE_MIT.txt`. Copyright (c) 2026 GEM Authors.
**The MIT licence of the source code does not relicense FLUX-derived weights.**

### FLUX.1 [dev] — Non-Commercial License v1.1.1
Full text: [manifests/FLUX.1-dev_LICENSE.md](manifests/FLUX.1-dev_LICENSE.md)
(fetched at the pinned revision; sha256 in the evidence index).

Redistribution of the trained checkpoint was checked **before** training.
§2(a) grants a licence to "access, use, create Derivatives of, and **Distribute**
the FLUX.1 [dev] Models and Derivatives solely for your Non-Commercial
Purposes", and §1(a) defines a Derivative to include "any customized or
**fine-tuned** version thereof". §3 then sets the conditions, which this release
meets:

* **§3(a)** — a copy of the licence is distributed with the artifacts, and any
  rights to use them are granted **directly by Black Forest Labs Inc. to the
  recipient** under that licence.
* **§3(b)** — the Attribution Notice is displayed with the Distribution:

  > The FLUX.1 [dev] Model is licensed by Black Forest Labs Inc. under the
  > FLUX.1 [dev] Non-Commercial License. Copyright Black Forest Labs Inc.
  > IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR
  > OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  > FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

* **§3(c)(i) — modification notice:** *The published transformer is a **modified**
  FLUX.1 [dev] Model. It was fine-tuned by this project using GEM LoRA adaptation
  on the `qk_dual` subset and the adapter was merged into the transformer
  weights. 76 weight tensors differ from the pristine base.*
* **§3(c)(iii) — no endorsement:** this is **not** an official Black Forest Labs
  product and has **not** been endorsed, approved or validated by Black Forest
  Labs Inc. Likewise it is not endorsed or validated by the GEM authors.
* **§2(e)(ii) — AI-generated content disclosure:** *all 40 published images were
  **generated by an artificial-intelligence model** (FLUX.1 [dev], and a GEM-
  fine-tuned derivative of it). They are not photographs and do not depict real
  objects, places or events.*
* **Non-Commercial Purpose:** this is public, non-commercial research published
  for the benefit of public knowledge, with no direct or indirect payment
  arising from it (§1(c)(i)).

### What is *not* published — stated precisely
* **The base model is not republished.** The release contains **only** the
  modified `transformer` component (and the LoRA adapter that produces it). The
  text encoders, VAE, tokenizers and scheduler are **not** included, and no
  working pipeline can be assembled from this release alone — gated base access
  is required.
* **Honest qualification:** a merged derivative *inherently* carries the
  unchanged base values for every tensor the training did not touch, and that is
  true here — 1084 of 1160 tensors hold pristine base values, and as noted above
  **one of the three published shards is byte-identical to its base shard**. This
  is what the FLUX licence calls a Derivative (§1(a): "any customized or
  fine-tuned version"), and distributing it is what §2(a)/§3 permit; it is not a
  republication of the base model as such. The smaller 27.6 MiB adapter route is
  offered precisely for anyone who prefers to carry no base bytes at all.
* **No credentials** of any kind are published.

### Requirements to use the checkpoint
* **Gated base access.** You must have accepted the FLUX.1 [dev] licence and
  have access to `black-forest-labs/FLUX.1-dev` at revision
  `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`. The adapter route additionally
  requires downloading that base yourself.
* **Software.** Python ≥ 3.11 with the locked environment
  ([manifests/uv.lock.frozen](manifests/uv.lock.frozen)); the pinned GEM source at
  `c328fc6`; a ≥ 40 GB GPU for 512×512 bfloat16 inference as configured.

## 10. Release assets

Tag **`modern-object-assets-v1`** — 26 assets, 23,861,428,095 bytes total, every
asset under GitHub's 2 GiB per-asset limit.

| asset | parts | bytes | sha256 (reconstructed) |
|---|---|---|---|
| `gem-flux-teapot-v1-transformer.tar` (native merged transformer) | **23** × ≤ 1 GiB | 23,803,095,040 | `32d8202d0bdae63674fb7ea9a530f1b13df5acfbf831526c764e12deb1c6381d` |
| `gem-flux-teapot-v1-lora-adapter.tar.gz` | 1 | 27,621,641 | `bc801fb9bb9830bff43c3f568e9cd1473ff1f8ef6169608e0cf73385e01bef96` |
| `gem-flux-teapot-v1-images.tar.gz` (40 primary PNGs + contact sheet + indices + noise tensors) | 1 | 18,265,516 | `8b254ab3f6cbc64330600c2918bcd921fd9c609b32f1905cc6db7539ebff6420` |
| `gem-flux-teapot-v1-evidence.tar.gz` | 1 | 12,445,898 | `49fdea2338028515e2ecf694d86ac0225684899a90759da3264886400bea7488` |

Per-part sizes and sha256: [manifests/ASSET_MANIFEST.json](manifests/ASSET_MANIFEST.json).

### Reconstructing the checkpoint

```bash
# 1. download all 23 parts from the release, then:
cat gem-flux-teapot-v1-transformer.tar.part* > gem-flux-teapot-v1-transformer.tar
sha256sum gem-flux-teapot-v1-transformer.tar
# expect 32d8202d0bdae63674fb7ea9a530f1b13df5acfbf831526c764e12deb1c6381d
tar -xf gem-flux-teapot-v1-transformer.tar          # -> transformer/

# 2. load it with the pinned base (GEM's own documented path):
#    PipelineConfig.default_for_model_type('flux'), model_dtype='bfloat16',
#    model_name_or_path=<dir containing transformer/>
#    -> overwrite_pipeline_components_from_directory replaces 'transformer'
```

Nothing was quantized or cast to fit hosting; the parts are a lossless split.

## 10b. Remote verification — performed, passed

Every published asset was downloaded into a **fresh directory** and checked
end-to-end (CPU only: **0 GPU device-hours, 0 decoded images**).

| check | result |
|---|---|
| Assets downloaded | **31 / 31** (1321 s) |
| Remote hashes verified against `ASSET_MANIFEST.json` | **30 / 30** |
| Lossless reconstruction of the 23-part checkpoint | **yes** — `32d8202d0bdae636…` expected **=** got |
| Inner checkpoint files match published per-file hashes | **5 / 5** |
| Remote checkpoint loads via the documented pinned-base path | **yes** (`overwrite: ['transformer']`, 1.0 s) |
| Remote reload vs local canonical post-merge | **bitwise-identical** — 0 of 1160 tensors differ, aggregate `71c8f08e…` |
| Remote reloaded tensors finite | **yes** |
| Primary file completeness | **40 / 40** PNGs (20 M0 + 20 edited) |
| PNG hash mismatches | **0** |
| Paired-input correspondence re-verified from the download | **yes** — 20 / 20 initial-noise tensors |
| Download failures | **none** |

Verdict: **`REMOTE_VERIFICATION_PASSED`**
([REMOTE_VERIFICATION.json](REMOTE_VERIFICATION.json)).

## 11. Reproducing the pairs

```bash
source scripts/env.sh                      # caches on /data, telemetry off
python scripts/make_noise.py               # 20 initial-noise tensors + hashes
python scripts/run_gen.py --condition m0     --out-dir gen/m0
python scripts/run_gen.py --condition edited --ckpt <ckpt> --out-dir gen/edited
python scripts/verify_pairs.py             # pairing + integrity + contact sheet
python scripts/verify_reload.py --run-dir <ckpt>
```

## 12. Relationship to other work in this repository

This project is **self-contained and additive**. All closed experiments,
surveys and assessments remain **unchanged**: RECE/SD1.x, DP/UCE, Minimalist,
EraseAnything, BLOCK, the modern erasure gap map, the changing-replacement
assessment and `measurement_study_v1`.

Specifically, this work did **not**: rerun their pipelines; reuse any edited
checkpoint, image, label or prediction; repair their implementations; populate
the frozen empty measurement package; expose `measurement_study_v1`'s private
materials; recruit reviewers; or execute any human-outcome analysis. Only
verified-pristine base-model caches and generic infrastructure were reused, and
those read-only. The pre-existing `block-object-sprint-v1` release is preserved.

Because this is a **public** asset release, no claim is made that any future
reviewer remains blinded to this provenance.
