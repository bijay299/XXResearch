# RUN_MANIFEST — modern FLUX DP/UCE reproduction + sequential extension

Frozen 2026-10-16, before any model edit or image generation, on host "hyperplane".

## Source code

- Repo: `https://github.com/czhang024/CVPR_DouleProjection`
- Frozen commit (checked out, detached HEAD): `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f`
- Local checkout: `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo`
- Paper: `https://arxiv.org/abs/2604.10032`
- Key file sha256 (pinned commit):
  - `SD/data/small_imagenet_prompts.csv`: `4ab01d9e98b0a23082b9cdbf52815cc639201fe6ba33b6ed8bbab58984c9fb2d`
  - `Flux/Double_proxy.py`: `832d986fd3f2a4dde2565bc8fe8d2a2747b7892645c593e4e57496537cb1a850`
  - `Flux/UCE_original.py`: `7cb88b5130d5fa2e2a4bf37a42e3b12ec09416cdaca058fb7dcbfbccc8204b33`
  - `Flux/generate_erased_flux.py`: `9f0d6ae74cc45e96f269d6d5ecab9209c7460f8eb99205e5fcd3363765697ad0`
  - `Flux/generate_original_flux.py`: `dd3e23d6e52741490d3239d630f31c24caf906340eadacc80760b077371cdd2c`
  - `Flux/compare_accuracy.py`: `3badb2e8bfc316f0b8b0f275e77de4262b8c35f5a497dd53664a437e1c7dc493`
  - `Flux/scripts/Double_CassettePlayer_flux.sh`: `5821ed9d5aa3dafffefdc79dc9ca9f05fcf7d0a24d18e9c7ea5e145c777685af`
  - `Flux/scripts/Double_ChainSaw_flux.sh`: `1690a213eed882a5814c6e23abb0afc22b6b10a8cd46982d2043f2958580de20`
  - `Flux/scripts/UCE_CassettePlayer_flux.sh`: `4affffb7315202f096e158f320674847d322b0cdaa3e09bea98b65276cc2ad6c`
  - `Flux/scripts/UCE_ChainSaw_flux.sh`: `064b46b6344d3646eac4877e22c6f94f5184bc4686c86414ff9b93e8697dffea`

No repairs to source were needed: pinned `requirements.txt` installed cleanly on
Python 3.10.12 / CUDA 12.1 wheels against driver 12.4; no patch file produced.

## Model

- `black-forest-labs/FLUX.1-schnell`, revision (full commit SHA) `741f7c3ce8b383c54771c7003378a50191e9efe9`
- Gated repo (`gated: auto`, Apache-2.0); access confirmed under HF account `CaptIce` using a
  user-supplied read token (stored only at `/data/bijaypandey/cuig_pilot/secrets/hf_token.env`,
  chmod 600, never logged).
- Diffusers-format files only downloaded (excluded the single-file `flux1-schnell.safetensors`,
  `ae.safetensors`, and `schnell_grid.jpeg`, which diffusers' `FluxPipeline` does not use).
- Local snapshot: `/data/bijaypandey/cuig_pilot/hf_home/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9`
- Downloaded weight-shard sha256, verified against the Hub's recorded LFS oids (exact match, all 7):
  - `transformer/diffusion_pytorch_model-00001-of-00003.safetensors`: `9b633dbe87316385c5b1c262bd4b5a01e3d955170661d63dcec8a01e89c0d820`
  - `transformer/diffusion_pytorch_model-00002-of-00003.safetensors`: `58b4434078f0c2567ddc54e3b5cbf39626ab55fbd9d5c22956e183668f535dec`
  - `transformer/diffusion_pytorch_model-00003-of-00003.safetensors`: `e2cbc25471ed5186e69a9b51098300cb2f612556453e38a372c851a220ed238d`
  - `text_encoder/model.safetensors`: `893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082`
  - `text_encoder_2/model-00001-of-00002.safetensors`: `ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e`
  - `text_encoder_2/model-00002-of-00002.safetensors`: `a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00`
  - `vae/diffusion_pytorch_model.safetensors`: `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`
- Classifier: `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` resolves to
  `ResNet50_Weights.IMAGENET1K_V2` (`resnet50-11ad3fa6.pth`, 1000 ImageNet categories).

## Environment

- Isolated venv (not shared with the RECE/SD1.x `venv-cuig`): `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/venv_flux_dp_uce`, Python 3.10.12.
- Installed exactly from the pinned `source_repo/requirements.txt`; full `pip freeze` saved at
  `manifest/pip_freeze.txt`. Headline versions: `torch==2.4.0+cu121`, `torchvision==0.19.0`,
  `diffusers==0.30.2`, `transformers==4.44.2`, `accelerate==0.34.2`, `safetensors==0.4.5`,
  `xformers==0.0.27.post2`, `huggingface-hub==0.29.1`.
- Caches/scratch all on `/data` via `source /data/bijaypandey/cuig_pilot/env.sh`
  (`HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `TMPDIR`, `PIP_CACHE_DIR`); the
  pre-existing RECE/SD1.x caches under the same `hf_home` were left untouched (only new
  `models--black-forest-labs--FLUX.1-schnell` blobs added).
- HF credential stored at `/data/bijaypandey/cuig_pilot/secrets/hf_token.env` (chmod 600),
  same pattern as the pre-existing `openai_api_key.env`.

## Hardware

- Host "hyperplane": 4x NVIDIA A100-SXM4-40GB, driver 550.144.03, CUDA 12.4 (runtime wheels: 12.1).
- Ubuntu 22.04.5 LTS, kernel 5.15.0-136-generic.
- Device used for this round: `CUDA_VISIBLE_DEVICES=2` (verified unoccupied — 0% util, ~14MiB
  used, driver overhead only — immediately before each allocation; GPUs 0/1/3 left for other
  users / rechecked at each phase boundary). `/data` had 654GB free immediately before the first
  generation (695GB before the model download; the ~34GB FLUX.1-schnell diffusers download
  accounts for the difference).

## Fixed targets and scope (per protocol)

| Symbol | Concept | Guide |
|---|---|---|
| A | cassette player | box |
| B | chain saw | stick |

Preserve lists used verbatim from the pinned shell scripts (source order):
- Erasing A (cassette player): `chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;parachute;french horn`
- Erasing B (chain saw): `cassette player;church;gas pump;tench;garbage truck;english springer;golf ball;parachute;french horn`

Native single-edit settings (verified against the actual pinned code, not just the shell comments):
- **DP** (`Flux/Double_proxy.py`): `concept_type=object`, `expand_prompts=false`, `replace_indices=all` (default — the chain-saw shell comment about index 1 is not actually passed as an argument), `torch_dtype=bfloat16` for model/text encoders, edit math done in float32 and cast back to bf16 (`W_0_f32.to(torch_dtype)`).
- **UCE** (`Flux/UCE_original.py`): `concept_type=object`, `expand_prompts=false`, `erase_scale=10`, `preserve_scale=1`, `lamb=0.1`, fp32 edit math, pristine text encoders.
- Both edit exactly two weight tensors, confirmed by source inspection:
  `context_embedder.weight` and `time_text_embed.text_embedder.linear_1.weight`
  (matched via substring check `'context_embedder' in name or 'text_embedder.linear_1' in name`
  in both `Double_proxy.py:48` and `UCE_original.py:26`).
- Saved edited weights are a 2-key safetensors file; `generate_erased_flux.py` loads a pristine
  `FluxPipeline`, takes `pipe.transformer.state_dict()`, overwrites only matched keys (cast to
  bf16), `strict=True` loads back, and prints a per-key matched/warning count — this count will
  be captured verbatim in each endpoint's generation log, not assumed.

Generation settings (all confirmed as actual defaults in the pinned scripts, not assumed from
the shell comments): resolution 512x512, `num_inference_steps=4`, `guidance_scale=0.0`,
`max_sequence_length=256`, bf16 inference, `batch_size=1`, `global_seed=42`,
`base_seed_offset=0`, per-image seed = `evaluation_seed + variation_index + base_seed_offset`
when the CSV row has an `evaluation_seed` (all 100 rows do).

## Case manifest

Built before any generation from the unmodified
`source_repo/SD/data/small_imagenet_prompts.csv` (100 rows, 10 classes x 10 cases), replicated
to 10 variations each (1,000 cases total), using the exact seed/filename formulas above:
- `manifest/full_case_manifest.csv`, sha256 `559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26`
- Verified: 1,000 rows, 100 per class, no duplicate filenames, no duplicate (case, variation) pairs.

## Scope ceiling

M0 + 4 native singles (DP-A, DP-B, UCE-A, UCE-B) + up to 4 sequential endpoints (conditional on
the eligibility gates below) = max 9 endpoints x 1,000 images = 9,000 images. No endpoint is
regenerated once produced; no SD1.x/RECE generation is touched or repeated.

## Eligibility gates for the sequential extension (recorded here before running; results to be
filled into the final report, not altered after the fact)

- M0 must semantically recognize each target (A, B) in >= 80/100 of its own cases.
- Each method's native single edit must reduce its own target's semantic recognition by >= 75%
  relative to the matching M0 count, with technical integrity holding (both weight keys matched,
  finite values, save/reload verified).
- For a given order, the first parent must still recognize the *next* target in >= 80/100 cases
  before that order's second edit is attempted.
