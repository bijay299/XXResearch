# RECE GolfBall_ep0 author-checkpoint screen -- prospective run manifest

Written BEFORE generation. Evaluation-only screen of an author-released
checkpoint for a credible first-edit baseline toward the sequential-
unlearning project. No training, no new editing, no hyperparameter search.

## 1. Source and provenance

| | value |
|---|---|
| HF repo | `ChaoGong/RECE` (author-linked), revision/sha `0c08d2bd30a800818d0e500987766ffeb2ec7105` |
| file downloaded | `GolfBall_ep0.pt` only (not the full collection) |
| HF x-linked-size | 3,438,384,718 bytes (~3.44GB, matches expected) |
| HF x-linked-etag (sha256) | `62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b` |
| local file sha256 (computed after download) | `62720e14f828e98f974e04de2e60f6133521b36126401438ec95b0e2b839f97b` -- **matches** |
| local path | `checkpoint/GolfBall_ep0.pt` |
| official source repo | `https://github.com/CharlesGong12/RECE`, pinned commit `403a67fd86dce6c124b88798340e001e625b28c9` (main HEAD at clone time), cloned to `repo_src/RECE/` |
| base model | `CompVis/stable-diffusion-v1-4` -- **the exact same local snapshot already cached for the UCE experiments**: `hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b`. This is not an assumption: the pinned `execs/generate_images.py` (line 231) hardcodes the authors' own base path ending in the identical snapshot hash `133a221b8aa7292a167afc5127cb63fb5005638b` -- confirming our cached base is bit-identical to what the authors used, not merely "compatible." |
| environment | reused `uce_apple_reference_screen/venv-uce/` unmodified (diffusers/transformers/torch already present and sufficient; no new environment built) |

## 2. Checkpoint structure (inspected, not assumed)

`torch.load(GolfBall_ep0.pt)` yields a flat `OrderedDict`, 686 keys, all
`torch.float32`. Compared directly against `UNet2DConditionModel`'s own
`state_dict()` loaded from the base snapshot's `unet/` subfolder:

- key sets are exactly equal (686 == 686, 0 keys only in base, 0 keys
  only in checkpoint);
- every shared key's shape matches exactly (0 mismatches);
- **32 of 686 tensors differ from the base weights; 654 are bit-identical
  to base.** The 32 changed tensors are exactly the set
  `{attn2.to_k.weight, attn2.to_v.weight}` across all 16 cross-attention
  blocks -- the identical key set our own UCE edits change. This is a
  genuine, verified provenance finding: the released file is a **full,
  complete 686-key UNet checkpoint by format** (every key present,
  loadable via one `load_state_dict(..., strict=True)` call, consistent
  with "do not treat it as our previous sparse UCE K/V edit or add it as
  a delta" -- it is loaded as a complete replacement, not a delta), but
  only the same narrow cross-attention K/V subset was actually altered by
  training, same as our own closed-form edits. This is reported as an
  observation, not used to change the loading procedure: the whole
  686-key state dict is loaded via strict full replacement regardless of
  which subset happens to differ numerically.
- Loading method: `unet.load_state_dict(ckpt, strict=True)` on a freshly
  loaded base UNet (float32); verified post-load that all 686 tensors in
  the resulting model equal the checkpoint's own saved values exactly
  (`torch.equal`, every key), and that this matches the 32-changed/654-
  unchanged split found above.

## 3. Resolved inference mechanics (from the pinned `execs/generate_images.py`)

| setting | resolved value | source |
|---|---|---|
| base pipeline load | `StableDiffusionPipeline.from_pretrained(dir_,)` -- **no `torch_dtype` argument** | line 237 -- defaults to float32, confirmed: this run uses float32 throughout, not fp16 (unlike the earlier UCE eval harness) |
| scheduler | `LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)` -- constructed directly, NOT the pipeline's bundled default scheduler | line 20 |
| resolution | 512 x 512 (`image_size` default, no per-row override columns present in `imagenette.csv`) | lines 42-43, confirmed CSV has no `sd_image_height`/`width` columns |
| inference steps | 50 (`ddim_steps` default) | line 208 |
| guidance scale | 7.5 (default; CSV has no `evaluation_guidance` column, confirmed) | line 204, 48 |
| seed mechanism | `generator = torch.cuda.manual_seed(seed)` -- **this call reseeds the GLOBAL per-device CUDA RNG and returns `None`**; the subsequent `torch.randn(..., generator=generator, device=torch_device)` therefore passes `generator=None` and draws from the just-reseeded global state, NOT from an independent `torch.Generator` object. Reproduced exactly as written (not substituted with an explicit `torch.Generator(device=...).manual_seed(seed)`, which would NOT produce the same sequence) -- this is what "paired initial noise" requires here. | lines 50, 66-69 |
| seed source | `row.evaluation_seed` (CSV column present, used directly, unmodified) | line 34 |
| prompt encoding | CLIP tokenizer `padding="max_length"`, `truncation=True`; uncond branch is a `[""]*batch_size` prompt, same `max_length` | lines 54-62 |
| classifier-free guidance | `noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)` | line 92 |
| latent scaling | `scheduler.scale_model_input` each step; final `latents = 1/0.18215 * latents` before VAE decode | lines 84, 98 |
| output | `vae.decode(latents).sample`, scaled to `[0,1]`, saved as uint8 PNG | lines 100-105 |

Both checkpoints (original SD1.4 UNet and the loaded `GolfBall_ep0` UNet)
share the identical tokenizer, text encoder, VAE, scheduler construction,
precision (float32), and per-row seed -- only the UNet weights differ.

**Compatibility accommodation, documented (encountered and fixed during
execution, not anticipated here):** the author's own
`StableDiffusionPipeline.from_pretrained(dir_,)` call omits
`safety_checker=None`. The first execution attempt failed with
`OSError: Error no file named model.safetensors, or pytorch_model.bin ...`
because this project's cached SD1.4 snapshot never downloaded the
(~1.2GB) `safety_checker` component's weights -- every prior use of this
snapshot in this project (UCE experiments) explicitly passed
`safety_checker=None`, so that subfolder is present but empty. Since
`generate_images.py` never calls the pipeline's own safety-checking path
(it extracts `vae`/`tokenizer`/`text_encoder`/`unet` directly and runs its
own manual sampling loop), loading or skipping the safety checker has
zero effect on the replicated generation mechanics or output pixels.
Fixed by passing `safety_checker=None, feature_extractor=None` to
`from_pretrained` rather than bulk-downloading an unused ~1.2GB
component -- a documented, necessary compatibility accommodation, not a
silent substitution of anything that affects the comparison.

**Second compatibility accommodation:** `LMSDiscreteScheduler` requires
`scipy`, not previously installed in `venv-uce` (the project's UCE work
never used this scheduler). Installed `scipy` via `pip install scipy`
into `venv-uce` only -- a small, non-training scientific-computing
dependency required to construct the authors' own documented scheduler
exactly as specified; `venv-cuig` and `SPEED/venv-speed` were not
touched.

## 4. Classifier mechanics (from the pinned `train_object.py`)

| setting | resolved value |
|---|---|
| model | `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` |
| **resolved weights enum (frozen, recorded)** | `ResNet50_Weights.IMAGENET1K_V2` (confirmed: `ResNet50_Weights.DEFAULT.name == "IMAGENET1K_V2"` in the installed torchvision 0.20.1+cu121) |
| preprocessing | `weights.transforms()` -- the official preprocessing bundled with this exact weights enum, used unmodified |
| top-k | top-1 only (this screen does not need top-5) |

**Explicit class -> ImageNet-1k category mapping** (verified by direct
lookup in `weights.meta["categories"]`, not assumed):

| imagenette class | ImageNet-1k index | ImageNet-1k category string |
|---|---|---|
| tench | 0 | `tench` |
| English springer | 217 | `English springer` |
| cassette player | 482 | `cassette player` |
| chain saw | 491 | `chain saw` |
| church | 497 | `church` |
| French horn | 566 | `French horn` |
| garbage truck | 569 | `garbage truck` |
| gas pump | 571 | `gas pump` |
| **golf ball (target)** | **574** | `golf ball` |
| parachute | 701 | `parachute` |

All ten map to a single, unambiguous ImageNet-1k index each (verified by
exact string match against the full 1000-category list; no collisions).

**A discrepancy from the author's own published comparison logic is
reported, not silently inherited**: `train_object.py`'s accuracy
computation (`image_classify`, lines 104-109) compares
`merged_df['category_top1'][i].lower() == merged_df['class'][i]` --
lowercasing only the *predicted* category string, not the CSV's `class`
column. Because ImageNet's official category strings preserve proper-noun
capitalization for "English springer" and "French horn" (confirmed above:
`weights.meta["categories"][217] == "English springer"`,
`[566] == "French horn"`, matching the CSV's own capitalization exactly),
`.lower()`-ing only one side turns these into `"english springer"` /
`"french horn"`, which can never equal the CSV's capitalized
`"English springer"` / `"French horn"`. As written, this comparison can
never register a correct prediction for these two retention classes,
regardless of what the classifier actually predicts. This screen does
**not** replicate that asymmetry: per this task's explicit instruction to
"map each manifest class explicitly to the classifier's category
identity," correctness is determined by **comparing the predicted top-1
ImageNet-1k index directly against the explicit index in the table
above** (index-based, not string-based) -- a robust equivalent that is
immune to this casing artifact, documented here as a difference from the
author's literal code rather than silently matched or silently fixed
without mention.

## 5. Frozen evaluation manifest

200 rows (10 classes x 20), first 20 rows per class from the pinned
`dataset/imagenette.csv` after sorting numerically by `case_number`,
prompts and `evaluation_seed` values retained unmodified. Verified: all
200 `case_number` values unique; each class contributes exactly 20
contiguous case numbers (e.g. golf ball: 4000-4019); no substitution of
prompts or seeds. Manifest: `manifest/frozen_manifest.json`
(sha256 `2fea62d13268bddeb70cf0e01f0885b7714aa90b03d3e341e8216f1b4c86e3b3`),
source CSV sha256 recorded inside the manifest file itself.

This is a **bounded reference screen using the authors' published prompt
pool** -- not a full paper reproduction, and not a guaranteed-untouched
confirmation set (the released checkpoint was itself selected using
validation performance by the authors).

## 6. Frozen screening gates (prospective, before generation)

- Baseline adequacy: original SD1.4 must register the golf-ball top-1
  label on >=16/20 target images. If this fails, the screen is classified
  **inconclusive for deletion**, not a success.
- Target suppression: >=75% relative reduction in golf-ball top-1 rate
  vs. that baseline.
- Preservation: no individual one of the 9 non-target classes loses more
  than 20 percentage points of top-1 accuracy vs. its own matching
  baseline.
- All nine retention classes reported individually plus their macro-
  average; a class with weak baseline recognition is flagged as weak
  preservation evidence, not dropped or treated as proof of preservation.
- Gates are not adjusted after seeing results.

## 7. Resource plan and scope

- Download, checkpoint inspection, repo cloning, and manifest construction
  all performed CPU-only, before any GPU is acquired.
- At most 1 genuinely idle GPU, selected via
  `/home/bijaypandey/FINMLResearch/scripts/gpu_select.sh --require 1`,
  released immediately after all generation and classification complete.
- Exactly 400 images maximum (200 original SD1.4 + 200 edited RECE
  checkpoint); no additional checkpoint, no alternate seed, no prompt
  revision, no second-stage edit, no confirmatory rerun.
- All downloads/caches under `/data` via the existing `env.sh`
  configuration (`HF_HUB_CACHE`, `TORCH_HOME`); no change to that
  configuration.
- All previous experiment directories preserved untouched.
