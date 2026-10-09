# Configuration deviations from upstream CUIG

Upstream reference: CUIG @ `9932ac3271a122f6d38d19e0b8c8908fe5237ff7`,
script `BashScripts/Independent/Style/Base/ConAbl.sh`.

Every deviation below is deliberate and recorded. **No training hyperparameter
is changed.**

---

## 1. Scheduler removed (Slurm → direct execution)

**Upstream.** `BashScripts/Independent/Style/Base/ConAbl.sh` is a *submission
generator*. It loops over 12 styles, writes a temporary `#SBATCH` job file per
style (`--gpus-per-node=1`, `--cpus-per-task=4`, `--time=02:00:00`) and calls
`sbatch` on each. `BashScripts/submit.sh` refuses to load its config unless
`CUIG_SLURM_ACCOUNT`, `CUIG_SLURM_CLUSTER` and `CUIG_SLURM_PARTITION` are all
set.

**Here.** This host has no Slurm and the brief asks for one concept on one GPU.
`scripts/run_single_concept_conabl.sh` keeps the three stages of the generated
job body — train → sample → evaluate — and executes them inline for a single
`--style`. The `sbatch` wrapper, the 12-style loop and the `#SBATCH` directives
are dropped. The three `CUIG_SLURM_*` variables are set to the literal
`none-direct-execution` in `scripts/pilot_config.sh` purely to satisfy
`submit.sh`'s presence check; they are never used, because we do not source
`submit.sh` for submission.

**Preserved exactly** (byte-identical argument list to the upstream job body):

```
--concept_type style
--iterations 1000
--num_anchor_images 200
--num_anchor_prompts 200
--anchor_dataset_dirs  <...>/anchor_datasets/style/Laion
--anchor_prompt_paths  <...>/anchor_prompts/style/Laion.txt
--scale_lr --hflip --noaug --enable_xformers_memory_efficient_attention
```

plus every upstream argparse default left untouched: `learning_rate 2e-6`
(scaled by `--scale_lr` to `2e-6 × 1 GPU × 1 accum × 4 batch = 8e-6`),
`anchor_batch_size 4`, `parameter_group kv-xattn`, `lr_scheduler constant`,
`lr_warmup_steps 500`, `seed 42`, `epochs 1`, `max_grad_norm 1.0`,
`adam_(beta1 0.9, beta2 0.999, weight_decay 1e-2, epsilon 1e-8)`,
`with_anchor_preservation` off, all regularizers off
(`l1sp_weight 0`, `l2sp_weight 0`, no projection, no SelFT, no `eval_interval`).

Sampling and evaluation keep the upstream concept split verbatim: target style +
12 retain styles as `--styles_subset`, 8 retain objects as `--objects_subset`,
and `--unlearn` / `--retain` / `--cross_retain` matching
`Evaluation/UnlearnCanvas/constants.py`'s `XLSX_ALL_*` lists. Sampler defaults
(5 seeds `188 288 588 688 888`, `guidance_scale 9.0`,
`num_inference_steps 100`, `resolution 512`) are untouched.

---

## 2. Accelerate config: `gpu_ids: '0'` → `gpu_ids: all`  ← safety-critical

**File.** `configs/accelerate_single_visible_gpu.yaml` is a copy of upstream
`Configs/Accelerator/single_gpu.yaml` with exactly one line changed:

```diff
-gpu_ids: '0'
+gpu_ids: all
```

**Why.** With `gpu_ids: '0'`, `accelerate launch` *overwrites*
`CUDA_VISIBLE_DEVICES` in the child environment. Verified on this host:

```
$ CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file Configs/Accelerator/single_gpu.yaml probe.py
CVD_INSIDE= '0'          # our pin discarded → would run on physical GPU 0

$ CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file configs/accelerate_single_visible_gpu.yaml probe.py
CVD_INSIDE= '3'          # our pin respected
```

On this shared, unscheduled host, physical GPU 0 belongs to another researcher.
Upstream's value would have silently moved training onto their device. With
`all`, accelerate leaves the variable alone and our outer pin is authoritative;
since exactly one device is visible, it is visible index 0 — which is what
upstream's `'0'` was expressing in the first place.

**Training-relevant fields are identical to upstream:**
`distributed_type: 'NO'`, `num_processes: 1`, `num_machines: 1`,
`mixed_precision: 'no'`, `downcast_bf16: 'no'`, `use_cpu: false`.
So the run is still single-process, single-GPU, full fp32 — no precision change.

---

## 3. Asset and output paths moved to `/data`

The filesystem carrying `/home` is at **100% (5.8 GB free)**, so the venv,
checkpoints, anchor images, generated samples and logs all live under
`/data/bijaypandey/cuig_pilot/` (1.4 TB free). Upstream defaults would have put
checkpoints inside the CUIG checkout (`Checkpoints/...`) and anchor datasets at
`UnlearningMethods/ConAbl/anchor_datasets/...`; these are redirected via
`scripts/pilot_config.sh` (`CUIG_UNLEARNCANVAS_GENERATOR_DIR`,
`CUIG_UNLEARNCANVAS_CLASSIFIER_DIR`, `--anchor_dataset_dirs`). `HF_HOME`,
`TORCH_HOME`, `PIP_CACHE_DIR` and `TMPDIR` are likewise redirected, otherwise
model downloads would fill the root filesystem. No code change; paths only.

---

## 4. Environment built with `venv` + pip instead of conda

Upstream ships `env.yaml` for `conda env create`. No conda/mamba/micromamba is
installed on this host and the system Python is already 3.10.12, matching
upstream's `python=3.10`. `scripts/setup_env.sh` therefore builds a `venv` and
pip-installs the **same pinned versions** from `env.yaml`, restricted to the
ConAbl + UnlearnCanvas import closure. Verified installed:

| Package | Pin in `env.yaml` | Installed |
|---|---|---|
| python | 3.10 | 3.10.12 |
| torch | 2.4.0 | 2.4.0+cu121 |
| torchvision | 0.19.0 | 0.19.0+cu121 |
| xformers | 0.0.27.post2 | 0.0.27.post2 |
| diffusers | 0.30.2 | 0.30.2 |
| transformers | 4.44.2 | 4.44.2 |
| tokenizers | 0.19.1 | 0.19.1 |
| accelerate | 0.34.2 | 0.34.2 |
| huggingface-hub | 0.32.4 | 0.32.4 |
| safetensors | 0.4.5 | 0.4.5 |
| timm | 1.0.15 | 1.0.15 |
| pytorch-lightning | 2.5.1.post0 | 2.5.1.post0 |
| numpy | 1.26.4 | 1.26.4 |
| pillow | 10.4.0 | 10.4.0 |
| openpyxl | 3.1.5 | 3.1.5 |
| openai | 0.28.0 | 0.28.0 |
| gdown | 5.2.1 | 5.2.1 |

Omitted from `env.yaml` because nothing on this code path imports them: the
Jupyter stack, `tensorflow-datasets`, `lm_eval`, `NudeNet`, `onnxruntime-gpu`,
`wandb`, `bitsandbytes`, `lpips`, `cleanfid`-adjacent and celebrity/nudity
evaluation dependencies. `wandb` and `bitsandbytes` are imported only inside
`if` branches we do not enable (`--report_to wandb`, `--use_8bit_adam`);
`openai` *is* a hard top-level import in `ConAbl/src/data.py`, so it is
installed even though no API call is made (see §5).

Exact resolved versions: `env/requirements.lock.txt`.

CUDA note: the system default `nvcc` is 11.5, but the driver is 550.144.03
(CUDA 12.4) and the torch wheels bundle their own CUDA 12.1 runtime, so the
system toolkit is irrelevant to these runs.

---

## 5. Things that look like deviations but are not

- **`--num_anchor_prompts 200` does not call OpenAI.**
  `anchor_prompts/style/Laion.txt` has **exactly 200** non-empty lines (the file
  has no trailing newline, so `wc -l` reports 199 and is misleading). CUIG only
  invokes `_generate_anchor_prompts` — which would need `OPENAI_API_KEY` and
  `gpt-4.1` — when the file holds *fewer* prompts than requested. At exactly
  200 the branch is not taken, so the released configuration runs unmodified
  with no API key. Any value above 200 would trigger a live OpenAI call.

- **The `images/` path segment is mandatory.**
  `Evaluation/UnlearnCanvas/evaluate.py::_record_prediction` computes
  `img_path.split("images/")[1]`, which raises `IndexError` if the sample
  directory path contains no `images/` segment. Upstream's
  `Results/<style>/images` layout satisfies this incidentally; our scripts keep
  the same layout deliberately, and the requirement is commented in-script.

- **`sample.py --device cuda:0` is correct under our pin.** It calls
  `torch.cuda.set_device("cuda:0")`, i.e. the first *visible* device, which is
  the single GPU we expose. No change needed.

---

## 6. Substitute-configuration mechanics check (not a baseline)

Because the UnlearnCanvas generator and classifiers could not be obtained
(`docs/ASSETS.md`), `scripts/mechanics_check.sh` exercises the code path with
base Stable Diffusion v1.5 and deliberately reduced settings
(`--iterations 8`, `--num_anchor_images 16`, `--num_anchor_prompts 16`,
`--epochs 4`, 4 sampled images). Its outputs are written to a separate root,
`/data/bijaypandey/cuig_pilot/mechanics_check/`, never to the experiment output
root, and its report is stamped `IS_BASELINE: false`,
`IS_REPRODUCTION: false`. UA/IRA/CRA are not computed there and remain
unavailable. These reduced settings are **not** a deviation in the baseline —
the baseline configuration in §1 is unchanged and simply has not been run.
