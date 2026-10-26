# REPAIR_BASELINES_v2 — Meta-Unlearning launcher, IMMA fidelity, ESD provenance

Agent: BASELINES repair. Date 2026-10-26. Project FROZEN (`FREEZE_ACTIVE`).
**No GPU work was launched.** Everything below is CPU-verified or is a bounded command for the lead.

Files I own and edited: `run_r2_meta_official.sh`, `run_r1_imma_official.sh`,
`run_r1b_imma_train.sh`, `run_r1c_eval.sh`, `scripts/r2_build_meta_pools.py`,
`scripts/r1_eval_imma.py`, and this document. `scripts/gpu_reserve.sh` is **called, never edited**;
findings about it are in section D.

---

## 0. One measurement that governs three separate items

```
$ venv_t2i/bin/python  # UNet2DConditionModel, CompVis/stable-diffusion-v1-4,
                       # revision=133a221b8aa7292a167afc5127cb63fb5005638b
tensors=686 bitwise_identical=0 differing=686
MAIN_vs_FP16VARIANT rel_frobenius = 2.074e-04
main.half().float() == fp16variant for 686/686 tensors
```

The local snapshot holds **both** `unet/diffusion_pytorch_model.safetensors` (fp32 main,
3,438,167,534 B) and `unet/diffusion_pytorch_model.fp16.safetensors` (1,719,125,304 B). Loading the
wrong one changes **every** tensor at a relative Frobenius of **2.07e-04**, and the fp16 variant is
bit-exactly `main.half()`.

Consequences, each handled below: it broke the R2 no-op guard (A-7), it contaminates the IMMA
composition's reported deltas (B-4), and it is the reason "all 686 tensors differ" was never
evidence about ESD's erasure scope (C).

---

## PART A — META LAUNCHER (`run_r2_meta_official.sh`)

Upstream pin verified: `git -C upstream/Meta-Unlearning rev-parse HEAD` =
`5a2a7600895a2885323cbbf599acaf3f74a96f64`, tree clean.

The script now has three modes: `R2_MODE=cpu` (default, no GPU, stages 1–4), `smoke` (a minimal
real update), `full`. **`R2_MODE=cpu` has been executed; its complete output is in A-9.**

### A-1 Reservation API replaces the retired selector — DONE

Removed: `while :; do G=$(bash scripts/campaign_free.sh | head -1); ...` (old L119).
Added, at stage 5:

```bash
source scripts/gpu_reserve.sh
gpu_reserve_hold "r2_esd_meta_${MODE}" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"} >/dev/null || die ...
export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
```

Same substitution in `run_r1_imma_official.sh` (2 sites), `run_r1b_imma_train.sh`,
`run_r1c_eval.sh`. `grep -c campaign_free` over the four runners now returns 1/2/1/1 — all inside
comments recording the retired call. The ad-hoc `FREEZE_ACTIVE` checks were also removed from the
runners: `gpu_reserve.sh`'s freeze gate runs inside its own mutex and is the single choke point.

### A-2 No silent default to the legacy pools — DONE

`META_POOL_SCHEME` has **no default**; unset prints the two definitions and exits 2.
`legacy` additionally requires `META_HISTORICAL_RECONSTRUCTION=1`. The same gate is duplicated in
the builder: `--pool-scheme` is now `required=True` (was `default="legacy"`) and `legacy` demands
`--historical-reconstruction` (`scripts/r2_build_meta_pools.py`).

Pool roots now default **per scheme**, so a coherent build cannot land on the preserved historical
pools: `legacy → upstream_runs/meta/dataset`, `coherent → upstream_runs/meta/dataset_coherent`.

```
$ bash run_r2_meta_official.sh                        # no scheme
REFUSING TO RUN: META_POOL_SCHEME is not set, and this script has no default.
rc=2
$ META_POOL_SCHEME=legacy bash run_r2_meta_official.sh
REFUSING TO RUN: META_POOL_SCHEME=legacy requires META_HISTORICAL_RECONSTRUCTION=1.
  The legacy pools measure TOKEN SUPPRESSION, not concept erasure (see header).
$ venv_t2i/bin/python scripts/r2_build_meta_pools.py --root ... --verify-only
r2_build_meta_pools.py: error: the following arguments are required: --pool-scheme
$ ... --pool-scheme legacy --total-per-pool 30 --verify-only
[r2pools] FATAL: --pool-scheme legacy is a HISTORICAL RECONSTRUCTION, not a scientific setup: ...
```

### A-3 Pool CONTENT, counts, split roles, fingerprints, resume compatibility — DONE

`scripts/r2_build_meta_pools.py` gained:

* **split-role validation** — every caption must be one of *that pool's* declared prompts, each
  prompt must contribute exactly `per_prompt` images, and no declared prompt may contribute zero.
  Previously a split could hold the right *number* of images of the *wrong concept* and validate,
  which is precisely the failure that would invalidate the forget/retain assignment.
* **content fingerprints** — sha256 over filename-sorted `(file_name, sha256(bytes), caption)`,
  recorded per split in `pools.json` under `fingerprints`, and re-checked on later runs
  (`fingerprint MATCHES manifest` / `fingerprint DRIFT`).
* **resume compatibility** — `check_resume_compat()` diffs the on-disk `pools.json` against the
  requested plan field by field and reports `RESUME INCOMPATIBLE` naming the field. Before, resume
  validated only the new plan's counts, so resuming the historical pools under a different sizing
  died with a bare count error that read like data corruption.
* **`--verify-only`** — validates pools on disk and exits, importing no torch and touching no GPU.

Measured state of the real pools at `upstream_runs/meta/dataset` (scheme `legacy`, sizing
`total_per_pool`, 30 per pool per split):

```
  hrm/train: n=30 prompts=1 per_prompt=30 sha=bdb3dfa5a482757c
  hrm/test : n=30 prompts=1 per_prompt=30 sha=9915bd23a326e2ca
  tgt/train: n=30 prompts=3 per_prompt=10 sha=656fec31e1f04a02
  tgt/test : n=30 prompts=3 per_prompt=10 sha=6d0cd3694f9b4ff7
  irt/train: n=30 prompts=6 per_prompt=5  sha=8db1efad8be17024
  irt/test : n=30 prompts=6 per_prompt=5  sha=2b2212b638a1ae2c
[r2pools] verify-only: OK
```

All six splits are intact: counts, per-prompt proportions, split roles and zero blank frames.
No fingerprint was recorded in the pre-v2 manifest, so these are the baseline values.

Resume-incompatibility behaves as specified:

```
$ ... --pool-scheme legacy --historical-reconstruction --verify-only   # default sizing=per_prompt
[r2pools] FATAL: RESUME INCOMPATIBLE -- the pools on disk were built to a DIFFERENT definition:
    sizing on disk='total_per_pool' requested='per_prompt'
    pool 'tgt' per_prompt on disk=10 requested=30
    pool 'irt' per_prompt on disk=5 requested=30
  The existing pools are NOT deleted. Either match the on-disk definition ... or build the new
  definition under a DIFFERENT --root so both survive.
```

**Note on the historical pools:** sizing `total_per_pool` is a compute adaptation, not the author
protocol. `src/gen_images.py:88-92` generates 30 train + 30 test **per prompt**, so upstream would
be 30/90/180 per split for hrm/tgt/irt. `pools.json` records which was used; the verifier prints
`<-- upstream would be N/split` on every divergence.

### A-4 Three hyperparameter tiers — DONE

The script separates **TIER P** (paper-supported), **TIER R** (README/shipped example, or the
argparse default where no example passes it) and **TIER I** (our inferred repairs). Every value is
echoed to the log and written to `upstream_runs/meta/out/<exp>_config.json` under
`tier_P_paper_supported` / `tier_R_readme_or_parser_default` / `tier_I_our_inferred_repairs`.

**TIER P — `fix_timesteps = False`.** README.md:43: *"We use **fix_timesteps = False** in our
paper, and the **fixed_timesteps = True** is just used for accelerating training"*.
`src/esd_meta.py:117` declares `--fix_timesteps` with `type=bool`, under which `bool("False")` is
`True`. Verified by running the real parser: `--fix_timesteps False` → `True`;
`--fix_timesteps True` → `True`; **omitted → `False`**. The flag is therefore deliberately absent
and the script *refuses to start* if it ever appears in the argv array. Companion hazard,
`esd_meta.py:118`: `--fixed_time_steps` is `type=list` with no `nargs`, so `1 2 5 10` parses to
`['1']` and leaves `2 5 10` as unrecognised positionals → hard `SystemExit 2`. Never passed.

**TIER I-1 — the gamma2 triplicate. Our values are an INFERENCE, not author settings.**
`scripts/esd_meta.sh:14-16` (and `scripts/meta.sh:13-15`) spell `--gamma2_1` **three times** and
never pass `--gamma2_2` or `--gamma2_3`; all five gammas default to `1.0` (`esd_meta.py:90-94`).

| arg | shipped example | **resolves to** | ours (TIER I) |
|---|---|---|---|
| gamma1_1 | `0.1` (esd_meta.sh:12) | 0.1 | 0.1 |
| gamma1_2 | `0.01` (esd_meta.sh:13) | 0.01 | 0.01 |
| gamma2_1 | `0.1`,`0.1`,`0.01` (L14,15,16) | **0.01** (last wins) | 0.1 |
| gamma2_2 | never passed | **1.0** (default) | 0.1 |
| gamma2_3 | never passed | **1.0** (default) | 0.01 |

Term semantics, `esd_meta.py`: `+gamma1_1·grad(irt/test)` (:983,:994), `+gamma1_2·grad(tgt/test)`
(:989,:999), then one relearn epoch on **hrm/train** (`train_set=True`, :1007), then
`−gamma2_1·grad(hrm/test)` (:1014,:1032), `−gamma2_2·grad(tgt/test)` (:1020,:1038),
`−gamma2_3·grad(irt/test)` (:1026,:1044); applied to the student at :1049-1050 on top of
`train_loss.backward()` (:1003).

**As scripted this is not a mild misconfiguration.** `tgt/test` carries both `+0.01` and `−1.0`,
and `irt/test` both `+0.1` and `−1.0`, so the shipped example **net-ascends both retain pools** by
100× and 10×. Our `(0.1, 0.1, 0.01)` reconstructs the evident intent of three successive
`--gamma2_*` flags. `config.json` records `upstream_example_resolves_to: {0.01, 1.0, 1.0}` beside
our values so the distinction cannot be lost downstream.

Other TIER I items: `seed=0` (upstream passes no `--seed`; default `None` — upstream runs are not
reproducible); `lr_warmup_steps=0` (parser default is **500**, `esd_meta.py:128`, but
`get_scheduler(name="constant", ...)` at `:874-879` ignores `num_warmup_steps`, so this is
immaterial while the scheduler stays `constant` — recorded as such); `save_every/eval_every`
cadence; `--revision`/`--variant` (A-6); pool construction (A-8).

### A-5 Selected upstream variant recorded, and meta.py's defects verified from source — DONE

`src/esd_meta.py` is recorded as the entry point in the header, in the log banner, and in
`config.json` (`entry_point` / `rejected_entry_point`). **Stage 1 re-verifies the justification at
runtime and aborts if it goes stale.** I verified all three claimed defects independently:

* **4-tuple return vs 11 three-value call sites — CONFIRMED.** `meta.py:586` defines `train_step`;
  its only return, `meta.py:626`, is
  `return task_unet,task_optimizer,task_lr_scheduler,np.mean(all_losses)`. The 11 call sites are
  `meta.py:967,973,978,983,990,995,1001,1007,1013,1019,1025`, **all unpacking 3**. First executed
  call is :967 (fix branch) or :978 → `ValueError` on step 1.
* **`train_loss` never assigned — CONFIRMED.** Read at `meta.py:1045,1047,1050`; `grep` returns
  only those three lines. Root cause: `train_unlearn_step` is defined at `meta.py:506` and
  **never called**, so meta.py's main loop contains no unlearning objective at all.
* **`param.grad` is None — CONFIRMED.** `meta.py:1033` `param.grad += sum_grads[i]` over
  `parameters` from `:860`. The only `.backward()` calls are `:615`/`:659`, both inside
  `train_step`/`train_step_deng` acting on `deepcopy(unet_student)` (`:933`). No `.backward()` in
  `main()`. `optimizer.zero_grad()` (`:1041`, `set_to_none=True`) re-Nones them every step.

`esd_meta.py` is internally consistent: 3-tuple return at `:626`, `train_loss` assigned at `:933`
from `train_unlearn_step`, back-propagated at `:1003`, `param.grad += ...` at `:1050`.

**Initial checkpoint = plain SD-1.4, confirmed.** `esd_meta.py:839-844` builds **both**
`unet_teacher` and `unet_student` from `args.pretrained_model_name_or_path`, same
`subfolder="unet"`, same revision/variant; teacher frozen at `:847`. The ESD erasure is internal,
`train_unlearn_step` at `:506-584`, target `noise_target = e_0 - concept_scale*(e_p - e_0)`
(`:571-582`). There is no externally erased checkpoint to supply.

Stage 1 output (executed):

```
    pin OK: 5a2a7600895a2885323cbbf599acaf3f74a96f64
    [OK ] meta.py train_step returns a 4-tuple
    [OK ] meta.py has 11 three-value call sites
    [OK ] meta.py never assigns train_loss
    [OK ] meta.py reads train_loss anyway
    [OK ] meta.py never back-propagates train_loss
    [OK ] meta.py never calls train_unlearn_step
    [OK ] esd_meta.py train_step returns a 3-tuple
    [OK ] esd_meta.py assigns train_loss
    [OK ] esd_meta.py back-propagates train_loss
    [OK ] esd_meta.py hardcodes ./dataset/{hrm,irt,tgt}
    [OK ] esd_meta.py forwards --revision and --variant
    [OK ] fix_timesteps is still type=bool (the hazard)
    meta.py 3-value train_step call sites found: 11
```

### A-6 Model PIN, and the reload trap — DONE (this one was live)

**Correction to the brief:** `esd_meta.py` **does** accept `--revision` and `--variant`
(`:49`, `:51`, both default `None`) and forwards them to `from_pretrained` at `:839-844`. The
previous script passed neither, so the run floated on whatever `main` resolved to. Both are now
pinned from `MODEL_PIN.txt` (`revision=133a221b8aa7292a167afc5127cb63fb5005638b`, `variant=fp16`).

**THE TRAP, CONFIRMED LIVE.** The old stage-6 reload check compared the saved UNet against base
SD-1.4 loaded with `variant="fp16"` and asserted `rel > 1e-6` as proof that training moved the
weights. Since the trainer was loading the **main** weights, a checkpoint that had not trained at
all would have shown `rel ≈ 2.07e-04` — **207× the threshold** — and the assert would have PASSED
on a no-op. Repaired stage 9:

* the reference uses the **same revision AND the same variant** actually passed to the trainer;
* the **other** variant is loaded purely for calibration, and the **variant-only floor** is
  computed and printed, so the comparison can be judged rather than trusted;
* the guard is `rel_delta_vs_same_init > 10 × variant_only_floor`, not a bare `> 1e-6`;
* all of it is written to `reload_check.json` (`rel_delta_vs_same_init`,
  `rel_delta_vs_other_variant`, `variant_only_floor`, `bitwise_identical_tensors`).

### A-7 CONFIRMED BLOCKING DEFECT found and fixed: wrong venv

`esd_meta.py:669` does a module-level `from datasets import load_dataset`. The previous script ran
the trainer under `venv_t2i`, which **has no `datasets`**:

```
venv_t2i :  torch 2.4.1+cu124  diffusers 0.31.0  transformers 4.46.3  datasets MISSING  wandb MISSING
venv_imma:  torch 2.4.1+cu124  diffusers 0.24.0  transformers 4.46.3  datasets 5.0.1    wandb 0.31.0
```

The R2 run would have died with `ModuleNotFoundError: No module named 'datasets'` before a single
step, after taking a GPU. The trainer now runs under `venv_imma`, which imports `esd_meta.py` and
parses its arguments cleanly, and stage 2 asserts the import instead of discovering it on a GPU.
`HF_DATASETS_CACHE` is now also exported to `/data/.../hf_home/datasets` — the root filesystem is
full and `datasets` would otherwise try `/datasets` (observed: `PermissionError: '/datasets'`).

### A-8 Verification without a full campaign — DONE, and it executes or fails clearly

| required check | stage | how it is verified | status |
|---|---|---|---|
| argument parsing | 3 | the trainer's **own** `parse_args()` on the exact argv; asserts `fix_timesteps is False`, revision/variant present, `len(devices)==2` | PASS |
| dataset paths | 4 + 6 | `load_dataset("imagefolder", data_dir="./dataset/<pool>")` really resolved through the staging symlink | PASS |
| pool content | 4 | counts, per-prompt split roles, blanks, fingerprints, resume compat | PASS |
| a MINIMAL REAL UPDATE | 7 | `R2_MODE=smoke`: few steps with `save_every=1`, so a `step=*/` checkpoint is written and diffable | **needs GPU** |
| saved-checkpoint reload | 9 | reload + the variant-floor guard above | **needs GPU** |
| evaluation compatibility | 10 | emits `adapters/R2_<exp>/unet.pt` (fp16 flat state dict) and asserts key/shape compatibility with the evaluator | **needs GPU** |

Dataset-path resolution, executed on CPU from the staging directory:

```
  hrm: {'train': 30, 'test': 30} prompts=1 first='a photo of a church'
  irt: {'train': 30, 'test': 30} prompts=6 first='a photo of fridge'
  tgt: {'train': 30, 'test': 30} prompts=3 first='a photo of a cathedral'
```

`esd_meta.py:671-673` hardcodes `./dataset/{hrm,irt,tgt}` relative to the process CWD with no CLI
override (byte-identical in `meta.py`); no `--data_dir`-style argument exists in either file. The
staging symlink is the repair, and it is now *tested*, not assumed.

Artifact layout, from source: `esd_meta.py:785-790` appends a timestamp to `exp_name`, so stage 8
discovers the output dir. `save_checkpoint(step=None)` writes a **full StableDiffusionPipeline** at
the run-dir root (`:340-349`); periodic saves write **unet-only** into `step=NNNNNN/` (`:328-339`),
pruned to `--max_checkpoints`. Stage 8 checks the right one per mode and fails clearly otherwise.

### A-9 `R2_MODE=cpu` full output — EXECUTED, rc=0

```
=== R2 2026-10-26T14:33:35-04:00 mode=cpu concept=church scheme=legacy sizing=total_per_pool
    pool root: upstream_runs/meta/dataset
    upstream variant: src/esd_meta.py @ 5a2a7600...  (NOT src/meta.py -- see header)
    TIER R: nipp=10 batch=10 steps=1500 / ft=full lr=1e-05 g1_1=0.1 g1_2=0.01 cfg=3.0
            concept_scale=3.0 res=512 sched=constant
    TIER P: fix_timesteps=False  (FLAG OMITTED -- the only way to obtain False)
    TIER I: g2_1=0.1 g2_2=0.1 g2_3=0.01  <-- OUR INFERRED REPAIR.
            upstream's example RESOLVES to g2_1=0.01 g2_2=1.0 g2_3=1.0; not author settings.
    TIER I: seed=0 warmup=0 save_every=500 eval_every=500
    TIER I: revision=133a221b8aa7292a167afc5127cb63fb5005638b variant=fp16 devices='0 0'
[stage 1: 12/12 OK, see A-5]
    venv_imma: datasets 5.0.1 diffusers 0.24.0 torch 2.4.1+cu124 transformers 4.46.3
    confirmed: venv_t2i has NO datasets -- running the trainer there would ImportError
    RESOLVED ARGS: {"pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
      "revision": "133a221b8aa7292a167afc5127cb63fb5005638b", "variant": "fp16",
      "removing_concepts": ["church"], "validation_prompts": ["a photo of a church"],
      "num_images_per_prompt": 10, "train_batch_size": 10, "devices": [0, 0],
      "num_train_steps": 1500, "finetuning_method": "full", "learning_rate": 1e-05,
      "lr_scheduler": "constant", "lr_warmup_steps": 0, "resolution": 512,
      "guidance_scale": 3.0, "concept_scale": 3.0, "gamma1_1": 0.1, "gamma1_2": 0.01,
      "gamma2_1": 0.1, "gamma2_2": 0.1, "gamma2_3": 0.01, "seed": 0, "save_every": 500,
      "eval_every": 500, "max_checkpoints": 5, "exp_name": "r2_esd_meta_church_legacy",
      "fix_timesteps": false, "fixed_time_steps": [1, 2, 5, 10],
      "gradient_accumulation_steps": 1, "use_fp16": false}
    fix_timesteps -> False (by omission). devices -> [0, 0]
[stage 4: all six splits OK, see A-3]
    config record -> upstream_runs/meta/out/r2_esd_meta_church_legacy_config.json
    CLAIM SCOPE: PROMPT-SPECIFIC TOKEN SUPPRESSION: ... Do NOT report results from this scheme
                 as 'church concept erasure'.
R2 CPU VERIFICATION COMPLETE -- stages 1-4 passed, no GPU was requested.
```

---

## PART B — IMMA FIDELITY

Upstream pin verified: `git -C upstream/IMMA rev-parse HEAD` =
`2f55041141705031d503134dec22f674f09e21bf`, tree clean.

### B-1 The completed run is preserved under a label recording its 5× budget — DONE

`upstream_runs/imma/REF_imma_vangogh/` is untouched. `run_r1b_imma_train.sh` stage 0 writes (CPU,
idempotent) `upstream_runs/imma/REF_imma_vangogh/BUDGET_5X.md`, titled
*"HISTORICAL ARTIFACT — 5× THE DOCUMENTED IMMUNISATION BUDGET"*, carrying the budget table, the
log evidence, the blank-image caveat and the revision caveat, and ending **DO NOT DELETE**.
The script refuses to write into that directory (`die` if the output name resolves to it).

Naming now encodes the budget: `IMMA_MAX_SAMPLES=20` (default) → `REF_imma_vangogh_s20`;
`IMMA_MAX_SAMPLES=all` → `REF_imma_vangogh_sall_5x` (the tag was `sall`, now `sall_5x`).
`run_r1c_eval.sh` defaults to the faithful `REF_imma_vangogh_s20`; selecting the historical arm
requires `R1C_ACK_5X=1` and tags every output `_5x`:

```
$ R1C_ARM=REF_imma_vangogh bash run_r1c_eval.sh
REFUSING: arm REF_imma_vangogh received 5x the documented immunisation budget
  (5000 steps / 2500 ascent updates vs the README's 1000 / 500). It is not a faithful
  reference run. See upstream_runs/imma/REF_imma_vangogh/BUDGET_5X.md. Set R1C_ACK_5X=1 ...
```

Historical artifact hashes, recorded:

| file | sha256 | bytes | contents |
|---|---|---|---|
| `REF_imma_vangogh/imma_unet_xatten_layer.pt` | `8ad7d87ad8668c65a2d1c96d07de04d17d4b5df8c3ada208f33109842d2b7a4a` | 374,210,382 | 160 fp32 tensors, 93,536,640 params, 80 attn1 + 80 attn2 |
| `adapters/REF_imma_vangogh/unet.pt` | `0614c404ce26fc568b4b0e5c4cabd03bdaf781c28acbb8ca88f1809ef10d0354` | 1,719,220,288 | 686 fp16 tensors, 859,520,964 params |

**Provenance caveat on the second one:** `logs/r1c_eval.log:13` does not match the current
`r1_eval_imma.py` print strings, so `adapters/REF_imma_vangogh/unet.pt` was produced by the
**pre-audit fuzzy-matching** version of the composer, not the strict one now on disk. Recorded in
the script header; I did not re-verify the earlier offline re-audit.

### B-2 Budget verified and documented on both sides — DONE

Derivation from source (`train/defend_text_to_image_lora.py`): `:697-699` subset selection,
`:710-716` dataloader length, `:720,:750` updates/epoch, `:722,:752` total steps, `:863,:945`
inner/outer alternation, `:950` LoRA step (descent, lr 1e-4), `:954-960` IMMA step on
`backward(-loss)` (**ascent**, lr 1e-5), `:563-567` scope, `:977` save.
With `bs=1`, `grad_accum=1`, `inner=outer=1`, even `N_subset`:
`total = epochs·N_subset`, `IMMA_ascent = epochs·N_subset/2`.

`run_r1b_imma_train.sh` stage 2 now **recomputes this from the real dataset on every run** and
writes `upstream_runs/imma/out/<arm>_budget.json`. Executed output:

```
$ R1_MODE=cpu bash run_r1b_imma_train.sh                     # the default, README-faithful
    OURS   : shuffle(seed=42).select([20,40)) -- the SECOND block
             n_subset=20 steps/epoch=20 total=1000 IMMA_ascent=500 LoRA=500
    README : n_subset=20 total=1000 IMMA_ascent=500
    RATIO (ours / README, on ASCENT updates): 1.00x

$ IMMA_MAX_SAMPLES=all R1_MODE=cpu bash run_r1b_imma_train.sh  # the historical setting
    OURS   : no subsetting (--max_train_samples absent)
             n_subset=100 steps/epoch=100 total=5000 IMMA_ascent=2500 LoRA=2500
    RATIO (ours / README, on ASCENT updates): 5.00x
```

| | samples | batch | epochs | inner/outer | steps/epoch | total steps | **IMMA ascent updates** |
|---|---|---|---|---|---|---|---|
| README (L50-66) | 20 | 1 | 50 | 1/1 | 20 | 1000 | **500** |
| `REF_imma_vangogh` | 100 | 1 | 50 | 1/1 | 100 | 5000 | **2500** |
| ratio | 5× | — | — | — | 5× | 5× | **5.00×** |

Independent corroboration, `logs/r1b_imma_train.log:30-35`:
`Num examples = 100 | Num Epochs = 50 | ... | Total optimization steps = 5000`. The literal argv
was never logged; `Num examples = 100` is the proof that `--max_train_samples` was absent. Run
wall time 11:59:23 → 12:21:06 = **21 m 43 s for 5000 steps ≈ 0.26 s/step** (basis for B-7).

The inner loop is **not** nested: with `inner=outer=1` it is a flat 1:1 alternation over `step`
within each epoch, each phase stepping its own optimizer. `step` resets per epoch and both 20 and
100 are even, so the phase never drifts.

Everything else in the historical run was README-faithful (lrs, scheduler, warmup, batch size,
resolution, seed, loop ratio). Only the sample count — hence the step budget — diverged.

### B-3 Blank-image filtering — CONFIRMED DEFECT (the filter never ran), now repaired

Exhaustively verified: **exactly three** of the 100 images are pure black (`max == 0`): `7.png`,
`29.png`, `41.png`. No borderline cases — the three lowest-std non-blank images are
`14.png (std 41.6)`, `60.png (45.3)`, `32.png (47.5)`. Root cause: upstream `eval/text2image.py`
builds its pipeline without `safety_checker=None`, so flagged samples are blanked.

**The repair in the previous version of `run_r1_imma_official.sh` never executed.** The whole
filter block (old L69-88) was nested inside `if [ ! -f "$R/data/vangogh/metadata.csv" ]` (old L61);
`metadata.csv` already existed, so the block was skipped, no `rejected_blank=` line appears in any
log, and all three blanks are still rows in `metadata.csv`. `run_r1b_imma_train.sh`'s header
claimed they had been excluded — **that claim was false and is corrected in both files.**
`train/defend_text_to_image_lora.py` filters nothing itself: no pixel/variance/NSFW check exists;
the only preprocessing is the geometric chain at `:681-695`.

The audit now **always runs** and is idempotent. It deliberately does **not** rewrite
`metadata.csv` — that is the input the historical artifact was produced from, and overwriting it
would make that artifact unreproducible. It writes alongside:
`blank_audit.json` (per-image max/std + the blank list + root cause) and
`metadata_noblank.csv` (97 rows, opt-in). Executed:

```
    on disk=100 blank(max==0)=3 ['29.png', '41.png', '7.png'] usable=97
    lowest-std non-blank: 14.png(std=41.6), 60.png(std=45.3), 32.png(std=47.5)
    metadata.csv: 100 rows, 100 distinct, prompts=['An artwork by Van Gogh']
    blank images PRESENT in metadata.csv: ['29.png', '41.png', '7.png']
    NOT rewriting metadata.csv: it is the input REF_imma_vangogh was produced from.
    -> upstream_runs/imma/data/vangogh/metadata_noblank.csv (97 rows)
    -> upstream_runs/imma/data/vangogh/blank_audit.json
```

**Does a blank image fall inside the `--max_train_samples 20` subset? NO.** And the selection is
**not** "the first N". `defend_text_to_image_lora.py:698-699`:

```python
dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples, 2*args.max_train_samples)) \
    if dataset["train"].shape[0] >= 2*args.max_train_samples else \
    dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
```

A seeded shuffle, then the **second** block `[N, 2N)` whenever the dataset has ≥ 2N rows. Resolved
concretely against the real 100-row dataset with `seed=42` (reproduced independently twice — by
the audit subagent and by my own stage 2):

```
95 43 35 82 16 61 81 33 79 80 44 0 88 51 19 36 73 50 25 45   (.png)
blank images inside the selection: NONE
```

Near-miss worth recording: `29.png` is at shuffled position 1, i.e. inside the **discarded**
`[0,20)` window. So "the first 20" *would* have contained a blank one, and so would the `< 2N`
fallback branch. For the historical `all` run the audit correctly reports all three inside the
selection (3 × 50 = 150 of its 5000 batches were pure black).

### B-4 Initial checkpoint identity, base revision/variant, composition — DONE

Two *different* initialisations are in play, and `scripts/r1_eval_imma.py` now records both under
`initial_checkpoint_identity`:

* **TRAINING** (`defend_text_to_image_lora.py:439-454`) loads with `revision=args.revision` and
  **no variant — the script has no `variant` argument at all** — i.e. the fp32 `main` weights. The
  historical run passed no `--revision` either (default `None` → `main`); it matched the pinned
  revision only because the local cache holds exactly one snapshot. `run_r1b_imma_train.sh` now
  passes `--revision` explicitly; the absent variant is a property of upstream, recorded not
  worked around. The UNet is then fully overwritten by the ESD checkpoint at `:452-454`, so only
  text_encoder/vae/scheduler actually come from the base.
* **COMPOSITION** (`r1_eval_imma.py:74-76`) references the base with `variant="fp16"`.

**For the composed UNet this is immaterial, and the script proves it rather than asserting it.**
The ESD checkpoint is a complete 686-tensor UNet overwriting every key (`:93`), and IMMA's 160
tensors overwrite a subset (`:115-120`); the `unexplained` check (`:132-140`) requires every output
tensor to be bit-identical (`rtol=0, atol=0`) to its ESD or IMMA source. The base contributes
nothing to the model.

**But it is material to the reported deltas**, which is the section-0 trap again:
`rel_delta_vs_base` was measured against the fp16-variant base and silently inherits a 2.07e-04
floor. Repaired: deltas are now reported against **both** base forms with the floor stated, all
written to `compose.json` (`rel_delta_vs_base_fp16variant`, `rel_delta_vs_base_main`,
`fp16_variant_floor` with per-class breakdown). CPU-tested on the historical checkpoint:

```
[r1eval] base forms: fp16-variant vs main differ in 686/686 tensors at rel=2.074e-04
         -- deltas at this magnitude are dtype, not model
[r1eval] ESD loaded: 686 tensors, 859.521M params
[r1eval] IMMA overlaid: 160/160 tensors, exact keys, exact shapes, 0 fuzzy, 0 reshaped, 0 skipped
floor per class  : {'attn1': 2.0737e-04, 'attn2': 2.0737e-04, 'other': 2.0731e-04}
delta vs variant : {'attn1': 0.032822, 'attn2': 0.040461, 'other': 0.003549}
delta vs main    : {'attn1': 0.032822, 'attn2': 0.040461, 'other': 0.003546}
```

The deltas are 17× (other) to 195× (attn2) the floor, and the two base choices agree to the fourth
decimal — so the historical composition's conclusions stand, but they are now *checkable*.

Scope naming, confirmed: the artifact is `imma_unet_xatten_layer.pt`, but `:563-567` is a plain
**substring** match of `to_q/to_k/to_v/to_out` over all `named_parameters()`, so it catches
self-attention too — 160 tensors = **80 attn1 (SELF) + 80 attn2 (CROSS)**. "xatten" is a misnomer;
half the immunised scope is self-attention. Asserted in both runners.

### B-5 Framing: IMMA's claim vs our additional measurement — DONE

`run_r1c_eval.sh` and the generated `QUARANTINE.md` now state this explicitly and separately:

> IMMA's claim is **resistance to RE-LEARNING**, and the authors evaluate **POST-ADAPTATION**.
> README L16: *"After IMMA training, the immunized model functions as the new pre-trained model for
> further fine-tuning adaptation."* The evaluation at README L146-151 compares `--base_dir`
> (generation **without** IMMA) against `--imma_dir` (generation **with** IMMA) on CLIP/DINO/LPIPS,
> and **both** directories are produced *after* an adaptation step (LoRA re-learning, README
> L68-86; DreamBooth, README L121-142). There is no README command that generates from an
> immunised checkpoint without first adapting it. **IMMA never evaluates an immunised model
> standalone.**
>
> Our standalone utility evaluation of an immunised model is therefore **a measurement we added**,
> on an axis the authors do not report. It is not an IMMA result, it reproduces nothing from the
> paper, and it can neither confirm nor refute IMMA's claim.

The mandatory ESD-only control (`adapters/ESD_vangogh_official/`) still runs first, and the
label-collapse detector is unchanged (`outputs/gen_REF_imma_vangogh.json`: target 0.867 with eight
of 11 non-targets at exactly 0.000 — a closed-set 12-way CLIP argmax artifact, `gen_eval.py` L48/81/127).

---

## PART C — ESD PROVENANCE (uncertainty preserved)

File: `upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt`

### C-1 Facts, with hashes

* **sha256 `8881d70b1bd376c5532e933f60c5a78fc76eb63e153edfbfaaf6d649106b167e`**, 3,438,307,109 bytes
  (independently reproduced twice). Recorded to `upstream_runs/imma/esd_identity.json` by
  `run_r1_imma_official.sh` stage 1.
* Bare `dict`, **686 keys, all `torch.float32`**, 859,520,964 params. **No wrapper keys**
  (`state_dict`/`ema`/`model_ema`/`model`/`module` all absent); no key contains "ema".
* Key set **exactly equal** to the SD-1.4 UNet, **0 shape mismatches**, 0 keys added or dropped.
* Obtaining it was correct and documented: `upstream/IMMA/README.md:44` supplies
  `wget https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt`.
  README:43 itself calls it *"Only for illustration purposes."*
* Upstream consumes it as a **complete UNet**, strict load, despite the `delta` in its argument
  name: `defend_text_to_image_lora.py:452-454` `unet.load_state_dict(torch.load(args.delta_ckpt))`
  (same at `train/train_text_to_image_lora.py:454-456`, `eval/text2image.py:59-61`).
* Base files for reference: `unet/diffusion_pytorch_model.safetensors` sha256
  `145a07e0f05ec5bbe6e2e9faf608bdb311caf708895cac8c8ed713c59864e1e8`;
  `unet/diffusion_pytorch_model.fp16.safetensors` sha256
  `a35404d03ec8f977715a4d2a080ddf72e2144f2ee49bb1ee213258bc64f9cc87`.

Deltas vs the pinned fp32 base, `‖Δ‖/‖w‖` per tensor:

| group | n | differing | median | min | max |
|---|---|---|---|---|---|
| attn2 (cross) | 80 | 80 | **2.770e-02** | 4.101e-03 | 1.674e-01 |
| non-attn2 | 606 | 606 | **4.094e-03** | 3.450e-04 | 2.024e-01 |

Global `‖Δ‖/‖w‖ = 2.797e-02`. Largest single deltas are **non**-attn2:
`up_blocks.0.resnets.2.conv_shortcut.weight` 0.2024 and `time_embedding.linear_2.weight` 0.1695,
both above the largest attn2 delta (0.1674).

### C-2 What the evidence RULES OUT

The brief's premise — that a variant difference, an EMA/non-EMA difference or a conversion could
make every tensor differ while the erasure stayed cross-attention-only — is **partly refuted by
measurement**. Three of the four candidates are now dead:

1. **fp16-variant vs main weights — RULED OUT.** The fp16 variant is bit-exactly `main.half()` for
   686/686 tensors, so it is not an independent base at all. Upcasting it changes the non-attn2
   median by < 0.2% (4.0935e-03 → 4.0873e-03). **0/686** tensors become identical either way.
2. **Dtype round-tripping — RULED OUT, but note which statistic carries the argument.**
   Measured floor (`main` vs `main.half().float()`): median 2.073e-04, max 3.071e-04.
   * The **median** argument is strong: non-attn2's median (4.094e-03) is **19.7×** the floor
     median, and 0/686 tensors become bitwise identical under either fp16 cast.
   * The **margin** argument is thin and is NOT what settles it: non-attn2's *minimum*
     (3.450e-04) exceeds the floor's *maximum* by only **1.12×**, and the floor median by
     **1.66×**. For the smallest handful of non-attn2 tensors, magnitude alone does not separate
     a real update from rounding, and I do not claim it does.
   * What actually settles it is **structural, not a margin**: per-element relative errors reach
     **2.55e+04** against an fp16 half-ulp bound of 4.88e-04 (bf16: 1.95e-03) — four to seven
     orders past any float format; and the delta is **absolute-scale, not proportional to weight
     magnitude** (`rms(Δ)` spans 6.4e-05…1.7e-03 while `rms(base)` spans 2.8e-03…8.3e-01, a 300×
     spread in weight scale against ~27× in delta scale). Rounding error is proportional by
     construction; this is not. That holds per tensor, independent of the aggregate margin.
3. **Format conversion — RULED OUT.** Pure diffusers naming (`down_blocks` 246, `up_blocks` 384,
   `mid_block` 46, `transformer_blocks` 320); **zero** CompVis/LDM markers
   (`model.diffusion_model`, `input_blocks`, `output_blocks`, `middle_block` all 0); key sets
   identical, no shape mismatches. No conversion residue of any kind.
4. **A different major revision — RULED OUT.** vs SD v1-5: attn2 median 2.759e-01, non-attn2
   1.023e-01 — two orders worse. The file is unambiguously an SD-1.4 derivative.

**A positive criterion now exists.** Our own ESD-u checkpoint (`scripts/erase_esd.py:72-75` freezes
cross-attention via `not ("attn2" in n)`) shows its frozen group at **2.0737e-04** — matching the
dtype floor to five significant figures. A genuinely frozen parameter group reproduces the floor
*exactly*, not merely "small". The target has **neither** group at the floor.

### C-3 What remains HYPOTHESIS

**H-A: the released artifact was trained with a scope wider than cross-attention**, the `x` in
`ESDx1` notwithstanding. Supporting: non-attn2 median (4.09e-03) is the same order as our genuine
full-scope ESD-u update (2.60e-03); both groups saturate the same `lr × steps ≈ 1e-2` Adam ceiling
(attn2 max 9.30e-03, non-attn2 max 1.12e-02) expected from ESD's published lr 1e-5 × 1000 iters;
`max/rms(Δ)` of 2.75–10.5 is optimizer-shaped, not the ≈1.7 of bounded uniform rounding.
Weakness: the 6.8× attn2:non-attn2 concentration is unexplained by a uniform-scope run, and the
ceiling argument assumes ESD's defaults.

**H-B: cross-attention-only, but trained from a different SD-1.4 instantiation** — EMA vs non-EMA
the leading candidate. Under this story non-attn2 Δ is a base-to-base offset and attn2 Δ is that
offset plus the erasure; it explains the 6.8× concentration naturally, and SD was trained with
AdamW so a base-to-base offset is also optimizer-shaped. **UNTESTED:** the non-EMA SD-1.4 weights
are not in the local cache, no `*ema*` file exists anywhere under `hf_home`, and the host is
offline. This is the single missing measurement.

**Not evidence either way:** norm comparison. Global norm ratio 0.99996470 against a 2.8e-02
relative difference, attn2 norm *grew* (1.000478), and `cos(Δ, w)` is small and sign-inconsistent
across sampled tensors (−0.611 … +0.203). This weakly argues against gross rescaling; it does not
refute EMA. Labelled weak.

### C-4 Standing language, and the decisive experiment

Do **not** write "a cross-attention-only ESD erasure", and do **not** write "mislabelled" or
"trained across the whole UNet". The supportable description is: *the published baulab
VanGogh-ESDx1 artifact, whose deltas against our pinned SD-1.4 cover the whole UNet, with a
provenance that is unresolved between H-A and H-B.* Verdict status: **OPEN HYPOTHESIS** (register
D-019 stays open, with three of its four benign explanations now eliminated).

Either experiment closes it; the second is cheaper and needs no base model:

1. Fetch the non-EMA / full-EMA SD-1.4 weights and re-diff. Under H-B the 606 non-attn2 tensors
   become bitwise identical, or land at the 2.07e-04 floor.
2. **Download a second `ESDx1` checkpoint for a different artist** from the same
   `erasing.baulab.info/weights/esd_models/art/` directory and diff the two *released files against
   each other*. Under H-B their 606 non-attn2 tensors must be **bitwise identical** (same frozen
   base, different concept); under H-A they will differ. Needs network, no GPU, ~3.5 GB.

**Caveat, recorded:** no sha256 was captured at download time. `8881d70b…` pins the bytes we hold;
it cannot prove they match what the URL served on 2026-10-26.

---

## PART D — `scripts/gpu_reserve.sh` (NOT mine; reported, not edited)

1. **`gpu_reserve_hold` lost the freeze override when the flag was passed second — FOUND, and
   independently FIXED by the Infrastructure agent mid-session.** The v1 signature was
   `<campaign> [wait_seconds] [--allow-frozen]`, so `gpu_reserve_hold "camp" --allow-frozen` bound
   the flag to `wait_seconds`, `extra` stayed empty, the override never reached `acquire`, and the
   call failed `rc=4` under the freeze. `run_v1_esd_alone_control.sh:31` used exactly that form.
   Had the freeze been lifted, the retry path would have run `sleep --allow-frozen`. The current
   version parses flags position-free and logs the odd call shape out loud. **Verified after their
   fix:** both `("camp" 120 --allow-frozen)` and `("camp" --allow-frozen)` now return rc=0 and
   grant a reservation; my four runners use the explicit 3-argument form and are compatible with
   either version.
2. **`GPU_RESERVE_GPUS=""` cannot express "no GPUs".** `ALLOWED_GPUS="${GPU_RESERVE_GPUS:-0 1}"`
   uses `:-`, which substitutes the default for an *empty* value too, so an empty allow-list
   silently becomes `0 1`. I discovered this by having it reserve GPU 0 during a probe I intended
   to be a no-op. Suggest `${GPU_RESERVE_GPUS-0 1}`. Minor; test-harness ergonomics only.
3. **A two-GPU reservation is not expressible.** `acquire` grants one index per call and
   `gpu_reserve_hold` sets a single `$GPU_RESERVED` and installs one `EXIT` trap, which a second
   call would overwrite. This matters here: `esd_meta.py:828-830` is written for **two** devices
   (`devices[0]` student, `devices[1]` teacher+text_encoder+vae) and `--devices 0` raises
   `IndexError` at `:886`. See D-5 below.
4. **`gpu_reserve_hold` overwrites any `EXIT` trap the caller already installed.** None of my
   runners install one, so this is a note for other owners, not a defect for me.
5. **Memory risk on the R2 full run — the lead should know before spending time.** Our GPUs are
   **A100-SXM4-40GB** ×4 (0 and 1 allowed); the Meta-Unlearning README:15 cites **80 GB** A100s,
   and both upstream scripts pass `--devices 0 0`, co-locating teacher and student on one device.
   With `--finetuning_method full` at fp32 the resident set is roughly: student 3.44 GB + grads
   3.44 + Adam states 6.88, teacher 3.44, a `deepcopy` task model 3.44 with its own full-parameter
   optimizer ≈ 10.3, plus `sum_grads` 3.44 — **≈ 34 GB before activations**, and `--train_batch_size
   10` at 512² on top. **I expect the first `R2_MODE=smoke` attempt to OOM at batch 10.** Hence
   the recommended smoke uses `META_BATCH=1`. If the full run cannot fit, the options are a smaller
   batch (changes the TIER R value — record it) or two reserved GPUs (needs item 3 resolved).

**Transparency:** my CPU probes of the reservation contract appear in
`logs/frozen_override.log` as campaigns `repair_probe_t4`, `repair_probe_t17`, `repair_probe_t18`,
`probe_t19b`. Each took GPU 0 for under a second and was released by the trap; `.gpulocks/` holds
only `.reserve.mutex`. **No GPU compute was launched.**

---

## PART E — BOUNDED GPU COMMANDS FOR THE LEAD

All require the documented override (`ALLOW_FROZEN_RUN=1` plus the flag the runner forwards), and
each is logged to `logs/frozen_override.log`. Run from `/data/bijaypandey/t2i_unlearn`.

### E-1 R2 minimal real update — **1 GPU, ~10 min, run this first**

```bash
ALLOW_FROZEN_RUN=1 META_POOL_SCHEME=legacy META_HISTORICAL_RECONSTRUCTION=1 \
  META_SIZING=total_per_pool META_TOTAL_PER_POOL=30 \
  META_BATCH=1 META_SMOKE_STEPS=2 META_NIPP=1 \
  R2_MODE=smoke bash run_r2_meta_official.sh
```

Reuses the existing validated pools, so no generation. Executes stages 1–10: argparse, dataset
path resolution, 2 real training steps with `save_every=1`, the `step=*/` checkpoint reload against
the **same** initialisation with the variant-floor guard, and the evaluator-compatibility export.
`META_BATCH=1` is a deliberate deviation from TIER R (10) to survive 40 GB; it is recorded in
`config.json`. Expect ~3–5 min of model loading, then seconds per step.
Outputs: `upstream_runs/meta/out/r2_esd_meta_church_legacy_smoke_<timestamp>/step=00000N/`
(+ `reload_check.json`), `adapters/R2_r2_esd_meta_church_legacy_smoke/unet.pt`,
`upstream_runs/meta/out/r2_esd_meta_church_legacy_smoke_config.json`, log `logs/r2_meta.log`.
**Read `reload_check.json` before anything else:** `rel_delta_vs_same_init` must exceed
`variant_only_floor` (~2.07e-04) by >10×, and `rel_delta_vs_other_variant` is calibration only.

### E-2 R2 full run — **1 GPU, duration MUST be measured from E-1, not guessed**

```bash
ALLOW_FROZEN_RUN=1 META_POOL_SCHEME=legacy META_HISTORICAL_RECONSTRUCTION=1 \
  META_SIZING=total_per_pool META_TOTAL_PER_POOL=30 META_BATCH=<as survived in E-1> \
  R2_MODE=full bash run_r2_meta_official.sh
```

I am deliberately not giving a wall-clock figure. Each of the 1500 steps runs the ESD unlearn step
**plus** five `train_step` calls that each iterate a whole 30-image dataloader (`esd_meta.py:981-1007`
and :1012-1044), i.e. ~15–18 batch iterations per step. Multiply the per-step time E-1 reports by
1500 and decide then; on the smoke evidence this is plausibly multi-hour. Outputs: as E-1 but under
`r2_esd_meta_church_legacy_<timestamp>/` with a full pipeline save (`unet/`, `vae/`,
`text_encoder/`, `model_index.json`) plus up to 5 `step=*/` dirs.

### E-3 R2 coherent pools — **1 GPU, ~40–60 min** (only for a run meant to support erasure)

```bash
ALLOW_FROZEN_RUN=1 META_POOL_SCHEME=coherent META_SIZING=per_prompt META_PER_PROMPT=30 \
  R2_MODE=smoke bash run_r2_meta_official.sh
```

Builds to its own root `upstream_runs/meta/dataset_coherent` (the legacy pools are untouched):
18 prompts × 30 × 2 splits = **1080 images** at 512² with blank rejection and retries, then the E-1
checks. Outputs: `upstream_runs/meta/dataset_coherent/{hrm,fgen,tgt,irt}/{train,test}/` +
`pools.json` with fingerprints. `fgen` (cathedral/chapel/basilica, scored as **forget**) is an
evaluation pool: `esd_meta.py` reads only `hrm/irt/tgt`.

### E-4 IMMA README-faithful training — **1 GPU, ~6 min**

```bash
ALLOW_FROZEN_RUN=1 IMMA_MAX_SAMPLES=20 R1_MODE=train bash run_r1b_imma_train.sh
```

1000 steps / 500 ascent updates. Duration from the historical run's own rate (0.26 s/step over
5000 steps) ⇒ ≈ 4 m 20 s compute + ~1 min load. The dataset already exists, so no generation.
Outputs: `upstream_runs/imma/REF_imma_vangogh_s20/imma_unet_xatten_layer.pt`,
`upstream_runs/imma/out/REF_imma_vangogh_s20_budget.json` (budget + artifact sha256 + rel delta vs
the ESD init), log `logs/r1b_imma_train.log`. It refuses to overwrite an existing arm, and
`REF_imma_vangogh/` cannot be a target.

### E-5 IMMA evaluation, both arms — **1 GPU, ~25–35 min**

```bash
ALLOW_FROZEN_RUN=1 R1C_ARM=REF_imma_vangogh_s20 bash run_r1c_eval.sh
```

ESD-only control first, then the immunised arm, then (gated on `legacy_retain8 >= 0.60`) three
relearning attacks at seeds 0/1/6, lr 1e-5, 300 steps. Each `gen_eval.py` arm is 12 concepts × 3
prompts × 5 images = 180 images at 25 steps (~3–4 min); the three attacks dominate the rest.
Outputs: `adapters/ESD_vangogh_official/{unet.pt,compose.json}`,
`adapters/REF_imma_vangogh_s20/{unet.pt,compose.json,QUARANTINE.md}`,
`outputs/gen_ESD_vangogh_official.json`, `outputs/gen_REF_imma_vangogh_s20.json`,
`outputs/atk_R1C_REF_imma_vangogh_s20_sd{0,1,6}_lr1e-5_s300.json`, log `logs/r1c_eval.log`.
Read `compose.json`'s `fp16_variant_floor` before interpreting any per-class delta.

### E-6 The queued D-019 behavioural control — **1 GPU, ~5 min**

`adapters/ESD_vangogh_official/` + `outputs/gen_ESD_vangogh_official.json` are produced as step 1
of E-5, so E-6 needs no separate launch. Its interpretation limit stands unchanged: a behavioural
control shows what the uncomposed checkpoint *does*; it establishes nothing about provenance, and
under D-030 a closed-set argmax with no reject option cannot separate erasure from degradation.

---

## Status summary

| item | status |
|---|---|
| A-1 reservation API replaces `campaign_free.sh` | **DONE**, CPU-verified |
| A-2 explicit forget/retain definition required; legacy gated | **DONE**, CPU-verified |
| A-3 pool content / counts / split roles / fingerprints / resume | **DONE**, CPU-verified on the real pools |
| A-4 three hyperparameter tiers; gamma2 inference labelled | **DONE** |
| A-5 upstream variant recorded; meta.py defects verified from source | **DONE**, runtime-asserted |
| A-6 model pinned; reload trap closed | **DONE** — the old guard would have passed a no-op |
| A-7 wrong venv (`datasets` missing) | **NEW BLOCKING DEFECT found and fixed** |
| A-8 CPU-verifiable path executes or fails clearly | **DONE**; 3 of 6 checks need GPU (E-1) |
| B-1 5× artifact preserved and labelled | **DONE**, `BUDGET_5X.md` written |
| B-2 budget resolved both sides (500 vs 2500 ascent, 5.00×) | **DONE**, recomputed per run |
| B-3 blank filtering | **CONFIRMED DEFECT: the filter never ran**; repaired. No blank in the 20-subset |
| B-4 checkpoint identity, revision/variant, composition | **DONE**; delta floor now reported |
| B-5 claim-vs-our-measurement framing | **DONE** |
| C ESD provenance | **OPEN HYPOTHESIS**; 4 benign explanations ruled out, EMA survives |
| D `gpu_reserve.sh` findings | **REPORTED, not edited**; item 1 fixed by its owner |

---

# REVIEW ROUND 2 — five gate blockers, all fixed and re-tested

An independent reviewer re-derived this work and found five blockers. All five are fixed. Every fix
below was executed on CPU; no GPU work was launched.

## F-1 (BLOCKING 1) The reload guard had a guaranteed FALSE NEGATIVE — FIXED

**Confirmed, exactly as reported.** Two versions of this guard failed in **opposite** directions:

| version | rule | outcome |
|---|---|---|
| v1 | `rel > 1e-6` vs the **fp16-variant** base | **passed a no-op** (a variant mismatch scores 2.074e-04 = 207× the threshold) |
| v2 | `rel > 10 × variant_floor` = **2.074e-03** | **failed genuine training** — 4.25× above the largest delta 2 steps can produce |

The reviewer's bound reproduces exactly. AdamW (`esd_meta.py:867 optim.AdamW`) has per-element step
bounded by ~`lr` (`m̂/(√v̂+ε) ∈ [-1,1]`, and `±1` exactly on the first bias-corrected step), so after
`S` updates `rel ≤ lr·S·√N/‖w‖`. Measured from this UNet: `N = 859,520,964`, `‖w‖ = 1201.5273`:

```
bound(S=2)    = 4.880053e-04      <- the most 2 steps at lr=1e-5 can do
bound(S=1500) = 3.660040e-01
v2 threshold 10*floor = 2.074001e-03   <-- 4.25x the S=2 bound
```

So `R2_MODE=smoke` — the only mode whose purpose is to prove the trainer **trains** rather than
merely **runs** — could never have passed.

**The fix is not another constant.** One number was being asked two questions; they are now asked
separately, and the magnitudes are **derived from the configured `lr` and steps**, not picked:

* **CHECK A — "is this above numerical noise?"** `rel > 1e-7` and not every tensor bitwise-identical
  to the initialisation.
* **CHECK B — "is my reference the actual initialisation?"** **Magnitude-free**: the checkpoint must
  be closer to the init we claim than to the other variant. For a true update of size `d` from
  `base_A`, `rel_vs_A = d` while `rel_vs_B = √(d² + floor²) > d`, so this holds for **any** `d`,
  including `d` far below the floor where no threshold could separate them. The variant floor is
  now **reported and never thresholded**.
* **CHECK C — "is this consistent with the configured amount of training?"** band
  `[bound(S=1)/100, 2·bound(S_configured)]`, `bound(S) = lr·S·√N/‖w‖`. The bound is independent of
  the gamma values: the gammas rescale `param.grad` and Adam normalises gradient scale away.

**Unit-tested against the real pinned base weights** (`scratchpad/baselines/guard_test.py`,
arithmetic copied verbatim from stage 9; 7 synthetic checkpoints, no GPU):

```
1. TRUE NO-OP, correct reference              rel_same=0.000000e+00 rel_other=2.074001e-04 id=686/686
                                              A=FAIL B=PASS C=FAIL => FAIL   (expected FAIL)
2. NO-OP vs the WRONG variant (the v1 defect) rel_same=2.074001e-04 rel_other=0.000000e+00 id=0/686
                                              A=PASS B=FAIL C=PASS => FAIL   (expected FAIL)
3. GENUINE 2-step saturating, |dw|=lr*2       rel_same=4.880007e-04 rel_other=5.302565e-04
                                              A=PASS B=PASS C=PASS => PASS   (expected PASS)  <-- v2 FAILED THIS
4. GENUINE 2-step at 10% saturation           rel_same=4.879542e-05  => PASS   (expected PASS)
5. 100x the saturating bound                  rel_same=4.880053e-02  => FAIL   (expected FAIL)
6. 1e-5 of a single update                    rel_same=6.560287e-10  => FAIL   (expected FAIL)
7. FULL RUN 1500 steps at 1% saturation       rel_same=3.660042e-03  => PASS   (expected PASS)
ALL GUARD CASES CORRECT
```

Case 3 is the reviewer's blocker: it now passes. Case 2 is the v1 defect, and it fails through the
**magnitude-free** CHECK B — which is why the fix cannot regress into either failure mode.

## F-2 (BLOCKING 2) `R2_MODE=cpu` reported a skipped stage as passed — FIXED

**Reproduced:** `META_POOL_SCHEME=bogus R2_MODE=cpu` → **rc=0** plus a config record.
Two independent causes, both fixed:

1. **An unknown scheme was never rejected.** It fell through to a non-existent pool root, so stage 4
   skipped. The scheme is now validated up front against the **builder's own** `SCHEMES` dict (asked
   of the builder, not hardcoded) → `rc=2` with `Known schemes: coherent legacy`.
2. **A skipped stage was reported as passed.** `STAGE_4_RAN` now tracks whether the pool contract was
   actually checked. If it was not, the script prints `R2 CPU VERIFICATION INCOMPLETE`, names the
   stage, and exits **3**. The success message no longer says "stages 1-4 passed" generically; it
   says which stages ran and which still need a GPU.
3. **No config record is written when the pools were not validated.** A record with `"pools": null`
   asserts nothing and reads like a pass. Validated records now also carry `"pools_validated": true`.

The two stale records produced by the defective path (`..._coherent_config.json` with `pools: null`,
and `..._bogus_config.json`) were **moved out of `upstream_runs/` into the scratchpad**. They were
generated by me earlier today through the buggy path and describe no run; no historical artifact was
touched.

```
  R2 cpu legacy (pools present+fingerprinted)  rc=0   COMPLETE -- stages 1,2,3,4 ALL RAN
  R2 cpu coherent (pools absent)               rc=3   INCOMPLETE -- STAGE 4 DID NOT RUN
  R2 cpu bogus scheme                          rc=2   REFUSING: not a scheme the builder knows
  R2 cpu no scheme                             rc=2   REFUSING: no default
  upstream_runs/meta/out/ now contains only r2_esd_meta_church_legacy_config.json
```

## F-3 (BLOCKING 3) `run_r1c_eval.sh` gated on output existence — FIXED

**The reviewer is right that this is structurally the bug I found myself** (the blank filter nested
inside `if [ ! -f metadata.csv ]`), and it was **live**: `adapters/REF_imma_vangogh/unet.pt` is on
disk with **no `compose.json` at all** — pre-audit, fuzzy-matched — and an existence gate would have
reused it silently.

All six cache gates are now **provenance-gated**. An artifact is reused only if its recorded
provenance matches the inputs about to be used:

* `compose_is_current <dir> <esd_sha> [imma_sha]` — requires `compose.json` to exist, to carry the
  REPAIR-v2 marker (`fp16_variant_floor`), to record exactly these input hashes, to have the right
  `mode`, and to show `imma_overlaid == imma_tensors`.
* `eval_is_current <label> <unet_sha>` — `gen_eval.py` records no input hash, so a sidecar
  `outputs/gen_<label>.provenance.json` records the sha256 of the UNet actually scored. Attacks are
  gated the same way.

Recorded input hashes (logged every run): ESD `8881d70b…`, IMMA `8ad7d87a…`.

**Quarantined, not deleted.** `adapters/REF_imma_vangogh/QUARANTINE_PREAUDIT.md` is now written,
marked *"produced by the PRE-AUDIT composer, provenance unverifiable"*, explaining that it is also
the composition of the 5× artifact and that the strict re-derivation lands in
`adapters/REF_imma_vangogh_5x/`. It is kept because `outputs/gen_REF_imma_vangogh.json` was scored
from it and that historical record must stay interpretable. The provenance helpers and the
quarantine were moved **ahead of the GPU reservation**, so they run on CPU even when the freeze
refuses a GPU — verified.

**Tested against the real functions** (extracted by line range, not re-typed):

```
=== compose gate ===
  [OK ] stale    PRE-AUDIT adapter, no compose.json (the LIVE case)
  [OK ] current  REPAIR-v2 composition, hashes match
  [OK ] stale    ESD hash differs
  [OK ] stale    IMMA hash differs
  [OK ] stale    esd_only expected, dir is esd_plus_imma
  [OK ] stale    correct hashes but NO v2 marker (old composer)
=== eval gate ===
  [OK ] stale    no gen_*.json at all
  [OK ] stale    historical gen_*.json exists but has NO provenance sidecar
  [OK ] current  after a sidecar is written with the matching sha
  [OK ] stale    sidecar present but the unet sha differs (adapter changed)
ALL GATE CASES CORRECT
```

## F-4 (BLOCKING 4) The collapse detector never fired on its own canonical case — FIXED

**Confirmed, including that my header's wrong number is what made the threshold look safe.**
The artifact has **seven** zero-accuracy concepts, not eight:

```
  concepts=12 non-target=11 target_acc=0.8667
  exact zeros: 7 total, 7 non-target ->
    ['church','english_springer','french_horn','garbage_truck','gas_pump','golf_ball','monet']
  OLD rule  zeros >= 0.6*12 = 7.2 -> needs >=8 -> DOES NOT FIRE   <-- the defect
  NEW rule  zeros_nontarget >= ceil(11/2) = 6 and target > 0.5 -> FIRES   <-- margin 1 concept
```

**Re-derivation.** Under collapse the target is *expected* to be non-zero, so the meaningful
population is the **non-target** concepts (11 of 12), not all 12. Collapse means the closed-set
argmax has stopped discriminating, so the signature is a **majority of non-targets at exactly
0.000** while the target stays high: `zeros_nontarget >= ceil(n_nontarget/2)` **and** `target > 0.5`.
The detector now also prints the threshold and the counts, and prints an explicit
"not present at the derived threshold" line when it does not fire. The "eight" claim is corrected in
`run_r1c_eval.sh` and in this document.

## F-5 (BLOCKING 5) Pool fingerprints were inert — FIXED

**Confirmed: present, cited, and incapable of firing.** `--verify-only` wrote nothing, the only
pools on disk carried no fingerprints, and a missing fingerprint was printed as a tolerated note.

Three changes make the check bind:

1. **`--backfill-fingerprints`** validates the pools and writes their fingerprints into the existing
   `pools.json`, leaving every other field verbatim. Regenerates no images, needs no GPU.
2. **A missing fingerprint is now an ERROR** in `--verify-only` (escape hatch:
   `--allow-missing-fingerprints`, which also prints `the check is NOT BINDING`).
3. **R2 stage 4 backfills automatically** right after building pools, so a newly built pool set can
   never be left unfingerprinted.

Executed — note the first command now *fails*, which is the point:

```
# before backfill: the check binds and REFUSES
[r2pools] FATAL: 6 split(s) carry no recorded fingerprint, so the drift check cannot bind and these
  pools are NOT provably the ones any earlier run used. Run `--backfill-fingerprints` ...   rc=1

# backfill
[r2pools] recorded fingerprints for 6 split(s) -> upstream_runs/meta/dataset/pools.json

# after backfill: binding and passing
  hrm/train ... fingerprint MATCHES manifest     (x6)
[r2pools] verify-only: OK

# and drift genuinely fires -- one recorded fingerprint corrupted, then restored
  hrm/train: sha=bdb3dfa5a482757c -- fingerprint DRIFT: manifest=0000000000000000
[r2pools] FATAL: 1 split(s) no longer match the recorded fingerprint   rc=1
```

Recorded fingerprints (now in `pools.json`, `fingerprints_backfilled_by` set):
`hrm/train bdb3dfa5a482757c…`, `hrm/test 9915bd23a326e2ca…`, `tgt/train 656fec31e1f04a02…`,
`tgt/test 6d0cd3694f9b4ff7…`, `irt/train 8db1efad8be17024…`, `irt/test 2b2212b638a1ae2c…`.

## Also-flagged items

**`args.max` bypasses the alternation — now drift-guarded.** `--max` (`:368`, *"Directly maximizing
model parameters as a vanilla way"*, default `None`) is tested at `:859` and `:936`, bypassing the
inner/outer alternation — every step would ascend and the budget arithmetic would be wrong. Two
assertions added to `run_r1b_imma_train.sh` stage 1: `--max` exists with `default=None`, and it still
gates exactly two branches. We never pass it. Both `[OK ]` in the re-run.

**`selected_files` ordering — now asserted, not assumed.** The subset resolution maps shuffled
dataset indices to filenames **by position**, so the imagefolder row order must equal the
`metadata.csv` row order; the no-blank conclusion depends on it. Stage 2 now reads the undecoded
image paths (`cast_column("image", HFImage(decode=False))`) and compares them element-by-element
against `metadata.csv`, exiting non-zero with the first divergence if they differ:

```
  ORDERING ASSERTED: imagefolder order == metadata.csv order over 100 rows
                     (first three ['0.png', '10.png', '11.png'])
```

The resolved subset is unchanged, so **the no-blank conclusion now rests on an assertion rather than
an assumption**.

**ESD floor rule-out — claim softened to match the statistic.** Section C-2 item 2 is rewritten to
separate three things: the **median** argument is strong (19.7× the floor, 0/686 identical under
either cast); the **margin** argument is thin and is explicitly disclaimed (non-attn2's minimum
3.450e-04 exceeds the floor's maximum by only **1.12×**, and the floor median by **1.66×**, so for
the smallest handful of tensors magnitude alone does *not* separate a real update from rounding);
and what actually settles it is **structural** — per-element relative errors to 2.55e+04 against an
fp16 half-ulp bound of 4.88e-04, and a delta that is absolute-scale rather than proportional to
weight magnitude, which holds per tensor independent of any aggregate margin.

## Round-2 regression

```
  syntax: 4 runners + 2 python modules            all OK
  R2 cpu legacy                                   rc=0  (COMPLETE, stages 1-4 all ran)
  R2 cpu coherent (pools absent)                  rc=3  (INCOMPLETE, stage 4 did not run)
  R2 cpu bogus scheme / no scheme                 rc=2 / rc=2
  R1 cpu (ESD identity + blank audit)             rc=0
  R1b cpu s20 / all                               rc=0 / rc=0  (ordering asserted, 1.00x / 5.00x)
  R1c 5x arm without acknowledgement              rc=2
  guard decision function, 7 cases                ALL CORRECT
  provenance gates, 10 cases                      ALL CORRECT
  collapse detector on its canonical case          FIRES (margin 1)
  pool fingerprints                                BOUND; drift fires; backfill recorded
  .gpulocks/                                       only .reserve.mutex -- no reservation held
```

## Could not fix, and why

* **C-3 / H-B (EMA vs non-EMA) remains untested.** It needs the non-EMA SD-1.4 weights, which are
  not in the local cache, and the host is offline. Unchanged: an **OPEN HYPOTHESIS**. The cheapest
  closing experiment is still to diff two released `ESDx1` artists against each other (network, no
  GPU, ~3.5 GB).
* **Stages 7, 9 and 10 of R2 have still never executed**, so the repaired guard is verified only by
  its unit test against synthetic checkpoints, not by a real trainer run. That is the E-1 command,
  and it is the right next step. I did not launch it.
* **`scripts/gpu_reserve.sh` items 2 and 3 of Part D** (empty `GPU_RESERVE_GPUS`; no two-GPU
  reservation) are not mine to edit and remain reported only. Item 3 still blocks the
  `--devices 0 1` option for the R2 memory problem.
