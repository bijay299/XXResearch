# Agent 2 — audit of the two official baseline reproductions (IMMA, Meta-Unlearning)

Scope: `run_r1_imma_official.sh`, `run_r1b_imma_train.sh`, `run_r1c_eval.sh`, `run_r2_meta_official.sh`,
`scripts/r1_eval_imma.py`, `scripts/r2_build_meta_pools.py`.
Method: read upstream source at the pinned commits; CPU tensor inspection; argparse replication.
No GPU work was launched (project FROZEN, `FREEZE_ACTIVE` 2026-10-26T12:43).

Upstream pins verified by `git log -1`:
* IMMA `2f55041141705031d503134dec22f674f09e21bf` (2025-01-17) — matches the stated pin.
* Meta-Unlearning `5a2a7600895a2885323cbbf599acaf3f74a96f64` (2025-04-22) — matches the stated pin.

Classification key: **(1) CONFIRMED DEFECT**, **(2) SUSPECTED**, **(3) VALID**,
**(4) HISTORICAL RESULT needing reinterpretation**, **(5) REPAIR PERFORMED**.

---

## A. IMMA

### A-1. Resolved arguments vs the README command — (1) CONFIRMED DEFECT (one material omission)

README (`upstream/IMMA/README.md` L50-66) against `run_r1b_imma_train.sh` L24-31.
Identical: `--mixed_precision=no`, `--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4`,
`--train_data_dir`, `--caption_column=prompt`, `--resolution=512`, `--random_flip`,
`--train_batch_size=1`, `--num_train_epochs=50`, `--learning_rate_lora=1e-04`,
`--learning_rate=1e-05`, `--lr_scheduler=constant`, `--lr_warmup_steps=0`, `--seed=42`,
`--output_dir`, `--delta_ckpt`.

| README arg | ours | resolved effect |
|---|---|---|
| `--max_train_samples 20` | **omitted** | default `None` → all 100 images used. **Material.** |
| `--inner_loop_steps=1` | omitted | parser default is `1` (L361) → identical. Immaterial. |
| `--outer_loop_steps=1` | omitted | parser default is `1` (L364) → identical. Immaterial. |
| `--report_to="tensorboard"` | omitted | parser default is `"tensorboard"` (L311) → identical. Immaterial. |

**Evidence of the material consequence**, `logs/r1b_imma_train.log`:
```
Num examples = 100 | Num Epochs = 50 | Total optimization steps = 5000
```
The README command resolves to 20 × 50 = **1000** steps. Ours ran **5000**. Because the loop
alternates inner/outer 1:1, our run performed **2500 gradient-ASCENT steps on the UNet attention
weights where the documented command performs 500** — a 5× over-budget immunisation. Every
artifact under `upstream_runs/imma/REF_imma_vangogh/` was produced at 5×.
Also: our runner passes no `--revision`/`--variant`, so the defend script loads SD-1.4 `main`
(fp32 `diffusion_pytorch_model.safetensors`, downloaded 2026-10-26 11:39), whereas the rest of the
project pins `revision=133a221b...` with `variant="fp16"`. Same commit, different weight file.

### A-2. The ESD initial checkpoint — (2) SUSPECTED (identity does not match its name)

`upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt`: a bare `state_dict`, 686 tensors,
859,520,964 params, all fp32. Key set is **exactly equal** to the SD-1.4 UNet key set. Target
concept is **Van Gogh**, which is the concept we train on (`An artwork by Van Gogh`) and the
concept we evaluate (`--target van_gogh`) → **concept-matched (3) VALID**.

But the erasure scope contradicts the "ESDx1" name. ESD-**x** trains cross-attention only, so
`attn1` and all non-attention tensors should be bit-identical to the base. Measured
`‖Δ‖/‖w‖` against pinned SD-1.4:

```
all 686 tensors differ (none bit-identical)
attn2 (cross): n=80  median 2.76e-2  max 1.67e-1
attn1 (self) : n=80  median 1.28e-2  max 3.21e-2
other        : n=526 median 3.54e-3  max 2.02e-1   (up_blocks.0.resnets.2.conv_shortcut.weight)
```
0.20 relative on a `conv_shortcut` is ~400× the fp16 round-trip floor (≈4.9e-4), so this is not a
precision artifact. Two candidate explanations, both consistent with the data:
(a) the published artifact was produced by a **full-parameter** run despite the `x` in its name; or
(b) it was trained from a **different SD-1.4 snapshot** (e.g. EMA vs non-EMA) than the one we
diff against. Decisive test (deferred, needs network): download a second `baulab` `ESDx1`
checkpoint for a different artist and check whether the non-attention deltas are *identical*
across the two — identical ⇒ (b) base mismatch; different ⇒ (a) full-parameter training.
Until then, do **not** describe this file as "a cross-attention-only ESD erasure"; it is
"the published baulab VanGogh-ESDx1 artifact, whose deltas cover the whole UNet".

### A-3. Dataset generation — (1) CONFIRMED DEFECT (NSFW-blanked images trained on)

`upstream_runs/imma/data/vangogh`: 100 PNGs, `metadata.csv` 100 rows + header, filenames match
disk exactly in both directions, single distinct prompt `An artwork by Van Gogh`. Metadata is
**(3) VALID**.

**3 of the 100 images are pure black** (`max pixel == 0`): `7.png`, `29.png`, `41.png`. No
near-black borderline cases (next-lowest std is 41.6). Root cause: upstream
`eval/text2image.py` builds `DiffusionPipeline.from_pretrained(...)` (L71-76) **without**
`safety_checker=None`, so SD-1.4's checker blanks flagged samples.

They were **included in training**: `run_r1_imma_official.sh` L41-42 writes one metadata row per
`*.png` with no content check, and `logs/r1b_imma_train.log` reports `Num examples = 100`.
So 3 % of the immunisation set is a black image captioned "An artwork by Van Gogh" — a data-quality
defect in both the smoke artifact and `REF_imma_vangogh`. Repaired: see **A-7**.

### A-4. What `defend_text_to_image_lora.py` actually does — (3) VALID vs our docs, with one naming correction

Read from source:

* **Trainable scope S** (L562-567): substring match of `['to_q','to_v','to_k','to_out']` over
  **every** `named_parameter`, excluding names containing `'lora'`. This catches `attn1`
  (self-attention) as well as `attn2`. Measured on the real checkpoint: **160 tensors, 93,536,640
  params — 80 `attn1` + 80 `attn2`, exactly the 160 non-LoRA `to_*` tensors in the UNet (93.54M).**
  The artifact name `imma_unet_xatten_layer.pt` and the variable `xattn_layer_names` are
  **misnomers**: half the scope is self-attention.
  `docs/UPSTREAM_FIDELITY_MANIFEST.md` §1.1 already states this correctly ("by substring over every
  `named_parameter`", 160 tensors / 93.5M) → **VALID**.
  `docs/PREREG_H82_imma_port.md` §2 states "S = cross-attention layers only … 80 `attn2`
  projections, 44.0M params" — that is the **paper's** claim and it does **not** match the released
  code. Both documents should carry that split explicitly.
* **Inner update** (L863-866, L945-952): when `step % (inner+outer) < inner` → `lora_layers`
  unfrozen, S frozen, `backward(loss)` (descent), `optimizer_lora.step()`, lr `1e-4`.
* **Outer update** (L867-868, L953-960): otherwise → `lora_layers` frozen, S unfrozen,
  `backward(-loss)` (**ascent** on the diffusion loss), `optimizer.step()`, lr `1e-5`.
  With `inner=outer=1`, strict alternation; `step` is the **in-epoch** dataloader index, so every
  epoch restarts on an inner step.
* **Two optimizers** (L593-608): `AdamW` over `lora_layers.parameters()` @ `learning_rate_lora`,
  and a separate `AdamW` over `unet_attn_params` @ `learning_rate`. Both `weight_decay=1e-2`,
  betas `(0.9, 0.999)`, `eps=1e-8`. Each has its own `constant` scheduler.
* **LoRA re-initialisation**: gated on `--reinitialize_lora` (L366, default `False`; L871-872).
  The README does not pass it and neither do we ⇒ **LoRA is NOT reinitialised; it is warm-started
  across the whole run.** This matches `PREREG_H82` ("WARM START, not reinitialised") and the
  manifest → **VALID**.
* **Artifact** (L977): `torch.save({name: param for name,param in zip(unet_attn_names, unet_attn_params)})`
  → S only, no LoRA, no optimizer state.

### A-5. The saved checkpoint and the composition in `r1_eval_imma.py` — (3) VALID. Composition is NOT the bug.

`REF_imma_vangogh/imma_unet_xatten_layer.pt`: 160 tensors, 93,536,640 params, all fp32,
80 `attn1` + 80 `attn2`, 32 `to_out.0.bias` + 128 weights. Every key is present verbatim in the
SD-1.4 UNet state dict and every shape matches.

The fuzzy-match + `reshape` path was the prime suspect. **Replicating the exact matching loop of
`r1_eval_imma.py` L23-26 offline:**
```
exact=160  fuzzy=0  reshaped(shape-differed)=0  skipped=0
```
and auditing the produced `adapters/REF_imma_vangogh/unet.pt` tensor by tensor:
```
composed tensors: 686, all fp16; key set identical to base SD-1.4
IMMA-scope tensors byte-equal to half(IMMA)      : 160 / 160
non-IMMA tensors byte-equal to half(ESD)         : 526 / 526
unexplained tensors                              : 0
non-finite tensors                               : 0
comp vs ESD: exactly 160 tensors with rel>1e-3, classes {attn2: 80, attn1: 80}
```
So: **no key fell back to fuzzy matching, nothing was reshaped, nothing was skipped, and every
IMMA tensor landed on the module it was trained on.** The composed UNet is ESD everywhere except
the 160 trained `to_*` tensors, which carry exactly the IMMA values. Item 4 of the coordinator's
request is answered affirmatively: **the S tensors are applied to their own modules.**

The composition is also **semantically identical to upstream's own**: `eval/text2image.py` L65-69
(`--defense_ckpt`) and `train_text_to_image_lora.py` L458-462 (`--imma_ckpt`) both do
`model_dict = unet.state_dict(); model_dict.update(torch.load(ckpt)); unet.load_state_dict(model_dict)`
after loading `--delta_ckpt`. `r1_eval_imma.py` reproduces that exactly.

**Direction / sign of the deltas** (coordinator item 3): the ESD delta is not inverted in any
detectable way, and the IMMA delta is modest and *targeted*, not divergent:
```
IMMA vs ESD over its own 160-tensor scope: min 0.0124  median 0.0268  max 0.2348
global ‖IMMA−ESD‖/‖ESD‖ over the scope   : 0.0274
largest deltas: attn2.to_out.0.weight (0.235, 0.190, 0.159, 0.153, 0.136…) then attn2.to_v
```
The perturbation concentrates in `attn2.to_out` and `attn2.to_v` — the text→image *value/output*
path, i.e. precisely the cross-attention conditioning channel — which is what 2500 ascent steps
on a single-prompt dataset would do.

Residual hardening still available without GPU: none. The remaining hypotheses are behavioural.

### A-6. `outputs/gen_REF_imma_vangogh.json` — (4) HISTORICAL RESULT needing reinterpretation. Keep quarantined; the cause is not a composition bug.

Observed: `van_gogh 0.867`; eight of eleven non-targets **exactly 0.000**; `parachute 0.200`,
`chain_saw 0.133`, `cassette_player 0.067`, `tench 0.067`; `monet 0.000`;
`utility_all_nontarget 0.0424`; `legacy_retain8 0.0333`; `clip_align` 0.165-0.217 (normal).
Controls `gen_base_sd14.json` / `gen_E6cal_BASE_sd14_on_DEV.json` score 1.000 on all twelve.

Given A-5, the artifact is validly composed, so the number is a real measurement of a real
checkpoint. It is the **interpretation** that is invalid, for three independent reasons:

1. **The scorer is a closed-set 12-way argmax.** `scripts/gen_eval.py` L48/L81/L127 classifies each
   image by `argmax` over `"a photo of {label}"` for twelve labels including `"van gogh painting"`
   and `"monet painting"`. If cross-attention conditioning is damaged and every image collapses to
   one painterly mode, the argmax lands on `van gogh painting` for *all* prompts — which produces
   exactly the observed signature: target high, `monet` 0.000, and eight objects at **exactly**
   0.000 rather than scattered around the 1/12 ≈ 0.083 chance line. A merely degraded-but-working
   model would give a spread, not a floor. (2) SUSPECTED mechanism — confirming it requires looking
   at the generated images, i.e. GPU work, which the freeze forbids.
2. **`target_suppression_acc` is not meaningful here.** 0.867 on the target cannot be read as
   "ESD+IMMA failed to erase Van Gogh" when the same classifier assigns `van_gogh` to prompts
   about garbage trucks.
3. **Standalone evaluation is outside IMMA's own threat model.** IMMA deliberately degrades the
   pre-trained weights so that *a later adaptation* fails; `README.md` L146-151 evaluates only
   post-adaptation CLIP/DINO/LPIPS on the fine-tuned model, never the immunised model alone.
   Aggravated by A-1 (5× the documented ascent budget) and A-3 (3 % black training images).

**CARRY-FORWARD: `outputs/gen_REF_imma_vangogh.json` must not be cited as an IMMA result, and in
particular must not be reported as "the IMMA reference destroys utility".** The defensible
statement is: *a 5×-over-budget IMMA immunisation of the published VanGogh-ESDx1 checkpoint, scored
standalone by a closed-set CLIP classifier, collapses to a single label; the IMMA paper does not
evaluate an immunised model standalone, so this configuration measures nothing about IMMA.*
The required controls before any IMMA claim: (i) ESD-only checkpoint through the same scorer
(isolates the ESD delta's direction behaviourally), (ii) a README-faithful
`--max_train_samples 20` run, (iii) the post-adaptation evaluation IMMA actually specifies,
(iv) saved image grids to test the label-collapse mechanism. `run_r1c_eval.sh` has been repaired
to run (i) and to refuse to emit a headline number without it (see A-8).

Note `legacy_retain8 = 0.0333 < 0.60`, so the K1 gate in `run_r1c_eval.sh` L32 correctly blocked
the attack grid; no `R1C_*` attack artifacts were produced.

### A-7. Environment change mid-flight — (3) VALID, no artifact invalidated, one provenance caveat

`venv_imma` site-packages mtimes: `diffusers-0.24.0` installed **11:47**, `huggingface_hub-0.25.2`
**11:48**, `datasets-5.0.1` 10:40; `torch 2.4.1+cu124`, `accelerate 1.1.1`, `transformers 4.46.3`
unchanged since 10-22. Artifact timeline:

| artifact | mtime | environment |
|---|---|---|
| `data/vangogh/*.png` + `metadata.csv` | 11:39-11:43 | **diffusers 0.31.0** (pre-downgrade) |
| `smoke/imma_unet_xatten_layer.pt` | 11:48 | diffusers 0.24.0 (log shows the 0.24-era `LoRAAttnProcessor` deprecation notice) |
| `REF_imma_vangogh/imma_unet_xatten_layer.pt` | 12:21 | diffusers 0.24.0 |
| `adapters/REF_imma_vangogh/unet.pt` | 12:39 | diffusers 0.24.0 (`venv_imma`) |

Only the **training images** were produced under 0.31.0; every trained and composed artifact is
0.24.0-consistent. The downgrade was necessary and is correctly documented in `run_r1b` L7-8:
`LoRAAttnProcessor` lost its `hidden_size` kwarg after 0.25, so `get_lora_procs` (L509-513) cannot
run on 0.31.0 at all. Nothing is invalidated. Two caveats to record:
(a) the 100 images carry a 0.31.0 scheduler/pipeline provenance that the trainer's environment no
longer reproduces, so regenerating them under 0.24.0 would not be bit-identical;
(b) in `venv_imma` (diffusers 0.24.0) `UNet2DConditionModel.from_pretrained(..., variant="fp16")`
resolves to `diffusion_pytorch_model.fp16.bin`, which the pinned snapshot does not have (only
`.fp16.safetensors`) — a latent failure mode for any future CPU-side audit in that venv; the
`StableDiffusionPipeline` path that `r1_eval_imma.py` uses does resolve correctly.

### A-8. (5) REPAIRS PERFORMED — IMMA

* `scripts/r1_eval_imma.py`: removed the silent fuzzy-match/`reshape` path. Now **strict** —
  requires exact key presence and exactly equal shapes, asserts the ESD state dict covers the full
  UNet key set, asserts all tensors finite, and exits non-zero with a diagnostic on any mismatch,
  skip, or shape disagreement. Added `--esd-only` to emit the ESD-only control UNet, and a
  `compose.json` provenance sidecar recording counts, per-class delta norms and the file hashes.
* `run_r1_imma_official.sh`: `set -euo pipefail`; explicit rc checks on data generation and the
  smoke run; **black-image rejection** — `metadata.csv` is now written only over images that are
  not NSFW-blanked, the count of rejects is logged, and the run aborts if fewer than
  `IMMA_MIN_IMAGES` usable images survive; all README args now passed explicitly
  (`--max_train_samples`, `--inner_loop_steps`, `--outer_loop_steps`, `--report_to`).
* `run_r1b_imma_train.sh`: `set -euo pipefail`, rc checks, **`--max_train_samples` now explicit**
  (default `20`, the README value) and the output directory is derived from it, so a README-faithful
  run cannot silently overwrite the 5× artifact; asserts the checkpoint exists and is non-trivial
  afterwards.
* `run_r1c_eval.sh`: `set -euo pipefail`, rc checks on every stage, composes **and evaluates the
  ESD-only control first**, and writes a `QUARANTINE` note next to the composed adapter stating
  that standalone suppression/utility numbers from an immunised checkpoint are not IMMA results.

---

## B. Meta-Unlearning

### B-1. `run_r2_meta_official.sh` never invoked the trainer — (1) CONFIRMED DEFECT, (5) REPAIRED

Verified: the script (34 lines, pre-repair) ran `scripts/r2_build_meta_pools.py`, echoed per-pool
counts, printed `R2 POOLS DONE` and exited. No reference to `upstream/Meta-Unlearning/src/meta.py`,
`esd_meta.py` or `sdd_meta.py` anywhere in it; `grep -c meta.py run_r2_meta_official.sh` → 0.
Its own header claimed "upstream `src/meta.py` is invoked, not our port" (L15) — false.
Matches `docs/AUDIT_REPAIR_REGISTER_v1.md` D-002. Repair implemented in **B-7**.

### B-2. Upstream `src/meta.py` cannot run at this commit — (1) CONFIRMED DEFECT (upstream)

Three independent fatal faults, read from source:

1. **Arity mismatch.** `train_step` returns a **4-tuple** (L629: `return task_unet, task_optimizer,
   task_lr_scheduler, np.mean(all_losses)`) but **all 11 call sites** unpack **3** values
   (L967, 973, 978, 983, 990, 995, 1001, 1007, 1013, 1019, 1025) →
   `ValueError: too many values to unpack (expected 3)` on the first call of the first step.
   `src/esd_meta.py` and `src/sdd_meta.py` return 3 and are internally consistent; only `meta.py`
   is broken.
2. **No main unlearning loss.** `grep -c 'train_loss *=' src/meta.py` → **0**. `train_loss` is read
   at L1045/1047/1050 and never assigned → `NameError`. `esd_meta.py` L933 assigns it from
   `train_unlearn_step` and back-propagates it at L1003; `meta.py` has no equivalent.
3. **`param.grad` is `None`.** With no main backward pass, `param.grad += sum_grads[i]` (L1033)
   raises `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'Tensor'`.

**Consequence for the task brief:** the named reference `src/meta.py` is not runnable, so an
"official reproduction" cannot be built on it without patching upstream (which is out of scope and
would no longer be the published artifact). The repair therefore targets **`src/esd_meta.py`**,
which is complete, consistent, and has its own shell script. Note the difference in what that
reproduces: `esd_meta.py` learns the ESD erasure *and* the meta-regulariser jointly from SD-1.4
(teacher and student both initialised from `--pretrained_model_name_or_path`, L839-844), so
**the correct "initial erased checkpoint" for this variant is plain `CompVis/stable-diffusion-v1-4`
— there is no separate erased checkpoint to supply.** `meta.py`'s `--pretrained_model_name_or_path
"your-model-path"` is the UCE/RECE path, which expects an *already*-erased model; that path is the
non-runnable one.

Other upstream quirks recorded: checkpoint saving is nested inside the validation branch
(L1061-1076 / `esd_meta.py` equivalent), so **intermediate checkpoints are only written when
`--validation_prompts` is non-empty**; without it only the final full-pipeline
`save_pretrained(args.output_dir)` is produced. `args.output_dir` is silently rewritten to
`os.path.join(output_dir, exp_name)` (L788-790). `src/gen_images.py` has no `hrm`/`irt`/`tgt`
entries in its `text_dict` and loops over `["style","paint"]`, so the shipped generator does not
produce the directory layout `meta.py` requires.

### B-3. Hardcoded dataset paths — (1) CONFIRMED DEFECT in our runner (upstream behaviour is as described)

`src/meta.py` L671-673 and `src/esd_meta.py` L671-673, identically:
```python
hrm_dataset = load_dataset("imagefolder", data_dir="./dataset/hrm")
irt_dataset = load_dataset("imagefolder", data_dir="./dataset/irt")
tgt_dataset = load_dataset("imagefolder", data_dir="./dataset/tgt")
```
Relative to the **process CWD**, with no CLI override. Splits come from the `train/`…`test/`
subdirectory names. Our pools live at `upstream_runs/meta/dataset/{hrm,irt,tgt}/{train,test}` and
the pre-repair runner never changed directory, so even if it had called the trainer, the load would
have failed. Repaired in B-7 by running the trainer from a staging directory whose `./dataset`
points at the built pools.

### B-4. The gamma arguments in `scripts/meta.sh` and `scripts/esd_meta.sh` — (1) CONFIRMED DEFECT (upstream examples); do not copy

Both scripts pass `--gamma2_1` **three times** and never pass `--gamma2_2` or `--gamma2_3`:
`meta.sh` L13-15 → `0.1, 0.1, 0.01`; `esd_meta.sh` L14-16 → `0.1, 0.1, 0.01`.
argparse keeps the **last** occurrence, so replicating the parser (`meta.py` L90-94, defaults 1.0):
```
repeated gamma2_1 -> 0.01   gamma2_2 -> 1.0   gamma2_3 -> 1.0
```
**Actually-resolved values** for the shipped example: `gamma1_1=0.1`, `gamma1_2=0.1` (`meta.sh`) /
`0.01` (`esd_meta.sh`), `gamma2_1=0.01`, `gamma2_2=1.0`, `gamma2_3=1.0`.
The evident intent (one value per term, matching the `gamma1_*` pattern) was
`gamma2_1=0.1, gamma2_2=0.1, gamma2_3=0.01`; the two retain-side terms therefore run at **100× and
10×** their intended weight. Semantics from `meta.py` L1010-1033: `gamma2_1` weights the `hrm/test`
term, `gamma2_2` the `tgt/test` term and `gamma2_3` the `irt/test` term, all **subtracted** from
`sum_grads`, while `gamma1_1` (`irt/test`) and `gamma1_2` (`tgt/test`) are **added**.
The repaired runner passes all five **once each, explicitly**, and logs the resolved values; it does
not reproduce the duplication.

### B-5. `type=bool` on `fix_timesteps` — (1) CONFIRMED DEFECT (parser hazard), verified

`src/meta.py` L117 and `src/esd_meta.py` L117: `parser.add_argument("--fix_timesteps", type=bool, default=False)`.
`bool(s)` is `True` for every non-empty string, so argparse cannot express `False`. Replicated:
```
--fix_timesteps False  -> True
--fix_timesteps 0      -> True
--fix_timesteps True   -> True
(omitted)              -> False
```
**The only way to obtain `fix_timesteps=False` is to omit the flag entirely.** The README (L43)
states the paper used `fix_timesteps = False` and that `True` exists only to accelerate training,
so the paper setting requires **omission**, and `esd_meta.sh` happens to get it right only because
the flag is commented out (L17-18). Any script that writes `--fix_timesteps False` silently selects
the *accelerated, non-paper* branch (`meta.py` L959-973 / L1010-1026, which iterates
`args.fixed_time_steps` instead of sampling timesteps) — a 100%-wrong configuration that produces no
error. The repaired runner never passes the flag and asserts in the log that the paper branch is
selected.

Companion hazard in the same parser: `--fixed_time_steps` is `type=list` with no `nargs`
(L118), so `--fixed_time_steps 1 2 5 10` parses as `['1']` (a list of *characters of the first
token*, as strings) and leaves `2 5 10` as unrecognised positionals → `parse_args` hard-errors.
`meta.sh` L17 passes exactly that form. Both upstream example scripts are unrunnable as written.

### B-6. Pool sizes — (4) HISTORICAL RESULT: an adaptation, not the author protocol, (5) REPAIRED

`src/gen_images.py` L90-92: `for text in text_all: train_data.extend(generate_images(..., num_images=30)); test_data.extend(...)`
→ **30 train + 30 test per PROMPT**, i.e. `30 × n_prompts` per split.
`scripts/r2_build_meta_pools.py` L52 (pre-repair): `per = max(1, a.n // len(texts))` → **30 per
POOL, split across prompts**. Built pools, verified on disk:

| pool | prompts | per prompt (ours) | per prompt (upstream) | per split (ours) | per split (upstream) |
|---|---|---|---|---|---|
| hrm | 1 | 30 | 30 | 30 | 30 |
| tgt | 3 | **10** | 30 | 30 | **90** |
| irt | 6 | **5** | 30 | 30 | **180** |

Our `tgt` and `irt` pools are **3× and 6× smaller per prompt** than upstream's. The header comment
in both `run_r2_meta_official.sh` L11 and `r2_build_meta_pools.py` L7 claimed "30 train + 30 test
per pool, … follows `src/gen_images.py`" — the two halves of that sentence contradict each other,
and the file implemented the smaller reading. **This is a deliberate compute adaptation and must be
labelled as such, not as "author protocol."** Repaired: the builder now takes `--per-prompt`
(upstream semantics, the default) or `--total-per-pool` (the historical behaviour) and records the
choice, the resolved per-prompt count and the upstream comparison in a `pools.json` manifest.

Pool integrity otherwise **(3) VALID**: 30 PNG + 30 metadata rows in each of the six directories,
filenames matching in both directions, **0 black images** (the builder passes `safety_checker=None`,
unlike the IMMA path — see A-3), train seeds `1000+100j` and test seeds `5000+100j` so the splits
are disjoint, and the author's exact negative prompt is used verbatim
(matches `gen_images.py` L46 character for character).

### B-7. forget = "church" vs related-retain = "cathedral/chapel/basilica" — (1) CONFIRMED DEFECT (incoherent), recommendation below

The current pools ask the method to erase `a photo of a church` while **retaining**
`a photo of a cathedral`, `a photo of a chapel`, `a photo of a basilica`. These are not "related"
concepts: a cathedral, a chapel and a basilica **are churches** — they are hyponyms, not
neighbours. The objective is therefore self-contradictory at the concept level: a genuine erasure
of the *concept* church must remove its instances, and anything that keeps them has only suppressed
the *token* "church".

What upstream actually means by `tgt`: `README.md` L18 defines `tgt` as "the concept **related** to
the unlearned concept", and `gen_images.py` L62-66 instantiates it as
`hrm = "Thomas Kinkade"` (a style) with `paint = "a painting"` — the **superordinate medium**, and
`style`/`ft_style` for the forget-generalisation prompts. Upstream's related-retain is a
*superordinate or sibling* category, never a hyponym of the forget concept.

**Recommended definition** (adopt in the repaired builder, selectable by `--pool-scheme`):

* **forget (hrm)** — the concept and its *paraphrases*: `a photo of a church`,
  `a photo of a church building`, `a photograph of a church exterior`.
* **forget-generalisation (fgen)** — hyponyms that a concept erasure **must also remove**:
  `cathedral`, `chapel`, `basilica`. Scored as **forget**, never as retain.
* **related-retain (tgt)** — co-hyponyms and the superordinate, which share visual and textual
  structure with the target but are **not instances of it**: `a photo of a mosque`,
  `a photo of a synagogue`, `a photo of a temple`, `a photo of a castle`, `a photo of a town hall`,
  `a photo of a stone building`.
* **unrelated-retain (irt)** — unchanged, sampled from the author's `data/coco_object_retain.csv`.

**Distinction to state in every report of this arm:** retaining cathedral/chapel/basilica while
suppressing "church" measures **prompt-specific token suppression** (does the model stop responding
to the string "church"?). Removing them alongside it measures **concept erasure** (is the visual
concept gone regardless of lexical route?). The existing pools, and every result computed from them,
measure the former. Any claim of "church erasure" built on the current `tgt` pool is a claim about
token suppression and must be worded that way. The `cathedral/chapel/basilica` prompts are a
perfectly good *probe* — they just belong on the forget side of the ledger, not the retain side.

### B-8. (5) REPAIRS PERFORMED — Meta-Unlearning

* `scripts/r2_build_meta_pools.py`: rewritten. `--per-prompt` (upstream semantics, default 30) vs
  `--total-per-pool` (the historical adaptation) with the choice recorded; `--pool-scheme`
  (`legacy` = the existing church/cathedral pools, `coherent` = the B-7 definition with a separate
  `fgen` pool); black-image rejection with regeneration on a fresh seed and a hard failure if the
  budget is exhausted; per-directory count validation after writing; a `pools.json` manifest with
  prompts, seeds, counts, the upstream comparison and the scheme; non-zero exit on any shortfall;
  no silent `skip` on a pre-existing directory (now `--resume` must be explicit).
* `run_r2_meta_official.sh`: implements the whole path, fail-fast throughout
  (`set -euo pipefail`, explicit rc check after every stage). It (1) builds the pools, (2)
  **validates** that each of the six directories holds the expected count with a matching
  `metadata.jsonl` and no black images, aborting otherwise, (3) prepares a staging directory with
  `./dataset` pointing at the pools so upstream's hardcoded relative paths resolve, (4) invokes
  **`upstream/Meta-Unlearning/src/esd_meta.py`** (not the non-runnable `meta.py` — B-2) from that
  CWD with every one of the five gammas passed exactly once, `--fix_timesteps` deliberately
  **omitted** so the paper branch is selected, `--validation_prompts` supplied so intermediate
  checkpoints are written at all, and all other hyperparameters resolved explicitly in the script
  rather than inherited from defaults, (5) verifies the saved checkpoint exists, **reloads it on
  CPU** and asserts the UNet key set and parameter count match SD-1.4 and that all tensors are
  finite, and (6) writes a run manifest. **It has NOT been executed** — the project is frozen and
  the script refuses to start unless `FREEZE_ACTIVE` is absent or `R2_FREEZE_OVERRIDE=1` is set by
  the implementation lead.

---

## C. Summary table

| # | item | class |
|---|---|---|
| A-1 | `--max_train_samples 20` omitted → 5000 steps instead of 1000, 2500 ascent steps instead of 500 | **1 CONFIRMED DEFECT** |
| A-1 | `--inner_loop_steps` / `--outer_loop_steps` / `--report_to` omitted but defaults identical | **3 VALID** |
| A-1 | no `--revision`/`--variant`: trainer used `main` fp32, project pins `133a221b` fp16 | **2 SUSPECTED** (provenance) |
| A-2 | ESD ckpt is the published VanGogh artifact, concept-matched to what we train/eval | **3 VALID** |
| A-2 | "ESDx1" but all 686 tensors differ from base, incl. resnets at 0.20 relative | **2 SUSPECTED** |
| A-3 | 3/100 training images NSFW-blanked to black and included in training | **1 CONFIRMED DEFECT** |
| A-3 | `metadata.csv` consistent with disk, single prompt | **3 VALID** |
| A-4 | scope is 160 tensors / 93.54M = 80 `attn1` + 80 `attn2`; "xatten" is a misnomer | **3 VALID** (manifest correct; `PREREG_H82` quotes the paper, not the code) |
| A-4 | LoRA warm-started, not reinitialised; two AdamW optimizers at 1e-4 / 1e-5; ascent on `-loss` | **3 VALID** |
| A-5 | composition: 160 exact matches, 0 fuzzy, 0 reshapes, 0 skips, 0 unexplained, 0 non-finite | **3 VALID** — composition is **not** the bug |
| A-6 | `gen_REF_imma_vangogh.json` — valid artifact, invalid interpretation; keep quarantined | **4 HISTORICAL RESULT** |
| A-7 | diffusers 0.31.0 → 0.24.0 mid-flight; only the images predate it | **3 VALID**, provenance caveat |
| A-8 | strict composition, black-image rejection, explicit README args, ESD-only control | **5 REPAIR PERFORMED** |
| B-1 | `run_r2_meta_official.sh` built pools and exited; never called any trainer | **1 CONFIRMED DEFECT** |
| B-2 | upstream `src/meta.py` is unrunnable: 4-vs-3 tuple unpack, undefined `train_loss`, `None.grad` | **1 CONFIRMED DEFECT** (upstream) |
| B-3 | `./dataset/{hrm,irt,tgt}` hardcoded relative to CWD, no override | **1 CONFIRMED DEFECT** in our runner |
| B-4 | `--gamma2_1` ×3, `gamma2_2`/`gamma2_3` never set → resolve to `0.01, 1.0, 1.0` | **1 CONFIRMED DEFECT** (upstream examples) |
| B-5 | `type=bool` ⇒ `--fix_timesteps False` yields `True`; paper setting requires omission | **1 CONFIRMED DEFECT** (parser hazard) |
| B-6 | 30 per pool, not 30 per prompt → `tgt` 3× and `irt` 6× smaller than upstream | **4 HISTORICAL RESULT** (adaptation) |
| B-6 | pool integrity: counts, metadata, disjoint seeds, 0 black, exact negative prompt | **3 VALID** |
| B-7 | forget "church" with related-retain "cathedral/chapel/basilica" (hyponyms) is incoherent | **1 CONFIRMED DEFECT** |
| B-8 | full R2 path implemented, fail-fast, validated, reload-verified; **not executed** | **5 REPAIR PERFORMED** |
