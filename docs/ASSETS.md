# Asset manifest — CUIG ConAbl / UnlearnCanvas pilot

Status date: 2026-10-08 · Host: `hyperplane`

Nothing in this file is tracked as a binary. Every asset below lives outside the
repository, under `/data/bijaypandey/cuig_pilot/`, because the filesystem that
carries `/home` is at **100% capacity (5.8 GB free)**.

---

## 1. Summary of availability

| Asset | Required for | Status |
|---|---|---|
| UnlearnCanvas generator (`style50`, diffusers format) | training, sampling | **NOT OBTAINED** |
| UnlearnCanvas style classifier (`style_classifier.pth`) | UA, IRA | **NOT OBTAINED** |
| UnlearnCanvas object classifier (`object_classifier.pth`) | CRA | **NOT OBTAINED** |
| ConAbl anchor prompts (`anchor_prompts/style/Laion.txt`) | training | **PRESENT** (in upstream checkout) |
| ConAbl anchor images (`anchor_datasets/style/Laion`) | training | **GENERATED ON DEMAND** by `train_conabl.py` |
| timm `vit_large_patch16_224.augreg_in21k` backbone | `evaluate.py` | **AVAILABLE** from HuggingFace |
| Base Stable Diffusion v1.5 (substitute generator) | mechanics check only | **PRESENT** (5.2 GB) |

**Consequence: UA / IRA / CRA are UNAVAILABLE.** No target-suppression or
retained-performance numbers can be produced until the two classifiers are in
place. This is an asset-availability blocker, not a code or environment problem.

---

## 2. Required filenames and exact destination paths

The pilot reads these paths from `scripts/pilot_config.sh`. Drop the files in
and nothing else needs to change.

```text
/data/bijaypandey/cuig_pilot/Checkpoints/
├── Classifiers/UnlearnCanvas/
│   ├── style_classifier.pth      # renamed from style50.pth
│   └── object_classifier.pth     # renamed from style50_cls.pth
└── Generators/UnlearnCanvas/
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── unet/
    └── vae/
```

Upstream's rename step (`Checkpoints/README.md`): from the downloaded
`cls_model` folder, delete `style60.pth`, then
`style50.pth → style_classifier.pth` and `style50_cls.pth → object_classifier.pth`.
The generator folder `style50` is renamed to `UnlearnCanvas`.

---

## 3. Documented sources and the errors observed

All four Google Drive folder IDs referenced by CUIG and by the upstream
UnlearnCanvas repository were probed from this host. Every one returns a
1652-byte Google **HTTP 404** page, including the dataset folder that the
official UnlearnCanvas README advertises as publicly available — so the failure
cannot be attributed to any single folder being withdrawn.

| Source | Referenced by | Observed |
|---|---|---|
| `drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61` | CUIG `Checkpoints/README.md` (classifiers) | HTTP 404 |
| `drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx` | CUIG `Checkpoints/README.md` (generator) | HTTP 404 |
| `drive.google.com/drive/folders/18dhkXyZQWjdMvlAlxZx3fZhdCZvlj2Hw` | UnlearnCanvas `README.md`, `machine_unlearning/README.md`, `style_transfer/README.md` (all checkpoints, incl. `cls_model`) | HTTP 404 |
| `drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy` | UnlearnCanvas `diffusion_model_finetuning/README.md` | HTTP 404 |
| `drive.google.com/drive/folders/1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM` | UnlearnCanvas `README.md` (dataset, advertised public) | HTTP 404 |

Tool-level errors:

```
$ gdown --folder https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61
Retrieving folder contents
Failed to retrieve folder contents          # gdown 5.2.1

$ curl https://www.googleapis.com/drive/v3/files?q='<id>'+in+parents
403 "Method doesn't allow unregistered callers ... Please use API Key"
```

Reachability controls (so this is not read as a general network outage):

```
https://www.google.com        -> 200
https://drive.google.com/     -> 302   (reachable)
https://huggingface.co        -> 200
```

**Interpretation:** the host reaches Google and HuggingFace. Unauthenticated
Google Drive *folder* access is refused here, so `gdown` cannot enumerate any
folder, and per-folder existence cannot be distinguished from per-folder denial.
Acquisition requires either an authenticated/browser session or an out-of-band
copy.

---

## 4. Mirrors evaluated

### 4.1 `sungjuncho/fade-unlearncanvas-full` — generator candidate, NOT adopted

Verified directly by range-reading the safetensors header (no full download).

| Property | Observed |
|---|---|
| File | `full.safetensors`, 7,703,324,308 B (7.17 GiB) |
| SHA-256 (HF `x-linked-etag`) | `bd11856118fd31a1aa4d5e1c1ecf1a165b6bb3db8ef1ef3864d15d7d21d38077` |
| Tensor count | 1831 (1829× F32, 1× I64, 1× I32) |
| Top-level prefixes | `model.` (686), `model_ema.` (688), `first_stage_model.` (248), `cond_stage_model.` (197), plus 12 diffusion-schedule buffers |
| Embedded metadata | `global_step: 7000`, `epoch: 778` |
| Architecture | SD v1.x — `cond_stage_model.transformer.text_model`, `token_embedding [49408, 768]` ⇒ CLIP ViT-L/14 text encoder, width 768 |
| Format | **CompVis / LDM**, not diffusers |
| Licence | CreativeML OpenRAIL-M; `base_model: runwayml/stable-diffusion-v1-5` |
| Card claim | "a mirror of the checkpoint released by the UnlearnCanvas authors", fine-tuned on all 50 UnlearnCanvas styles, pre-unlearning |

**Why it was not adopted.** Unresolved differences, each of which would change
what a baseline number means:

1. **Format incompatibility.** CUIG loads a diffusers pipeline directory
   (`AutoencoderKL`/`UNet2DConditionModel`/`CLIPTextModel` with `subfolder=`).
   This checkpoint is a monolithic CompVis state dict and would need conversion
   (`diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py`).
2. **EMA ambiguity.** It carries both `model.` and `model_ema.` weights.
   Conversion must pick one; which set the authors' released *diffusers*
   `style50` folder corresponds to is not documented, and the two give
   different generators.
3. **Equivalence unverifiable.** The authors publish no checksum for the
   original, and the original is unreachable here, so the mirror's claim to be
   the same checkpoint cannot be confirmed. Provenance is third-party (the FADE
   authors, arXiv:2510.12981), not the UnlearnCanvas authors.
4. **Conversion risk.** Scheduler and VAE config choices made during conversion
   silently affect sampling.

It remains the most promising recovery route if the Drive folders stay
unreachable, but it must be converted and validated before any numbers from it
are treated as a baseline.

### 4.2 Classifiers — no mirror found

Searched HuggingFace models and datasets for `unlearncanvas`, `unlearn-canvas`,
`style50`, `style_classifier`, `uc_classifier`, and author `OPTML-Group`.

- `OPTML-Group/UnlearnCanvas` (dataset) — 333 files, **all** `data/*.parquet`
  plus `.gitattributes`/`README.md`. Images only; **no checkpoints**.
- `OPTML-Group/UnlearnCanvas-Benchmark` (Space) — gradio leaderboard, 22 source
  files, **no checkpoints**.
- `sungjuncho/fade-unlearncanvas-*` — UNet-only `.safetensors` for a different
  paper, covering only `monet` / `picasso` / `van_gogh`; not the ViT-L
  style/object classifiers.
- `leno3003/SDXL_unet_UnlearnCanvas` — SDXL UNet; wrong architecture.

No mirror of the UnlearnCanvas ViT-Large style/object classifiers exists on
HuggingFace.

---

## 5. Validation requirements before any asset is trusted

Run `python scripts/preflight_check.py --gpu <idle>` once assets are in place.
It must report PASS for all of:

1. **Generator loads** — `model_index.json` plus all five subdirectories
   present; `StableDiffusionPipeline.from_pretrained` succeeds.
2. **Classifier head dimensions match the label lists** —
   `style_classifier.pth` head must be `out_features == 51`, and
   `object_classifier.pth` head `out_features == 20`, matching
   `len(STYLES_AVAILABLE)` and `len(OBJECTS_AVAILABLE)` in
   `Evaluation/UnlearnCanvas/constants.py`. `evaluate.py` builds the head as
   `torch.nn.Linear(1024, num_classes)` and reads weights from
   `ckpt["model_state_dict"]`, so a mismatch fails loudly — but a *silently
   permuted* label order would not. See §6.
3. **Generation works** — one image generates at 512×512.
4. **Classifier scores that image** — both classifiers return logits of shape
   `[1, 51]` and `[1, 20]`.

Additionally, before trusting UA/IRA/CRA, confirm **label-order agreement**
empirically (§6).

---

## 6. Label-order agreement — what is and is not established

`evaluate.py` converts a concept name to an integer with
`STYLES_AVAILABLE.index(name)` / `OBJECTS_AVAILABLE.index(name)` and maps a
prediction back with `STYLES_AVAILABLE[pred]`. Correct metrics therefore require
the classifier's output index order to equal the order of those two lists.

**Established statically (no assets needed):**

- `STYLES_AVAILABLE` has 51 entries and is exactly alphabetically sorted.
- `OBJECTS_AVAILABLE` has 20 entries and is exactly alphabetically sorted.
- Index 0 is `Abstractionism` (style) and `Architectures` (object).
- `sample.py` writes `"<style>_<object>_seed<seed>.jpg"` from the prompt
  template `f"A {obj} image in {style} style"`; `evaluate.py` reconstructs the
  same filename, so sampling and evaluation agree with each other.
- Alphabetical ordering is consistent with `torchvision.datasets.ImageFolder`
  class ordering, which is the conventional way these classifiers were trained.

**Not established, and requiring the real classifiers:** that the checkpoints
were in fact trained with that ordering. The agreement check is necessary, not
sufficient. Once the classifiers exist, verify by sampling from the
*un-modified* generator and confirming high top-1 accuracy against the known
style/object of each generated image — a permuted label order shows up as
near-chance accuracy on an otherwise healthy generator. A permutation would
corrupt UA/IRA/CRA without raising any error.

Note also that `STYLES_AVAILABLE` contains 51 entries — the 50 painting styles
plus `Seed_Images` — while the benchmark's unlearn/retain splits use 12 + 12
styles. Upstream deletes the downloaded `style60.pth` and uses `style50.pth`,
so the 51-way head is the expected one.

---

## 7. Re-fetch instructions

```bash
# Once Drive is reachable (or from a machine that can reach it):
bash scripts/fetch_assets.sh          # follows upstream's documented steps

# Then validate:
python scripts/preflight_check.py --gpu <idle-gpu-index> \
    --json /data/bijaypandey/cuig_pilot/preflight/preflight_assets.json
```

If Drive remains unreachable, download the two folders through a browser and
copy them to the §2 paths; `fetch_assets.sh` skips anything already present.

---

## 8. Other large local paths (all git-excluded)

| Path | Contents | Size |
|---|---|---|
| `/data/bijaypandey/cuig_pilot/CUIG` | upstream checkout @ `9932ac3` | 68 MB |
| `/data/bijaypandey/cuig_pilot/venv-cuig` | isolated Python env | ~9 GB |
| `/data/bijaypandey/cuig_pilot/Generators_substitute/sd-v1-5` | base SD v1.5 (mechanics check only) | 5.2 GB |
| `/data/bijaypandey/cuig_pilot/hf_home` | HuggingFace cache | varies |
| `/data/bijaypandey/cuig_pilot/mechanics_check` | substitute-run outputs | small |
| `/data/bijaypandey/cuig_pilot/outputs` | experiment outputs (empty until assets arrive) | — |
