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
| Labelled held-out images (UnlearnCanvas HF dataset) | classifier label-order + accuracy validation | **PRESENT** (137 imgs, 10 styles, 20 objects) |
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

All five Google Drive folder IDs referenced by CUIG and by the upstream
UnlearnCanvas repository were probed from this host. **Every one returns a
1652-byte Google HTTP 404 page.** That is the observation. The cause is **not
established** — see the interpretation below.

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

### Interpretation — the cause is UNRESOLVED

What is established:

- The host has working general connectivity to Google and HuggingFace.
- All five documented Drive folder IDs return HTTP 404 from this host.
- `gdown` cannot enumerate any of them.
- Google's Drive REST API refuses unauthenticated callers (403), so the 404 page
  cannot be cross-checked against an authoritative existence query from here.

What is **not** established — and must not be asserted:

- whether the folders still exist;
- whether their sharing permissions changed;
- whether unauthenticated Drive *folder* access is blocked from this network or IP;
- whether `gdown` 5.2.1 is simply broken against Drive's current folder endpoint.

Google returns 404 both for a folder that does not exist and for one the caller
may not view, so these cases are **indistinguishable from outside**. One data
point argues against simple deletion — the last row is the dataset folder the
official UnlearnCanvas README advertises as public, and it 404s too — but that is
suggestive, not conclusive, and it is equally consistent with a network-side
restriction.

**Conclusion: the download path is blocked and the reason is unknown.**
Acquisition requires an authenticated/browser session, an out-of-band copy, or
working links from the maintainers. A request to the maintainers that states the
ambiguity plainly is drafted at
[`outbox/asset-request-DRAFT.md`](outbox/asset-request-DRAFT.md) (not sent).

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

### 4.2 Classifiers — no accessible, verified mirror found in the sources searched

This is a **bounded negative result**, not proof that no mirror exists. The
search covered only the sources listed below; other hosts (institutional pages,
Zenodo, personal mirrors, private copies) were not searched and may well hold
them.

Sources searched: HuggingFace **models** and **datasets** indexes for the queries
`unlearncanvas`, `unlearn-canvas`, `style50`, `style_classifier`,
`uc_classifier`, `vit_large style`; the `OPTML-Group` author namespace; the
`OPTML-Group/UnlearnCanvas-Benchmark` Space; and the download links cited in the
CUIG and UnlearnCanvas repositories.

Result: **no accessible mirror of the UnlearnCanvas ViT-Large style/object
classifiers was found in those sources.** Nothing found anywhere was both
accessible and verifiable as the authors' checkpoint.

- `OPTML-Group/UnlearnCanvas` (dataset) — 333 files, **all** `data/*.parquet`
  plus `.gitattributes`/`README.md`. Images only; **no checkpoints**.
- `OPTML-Group/UnlearnCanvas-Benchmark` (Space) — gradio leaderboard, 22 source
  files, **no checkpoints**.
- `sungjuncho/fade-unlearncanvas-*` — UNet-only `.safetensors` for a different
  paper, covering only `monet` / `picasso` / `van_gogh`; not the ViT-L
  style/object classifiers.
- `leno3003/SDXL_unet_UnlearnCanvas` — SDXL UNet; wrong architecture.

None of the above is the UnlearnCanvas ViT-Large style/object classifiers. The
HuggingFace indexes surfaced no such checkpoint for the queries listed; whether
one exists elsewhere is unknown.

**One useful positive result:** `OPTML-Group/UnlearnCanvas` (the image dataset)
**is reachable** from this host. Its captions have the form
`"A {Object} image in {Style} style"`, so ground-truth style and object labels
are recoverable exactly. That makes it a legitimate source of **labelled
held-out images** for validating the classifiers once they are obtained — see
§6 and `scripts/build_heldout_eval_set.py`. A 137-image, 10-style, 20-object
probe set has already been built and is in place, so classifier validation can
run the moment the checkpoints land.

Note that the dataset covers **60** styles while `STYLES_AVAILABLE` holds 51
entries (50 styles + `Seed_Images`); images whose style is outside that list
(e.g. `Dreamweave`, `Pointillism`) are excluded from the style test rather than
counted as errors.

---

## 5. Validation gate — what must pass before any asset is trusted

Assets have three states. **Presence is never sufficient.**

| State | Meaning |
|---|---|
| `MISSING` | not on this host |
| `PRESENT_UNVERIFIED` | bytes present, hashed; behaviour and provenance unestablished |
| `VERIFIED` | passed `scripts/validate_assets.py`; receipt recorded |

Only `scripts/validate_assets.py` may write `VERIFIED`. Both
`scripts/preflight_check.py` and `scripts/run_single_concept_conabl.sh` refuse to
proceed without that receipt, so no UA/IRA/CRA can be produced from an unverified
asset even by accident.

### Order of operations once the assets arrive

```bash
python scripts/record_asset_hashes.py       # -> PRESENT_UNVERIFIED (+ SHA-256)
python scripts/build_heldout_eval_set.py    # already done; rerun to enlarge
python scripts/validate_assets.py --gpu <idle> --update-manifest
python scripts/preflight_check.py --gpu <idle>     # gate must read OPEN
```

### What `validate_assets.py` checks

| # | Check | Decides |
|---|---|---|
| A | generator dir and both classifier files present | `MISSING` vs present |
| B | classifier heads are `Linear(1024, 51)` / `Linear(1024, 20)`, weights under `model_state_dict` | **shape compatibility only** |
| C | top-1 on labelled held-out UnlearnCanvas images under the `constants.py` ordering; plus modal-predicted-index per true class | label order `confirmed` / `refuted_permuted` / `inconclusive` |
| D | that top-1 clears a threshold (default 0.70; local policy, recorded in the receipt, not a published value) | accuracy pass/fail |
| E | sample from the **untouched** generator with the released sampling config and score it | pre-unlearning reference; catches a wrong generator |

**Check B is explicitly not validation.** A checkpoint whose classes are permuted
has exactly the same head shape as a correct one, so `preflight_check.py` reports
the shape check as `WARN`, never `PASS`, to keep it from being read as a green
light. This was verified by a negative control: randomly-initialised heads of the
correct shape scored 3.5% (style, chance 2.0%) and 2.9% (object, chance 5.0%)
with modal-index agreement 0/8 and 0/20, and were **rejected** —
`shape_compatible: true`, `classifiers_verified: false`,
`UA_IRA_CRA: UNAVAILABLE`.

### Fields that stay `unknown` even after a full PASS

Behavioural agreement does not establish origin. These remain `unknown` in both
the manifest and the receipt until a maintainer supplies them:

- `classifier_provenance`, `generator_provenance`
- `generator_ema_selection` (EMA vs non-EMA)
- `preprocessing_matches_classifier_training`

> A locally computed SHA-256 identifies the bytes we hold. It does **not**
> independently prove equivalence to the authors' checkpoint. Only a checksum
> published by the authors, or a copy obtained directly from them, can do that.

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
were in fact trained with that ordering. The alphabetical-sorting argument is
necessary, not sufficient. A permutation would corrupt UA/IRA/CRA without raising
any error.

**How it is now tested** (`scripts/validate_assets.py`, check C). Rather than
relying on generated images alone, the probe uses **real labelled UnlearnCanvas
images** from the HuggingFace dataset, whose captions carry ground truth
(`scripts/build_heldout_eval_set.py`; 137 images, 10 styles, 20 objects already
built at `/data/bijaypandey/cuig_pilot/heldout_eval_set`). For each classifier it
computes:

- **top-1 accuracy under the assumed ordering** — high accuracy confirms it;
- **the modal predicted index per true class** — if predictions are consistent
  and distinct per class but land on the *wrong* indices, the checkpoint is
  informative with a **different** ordering. That is reported as
  `refuted_permuted`, which is a materially different diagnosis from a bad model
  and is not allowed to pass.

Verdicts: `confirmed` · `refuted_permuted` · `inconclusive`. Only `confirmed`
opens the gate; the other two leave `*_label_order: unknown`.

Check E then samples from the **untouched generator** with the released sampling
config and scores it. Near-chance accuracy there, with classifiers that already
passed C and D, indicates the generator is not the UnlearnCanvas-fine-tuned
model — the failure mode that would otherwise stay invisible until the baseline
numbers turned out meaningless.

Note also that `STYLES_AVAILABLE` contains 51 entries — the 50 painting styles
plus `Seed_Images` — while the benchmark's unlearn/retain splits use 12 + 12
styles. Upstream deletes the downloaded `style60.pth` and uses `style50.pth`,
so the 51-way head is the expected one.

---

## 7. Re-fetch instructions

```bash
# 1. obtain (any route): upstream's documented steps, a browser download copied
#    to the §2 paths, or links supplied by the maintainers.
bash scripts/fetch_assets.sh          # skips anything already present

# 2. record what we actually hold  -> PRESENT_UNVERIFIED
python scripts/record_asset_hashes.py \
    --source-url "<where it actually came from>" --revision "<rev, if any>"

# 3. (re)build the labelled held-out probe set, if not already present
python scripts/build_heldout_eval_set.py --shards 8 --per-class 2

# 4. validate  -> VERIFIED, or an explicit failure with unknowns retained
python scripts/validate_assets.py --gpu <idle-gpu-index> --update-manifest

# 5. confirm the gate is open
python scripts/preflight_check.py --gpu <idle-gpu-index> \
    --json /data/bijaypandey/cuig_pilot/preflight/preflight_assets.json

# 6. only now is the baseline runnable
bash scripts/run_single_concept_conabl.sh --style Abstractionism
```

Steps 4 and 5 are not optional: `run_single_concept_conabl.sh` reads the
validation receipt and exits 5 if the assets are not verified.

---

## 8. Other large local paths (all git-excluded)

| Path | Contents | Size |
|---|---|---|
| `/data/bijaypandey/cuig_pilot/CUIG` | upstream checkout @ `9932ac3` | 68 MB |
| `/data/bijaypandey/cuig_pilot/venv-cuig` | isolated Python env | ~9 GB |
| `/data/bijaypandey/cuig_pilot/Generators_substitute/sd-v1-5` | base SD v1.5 (mechanics check only) | 5.2 GB |
| `/data/bijaypandey/cuig_pilot/hf_home` | HuggingFace cache | varies |
| `/data/bijaypandey/cuig_pilot/mechanics_check` | substitute-run outputs | small |
| `/data/bijaypandey/cuig_pilot/heldout_eval_set` | labelled held-out probe images + `labels.json` | 15 MB |
| `/data/bijaypandey/cuig_pilot/uc_dataset_probe` | cached UnlearnCanvas parquet shards | ~4 GB |
| `/data/bijaypandey/cuig_pilot/outputs` | experiment outputs (empty until assets arrive) | — |

---

## 9. Machine-readable manifest

[`../assets/asset_manifest.json`](../assets/asset_manifest.json) is the
authoritative structured record, regenerated by
`scripts/make_asset_manifest.py` (class lists are read from the pinned upstream
`constants.py`, never transcribed). It carries, per asset: role, required
filename, expected local path, expected architecture, documented sources with
observed HTTP status, and a `download` block for source URL, revision, local
path, size and SHA-256. It also carries the ordered 51-style and 20-object class
mappings and the exact evaluation preprocessing.

Unestablished values are stored as
`{"value": null, "status": "unknown", "how_to_resolve": "..."}` — notably the
generator's version, EMA/non-EMA selection and finetune config, and both
classifiers' class index order. They are never filled with a guess.
