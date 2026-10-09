# FINMLResearch — sequential concept erasure in diffusion models

Research pilot on **sequential concept erasure** in text-to-image diffusion
models. This stage establishes **feasibility and measurement reliability** for a
single-concept ConAbl baseline on UnlearnCanvas, before any new method is
developed. It is not a reproduction of published results.

Upstream method/benchmark code is **not vendored here**. It is cloned at a
pinned commit outside the repo; this repository holds only our configuration,
launch scripts, validators, measurements and notes.

---

## Current status (2026-10-08)

| Item | Status |
|---|---|
| Isolated environment, versions pinned to upstream `env.yaml` | **done** |
| Upstream CUIG pinned, licence + attribution preserved | **done** |
| Training / evaluation code inspected, label agreement analysed | **done** |
| Single-GPU single-concept launcher (Slurm removed, settings preserved) | **done** |
| GPU idle-detection and launch guard | **done** |
| UnlearnCanvas generator + classifiers | **BLOCKED — not obtainable on this host** |
| Pipeline mechanics check (substitute SD-1.5, GPU 2) | **done, passed** |
| `evaluate.py` code-path software test (synthetic, isolated) | **done, passed** |
| Machine-readable asset manifest with explicit `unknown` fields | **done** |
| Asset gate — preflight and runner reject unverified assets | **done, negative-control tested** |
| Labelled held-out validation set (137 imgs / 10 styles / 20 objects) | **done** |
| Asset-request draft to maintainers | **done — DRAFT, not sent** |
| UnlearnCanvas generator + classifiers | **BLOCKED — not obtainable on this host; cause unresolved** |
| **Single-concept ConAbl baseline** | **NOT RUN — waiting on assets** |
| **UA / IRA / CRA (target suppression, retention)** | **UNAVAILABLE — waiting on assets** |

The blocker is asset availability, not code or environment. See
[`docs/ASSETS.md`](docs/ASSETS.md) for the manifest, the exact errors observed,
and what must be dropped where to unblock.

### The asset gate

Assets are `MISSING` → `PRESENT_UNVERIFIED` → `VERIFIED`. Presence is never
enough: a classifier with a permuted class order, or a generator never fine-tuned
on UnlearnCanvas, is shape-identical to the real thing and would produce
confident, wrong numbers. Only `scripts/validate_assets.py` can write `VERIFIED`,
and both `preflight_check.py` and `run_single_concept_conabl.sh` refuse to
proceed without its receipt (the runner exits 5).

Verified by negative control: randomly-initialised heads of the *correct shape*
score at chance and are rejected — `shape_compatible: true`,
`classifiers_verified: false`, `UA_IRA_CRA: UNAVAILABLE`.

---

## Layout

```text
.
├── assets/
│   └── asset_manifest.json                  # machine-readable; explicit "unknown" fields
├── configs/
│   └── accelerate_single_visible_gpu.yaml   # upstream single_gpu.yaml, gpu_ids: all
├── docs/
│   ├── ASSETS.md                            # sources, errors, validation gate, manifest guide
│   ├── DEVIATIONS.md                        # upstream differences; preserved vs changed
│   ├── CODE_NOTES.md                        # findings from reading the CUIG code
│   └── outbox/
│       └── asset-request-DRAFT.md           # maintainer request — DRAFT, NOT SENT
├── env/
│   └── requirements.lock.txt                # exact resolved versions (pip freeze)
├── scripts/
│   ├── pilot_config.sh                      # paths; no secrets
│   ├── setup_env.sh                         # build the isolated venv
│   ├── fetch_assets.sh                      # download UnlearnCanvas assets
│   ├── gpu_select.sh                        # idle-GPU detection for a shared host
│   ├── make_asset_manifest.py               # regenerate the manifest from pinned constants
│   ├── record_asset_hashes.py               # source/revision/path/size/SHA-256 once downloaded
│   ├── build_heldout_eval_set.py            # labelled held-out images from the HF dataset
│   ├── validate_assets.py                   # THE gate: label order, accuracy, generator
│   ├── preflight_check.py                   # env / upstream / assets / receipt / labels
│   ├── run_single_concept_conabl.sh         # THE baseline launcher (one GPU, one concept)
│   ├── mechanics_check.sh                   # substitute-config smoke run (not a baseline)
│   ├── software_test_evaluate_path.sh       # isolated evaluate.py code-path test
│   └── verify_checkpoint.py                 # delta.bin save/reload verification
├── results/
│   ├── assets/validation_receipt.json       # asset gate verdict
│   ├── mechanics_check/                     # substitute-run measurements (JSON only)
│   └── software_test/                       # evaluate.py code-path test
└── third_party/CUIG/
    ├── UPSTREAM.md                          # pinned commit, provenance, attribution
    └── LICENSE.upstream                     # verbatim upstream MIT licence
```

Large artefacts live on `/data/bijaypandey/cuig_pilot/` and are git-excluded —
the filesystem holding `/home` is at 100% capacity. Paths are documented in
[`docs/ASSETS.md`](docs/ASSETS.md).

---

## Host

| | |
|---|---|
| Host | `hyperplane`, Ubuntu 22.04.5, Linux 5.15.0-136 |
| GPUs | **4 × NVIDIA A100-SXM4-40GB** — verified |
| GPU memory | 40960 MiB per device (40 GB) |
| Driver / CUDA | 550.144.03 / 12.4 |
| CPU / RAM | 255 cores / 503 GB |
| Scheduler | **none** — GPUs are shared informally; always check before launching |
| Storage | `/` 1.8 T **100% used, 5.8 G free** · `/data` 14 T, 1.4 T free |

**Verified GPU inventory** (`nvidia-smi --query-gpu=index,name,memory.total,uuid`):

| Idx | Name | Memory | UUID |
|---|---|---|---|
| 0 | NVIDIA A100-SXM4-40GB | 40960 MiB | `GPU-b2f89296-447e-dc7a-96b6-82ac699a02d7` |
| 1 | NVIDIA A100-SXM4-40GB | 40960 MiB | `GPU-16080ca5-6073-f2c8-04d7-368d4a510a2a` |
| 2 | NVIDIA A100-SXM4-40GB | 40960 MiB | `GPU-3594d682-af4b-13e1-3266-8a55455d669d` |
| 3 | NVIDIA A100-SXM4-40GB | 40960 MiB | `GPU-17211a38-e5ad-e9bc-eeda-aa3caa659d27` |

> The original project brief specified 4 × RTX A6000 48 GB. The hardware actually
> present is **A100-SXM4-40 GB**, confirmed by direct query. The per-device
> memory budget is **40 GB, not 48**. This table is the record of record.

### GPU etiquette

Other researchers share these GPUs. Never terminate another workload.
`scripts/gpu_select.sh` treats a GPU as idle only when it has **no compute
process**, stays under 1 GB used, **and** stays under 5% utilisation across
repeated samples — because zero instantaneous utilisation does not mean idle,
and free memory does not mean idle either. Both launchers re-verify
immediately before starting and refuse to run otherwise.

---

## Quick start

```bash
# 1. environment (once)
bash scripts/setup_env.sh
source /data/bijaypandey/cuig_pilot/venv-cuig/bin/activate

# 2. assets  (currently blocked — see docs/ASSETS.md)
bash scripts/fetch_assets.sh

# 3. record what we actually hold -> PRESENT_UNVERIFIED (+ SHA-256)
python scripts/record_asset_hashes.py --source-url "<origin>" --revision "<rev>"

# 4. labelled held-out probe set (already built; rerun to enlarge)
python scripts/build_heldout_eval_set.py --shards 8 --per-class 2

# 5. THE GATE -> VERIFIED, or explicit failure with unknowns retained
python scripts/validate_assets.py --gpu "$(bash scripts/gpu_select.sh --pick)" \
    --update-manifest

# 6. confirm the gate reads OPEN
python scripts/preflight_check.py --gpu "$(bash scripts/gpu_select.sh --pick)"

# 7. only now is the baseline runnable (exits 5 otherwise)
bash scripts/run_single_concept_conabl.sh --style Abstractionism
```

`run_single_concept_conabl.sh` picks an idle GPU itself (or takes `--gpu N`),
pins it with `CUDA_VISIBLE_DEVICES`, runs train → sample → evaluate, and writes
`run_manifest.json` with GPU identity, peak memory, per-stage timings and the
checkpoint location.

---

## What this pilot does not claim

- The mechanics check used **base Stable Diffusion v1.5**, not the UnlearnCanvas
  generator. It demonstrates that the code path runs; it says nothing about
  unlearning quality. Preserved-vs-changed settings are tabulated in
  [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) §6.
- The `evaluate.py` software test used **randomly-initialised** classifier heads
  in an isolated directory. Its accuracies are noise.
- No UA / IRA / CRA has been measured. No baseline has been run.
- The Drive 404s are an **observation, not a diagnosis** — we have not
  established whether the folders were withdrawn, re-permissioned, or are simply
  unreachable from this network.
- "No classifier mirror found" is bounded to the sources searched
  ([`docs/ASSETS.md`](docs/ASSETS.md) §4.2), not a claim that none exists.
- A locally computed SHA-256 identifies a file; it does **not** prove the file is
  the authors' checkpoint. Provenance stays `unknown` until a maintainer
  supplies a checksum or a direct copy.

---

## Credits

Built on **CUIG** (Lee et al., ICLR 2026) — MIT licence — which builds on
**Concept Ablation** (Kumari et al.) and the **UnlearnCanvas** benchmark
(Zhang et al., 2024). Full attribution in
[`third_party/CUIG/UPSTREAM.md`](third_party/CUIG/UPSTREAM.md).
