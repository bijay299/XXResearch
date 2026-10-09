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
| **Single-concept ConAbl baseline** | **NOT RUN — waiting on assets** |
| **UA / IRA / CRA (target suppression, retention)** | **UNAVAILABLE — waiting on assets** |

The blocker is asset availability, not code or environment. See
[`docs/ASSETS.md`](docs/ASSETS.md) for the manifest, the exact errors observed,
and what must be dropped where to unblock.

---

## Layout

```text
.
├── configs/
│   └── accelerate_single_visible_gpu.yaml   # upstream single_gpu.yaml, gpu_ids: all
├── docs/
│   ├── ASSETS.md                            # asset manifest, sources, errors, validation
│   ├── DEVIATIONS.md                        # every difference from upstream, with reasons
│   └── CODE_NOTES.md                        # findings from reading the CUIG code
├── env/
│   └── requirements.lock.txt                # exact resolved versions (pip freeze)
├── scripts/
│   ├── pilot_config.sh                      # paths; no secrets
│   ├── setup_env.sh                         # build the isolated venv
│   ├── fetch_assets.sh                      # download UnlearnCanvas assets
│   ├── gpu_select.sh                        # idle-GPU detection for a shared host
│   ├── preflight_check.py                   # validate env / upstream / assets / labels
│   ├── run_single_concept_conabl.sh         # THE baseline launcher (one GPU, one concept)
│   ├── mechanics_check.sh                   # substitute-config smoke run (not a baseline)
│   ├── software_test_evaluate_path.sh       # isolated evaluate.py code-path test
│   └── verify_checkpoint.py                 # delta.bin save/reload verification
├── results/
│   └── mechanics_check/                     # substitute-run measurements (JSON only)
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
| GPUs | **4 × NVIDIA A100-SXM4-40GB** (driver 550.144.03, CUDA 12.4) |
| CPU / RAM | 255 cores / 503 GB |
| Scheduler | **none** — GPUs are shared informally; always check before launching |
| Storage | `/` 1.8 T **100% used, 5.8 G free** · `/data` 14 T, 1.4 T free |

> The project brief specified 4 × RTX A6000 48 GB. The host actually has
> A100-SXM4-**40 GB**, so the per-device memory budget is 40 GB, not 48.

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

# 2. assets  (currently blocked — see docs/ASSETS.md)
bash scripts/fetch_assets.sh

# 3. validate before spending GPU time
source /data/bijaypandey/cuig_pilot/venv-cuig/bin/activate
python scripts/preflight_check.py --gpu "$(bash scripts/gpu_select.sh --pick)"

# 4. the single-concept baseline (one GPU, one concept)
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
  unlearning quality.
- The `evaluate.py` software test used **randomly-initialised** classifier heads
  in an isolated directory. Its accuracies are noise.
- No UA / IRA / CRA has been measured. No baseline has been run.

---

## Credits

Built on **CUIG** (Lee et al., ICLR 2026) — MIT licence — which builds on
**Concept Ablation** (Kumari et al.) and the **UnlearnCanvas** benchmark
(Zhang et al., 2024). Full attribution in
[`third_party/CUIG/UPSTREAM.md`](third_party/CUIG/UPSTREAM.md).
