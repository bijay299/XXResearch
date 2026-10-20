# LLaDA Access & Engineering Feasibility Check (`llada_access_smoke_v1`)

Bounded smoke test run on hyperplane, 2026-10-19 (~0.38 h, 0.042 GPU-h of a 2 GPU-h cap).

> **Amended 2026-10-20 — see [`AMENDMENT_01.md`](AMENDMENT_01.md).** Documentation and
> evidence-publication corrections only: the four run logs are now published (a `.gitignore`
> rule had silently excluded them), the full-parameter memory arithmetic is corrected, absence
> claims are narrowed to what was inspected, the TOFU corpus size is corrected to 4,000 QA
> pairs, and accounting figures are labelled measured vs estimated vs unmetered.
> **All five statuses and every measured figure are unchanged. No new model execution occurred.**

**Outcome: inference and LoRA training passed; benchmark reproduction remains blocked by an
unresolved starting checkpoint.**

| Check | Status |
|---|---|
| Model access | ✅ PASS — both checkpoints ungated/MIT, Instruct downloaded, 6/6 shard hashes verified |
| Native inference | ✅ PASS — 4/4 prompts valid via the author's masked-diffusion `generate()`, 15.0 GiB peak |
| One-step LoRA training | ✅ PASS — **training mechanics only**; 5/5 gates, base model byte-identical across all 291 tensors, 17.1 GiB peak |
| Benchmark asset readiness | ⚠️ PARTIAL — data/code/licenses accessible; **no author-matched LLaDA TOFU checkpoint was located in the inspected release or recorded search** |
| Full baseline reproduction | ⛔ **NOT ATTEMPTED** |

This is a smoke test. It is **not** evidence of unlearning efficacy, **not** benchmark accuracy
or a paper reproduction, and does **not** show that full-parameter MDU training fits our
hardware (the training gate used LoRA at 0.10% of parameters).

## Contents

| File | What it is |
|---|---|
| [`AMENDMENT_01.md`](AMENDMENT_01.md) | **Amendment 01** (2026-10-20): corrected claims, recovered-log inventory, remaining uncertainty, frozen-manifest qualifications |
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Full findings, per-check scope, failures/retries, resource use, next recommendation |
| [`BENCHMARK_ASSET_MATRIX.md`](BENCHMARK_ASSET_MATRIX.md) | Asset matrix for a possible future LLaDA unlearning comparison, plus the three blockers |
| [`RUN_MANIFEST.md`](RUN_MANIFEST.md) | Frozen **before** model execution (sha256 `3c9fee37…`): planned inference, training fixture, settings, limits, retry policy |
| [`RESOURCE_LEDGER.md`](RESOURCE_LEDGER.md) | Final accounting, every GPU charge incl. the failed attempt, and what the enforcement actually covers |
| [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) | SHA-256 of every published file; pinned revisions; local-only and unavailable artifacts |
| `scripts/` | `run_inference.py`, `run_lora_step.py`, plus `env.sh` / `bounded.sh` / `ledger.sh` / `download.sh` |
| `evidence/` | Inference outputs, LoRA gate results, supervised-mask breakdown, HF API metadata, TOFU row counts, adapter config, `pip_freeze.txt` |
| `logs/` | **Original, unfiltered** run logs plus the bounded-timeout charge log (published by Amendment 01). Note: the failed inference attempt's own log was overwritten at run time and is not recoverable — `bounded.log` still records that attempt |

## Reproducing

Requires an A100-40GB-class GPU and ~15 GiB for weights.

```sh
bash scripts/download.sh            # pinned to revision 08b83a6f, verify hashes vs EVIDENCE_INDEX.md
source scripts/env.sh               # redirect all caches under /data
bash scripts/bounded.sh 1800 gpu_inference python scripts/run_inference.py
bash scripts/bounded.sh 1800 gpu_lora      python scripts/run_lora_step.py
```

Environment: Python 3.10.12, torch 2.4.1+cu124, transformers 4.49.0, peft 0.14.0
(see `evidence/pip_freeze.txt`). Note this is **not** MDU's pinned stack (torch 2.7.0 /
transformers 4.57.0); paths in the scripts are absolute to this host.
