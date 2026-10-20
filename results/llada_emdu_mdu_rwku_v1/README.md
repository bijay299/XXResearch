# EMDU vs MDU on RWKU, LLaDA-8B (`llada_emdu_mdu_rwku_v1`)

**Outcome: `PAPER_ALIGNED_RUN_BLOCKED`.** The protocol, data, both objectives and the
evaluation are implemented, frozen and verified — but the comparison was **not run**,
because full-parameter training of the paper-aligned recipe measurably does not fit the
authorized envelope on 2×40 GiB devices.

| Phase | Status |
|---|---|
| EMDU author code available | ❌ No — labeled paper-based reimplementation |
| Data (582 pairs/subject) | ✅ PASS — exactly 582 verified, all 3 subjects |
| Frozen protocol + eval IDs | ✅ FROZEN before any Stage B output |
| Objective correctness | ✅ PASS — 24/24, incl. vs MDU's **executed** native KL |
| Runtime correctness | see `outputs/stageb_runtime_checks.json` |
| Full-parameter feasibility | 🔴 **BLOCKED** (measured) |
| EMDU vs MDU comparison | ⛔ **NOT RUN** |

Not a reproduction of the paper's tables, not SOTA, not a new contribution. No
forgetting, locality or utility number is produced. The earlier qualification study
remains `INSUFFICIENT_BASE_RECALL`, unchanged.

## The measured blocker in one table

Memory **fits** (peak 34.71 / 15.07 GiB of 40). Model math is cheap (~1.0–1.6 s/update).
But 8 B params of fp32 AdamW state (~90 GiB) cannot sit on 2×40 GiB cards, so it must go
to host RAM — and the resulting update cost is bandwidth-bound:

| CPU threads | per update | 240 updates → GPU-s (cap 20,729) | → CPU core-h (cap 20) | fits both |
|---|---|---|---|---|
| 255 | 51.9 s | 24,927 ✗ | 873.3 ✗ | no |
| 32 | 43.6 s | 20,923 ✗ | 98.4 ✗ | no |
| 8 | 43.4 s | 20,831 ✗ | 30.2 ✗ | no |

The one permitted repair gave **1.01×** and wall time is flat across 8→255 threads, so
this is a bandwidth limit, not an implementation defect. Training alone exceeds the GPU
budget before any of the 1,029-generation evaluation.

**Not done:** no LoRA/quantization substitution, no trimmed subjects or updates, no
shortened recipe, no open-ended FSDP/ZeRO rescue.

## Contents

| Path | What |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Full findings, method difference, reconstruction list, measured blocker, accounting |
| [`RESOURCE_LEDGER.md`](RESOURCE_LEDGER.md) | Carried-forward budgets, attempt ledger, measured vs unmetered |
| `frozen/` | Protocol spec, complete panel row IDs/counts, MMLU and TriviaQA panels (frozen pre-output) |
| `scripts/` | `objectives.py` (MDU + EMDU), verifiers, feasibility probes, attempt runner, ledger |
| `outputs/` | Feasibility measurements (both attempts), objective and runtime check records |
| `logs/` | One preserved log per attempt, including the failed repair |
| `evidence/` | Attempt ledger, extracted paper text, exit-code negative controls (valid and invalid) |
| [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) | Every published file with SHA-256, generated from tracked files |
