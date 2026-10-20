# EMDU vs MDU on RWKU, LLaDA-8B (`llada_emdu_mdu_rwku_v1`)

**Outcome: `PAPER_ALIGNED_RUN_BLOCKED`** — scoped to **the tested custom CPU-offload
implementation**. The protocol and data are frozen and both objectives are implemented and
checked; the evaluation panels and scorers are **configured but never executed**. The
comparison was **not run** because that implementation (including its one permitted repair)
did not fit the remaining envelope, as measured.

> **Amended 2026-10-20 — see [`AMENDMENT_01.md`](AMENDMENT_01.md).** Corrected CPU total
> (octal bug: **58,340 core-s = 16.2056 core-h**, previously 43,592 / 12.1089 — a 4.0967
> core-h undercount); **withdrew** the single-80 GB-GPU recommendation and its "minutes"
> prediction; **withdrew** the bandwidth-bound attribution and the native batch-reduction
> equivalence claim; narrowed fidelity, coverage and chronology. Outcome unchanged.

The measurements do **not** prove universal full-parameter infeasibility on two 40 GB GPUs
and qualify **no** optimized offload or sharding solution. **No EMDU-versus-MDU behavioral
comparison occurred**, and neither this outcome nor Stage A's establishes which method is
better.

| Phase | Status |
|---|---|
| EMDU author code available | ❌ Not located in sources inspected 2026-10-20 — labeled paper-based reimplementation |
| Data (582 pairs/subject) | ✅ PASS — exactly 582 verified, all 3 subjects |
| Frozen protocol + eval IDs | ✅ FROZEN before any Stage B output |
| Objective correctness | ✅ PASS — 24/24 observed, incl. vs MDU's **extracted** native KL helper (complete native trainer never executed) |
| Runtime correctness | ✅ PASS — 12/12 observed; both forwards ran, optimizer updates **EMDU-only (2)**, digests **40/291** tensors |
| Full-parameter feasibility, **tested CPU-offload impl.** | 🔴 **BLOCKED** (measured; not a universal result) |
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

The one permitted repair gave **1.01×** and wall time is flat across 8→255 threads. That is
*consistent with* a transfer limit but **does not establish one** — thread count was the only
dimension varied, and pinned/overlapped transfers, fused kernels, sharded optimizer state,
ZeRO/FSDP and 8-bit states were never tried. Training alone exceeds the GPU budget before any
of the 1,029-generation evaluation.

**Not done:** no LoRA/quantization substitution, no trimmed subjects or updates, no
shortened recipe, no open-ended FSDP/ZeRO rescue.

## Contents

| Path | What |
|---|---|
| [`AMENDMENT_01.md`](AMENDMENT_01.md) | **Amendment 01** (2026-10-20): corrected accounting, withdrawn hardware claim, narrowed fidelity/coverage/chronology |
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Full findings, method difference, reconstruction list, measured blocker, accounting |
| [`RESOURCE_LEDGER.md`](RESOURCE_LEDGER.md) | Carried-forward budgets, attempt ledger, measured vs unmetered |
| `frozen/` | Protocol spec, complete panel row IDs/counts, MMLU and TriviaQA panels (frozen pre-output) |
| `scripts/` | `objectives.py` (MDU + EMDU), verifiers, feasibility probes, attempt runner, ledger |
| `outputs/` | Feasibility measurements (both attempts), objective and runtime check records |
| `logs/` | One preserved log per attempt, including the failed repair |
| `evidence/` | Attempt ledger, extracted paper text, exit-code negative controls (valid and invalid) |
| [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) | Every published file with SHA-256, generated from tracked files |
