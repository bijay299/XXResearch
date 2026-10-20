# Resource Ledger — LLaDA access & feasibility smoke test

Established before GPU execution. `START_EPOCH` frozen at the true task start
(2026-10-19T23:22:10-04:00), not at ledger-creation time.

> **Amended 2026-10-20** ([`AMENDMENT_01.md`](AMENDMENT_01.md) Correction 4): every figure is
> now labelled **measured / estimated / unmetered**, and **enforced** limits are separated from
> **advisory** checks. The recovered `logs/bounded.log` corroborates the GPU-second accounting
> exactly. **No figure was revised and no retrospective measurement was invented.**

## Final accounting

| Budget | Cap | Actual | Classification | Fraction |
|---|---|---|---|---|
| Elapsed | 3.00 h | ~0.38 h | **measured** — wall-clock from frozen `START_EPOCH` | 13% |
| Aggregate GPU | 2.00 GPU-h | **152 GPU-s = 0.0422 GPU-h** | **measured** — corroborated by `logs/bounded.log` | 2.1% |
| CPU (GPU jobs only) | — | **201.25 core-s** | **measured** — `/usr/bin/time -v` / `getrusage`, two successful jobs | — |
| CPU (total) | 8.00 core-h | ~0.12 core-h | ⚠️ **estimate, incomplete coverage** — not comprehensive metering | ~1.5% |
| Additional storage | 60 GiB | ~21 GiB | **measured** — `du -sb` of workspace | 35% |
| Concurrent GPUs | 1 default / 2 max | 1 (`CUDA_VISIBLE_DEVICES=0`) | **measured** | — |

## GPU charges (every attempt, failures included)

Reproduced verbatim from the recovered `logs/bounded.log`
(sha256 `57106f0552675740c1921741fa092788e6c12d42b6b19fe6593a2b0917f6c747`):

| # | Label | rc | Duration | Timeout | Charged |
|---|---|---|---|---|---|
| 1 | `gpu_inference` | **1 (FAILED)** | 3 s | 1800 s | ✅ yes |
| 2 | `gpu_inference` | 0 | 18 s | 1800 s | ✅ yes |
| 3 | `gpu_lora` | 0 | 131 s | 1800 s | ✅ yes |
| | | | **152 s total** | | |

The failed attempt #1 is charged, not forgiven. Budgets were never extended or reset.
**The recovered log corroborates all three charges**, so the 152 GPU-s figure is independently
verifiable from published evidence.

## CPU accounting — what is measured and what is not

| Component | Core-seconds | Source |
|---|---|---|
| Inference (successful attempt) | 44.09 (31.57 user + 12.52 sys) | **measured** — `logs/inference.log` |
| LoRA step | 157.16 (76.53 user + 80.63 sys) | **measured** — `logs/lora_step.log` |
| **Measured subtotal** | **201.25** | both corroborated by recovered logs |
| Failed inference attempt | ≈10.2 | ⚠️ **transcript-only** — its log was overwritten; **excluded** from the subtotal |
| venv creation + pip installs | not instrumented | ⚠️ **unmetered** |
| SHA-256 hashing (6-way parallel, model shards) | not instrumented | ⚠️ **unmetered** |
| `curl` downloads (14.94 GiB) | not instrumented | ⚠️ **unmetered** |
| Repo clones, audit queries, publication | not instrumented | ⚠️ **unmetered** |

**The ~0.12 core-h total is an estimate that combines the 201.25 measured core-seconds with
unmetered work. It must not be described as comprehensive metering.** The true total is
certainly higher than 201.25 core-s and remains comfortably below the 8 core-h cap, but the
margin is not precisely known.

## What the enforcement actually covers

**Enforced (a real limit):**
- `scripts/bounded.sh` wraps each GPU job in coreutils `timeout` (SIGTERM, then SIGKILL after a
  30 s grace) with a **1800 s per-job ceiling**. A single job could not exceed it.
- The same wrapper adds each attempt's wall-clock to a cumulative `GPU_SECONDS` counter
  **including failures**, so no attempt escaped accounting.

**Advisory only (not enforced):**
- The **aggregate** elapsed-time, aggregate GPU-hour, CPU-core-hour and storage limits were
  **not scheduler quotas and not cgroup limits**. `scripts/ledger.sh` computed and reported them
  between phases; nothing would have terminated the session had an aggregate been crossed
  mid-phase.
- CPU-core-hours and storage were **observed and reported, never capped**.

Every budget finished far under cap, so the distinction was never load-bearing in this run — but
a hard cap must not be inferred from advisory monitoring.

## Timeout and retry policy (as applied)

- Task-specific timeout established before GPU execution: 1800 s per GPU job.
- At most **one** technical retry before a successful training update. The LoRA training
  gate **passed first time** — that retry was never consumed, and no configuration was
  changed to obtain a passing result.
- The one failure (inference attempt #1, 3 s) was a cache-path failure on the full root
  filesystem, fixed by redirecting caches under `/data`. Both configurations are preserved:
  attempt #1 used the inherited `$HOME`-based cache defaults, attempt #2 used
  `scripts/env.sh`. **Attempt #2 passed.** The Python script itself was unchanged between
  them — only environment variables differed.
- ⚠️ **Attempt #1's own stdout/stderr was not preserved** (overwritten by attempt #2's redirect).
  Its existence, exit status, duration and GPU charge are corroborated by `logs/bounded.log`;
  its error text and CPU time survive only in the session transcript. No log was recreated.

## GPU selection and courtesy

- Idleness re-checked immediately before each allocation.
- GPU 0 used throughout (14 MiB used, 0% util at each check).
- GPUs 2 and 3 carried another user's processes (PIDs 2059679, 2064108, 2064109) at
  ~81-94% utilization for the entire window. Never killed, never interfered with.
- A suitable GPU was available immediately, so the 10-minute availability rule was never
  triggered and no poller was left running.

## Amendment 01 resource use

The 2026-10-20 documentation amendment consumed **zero GPU-seconds**: no model execution, no
training, no inference, no benchmark evaluation, no downloads of model weights. Work was limited
to filesystem inspection, hashing of already-present files, two small read-only metadata queries,
and `git` operations.
