# Resource Ledger — LLaDA access & feasibility smoke test

Established before GPU execution. `START_EPOCH` frozen at the true task start
(2026-10-19T23:22:10-04:00), not at ledger-creation time.

## Final accounting

| Budget | Cap | Actual | Fraction |
|---|---|---|---|
| Elapsed | 3.00 h | ~0.38 h | 13% |
| Aggregate GPU | 2.00 GPU-h | **152 GPU-s = 0.0422 GPU-h** | 2.1% |
| CPU | 8.00 core-h | ~0.12 core-h (201 core-s measured on GPU jobs) | ~1.5% |
| Additional storage | 60 GiB | ~21 GiB | 35% |
| Concurrent GPUs | 1 default / 2 max | 1 (`CUDA_VISIBLE_DEVICES=0`) | — |

## GPU charges (every attempt, failures included)

| # | Label | rc | Duration | Timeout | Charged |
|---|---|---|---|---|---|
| 1 | `gpu_inference` | **1 (FAILED)** | 3 s | 1800 s | ✅ yes |
| 2 | `gpu_inference` | 0 | 18 s | 1800 s | ✅ yes |
| 3 | `gpu_lora` | 0 | 131 s | 1800 s | ✅ yes |
| | | | **152 s total** | | |

Failed attempt #1 is charged, not forgiven. Budgets were never extended or reset.

## What the enforcement actually covers

- `scripts/bounded.sh` — wraps each GPU job in coreutils `timeout` (SIGTERM, then SIGKILL
  after a 30 s grace) and adds that attempt's wall-clock to a cumulative `GPU_SECONDS`
  counter **including failures**. Per-job ceiling: 1800 s. This is a real, enforced limit
  on any single job.
- `scripts/ledger.sh` — reports elapsed time (from the frozen `START_EPOCH`) and workspace
  size against the caps; checked between phases.
- **Not covered:** these are not scheduler quotas or cgroup limits. Nothing would have
  killed the session mid-phase if an *aggregate* had been crossed. CPU-core-hours and
  storage were **measured and reported, not hard-capped**. Every budget finished far under
  cap, so the distinction was never load-bearing in this run.

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

## GPU selection and courtesy

- Idleness re-checked immediately before each allocation.
- GPU 0 used throughout (14 MiB used, 0% util at each check).
- GPUs 2 and 3 carried another user's processes (PIDs 2059679, 2064108, 2064109) at
  ~81-94% utilization for the entire window. Never killed, never interfered with.
- A suitable GPU was available immediately, so the 10-minute availability rule was never
  triggered and no poller was left running.
