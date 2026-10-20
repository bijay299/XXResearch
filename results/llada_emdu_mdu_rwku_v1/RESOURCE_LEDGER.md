# Resource Ledger — shared envelope, carried forward

The original task clock was **not restarted**. Budgets carry forward from
`results/llada_rwku_baseline_v1` (which charged 323 GPU-s).

```
=== SHARED LEDGER (carried forward, clock NOT restarted) ===
elapsed        :   1.199 h / 24 h      remaining  22.801 h
GPU total      :   1883 s = 0.5231 h / 6 h   remaining  19717 s = 5.4769 h
  of which carried from llada_rwku_baseline_v1 : 323 s
  charged this task                            : 1560 s
CPU (new work) :  43592 core-s = 12.1089 core-h / 20 core-h   [MEASURED from here on]
  historical CPU (previous study)              : UNMETERED - no retrospective charge invented
storage        :   0.243 GiB / 10 GiB
attempts: 6
```

## Classification

| Figure | Basis |
|---|---|
| Aggregate GPU seconds | **measured** — wall-clock × devices allocated, charged for **every** attempt including the failed repair |
| Elapsed | **measured** — from the original frozen `START_EPOCH` (2026-10-20T00:31:58), not a restarted clock |
| CPU core-seconds, this task | **measured** — `/usr/bin/time` user+sys per attempt |
| CPU core-seconds, previous study | ⚠️ **UNMETERED and unknown** — no retrospective charge invented, and no claim of comprehensive CPU accounting |
| Storage | **measured** — `du -sb` across both workspaces |
| Concurrent GPUs | **measured** — 2 maximum (devices 2 and 3) |

## Enforcement

**Enforced:** per-attempt wall-clock timeouts via coreutils `timeout`, with each GPU
attempt's timeout **clamped to the remaining aggregate GPU budget** before launch; a
refusal guard that aborts rather than overwrite an existing log; and atomic `mkdir`-based
attempt-ID allocation (fixing the previous study's racy line-count allocator).

**Advisory, not enforced:** the aggregate elapsed, CPU-core-hour and storage caps are not
scheduler or cgroup quotas. They are computed and checked between attempts; nothing would
terminate the session mid-attempt if an aggregate were crossed.

## Device restrictions

GPUs **2 and 3** only, per the directly recorded user override ("override and use gpu 2 3",
"leave gpu 0 1"). Availability was rechecked immediately before each allocation. No other
user's process was signalled or killed at any point. All work, caches and temporary files
under `/data`.

## Attempt ledger

See `evidence/ATTEMPTS.tsv`. Every attempt has its own preserved log, including:

- `b0002` full-parameter feasibility probe, attempt 1 (rc=0) — 43.9–67.8 s/update
- `b0003` full-parameter feasibility **repair**, attempt 2 (**rc=3**, by design: no thread
  setting fits both budgets) — the single permitted technical repair; **both attempts preserved**
- `b0005` objective checks (24/24), `b0006` runtime checks (12/12)

*Disclosed cosmetic defect:* one `cpu_core_s` cell renders as `026063` (leading zero from
`awk` formatting) instead of `26063`. Other values are correct; the raw logs are authoritative.
