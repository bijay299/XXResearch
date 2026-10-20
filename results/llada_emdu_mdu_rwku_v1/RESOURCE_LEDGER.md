# Resource Ledger — CORRECTED (derived)

> **Corrected 2026-10-20 by [`AMENDMENT_01.md`](AMENDMENT_01.md).** The previously published
> CPU total was wrong: `scripts/ledger.sh` summed the attempt column with POSIX shell
> arithmetic, and attempt `b0003`'s raw cell `026063` was parsed as **octal**
> (`026063₈ = 11315₁₀`), silently losing 14,748 core-seconds.
>
> The original attempt rows, the historical counter files and `scripts/ledger.sh` are
> **preserved unchanged as historical evidence**. This file is the corrected *derived*
> ledger; the historical measurements are not edited or replaced.

The original task clock was **not restarted**. Budgets carry forward from
`results/llada_rwku_baseline_v1` (which charged 323 GPU-s). Original start
**2026-10-20T00:31:58**.

## CPU — corrected reconciliation

| attempt | label | raw cell | decimal core-s |
|---|---|---|---|
| `b0001` | stageA audit offline_recompute | `0` | 0 |
| `b0002` | feasibility fullparam_probe | `27233` | 27,233 |
| `b0004` | freeze protocol_and_eval_ids | `1` | 1 |
| `b0005` | verify objectives_cpu | `152` | 152 |
| `b0003` | feasibility fullparam_repair | **`026063`** | **26,063** |
| `b0006` | verify runtime_frozen_teacher | `4891` | 4,891 |

| | core-seconds | core-hours |
|---|---|---|
| **Corrected measured new CPU usage** | **58,340** | **16.2056** |
| Previously reported (octal-corrupted) | 43,592 | 12.1089 |
| **Undercount** | **14,748** | **4.0967** |

```
$((0+27233+1+152+026063+4891)) = 43592      # the bug: 026063 read as octal
0+27233+1+152+26063+4891       = 58340      # decimal truth
```

### Budget-compliance caveat — read this before quoting any remainder

**Historical CPU usage remains unmetered.** The earlier qualification study had no CPU
instrumentation, so its consumption is unknown and **no retrospective charge is invented**.
Consequently:

> **3.7944 core-hours is at most the remaining allowance before unknown earlier usage —
> NOT a verified remaining whole-envelope budget.**

**No claim of overall CPU-budget compliance is made.**

## GPU — reconciled, unchanged

| component | GPU-seconds |
|---|---|
| carried from `llada_rwku_baseline_v1` | 323 |
| `b0002` feasibility probe (274 s wall × 2 devices) | 548 |
| `b0003` feasibility repair (330 s wall × 2 devices) | 660 |
| `b0006` runtime checks (176 s wall × 2 devices) | 352 |
| **total** | **1,883** ≈ **0.5231 GPU-h** |
| remaining against the original 6 h (21,600 s) ceiling | **19,717 s** |

**Maximum two concurrent GPUs** throughout (devices 2 and 3, per the recorded user
override; devices 0–1 untouched). Every GPU attempt is charged wall × 2, failures included.

## Other budgets

| Budget | Cap | Actual | Basis |
|---|---|---|---|
| Elapsed | 24 h from 00:31:58 | ~1.2 h at publication | **measured** |
| Storage | 10 GiB | ~0.24 GiB | **measured** (`du -sb`) |
| Concurrent GPUs | 2 | 2 | **measured** |

## Measured versus estimated — important distinction

- **Measured CPU:** only the `/usr/bin/time` user+sys values in `evidence/ATTEMPTS.tsv`.
- **Estimated CPU:** the feasibility trade-off table's "CPU core-h" figures are
  **thread count × optimizer wall time** — a *cost estimate*, not measured core usage. It
  assumes every configured thread is fully busy for the whole optimizer phase.
- **Estimated:** all 240-update projections (training wall, GPU-s, core-h) are
  extrapolations from 1–2 measured updates per setting.

## Enforcement

**Enforced:** per-attempt wall-clock timeouts (coreutils `timeout`), with each GPU
attempt's timeout clamped to the remaining aggregate GPU budget before launch; a refusal
guard that aborts rather than overwrite an existing log; atomic `mkdir`-based attempt-ID
allocation.

**Advisory, not enforced:** the aggregate elapsed, CPU-core-hour and storage caps are not
scheduler or cgroup quotas. They were computed between attempts; nothing would terminate
the session mid-attempt if an aggregate were crossed. The CPU cap in particular was being
checked against an **undercounted** total, so it provided less protection than it appeared
to.

## Attempt ledger

`evidence/ATTEMPTS.tsv` — preserved verbatim, including the raw `026063` cell. Every
attempt has its own log in `logs/`, including `b0003` (the permitted repair, `rc=3` by
design). 11 engineering optimizer updates occurred in total (3 + 6 + 2); **0** of the
planned 240 scientific comparison updates ran.
