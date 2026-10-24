# Resource note — two accounting failures, both mine, and where things stand

**Date:** 2026-10-24

## 1. This phase had no resource accounting at all

The authorization for this work includes *"continue recording all resource consumption without
resetting previous records."* The `mdlm_unlearn` phase has a proper ledger — `GPU_SECONDS`,
`CPU_SECONDS`, `ATTEMPTS.tsv`, `PROJECT_START`, and a `ledger.sh` that deliberately shows both the
counter total and the attempts-derived total so the gap between them stays visible.

**`t2i_unlearn` had none of that.** No counters, no attempts log, no start timestamp. The
`./ledger.sh` figures I quoted in status reports throughout this phase were `mdlm_unlearn`'s and did
**not** cover this work. That is a reporting error on my part, not a gap in the record I inherited.

`ledger.sh` and `PROJECT_START` now exist here. Nothing was reset. The GPU figure it reports is
**reconstructed after the fact and is a floor**: only 328 of 925 result JSONs carry a runtime field,
so

> **≥ 21.7 GPU device-hours over a 47.9 h span** — a floor, with the true figure higher by an
> unknown amount.

It is labelled as a floor in the tool's own output and should be quoted that way, never as a
measurement.

## 2. Storage exceeded the 60 GiB authorization

Authorized: 60 GiB. Measured at the time of writing: **120 GiB**, i.e. **2.0× over**. `/data` was
simultaneously at **100 % capacity with 108 GB free on a 14 TB shared filesystem**, which is a risk
to other users' jobs, not only to this project.

Reclaimed immediately, keeping only artifacts whose measurements were **not** already recorded:

| removed | size | why it was safe |
|---|---|---|
| 16 `attacked/DOSE*` attacker-cost checkpoints | ~27 GiB | all 94 measurements recorded in `outputs/`; curves published in `FINDING37`, `FINDING52` |
| 2 `attacked/DIAG_H66_*` checkpoints | ~3.4 GiB | the re-zero study they fed is **complete** (`FINDING57`) |
| 3 `adapters/_rezero*` re-zeroed copies | ~5 GiB | built *and* evaluated; both recorded |
| `scope_slices/` | 4.1 GiB | derived slices, regenerable from `adapters/` |

**Now at 81.7 GiB — still 1.36× over the authorization.** The remainder is:

| | size | notes |
|---|---|---|
| `adapters/` | 57 GiB | 61 erasure checkpoints × ~1.7 GiB (full UNet state dicts) |
| `attacked/` | 13 GiB | 8 `KEEP_*` post-attack checkpoints for cross-arm weight diagnostics |
| `hf_home/` | 7.5 GiB | the pinned SD-1.4 download; needed |
| `venv_t2i/` | 5.2 GiB | the environment; needed |

**I have not deleted any `adapters/` entry.** Deleting one is irreversible and forfeits the ability
to re-attack that arm — and `FINDING35` §4's ratchet means every sub-ceiling number in this phase is
provisional precisely *because* arms may need to be attacked harder later. Throwing away the arm
makes that impossible. So this is flagged for a decision rather than resolved unilaterally.

**The proposal, if a cut is wanted:** ~30 GiB is recoverable by deleting the checkpoints of
superseded ladder rungs and closed-study arms whose selected operating point was never the headline
(the `h68v2_*`/`h69v2_*` sweeps, the destroyed ρ = 10 %/20 % rungs, the `undo_a*` series). Each
build recipe is recorded, so any of them can be rebuilt from `scripts/` at known cost rather than
recovered. That would land the phase at ~52 GiB, inside the authorization.

## 3. What is in place so the first failure cannot recur

`ledger.sh` reports the floor **and** says it is a floor and why, in its own output, rather than
printing a clean number that overstates what is known — the same reason `mdlm_unlearn`'s ledger
prints two totals instead of one.
