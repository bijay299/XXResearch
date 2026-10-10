# Resource policy change, 2026-10-10 — the GPU-hour ceiling is retired

**This is an operational record, not a protocol change.** Nothing about what we
measure, how a checkpoint is selected, which set is frozen, or what counts as
evidence changes here. Resource availability and protocol decisions are kept
apart on purpose: having free GPUs approves no experiment, and this document
approves none.

## 1. What the PI withdrew

> "We may use as much GPU compute as the research needs, provided the GPUs are
> not occupied by another user. The four-GPU-hour ceiling, test-budget
> withholding and requests for permission based solely on GPU-hours are
> withdrawn. Do not invent a replacement cap."

So, concretely:

| withdrawn | was |
|---|---|
| the hard ceiling | 4.000 device-hours total, counting failures and setup |
| the confirmatory reserve | 1.3474 device-hours withheld for the frozen-test stage |
| budget-derived runtime cutoffs | each stage's cost estimate doubled as a `timeout` |
| asking permission by GPU-hours | a request to the PI per block of hours |

**No replacement cap has been defined.** There is deliberately no new number,
no soft limit and no quota anywhere in the code.

## 2. What gates compute now

Verified GPU availability, and nothing else:

* `scripts/seq/run_diagnostic.sh:acquire_gpus` polls the real fail-closed
  selector (`scripts/gpu_select.sh`) over several samples before it will call a
  device idle. An unreadable or ambiguous reading is **not** availability — the
  selector fails closed.
* If the devices wanted are not verifiably idle, the runner **waits** (poll
  interval `SEQ_DIAG_WAIT_POLL_SECONDS`, bound `SEQ_DIAG_WAIT_SECONDS`) and then
  reports `QUEUED/BLOCKED` and exits. On this host, that poll *is* the queue.
* Nothing pre-empts, signals, or reconfigures another user's job, ever. Cleanup
  (`kill_tree`) only ever signals descendants of our own run.

Checked before this document was written: all four A100-SXM4-40GB devices read
14 MiB used, 0 % utilisation, no compute apps — i.e. idle. That is recorded as
status, not as an approval.

## 3. What changed in the code

| file | change |
|---|---|
| `scripts/seq/gpu_budget.py` | two modes. `recording_only` (new default) never refuses an admission, returns **no** `max_wall_seconds`, and never fails on a ceiling. `ceiling` is the retired mode, kept so the admission-race fix stays covered by tests. New `retire` subcommand switches an existing ledger over, idempotently. |
| `scripts/seq/run_diagnostic.sh` | `charge` records instead of gating; no `STAGE_TIMEOUT` is derived from an estimate; preflight explicitly **retires** whatever ledger it finds rather than leaving a 4.0-hour ceiling armed; `bounded` applies a limit only if `SEQ_STAGE_WATCHDOG_SECONDS` is set; interrupt cleanup retained. |
| `scripts/seq/exp_config.sh` | `SEQ_DIAG_CEILING_GPU_HOURS` and `SEQ_DIAG_TEST_RESERVE_GPU_HOURS` removed as live settings, with the figures recorded as history. `SEQ_STAGE_WATCHDOG_SECONDS` added, defaulting to unset. |

### The review finding this closes

> "Each child receives the entire evaluation slot timeout, so it does not
> enforce a whole-slot bound."

Correct, and it is now moot rather than re-tuned. An evaluation slot calls
`bounded` twice — once for generation, once for detection — so the old
budget-derived allowance could be spent twice over by one slot. With the cap
withdrawn there is nothing to enforce, so the cutoff is **removed** rather than
rebuilt: no stage inherits a time limit at all. `SEQ_STAGE_WATCHDOG_SECONDS`
remains available for an operator who wants one, and its docstring states
plainly that it is per call, not per slot. No budget controller was rebuilt.

## 4. What was explicitly preserved

* **Every historical usage record.** `retire` keeps all entries byte for byte
  and stamps the prior ceiling (4.0), the prior reserve (1.3474), the entry
  count and the spend at the moment of the switch, so the policy change cannot
  become a cover for editing the record. The original run's 27 stages and
  1.2022 device-hours are intact.
* **Carry-forward.** A new output directory still cannot restore a zero
  balance, and `init` still refuses to reset an existing ledger.
* **Recording of failures and orphans.** Failed stages are still recorded at
  their actual occupancy; a stage interrupted between start and finish is still
  reconciled conservatively rather than vanishing.
* **Ordinary process cleanup.** An interrupt still stops this run's own
  trainers and generators so no device is left held.
* **Estimates**, as *information*: they sit beside the actual occupancy so
  estimate-versus-actual stays reportable and the cost model can improve. They
  admit nothing and bound nothing.

## 5. Verification

CPU only; no GPU work was done for this change.

| suite | assertions |
|---|---|
| `scripts/seq/test_gpu_budget.py` | 59 (was 37) |
| `scripts/seq/test_diagnostic_runner.sh` | 104 (was 62) |

New coverage specific to this change: a new ledger defaults to recording-only
and says so on disk; the two 2.5-hour requests the old ceiling refused are both
admitted; no admission returns a derived cutoff; a failed stage is still
recorded; `report` exits 0 with nothing to breach; retiring a 4.0-ceiling ledger
preserves every entry and hour and stamps the retired figures; retiring twice is
a no-op; preflight retires rather than leaves armed; a slow stage runs to
completion by default; an explicit watchdog still stops one and the occupancy is
still recorded; a new output directory still does not reset prior spend.

## 6. What this does not do

* It does not approve any experiment. The fixed early-grid amendment
  ([`AMENDMENT_01_EARLY_GRID.md`](AMENDMENT_01_EARLY_GRID.md)) is still awaiting
  the PI's separate decision, and nothing in this clarification changes its
  selection rules or certifies its implementation.
* It does not expand any study. Scope is set by the research question, not by
  what the hardware happens to permit.
* It does not shorten any study to fit a budget. The early-grid rerun is short
  because 100 steps is where the match point lies, not to save hours.
