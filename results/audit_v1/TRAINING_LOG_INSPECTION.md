# Bounded CPU inspection of the saved pilot's training logs

**AUDIT-01d.** CPU only. No GPU, no model, no training, no generation. Nothing
was written into any training report and no completed-step counter was derived
from this. Machine-readable results:
[`training_log_inspection.json`](training_log_inspection.json), produced by
`python scripts/seq/inspect_training_logs.py`.

## Why this exists, and what it does not do

The ten saved pilot training reports carry **no structured counter of completed
optimizer steps**, so **training completion is UNVERIFIED for the entire saved
pilot**. That has not changed and is not changed by anything below.

A previous Claude round reported that the logs nonetheless contain terminal
`1000/1000` progress output. The independent review correctly treated that as a
*reported* additional evidence source rather than a verified one, and asked for
the logs to be inspected on CPU with their provenance and limits stated first —
clearly separate from direct counters.

**This is corroborating provenance, not a counter.**

- `Opt. Steps: 100%|██████████| 1000/1000` is a **tqdm progress bar**: terminal
  display output, rendering the bar's own configured total. It is not the
  trainer's internal completed-step variable.
- It was written by the **same process** that would have recorded a counter, so
  it is not an independent witness to that process.
- It was **never parsed or validated** when it was produced. Nothing compared it
  to anything.
- **No contract hashes these logs.** They are plain text files in a directory
  and could be edited or truncated with no check noticing. The digests in the
  JSON fix them as of this inspection only.

Consequently: `--steps_evidence legacy_optional` still reports
`training_completion_verified: false` for all ten artifacts, and
[`configs/legacy_training_artifacts.json`](../../configs/legacy_training_artifacts.json)
still records the gap. Nothing here licenses a stronger completion claim.

## What was measured

| measurement | result |
|---|---|
| training logs expected (2 seeds × 5 checkpoints) | 10 |
| training logs found | **10** |
| logs showing a terminal `1000/1000` progress line, `100%` | **10 / 10** |
| distinct progress-bar totals seen, any log | `[1000]` only |
| maximum step shown, any log | `1000` |
| logs embedding the **registered `delta.bin` digest** of the artifact they belong to | **10 / 10** |

Seed 17 wrote into `logs/`, seed 29 into `logs/seed29/`, matching the
evaluation-layout history recorded in `exp_config.sh`.

## The one thing this does add: linkage

Each log's tail embeds the JSON report the run printed, including
`child_checkpoint.sha256`. For all ten, that digest equals the `delta_sha256`
registered for the artifact the log is supposed to belong to. So these are not
merely "ten logs that exist somewhere": each one is tied by content to a
specific saved checkpoint.

That is a **provenance** fact — this log belongs to this checkpoint — not a
**completion** fact.

## What this does NOT establish

1. **That 1000 optimizer updates were applied.** The progress bar's total is
   configuration, not measurement. A bar can reach its total while something
   downstream of the step counter is wrong, and the circular runtime check in
   use at the time would not have caught it either.
2. **That the logs are unmodified.** No digest of them was recorded before now.
3. **Any change to the saved pilot's status.** The published rates, the
   checkpoints and the reports are untouched. `legacy_optional` still means
   UNVERIFIED.

## Consequence for the proposed diagnostic

None that changes its cost or scope. The matched-effectiveness diagnostic
compares **exact saved L2 endpoints** with new unregularised checkpoints at
**comparable measured dog suppression**. Equal verified training duration is not
its matching variable, so the logging asymmetry — the L2 arms' completion
unverified, the new unregularised arms' completion recorded by counter — does
**not** justify retraining the two L2 arms.

The ~0.46 GPU-hour L2 retraining suggestion from an earlier round remains an
**unapproved scope expansion** and is not included: the design stays at two
unregularised trajectories, 1,920 development + 3,360 test images, an estimated
**3.095 GPU-hours** and a proposed **four-hour ceiling**. The asymmetry is
disclosed in the protocol rather than spent against.

## Reproduce

    python scripts/seq/inspect_training_logs.py \
        --logs_root /data/bijaypandey/cuig_pilot/seq_pilot/logs \
        --registry configs/legacy_training_artifacts.json \
        --out results/audit_v1/training_log_inspection.json
