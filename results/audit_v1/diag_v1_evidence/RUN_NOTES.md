# Matched-effectiveness diagnostic — run notes

Execution commit: `3a5a69c650a0f1f14e873aee3f374d1ff8a71fed`
Started: 2026-10-09T17:13 -04:00. Ceiling: 4.0 GPU-hours (device-hours).

## Intermediate dumps are 2,156 bytes larger than `delta.bin`, by design

Measured in the pre-launch GPU smoke test:

    delta-10         76,688,346 bytes
    delta-20         76,688,346 bytes
    delta.bin        76,686,190 bytes   <- configs/checkpoint_contract.json

Verified on CPU that this is **pickle metadata, not content**. Both files carry
the same single top-level `unet` key, the same 32 kv-xattn tensors, the same
19,169,280 parameters, float32, all finite, no missing and no unexpected keys;
and at step 20 the dump's values are **bit-identical** to the final `delta.bin`
(max abs difference 0.0). The difference comes from the periodic save path
constructing its pipeline differently from the final save, not from the weights.

**Consequence for validation, recorded so nobody "fixes" it the wrong way.**
`validate_stage.py train` checks `expected_file_bytes` and is correct to: it
validates a FINAL `delta.bin`. The dumps are therefore validated by CONTENT
instead — tensor names, shapes, dtypes, finiteness, parameter count, digest
stability and schedule completeness — inside `run_diagnostic.sh`. Do **not**
loosen or widen `expected_file_bytes` in the checkpoint contract to accommodate
dumps; that would weaken the check that protects the final checkpoints.

## What is reused, and read-only

Bound per (seed, slot) against `configs/legacy_training_artifacts.json` at
preflight, so one seed's parent cannot stand in for the other's:

    seed17/MA      3cb72ee162b9…      seed17/MAB_L2  8ee7d56f62b0…
    seed29/MA      fbf49199a306…      seed29/MAB_L2  c4f3ecdcb945…

Nothing in `${SEQ_ROOT}/models`, `eval/` or `eval_seed29/` is written by this
run. All diagnostic output lands in this directory.

## Detector results are provisional

The blinded human annotation is being completed in parallel and is **not** an
input to any step here. No final scientific conclusion follows from the detector
alone.

## Outcome: INFEASIBLE on both seeds — the test set was not evaluated

Stopped at selection by the rule declared in advance. 1.2022 of 4.000 GPU-hours
spent, 0 failed stages, the 1.008 GPU-hour test reserve **unspent**, and
`eval_test/` empty.

| seed | MA dog residue | L2 endpoint | L2 suppression | first dump (step 100) | closest mismatch |
|---|---|---|---|---|---|
| 17 | 82.5% (66/80) | 52.5% | **30.0 pp** | 43.8% → **38.8 pp** | **8.8 pp** |
| 29 | 82.5% (66/80) | 48.8% | **33.8 pp** | 40.0% → **42.5 pp** | **8.8 pp** |

All ten dumps pass both reference gates on both seeds. None lands within 5 pp of
its seed's L2 level, because **the unregularised arm has already overshot that
level by step 100** — the earliest dump the frozen 100-step grid provides. The
match point therefore falls between the parent (step 0, 0 pp by definition) and
step 100, a gap of 38.8 pp (s17) / 42.5 pp (s29).

The tolerance was not widened, neither seed was dropped, and no unmatched
comparison was run.
