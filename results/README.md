# Results

Only small JSON measurement artefacts are tracked here. Generated images,
checkpoints, Excel reports and logs are git-excluded and live under
`/data/bijaypandey/cuig_pilot/`.

## What is here

| File | What it is | Scientific status |
|---|---|---|
| `assets/validation_receipt.json` | asset gate verdict | **FAIL** — classifier checkpoints absent; `UA_IRA_CRA: UNAVAILABLE` |
| `assets/negative_control_receipt.json` | gate run against shape-correct **random** heads | **FAIL by design** — proves shape compatibility alone cannot open the gate |
| `preflight/preflight_benchmark_paths.json` | preflight against the real benchmark asset paths | environment/upstream PASS; assets `MISSING`; **baseline gate CLOSED** |
| `mechanics_check/mechanics_check_report.json` | substitute-configuration smoke run on GPU 2 | **NOT a baseline.** Generator was base SD-1.5, 8 optimizer steps, 16 anchor images |
| `mechanics_check/checkpoint_verify.json` | `delta.bin` save/reload verification | PASS — 32 kv-xattn tensors, 0 unexpected keys, all finite |
| `software_test/software_test_report.json` | `evaluate.py` code-path test with synthetic classifiers | **Software test.** Accuracies are random noise |

## The negative control, in one line

Randomly-initialised heads of the *correct* shape scored 3.5% (style, chance
2.0%) and 2.9% (object, chance 5.0%), modal-index agreement 0/8 and 0/20 →
`shape_compatible: true`, `classifiers_verified: false`. The gate rejects them.

## What is NOT here

**No UA, IRA or CRA.** The UnlearnCanvas style and object classifiers could not
be obtained on this host, so target-suppression and retention metrics have not
been measured. The single-concept ConAbl baseline has not been run. See
[`../docs/ASSETS.md`](../docs/ASSETS.md).

Any accuracy value appearing in `software_test/` comes from randomly-initialised
classifier heads and means nothing. It exists only to show the evaluation code
executes end to end.
