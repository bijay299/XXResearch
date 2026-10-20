# LLaDA/RWKU Baseline Qualification (`llada_rwku_baseline_v1`)

**Outcome: `INSUFFICIENT_BASE_RECALL` — stopped without training, as the protocol requires.**

Pristine LLaDA-8B-Instruct does not recall enough RWKU target knowledge to support a
forgetting measurement: **0 of 10** candidate entities reached the ≥6/10 threshold
(best 5/10; three were required). The infrastructure built to test this is verified
and working.

| Phase | Status |
|---|---|
| Pristine recall | 🔴 **INSUFFICIENT_BASE_RECALL** — 0/10 entities qualify, max 5/10 |
| Evaluator correctness | ✅ **PASS** — 48/48 assertions |
| Objective correctness | ✅ **PASS** — 10/10 vs MDU's native source |
| Training / reload | ⛔ **NOT ATTEMPTED** — forbidden by the stop condition; zero optimizer steps |
| Behavioral effect | ⛔ **NOT REACHED** |

**Not** `QUALIFIED_FOR_BOUNDED_DEVELOPMENT_COMPARISON`. Nothing here is a baseline
result, an unlearning claim, or a statement about MDU's published numbers.

Resources: **0.090 of 6.0 GPU-h** · 0.32 of 24 h · 0.227 of 10 GiB · 1 GPU.

## Why the shortfall is genuine

Zero mechanical failures across all 200 generations (`empty/refusal/repetition/
truncation/parse_failure` all 0); the misses are confident *wrong facts* ("Dan Simmons"
for Richard Bachman, "Harvard" for Columbia); and the same harness scores **75% on
MMLU** against a 50% floor. The pipeline works — the model simply does not know these
facts reliably.

## Contents

| Path | What it is |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Full findings, per-phase scope, objective verification, accounting |
| [`FREEZE_NOTE.md`](FREEZE_NOTE.md) | The pre-inference freeze, committed in `203525b` before the model ran once |
| `frozen/` | Frozen spec, panels, MMLU panel, neighbor availability (hashed, pre-inference) |
| `scripts/` | `freeze_eval.py`, `evaluate.py`, `mdu_objective.py`, `verify_objective.py`, `verify_evaluator.py`, `verify_teacher_wiring.py`, attempt runner + ledger |
| `outputs/` | Every generation and score, both decoders; objective/evaluator/wiring verification results |
| `logs/` | All 8 attempt logs, including both failures, none overwritten |
| `evidence/ATTEMPTS.tsv` | Attempt ledger |
| [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) | Every published file with SHA-256, generated from tracked files |
