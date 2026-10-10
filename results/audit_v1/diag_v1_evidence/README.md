# Diagnostic 01 evidence bundle

Everything needed to re-derive the decision of the matched-effectiveness
diagnostic without access to the private data root. **824 KB, 21 files.**
Every file is listed in [`MANIFEST.json`](MANIFEST.json) with its sha256.

**Outcome: a GRID-MATCH FAILURE.** No dump matched on the frozen 100-step grid,
on either seed. The matched bird-retention question is **inconclusive** — it was
never measured. See [`../DIAGNOSTIC_01_FINDINGS.md`](../DIAGNOSTIC_01_FINDINGS.md).

**Provisional.** Detector output is a proxy. No scientific conclusion follows
until the blinded human annotation is complete; no labels exist yet.

## Contents

| file | what it is |
|---|---|
| `selection.json` | **current** decision record: per-dump table, bracketing, and the L2-reference eligibility gate |
| `selection.20261009T215925.superseded.json` | **preserved** earlier decision record — not overwritten |
| `run_all.log` | preflight, start, every stage, selection and stop records, verbatim |
| `per_dump_table.csv` | every scored checkpoint, both seeds, with gate outcomes and mismatches |
| `development_slots.tar.gz` | all **24** development slots: `detections.jsonl` (per-image detector output, **1,920 rows**), `image_report.json`, `detect_report.json`, validation marker |
| `development_slots_index.json` | per-slot digests, row counts, generating-checkpoint digest, manifest identity, settings source |
| `trajectories.json` | trajectory counters, dump inventories, endpoint digests, all 32 effective hyperparameters — **no weights** |
| `u_vs_mab_comparison.json` | new U endpoint vs saved MAB: configuration, parent identity, serialisation, numerical agreement |
| `gpu_budget_ledger.json` | the experiment's ledger, with all prior spend carried forward |
| `gpu_budget_ledger_original_location.json` | the original in-output-directory ledger, superseded but retained |
| `cost_model_measured.json` | conservative measured per-stage estimates and the reconciled reserve |
| `frozen_test_inventory.json` | the frozen-test output location, inventoried **empty (0 files)** |
| `RUN_NOTES.md` | run-time provenance notes, including the dump-size finding |
| `test_logs/` | the eight CPU suites' exact output, each stamped with the commit it ran at |

## Deliberately excluded

- **Model weights** — ~1.6 GB of `delta.bin` and intermediate dumps. Their
  sha256 digests are in `trajectories.json`.
- **Generated images** — ~208 MB. Per-image digests are inside each slot's
  `image_report.json` in the archive.
- **The annotation arm key** — not included and not referenced. Nothing from
  the blinded annotation packet appears here; verified by probing the bundle and
  the archive for the packet's blinded item ids (0 hits) and for any reference to
  its path (0 hits). The key stays closed until labels are frozen.

## Re-deriving the decision

```bash
tar -xzf development_slots.tar.gz -C /tmp/dev_slots
python scripts/seq/select_matched_dump.py \
    --dev_root /tmp/dev_slots --seeds "17 29" \
    --dump_steps 100,200,300,400,500,600,700,800,900,1000 \
    --out /tmp/reselection.json
```

Both seeds return `INFEASIBLE`, both L2 references pass eligibility, and the
frozen test set stays forbidden.

## Correction notice (2026-10-10)

Two **interpretations** in this bundle's accompanying prose were withdrawn after
it was assembled. Every measurement here stands and nothing has been re-run,
re-measured or re-hashed: the bundle is deliberately left **byte-identical** so
that its independent verification remains valid.

The superseded wording survives in `u_vs_mab_comparison.json` →
`conclusion`, which names run-to-run nondeterminism as the cause of the
endpoint difference. That attribution is **not established** — equal recorded
configuration and an equal parent do not isolate a cause. Likewise, the measured
points straddling the L2 level at steps 0 and 100 does **not** guarantee that a
matching discrete checkpoint exists in between.

**[`../CLAIM_CORRECTIONS_V4.md`](../CLAIM_CORRECTIONS_V4.md) governs.** This file
is not covered by `MANIFEST.json`, which is why the notice can be added here
without disturbing any verified digest.
