# Sequential-erasure diagnostic — EXECUTED (chain saw -> parachute, FLUX.1-schnell)

Execution of the accepted revision-02 protocol
([`c2c0388`](../sequential_diagnostic_protocol/PROTOCOL.md)). **Outcome:
STOPPED_AT_FROZEN_GATE** — both methods hit `C=0, U=0` on the historical target, so
neither batch comparator was triggered; UCE additionally failed one retention bound.

| start here | for |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | the answered primary question, every frozen gate per method, measured preservation residuals, rendering observations, limits, resource accounting and the one recommended next step |
| [`EXECUTION_MANIFEST.md`](EXECUTION_MANIFEST.md) | identities and hashes verified before generation, exact commands/settings, the only implementation changes, endpoint accounting |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Headline numbers

- Chain saw (historical target): **100 -> 0 at the A-parent -> 0 after the B edit**, for
  both DP and UCE; paired table shows **absent->absent in 100/100 cases, zero
  absent->present**. `C=0, U=0` for both — not a semantic-absence certificate.
- Parachute (new target): **100 -> 0** both directly and sequentially, both methods;
  direct-vs-sequential deletion difference `|0-0| = 0`.
- Preservation residual on chain saw, post-cast relative: **DP 0.42%**, **UCE 38.7%** —
  a ~90x gap in weight-level preservation that produced **no** difference in relapse.
- Retention: DP passes both bounds; **UCE fails the cumulative bound** (cassette player
  −24 vs a 15-case limit).
- **4,000 new images of the 6,000 ceiling**; 4 of 6 endpoints run.

## What is here

- **`evidence/per_image/`** — per-image classifier predictions for all 4,000 new images
  (`{DP,UCE}_{B_direct,SEQ}_per_image.csv`) plus per-endpoint summaries.
- **`evidence/`** — `ANALYSIS.json` (all gates, raw transitions, retention),
  `CASE_JUDGMENTS.json` (candidate set and C/U record), the two chain-saw paired
  transition tables, `panel_hashes.csv`, and `HOST_ONLY_ARTIFACTS.json` (hashes of the
  deliberately unpublished checkpoints and raw generation output).
- **`evidence/verification/`** — parent sha256 / bitwise-load / key / dtype / finiteness /
  save-reload records for both SEQ edits, and the direct-edit integrity record.
- **`evidence/residuals/`** — measured pre- and post-cast preservation residuals with
  numerical context, per module and per preserve vector.
- **`evidence/code/`** — the parent-loading adapter, residual logger, classifier wrapper,
  analysis and panel builder actually used.
- **`images/full_resolution/`** — the frozen 20-case panel at 512x512 for all seven
  compared endpoints (M0, both A-parents, both B-direct, both SEQ).
- **`images/grids/`** — 20 readable per-class comparison grids (M0 | A-parent | B-direct | SEQ).

## What is NOT here

The four edited `.safetensors` checkpoints and the full 4,000-image raw generation
output stay on `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/`, hashed and
listed in `evidence/HOST_ONLY_ARTIFACTS.json`. The pinned upstream source tree (MIT,
unmodified) is cited by commit, not vendored. The protocol itself and all prior reports
are unchanged.
