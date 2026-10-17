# Sequential-erasure diagnostic — EXECUTED (chain saw -> parachute, FLUX.1-schnell)

Execution of the accepted revision-02 protocol
([`c2c0388`](../sequential_diagnostic_protocol/PROTOCOL.md)). **Outcome:
STOPPED_AT_FROZEN_GATE; diagnostic now CLOSED.** Both methods hit `C=0, U=0` on the
historical target, so neither batch comparator was triggered. **DP satisfies the full
frozen gate conjunction; UCE does not** — it shows the same detector-defined zero relapse
but fails the cumulative retention gate on two classes independently.

> **Carries a post-publication reporting correction.** Executed numbers, gates, candidate
> sets and the stopping decision are **unchanged**; several interpretations published at
> execution commit `ffc133f` are corrected — see [`CORRECTIONS.md`](CORRECTIONS.md).

| start here | for |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | the answered primary question, every frozen gate per method, measured preservation residuals (with the corrected metric definition), rendering observations, limits, resource accounting and the closeout decision |
| [`EXECUTION_MANIFEST.md`](EXECUTION_MANIFEST.md) | identities and hashes verified before generation, exact commands/settings, the only implementation changes, endpoint accounting |
| [`CORRECTIONS.md`](CORRECTIONS.md) | **read alongside the report.** Ten corrected claims (residual metric meaning, the scope of UCE's retention failure, the springer explanation, provenance wording, the withdrawn next-step recommendation), each with whether the underlying number changed — none did |
| [`PUBLISHED_EVIDENCE_INDEX.md`](PUBLISHED_EVIDENCE_INDEX.md) | path, byte count and sha256 of every published file |

## Headline numbers

- Chain saw (historical target): **100 -> 0 at the A-parent -> 0 after the B edit**, for
  both DP and UCE; paired table shows **absent->absent in 100/100 cases, zero
  absent->present**. `C=0, U=0` for both — not a semantic-absence certificate.
- Parachute (new target): **100 -> 0** both directly and sequentially, both methods;
  direct-vs-sequential deletion difference `|0-0| = 0`.
- Measured preservation residuals on chain-saw probe vectors, as the archived logger
  defines them (`q = ‖(W_child−W_parent)c‖₂ / ‖c‖₂`, **input-normalized — not a
  parent-response percentage**): DP post-cast `q` = 0.0042419 (T5) / 0.0013254 (CLIP);
  UCE post-cast `q` = 0.3869613 (T5) / 0.0319098 (CLIP). Differing residuals coexist with
  zero detector-discovered relapse; **no preservation-strength ranking follows**
  (`CORRECTIONS.md` C2–C4).
- Retention: DP passes both bounds; **UCE fails the cumulative bound on two classes
  independently** — cassette player −24 and english springer −17, against a 15-case
  per-class limit.
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

The four edited `.safetensors` checkpoints and the DP pre-cast fp32 capture stay on
`/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/`, **hash-identified** in
`evidence/HOST_ONLY_ARTIFACTS.json`; the full 4,000-image raw generation output stays
there too but is inventoried **only by directory, image count and total bytes — not
hash-indexed per image** (`CORRECTIONS.md` C7–C8). The pinned upstream source tree (MIT,
unmodified) is cited by commit, not vendored. The protocol itself and all prior reports
are unchanged.
