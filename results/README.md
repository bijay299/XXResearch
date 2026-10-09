# Results

Only small JSON measurement artefacts are tracked here. Generated images,
checkpoints, Excel reports and logs are git-excluded and live under
`/data/bijaypandey/cuig_pilot/`.

> ## Current status — read `audit_v1/` first
>
> **AUDIT-01 (2026-10-09)** re-derived the whole two-seed snapshot `a9de625`
> from the raw per-image detector predictions. **The arithmetic reproduces
> exactly — 0 mismatches over 1026 values — and completeness is exact: 0
> duplicate rows, 0 missing rows, no de-duplication rule needed anywhere.**
>
> Nine **narrative** claims were overstated and are corrected in
> [`audit_v1/CLAIM_CORRECTIONS.md`](audit_v1/CLAIM_CORRECTIONS.md). The
> headline ones: 3080 images were generated (not 3360 — M0 is reused via a
> symlink); the sandwich non-replication was **not** a false positive and is
> largely a `D_cat(MA)` reference-term effect confined to paraphrase prompts;
> "robust" is unavailable at n=2; and the L2-SP retention comparisons are
> **confounded by unequal deletion strength**.
>
> Documents carrying inline corrections are marked **[corrected]**; pre-audit
> text is preserved at
> [`audit_v1/preserved_originals/`](audit_v1/preserved_originals/) and in git
> at `a9de625`.
>
> | start here | for |
> |---|---|
> | [`audit_v1/AUDIT_REPORT.md`](audit_v1/AUDIT_REPORT.md) | what was verified, what was corrected, what is unresolved |
> | [`audit_v1/CORRECTED_TABLES.md`](audit_v1/CORRECTED_TABLES.md) | corrected counts and the direct L2 contrasts the snapshot never computed |
> | [`audit_v1/ARTIFACT_INVENTORY.md`](audit_v1/ARTIFACT_INVENTORY.md) | what is reusable; **no intermediate checkpoints exist** |
> | [`../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md`](../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md) | the proposed next experiment (≈1.93 GPU-h, **pending review**) |
>
> **No human annotation has been performed.** A 236-item blinded packet is built
> and empty at [`audit_v1/annotation_packet/`](audit_v1/annotation_packet/).
> Every visual statement in these reports is AI inspection, not human review.

## What is here

| File | What it is | Scientific status |
|---|---|---|
| `audit_v1/` | AUDIT-01 re-derivation, corrections, inventory, annotation packet | **current** — supersedes the narrative framing of the files below |
| `COMBINED_RESULTS.md`, `SEED_COMPARISON.md`, `seq17/`, `seq29/` | two-seed sequential pilot | numbers verified; narrative **[corrected]** inline by AUDIT-01 |
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
