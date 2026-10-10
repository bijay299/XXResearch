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
> symlink); the sandwich non-replication was **not** a false positive; "robust"
> is unavailable at n=2; and the L2-SP retention comparisons are **confounded by
> unequal deletion strength**.
>
> **A second round** then corrected eight overstatements in the audit's own
> write-up — [`audit_v1/CLAIM_CORRECTIONS_V2.md`](audit_v1/CLAIM_CORRECTIONS_V2.md).
> Among them: the two runs differ **only in the training RNG seed** and are not
> independent replications; the child-vs-child contrast is a **different
> estimand**, not a parent-free recovery and not an isolation of the L2 effect;
> the non-replication is **half child and half parent** movement, not "largely"
> the parent; the +40 to +47.5 pp result is a **newest-target** residual
> difference and says nothing about cat history; the evaluation set is **70
> prompt texts** (280 prompt × generation-seed pairs), not 280 prompts; and
> upstream stopping is **threshold-or-patience**, not a certified per-arm
> maximum, and is not wired into the sequential setting at all.
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
> | [`audit_v1/CLAIM_CORRECTIONS_V2.md`](audit_v1/CLAIM_CORRECTIONS_V2.md) | corrections to the audit's own first write-up |
> | [`audit_v1/AMENDMENT_01_CLOSEOUT.md`](audit_v1/AMENDMENT_01_CLOSEOUT.md) | **start here for the latest result.** Amendment 01 run and closed: development matched at step 70, **test conditions failed on both seeds**, matched-retention question **inconclusive in the intended regime** |
> | [`audit_v1/AMENDMENT_01_RESULTS.md`](audit_v1/AMENDMENT_01_RESULTS.md) | the corrected results in full, with the evidence index |
> | [`audit_v1/CLAIM_CORRECTIONS_V5.md`](audit_v1/CLAIM_CORRECTIONS_V5.md) | the withdrawn practical-equivalence reading, the annotation-deliverable reconciliation, and the archive gap |
> | [`../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md`](../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md) | the bounded experiment — **approved and run**; 2.9013889 device-hours total |
> | [`../docs/research/NEXT_SCREEN_PROPOSAL.md`](../docs/research/NEXT_SCREEN_PROPOSAL.md) | the next scientific screen — CPU-only, **not authorised to run** |
> | [`audit_v1/draft_manifests/FREEZE.md`](audit_v1/draft_manifests/FREEZE.md) | the frozen draft prompt manifests and analysis rules |
> | [`audit_v1/annotation_packet/ACCESS.md`](audit_v1/annotation_packet/ACCESS.md) | how the PI gets the packet images and blinded sheet |
>
> **No human annotation has been performed.** A 236-item blinded packet is built
> and empty at [`audit_v1/annotation_packet/`](audit_v1/annotation_packet/).
> Amendment 01 adds two more, also empty and also unpooled: the prescribed
> 180-item paired audit
> [`audit_v1/annotation_packet_paired180/`](audit_v1/annotation_packet_paired180/)
> and the 228-item supplementary audit
> [`audit_v1/annotation_packet_v2/`](audit_v1/annotation_packet_v2/).
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
