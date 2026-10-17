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
> | [`audit_v1/PUBLISHED_EVIDENCE_INDEX.md`](audit_v1/PUBLISHED_EVIDENCE_INDEX.md) | exact path and sha256 of every published, key-free file — verify any row without trusting the table |
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
| [`rece_seq_golfball_englishspringer/`](rece_seq_golfball_englishspringer/) | RECE (closed-form SD1.4 erasure) direct-vs-sequential golf-ball→English-springer editing, a stage-ablation diagnostic, a reference-only diagnostic (parent A vs. M0 as the adversarial-embedding reference), an embedding/working-checkpoint cross-swap diagnostic, a CPU-only numerical-optimality audit of the Stage-2 refinement solver, a solver-only intervention (`AB_STABLE_SOLVE`) testing whether solving that same solver accurately actually fixes anything, and a CPU-only recovery-optimality audit of the separate embedding-recovery solve (`close_form_emb_regzero`) that produced `E_D`/`E_S` in the first place, and a preservation-content control testing whether RECE's existing preservation term (pointed at the already-erased golf-ball response, vs. an unrelated control) prevents the golf-ball corruption | **separate research line, unrelated to UnlearnCanvas/ConAbl above.** Both original gates pass; visual review finds sequential-specific quality costs the numbers miss. The reference-A swap partially, not fully, mitigates that quality cost at a measured deletion-strength cost. The cross-swap diagnostic's own report overstated the erasure-target mechanism as "any cell using the recovered sequential embedding corrupts" — corrected (`CORRECTIONS.md` C16) to a single direction-specific cell, not a clean embedding effect; the historical-class interaction finding stands. The numerical audit (`numerical_optimality_audit/FINAL_REPORT.md`) finds the native float32 closed-form Stage-2 refinement solve materially fails to solve its own stated objective in all four cross-swap cells — a concrete candidate mechanism. The solver-only intervention (`solver_only_intervention/FINAL_REPORT.md`) tests it directly: holding everything else fixed and solving exactly, English-springer deletion is unchanged and its generic-dog leakage drops 6/20→2/20, but golf-ball degeneration only falls 35/40→**31/40** (`CORRECTIONS.md` C17 corrects the originally reported 30/40/"5 recovered" after full-resolution re-inspection of case 38; **77.5%** still corrupted, mostly refinement-introduced not inherited) at a small retained-class recognition cost. The **recovery-optimality audit** (`recovery_optimality_audit/FINAL_REPORT.md`) tests the separate upstream computation — the embedding-recovery solve itself — and finds it is **already close to its own fp64 optimum** (relative forward discrepancy ≈0.004%, angle ≈0.002° in both branches) and that the large direct/sequential embedding contrast (≈94° angle, ≈3.9× magnitude) **survives accurate recovery essentially unchanged** — closing large recovery-solve numerical error as the leading explanation for that contrast, in contrast to the Stage-2 refinement solve, which *is* materially inaccurate. The **preservation-content control** (`preservation_content_control/FINAL_REPORT.md`) then tests an existing-method fix directly: holding everything else fixed (including `AB_UCE` as `W` and the saved native `E_S`) and replacing only the Stage-2 refinement's retain text, pointing preservation at `AB_UCE`'s own already-erased golf-ball response collapses degeneration **31/40→1/40** (96.8% relative reduction, landing near Stage-1's own 2/40 baseline) without restoring the golf ball (0/40 visual recognition in every arm, including a confirmed classifier false positive on the one exact-match case) — raw classifier predictions on English-springer exact recognition are GOLF 0/20, HORN 1/20 (case 86); the gate (>=75% suppression vs. A's 20/20) passes both (100% / 95%) but "0/20 in both" is not the full picture — while an unrelated-content control (`'French horn'`) shows, after closeout reinspection (`CORRECTIONS.md` C25), **no improvement and a one-case net worsening** versus blank (31/40→32/40, 0 recovered/1 newly damaged — not the originally reported 31/40→31/40 nominal wash), showing preservation *content*, not just a nonzero preserve term, is what matters. The real cost: generic-dog leakage on the erasure target more than triples (2/20→7/20) under historical preservation; the unrelated control also shows elevated generic-dog leakage (5/20→6/20 after closeout correction). Numerical inaccuracy (isolated to the Stage-2 refinement step) and preservation-content choice are now both established as real, partial levers on this failure — neither is a new method; both are existing-mechanism controls. A follow-up, `history_preservation_confirmation/FINAL_REPORT.md`, tests the same refinement-only preservation control on two **fresh** sequences (S1: golf ball→French horn; S2: English springer→golf ball) with a freshly frozen 80-case panel, and finds **it does not help on either**: historical-class degeneration is 19/20→20/20 (S1, net −1) and 20/20→20/20 (S2, net 0) going from blank to historical preservation — most of the damage (18/20 and 20/20 respectively) is already introduced at the native Stage-1 edit itself, before refinement's preservation choice ever runs (`CORRECTIONS.md` C26–C27; narrow conclusion only — this does not establish that history preservation fails in general, nor that the original benefit is pair-specific, since both the sequence and the panel changed). That premise motivated a direct **Stage-1 intervention**, `stage1_history_intervention/FINAL_REPORT.md`: applying the identical preservation choice (`retain_texts=['golf ball']`) at the native Stage-1 edit itself — not at refinement — **eliminates** the damage: golf-ball degeneration **14/20→0/20** (corrected from an originally reported 18/20 after a rubric-consistency re-inspection of 4 macro-texture cases, `CORRECTIONS.md` C29 — the qualitative result, a complete paired recovery with zero newly damaged, is unchanged), with French-horn deletion unchanged (100% suppression against the parent's own verified 16/20) and retained-class recognition unchanged or slightly higher. A narrow but clean existing-method result: applying preservation *before* the damage occurs works, on this one sequence and preservation choice, where applying it only *after* does not (`CORRECTIONS.md` C28). The identical intervention was then run on **S2**, a second, independently selected sequence (not S1's structural reverse), `stage1_history_intervention_s2/FINAL_REPORT.md`: English-springer (historical, *deliberately erased*) degeneration drops substantially, **20/20→2/20** (corrected from an originally reported 3/20 after re-inspecting case 63 against its inherited parent pattern, `CORRECTIONS.md` C31), while exact recognition and visual breed-specific presence correctly stay at **0/20 throughout** — springer's continued absence is the *desired* outcome here, the same role golf ball played in S1, not a shortfall. Golf-ball (new-target) deletion also shows one genuine, independently confirmed return (not present in the blank comparator, whose own single raw hit was a confirmed false positive) plus two visually ambiguous cases, still comfortably inside the ≥75% suppression gate. A **both-stages completion**, `history_both_stages_s2/FINAL_REPORT.md`, then applied the already-validated Stage-2 refinement to this improved checkpoint: golf-ball suppression reaches **0/20 exact** (best of any S2 checkpoint), springer stays correctly erased, French horn recognition ties the best S2 result (14/20), and retained-class degeneration holds at 0/40 — but **springer degeneration rises from 2/20 to 6/20** (18 recovered/2 persisted/4 newly damaged against that baseline), so the Stage-1 quality gain does not fully survive full-pipeline refinement, concentrated in 3 of 5 scenes and one seed. Three sequence/stage combinations tested now: S1's Stage-1 intervention a near-complete, clean fix; S2's Stage-1 intervention a substantial but incomplete one; S2's full pipeline mostly holds that gain with one genuine, disclosed quality cost from refinement — applying existing preservation earlier helps in every case tried, but is not a uniform, complete fix. A **blinded human review package**, `blinded_quality_review_s2/`, checked that claimed springer-degeneration regression (2/20→6/20) independently: all 20 matched case-60–79 pairs, randomized once with a private mapping and a published commitment hash, individual-image judgments collected before any paired comparison. One selected human response has now been unblinded (`closeout/HUMAN_REVIEW_REPORT.md`) — it does **not** reproduce the claimed regression: whole-frame degeneration goes 2/20→**1/20** (1 recovered, 1 persisted, 0 newly damaged), the opposite direction, with case-level agreement against the assistant's own labels low (16/20 and 15/20, overlapping in exactly one flagged case across both endpoints) — one reviewer, not yet a settled result; an earlier survey submission was explicitly withdrawn and is not used. A **second, context-aware package**, `context_quality_review_s2_v2/`, was then prepared (not yet answered, v1 unchanged) reusing the identical 20 pairs/40 images with the generation-prompt context v1 lacked, explicit erasure-objective framing, a required reason for "breakdown" answers, and new independent randomization (`R01`–`R20`) — any future label difference from v1 must not be attributed to "the addition of context" without further evidence. v2 was then **repaired** (`-v1`→`-v2`, `CORRECTIONS.md` C36): two interaction/resume software defects (reason entry not refreshing the UI; imported exports crashing on Begin) were found, independently reproduced, fixed, and independently re-verified with a fresh regression harness — a software fix, not a new experiment; no response exists for either revision, and the mapping/commitment/images/v1 are all unchanged. A **third, baseline-context package**, `context_quality_review_s2_baseline/`, then added a visible pristine-M0 Original image alongside each of v2's existing 20 pairs (reusing those 40 images byte-identical, v2's mapping unchanged, incorporating the v2 repair directly), with a new, separate relative-deformation question kept apart from the existing absolute judgments — Original identity is intentionally visible (disclosed as partial masking, not full blinding, since the edited images were already public). No response exists for any of the three packages; v1's and v2's own data are untouched. See `CORRECTIONS.md` C30–C37, and each report's own `README.md`/`CORRECTIONS.md` |
| [`modern_flux_dp_uce/`](modern_flux_dp_uce/) | **separate research line, modern model (FLUX.1-schnell), not SD1.x/RECE.** First reproduction of the released DP/UCE FLUX adaptations (`czhang024/CVPR_DouleProjection`), cassette player / chain saw | Native single-edit reproduction only — both methods' chain-saw edits pass every gate (100%→0% semantic recognition); the conditional sequential extension did **not** run because cassette player's own M0 baseline (28/100 semantic) fails the 80/100 reliability gate for both methods, despite both single edits showing 96–100% relative suppression |
| [`modern_flux_dp_uce/sequential_diagnostic_protocol/`](modern_flux_dp_uce/sequential_diagnostic_protocol/) | **protocol only — not executed, revision 02.** Prospective chain-saw→parachute sequential-erasure diagnostic (DP + UCE), built entirely from the existing reproduction's tables | No model/encoder/GPU work ran for the protocol or either correction pass. Selected pair, one mechanism-based prediction with exact source lines (including the native batch-edit comparators' real guide-pooling/preservation/precision differences from the sequential workflow), a hard 6-endpoint/6,000-image ceiling, prospective gates with explicit numeric bounds, and an authoritative ambiguity-aware decision table (`C`/`U`, `[C, C+U]`, `D_min`/`D_max`) governing every conclusion — all pending a separate future execution instruction. See `REVISION_01.md` and `REVISION_02.md` for the two coordinator-requested correction mappings |

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
