# Amendment 01 — the fixed early grid at steps 10…90: results

**Detector-based and PROVISIONAL.** The blinded human annotation is a
prerequisite for any final scientific conclusion. No labels exist yet; the
packets are built with every label cell empty and their keys stored outside them.

> **CORRECTED 2026-10-10 (v2 of this document).** The on-test check in §7.1 and
> the readings in §7.2 originally tested the **match tolerance only**. Protocol
> §5 requires the **same gates and the same match** on TEST. Re-applying the
> gates shows **both arms of both seeds fall below the 30 pp suppression gate**,
> so **neither seed supplies a valid matched comparison** in the prespecified
> regime — including seed 17, which was previously reported as a matched
> practical-equivalence result. That reading is **withdrawn**. The bird
> contrasts and intervals are retained as **descriptive** results. Full record:
> [`CLAIM_CORRECTIONS_V5.md`](CLAIM_CORRECTIONS_V5.md); one-page closeout:
> [`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md). No GPU work was done
> for the correction, nothing was reselected, retrained or re-evaluated, and
> every earlier analysis record is preserved.

Execution commit `0cd9dc611b737e32774c631930ddf5a4199edc27`. Output root
`/data/bijaypandey/cuig_pilot/seq_pilot/diag_v2_early_grid`. The original run
(`diag_v1`) is untouched. Protocol:
[`AMENDMENT_01_EARLY_GRID.md`](../../docs/research/AMENDMENT_01_EARLY_GRID.md).

---

## 1. The hypothesis this tested, and the answer

The original 100-step grid found no dump within 5 pp of its seed's L2 endpoint
on either seed: the measured points **straddled** the target (0 pp at step 0,
38.75/42.50 pp at step 100). Straddling does **not** guarantee a matching
discrete checkpoint, so this amendment **tested** whether some step on the fixed
grid 10, 20, …, 90 lands inside the band.

**It does, on both seeds, at step 70 — on the development set.** On the frozen
test set the prespecified conditions of §5 then **fail on both seeds**: the
reference gates are not met by either arm of either seed, and on seed 29 the
match tolerance is exceeded as well. **Neither seed supplies a valid matched
comparison**, so the matched-retention question is **inconclusive in the
intended regime**. See §7.1.

| seed | fixed L2 target | step-70 suppression | mismatch | within 5 pp |
|---|---|---|---|---|
| 17 | 30.00 pp (eligible, exactly on the gate) | 31.25 pp | **1.25 pp** | yes |
| 29 | 33.75 pp (eligible) | 31.25 pp | **2.50 pp** | yes |

Granularity is 1.25 pp (one detection in 80 dog pairs).

---

## 2. Training: the two declared changes, and nothing else

Each seed's trajectory was retrained from its **saved MA** (bound by digest
`3cb72ee162b9…` s17, `fbf49199a306…` s29) for **100 optimizer steps** with a
dump **every 10**. Both reached `optimizer_steps_completed = 100`, and both
wrote **10/10 dumps**, each validating against the checkpoint contract with all
digests distinct.

Every setting that determines the first 100 updates was preserved and
**verified from the reports**, not assumed; the learning-rate schedule is
`constant`, which diffusers builds without reference to the horizon, so the
shortened stopping limit cannot move the learning rate at any early step. The
declared contract was enforced by value: seeds 17/29, 100 steps, cadence 10,
bridge step 100, tolerance 5 pp, same parent digest on both runs.

---

## 3. The bridge at step 100: HELD

The one step both runs write, compared three ways with the stop condition
declared in advance.

| seed | original step-100 | rerun step-100 | difference | ≤ 5 pp |
|---|---|---|---|---|
| 17 | 38.75 pp | 37.50 pp | **1.25 pp** | yes |
| 29 | 42.50 pp | 43.75 pp | **1.25 pp** | yes |

Weights at step 100 are **not** bitwise identical (0/32 tensors; max abs
3.297e-04 s17, 3.340e-04 s29; relative Frobenius ~1.2e-03). Two things follow,
and only these two:

* the early dumps are a **second realisation**, not points on the original run's
  path, so they are never mixed with the original run's dumps in the match;
* **the cause of the difference is not established.** This comparison cannot
  separate nondeterministic kernels from library, driver or upstream-code state,
  from the changed save cadence, or from dataloader ordering. No controlled
  repeat was run and none is proposed.

Equally, **identical step-100 weights would not have proved identical earlier
states**: one shared later point is not a shared path, and the candidates come
from the unmeasured interior at steps 1–99.

---

## 4. The reference target could not move

MA and MAB_L2 were **not regenerated**. They were read read-only from the
original run's verified development evaluations and bound four ways before
anything was selected (`reference_bindings.json`, `ok: true`):

| check | result |
|---|---|
| generating checkpoint == the registered saved artifact | 4/4 slots |
| manifest == the frozen development manifest `1cd1f902…` | 4/4 slots |
| detections and image-report digests == the published `development_slots_index.json` | 4/4 slots |
| measured L2 suppression == the declared target | 30.00 pp (s17), 33.75 pp (s29) |

The development scan therefore scored **20 slots — the ten dumps per seed only**
and wrote no reference evaluation beside the candidates.

---

## 5. Selection on the development set

Dog-only, 80 pairs per checkpoint, t=0.5, against each seed's own fixed MA
evaluation. Gates ≥30 pp suppression and ≤60% residue, applied to the candidates
**and to the L2 reference**; both references were eligible (seed 17 exactly on
the suppression gate).

| step | seed 17 suppression (gates) | seed 29 suppression (gates) |
|---|---|---|
| 10 | 7.50 pp (fail) | 10.00 pp (fail) |
| 20 | 10.00 pp (fail) | 16.25 pp (fail) |
| 30 | 16.25 pp (fail) | 20.00 pp (fail) |
| 40 | 21.25 pp (fail) | 26.25 pp (fail) |
| 50 | 26.25 pp (fail) | 30.00 pp (PASS) |
| 60 | 28.75 pp (fail) | 28.75 pp (fail) |
| **70** | **31.25 pp (PASS) — SELECTED** | **31.25 pp (PASS) — SELECTED** |
| 80 | 35.00 pp (PASS) | 38.75 pp (PASS) |
| 90 | 36.25 pp (PASS) | 40.00 pp (PASS) |

Both seeds `MATCHED`; step 70 is the qualifying dump closest to its seed's L2
level. Two dumps lay inside the band for seed 17 and three for seed 29; the
prespecified ambiguity rule (separated dumps inside the band with a ≥10 pp
residue spread) was evaluated and **not** triggered — the spread was 8.75 pp for
seed 29.

**Worth stating plainly:** suppression is **not monotone** in steps on seed 29
(30.00 pp at step 50, 28.75 pp at step 60, 31.25 pp at step 70). The selection
rule is indifferent to that, but it is a reminder that these are noisy discrete
measurements at 1.25 pp granularity, not a smooth curve.

Every development input the decision read is recorded with its sha256
(11 per seed), and the decision was **re-bound** to those bytes immediately
before the frozen test was touched.

---

## 6. Gates that had to pass before the frozen test was evaluated

All three were recomputed at that moment, not read from disk:

1. the bridge, **recomputed** and then re-bound — `BRIDGE BINDINGS OK` on both seeds;
2. the selection record, re-bound to its development inputs — `SELECTION BINDINGS OK`;
3. both seeds matched.

Selection is now refused outright while the frozen-test output holds any file,
so there is no path to an automatic reselection after test access.

---

## 7. Frozen test, evaluated once

6 slots (MA, L2, U*=step 70 per seed) × 560 pairs = **3,360 images, 3,360
detector rows**. The manifest identity `5dd87dbb5a77c0f4…` is unchanged, the set
was evaluated **once**, and selection is now locked out of running at all.

### 7.1 The same gates and the same match, re-checked on TEST — and they fail on both seeds

Protocol §5's reporting rule is explicit: *"Re-check the same gates and the same
match on TEST and report the achieved values beside every retention contrast."*
That is **three separate questions**, and collapsing them is what produced the
first version of this section:

| | question | on this run |
|---|---|---|
| **(a) tolerance** | did the two arms delete the **same** amount here (≤ 5 pp apart)? | seed 17 yes, seed 29 no |
| **(b) eligibility** | does **each arm on its own** sit in the declared regime here (**≥ 30 pp** suppression vs its own MA **and** ≤ 60% residue)? | **no — all four arms** |
| **(c) validity** | (a) **and** (b) — the only thing that licenses the word "matched" and the §7 decision table | **no — neither seed** |

**(b), the check that was missing.** Deletion measured on the test prompts is a
different measurement from deletion measured on the development prompts, so a
development pass does not carry over. It did not:

| seed | arm | dog suppression vs own MA | ≥ 30 pp gate | dog residue | ≤ 60% gate | eligible |
|---|---|---|---|---|---|---|
| 17 | L2 (reference) | **17.50 pp** | **FAIL** | 51.25% | pass | **no** |
| 17 | U* (candidate) | **16.25 pp** | **FAIL** | 52.50% | pass | **no** |
| 29 | L2 (reference) | **22.50 pp** | **FAIL** | 52.50% | pass | **no** |
| 29 | U* (candidate) | **28.75 pp** | **FAIL** | 46.25% | pass | **no** |

On the development set the same quantities were 30.00/31.25 pp (seed 17) and
33.75/31.25 pp (seed 29) — at or just above the gate. On the test prompts every
arm lands 1.25–13.75 pp *below* it. The residue gate is met everywhere; it is the
**suppression** gate that fails, in all four arms.

**(a), the tolerance, reported separately:**

| seed | dog deletion, L2 | dog deletion, U* | mismatch on TEST | dev mismatch | tolerance |
|---|---|---|---|---|---|
| 17 | 17.50 pp | 16.25 pp | **1.25 pp** | 1.25 pp | **pass** |
| 29 | 22.50 pp | 28.75 pp | **6.25 pp** (U* deleted more) | 2.50 pp | **FAIL** |

**(c), the consequence.** Seed 17's arms agree with each other to 1.25 pp — but
they agree *outside the authorised regime*, both having deleted about half of
what the gates require. Two arms that both under-deleted can match each other
perfectly; that is agreement, not a matched comparison. Seed 29 fails on both
counts. **Neither seed supplies a valid matched comparison**, so **the comparison is
rejected, not the hypothesis** — §7's last row says exactly that for a failed
match, and §10 makes a reference-gate failure a stop-and-report condition in its
own right.

No reselection was performed, the tolerance was not widened, the gates were not
relaxed, nothing was retrained or re-evaluated, and the step-70 checkpoints
stand. Every earlier analysis record is preserved
(`analysis_test.20261010T205055.superseded.json`,
`analysis_test.20261010T205657.superseded.json`).

> **Why the development set passed and the test set did not is not established
> here.** The development set is dog-only, 80 pairs, 20 prompt texts; the test
> set is 560 pairs over 7 categories with 20 dog prompt texts the model has
> never been selected on. A level that transfers poorly between two different
> prompt samples is the kind of thing this run can report but not explain. No
> determinism or repeatability study was requested and none was run.

### 7.2 Primary endpoint: the paired bird contrast, t=0.5 — DESCRIPTIVE ONLY

Per training seed, never pooled. Frozen scene-cluster bootstrap
`2ba0e8df5034d011…`, 70 clusters, 7 category strata, 10,000 draws, RNG
2026100901, 95% percentile intervals.

| seed | D_bird(L2) | D_bird(U*) | contrast L2 − U* | 95% interval | valid matched comparison | reading |
|---|---|---|---|---|---|---|
| 17 | 73.75% | 72.50% | **+1.25 pp** | [−2.50, +5.00] | **no** (eligibility) | **descriptive only** |
| 29 | 76.25% | 77.50% | **−1.25 pp** | [−6.25, +3.75] | **no** (eligibility + tolerance) | **descriptive only** |

**What is withdrawn.** Both intervals lie inside ±10 pp and both upper bounds are
below +10 pp. Under §7 that *would* read as practical equivalence at the declared
margin and as ruling out a 10 pp benefit. **It does not read that way here, for
either seed**, because §7 applies only where §5's conditions hold. Three claims
are therefore withdrawn, for seed 17 as well as seed 29:

- practical equivalence at the declared margin;
- exclusion of a material (≥ 10 pp) bird benefit;
- any statement of the form "at matched dog deletion, L2-SP shows no retention
  advantage".

**What stands.** The two checkpoints were compared on a frozen set, paired at
identical `(prompt_id, gen_seed)`, and differed on bird by +1.25 pp (seed 17) and
−1.25 pp (seed 29) with the intervals above. That is a measurement of what these
particular checkpoints do. It is not a matched-effectiveness result.

**Withheld is not disproved.** Nothing here shows that L2-SP *does* retain more
bird, and nothing shows it does not. The question is **inconclusive in the
intended regime**, which is a different statement from either answer. The arms'
directions also disagree between seeds (+1.25 vs −1.25 pp), which §7 reads as
unresolved on its own terms.

### 7.3 Every category separately, never averaged (t=0.5, contrast L2 − U*)

**All of it is descriptive** (§7.2): these are differences between two
checkpoints that were not a valid matched pair on this set. They are retained
because they are the measurements that were made, and because the retained and
non-target columns are informative about what the two arms do — not because any
of them is a matched-effectiveness contrast.

| category | role | seed 17 | seed 29 |
|---|---|---|---|
| **bird** | **primary retained** | +1.25 [−2.50, +5.00] | −1.25 [−6.25, +3.75] |
| horse | retained (anchor) | +1.25 [−2.50, +5.00] | +1.25 [0.00, +3.75] |
| chair | retained | −2.50 [−8.75, +3.75] | −3.75 [−10.00, +2.50] |
| bicycle | retained | +2.50 [0.00, +6.25] | +2.50 [−2.50, +8.75] |
| sandwich | never-targeted control | −2.50 [−7.50, +2.50] | 0.00 [−7.50, +6.25] |
| cat | earlier deletion target | −2.50 [−8.75, +2.50] | +1.25 [−3.75, +6.25] |
| dog | current target (matching check) | −1.25 [−6.25, +3.75] | **+6.25 [−1.25, +13.75]** |

Thresholds 0.3 and 0.7, and the literal/paraphrase split, are reported in
`analysis_test.json`; 0.5 is primary. Every interval is **descriptive**: no
multiplicity correction is applied, an interval including zero is not evidence of
no effect, and a difference in labels between seeds is not a significant
difference.

### 7.4 Blinded human-audit packets — the prescribed one, and the supplementary one

Two packets now exist, and they are **separate instruments that must not be
pooled**. The delivered packet did not match the approved audit; that is
reconciled here rather than re-described.

**(1) PRESCRIBED — the approved audit, 180 paired items.**
Ten matched `(prompt, generation-seed)` tuples for **each training seed** and
each of **cat / dog / bird**, with **all three arms** per tuple:
2 × 3 × 10 × 3 = **180**. Built from images that already exist; nothing was
generated.

| property | value |
|---|---|
| items / tuples | **180** / 60, exactly 3 arms per tuple, 180 distinct images |
| cells | 18 `(seed, category)` × `arm`, exactly 10 items each |
| tuple universe | the frozen TEST manifest `5dd87dbb5a77c0f4…` |
| sampling | uniform without replacement, **declared RNG seed 20261012** |
| detector independence | **structural**: the builder opens only each slot's `image_report.json` (the generation-side record) and the manifest. `detections.jsonl` is never read, so no score, verdict or threshold can reach the draw |
| realised literal/paraphrase split | reported per cell in the manifest, not forced (4–6 literal of 10) |
| labels | **empty**, asserted before the build is allowed to finish |
| key | outside the packet, `annotation_paired_180/KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv` |
| blinding salt | a fresh **32-byte secret** per packet, stored beside the key (mode 600, outside the packet); only its sha256 is published, and the drawn tuples are withheld from the published manifest. Blinded ids are `sha256(salt \| image_path)[:12]` over *deterministic* image paths, so a published salt would make the whole id space enumerable and the blind recoverable. The earlier two packets published their salts and are **preserved as delivered**; that limitation is disclosed in the packet README rather than fixed retroactively |
| blinding probe | the packet's text files are scanned for arm, seed, detector strings and the salt; a hit aborts the build, as does finding the key, salt or overlap report inside the packet, or any non-empty label cell |

**Why paired matters.** The same prompt and generation seed is judged under all
three arms, so a human can see whether a detector verdict *change between arms*
is a real change in the image. An independently stratified sample cannot answer
that, because it does not hold the tuple fixed.

**Disclosed limitation.** A paired packet necessarily contains three images of
the same prompt and seed. An annotator may notice the similarity; nothing
identifies which of a similar group came from which arm, and the order is
shuffled. This is a property of the approved design, recorded rather than
concealed.

Committed key-free metadata:
[`annotation_packet_paired180/`](annotation_packet_paired180/).

**(2) SUPPLEMENTARY — preserved as delivered, 228 items.**
168 independently stratified items across **seven** categories (2 per
`(seed, arm, category, family)` cell) plus **60** enriched diagnostics (target
score in [0.3, 0.7], or a 0.5-threshold verdict flip against that seed's own
MA), shuffled once under RNG 20261011, key outside the packet. It is **retained
unchanged** as a separate supplementary audit —
[`annotation_packet_v2/`](annotation_packet_v2/) — and is **not** the approved
audit. Its enriched subset is biased upward by construction, so an error rate
computed over the two packets together is not an error rate.

**Overlap, so no human work is repeated.** **13** of the 180 paired images also
appear in the 228-item packet; **0** of them carry a label, because no
annotation has been performed anywhere in this project. The item-level mapping
and any label already entered live with the key at
`annotation_paired_180/OVERLAP_coordinator_only.csv` — **outside** both packets,
because overlap reveals set membership. Aggregate counts only are committed, in
`annotation_packet_paired180/overlap_summary.json`. The paired packet's own cells
stay empty: pre-filling them would import the other packet's sampling design
into this one. **No key is disclosed, and the two samples are not mixed.**

### 7.5 Compute

**2.9013889** device-hours recorded in total across both runs — **1.6991667**
for this amendment and **1.2022222** carried forward over the original run's
**27** entries, all of which are intact. 56 stages, **0 failed, 0 orphaned**.
Recomputed from each entry's `wall_seconds`; the ledger's stored
`spent_gpu_hours` of 2.901391 is the sum of the per-entry values as rounded at
write time and is ~2e-6 h (about 8 ms) higher. Both are recorded in
[`amendment01_evidence/budget_reconciliation.json`](amendment01_evidence/budget_reconciliation.json);
**no entry was added, removed, re-weighted or re-dated**. No ceiling applied.
GPUs 0 and 1 only, both verified idle before assignment; GPUs 2 and 3 were never
touched. **No GPU time was spent on this correction**, and none is authorised by
it.

---

## 8. What no outcome here can support

* **No mechanism claim.** Nothing here speaks to *why* either arm behaves as it
  does, and none was sought.
* **No causal account of run-to-run differences** (§3).
* **Asymmetric completion evidence, disclosed not corrected.** The saved L2
  endpoints' training completion is **UNVERIFIED** legacy evidence; the new
  unregularised arms' completion is counter-verified at 100 steps. Nothing was
  backfilled.
* **The grid was chosen after seeing development results** (disclosed in the
  amendment, §1). The development set is a selection instrument and no
  confirmatory claim rests on it, but the grid lacks the "fixed before looking"
  property the original 100-step grid had, and any report must say so in the
  same sentence as the result.
* **Detector output is a proxy.** Every number above is provisional until the
  blinded human annotation is complete.
* **No general defect in matched-effectiveness designs.** What is established is
  that **this** selected pair did not satisfy the declared test conditions in
  **this** pilot. Whether a matched-effectiveness design is generally reliable,
  and why this one's development level did not transfer, are **unresolved** and
  are not claimed either way.
* **No matched-retention answer.** The question "at matched dog deletion, does
  L2-SP retain more bird?" is **inconclusive in the intended regime**. That is
  neither a positive nor a negative result for L2-SP.
* **The regime that was reached was weaker than the one authorised.** The gates
  were set at ≥ 30 pp to make the comparison a meaningful partial-suppression
  test; the test-set levels of 16–29 pp are below that by design of the gates,
  not by reinterpretation of them. Any follow-up must reach a deletion level
  that is meaningful **on the set it reports**, and verify it there.

---

## 9. Evidence

[`amendment01_evidence/`](amendment01_evidence/) — 15 payloads, each listed in
`MANIFEST.json` with its sha256 and byte count, independently re-checked:

| payload | what it is |
|---|---|
| `selection.json` | the development decision, gate outcomes, and the sha256 of all 11 inputs per seed it read |
| `bridge_check.json`, `reference_bindings.json` | the step-100 bridge and the four-way binding of the reused references |
| `analysis_test.json` | **v3, current**: the same gates and the same match, re-checked on TEST |
| `analysis_test.20261010T205657.superseded.json` | **v2**: tolerance only — the version published at `7e01fbd`, preserved verbatim |
| `analysis_test.20261010T205055.superseded.json` | **v1**: no on-test re-check at all, preserved verbatim |
| `development_slots.tar.gz` + index | raw per-image detector rows and stage reports for the **20** development candidate dumps |
| `frozen_test_slots.tar.gz` + index | raw per-image detector rows and stage reports for the **6** frozen-test slots (3,360 rows) |
| `unattended.log` | the supervisor log — listed in the manifest published at `7e01fbd` but **absent** from that commit (`.gitignore` excludes `*.log`); supplied here and verified against the identity already recorded, `8ab803ee4379…`, 73,561 bytes |
| `gpu_usage_record.json`, `budget_reconciliation.json` | the device-hour ledger and its recomputation |
| `STATUS.json`, `MANIFEST.published_at_7e01fbd.json` | the supervisor state machine, and the earlier manifest kept verbatim |

With the two archives, **selection and every interval can be re-derived without
the bulk images (≈592 MB) and without the weights** — the frozen bootstrap
grouping is committed at
[`draft_manifests/bootstrap_grouping.json`](draft_manifests/bootstrap_grouping.json).
Both archives are built with normalised mtimes, so rebuilding from the same
bytes reproduces the same digest.

**Access limitations: none.** Every listed payload is present and hash-checked.
Deliberately excluded, unchanged: model weights (digests only), generated images
(per-image digests inside the archives), and **both** annotation packets' keys.

Rebuild:

    python scripts/seq/diag_analysis.py --diag_root <root> --set test
    python scripts/seq/build_amendment01_evidence.py \
        --diag_root <root> --out results/audit_v1/amendment01_evidence
    python scripts/seq/build_paired_annotation_packet.py \
        --eval_root <root>/eval_test --out_root <root>/annotation_paired_180 \
        --repo_out results/audit_v1/annotation_packet_paired180
    python scripts/seq/test_diag_test_conditions.py     # 59 checks, CPU only
