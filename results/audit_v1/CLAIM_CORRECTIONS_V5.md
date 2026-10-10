# Claim corrections, V5 — the on-test check tested the wrong thing

Raised by the coordinator on 2026-10-10 against the Amendment 01 result commit
`7e01fbd`. This is an **interpretation and analysis-logic** correction: no number
was re-measured, no checkpoint was reselected, no tolerance was widened, no grid
was refined, nothing was retrained, and the frozen test set was **not** touched
again. All GPU-derived quantities stand exactly as recorded. The only new
computation is a CPU re-run of `diag_analysis.py` over the **existing** detector
rows.

Earlier rounds: [`CLAIM_CORRECTIONS_V2.md`](CLAIM_CORRECTIONS_V2.md),
[`CLAIM_CORRECTIONS_V3.md`](CLAIM_CORRECTIONS_V3.md),
[`CLAIM_CORRECTIONS_V4.md`](CLAIM_CORRECTIONS_V4.md).

---

## C5.1 — "Matched on TEST" was decided by the tolerance alone

**What was claimed**

> "| 17 | 17.50 pp | 16.25 pp | **1.25 pp** | 1.25 pp | **MATCHED ON TEST** |"
>
> "**What this supports, on the one seed where matching holds on the test set:**
> at matched dog deletion, the L2-SP arm shows **no material retention
> advantage** on bird. The interval lies wholly inside ±10 pp and its upper bound
> is +5.0 pp, so a benefit of 10 pp or more is ruled out at that margin,
> conditionally."
> — `AMENDMENT_01_RESULTS.md` §7.1–§7.2 at `7e01fbd`, and the
> `test_matching_check` / `primary_reading_per_seed` fields of
> `analysis_test.json`

**What the protocol actually requires.**
`MATCHED_EFFECTIVENESS_PROTOCOL.md` §5, *Reporting imperfect matching on the test
set without reselecting*:

> "Re-check **the same gates and the same match** on TEST and report the achieved
> values beside every retention contrast."

§5's gates are stated two paragraphs earlier and were applied at selection to
both the candidate **and** the L2 reference: **≥ 30 pp dog suppression** against
that seed's own MA **and** **≤ 60% dog residue**. The check added to
`diag_analysis.py` at `7e01fbd` implemented only the **≤ 5 pp mismatch**. The
gates were never re-applied on the test set.

**Why the omission is consequential and not cosmetic.** The tolerance asks
whether the two arms deleted the *same* amount. The gates ask whether that amount
is inside the declared partial-suppression regime *at all*. Two arms that both
under-deleted can agree with each other perfectly — and that is exactly what
happened on seed 17.

**The numbers, re-derived from the existing detector rows:**

| seed | arm | dog suppression vs own MA | ≥ 30 pp | dog residue | ≤ 60% | eligible |
|---|---|---|---|---|---|---|
| 17 | L2 (reference) | **17.50 pp** | **FAIL** | 51.25% | pass | **no** |
| 17 | U* (candidate) | **16.25 pp** | **FAIL** | 52.50% | pass | **no** |
| 29 | L2 (reference) | **22.50 pp** | **FAIL** | 52.50% | pass | **no** |
| 29 | U* (candidate) | **28.75 pp** | **FAIL** | 46.25% | pass | **no** |

The residue gate is met in all four arms. The **suppression** gate fails in all
four. On the development set the same arms were at or above it (30.00/31.25 pp
seed 17; 33.75/31.25 pp seed 29), so this is a **transfer** failure between two
different prompt samples, not a disagreement about the gate.

**Status of each claim:**

| claim | status |
|---|---|
| seed 17 mismatch on TEST is 1.25 pp, within the 5 pp tolerance | **STANDS** — and is now reported as *tolerance agreement*, not as "matched" |
| seed 29 mismatch on TEST is 6.25 pp, outside the tolerance | **STANDS** |
| seed 17 is a **matched comparison** on TEST | **WITHDRAWN.** Both its arms fail the suppression gate on TEST |
| seed 17 shows **practical equivalence** at the declared margin | **WITHDRAWN** |
| a ≥ 10 pp bird benefit is **ruled out** on seed 17 | **WITHDRAWN** |
| "at matched dog deletion, L2-SP shows no material retention advantage" | **WITHDRAWN** for both seeds |
| the bird contrast is +1.25 pp [−2.50, +5.00] (s17) and −1.25 pp [−6.25, +3.75] (s29) | **STANDS as a descriptive measurement** of these two checkpoints |
| every per-category, per-threshold and per-family contrast and interval | **STANDS as descriptive**, unchanged in value — **all 378 estimates per seed are bit-identical** between the published `analysis_test.json` and the corrected one |
| the matched-retention question is answered | **NO.** It is **inconclusive in the intended regime** |

**Withheld is not disproved.** Withdrawing the equivalence claim does **not**
assert that L2-SP retains more bird. It asserts that this run cannot say either
way, because the comparison it ran was not the comparison that was authorised.

**What is now reported, separately and never collapsed:**

| field in `analysis_test.json` | what it decides |
|---|---|
| `test_matching_check` | **tolerance agreement** between the arms — necessary, not sufficient |
| `test_eligibility_check` | each arm's **own** §5 reference gates on the set being reported |
| `test_protocol_validity` | both of the above — the only thing that licenses "matched" and the §7 decision table |
| `primary_reading_per_seed.interval_*` | the **arithmetic** facts about the interval, carrying no protocol claim |
| `primary_reading_per_seed.withdrawn_claims` | the protocol-level claims explicitly withheld for that seed |

`valid_matched_comparison_any_seed` is **false**.

---

## C5.2 — The annotation deliverable was not the approved audit

**What was claimed**

> "**228 items** (SET R 168 stratified-random, SET D 60 enriched …). … Key-free
> metadata is committed at `annotation_packet_v2/`."
> — `AMENDMENT_01_RESULTS.md` §7.4 at `7e01fbd`

**What was approved.** 180 paired image items: ten matched
`(prompt, generation-seed)` tuples for each training seed and each of
cat / dog / bird, **all three arms for each tuple** — 2 × 3 × 10 × 3 = 180.

**Why the difference matters.** The approved design is **paired**: the same
prompt and generation seed under all three arms, so a human can judge whether a
detector verdict *change between arms* is a real change in the image. The
delivered packet sampled arms and categories independently, over seven
categories, with 60 items enriched for detector-hard cases. It answers a
different question — detector error rate over strata, and failure modes — and
cannot answer the paired one, because it does not hold the tuple fixed.

**Status:**

| claim | status |
|---|---|
| a blinded packet with empty labels and an external key was delivered | **STANDS** |
| the delivered 228-item packet **is** the approved audit | **WITHDRAWN** |
| the 228-item packet is useful | **STANDS** — retained unchanged as a **separate supplementary** audit |
| an error rate may be computed across the two packets | **NO.** The enriched subset is biased upward by construction; the paired packet is not category-stratified |

**Reconciliation.** The prescribed 180-item paired packet is now built from the
**same existing images** — nothing generated — at
[`annotation_packet_paired180/`](annotation_packet_paired180/):
60 tuples × 3 arms, 18 `(seed, category, arm)` cells of exactly 10, 180 distinct
images, labels **empty**, key outside the packet, blinding asserted by a probe
that aborts the build on a leak. Detector independence is **structural**: the
builder reads only each slot's `image_report.json` and the frozen test manifest,
never `detections.jsonl`, under declared sampling seed **20261012**.

**Overlap: 13** of 180 images also appear in the 228-item packet; **0** carry a
label, because no human annotation has been performed anywhere in this project.
The item-level mapping and any prior labels live with the key, **outside** both
packets, because overlap reveals set membership; only aggregate counts are
committed. No key is disclosed and the two samples are not mixed.

**Additionally disclosed — the earlier packets' blinded ids are enumerable.**
Blinded ids are `sha256(salt | image_path)[:12]` over *deterministic* image
paths, and the v1 and v2 packets **published their salts** (`AUDIT-01`,
`AMENDMENT-01`). A few hundred hashes per cell therefore recover the arm of
every item in those packets for anyone with repository access. The paired packet
uses a **fresh 32-byte secret salt** stored beside its key, publishing only the
digest, and withholds the drawn tuples from the published manifest. The two
earlier packets are **preserved exactly as delivered and were not rebuilt**;
this is recorded so no reader assumes the three blind equally. It is a
theoretical path requiring deliberate effort, not an observed de-blinding, and
the annotator is given only the packet directory.

---

## C5.3 — The published evidence manifest listed a file the commit did not carry

**What was claimed.** `amendment01_evidence/MANIFEST.json` at `7e01fbd` listed
eight payloads including `unattended.log` (sha256 `8ab803ee4379…`, 73,561 bytes).
Seven were present. `unattended.log` was **not in the commit**: `.gitignore`
excludes `*.log`, and unlike `diag_v1_evidence/run_all.log` it was never
force-added.

**Status:**

| claim | status |
|---|---|
| the seven committed payloads match their recorded identities | **STANDS** — independently re-checked |
| `unattended.log` was available at `7e01fbd` | **WITHDRAWN.** It was listed but absent |

**Resolution.** The file is supplied and force-added. Its bytes hash to
`8ab803ee4379…` at 73,561 bytes — **the identity the published manifest already
recorded** — so it is proved to be the file that was listed rather than a later
substitute. The bundle also now carries the raw development and frozen-test
detector rows with their generation and detection reports, so selection and every
interval are re-derivable without the images or the weights, and the manifest
published at `7e01fbd` is preserved verbatim as
`MANIFEST.published_at_7e01fbd.json`. **Access limitations: none.**

---

## C5.4 — What must not be concluded from any of this

* **This is not a general defect of matched-effectiveness designs.** The
  observation is that **this** selected pair did not satisfy the declared test
  conditions in **this** pilot. Causes, and the general reliability of such
  designs, are **unresolved**.
* **Why the development level did not transfer is not established.** The
  development set is dog-only (80 pairs, 20 prompt texts); the test set has 20
  dog prompt texts never used for selection. No determinism or repeatability
  study was requested, and none was run.
* **The pilot's L2-SP reading is still unresolved**, not refuted. The confound
  AUDIT-01 identified — unequal deletion strength — remains unresolved, because
  the diagnostic that was meant to resolve it did not reach the authorised
  regime on the set it reported.
* **No GPU work is authorised by this correction**, and none was performed for
  it.

---

## Regression evidence

`scripts/seq/test_diag_test_conditions.py` — **59 CPU checks, 0 failures**:

* the gate constants are the protocol's values, with inclusive boundaries
  (exactly 30.00 pp passes; 29.99 pp fails; exactly 60.00% passes);
* **CASE A — arms close, both under-suppressed**: MA 50% → both arms 30% gives
  20 pp suppression each (gate FAIL), 30% residue each (gate pass), 0.00 pp
  mismatch (tolerance pass), and an identical-bird contrast whose interval is
  exactly [0, 0]. This is the shape the tolerance-only code read as practical
  equivalence; it must now report **no protocol-level claim** while retaining
  the interval;
* **CASE B — a valid positive**: MA 90% → both arms 40% gives 50 pp suppression
  (pass), 40% residue (pass), 0.00 pp mismatch (pass), so the §7 decision table
  **does** apply and the readings are available;
* **CASE C** — suppression exactly on the gate with residue over it, confirming
  the two gates are independent;
* **source-bound** checks that read the committed
  `amendment01_evidence/analysis_test.json` and assert the real values
  (17.50 / 16.25 / 22.50 / 28.75 pp), that neither seed is a valid matched
  comparison, and that no practical-equivalence claim stands for either.

`scripts/seq/test_diag_analysis.py` — **41 checks, 0 failures**, unchanged.
