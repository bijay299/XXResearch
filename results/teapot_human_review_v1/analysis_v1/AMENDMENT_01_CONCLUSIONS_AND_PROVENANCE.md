# AMENDMENT 01 to analysis_v1 — conclusions, provenance and evidence index

**Dated 2026-10-18. Authoritative over earlier wording on every point below,
including wording in the unchanged [`summary.json`](summary.json) and in
[`FINAL_REPORT.md`](FINAL_REPORT.md) as first published.**

Scope: **documentation, provenance and evidence-index corrections only.**

**No response, count, table, denominator, derived label, reporting convention
or scientific rule is changed by this amendment.** The numerical results have
been independently verified and are accepted as published. The analysis was
**not** re-run, no statistical test was added, no label was inspected or
adjudicated, no model/GPU/API was used, no image was generated, and no
participant was contacted.

Closeout: **`ANALYSIS_COMPLETE` is retained**, and this analysis is now
**`CLOSED WITH QUALIFIED CONCLUSIONS`**. That status records that the numbers
stand and that several *interpretive* claims around them were overstated and
are withdrawn here. **It is not a finding of scientific failure, and it is not
a new success threshold.**

---

## 1. The accepted counts — unchanged

Restated here for the record. Every figure is byte-identical to what
[`summary.json`](summary.json) and [`tables/`](tables/) already publish.

| | |
|---|---|
| M0 target present (consensus) | **20 / 20** |
| Edited consensus | **18 absent, 1 present, 1 unresolved** |
| Conditional demonstrated-removal count | **18 / 20** — denominator 20 **includes the unresolved edited case** |
| Rendering among the 18 edited-absent cases | **13 agreed coherent, 5 unresolved** |
| Task-A presence agreement | **39 / 40** |
| Task-A rendering agreement | **34 / 40** |
| Task-B ratings across 40 | **36 `about_the_same`, 3 `more`, 1 `less`, 0 `cannot_fairly_compare`** |
| Task-B exact per-pair agreement | **18 / 20** |

All per-reviewer cells, stratum cells and `unresolved` labels stand exactly as
published. The frozen rules are untouched: two agreeing definite votes make
consensus; `absent → absent` is not demonstrated removal; `unable_to_judge`
stays distinct from `unresolved`; the removal count is never merged with any
rendering outcome into a combined score.

## 2. Correction — how many records, and how many answers

**Earlier prose said "120 individual categorical votes." That arithmetic was
wrong.**

There are **120 reviewer-item records** carrying **200 categorical answers**:

* **80 Task-A records** (2 reviewers × 40 images), each with **two** answers —
  `target` **and** `rendering` — - **160 answers**;
* **40 Task-B records** (2 reviewers × 20 pairs), each with **one**
  `comparison` answer - **40 answers**.

160 + 40 = **200 categorical answers across 120 records**. The earlier figure
conflated records with answers. **No count, table or denominator depends on
this: it is a prose error only**, and every published table already held all
200 answers.

> The "120" shorthand also appears in earlier frozen documents
> ([`../protocol/PROTOCOL.md`](../protocol/PROTOCOL.md) §7,
> [`../MASKING_AND_LIMITATIONS.md`](../MASKING_AND_LIMITATIONS.md),
> [`../README.md`](../README.md)) in the form "20 paired cases are not 40
> independent comparisons or 120 independent reviewer observations." Those
> files are **historical and are deliberately left unedited**. Their
> non-independence point is correct and still stands; only the arithmetic
> reading is superseded by this section.

## 3. Withdrawn and qualified interpretive claims

Each claim below appeared in `FINAL_REPORT.md` §9–§10 as first published. The
**counts that prompted them are unchanged**; what is withdrawn is the
*inference* layered on them.

### 3.1 Instrument ceiling and discriminative range — WITHDRAWN

**Withdrawn:** that the measurement is "near the ceiling of what this
instrument can resolve"; that "Task B as designed has almost no discriminative
range on this material"; and that either is "a finding about the instrument."

**Why:** the observation is that **36 of 40** Task-B ratings fell on
`about_the_same` and `cannot_fairly_compare` was unused. A concentrated rating
distribution on **one fixed 20-case panel judged by two people** does not
establish a property of the instrument. **No instrument-sensitivity study,
no discrimination test and no comparison against a panel with known
differences was performed**, and none is performed now.

**Retained as observation only:** Task-B ratings were **concentrated at
`about_the_same`** (36/40), with 3 `more`, 1 `less` and 0
`cannot_fairly_compare`.

### 3.2 "Badly under-powered" — WITHDRAWN

**Withdrawn:** the phrase "positive but badly under-powered."

**Why:** **no inferential test, estimand, meaningful effect size or power
calculation was ever specified** for this analysis — by design, since it is
descriptive. Calling it under-powered asserts a quantity that was never
defined. **No power calculation is run now.**

**Replacement wording:** this is a **small fixed descriptive panel with limited
generalization** — 20 paired cases, 200 categorical answers, 120 records, two
reviewers, one object, one endpoint.

### 3.3 "One-directional" rendering difference — CORRECTED SCOPE

**Withdrawn:** that the rendering-strictness difference between reviewers is
"one-directional."

**Correct statement:** across the **six** rendering disagreements, **five** have
R2 marking `localized_deformation` where R1 marked `coherent`, and **one runs
the other way**. A stricter-threshold explanation for R2 **remains a
hypothesis**, not a finding; it is equally consistent with ordinary rater
variation, and nothing here separates the two.

**Preserved, correctly scoped:** the **five** rendering disagreements that fall
**within the 18 edited-absent subset** do all run the same direction — R2
`localized_deformation` where R1 `coherent`. That narrower statement was
accurate and stands. All six disagreements are `coherent` ↔
`localized_deformation`; no other pair of rendering values was ever in
conflict.

### 3.4 Diagnosed calibration/anchor failure — WITHDRAWN

**Withdrawn:** that there is a "rendering-anchor problem" to "fix," and the
implication that missing calibration *caused* the six unresolved rendering
labels.

**Why:** that the six unresolved labels are all `coherent` ↔
`localized_deformation` is an observation. Attributing them to absent shared
anchors is an untested explanation. **No calibration study was run.**

### 3.5 Binding constraints and categorical invalidity — WITHDRAWN

**Withdrawn:** that "the binding constraint is now case count, reviewer count
and rendering calibration"; that adding methods or baselines "would produce
rankings that this design cannot support"; and any categorical claim that a
method comparison is invalid or that **no** bounded efficacy description is
permissible.

**Why:** these were presented as established constraints. They were
judgements about hypothetical future designs, not results of this analysis.
What is accurate is narrower and is stated in §4: this panel supports a
**bounded descriptive finding** and does not support general claims. Whether
some *other* bounded comparison could be justified is a question this analysis
did not examine and does not foreclose.

## 4. The supported conclusion

**On this fixed panel:** both reviewers marked the target **present in all 20
M0 images**. Applying the frozen consensus rule to the edited images yields
**18 absent, one present and one unresolved**; among the 18 absent cases, **13
are agreed coherent and five are rendering-unresolved**. Task-B ratings were
**concentrated at `about_the_same`** (36 of 40).

**Three explanations remain compatible** with the Task-B concentration, and this
analysis does not choose between them:

1. genuine similarity in quality between the two conditions;
2. limited stimulus variation across this particular 20-case panel;
3. limited sensitivity of the comparison instrument.

**This supports a bounded descriptive finding about visible target depiction at
one fixed endpoint, on this panel, as judged by these two reviewers.**

It does **not** establish general unlearning, removal of concept knowledge,
retention of unrelated concepts, adversarial robustness, evaluator bias,
novelty, superiority over any method, or paper readiness.

**The 13 agreed-coherent absences are not a success score.** They are not
combined with the 18/20 removal count, or with each other, into any composite
figure — exactly as the frozen rules require.

## 5. Closure replaces the prescriptive recommendation

**Withdrawn:** the recommendation that the study "should" be re-run on a larger
calibrated panel, and the framing of that as the necessary next step.

**Replacement:** **this fixed analysis is closed.** Any future enlargement,
calibration pass, additional reviewers or method comparison is an
**unselected prospective option** that would require its own justified
question, design, resourcing and authorization. None is selected here, none is
recommended as mandatory, and **no follow-up is launched**.

## 6. Masking — procedural account, not proof

**Corrected.** Earlier wording said "the masking that mattered was masking at
judging time," presented as something that held.

**Accurate statement:** both reviewers **declared** no pre-Task-A access to
source pairings or condition labels and no image-specific coaching, and
independence was **user-reported**. That supports a **procedural masking
account** — the protocol provided for masked judging and the declarations are
consistent with it.

**It is not proof that judging-time masking actually held.** Declarations and
aliases cannot establish what a reviewer knew. Equally, **nothing here
indicates that masking failed**; this amendment asserts neither.

## 7. Reproducibility — two different limits, not one

Earlier wording blurred two distinct facts. Separated here.

**(a) Condition and pair assignment IS recoverable.** It can be reconstructed
by hashing the already-public Task-A, Task-B and source-asset images, because
the source release publishes each case's `m0`/`edited` filenames with their
SHA256 and both bundles ship byte-identical images. **An independent review
reconstructed the assignment that way.** Withholding a direct crosswalk
**does not make that assignment unreproducible** — and this amendment does not
claim otherwise.

**(b) Export-level reproduction IS limited.** The exact original export files,
free-text answers, original export metadata and the reviewer↔`R1`/`R2` linkage
remain **private**. A third party therefore cannot reproduce the published
tables from the original artifacts, and cannot verify the reviewer linkage.

These are different limits: **(a) is recoverable, (b) is not.** Earlier text
presented a single undifferentiated "partial reproducibility."

**On the withheld-fields list.** [`summary.json`](summary.json) is unchanged and
lists "case IDs" and "neutral-ID → condition associations" among withheld
fields. **Clarified, not edited:** that wording refers to **private source
identifiers and direct crosswalks** — it does **not** mean that every case
identifier is withheld, and it does **not** claim that no association is
inferable. **The Task-B pair IDs `B-01`…`B-20` are published**, in
[`tables/taskB_responses.csv`](tables/taskB_responses.csv) and
[`tables/paired_case_ledger.csv`](tables/paired_case_ledger.csv), and the
condition assignment is inferable as described in (a). This amendment
supersedes that wording rather than rewriting `summary.json`'s bytes.

**Pseudonyms.** `R1`/`R2` are pseudonyms, not anonymity. They **do not erase
the earlier roster disclosure** recorded in
[`../AMENDMENT_03_TASKA_CLOSURE_AND_TASKB_RELEASE.md`](../AMENDMENT_03_TASKA_CLOSURE_AND_TASKB_RELEASE.md)
§7b and **do not guarantee anonymity**.

## 8. Chronology — preserved exactly as disclosed

The honest chronology in [`ANALYSIS_CONTRACT.md`](ANALYSIS_CONTRACT.md) §0 is
**preserved unchanged and is not backdated**:

* the four answer files were **already visible** before the implementation
  contract was written, because they arrived attached to the request;
* the contract was written **before any aggregation, table or count** was
  computed;
* the **underlying scientific rules came from the earlier frozen protocol**
  (`TEAPOT-HR-PROTO-1.0.0`) and its authoritative amendments — the contract
  implemented them and clarified their shorthand.

**This is not described as a blind preregistration**, and this amendment does
not re-characterize it as one. **No participant repeat, re-review or revised
judgment follows from this disclosed deviation**, and none was performed.

## 9. Manifest and evidence-index repair

Two mechanical defects in the previous publication, both now fixed.

### 9.1 What was wrong

The top-level [`../EVIDENCE_INDEX.md`](../EVIDENCE_INDEX.md) at
`5b25219` was generated from **working-tree bytes paired with an unrelated
commit** (`55ccc4c`). Consequently:

* it listed **stale hashes for four analysis artifacts** that were updated in
  the later commit — `FINAL_REPORT.md`, `provenance.json`, `public_audit.json`
  and `analysis_v1/MANIFEST_SHA256.md`;
* its **own self-row was unverifiable**: it claimed 27,517 bytes and a digest
  belonging to the *previous* index version, while the published file was
  30,091 bytes. **That row failed even at its declared pin.**

Verified mechanically before repair: **2 of 90 rows failed at the declared
pin** and **5 of 90 failed against the commit readers actually fetch**.

### 9.2 The non-circular sequence now used

1. Finalize the substantive payload — amendment, report, provenance, pointers,
   new scoped privacy-audit record.
2. Generate [`MANIFEST_SHA256.md`](MANIFEST_SHA256.md) with **explicit
   exclusions** for itself, the top-level index and later verification outputs.
   **No circular coverage is claimed.**
3. Commit the complete finalized payload.
4. Generate the top index **from exact Git objects at that payload commit** —
   never working-tree bytes — **excluding the index's own path**, with **no
   historical self-hash**.
5. Publish the index in a **child commit**. Indexed payload files are not
   edited afterwards; any further edit requires a new payload commit and a
   regenerated index.

`scripts/gen_evidence_index.py` was repaired to enforce this. **The numerical
pipeline was not touched.**

### 9.3 The earlier manifest result is retained as a historical fact

The previous 22-file analysis manifest **was verified correct** at the time:
every row matched the live published bytes. That remains true of those bytes
and is recorded here. The manifest is regenerated for the amended
documentation, with its coverage and exclusions stated explicitly.

## 10. What changed on disk

Only: this amendment; corrections to `FINAL_REPORT.md` and the narrative fields
of `provenance.json`; current pointer text in the two READMEs; the regenerated
manifest and evidence index; `gen_evidence_index.py`; and new mechanical
audit/verification/accounting records.

**Historical audit reports are not overwritten.**
[`public_audit.json`](public_audit.json),
[`verification.json`](verification.json),
[`synthetic_checks.json`](synthetic_checks.json) and
[`mapping_verification.json`](mapping_verification.json) are **unchanged**; the
new scan for this change set is a **separate** record,
[`amendment_public_audit.json`](amendment_public_audit.json), scoped to the
actual final change set and stating honestly which private checks ran and which
were skipped.

The **package-level** [`../MANIFEST_SHA256.md`](../MANIFEST_SHA256.md) is a
**historical record and is left byte-identical** — it is not regenerated here,
and nothing in this amendment claims it covers the amended files. Current
coverage of every tracked file comes from the regenerated
[`../EVIDENCE_INDEX.md`](../EVIDENCE_INDEX.md), generated from Git objects at
the payload commit as described in §9.2.

> While staging this amendment I briefly overwrote that package-level manifest
> with the analysis manifest's contents by mistake, and left a stray copy of
> `gen_analysis_manifest.py` at the package root. Both were caught by the
> change-set review before any commit, the manifest was restored to its exact
> published bytes, and the stray file was removed. The preservation check now
> covers that manifest explicitly so the same slip cannot pass silently.

In `provenance.json`, **only inaccurate narrative fields were changed.**
Numeric results, code hashes, input hashes and the original chronology fields
are **preserved byte-identically in value**, and the amendment timestamp is
recorded in a **separate** field rather than by altering the original one.

## 11. Preservation — mechanically verified before and after

See [`preservation_check.json`](preservation_check.json).

| | |
|---|---|
| Four original export hashes | **unchanged** (hashed only; **not read or aggregated**) |
| Private read-only copies and verified backup | **unchanged** |
| Every published CSV and `summary.json` | **unchanged** |
| `ANALYSIS_CONTRACT.md`, mappings, commitments, IDs, reviewer linkage | **unchanged** |
| Analysis / rule / mapping / test / verification code | **unchanged** |
| Historical `synthetic_checks.json`, `verification.json`, `public_audit.json`, `mapping_verification.json` | **unchanged** |
| All scientific definitions, denominators, reporting conventions | **unchanged** |
| All three questionnaire releases — IDs, assets, sizes, digests, timestamps, descriptions, tag targets | **unchanged** (download counters excluded) |
| Source asset release — **31 assets** | **unchanged** |
| Closed projects (`modern_object_assets_v1`, `measurement_study_v1`) | **unchanged** |
| Questionnaire viewer files | **unchanged**; neither viewer rebuilt |

## 12. Accounting — appended; historical clocks not reset

| | used | cap |
|---|---|---|
| Elapsed (this amendment) | **under cap** | 2 h |
| Worker threads | 1 (CPU only) | 8 |
| Additional scratch under `/data` | far under cap | 1 GiB |
| GPU / model / API judging / generation | **0** | 0 |
| Numerical reanalysis, statistical tests, browser runs | **0** | 0 |
| Label inspection or adjudication, participant contact | **0** | 0 |
| Responses read or aggregated | **0** (hashed only) | 0 |

All historical preparation, release, maintenance and analysis clocks are
preserved and were **not reset or retroactively extended**.

## 13. Status

**`ANALYSIS_COMPLETE`** (retained) · **`CLOSED WITH QUALIFIED CONCLUSIONS`**.

The accepted numerical findings stand. The interpretive overreach around them
is withdrawn and replaced with the bounded descriptive finding in §4. **No new
study was started, no judgment was revised, and no completed work was
reopened.**
