# AMENDMENT 03 — Task-A closure and authorized Task-B release

**Dated 2026-10-18. Authoritative over earlier wording on the points below.**

Scope: **closure of Task A and preparation/release of the second-stage
questionnaire.** This is not a new scientific experiment. **No outcome is
reported, no response was collected or inspected on this host, and no image was
judged.**

---

## 1. Task A is CLOSED

| | |
|---|---|
| Status | **CLOSED** |
| Planned cohort | 3 independent reviewers |
| **Actual cohort** | **2 independent reviewers** |
| Frozen minimum | ≥ 2 distinct, complete, independent returns |
| Meets the frozen minimum | **Yes** |
| When the cohort change was recorded | **after receipt of the returns, before any outcome analysis** |

Both reviewers returned **complete 40-item Finals** from the corrected
`patch01` bundle, at schema/app `1.1.0`, with **no declared pre-Task-A access to
pairings or condition labels and no image-specific coaching**.

> **Provenance of these closure facts.** They are **coordinator-reported**:
> raw response snapshots are preserved read-only on the user's machine. They
> were **not independently verified on this host**. The raw files were never
> requested, transferred, opened, published, edited or migrated, and were not
> needed for this release. **Zero response files exist on this host**, and none
> is part of any public artifact.
>
> The roster (aliases) and the coordinator-reported file identities are held in
> a **private** coordination record. Public reporting carries only the cohort
> count and this closure/authorization account.

**The original Task-A responses are locked and unchanged.** Nothing in this
release reads, alters, migrates or reinterprets them.

## 2. Both Task-B release prerequisites are satisfied

1. **Task A closed with original responses preserved** — §1.
2. **Explicit later authorization from the user** — given for this release.

Task B is therefore released. It is published as the new tag
**`teapot-human-review-v1-taskB`**.

## 3. Task B is partially unmasked — stated plainly

**This publication intentionally reveals matched pairs and reference identity.**
Every pair shows which image is the original-model **reference** and which is the
modified-model **candidate**, because the comparison requires it.

Three things follow, and none of them is hedged:

* **Task B does not conceal pairing or condition roles.** It is designed to
  disclose them.
* Because the 40 images are **already public** in the
  `modern-object-assets-v1` release, this publication **can enable matching back
  to the public Task-A images**, even though no A-to-B or source crosswalk is
  published.
* **Publishing on GitHub does not restrict access to any particular people.**
  Anyone can download it. **Eligibility is procedural, not technical**: the
  intended primary Task-B cohort is exactly the two locked Task-A reviewers, and
  extra public downloads or returns are **not** automatically eligible primary
  reviewers.

**Aliases and files cannot establish identity or independence.** Future primary
Task-B reporting requires two complete returns **from those same two people**;
missing or partial participation is retained explicitly.

## 4. What was verified

### Package integrity — aggregate only

| check | result |
|---|---|
| Pairs | **20 / 20** |
| Images | **40 / 40** |
| Role assignment (M0 = reference, edited = candidate) | **20 / 20** |
| Pairing matches the approved source | **20 / 20** |
| Image hashes match the approved source | **40 / 40** |
| Pairs with byte-identical initial noise | **20 / 20** |
| Verdict | **`TASKB_INTEGRITY_VERIFIED`** |

No A-to-B or source crosswalk is published. These are aggregate results only.

### Browser QA — synthetic fixtures only

**174 / 174 checks pass** across **three engine configurations** (Chromium,
Firefox, and Firefox with `privacy.file_unique_origin` and
`security.fileuri.strict_origin_policy`), offline via `file://` with every
non-`file://` request aborted — **0 network requests attempted**.

Covered: all 20 pairs displayed with correct reference/candidate pairing and
labels; native 512×512 rendering with **independent per-image zoom**; no answer
preselected; Final disabled until complete; `cannot_fairly_compare` counting as
complete; draft and complete-Final export → **fresh-session re-import** →
reload/resume with asserted equality of raw answers, alias, declarations and
order; 16 rejection cases each asserted to leave **memory and `localStorage`
unchanged**; **A/B isolation** — Task B never alters an existing Task A session
and refuses a Task A export with a clear message; companion-validator acceptance;
and mobile layout.

> **A/B isolation was tested with SYNTHETIC Task-A state only.** No real Task-A
> session or response was read at any point.

### One defect found and fixed during this phase

The pair view **overflowed horizontally on a 390 px screen** (scrollWidth 557).
Figures now scroll internally instead of widening the page. The review panel was
also widened so a pair sits **side by side** on desktop rather than stacking with
a scroll between the two images — material for a comparison task. Both fixes are
Task-B-only.

## 5. Versions and the legacy-schema resolution

| | |
|---|---|
| Package | `THR-V1-B` |
| Protocol | `TEAPOT-HR-PROTO-1.0.0` (unchanged) |
| Export schema | **`THR-EXPORT-SCHEMA-1.1.0`** |
| App | **`1.1.0`, build `taskB-b1`** |
| Order variants | `ALIAS` (default) and `FIXED`, reconstructed from trusted B data |

**The legacy-metadata problem, and how it was resolved.** The old B builder
inherited `schema_id: THR-EXPORT-SCHEMA-1.0.0` from the frozen private mapping,
which was current when the IDs were frozen and is **covered by the published
commitment**. The mapping was **not edited**. Instead the **B software and
support files now declare the export contract they actually implement**
(`1.1.0`), and record the mapping's legacy value alongside for transparency.
Both published commitments were re-verified against the untouched mapping and
still match.

**Task A's 40-item permutations and its pair-spacing rule do not apply to Task B
and are not used here.** B's order is reconstructed from B's own 20 pair IDs plus
the reviewer alias.

The shared companion validator is at revision **`validator-r3`**: stage B now
requires `reference_identity_disclosed_in_this_stage: true` and uses
stage-aware order variants. **Stage-A behaviour is unchanged**, re-verified by
the 45/45 stage-A regression suite.

## 6. Eligibility handling

B exports identify package, protocol, schema, stage, alias, the trusted
presentation order, explicit declarations, raw comparisons and **recomputed**
completion (supplied counts are never trusted). Required declarations:

* Task A completed and its Final exported **before** Task B;
* prior condition access and image-specific coaching **before Task A, including
  same-day exposure**;
* `reference_identity_disclosed_in_this_stage: true`.

**Reference identity revealed by this authorized Task-B stage is not treated as
disqualifying pre-A exposure.** Any other between-stage exposure can be recorded
separately, without rewriting Task A and without inventing a new exclusion rule.

Finals require exactly **20 distinct expected pair IDs with 20 valid
comparisons**; drafts may be partial. Duplicate or unknown IDs, invalid
values/types, unsupported versions and wrong package/stage are rejected without
mutation, and an import **never merges into or overwrites Task A**.

## 7. What was preserved

| | |
|---|---|
| Locked Task-A responses | **unchanged**; never read on this host |
| Both Task-A releases, assets, tag targets | **unchanged** |
| Active patch01 ZIP | 13,923,703 B / `837770ae…` — unchanged |
| Restored original ZIP | 13,917,868 B / `45e3eb7a…` — unchanged |
| 40 approved PNG bytes | **unchanged** |
| `B-01`…`B-20` pair IDs and role assignments | **unchanged** |
| Private mappings, salts, commitments | **unchanged**; commitments re-verified |
| Task-A schema/app/protocol | **unchanged** |
| All completed projects and source releases | **unchanged** |

`freeze_ids.py` was **not** re-run; no IDs were regenerated; no private committed
mapping field was edited to change a software version. No new generation, model
loading, training, GPU allocation, judging, image-content labelling or outcome
analysis occurred.

## 7b. One mistake of mine during this release, and its fix

While building the Task-B audit tooling I **hard-coded the two reviewer aliases
as literals** in two scripts that are published in this repository
(`scripts/audit_taskB.py` and `scripts/verify_taskB_published.py`). They were
there so the audit could scan public artifacts *for* those aliases — but putting
them in published code **disclosed the roster that is supposed to stay
private**.

They were never in the reviewer bundle or on any release asset (the bundle audit
confirms zero alias findings), but they were committed to Git and pushed. Both
scripts now **load the roster from the private coordination record** instead, and
report whether the alias scan actually ran rather than silently passing when the
private record is unavailable. No alias literal remains in any published file.

The aliases were briefly present in the public Git history for those commits.
Rewriting published history would be worse than disclosing this, so the commits
stand and this note records what happened.

## 8. Accounting — appended, historical clocks not reset

| | used | cap |
|---|---|---|
| Elapsed (Task-B release phase) | within cap | 4 h |
| Worker threads | 8 | 8 |
| Additional scratch under `/data` | well under cap | 2 GiB |
| GPU / model / training / generation / judging | **0** | 0 |
| Task-A changes | **0** | 0 |
| Real participant files read | **0** | 0 |

Browsers and caches installed in earlier phases were reused. The historical
preparation and maintenance clocks were **not reset or retroactively extended**.

## 9. What this does and does not establish

It establishes that the **second-stage instrument is built, verified and
published**, and that Task A closed with a two-reviewer cohort meeting the
frozen minimum.

It establishes **nothing** about erasure efficacy, retention, quality, evaluator
bias, novelty or paper readiness, and it reports **no outcome, no label and no
image judgement**. Whether two eligible Task-B returns arrive is a future
question this release does not answer.
