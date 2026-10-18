# AMENDMENT 02 — coordinator-side cleanup

**Dated 2026-10-18. Authoritative over earlier wording on the points below.**

**Scope: documentation and companion-Python-validator corrections only.** This
was done **while independent reviews are in progress**, using the corrected
Task-A `patch01` bundle the user is distributing.

> **Nothing respondent-facing changed.** No questionnaire byte, release, tag,
> asset, image, neutral ID, permutation, mapping, commitment, protocol or schema
> was modified. No reviewer was contacted, no repeat or re-export was requested,
> no response was collected or inspected, and no outcome is reported. Task B
> remains private, unchanged and unreleased. All closed projects are untouched.

Status **`READY_FOR_INDEPENDENT_REVIEW`** is retained **for the existing
corrected Task A only**. Validator maintenance is **not** a new questionnaire
certification.

---

## 1. Frozen respondent-facing material — verified before and after

| | |
|---|---|
| Active tag | `teapot-human-review-v1-taskA-patch01` |
| Active ZIP | `teapot-human-review-v1-taskA-patch01.zip` |
| Bytes / sha256 | 13,923,703 / `837770ae23e6b1fb56b7269a87e2a2bc23f40f6e2090e5e172ea57a8719ab7e7` |
| App / schema | `1.1.0` / `THR-EXPORT-SCHEMA-1.1.0` |
| Protocol | `TEAPOT-HR-PROTO-1.0.0` |
| Original restored ZIP | 13,917,868 / `45e3eb7aeb9443e35a4ab990969eb9aa450e2f36b2238192244050b320dbd005` |

Both releases were snapshotted **before and after** this maintenance — release
IDs, asset IDs, names, sizes, content types, created/updated timestamps, body
digests and tag targets — and compared. Download counters are recorded but
excluded from the comparison, since they legitimately change while reviewers
work. See `maint/release_identity_check.json`.

`app.js`, the HTML, CSS, bundled instructions and items, image bytes, neutral
IDs, fixed permutations, the alias-order algorithm, the scientific
protocol/schema, the mappings and the commitments are **all unchanged**. No
questionnaire ZIP or release was rebuilt, replaced, uploaded or created.

---

## 2. Companion validator repaired (`validator-r2`)

`scripts/validate_export.py` **crashed with a traceback** on malformed JSON
*types*, sometimes after already recording a type error. All reported cases were
reproduced first:

| input | previous behaviour |
|---|---|
| `schema_id=[]` | `TypeError: unhashable type: 'list'` |
| `completion="bad"` | `AttributeError: 'str' object has no attribute 'get'` |
| `exposure.other_exposure_note=7` | `AttributeError: 'int' object has no attribute 'strip'` |
| `response.item_id=[]` | `TypeError: unhashable type: 'list'` |
| `response.target=[]` | `TypeError: unhashable type: 'list'` |
| `export_kind=[]` | `TypeError: unhashable type: 'list'` |
| `response.rendering={}` | `TypeError: unhashable type: 'dict'` |
| mixed unknown-ID types | `TypeError: unhashable type: 'dict'` |

**Repair.** Explicit type guards were added **before** every set-membership
test, set insertion, sort key, `.get()` and `.strip()` — covering `schema_id`,
`export_kind`, `reviewer_alias`, `item_id`/`pair_id`, `target`, `rendering`,
`comparison`, `exposure.other_exposure_note`, the supplied `completion` block,
and the `--cohort` alias path. Malformed input now yields a **structured invalid
result with no traceback and no partial processing**, and a batch **continues
processing neighbouring files** and reports the failures.

**Unchanged:** acceptance of correctly generated schema-1.1.0 exports, and every
frozen rule — exact expected IDs, required declarations, allowed enums, partial
drafts, complete Finals, `uncertain`/`unable_to_judge` as valid completed
answers, completion recomputed rather than trusted, and legacy exports rejected
without relabelling or migration. **No new eligibility rule, scoring or analysis
was introduced**, and the cohort audit still states that it **cannot establish
distinct independent humans**.

The validator carries its own maintenance revision marker; **the questionnaire
app version and export schema are untouched at `1.1.0`.**

### Regression evidence

**45/45 CPU-only synthetic checks pass** — `maint/validator_r2_report.json`.
They cover every crash class above plus wrong-type enums, alias and cohort
paths, malformed/non-object/empty input, the full frozen rule set, legacy
rejection wording, mixed valid+invalid batches, `--cohort`, and a check that the
validator **never modifies an input file**.

Acceptance parity was demonstrated by running the **pre-repair** validator and
the repaired one over the same valid fixtures and comparing validity, recomputed
completion and missing-ID sets: **identical**.

> Every fixture is machine-generated and clearly synthetic. **None is a human
> judgement, an image assessment or a reviewer response**, and none may be used
> as such. No real participant file or browser storage was read.
>
> **These are Python structural tests, not browser or scientific validation.**
> The questionnaire is unchanged, so **no browser rerun was needed**; the
> existing **150/150** browser evidence stands as the host-performed browser
> record.

---

## 3. Documentation corrections

### 3.1 Current export schema

The README's schema table still showed `THR-EXPORT-SCHEMA-1.0.0`. The current
export schema is **`THR-EXPORT-SCHEMA-1.1.0`** (app `1.1.0`, patch01).

### 3.2 Ordering provenance

The README still said all orderings were "computed from the public IDs and alias
only, **independent of the arm mapping**". Corrected:

* **`V1`/`V2`/`V3` are pair-aware** — built **using private pair membership** to
  keep paired items ≥4 positions apart. They are **not** computed from public
  data alone and **not** independent of pairing. They do **not** explicitly use
  which member is M0 and which is edited; the spacing rule reveals **negative
  pairing constraints**, not the key.
* **Only the alias-derived shuffle is pairing-independent**, and it carries
  **no minimum-spacing guarantee**.

### 3.3 Commitment scope

`PRIVATE_RECOVERY.md` now states that the commitments cover the **populated
mappings, order variants and item-ID list — not the whole Task-B package**, and
that they support a **future opening anchored to public publication** of the
commitments, **not** independent proof of a self-declared private freeze
timestamp.

### 3.4 Original-release preservation — claim narrowed

Amendment 01 said the original release's "tag and its assets are preserved
unchanged". **The bytes were restored exactly and the tag target never moved**,
but the asset was **deleted and re-uploaded**, so its **asset ID and
created/updated timestamps changed**. Literal claims that all asset identities
remained unchanged are **withdrawn**. The incident disclosure is **retained**.

### 3.5 Historical QA preservation — claim corrected

Amendment 01 said the original 18-fixture and 30-check QA reports were "retained
as historical evidence". Accurately:

* `qa/validator_qa_report.json` — **unchanged** at its original path.
* `qa/browser_qa_report.json` — **overwritten in place** when the suite was
  re-run.
* `verify/published_verification.json` — **removed** from the current tree.
* `verify/masking_audit.json` — **overwritten in place**.

Those earlier versions remain in Git at `d7921df8…`. **Exact byte-identical
copies are now restored under [`archive/d7921df/`](archive/d7921df/)**, each
recording its source commit, source path, byte count and sha256 in
`archive/d7921df/ARCHIVE_MANIFEST.json`. Current QA files were **not**
overwritten and historical contents were **not** edited. No claim is made that
they stayed unchanged at their original paths.

### 3.6 Unintended previews — scope of the disclosure

The publication and removal of two item-view previews is **retained** as
disclosed. Two things are now stated explicitly:

* **Removal does not establish that nobody accessed them** while they were live.
* **A real image shown beside its neutral ID establishes only that one
  ID-to-image link was visible.** It does **not** by itself reveal whether that
  image was the **M0** or the **edited** member of its pair, and it is **not** a
  blanket unblinding.

No guarantee of privacy is claimed, and **no reviewer restart is implied or
needed** on this basis.

### 3.7 Verification snapshots vs final inventories

Earlier verification records are **snapshots taken at a point in time**, not
final inventories. Each audit's actual coverage is now stated where it is cited:

| record | what it actually covered |
|---|---|
| `verify/source_verification_summary.json` | the 40 source PNGs and pairing records, before packaging |
| `archive/d7921df/verify_published_verification.json` | the **original** (superseded) ZIP as published at that time |
| `verify/published_verification_patch01.json` | the **current** patch01 ZIP: fresh download, 40 image hashes, support files, private-association audit, browser re-run on the downloaded bytes |
| `archive/d7921df/verify_masking_audit.json` | the masking audit **as of `d7921df`**, before preview/release-asset coverage existed |
| `verify/masking_audit.json` | the current audit, which **also** covers staged previews and the live asset list of both releases |
| `archive/d7921df/qa_browser_qa_report.json` | the original 30-check, Firefox-only browser run |
| `qa/qa_patch01_downloaded_report.json` | 150 checks across three engine configurations, on the downloaded patch01 bytes |

### 3.8 Participation wording

Earlier text said no reviewer had been recruited or contacted and no participant
existed. **That is no longer current**: the user is distributing the corrected
Task A independently of this repository.

Corrected wording states only what this repository can support: **this package
and this maintenance collect no responses and report no outcomes**, and no
participant information, label or result is held or described here. It makes no
claim about whether participants exist or who has been invited.

---

## 4. Maintenance accounting

Recorded **separately**; the historical preparation clock is **not reset and not
retroactively extended** (`maint/maintenance_clock.json`).

| | used | cap |
|---|---|---|
| Elapsed | well under the cap | 2 h |
| Worker threads | 8 | 8 |
| Additional scratch under `/data` | a few MiB | 1 GiB |
| GPU / model / training / inference / judging | **0** | 0 |
| Questionnaire changes | **0** | 0 |
| Release, tag or asset changes | **0** | 0 |
| Browser runs | **0** (not needed; questionnaire unchanged) | — |
| Real participant files read | **0** | 0 |

---

## 5. What this does and does not establish

It establishes that the **coordinator-side tooling and documentation** are
correct: the validator handles malformed input without crashing, its acceptance
semantics are unchanged, and the current documentation no longer contains the
stale or overstated claims listed above.

It establishes **nothing** about erasure efficacy, retention, quality, evaluator
bias, novelty or paper readiness, and it is **not** evidence about any
participant, label or outcome.
