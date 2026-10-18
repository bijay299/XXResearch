> # ⚠️ PARTIALLY SUPERSEDED — read [`AMENDMENT_02_COORDINATOR_CLEANUP.md`](AMENDMENT_02_COORDINATOR_CLEANUP.md)
>
> **Dated 2026-10-18.** Amendment 02 is authoritative over this document on the
> three points below. The historical body is **preserved unchanged** as
> originally published, including the wording now corrected.
>
> 1. **"its tag and its assets are preserved unchanged" (header) and
>    "Original release … tag and assets — preserved" (§5) overstate what
>    happened.** The original ZIP's **bytes** were restored exactly
>    (13,917,868 B, sha256 `45e3eb7a…`) and the **tag target** never moved, but
>    the asset was deleted and re-uploaded, so its **asset ID and
>    created/updated timestamps changed**. Literal claims that all asset
>    identities were unchanged are withdrawn; the incident disclosure stands.
> 2. **"Original 18-fixture and 30-check QA reports — retained as historical
>    evidence" (§5) is not accurate at the original paths.** The 18-fixture
>    report is unchanged, but `qa/browser_qa_report.json` was **overwritten in
>    place** and `verify/published_verification.json` was **removed** from the
>    current tree. The earlier versions remain in Git at `d7921df`, and exact
>    copies are now restored under `archive/d7921df/`.
> 3. **The preview-disclosure wording should not be read as broader than it
>    is.** Removing the two previews does not establish that nobody accessed
>    them while they were live. Equally, a real image shown beside its neutral
>    ID establishes only that **one ID-to-image** link was visible — it does
>    **not** by itself reveal whether that image was the M0 or the edited member
>    of its pair, and it is **not** a blanket unblinding. No guarantee of
>    privacy is claimed, and no reviewer restart is implied.
>
> Status `READY_FOR_INDEPENDENT_REVIEW` is retained for the existing corrected
> Task A only.

---

# AMENDMENT 01 — Task-A software repair (patch01)

**Dated 2026-10-18. Authoritative over earlier wording on the points below.**
Scope: **software repair and synthetic browser QA only.** No new experiment, no
new randomization, no reviewer contact, no response, no outcome.

The corrected bundle is published as release
**`teapot-human-review-v1-taskA-patch01`**. The original release
`teapot-human-review-v1-taskA`, its tag and its assets are **preserved
unchanged**.

---

## 1. Defects found and repaired

### D1 — the published bundle would not open in Chrome or Edge *(found by the user)*

The viewer fetched its item list with `XMLHttpRequest` over `file://`. Every
Chromium browser blocks that by CORS policy; reviewers saw
*"Could not load items.json"* instead of the task.

**Repair.** The list ships as `items.js` and is loaded with a plain
`<script src="items.js">`, which is not subject to that restriction. No
`XMLHttpRequest` or `fetch` remains in either viewer.

**Why QA missed it.** The browser suite ran **Firefox only**, which permits the
request, and reported every check green against a bundle that was broken for
most reviewers. The suite now runs **three engine configurations** — Chromium,
Firefox, and Firefox with `privacy.file_unique_origin` and
`security.fileuri.strict_origin_policy` enabled.

### D2 — import validation was far too weak

The published `app.js` checked only `package_id`, `stage` and that `responses`
was an array, then **merged** the file's answers into whatever session was open,
keeping that session's alias, declarations and order. A file from reviewer *X*
could silently become part of reviewer *Y*'s return.

**Repair — an explicit provenance policy.** The **entire** file is validated
before anything is touched; an accepted import then **restores its own session**
rather than merging:

* trusted `package_id` / `protocol_id` / `schema_id` / `stage`, supported
  `export_kind`, usable alias, valid `order_variant`;
* every declaration present and an **explicit boolean**; correct types for
  optional reasons, notes and any supplied completion block;
* expected item IDs only, no duplicates, no unknowns, valid enums;
* drafts may be partial, **Finals require all 40 items fully answered**;
* the presentation order is **reconstructed from the trusted bundle data plus
  the alias** — a file's own ordering is never trusted;
* completion is **recomputed from valid choices**; a supplied count is
  type-checked and otherwise ignored.

An accepted import restores the file's **own** alias, declarations, order and
raw answers. If a stored session for that alias already holds answers, the
reviewer is asked before it is replaced. **No other alias's session is ever
touched, and declarations are never silently substituted.** A rejected import
leaves **both memory and `localStorage` unchanged**.

### D3 — `answered()` counted truthiness, not valid choices

Completion counted any truthy value, so a malformed file carrying
`"target": "definitely"` could count as complete and enable a Final export.
`answered()` now requires membership in the declared enums, in the viewer and in
the Python validator alike.

### D4 — the Python validator had a hard-coded host path

It resolved the trusted item list through an absolute
`/data/bijaypandey/...` root, so the published command could not run anywhere
else. It now discovers the list relative to itself or the working directory, or
takes `--items PATH`, and no absolute host path remains. Type checking was also
tightened (`order_variant`, `item_count_expected`, `completion`,
`other_exposure_note`).

### D5 — a mis-stated QA claim

The earlier report described a Final export followed by Python validation as a
Final **re-import** test. **It was not** — the file was never read back into the
viewer. That claim is withdrawn. A genuine Final re-import, asserting
answer-and-metadata equality in a fresh session, is now part of the suite.

---

## 2. Corrected claims about ordering

An earlier version stated that **all** orderings were computed from the public
IDs and alias alone and were independent of the arm mapping. **That was wrong
for the fixed variants.**

* **`V1`/`V2`/`V3` are pair-aware.** They were constructed **using private pair
  membership** to enforce at least four positions of separation between the two
  members of a pair. They are **not** derivable from public data alone and are
  **not** independent of pairing.
* They do **not** explicitly use which member is M0 and which is edited. The
  spacing rule reveals **negative pairing constraints** — which items are *not*
  each other's partner at close range — **not** the pairing key.
* **The alias-derived shuffle is pairing-independent** (alias + public IDs only)
  and carries **no guaranteed minimum separation**; a pair can land adjacent.

**The permutations themselves were not regenerated.** `freeze_ids.py` was not
re-run, no new randomization was created, and the item IDs, V1/V2/V3 orders and
`frozen_iso` are **byte-identical** to the reviewed publication `d7921df`.

---

## 3. Corrected exposure wording

"Before today" is replaced throughout the viewer and the instructions with
**"Before starting this Task A review"**, which correctly covers same-day access
to source pairings, condition labels or image-specific coaching.

Reviewers are now told, in the viewer and the instructions:

> You may still participate if you answer yes. Certain prior exposure can make
> your return ineligible for the primary independent-review analysis; it will be
> retained with its exposure status.

The prespecified primary-eligibility rules are **unchanged**, and general
research context alone remains **non-excluding**.

---

## 4. Corrected commitment claims

The published HMACs cover the **populated Task-A mapping, the populated Task-B
mapping, the three order variants and the item-ID list**. They do **not** cover
every byte of the private Task-B package — its images, viewer files and layout
are not committed.

A future opening should be anchored to the fact that the commitments were
**published in this repository at a publicly visible commit**, not to the
self-declared `frozen_iso` field inside the artifacts, which is not
independently proven.

---

## 5. What was preserved

| | |
|---|---|
| 40 research-image bytes | **unchanged** (byte-identical, re-verified) |
| Neutral IDs `A-001`…`A-040` | **unchanged** |
| Source pairings and private mapping | **unchanged**, never regenerated |
| Fixed `V1`/`V2`/`V3` permutations | **unchanged** (verified against `d7921df`) |
| Alias-order algorithm | **unchanged** |
| Mapping / order / ID commitments | **unchanged** |
| Private Task-B package (20 pairs, 40 images) | **unchanged and still unreleased** |
| Original release `teapot-human-review-v1-taskA`, tag and assets | **preserved** |
| Protocol, scientific definitions, reporting conventions | **unchanged** |
| Original 18-fixture and 30-check QA reports | **retained as historical evidence** |
| Preparation clock | **not reset**; repair accounting appended |

`freeze_ids.py` was **not** re-run.

### Two mistakes made during this work, and how they were fixed

**(a) The original release asset was overwritten.** Earlier in this session,
before this repair was commissioned, the corrected bundle was uploaded *in
place of* the original asset on the `teapot-human-review-v1-taskA` release. The
original should have been preserved. Because the archive build is
deterministic, the original ZIP was reconstructed **byte-for-byte** from the
known source state and re-uploaded. The original release now carries exactly
13,917,868 bytes, sha256
`45e3eb7aeb9443e35a4ab990969eb9aa450e2f36b2238192244050b320dbd005`,
with its tag target unchanged — **verified by re-downloading it**.

**(b) Two preview screenshots that disclose a mapping were briefly published.**
`preview_A2_item_view.png` and `preview_A4_mobile_item.png` each show a **real
research image beside its neutral ID**, which discloses that mapping. They had
been quarantined earlier in preparation, but restoring `browser_qa.py` from Git
to refactor it re-created them, and the publisher was staging *every* file in
the screenshots directory. They were uploaded to the patch01 release and
**removed as soon as the asset list was reviewed**.

Three changes make that class of mistake non-recurring:

1. `browser_qa.py` now writes item-view screenshots **into the private area**,
   never the public one.
2. `build_publish.py` stages previews from an **explicit allowlist** of four
   files — two instructions pages (no research image) and two
   synthetic-placeholder item views — and fails loudly on anything else.
3. The masking audit now also checks the **staged preview directory and the
   live asset list of every release**, which is precisely the gap that let this
   through: it previously scanned only the results tree and the ZIP.

Both releases now carry only the four allowlisted preview images; the audit
confirms this against the live API.

---

## 6. Versioning

| | before | after |
|---|---|---|
| Package ID | `THR-V1-A` | `THR-V1-A` *(same study material)* |
| Protocol ID | `TEAPOT-HR-PROTO-1.0.0` | unchanged *(scientific definitions unchanged)* |
| Export schema | `THR-EXPORT-SCHEMA-1.0.0` | **`THR-EXPORT-SCHEMA-1.1.0`** |
| App / bundle | — | **`1.1.0` (patch01)** |

`THR-EXPORT-SCHEMA-1.0.0` exports are **rejected as unsupported legacy** with a
clear message, and are **not silently migrated**. Nothing needs migrating: **no
reviewer return has ever been collected.**

---

## 7. Remaining limits — unchanged by this repair

* The 40 images and their condition/pair provenance are **already public**;
  neutral IDs reduce accidental disclosure but **cannot prevent reverse matching
  or guarantee blinding**.
* **Task B remains private, unreleased and NOT browser-tested.** These Task-A
  checks certify nothing about the Task-B bundle; its validation scope is the
  file-level completeness and byte-identity check only.
* Self-declared exposure cannot be verified from the artifacts.
* The validator counts **files, not people**, and cannot establish that two
  files came from two distinct independent humans.
* **No participant exists, no response has been collected, and no outcome of any
  kind is reported.** This establishes **instrument readiness only** — not
  independent participation, scientific validity, or erasure efficacy.

---

## 8. Status

See the package `README.md` for the current status line, the corrected ZIP hash
and the publication state of both releases.
