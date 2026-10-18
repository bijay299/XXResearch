# EXPORT SCHEMA — `THR-EXPORT-SCHEMA-1.1.0`

Reviewer exports are plain JSON files produced entirely in the browser. Nothing
is uploaded; the reviewer saves a file and sends it however they choose.

The reference implementation of the checks below is
[`scripts/validate_export.py`](../scripts/validate_export.py). It contains **no
absolute host path**: it discovers the trusted item list relative to itself or
the working directory, or takes `--items PATH`. From this results directory:

```bash
python3 scripts/validate_export.py EXPORT.json --expect-stage A \
        --items protocol/taskA_items.json
```

## Common envelope

| field | type | notes |
|---|---|---|
| `package_id` | string | `THR-V1-A` (stage A) or `THR-V1-B` (stage B) |
| `protocol_id` | string | `TEAPOT-HR-PROTO-1.0.0` |
| `schema_id` | string | `THR-EXPORT-SCHEMA-1.1.0` (`1.0.0` is legacy: **rejected, not migrated**) |
| `app_version` | string | bundle app version, e.g. `1.1.0` |
| `stage` | `"A"` \| `"B"` | |
| `export_kind` | `"draft"` \| `"final"` | |
| `reviewer_alias` | string | non-empty; a self-chosen label, **not** an identity |
| `order_variant` | string | `V1`/`V2`/`V3`/`ALIAS` |
| `exposure` | object | **required, explicit** — see below |
| `exported_iso` | string | ISO timestamp written by the browser |
| `item_count_expected` | integer | |
| `responses` | array | see per-stage shape |
| `completion` | object | **self-reported; never trusted** — recomputed |

No name, email, IP, device identifier, cookie or analytics field is collected.
The alias is the only reviewer-supplied identifier.

### Exposure — must be explicit booleans

Stage A requires `declared: true` plus booleans for
`prior_access_to_source_pairings`, `prior_access_to_condition_labels`,
`image_specific_coaching_before_task_a`.
Stage B requires `declared: true` plus
`task_a_final_exported_before_task_b`, `prior_access_to_condition_labels`,
`image_specific_coaching_before_task_a`, and records
`reference_identity_disclosed_in_this_stage: true`.

An optional free-text `other_exposure_note` captures other exposure or
background. A missing key, a `null`, or a string such as `"no"` instead of a
boolean is a **validation error** — declarations must be explicit.

## Stage A responses

```json
{ "item_id": "A-017", "target": "present",
  "rendering": "localized_deformation", "reason": "" }
```

* `target` ∈ `present` | `absent` | `uncertain`
* `rendering` ∈ `coherent` | `localized_deformation` | `whole_frame_breakdown` | `unable_to_judge`
* `reason` optional string

**`uncertain` and `unable_to_judge` are valid COMPLETED answers**, not missing
data. An item counts as answered only when *both* questions carry a valid enum.

## Stage B responses

```json
{ "pair_id": "B-07", "comparison": "about_the_same", "reason": "" }
```

* `comparison` ∈ `less` | `about_the_same` | `more` | `cannot_fairly_compare`
* `cannot_fairly_compare` is likewise a valid completed answer.

## Validation rules

| rule | behaviour |
|---|---|
| **Task-A Final** | must carry **exactly the 40 expected IDs**, each distinct, each with **both** answers valid |
| **Task-B Final** | must carry **exactly the 20 expected pair IDs**, each with a valid `comparison` |
| **Drafts** | may be partial |
| **Empty or partial Final** | **fails** |
| **Trusted sets** | `package_id`, `protocol_id`, `schema_id`, `stage` and the expected ID set are enforced **by default** |
| **Duplicate IDs** | rejected |
| **Extra / unknown IDs** | rejected |
| **Wrong stage or package** | rejected (and the viewer refuses such an import) |
| **Invalid enum value** | rejected |
| **Malformed JSON** | rejected with an error, never a crash |
| **Supplied `completion`** | **never trusted** — completion is recomputed from the raw choices; a disagreement is reported as a warning and the recomputed value governs |
| **Duplicate files / repeated alias** | flagged by the cohort audit; **never counted as independent reviewers** |

### What the validator does not do

It validates **file structure only**. It makes no judgement about image content,
applies no automated scoring, and **cannot establish that two files came from
two distinct independent humans**. Independence is a procedural claim about
people, not a property the schema can check.

## Import policy (viewer) — restore, never merge

An accepted import **restores a validated snapshot of its own session**. It is
never merged into whatever session happens to be open.

1. **Validate the whole file first.** Trusted package / protocol / schema /
   stage, supported `export_kind`, a usable alias, a valid `order_variant`, all
   declarations present and explicitly boolean, correct types for optional
   reasons and notes and for any supplied completion block, expected item IDs
   with no duplicates or unknowns, and valid enums. Finals additionally require
   all 40 items fully answered. **Nothing is touched until every check passes.**
2. **Reconstruct the order from trusted data** — the bundle's own item list plus
   the alias. An imported file's ordering is never trusted.
3. **Recompute completion** from valid choices; a supplied count is only
   type-checked, never believed.
4. **Restore under the file's own alias**, keeping its declarations, order and
   raw answers. If a stored session for that alias already holds answers, the
   reviewer is asked to confirm the replacement. Any other alias's session is
   left untouched.
5. **A rejected import changes nothing** — not the in-memory session, not
   `localStorage`. The reason is shown.

### Supported workflow for a new computer or browser

Export a draft (or final) on the first machine → open `index.html` on the second
→ use **"Continue from an exported file"** on the first screen → the session is
restored with its **original** alias, declarations, presentation order and
answers. No form needs to be filled in first, and no provenance is rewritten.

**Unsupported legacy exports (`THR-EXPORT-SCHEMA-1.0.0`) are rejected with a
clear message and are not silently migrated.** No real reviewer return exists to
migrate — none has ever been collected.

## Mechanical QA

18 synthetic fixtures exercise the original rules (valid final, all-`uncertain`
final, partial draft, Task-B final, missing item, empty final, duplicate ID,
extra ID, bad enum, missing answer, missing exposure key, non-boolean exposure,
wrong package, wrong stage, wrong schema, partial B final, malformed JSON, and a
duplicate-file cohort audit). All 18 pass; see [`qa/validator_qa_report.json`](../qa/validator_qa_report.json)
(retained as historical evidence for schema `1.0.0`).

The corrected bundle is covered by **150 focused browser checks across three
engine configurations** — [`qa/qa_patch01_report.json`](../qa/qa_patch01_report.json).

**Every fixture is explicitly synthetic and machine-generated.** Fixture choices
come from a fixed rotation and encode no assessment of any image. No fixture is
a human judgement, none may be used as reviewer evidence, and none is placed in
a reviewer bundle or a real-response directory.
