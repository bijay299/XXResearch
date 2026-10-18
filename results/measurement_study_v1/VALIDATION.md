# Preparation-validation record

All validation used **synthetic fixtures only**. No human response was simulated as data:
the scripted clicks exist to exercise the instrument, and their outputs were discarded.
Synthetic fixtures, fixture bundles and test answer keys are **not** in the reviewer ZIP.

## Browser validation — performed, not claimed

| | |
|---|---|
| browser | **Chromium 131.0.6778.33**, headless, driven by Playwright 1.49.0 |
| loaded from | `file://` (exactly how a reviewer opens it after unzipping) |
| result | **47 / 47 checks passed** |
| record | `validation/validation_results.json`, suite in `validation/run_validation.js` |

### Real defects the validation found and fixed

The suite was worth running: it found three genuine defects that inspection had missed.

1. **A reviewer's typed explanation could be lost.** The free-text box saved only on `blur`,
   so a note typed immediately before closing the tab was never persisted. Fixed: debounced
   save on `input`, plus saves on `blur`, `pagehide` and `visibilitychange`.
2. **The assigned order variant was silently ignored on resume.** Restoring a session
   overwrote the variant chosen at the gate with the stored one, so a reviewer assigned
   variant B or C would silently keep variant A on every resume. Fixed: the
   coordinator-assigned selection wins; answers are keyed by item id and survive.
3. **Export controls were unreachable on an empty package.** They sat inside the review card,
   which is hidden when there are no study items. Fixed: moved to their own always-visible
   card, so the export mechanism can be exercised even now.

A fourth and fifth defect were in the test suite itself (all malformed-import fixtures
written eagerly to one filename; a tap-target measured on a hidden element) and were fixed
before the suite was trusted.

### What was checked

- **Gate**: empty alias, malformed alias and missing prior-exposure answer are all rejected.
- **No inferred answers**: nothing is preselected; an unanswered item exports as `null` and
  is counted as missing, never guessed or defaulted.
- **Uncertainty**: `uncertain` and `unable to judge` persist as real answers.
- **Autosave / resume**: position, radio answers and free text survive a reload.
- **Order**: stable for the same alias+variant; different between reviewers; three variants
  give three different orders; derived purely from alias+variant.
- **Export**: draft always available; final blocked until every item is answered; carries
  stage, package, protocol and schema ids, alias, order variant, exposure declaration,
  neutral ids and positions; contains no prompt, seed, file, arm or method field.
- **Import**: round-trips into a clean session; rejects malformed JSON, wrong stage, wrong
  package id, wrong schema version and a missing responses array; reports unknown ids and
  duplicates instead of silently accepting them.
- **Per-reviewer isolation**: a second alias starts clean on the same browser; storage is
  namespaced per stage / package / alias.
- **Zoom and navigation**: full-resolution zoom toggles both ways; first/last bounds hold.
- **Mobile**: no horizontal overflow at 390 px; answer targets 43 px tall.
- **Stage separation**: stage B shows the reference pane and its own question set, tags its
  export `stage: "B"`, and refuses a stage-A file.
- **Public bundle**: shows the 12 practice items, states plainly that it holds no study
  items, raises no JavaScript errors, contains no `ref_file` and no arm/method/seed/prompt
  key, and still exports a well-formed empty response file.
- **Offline**: every `http(s)` request was intercepted; **zero** were attempted.

## Analysis-code validation (synthetic fixtures only)

- `analysis/validate_responses.py` correctly flagged a malformed fixture: non-conforming
  alias, an invalid enum value, a duplicate `neutral_id`, and a disagreeing count; and
  warned that a `draft` file is not admissible.
- `analysis/analyze.py` on fixtures: produced the reference convention counts with
  `no_reference` preserved; reported sensitivity/specificity **only** where a denominator
  existed (returning `null` where it did not); attached coverage and uncertain-case bounds;
  ran the cluster bootstrap over targets then cases; and emitted paired transitions.
- **Refusal guard verified**: given a response file whose `package_id` is not a fixture, the
  analysis refuses to run and explains that human-outcome analysis requires first-stage
  closure, locked responses and separate explicit authorisation.
- With fewer than two `final` returns it reports the primary analysis as **NOT RUN** rather
  than approximating it.

## Masking audit

`MASKING_AUDIT.json` — run over the public build **and** over the extracted ZIP:
**0 blocking findings.** Checked for the private salt and seed, arm/method/condition/prompt/
seed/source mapping keys bound to literal values, Task-B reference associations, automated
predictions or scores, external `http(s)` references, base64 blobs (decoded and inspected),
PNG `tEXt`/`iTXt`/`zTXt` metadata chunks, and non-neutral filenames.

Two matches are **allowlisted with printed justification**: in `analysis/analyze.py` the
words `method` and `condition` appear as output **field names** (one bound to a descriptive
label, one to a variable). They carry no mapping, and the file contains no condition
literals bound to items and no neutral-id literals.

The audit was run again over the **published directory**, which additionally contains the
build code. There it allowlists the prompts and seeds in `code/practice_manifest.json` and
`code/gen_practice.py`: these are **practice** provenance — non-study instructional aids
built on an object (*bicycle*) declared not to be a study target, not shipped inside the
reviewer ZIP, and deliberately public so that readers can see exactly how the practice
examples (including the two intentionally degraded ones) were produced. They cannot leak a
study arm, since no study item exists. Every allowlisted match is printed with its
justification in `MASKING_AUDIT.json` rather than silently dropped.

Practice PNGs were re-encoded to drop container metadata; **decoded pixels verified identical
for all 12** (source and published hashes recorded privately).

## Limitations of this validation

- Chromium only. Firefox, Safari and Edge were **not** exercised; the viewer uses no
  browser-specific API, but that is an expectation, not a measurement.
- `localStorage` under `file://` shares one origin across local pages, so per-reviewer
  isolation rests on the key namespace, not on browser origin separation. A reviewer using
  the same browser profile for two aliases is isolated by key; a reviewer clearing site data
  loses the draft. Both are stated in the reviewer instructions.
- No real reviewer has used the package. Usability beyond these mechanical checks is unknown.
- The study flow was exercised on **synthetic** items; it has never run on real study images,
  because none exist yet (see `ACCESS_CHECK.md`).
