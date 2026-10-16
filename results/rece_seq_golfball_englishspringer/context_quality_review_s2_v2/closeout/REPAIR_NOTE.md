# v2 viewer repair note (`-v1` → `-v2`, 2026-10-16)

**This is a questionnaire software/documentation correction, not a new
research experiment or a new package design.** No model endpoint, GPU
work, image generation, classifier run, pair membership, pair order,
A/B assignment, success threshold, or qualitative criterion changed.
No v2 response is known to exist for either revision.

## Source of the report

An external review (Codex) downloaded and independently verified the
actual published `-v1` zip (bundle `V2_VIEWER_REPAIR_BUNDLE.zip`,
supplied by the user). Its accepted checks — live branch match at
`321b83679f2b4b81b1db5e51a59abff8dd7e1a59`, all 41 archive file hashes,
the unchanged 40-image multiset shared with v1, the prompt multiset,
v1's own subtree untouched — were **independently re-confirmed here**
before any repair was made (the bundle's "original" `index.html` was
byte-identical to the actually-published file, sha256
`6580cfcc95147d5bb0749098eb827c67bd5d5772bdbaffea0baa594f883431f3`).
The bundle's own synthetic test harness and proposed patch were treated
as a reference proposal, not applied blindly — both defects were
independently reproduced here, the proposed patch was reviewed in full
before being applied, and the fix was independently re-verified with a
separately written regression harness (below), not by trusting the
bundle's own reported numbers.

## Defect A — reason entry did not refresh dependent UI

**Symptom**: `reviewer_package/index.html`'s `reason-a-box`/`reason-b-box`
`input` handlers updated `state` and `localStorage` but never called
`render()`. If a reason was the *last* missing mandatory field, the
paired-comparison question and the progress/export controls stayed
stale (hidden, or still-visible-when-incomplete) until the reviewer
navigated to another pair and back.

**Fix**: a new `saveReasonAndRefresh(pairId, field, element)` function
saves the field, calls `render()` immediately, then restores input
focus, caret position (`selectionStart`/`selectionEnd`), and page
scroll position on the freshly re-rendered textarea. Both reason
listeners now call this function instead of updating state directly.

## Defect B — imported exports could not resume

**Symptom**: `doExport()` produces `{..., responses: [ {neutral_id, ...},
... ]}` — an **array** keyed by row. The `import-file` change handler
did `state = imported;`, directly assigning this export shape to the
live UI state, which instead needs `responses` keyed **by neutral ID**
plus defined `currentIndex`/`setupComplete`. After import,
`state.currentIndex` was `undefined`; clicking "Begin" then crashed
with `TypeError: Cannot read properties of undefined (reading 'id')`
(`PAIRS[undefined]` is `undefined`, then `.id` throws).

**Fix**: a new `restoreExport(payload)` function rebuilds a clean
`freshState()`, then: rejects a `packageVersion`/`protocolVersion`
mismatch; rejects a `responses` array whose length isn't exactly 20;
rejects any missing, unknown, or duplicate `neutral_id`; validates every
field's value against its allowed answer set (rejecting anything else,
e.g. `"maybe"`); preserves every field's actual string value verbatim,
including empty strings (a deliberately missing answer), `"unable"`
answers, and optional style fields; validates and copies `reviewer`
fields; preserves `startedAt`/`exportedAt`; and sets `currentIndex = 0`,
`setupComplete = false` (the reviewer resumes via the normal Begin
button, at a defined pair). The import handler now calls
`state = restoreExport(imported);` inside its existing `try`/`catch`,
so a rejected import still surfaces the existing "Could not read that
file" alert rather than crashing later on Begin.

**No human judgment is inferred, rewritten, or filled in by either
fix.** Malformed or incompatible imports are refused, not repaired.

## Wording clarifications (HTML, preview, protocol — applied consistently)

| issue | fix |
|---|---|
| Claimed a "subject mismatch" field exists and tracks compliance separately | Removed; states plainly the questionnaire scores breakdown, local defects, and optional style only, and does not score subject/scene compliance |
| "a photo might be implied" suggested photography was a default expectation | Replaced with "style mismatch applies only when the prompt explicitly requests a style" |
| Could be read as implying arbitrary loss of all prompt context is itself desirable | Kept the "absence expected, substitute not a defect" framing, removed language that could be read as endorsing arbitrary context loss |
| Local-defect wording could be read as requiring the rest of the image to be fully coherent | Clarified: local defects affect a specific part; both the local and widespread questions can independently be "Yes" |
| "answering consistently [with the generic examples] is not required or checked" | Replaced: the examples explain the definitions to apply throughout; study images need not resemble them, but the same definitions apply |
| "these 20 specific study images" (case-feedback question, in HTML, preview, protocol, README) | Corrected to "this review's 40 images (20 pairs)" everywhere |
| Protocol's numbering implied the paired question needs items 1–8 (including optional style) answered | Corrected: unlocked by the mandatory fields only (both breakdown questions, both local-defects questions, and each reason whenever applicable) — optional style and an inapplicable reason never block it |

Applied in `reviewer_package/index.html`, `INSTRUCTIONS_PREVIEW.md`, and
`FROZEN_PROTOCOL_V2.md` consistently. No new metric, score, or threshold
was added anywhere in this pass.

## Independent repair verification (fresh harness, not the bundle's own)

A separate Node.js regression script
(`/data/bijaypandey/claude_tmp/scratch_check/independent_repair_verification.js`,
not published — scratch only) was written from scratch and run against
**both** the pre-repair and post-repair published files, exercising the
actual shipped functions (`saveReasonAndRefresh`, `restoreExport`,
`render`) rather than reimplemented completeness predicates, with
clearly marked synthetic data (`"SYNTHETIC: ..."` strings, never
published):

| check | pre-repair (`-v1`) | post-repair (`-v2`) |
|---|---|---|
| Paired question hidden before reason entered | true (correct) | true (correct) |
| Paired question visible **immediately** after typing the last required reason, no navigation | **false (defect reproduced)** | **true (fixed)** |
| Paired question hides again immediately after clearing that reason | n/a (never updated) | **true (fixed)** |
| Importing a real final export, then clicking Begin | **throws `TypeError: Cannot read properties of undefined (reading 'id')`** (exact reported error, reproduced independently) | **no error; first pair's preserved reason text, "unable" value, and optional-blank style all confirmed intact** |
| Duplicate neutral ID in an import | n/a | **rejected** |
| Invalid field value (e.g. `"maybe"`) in an import | n/a | **rejected** |
| Wrong `packageVersion` in an import | n/a | **rejected** |
| Incomplete draft import: a missing reason stays missing (not auto-filled) | n/a | **confirmed — stays `""`, not silently completed** |

Separately, `node --check` confirms valid JavaScript syntax, and brace/
paren/backtick counts and `id=`/`getElementById` cross-references are
balanced and consistent, same as every prior check in this results
line.

**What remains unverified, disclosed rather than assumed**: no browser
or headless-browser tool was available in this environment (same
limitation as every prior package in this line). Actual visual
rendering, layout, real click/keyboard event wiring, the native file
picker, and native file download were **not** exercised in a real
browser. The user should open the repaired `index.html` directly to
confirm visual/interactive behavior before arranging a review.

## What did not change

- The 40 source images: byte-identical (hashes re-verified, spot-checked
  against the private mapping's own recorded hashes).
- The v2 private mapping and its random seed: **not regenerated, not
  touched.** `build_v2_reviewer_package.py` was not re-run.
- The published mapping commitment,
  `bd360bc35c45d5eb68f5ff6e91aedb1db30cd70e10a9b55023bb2d330fbbc529`:
  unchanged, re-verified against the still-unchanged private mapping
  file before publishing this repair.
- v1 (`blinded_quality_review_s2/`), its selected response, and its
  verified unblinding result: untouched (not part of this commit's
  diff). The withdrawn v1 survey response remains unanalyzed and
  unpublished.
- Pair membership, R01–R20 order, A/B assignment, the qualitative
  degeneration/local-defect criterion, and every prompt: unchanged.

## Version identifiers incremented

| | `-v1` (original) | `-v2` (this repair) |
|---|---|---|
| `PACKAGE_VERSION` | `context_quality_review_s2_v2-v1` | `context_quality_review_s2_v2-v2` |
| `PROTOCOL_VERSION` | `FROZEN_PROTOCOL_V2-v1` | `FROZEN_PROTOCOL_V2-v2` |
| localStorage key | `context_quality_review_s2_v2_progress_v1` | `context_quality_review_s2_v2_progress_v2` |
| zip sha256 | `e9c9906a8b5633aafc5e718b727e8d493d65d83b7660b6251464c512acc66953` | `b73caf1310d4e9533b5660341e7d46a5b7a8cff2790e4b1d367bcb262b14227d` |

The viewer's own version check means a `-v1` export (none is known to
exist) **cannot** be imported into the `-v2` package, and vice versa —
no compatibility shim is added; an incompatible import is refused with
its existing alert, now backed by a handler that cannot crash on it.
The prior `-v1` revision remains in git history at commit
`321b83679f2b4b81b1db5e51a59abff8dd7e1a59`, unmodified.
