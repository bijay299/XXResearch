# DIGEST RULE MIGRATION — AUDIT-01c

**Nothing about the prompt sets changed.** The records, prompt texts, prompt ids,
generation seeds and counts are byte-for-byte identical before and after. Only
the **digest rule** changed, so both manifests have new identities.

## Why

AUDIT-01b built the drafts with `json.dumps(payload, indent=2, sort_keys=True)`,
while the frozen pilot manifest had been built with
`json.dumps(payload, sort_keys=True)` (no indent). Two rules in the codebase
meant no single recomputable identity. Validation now **recomputes** a
manifest's digest and compares it with a frozen expected identity, so exactly
one rule can exist.

## The canonical rule

```python
sha256(json.dumps({k: v for k, v in manifest.items()
                   if k != "manifest_sha256"}, sort_keys=True).encode())
```

Defined once, in `scripts/seq/validate_stage.py:canonical_manifest_digest`, and
imported by `scripts/seq/build_draft_manifests.py` so a second copy cannot
drift. The rule was chosen to match the **pilot** manifest, which therefore
needs **no migration**: its stored digest
`0020c81c4a4dd3580a6574c91b4aea1a7e349e9e89fc845ed8c2077d47564892` recomputes
exactly.

## Digests

| manifest | superseded (indent=2 rule) | current (canonical rule) |
|---|---|---|
| `dev_manifest_DRAFT.json` | `ebfdd37270dea75012756fdc877fb614a9bd5ae0dd78af76ed426844136e5b89` | **`1cd1f902669cefdc2d2f96ccae8641d02c831bf4f5ecfa09b5844da835063e6b`** |
| `test_manifest_DRAFT.json` | `170696eb5fc2832b832e7e2baca6c820344ac9636c66e510398c3a19e6ef8ae1` | **`5dd87dbb5a77c0f4c75de79a19b15b0b2e9cddc90d08172857700dac1bcc2a95`** |
| pilot `eval_manifest.json` | — | `0020c81c4a4dd3580a6574c91b4aea1a7e349e9e89fc845ed8c2077d47564892` (unchanged) |

## Provenance preserved

The pre-migration drafts are kept verbatim in
[`superseded_indent2_rule/`](superseded_indent2_rule/) — both manifests, the
previous `FREEZE.md` and the previous disjointness report. They are **not**
current and must not be used; they are retained so the migration can be checked
rather than taken on trust.

`scripts/seq/test_validate_stage.py` asserts that the superseded draft **fails**
under the canonical rule and that the current drafts and the pilot manifest
pass — so the migration is demonstrated by test, not asserted here.

Verify by hand:

```bash
python scripts/seq/validate_stage.py manifest dev \
    --manifest results/audit_v1/draft_manifests/dev_manifest_DRAFT.json \
    --expect_manifest_sha 1cd1f902669cefdc2d2f96ccae8641d02c831bf4f5ecfa09b5844da835063e6b \
    --expect_records 80 --expect_prompt_texts 20
```
