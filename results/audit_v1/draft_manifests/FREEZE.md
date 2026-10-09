# FROZEN DRAFT MANIFESTS — AUDIT-01b

**Status: DRAFT, for review. Nothing has been generated from either manifest and
no GPU work is authorised.** These exist so the prompt sets and the analysis
rules are fixed *before* any approved launch, not chosen afterwards.

## Hashes

| manifest | distinct prompt texts | gen seeds | pairs/checkpoint | sha256 |
|---|---|---|---|---|
| `dev_manifest_DRAFT.json` | 20 | [511, 622, 733, 844] | 80 | `1cd1f902669cefdc2d2f96ccae8641d02c831bf4f5ecfa09b5844da835063e6b` |
| `test_manifest_DRAFT.json` | 140 | [1301, 1402, 1503, 1604] | 560 | `5dd87dbb5a77c0f4c75de79a19b15b0b2e9cddc90d08172857700dac1bcc2a95` |

Counting, stated explicitly because the pilot conflated these: the development
set is **20 distinct prompt texts** over 1
categories giving **80 prompt x generation-seed pairs** per checkpoint;
the test set is **140 texts** over 7
categories giving **560 pairs** per checkpoint.

## Disjointness

- Exact collisions, all five pairings (dev/test, dev/pilot, test/pilot,
  dev/anchor-training, test/anchor-training): **0**
- Near-duplicates at similarity >= 0.90, all five pairings:
  **0**
- Anchor training strings checked: **400**
- Generation-seed sets pairwise disjoint across pilot/dev/test:
  **True**
- No generation seed equals a training seed (17, 29):
  **True**

**Shared surface structure is reported, not denied.** Both sets are
template-built. Dev and test share **0** scene tails; see
`disjointness_report.json` for the full analysis. The bootstrap's resampling
unit is the **scene cluster**, carrying both family wordings and all four
generation seeds together (`bootstrap_grouping.json`), which absorbs dependence
*within* a scene. A template shared *across* clusters is not absorbed — the test
set has one, `"a photo of a"` across all 70 literal prompts — and **the
direction and size of its effect on interval width are not established, and no
claim is made about them**. The limitation applies to the pilot set too.

## Rules frozen with these hashes

1. **Selection happens on the development set only.** The matched unregularised
   step is chosen by closest dog suppression to that seed's L2 endpoint,
   earliest step breaking ties, mismatch <= 5 points.
2. **Reference gates**: >= 30 points suppression and <= 60% residue, on the
   development set, against that seed's own MA. These gates make the comparison
   an explicit **partial-suppression** test — they do not describe a strongly
   suppressed regime, and no claim about strong suppression follows from it.
3. **If either seed has no match within 5 points, the planned paired test
   stops** and the bracketing steps are reported.
4. **The frozen test set is evaluated once.** Gates and match are re-checked on
   it and reported as achieved; **nothing is reselected using it**.
5. **Primary endpoint**: the paired bird contrast. Dog residue/suppression, cat
   change from MA, and every retained category are reported **separately and
   never averaged**.
6. **Prespecified material bird advantage: 10 points.** Upper 95% bound below
   +10 in both seeds rules out a benefit of that size, conditionally. Lower bound
   above +10 in both carries a material benefit forward. Anything else is
   inconclusive. A practical-equivalence claim additionally requires intervals
   to lie **wholly within [-10, +10]** in both seeds.
7. **Uncertainty is conditional paired prompt-cluster uncertainty** within one
   training run. Two runs support a direction check; no broad inference from
   n=2, and no pooling across seeds.
8. Generation seeds are evaluation-side RNG only and are never reused as
   training seeds.

## Digest rule

Identities use the ONE canonical rule, `sha256` over
`json.dumps(manifest_without_its_digest_field, sort_keys=True)`, defined in
`scripts/seq/validate_stage.py:canonical_manifest_digest` and imported here so a
second copy cannot drift. Validation **recomputes** this digest from the
manifest's contents and compares it with the frozen identity above, so altering
a prompt text is detected even when prompt ids, the row count and the stored
digest field are preserved.

These drafts were migrated to that rule from an earlier `indent=2` rule; the
prompt sets are byte-identical and only the identities changed. Both digests,
the reason, and the verbatim superseded drafts are recorded in
[`DIGEST_MIGRATION.md`](DIGEST_MIGRATION.md) and
[`superseded_indent2_rule/`](superseded_indent2_rule/). The frozen pilot
manifest needs no migration under this rule.

Re-derive with `python scripts/seq/build_draft_manifests.py` (CPU, no GPU), and
check an identity with:

    python scripts/seq/validate_stage.py manifest dev \
        --manifest results/audit_v1/draft_manifests/dev_manifest_DRAFT.json \
        --expect_manifest_sha <the hash above> \
        --expect_records 80 --expect_prompt_texts 20
