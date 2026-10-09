# FROZEN DRAFT MANIFESTS — AUDIT-01b

**Status: DRAFT, for review. Nothing has been generated from either manifest and
no GPU work is authorised.** These exist so the prompt sets and the analysis
rules are fixed *before* any approved launch, not chosen afterwards.

## Hashes

| manifest | distinct prompt texts | gen seeds | pairs/checkpoint | sha256 |
|---|---|---|---|---|
| `dev_manifest_DRAFT.json` | 20 | [511, 622, 733, 844] | 80 | `ebfdd37270dea75012756fdc877fb614a9bd5ae0dd78af76ed426844136e5b89` |
| `test_manifest_DRAFT.json` | 140 | [1301, 1402, 1503, 1604] | 560 | `170696eb5fc2832b832e7e2baca6c820344ac9636c66e510398c3a19e6ef8ae1` |

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
`disjointness_report.json` for the full analysis. Where prompt texts share a
template the bootstrap's prompt clusters are not fully independent and intervals
are mildly optimistic. That limitation applies to the pilot set too and is
stated rather than assumed away.

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

Re-derive with `python scripts/seq/build_draft_manifests.py` (CPU, no GPU).
