# RWKU local screen — 860 probes (development evidence)

Run 2026-10-20 on GPUs 2-3, `scripts/screen_rwku_full.py`, raw generations in
`outputs/rwku_screen_shard{0,1}.json`. Pinned model `08b83a6f…963b07`, author's sampler
unmodified, decoder gen 64 / steps 64 / block 32 / temp 0 / low_confidence — identical to
the prior closed study's primary decoder so the numbers are comparable.

**This does not revise the closed study** `research/llada-rwku-baseline-v1 @ aecefcd662c5`.
That study screened 100 probes against its own pre-registered exact-match gate and returned
`INSUFFICIENT_BASE_RECALL`; that verdict stands as published. What follows is a wider
measurement (every local forget *and* neighbor probe — 860 vs 100) scored against the
metric RWKU itself defines.

## Results

| probe set | n | EM | **ROUGE-L recall** | **span containment** | answer in prompt | degenerate |
|---|---|---|---|---|---|---|
| forget_level1 (cloze) | 166 | 0.012 | 0.493 | 0.452 | 0.024 | 0 |
| forget_level2 (QA) | 131 | 0.008 | 0.448 | 0.321 | 0.015 | 3 |
| neighbor_level1 | 288 | 0.000 | 0.545 | 0.535 | 0.059 | 3 |
| neighbor_level2 | 275 | 0.000 | 0.359 | 0.287 | 0.004 | 1 |

## Exact match was measuring formatting, not knowledge

EM is ~0.01 while span containment is 0.32-0.45 on the same probes, because the model
answers in sentences and the gold answers are phrases:

```
gold "Meghan Markle"            gen "Prince Harry married Meghan Markle on May 19, 2018."
gold "Omaha, Nebraska"          gen "Warren Buffett was born in Omaha, Nebraska."
gold "Royal Military Academy…"  gen "Prince Harry attended the Royal Military Academy Sand…"
```

EM scores all three as 0. The evaluation lead established independently that **RWKU's
official primary metric is ROUGE-L recall** (paper §3.3.1 and `eval_forget.py` returning
`['rouge-l']['r']`), not EM. So an EM-anchored recall gate is mis-specified relative to the
benchmark, and yield estimates derived from EM survivorship understate what is available.

## Clean candidate forget targets

Filter: answer ≥ 2 tokens **and** answer not present in the prompt **and** span-recalled.

**40 of 297 forget probes (13.5%), spread over 9 of the 10 entities** —
`outputs/rwku_clean_candidates.json`. Per entity: Justin Bieber 12, Warren Buffett 9,
Prince Harry 8, Stephen King 2, Bruce Lee 2, Christina Aguilera 2, Cindy Crawford 2,
Paris Hilton 2, Confucius 1, Marie Osmond 0.

That is ~4 clean facts per entity rather than the ~0.5 projected from EM-based criteria.

**Not yet applied, and required before this count is used:** the cross-decoder stability
filter (the evaluation lead measured conditional stability of 0.727 for EM survivors under
a second decoder; it must be recomputed for span-recalled multi-token targets), and a
hand-check for the spurious-credit mechanism the same lead documented, where ROUGE-L recall
awards 0.50 to "New York University" against gold "Northwestern University". Span
containment is stricter than ROUGE-L but is not immune — `Duke of Sussex` appearing inside
a generation about the Sussex title is a candidate false positive. **The 40 is an upper
bound pending both filters.**

## Two things the prior study had no data on

- **Neighbor probes are strongly recalled** (level-1 span 0.535, ROUGE-L recall 0.545, n=288).
  These are RWKU's own adjacent-knowledge set, i.e. ready-made preservation measurements on
  real pretrained knowledge — the thing synthetic facts cannot supply.
- **The base model is coherent**: 7 degenerate outputs in 860 probes (0.8%). So any
  degeneracy observed after unlearning is attributable to the unlearning, not to the model
  or the decoder.

## Cost

860 probes, 2 shards × 430, 908 s and 900 s wall, 15.04 GiB peak each, ~2.1 s/probe.
Recorded in `ATTEMPTS.tsv` as p001/p002.
