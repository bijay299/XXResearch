# AUDIT-01 — corrected counts and direct contrast tables

Re-derived from the raw per-image detector predictions (`detections.jsonl`) and the frozen evaluation manifest. All rates are detector-based object-presence proxies: not certified erasure, not image quality, not a causal mechanism.

## 1. Corrected counts

Frozen evaluation manifest: 280 records, 280 unique (category, prompt_id, gen_seed) identities, `frozen_before_any_editing = True`, manifest SHA-256 `0020c81c4a4dd358…`. Disjointness from training strings: 0 exact collisions, 0 near-duplicates over 999 training strings checked.

| quantity | value | basis |
|---|---|---|
| checkpoint-evaluation table slots | 12 | 2 seeds × 6 checkpoints |
| **distinct** checkpoint evaluations | **11** | M0 counted once; `eval_seed29/M0` is a symlink to `eval/M0` |
| table rows across both seeds | 3360 | 12 × 280 |
| **newly generated** evaluation images | **3080** | the 280 M0 images are reused, not regenerated |
| training runs with a saved final delta | 10 | 2 seeds × 5 requests |

> **Correction.** A total of “3360 evaluation images” counts the reused M0 set twice. The number of **generated** images is **3080**; 3360 is the row count of the two stacked tables. The M0 rows are one evaluation set appearing under both seed labels, so they carry no cross-seed replication information.

### Completeness and identity audit

| seed / checkpoint | rows | unique identities | dup. identities | unique image SHA-256 | missing vs manifest | extra vs manifest |
|---|---|---|---|---|---|---|
| seed17 / M0 | 280 | 280 | 0 | 280 | 0 | 0 |
| seed17 / MA | 280 | 280 | 0 | 280 | 0 | 0 |
| seed17 / MAB | 280 | 280 | 0 | 280 | 0 | 0 |
| seed17 / MAB_L2 | 280 | 280 | 0 | 280 | 0 | 0 |
| seed17 / MAC | 280 | 280 | 0 | 280 | 0 | 0 |
| seed17 / MAC_L2 | 280 | 280 | 0 | 280 | 0 | 0 |
| seed29 / M0 | 280 | 280 | 0 | 280 | 0 | 0 |
| seed29 / MA | 280 | 280 | 0 | 280 | 0 | 0 |
| seed29 / MAB | 280 | 280 | 0 | 280 | 0 | 0 |
| seed29 / MAB_L2 | 280 | 280 | 0 | 280 | 0 | 0 |
| seed29 / MAC | 280 | 280 | 0 | 280 | 0 | 0 |
| seed29 / MAC_L2 | 280 | 280 | 0 | 280 | 0 | 0 |

Integrity issues raised: **0**.
Every checkpoint has exactly 280 rows, 280 distinct (category, prompt_id, gen_seed) identities and 280 distinct image hashes, with 7 categories × 10 prompts × 4 generation seeds and a 140/140 literal/paraphrase split. **No duplicate rows and no missing rows were found, so no de-duplication or dropping rule was applied to anything.**

Images byte-identical across two different table slots: 280, of which **0** are anything other than the known M0 reuse. Seed 17 and seed 29 therefore produced genuinely distinct images at every edited checkpoint — the two runs are independent everywhere except the shared M0 set and the shared anchor caches.

### Cross-check of the committed summaries against raw predictions

| seed | rate cells checked | contrast values checked | mismatches |
|---|---|---|---|
| seed17 | 126 | 387 | 0 |
| seed29 | 126 | 387 | 0 |

Every rate and every headline point estimate in the committed `rates.csv` and `contrasts.json` reproduces exactly from the raw predictions. **The arithmetic of the snapshot is correct; the problems found by this audit are in the narrative framing, not in the data.**

## 2. Direct paired L2-SP versus unregularised contrasts

Resampling unit: prompt (10 per category), carrying its 4 generation seeds. one resampled prompt list indexes both arms of every contrast. 10000 draws, 95% percentile interval. Training seeds analysed separately; never pooled, never resampled jointly.

- **Interval meaning.** sampling uncertainty over the 10 evaluation prompts within ONE training run. With 10 clusters of 4 images the attainable resolution is coarse: a 95% interval cannot be narrower than a few percentage points, and an interval that includes zero is NOT evidence that the effect is zero.

- **Multiplicity.** no multiple-comparison correction is applied; many contrasts are reported, so individual interval-excludes-zero labels must be read as descriptive, not as tests.

- **Why these are the right contrasts.** L2 contrasts are child-vs-child, so the MA parent term cancels and these estimates do not inherit any D_cat(MA) instability.

`ns` marks an interval that includes zero. A `ns` label is **not** a finding of no effect, and a change of `ns` label between two seeds is **not** a demonstration that the two differ.

### MAB_L2 − MAB (second request = dog)

Positive = the L2-SP arm left **more** of that category standing.

| contrast | thr | seed 17 | seed 29 |
|---|---|---|---|
| new target (dog) residual | 0.3 | +40.0 [+22.5, +57.5] | +42.5 [+22.5, +62.5] |
| new target (dog) residual | 0.5 | +40.0 [+22.5, +57.5] | +40.0 [+17.5, +62.5] |
| new target (dog) residual | 0.7 | +40.0 [+22.5, +57.5] | +45.0 [+25.0, +65.0] |
| historical cat residual | 0.3 | +5.0 [-5.0, +15.0] ns | +7.5 [+0.0, +15.0] ns |
| historical cat residual | 0.5 | +5.0 [+0.0, +12.5] ns | +5.0 [+0.0, +12.5] ns |
| historical cat residual | 0.7 | +2.5 [+0.0, +7.5] ns | +2.5 [+0.0, +7.5] ns |
| retained: horse | 0.5 | +2.5 [+0.0, +7.5] ns | +2.5 [+0.0, +7.5] ns |
| retained: bird | 0.5 | +40.0 [+15.0, +65.0] | +35.0 [+7.5, +62.5] |
| retained: chair | 0.5 | +5.0 [-5.0, +15.0] ns | +2.5 [-7.5, +15.0] ns |
| retained: bicycle | 0.5 | +2.5 [+0.0, +7.5] ns | +0.0 [+0.0, +0.0] ns |
| sibling target: sandwich (reported separately) | 0.5 | +5.0 [+0.0, +12.5] ns | +10.0 [+0.0, +20.0] ns |

Literal versus paraphrase, t=0.5 (point estimates of two separate contrasts; no interaction interval and no interaction test):

| quantity | seed 17 literal | seed 17 paraphrase | seed 29 literal | seed 29 paraphrase |
|---|---|---|---|---|
| new target (dog) residual, L2 − no-L2 | +40.0 | +40.0 | +45.0 | +35.0 |
| historical cat residual, L2 − no-L2 | +5.0 | +5.0 | +5.0 | +5.0 |

### MAC_L2 − MAC (second request = sandwich)

Positive = the L2-SP arm left **more** of that category standing.

| contrast | thr | seed 17 | seed 29 |
|---|---|---|---|
| new target (sandwich) residual | 0.3 | +47.5 [+35.0, +60.0] | +47.5 [+25.0, +67.5] |
| new target (sandwich) residual | 0.5 | +42.5 [+27.5, +57.5] | +45.0 [+17.5, +65.0] |
| new target (sandwich) residual | 0.7 | +40.0 [+25.0, +52.5] | +42.5 [+20.0, +62.5] |
| historical cat residual | 0.3 | +10.0 [+0.0, +20.0] ns | +0.0 [-10.0, +10.0] ns |
| historical cat residual | 0.5 | +7.5 [+0.0, +17.5] ns | +2.5 [-7.5, +15.0] ns |
| historical cat residual | 0.7 | +7.5 [+0.0, +22.5] ns | -2.5 [-7.5, +0.0] ns |
| retained: horse | 0.5 | +0.0 [+0.0, +0.0] ns | +0.0 [+0.0, +0.0] ns |
| retained: bird | 0.5 | -2.5 [-7.5, +0.0] ns | -2.5 [-15.0, +10.0] ns |
| retained: chair | 0.5 | +5.0 [-10.0, +20.0] ns | +0.0 [-10.0, +12.5] ns |
| retained: bicycle | 0.5 | +0.0 [+0.0, +0.0] ns | +0.0 [+0.0, +0.0] ns |
| sibling target: dog (reported separately) | 0.5 | +0.0 [+0.0, +0.0] ns | -2.5 [-7.5, +0.0] ns |

Literal versus paraphrase, t=0.5 (point estimates of two separate contrasts; no interaction interval and no interaction test):

| quantity | seed 17 literal | seed 17 paraphrase | seed 29 literal | seed 29 paraphrase |
|---|---|---|---|---|
| new target (sandwich) residual, L2 − no-L2 | +55.0 | +30.0 | +65.0 | +25.0 |
| historical cat residual, L2 − no-L2 | +0.0 | +15.0 | +0.0 | +5.0 |

## 3. The two estimands side by side

`historical cat recovery` is referenced to the parent, `D_cat(child) − D_cat(MA)` of the **same training seed**. The direct contrast `D_cat(L2) − D_cat(no-L2)` measures the L2 effect with the parent term algebraically cancelled. Where the two disagree across seeds, the disagreement is located in the reference term.

| child | thr | recovery vs own MA, seed 17 | recovery vs own MA, seed 29 |
|---|---|---|---|
| MAB | 0.3 | -17.5 [-37.5, +0.0] ns | -27.5 [-47.5, -10.0] ← label differs |
| MAB | 0.5 | -17.5 [-32.5, -5.0] | -27.5 [-47.5, -10.0] |
| MAB | 0.7 | -17.5 [-32.5, -5.0] | -25.0 [-42.5, -10.0] |
| MAB_L2 | 0.3 | -12.5 [-22.5, -2.5] | -20.0 [-35.0, -7.5] |
| MAB_L2 | 0.5 | -12.5 [-22.5, -2.5] | -22.5 [-42.5, -7.5] |
| MAB_L2 | 0.7 | -15.0 [-27.5, -2.5] | -22.5 [-40.0, -7.5] |
| MAC | 0.3 | +5.0 [+0.0, +12.5] ns | +0.0 [-10.0, +10.0] ns |
| MAC | 0.5 | +7.5 [+0.0, +15.0] ns | -7.5 [-25.0, +5.0] ns |
| MAC | 0.7 | +5.0 [-5.0, +15.0] ns | -5.0 [-17.5, +5.0] ns |
| MAC_L2 | 0.3 | +15.0 [+5.0, +27.5] | +0.0 [-7.5, +7.5] ns ← label differs |
| MAC_L2 | 0.5 | +15.0 [+2.5, +27.5] | -5.0 [-12.5, +0.0] ns ← label differs |
| MAC_L2 | 0.7 | +12.5 [+2.5, +22.5] | -7.5 [-17.5, +0.0] ns ← label differs |

### The reference term itself

`D_cat(MA)` (%), the quantity that cancels in every direct contrast:

| thr | family | seed 17 | seed 29 | gap |
|---|---|---|---|---|
| 0.3 | all | 20.0 | 27.5 | +7.5 pp |
| 0.3 | literal | 15.0 | 15.0 | +0.0 pp |
| 0.3 | paraphrase | 25.0 | 40.0 | +15.0 pp |
| 0.5 | all | 17.5 | 27.5 | +10.0 pp |
| 0.5 | literal | 15.0 | 15.0 | +0.0 pp |
| 0.5 | paraphrase | 20.0 | 40.0 | +20.0 pp |
| 0.7 | all | 17.5 | 25.0 | +7.5 pp |
| 0.7 | literal | 15.0 | 15.0 | +0.0 pp |
| 0.7 | paraphrase | 20.0 | 35.0 | +15.0 pp |

The parent difference between the two training runs is **entirely on paraphrase prompts**: literal `D_cat(MA)` is identical in both seeds at all three thresholds, while the paraphrase value differs by 15–20 pp. Any estimand referenced to `D_cat(MA)` inherits that difference; the direct child-vs-child contrasts do not.

## 4. Deletion effectiveness context

A retention comparison is only interpretable next to how much deletion each arm achieved. Suppression is `D_new(MA) − D_new(child)` against the same seed's parent; residual is the child's own rate.

| seed | child | target | thr | suppression (pp) | residual (%) |
|---|---|---|---|---|---|
| seed17 | MAB | dog | 0.3 | +82.5 | 15.0 |
| seed17 | MAB_L2 | dog | 0.3 | +42.5 | 55.0 |
| seed17 | MAC | sandwich | 0.3 | +50.0 | 42.5 |
| seed17 | MAC_L2 | sandwich | 0.3 | +2.5 | 90.0 |
| seed17 | MAB | dog | 0.5 | +82.5 | 10.0 |
| seed17 | MAB_L2 | dog | 0.5 | +42.5 | 50.0 |
| seed17 | MAC | sandwich | 0.5 | +50.0 | 37.5 |
| seed17 | MAC_L2 | sandwich | 0.5 | +7.5 | 80.0 |
| seed17 | MAB | dog | 0.7 | +85.0 | 7.5 |
| seed17 | MAB_L2 | dog | 0.7 | +45.0 | 47.5 |
| seed17 | MAC | sandwich | 0.7 | +42.5 | 35.0 |
| seed17 | MAC_L2 | sandwich | 0.7 | +2.5 | 75.0 |
| seed29 | MAB | dog | 0.3 | +80.0 | 15.0 |
| seed29 | MAB_L2 | dog | 0.3 | +37.5 | 57.5 |
| seed29 | MAC | sandwich | 0.3 | +50.0 | 42.5 |
| seed29 | MAC_L2 | sandwich | 0.3 | +2.5 | 90.0 |
| seed29 | MAB | dog | 0.5 | +82.5 | 10.0 |
| seed29 | MAB_L2 | dog | 0.5 | +42.5 | 50.0 |
| seed29 | MAC | sandwich | 0.5 | +52.5 | 35.0 |
| seed29 | MAC_L2 | sandwich | 0.5 | +7.5 | 80.0 |
| seed29 | MAB | dog | 0.7 | +90.0 | 0.0 |
| seed29 | MAB_L2 | dog | 0.7 | +45.0 | 45.0 |
| seed29 | MAC | sandwich | 0.7 | +55.0 | 30.0 |
| seed29 | MAC_L2 | sandwich | 0.7 | +12.5 | 72.5 |

## 5. Recomputed detection rates, t=0.5, all prompts

| seed | checkpoint | cat | dog | sandwich | horse | bird | chair | bicycle |
|---|---|---|---|---|---|---|---|---|
| seed17 | M0 | 100.0 (40/40) | 100.0 (40/40) | 95.0 (38/40) | 92.5 (37/40) | 92.5 (37/40) | 90.0 (36/40) | 90.0 (36/40) |
| seed17 | MA | 17.5 (7/40) | 92.5 (37/40) | 87.5 (35/40) | 100.0 (40/40) | 90.0 (36/40) | 90.0 (36/40) | 95.0 (38/40) |
| seed17 | MAB | 0.0 (0/40) | 10.0 (4/40) | 82.5 (33/40) | 97.5 (39/40) | 45.0 (18/40) | 82.5 (33/40) | 90.0 (36/40) |
| seed17 | MAB_L2 | 5.0 (2/40) | 50.0 (20/40) | 87.5 (35/40) | 100.0 (40/40) | 85.0 (34/40) | 87.5 (35/40) | 92.5 (37/40) |
| seed17 | MAC | 25.0 (10/40) | 95.0 (38/40) | 37.5 (15/40) | 100.0 (40/40) | 97.5 (39/40) | 80.0 (32/40) | 95.0 (38/40) |
| seed17 | MAC_L2 | 32.5 (13/40) | 95.0 (38/40) | 80.0 (32/40) | 100.0 (40/40) | 95.0 (38/40) | 85.0 (34/40) | 95.0 (38/40) |
| seed29 | M0 | 100.0 (40/40) | 100.0 (40/40) | 95.0 (38/40) | 92.5 (37/40) | 92.5 (37/40) | 90.0 (36/40) | 90.0 (36/40) |
| seed29 | MA | 27.5 (11/40) | 92.5 (37/40) | 87.5 (35/40) | 100.0 (40/40) | 95.0 (38/40) | 85.0 (34/40) | 92.5 (37/40) |
| seed29 | MAB | 0.0 (0/40) | 10.0 (4/40) | 75.0 (30/40) | 97.5 (39/40) | 55.0 (22/40) | 85.0 (34/40) | 92.5 (37/40) |
| seed29 | MAB_L2 | 5.0 (2/40) | 50.0 (20/40) | 85.0 (34/40) | 100.0 (40/40) | 90.0 (36/40) | 87.5 (35/40) | 92.5 (37/40) |
| seed29 | MAC | 20.0 (8/40) | 97.5 (39/40) | 35.0 (14/40) | 100.0 (40/40) | 95.0 (38/40) | 87.5 (35/40) | 92.5 (37/40) |
| seed29 | MAC_L2 | 22.5 (9/40) | 95.0 (38/40) | 80.0 (32/40) | 100.0 (40/40) | 92.5 (37/40) | 87.5 (35/40) | 92.5 (37/40) |

Full grid at all three thresholds and all three prompt families: `recomputed_rates.csv` (756 rows).

