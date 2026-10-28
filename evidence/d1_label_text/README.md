# D1 resolved — the inconsistent label text moves the reported metrics by ~0.01

**2026-10-28.** Defect D1 has been open since FINDING65, whose banner states: *"The evaluator used
inconsistent label text. The effect on scores and conclusions is **unresolved** pending corrected
evaluation."* It is the first of the four defects blocking a submission. This closes it with a
measurement rather than an argument.

## The defect, precisely

`gen_eval.py` defines two parallel sequences: `LABELS`, the text intended for CLIP, and `LAB_KEY`,
the underscored record keys. **Nine of twelve differ:**

| CLIP text (`LABELS`) | record key (`LAB_KEY`) |
|---|---|
| garbage truck | garbage_truck |
| van gogh painting | van_gogh |
| cassette player | cassette_player |
| chain saw | chain_saw |
| english springer spaniel | english_springer |
| french horn | french_horn |
| gas pump | gas_pump |
| golf ball | golf_ball |
| monet painting | monet |

`gen_eval_final.py` passed the **key** strings to CLIP, so some published numbers were scored
against different label text than others.

**`church` is identical in both.** That bounds the problem but does not remove it: suppression and
recovery are a 12-way **argmax**, so an image can cross between `church` and a competitor purely
because the competitor's text changed.

## Measurement

The same **saved pixels** from existing evidence directories, re-scored under both label sets with
everything else held fixed. Nothing regenerated; no model loaded.

| split | images | argmax agreement | acc with `LABELS` | acc with `LAB_KEY` | **difference** |
|---|---:|---:|---:|---:|---:|
| **H** (the recovery metric, church) | 900 | 0.9022 | 0.6633 | 0.6533 | **0.0100** |
| **util** (the retained-capability metric) | 1,152 | 0.9627 | 0.8455 | 0.8576 | **0.0121** |

## Reading

**Individual labels do flip** — 9.8% of target-panel images and 3.7% of utility images land on a
different label. So the defect is real and was worth tracking.

**The reported aggregates barely move.** The target metric differs by **0.0100**, against a measured
noise floor of one quantum, 1/15 = **0.067**. The utility metric differs by **0.0121**, against a
measured floor of **0.042**. Both differences are roughly **a fifth to a quarter of the noise the
metric already carries**, i.e. below the threshold at which this project treats a difference as a
difference.

**Conclusion: D1 does not change any conclusion in this project.** It should still be fixed in the
code and disclosed in the paper, but it is not a blocking defect and the claims built on these
metrics stand.

## What this does not cover

- Measured on U33 ladder evidence (church, SD-1.4). Other concepts are not covered, though the
  mechanism — competing-label text — is concept-independent.
- Closes **D1 only**. D2 (the validity gate's exclusion rate is non-monotone in pre-attack
  suppression, n=520), D3 (the UCE port vs upstream discrepancy) and D4 (pervasive n=1) remain open.
- The flips themselves are recorded in `outputs/d1_label_text_impact.json` for anyone who wants to
  check which images moved and where.
