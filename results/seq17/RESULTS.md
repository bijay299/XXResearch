# SD-1.5 exploratory sequential object-erasure pilot — results

**This is not a reproduction of any published result, and these are not UA/IRA/CRA.** The backbone is base Stable Diffusion v1.5 and the evaluator is a public COCO detector; every rate below is a detector-based object-presence proxy, not ground-truth erasure and not an image-quality score. The blocked UnlearnCanvas baseline and its validation gate are untouched and remain blocked.

## Research question

Can an earlier concept deletion persist through a later deletion while preserving both new-request effectiveness and retained utility? L2-SP is an existing baseline regulariser, not a proposed new method. The question is tested, not assumed.

## Configuration

- Upstream CUIG `9932ac3271a122f6d38d19e0b8c8908fe5237ff7`, native ConAbl object training, mappings `horse+cat`, `horse+dog`, `flower+sandwich`.
- Backbone M0: base Stable Diffusion v1.5.
- 1000 optimizer steps per request (upstream sequential uses 2000; reduced exploratory budget), 50 steps/epoch, epoch cap 20 → capacity 1000 steps.
- Effective LR 8e-06 (CLI default 2e-06 × grad_accum 1 × batch 4 × procs 1), optimizer AdamW (torch.optim.AdamW; use_8bit_adam=False), precision fp32 (accelerate mixed_precision: 'no'), parameter group `kv-xattn`.
- 200 anchor images/prompts, generated once from untouched M0 (seed 17) and shared byte-identically across matched arms.
- Training seed 17; regularised and unregularised arms differ only in `--l2sp_weight 25000` (a documented example coefficient, not an SD-1.5-tuned optimum; no coefficient search was run).

## Pre-registration-style checks that passed before any editing

- **M0 generates the targets**: cat 100%, dog 100%, sandwich 100% at t=0.5 on a separate calibration set.
- **Detector does not over-trigger**: negative controls fire 0% of the time.
- **No-update reload is inert**: 8/8 regenerated images byte-identical after loading a no-op delta through the real checkpoint path.
- **Evaluation manifest frozen before editing**: 7 categories × 10 prompts (5 literal + 5 paraphrase) × 4 seeds = 280 images per checkpoint; zero overlap with training prompts or their target-substituted forms.

## Headline findings

1. **The first deletion worked.** Cat fell 100% → 18% (+82.5 pp [+67.5, +95.0]), so historical recovery is a well-posed question here rather than a measurement of noise.

2. **Historical interference is real but direction depends on the second request.** Cat detection rose credibly after MAC_L2 and fell further after MAB, MAB_L2. Deleting dog — which shares the `horse` anchor with cat and is semantically close — pushed cat *further down*; deleting sandwich, with a different (`flower`) anchor, let cat partially return. A single 'does deletion survive?' answer would misdescribe this.

3. **L2-SP did not protect the earlier deletion; it bought retention and paid in new-request effectiveness.** New-target suppression fell in both branches (MAB +82.5 → MAB_L2 +42.5 pp; MAC +50.0 → MAC_L2 +7.5 pp), and cat recovery did not decrease — in the sandwich branch it *increased* (+7.5 → +15.0 pp). What L2-SP clearly did protect is collateral damage to controls: bird fell -45.0 pp under MAB but only -5.0 pp under MAB_L2. This is a **tradeoff**, not an improved method.

4. **Recovery is larger under paraphrased prompts.** For MAC_L2 the paraphrase family recovered 20.0 pp more than the literal family. Measuring erasure only with prompts that name the concept would understate what the edited model still produces.


## Measured results

### Raw detection rate D (%) at t=0.5, all prompts

| checkpoint | cat | dog | sandwich | horse | bird | chair | bicycle |
|---|---|---|---|---|---|---|---|
| M0 | 100 (n=40) | 100 (n=40) | 95 (n=40) | 92 (n=40) | 92 (n=40) | 90 (n=40) | 90 (n=40) |
| MA | 18 (n=40) | 92 (n=40) | 88 (n=40) | 100 (n=40) | 90 (n=40) | 90 (n=40) | 95 (n=40) |
| MAB | 0 (n=40) | 10 (n=40) | 82 (n=40) | 98 (n=40) | 45 (n=40) | 82 (n=40) | 90 (n=40) |
| MAB_L2 | 5 (n=40) | 50 (n=40) | 88 (n=40) | 100 (n=40) | 85 (n=40) | 88 (n=40) | 92 (n=40) |
| MAC | 25 (n=40) | 95 (n=40) | 38 (n=40) | 100 (n=40) | 98 (n=40) | 80 (n=40) | 95 (n=40) |
| MAC_L2 | 32 (n=40) | 95 (n=40) | 80 (n=40) | 100 (n=40) | 95 (n=40) | 85 (n=40) | 95 (n=40) |

### Experiment 1 — historical interference

**Initial cat suppression** D_cat(M0) − D_cat(MA) = **+82.5 pp [+67.5, +95.0]** (M0 100% → MA 18%).


**Historical cat recovery** D_cat(child) − D_cat(MA):

| child | new deletion | cat recovery (pp, 95% CI) | D_cat(MA) | D_cat(child) |
|---|---|---|---|---|
| MAB | dog | -17.5 pp [-32.5, -5.0] | 18% | 0% |
| MAB_L2 | dog | -12.5 pp [-22.5, -2.5] | 18% | 5% |
| MAC | sandwich | +7.5 pp [+0.0, +15.0] | 18% | 25% |
| MAC_L2 | sandwich | +15.0 pp [+2.5, +27.5] | 18% | 32% |

**New-request suppression** D_new(MA) − D_new(child), with D_new(M0) shown because the cat deletion may already move the later target:

| child | target | D_M0 | D_MA | D_child | suppression (pp, 95% CI) |
|---|---|---|---|---|---|
| MAB | dog | 100% | 92% | 10% | +82.5 pp [+65.0, +97.5] |
| MAB_L2 | dog | 100% | 92% | 50% | +42.5 pp [+25.0, +62.5] |
| MAC | sandwich | 95% | 88% | 38% | +50.0 pp [+35.0, +62.5] |
| MAC_L2 | sandwich | 95% | 88% | 80% | +7.5 pp [+0.0, +17.5] |

**Retention change vs MA** on the four common controls (never a deletion target in any branch):

| child | horse (anchor) | bird | chair | bicycle |
|---|---|---|---|---|
| MAB | -2.5 pp [-7.5, +0.0] | -45.0 pp [-72.5, -17.5] | -7.5 pp [-20.0, +5.0] | -5.0 pp [-12.5, +0.0] |
| MAB_L2 | +0.0 pp [+0.0, +0.0] | -5.0 pp [-15.0, +0.0] | -2.5 pp [-7.5, +0.0] | -2.5 pp [-7.5, +0.0] |
| MAC | +0.0 pp [+0.0, +0.0] | +7.5 pp [-2.5, +20.0] | -10.0 pp [-17.5, -2.5] | +0.0 pp [+0.0, +0.0] |
| MAC_L2 | +0.0 pp [+0.0, +0.0] | +5.0 pp [-5.0, +15.0] | -5.0 pp [-17.5, +5.0] | +0.0 pp [+0.0, +0.0] |

**Branch-specific undeleted target** (reported separately, never folded into a common retain average):

| child | undeleted target | vs M0 | vs MA |
|---|---|---|---|
| MAB | sandwich | -12.5 pp [-27.5, +0.0] | -5.0 pp [-12.5, +0.0] |
| MAB_L2 | sandwich | -7.5 pp [-20.0, +5.0] | +0.0 pp [-7.5, +7.5] |
| MAC | dog | -5.0 pp [-12.5, +0.0] | +2.5 pp [-5.0, +10.0] |
| MAC_L2 | dog | -5.0 pp [-12.5, +0.0] | +2.5 pp [-5.0, +10.0] |

### Experiment 2 — preservation versus new deletion

| contrast | cat recovery (pp) | new-target suppression (pp) |
|---|---|---|
| MAB (no reg.) | -17.5 pp [-32.5, -5.0] | +82.5 pp [+65.0, +97.5] |
| MAB_L2 (L2-SP 25000) | -12.5 pp [-22.5, -2.5] | +42.5 pp [+25.0, +62.5] |
| MAC (no reg.) | +7.5 pp [+0.0, +15.0] | +50.0 pp [+35.0, +62.5] |
| MAC_L2 (L2-SP 25000) | +15.0 pp [+2.5, +27.5] | +7.5 pp [+0.0, +17.5] |

### Literal vs paraphrased prompts

| contrast | literal | paraphrase |
|---|---|---|
| initial cat suppression | +85.0 pp [+65.0, +100.0] | +80.0 pp [+60.0, +100.0] |
| cat recovery MAB | -15.0 pp [-35.0, +0.0] | -20.0 pp [-40.0, +0.0] |
| new-target suppression MAB | +65.0 pp [+40.0, +85.0] | +100.0 pp [+100.0, +100.0] |
| cat recovery MAB_L2 | -10.0 pp [-20.0, +0.0] | -15.0 pp [-35.0, +0.0] |
| new-target suppression MAB_L2 | +25.0 pp [+5.0, +50.0] | +60.0 pp [+40.0, +80.0] |
| cat recovery MAC | +5.0 pp [+0.0, +15.0] | +10.0 pp [+0.0, +20.0] |
| new-target suppression MAC | +60.0 pp [+40.0, +75.0] | +40.0 pp [+20.0, +50.0] |
| cat recovery MAC_L2 | +5.0 pp [+0.0, +15.0] | +25.0 pp [+5.0, +45.0] |
| new-target suppression MAC_L2 | +5.0 pp [+0.0, +15.0] | +10.0 pp [+0.0, +30.0] |

### Parameter movement (diagnostic association only)

| checkpoint | relative L2 vs M0 | relative L2 vs MA |
|---|---|---|
| MA | 0.01184 | 0.00000 |
| MAB | 0.02078 | 0.01358 |
| MAB_L2 | 0.01225 | 0.00154 |
| MAC | 0.01673 | 0.01175 |
| MAC_L2 | 0.01192 | 0.00147 |

This is a description of how far the edited tensors moved. It is **not** evidence that weight movement causes recovery.

## Runtime and peak memory

| checkpoint | GPU | train s | s/step | peak GPU mem (MiB) |
|---|---|---|---|---|
| MA | GPU-3594d682-af4b-13… | 828.6 | 0.829 | 10442 |
| MAB | GPU-3594d682-af4b-13… | 828.5 | 0.828 | 10442 |
| MAB_L2 | GPU-17211a38-e5ad-e9… | 831.3 | 0.831 | 10516 |
| MAC | GPU-3594d682-af4b-13… | 828.3 | 0.828 | 10443 |
| MAC_L2 | GPU-17211a38-e5ad-e9… | 831.7 | 0.832 | 10516 |

## Limitations

- One training seed (17). The bootstrap intervals are **sampling** uncertainty over prompts; generation seeds are not independent training repeats, and no population variance over training runs is claimed.
- 1000 optimizer steps is a reduced exploratory budget; upstream's sequential object script uses 2000.
- Detector-based proxies only. A detection is not ground truth, and absence of a detection is not proof of erasure.
- Dog vs sandwich is one selected request contrast with different anchor mappings; it does not isolate semantic similarity.
- L2-SP coefficient 25000 is a documented example, not tuned for SD-1.5. No coefficient search was performed against final evaluation results.
- Two regularisation settings cannot define a Pareto frontier.

## Visual inspection status

Representative calibration images were inspected **by the AI assistant**, not by a human annotator. Blinded, checkpoint-shuffled grids and an empty annotation sheet are prepared under `grids/blinded/` for human scoring. **No human annotation has been performed.**

## What this motivates next

- **Separate anchor sharing from semantic similarity.** The two branches differ both in anchor (`horse` vs `flower`) and in semantic distance from cat, and they moved cat in opposite directions. The decisive next run is a second request that is semantically far from cat but *shares* the `horse` anchor, and one that is semantically close but uses a different anchor. Only that factorisation can tell which caused the sign flip.
- **Does recovery accumulate?** Extend the stream past two requests and test whether cat keeps returning in the sandwich-like branches.
- **Does the L2-SP tradeoff have a usable setting?** At 25000 it cost most of the sandwich deletion (+50.0 → +7.5 pp). A coefficient sweep — chosen on a held-out split, not on these final numbers — would show whether any value keeps retention without surrendering the new request.

- A second training seed is required before any numeric magnitude is treated as stable.

