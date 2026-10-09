# SD-1.5 exploratory sequential object-erasure pilot — combined results (seeds 17 and 29)

**Not UA/IRA/CRA. Not a reproduction of any published result.** The backbone is
base Stable Diffusion v1.5; the evaluator is the official Torchvision COCO
Faster R-CNN. Every rate is a detector-based object-presence proxy — not
ground-truth erasure, not an image-quality score. The blocked UnlearnCanvas
baseline and its validation gate are untouched.

Two complete cores were run: **seed 17** and **seed 29**, five training runs
each, six evaluated checkpoints each, 280 images per checkpoint.
**3360 evaluation images in total, zero generation or detection failures.**

---

## The question

> Can an earlier concept deletion persist through a later deletion while
> preserving both new-request effectiveness and retained utility?

A = cat, then B = dog or C = sandwich, from a shared parent MA. L2-SP (25000) is
an **existing baseline** added at the second request only.

## The short answer

**No evidence that an earlier deletion reverses.** At this budget the first
deletion holds, and in the semantically-near branch it is *reinforced* rather
than undone. The one apparently credible recovery in seed 17 **did not
replicate** in seed 29 — see §3.

**L2-SP is a tradeoff, not an improvement.** It reliably protects collateral
damage and reliably costs new-request effectiveness. It did not protect the
earlier deletion.

---

## 1. What replicates across both training seeds

| Finding | seed 17 | seed 29 | status |
|---|---|---|---|
| **First deletion works.** cat D(M0)−D(MA) | +82.5 pp [+67.5,+95.0] | +72.5 pp [+50.0,+90.0] | **robust** |
| **Dog branch pushes cat further down**, not up. MAB | −17.5 pp [−32.5,−5.0] | −27.5 pp [−50.0,−10.0] | **robust** |
| same, with L2-SP. MAB_L2 | −12.5 pp [−22.5,−2.5] | −22.5 pp [−42.5,−7.5] | **robust** |
| **New-request suppression, dog, no reg.** | +82.5 pp | +82.5 pp | **robust** |
| **New-request suppression, dog, L2-SP** | +42.5 pp | +42.5 pp | **robust** |
| **New-request suppression, sandwich, no reg.** | +50.0 pp | +52.5 pp | **robust** |
| **New-request suppression, sandwich, L2-SP** | +7.5 pp [+0.0,+17.5] | +7.5 pp [+0.0,+15.0] | **robust (both ns)** |
| **Collateral damage to bird under MAB** | −45.0 pp | −40.0 pp | **robust** |
| **L2-SP prevents it** (MAB_L2 bird) | −5.0 pp | −5.0 pp | **robust** |

The new-request suppression numbers are reproducible to within 2.5 pp across
independent training runs. That is the most stable quantity in the study.

## 2. The L2-SP tradeoff — the clearest result

Adding L2-SP at the second request, in both seeds:

| | dog branch | sandwich branch |
|---|---|---|
| new-target suppression, no reg. | +82.5 / +82.5 pp | +50.0 / +52.5 pp |
| new-target suppression, L2-SP | **+42.5 / +42.5 pp** | **+7.5 / +7.5 pp (ns)** |
| bird collateral, no reg. | −45.0 / −40.0 pp | +7.5 / +0.0 pp |
| bird collateral, L2-SP | **−5.0 / −5.0 pp** | +5.0 / −2.5 pp |
| movement from MA (relative L2) | 0.0136 / 0.0135 | 0.0118 / 0.0121 |
| movement from MA, L2-SP | **0.0015 / 0.0015** | **0.0015 / 0.0016** |

L2-SP cuts movement from the parent by roughly 9x. In the dog branch that buys
a large reduction in collateral damage (bird −45 pp → −5 pp) at roughly half the
deletion strength. In the sandwich branch it **eliminates the deletion
altogether** (+7.5 pp, interval touching zero in both seeds) while buying almost
nothing, because that branch had little collateral damage to prevent.

So the coefficient 25000 — a documented example, never tuned for SD-1.5 — is far
too strong for the sandwich branch and arguably reasonable for the dog branch.
A single fixed coefficient is not appropriate across requests.

## 3. What did NOT replicate — and must not be presented as a finding

Seed 17 showed a credible increase in cat detection in the regularised sandwich
branch: **MAC_L2 +15.0 pp [+2.5, +27.5]**. In seed 29 the same arm gives
**−5.0 pp [−12.5, +0.0]** — a different credibility class.

| child | seed 17 | seed 29 | agreement |
|---|---|---|---|
| MAB | −17.5 [−32.5,−5.0] | −27.5 [−50.0,−10.0] | yes (decrease) |
| MAB_L2 | −12.5 [−22.5,−2.5] | −22.5 [−42.5,−7.5] | yes (decrease) |
| MAC | +7.5 [+0.0,+15.0] | −7.5 [−25.0,+5.0] | both ns |
| MAC_L2 | **+15.0 [+2.5,+27.5]** | **−5.0 [−12.5,+0.0]** | **NO** |

Much of the gap is the parent itself moving: D_cat(MA) was 17.5% in seed 17 and
27.5% in seed 29, so the same child rate produces a different signed difference.
That is exactly the instability one training seed cannot reveal, and it is why
the single-seed headline was wrong.

**Conclusion: historical recovery is not established by these runs.** Reporting
it from seed 17 alone would have been a false positive.

## 4. Prompt family matters

Recovery and residual presence are consistently larger under **paraphrased**
prompts than under prompts that name the category. In seed 17, MAC_L2 recovered
+25.0 pp on paraphrases versus +5.0 pp on literals. Evaluating erasure only with
prompts that name the concept understates what the edited model still produces.
This is a generalisation check, not an adaptive attack benchmark.

## 5. What the images show

After the cat deletion, cat prompts render **horses** — the anchor concept
(`horse+cat`). The deletion works by redirecting the target onto its anchor
rather than by degrading the image. Residual cats appear on a minority of
prompt/seed pairs. See `seq17/grids/grid_cat.jpg`.

Visual inspection so far is **by the AI assistant**, not a human. Blinded,
checkpoint-shuffled grids and an empty annotation sheet are prepared in
`seq17/grids/` for human scoring. **No human annotation has been performed.**

---

## Configuration (identical for both seeds)

- CUIG `9932ac3271a122f6d38d19e0b8c8908fe5237ff7`, native ConAbl object training,
  upstream mappings `horse+cat`, `horse+dog`, `flower+sandwich`.
- 1000 optimizer steps per request (upstream sequential uses 2000), 200 anchor
  images/prompts, anchor caches generated once from untouched M0 and shared
  byte-identically across matched arms.
- Effective LR 8e-6 (CLI default 2e-6 × accum 1 × batch 4 × 1 process), AdamW,
  fp32, `kv-xattn` (32 tensors, 19,169,280 params), constant schedule, warmup 500.
- Regularised and unregularised arms differ **only** in `--l2sp_weight 25000`;
  matched arms drew identical target prompts.
- Evaluation: 7 categories × 10 prompts (5 literal + 5 paraphrase) × 4 seeds =
  280 images/checkpoint, identical prompt/seed pairs, CFG 7.5, 30 steps, 512², fp16.
- Detector: `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, SHA-256
  `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`.

### Checks that passed before any editing
M0 generates cat/dog/sandwich at 100/100/100% (separate calibration set);
negative controls fire 0%; a no-op delta reload leaves outputs byte-identical
(8/8); evaluation prompts disjoint from 999 training strings; every child
verified against the same MA parent (deviation 0.0); L2-SP reference asserted
equal to MA before the first optimizer step.

### Runtime
828–832 s per 1000-step request, peak 10442–10516 MiB on A100-SXM4-40GB;
0.99 s per evaluation image. Ten training runs and eleven evaluation sweeps
across four GPUs.

---

## Limitations

- **Two training seeds.** Intervals are sampling uncertainty over prompts within
  a single run. n=2 supports a direction check only — no population variance.
- 1000 steps is a reduced exploratory budget; upstream uses 2000.
- Detector proxies: a detection is not ground truth, and absence of a detection
  is not proof of erasure.
- Dog vs sandwich differs in **both** anchor (`horse` vs `flower`) and semantic
  distance from cat; this design cannot separate the two.
- L2-SP 25000 is a documented example, not tuned. No coefficient search was run.
- Two regularisation settings cannot define a Pareto frontier.

## What this motivates next

1. **Separate anchor sharing from semantic similarity.** The branches differ on
   both axes and behave differently. Run a second request that is semantically
   far from cat but *shares* the `horse` anchor, and one semantically close but
   using a different anchor. Only that factorisation explains the dog/sandwich
   asymmetry.
2. **More training seeds before any recovery claim.** D_cat(MA) moved 10 pp
   between two seeds; that alone can flip the sign of a recovery estimate.
3. **Per-request L2-SP strength.** 25000 was roughly reasonable for dog and
   destroyed the sandwich deletion. Sweep on a held-out split — never on these
   final numbers.
4. **Longer streams**, to test whether suppression continues to deepen in near
   branches and whether drift accumulates in far ones.
