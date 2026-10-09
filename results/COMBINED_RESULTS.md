# SD-1.5 exploratory sequential object-erasure pilot — combined results (seeds 17 and 29)

> ### ⚠ AUDITED AND PARTIALLY CORRECTED — read with `results/audit_v1/`
>
> AUDIT-01 re-derived every number in this document from the raw per-image
> detector predictions. **All 1026 values reproduce exactly — zero mismatches.**
> The arithmetic is sound. Nine **narrative** claims were overstated and are
> corrected in
> [`audit_v1/CLAIM_CORRECTIONS.md`](audit_v1/CLAIM_CORRECTIONS.md); the
> pre-audit text is preserved at
> [`audit_v1/preserved_originals/`](audit_v1/preserved_originals/) and in git at
> `a9de625`.
>
> Corrections applied inline below are marked **[corrected]**. The four that
> change how this document should be read:
>
> 1. **3080 images were generated**, not 3360. `eval_seed29/M0` is a symlink to
>    `eval/M0`; 3360 is the row count of two stacked tables and counts one
>    reused evaluation set twice.
> 2. **"False positive" was wrong** (§3). The two intervals overlap; a differing
>    significance label is not a significant difference. And the **parent-free**
>    contrast `D_cat(MAC_L2) − D_cat(MAC)` *does* agree across both seeds
>    (`+7.5` / `+2.5` pp) — the non-replication is largely in the `D_cat(MA)`
>    reference term, which differs between runs **only on paraphrase prompts**
>    (literal rates are identical at all three thresholds).
> 3. **"Robust" is not available at n=2** (§1). Two runs support a direction
>    check only. "Robust (both ns)" labels two runs each failing to resolve an
>    effect as a finding.
> 4. **The retention comparisons are confounded by unequal deletion strength**
>    (§2). L2-SP arms deleted far less than the unregularised arms, so their
>    greater retention is not yet a preservation result. The missing direct
>    contrast is now computed in
>    [`audit_v1/CORRECTED_TABLES.md`](audit_v1/CORRECTED_TABLES.md), and the
>    proposed fix is
>    [`docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md`](../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md).

**Not UA/IRA/CRA. Not a reproduction of any published result.** The backbone is
base Stable Diffusion v1.5; the evaluator is the official Torchvision COCO
Faster R-CNN. Every rate is a detector-based object-presence proxy — not
ground-truth erasure, not an image-quality score. The blocked UnlearnCanvas
baseline and its validation gate are untouched.

Two complete cores were run: **seed 17** and **seed 29**, five training runs
each, six evaluated checkpoints each, 280 images per checkpoint.
**3080 evaluation images were generated, zero generation or detection failures** — across 11 distinct checkpoint evaluations. The two stacked seed tables have 3360 rows because M0's 280 samples are reused by both. **[corrected]**

---

## The question

> Can an earlier concept deletion persist through a later deletion while
> preserving both new-request effectiveness and retained utility?

A = cat, then B = dog or C = sandwich, from a shared parent MA. L2-SP (25000) is
an **existing baseline** added at the second request only.

## The short answer

**No reversal of the earlier deletion was resolved at this precision.** **[corrected]** — an absence of resolution over 10 prompt clusters, not a demonstration that reversal does not occur. At this budget the first
deletion holds, and in the semantically-near branch it is *reinforced* rather
than undone. The one apparently credible recovery in seed 17 **did not
replicate** in seed 29 — see §3.

**L2-SP traded new-request effectiveness for retention at this one untuned
coefficient.** **[corrected]** It cost ~40–47 pp of new-target deletion in every
seed, branch and threshold — the most stable quantity in the study — and in the
dog branch retained ~35–40 pp more bird. But the arms being compared **deleted
different amounts**, so this is not yet a preservation result; see the banner
above and `audit_v1/CORRECTED_TABLES.md` §4.

---

## 1. What agrees in direction across both training seeds

**[corrected]** "Robust" is not available at n=2 and has been replaced
throughout this table by the only cross-seed claim two runs support — whether
the **direction** agrees — at t=0.5. Where both intervals include zero the
label is "neither run resolved this", which is not a finding. Several of these
rows are also threshold-dependent: `MAB` changes credibility class at t=0.3
(`−17.5 [−37.5, 0.0]`). See `audit_v1/CORRECTED_TABLES.md` §3.

| Finding | seed 17 | seed 29 | direction agrees (t=0.5)? |
|---|---|---|---|
| **First deletion works.** cat D(M0)−D(MA) | +82.5 pp [+67.5,+95.0] | +72.5 pp [+50.0,+90.0] | yes (increase) |
| **Dog branch pushes cat further down**, not up. MAB | −17.5 pp [−32.5,−5.0] | −27.5 pp [−50.0,−10.0] | yes (decrease), but class-flips at t=0.3 |
| same, with L2-SP. MAB_L2 | −12.5 pp [−22.5,−2.5] | −22.5 pp [−42.5,−7.5] | yes (decrease) |
| **New-request suppression, dog, no reg.** | +82.5 pp | +82.5 pp | yes (increase) |
| **New-request suppression, dog, L2-SP** | +42.5 pp | +42.5 pp | yes (increase) |
| **New-request suppression, sandwich, no reg.** | +50.0 pp | +52.5 pp | yes (increase) |
| **New-request suppression, sandwich, L2-SP** | +7.5 pp [+0.0,+17.5] | +7.5 pp [+0.0,+15.0] | neither run resolved this **[corrected]** |
| **Collateral damage to bird under MAB** | −45.0 pp | −40.0 pp | yes (decrease) |
| **L2-SP arm shows far less of it** (MAB_L2 bird) **[corrected]** | −5.0 pp | −5.0 pp | yes, but at **unequal deletion strength** |

The new-request suppression numbers are reproducible to within 2.5 pp across
independent training runs. The single most stable quantity in the study is the
**direct** L2-versus-unregularised contrast on the new target, which was absent
from the committed `contrasts.json` entirely: `D_new(L2) − D_new(no-L2)` is
**+40.0 to +47.5 pp** in every cell — both seeds, both branches, all three
thresholds. **[corrected]**

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

L2-SP cuts movement from the parent by roughly 9x. In the dog branch that
coincides with a large reduction in collateral damage (bird −45 pp → −5 pp) **at
roughly half the deletion strength — which is exactly why the two are not yet
comparable**. In the sandwich branch it leaves only **+7.5 pp** of measured
suppression, with an interval reaching zero: at or below the resolution floor,
so its true deletion strength is unknown and may be near zero. **[corrected]**
It has *not* been shown to be eliminated, and because that arm barely deleted,
its retention must not be scored against an arm that did.

So the coefficient 25000 — a documented example, never tuned for SD-1.5 —
**behaves differently in these two branches**: +42.5 pp of dog suppression
versus +7.5 pp of sandwich suppression. Whether any single coefficient serves
both is **untested**; only one value was ever run. **[corrected]** The earlier
claim that a fixed coefficient "is not appropriate across requests" does not
follow from a single tested value.

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
That is exactly the instability one training seed cannot reveal.

**[corrected] — and the gap is specifically a paraphrase effect.** Split by
prompt family, the two runs produced **identical** literal-prompt parent rates
at every threshold, with the whole divergence on paraphrases:

| threshold | literal s17 / s29 | paraphrase s17 / s29 |
|---|---|---|
| 0.3 | 15.0 / 15.0 **identical** | 25.0 / 40.0 |
| 0.5 | 15.0 / 15.0 **identical** | 20.0 / 40.0 |
| 0.7 | 15.0 / 15.0 **identical** | 20.0 / 35.0 |

So the instability sits in the reference term, on exactly the prompt family
where 10 prompts resolve least. The **parent-free** contrast
`D_cat(MAC_L2) − D_cat(MAC)` removes that term algebraically and *does* agree
across both runs.

**Conclusion: this pair of runs does not resolve the sign of parent-referenced
cat change in the sandwich branch.** **[corrected]** That is a statement about
resolution, not about absence — "false positive" was wrong. The two intervals
overlap, the reading is specific to t=0.5, and the **parent-free** contrast
`D_cat(MAC_L2) − D_cat(MAC)` agrees in direction and credibility class across
both seeds (`+7.5 [0.0,+17.5]` and `+2.5 [−7.5,+15.0]`). More training runs are
required before this is reported in either direction. See
`audit_v1/CLAIM_CORRECTIONS.md` C2.

## 4. Prompt family matters

**Residual presence** and **parent-referenced change** are two different
quantities and are separated here. **[corrected]** Residual presence is higher
under paraphrases. Parent-referenced change is also larger under paraphrases —
but partly because the *parent's own* paraphrase rate differs between runs:
`D_cat(MA)` is 20% (s17) versus 40% (s29) on paraphrases, while the literal
rates are **identical** at 15% in both runs at all three thresholds. In seed 17
MAC_L2 sits +25.0 pp above its parent on paraphrases versus +5.0 pp on literals.
What follows is only that evaluating erasure with literal prompts alone
understates what the edited model still produces. This is a fixed two-family
generalisation check, not an adaptive attack benchmark, and not evidence of
greater recovery.

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
3. **Matched deletion effectiveness — now the first priority. [corrected]**
   Before any L2-SP preservation claim, compare it against an unregularised run
   stopped early at the *same* deletion level. The pilot's retention comparison
   is confounded by unequal deletion strength, and no intermediate checkpoint
   exists to resolve it, so this needs a new trajectory. Proposed in
   [`docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md`](../docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md)
   (≈1.93 GPU-hours, pending review).
4. **L2-SP coefficient behaviour. [corrected]** 25000 left +42.5 pp of dog
   suppression and +7.5 pp of sandwich suppression. Whether any single value
   serves both is **untested** — one value was run. Sweep on a development
   split, never on final numbers.
5. **Longer streams**, to test whether suppression continues to deepen in near
   branches and whether drift accumulates in far ones.
6. **Human annotation.** A 236-item blinded packet is built and empty at
   `results/audit_v1/annotation_packet/`; detector validity is still unmeasured.
