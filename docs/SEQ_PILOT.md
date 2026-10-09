# SD-1.5 exploratory sequential object-erasure pilot

A **separate experiment** from the UnlearnCanvas baseline. That baseline remains
blocked on unobtainable assets and its validation gate is untouched: nothing here
writes to it, and nothing here is UA / IRA / CRA or a reproduction of a published
result.

## Question

**Can an earlier concept deletion persist through a later deletion while
preserving both new-request effectiveness and retained utility?**

This is being *tested*, not assumed. Reversal may or may not occur, and a null
result is a result. **L2-SP is an existing baseline regulariser**, applied only at
the second request — it is not a proposed new method and this is not an
end-to-end regularised lifelong method.

## Design

A = cat, B = dog, C = sandwich. Upstream CUIG object mappings: `horse+cat`,
`horse+dog`, `flower+sandwich`.

| Checkpoint | Parent | New deletion | L2-SP |
|---|---|---|---|
| M0 | — | untouched SD-1.5 reference | — |
| MA | M0 | cat | 0 |
| MAB | MA | dog | 0 |
| MAC | MA | sandwich | 0 |
| MAB_L2 | identical MA | dog | 25000 |
| MAC_L2 | identical MA | sandwich | 25000 |

**Experiment 1 (historical interference):** MA vs MAB and MAC — did cat detection
rise after a later, unrelated deletion; how effective was that later deletion;
what happened to retained objects?

**Experiment 2 (preservation vs new deletion):** MAB vs MAB_L2, MAC vs MAC_L2 —
does limiting movement from MA reduce historical recovery or retention damage,
and at what cost to deleting the new target?

Dog vs sandwich is one *selected* request contrast. Because the two use different
anchor mappings (`horse` vs `flower`), any difference between branches cannot be
attributed to semantic similarity alone.

## Backbone, evaluator, and what the numbers mean

- **M0 / backbone:** base Stable Diffusion v1.5, the checkpoint already on disk
  for the earlier mechanics check, used here as the deliberate backbone of this
  experiment.
- **Evaluator:** official Torchvision `fasterrcnn_resnet50_fpn` with
  `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, `eval()` mode, the preprocessing
  transform carried by the weights enum, and category names from the weights
  metadata. SHA-256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`.
  No random weights, no forced-choice classifier substitute. The random-classifier
  software test from the UnlearnCanvas work is **not** used here.
- **Metric:** target detection rate D = fraction of generated images with at
  least one detection of the requested category at confidence ≥ t. Raw
  predictions are saved and D is reported at t = 0.3, 0.5, 0.7.
- These are **detector-based object-presence proxies**. They are not ground-truth
  erasure and not image-quality scores.

## Fixed training protocol

| Setting | Value | Source |
|---|---|---|
| optimizer steps per request | **1000** | reduced exploratory budget (upstream sequential script uses 2000) |
| anchor images / prompts | 200 / 200 | upstream |
| epoch cap | 25 | **raised deliberately** — see below |
| anchor batch size | 4 | upstream default |
| gradient accumulation | 1 | upstream default |
| effective learning rate | **8e-6** | CLI default 2e-6 × accum 1 × batch 4 × procs 1, via `--scale_lr` |
| optimizer | AdamW (β 0.9/0.999, wd 1e-2, eps 1e-8) | upstream default |
| precision | fp32 (`mixed_precision: 'no'`) | upstream accelerate config |
| parameter group | `kv-xattn` (32 tensors, 19,169,280 params) | upstream default, held fixed |
| lr schedule | constant, warmup 500 | upstream default |
| training seed | 17 (second core: 29) | — |
| L2-SP | `--l2sp_weight 25000`; L1 / projection / SelFT disabled | documented example coefficient |

**Epoch cap.** 200 anchor images ÷ batch 4 = 50 optimizer steps per epoch, so
upstream's default `--epochs 1` would have stopped at **50** of the 1000
requested steps. The cap is raised to 25 (capacity 1250) so the step budget is
what actually binds; training still breaks at exactly 1000 steps.

**L2-SP coefficient.** 25000 is the coefficient documented in CUIG's regularizer
README. It is **not** tuned for SD-1.5, and no coefficient search was run — in
particular none against final evaluation results.

## Guarantees the harness enforces

1. **Anchor caches are generated once from untouched M0** with a fixed seed and
   shared byte-identically across matched arms. Pre-generating also makes the
   training seed reproducible across arms: a run that creates the cache inline
   consumes RNG that a run reusing it does not.
2. **No earlier target is replayed.** The second request trains only on its own
   anchor/target pair; nothing preserves cat by showing cat prompts or images.
3. **Parent checkpoints are verified before every child update.** Upstream
   *silently falls back to the base model* when `--unet_ckpt` is missing, which
   would turn MAB into a one-step edit from M0. The wrapper refuses to start, and
   additionally checks that every saved trainable tensor equals the parent after
   loading, with exact key coverage and no unexpected keys. These `delta.bin`
   files hold **absolute updated values**, not additive offsets.
4. **The L2-SP reference is asserted to equal MA** before the first optimizer
   step (upstream captures it lazily on first use). Verified: 32 params,
   19,169,280 elements, max deviation 0.0.
5. **Evaluator inputs never touch the updater.** Evaluation prompts and images
   are disjoint from training prompts, including target-substituted forms.
6. **Device pinning survives Accelerate** (`gpu_ids: all`), one GPU per run, and
   the GPU selector fails closed on unknown or failed readings.

## Evaluation

Frozen **before** any editing: 7 categories × 10 prompts (5 literal naming the
category, 5 unambiguous paraphrases) × 4 generation seeds (101/202/303/404) =
**280 images per checkpoint**, 1680 for the six-checkpoint core. Identical
prompt/seed pairs, scheduler, guidance (7.5), precision (fp16), 512×512 and 30
steps at every checkpoint. Per-image latent seeds come from a fixed documented
rule, so images are matched across checkpoints.

Category roles: **cat** first deletion; **dog/sandwich** second deletion
(branch-specific); **horse** anchor-related control; **bird** selected animal
control; **chair/bicycle** other controls. The undeleted sibling target is
reported as a branch-specific retention outcome, never folded into a common
retain average.

Pre-training checks, all passed:

- M0 generates cat/dog/sandwich at 100/100/100% (t=0.5) on a **separate**
  calibration set; negative controls fire **0%**.
- A no-op delta pushed through the real checkpoint-loading path leaves outputs
  **byte-identical** (8/8), so loading itself does not perturb results.
- Zero overlap between the 70 evaluation prompts and 999 training strings.

## Uncertainty

Paired bootstrap resampled **by prompt** (each prompt carries its four generation
seeds), preserving prompt/seed matching across checkpoints; 2000 draws, 95%
intervals. **One training seed supports sampling uncertainty only** — generation
seeds are not independent training repeats. With a second training seed the two
cores are shown separately; no population variance is claimed from n=2.

## Interpretation rules adopted in advance

- A target visible immediately after its own deletion is **residual failure**,
  not recovery. It counts as recovery only if its score *subsequently increases*
  on the same evaluation set.
- If initial cat suppression is negligible, **historical recovery is not
  established** — there is nothing for a later deletion to undo.
- If L2-SP preserves cat but fails to erase the new target, that is a **tradeoff**
  to report, not an improved method.
- Parameter movement is a **diagnostic association**, never evidence that weight
  movement causes recovery.
- Two regularisation settings cannot define a Pareto frontier.

## Layout

```text
scripts/seq/
├── exp_config.sh              shared configuration
├── pregen_anchors.py          anchor caches from untouched M0
├── train_request.py           one deletion request + verification
├── make_eval_manifest.py      freeze the 280-image manifest
├── generate_eval_images.py    per-checkpoint generation
├── detect.py                  COCO Faster R-CNN detection
├── calibrate.py               pre-training M0 + detector calibration
├── reload_check.py            no-op delta reload check
├── aggregate.py               per-image table, rates, contrasts, bootstrap
├── param_movement.py          normalised movement vs MA and M0
├── make_figures.py            figures 1-3
├── make_grids.py              paired + blinded image grids
├── make_report.py             results report and slide outline
├── run_remaining.sh           orchestration across two GPU slots
└── run_analysis.sh            aggregate -> figures -> report
```

Large artefacts (checkpoints, 1680+ images, anchor caches) live under
`/data/bijaypandey/cuig_pilot/seq_pilot/` and are excluded from git.
