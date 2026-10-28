# U40 — ResAlign's un-attacked claim **reproduces exactly**, and the control proves the instrument works

**2026-10-28.** Before testing whether ResAlign's resistance survives a *targeted* attack — the
question their paper does not ask — their own un-attacked figure had to reproduce here. It does.

## Result

Their protocol (`scripts/eval_safety.sh`): `data/unsafe.csv`, 3 images per prompt, 30 inference
steps, their MHSC "sexual" detector (ViT-L-14/openai + their trained projection head), threshold 0.5.
600 images per arm.

| arm | US rate | mean detector score |
|---|---:|---:|
| **ResAlign higher-safety** (authors' released checkpoint) | **0.0000** (0/600) | 0.0143 |
| **base SD-1.4** (positive control) | **0.1817** (109/600) | 0.2053 |

**Their reported un-attacked US rate is 0.0000. We measure 0.0000.** Exact reproduction, on the
authors' own weights, with the authors' own detector and prompt set.

## The control is why this means anything

A detector reading 0.0000 on a defended model proves nothing unless it reads clearly non-zero on an
undefended one. Base SD-1.4, through the identical pipeline, reads **0.1817** — 109 of 600 images
flagged, mean score 14× higher. **The instrument works and the zero is real.**

This is the first result in this project where a defence holds at zero under a measurement whose
positive control has been verified.

## Why it matters for the programme

Two things follow, and they cut in opposite directions.

**For ResAlign:** their defence is genuine and their reporting is accurate. Nothing in our
criticism of this field's measurement applies to them — their metric is a *trained binary detector
with a threshold*, not a closed-set argmax with no reject option, so the U29/U30 defect does not
touch it. **Their evaluation is sounder than the one we have been criticising, and the write-up must
say so.**

**For our central claim:** the tier-3 question is now well-posed and worth asking. If targeted
relearning breaks a checkpoint that genuinely holds at 0.0000 under a validated detector, that is
the strongest form of our result. If it does not break, our claim has a real exception and the paper
changes shape. Either outcome is informative; neither was before this reproduction.

## Deviations from their pipeline, stated

- **Images scored in memory, never written to disk.** Their script generates to a directory and
  scores it afterwards. Seeds are recorded (`1000*prompt_index + k`) so any image can be regenerated
  exactly. This changes nothing measured.
- **Safety checker disabled**, as their protocol requires — leaving it on measures the filter, not
  the model.
- `unsafe.csv` / MHSC only. Their **IP** figure uses `i2p.csv` with NudeNet, which is not installed
  here; that half is not reproduced.
- One checkpoint (higher-safety). The higher-utility checkpoint is downloaded but not yet evaluated.

## Provenance

Checkpoints are the authors' released files, relayed via the owner's Hugging Face account: 686
tensors, 859,520,964 parameters, fp32, 0 missing keys against SD-1.4's UNet, differing from base by
0.32% of weight norm. **Not our implementation** — unlike every other baseline in this project.
