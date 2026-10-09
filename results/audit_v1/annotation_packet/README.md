# Blinded annotation packet — AUDIT-01 v1

236 items built **only** from evaluation images that already existed.
Nothing here was generated, and **no human annotation has been performed**:
`label_sheet_EMPTY.csv` is empty by design.

## How to annotate

1. Work through `label_sheet_EMPTY.csv` in the order given (save your copy
   under a new name). For each row, open `images/<item_id>.jpg` and judge only
   the category named in `category_to_judge`.
2. Fill the four judgement columns:
   - `target_present__yes_no_unsure` — is that category visibly present?
   - `ambiguity__clear_ambiguous_undecidable` — how hard was the call?
   - `quality_concern__none_minor_severe` — artefacts, blur, distortion.
     **A quality concern is not a presence judgement**; record it here only.
   - `context_concern__none_describe_in_notes` — anything else visible that
     matters (wrong object substituted, odd scene, text artefacts).
3. Do not open the key until the sheet is complete. It is deliberately stored
   outside this directory, at:
   `/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/key/KEY_do_not_open_before_annotating.csv`

## What is hidden and what is shown

Hidden: which checkpoint and training seed produced the image, the detector's
score and verdict, and whether the item is from the random or the diagnostic
set. Shown: the image and the category to judge — without the category there
is no question to answer.

## Two sets, which must not be pooled

| set | n | what it is | what it may be used for |
|---|---|---|---|
| R (random) | 176 | stratified random sample over 88 (seed, checkpoint, category, family) cells, `rng_seed=20261009` | the only set for an overall detector error rate, over its strata, with the weights in `MANIFEST.json` |
| D (diagnostic) | 60 of 494 eligible | **enriched** for threshold-boundary cases (hit at 0.3 but not 0.7) and parent→child verdict flips | characterising failure modes only |

**SET D is biased upward by construction.** It over-samples exactly the images
the detector finds hardest. An error rate computed on SET D, or on SET R and
SET D pooled, is not an overall error rate and must not be reported as one.

Items are shuffled once under the fixed RNG seed, so the packet is reproducible
and carries no condition ordering. Blinded ids are
`sha256(salt | seed | checkpoint | image_sha256)[:10]`, salt `AUDIT-01`.

## Reproducing this packet

    python scripts/seq/build_annotation_packet.py --copy_images

Images are **not** committed to the repository. Only this README, the empty
sheet and the manifest are. The packet with images lives at `/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/packet`.
