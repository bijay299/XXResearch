# Prescribed paired audit packet — 180 items

The **approved** human audit, built from frozen-test images that already exist.
Nothing was generated. **No human annotation has been performed**: every label
cell in `label_sheet_EMPTY.csv` is empty by design.

    ten matched (prompt, generation-seed) tuples
      x 2 training seeds        (17, 29)
      x 3 categories            (cat, dog, bird)
      x 3 arms per tuple        (MA, MAB_L2, U* = step 70)
      = 180 paired image items

## Why paired

The same prompt **and** the same generation seed is judged under all three arms.
That is the only way a human can tell whether a detector verdict *change between
arms* is a real change in the image, which is the question the detector numbers
need checking against. An independently stratified sample cannot answer it,
because it does not hold the tuple fixed.

## What was built

| property | value |
|---|---|
| items | **180** — 60 tuples × 3 arms, 180 distinct images |
| cells | 18 `(training seed, category, arm)`, exactly **10** items each |
| tuple universe | the frozen TEST manifest `5dd87dbb5a77c0f4…` |
| sampling | uniform without replacement over tuples present in all three arms, **declared RNG seed 20261012** |
| realised literal/paraphrase split | **not forced** — reported per cell in `packet_manifest.json` (4–6 literal of 10) |
| labels | **empty**, asserted before the build completes |
| key | **outside** the packet, at `<data root>/annotation_paired_180/KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv` |

## Detector independence is structural, not just declared

The builder opens **only** each slot's `image_report.json` — the generation-side
record — and the frozen test manifest. It never opens `detections.jsonl`, so no
detector score, verdict or threshold can enter the draw even by accident. The
sampling seed is a command-line argument, fixed before any tuple is drawn and
recorded in the manifest.

## Blinding, and one disclosed limitation

Hidden from the annotator: which arm produced an image, which training seed,
everything the detector produced, and the tuple grouping. Shown: the category to
judge, because without it there is no question. Order is shuffled once under the
sampling RNG. The build **aborts** if a probe finds any arm, seed, detector
string or the salt in the packet's text files, if the key, salt or overlap report
is inside the packet, or if any label cell is non-empty.

**The blinding salt is a secret, and it has to be.** Blinded ids are
`sha256(salt | image_path)[:12]`, and image paths are deterministic —
`<eval_root>/seed<S>_<arm>/images/<prompt_id>_seed<gs>.jpg`. With a *published*
salt, anyone who can read the manifest could hash the few hundred candidate
paths per cell and recover the arm of every item: a published salt is not a
blind at all. A fresh 32-byte secret is therefore generated per packet, written
beside the key at `BLINDING_SALT_SECRET.txt` (mode 600, outside the packet), and
only its **sha256** is published. For the same reason the published manifest
**withholds the drawn `(prompt_id, gen_seed)` tuples** and their generation
seeds — the counts and realised family splits, which are what a reviewer needs,
are published in full. Passing the stored secret back via `--salt` reproduces
the packet exactly.

> **Disclosed, not silently fixed:** the earlier packets
> ([`../annotation_packet/`](../annotation_packet/), salt `AUDIT-01`;
> [`../annotation_packet_v2/`](../annotation_packet_v2/), salt `AMENDMENT-01`)
> published their salts, so their blinded ids have the enumerability described
> above. Those packets are **preserved as delivered** and were not rebuilt. It
> is a theoretical de-blinding path, not an observed one: it needs repository
> access and deliberate effort, and the person being blinded is the annotator,
> who is given only the packet directory. It is recorded here so a future reader
> does not assume all three packets blind equally.

**Disclosed:** a paired packet necessarily contains three images of the same
prompt and generation seed. An annotator may notice the similarity and infer
that items belong together. Nothing identifies *which* of a similar group came
from which arm, and the order is shuffled. This is a property of the approved
design and is recorded rather than concealed.

## Relationship to the 228-item supplementary packet

The previously delivered packet — 168 independently stratified items across
**seven** categories plus **60** items enriched for detector-hard cases — is
**preserved unchanged** as a separate supplementary audit at
[`../annotation_packet_v2/`](../annotation_packet_v2/). It is **not** the
approved audit, and the two are **not pooled**: its enriched subset is biased
upward by construction, and this packet's sample is paired rather than
category-stratified. An error rate computed across them is not an error rate.

**Overlap: 13** of these 180 images also appear in that packet, and **0** carry
a label. `overlap_summary.json` here holds the aggregate counts; the item-level
mapping and any label already entered live with the key, **outside** both
packets, because overlap reveals set membership — it is for the coordinator, not
the annotator. Prior labels are carried into that report rather than into this
sheet: pre-filling cells here would import the other packet's sampling design
into this one.

## Reproduce

    python scripts/seq/build_paired_annotation_packet.py \
        --eval_root <data root>/diag_v2_early_grid/eval_test \
        --out_root  <data root>/diag_v2_early_grid/annotation_paired_180 \
        --repo_out  results/audit_v1/annotation_packet_paired180 \
        --compare_packets "<data root>/diag_v2_early_grid/annotation_v2/packet/label_sheet.csv"

Images are **not** committed. Only this README, the empty sheet, the
instructions, the manifest and the overlap summary are. The packet with images
lives at `<data root>/diag_v2_early_grid/annotation_paired_180/packet/`.

**Detector results stay PROVISIONAL until this audit is complete.**
