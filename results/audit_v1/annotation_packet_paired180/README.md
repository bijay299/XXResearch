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
| image files | **re-encoded**, not copied: lossless PNG from the decoded pixels, no EXIF/JFIF/PNG-text, **180/180 pixel-verified** against the source, **0/180** hashing to any published digest |

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

**Three channels could de-blind this packet; all three were attacked, not
assumed** — full report and threat model: [`BLINDING_AUDIT.md`](BLINDING_AUDIT.md),
verbatim output in [`blinding_audit.log`](blinding_audit.log). The attacks model
**one adversary: a packet holder who also has the repository.** A holder of the
private source images can still identify every item by pixel comparison, and no
re-encoding can prevent that — no universal blinding claim is made.

1. **The identifier.** Blinded ids are `sha256(salt | image_path)[:12]` and image
   paths are deterministic — `<eval_root>/seed<S>_<arm>/images/<prompt_id>_seed<gs>.jpg`
   — so a *published* salt makes the whole id space enumerable. The salt is
   therefore a fresh 32-byte **secret**, stored beside the key at
   `BLINDING_SALT_SECRET.txt` (mode 600, outside the packet); only its sha256 is
   published.
2. **The row order.** This broke the packet's **first build**: one
   `random.Random(20261012)` both drew the sample and shuffled the rows, and that
   seed was published, so the attack replayed **180/180 positions exactly** and
   recovered the arm of every row — the secret salt was irrelevant, because the
   channel was position. The row order is now shuffled by a **separate RNG seeded
   from the secret**, published only as `order_seed_sha256`.
3. **The image bytes.** This broke the **second build**, and it was the serious
   one: the packet copied source bytes unchanged, and the committed image reports
   and detector rows associate every image's `sha256` with its source slot, so
   **180/180** delivered files resolved to a unique slot by hashing alone — no
   key, no salt, no order needed. Each annotation copy is now **re-encoded from
   the decoded pixels** into a fresh lossless PNG with no EXIF, JFIF or PNG text
   chunk; **all 180 exported images are pixel-verified element-by-element against
   their source**, so what the annotator sees is unchanged, and the build refuses
   to finish if any exported byte stream equals a published digest. Re-verified
   after the rebuild: **0/180** matches across 6,880 slot-resolving digests and
   243 committed text files.

The **sampling** seed stays public deliberately: a reader must be able to replay
the draw and confirm it consumed no detector output. So tuple **membership is
public by design**, and the drawn tuples are published in full; what is protected
is the **assignment** of arm and training seed to a row. Knowing membership, a
reader still faces a 1-in-6 guess per row within that row's visible category.
Passing the stored secret back via `--salt` reproduces the packet exactly.

> **The 228-item packet is BROKEN on two channels, and preserved anyway.**
> Attack 1 regenerated **228/228** of its ids using its published salt
> `AMENDMENT-01`, reproducibly from **only the files committed at `7e01fbd`**;
> attack 3 resolved **228/228** of its delivered files to a unique slot, because
> it ships original bytes. Neither was caused by the closeout's evidence
> archive. It is preserved as delivered and **not** rebuilt, per the handoff;
> **0 of its 228 label cells are filled**, so nothing is lost today.
> **Re-salting it alone cannot fix this**: new ids leave the image bytes
> unchanged and attack 3 still resolves 228/228 — making it blind would require
> re-exporting its images, changing every id and file, which is a decision left
> to the PI. Until then treat it as an **unblinded** instrument. The pilot
> packet publishes its salt too, under a different id rule this attack did not
> break. **The three packets do not blind equally.**
>
> **Overlap exposure, quantified:** **13 of these 180** images also appear in the
> 228-item packet, whose originals are hashable, so a holder of that packet can
> locate those 13 here by pixel comparison — **7% of this packet.** Give this
> audit to an annotator who has **not** received the 228-item or 236-item
> packet. None of the 13 carries a label.

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
