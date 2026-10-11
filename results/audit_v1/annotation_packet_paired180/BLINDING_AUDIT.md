# Delivery gate — three channels, four packets, one stated threat model

Whether a packet holder can work out which arm produced which row. Asserting
they cannot is worthless, so the packets are **attacked** with
[`scripts/seq/audit_packet_blinding.py`](../../../scripts/seq/audit_packet_blinding.py).
**No key is opened, no salt is printed, and no item-to-arm assignment is ever
reported** — only aggregate counts and, for the text sweep, the file paths
involved.

Verbatim output: [`blinding_audit.log`](blinding_audit.log). Regressions:
[`scripts/seq/test_packet_delivery_gate.py`](../../../scripts/seq/test_packet_delivery_gate.py)
— **27 checks, 0 failures**. CPU only. **No images were generated.**

## The gate certified deliveries it had never seen — fixed

Handed the real 180-row published sheet together with an **empty image
directory**, or with **one unrelated PNG**, the previous gate scored 0/0 and 0/1
digest matches and printed *"BLINDING HOLDS against all 3 attacks"*, exit 0.
Both reproduced before repair (cases 2–3 in the log).

**The gate is now bound to the published sheet** and refuses anything that does
not match it row-for-row. Five statuses replace a boolean, and only one is a
pass:

| status | meaning | exit |
|---|---|---|
| `holds` | the attack was repelled | 0 |
| `BROKEN` | the channel de-blinds rows | 1 |
| `UNRESOLVED` | an occurrence that **requires assessment** — not proven de-blinding, not a pass | 2 |
| `NOT MODELLED` | the attack does not apply to this packet — **untested**, not passed | 2 |
| `REJECTED` | the input is not fit to certify; **no blinding claim either way** | 3 |

Rejected inventories, each with its own regression: empty directory; one
unrelated file; a referenced file missing; an unreferenced extra file; two rows
referencing one file; an unreadable referenced file; an empty comparison corpus.
File count and **distinct-hash** count are now reported separately, and two
items sharing byte-identical content is itself `UNRESOLVED`.

**Builder, correspondingly:** a missing source image is now **fatal** instead of
silently producing no file; an existing destination is **re-exported and
re-verified** rather than trusted; and the byte guard now covers the **stated
published corpus — 6,880 digests from the committed evidence archives**, not
just the 180 selected originals, aborting on any collision, any duplicate
exported digest, or any count mismatch against the item list.

## Threat model — what `holds` is scoped to

| adversary | has | status |
|---|---|---|
| **A — packet holder with repository access** | the packet, plus every committed key-free record | **what the three attacks model.** Closed for the repaired paired packet |
| **B — holder of the private source images** | the originals the packet is made from | **not closed, and cannot be.** Pixel comparison identifies every item; re-encoding changes bytes, not content. Access control only |
| **C — holder of an earlier packet** | an earlier packet's *original* bytes | **partially open**, quantified below |

**No universal blinding claim is made.** Only adversary A is tested.

## The three channels

1. **Identifier hashing** — ids are `sha256(salt | image_path)[:12]` over
   deterministic paths; a *published* salt makes the id space enumerable.
2. **Published RNG state plus row order** — if one RNG both draws the sample and
   shuffles the rows and its seed is published, the ordered item list replays
   from published data. A secret salt is irrelevant: the channel is row
   *position*.
3. **Delivered image bytes** — the committed image reports and detector rows tie
   every image's `sha256` to its source slot. Hashing a renamed file reads off
   the arm. The sweep covers **6,880 slot-resolving published digests** and
   **246 committed text files**. A digest occurrence is **graded**: within two
   lines of a checkpoint/arm/slot/training-seed identifier it is an
   **identifying association** and the packet is BROKEN for those rows; a bare
   mention elsewhere is **UNRESOLVED and requires assessment**, because a digest
   alone proves nothing.

## Results

| packet | 1 identifier | 2 RNG + row order | 3 delivered bytes | verdict |
|---|---|---|---|---|
| **paired 180 — repaired, the packet to hand out** | holds | holds | **holds — 0/180**, 180 distinct hashes, 0 text-corpus occurrences | **holds against all three** |
| **supplementary 228 — preserved as delivered** | **BROKEN — 228/228** ids regenerated | not modelled | **BROKEN — 228/228** resolve to a unique slot | **BROKEN on two channels** |
| **pilot 236 — preserved as delivered** | **UNRESOLVED** — salt published under a different id rule, 0/236 regenerated | not modelled | **BROKEN — 2 of 236** | **BROKEN on one channel** |
| gate fed an empty dir / one unrelated file | holds | holds | **REJECTED** | **no claim made** |

### The paired packet: channel 3 was live, and it was the serious one

The second build **copied source bytes unchanged**: all **180/180** files hashed
to a published digest resolving to a unique slot. The secret salt and secret row
order were irrelevant — the file content *is* the identifier.

**Repair, under adversary model A.** Each copy is re-encoded from the **decoded
pixels** into a fresh lossless **PNG**, no EXIF, no JFIF block, no PNG text
chunk. Then:

- **displayed pixels verified, not asserted** — every exported image's pixel
  array compared element-by-element against the source's: **180/180 identical**,
  in the builder and again independently;
- **0 of 180** files carry EXIF or any stray `info` key beyond rendering hints;
- the build **refuses to finish** if any exported byte stream equals a published
  digest, if pixels moved, if two exported digests collide, or if the file count
  misses the item count;
- **re-verified after the rebuild**: 0/180 matches across 6,880 digests and 246
  text files, 180 distinct hashes.

The replacement was rebuilt under a **fresh** secret salt and row order, so its
ids are unrelated to the defective build's. Rebuilding again with the stored
secret **reproduces the same ids** — verified: the published sheet's digest is
unchanged across the hardened rebuild.

### The pilot 236 packet: a new, small, genuine break

Channel 3's graded sweep flagged **2 of 236** delivered digests appearing in
`results/seq17/calibration/reload_check.json`. The gate reported this as
`UNRESOLVED`; **assessed by hand, it is identifying.** That file records a
reload-determinism check and lists each digest beside `sha256_m0_direct` /
`sha256_noop_reload` — naming the producing checkpoint as **M0**, which is one
of the **six** checkpoints in that packet's cells, and whose identity the packet
explicitly blinds. So the condition of **2 of 236 items (0.8%)** is readable from
committed data.

The pattern now matches those field names, so the gate reports it as BROKEN
without needing a human. **The packet is preserved exactly as delivered and was
not rebuilt**, per the handoff; **0 of its 236 label cells are filled**.

### The 228 packet: broken on two channels, and re-salting cannot fix it

Its identifier exposure reproduces from **only the files committed at
`7e01fbd`** (case 6), so neither channel was caused by the closeout's evidence
archive. **Re-salting changes ids and leaves the bytes unchanged**, so attack 3
would still resolve 228/228. Making it blind under adversary A would require
re-exporting its images the way the paired packet now does, changing every id and
file. That is a decision about a preserved artifact and is **left to the PI.**
Until then treat it as an **unblinded** instrument: usable for inspecting failure
modes, not for an unbiased detector error rate.

### History and overlap

**History.** No frozen-test image has ever been committed: **0** image blobs
under any `*/images/` path in the full object history. The only committed pixels
are `results/*/figures/*` and five `results/seq17/grids/grid_*.jpg` contact
sheets built from the **pilot** set, a different prompt set from which the paired
packet draws nothing.

**Overlap — adversary C, quantified.** **13 of the paired packet's 180** images
also appear in the 228-item packet. That packet holds **original** bytes, so a
holder of it can hash those 13, learn their arm, and locate them among the
paired packet's re-encoded copies by **pixel comparison** — **13/180 ≈ 7%.**
None carries a label. Re-salting the 228 packet would not change this: the
exposure is its image bytes.

**Recommendation, unchanged:** the paired audit must go to an annotator who has
seen **neither earlier packet nor any source image**.

### Exposure status

**No annotator has seen anything.** 0 of 180, 0 of 228 and 0 of 236 label cells
are filled; no human annotation has been performed anywhere in this project, and
no packet has left this host. **If any copy of an original or a de-blinded packet
ever did reach an annotator**, their labels on affected items could not be
treated as blind and those items would need re-judging by an unexposed
annotator. Nothing observed indicates that happened.

The defective build's record — note, empty sheet, manifest, key — is preserved at
`annotation_paired_180.superseded_unblinded_export/`. Its 180 image copies were
verified **180/180 byte-identical duplicates of preserved originals** and then
removed, because the replacement draws the same 180 sources and a holder of the
old copies could have pixel-matched into the new packet. **No further artifact
has been deleted**, and none will be.

## What this does not establish

- **Three channels, not all channels.** Untested: file size or timing
  correlations, filesystem ordering, and an annotator clustering the three
  near-identical images of a tuple (disclosed in the packet README as a property
  of the paired design).
- **`NOT MODELLED` is not a pass.** Attack 2 is specialised to the paired
  design; the earlier packets' draws consume detector output, so replaying them
  needs the raw rows, and that was not attempted.
- **`UNRESOLVED` is not a pass either**, and is not proof of de-blinding. It
  means a human must look — as happened with the pilot packet above.
- **Adversary B is out of scope by construction.**
- **Nothing about operational handling** — who holds the key, where the packet is
  copied, what the annotator is told.
- The private 180-image pixel-equality, exposure and history results above are
  **this project's reported evidence**, not an independent party's verification.

## Reproduce

    python scripts/seq/test_packet_delivery_gate.py          # 27 checks
    python scripts/seq/audit_packet_blinding.py \
        --packet_dir results/audit_v1/annotation_packet_paired180 \
        --delivered_images <data root>/annotation_paired_180/packet/images

Exit codes: `0` all repelled, `1` broken, `2` unresolved or not modelled, `3`
input rejected. Omitting `--delivered_images` reports attack 3 as **not
modelled** rather than passed.
