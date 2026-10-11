# Blinding audit — three channels, four packets, one stated threat model

The question is whether a packet holder can work out which arm produced which
row. Asserting they cannot is worthless, so the packets are **attacked** with
[`scripts/seq/audit_packet_blinding.py`](../../../scripts/seq/audit_packet_blinding.py).
**No key is opened, no salt value is printed, and attack 3 reports aggregate
counts only** — never which item resolved to which slot.

Verbatim output: [`blinding_audit.log`](blinding_audit.log). CPU only. **No
images were generated.**

## Threat model — what "holds" is scoped to

| adversary | has | status |
|---|---|---|
| **A — packet holder with repository access** | the packet directory, plus every committed key-free record | **this is what the three attacks model.** Closed for the repaired paired packet |
| **B — holder of the private source images** (`/data/.../eval_test/*/images/`) | the originals the packet is made from | **not closed, and cannot be.** Pixel-for-pixel comparison identifies every item; re-encoding changes bytes, not content. Mitigated by access control alone |
| **C — holder of an earlier packet** | an earlier packet's *original* bytes | **partially open.** Quantified below |

**No universal blinding claim is made.** Only adversary A is tested.

## The three channels

**Attack 1 — identifier hashing.** Ids are `sha256(salt | image_path)[:12]`
over deterministic paths; a *published* salt makes the id space enumerable.

**Attack 2 — published RNG state plus row order.** If one RNG both draws the
sample and shuffles the rows and its seed is published, the ordered item list —
arm included — replays from published data. A secret salt does nothing here:
the channel is row *position*.

**Attack 3 — the delivered image bytes.** The committed image reports and
detector rows associate every image's `sha256` with its source slot. If the
packet ships source bytes unchanged, hashing a renamed file reads off the arm —
no key, no salt, no order needed. This attack hashes the **actual delivered
files** and looks each digest up in **6,880 slot-resolving published digests**
plus **243 other committed text files**.

## Results

| packet | 1 identifier | 2 RNG + row order | 3 delivered bytes | verdict |
|---|---|---|---|---|
| **paired 180 — repaired, the packet to hand out** | holds | holds (best replay 67/180, chance ≈ 60) | **holds — 0/180** match any published digest | **holds against all three** |
| paired 180 — *superseded* defective build | holds | holds | **BROKEN — 180/180** resolved to a unique slot | superseded; image copies since deleted |
| **supplementary 228 — preserved as delivered** | **BROKEN — 228/228** ids regenerated | not modelled | **BROKEN — 228/228** resolved to a unique slot | **BROKEN on two channels** |
| pilot 236 — preserved as delivered | salt published under a different id rule; enumeration got 0/236 — a weakness, not a pass | not modelled | holds — 0/236; its digests appear in **no** committed file or archive | 2 of 3 repelled, 1 **not modelled** — not a clean pass |

## Each defect, and what was done

### Channel 3 was live, and it was the serious one

The second build of the paired packet **copied source bytes unchanged**. All
**180/180** delivered files hashed to a published digest that resolves to a
unique slot, so any packet holder with the repository could recover the arm and
training seed of every row. The secret salt and secret row order added in the
previous round were irrelevant — **the file content is the identifier.**

**Repair, under adversary model A.** Each annotation copy is now **re-encoded
from the decoded pixels** into a fresh lossless **PNG**, carrying no EXIF, no
JFIF block and no PNG text chunk. Then:

- **displayed pixels verified, not asserted** — every exported image's decoded
  pixel array is compared element-by-element against the source's:
  **180/180 identical**, in the builder and again independently afterwards. A
  mismatch aborts the build;
- **metadata checked** — 0 of 180 files carry EXIF or any stray `info` key
  beyond harmless rendering hints; anything else aborts the build;
- **the channel is closed by assertion, not by intention** — the build refuses
  to finish if any exported byte stream equals a published source digest;
- **re-verified after the rebuild**: 0/180 matches across 6,880 slot-resolving
  digests and 243 committed text files.

The replacement was rebuilt under a **fresh** secret salt and fresh row order,
so its item ids are unrelated to the defective build's. That is deliberate:
shared ids would have let a holder of the old build de-blind the new one by
cross-reference.

### History check

No frozen-test image has ever been committed: **0** image blobs under any
`*/images/` path exist in the full object history (`git rev-list --objects
--all`). The only committed pixels are `results/*/figures/*` and five
`results/seq17/grids/grid_*.jpg` contact sheets, which are built from the
**pilot** evaluation set — a different prompt set (`test_*` ids do not appear in
it) from which the paired packet draws nothing.

### Known packet overlap — adversary C, quantified

**13 of the 180** paired images also appear in the 228-item packet. The 228
packet holds **original** bytes, so a holder of it can hash those 13 and learn
their arm — and can then locate them among the paired packet's re-encoded copies
by **pixel comparison**, not by hashing. So **13/180 ≈ 7% of the paired packet
is de-blindable by a holder of the 228 packet.** None of the 13 carries a label.

**Consequence, and the recommendation:** the paired audit should go to an
annotator who has **not** received the 228-item or 236-item packet. Re-salting
the 228 packet would not change this — the exposure is its *image bytes*, not
its ids, which is exactly why re-salting alone cannot solve this channel.

### Exposure status of the three packets

**No annotator has seen anything.** 0 of 180, 0 of 228 and 0 of 236 label cells
are filled; no human annotation has been performed anywhere in this project, and
no packet has been distributed off this host. The defective build's 180 image
copies were **byte-identical duplicates of preserved originals** (verified
180/180 before removal) and were deleted after the defect was demonstrated and
logged, because the replacement draws the same 180 source images and a holder of
the old copies could have pixel-matched into the new packet. Its record — note,
empty sheet, manifest, key — is preserved at
`annotation_paired_180.superseded_unblinded_export/`.

**If any copy of an original or a de-blinded packet ever did reach an
annotator**, their labels on the affected items cannot be treated as blind, and
those items must be re-judged by an **unexposed** annotator. Nothing observed
here indicates that happened.

### The 228-item packet: preserved, broken, and not re-salted

It is broken on **two** channels and is **preserved exactly as delivered**, per
the handoff. Its identifier exposure is reproducible from **only the files
committed at `7e01fbd`** (case 5 in the log), so neither it nor channel 3 was
caused by the closeout's evidence archive.

**Re-salting it cannot fix it.** New ids would leave the image bytes unchanged,
and attack 3 would still resolve 228/228. Making it blind under adversary A
would require re-exporting its images the way the paired packet now does, which
changes every id and every file. That is a decision about a preserved artifact
and is **left to the PI, not taken here.** Until then it should be treated as an
**unblinded** instrument: usable for inspecting failure modes, not for an
unbiased detector error rate.

## What this audit does not establish

- **Three channels, not all channels.** Untested: file size or timing
  correlations, filesystem ordering, and an annotator clustering the three
  near-identical images of a tuple (disclosed in the packet README as a property
  of the paired design).
- **"Not modelled" is not "passed."** Attack 2 is specialised to the paired
  design; the two earlier packets' draws consume detector output, so replaying
  them needs the raw rows rather than published metadata, and that was not
  attempted.
- **Adversary B is out of scope by construction**, and attack 3's "holds" is
  explicitly scoped to the digest channel.
- **Nothing about operational handling** — who holds the key, where the packet is
  copied, what the annotator is told.

## Reproduce

    python scripts/seq/audit_packet_blinding.py \
        --packet_dir results/audit_v1/annotation_packet_paired180 \
        --delivered_images <data root>/annotation_paired_180/packet/images

Exit codes: `0` all attacks repelled, `1` at least one broke it, `2` some attack
not modelled. Omitting `--delivered_images` reports attack 3 as **not modelled**
rather than passed.
