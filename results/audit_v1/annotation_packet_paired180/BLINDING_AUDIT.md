# Blinding audit — all three annotation packets

The question asked was whether the committed, key-free records let a reader work
out which arm produced which row. Asserting they don't is worthless, so the
packets were **attacked** instead, with
[`scripts/seq/audit_packet_blinding.py`](../../../scripts/seq/audit_packet_blinding.py).
**No key was opened and no salt value is printed anywhere** — a reconstruction is
reported as a count and a verdict, confirmed against the sheet's *visible*
`category_to_judge` column, which a 180- or 228-long sequence cannot match by
chance.

Verbatim output: [`blinding_audit.log`](blinding_audit.log). CPU only.

## The two channels

**Attack 1 — identifier hashing.** Blinded ids are
`sha256(salt | image_path)[:12]`, and image paths are deterministic:
`<eval_root>/seed<S>_<arm>/images/<prompt_id>_seed<gs>.jpg`. The eval root, the
slot names and the frozen prompt manifest are all published, so the candidate
path space is a few hundred per cell. **If the salt is published too, every id
is enumerable.**

**Attack 2 — published RNG state plus row order.** If one RNG both *draws* the
sample and *shuffles* the rows, and its seed is published, the whole ordered
item list — arm included — replays from published data. **A secret salt does
nothing about this**, because the channel is row *position*, not the identifier.

## Results

| packet | attack 1 — identifier | attack 2 — RNG + row order | verdict |
|---|---|---|---|
| **paired 180** (prescribed) | **holds** — no salt value in the published records; only `salt_sha256` | **holds** — none of 3 reconstructions matches (best 57/180, below the ~60 expected by chance) | **blinding holds** against both |
| **supplementary 228** | **BROKEN — 228/228 ids regenerated** from published records alone, using the published salt `AMENDMENT-01`. Every hit reveals that row's arm and training seed | not modelled (its draw consumes detector output) | **BROKEN** |
| **pilot 236** | salt published (`blinding_salt`), but its id rule is `sha256(salt | seed | checkpoint | image_sha256)[:10]` — path-based enumeration regenerated **0/236**. **A published salt is still a weakness**: only the rule, not a secret, stands between a reader and the mapping | not modelled | **1 of 2 repelled, 1 untested — not a clean pass** |

### The paired packet was broken on its first build, and that is why this exists

The first build of the 180-item paired packet used **one** `random.Random(20261012)`
for both the tuple draw and the row shuffle, and published that seed. Attack 2
replayed **180/180 positions exactly**, recovering the arm and training seed of
every row — *despite* the secret salt, which had been added specifically to
protect the identifier. The identifier was not the live channel; order was.

**The fix, now in the builder:** the **sampling** seed stays public, so a reader
can replay the draw and confirm it consumed no detector output — tuple
**membership is public by design**. The **row order** is shuffled by a separate
RNG seeded from the secret, published only as `order_seed_sha256`. What is
protected is the **assignment** of arm and training seed to a sheet row. Knowing
membership, a reader faces a 1-in-6 guess per row for `(seed, arm)` within the
row's visible category.

Consequence for the published manifest: the drawn tuples **are** published
again. Withholding them was security by omission — the published sampling seed
replays the draw anyway — and omission was concealing the real weakness rather
than fixing it.

### The 228-item packet's exposure predates this work

Attack 1 was re-run against **only the files committed at `7e01fbd`**
(`annotation_packet_v2/packet_manifest.json`, its `label_sheet.csv`, and
`draft_manifests/test_manifest_DRAFT.json`) and broke it there too, 228/228.
So this is **not** a consequence of the evidence archive added in the closeout:
publishing the raw detector rows did not cause it, and the packet was already
fully de-blindable from the original result commit. The cause is the published
salt plus a deterministic path rule.

**It is preserved as delivered and was not rebuilt**, per the handoff. Nothing
is lost by that today: **0 of its 228 label cells are filled**, as are 0 of the
pilot packet's 236 and 0 of the paired packet's 180 — no human annotation has
been performed anywhere in this project. The exposure is a theoretical path
requiring repository access, the documented rule and deliberate effort; the
person being blinded is the annotator, who receives only the packet directory.

**For the PI, not decided here:** if the 228-item packet is to be annotated, it
should be re-salted and re-ordered first on the terms above, which changes its
item ids. That is a decision about a preserved artifact and is left open rather
than taken.

## What this audit does not establish

- It tests **two** channels. Others may exist — image file mtimes, EXIF, file
  sizes correlating with arm, or an annotator clustering the three near-identical
  images of a tuple (disclosed separately in the packet README as a property of
  the paired design).
- "Not modelled" is **not** "passed". Attack 2 is specialised to the paired
  design; for the two earlier packets the draw consumes detector output, so
  replaying it needs the raw rows rather than published metadata, and that was
  not attempted.
- A clean result here says the published records do not de-blind the packet. It
  says nothing about operational handling — who holds the key, where the packet
  is copied, or what the annotator is told.

## Reproduce

    python scripts/seq/audit_packet_blinding.py \
        --packet_dir results/audit_v1/annotation_packet_paired180
    python scripts/seq/audit_packet_blinding.py \
        --packet_dir results/audit_v1/annotation_packet_v2 \
        --sheet results/audit_v1/annotation_packet_v2/label_sheet.csv

Exit codes: `0` both attacks repelled, `1` at least one broke it, `2` some attack
not modelled for that packet.
