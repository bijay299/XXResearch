# How to get the annotation packet — for the PI

The packet is **236 images (26 MB)** and is deliberately **not in this
repository**. Images are private assets and bulk image archives are not pushed
to GitHub. Nothing has been sent to any external service.

**No human annotation has been performed.** The label sheet is empty by design,
and nothing in the current reports may be described as human-validated.

---

## Where it is, on `hyperplane`

| what | path |
|---|---|
| packet directory (images + empty sheet + manifest + README) | `/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/packet/` |
| **ready-made archive, key excluded** | `/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/annotation_packet_v1.tar.gz` |
| **the arm key — withheld, do not open before annotating** | `/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/key/KEY_do_not_open_before_annotating.csv` |

Archive: 25,794,644 bytes,
sha256 `a9ee2958a38192bb93e2e96830cd73e03d480357f061725cf7451193b04c010b`.
It contains **236 jpgs + the empty sheet + manifest + README** and
**zero key entries** — verified by listing the archive.

## Three ways to work through it

**1. Copy the archive to your own machine** (recommended — the key stays on the
server, so it cannot be opened by accident):

```bash
scp bijaypandey@hyperplane:/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/annotation_packet_v1.tar.gz .
tar -xzf annotation_packet_v1.tar.gz
cd packet
# open label_sheet_EMPTY.csv in a spreadsheet, save as label_sheet_<yourname>.csv
# open images/<item_id>.jpg as you go
```

**2. Browse the images in a browser over an SSH tunnel**, without copying
anything. On the server:

```bash
cd /data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1/packet
python3 -m http.server 8765 --bind 127.0.0.1
```

Then on your own machine:

```bash
ssh -N -L 8765:127.0.0.1:8765 bijaypandey@hyperplane
# browse http://127.0.0.1:8765/images/
```

Bound to `127.0.0.1` so it is reachable only through your own tunnel, and
started from `packet/` so the server's document root **cannot reach the key
directory above it**. Stop it with Ctrl-C when done.

**3. Annotate in place on the server**, editing a copy of the sheet there.
Workable but slower for image viewing.

## The sheet

`label_sheet_EMPTY.csv`, one row per item, in the shuffled order:

| column | what to record |
|---|---|
| `item_id`, `image_file` | identifiers — leave alone |
| `category_to_judge` | the category to look for — **given**, not blinded |
| `target_present__yes_no_unsure` | is that category visibly present? |
| `ambiguity__clear_ambiguous_undecidable` | how hard was the call? |
| `quality_concern__none_minor_severe` | artefacts, blur, distortion |
| `context_concern__none_describe_in_notes` | wrong object substituted, odd scene, text artefacts |
| `notes` | free text |

**Quality and context are separate columns from presence on purpose.** A badly
rendered image is not an absent object; recording a quality concern must not
double as a presence judgement.

## What is hidden, and what is not

Hidden: which checkpoint produced the image, which training seed, the detector's
score, the detector's verdict, and whether the item is from the random or the
diagnostic set.

Shown: the image and the category to judge — without the category there is no
question to answer.

## The two sets, which must not be pooled

| set | n | what it is | what it may be used for |
|---|---|---|---|
| **R** random | 176 | stratified random sample over 88 (seed, checkpoint, category, family) cells, `rng_seed=20261009`, per-cell weights in `MANIFEST.json` | the **only** set for an overall detector error rate, over its strata, with those weights |
| **D** diagnostic | 60 of 494 eligible | **enriched**: detector hit at 0.3 but not 0.7, and MA→child verdict flips at the same prompt and generation seed | characterising failure modes **only** |

Set membership is in the key, so while annotating you cannot tell R from D —
which is the point. **Set D is biased upward by construction**: it over-samples
exactly the images the detector finds hardest. An error rate computed on set D,
or on R and D pooled, is **not** an overall error rate and must not be reported
as one.

## After the sheet is filled

1. Keep your filled copy under a new name; leave `label_sheet_EMPTY.csv` empty.
2. Only then open the key, and join on `item_id`.
3. Estimate detector error from **set R only**, applying the per-cell weights,
   and report it per prompt family. M0 appears once in set R, not twice, because
   it is one evaluation set shared by both seed tables.
4. Report set D separately as failure modes.

Rebuild the packet at any time (CPU only, no GPU):

```bash
python scripts/seq/build_annotation_packet.py --copy_images
```
