#!/usr/bin/env python3
"""The PRESCRIBED paired human-audit packet: 180 items, as approved.

The approved audit
------------------
    ten matched (prompt, generation-seed) tuples
      for each training seed            x2   (17, 29)
      and each of cat / dog / bird      x3
    all three arms for each tuple       x3   (MA, MAB_L2, U*)
    = 2 x 3 x 10 x 3 = 180 paired image items

This is a PAIRED design, and that is its whole point: the same prompt and the
same generation seed is judged under all three arms, so a human can see whether
a detector verdict change between arms is a real change in the image. An
independently stratified sample cannot answer that, because it does not hold the
tuple fixed.

Detector independence is STRUCTURAL here, not merely declared
-------------------------------------------------------------
The tuple universe comes from the frozen TEST manifest, and the per-image
records come from each slot's ``image_report.json`` -- the GENERATION-side
report. ``detections.jsonl`` is never opened by this script, so no detector
score, verdict or threshold can reach the draw even by accident. The sampling
RNG seed is a CLI argument, recorded in the manifest, and fixed before any tuple
is drawn.

What is blinded
---------------
Hidden: which arm produced an image, which training seed, and everything the
detector produced. Shown: the category to judge, because without it there is no
question. Order is shuffled once under the sampling RNG.

TWO channels can de-blind a packet, and both are closed here.

1. THE IDENTIFIER. Blinded ids are ``sha256(salt | image_path)[:12]`` and image
   paths are deterministic, so a PUBLISHED salt lets anyone who can read the
   manifest enumerate a few hundred hashes and recover the arm of every item.
   The salt is therefore a fresh 32-byte secret, stored beside the KEY outside
   the packet; only its sha256 is published.

2. THE ROW ORDER. If one RNG both draws the sample and shuffles the rows, and
   its seed is published, the ordered item list -- arm included -- replays from
   published data, and a secret salt does nothing about it. ``audit_packet_
   blinding.py`` reconstructed 180/180 rows of this packet's first build exactly
   that way. The row order is therefore shuffled by a SEPARATE RNG seeded from
   the secret.

3. THE IMAGE BYTES. This was the live channel in the second build. The packet
   copied source bytes unchanged, and the committed image reports and detector
   rows associate every image's sha256 with its source slot -- so a packet
   holder could hash a renamed file and read off the arm. All 180 of the second
   build's files resolved to a unique slot that way. Neither the secret salt nor
   the secret row order touches it. Each annotation copy is therefore RE-ENCODED
   from the decoded pixels into a fresh lossless PNG, stripped of EXIF/JFIF/PNG
   text, with the decoded pixels compared element-by-element against the
   source's so that "the annotator sees the same image" is verified, not
   asserted. The build aborts if any exported file hashes to a published digest.

The SAMPLING seed stays public on purpose: a reader must be able to replay the
draw and confirm it consumed no detector output. So tuple MEMBERSHIP is public
by design; what is protected is the ASSIGNMENT of arm and training seed to a
sheet row.

THREAT MODEL, stated rather than implied. Channels 1-3 are closed against a
packet holder who ALSO has the repository. They are NOT closed against a holder
of the private source images under the data root: the packet is made of those
images, so pixel comparison still identifies every item, and no re-encoding can
change that. Verify with ``scripts/seq/audit_packet_blinding.py``, which attacks
the published records and the delivered files, and never opens a key.

Disclosed limitation of a paired packet: it necessarily contains three images of
the same prompt and generation seed. An annotator may notice the visual
similarity and infer that items belong together. Arm identity is still hidden,
the order is shuffled, and nothing identifies WHICH of a similar group came from
which arm. This is a property of the approved design, recorded rather than
concealed.

Relationship to the 228-item supplementary packet
-------------------------------------------------
The previously delivered 228-item packet (168 independently stratified items
over seven categories + 60 enriched diagnostics) is PRESERVED as a separate
supplementary audit. The two packets are not pooled and not mixed: an error rate
from the enriched set is biased upward by construction, and this packet's own
sample is paired rather than category-stratified. Overlap between them is
reported -- outside the packet, because overlap reveals set membership -- so no
human work is repeated and any label already entered is carried forward rather
than lost.

CPU only. Generates nothing: every item is an image that already exists.

    python scripts/seq/build_paired_annotation_packet.py \
        --eval_root <diag>/eval_test \
        --out_root <diag>/annotation_paired_180 \
        --repo_out results/audit_v1/annotation_packet_paired180
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import random
import secrets
import shutil
from pathlib import Path

CATEGORIES = ("cat", "dog", "bird")
ARMS = ("MA", "MAB_L2", "U")          # U is resolved to U_step<selected>
TUPLES_PER_CELL = 10
# Columns that must never appear inside the packet directory.
FORBIDDEN_IN_PACKET = ("MAB_L2", "_MA/", "U_step", "max_score", "detections",
                       "seed17", "seed29", "training_seed", "arm")
# The secret salt must never reach the packet either; checked separately,
# because its value is not a literal known at import time.


def export_image(src: Path, dst: Path) -> dict:
    """Write an annotation copy whose BYTES differ but whose PIXELS do not.

    Copying source bytes is a de-blinding channel: the published image reports
    and detector rows associate each image's sha256 with its source slot, and
    those digests are committed, so a packet holder can hash a renamed file and
    read off the arm. The salt and the secret row order do nothing about it --
    the file content itself is the identifier.

    So the annotation copy is re-encoded from the DECODED pixels into a fresh
    lossless PNG, carrying no EXIF, no JFIF block and no PNG text chunk. The
    decoded pixels are then compared element-by-element against the source's
    decoded pixels, so "what the annotator sees is unchanged" is verified rather
    than asserted. Returns the verification record; raises if pixels moved.
    """
    from PIL import Image
    import numpy as np

    with Image.open(src) as im:
        src_mode, src_size, src_fmt = im.mode, im.size, im.format
        a = np.array(im.convert("RGB"))
    # A fresh image from the array: nothing from im.info travels with it.
    Image.fromarray(a, mode="RGB").save(dst, format="PNG", optimize=True)
    with Image.open(dst) as im2:
        b = np.array(im2.convert("RGB"))
        leftover = dict(im2.info)
        n_exif = len(im2.getexif())
    if a.shape != b.shape or not np.array_equal(a, b):
        raise SystemExit(f"FATAL: re-encoding changed pixels for {src.name}")
    # Only harmless rendering hints may survive; anything else is a leak risk.
    stray = {k: v for k, v in leftover.items()
             if k not in ("dpi", "gamma", "aspect", "srgb", "icc_profile")}
    if stray or n_exif:
        raise SystemExit(f"FATAL: metadata survived re-encoding for "
                         f"{src.name}: {sorted(stray)} exif={n_exif}")
    return {"source_format": src_fmt, "source_mode": src_mode,
            "source_size": list(src_size), "pixels_identical": True,
            "exported_format": "PNG", "exif_entries": n_exif}


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def slot_images(slot: Path) -> dict[tuple[str, int], dict]:
    """Per-image GENERATION records for one slot, keyed by (prompt_id, gen_seed).

    Reads image_report.json ONLY. detections.jsonl is deliberately not opened.
    """
    rep = slot / "image_report.json"
    if not rep.is_file():
        raise SystemExit(f"FATAL: {rep} absent")
    j = json.loads(rep.read_text())
    if not j.get("complete"):
        raise SystemExit(f"FATAL: {rep} is not marked complete")
    out = {}
    for im in j.get("images") or []:
        out[(im["prompt_id"], int(im["gen_seed"]))] = im
    return out


def resolve_slots(eval_root: Path, seed: int) -> dict[str, Path]:
    """The three arms for one training seed, by directory name."""
    found: dict[str, Path] = {}
    for d in sorted(eval_root.glob(f"seed{seed}_*")):
        if not d.is_dir():
            continue
        tail = d.name.split("_", 1)[1]
        if tail == "MA":
            found["MA"] = d
        elif tail == "MAB_L2":
            found["MAB_L2"] = d
        elif tail.startswith("U_step"):
            found["U"] = d
    missing = [a for a in ARMS if a not in found]
    if missing:
        raise SystemExit(f"FATAL: seed {seed} is missing arm(s) {missing} under "
                         f"{eval_root}; a paired packet needs all three")
    return found


def read_prior_labels(sheets: list[Path]) -> dict[str, dict]:
    """Any label already entered in an earlier packet, keyed by image sha256.

    Earlier packets carry blinded item ids, not image paths, so the join runs
    through that packet's KEY (which the coordinator holds). Where no key is
    available the sheet is reported as unjoinable rather than silently skipped.
    """
    out: dict[str, dict] = {}
    for sheet in sheets:
        if not sheet.is_file():
            continue
        key = None
        for cand in (sheet.parent.parent / "KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv",
                     sheet.parent.parent / "key"
                     / "KEY_do_not_open_before_annotating.csv"):
            if cand.is_file():
                key = cand
                break
        if key is None:
            continue
        by_item = {}
        with key.open(newline="") as fh:
            for row in csv.DictReader(fh):
                by_item[row["item_id"]] = row.get("image_sha256")
        with sheet.open(newline="") as fh:
            for row in csv.DictReader(fh):
                ans = (row.get("answer_yes_no_unsure")
                       or row.get("target_present__yes_no_unsure") or "").strip()
                notes = (row.get("notes") or "").strip()
                if not ans and not notes:
                    continue
                sha = by_item.get(row["item_id"])
                if sha:
                    out[sha] = {"source_packet": str(sheet.parent),
                                "source_item_id": row["item_id"],
                                "answer": ans, "notes": notes}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--manifest",
                    default="results/audit_v1/draft_manifests/test_manifest_DRAFT.json",
                    help="the frozen TEST manifest: the tuple universe")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--repo_out", default=None)
    ap.add_argument("--tuples_per_cell", type=int, default=TUPLES_PER_CELL)
    ap.add_argument("--sampling_seed", type=int, default=20261012,
                    help="declared BEFORE any tuple is drawn; the draw reads "
                         "only generation-side records, never detector output")
    ap.add_argument("--salt", default=None,
                    help="blinding salt. Omitted: a fresh 32-byte secret is "
                         "generated and stored WITH THE KEY, outside the "
                         "packet, and only its sha256 is published. Pass the "
                         "stored secret to reproduce an existing packet.")
    ap.add_argument("--compare_packets", default="",
                    help="space-separated label_sheet.csv paths of previously "
                         "delivered packets, for the overlap report")
    ap.add_argument("--copy_images", action="store_true", default=True)
    a = ap.parse_args()

    eval_root = Path(a.eval_root)
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    man = json.loads(Path(a.manifest).read_text())

    # ---- the blinding secret, resolved BEFORE anything is drawn ----------
    #
    # TWO channels can de-blind a packet, and both must be closed.
    #
    #  (1) THE IDENTIFIER. Blinded ids are sha256(salt | image_path)[:12] and
    #      image paths are deterministic --
    #      <eval_root>/seed<S>_<arm>/images/<prompt_id>_seed<gs>.jpg -- so a
    #      PUBLISHED salt makes the whole id space enumerable.
    #
    #  (2) THE ROW ORDER. If one RNG both draws the sample and shuffles the
    #      rows, and its seed is published, the ordered item list -- arm
    #      included -- replays from published data and position alone de-blinds
    #      every row. A secret salt does nothing about this. This is not
    #      hypothetical: scripts/seq/audit_packet_blinding.py reconstructed
    #      180/180 rows of the first build of this packet that way.
    #
    # So: the SAMPLING seed stays PUBLIC, because a reader must be able to
    # replay the draw and confirm it used no detector input -- tuple MEMBERSHIP
    # is public by design. The ORDER seed is derived from the secret and is not
    # published, so the ASSIGNMENT of arm to row is what stays hidden.
    out_root_early = Path(a.out_root)
    out_root_early.mkdir(parents=True, exist_ok=True)
    salt_p = out_root_early / "BLINDING_SALT_SECRET.txt"
    if a.salt:
        salt = a.salt
        salt_source = "supplied on the command line (reproducing an existing packet)"
    elif salt_p.is_file():
        salt = salt_p.read_text().strip()
        salt_source = f"read from the stored secret at {salt_p}"
    else:
        salt = secrets.token_hex(32)
        salt_p.write_text(salt + "\n")
        salt_p.chmod(0o600)
        salt_source = f"freshly generated and stored at {salt_p}"
    salt_sha = hashlib.sha256(salt.encode()).hexdigest()
    order_seed = hashlib.sha256(f"{salt}|row-order".encode()).hexdigest()
    order_sha = hashlib.sha256(order_seed.encode()).hexdigest()

    # ---- the tuple universe, from the FROZEN MANIFEST ---------------------
    universe: dict[tuple[int, str], list[tuple[str, int]]] = {}
    by_tuple: dict[tuple[str, int], dict] = {}
    for r in man["records"]:
        if r["category"] not in CATEGORIES:
            continue
        by_tuple[(r["prompt_id"], int(r["gen_seed"]))] = r
    for s in seeds:
        for cat in CATEGORIES:
            pool = sorted(k for k, r in by_tuple.items() if r["category"] == cat)
            universe[(s, cat)] = pool

    # ---- the per-slot generation records ----------------------------------
    slots: dict[int, dict[str, Path]] = {s: resolve_slots(eval_root, s)
                                         for s in seeds}
    images: dict[tuple[int, str], dict] = {}
    for s in seeds:
        for arm, d in slots[s].items():
            images[(s, arm)] = slot_images(d)

    # ---- the draw ---------------------------------------------------------
    rng = random.Random(a.sampling_seed)
    picked: dict[str, list] = {}
    items: list[dict] = []
    for s in seeds:
        for cat in CATEGORIES:
            pool = [k for k in universe[(s, cat)]
                    if all(k in images[(s, arm)] for arm in ARMS)]
            dropped = len(universe[(s, cat)]) - len(pool)
            if len(pool) < a.tuples_per_cell:
                raise SystemExit(
                    f"FATAL: seed {s}/{cat} has only {len(pool)} tuples present "
                    f"in all three arms; {a.tuples_per_cell} required")
            chosen = sorted(rng.sample(pool, a.tuples_per_cell))
            fams = [by_tuple[k]["prompt_family"] for k in chosen]
            picked[f"{s}|{cat}"] = {
                "tuple_universe": len(universe[(s, cat)]),
                "available_in_all_three_arms": len(pool),
                "dropped_not_in_all_arms": dropped,
                "drawn": a.tuples_per_cell,
                "realised_family_split": {f: fams.count(f)
                                          for f in sorted(set(fams))},
                "distinct_prompt_ids": len({k[0] for k in chosen}),
                "distinct_gen_seeds": sorted({k[1] for k in chosen}),
                "tuples": [{"prompt_id": k[0], "gen_seed": k[1]} for k in chosen],
            }
            for k in chosen:
                for arm in ARMS:
                    im = images[(s, arm)][k]
                    items.append({
                        "training_seed": s, "arm": arm,
                        "slot": slots[s][arm].name,
                        "tuple_id": f"s{s}|{k[0]}|{k[1]}",
                        "prompt_id": k[0], "gen_seed": k[1],
                        "category": cat,
                        "prompt_family": by_tuple[k]["prompt_family"],
                        "prompt_index": by_tuple[k]["prompt_index"],
                        "image_path": im["image_path"],
                        "image_sha256": im["sha256"],
                    })
    expected = len(seeds) * len(CATEGORIES) * a.tuples_per_cell * len(ARMS)
    if len(items) != expected:
        raise SystemExit(f"FATAL: built {len(items)} items, expected {expected}")
    # A SEPARATE RNG, seeded from the secret: see the note above. Reusing `rng`
    # here would publish the row order along with the sampling seed.
    random.Random(order_seed).shuffle(items)

    # ---- prior labels, so no human work is repeated or lost ---------------
    prior_sheets = [Path(x) for x in a.compare_packets.split() if x]
    prior = read_prior_labels(prior_sheets)
    prior_images = set()
    for sheet in prior_sheets:
        key = sheet.parent.parent / "KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv"
        alt = sheet.parent.parent / "key" / "KEY_do_not_open_before_annotating.csv"
        kp = key if key.is_file() else (alt if alt.is_file() else None)
        if kp is None:
            continue
        with kp.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("image_sha256"):
                    prior_images.add(row["image_sha256"])

    # ---- write the packet -------------------------------------------------
    out_root = Path(a.out_root)
    packet = out_root / "packet"
    (packet / "images").mkdir(parents=True, exist_ok=True)

    sheet_rows, key_rows, overlap_rows = [], [], []
    reencoded = pixel_verified = 0
    src_formats: set = set()
    exported_digests: set = set()
    for r in items:
        blind = "item_" + hashlib.sha256(
            f"{salt}|{r['image_path']}".encode()).hexdigest()[:12]
        src = Path(r["image_path"])
        # NOT the source extension and NOT the source bytes: see export_image.
        ext = ".png"
        dst = packet / "images" / f"{blind}{ext}"
        if a.copy_images and src.is_file() and not dst.exists():
            checks = export_image(src, dst)
            reencoded += 1
            pixel_verified += int(checks["pixels_identical"])
            src_formats.add(checks["source_format"])
        exported_digests.add(sha256_file(dst) if dst.is_file() else "")
        sheet_rows.append({
            "item_id": blind,
            "image": f"images/{blind}{ext}",
            "category_to_judge": r["category"],
            "question": (f"Is a clearly recognisable {r['category']} present "
                         f"in this image?"),
            "answer_yes_no_unsure": "",       # left EMPTY, deliberately
            "notes": "",
        })
        key_rows.append({
            "item_id": blind, "tuple_id": r["tuple_id"],
            "training_seed": r["training_seed"], "arm": r["arm"],
            "slot": r["slot"], "category": r["category"],
            "prompt_family": r["prompt_family"],
            "prompt_index": r["prompt_index"],
            "prompt_id": r["prompt_id"], "gen_seed": r["gen_seed"],
            "image_path": r["image_path"], "image_sha256": r["image_sha256"],
        })
        if r["image_sha256"] in prior_images:
            p = prior.get(r["image_sha256"], {})
            overlap_rows.append({
                "item_id": blind, "image_sha256": r["image_sha256"],
                "also_in_packet": p.get("source_packet", "(earlier packet)"),
                "earlier_item_id": p.get("source_item_id", ""),
                "earlier_answer": p.get("answer", ""),
                "earlier_notes": p.get("notes", ""),
                "already_labelled": bool(p.get("answer") or p.get("notes")),
            })

    with (packet / "label_sheet.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet_rows[0].keys()))
        w.writeheader()
        w.writerows(sheet_rows)

    (packet / "INSTRUCTIONS.md").write_text(
        "# Blinded image audit\n\n"
        "Judge each image on its own. For every row in `label_sheet.csv`, open\n"
        "the image named in the `image` column and answer the question in\n"
        "`question` by writing `yes`, `no` or `unsure` in\n"
        "`answer_yes_no_unsure`. Use `notes` for anything ambiguous.\n\n"
        "Judge **only** the category named in `category_to_judge`. Whether the\n"
        "image is beautiful, odd or damaged is not the question; record that in\n"
        "`notes` instead.\n\n"
        "You are not told which model produced an image, which condition it\n"
        "belongs to, or what an automatic detector thought. That is deliberate:\n"
        "your labels are the reference the detector numbers are checked\n"
        "against, so they must be formed independently.\n\n"
        "Some images will look similar to each other. Judge each one on its\n"
        "own and do not try to group them.\n\n"
        "Do not look for the key. It is stored outside this directory.\n")

    # The key and the overlap report live OUTSIDE the packet: the key reveals
    # arm and seed, and the overlap report reveals set membership.
    keyp = out_root / "KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv"
    with keyp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    overlapp = out_root / "OVERLAP_coordinator_only.csv"
    fields = ["item_id", "image_sha256", "also_in_packet", "earlier_item_id",
              "earlier_answer", "earlier_notes", "already_labelled"]
    with overlapp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(overlap_rows)

    # ---- blinding probe: assert, do not assume ----------------------------
    leaks = []
    for p in sorted(packet.rglob("*")):
        if p.is_dir() or p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            continue
        txt = p.read_text(errors="ignore")
        for bad in FORBIDDEN_IN_PACKET:
            if bad in txt:
                leaks.append(f"{p.relative_to(packet)}: {bad!r}")
    if leaks:
        raise SystemExit("FATAL: the packet leaks condition information: "
                         + "; ".join(leaks))
    for p_ in sorted(packet.rglob("*")):
        if p_.is_file() and p_.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            if salt in p_.read_text(errors="ignore"):
                raise SystemExit(f"FATAL: the blinding salt appears in "
                                 f"{p_.relative_to(packet)}")
    for leaked in (keyp.name, overlapp.name, salt_p.name):
        if (packet / leaked).exists():
            raise SystemExit(f"FATAL: {leaked} is inside the packet")
    labelled = sum(1 for r in sheet_rows if r["answer_yes_no_unsure"])
    if labelled:
        raise SystemExit(f"FATAL: {labelled} label cells are not empty")
    # The source digests are PUBLISHED and map to a slot. No exported byte
    # stream may equal one of them; checked here so the build cannot ship the
    # channel again.
    source_digests = {r["image_sha256"] for r in items if r.get("image_sha256")}
    collide = exported_digests & source_digests
    if collide:
        raise SystemExit(f"FATAL: {len(collide)} exported file(s) hash to a "
                         f"PUBLISHED source digest; the arm is recoverable by "
                         f"hashing the packet")
    if a.copy_images and reencoded and pixel_verified != reencoded:
        raise SystemExit(f"FATAL: pixels verified for only {pixel_verified} of "
                         f"{reencoded} exported images")

    meta = {
        "_what_this_is": (
            "The PRESCRIBED paired human-audit packet, built to the approved "
            "audit: ten matched (prompt, generation-seed) tuples for each "
            "training seed and each of cat/dog/bird, all three arms per tuple. "
            "Nothing was generated. Labels are EMPTY."),
        "approved_design": {
            "training_seeds": seeds,
            "categories": list(CATEGORIES),
            "tuples_per_training_seed_and_category": a.tuples_per_cell,
            "arms_per_tuple": list(ARMS),
            "n_items": len(items),
            "arithmetic": (f"{len(seeds)} seeds x {len(CATEGORIES)} categories x "
                           f"{a.tuples_per_cell} tuples x {len(ARMS)} arms = "
                           f"{len(items)}"),
            "why_paired": (
                "the same prompt and generation seed is judged under all three "
                "arms, so a human can see whether a detector verdict change "
                "BETWEEN arms is a real change in the image. An independently "
                "stratified sample cannot answer that question, because it does "
                "not hold the tuple fixed."),
        },
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eval_root": str(eval_root),
        "slots": {str(s): {k: v.name for k, v in slots[s].items()} for s in seeds},
        "tuple_universe_source": {
            "manifest": str(a.manifest),
            "manifest_sha256_as_stored": man.get("manifest_sha256"),
            "note": "the frozen TEST manifest supplies the tuple universe",
        },
        "sampling": {
            "rng_seed": a.sampling_seed,
            "declared": ("fixed as a CLI argument and recorded here; the draw "
                         "is uniform without replacement over the tuples "
                         "present in all three arms"),
            "detector_independence": (
                "STRUCTURAL: this script opens only each slot's "
                "image_report.json (the generation-side record) and the frozen "
                "test manifest. detections.jsonl is never read, so no detector "
                "score, verdict or threshold can enter the draw."),
            "per_cell": picked,
        },
        "blinding": {
            "hidden_from_annotator": ["arm", "training seed", "detector score",
                                      "detector verdict", "tuple grouping"],
            "shown_to_annotator": ["the category to judge"],
            "order": ("shuffled once under a SEPARATE RNG seeded from the "
                      "secret; the sampling seed does NOT reproduce the row "
                      "order"),
            "order_seed_sha256": order_sha,
            "what_is_public_by_design": (
                "tuple MEMBERSHIP. The sampling seed is published so a reader "
                "can replay the draw and confirm it used no detector input; "
                "that replay reveals WHICH 60 tuples were drawn."),
            "what_is_protected": (
                "the ASSIGNMENT of arm and training seed to a sheet row. Both "
                "channels that could leak it are closed: the identifier (secret "
                "salt) and the row order (separate secret-seeded RNG). Verify "
                "with scripts/seq/audit_packet_blinding.py, which attacks the "
                "published records and opens no key."),
            "blinded_id_rule": "sha256(<secret salt> | <image_path>)[:12]",
            "salt_is_secret": (
                "The salt is NOT published. Image paths are deterministic, so a "
                "published salt would make the entire id space enumerable and "
                "the blind recoverable by anyone who can read this manifest. "
                "Only the salt's digest appears here."),
            "salt_sha256": salt_sha,
            "salt_location": str(salt_p),
            "salt_source": salt_source,
            "key_location": str(keyp),
            "key_inside_packet": False,
            "probe": ("the packet's text files were scanned for arm, seed and "
                      "detector strings; no match, or the build aborts"),
            "image_export": {
                "why": ("copying source bytes is a de-blinding channel: the "
                        "committed image reports and detector rows associate "
                        "each image's sha256 with its source slot, so a packet "
                        "holder could hash a renamed file and read off the arm. "
                        "Neither the salt nor the secret row order affects "
                        "this -- the file content is the identifier."),
                "method": ("each annotation copy is re-encoded from the DECODED "
                           "pixels into a fresh lossless PNG with no EXIF, no "
                           "JFIF block and no PNG text chunk"),
                "pixels_verified": (
                    f"{pixel_verified}/{reencoded} exported images compared "
                    f"element-by-element against the source's decoded pixels; "
                    f"a mismatch aborts the build"),
                "source_formats": sorted(src_formats),
                "no_exported_file_matches_a_published_digest": True,
                "closed_against": (
                    "a packet holder who also has the repository: no exported "
                    "byte stream equals any published image digest, and no "
                    "frozen-test pixels are committed anywhere"),
                "NOT_closed_against": (
                    "a holder of the PRIVATE source images under the data root. "
                    "The packet is made of those images, so pixel-for-pixel "
                    "comparison still identifies every item. Re-encoding cannot "
                    "close that and does not claim to; it is mitigated by "
                    "access control alone."),
            },
            "disclosed_limitation": (
                "a paired packet necessarily contains three images of the same "
                "prompt and generation seed. An annotator may notice the "
                "similarity and infer that items belong together; nothing "
                "identifies WHICH of a similar group came from which arm. This "
                "is a property of the approved design and is recorded, not "
                "concealed."),
        },
        "relationship_to_other_packets": {
            "supplementary_packet_preserved": (
                "the previously delivered 228-item packet (168 independently "
                "stratified items over seven categories + 60 enriched "
                "diagnostics) is retained unchanged as a SEPARATE supplementary "
                "audit"),
            "must_not_be_pooled": (
                "the enriched subset is biased upward by construction and this "
                "packet's sample is paired rather than category-stratified; an "
                "error rate computed across them is not an error rate"),
            "compared_against": [str(p) for p in prior_sheets],
            "n_overlapping_images": len(overlap_rows),
            "n_overlapping_already_labelled":
                sum(1 for r in overlap_rows if r["already_labelled"]),
            "overlap_report": str(overlapp),
            "overlap_report_is_outside_the_packet_because":
                "overlap reveals set membership, which the annotator must not see",
            "prior_labels_preserved": (
                "any answer already entered in an earlier packet is carried into "
                "the overlap report with its source item id, so no human work is "
                "lost. The new packet's own cells stay EMPTY: pre-filling them "
                "would import another packet's sampling design into this one."),
        },
        "label_state": "EMPTY -- no labels are supplied or implied",
        "provisional": ("Detector results stay PROVISIONAL until this audit is "
                        "complete. No labels exist yet."),
    }
    (out_root / "packet_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    if a.repo_out:
        rp = Path(a.repo_out)
        rp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packet / "label_sheet.csv", rp / "label_sheet_EMPTY.csv")
        shutil.copy2(packet / "INSTRUCTIONS.md", rp / "INSTRUCTIONS.md")
        # The drawn tuples ARE published. Withholding them would be security
        # by omission, not security: the published sampling seed replays the
        # draw anyway, and membership is meant to be auditable. What protects
        # the blind is the secret salt and the secret row order, not secrecy
        # about which tuples were chosen.
        shutil.copy2(out_root / "packet_manifest.json", rp / "packet_manifest.json")
        (rp / "overlap_summary.json").write_text(json.dumps({
            "_what_this_is": ("Aggregate overlap between this paired packet and "
                              "the previously delivered supplementary packet. "
                              "Counts only: the item-level mapping and any "
                              "prior labels stay beside the key, outside the "
                              "repository and outside the packet."),
            "compared_against": [str(p) for p in prior_sheets],
            "n_items_this_packet": len(items),
            "n_overlapping_images": len(overlap_rows),
            "n_overlapping_already_labelled":
                sum(1 for r in overlap_rows if r["already_labelled"]),
            "item_level_mapping": str(overlapp),
        }, indent=2) + "\n")
        print(f"[repo copy] key-free packet metadata -> {rp}")

    print("=== prescribed paired audit packet ===")
    print(f"  items            : {len(items)}  "
          f"({len(seeds)}x{len(CATEGORIES)}x{a.tuples_per_cell}x{len(ARMS)})")
    for k, v in picked.items():
        print(f"  {k:<10} {v['drawn']} tuples from {v['available_in_all_three_arms']}"
              f"  families {v['realised_family_split']}  "
              f"distinct prompts {v['distinct_prompt_ids']}")
    print(f"  sampling seed    : {a.sampling_seed} (no detector input)")
    print(f"  packet           : {packet}")
    print(f"  key              : {keyp}  (outside the packet)")
    print(f"  blinding salt    : {salt_p}  (SECRET, outside the packet; "
          f"digest {salt_sha[:12]}…)")
    print(f"  row order        : separate RNG seeded from the secret "
          f"(digest {order_sha[:12]}…); the sampling seed does NOT reproduce it")
    print(f"  image export     : {reencoded} re-encoded to PNG from decoded "
          f"pixels, {pixel_verified} pixel-verified, 0 matching a published "
          f"digest")
    print(f"  overlap report   : {overlapp}  (outside the packet)")
    print(f"  overlapping imgs : {len(overlap_rows)} "
          f"({sum(1 for r in overlap_rows if r['already_labelled'])} already "
          f"labelled)")
    print(f"  labels           : EMPTY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
