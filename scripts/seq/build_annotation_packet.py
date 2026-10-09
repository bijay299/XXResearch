#!/usr/bin/env python3
"""Build a blinded human-annotation packet from images that already exist.

Generates nothing: every item is an existing evaluation image, copied under a
blinded name. CPU only, no model, no GPU.

Two sets, kept separate on purpose
----------------------------------
SET R (random)      A reproducible stratified random sample across every
                    (training seed, checkpoint, category, prompt family) cell.
                    This is the ONLY set from which an overall detector error
                    rate may be estimated, and then only for the strata it
                    covers, with the per-stratum sampling weights applied.

SET D (diagnostic)  Deliberately ENRICHED for cases where the detector is
                    least trustworthy: scores between the 0.3 and 0.7
                    thresholds, and parent->child verdict flips at the same
                    prompt and generation seed. It is a magnifier for failure
                    modes. Error rates computed on SET D are biased upward by
                    construction and MUST NOT be reported as overall accuracy.

What is blinded
---------------
Hidden from the annotator: which checkpoint produced the image, which training
seed, the detector's score and verdict, and which set the item belongs to.
Shown to the annotator: the category to judge -- without it there is no
question to answer. Item order is shuffled once under a fixed RNG seed, so the
packet is reproducible but carries no condition ordering.

The key mapping blinded ids back to conditions is written OUTSIDE the packet
directory so it cannot be read by accident while annotating.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ANNOTATED = ["cat", "bird", "dog", "sandwich"]
CKS = ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]
FAMILIES = ["literal", "paraphrase"]


def load_all(roots: dict[str, Path]) -> list[dict]:
    """Every prediction row, tagged with its training seed.

    M0 is byte-identical between the two seed tables (eval_seed29/M0 is a
    symlink), so it is loaded ONCE and tagged 'shared' rather than being
    duplicated into both seeds. Sampling it twice would put the same image in
    the packet twice and double-weight it in any error estimate.
    """
    rows: list[dict] = []
    seen_m0 = False
    for sn, root in roots.items():
        for ck in CKS:
            p = root / ck / "detections.jsonl"
            if not p.exists():
                continue
            if ck == "M0":
                if seen_m0:
                    continue
                seen_m0 = True
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["_seed"] = "shared" if ck == "M0" else sn
                r["_ck"] = ck
                rows.append(r)
    return rows


def blind_id(r: dict, salt: str) -> str:
    h = hashlib.sha256(
        (salt + "|" + r["_seed"] + "|" + r["_ck"] + "|" + r["image_sha256"]).encode())
    return "I" + h.hexdigest()[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--out_root", default="/data/bijaypandey/cuig_pilot/seq_pilot/annotation_v1")
    ap.add_argument("--repo_out", default="results/audit_v1/annotation_packet")
    ap.add_argument("--per_cell", type=int, default=2,
                    help="random-sample items per (seed, checkpoint, category, family) cell")
    ap.add_argument("--n_diagnostic", type=int, default=60)
    ap.add_argument("--rng_seed", type=int, default=20261009)
    ap.add_argument("--salt", default="AUDIT-01")
    ap.add_argument("--copy_images", action="store_true",
                    help="copy image files into the packet (default: manifest only)")
    args = ap.parse_args()

    seq = Path(args.seq_root)
    roots = {"seed17": seq / "eval", "seed29": seq / "eval_seed29"}
    missing = [str(p) for p in roots.values() if not p.exists()]
    if missing:
        raise SystemExit("missing evaluation roots: " + ", ".join(missing))

    rows = [r for r in load_all(roots) if r["category"] in ANNOTATED]
    print(f"candidate rows in the four annotated categories: {len(rows)}")

    # Absent images are recorded by exact path, never substituted or invented.
    absent = [r["image_path"] for r in rows if not Path(r["image_path"]).exists()]

    rng = random.Random(args.rng_seed)

    # ---- SET R: stratified random sample ----------------------------------
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["_seed"], r["_ck"], r["category"], r["prompt_family"])].append(r)
    set_r: list[dict] = []
    weights: dict[str, dict] = {}
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda x: x["image_sha256"])
        take = min(args.per_cell, len(pool))
        picked = rng.sample(pool, take)
        for r in picked:
            r["_set"] = "R"
            set_r.append(r)
        weights["|".join(key)] = {"cell_size": len(pool), "sampled": take,
                                  "sampling_weight": round(len(pool) / take, 4)}
    print(f"SET R: {len(set_r)} items over {len(cells)} strata")

    # ---- SET D: enriched diagnostic set -----------------------------------
    chosen = {id(r) for r in set_r}
    diag: list[dict] = []

    # (a) scores strictly between the lowest and highest reported thresholds:
    #     the verdict changes with the threshold, so the detector is undecided.
    for r in rows:
        if id(r) in chosen:
            continue
        if r["hit"]["0.3"] and not r["hit"]["0.7"]:
            r["_diag_reason"] = "threshold_boundary_hit_at_0.3_not_0.7"
            diag.append(r)

    # (b) parent -> child verdict flips at the SAME prompt and generation seed.
    by_ident: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for r in rows:
        by_ident[(r["category"], r["prompt_id"], r["gen_seed"])][(r["_seed"], r["_ck"])] = r
    for ident, per_ck in by_ident.items():
        for sn in ("seed17", "seed29"):
            parent = per_ck.get((sn, "MA"))
            if not parent:
                continue
            for ck in ("MAB", "MAB_L2", "MAC", "MAC_L2"):
                child = per_ck.get((sn, ck))
                if not child:
                    continue
                if parent["hit"]["0.5"] != child["hit"]["0.5"]:
                    for r, role in ((parent, "parent"), (child, "child")):
                        if id(r) in chosen or any(id(r) == id(d) for d in diag):
                            continue
                        r["_diag_reason"] = f"MA_to_{ck}_verdict_flip_at_0.5_{role}_side"
                        diag.append(r)

    uniq: dict[int, dict] = {}
    for r in diag:
        uniq.setdefault(id(r), r)
    diag = sorted(uniq.values(), key=lambda x: x["image_sha256"])
    n_avail = len(diag)
    if len(diag) > args.n_diagnostic:
        diag = rng.sample(diag, args.n_diagnostic)
    for r in diag:
        r["_set"] = "D"
    print(f"SET D: {len(diag)} items sampled from {n_avail} eligible")

    # ---- blind, shuffle, write -------------------------------------------
    items = set_r + diag
    for r in items:
        r["_id"] = blind_id(r, args.salt)
    seen: dict[str, int] = defaultdict(int)
    for r in items:
        seen[r["_id"]] += 1
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        raise SystemExit(f"blinded id collision ({len(dups)}); change --salt")
    rng.shuffle(items)

    out = Path(args.out_root)
    pack = out / "packet"
    pack.mkdir(parents=True, exist_ok=True)
    (out / "key").mkdir(parents=True, exist_ok=True)

    n_copied = 0
    if args.copy_images:
        (pack / "images").mkdir(exist_ok=True)
        for r in items:
            src = Path(r["image_path"])
            if src.exists():
                shutil.copy2(src, pack / "images" / f"{r['_id']}.jpg")
                n_copied += 1

    # The sheet the annotator fills in. Presence, ambiguity and visible quality
    # or context concerns are three separate columns: a quality concern is not
    # evidence about presence, and must not be recorded as if it were.
    sheet = pack / "label_sheet_EMPTY.csv"
    with sheet.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "item_id", "image_file", "category_to_judge",
            "target_present__yes_no_unsure",
            "ambiguity__clear_ambiguous_undecidable",
            "quality_concern__none_minor_severe",
            "context_concern__none_describe_in_notes",
            "notes",
        ])
        for r in items:
            w.writerow([r["_id"], f"images/{r['_id']}.jpg", r["category"], "", "", "", "", ""])

    # The key: blinded id -> condition and detector verdict. Written outside the
    # packet directory so it is not opened by accident during annotation.
    with (out / "key" / "KEY_do_not_open_before_annotating.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "set", "diagnostic_reason", "training_seed", "checkpoint",
                    "category", "prompt_id", "prompt_family", "gen_seed",
                    "max_score_target", "det_hit_0.3", "det_hit_0.5", "det_hit_0.7",
                    "image_sha256", "source_image_path"])
        for r in sorted(items, key=lambda x: x["_id"]):
            w.writerow([r["_id"], r["_set"], r.get("_diag_reason", ""), r["_seed"],
                        r["_ck"], r["category"], r["prompt_id"], r["prompt_family"],
                        r["gen_seed"], f"{r['max_score_target']:.4f}",
                        int(r["hit"]["0.3"]), int(r["hit"]["0.5"]), int(r["hit"]["0.7"]),
                        r["image_sha256"], r["image_path"]])

    manifest = {
        "packet": "AUDIT-01 blinded annotation packet v1",
        "built_from": "existing evaluation images only; nothing was generated",
        "n_items": len(items),
        "n_set_R_random": len(set_r),
        "n_set_D_diagnostic": len(diag),
        "n_set_D_eligible_total": n_avail,
        "rng_seed": args.rng_seed,
        "blinding_salt": args.salt,
        "per_cell_random": args.per_cell,
        "categories_annotated": ANNOTATED,
        "images_copied_into_packet": n_copied if args.copy_images else 0,
        "images_absent_on_disk": absent,
        "n_images_absent_on_disk": len(absent),
        "packet_dir": str(pack),
        "key_path": str(out / "key" / "KEY_do_not_open_before_annotating.csv"),
        "set_R_sampling_weights": weights,
        "blinded_from_annotator": [
            "checkpoint identity", "training seed", "detector score",
            "detector verdict", "set membership (R or D)"],
        "shown_to_annotator": ["the category to judge", "the image"],
        "human_annotation_status": "NOT STARTED — the label sheet is empty by design",
        "estimation_rules": {
            "set_R": ("the only set usable for an overall detector error rate, and "
                      "only over the strata it covers, with set_R_sampling_weights "
                      "applied; M0 appears once because it is shared between the "
                      "two seed tables"),
            "set_D": ("ENRICHED for threshold-boundary and verdict-flip cases. Error "
                      "rates on SET D are biased upward by construction and must "
                      "never be reported as overall detector accuracy. Use it to "
                      "characterise failure modes only."),
            "pooling": "SET R and SET D must not be pooled into one error rate",
        },
    }
    (pack / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    readme = f"""# Blinded annotation packet — AUDIT-01 v1

{len(items)} items built **only** from evaluation images that already existed.
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
   `{out / 'key' / 'KEY_do_not_open_before_annotating.csv'}`

## What is hidden and what is shown

Hidden: which checkpoint and training seed produced the image, the detector's
score and verdict, and whether the item is from the random or the diagnostic
set. Shown: the image and the category to judge — without the category there
is no question to answer.

## Two sets, which must not be pooled

| set | n | what it is | what it may be used for |
|---|---|---|---|
| R (random) | {len(set_r)} | stratified random sample over {len(cells)} (seed, checkpoint, category, family) cells, `rng_seed={args.rng_seed}` | the only set for an overall detector error rate, over its strata, with the weights in `MANIFEST.json` |
| D (diagnostic) | {len(diag)} of {n_avail} eligible | **enriched** for threshold-boundary cases (hit at 0.3 but not 0.7) and parent→child verdict flips | characterising failure modes only |

**SET D is biased upward by construction.** It over-samples exactly the images
the detector finds hardest. An error rate computed on SET D, or on SET R and
SET D pooled, is not an overall error rate and must not be reported as one.

Items are shuffled once under the fixed RNG seed, so the packet is reproducible
and carries no condition ordering. Blinded ids are
`sha256(salt | seed | checkpoint | image_sha256)[:10]`, salt `{args.salt}`.

## Reproducing this packet

    python scripts/seq/build_annotation_packet.py --copy_images

Images are **not** committed to the repository. Only this README, the empty
sheet and the manifest are. The packet with images lives at `{pack}`.
"""
    (pack / "README.md").write_text(readme)

    # Small, commitable copies: sheet, manifest, README. Never the images, never the key.
    repo = Path(args.repo_out)
    repo.mkdir(parents=True, exist_ok=True)
    for f in ("label_sheet_EMPTY.csv", "MANIFEST.json", "README.md"):
        shutil.copy2(pack / f, repo / f)

    print(f"\npacket : {pack}")
    print(f"key    : {out / 'key' / 'KEY_do_not_open_before_annotating.csv'} (outside the packet)")
    print(f"repo   : {repo} (sheet + manifest + README only)")
    if absent:
        print(f"\nWARNING: {len(absent)} source images absent on disk; "
              f"exact paths recorded in MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
