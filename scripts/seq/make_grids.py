#!/usr/bin/env python3
"""Paired image grids across checkpoints, plus blinded grids for annotation.

Selection is by a FIXED RULE, decided before looking at any image: prompt index 0
of each family, generation seeds 101 and 202. Nothing is picked for being
dramatic. Retained-object categories are included alongside the deletion targets.

Two outputs per category:
  * labelled grid  - rows are checkpoints in fixed order, for the report;
  * blinded grid   - rows shuffled with a recorded seed and labelled A/B/C...,
                     with a key file and a CSV annotation sheet, so a person can
                     score presence without knowing which row is which.

No claim of human review is made anywhere by this script. It only prepares the
sheet; whether a human filled it in is recorded separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

CKPT_ORDER = ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]
SEEDS = [101, 202]
FAMILIES = ["literal", "paraphrase"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--categories", default="cat,dog,sandwich,horse,bird")
    ap.add_argument("--blind_seed", type=int, default=20261009)
    ap.add_argument("--thumb", type=int, default=256)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    out = Path(args.out_dir)
    (out / "labelled").mkdir(parents=True, exist_ok=True)
    (out / "blinded").mkdir(parents=True, exist_ok=True)
    root = Path(args.eval_root)

    present = [c for c in CKPT_ORDER if (root / c / "images").is_dir()]
    cats = args.categories.split(",")
    T, PAD, LAB_W, HDR = args.thumb, 8, 120, 34

    cells = [(fam, s) for fam in FAMILIES for s in SEEDS]
    keys = {}
    sheet_rows = []

    for cat in cats:
        names = [f"{cat}_{fam}_0_seed{s}.jpg" for fam, s in cells]

        def build(rows_order, labels, path, title):
            W = LAB_W + len(names) * (T + PAD) + PAD
            H = HDR + len(rows_order) * (T + PAD) + PAD
            canvas = Image.new("RGB", (W, H), "#fcfcfb")
            d = ImageDraw.Draw(canvas)
            d.text((PAD, 10), title, fill="#0b0b0b")
            for j, (fam, s) in enumerate(cells):
                d.text((LAB_W + j * (T + PAD), HDR - 14),
                       f"{fam[:4]} seed{s}", fill="#52514e")
            for i, ck in enumerate(rows_order):
                y = HDR + i * (T + PAD)
                d.text((PAD, y + T // 2 - 6), labels[i], fill="#0b0b0b")
                for j, nm in enumerate(names):
                    p = root / ck / "images" / nm
                    x = LAB_W + j * (T + PAD)
                    if p.exists():
                        im = Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS)
                        canvas.paste(im, (x, y))
                    else:
                        d.rectangle([x, y, x + T, y + T], outline="#e34948")
                        d.text((x + 8, y + T // 2), "missing", fill="#e34948")
            canvas.save(path, quality=95)

        build(present, present, out / "labelled" / f"grid_{cat}.jpg",
              f"{cat} - identical prompts/seeds across checkpoints "
              f"(prompt index 0, seeds 101/202)")

        rng = random.Random(args.blind_seed + hash(cat) % 1000)
        shuffled = present[:]
        rng.shuffle(shuffled)
        labels = [chr(ord("A") + i) for i in range(len(shuffled))]
        build(shuffled, labels, out / "blinded" / f"blind_{cat}.jpg",
              f"{cat} - blinded rows (score presence of '{cat}' in each cell)")
        keys[cat] = dict(zip(labels, shuffled))
        for lab in labels:
            for fam, s in cells:
                sheet_rows.append({"category": cat, "blind_row": lab,
                                   "prompt_family": fam, "gen_seed": s,
                                   "target_visible__fill_in_yes_no_unsure": "",
                                   "notes": ""})

    (out / "blinded" / "KEY_do_not_open_before_annotating.json").write_text(
        json.dumps({"blind_seed": args.blind_seed, "mapping": keys}, indent=2))

    with open(out / "blinded" / "annotation_sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet_rows[0].keys()))
        w.writeheader()
        w.writerows(sheet_rows)

    (out / "blinded" / "README.md").write_text(
        "# Blinded visual check\n\n"
        "Rows are shuffled per category and labelled A/B/C...; the mapping is in "
        "`KEY_do_not_open_before_annotating.json`.\n\n"
        "1. Open `blind_<category>.jpg`.\n"
        "2. For each row and cell, record whether the named category is visibly "
        "present, in `annotation_sheet.csv`.\n"
        "3. Only then open the key.\n\n"
        "**No human annotation has been performed yet.** The sheet is empty by "
        "design. Any statement that images were human-reviewed must be backed by a "
        "filled-in copy of this sheet.\n")

    print(f"wrote labelled grids : {out/'labelled'}")
    print(f"wrote blinded grids  : {out/'blinded'} (+ key, annotation_sheet.csv)")
    print(f"checkpoints included : {present}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
