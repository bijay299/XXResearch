#!/usr/bin/env python3
"""Builds BLINDED contact sheets for the agent visual audit, plus the key that is only
read AFTER the judgments are written down.

Arm identity is hidden: tiles are shuffled by a fixed seed and labelled with opaque slot
ids. The audit is recorded as an AGENT JUDGMENT, never as independent human review.
"""
import os, json, argparse, random
from PIL import Image, ImageDraw

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"


def build(items, out_png, out_key, cols=4, tile=320, seed=1234):
    random.Random(seed).shuffle(items)
    rows = (len(items) + cols - 1) // cols
    lab = 26
    sheet = Image.new("RGB", (cols * tile, rows * (tile + lab)), "white")
    d = ImageDraw.Draw(sheet)
    key = []
    for i, it in enumerate(items):
        im = Image.open(it["image"]).convert("RGB").resize((tile, tile), Image.LANCZOS)
        r, c = divmod(i, cols)
        x, y = c * tile, r * (tile + lab)
        sheet.paste(im, (x, y + lab))
        slot = f"S{i+1:02d}"
        d.text((x + 4, y + 6), slot, fill="black")
        d.rectangle([x, y, x + tile - 1, y + tile + lab - 1], outline="black")
        key.append({"slot": slot, **it})
    sheet.save(out_png)
    json.dump(key, open(out_key, "w"), indent=1)
    print(f"wrote {out_png} ({len(items)} tiles, {cols}x{rows}) and key {out_key}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="json: list of {arm,object,stratum,theme,class}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=320)
    a = ap.parse_args()
    build(json.load(open(a.spec)), a.out, a.key, a.cols, a.tile)
