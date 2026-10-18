#!/usr/bin/env python3
"""Fixed-order contact grids: every scored image for one (arm, object), laid out in the
frozen case order from cases.json (target 40, then related 20, then unrelated 20).
Downscaled JPEGs so they are accessible image evidence inside git; the full-resolution
PNG originals ship as a release asset.
"""
import os, json, sys
from PIL import Image, ImageDraw

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
OUT = f"{SPRINT}/artifacts/grids"
TILE, COLS, LAB = 168, 10, 16


def grid(arm, obj):
    man = json.load(open(f"{SPRINT}/artifacts/eval_views/{arm}_{obj}/presence_manifest.json"))
    rows = (len(man) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE, rows * (TILE + LAB)), "white")
    d = ImageDraw.Draw(sheet)
    for i, m in enumerate(man):
        im = Image.open(m["image"]).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
        r, c = divmod(i, COLS)
        x, y = c * TILE, r * (TILE + LAB)
        sheet.paste(im, (x, y + LAB))
        d.text((x + 2, y + 3), f"{i+1:02d} {m['stratum'][:3]} {m['theme'][:9]}", fill="black")
        d.rectangle([x, y, x + TILE - 1, y + TILE + LAB - 1], outline="#999")
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{arm}_{obj}_fixed_order.jpg"
    sheet.save(p, quality=86, optimize=True)
    print(f"  {p}  ({len(man)} tiles, {os.path.getsize(p)//1024} KB)")


if __name__ == "__main__":
    for obj in ("Bears", "Cats", "Horses"):
        for arm in ("M0", "T0", "N"):
            if os.path.exists(f"{SPRINT}/artifacts/eval_views/{arm}_{obj}"):
                grid(arm, obj)
