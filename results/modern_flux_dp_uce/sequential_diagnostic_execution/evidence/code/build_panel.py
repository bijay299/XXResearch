"""Build the protocol's frozen 20-case rendering panel plus readable comparison grids.

Panel (frozen before execution, unchanged): variation 0 of the first two CSV rows of
each of the 10 classes -> cases 0,1,10,11,...,90,91.

Columns per method: M0 | A-parent (chain-saw-erased) | B-direct (parachute) | SEQ.
Batch comparators are not included because they were not triggered.
"""
from __future__ import annotations

import hashlib
import os
import shutil

from PIL import Image, ImageDraw

EXEC = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02"
OLDRUNS = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/runs"
OUT = f"{EXEC}/panel"

CASES = {0: "cassette_player", 1: "cassette_player", 10: "chain_saw", 11: "chain_saw",
         20: "church", 21: "church", 30: "gas_pump", 31: "gas_pump",
         40: "tench", 41: "tench", 50: "garbage_truck", 51: "garbage_truck",
         60: "english_springer", 61: "english_springer", 70: "golf_ball", 71: "golf_ball",
         80: "parachute", 81: "parachute", 90: "french_horn", 91: "french_horn"}

SRC = {
    "M0":            f"{OLDRUNS}/M0/images",
    "DP_A_parent":   f"{OLDRUNS}/DP_B/images",
    "DP_B_direct":   f"{EXEC}/runs/DP_B_direct/images",
    "DP_SEQ":        f"{EXEC}/runs/DP_SEQ/images",
    "UCE_A_parent":  f"{OLDRUNS}/UCE_B/images",
    "UCE_B_direct":  f"{EXEC}/runs/UCE_B_direct/images",
    "UCE_SEQ":       f"{EXEC}/runs/UCE_SEQ/images",
}
COLS = {"DP": ["M0", "DP_A_parent", "DP_B_direct", "DP_SEQ"],
        "UCE": ["M0", "UCE_A_parent", "UCE_B_direct", "UCE_SEQ"]}


def fn(case: int) -> str:
    return f"{case:03d}_{CASES[case]}_case{case:03d}_var00.png"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ep, d in SRC.items():
        od = os.path.join(OUT, "full_resolution", ep)
        os.makedirs(od, exist_ok=True)
        for c in sorted(CASES):
            s = os.path.join(d, fn(c))
            assert os.path.isfile(s), s
            t = os.path.join(od, fn(c))
            shutil.copy2(s, t)
            b = open(t, "rb").read()
            rows.append((ep, fn(c), len(b), hashlib.sha256(b).hexdigest()))
    with open(os.path.join(OUT, "panel_hashes.csv"), "w") as f:
        f.write("endpoint,filename,bytes,sha256\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    # readable grids: one per method per class (2 cases x 4 endpoints), labeled
    gd = os.path.join(OUT, "grids")
    os.makedirs(gd, exist_ok=True)
    classes = sorted(set(CASES.values()))
    for method, cols in COLS.items():
        for cls in classes:
            cs = [c for c in sorted(CASES) if CASES[c] == cls]
            W = H = 512
            pad, hdr = 6, 34
            img = Image.new("RGB", (len(cols) * (W + pad) + pad,
                                    len(cs) * (H + pad) + hdr + pad), "white")
            dr = ImageDraw.Draw(img)
            for j, ep in enumerate(cols):
                dr.text((pad + j * (W + pad) + 6, 10), f"{ep}", fill="black")
            for i, c in enumerate(cs):
                for j, ep in enumerate(cols):
                    im = Image.open(os.path.join(OUT, "full_resolution", ep, fn(c))).convert("RGB")
                    img.paste(im, (pad + j * (W + pad), hdr + pad + i * (H + pad)))
                dr.text((pad + 6, hdr + pad + i * (H + pad) + 6), f"case {c}", fill="yellow")
            p = os.path.join(gd, f"{method}_{cls}.png")
            img.save(p, optimize=True)
    print(f"panel files: {len(rows)}; grids: {len(classes)*2}")
    print("->", OUT)


if __name__ == "__main__":
    main()
