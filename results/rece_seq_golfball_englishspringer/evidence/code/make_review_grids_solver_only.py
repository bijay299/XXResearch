#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/solver_only_intervention")
MANIFEST = json.loads(Path("/data/bijaypandey/cuig_pilot/rece_golfball_confirmation/manifest/frozen_manifest.json").read_text())
rows_by_class = {}
for r in MANIFEST["rows"]:
    rows_by_class.setdefault(r["class"], []).append(r)
for c in rows_by_class:
    rows_by_class[c].sort(key=lambda r: r["case_id"])

THUMB = 180
COLS = 8

for ckpt in ["AB_STABLE_SOLVE"]:
    img_dir = HERE / "eval" / ckpt / "imgs"
    out_dir = HERE / "review_grids" / ckpt
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls, rows in rows_by_class.items():
        n = len(rows)
        cols = min(COLS, n)
        rows_n = (n + cols - 1) // cols
        sheet = Image.new("RGB", (cols * THUMB, rows_n * (THUMB + 18)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            case_id = row["case_id"]
            fpath = img_dir / f"{case_id}_0.png"
            r, c = divmod(i, cols)
            x, y = c * THUMB, r * (THUMB + 18)
            if fpath.exists():
                im = Image.open(fpath).convert("RGB").resize((THUMB, THUMB))
                sheet.paste(im, (x, y))
            else:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline="red", width=3)
            draw.text((x + 2, y + THUMB + 2), f"{case_id} f={row['family'][:3]}", fill="black")
        safe_cls = cls.replace(" ", "_")
        out_path = out_dir / f"{safe_cls}.png"
        sheet.save(out_path)
        print(f"wrote {out_path} ({n} cells, {cols}x{rows_n})")
