#!/usr/bin/env python3
"""Build labeled contact-sheet grids for S2_STAGE1_HISTORY's two purely-
retained classes (parachute 20-39, French horn 40-59), same layout
convention as S1_STAGE1_HISTORY's review_grids/. 160px cells, case_id
labeled, no content alteration beyond resize."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE.parent / "history_preservation_confirmation" / "panel" / "FRESH_PANEL_MANIFEST.json").read_text())
rows_by_class = {}
for r in MANIFEST["rows"]:
    rows_by_class.setdefault(r["class"], []).append(r)
for c in rows_by_class:
    rows_by_class[c].sort(key=lambda r: r["case_id"])

THUMB = 160
COLS = 5
img_dir = HERE / "eval" / "S2_STAGE1_HISTORY" / "imgs"
out_dir = HERE / "review_grids"
out_dir.mkdir(parents=True, exist_ok=True)

for cls in ["parachute", "French horn"]:
    rows = rows_by_class[cls]
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
    safe_cls = cls.replace(" ", "_").lower()
    out_path = out_dir / f"{safe_cls}.png"
    sheet.save(out_path)
    print(f"wrote {out_path} ({n} cells, {cols}x{rows_n})")
