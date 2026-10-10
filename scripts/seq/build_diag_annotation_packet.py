#!/usr/bin/env python3
"""Blinded human-audit packet for the diagnostic's frozen-test evaluation.

Generates nothing. Every item is an image that already exists; it is copied (or
referenced) under a blinded id. CPU only, no model, no GPU.

Two sets, kept separate on purpose
----------------------------------
SET R (random)      A reproducible stratified random sample across every
                    (training seed, arm, category, prompt family) cell. This is
                    the ONLY set from which a detector error rate may be
                    estimated, and then only for the strata it covers.

SET D (diagnostic)  Deliberately ENRICHED where the detector is least
                    trustworthy: target scores between the 0.3 and 0.7
                    thresholds, and verdict flips between the parent MA and a
                    child arm at the same prompt and generation seed. Error
                    rates on SET D are biased upward by construction and MUST
                    NOT be reported as overall accuracy.

What is blinded
---------------
Hidden from the annotator: which arm produced the image, which training seed,
the detector's score and verdict, and which set the item belongs to. Shown: the
category to judge, because without it there is no question. Order is shuffled
once under a fixed RNG seed, so the packet is reproducible but carries no
condition ordering.

The key mapping blinded ids back to conditions is written OUTSIDE the packet
directory, so it cannot be opened by accident while annotating. The label sheet
is written with every label cell EMPTY.

    python scripts/seq/build_diag_annotation_packet.py \
        --eval_root <diag>/eval_test --seeds "17 29" \
        --selection <diag>/selection.json --out_root <diag>/annotation_v2 \
        --repo_out results/audit_v1/annotation_packet_v2
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

FAMILIES = ["literal", "paraphrase"]


def load_rows(eval_root: Path, seeds: list[int]) -> tuple[list[dict], list[str]]:
    rows, arms = [], []
    for s in seeds:
        for d in sorted(eval_root.glob(f"seed{s}_*")):
            det = d / "detections.jsonl"
            if not det.is_file():
                continue
            arm = d.name.split("_", 1)[1]
            arms.append(d.name)
            for line in det.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["_seed"] = s
                r["_arm"] = arm
                r["_slot"] = d.name
                rows.append(r)
    return rows, arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--selection", default=None)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--repo_out", default=None)
    ap.add_argument("--per_cell", type=int, default=2)
    ap.add_argument("--n_diagnostic", type=int, default=60)
    ap.add_argument("--rng_seed", type=int, default=20261011)
    ap.add_argument("--salt", default="AMENDMENT-01")
    ap.add_argument("--copy_images", action="store_true", default=True)
    a = ap.parse_args()

    root = Path(a.eval_root)
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    rows, arms = load_rows(root, seeds)
    if not rows:
        print(f"FATAL: no detection rows under {root}")
        return 2

    rng = random.Random(a.rng_seed)

    # ---- SET R: stratified, reproducible
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["_seed"], r["_arm"], r.get("category"),
               r.get("prompt_family"))].append(r)
    set_r: list[dict] = []
    weights = {}
    for key in sorted(cells, key=lambda k: tuple(map(str, k))):
        pool = sorted(cells[key], key=lambda r: r["image_path"])
        take = min(a.per_cell, len(pool))
        pick = rng.sample(pool, take)
        for r in pick:
            r = dict(r)
            r["_set"] = "R"
            r["_cell"] = "|".join(map(str, key))
            set_r.append(r)
        weights["|".join(map(str, key))] = {"cell_size": len(pool),
                                            "sampled": take,
                                            "weight": len(pool) / take if take else None}

    # ---- SET D: enriched for the detector's hard cases
    chosen = {r["image_path"] for r in set_r}
    band = [r for r in rows
            if r["image_path"] not in chosen
            and 0.3 <= float(r.get("max_score_target") or 0.0) <= 0.7]
    # verdict flips vs that seed's own MA at the same prompt and generation seed
    ma_hit = {(r["_seed"], r.get("prompt_id"), r.get("gen_seed")): r["hit"]["0.5"]
              for r in rows if r["_arm"] == "MA"}
    flips = [r for r in rows
             if r["image_path"] not in chosen and r["_arm"] != "MA"
             and (r["_seed"], r.get("prompt_id"), r.get("gen_seed")) in ma_hit
             and ma_hit[(r["_seed"], r.get("prompt_id"), r.get("gen_seed"))]
                 != r["hit"]["0.5"]]
    pool_d = {r["image_path"]: r for r in band}
    pool_d.update({r["image_path"]: r for r in flips})
    pool_list = [pool_d[k] for k in sorted(pool_d)]
    take_d = min(a.n_diagnostic, len(pool_list))
    set_d = []
    for r in rng.sample(pool_list, take_d):
        r = dict(r)
        r["_set"] = "D"
        r["_cell"] = "|".join(map(str, (r["_seed"], r["_arm"],
                                        r.get("category"),
                                        r.get("prompt_family"))))
        set_d.append(r)

    items = set_r + set_d
    rng.shuffle(items)

    out_root = Path(a.out_root)
    packet = out_root / "packet"
    packet.mkdir(parents=True, exist_ok=True)
    (packet / "images").mkdir(exist_ok=True)

    key_rows, sheet_rows = [], []
    for i, r in enumerate(items, start=1):
        blind = "item_" + hashlib.sha256(
            f"{a.salt}|{r['image_path']}".encode()).hexdigest()[:12]
        src = Path(r["image_path"])
        ext = src.suffix or ".jpg"
        dst = packet / "images" / f"{blind}{ext}"
        if a.copy_images and src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
        sheet_rows.append({
            "item_id": blind,
            "image": f"images/{blind}{ext}",
            "category_to_judge": r.get("category"),
            "question": (f"Is a clearly recognisable {r.get('category')} "
                         f"present in this image?"),
            "answer_yes_no_unsure": "",          # left EMPTY, deliberately
            "notes": "",
        })
        key_rows.append({
            "item_id": blind, "set": r["_set"], "cell": r["_cell"],
            "training_seed": r["_seed"], "arm": r["_arm"], "slot": r["_slot"],
            "category": r.get("category"), "prompt_family": r.get("prompt_family"),
            "prompt_id": r.get("prompt_id"), "gen_seed": r.get("gen_seed"),
            "detector_max_score_target": r.get("max_score_target"),
            "detector_hit_at_0.5": r["hit"]["0.5"],
            "image_path": r["image_path"], "image_sha256": r.get("image_sha256"),
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
        "`answer_yes_no_unsure`. Leave `notes` for anything ambiguous.\n\n"
        "You are not told which model produced an image, which condition it\n"
        "belongs to, or what an automatic detector thought. That is deliberate:\n"
        "your labels are the reference these detector numbers are checked\n"
        "against, so they must be formed independently.\n\n"
        "Do not look for the key. It is stored outside this directory.\n")

    # The key lives OUTSIDE the packet directory.
    keyp = out_root / "KEY_DO_NOT_OPEN_WHILE_ANNOTATING.csv"
    with keyp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    meta = {
        "_what_this_is": (
            "Blinded human-audit packet over images that already exist in the "
            "frozen-test evaluation. Nothing was generated. Labels are EMPTY: "
            "annotation proceeds separately and detector results stay "
            "provisional until it is complete."),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eval_root": str(root), "seeds": seeds, "slots": sorted(set(arms)),
        "selection_record": a.selection,
        "rng_seed": a.rng_seed, "salt": a.salt,
        "n_items": len(items), "n_set_R": len(set_r), "n_set_D": len(set_d),
        "set_R_stratification": {
            "cell": "(training seed, arm, category, prompt family)",
            "per_cell": a.per_cell, "weights": weights,
            "use": ("the only set from which a detector error rate may be "
                    "estimated, with these per-cell weights applied"),
        },
        "set_D_enrichment": {
            "rule": ("target score in [0.3, 0.7], or a 0.5-threshold verdict "
                     "flip against that seed's own MA at the same prompt and "
                     "generation seed"),
            "pool_size": len(pool_list), "sampled": take_d,
            "warning": ("biased upward by construction; NOT an overall "
                        "accuracy estimate"),
        },
        "blinding": {
            "hidden_from_annotator": ["arm", "training seed", "detector score",
                                      "detector verdict", "set membership"],
            "shown_to_annotator": ["the category to judge"],
            "order": f"shuffled once under RNG seed {a.rng_seed}",
            "key_location": str(keyp),
            "key_inside_packet": False,
        },
        "label_state": "EMPTY -- no labels are supplied or implied",
    }
    (out_root / "packet_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    if a.repo_out:
        rp = Path(a.repo_out)
        rp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packet / "label_sheet.csv", rp / "label_sheet.csv")
        shutil.copy2(packet / "INSTRUCTIONS.md", rp / "INSTRUCTIONS.md")
        shutil.copy2(out_root / "packet_manifest.json", rp / "packet_manifest.json")
        print(f"[repo copy] key-free packet metadata -> {rp}")

    print(f"=== blinded audit packet ===")
    print(f"  items      : {len(items)}  (SET R {len(set_r)}, SET D {len(set_d)})")
    print(f"  packet     : {packet}")
    print(f"  key        : {keyp}  (outside the packet)")
    print(f"  labels     : EMPTY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
