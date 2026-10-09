#!/usr/bin/env python3
"""Run the official Torchvision Faster R-CNN COCO detector over a checkpoint's images.

Evaluator: `fasterrcnn_resnet50_fpn` with `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`,
downloaded from its official source, used in `eval()` mode with the documented
preprocessing transform attached to the weights enum and the category names from
the weights metadata. No random weights, no forced-choice classifier.

Target detection rate D = fraction of images with >=1 detection of the requested
category at confidence >= threshold. Raw predictions are saved so the metric can
be recomputed at 0.3 / 0.5 / 0.7 without re-running the detector.

These are detector-based object-presence proxies. They are not ground-truth
erasure measurements and not image-quality scores.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

THRESHOLDS = [0.3, 0.5, 0.7]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_report", required=True, help="generate_eval_images.py report")
    ap.add_argument("--checkpoint_name", required=True)
    ap.add_argument("--out", required=True, help="per-image detection JSONL")
    ap.add_argument("--report", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keep_topk", type=int, default=100)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                              FasterRCNN_ResNet50_FPN_Weights)

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    categories = weights.meta["categories"]          # index -> COCO name, incl. "__background__"
    preprocess = weights.transforms()                # documented preprocessing

    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Record evaluator identity: URL, local cached file and its SHA-256.
    url = weights.url
    ckpt_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    wsha = None
    if ckpt_path.exists():
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as fh:
            while c := fh.read(1 << 20):
                h.update(c)
        wsha = h.hexdigest()

    detector_identity = {
        "architecture": "fasterrcnn_resnet50_fpn",
        "weights_enum": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        "url": url,
        "local_file": str(ckpt_path),
        "file_sha256": wsha,
        "num_categories_incl_background": len(categories),
        "torchvision": __import__("torchvision").__version__,
        "mode": "eval()",
        "preprocessing": str(preprocess),
        "meta_min_size": weights.meta.get("min_size"),
    }
    print(json.dumps(detector_identity, indent=2))

    img_report = json.loads(Path(args.image_report).read_text())
    rows = img_report["images"]

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with open(out_p, "w") as fh:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i: i + args.batch]
            imgs, keep = [], []
            for r in chunk:
                p = Path(r["image_path"])
                if not p.exists():
                    print(f"[MISSING] {p}", file=sys.stderr)
                    continue
                imgs.append(preprocess(Image.open(p).convert("RGB")).to(device))
                keep.append(r)
            if not imgs:
                continue
            with torch.no_grad():
                preds = model(imgs)
            for r, pr in zip(keep, preds):
                boxes = pr["boxes"].cpu().tolist()[: args.keep_topk]
                labels = pr["labels"].cpu().tolist()[: args.keep_topk]
                scores = pr["scores"].cpu().tolist()[: args.keep_topk]
                dets = [{"label_id": int(l), "label": categories[int(l)],
                         "score": round(float(s), 5),
                         "box": [round(float(b), 1) for b in bx]}
                        for l, s, bx in zip(labels, scores, boxes) if s >= 0.05]
                tgt = r["category"]
                rec = {
                    "checkpoint": args.checkpoint_name,
                    "prompt_id": r["prompt_id"], "category": tgt,
                    "prompt_family": r["prompt_family"], "prompt_index": r["prompt_index"],
                    "prompt": r["prompt"], "gen_seed": r["gen_seed"],
                    "image_seed": r.get("image_seed"),
                    "image_path": r["image_path"], "image_sha256": r.get("sha256"),
                    "detections": dets,
                    "max_score_target": max([d["score"] for d in dets
                                             if d["label"] == tgt], default=0.0),
                    "n_detections_target": {
                        str(t): sum(1 for d in dets if d["label"] == tgt and d["score"] >= t)
                        for t in THRESHOLDS},
                    "hit": {str(t): bool(any(d["label"] == tgt and d["score"] >= t
                                             for d in dets)) for t in THRESHOLDS},
                }
                fh.write(json.dumps(rec) + "\n")
                n_done += 1
            if (i + args.batch) % 80 == 0:
                print(f"  {min(i+args.batch, len(rows))}/{len(rows)}", flush=True)

    report = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "metric_definition": ("target detection rate D = fraction of images with >=1 "
                              "detection of the requested category at confidence >= t"),
        "caveat": ("detector-based object-presence proxy; not ground-truth erasure, "
                   "not an image-quality score; not UA/IRA/CRA"),
        "checkpoint": args.checkpoint_name,
        "detector": detector_identity,
        "thresholds": THRESHOLDS,
        "images_scored": n_done,
        "images_expected": len(rows),
        "complete": n_done == len(rows),
        "per_image_jsonl": str(out_p),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"[{args.checkpoint_name}] scored {n_done}/{len(rows)} -> {out_p}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
