#!/usr/bin/env python3
"""Pre-training calibration of the M0 generator and the COCO detector.

Two questions must be answered BEFORE any editing, or a later "erasure" claim is
uninterpretable:

  1. Does untouched M0 actually generate the target categories reliably? If M0
     rarely produces a cat, a low cat rate after editing says nothing.
  2. Does the detector find targets that are visibly present, and does it stay
     quiet on negative controls? A detector that misses visible cats would
     manufacture apparent erasure.

Calibration prompts are deliberately SEPARATE from the frozen evaluation
manifest and from the ConAbl training prompts. Negative controls are scenes that
should contain none of the seven categories.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

CATEGORIES = ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]

POSITIVE = {
    "cat": ["a cat on a kitchen counter", "a cat beside a bookshelf",
            "a cat sitting on a stone step"],
    "dog": ["a dog on a sandy beach", "a dog beside a red door",
            "a dog sitting in tall grass"],
    "sandwich": ["a sandwich on a picnic blanket", "a sandwich beside a mug of coffee",
                 "a sandwich on a metal tray"],
    "horse": ["a horse near a stone wall", "a horse in a snowy field",
              "a horse beside a wooden gate"],
    "bird": ["a bird on a balcony railing", "a bird beside a bird bath",
             "a bird on a mossy log"],
    "chair": ["a chair on a wooden porch", "a chair beside a fireplace",
              "a chair in a tiled hallway"],
    "bicycle": ["a bicycle on a bridge", "a bicycle beside a brick wall",
                "a bicycle on a country lane"],
}

# Scenes that should contain none of the seven categories.
NEGATIVE = [
    "an empty desert landscape at midday",
    "a close-up of ocean waves breaking",
    "a plain blue sky with scattered clouds",
    "a rocky mountain ridge under grey light",
    "a dense green forest canopy from below",
    "a calm lake surface reflecting trees",
]

SEEDS = [11, 22]
THRESHOLDS = [0.3, 0.5, 0.7]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                              FasterRCNN_ResNet50_FPN_Weights)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    pipe = StableDiffusionPipeline.from_pretrained(args.base_model_dir,
                                                   torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = lambda images, **kw: (images, [False] * len(images))

    jobs = []
    for cat, prompts in POSITIVE.items():
        for i, p in enumerate(prompts):
            for s in SEEDS:
                jobs.append({"kind": "positive", "category": cat, "prompt": p,
                             "prompt_index": i, "seed": s,
                             "name": f"pos_{cat}_{i}_seed{s}.jpg"})
    for i, p in enumerate(NEGATIVE):
        for s in SEEDS:
            jobs.append({"kind": "negative", "category": None, "prompt": p,
                         "prompt_index": i, "seed": s,
                         "name": f"neg_{i}_seed{s}.jpg"})

    print(f"generating {len(jobs)} calibration images...")
    for j in jobs:
        p = out / j["name"]
        if p.exists():
            continue
        g = torch.Generator(device=device).manual_seed(j["seed"] * 7919 + j["prompt_index"])
        img = pipe(prompt=j["prompt"], width=512, height=512,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=g).images[0]
        img.save(p, quality=95)
    del pipe
    torch.cuda.empty_cache()

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    cats_meta = weights.meta["categories"]
    pre = weights.transforms()
    model = fasterrcnn_resnet50_fpn(weights=weights).eval().to(device)

    for j in jobs:
        x = pre(Image.open(out / j["name"]).convert("RGB")).to(device)
        with torch.no_grad():
            pr = model([x])[0]
        dets = [{"label": cats_meta[int(l)], "score": round(float(s), 4)}
                for l, s in zip(pr["labels"].cpu().tolist(), pr["scores"].cpu().tolist())
                if s >= 0.05]
        j["detections"] = dets[:20]
        if j["kind"] == "positive":
            j["hit"] = {str(t): any(d["label"] == j["category"] and d["score"] >= t
                                    for d in dets) for t in THRESHOLDS}
            j["max_score_target"] = max([d["score"] for d in dets
                                         if d["label"] == j["category"]], default=0.0)
        else:
            j["any_of_seven"] = {str(t): sorted({d["label"] for d in dets
                                                 if d["label"] in CATEGORIES and d["score"] >= t})
                                 for t in THRESHOLDS}

    summary = {}
    for cat in CATEGORIES:
        sub = [j for j in jobs if j["kind"] == "positive" and j["category"] == cat]
        summary[cat] = {
            "n": len(sub),
            "detection_rate": {str(t): round(sum(j["hit"][str(t)] for j in sub) / len(sub), 3)
                               for t in THRESHOLDS},
            "mean_max_score": round(sum(j["max_score_target"] for j in sub) / len(sub), 4),
        }
    negs = [j for j in jobs if j["kind"] == "negative"]
    neg_summary = {str(t): round(sum(1 for j in negs if j["any_of_seven"][str(t)]) / len(negs), 3)
                   for t in THRESHOLDS}

    # Calibration errors: a target M0 cannot reliably draw, or the detector cannot see.
    errors = []
    for cat in ["cat", "dog", "sandwich"]:
        if summary[cat]["detection_rate"]["0.5"] < 0.75:
            errors.append(f"{cat}: M0 detection rate "
                          f"{summary[cat]['detection_rate']['0.5']:.2f} < 0.75 at t=0.5 - "
                          "erasure of this target would not be interpretable")
    for cat in CATEGORIES:
        if summary[cat]["detection_rate"]["0.5"] < 0.5:
            errors.append(f"{cat}: control category weakly generated/detected "
                          f"({summary[cat]['detection_rate']['0.5']:.2f})")
    if neg_summary["0.5"] > 0.34:
        errors.append(f"negative controls fire at {neg_summary['0.5']:.2f} (>0.34) - "
                      "possible detector over-triggering")

    report = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "stage": "pre-training calibration of M0 and the COCO detector",
        "base_model_dir": args.base_model_dir,
        "generation": {"num_inference_steps": args.steps, "guidance_scale": args.guidance,
                       "resolution": 512, "seeds": SEEDS},
        "detector": {"weights_enum": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
                     "url": weights.url},
        "prompts_are_separate_from_eval_manifest": True,
        "positive_summary": summary,
        "negative_control_any_of_seven_rate": neg_summary,
        "calibration_errors": errors,
        "usable": not errors,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "jobs": jobs,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))

    print("\n=== M0 + detector calibration (t=0.5) ===")
    for cat in CATEGORIES:
        s = summary[cat]
        print(f"  {cat:9s} D={s['detection_rate']['0.5']:.2f}  "
              f"(0.3:{s['detection_rate']['0.3']:.2f} 0.7:{s['detection_rate']['0.7']:.2f})  "
              f"mean_max_score={s['mean_max_score']:.3f}  n={s['n']}")
    print(f"  negative controls firing any-of-seven: {neg_summary}")
    if errors:
        print("\nCALIBRATION ERRORS:")
        for e in errors:
            print("  - " + e)
    else:
        print("\ncalibration OK")
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
