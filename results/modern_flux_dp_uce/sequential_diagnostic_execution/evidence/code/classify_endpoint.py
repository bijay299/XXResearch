"""Classify one endpoint's 1,000 images with the PINNED evaluator, unchanged.

Imports ErasureAccuracyEvaluator / SEMANTIC_EQUIVALENT_CLASSES directly from the pinned
Flux/compare_accuracy.py (same ResNet50 IMAGENET1K_V2 weights, same preprocessing, same
semantic-equivalence mapping, same merge logic) so every number traces to that code.

Asserts no row loss across the prompts<->predictions inner join, and keeps strict and
semantic correctness as separate columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

SRC = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo/Flux"
sys.path.insert(0, SRC)
from compare_accuracy import ErasureAccuracyEvaluator, SEMANTIC_EQUIVALENT_CLASSES  # noqa: E402

PROMPTS = ("/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/source_repo/SD/data/"
           "small_imagenet_prompts.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    n_disk = len([f for f in os.listdir(a.images) if f.endswith(".png")])
    assert n_disk == 1000, f"[{a.name}] expected 1000 images, found {n_disk}"

    ev = ErasureAccuracyEvaluator(device=a.device, batch_size=250)
    prompts = pd.read_csv(PROMPTS)
    assert len(prompts) == 100

    merged = ev.classify_images(a.images, prompts.copy())
    assert len(merged) == 1000, (
        f"[{a.name}] inner join shrank the panel: {n_disk} images -> {len(merged)} rows")

    merged["pred_top1_lower"] = merged["category_top1"].str.strip().str.lower()
    merged["true_class_lower"] = merged["class"].str.strip().str.lower()
    merged["correct_strict"] = merged["pred_top1_lower"] == merged["true_class_lower"]

    def sem(row):
        if row["correct_strict"]:
            return True
        eq = SEMANTIC_EQUIVALENT_CLASSES.get(row["true_class_lower"])
        return bool(eq) and row["pred_top1_lower"] in [c.lower() for c in eq]

    merged["correct_semantic"] = merged.apply(sem, axis=1)

    cols = (["case_number", "class", "prompt", "evaluation_seed", "filename",
             "correct_strict", "correct_semantic"]
            + [f"category_top{k}" for k in range(1, 6)]
            + [f"scores_top{k}" for k in range(1, 6)])
    merged = merged[cols].sort_values(["case_number", "filename"]).reset_index(drop=True)
    merged.to_csv(a.out_csv, index=False)

    summary = {
        "endpoint": a.name,
        "images_on_disk": n_disk,
        "rows_after_join": len(merged),
        "semantic_correct_per_class": merged.groupby("class")["correct_semantic"].sum().astype(int).to_dict(),
        "strict_correct_per_class": merged.groupby("class")["correct_strict"].sum().astype(int).to_dict(),
    }
    with open(a.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["semantic_correct_per_class"], indent=2))
    print("->", a.out_csv)


if __name__ == "__main__":
    main()
