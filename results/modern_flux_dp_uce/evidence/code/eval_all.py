"""
Evaluation orchestrator for the native FLUX DP/UCE reproduction.

Reuses the pinned repo's own ErasureAccuracyEvaluator / SEMANTIC_EQUIVALENT_CLASSES
(imported unmodified from source_repo/Flux/compare_accuracy.py -- same ResNet50
weights, same preprocessing, same semantic-equivalence mapping, same merge logic)
so that every classification number in this run traces back to the pinned
evaluation code, not a reimplementation.

Classifies each of the 5 generated endpoints (M0, DP_A, DP_B, UCE_A, UCE_B) exactly
once (avoids re-classifying the shared M0 set four times), saves full per-image
top-5 records (not just the aggregate accuracy table), and asserts no row loss
across the prompts<->predictions inner join before any downstream analysis runs.
"""
import os
import sys
import json
import pandas as pd

ROOT = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce"
sys.path.insert(0, os.path.join(ROOT, "source_repo", "Flux"))
from compare_accuracy import ErasureAccuracyEvaluator, SEMANTIC_EQUIVALENT_CLASSES  # noqa: E402

PROMPTS_CSV = os.path.join(ROOT, "source_repo", "SD", "data", "small_imagenet_prompts.csv")
ENDPOINTS = {
    "M0": os.path.join(ROOT, "runs", "M0", "images"),
    "DP_A": os.path.join(ROOT, "runs", "DP_A", "images"),
    "DP_B": os.path.join(ROOT, "runs", "DP_B", "images"),
    "UCE_A": os.path.join(ROOT, "runs", "UCE_A", "images"),
    "UCE_B": os.path.join(ROOT, "runs", "UCE_B", "images"),
}
OUT_DIR = os.path.join(ROOT, "eval")
os.makedirs(OUT_DIR, exist_ok=True)

evaluator = ErasureAccuracyEvaluator(device="cuda:0", batch_size=250)
prompts_df = pd.read_csv(PROMPTS_CSV)
assert len(prompts_df) == 100, f"expected 100 prompt rows, got {len(prompts_df)}"

per_endpoint_df = {}
for name, folder in ENDPOINTS.items():
    n_images_on_disk = len([f for f in os.listdir(folder) if f.endswith(".png")])
    merged = evaluator.classify_images(folder, prompts_df.copy())
    n_after_join = len(merged)
    print(f"[{name}] images_on_disk={n_images_on_disk} rows_after_join={n_after_join}")
    assert n_images_on_disk == 1000, f"[{name}] expected 1000 images on disk, found {n_images_on_disk}"
    assert n_after_join == 1000, (
        f"[{name}] inner join shrank the panel: {n_images_on_disk} images -> "
        f"{n_after_join} rows after merging with prompts_df (case_number mismatch?)"
    )
    # correctness flags against this row's OWN true class (strict + semantic,
    # identical logic to compute_accuracy_by_class, but kept per-image here)
    merged["pred_top1_lower"] = merged["category_top1"].str.strip().str.lower()
    merged["true_class_lower"] = merged["class"].str.strip().str.lower()
    merged["correct_strict"] = merged["pred_top1_lower"] == merged["true_class_lower"]
    def _semantic_ok(row):
        if row["correct_strict"]:
            return True
        eq = SEMANTIC_EQUIVALENT_CLASSES.get(row["true_class_lower"])
        if eq is None:
            return False
        return row["pred_top1_lower"] in [c.lower() for c in eq]
    merged["correct_semantic"] = merged.apply(_semantic_ok, axis=1)

    keep_cols = ["case_number", "class", "prompt", "evaluation_seed", "filename",
                 "correct_strict", "correct_semantic"] + \
                [f"category_top{k}" for k in range(1, 6)] + \
                [f"scores_top{k}" for k in range(1, 6)]
    merged = merged[keep_cols].sort_values(["case_number"]).reset_index(drop=True)
    out_csv = os.path.join(OUT_DIR, f"{name}_per_image.csv")
    merged.to_csv(out_csv, index=False)
    per_endpoint_df[name] = merged
    print(f"  saved {out_csv} ({len(merged)} rows)")

# Per-class semantic recognition counts (out of 100) for every endpoint, used for
# the M0 and single-edit eligibility gates -- recorded in full regardless of outcome.
summary = {}
for name, df in per_endpoint_df.items():
    by_class = df.groupby("class")["correct_semantic"].sum().astype(int).to_dict()
    by_class_strict = df.groupby("class")["correct_strict"].sum().astype(int).to_dict()
    summary[name] = {"semantic_correct_per_class": by_class, "strict_correct_per_class": by_class_strict}

with open(os.path.join(OUT_DIR, "recognition_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Recognition counts (semantic / strict), per class, per endpoint ===")
for name in ENDPOINTS:
    print(f"\n{name}:")
    for cls in sorted(summary[name]["semantic_correct_per_class"]):
        sem = summary[name]["semantic_correct_per_class"][cls]
        strict = summary[name]["strict_correct_per_class"][cls]
        print(f"  {cls:20s} semantic={sem:3d}/100  strict={strict:3d}/100")

print("\nDone.")
