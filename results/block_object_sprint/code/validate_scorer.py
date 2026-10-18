#!/usr/bin/env python3
"""Validates the FROZEN binary presence rule against independently annotated public images.

Ground truth = COCO val2014 human instance annotations (not prompts, not filenames, not a
second pass of the judge, not any closed survey labels). 20 target-present and 20
target-absent images per object, 10 of each negative panel containing a DIFFERENT animal,
selected by rules fixed in code/build_coco_assets.py before any scorer output was seen.

GATE: >= 18/20 sensitivity AND >= 18/20 specificity, SEPARATELY for every object.
"uncertain" and unparseable answers count as ERRORS.

Builds the manifest, runs the scorer, writes confusion counts. The rule is NOT tuned here:
it is run once, as frozen, and the result is reported whatever it is.
"""
import os, json, subprocess, sys, argparse

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
NOUN = {"Bears": "bear", "Cats": "cat", "Horses": "horse"}


def build_manifest():
    panels = json.load(open(f"{SPRINT}/assets/coco/calibration_panels.json"))
    items = []
    for obj, p in panels["panels"].items():
        for r in p["rows"]:
            img = f"{SPRINT}/assets/calib_images/{obj}/{r['file_name']}"
            if not os.path.exists(img):
                raise SystemExit(f"missing calibration image {img}")
            items.append({
                "id": f"calib|{obj}|{r['coco_image_id']}",
                "image": img, "object": obj,
                "target_noun": NOUN[obj],
                "ground_truth_target_present": r["ground_truth_target_present"],
                "panel": r["panel"],
                "coco_image_id": r["coco_image_id"],
                "coco_annotated_categories": r["coco_annotated_categories"],
            })
    m = f"{SPRINT}/artifacts/scorer_validation/manifest.json"
    os.makedirs(os.path.dirname(m), exist_ok=True)
    json.dump(items, open(m, "w"), indent=1)
    print(f"manifest: {len(items)} images -> {m}")
    return m


def summarise(jsonl):
    rows = [json.loads(l) for l in open(jsonl)]
    out = {"gate": {"sensitivity_min": "18/20", "specificity_min": "18/20",
                    "uncertain_counts_as_error": True},
           "per_object": {}, "overall_pass": True}
    for obj in sorted({r["object"] for r in rows}):
        rs = [r for r in rows if r["object"] == obj]
        pos = [r for r in rs if r["ground_truth_target_present"]]
        neg = [r for r in rs if not r["ground_truth_target_present"]]
        tp = sum(1 for r in pos if r["presence_target"] == "yes")
        fn = sum(1 for r in pos if r["presence_target"] == "no")
        pu = sum(1 for r in pos if r["presence_target"] == "uncertain")
        tn = sum(1 for r in neg if r["presence_target"] == "no")
        fp = sum(1 for r in neg if r["presence_target"] == "yes")
        nu = sum(1 for r in neg if r["presence_target"] == "uncertain")
        rel = [r for r in neg if r["panel"] == "negative_related_animal"]
        rel_fp = sum(1 for r in rel if r["presence_target"] == "yes")
        rel_unc = sum(1 for r in rel if r["presence_target"] == "uncertain")
        d = {"n_pos": len(pos), "n_neg": len(neg),
             "sensitivity_correct": tp, "sensitivity_denom": len(pos),
             "specificity_correct": tn, "specificity_denom": len(neg),
             "confusion": {"TP": tp, "FN": fn, "pos_uncertain": pu,
                           "TN": tn, "FP": fp, "neg_uncertain": nu},
             "related_animal_negatives": {"n": len(rel), "false_positive": rel_fp,
                                          "uncertain": rel_unc},
             "sensitivity_pass": tp >= 18, "specificity_pass": tn >= 18}
        d["pass"] = d["sensitivity_pass"] and d["specificity_pass"]
        out["per_object"][obj] = d
        out["overall_pass"] &= d["pass"]
        print(f"{obj:7s} sens {tp}/{len(pos)} (FN {fn}, unc {pu})  "
              f"spec {tn}/{len(neg)} (FP {fp}, unc {nu})  "
              f"related-animal FP {rel_fp}/{len(rel)}  -> {'PASS' if d['pass'] else 'FAIL'}")
    out["caveat"] = ("These are small frozen controls on real photographs. They do NOT "
                     "establish universal accuracy, and they do not establish validity on "
                     "the synthetic, heavily stylised FLUX domain being measured.")
    p = f"{SPRINT}/artifacts/scorer_validation/validation_summary.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\noverall_pass={out['overall_pass']} -> {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["manifest", "summarise"], required=True)
    ap.add_argument("--jsonl", default=f"{SPRINT}/artifacts/scorer_validation/answers.jsonl")
    a = ap.parse_args()
    if a.stage == "manifest":
        build_manifest()
    else:
        summarise(a.jsonl)
