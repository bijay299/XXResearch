#!/usr/bin/env python3
"""Annotation-based evaluator qualification for both frozen detectors.

Downloads only the selected COCO val2017 photographs (counted against the
600-photograph ceiling), runs each detector with its official pre/post
processing at the fixed 0.30 threshold, and applies the frozen operational
gate: for EVERY scored class, >=16/20 positives with a matching detection and
>=16/20 negatives without a detection.

Cases were chosen from annotations alone; nothing here selects by prediction.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ["CPRD_ROOT"])
sys.path.insert(0, os.path.join(os.environ["CPRD_ROOT"], "pydeps"))
import torch
from PIL import Image

from cprd import ledger as LG
from cprd import detect as DT
from cprd import labels as LB

ROOT = os.environ["CPRD_ROOT"]
CFG = json.load(open(os.path.join(ROOT, "configs", "categories_and_qualification.json")))
ANN = json.load(open(os.path.join(ROOT, "coco", "annotations", "instances_val2017.json")))
IMGDIR = os.path.join(ROOT, "coco", "val2017")
MIN_HIT = 16
N = 20


def fetch(images):
    """Download only the needed photographs, once each."""
    import urllib.request
    os.makedirs(IMGDIR, exist_ok=True)
    new = 0
    for rec in images:
        p = os.path.join(IMGDIR, rec["file_name"])
        if os.path.exists(p):
            continue
        urllib.request.urlretrieve(rec["coco_url"], p)
        new += 1
    if new:
        LG.count_coco(new, detail="qualification photographs")
    return new


def main():
    t0 = time.time()
    quals = CFG["qualification"]
    scored = CFG["scored_classes"]
    name2id = {c["name"]: c["id"] for c in ANN["categories"]}

    # ---- gather the unique photograph set and download it ----
    need, seen = [], set()
    for nm in scored:
        for grp in ("positives", "negatives"):
            for r in quals[nm][grp]:
                if r["image_id"] not in seen:
                    seen.add(r["image_id"]); need.append(r)
    print(f"unique photographs required: {len(need)}")
    fetch(need)

    # ---- ground-truth boxes for IoU matching (annotations only) ----
    gt = {}
    for a in ANN["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, hh = a["bbox"]
        gt.setdefault((a["image_id"], a["category_id"]), []).append([x, y, x + w, y + hh])

    out = {"threshold": DT.THRESHOLD, "iou_match": DT.IOU_MATCH,
           "min_hit": MIN_HIT, "n_per_group": N, "detectors": {}}

    for tag in ("rtdetr_r50vd", "detr_resnet50_no_timm"):
        det = DT.Detector(tag, device="cuda:0")
        lmap = LB.build_map(det.id2label, ANN["categories"], tag)
        per_class, raw = {}, {}
        for nm in scored:
            cid = name2id[nm]
            det_lid = lmap["coco_id_to_detector_id"][cid]["detector_label_id"]
            pos_hit = neg_clean = 0
            pos_boxmatch = 0
            rows = []
            for grp, want_present in (("positives", True), ("negatives", False)):
                for rec in quals[nm][grp]:
                    p = os.path.join(IMGDIR, rec["file_name"])
                    try:
                        im = Image.open(p).convert("RGB")
                        r = det.detect([im])[0]
                        miss = False
                    except Exception as exc:                      # missing measurement
                        r, miss = {"scores": [], "labels": [], "label_strs": [],
                                   "boxes_xyxy": []}, f"{type(exc).__name__}: {exc}"
                    d = DT.detected(r, det_lid)
                    bm, best_iou = (DT.box_match(r, det_lid, gt.get((rec["image_id"], cid), []))
                                    if want_present else (None, None))
                    if miss is False:
                        if want_present:
                            pos_hit += int(d)
                            pos_boxmatch += int(bool(bm))
                        else:
                            neg_clean += int(not d)
                    rows.append({"image_id": rec["image_id"], "group": grp,
                                 "detected": d, "best_score": DT.best_score(r, det_lid),
                                 "box_match_iou50": bm, "best_iou": best_iou,
                                 "missing_measurement": miss,
                                 "is_supercategory_confounder": rec.get("is_supercategory_confounder"),
                                 "n_detections": len(r["labels"])})
            passed = pos_hit >= MIN_HIT and neg_clean >= MIN_HIT
            per_class[nm] = {"coco_id": cid, "detector_label_id": det_lid,
                             "positives_detected": pos_hit,
                             "positives_box_matched_iou50": pos_boxmatch,
                             "negatives_clean": neg_clean,
                             "n_positives": N, "n_negatives": N,
                             "sensitivity": pos_hit / N, "specificity": neg_clean / N,
                             "passed": bool(passed)}
            raw[nm] = rows
            print(f"  [{tag}] {nm:14s} pos {pos_hit}/{N} (box {pos_boxmatch}/{N})  "
                  f"neg {neg_clean}/{N}  {'PASS' if passed else 'FAIL'}")
        allp = all(v["passed"] for v in per_class.values())
        out["detectors"][tag] = {
            "spec": DT.DETECTORS[tag], "label_map": lmap,
            "per_class": per_class, "all_classes_passed": bool(allp),
            "verdict": ("QUALIFIED" if allp else "NOT_QUALIFIED"),
            "failing_classes": sorted(k for k, v in per_class.items() if not v["passed"]),
        }
        json.dump(raw, open(os.path.join(ROOT, "artifacts", f"qualification_raw_{tag}.json"), "w"),
                  indent=2, sort_keys=True, default=str)
        del det
        torch.cuda.empty_cache()

    out["label_ids_differ_across_detectors"] = LB.check_scored_classes(
        {t: out["detectors"][t]["label_map"] for t in out["detectors"]},
        scored, name2id)
    out["primary_verdict"] = out["detectors"]["rtdetr_r50vd"]["verdict"]
    out["secondary_verdict"] = out["detectors"]["detr_resnet50_no_timm"]["verdict"]
    out["limits"] = [
        "Both detectors were developed and evaluated on COCO; this is a "
        "benchmark/implementation check, NOT independent out-of-domain validation.",
        "Annotation-negative is an imperfect absence reference.",
        "Real-photo qualification does NOT establish accuracy on edited generations.",
        "Detector agreement is not ground truth.",
        "Prompts, Claude judgments and the failed pilot's calibration images are "
        "not qualification labels and were not used.",
        "Detector scores are not calibrated presence probabilities and the two "
        "models' score scales are not directly comparable.",
    ]
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "qualification.json"), "w"),
              indent=2, sort_keys=True, default=str)
    print(f"\nPRIMARY   {out['primary_verdict']}")
    print(f"SECONDARY {out['secondary_verdict']}")
    return 0 if out["primary_verdict"] == "QUALIFIED" else 2


if __name__ == "__main__":
    sys.exit(main())
