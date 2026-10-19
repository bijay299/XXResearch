#!/usr/bin/env python3
"""Detector qualification on the frozen manifest (20 positives + 20 annotation-negatives per scored class).
Binding assertions (both detectors, EVERY scored class): >=16/20 pos_boxmatch (correct class, IoU>=0.50 with a
reference box) AND >=16/20 neg_clean (no threshold-reaching detection of the class). Technical errors are
MISSING measurements: they stay in the denominator of 20 and entry fails; no substitution."""
import json
import os
import sys
import urllib.request

from PIL import Image

from cprd2 import detect as D
from cprd2 import labels as LB
from cprd2 import protocol as P
from cprd2.common import ROOT, dump, load, sha256_file, utc

PHOTO_DIR = os.path.join(ROOT, "coco", "val2017")


def fetch(entry):
    path = os.path.join(PHOTO_DIR, entry["file_name"])
    if not os.path.exists(path):
        os.makedirs(PHOTO_DIR, exist_ok=True)
        req = urllib.request.Request(entry["coco_url"], headers={"User-Agent": "cpr-direct-agreement-controls-v1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path


def main():
    qm = load(os.path.join(ROOT, "configs", "qualification_manifest.json"))
    out = {"project_id": "cpr_direct_agreement_controls_v1", "started_utc": utc(),
           "coordinate_convention": "boxes are [x0,y0,x1,y1] in ORIGINAL photograph pixels (official "
                                    "post_process_object_detection with target_sizes=(H,W)); reference boxes are "
                                    "COCO bbox [x,y,w,h] converted to xyxy",
           "threshold": P.THRESHOLD, "iou_match": P.IOU_MATCH, "detectors": {}, "photographs": {}}
    unique = {}
    for cls, v in qm["classes"].items():
        for k in ("positives", "negatives"):
            for e in v[k]:
                unique[e["file_name"]] = e
    paths, fetch_errors = {}, {}
    for fn, e in sorted(unique.items()):
        try:
            paths[fn] = fetch(e)
        except Exception as ex:                               # noqa: BLE001
            fetch_errors[fn] = repr(ex)[:200]
    out["n_unique_photographs"] = len(unique)
    out["fetch_errors"] = fetch_errors
    out["photograph_sha256"] = {fn: sha256_file(p) for fn, p in paths.items()}
    all_pass = True
    for tag in (P.PRIMARY_DET, P.SECONDARY_DET):
        det = D.Detector(tag, "cuda:0")
        res = {"repo": det.spec["repo"], "revision": det.spec["revision"], "classes": {}, "label_ids": {}}
        pred_cache = {}
        errs = {}
        for fn, p in paths.items():
            try:
                im = Image.open(p).convert("RGB")
                pred_cache[fn] = det.detect(im) | {"image_size_wh": [im.width, im.height]}
            except Exception as ex:                           # noqa: BLE001
                errs[fn] = repr(ex)[:200]
        res["technical_errors"] = errs
        for cls, v in qm["classes"].items():
            lid = LB.resolve(det.id2label, cls)
            res["label_ids"][cls] = {"label_id": lid, "label_str": det.id2label[lid]}
            pos_rows, neg_rows = [], []
            for e in v["positives"]:
                fn = e["file_name"]
                row = {"image_id": e["image_id"], "file_name": fn, "gt_annotation_ids": e["gt_annotation_ids"],
                       "gt_boxes_xyxy": e["gt_boxes_xyxy"], "role": "positive"}
                if fn in pred_cache:
                    m, bi, bidx = D.box_match(pred_cache[fn], lid, e["gt_boxes_xyxy"])
                    row.update({"pos_boxmatch": bool(m), "best_iou": bi, "best_det_idx": bidx,
                                "pos_hit_image_level": D.detected(pred_cache[fn], lid), "prediction": pred_cache[fn],
                                "measured": True})
                else:
                    row.update({"measured": False, "pos_boxmatch": None})
                pos_rows.append(row)
            for e in v["negatives"]:
                fn = e["file_name"]
                row = {"image_id": e["image_id"], "file_name": fn, "role": "negative",
                       "is_supercategory_confounder": e["is_supercategory_confounder"]}
                if fn in pred_cache:
                    row.update({"neg_clean": not D.detected(pred_cache[fn], lid), "best_class_score": D.best_score(pred_cache[fn], lid),
                                "prediction": pred_cache[fn], "measured": True})
                else:
                    row.update({"measured": False, "neg_clean": None})
                neg_rows.append(row)
            n_pos = sum(1 for r in pos_rows if r["pos_boxmatch"] is True)
            n_neg = sum(1 for r in neg_rows if r["neg_clean"] is True)
            cp = {"n_positives": len(pos_rows), "n_negatives": len(neg_rows),
                  "pos_measured": sum(r["measured"] for r in pos_rows), "neg_measured": sum(r["measured"] for r in neg_rows),
                  "pos_boxmatch_count": n_pos, "neg_clean_count": n_neg,
                  "confounder_negatives": sum(1 for r in neg_rows if r["is_supercategory_confounder"]),
                  "confounder_deficit": v["confounder_deficit"],
                  "pass": bool(n_pos >= P.QUAL["min_pos_boxmatch"] and n_neg >= P.QUAL["min_neg_clean"]
                               and len(pos_rows) == P.QUAL["n_pos"] and len(neg_rows) == P.QUAL["n_neg"]),
                  "positives": pos_rows, "negatives": neg_rows}
            res["classes"][cls] = cp
            all_pass &= cp["pass"]
            print(f"[{'PASS' if cp['pass'] else 'FAIL'}] {tag} {cls} pos {n_pos}/20 neg {n_neg}/20", flush=True)
        out["detectors"][tag] = res
        del det
    out["entry_qualification_pass"] = bool(all_pass and not fetch_errors
                                           and all(not r["technical_errors"] for r in out["detectors"].values()))
    out["finished_utc"] = utc()
    os.makedirs(os.path.join(ROOT, "artifacts", "qualification"), exist_ok=True)
    dump(out, os.path.join(ROOT, "artifacts", "qualification", "qualification_full.json"))
    summ = {t: {c: {k: v for k, v in cp.items() if k not in ("positives", "negatives")} for c, cp in d["classes"].items()}
            for t, d in out["detectors"].items()}
    dump({"entry_qualification_pass": out["entry_qualification_pass"], "n_unique_photographs": len(unique),
          "summary": summ}, os.path.join(ROOT, "artifacts", "qualification", "qualification_summary.json"))
    print("QUALIFICATION", "PASS" if out["entry_qualification_pass"] else "FAIL")
    if not out["entry_qualification_pass"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
