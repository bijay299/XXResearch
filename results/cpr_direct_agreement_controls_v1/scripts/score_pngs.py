#!/usr/bin/env python3
"""Score saved PNGs with BOTH frozen detectors: output flags + full postprocessed predictions per image.
Usage: score_pngs.py <plan.json> <out.json>; plan = list of {label,image_dir,cases:[...]} (same format as gen plans)."""
import json
import os
import sys

from cprd2 import detect as D
from cprd2 import labels as LB
from cprd2 import protocol as P
from cprd2.common import ROOT, dump, load, utc


def main(plan_path, out_path):
    import time
    plan = load(plan_path)
    t_load = time.time()
    dets = {t: D.Detector(t, "cuda:0") for t in (P.PRIMARY_DET, P.SECONDARY_DET)}
    lids = {t: {c: LB.resolve(d.id2label, c) for c in P.SCORED_CLASSES} for t, d in dets.items()}
    out = {"started_utc": utc(), "threshold": P.THRESHOLD, "label_ids": lids, "sets": [],
           "load_seconds": round(time.time() - t_load, 1)}
    t_det, n_img = time.time(), 0
    for s in plan:
        rows = []
        for c in s["cases"]:
            path = os.path.join(ROOT, s["image_dir"], c["case_id"] + ".png")
            flags, im = D.output_flags(path)
            row = {"case_id": c["case_id"], "cls": c["cls"], "kind": c["kind"], "path": os.path.relpath(path, ROOT),
                   "flags": flags, "detections": {}}
            if im is not None:
                for t, d in dets.items():
                    try:
                        row["detections"][t] = d.detect(im)
                    except Exception as ex:                    # noqa: BLE001
                        row["detections"][t] = {"error": repr(ex)[:200]}
            rows.append(row)
            n_img += 1
        out["sets"].append({"label": s["label"], "image_dir": s["image_dir"], "rows": rows})
        print("scored", s["label"], len(rows), flush=True)
    out["detect_seconds_total"] = round(time.time() - t_det, 1)
    out["n_images"] = n_img
    out["finished_utc"] = utc()
    dump(out, out_path)
    print("SCORE_DONE", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
