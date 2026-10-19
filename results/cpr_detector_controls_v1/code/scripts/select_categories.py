#!/usr/bin/env python3
"""Annotation-only category selection and qualification-case construction.

METADATA ONLY. No detector, no model, no image is loaded or decoded here.
Cases are NEVER selected by detector predictions.
"""
import hashlib
import json
import os
import sys

ROOT = os.environ["CPRD_ROOT"]
ANN = os.path.join(ROOT, "coco", "annotations", "instances_val2017.json")
PROJECT_ID = "cpr_detector_controls_v1"

# ordered candidate list, exactly as given by the contract
CANDIDATES = ["traffic light", "stop sign", "parking meter", "bench", "fire hydrant",
              "microwave", "refrigerator", "oven", "sink", "toilet", "bed",
              "dining table", "couch", "train", "bus", "truck", "boat",
              "tennis racket", "baseball bat", "surfboard"]

# Categories closed by ACTUALLY EXECUTED earlier work (see CLOSED_CATEGORIES.md).
CLOSED_EXECUTED = {
    # executed erasure targets
    "english springer", "french horn", "frida kahlo", "golf ball",
    "katsushika hokusai", "vincent van gogh", "bear", "bears", "cat", "cats",
    "horse", "horses", "dog", "sandwich", "teapot",
    # executed retention-panel classes (seq17/seq29)
    "bird", "chair", "bicycle",
    # v1 pilot: calibration images were actually generated for these three
    "backpack", "violin",
}

MIN_POS = 20
MIN_NEG = 20
MIN_BOX_FRACTION = 0.05
N_CONFOUNDER_TARGET = 10


def h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def main():
    ann = json.load(open(ANN))
    cats = {c["id"]: c for c in ann["categories"]}
    name2id = {c["name"]: c["id"] for c in cats.values()}
    imgs = {i["id"]: i for i in ann["images"]}
    supercat = {c["name"]: c["supercategory"] for c in cats.values()}

    # ---- per-class annotation index (metadata only) ----
    pos_by_cat = {cid: set() for cid in cats}          # >=5% noncrowd box
    any_by_cat = {cid: set() for cid in cats}          # ANY annotation incl. crowd
    for a in ann["annotations"]:
        cid, iid = a["category_id"], a["image_id"]
        any_by_cat[cid].add(iid)
        if a.get("iscrowd", 0) == 0:
            im = imgs[iid]
            frac = a["bbox"][2] * a["bbox"][3] / float(im["width"] * im["height"])
            if frac >= MIN_BOX_FRACTION:
                pos_by_cat[cid].add(iid)
    all_ids = set(imgs)

    def eligible(name):
        cid = name2id[name]
        return len(pos_by_cat[cid]) >= MIN_POS and len(all_ids - any_by_cat[cid]) >= MIN_NEG

    report = {"project_id": PROJECT_ID,
              "annotation_sha256": hashlib.sha256(open(ANN, "rb").read()).hexdigest(),
              "n_images": len(imgs), "n_categories": len(cats),
              "min_box_fraction": MIN_BOX_FRACTION,
              "closed_executed": sorted(CLOSED_EXECUTED), "candidate_scan": []}

    for nm in CANDIDATES:
        cid = name2id.get(nm)
        row = {"name": nm, "coco_id": cid,
               "supercategory": supercat.get(nm),
               "closed": nm.lower() in CLOSED_EXECUTED,
               "n_positive_ge5pct": len(pos_by_cat[cid]) if cid else 0,
               "n_annotation_negative": len(all_ids - any_by_cat[cid]) if cid else 0}
        row["metadata_eligible"] = bool(cid) and not row["closed"] and eligible(nm)
        report["candidate_scan"].append(row)

    # ---- first three eligible, non-colliding, in fixed list order ----
    chosen = [r["name"] for r in report["candidate_scan"] if r["metadata_eligible"]][:3]
    assert len(chosen) == 3, f"could not find three eligible targets: {chosen}"
    dev_target, ho1, ho2 = chosen
    targets = [dev_target, ho1, ho2]
    report["targets"] = {"development": dev_target, "held_out": [ho1, ho2]}

    # ---- allowed panels: 2 same-supercategory + 2 other, ascending COCO id ----
    banned = {t.lower() for t in targets} | CLOSED_EXECUTED
    def panel(target):
        """Two same-supercategory, then two other, ascending COCO id.

        If the supercategory is exhausted (all three targets here are
        `outdoor`, and the remaining outdoor classes fail the >=20-positive
        floor), the shortfall is filled by CONTINUING down the `other` list in
        ascending COCO id. The deficit is predeclared and disclosed rather than
        silently shrinking the panel.
        """
        sc = supercat[target]
        same = sorted((c for c in cats.values()
                       if c["supercategory"] == sc and c["name"].lower() not in banned
                       and eligible(c["name"])), key=lambda c: c["id"])[:2]
        other_pool = sorted((c for c in cats.values()
                             if c["supercategory"] != sc and c["name"].lower() not in banned
                             and eligible(c["name"])), key=lambda c: c["id"])
        need = 4 - len(same)
        other = other_pool[:need]
        return ([c["name"] for c in same] + [c["name"] for c in other],
                {"same_supercategory_used": [c["name"] for c in same],
                 "same_supercategory_deficit": 2 - len(same),
                 "deficit_filled_from_other_classes": [c["name"] for c in other[2 - len(same):]]
                 if len(same) < 2 else []})

    panels, panel_meta = {}, {}
    for t in targets:
        panels[t], panel_meta[t] = panel(t)
    report["allowed_panels"] = panels
    report["allowed_panel_construction"] = panel_meta
    for t, p in panels.items():
        assert len(p) == 4, f"panel for {t} is {p}"
        assert not (set(p) & set(targets)), f"panel for {t} collides with a target"

    scored = sorted({*targets, *(c for p in panels.values() for c in p)})
    report["scored_classes"] = scored
    report["n_scored_classes"] = len(scored)
    assert len(scored) <= 15, f"too many scored classes: {len(scored)}"

    # ---- deterministic qualification cases, metadata only ----
    quals, used_images = {}, set()
    for nm in scored:
        cid = name2id[nm]
        sc = supercat[nm]
        conf_cats = [c["id"] for c in cats.values()
                     if c["supercategory"] == sc and c["id"] != cid]
        conf_pool = set().union(*[any_by_cat[c] for c in conf_cats]) if conf_cats else set()

        pos = sorted(pos_by_cat[cid], key=lambda i: h(PROJECT_ID, cid, i))[:MIN_POS]
        neg_all = all_ids - any_by_cat[cid]
        # same-supercategory confounder negatives first, then plain negatives
        conf = sorted(neg_all & conf_pool, key=lambda i: h(PROJECT_ID, cid, i))
        plain = sorted(neg_all - conf_pool, key=lambda i: h(PROJECT_ID, cid, i))
        n_conf = min(N_CONFOUNDER_TARGET, len(conf))
        neg = conf[:n_conf] + plain[:MIN_NEG - n_conf]
        deficit = N_CONFOUNDER_TARGET - n_conf
        assert len(pos) == MIN_POS and len(neg) == MIN_NEG, nm
        used_images |= set(pos) | set(neg)
        quals[nm] = {
            "coco_id": cid, "supercategory": sc,
            "n_confounder_negatives": n_conf,
            "confounder_deficit_filled_from_plain_negatives": deficit,
            "positives": [{"image_id": i, "file_name": imgs[i]["file_name"],
                           "coco_url": imgs[i]["coco_url"],
                           "license": imgs[i]["license"],
                           "width": imgs[i]["width"], "height": imgs[i]["height"]}
                          for i in pos],
            "negatives": [{"image_id": i, "file_name": imgs[i]["file_name"],
                           "coco_url": imgs[i]["coco_url"],
                           "license": imgs[i]["license"],
                           "width": imgs[i]["width"], "height": imgs[i]["height"],
                           "is_supercategory_confounder": i in conf_pool}
                          for i in neg],
        }

    report["qualification"] = quals
    report["n_image_class_pairs"] = sum(len(v["positives"]) + len(v["negatives"])
                                        for v in quals.values())
    report["n_unique_photographs"] = len(used_images)
    report["licenses"] = {str(l["id"]): l["name"] for l in ann["licenses"]}
    assert report["n_image_class_pairs"] <= 600, report["n_image_class_pairs"]
    assert report["n_unique_photographs"] <= 600, report["n_unique_photographs"]

    out = os.path.join(ROOT, "configs", "categories_and_qualification.json")
    blob = json.dumps(report, indent=2, sort_keys=True)
    open(out, "w").write(blob)
    print("wrote", out, "sha256", hashlib.sha256(blob.encode()).hexdigest())
    print("\ncandidate scan (ordered):")
    for r in report["candidate_scan"]:
        if r["closed"]: flag = "CLOSED"
        elif r["metadata_eligible"]: flag = "ELIGIBLE"
        elif r["n_positive_ge5pct"] < 20: flag = f"ineligible(pos {r['n_positive_ge5pct']}<20)"
        else: flag = "ineligible"
        print(f"  {r['name']:16s} id={str(r['coco_id']):>3s} super={str(r['supercategory']):10s} "
              f"pos>=5%={r['n_positive_ge5pct']:>4d} neg={r['n_annotation_negative']:>4d}  {flag}")
    print(f"\nTARGETS  dev={dev_target}  held-out={ho1}, {ho2}")
    for t, p in panels.items():
        print(f"  allowed[{t:14s}] = {p}")
    print(f"scored classes ({len(scored)}): {scored}")
    print(f"image/class pairs: {report['n_image_class_pairs']}   unique photos: {report['n_unique_photographs']}")
    for nm, v in quals.items():
        print(f"  {nm:16s} confounder_negs={v['n_confounder_negatives']:>2d} "
              f"deficit_filled={v['confounder_deficit_filled_from_plain_negatives']}")


if __name__ == "__main__":
    sys.exit(main())
