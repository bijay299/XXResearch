#!/usr/bin/env python3
"""METADATA-ONLY design selection. No image is opened; no model, detector or
GPU is touched. Reads: official COCO val2017 instance annotations, each
detector's id2label from its pinned config.json, and committed records of
closed work at the accepted HEAD (via `git show`).

Outputs configs/exclusion_inventory.json and configs/design.json.
Status `DESIGN_BLOCKED` is published without expanding the search.
"""
import csv
import io
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.environ["CPRD2_ROOT"], "code"))
from cprd2.common import (ACCEPTED_HEAD, PROJECT_ID, REPO, ROOT, dump, sha256_file, utc)
from cprd2.labels import build_map

ANN = os.path.join(ROOT, "coco", "annotations", "instances_val2017.json")
HUB = "/data/bijaypandey/cuig_pilot/hf_home/hub"
DETECTOR_CONFIGS = {
    "rtdetr_r50vd": f"{HUB}/models--PekingU--rtdetr_r50vd/snapshots/df939e661d8c52e80608d1ec566561aabd25a4e7/config.json",
    "detr_resnet50_no_timm": f"{HUB}/models--facebook--detr-resnet-50/snapshots/70120ba84d68ca1211e007c4fb61d0cd5424be54/config.json",
}
CANDIDATES = ["traffic light", "stop sign", "parking meter", "bench", "fire hydrant",
              "microwave", "refrigerator", "oven", "sink", "toilet", "bed",
              "dining table", "couch", "train", "bus", "truck", "boat",
              "tennis racket", "baseball bat", "surfboard"]
MIN_POS, MIN_NEG, MIN_BOX_FRAC = 20, 20, 0.05


def git_show(path):
    return subprocess.check_output(["git", "-C", REPO, "show", f"{ACCEPTED_HEAD}:{path}"])


# ---------------------------------------------------------------- inventory
def build_inventory():
    inv = {"project_id": PROJECT_ID, "source_commit": ACCEPTED_HEAD, "generated_utc": utc(),
           "rule": "COCO classes excluded as new targets AND as panel classes. Incidental "
                   "scene words alone are not historical panel membership.",
           "entries": [], "considered_not_excluded": []}
    verified = {}

    def seq_cats(study):
        p = f"results/{study}/per_image.csv"
        rows = list(csv.DictReader(io.StringIO(git_show(p).decode())))
        return p, sorted({r["category"] for r in rows})

    def add(coco, reason, path, evidence, alias_of=None):
        inv["entries"].append({"coco_class": coco, "reason": reason, "provenance_path": path,
                               "evidence": evidence, "alias_of": alias_of})

    for study in ("seq17", "seq29"):
        p, cats = seq_cats(study)
        for c in ("bicycle", "bird", "cat", "chair", "dog", "horse", "sandwich"):
            assert c in cats, (study, c, cats)
            add(c, f"executed {study} SD-1.5 sequential pilot target/panel category",
                p, f"`category` column of per_image.csv contains {cats}")
    bs = json.loads(git_show("results/block_object_sprint/cases.json"))
    assert bs["objects"] == {"development": "Bears", "confirmation": ["Cats", "Horses"]}
    assert bs["retained_rule"]["related"] == ["Dogs", "Rabbits"]
    for coco, word in (("bear", "Bears"), ("cat", "Cats"), ("horse", "Horses"), ("dog", "Dogs")):
        add(coco, "executed BLOCK object sprint target/related-panel object (plural alias)",
            "results/block_object_sprint/cases.json", f"objects/retained_rule name {word!r}",
            alias_of=word)
    cal = json.loads(git_show("results/cpr_real_model_controls_v1/artifacts/calibration_images.json"))
    objs = sorted({x["object"] for x in cal if x.get("object")})
    assert objs == ["backpack", "bicycle", "violin"], objs
    for coco in ("backpack", "bicycle"):
        add(coco, "calibration images actually generated in cpr_real_model_controls_v1",
            "results/cpr_real_model_controls_v1/artifacts/calibration_images.json",
            f"calibration positive objects {objs}")
    rece = git_show("results/rece_seq_golfball_englishspringer/CORRECTIONS.md").decode().lower()
    assert "golf ball" in rece and "english springer" in rece
    add("sports ball", "ambiguous COCO alias of executed target `golf ball` (RECE golf-ball -> "
        "English-springer edits)", "results/rece_seq_golfball_englishspringer/CORRECTIONS.md",
        "golf ball and English springer named as executed targets", alias_of="golf ball")
    add("dog", "ambiguous COCO alias of executed target `English springer` (a dog breed)",
        "results/rece_seq_golfball_englishspringer/CORRECTIONS.md",
        "English springer named as executed target", alias_of="english springer")
    prev = json.loads(git_show("results/cpr_detector_controls_v1/configs/categories_and_qualification.json"))
    assert sorted(prev["scored_classes"]) == ["airplane", "bench", "car", "fire hydrant",
                                              "motorcycle", "person", "stop sign"]
    for c in prev["scored_classes"]:
        add(c, "scored class of the closed cpr_detector_controls_v1 (targets + panel)",
            "results/cpr_detector_controls_v1/configs/categories_and_qualification.json",
            f"scored_classes={prev['scored_classes']}")
    add("bench", "ambiguous alias: `park bench` (planned cpr_real_model_controls_v1 panel word)",
        "results/cpr_real_model_controls_v1/code/configs/cases.json", "heldout_allowed_objects contains 'park bench'",
        alias_of="park bench")
    inv["considered_not_excluded"] = [
        {"term": "bird house", "reason": "incidental scene/object word in an unexecuted planned panel; not the COCO class `bird`"},
        {"term": "folding chair", "reason": "incidental planned-panel word, no images generated; not historical membership of `chair` by this word (chair is excluded via seq17/29)"},
        {"term": "teapot, french horn, violin, golf ball, frida kahlo, katsushika hokusai, vincent van gogh",
         "reason": "executed targets with no COCO category (golf ball only via alias `sports ball`, excluded above)"},
        {"term": "cpr_real_model_controls_v1 planned panel objects (suitcase, handbag, umbrella, laptop, skateboard, ...)",
         "reason": "planned and frozen, but the study stopped at its entry gate and generated no panel images; not executed"},
    ]
    names = sorted({e["coco_class"] for e in inv["entries"]})
    inv["excluded_coco_classes"] = names
    return inv


# ------------------------------------------------------------------- design
def main():
    inv = build_inventory()
    dump(inv, os.path.join(ROOT, "configs", "exclusion_inventory.json"))
    excluded = set(inv["excluded_coco_classes"])

    ann = json.load(open(ANN))
    cats = {c["id"]: c for c in ann["categories"]}
    name2id = {c["name"]: c["id"] for c in cats.values()}
    imgs = {i["id"]: i for i in ann["images"]}
    pos, anyc = {c: set() for c in cats}, {c: set() for c in cats}
    for a in ann["annotations"]:
        anyc[a["category_id"]].add(a["image_id"])
        if a.get("iscrowd", 0) == 0:
            im = imgs[a["image_id"]]
            if a["bbox"][2] * a["bbox"][3] / float(im["width"] * im["height"]) >= MIN_BOX_FRAC:
                pos[a["category_id"]].add(a["image_id"])
    all_ids = set(imgs)

    maps, unsup = {}, {}
    for tag, p in DETECTOR_CONFIGS.items():
        id2label = {int(k): v for k, v in json.load(open(p))["id2label"].items()}
        maps[tag], unsup[tag] = build_map(id2label, ann["categories"])

    rows = {}
    for cid, c in cats.items():
        n_pos, n_neg = len(pos[cid]), len(all_ids - anyc[cid])
        supported = all(cid in m for m in maps.values())
        rows[c["name"]] = {"coco_id": cid, "supercategory": c["supercategory"],
                           "n_positive_noncrowd_ge5pct": n_pos, "n_annotation_negative": n_neg,
                           "detector_supported_both": supported,
                           "excluded": c["name"] in excluded,
                           "eligible": (n_pos >= MIN_POS and n_neg >= MIN_NEG and supported
                                        and c["name"] not in excluded)}

    cand_index = {nm: i for i, nm in enumerate(CANDIDATES)}
    enumerated, accepted = [], None
    for i, j, k in itertools.combinations(range(len(CANDIDATES)), 3):
        trio = [CANDIDATES[i], CANDIDATES[j], CANDIDATES[k]]
        rec = {"indices": [i, j, k], "targets": trio, "ok": False, "reason": None}
        bad = [t for t in trio if not rows[t]["eligible"]]
        if bad:
            rec["reason"] = f"ineligible target(s): {bad}"
            enumerated.append(rec)
            continue
        banned = set(trio) | excluded
        panels, fail = {}, None
        for t in trio:
            sc = rows[t]["supercategory"]
            pool = sorted((r["coco_id"], nm) for nm, r in rows.items()
                          if r["eligible"] and nm not in banned)
            same = [nm for cid, nm in pool if rows[nm]["supercategory"] == sc][:2]
            other = [nm for cid, nm in pool if rows[nm]["supercategory"] != sc][:2]
            if len(same) < 2 or len(other) < 2:
                fail = f"target {t}: same-supercategory available={len(same)}, other={len(other)}"
                break
            panels[t] = {"same_supercategory": same, "other_supercategory": other}
        if fail:
            rec["reason"] = fail
            enumerated.append(rec)
            continue
        rec["ok"] = True
        rec["panels"] = panels
        enumerated.append(rec)
        accepted = rec
        break

    out = {"project_id": PROJECT_ID, "generated_utc": utc(),
           "annotation_sha256": sha256_file(ANN), "n_images": len(imgs),
           "detector_unsupported_coco_classes": unsup,
           "detector_configs": {k: os.path.relpath(v, HUB) for k, v in DETECTOR_CONFIGS.items()},
           "excluded_coco_classes": sorted(excluded),
           "class_table": rows, "candidate_order": CANDIDATES,
           "triples_enumerated_until_acceptance": len(enumerated),
           "rejected_triples_sample_first20": enumerated[:20],
           "enumeration_log_path": "configs/design_enumeration.json"}
    dump(enumerated, os.path.join(ROOT, "configs", "design_enumeration.json"))
    if accepted is None:
        out["status"] = "DESIGN_BLOCKED"
    else:
        t = accepted["targets"]
        scored = list(t)
        for tt in t:
            for nm in accepted["panels"][tt]["same_supercategory"] + accepted["panels"][tt]["other_supercategory"]:
                if nm not in scored:
                    scored.append(nm)
        out.update({"status": "DESIGN_SELECTED",
                    "accepted_triple": accepted["indices"],
                    "development_target": t[0], "heldout_targets": t[1:],
                    "allowed_panels": accepted["panels"],
                    "scored_classes": scored, "n_scored_classes": len(scored)})
        assert len(scored) <= 15
    dump(out, os.path.join(ROOT, "configs", "design.json"))
    print(out["status"], out.get("accepted_triple"), out.get("scored_classes"))


if __name__ == "__main__":
    main()
