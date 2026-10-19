#!/usr/bin/env python3
"""Build the frozen qualification manifest and the case/noise/trajectory manifest.

Metadata and annotations only. No image is opened, no detector or FLUX is run. The only tensor work
is CPU randn for the shared image-noise hashes (a pure function of the frozen seed).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ["CPRD2_ROOT"], "code"))
from cprd2 import protocol as P
from cprd2.common import PROJECT_ID, ROOT, dump, h_ranked, load, sha256_file, utc

ANN = os.path.join(ROOT, "coco", "annotations", "instances_val2017.json")
from cprd2.noise import NOISE_SHAPE, noise_hash


def build_qualification():
    ann = json.load(open(ANN))
    cats = {c["name"]: c for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    design = load(os.path.join(ROOT, "configs", "design.json"))
    by_img_cls = {}
    for a in ann["annotations"]:
        by_img_cls.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(a)
    out = {"project_id": PROJECT_ID, "generated_utc": utc(), "annotation_sha256": sha256_file(ANN),
           "ordering_key": "SHA256('|'.join([project_id, class_id, image_id]))", "classes": {}}
    for nm in P.SCORED_CLASSES:
        cid = cats[nm]["id"]
        sup = cats[nm]["supercategory"]
        same_sup_ids = {c["id"] for c in ann["categories"] if c["supercategory"] == sup and c["id"] != cid}
        pos_c, neg_c, conf_c = [], [], []
        for iid, d in by_img_cls.items():
            im = imgs[iid]
            area = im["width"] * im["height"]
            if cid in d:
                if any((not a.get("iscrowd", 0)) and a["bbox"][2] * a["bbox"][3] >= P.QUAL["min_box_frac"] * area
                       for a in d[cid]):
                    pos_c.append(iid)
        for iid, im in imgs.items():
            d = by_img_cls.get(iid, {})
            if cid not in d:                               # no annotation of the class, crowd included
                neg_c.append(iid)
                if any(k in same_sup_ids for k in d):
                    conf_c.append(iid)
        key = lambda i: h_ranked(PROJECT_ID, cid, i)
        pos = sorted(pos_c, key=key)[:P.QUAL["n_pos"]]
        conf = sorted(conf_c, key=key)[:P.QUAL["n_confounders"]]
        conf_set = set(conf)
        rest = [i for i in sorted(neg_c, key=key) if i not in conf_set]
        negs = conf + rest[:P.QUAL["n_neg"] - len(conf)]
        deficit = P.QUAL["n_confounders"] - len(conf)

        def rec(iid, with_gt):
            im = imgs[iid]
            r = {"image_id": iid, "file_name": im["file_name"], "coco_url": im["coco_url"],
                 "width": im["width"], "height": im["height"],
                 "is_supercategory_confounder": iid in conf_set}
            if with_gt:
                gts = [a for a in by_img_cls[iid][cid] if not a.get("iscrowd", 0)]
                r["gt_boxes_xyxy"] = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                       a["bbox"][1] + a["bbox"][3]] for a in gts]
                r["gt_annotation_ids"] = [a["id"] for a in gts]
                r["gt_min_box_frac_max"] = max(a["bbox"][2] * a["bbox"][3] for a in gts) / (im["width"] * im["height"])
            return r
        out["classes"][nm] = {
            "coco_id": cid, "supercategory": sup,
            "n_pool_positive": len(pos_c), "n_pool_negative": len(neg_c), "n_pool_confounders": len(conf_c),
            "confounder_deficit": deficit,
            "positives": [rec(i, True) for i in pos], "negatives": [rec(i, False) for i in negs]}
        assert len(pos) == 20 and len(negs) == 20, (nm, len(pos), len(negs))
    uniq = {r["image_id"] for v in out["classes"].values() for g in ("positives", "negatives") for r in v[g]}
    out["n_unique_photographs"] = len(uniq)
    assert len(uniq) <= 600, len(uniq)
    dump(out, os.path.join(ROOT, "configs", "qualification_manifest.json"))
    print("qualification manifest: unique photographs", len(uniq),
          {k: v["confounder_deficit"] for k, v in out["classes"].items()})


def mk_case(stage_tag, stage, kind, cls, j, base=P.EVAL_BASE):
    idx = P.prompt_index(stage, cls, j) if kind == "allowed" else j
    seed = P.ev_seed(stage_tag, kind, cls, j, base=base)
    return {"case_id": f"{stage_tag}.{kind}.{cls.replace(' ', '_')}.{j:02d}", "kind": kind, "cls": cls,
            "prompt": P.TEMPLATES[cls][idx], "prompt_index": idx, "eval_seed": seed,
            "noise_sha256": noise_hash(seed)}


def cases_for(target, stage, stage_tag, n_t, n_a):
    out = [mk_case(stage_tag, stage, "target", target, i) for i in range(n_t)]
    for cls in P.PANELS[target]:
        out += [mk_case(stage_tag, stage, "allowed", cls, j) for j in range(n_a)]
    return out


def build_cases():
    dev = cases_for(P.DEV_TARGET, "dev", "dev", 12, 3)
    ho = {t: cases_for(t, "ho", f"ho.{t}", 20, 5) for t in P.HELDOUT_TARGETS}
    tech = [mk_case("tech", "dev", "target", P.DEV_TARGET, 0, base=P.TECH_BASE),
            mk_case("tech", "dev", "target", P.DEV_TARGET, 1, base=P.TECH_BASE),
            mk_case("tech", "dev", "allowed", "oven", 0, base=P.TECH_BASE),
            mk_case("tech", "dev", "allowed", "bus", 0, base=P.TECH_BASE)]
    assert len(dev) == 24 and all(len(v) == 40 for v in ho.values()) and len(tech) == 4
    seeds = [c["eval_seed"] for c in dev + tech] + [c["eval_seed"] for v in ho.values() for c in v]
    assert len(set(seeds)) == len(seeds), "duplicate eval seed"
    assert not (set(seeds) & {P.TRAIN_SEED_DEV, *P.TRAIN_SEEDS_HELDOUT})
    dev_traj = [{"traj_id": f"dev.{a}.{ln}", "arm": a, "lr_name": ln, "lr": lr, "target": P.DEV_TARGET,
                 "train_seed": P.TRAIN_SEED_DEV}
                for a in P.ARMS for ln, lr in P.SETTINGS["learning_rates"].items()]
    ho_traj = [{"traj_id": f"ho.{a}.{t}.s{s}", "arm": a, "target": t, "train_seed": s}
               for a in P.ARMS for t in P.HELDOUT_TARGETS for s in P.TRAIN_SEEDS_HELDOUT]
    counts = {"dev_trajectories": len(dev_traj), "dev_edited": len(dev_traj) * 3 * len(dev),
              "dev_m0": len(dev), "heldout_endpoints": len(ho_traj),
              "heldout_edited": len(ho_traj) * 40, "heldout_m0": 80,
              "technical_batch": len(tech)}
    counts["primary_total"] = counts["dev_edited"] + counts["dev_m0"] + counts["heldout_edited"] + counts["heldout_m0"]
    assert counts["dev_edited"] == 864 and counts["heldout_edited"] == 960 and counts["primary_total"] == 1928, counts
    bundle = {"project_id": PROJECT_ID, "arms": P.ARMS, "dev_target": P.DEV_TARGET,
              "heldout_targets": P.HELDOUT_TARGETS, "panels": P.PANELS, "dev_cases": dev,
              "heldout_cases": ho, "technical_cases": tech, "dev_trajectories": dev_traj,
              "heldout_trajectories": ho_traj, "counts": counts,
              "train_seed_dev": P.TRAIN_SEED_DEV, "train_seeds_heldout": P.TRAIN_SEEDS_HELDOUT,
              "bootstrap_seed": P.BOOTSTRAP_SEED, "noise_shape_unpacked": list(NOISE_SHAPE),
              "noise_hash_definition": "sha256 of float32 bytes of the PACKED CPU torch.randn(1,16,64,64; "
                                       "torch.Generator('cpu').manual_seed(eval_seed))"}
    dump(bundle, os.path.join(ROOT, "configs", "cases.json"))
    print("cases manifest", counts, "bootstrap_seed", P.BOOTSTRAP_SEED)


if __name__ == "__main__":
    build_qualification()
    build_cases()
