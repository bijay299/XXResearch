#!/usr/bin/env python3
"""
Builds the two COCO-derived assets this sprint needs, with frozen seeds.

(1) prompts/coco_train2014_30k_prompts.txt -- the file referenced by the BLOCK release's
    get_activation_diffusion_mlp.py:161 but ABSENT from the release (gap A4). The pooled-text
    (modulation MLP) transcoder's training corpus. Paper E.1: "we instead sample 30,000 prompts
    from MS-COCO and apply the same style augmentation, resulting in 330,000 training pairs."
    The authors' exact 30k caption list is not published, so it is reconstructed by a documented
    seeded sample. This is a recorded deviation, not a claim of identity.

(2) Calibration panels for the binary target-presence scorer: per object, 20 target-present and
    20 target-absent COCO val2014 images with INDEPENDENT ground-truth labels from COCO's own
    human instance annotations (not from prompts, filenames, or a second pass of the judge).
    Half of each negative panel is "related-object" (another animal, no target) so the panel
    directly probes related-animal confusion.

ALL SELECTION RULES HERE ARE FROZEN BEFORE ANY SCORER OUTPUT IS SEEN.
"""
import json, random, hashlib, os, sys, argparse

ROOT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
ANN = f"{ROOT}/assets/coco/annotations"

# ---- frozen parameters -------------------------------------------------------
CAPTION_SEED = 42            # matches the release's own random.seed(42) convention
N_CAPTIONS = 30_000
PANEL_SEED = 20261018
N_POS = 20                   # gate is ">=18/20 sensitivity", so the panel is exactly 20
N_NEG = 20                   # ">=18/20 specificity"
N_NEG_RELATED = 10           # half the negatives must contain a related animal
MIN_AREA_FRAC = 0.02         # a positive's largest target instance must cover >=2% of the image
                             # so that "target present" is not a pixel-hunt; fixed in advance

# sprint objects -> COCO category names.  UnlearnCanvas class -> COCO category.
TARGETS = {"Bears": "bear", "Cats": "cat", "Horses": "horse"}
# "related" = other animals; used to build related-object negatives for every target
ANIMALS = ["bear", "cat", "dog", "horse", "sheep", "cow", "elephant", "zebra", "giraffe", "bird"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_captions():
    with open(f"{ANN}/captions_train2014.json") as f:
        d = json.load(f)
    # one caption per image: the lowest annotation id, so the choice is deterministic
    per_img = {}
    for a in d["annotations"]:
        i = a["image_id"]
        if i not in per_img or a["id"] < per_img[i]["id"]:
            per_img[i] = a
    img_ids = sorted(per_img)
    rng = random.Random(CAPTION_SEED)
    pick = rng.sample(img_ids, N_CAPTIONS)
    out_dir = f"{ROOT}/code/block-concept-removal/prompts"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/coco_train2014_30k_prompts.txt"
    rows = []
    with open(path, "w") as f:
        for i in pick:
            c = " ".join(per_img[i]["caption"].split())   # collapse whitespace/newlines
            f.write(c + "\n")
            rows.append({"coco_image_id": i, "caption_id": per_img[i]["id"], "caption": c})
    with open(f"{ROOT}/assets/coco/coco_30k_provenance.json", "w") as f:
        json.dump({"source": "COCO captions_train2014.json",
                   "zip_sha256_of_annotations": "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009",
                   "rule": "one caption per image (lowest annotation id); random.Random(42).sample over sorted image ids",
                   "n": N_CAPTIONS, "rows": rows}, f)
    print(f"[captions] wrote {path}  n={len(pick)}  sha256={sha256(path)}")
    print(f"[captions] -> style augmentation at load time yields {len(pick)*11} training pairs")
    return path


def build_panels():
    with open(f"{ANN}/instances_val2014.json") as f:
        d = json.load(f)
    cats = {c["name"]: c["id"] for c in d["categories"]}
    licenses = {l["id"]: l for l in d["licenses"]}
    imgs = {im["id"]: im for im in d["images"]}

    # image_id -> set of category ids present (iscrowd instances excluded from "present"
    # evidence but still counted as contamination for negatives)
    present, crowd = {}, {}
    area_of = {}
    for a in d["annotations"]:
        i, c = a["image_id"], a["category_id"]
        if a.get("iscrowd", 0):
            crowd.setdefault(i, set()).add(c)
            continue
        present.setdefault(i, set()).add(c)
        key = (i, c)
        area_of[key] = max(area_of.get(key, 0.0), a["area"])

    animal_ids = {cats[a] for a in ANIMALS}
    out = {}
    rng = random.Random(PANEL_SEED)

    for obj, catname in TARGETS.items():
        tid = cats[catname]
        # ---- positives: target present, largest instance >= MIN_AREA_FRAC of image
        pos_pool = []
        for i, cs in present.items():
            if tid not in cs:
                continue
            im = imgs[i]
            frac = area_of[(i, tid)] / float(im["width"] * im["height"])
            if frac >= MIN_AREA_FRAC:
                pos_pool.append((i, frac))
        pos_pool.sort()
        pos = rng.sample([i for i, _ in pos_pool], N_POS)

        # ---- negatives: target absent entirely (no instance, no crowd instance)
        def has_target(i):
            return tid in present.get(i, set()) or tid in crowd.get(i, set())

        rel_pool, plain_pool = [], []
        for i in sorted(imgs):
            if has_target(i):
                continue
            cs = present.get(i, set())
            if not cs:
                continue
            others = cs & animal_ids
            if others:
                # related-object negative: contains some OTHER animal, never the target
                frac = max(area_of[(i, c)] for c in others) / float(imgs[i]["width"] * imgs[i]["height"])
                if frac >= MIN_AREA_FRAC:
                    rel_pool.append(i)
            elif not (cs & animal_ids):
                plain_pool.append(i)
        neg_rel = rng.sample(sorted(rel_pool), N_NEG_RELATED)
        neg_plain = rng.sample(sorted(plain_pool), N_NEG - N_NEG_RELATED)

        def row(i, label, kind):
            im = imgs[i]
            cs = sorted(present.get(i, set()) | crowd.get(i, set()))
            names = sorted({c["name"] for c in d["categories"] if c["id"] in cs})
            return {"coco_image_id": i, "file_name": im["file_name"],
                    "coco_url": im["coco_url"], "flickr_url": im.get("flickr_url"),
                    "license_id": im["license"], "license_name": licenses[im["license"]]["name"],
                    "license_url": licenses[im["license"]]["url"],
                    "ground_truth_target_present": label, "panel": kind,
                    "coco_annotated_categories": names}

        rows = ([row(i, True, "positive") for i in pos]
                + [row(i, False, "negative_related_animal") for i in neg_rel]
                + [row(i, False, "negative_other") for i in neg_plain])
        out[obj] = {"target_coco_category": catname, "rows": rows}
        nrel = len(neg_rel)
        print(f"[panel {obj:7s}] pos={len(pos)} (pool {len(pos_pool)})  "
              f"neg={len(neg_rel)+len(neg_plain)} (related {nrel}, other {len(neg_plain)})")

    meta = {"source": "COCO instances_val2014.json (human instance annotations)",
            "independence": "labels come from COCO annotators, not from prompts, filenames, or the judge",
            "frozen_rules": {"panel_seed": PANEL_SEED, "n_pos": N_POS, "n_neg": N_NEG,
                             "n_neg_related": N_NEG_RELATED, "min_area_frac": MIN_AREA_FRAC,
                             "positive": ">=1 non-crowd target instance with largest area >= 2% of image",
                             "negative": "zero target instances of any kind (incl. iscrowd)",
                             "negative_related_animal": "contains another animal >=2% area, never the target"},
            "targets": TARGETS, "related_animal_set": ANIMALS, "panels": out}
    p = f"{ROOT}/assets/coco/calibration_panels.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[panels] wrote {p}  sha256={sha256(p)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["captions", "panels", "both"], default="both")
    a = ap.parse_args()
    if a.what in ("captions", "both"):
        build_captions()
    if a.what in ("panels", "both"):
        build_panels()
