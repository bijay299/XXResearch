"""Frozen canonical COCO <-> detector label mapping.

The two detectors DO NOT share a label space. Measured on the pinned
revisions:

  * `PekingU/rtdetr_r50vd`   -> 80 contiguous ids 0..79, VOC-style names for
                                two classes (`motorbike`, `aeroplane`).
  * `facebook/detr-resnet-50`-> 91 ids following COCO category ids 1..90,
                                with 11 unused `N/A` slots.

Example of the trap this module exists to prevent: `stop sign` is id **11** in
RT-DETR and id **13** in DETR; `fire hydrant` is **10** vs **11**. Nothing here
assumes the two id spaces agree -- every lookup is resolved from each model's
own `id2label`, keyed by the canonical COCO category id.
"""

# Detector label strings that denote the same COCO category under a different
# name. Frozen; applied in BOTH directions when resolving.
NAME_SYNONYMS = {
    "motorcycle": {"motorcycle", "motorbike"},
    "airplane": {"airplane", "aeroplane"},
    "couch": {"couch", "sofa"},
    "tv": {"tv", "tvmonitor", "tv monitor"},
    "dining table": {"dining table", "diningtable"},
    "potted plant": {"potted plant", "pottedplant"},
}


def _variants(coco_name):
    return {v.lower() for v in NAME_SYNONYMS.get(coco_name, {coco_name})}


def resolve(id2label, coco_name):
    """Return the detector-internal label id for a canonical COCO class name.

    Raises if the class is unsupported -- never guesses an id.
    """
    want = _variants(coco_name)
    hits = [int(k) for k, v in id2label.items() if str(v).strip().lower() in want]
    if len(hits) != 1:
        raise KeyError(
            f"canonical COCO class {coco_name!r} resolves to {hits} in this "
            f"detector's id2label (expected exactly one); variants tried={sorted(want)}")
    return hits[0]


def build_map(id2label, coco_categories, detector_tag):
    """Map every canonical COCO category id -> this detector's label id.

    `coco_categories` is the official `categories` list from
    `instances_val2017.json`. Unsupported categories are recorded, not
    silently dropped.
    """
    out, unsupported = {}, []
    for c in coco_categories:
        try:
            out[int(c["id"])] = {"coco_name": c["name"],
                                 "detector_label_id": resolve(id2label, c["name"]),
                                 "detector_label_str": None}
        except KeyError:
            unsupported.append(c["name"])
    for cid, rec in out.items():
        rec["detector_label_str"] = str(id2label[rec["detector_label_id"]]
                                        if rec["detector_label_id"] in id2label
                                        else id2label[str(rec["detector_label_id"])])
    unused = sorted(int(k) for k, v in id2label.items()
                    if str(v).strip().lower() in ("n/a", "na", "none", "__background__"))
    return {"detector": detector_tag,
            "n_detector_labels": len(id2label),
            "coco_id_to_detector_id": out,
            "unsupported_coco_categories": unsupported,
            "unused_detector_label_ids": unused}


def check_scored_classes(maps, scored_names, coco_name_to_id):
    """Every scored class must resolve in EVERY detector, or the study cannot
    use it. Returns a per-detector table and a hard verdict."""
    table, ok = {}, True
    for tag, m in maps.items():
        row = {}
        for nm in scored_names:
            cid = coco_name_to_id[nm]
            rec = m["coco_id_to_detector_id"].get(cid)
            row[nm] = None if rec is None else rec["detector_label_id"]
            ok &= rec is not None
        table[tag] = row
    agree = {nm: len({table[t][nm] for t in table}) == 1 for nm in scored_names}
    return {"per_detector_label_ids": table,
            "label_ids_agree_across_detectors": agree,
            "all_scored_classes_supported": bool(ok)}
