"""Canonical COCO category -> detector label mapping, resolved from each
detector's own id2label. Raises on ambiguity; never guesses an id."""

NAME_SYNONYMS = {
    "motorcycle": {"motorcycle", "motorbike"},
    "airplane": {"airplane", "aeroplane"},
    "couch": {"couch", "sofa"},
    "tv": {"tv", "tvmonitor", "tv monitor"},
    "dining table": {"dining table", "diningtable"},
    "potted plant": {"potted plant", "pottedplant"},
}


def variants(coco_name):
    return {v.lower() for v in NAME_SYNONYMS.get(coco_name, {coco_name})}


def resolve(id2label, coco_name):
    want = variants(coco_name)
    hits = [int(k) for k, v in id2label.items() if str(v).strip().lower() in want]
    if len(hits) != 1:
        raise KeyError(f"{coco_name!r} resolves to {hits} (variants {sorted(want)})")
    return hits[0]


def build_map(id2label, coco_categories):
    """{coco_id: detector_label_id} for supported categories + unsupported names."""
    out, unsupported = {}, []
    for c in coco_categories:
        try:
            out[int(c["id"])] = resolve(id2label, c["name"])
        except KeyError:
            unsupported.append(c["name"])
    return out, unsupported
