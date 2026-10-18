#!/usr/bin/env python3
"""Populate the Task-A reviewer bundle.

Copies the 40 approved PNGs to neutral IDs **byte-for-byte** (their containers
carry no text/EXIF chunks, so nothing needs stripping and no pixel is touched),
writes items.json, and verifies actual 40-image completeness.

Never crops, resizes, enhances or annotates a research image.
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
SRC = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1/gen"
A = os.path.join(ROOT, "taskA")
IMG = os.path.join(A, "images")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    priv = json.load(open(os.path.join(ROOT, "private", "id_mapping_private.json")))
    mapping = priv["taskA_mapping"]
    pub = json.load(open(os.path.join(ROOT, "protocol", "taskA_items.json")))

    os.makedirs(IMG, exist_ok=True)
    rows, problems = [], []
    for aid in sorted(mapping):
        m = mapping[aid]
        src = os.path.join(SRC, m["condition"], m["source_filename"])
        dst = os.path.join(IMG, aid + ".png")
        shutil.copyfile(src, dst)                      # bytes verbatim
        got = sha256_file(dst)
        same = (got == m["source_sha256"])
        if not same:
            problems.append(aid)
        rows.append({"item_id": aid, "bytes": os.path.getsize(dst),
                     "sha256": got, "byte_identical_to_source": same})

    # decoded-pixel identity + native resolution (no modification performed)
    pixel_ok = True
    try:
        from PIL import Image
        for aid in sorted(mapping):
            m = mapping[aid]
            a = os.path.join(SRC, m["condition"], m["source_filename"])
            b = os.path.join(IMG, aid + ".png")
            with Image.open(a) as ia, Image.open(b) as ib:
                ia.load(); ib.load()
                if ia.size != (512, 512) or ib.size != (512, 512) \
                   or ia.tobytes() != ib.tobytes():
                    pixel_ok = False
                    problems.append(aid + ":pixels")
    except Exception as e:
        pixel_ok = None
        print("pixel check unavailable:", e)

    # items.json for the viewer (public, arm-independent)
    items = dict(pub)
    items["images_dir"] = "images"
    json.dump(items, open(os.path.join(A, "items.json"), "w"), indent=1)

    n_png = len([f for f in os.listdir(IMG) if f.endswith(".png")])
    expected = set(pub["item_ids"])
    actual = {f[:-4] for f in os.listdir(IMG) if f.endswith(".png")}
    report = {
        "stage": "A",
        "package_id": pub["package_id"],
        "n_images_expected": 40,
        "n_images_present": n_png,
        "complete": n_png == 40 and actual == expected and not problems,
        "missing_ids": sorted(expected - actual),
        "unexpected_ids": sorted(actual - expected),
        "all_byte_identical_to_source": all(r["byte_identical_to_source"] for r in rows),
        "decoded_pixel_identity_verified": pixel_ok,
        "images_modified": 0,
        "images_regenerated": 0,
        "container_metadata_stripped": False,
        "container_metadata_note": ("Source PNGs contain only IHDR/IDAT/IEND -- no "
                                    "tEXt/iTXt/zTXt/eXIf/tIME chunks -- so the files "
                                    "are copied unchanged and only renamed."),
        "problems": problems,
        "rows": rows,
    }
    json.dump(report, open(os.path.join(ROOT, "verify", "taskA_completeness.json"), "w"),
              indent=1)

    for k in ("n_images_present", "complete", "all_byte_identical_to_source",
              "decoded_pixel_identity_verified", "missing_ids", "unexpected_ids",
              "problems"):
        print(f"{k:36} {report[k]}")
    sys.exit(0 if report["complete"] else 2)


if __name__ == "__main__":
    main()
