#!/usr/bin/env python3
"""Populate the PRIVATE Task-B package (20 matched pairs).

Written under private/ with restricted permissions. Never published, never
committed, never embedded in the Task-A bundle. Release requires BOTH that
Task-A collection is closed with originals locked AND explicit later
authorization from the user.

Images are copied byte-for-byte; nothing is cropped, resized, enhanced or
annotated.
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
SRC = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1/gen"
PRIV = os.path.join(ROOT, "private")
B = os.path.join(PRIV, "taskB_package")
IMG = os.path.join(B, "images")
SRCUI = os.path.join(ROOT, "taskB_src")


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    priv = json.load(open(os.path.join(PRIV, "id_mapping_private.json")))
    bmap = priv["taskB_mapping"]

    os.makedirs(IMG, exist_ok=True)
    rows, problems = [], []
    for pid in sorted(bmap):
        m = bmap[pid]
        for role, cond, fn, exp in (
                ("reference", m["reference_condition"], m["reference_filename"], m["reference_sha256"]),
                ("candidate", m["candidate_condition"], m["candidate_filename"], m["candidate_sha256"])):
            src = os.path.join(SRC, cond, fn)
            dst = os.path.join(IMG, f"{pid}_{role}.png")
            shutil.copyfile(src, dst)
            got = sha256_file(dst)
            ok = got == exp
            if not ok:
                problems.append(f"{pid}/{role}")
            rows.append({"pair_id": pid, "role": role, "bytes": os.path.getsize(dst),
                         "sha256": got, "byte_identical_to_source": ok})

    # viewer bundle: items.json carries ONLY neutral pair IDs
    items = {
        "package_id": priv["package_b"],
        "protocol_id": priv["protocol_id"],
        "schema_id": priv["schema_id"],
        "stage": "B",
        "n_items": len(bmap),
        "item_ids": sorted(bmap),
        "images_dir": "images",
        "note": ("Pair IDs are neutral. This bundle deliberately discloses which "
                 "side is the reference, so this stage is only partially masked. "
                 "It carries no case IDs, seeds, prompts, machine scores or "
                 "Task-A associations."),
    }
    json.dump(items, open(os.path.join(B, "items.json"), "w"), indent=1)
    with open(os.path.join(B, "items.js"), "w") as fh:
        fh.write("window.THR_ITEMS = ")
        json.dump(items, fh, indent=1)
        fh.write(";\n")
    for f in ("index.html", "app.js", "style.css"):
        shutil.copyfile(os.path.join(SRCUI, f), os.path.join(B, f))

    n_png = len([f for f in os.listdir(IMG) if f.endswith(".png")])
    expected = {f"{p}_{r}" for p in bmap for r in ("reference", "candidate")}
    actual = {f[:-4] for f in os.listdir(IMG) if f.endswith(".png")}

    # decoded-pixel identity
    pixel_ok = True
    try:
        from PIL import Image
        for pid in sorted(bmap):
            m = bmap[pid]
            for role, cond, fn in (("reference", m["reference_condition"], m["reference_filename"]),
                                   ("candidate", m["candidate_condition"], m["candidate_filename"])):
                a = os.path.join(SRC, cond, fn)
                b = os.path.join(IMG, f"{pid}_{role}.png")
                with Image.open(a) as ia, Image.open(b) as ib:
                    ia.load(); ib.load()
                    if ia.size != (512, 512) or ia.tobytes() != ib.tobytes():
                        pixel_ok = False
                        problems.append(f"{pid}/{role}:pixels")
    except Exception as e:
        pixel_ok = None
        print("pixel check unavailable:", e)

    report = {
        "stage": "B",
        "package_id": priv["package_b"],
        "location": "PRIVATE (work-directory /private/) -- not published",
        "n_pairs_expected": 20,
        "n_pairs_present": len(bmap),
        "n_images_expected": 40,
        "n_images_present": n_png,
        "complete": (len(bmap) == 20 and n_png == 40 and actual == expected and not problems),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "all_byte_identical_to_source": all(r["byte_identical_to_source"] for r in rows),
        "decoded_pixel_identity_verified": pixel_ok,
        "images_modified": 0,
        "images_regenerated": 0,
        "problems": problems,
        "release_preconditions": [
            "All planned Task-A collection is closed and original Task-A responses are locked.",
            "The user gives later explicit authorization.",
        ],
        "released": False,
    }
    # the per-file row detail stays private (it maps pair IDs to real hashes)
    json.dump({**report, "rows": rows},
              open(os.path.join(PRIV, "taskB_completeness_private.json"), "w"), indent=1)
    # a publishable summary with no associations
    json.dump({k: v for k, v in report.items()},
              open(os.path.join(ROOT, "verify", "taskB_completeness_summary.json"), "w"), indent=1)

    for k in ("n_pairs_present", "n_images_present", "complete",
              "all_byte_identical_to_source", "decoded_pixel_identity_verified",
              "missing", "unexpected", "problems"):
        print(f"{k:36} {report[k]}")
    sys.exit(0 if report["complete"] else 2)


if __name__ == "__main__":
    main()
