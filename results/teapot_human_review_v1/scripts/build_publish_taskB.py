#!/usr/bin/env python3
"""Build the public Task-B reviewer ZIP. B-only: never touches Task A.

Deterministic archive (fixed member timestamps) so the published hash is
reproducible. Previews come from an EXPLICIT ALLOWLIST -- never a directory
glob -- and the item-view preview uses a clearly synthetic placeholder.
"""
import hashlib
import json
import os
import shutil
import zipfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
SRC = os.path.join(ROOT, "private", "taskB_package")
OUT = os.path.join(ROOT, "publishB")
ZIPNAME = "teapot-human-review-v1-taskB.zip"
ZROOT = "teapot-human-review-v1-taskB"
FIXED = (2026, 10, 18, 0, 0, 0)

PUBLISHABLE_PREVIEWS = [
    "B1_instructions.png",
    "B2_pair_view_synthetic.png",
]


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    zpath = os.path.join(OUT, ZIPNAME)
    if os.path.exists(zpath):
        os.remove(zpath)

    files = ["index.html", "app.js", "items.js", "style.css", "items.json"]
    imgs = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
                  if f.endswith(".png"))
    assert len(imgs) == 40, f"expected 40 images, found {len(imgs)}"
    pair_ids = sorted({f.rsplit("_", 1)[0] for f in imgs})
    assert len(pair_ids) == 20, f"expected 20 pairs, found {len(pair_ids)}"
    for pid in pair_ids:
        for role in ("reference", "candidate"):
            assert f"{pid}_{role}.png" in imgs, f"missing {pid}_{role}.png"

    def add(z, data, arc):
        zi = zipfile.ZipInfo(arc, date_time=FIXED)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, data)

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            add(z, open(os.path.join(SRC, f), "rb").read(), f"{ZROOT}/{f}")
        add(z, open(os.path.join(ROOT, "protocol",
                                 "INSTRUCTIONS_TASK_B.md"), "rb").read(),
            f"{ZROOT}/INSTRUCTIONS.md")
        for im in imgs:
            add(z, open(os.path.join(SRC, "images", im), "rb").read(),
                f"{ZROOT}/images/{im}")
        add(z, ("Task B reviewer bundle\n"
                "======================\n\n"
                "Extract this folder fully, then open index.html in a browser.\n"
                "It works entirely offline: no account, no upload, no tracking.\n"
                "Your answers stay in your browser until you export a file.\n\n"
                "This stage SHOWS which image is the reference (original model)\n"
                "and which is the candidate (modified model). Please leave your\n"
                "Task A answers exactly as you submitted them.\n\n"
                "See INSTRUCTIONS.md for what to do.\n\n"
                "package THR-V1-B | protocol TEAPOT-HR-PROTO-1.0.0 | "
                "schema THR-EXPORT-SCHEMA-1.1.0 | app 1.1.0 taskB-b1 | "
                "20 pairs / 40 images\n").encode(),
            f"{ZROOT}/README.txt")

    # allowlisted previews only
    stage = os.path.join(OUT, "preview_assets")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    shots = os.path.join(ROOT, "qa", "screenshotsB")
    staged, skipped = [], []
    if os.path.isdir(shots):
        for f in sorted(os.listdir(shots)):
            if f in PUBLISHABLE_PREVIEWS:
                shutil.copyfile(os.path.join(shots, f),
                                os.path.join(stage, "preview_" + f))
                staged.append("preview_" + f)
            else:
                skipped.append(f)
    if skipped:
        print(f"[publishB] NOT staging {len(skipped)} non-allowlisted "
              f"screenshot(s): {skipped}")

    ident = {
        "release_asset": ZIPNAME, "zip_root": ZROOT,
        "bytes": os.path.getsize(zpath), "sha256": sha256_file(zpath),
        "n_members": 40 + len(files) + 2, "n_images": 40, "n_pairs": 20,
        "package_id": "THR-V1-B", "protocol_id": "TEAPOT-HR-PROTO-1.0.0",
        "schema_id": "THR-EXPORT-SCHEMA-1.1.0",
        "app_version": "1.1.0", "app_build": "taskB-b1",
        "deterministic_build": True,
        "contains_task_a": False, "contains_answer_key": False,
        "contains_mapping_or_salt": False,
        "contains_reviewer_roster_or_responses": False,
        "staged_previews": staged,
        "partially_unmasked": True,
        "disclosure": ("This bundle intentionally reveals matched pairs and "
                       "reference identity. It does not conceal pairing or "
                       "condition roles, and publishing it on GitHub does not "
                       "restrict access to any particular people."),
    }
    json.dump(ident, open(os.path.join(OUT, "TASKB_ZIP_IDENTITY.json"), "w"),
              indent=1)
    print(f"zip: {ZIPNAME}  {ident['bytes']:,} bytes  sha256={ident['sha256']}")
    print(f"members={ident['n_members']} pairs={ident['n_pairs']} "
          f"images={ident['n_images']} previews={staged}")


if __name__ == "__main__":
    main()
