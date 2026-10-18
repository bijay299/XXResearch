#!/usr/bin/env python3
"""Audit every Task-B public artifact before release.

Checks the ZIP members, the staged previews and the B documentation for
anything that must stay private: the A-to-B/source crosswalk, case IDs, seeds,
prompts, source filenames, the salt, private mapping fields, Task-A responses,
the reviewer roster/aliases, answer keys and synthetic test exports.

Task B intentionally reveals matched pairs and reference identity; that is the
point of the stage and is NOT treated as a leak. What must not appear is the
link back to the source case/condition filenames or any private material.
"""
import hashlib
import json
import os
import re
import struct
import sys
import zipfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
SRCMAN = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1/manifests"
ZPATH = os.path.join(ROOT, "publishB", "teapot-human-review-v1-taskB.zip")
PREVIEWS = os.path.join(ROOT, "publishB", "preview_assets")

# The locked-A roster is private. Aliases must never appear in public artifacts.
ROSTER = ["los", "Shree"]

PATTERNS = [
    ("source filename", re.compile(r"CASE-\d{2}_(m0|edited)_seed\d+\.png")),
    ("case id", re.compile(r"\bCASE-\d{2}\b")),
    ("eval seed", re.compile(r"\bseed ?7\d{3}\b", re.I)),
    ("source condition token", re.compile(r"_(m0|edited)_seed")),
    ("crosswalk key", re.compile(r'"(reference_filename|candidate_filename|'
                                 r'reference_sha256|candidate_sha256|case_id|'
                                 r'taskA_mapping|taskB_mapping|source_filename)"')),
    ("answer key", re.compile(r'"(answer_key|ground_truth|gold|label_key)"')),
    ("A item id", re.compile(r"\bA-\d{3}\b")),
]


def png_meta(data):
    found = []
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return ["not-a-png"]
    i = 8
    while i + 8 <= len(data):
        ln, typ = struct.unpack(">I4s", data[i:i + 8])
        t = typ.decode("latin1")
        if t in ("tEXt", "iTXt", "zTXt", "eXIf", "tIME"):
            found.append(t)
        i += 8 + ln + 4
        if t == "IEND":
            break
    return found


def main():
    prompts = [c["prompt"] for c in
               json.load(open(os.path.join(SRCMAN, "CASES.json")))["cases"]]
    salt_hex = open(os.path.join(ROOT, "private", "salt.bin"), "rb").read().hex()

    rep = {"zip": ZPATH, "findings": [], "previews": {}, "docs": {}}
    hits = []

    with zipfile.ZipFile(ZPATH) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        pngs = [n for n in names if n.endswith(".png")]
        pair_ids = sorted({os.path.basename(n).rsplit("_", 1)[0] for n in pngs})
        roles = sorted({os.path.basename(n).rsplit("_", 1)[1][:-4] for n in pngs})
        meta = {}
        for n in names:
            data = z.read(n)
            if n.endswith(".png"):
                m = png_meta(data)
                if m:
                    meta[n] = m
                continue
            try:
                txt = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            is_code = n.endswith((".py", ".js", ".css", ".html"))
            for label, rx in PATTERNS:
                if is_code and label == "crosswalk key":
                    continue
                for mm in rx.finditer(txt):
                    hits.append({"member": n, "kind": label,
                                 "match": mm.group(0)[:60]})
            for pr in prompts:
                if len(pr) >= 15 and pr.lower() in txt.lower():
                    hits.append({"member": n, "kind": "verbatim case prompt"})
            if salt_hex in txt:
                hits.append({"member": n, "kind": "SALT"})
            for a in ROSTER:
                if re.search(r"\b" + re.escape(a) + r"\b", txt):
                    hits.append({"member": n, "kind": "reviewer alias"})
            if re.search(r'"_fixture"|SYNTHETIC QA FIXTURE', txt):
                hits.append({"member": n, "kind": "synthetic test export"})
            if '"export_kind"' in txt:
                hits.append({"member": n, "kind": "completed export in bundle"})

        rep["zip_inventory"] = {
            "n_members": len(names), "n_png": len(pngs),
            "n_pairs": len(pair_ids), "pair_ids_contiguous":
                pair_ids == [f"B-{i:02d}" for i in range(1, 21)],
            "roles": roles, "roles_exact": roles == ["candidate", "reference"],
            "two_images_per_pair": all(
                sum(1 for n in pngs if os.path.basename(n).startswith(p + "_")) == 2
                for p in pair_ids),
            "support_files": sorted(os.path.basename(n) for n in names
                                    if "/images/" not in n),
            "png_metadata_chunks": meta,
            "png_containers_clean": not meta,
        }

    # previews
    pv = {}
    for f in sorted(os.listdir(PREVIEWS)) if os.path.isdir(PREVIEWS) else []:
        p = os.path.join(PREVIEWS, f)
        pv[f] = {"bytes": os.path.getsize(p),
                 "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    allowed = {"preview_B1_instructions.png", "preview_B2_pair_view_synthetic.png"}
    rep["previews"] = {"staged": sorted(pv), "identities": pv,
                       "disallowed": sorted(set(pv) - allowed)}

    rep["findings"] = hits
    rep["clean"] = (not hits
                    and rep["zip_inventory"]["pair_ids_contiguous"]
                    and rep["zip_inventory"]["roles_exact"]
                    and rep["zip_inventory"]["two_images_per_pair"]
                    and rep["zip_inventory"]["png_containers_clean"]
                    and not rep["previews"]["disallowed"])
    rep["intentional_disclosure"] = (
        "Task B intentionally reveals matched pairs and reference identity. It "
        "does NOT conceal pairing or condition roles. Because the 40 images are "
        "already public, this can enable matching to the public Task-A images "
        "even though no crosswalk is published. Publishing on GitHub does not "
        "restrict access to any particular people; eligibility is procedural.")
    json.dump(rep, open(os.path.join(ROOT, "relB", "taskB_public_audit.json"), "w"),
              indent=1)
    inv = rep["zip_inventory"]
    print(f"  members={inv['n_members']} png={inv['n_png']} pairs={inv['n_pairs']} "
          f"contiguous={inv['pair_ids_contiguous']}")
    print(f"  roles={inv['roles']} exact={inv['roles_exact']} "
          f"two_per_pair={inv['two_images_per_pair']}")
    print(f"  support={inv['support_files']}")
    print(f"  png containers clean={inv['png_containers_clean']}")
    print(f"  previews staged={rep['previews']['staged']} "
          f"disallowed={rep['previews']['disallowed']}")
    print(f"  findings={len(hits)} {hits[:3]}")
    print(f"  CLEAN={rep['clean']}")
    sys.exit(0 if rep["clean"] else 5)


if __name__ == "__main__":
    main()
