#!/usr/bin/env python3
"""Builds, per (arm, object), one directory of symlinks holding that object's full frozen
80-case panel, so the release's eval/llava.py can compute UA/IRA/CRA over exactly those
cases. Also writes the manifest the frozen binary presence scorer consumes.

M0 and T0 retained images are the SHARED panel (identical cases for all three objects);
the reuse is recorded in each view's reuse.json rather than regenerated.
"""
import os, json, argparse, hashlib

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
I = f"{SPRINT}/artifacts/images"
NOUN = {"Bears": "bear", "Cats": "cat", "Horses": "horse",
        "Dogs": "dog", "Rabbits": "rabbit",
        "Architectures": "building", "Flowers": "flower"}


def build(arm, obj, target_dir, retained_dir, out):
    spec = json.load(open(f"{SPRINT}/cases.json"))
    os.makedirs(out, exist_ok=True)
    manifest, reuse = [], {"retained_source": retained_dir,
                           "reused_from_shared_panel": "SHARED" in retained_dir}
    for stratum in ("target", "related", "unrelated"):
        cases = (spec["cases"][obj]["target"] if stratum == "target"
                 else spec["shared_retained"][stratum])
        src_dir = target_dir if stratum == "target" else retained_dir
        for c in cases:
            src = os.path.join(src_dir, c["filename"])
            if not os.path.exists(src):
                raise SystemExit(f"missing image {src}")
            dst = os.path.join(out, c["filename"])
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            manifest.append({
                "id": f"{arm}|{obj}|{c['case_id']}",
                "image": os.path.abspath(src),
                "arm": arm, "object": obj, "stratum": stratum,
                "class": c["class"], "theme": c["theme"],
                "init_seed": c["init_seed"], "prompt": c["prompt"],
                "filename": c["filename"],
                # Ask only the question each stratum needs, so no image is queried twice
                # with the identical string (for target cases the intended object IS the
                # erased object). Frozen with the rest of the measurement spec.
                #   target  stratum -> presence_target  + render_failure
                #   retained strata -> presence_intended + render_failure
                "target_noun": NOUN[obj] if stratum == "target" else None,
                "intended_noun": None if stratum == "target" else NOUN[c["class"]],
            })
    json.dump(manifest, open(f"{out}/presence_manifest.json", "w"), indent=1)
    json.dump(reuse, open(f"{out}/reuse.json", "w"), indent=1)
    print(f"{arm}/{obj}: {len(manifest)} cases -> {out}")
    return len(manifest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--retained-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.arm, a.object, a.target_dir, a.retained_dir, a.out)
