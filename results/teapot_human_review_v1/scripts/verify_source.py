#!/usr/bin/env python3
"""Verify the 40 approved teapot PNGs against the accepted hashes and pairing
records of the completed modern_object_assets_v1 asset build.

Read-only. Prefers existing verified local copies; never downloads the model,
never regenerates anything, never alters a research image.
"""
import hashlib
import json
import os
import sys

SRC = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1"
ACCEPTED_ARCHIVE = {
    "name": "gem-flux-teapot-v1-images.tar.gz",
    "bytes": 18265516,
    "sha256": "8b254ab3f6cbc64330600c2918bcd921fd9c609b32f1905cc6db7539ebff6420",
}
OUT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1/verify/source_verification.json"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rep = {"source_release_tag": "modern-object-assets-v1",
           "mode": "read-only reuse of approved images and pairing records",
           "model_loaded": False, "images_regenerated": 0, "gpu_seconds": 0.0}

    # ---- 1. local archive identity against the accepted values ----
    arch = os.path.join(SRC, "publish", "assets", ACCEPTED_ARCHIVE["name"])
    if os.path.isfile(arch):
        got_b, got_h = os.path.getsize(arch), sha256_file(arch)
        rep["local_archive"] = {
            "path": arch, "bytes": got_b, "sha256": got_h,
            "bytes_match": got_b == ACCEPTED_ARCHIVE["bytes"],
            "sha256_match": got_h == ACCEPTED_ARCHIVE["sha256"],
        }
    else:
        rep["local_archive"] = {"path": arch, "present": False}

    # ---- 2. pairing records from the accepted asset build ----
    pv = json.load(open(os.path.join(SRC, "manifests", "PAIR_VERIFICATION.json")))
    ni = json.load(open(os.path.join(SRC, "manifests", "initial_noise_index.json")))
    cases = json.load(open(os.path.join(SRC, "manifests", "CASES.json")))["cases"]
    noise = {r["case_id"]: r for r in ni["cases"]}

    rows, problems = [], []
    n_ok = 0
    for r in pv["rows"]:
        cid = r["case_id"]
        rec = {"case_id": cid, "eval_seed": r["eval_seed"],
               "class": r["class"], "category": r["category"],
               "noise_sha256": r.get("noise_sha256"),
               "noise_identical_across_conditions": r.get("noise_identical_across_conditions")}
        for cond, sub in (("m0", "m0"), ("edited", "edited")):
            fn = r[cond]["filename"]
            p = os.path.join(SRC, "gen", sub, fn)
            present = os.path.isfile(p)
            got = sha256_file(p) if present else None
            match = present and got == r[cond]["file_sha256"]
            rec[cond] = {"filename": fn, "present": present,
                         "expected_sha256": r[cond]["file_sha256"],
                         "got_sha256": got, "sha256_match": match,
                         "bytes": os.path.getsize(p) if present else None}
            if match:
                n_ok += 1
            else:
                problems.append(f"{cid}/{cond}")
        # cross-check the frozen case record
        c = next((c for c in cases if c["case_id"] == cid), None)
        rec["case_record_present"] = c is not None
        rec["prompt_matches_frozen_case"] = bool(c and c["prompt"] == r["prompt"])
        rec["seed_matches_frozen_case"] = bool(c and c["eval_seed"] == r["eval_seed"])
        rec["noise_record_present"] = cid in noise
        rows.append(rec)

    rep["cases"] = len(rows)
    rep["pngs_verified"] = n_ok
    rep["pngs_expected"] = 40
    rep["all_pngs_match_accepted_hashes"] = (n_ok == 40 and not problems)
    rep["problems"] = problems
    rep["all_pairing_records_present"] = all(
        x["case_record_present"] and x["noise_record_present"]
        and x["prompt_matches_frozen_case"] and x["seed_matches_frozen_case"]
        for x in rows)
    rep["all_noise_identical_across_conditions"] = all(
        x["noise_identical_across_conditions"] for x in rows)
    rep["distinct_noise_hashes"] = len({x["noise_sha256"] for x in rows})
    rep["rows"] = rows

    # ---- 3. decoded-pixel readability + native resolution (no modification) ----
    try:
        from PIL import Image
        sizes, modes = set(), set()
        for r in rows:
            for cond, sub in (("m0", "m0"), ("edited", "edited")):
                p = os.path.join(SRC, "gen", sub, r[cond]["filename"])
                with Image.open(p) as im:
                    im.load()
                    sizes.add(im.size)
                    modes.add(im.mode)
        rep["decoded_sizes"] = sorted(str(s) for s in sizes)
        rep["decoded_modes"] = sorted(modes)
        rep["all_native_512"] = sizes == {(512, 512)}
    except Exception as e:
        rep["decode_error"] = f"{type(e).__name__}: {e}"

    ok = (rep["all_pngs_match_accepted_hashes"] and rep["all_pairing_records_present"]
          and rep.get("all_native_512") and rep["distinct_noise_hashes"] == 20)
    rep["verdict"] = "SOURCE_VERIFIED" if ok else "SOURCE_VERIFICATION_FAILED"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    for k in ("verdict", "cases", "pngs_verified", "all_pngs_match_accepted_hashes",
              "all_pairing_records_present", "all_noise_identical_across_conditions",
              "distinct_noise_hashes", "all_native_512", "problems"):
        print(f"{k:38} {rep.get(k)}")
    print(f"{'local_archive sha256_match':38} {rep['local_archive'].get('sha256_match')}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
