#!/usr/bin/env python3
"""Verify the UNCHANGED private mappings before they are used to join responses.

Reads the private mapping and salt, re-computes the two published commitments,
and checks the structural properties the analysis contract (§6) requires:
one-to-one neutral IDs, 40 Task-A images, 20 Task-B cases, m0 = reference and
edited = candidate in every pair, and Task-A/Task-B case agreement.

Nothing is regenerated. Nothing secret is printed or written: the public
verification record carries verdicts, counts and the already-published
commitment digests only -- no key, no salt, no crosswalk, no filename.
"""
import datetime
import hashlib
import hmac
import json
import os
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
PRIV = os.path.join(ROOT, "private")
PUB = "/data/bijaypandey/xxresearch_standalone/results/teapot_human_review_v1"
OUT = os.path.join(ROOT, "analysis_v1", "mapping_verification.json")


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def main():
    mp = json.load(open(os.path.join(PRIV, "id_mapping_private.json")))
    salt = open(os.path.join(PRIV, "salt.bin"), "rb").read()
    pubc = json.load(open(os.path.join(PUB, "protocol", "COMMITMENTS.json")))
    a_items = json.load(open(os.path.join(PUB, "protocol", "taskA_items.json")))

    a_map, b_map = mp["taskA_mapping"], mp["taskB_mapping"]
    by_label = {c["label"]: c["hmac_sha256"] for c in pubc["commitments"]}

    def recommit(blob):
        return hmac.new(salt, blob, hashlib.sha256).hexdigest()

    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    # --- commitments still match the untouched mapping -----------------------
    ra = recommit(canon(a_map))
    rb = recommit(canon(b_map))
    ids = sorted(a_map.keys())
    ri = recommit(canon(ids))
    chk("taskA_mapping_40_items commitment matches",
        ra == by_label["taskA_mapping_40_items"], by_label["taskA_mapping_40_items"])
    chk("taskB_mapping_20_pairs commitment matches",
        rb == by_label["taskB_mapping_20_pairs"], by_label["taskB_mapping_20_pairs"])
    chk("taskA_item_ids commitment matches",
        ri == by_label["taskA_item_ids"], by_label["taskA_item_ids"])

    # --- counts --------------------------------------------------------------
    chk("Task-A mapping has 40 images", len(a_map) == 40, str(len(a_map)))
    chk("Task-B mapping has 20 cases", len(b_map) == 20, str(len(b_map)))
    chk("Task-A IDs are A-001..A-040 contiguous",
        ids == ["A-%03d" % i for i in range(1, 41)])
    chk("Task-B IDs are B-01..B-20 contiguous",
        sorted(b_map) == ["B-%02d" % i for i in range(1, 21)])
    chk("public taskA_items.json agrees with the private mapping IDs",
        sorted(a_items["item_ids"]) == ids)

    # --- one-to-one ----------------------------------------------------------
    a_files = [v["source_filename"] for v in a_map.values()]
    a_hashes = [v["source_sha256"] for v in a_map.values()]
    chk("Task-A source filenames are one-to-one with neutral IDs",
        len(set(a_files)) == 40, "%d distinct" % len(set(a_files)))
    chk("Task-A source hashes are 40 distinct images",
        len(set(a_hashes)) == 40, "%d distinct" % len(set(a_hashes)))

    conds = sorted({v["condition"] for v in a_map.values()})
    nM0 = sum(1 for v in a_map.values() if v["condition"] == "m0")
    nED = 40 - nM0
    chk("Task-A conditions are exactly {m0, edited}", conds == ["edited", "m0"],
        ",".join(conds))
    chk("Task-A splits 20 m0 / 20 edited", nM0 == 20 and nED == 20,
        "m0=%d edited=%d" % (nM0, nED))

    # --- roles: m0 = reference, edited = candidate ---------------------------
    role_ok = sum(1 for v in b_map.values()
                  if v["reference_condition"] == "m0"
                  and v["candidate_condition"] == "edited")
    chk("every pair has m0 = reference and edited = candidate", role_ok == 20,
        "%d / 20" % role_ok)

    # --- Task-A <-> Task-B case agreement ------------------------------------
    a_by_case = {}
    for iid, v in a_map.items():
        a_by_case.setdefault(v["case_id"], {})[v["condition"]] = (
            iid, v["source_sha256"], v.get("class"), v.get("category"))
    chk("Task-A covers 20 distinct cases", len(a_by_case) == 20,
        "%d cases" % len(a_by_case))
    chk("every Task-A case has exactly one m0 and one edited image",
        all(sorted(d) == ["edited", "m0"] for d in a_by_case.values()))

    b_cases = {v["case_id"] for v in b_map.values()}
    chk("Task-B case set equals the Task-A case set",
        b_cases == set(a_by_case), "%d vs %d" % (len(b_cases), len(a_by_case)))

    hash_match = 0
    meta_match = 0
    for pid, v in b_map.items():
        d = a_by_case.get(v["case_id"])
        if not d:
            continue
        if (d["m0"][1] == v["reference_sha256"]
                and d["edited"][1] == v["candidate_sha256"]):
            hash_match += 1
        if d["m0"][2] == v.get("class") and d["m0"][3] == v.get("category"):
            meta_match += 1
    chk("pair reference/candidate hashes equal the Task-A m0/edited hashes",
        hash_match == 20, "%d / 20" % hash_match)
    chk("pair class/category agree with the Task-A record", meta_match == 20,
        "%d / 20" % meta_match)

    # --- descriptive strata come from the frozen source record ---------------
    cats, classes = {}, {}
    for v in b_map.values():
        cats[v.get("category")] = cats.get(v.get("category"), 0) + 1
        classes[v.get("class")] = classes.get(v.get("class"), 0) + 1
    chk("case categories recorded in the frozen mapping sum to 20",
        sum(cats.values()) == 20, json.dumps(cats, sort_keys=True))
    chk("case strata (ordinary/challenging) recorded in the frozen mapping "
        "sum to 20", sum(classes.values()) == 20,
        json.dumps(classes, sort_keys=True))

    ok = all(c["ok"] for c in checks)
    out = {
        "generated_iso": datetime.datetime.now().astimezone().isoformat(),
        "what_this_is": (
            "Host-performed verification that the private Task-A/Task-B "
            "mappings are unchanged and structurally sound before they are used "
            "to join responses. Mappings were NOT regenerated. No key, salt, "
            "filename or neutral-ID-to-condition association is published."),
        "mapping_generated_iso": mp.get("generated_iso"),
        "mapping_schema_id_recorded": mp.get("schema_id"),
        "commitment_scheme": pubc["scheme"],
        "committed_iso": pubc["committed_iso"],
        "counts": {"taskA_images": len(a_map), "taskB_cases": len(b_map),
                   "m0_images": nM0, "edited_images": nED},
        "case_category_counts": dict(sorted(cats.items())),
        "case_class_counts": dict(sorted(classes.items())),
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["ok"]),
        "verdict": "MAPPINGS_VERIFIED_UNCHANGED" if ok else "MAPPING_BLOCKER",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"]
              + (("  [" + c["detail"] + "]") if c["detail"] else ""))
    print("\n%d/%d  %s" % (out["n_passed"], out["n_checks"], out["verdict"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
