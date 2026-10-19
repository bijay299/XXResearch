#!/usr/bin/env python3
"""VISUAL_AUDIT_ADDENDUM_01 tooling. Stage `sample` is a pure function of the frozen case manifest (no model output)."""
import hashlib, json, os, sys
ROOT = os.environ["CPRX_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "code"))
from cprx import protocol as P

N_TARGET, N_PER_ALLOWED = 4, 1


def key(obj, stratum, case_id):
    return f"CPR-VISUAL-01|{obj}|{stratum}|{case_id}"


def select(cases_by_obj):
    out = []
    for obj in P.HELDOUT_TARGETS:
        cs = cases_by_obj[obj]
        strata = [("target", [c for c in cs if c["kind"] == "target"], N_TARGET)]
        for cls in P.PANELS[obj]:
            strata.append((cls, [c for c in cs if c["kind"] == "allowed" and c["cls"] == cls], N_PER_ALLOWED))
        for st, pool, n in strata:
            ranked = sorted(pool, key=lambda c: (hashlib.sha256(key(obj, st, c["case_id"]).encode()).hexdigest(), c["case_id"]))
            for rank, c in enumerate(ranked[:n]):
                out.append({"object": obj, "stratum": st, "rank_in_stratum": rank, "case_id": c["case_id"], "cls": c["cls"],
                            "kind": c["kind"], "prompt": c["prompt"],
                            "sort_key_sha256": hashlib.sha256(key(obj, st, c["case_id"]).encode()).hexdigest()})
            if len(ranked) < n:
                out.append({"object": obj, "stratum": st, "missing_slots": n - len(ranked)})
    return out


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        c = json.load(open(os.path.join(ROOT, "configs", "cases.json")))
        s = select(c["heldout_cases"])
        os.makedirs(os.path.join(ROOT, "visual_audit"), exist_ok=True)
        p = os.path.join(ROOT, "visual_audit", "sample.json")
        json.dump({"rule": "ascending SHA256 of 'CPR-VISUAL-01|test_object|stratum|case_id'; lexical case-id tiebreak; "
                           "4 target + 1 per allowed class per test object; same cases across 6 arms x 2 seeds",
                   "sample": s}, open(p, "w"), indent=1, sort_keys=True)
        print(len([x for x in s if "case_id" in x]), "cases;", sum(x.get("missing_slots", 0) for x in s), "missing slots;",
              hashlib.sha256(open(p, "rb").read()).hexdigest())


# ------------------------------------------------------------------ package / analysis stages (added after FREEZE_01, before any judgment)
def _png_rgb(path):
    from PIL import Image
    import numpy as np
    im = Image.open(path)
    im.load()
    return np.array(im.convert("RGB")), im.size


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def package():
    import random, secrets, shutil
    from PIL import Image
    import numpy as np
    S = json.load(open(os.path.join(ROOT, "visual_audit", "sample.json")))["sample"]
    cases = [x for x in S if "case_id" in x]
    recs = []   # one per underlying image
    for x in cases:
        o = x["object"]
        recs.append({"role": "m0", "arm": None, "seed": None, "object": o, "stratum": x["stratum"], "case_id": x["case_id"],
                     "prompt": x["prompt"], "named_object": x["cls"], "orig": f"images/m0/ho/{o}/{x['case_id']}.png"})
        for arm in P.ARMS:
            for sd in P.TRAIN_SEEDS_HELDOUT:
                recs.append({"role": "edited", "arm": arm, "seed": sd, "object": o, "stratum": x["stratum"], "case_id": x["case_id"],
                             "prompt": x["prompt"], "named_object": x["cls"],
                             "orig": f"images/ho/ho.{arm}.{o}.s{sd}/{x['case_id']}.png"})
    for r in recs:
        r["exists"] = os.path.exists(os.path.join(ROOT, r["orig"]))
        if r["exists"]:
            r["orig_sha256"] = _sha(os.path.join(ROOT, r["orig"]))
    salt = secrets.token_hex(32)
    priv = os.path.join(ROOT, "private_audit")
    pkg = os.path.join(ROOT, "visual_audit_package")
    assert not os.path.exists(pkg) and not os.path.exists(priv), "package/private already exist"
    os.makedirs(os.path.join(pkg, "passA")); os.makedirs(os.path.join(pkg, "passB")); os.makedirs(priv)
    rngA = random.Random(int(hashlib.sha256((salt + "|A").encode()).hexdigest(), 16))
    rngB = random.Random(int(hashlib.sha256((salt + "|B").encode()).hexdigest(), 16))
    exists = [r for r in recs if r["exists"]]
    orderA = exists[:]; rngA.shuffle(orderA)
    m0 = {(r["object"], r["case_id"]): r for r in recs if r["role"] == "m0"}
    edited = [r for r in exists if r["role"] == "edited" and m0[(r["object"], r["case_id"])]["exists"]]
    orderB = edited[:]; rngB.shuffle(orderB)

    def disp(src, dst):
        arr, size = _png_rgb(src)
        Image.fromarray(arr).save(dst, format="PNG")                # no metadata, lossless
        arr2, size2 = _png_rgb(dst)
        return bool(np.array_equal(arr, arr2) and size == size2)

    mapping = {"pass_a": {}, "pass_b": {}}
    tA, tB, ident = [], [], []
    for i, r in enumerate(orderA, 1):
        iid = f"A-{i:04d}"
        ok = disp(os.path.join(ROOT, r["orig"]), os.path.join(pkg, "passA", iid + ".png"))
        ident.append(ok)
        mapping["pass_a"][iid] = {k: r[k] for k in ("role", "arm", "seed", "object", "stratum", "case_id", "orig", "orig_sha256")}
        tA.append({"image_id": iid, "file": f"passA/{iid}.png", "prompt": r["prompt"], "named_object": r["named_object"],
                   "object_presence": None, "rendering": None, "reason": None})
    for i, r in enumerate(orderB, 1):
        iid = f"B-{i:03d}"
        ref = m0[(r["object"], r["case_id"])]
        ok1 = disp(os.path.join(ROOT, r["orig"]), os.path.join(pkg, "passB", iid + ".png"))
        ok2 = disp(os.path.join(ROOT, ref["orig"]), os.path.join(pkg, "passB", iid + "_ref.png"))
        ident += [ok1, ok2]
        mapping["pass_b"][iid] = {k: r[k] for k in ("role", "arm", "seed", "object", "stratum", "case_id", "orig", "orig_sha256")}
        tB.append({"image_id": iid, "edited_file": f"passB/{iid}.png", "reference_m0_file": f"passB/{iid}_ref.png",
                   "prompt": r["prompt"], "named_object": r["named_object"],
                   "relative_rendering": None, "damage_extent": None, "reason": None})
    json.dump(tA, open(os.path.join(pkg, "passA_template.json"), "w"), indent=1)
    json.dump(tB, open(os.path.join(pkg, "passB_template.json"), "w"), indent=1)
    shutil.copy(os.path.join(ROOT, "records", "VISUAL_AUDIT_RUBRIC.md"), os.path.join(pkg, "RUBRIC.md"))
    canon = json.dumps({"mapping": mapping, "missing": [{k: r[k] for k in ("role", "arm", "seed", "object", "case_id", "orig")}
                                                         for r in recs if not r["exists"]]}, sort_keys=True)
    open(os.path.join(priv, "mapping.json"), "w").write(canon)
    open(os.path.join(priv, "salt.txt"), "w").write(salt)
    commit = hashlib.sha256((salt + canon).encode()).hexdigest()
    files = {}
    for rt, _, fs in os.walk(pkg):
        for f in fs:
            p = os.path.join(rt, f); files[os.path.relpath(p, pkg)] = _sha(p)
    pub = os.path.join(ROOT, "visual_audit", "public"); os.makedirs(pub, exist_ok=True)
    out = {"commitment_sha256": commit, "commitment_definition": "SHA256(salt_hex_utf8 + canonical_mapping_json_utf8); mapping.json bytes are exactly that canonical JSON",
           "n_images_pass_a": len(orderA), "n_pairs_pass_b": len(orderB), "n_missing_underlying_images": sum(1 for r in recs if not r["exists"]),
           "n_expected_edited": 192, "n_expected_m0": 16, "pixel_identity_all_ok": all(ident), "n_pixel_identity_checks": len(ident),
           "package_file_sha256": files}
    json.dump(out, open(os.path.join(pub, "package_commitment.json"), "w"), indent=1, sort_keys=True)
    print({k: v for k, v in out.items() if k != "package_file_sha256"})


def unblind_and_analyze(judg_a, judg_b):
    """Join locked judgments to the private mapping and write per-arm/object/seed summaries (see addendum)."""
    raise NotImplementedError


if __name__ == "__main__" and sys.argv[1] == "package":
    package()
