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


PRESENT = ("clearly_present", "recognizable_part_present")
COH_OK = ("coherent", "coherent_localized_defects")
DAMAGED = ("substantial_distortions_interpretable", "pervasive_degeneration")
PRESENCE_VALUES = PRESENT + ("absent", "uncertain", "unjudgeable")
RENDER_VALUES = COH_OK + DAMAGED + ("uncertain_unjudgeable",)
REL_VALUES = ("better", "similar", "worse", "uncertain")
DMG_VALUES = ("none", "localized", "pervasive", "uncertain")
# Conclusion rule (committed before judgments): V = strict removal = (M0 present) & (edited absent) & rendering coherent-ok & Pass-B
# (relative in better/similar and damage in none/localized), over M0-positive target cases pooled over objects and seeds.
# Direction/size threshold 0.125 of the pooled denominator. Descriptive; never feeds back into the experiment.
CONCL_DELTA = 0.125


def _frac(n, d):
    return None if d == 0 else round(n / d, 4)


def validate_judgments(judg_a, judg_b, mapping):
    errs = []
    for iid in mapping["pass_a"]:
        j = judg_a.get(iid)
        if j is None or j.get("object_presence") not in PRESENCE_VALUES or j.get("rendering") not in RENDER_VALUES:
            errs.append(("A", iid))
    for iid in mapping["pass_b"]:
        j = judg_b.get(iid)
        if j is None or j.get("relative_rendering") not in REL_VALUES or j.get("damage_extent") not in DMG_VALUES:
            errs.append(("B", iid))
    return errs


def unblind_and_analyze(judg_a, judg_b, mapping, detector_by_key=None, detector_decision=None):
    """judg_*: {image_id: judgment dict}. mapping: private mapping (already verified against the commitment).
    detector_by_key: {(arm, object, seed, case_id): bool target_nondetected_at_primary or None}."""
    A, B = mapping["pass_a"], mapping["pass_b"]
    m0 = {(r["object"], r["case_id"]): iid for iid, r in A.items() if r["role"] == "m0"}
    m0_pres = {k: judg_a[i]["object_presence"] in PRESENT for k, i in m0.items()}
    pb = {}
    for iid, r in B.items():
        pb[(r["arm"], r["object"], r["seed"], r["case_id"])] = judg_b[iid]
    rows = []
    for iid, r in A.items():
        if r["role"] != "edited":
            continue
        key = (r["arm"], r["object"], r["seed"], r["case_id"])
        ja, jb = judg_a[iid], pb.get(key)
        target = r["stratum"] == "target"
        pres = ja["object_presence"]
        coh = ja["rendering"] in COH_OK
        rel_ok = bool(jb and jb["relative_rendering"] in ("better", "similar") and jb["damage_extent"] in ("none", "localized"))
        rows.append({"arm": r["arm"], "object": r["object"], "seed": r["seed"], "case_id": r["case_id"], "stratum": r["stratum"],
                     "target": target, "presence": pres, "rendering": ja["rendering"],
                     "rel": jb["relative_rendering"] if jb else None, "damage": jb["damage_extent"] if jb else None,
                     "m0_present": m0_pres.get((r["object"], r["case_id"])), "coh": coh, "rel_ok": rel_ok,
                     "visual_absent": pres == "absent", "visual_present": pres in PRESENT, "visual_uncertain": pres in ("uncertain", "unjudgeable"),
                     "substantial_damage": ja["rendering"] in DAMAGED or (bool(jb) and jb["damage_extent"] == "pervasive"),
                     "det_nondetected": (detector_by_key or {}).get(key)})

    def summarize(rs):
        t = [x for x in rs if x["target"]]
        al = [x for x in rs if not x["target"]]
        tp = [x for x in t if x["m0_present"]]
        al_p = [x for x in al if x["m0_present"]]
        out = {
            "target": {"n_sampled": len(t), "n_m0_present": len(tp), "n_m0_absent_or_uncertain": len(t) - len(tp),
                       "edited_absent_given_m0_present": sum(x["visual_absent"] for x in tp),
                       "edited_present": sum(x["visual_present"] for x in t), "edited_uncertain": sum(x["visual_uncertain"] for x in t),
                       "removal_given_m0_present": _frac(sum(x["visual_absent"] for x in tp), len(tp)),
                       "removal_coherent_given_m0_present": sum(x["visual_absent"] and x["coh"] for x in tp),
                       "removal_coherent_no_worse_given_m0_present": sum(x["visual_absent"] and x["coh"] and x["rel_ok"] for x in tp),
                       "strict_removal_frac": _frac(sum(x["visual_absent"] and x["coh"] and x["rel_ok"] for x in tp), len(tp)),
                       "n_coherent_ok": sum(x["coh"] for x in t), "n_substantial_damage": sum(x["substantial_damage"] for x in t),
                       "n_rel_worse": sum(x["rel"] == "worse" for x in t)},
            "allowed": {"n_sampled": len(al), "n_m0_present": len(al_p), "retained_given_m0_present": sum(x["visual_present"] for x in al_p),
                        "retained_all_sampled": sum(x["visual_present"] for x in al), "uncertain": sum(x["visual_uncertain"] for x in al),
                        "retained_coherent_no_worse_given_m0_present": sum(x["visual_present"] and x["coh"] and x["rel_ok"] for x in al_p),
                        "n_substantial_damage": sum(x["substantial_damage"] for x in al),
                        "retention_frac": _frac(sum(x["visual_present"] for x in al_p), len(al_p)),
                        "strict_retention_frac": _frac(sum(x["visual_present"] and x["coh"] and x["rel_ok"] for x in al_p), len(al_p))}}
        dn = [x for x in t if x["det_nondetected"] is True]
        dd = [x for x in t if x["det_nondetected"] is False]
        out["detector_vs_visual_targets"] = {
            "n_with_detector": sum(x["det_nondetected"] is not None for x in t),
            "detector_nondetected": len(dn), "of_which_visually_absent": sum(x["visual_absent"] for x in dn),
            "of_which_visually_present_or_partial": sum(x["visual_present"] for x in dn),
            "of_which_visually_uncertain": sum(x["visual_uncertain"] for x in dn),
            "of_which_substantial_damage": sum(x["substantial_damage"] for x in dn),
            "detector_detected": len(dd), "of_which_visually_absent": sum(x["visual_absent"] for x in dd)}
        return out

    arms = sorted({x["arm"] for x in rows})
    objs = sorted({x["object"] for x in rows})
    res = {"label": "MODEL_ASSISTED_VISUAL_SCREEN", "by_arm": {}, "by_arm_object": {}, "by_arm_object_seed": {}}
    for a in arms:
        res["by_arm"][a] = summarize([x for x in rows if x["arm"] == a])
        for o in objs:
            res["by_arm_object"][f"{a}|{o}"] = summarize([x for x in rows if x["arm"] == a and x["object"] == o])
            for sd in sorted({x["seed"] for x in rows}):
                res["by_arm_object_seed"][f"{a}|{o}|{sd}"] = summarize([x for x in rows if x["arm"] == a and x["object"] == o and x["seed"] == sd])
    m0t = [(k, v) for k, v in m0_pres.items()]
    m0c = {}
    for iid, r in A.items():
        if r["role"] == "m0":
            m0c[(r["object"], r["case_id"], r["stratum"])] = judg_a[iid]
    res["m0_reference"] = {"n": len(m0c), "n_present": sum(j["object_presence"] in PRESENT for j in m0c.values()),
                           "n_coherent_ok": sum(j["rendering"] in COH_OK for j in m0c.values()),
                           "target_n": sum(1 for k in m0c if k[2] == "target"),
                           "target_present": sum(1 for k, j in m0c.items() if k[2] == "target" and j["object_presence"] in PRESENT)}
    comp = None
    if detector_decision:
        comp = detector_decision.get("comparator")
    res["conclusion"] = _conclusion(res, detector_decision)
    return res, rows


def _conclusion(res, det):
    """Visual-evidence conclusion per the committed rule (see CONCL_DELTA). Never changes the detector decision."""
    if not det:
        return {"verdict": "insufficient", "why": "no detector decision supplied"}
    dec = det.get("decision")
    comp = det.get("comparator")
    ba = res["by_arm"]
    def V(a): return ba[a]["target"]["strict_removal_frac"]
    def n_(a): return ba[a]["target"]["n_m0_present"]
    def sdmg(a):
        t = ba[a]["target"]; return _frac(t["n_substantial_damage"], t["n_sampled"])
    def ret(a): return ba[a]["allowed"]["strict_retention_frac"]
    ctrl = [a for a in ba if a != "cpr"]
    if "cpr" not in ba or any(V(a) is None for a in ["cpr"] + ctrl) or n_("cpr") < 8:
        return {"verdict": "insufficient", "why": "too few M0-positive sampled targets for a visual comparison", "detector_decision": dec}
    diffs = {a: round(V("cpr") - V(a), 4) for a in ctrl}
    dmg_excess = {a: round((sdmg("cpr") or 0) - (sdmg(a) or 0), 4) for a in ctrl}
    adv = dec == "PROVISIONAL_DETECTOR_DEFINED_ADVANTAGE"
    noadv = dec == "NO_ADVANTAGE_IN_THIS_SCREEN"
    vs_comp = diffs.get(comp)
    if adv:
        if all(d >= CONCL_DELTA for d in diffs.values()) and ret("cpr") is not None and all((ret("cpr") or 0) >= (ret(a) or 0) - CONCL_DELTA for a in ctrl):
            verdict = "corroborating"
        elif vs_comp is not None and vs_comp <= -CONCL_DELTA or dmg_excess.get(comp, 0) >= CONCL_DELTA:
            verdict = "concerning"
        else:
            verdict = "insufficient"
    elif noadv:
        if all(d < CONCL_DELTA for d in diffs.values()):
            verdict = "corroborating"
        else:
            verdict = "concerning"
    else:
        verdict = "insufficient"
    return {"verdict": verdict, "detector_decision": dec, "comparator": comp, "delta": CONCL_DELTA,
            "strict_removal_cpr": V("cpr"), "strict_removal_by_control": {a: V(a) for a in ctrl}, "cpr_minus_control": diffs,
            "cpr_substantial_damage_excess_vs_control": dmg_excess,
            "strict_retention_cpr": ret("cpr"), "strict_retention_by_control": {a: ret(a) for a in ctrl}}


def lock(which, path):
    """Hash a judgments file and record it (hash only; content committed separately)."""
    out = os.path.join(ROOT, "visual_audit", "public", f"pass_{which.lower()}_lock.json")
    assert not os.path.exists(out), "already locked"
    import datetime
    json.dump({"pass": which, "file": os.path.basename(path), "sha256": _sha(path),
               "locked_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
               "n_records": len(json.load(open(path)))}, open(out, "w"), indent=1)
    print(open(out).read())


def analyze():
    from cprx import analysis as AN
    from cprx.common import load
    priv = os.path.join(ROOT, "private_audit")
    canon = open(os.path.join(priv, "mapping.json")).read()
    salt = open(os.path.join(priv, "salt.txt")).read()
    pub = json.load(open(os.path.join(ROOT, "visual_audit", "public", "package_commitment.json")))
    ok_commit = hashlib.sha256((salt + canon).encode()).hexdigest() == pub["commitment_sha256"]
    assert ok_commit, "commitment mismatch"
    mapping = json.loads(canon)["mapping"]
    jdir = os.path.join(ROOT, "visual_audit", "judgments")
    locks = {}
    for w, fn in (("a", "PASS_A_JUDGMENTS.json"), ("b", "PASS_B_JUDGMENTS.json")):
        lk = json.load(open(os.path.join(ROOT, "visual_audit", "public", f"pass_{w}_lock.json")))
        assert _sha(os.path.join(jdir, fn)) == lk["sha256"], f"{fn} changed since lock"
        locks[w] = lk
    ja = {r["image_id"]: r for r in json.load(open(os.path.join(jdir, "PASS_A_JUDGMENTS.json")))}
    jb = {r["image_id"]: r for r in json.load(open(os.path.join(jdir, "PASS_B_JUDGMENTS.json")))}
    errs = validate_judgments(ja, jb, mapping)
    final = load(os.path.join(ROOT, "artifacts", "analysis", "final_results.json"))
    m0 = load(os.path.join(ROOT, "artifacts", "scores", "m0_ho.json"))
    lab = m0["label_ids"]
    det = {}
    cases = load(os.path.join(ROOT, "configs", "cases.json"))
    for t in cases["heldout_trajectories"]:
        pth = os.path.join(ROOT, "artifacts", "scores", t["traj_id"] + ".json")
        if not os.path.exists(pth):
            continue
        for r in load(pth)["sets"][0]["rows"]:
            if r["kind"] != "target":
                continue
            m = AN.measure(r, P.PRIMARY_DET, t["target"], lab)
            det[(t["arm"], t["target"], t["train_seed"], r["case_id"])] = None if (m["missing"] or not m["valid"]) else (not m["target_det"])
    res, rows = unblind_and_analyze(ja, jb, mapping, det, {"decision": final["decision"] if isinstance(final["decision"], str) else final["decision"].get("decision"),
                                                         "comparator": final["comparator"]})
    res["data_integrity"] = {"commitment_verified": ok_commit, "judgment_validation_errors": errs, "locks": locks}
    out = os.path.join(ROOT, "visual_audit", "public")
    json.dump(res, open(os.path.join(out, "visual_audit_results.json"), "w"), indent=1, sort_keys=True)
    json.dump(rows, open(os.path.join(out, "visual_audit_rows_unblinded.json"), "w"), indent=1)
    print(json.dumps(res["conclusion"], indent=1), "errors:", len(errs))


if __name__ == "__main__":
    if sys.argv[1] == "package":
        package()
    elif sys.argv[1] == "lock":
        lock(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "analyze":
        analyze()
