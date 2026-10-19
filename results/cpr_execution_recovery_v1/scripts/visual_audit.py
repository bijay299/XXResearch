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
