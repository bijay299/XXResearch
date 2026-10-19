#!/usr/bin/env python3
"""Synthetic-data tests of the frozen analysis predicates (no model, no detector, no image)."""
import json
import os
import sys

import numpy as np

from cprd2 import analysis as A
from cprd2 import protocol as P

LID = {t: {c: i for i, c in enumerate(P.SCORED_CLASSES)} for t in (P.PRIMARY_DET, P.SECONDARY_DET)}
TGT = "toilet"
ALLOWED = P.PANELS["toilet"]


def case_list(n_t=20, n_a=5):
    out = [("target", TGT, f"c.t.{i:02d}") for i in range(n_t)]
    for c in ALLOWED:
        out += [("allowed", c, f"c.{c}.{i:02d}") for i in range(n_a)]
    return out


def row(kind, cls, cid, det_target, det_own, valid=True, miss=False):
    det = {}
    for t in (P.PRIMARY_DET, P.SECONDARY_DET):
        labs = []
        if det_target: labs.append(LID[t][TGT])
        if det_own and kind == "allowed": labs.append(LID[t][cls])
        det[t] = {"labels": labs, "scores": [0.9] * len(labs), "boxes_xyxy": [[0, 0, 1, 1]] * len(labs)}
        if miss: det[t] = {"error": "x"}
    return {"case_id": cid, "cls": cls, "kind": kind, "flags": {"valid": valid, "flags": [] if valid else ["collapse"]},
            "detections": det}


def m0():
    return [row(k, c, i, True, True) for k, c, i in case_list()]


def endpoint(rng, s_rate, r_rate):
    rows = []
    for k, c, i in case_list():
        if k == "target":
            rows.append(row(k, c, i, det_target=rng.random() > s_rate, det_own=False))
        else:
            rows.append(row(k, c, i, det_target=False, det_own=rng.random() < r_rate))
    return rows


def table(spec, seed0=0):
    rng = np.random.default_rng(seed0)
    return {arm: {s: endpoint(rng, *spec[arm]) for s in (5201, 5202)} for arm in P.ARMS}


def main():
    fails = []

    def chk(n, ok):
        print(("PASS " if ok else "FAIL ") + n)
        if not ok: fails.append(n)

    # 1. planted CPR advantage -> PROMISING
    spec = {a: (1.0, 0.6) for a in P.ARMS}; spec["cpr"] = (1.0, 1.0)
    E = A.build_object_table(TGT, m0(), table(spec, 1), LID, n_boot=2000)
    pr = A.object_predicates(TGT, E, "native_gem")
    chk("planted advantage -> promising object", pr["promising"]["ok"])
    # 2. planted parity -> NO_ADVANTAGE for chosen comparator
    spec = {a: (1.0, 1.0) for a in P.ARMS}
    E2 = A.build_object_table(TGT, m0(), table(spec, 2), LID, n_boot=2000)
    pr2 = A.object_predicates(TGT, E2, "native_gem")
    chk("parity -> not promising", not pr2["promising"]["ok"])
    chk("parity -> no_advantage predicate", pr2["no_advantage"]["ok"])
    chk("planted advantage -> not no_advantage", not pr["no_advantage"]["ok"])
    # 3. determinism of bootstrap
    E3 = A.build_object_table(TGT, m0(), table(spec, 2), LID, n_boot=2000)
    chk("bootstrap deterministic", E3["bootstrap"] == E2["bootstrap"])
    # 4. missing measurement -> no decision
    t = table(spec, 3)
    t["cpr"][5201][0] = row("target", TGT, "c.t.00", False, False, miss=True)
    Em = A.build_object_table(TGT, m0(), t, LID, n_boot=500)
    chk("missing measurement flagged", Em["missing"])
    chk("final decision on missing", A.final_decision({"toilet": {"missing": True}, "bed": {"missing": True}}, None).startswith("NO_SCIENTIFIC"))
    # 5. invalid outputs count as failures (not dropped, not missing)
    t = table({a: (1.0, 1.0) for a in P.ARMS}, 4)
    for r in t["cpr"][5201][:4]:
        r["flags"] = {"valid": False, "flags": ["collapse"]}
    Ei = A.build_object_table(TGT, m0(), t, LID, n_boot=500)
    chk("invalid rows are failures with fixed denominator", (not Ei["missing"]) and abs(Ei["primary"]["seed_S"][5201]["cpr"] - 16 / 20) < 1e-9
        and Ei["primary"]["n_target"] == 20)
    # 6. eligibility gate
    mm = m0()
    for r in mm[:11]:                      # 11/20 targets undetected in M0 -> 9 eligible <16
        for tg in r["detections"]: r["detections"][tg]["labels"] = []
    es = A.eligibility_summary(mm, TGT, LID, "ho")
    chk("held-out eligibility failure detected", es["counts"]["primary"]["target"] == 9 and not es["pass"])
    chk("held-out eligibility pass", A.eligibility_summary(m0(), TGT, LID, "ho")["pass"])
    # 7. dev selection rules
    def C(i, arm, lr, st, S, R): return {"id": i, "arm": arm, "lr": lr, "states": st, "S": S, "R": R}
    cands = {a: [C(f"{a}.a", a, 5e-4, 66, 0.9, 0.5), C(f"{a}.b", a, 1e-3, 66, 0.9, 0.7), C(f"{a}.c", a, 1e-3, 250, 0.95, 0.7)]
             for a in P.ARMS}
    sel = A.dev_selection(cands)
    chk("tie -> fewer states", sel["selected"]["cpr"]["id"] == "cpr.b")
    cands["anchor_only"] = [C("x", "anchor_only", 1e-3, 66, 0.5, 0.99)]
    cands["native_gem"] = [C("n1", "native_gem", 1e-3, 66, 0.85, 0.75)]
    sel = A.dev_selection(cands)
    chk("unmatched arm recorded", sel["selected"]["anchor_only"]["efficacy_unmatched"])
    chk("strongest comparator by R among floor-qualified", sel["strongest_comparator"] == "native_gem" or sel["selected"][sel["strongest_comparator"]]["R"] >= 0.7)
    cands["cpr"] = [C("c", "cpr", 1e-3, 66, 0.7, 0.9)]
    chk("cpr below floor closes dev", A.dev_selection(cands)["status"] == "DEVELOPMENT_CLOSED_NATIVE_OR_CPR_BELOW_S_FLOOR")
    # 8. final decisions
    po = {"promising": {"ok": True}, "no_advantage": {"ok": False}}
    na = {"promising": {"ok": False}, "no_advantage": {"ok": True}}
    chk("final promising", A.final_decision({"a": po, "b": po}, "native_gem") == "PROMISING_DETECTOR_DEFINED_TRADEOFF")
    chk("final no advantage", A.final_decision({"a": na, "b": na}, "native_gem") == "NO_ADVANTAGE_IN_THIS_SCREEN")
    chk("final inconclusive", A.final_decision({"a": po, "b": na}, "native_gem") == "INCONCLUSIVE")
    os.makedirs(os.path.join(os.environ["CPRD2_ROOT"], "artifacts"), exist_ok=True)
    json.dump({"fails": fails, "all_pass": not fails}, open(os.path.join(os.environ["CPRD2_ROOT"], "artifacts", "analysis_synthetic_tests.json"), "w"), indent=1)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
