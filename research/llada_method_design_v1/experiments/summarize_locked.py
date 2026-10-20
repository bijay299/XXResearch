"""Mechanical summary of the locked validation (VALIDATION_SPEC.md section 5 and section 6) -> final_summary.json.

Input: the run_locked.py output dir (refs/refs_wW.json, units/<arm>_wW_sS.json) and the locked config.
Rules implemented exactly as written in VALIDATION_SPEC section 5; where the text leaves a case open, the choice made
here is listed under "interpretations" in the output so that G/coordinator can confirm or override it before the lock:
  I1 Cut A ties: higher R, then fewer steps, then lower LR.  Cut B ties: lower held-out leak, fewer steps, lower LR.
  I2 Delta_A when BOTH comparator and FX-Mix are INFEASIBLE: 0 (flagged); comparator-only => +1; FX-Mix-only => -1.
  I3 Worlds failing R5 are excluded; R1's "5 of 6" becomes ceil(5n/6) of the n counted units (flagged if n != 6).
  I4 R3 averages off_manifold_frac over the H1-H5 cells that have >= 1 first-commit decision state, then over units.
  I5 R4 means are over units whose Cut A exists (counts reported).
  I6 A unit missing from units/ (not completed) counts as INFEASIBLE for that arm (section 5 coverage rule).
Usage: summarize_locked.py --config configs/validation_locked.json [--runs DIR] [--integration-dryrun pass|fail]
"""
from __future__ import annotations

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HELDOUT_PRIMARY = ["H1", "H2", "H3", "H4", "H5"]
D_RET, D_COPY = 0.05, 0.02


def points(unit):
    for r in unit["runs"]:
        lr = r["config"]["lr"]
        for ev in r["curve"]:
            if ev["step"] > 0:
                yield {"lr": lr, "step": ev["step"], "ev": ev}


def feasible(agg, base):
    return agg["mean_R_acc"] >= base["mean_R_acc"] - D_RET and agg["min_copy"] >= base["min_copy"] - D_COPY


def cut_A(unit, base):
    c = [p for p in points(unit) if feasible(p["ev"]["agg"], base)]
    if not c:
        return None
    return min(c, key=lambda p: (p["ev"]["agg"]["max_F_leak_any_heldout"], -p["ev"]["agg"]["mean_R_acc"], p["step"],
                                 p["lr"]))


def cut_B(unit, retrain):
    thr = retrain["max_F_leak_any_heldout"]
    c = [p for p in points(unit) if p["ev"]["agg"]["max_F_leak_any_heldout"] <= thr]
    if not c:
        return None
    return max(c, key=lambda p: (p["ev"]["agg"]["mean_R_acc"], -p["ev"]["agg"]["max_F_leak_any_heldout"], -p["step"],
                                 -p["lr"]))


def describe(p):
    if p is None:
        return None
    a, ev = p["ev"]["agg"], p["ev"]
    d = {"lr": p["lr"], "step": p["step"], "HO5_leak_any": a["max_F_leak_any_heldout"], "H6_leak_any": a["max_F_leak_any_H6"],
         "HO6_leak_any": a["max_F_leak_any_heldout6"], "pool_leak_any": a["max_F_leak_any_train"],
         "pool_leak_full": a["max_F_leak_full_train"], "R": a["mean_R_acc"], "S": a["mean_S_acc"],
         "F_wf": a["mean_F_wf"], "min_copy": a["min_copy"], "counters": ev.get("counters")}
    mech = ev.get("mechanism")
    if mech:
        offs = [mech[g]["off_manifold_frac"] for g in HELDOUT_PRIMARY if mech.get(g, {}).get("off_manifold_frac") is not None]
        d["off_manifold_H1_H5_mean"] = sum(offs) / len(offs) if offs else None
        d["off_manifold_cells"] = len(offs)
        d["certificate"] = {g: {"B_hat": mech[g]["cert_B"], "se": mech[g].get("cert_B_se"), "leak": mech[g]["leak_rate"]}
                            for g in mech}
    return d


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "validation_locked.json"))
    ap.add_argument("--runs", default=None)
    ap.add_argument("--integration-dryrun", choices=["pass", "fail", "unknown"], default="unknown")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    rdir = a.runs or os.path.join(ROOT, cfg["out_dir"])
    arms = list(cfg["arms"])
    S = {"rules": "VALIDATION_SPEC.md section 5 (mechanical)", "interpretations": __doc__.split("Usage:")[0].strip(),
         "worlds": {}, "units": {}, "delta_A": {}, "R": {}, "flags": []}
    counted = []
    for W in cfg["worlds"]:
        p = os.path.join(rdir, "refs", f"refs_w{W}.json")
        if not os.path.exists(p):
            S["worlds"][W] = {"status": "MISSING_REFS"}
            S["flags"].append(f"world {W}: refs missing")
            continue
        refs = json.load(open(p))
        base, rt = refs["base"]["agg"], refs["retrain"]["agg"]
        r5 = base["max_F_leak_any_heldout"] >= 0.9 and rt["max_F_leak_full_heldout"] == 0
        S["worlds"][W] = {"R5_pass": r5, "base_HO5_leak_any": base["max_F_leak_any_heldout"],
                          "retrain_HO5_leak_any": rt["max_F_leak_any_heldout"],
                          "retrain_HO5_leak_full": rt["max_F_leak_full_heldout"], "base_R": base["mean_R_acc"],
                          "base_min_copy": base["min_copy"], "retrain_R": rt["mean_R_acc"],
                          "retrain_F_wf": rt["mean_F_wf"], "fban_F_wf": refs["base_fban"]["agg"]["mean_F_wf"]}
        if r5:
            counted.append((W, base, rt))
        else:
            S["flags"].append(f"world {W} fails R5 sanity -> excluded")
    unit_keys = [(W, s) for W, _, _ in counted for s in cfg["unlearn_seeds"]]
    cuts = {}
    for arm in arms:
        for (W, base, rt) in counted:
            for s in cfg["unlearn_seeds"]:
                up = os.path.join(rdir, "units", f"{arm}_w{W}_s{s}.json")
                if not os.path.exists(up):
                    cuts[(arm, W, s)] = (None, None, "MISSING")       # I6
                    continue
                u = json.load(open(up))
                cuts[(arm, W, s)] = (cut_A(u, base), cut_B(u, rt), "OK")
                S["units"][f"{arm}_w{W}_s{s}"] = {"cutA": describe(cuts[(arm, W, s)][0]),
                                                 "cutB": describe(cuts[(arm, W, s)][1]),
                                                 "wall_s_per_run": [r.get("wall_s") for r in u["runs"]]}
            for s in cfg["unlearn_seeds"]:
                if cuts[(arm, W, s)][2] == "MISSING":
                    S["units"][f"{arm}_w{W}_s{s}"] = {"cutA": None, "cutB": None, "status": "MISSING (counts INFEASIBLE)"}

    def cutA_val(arm, W, s):
        c = cuts.get((arm, W, s), (None,))[0]
        return None if c is None else c["ev"]["agg"]["max_F_leak_any_heldout"]

    n = len(unit_keys)
    need = 5 if n == 6 else math.ceil(5 * n / 6)
    if n != 6:
        S["flags"].append(f"{n} counted units (not 6): R1 positivity threshold = {need} of {n} (I3)")
    for X in arms:
        if X == "fx_mix":
            continue
        ds = []
        for (W, s) in unit_keys:
            x, f = cutA_val(X, W, s), cutA_val("fx_mix", W, s)
            if x is None and f is None:
                ds.append(0.0)
                S["flags"].append(f"Delta_A({X}) w{W} s{s}: both INFEASIBLE -> 0 (I2)")
            elif x is None:
                ds.append(1.0)
            elif f is None:
                ds.append(-1.0)
            else:
                ds.append(x - f)
        S["delta_A"][X] = {"per_unit": dict(zip([f"w{W}_s{s}" for W, s in unit_keys], ds)), "mean": mean(ds),
                           "n_positive": sum(d > 0 for d in ds), "n": len(ds)}
    infB = {arm: sum(cuts.get((arm, W, s), (None, None))[1] is None for (W, s) in unit_keys) for arm in arms}
    S["cutB_infeasible_count"] = infB
    S["cutA_infeasible_count"] = {arm: sum(cuts.get((arm, W, s), (None,))[0] is None for (W, s) in unit_keys)
                                  for arm in arms}
    # R1
    r1 = {}
    for X in ("sc_tf", "sc"):
        d = S["delta_A"].get(X)
        ok = d is not None and n > 0 and d["mean"] is not None and d["mean"] >= 0.03 and d["n_positive"] >= need
        r1[X] = {"mean_delta_A": d and d["mean"], "n_positive": d and d["n_positive"], "need": need,
                 "pass_delta": bool(ok)}
    cutB_ok = infB.get("fx_mix", 0) <= infB.get("sc_tf", 0)
    R1 = all(v["pass_delta"] for v in r1.values()) and cutB_ok
    S["R"]["R1"] = {"PASS": R1, "per_comparator": r1, "fxmix_cutB_infeasible": infB.get("fx_mix"),
                    "sctf_cutB_infeasible": infB.get("sc_tf"), "cutB_condition": cutB_ok}
    # R2
    S["R"]["R2"] = {"mean_delta_A_fx_mix_greedy": S["delta_A"].get("fx_mix_greedy", {}).get("mean"), "descriptive": True}
    # R2b (pre-registered secondary, coordinator 05:35Z): the R1 rule with FX-Mix-greedy as the candidate; reported only,
    # never changes the decision.
    r2b = {}
    if "fx_mix_greedy" in arms:
        for X in ("sc_tf", "sc"):
            ds = []
            for (W, s) in unit_keys:
                x, g = cutA_val(X, W, s), cutA_val("fx_mix_greedy", W, s)
                ds.append(0.0 if (x is None and g is None) else 1.0 if x is None else -1.0 if g is None else x - g)
            m = mean(ds)
            npos = sum(d > 0 for d in ds)
            r2b[X] = {"mean_delta_A": m, "n_positive": npos, "need": need,
                      "pass_delta": bool(n > 0 and m is not None and m >= 0.03 and npos >= need)}
        okB = infB.get("fx_mix_greedy", 0) <= infB.get("sc_tf", 0)
        S["R"]["R2b_secondary"] = {"PASS": all(v["pass_delta"] for v in r2b.values()) and okB, "per_comparator": r2b,
                                   "cutB_condition": okB, "decision_relevant": False}
    # R3
    offs = []
    for (W, s) in unit_keys:
        c = cuts.get(("sc_tf", W, s), (None,))[0]
        if c is not None:
            dd = describe(c)
            offs.append(dd.get("off_manifold_H1_H5_mean"))
    m3 = mean(offs)
    S["R"]["R3"] = {"sc_tf_off_manifold_cutA_mean_H1_H5": m3, "n_units_defined": sum(o is not None for o in offs),
                    "coverage_gap_present": (m3 is not None and m3 >= 0.5)}
    # R4
    def wf_mean(arm):
        v = [describe(cuts[(arm, W, s)][0])["F_wf"] for (W, s) in unit_keys
             if cuts.get((arm, W, s), (None,))[0] is not None]
        return mean(v), len(v)
    fx_wf, nfx = wf_mean("fx_mix") if "fx_mix" in arms else (None, 0)
    lit = {X: wf_mean(X) for X in ("npo", "mdu_t1", "emdu") if X in arms}
    lit_vals = [v for v, _ in lit.values() if v is not None]
    gap = None if fx_wf is None or not lit_vals else fx_wf - max(lit_vals)
    S["R"]["R4"] = {"fx_mix_F_wf": fx_wf, "n_fx": nfx, "literature_F_wf": {k: {"mean": v, "n": c} for k, (v, c) in lit.items()},
                    "gap": gap, "exclusion_target_avoids_degenerate_forgetting": gap is not None and gap >= 0.2,
                    "credit": "property of the exclusion target family (TNT/UNDIAL), not of FX-Mix's state design"}
    S["R"]["R5"] = {str(W): S["worlds"][W].get("R5_pass") for W in cfg["worlds"]}
    expected = len(cfg["worlds"]) * len(cfg["unlearn_seeds"])
    S["coverage"] = {arm: {"completed_units": sum(os.path.exists(os.path.join(rdir, "units", f"{arm}_w{W}_s{s}.json"))
                                                  for W in cfg["worlds"] for s in cfg["unlearn_seeds"]),
                           "expected_units": expected} for arm in arms}
    if R1 and a.integration_dryrun == "pass":
        dec = "READY_FOR_BOUNDED_REAL_MODEL_TEST"
    elif R1 and a.integration_dryrun == "fail":
        dec = "DESIGNED_WITH_SPECIFIC_BLOCKER"
    elif not R1:
        dec = "NO_DISTINCT_METHOD_AFTER_CONSTRUCTION"
    else:
        dec = "PENDING_INTEGRATION_DRYRUN_STATUS"
    S["decision"] = dec
    out = a.out or os.path.join(rdir, "final_summary.json")
    json.dump(S, open(out, "w"), indent=1, default=str)
    print(json.dumps({"decision": dec, "R1": S["R"]["R1"]["PASS"], "delta_A": {k: v["mean"] for k, v in S["delta_A"].items()},
                      "R3": S["R"]["R3"], "R4_gap": gap, "flags": S["flags"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
