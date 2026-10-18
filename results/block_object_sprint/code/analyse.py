#!/usr/bin/env python3
"""Turns the frozen per-image labels into the gate decisions and the arm comparisons.

All definitions below are fixed in RUN_MANIFEST.md and applied identically to every arm.

  definitely_target_present   presence_target == "yes"
  definitely_retained_ok      presence_intended == "yes" AND render_failure == "no"
                              (strict: an uncertain rendering judgment is not "definitely
                               successful"; a looser variant is also reported)
  severe_render_failure       render_failure == "yes"
  uncertainty                 any field parsed as "uncertain"

Conservative suppression bounds (fixed before outcomes, least favourable to the edit):
  uncertain M0 target cases count as ABSENT   -> smaller M0 denominator
  uncertain edited target cases count PRESENT -> larger edited numerator
so reported relative suppression is a lower bound.
"""
import os, json, glob, argparse, itertools

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
SC = f"{SPRINT}/artifacts/scores"


def load(arm, obj):
    p = f"{SC}/{arm}_{obj}/answers.jsonl"
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def strat(rows, s):
    return [r for r in rows if r["stratum"] == s]


def counts(rows):
    """presence/retention/rendering tallies for one list of rows."""
    n = len(rows)
    tp = sum(1 for r in rows if r.get("presence_target") == "yes")
    ta = sum(1 for r in rows if r.get("presence_target") == "no")
    tu = sum(1 for r in rows if r.get("presence_target") == "uncertain")
    ip = sum(1 for r in rows if r.get("presence_intended") == "yes")
    iu = sum(1 for r in rows if r.get("presence_intended") == "uncertain")
    rf = sum(1 for r in rows if r.get("render_failure") == "yes")
    ru = sum(1 for r in rows if r.get("render_failure") == "uncertain")
    ok_strict = sum(1 for r in rows
                    if r.get("presence_intended") == "yes" and r.get("render_failure") == "no")
    ok_loose = sum(1 for r in rows
                   if r.get("presence_intended") == "yes" and r.get("render_failure") != "yes")
    return {"n": n,
            "target_present": tp, "target_absent": ta, "target_uncertain": tu,
            "intended_present": ip, "intended_uncertain": iu,
            "render_failure": rf, "render_uncertain": ru,
            "retained_ok_strict": ok_strict, "retained_ok_loose": ok_loose,
            "target_present_rate": tp / n if n else None,
            "retained_ok_strict_rate": ok_strict / n if n else None,
            "render_failure_rate": rf / n if n else None}


def arm_summary(arm, obj):
    rows = load(arm, obj)
    if rows is None:
        return None
    out = {"arm": arm, "object": obj, "n_rows": len(rows)}
    for s in ("target", "related", "unrelated"):
        out[s] = counts(strat(rows, s))
    out["all80_render"] = counts(rows)
    return out


def pct(x):
    return None if x is None else round(100.0 * x, 1)


def suppression(m0, ed):
    """conservative relative suppression of the target vs M0, both from target-stratum rows."""
    m0_present = m0["target"]["target_present"]                      # uncertain M0 -> absent
    ed_present = ed["target"]["target_present"] + ed["target"]["target_uncertain"]  # unc -> present
    if m0_present == 0:
        return {"relative_suppression_pct": None,
                "note": "M0 denominator is zero; suppression undefined"}
    return {"m0_present_conservative": m0_present,
            "edited_present_conservative": ed_present,
            "relative_suppression_pct": round(100.0 * (1 - ed_present / m0_present), 1),
            "point_estimate_pct": round(100.0 * (
                1 - ed["target"]["target_present"] / max(1, m0["target"]["target_present"])), 1),
            "bound": "lower bound: uncertain M0 counted absent, uncertain edited counted present"}


def gates(objs=("Bears", "Cats", "Horses")):
    g, res = {}, {}
    for obj in objs:
        m0, t0 = arm_summary("M0", obj), arm_summary("T0", obj)
        if not m0 or not t0:
            continue
        res[obj] = {"M0": m0, "T0": t0}
        g4 = {"M0_definitely_target_present": m0["target"]["target_present"],
              "T0_definitely_target_present": t0["target"]["target_present"],
              "denominator": m0["target"]["n"], "threshold": 32,
              "M0_pass": m0["target"]["target_present"] >= 32,
              "T0_pass": t0["target"]["target_present"] >= 32}
        g4["pass"] = g4["M0_pass"] and g4["T0_pass"]
        g5 = {s: {"M0_retained_ok_strict": m0[s]["retained_ok_strict"],
                  "denominator": m0[s]["n"], "threshold": 16,
                  "pass": m0[s]["retained_ok_strict"] >= 16}
              for s in ("related", "unrelated")}
        g5["pass"] = all(g5[s]["pass"] for s in ("related", "unrelated"))
        g6 = {}
        for s in ("related", "unrelated"):
            d = pct(m0[s]["retained_ok_strict_rate"]) - pct(t0[s]["retained_ok_strict_rate"])
            g6[f"{s}_retention_loss_points"] = round(d, 1)
            g6[f"{s}_pass"] = d <= 10.0
        d_rf = pct(t0["all80_render"]["render_failure_rate"]) - \
            pct(m0["all80_render"]["render_failure_rate"])
        g6["render_failure_increase_points"] = round(d_rf, 1)
        g6["render_pass"] = d_rf <= 10.0
        g6["threshold_points"] = 10.0
        g6["pass"] = g6["related_pass"] and g6["unrelated_pass"] and g6["render_pass"]
        g[obj] = {"gate4_target_present": g4, "gate5_M0_retention": g5,
                  "gate6_T0_substitution_cost": g6,
                  "pass": g4["pass"] and g5["pass"] and g6["pass"]}
    return g, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gates", choices=["gates", "compare"])
    ap.add_argument("--objects", nargs="+", default=["Bears", "Cats", "Horses"])
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.mode == "gates":
        g, res = gates(tuple(a.objects))
        print(json.dumps(g, indent=1))
        allp = all(v["pass"] for v in g.values()) if g else False
        print(f"\nGATES 4-6 PASS (all objects): {allp}")
        out = a.out or f"{SPRINT}/artifacts/scores/gates_4_6.json"
        json.dump({"gates": g, "summaries": res, "all_pass": allp}, open(out, "w"), indent=1)
        print("wrote", out)
    else:
        table = {}
        for obj in a.objects:
            m0 = arm_summary("M0", obj)
            row = {}
            for arm in a.arms:
                s = arm_summary(arm, obj)
                if not s:
                    continue
                e = {"target_present": s["target"]["target_present"],
                     "target_present_pct": pct(s["target"]["target_present_rate"]),
                     "target_uncertain": s["target"]["target_uncertain"],
                     "related_retention_pct": pct(s["related"]["retained_ok_strict_rate"]),
                     "unrelated_retention_pct": pct(s["unrelated"]["retained_ok_strict_rate"]),
                     "render_failure_pct": pct(s["all80_render"]["render_failure_rate"]),
                     "counts": s}
                if m0 and arm != "M0":
                    e["suppression_vs_M0"] = suppression(m0, s)
                row[arm] = e
            table[obj] = row
        print(json.dumps({o: {k: {kk: vv for kk, vv in v.items() if kk != "counts"}
                              for k, v in r.items()} for o, r in table.items()}, indent=1))
        out = a.out or f"{SPRINT}/artifacts/scores/comparison.json"
        json.dump(table, open(out, "w"), indent=1)
        print("wrote", out)
