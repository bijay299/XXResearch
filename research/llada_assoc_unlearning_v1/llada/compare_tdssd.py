"""Predeclared comparison and verdict (agents/O_FINAL.md §2 "Metrics", "Matched comparison", "P").

  python llada/compare_tdssd.py --theta0 eval_p0.json --x50 X/eval_50.json --x150 X/eval_150.json \
      --z50 Z/eval_50.json --z150 Z/eval_150.json --out compare.json

Definitions (relative to the memorised checkpoint theta0, same functional on both sides):
  D_F    = mean over forget facts and phrasings of mean_j KL(t_j || p_theta,j) (eval summary "D_F_mean")
  Delta_R = mean over held-out retain items (in_retain_loss = False) that share an answer word with a forget fact, both
           phrasings, of [log p0 - log p_theta](answer-word span)        ("Y"; larger = more damage)
  rho    = mean_{facts, eval strangers} |Delta log p(object span | stranger canvas)|
           / mean_facts |Delta log p(object span | subject canvas)|      ("W", stranger drift)
Z frontier: piecewise-linear in D_F through (D_F(theta0), Delta_R = 0), Z@50, Z@150 (sorted by D_F); rho^Z is linearly
interpolated between the Z checkpoints and held constant beyond them. X checkpoint comparable iff
D_F(X) in [min D_F(Z), D_F(theta0)].
Verdict: SUPPORTED iff >= 1 comparable X checkpoint and at every comparable one Delta_R(X) <= 0.7 Delta_R^Z(D_F(X))
and rho(X) <= 0.3 rho^Z(D_F(X)); CONTRADICTED iff Delta_R(X) >= Delta_R^Z(D_F(X)) at every comparable one;
INCONCLUSIVE otherwise; NOT_COMPARABLE (L2-type: X did not forget enough) if none is comparable. If Y passes and W
fails, the gain is not credited to the mechanism ("SUPPORTED_Y_ONLY"). One seed, 8 facts: directional only.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List


def _lp(recs, set_name):
    return {(r["id"], r.get("form"), r.get("stranger")): r["key_logprob"] for r in recs if r["set"] == set_name}


def metrics(ev: Dict, ev0: Dict) -> Dict:
    r, r0 = ev["records"], ev0["records"]
    out = {"D_F": ev["summary"]["D_F_mean"]}
    ho = {(x["id"], x.get("form")) for x in r0 if x["set"] == "retain_lik" and x["shares_answer_word"]
          and not x["in_retain_loss"]}
    tr = {(x["id"], x.get("form")) for x in r0 if x["set"] == "retain_lik" and x["shares_answer_word"]
          and x["in_retain_loss"]}
    lp, lp0 = _lp(r, "retain_lik"), _lp(r0, "retain_lik")
    for nm, keys in (("Delta_R", ho), ("Delta_R_train", tr)):
        d = [lp0[k + (None,)] - lp[k + (None,)] for k in keys if k + (None,) in lp]
        out[nm] = sum(d) / len(d) if d else None
        out[nm + "_n"] = len(d)
    s, s0 = _lp(r, "stranger_lik"), _lp(r0, "stranger_lik")
    f, f0 = _lp(r, "forget_lik"), _lp(r0, "forget_lik")
    ds = [abs(s[k] - s0[k]) for k in s if k in s0]
    df = [abs(f[k] - f0[k]) for k in f if k in f0]
    out["stranger_drift_abs"] = sum(ds) / len(ds) if ds else None
    out["subject_change_abs"] = sum(df) / len(df) if df else None
    out["rho"] = (out["stranger_drift_abs"] / out["subject_change_abs"]) if ds and df and out["subject_change_abs"] \
        else None
    out["generation"] = {k: ev["summary"].get(k) for k in ev["summary"] if not k.startswith(("D_F", "retain_key",
                                                                                              "forget_key"))}
    return out


def _interp(xs: List[float], ys: List[float], x: float) -> float:
    pts = sorted(zip(xs, ys))
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 > x1 else y1
    return pts[-1][1]


def verdict(m0: Dict, X: Dict[str, Dict], Z: Dict[str, Dict]) -> Dict:
    zD = [m0["D_F"]] + [Z[k]["D_F"] for k in sorted(Z)]
    zY = [0.0] + [Z[k]["Delta_R"] for k in sorted(Z)]
    zrD = [Z[k]["D_F"] for k in sorted(Z)]
    zrho = [Z[k]["rho"] for k in sorted(Z)]
    lo, hi = min(zD[1:]), m0["D_F"]
    rows, comp = [], []
    for k in sorted(X):
        x = X[k]
        ok = lo <= x["D_F"] <= hi
        row = {"ckpt": k, "D_F": x["D_F"], "comparable": ok, "Delta_R": x["Delta_R"], "rho": x["rho"]}
        if ok:
            row["Delta_R_Z_matched"] = _interp(zD, zY, x["D_F"])
            row["rho_Z_matched"] = _interp(zrD, zrho, x["D_F"]) if None not in zrho else None
            row["Y_pass"] = x["Delta_R"] <= 0.7 * row["Delta_R_Z_matched"]
            row["Y_contra"] = x["Delta_R"] >= row["Delta_R_Z_matched"]
            row["W_pass"] = (x["rho"] is not None and row["rho_Z_matched"] is not None
                             and x["rho"] <= 0.3 * row["rho_Z_matched"])
            comp.append(row)
        rows.append(row)
    if not comp:
        v = "NOT_COMPARABLE"
    elif all(r["Y_pass"] and r["W_pass"] for r in comp):
        v = "SUPPORTED"
    elif all(r["Y_pass"] for r in comp):
        v = "SUPPORTED_Y_ONLY"
    elif all(r["Y_contra"] for r in comp):
        v = "CONTRADICTED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "rows": rows, "Z_frontier": {"D_F": zD, "Delta_R": zY}, "comparable_D_F_range": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    for k in ("theta0", "x50", "x150", "z50", "z150"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ld = lambda p: json.load(open(p))  # noqa: E731
    ev0 = ld(a.theta0)
    m0 = {"D_F": ev0["summary"]["D_F_mean"], "generation": ev0["summary"]}
    X = {"50": metrics(ld(a.x50), ev0), "150": metrics(ld(a.x150), ev0)}
    Z = {"50": metrics(ld(a.z50), ev0), "150": metrics(ld(a.z150), ev0)}
    res = {"theta0": m0, "X": X, "Z": Z, **verdict(m0, X, Z)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"verdict": res["verdict"], "rows": res["rows"]}, indent=1))


if __name__ == "__main__":
    main()
