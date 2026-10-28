"""D2 — does the validity gate's bias change any conclusion?

THE DEFECT. FINDING72 measured (n=520) that the gate's exclusion rate is **non-monotone in
`T_before`**. Since `max-over-valid H_after` is this project's primary robustness quantity, it is
computed over a sample whose filtering correlates with how well-erased the arm is. A reviewer's
question is immediate: does the headline survive an estimator that does not filter?

WHY THE GATE EXISTS. A cell where the attacker never relearned the prompts it TRAINED on
(`T_rise < 0.20`) is inconclusive: a low `H_after` there is not evidence of robustness. Dropping the
gate therefore admits cells where the attack never started, which pushes measured recovery DOWN.
So the unfiltered estimator is biased AGAINST the project's central claim -- if the claim survives
it, the gate is not holding the result up.

WHAT THIS DOES. Recomputes every arm's recovery under four estimators on the same cells:
  gated        max over gate-valid, non-destroyed      (the published quantity)
  ungated      max over non-destroyed, no T_rise gate  (admits never-started attacks)
  raw          max over every cell, no filtering at all
  mean_gated   mean rather than max over gate-valid    (a max is itself a biased statistic)
and reports exclusion rate against `T_before` so the non-monotonicity is visible rather than cited.

Reads existing JSON only. No GPU, no regeneration.
"""
import json, glob, os, argparse, collections

ROOT = "/data/bijaypandey/t2i_unlearn"


def classify(d):
    av = d.get("attack_validity") or {}
    rc = d.get("recovery") or {}
    pu = (d.get("post_attack_utility") or {}).get("retained_capability") or {}
    rb, ra = pu.get("before"), pu.get("after")
    destroyed = (ra is not None and rb is not None and ra < 0.55 and (rb - ra) >= 0.25)
    return {
        "T_before": av.get("T_before"), "T_rise": av.get("T_rise"),
        "valid": bool(av.get("valid")), "destroyed": destroyed,
        "H": rc.get("H_after"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--out", default="outputs/d2_gate_bias.json")
    a = ap.parse_args()

    by_arm = collections.defaultdict(list)
    tb_bucket = collections.defaultdict(lambda: [0, 0])   # T_before bucket -> [excluded, total]

    for f in glob.glob(os.path.join(ROOT, "outputs/atk_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "attack_validity" not in d:
            continue
        if d.get("concept") != a.concept:
            continue
        unet = d.get("unet") or ""
        if "adapters/" not in unet:
            continue
        arm = unet.split("adapters/")[1].split("/")[0]
        c = classify(d)
        if c["H"] is None:
            continue
        by_arm[arm].append(c)
        tb = c["T_before"]
        if tb is not None:
            b = round(min(max(tb, 0.0), 1.0) * 5) / 5.0      # 0.0,0.2,...,1.0
            tb_bucket[b][1] += 1
            if (not c["valid"]) or c["destroyed"]:
                tb_bucket[b][0] += 1

    rows = []
    for arm, cs in sorted(by_arm.items()):
        gated = [c["H"] for c in cs if c["valid"] and not c["destroyed"]]
        ungated = [c["H"] for c in cs if not c["destroyed"]]
        raw = [c["H"] for c in cs]
        rows.append({
            "arm": arm, "n_cells": len(cs),
            "n_gated": len(gated), "n_ungated": len(ungated),
            "gated_max": max(gated) if len(gated) >= a.min_cells else None,
            "ungated_max": max(ungated) if len(ungated) >= a.min_cells else None,
            "raw_max": max(raw) if len(raw) >= a.min_cells else None,
            "gated_mean": (sum(gated) / len(gated)) if len(gated) >= a.min_cells else None,
        })

    reach = lambda key: sum(1 for r in rows if r[key] is not None and r[key] >= 0.999)
    have = lambda key: sum(1 for r in rows if r[key] is not None)
    summary = {
        "concept": a.concept, "n_arms": len(rows), "n_cells": sum(r["n_cells"] for r in rows),
        "arms_reaching_1.000_gated": [reach("gated_max"), have("gated_max")],
        "arms_reaching_1.000_ungated": [reach("ungated_max"), have("ungated_max")],
        "arms_reaching_1.000_raw": [reach("raw_max"), have("raw_max")],
        "exclusion_rate_by_T_before": {
            f"{k:.1f}": {"excluded": v[0], "total": v[1], "rate": round(v[0] / v[1], 4)}
            for k, v in sorted(tb_bucket.items()) if v[1]
        },
    }
    rec = {"study": "d2_gate_bias", "summary": summary, "per_arm": rows,
           "note": ("the ungated estimator admits cells where the attack never started, which pushes "
                    "measured recovery DOWN; it is therefore biased against this project's claim")}
    with open(os.path.join(ROOT, a.out), "w") as fh:
        json.dump(rec, fh, indent=1)

    print(f"[d2] concept={a.concept}  arms={len(rows)}  cells={summary['n_cells']}")
    print("[d2] arms reaching H=1.000:")
    for k in ("gated", "ungated", "raw"):
        n, t = summary[f"arms_reaching_1.000_{k}"]
        print(f"       {k:8s} {n}/{t}")
    print("[d2] gate+destroyed exclusion rate by T_before:")
    for k, v in summary["exclusion_rate_by_T_before"].items():
        print(f"       T_before~{k}  {v['excluded']:4d}/{v['total']:4d}  = {v['rate']:.3f}")
    dis = [r for r in rows if r["gated_max"] is not None and r["ungated_max"] is not None
           and abs(r["gated_max"] - r["ungated_max"]) > 0.067]
    print(f"[d2] arms where gated and ungated differ by more than one quantum (0.067): {len(dis)}")
    for r in dis[:12]:
        print(f"       {r['arm']:34s} gated {r['gated_max']:.4f} ({r['n_gated']}) "
              f"ungated {r['ungated_max']:.4f} ({r['n_ungated']})")
    print(f"[d2] -> {a.out}")


if __name__ == "__main__":
    main()
