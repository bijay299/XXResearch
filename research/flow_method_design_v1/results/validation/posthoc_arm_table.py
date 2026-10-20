"""POST-HOC DESCRIPTIVE analysis (coordinator, after the verdict was computed). Not part of the frozen decision rules.
For each (family, cell, seed, arm), select the config with the lowest interpolated rho_sel at E_sel = 0.8 using the
frozen runner's own select_config/matched; report per family/cell the median over seeds of rho_test at that crossing,
the number of seeds whose arm reached E = 0.8, and the arm with the lowest median. No run is executed."""
import collections, glob, gzip, json, math, os, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import run_validation as rv

E = 0.8
FAMS = sys.argv[1:] or ["G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9"]
out = {}
for fam in FAMS:
    for p in sorted(glob.glob(os.path.join(HERE, "raw", fam, "*.jsonl.gz"))):
        cell = os.path.basename(p).split("__")[0]
        runs = collections.defaultdict(list)
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            if r.get("arm") and r.get("status") == "ok":
                runs[r["arm"]].append(r)
        for arm, rs in runs.items():
            b = rv.select_config(rs, E)
            d = out.setdefault(f"{fam}/{cell}", {}).setdefault(arm, {"rho": [], "reached": 0, "n": 0})
            d["n"] += 1
            if b is not None and math.isfinite(b[0].get("rho_test", math.nan)):
                d["rho"].append(b[0]["rho_test"]); d["reached"] += 1
table = {}
for fc, arms in out.items():
    rows = {a: {"median_rho_test": (statistics.median(v["rho"]) if v["rho"] else None),
                "seeds_reached": v["reached"], "seeds": v["n"]} for a, v in arms.items()}
    ok = {a: r["median_rho_test"] for a, r in rows.items() if r["median_rho_test"] is not None and r["seeds_reached"] == r["seeds"]}
    table[fc] = {"arms": rows, "lowest_median_all_seeds_reached": min(ok, key=ok.get) if ok else None}
json.dump(table, open(os.path.join(HERE, "posthoc_arm_table.json"), "w"), indent=1)
for fc, t in table.items():
    v = t["arms"].get("vau", {})
    best = t["lowest_median_all_seeds_reached"]
    print(f"{fc:28s} best={best!s:26s} vau={v.get('median_rho_test')!s:.10s} reached={v.get('seeds_reached')}/{v.get('seeds')}"
          f"  best_rho={(t['arms'][best]['median_rho_test'] if best else None)!s:.10s}")
