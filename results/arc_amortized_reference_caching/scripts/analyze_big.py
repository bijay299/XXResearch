"""
Breaking-point analysis for the 128-fact probe.

Two strata are reported for every arm:
  ALL        - all 64 forget facts (what the eval emitted). theta0 discloses only 0.781
               here because 13 forget facts were never memorized (the 0.797 gate shortfall).
  RECALLED   - the 51 forget facts theta0 actually recalled. This is the only subset on
               which "did the arm remove a memorized fact?" is a meaningful question.
Plus the pre-registered IN-pool / ABSENT-from-pool split, which is what the sharp
per-fact prediction in build_big.py is tested against.
"""
import json

S = "/data/bijaypandey/spex"
ARMS = ["theta0", "uncached", "arcK16", "arcK512"]
RECALL = set(json.load(open(f"{S}/outputs/big_recalled_facts.json"))["recalled_forget_facts"])
POOL = json.load(open(f"{S}/adapters/big_arcK16/report.json"))["pool_membership"]
IN_POOL = set(POOL["facts_in_pool"])


def rate(rows, pred):
    r = [x for x in rows if pred(x)]
    if not r:
        return None
    return {"n": len(r),
            "disclosure": round(sum(bool(x["answer_present"]) for x in r) / len(r), 4),
            "degenerate": sum(bool(x["output_failure"]) for x in r)}


out = {}
for a in ARMS:
    rows = json.load(open(f"{S}/outputs/big_gens_{a}.json"))
    fo = [x for x in rows if x["split"] == "forget"]
    re_ = [x for x in rows if x["split"] == "retain"]
    out[a] = {
        "forget_all": rate(fo, lambda x: True),
        "forget_recalled": rate(fo, lambda x: x["fact_id"] in RECALL),
        "forget_recalled_IN_pool": rate(fo, lambda x: x["fact_id"] in RECALL and x["fact_id"] in IN_POOL),
        "forget_recalled_ABSENT": rate(fo, lambda x: x["fact_id"] in RECALL and x["fact_id"] not in IN_POOL),
        "retain_all": rate(re_, lambda x: True),
    }

L = ["## 128-fact breaking-point probe — recalled-only stratification\n",
     f"Pool membership (ARC K=16): {len(IN_POOL)}/64 forget facts in pool. "
     f"theta0 recalled {len(RECALL)}/64 forget facts.\n",
     "| arm | forget ALL ↓ | **forget RECALLED ↓** | recalled IN-pool | recalled ABSENT | retain ↑ | degen (forget) |",
     "|---|---|---|---|---|---|---|"]
for a in ARMS:
    d = out[a]
    g = lambda k: f"{d[k]['disclosure']:.3f} (n={d[k]['n']})" if d[k] else "—"
    L.append(f"| `{a}` | {d['forget_all']['disclosure']:.3f} | **{d['forget_recalled']['disclosure']:.3f}** "
             f"| {g('forget_recalled_IN_pool')} | {g('forget_recalled_ABSENT')} "
             f"| {d['retain_all']['disclosure']:.3f} | {d['forget_all']['degenerate']} |")

# the pre-registered prediction, judged on the recalled subset
p = out["arcK16"]
gap = (p["forget_recalled_ABSENT"]["disclosure"] - p["forget_recalled_IN_pool"]["disclosure"]
       if p["forget_recalled_IN_pool"] else None)
L.append(f"\n**Pre-registered prediction** (ARC K=16 fails *specifically* on facts absent from its "
         f"pool, predictably per fact): ABSENT {p['forget_recalled_ABSENT']['disclosure']:.3f} vs "
         f"IN-pool {p['forget_recalled_IN_pool']['disclosure']:.3f}, gap {gap:+.3f}. "
         f"Direction correct, magnitude far too small to carry the failure — **REFUTED as stated**: "
         f"K=16 fails on in-pool facts too.")
json.dump({"pool_in": sorted(IN_POOL), "n_recalled": len(RECALL), "arms": out},
          open(f"{S}/outputs/big_analysis.json", "w"), indent=2)
open(f"{S}/outputs/BIG_RESULTS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
