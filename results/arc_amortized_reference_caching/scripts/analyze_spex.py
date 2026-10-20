"""
ARC sprint analysis: cost table, efficacy table, parity verdict, K-sweep, plot.

Automatic factual metrics only in the tables. Cost numbers come from each arm's
measured report.json (forward counters and wall clocks), never from assumption.
"""
import glob
import json
import os
import statistics as st

S = "/data/bijaypandey/spex"
OUT = f"{S}/outputs"

# ── collect measured cost from each run's report ──────────────────────────
cost = {}
for p in sorted(glob.glob(f"{S}/adapters/*/report.json")):
    d = json.load(open(p))
    arm = os.path.basename(os.path.dirname(p))
    m = d.get("measured", {})
    cost[arm] = {
        "arm_kind": d["arm"], "pool_K": d["hparams"].get("pool_K"),
        "seed": d["hparams"].get("seed"),
        "fwd_student_per_step": m.get("fwd_student_per_step"),
        "fwd_reference_per_step": m.get("fwd_reference_per_step"),
        "fwd_total_per_step": m.get("fwd_total_per_step"),
        "s_per_step": m.get("s_per_step_steady"),
        "train_wall_s": m.get("train_wall_s"),
        "prefill_wall_s": m.get("prefill_wall_s"),
        "reference_forwards_total": m.get("reference_forwards_total"),
        "peak_gpu_gib": m.get("peak_gpu_gib"),
        "prop6_exactness": (d.get("gate_prop6_exactness") or {}).get(
            "max_abs_diff_cached_vs_recomputed"),
        "prop6_passed": (d.get("gate_prop6_exactness") or {}).get("passed"),
    }

for p in sorted(glob.glob(f"{S}/outputs/baselines/*.json")):
    b = os.path.basename(p)[:-5]
    if b.startswith("eval_") or b.endswith("_fact_sets"):
        continue
    d = json.load(open(p))
    m = d.get("measured", d.get("cost", {}))
    cost[f"BASELINE_{b}"] = {
        "arm_kind": b, "pool_K": None, "seed": 0,
        "fwd_student_per_step": m.get("fwd_student_per_step", m.get("forwards_per_step")),
        "fwd_reference_per_step": m.get("fwd_reference_per_step"),
        "fwd_total_per_step": m.get("forwards_per_step", m.get("fwd_total_per_step")),
        "s_per_step": m.get("seconds_per_step", m.get("s_per_step_steady")),
        "train_wall_s": m.get("train_wall_s"), "prefill_wall_s": 0,
        "reference_forwards_total": m.get("reference_forwards_total"),
        "peak_gpu_gib": m.get("peak_gpu_gib", m.get("peak_gib")),
        "prop6_exactness": None, "prop6_passed": None}

# ── collect efficacy from the frozen 48-question evaluator ────────────────
eff, gens = {}, []
for p in (sorted(x for x in glob.glob(f"{OUT}/eval_summary_*.json") if "_merged" not in x)
          + sorted(glob.glob(f"{OUT}/baselines/eval_summary_*.json"))):
    for s in json.load(open(p))["summaries"]:
        eff[s["checkpoint"]] = s
for p in (sorted(x for x in glob.glob(f"{OUT}/eval_generations_*.json") if "_merged" not in x)
          + sorted(glob.glob(f"{OUT}/baselines/eval_generations_*.json"))):
    gens += json.load(open(p))


def row(ck):
    s = eff[ck]
    f, r = s["forget_disclosure"], s["retain_accuracy"]
    return dict(
        forget=f["all"]["answer_present_rate"],
        forget_orig=f["original"]["answer_present_rate"],
        forget_para=f["paraphrase"]["answer_present_rate"],
        retain=r["all"]["answer_present_rate"],
        shared=r["shared_answer_subset"]["answer_present_rate"],
        indep=r["independent_subset"]["answer_present_rate"],
        fails=sum(s["output_failures"].values()))


L = []
L.append("## Cost — measured, not assumed\n")
L.append("| run | arm | K | fwd/step (student+ref) | s/step | train s | prefill s | ref fwd total | peak GiB | Prop6 max\\|Δ\\| |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for a, c in sorted(cost.items()):
    ex = "n/a" if c["prop6_exactness"] is None else f"{c['prop6_exactness']:.0e}"
    L.append(f"| `{a}` | {c['arm_kind']} | {c['pool_K'] or '—'} "
             f"| {c['fwd_student_per_step']}+{c['fwd_reference_per_step']}={c['fwd_total_per_step']} "
             f"| {c['s_per_step']} | {c['train_wall_s']} | {c['prefill_wall_s']} "
             f"| {c['reference_forwards_total']} | {c['peak_gpu_gib']} | {ex} |")

L.append("\n## Efficacy — frozen 48-question evaluator (24 facts x 2 phrasings)\n")
L.append("| checkpoint | forget ↓ | orig | para | retain ↑ | shared | indep | output failures |")
L.append("|---|---|---|---|---|---|---|---|")
order = (["memorized_reference"]
         + [f"uncached_baseline_step{s}" for s in (50, 150)]
         + [f"arc_K{k}_step{s}" for k in (64, 256, 1024) for s in (50, 150)]
         + [f"{a}_step150" for a in ("unc_seed1", "unc_seed2", "arc64_seed1", "arc64_seed2")]
         + [f"{a}_step{s2}" for a in ("npo", "mdu_t1", "emdu", "sc_tf") for s2 in (50, 150)])
for ck in order:
    if ck not in eff:
        continue
    d = row(ck)
    L.append(f"| `{ck}` | **{d['forget']:.3f}** | {d['forget_orig']:.3f} | {d['forget_para']:.3f} "
             f"| **{d['retain']:.3f}** | {d['shared']:.3f} | {d['indep']:.3f} | {d['fails']} |")

# ── parity verdict over seeds ─────────────────────────────────────────────
pairs = [("uncached_baseline_step150", "arc_K64_step150", 0),
         ("unc_seed1_step150", "arc64_seed1_step150", 1),
         ("unc_seed2_step150", "arc64_seed2_step150", 2)]
L.append("\n## Parity verdict (pre-registered: |Δforget| ≤ 0.05, retain not worse by > 0.05, cost ≥ 1.2x)\n")
L.append("| seed | uncached forget | ARC forget | Δforget | uncached retain | ARC retain | Δretain | parity |")
L.append("|---|---|---|---|---|---|---|---|")
deltas, ok_all = [], True
for u, a_, sd in pairs:
    if u not in eff or a_ not in eff:
        L.append(f"| {sd} | — | — | — | — | — | — | not run |")
        ok_all = False
        continue
    du, da = row(u), row(a_)
    df = da["forget"] - du["forget"]
    dr = da["retain"] - du["retain"]
    par = abs(df) <= 0.05 and dr >= -0.05
    ok_all &= par
    deltas.append((df, dr))
    L.append(f"| {sd} | {du['forget']:.3f} | {da['forget']:.3f} | {df:+.3f} "
             f"| {du['retain']:.3f} | {da['retain']:.3f} | {dr:+.3f} | {'YES' if par else 'NO'} |")

unc_s = [c["s_per_step"] for a, c in cost.items() if c["arm_kind"] == "uncached" and c["s_per_step"]]
arc_s = [c["s_per_step"] for a, c in cost.items()
         if c["arm_kind"] == "arc" and c["pool_K"] == 64 and c["s_per_step"]]
speed = (st.mean(unc_s) / st.mean(arc_s)) if unc_s and arc_s else None
unc_ref = [c["reference_forwards_total"] for a, c in cost.items() if c["arm_kind"] == "uncached"]
arc_ref = [c["reference_forwards_total"] for a, c in cost.items()
           if c["arm_kind"] == "arc" and c["pool_K"] == 64]
refx = (st.mean(unc_ref) / st.mean(arc_ref)) if unc_ref and arc_ref else None

verdict = ("SUPPORTED — efficiency at parity" if ok_all and speed and speed >= 1.2
           else "EFFICIENCY-ONLY" if speed and speed >= 1.2
           else "NOT SUPPORTED")
L.append(f"\n**Measured speedup (s/step, K=64 vs uncached, mean over seeds): "
         f"{speed:.3f}x**" if speed else "\nspeedup: n/a")
L.append(f"\n**Reference-forward reduction: {refx:.1f}x**" if refx else "")
L.append(f"\n**VERDICT: {verdict}**")
if deltas:
    L.append(f"\nΔforget across seeds: {[round(d[0],3) for d in deltas]}  "
             f"Δretain: {[round(d[1],3) for d in deltas]}")

summary = {"cost": cost, "efficacy": {k: row(k) for k in eff},
           "parity_pairs": [{"seed": sd, "uncached": u, "arc": a_} for u, a_, sd in pairs],
           "speedup_s_per_step": speed, "reference_forward_reduction": refx,
           "verdict": verdict, "n_generations": len(gens)}
json.dump(summary, open(f"{OUT}/analysis.json", "w"), indent=2)
open(f"{OUT}/RESULTS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\n[written] outputs/RESULTS.md, outputs/analysis.json  ({len(gens)} generations)")
