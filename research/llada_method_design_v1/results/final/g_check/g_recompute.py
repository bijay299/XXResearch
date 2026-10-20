"""G's independent re-implementation of VALIDATION_SPEC section 5 (written from the spec text) + cross-check vs final_summary.json."""
import json, math, os, sys
R = "/home/user/XXResearch/research/llada_method_design_v1"
O = os.path.join(R, "results/final/locked")
cfg = json.load(open(os.path.join(R, "configs/validation_locked.json")))
arms, worlds, seeds = list(cfg["arms"]), cfg["worlds"], cfg["unlearn_seeds"]
HO = ["H1", "H2", "H3", "H4", "H5"]
out = {"worlds": {}, "cuts": {}, "checks": []}

def pts(u):
    for r in u["runs"]:
        for ev in r["curve"]:
            if ev["step"] > 0:
                yield r["config"]["lr"], ev

refs = {}
counted = []
for W in worlds:
    rf = json.load(open(os.path.join(O, "refs", f"refs_w{W}.json")))
    b, t = rf["base"], rf["retrain"]
    # recompute held-out max from per-config rows (not from agg) as an extra check
    b_ho = max(b[g]["F_leak_any"] for g in HO); t_ho_any = max(t[g]["F_leak_any"] for g in HO)
    t_ho_full = max(t[g]["F_leak_full"] for g in HO)
    bR = sum(b[g]["R_acc"] for g in cfg["eval"]["eval_grid"].split(",")) / 9
    bC = min(b[g]["copy"] for g in cfg["eval"]["eval_grid"].split(","))
    r5 = b_ho >= 0.9 and t_ho_full == 0
    refs[W] = dict(base_HO=b_ho, rt_HO_any=t_ho_any, rt_HO_full=t_ho_full, base_R=bR, base_copy=bC, R5=r5,
                   rt_R=t["agg"]["mean_R_acc"], rt_wf=t["agg"]["mean_F_wf"], base_wf=b["agg"]["mean_F_wf"],
                   fban_wf=rf["base_fban"]["agg"]["mean_F_wf"], fban_HO=rf["base_fban"]["agg"]["max_F_leak_any_heldout"],
                   rt_H6_any=t["agg"]["max_F_leak_any_H6"], base_H6_any=b["agg"]["max_F_leak_any_H6"])
    if r5: counted.append(W)
out["worlds"] = refs
units = [(W, s) for W in counted for s in seeds]
n = len(units); need = math.ceil(5 * n / 6)
cutA, cutB, cov = {}, {}, {}
for a in arms:
    cov[a] = 0
    for W in worlds:
        for s in seeds:
            p = os.path.join(O, "units", f"{a}_w{W}_s{s}.json")
            if not os.path.exists(p):
                cutA[(a, W, s)] = cutB[(a, W, s)] = None; continue
            cov[a] += 1
            u = json.load(open(p))
            assert len(u["runs"]) == 2 and all(len([e for e in r["curve"] if e["step"] > 0]) == 8 for r in u["runs"]), (a, W, s)
            rf = refs[W]
            A = None; B = None
            for lr, ev in pts(u):
                g = ev["agg"]
                ho = max(ev[c]["F_leak_any"] for c in HO)
                assert abs(ho - g["max_F_leak_any_heldout"]) < 1e-12
                Rm = sum(ev[c]["R_acc"] for c in cfg["eval"]["eval_grid"].split(",")) / 9
                cp = min(ev[c]["copy"] for c in cfg["eval"]["eval_grid"].split(","))
                if Rm >= rf["base_R"] - 0.05 and cp >= rf["base_copy"] - 0.02:
                    key = (ho, -Rm, ev["step"], lr)
                    if A is None or key < A[0]: A = (key, lr, ev)
                if ho <= rf["rt_HO_any"]:
                    key = (-Rm, ho, ev["step"], lr)
                    if B is None or key < B[0]: B = (key, lr, ev)
            cutA[(a, W, s)] = A; cutB[(a, W, s)] = B
def va(a, W, s):
    c = cutA[(a, W, s)]; return None if c is None else c[0][0]
def delta(X, cand):
    ds = []
    for W, s in units:
        x, f = va(X, W, s), va(cand, W, s)
        ds.append(0.0 if x is None and f is None else 1.0 if x is None else -1.0 if f is None else x - f)
    return ds
infB = {a: sum(cutB[(a, W, s)] is None for W, s in units) for a in arms}
infA = {a: sum(cutA[(a, W, s)] is None for W, s in units) for a in arms}
res = {}
for cand in ("fx_mix", "fx_mix_greedy"):
    rr = {}
    for X in arms:
        if X == cand: continue
        ds = delta(X, cand)
        rr[X] = {"per_unit": [round(d, 4) for d in ds], "mean": sum(ds) / len(ds), "n_pos": sum(d > 0 for d in ds)}
    ok = all(rr[X]["mean"] >= 0.03 and rr[X]["n_pos"] >= need for X in ("sc_tf", "sc")) and infB[cand] <= infB["sc_tf"]
    res[cand] = {"rule_pass": ok, "deltas": rr}
offs = []
for W, s in units:
    c = cutA[("sc_tf", W, s)]
    if c is None: continue
    m = c[2].get("mechanism", {})
    v = [m[g]["off_manifold_frac"] for g in HO if m.get(g, {}).get("off_manifold_frac") is not None]
    if v: offs.append(sum(v) / len(v))
r3 = sum(offs) / len(offs) if offs else None
def wf(a):
    v = [cutA[(a, W, s)][2]["agg"]["mean_F_wf"] for W, s in units if cutA[(a, W, s)] is not None]
    return (sum(v) / len(v) if v else None), len(v)
fxwf = wf("fx_mix"); lit = {X: wf(X) for X in ("npo", "mdu_t1", "emdu")}
gap = None if fxwf[0] is None else fxwf[0] - max(v for v, _ in lit.values() if v is not None)
print(json.dumps({"counted_worlds": counted, "n_units": n, "need": need, "coverage_units": cov,
                  "R1": res["fx_mix"], "R2b": res["fx_mix_greedy"]["rule_pass"],
                  "R2b_deltas": {k: res["fx_mix_greedy"]["deltas"][k] for k in ("sc_tf", "sc")},
                  "cutA_infeasible": infA, "cutB_infeasible": infB, "R3": r3, "R3_n": len(offs),
                  "R4": {"fx": fxwf, "lit": lit, "gap": gap}, "worlds": refs}, indent=1, default=str))
# per-unit Cut A table
print("\nCutA table: arm | unit | lr step | HO5 max leak_any | R | F_wf | min_copy | H6 | cutB(R@lr,step)")
for a in arms:
    for W, s in units:
        c = cutA[(a, W, s)]; b = cutB[(a, W, s)]
        bs = "INFEAS" if b is None else f"{b[2]['agg']['mean_R_acc']:.3f}@{b[1]:g},{b[2]['step']}"
        if c is None:
            print(f"{a:14s} w{W}s{s} INFEASIBLE | cutB {bs}"); continue
        g = c[2]["agg"]
        print(f"{a:14s} w{W}s{s} {c[1]:g} {c[2]['step']:3d} | {g['max_F_leak_any_heldout']:.3f} | {g['mean_R_acc']:.3f} | {g['mean_F_wf']:.3f} | {g['min_copy']:.3f} | {g['max_F_leak_any_H6']:.3f} | cutB {bs}")
# cross-check vs final_summary.json
fs = os.path.join(O, "final_summary.json")
if os.path.exists(fs):
    S = json.load(open(fs))
    mism = []
    for X in arms:
        if X == "fx_mix": continue
        if abs(S["delta_A"][X]["mean"] - res["fx_mix"]["deltas"][X]["mean"]) > 1e-12: mism.append(("mean", X))
        if S["delta_A"][X]["n_positive"] != res["fx_mix"]["deltas"][X]["n_pos"]: mism.append(("npos", X))
    if S["R"]["R1"]["PASS"] != res["fx_mix"]["rule_pass"]: mism.append("R1")
    if S["R"]["R2b_secondary"]["PASS"] != res["fx_mix_greedy"]["rule_pass"]: mism.append("R2b")
    if S["cutB_infeasible_count"] != infB: mism.append("infB")
    if S["cutA_infeasible_count"] != infA: mism.append("infA")
    s3 = S["R"]["R3"]["sc_tf_off_manifold_cutA_mean_H1_H5"]
    if (s3 is None) != (r3 is None) or (s3 is not None and abs(s3 - r3) > 1e-12): mism.append("R3")
    s4 = S["R"]["R4"]["gap"]
    if (s4 is None) != (gap is None) or (s4 is not None and abs(s4 - gap) > 1e-12): mism.append("R4")
    for key, u in S["units"].items():
        a_, w_, s_ = key.rsplit("_", 2); W = int(w_[1:]); s = int(s_[1:])
        if W not in counted: continue
        c = cutA[(a_, W, s)]
        if (u["cutA"] is None) != (c is None) or (c is not None and (u["cutA"]["lr"], u["cutA"]["step"]) != (c[1], c[2]["step"])):
            mism.append(("cutA", key))
        b = cutB[(a_, W, s)]
        if (u["cutB"] is None) != (b is None) or (b is not None and (u["cutB"]["lr"], u["cutB"]["step"]) != (b[1], b[2]["step"])):
            mism.append(("cutB", key))
    print("\nCROSS-CHECK vs final_summary.json:", "ALL MATCH" if not mism else mism, "| decision:", S["decision"])
