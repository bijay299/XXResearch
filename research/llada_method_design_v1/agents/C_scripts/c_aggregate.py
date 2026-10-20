#!/usr/bin/env python3
"""Aggregate out_main_seed*.json and out_sweep_*.json into a text table (c_summary.txt)."""
import json, glob, os, statistics as st, sys
S = os.path.dirname(os.path.abspath(__file__))
lines = []
P = lambda *a: lines.append(" ".join(str(x) for x in a))
mains = [json.load(open(f)) for f in sorted(glob.glob(f"{S}/out_main_seed*.json"))]
def ms(v):
    v = [x for x in v if x is not None]
    if not v: return "   n/a      "
    return f"{st.mean(v):6.3f}±{(st.pstdev(v) if len(v)>1 else 0):.3f}"
if mains:
    P(f"MAIN: seeds={[m['seed'] for m in mains]}  matched forgetting = mean p(true) over forget TF states <= THR (chance 1/12=0.083)")
    pre = [m["main"]["pre"] for m in mains]
    P("pre-edit:", {k: ms([p[k] for p in pre]) for k in pre[0]})
    for key in ("sep_delta", "sep_raw"):
        P(key, [[round(x, 1) for x in m["main"][key]["mu_top"]] + [round(m["main"][key]["mu_16"], 1)] for m in mains])
    methods = [k for k in mains[0]["main"] if k not in ("pre", "sep_delta", "sep_raw")]
    cols = ["ho_kl_mean", "ho_subj_nb_kl", "ho_rel_nb_kl", "ho_other_kl", "ho_copy_kl", "ho_dp_mean", "rtr_kl",
            "forget_alt_p", "forget_gen_par", "forget_gen_l2r", "forget_gen_conf", "ho_fact_gen_l2r"]
    for thr in ("0.2", "0.15", "0.1"):
        P(f"\n=== THR {thr} ===  (mean±sd over seeds; feas = #seeds reaching THR)")
        P(f"{'method':14s} feas " + " ".join(f"{c[:14]:>13s}" for c in cols))
        for mth in methods:
            rows = [m["main"][mth][thr] for m in mains]
            ok = [r for r in rows if r.get("feasible")]
            P(f"{mth:14s} {len(ok)}/{len(rows)} " + " ".join(f"{ms([r.get(c) for r in ok]):>13s}" for c in cols))
sw = sorted(glob.glob(f"{S}/out_sweep_*.json"))
if sw:
    P("\nSWEEP (failure boundary): forget-subject embedding -> (1-a) own + a partner(held-out, same relation)")
    P(f"{'alpha':>5s} {'seed':>4s} {'cosE':>6s} {'rhoA':>6s} {'mu1':>7s} {'mu16':>6s} {'pre_fgt_p':>9s} | "
      + " | ".join(f"{m}@{t}: hoKL relKL" for m in ("MSDA16raw", "LoRA16", "FT") for t in ("0.15", "0.1")))
    for f in sw:
        d = json.load(open(f)); r = d["sweep"]
        cells = []
        for m in ("MSDA16raw", "LoRA16", "FT"):
            for t in ("0.15", "0.1"):
                x = r[m][t]
                cells.append(f"{x['ho_kl_mean']:.3f} {x['ho_rel_nb_kl']:.3f}" if x.get("feasible") else "  infeasible ")
        P(f"{d['alpha']:5.2f} {d['seed']:4d} {r['cos_euclid']:6.3f} {r['rho_in_MSDA_space']:6.3f} {r['mu_top'][0]:7.1f} "
          f"{r['mu_16']:6.1f} {r['pre']['forget_tf_p0']:9.3f} | " + " | ".join(cells))
open(f"{S}/c_summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- paired per-seed comparisons (held-out retain KL and copy KL), incl. GE-init arm
gi = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{S}/out_geinit_seed*.json")}
if mains:
    L2 = []
    pairs = [("GEraw16", "LoRA16"), ("GEraw48", "LoRA48"), ("MSDA16", "LoRA16"), ("MSDA48", "LoRA48"),
             ("GEraw16", "FT"), ("GEraw48", "RandFixed48"), ("GEinit16", "LoRA16"), ("GEinit48", "LoRA48"),
             ("MSDA48-g", "MSDA48"), ("FT-g", "FT"), ("CF-ours", "CF-iso"), ("CF-ours", "CF-visC")]
    for thr in ("0.15", "0.1"):
        L2.append(f"\nPAIRED (per seed) THR {thr}: metric a | b  [ho_kl_mean ; ho_copy_kl ; forget_alt_p]")
        for a, b in pairs:
            cells = []
            for m in mains:
                sd = m["seed"]
                get = lambda k: (gi[sd][k][thr] if k.startswith("GEinit") else m["main"][k][thr])
                ra, rb = get(a), get(b)
                if ra.get("feasible") and rb.get("feasible"):
                    cells.append(f"s{sd}: {ra['ho_kl_mean']:.3f}|{rb['ho_kl_mean']:.3f} ; {ra['ho_copy_kl']:.3f}|{rb['ho_copy_kl']:.3f} ; {ra['forget_alt_p']:.3f}|{rb['forget_alt_p']:.3f}")
                else:
                    cells.append(f"s{sd}: infeasible({a if not ra.get('feasible') else b})")
            L2.append(f"  {a:9s} vs {b:11s} " + "   ".join(cells))
    open(f"{S}/c_summary.txt", "a").write("\n".join(L2) + "\n")
    print("\n".join(L2))
