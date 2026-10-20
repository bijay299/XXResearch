#!/usr/bin/env python3
"""Compact text summary of tabular_commit_hazard_out.json (matched-forgetting table, worst configs, bounds)."""
import json, sys

d = json.load(open(sys.argv[1]))
TRAIN = [[4, 4, 'low_confidence', 0.0], [4, 4, 'random', 1.0], [2, 2, 'random', 1.0]]


def fmt_cfg(c):
    return f"N{c[0]}/b{c[1]}/{'lc' if c[2] == 'low_confidence' else 'rnd'}/T{c[3]}"


for key, res in d.items():
    if key == 'params':
        continue
    cfgs = res['eval_cfgs']
    held = [i for i, c in enumerate(cfgs) if c not in TRAIN]
    print(f"\n######## {key}   ({len(cfgs)} native configs, {len(held)} held out from OP_mix training)")
    print(f"{'arm':10s} | {'first round leak_max<=0.05':>34s} | {'first round leak_mean<=0.01':>34s} | final: mean / max / max_heldout / footKL / nEdit / retainF")
    for a, h in res['arms'].items():
        def first(pred):
            x = [e for e in h if pred(e)]
            if not x:
                return 'INFEASIBLE in budget'.rjust(34)
            e = x[0]
            return f"r{e['round']:2d} mean={e['leak_mean']:.4f} KL={e['footprint_kl']:6.0f} nE={e['n_edited']:4d}"
        f = h[-1]
        mh = max(f['leak_by_cfg'][i] for i in held)
        print(f"{a:10s} | {first(lambda e: e['leak_max'] <= 0.05):>34s} | {first(lambda e: e['leak_mean'] <= 0.01):>34s} | "
              f"{f['leak_mean']:.4f} / {f['leak_max']:.4f} / {mh:.4f} / {f['footprint_kl']:.0f} / {f['n_edited']} / {f['retain_recall_mean']:.3f}")
        worst = sorted(zip(f['leak_by_cfg'], [fmt_cfg(c) for c in cfgs]), reverse=True)[:3]
        print(f"{'':10s}   worst final configs: " + ", ".join(f"{c}={v:.3f}" for v, c in worst))
    print("\nTF-trained model: fraction of fact-free visitation mass outside supp(TF), per config (min/max):",
          f"{min(res['TF_model_ff_visit_mass_outside_suppTF']):.3f} / {max(res['TF_model_ff_visit_mass_outside_suppTF']):.3f}")
    for a, rows in res['bounds_after'].items():
        tot = sum(r['leak'] for r in rows)
        out = sum(r['leak_from_outside_TF'] for r in rows)
        print(f"[{a}] share of total leak (summed over configs) committed from states outside supp(TF): "
              f"{(out / tot if tot > 0 else 0):.3f}  (total leak {tot:.3f})")
    print("\nBound tightness (exact leak vs B1=q-union, B2=rho-union [all T<=1], B3=sup bound)")
    for a in ['base', 'TF', 'OP_mix']:
        rows = res['base_bounds'] if a == 'base' else res['bounds_after'][a]
        print(f"  -- {a}")
        for r in rows:
            c = r['cfg']
            if c[1] == 4 and c[0] in (4, 1) or (c[0] == 2 and c[1] == 2 and c[2] == 'random'):
                print(f"     {fmt_cfg(c):18s} leak={r['leak']:.4f} B1={r['B1']:.4f} B2={r['B2']:.4f} B3={r['B3']:.4f} "
                      f"B1/leak={(r['B1'] / r['leak'] if r['leak'] > 1e-12 else float('inf')):.2f}")
print("\nparams:", d['params'])
