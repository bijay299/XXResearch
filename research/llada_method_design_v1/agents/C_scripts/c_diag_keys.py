#!/usr/bin/env python3
"""Diagnostics for section 1: shared [MASK]/template component of masked-position keys, state-dependence of keys,
and the visible-path share of the edit-site effect at [MASK] readouts (toy of c_toy_mswe.py)."""
import sys, json, math, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import c_toy_mswe as T
out = {}
for seed in (0, 1, 2):
    D = T.build(seed)
    F, R = D["F"], D["Rtr"]
    KR = R.kt; KF = F.kt
    CR = KR.T @ KR / KR.shape[0]
    ev, U = torch.linalg.eigh(CR); u1 = U[:, -1]
    mean_r = KR.mean(0)
    r = {}
    r["retain_top1_var_frac"] = (ev[-1] / ev.sum()).item()
    r["cos_top1_vs_retain_mean"] = abs((u1 @ mean_r) / mean_r.norm()).item()
    r["forget_energy_frac_on_top1"] = ((KF @ u1).pow(2).sum() / KF.pow(2).sum()).item()
    dK = F.kt - F.akt
    r["delta_energy_frac_on_top1"] = ((dK @ u1).pow(2).sum() / dK.pow(2).sum()).item()
    r["delta_over_key_energy"] = (dK.pow(2).sum() / KF.pow(2).sum()).item()
    # state dependence: cosine between the all-masked key and the TF / alternative-visible keys at the same position
    toy, o = D["toy"], D["o"]
    cs_tf, cs_alt, wh_tf, wh_alt = [], [], [], []
    CRr = CR + 1e-2 * torch.trace(CR) / CR.shape[0] * torch.eye(CR.shape[0])
    for s in range(4):
        (p, a) = T.base_fact(o, s, 0)
        toks = [list(p) + [T.MASK, T.MASK], list(p) + [a[0], T.MASK], list(p) + [T.OBJ0 + (a[0] - T.OBJ0 + 1) % T.NO, T.MASK]]
        kt, _, _, _, _ = toy.feats(torch.tensor(toks))
        dk = kt[:, 3] - toy.feats(torch.tensor([T.anchor_of(tuple(t)) for t in toks]))[0][:, 3]
        cos = lambda x, y: (x @ y / x.norm() / y.norm()).item()
        whc = lambda x, y: (x @ torch.linalg.solve(CRr, y) / math.sqrt((x @ torch.linalg.solve(CRr, x)) * (y @ torch.linalg.solve(CRr, y)))).item()
        cs_tf.append(cos(dk[0], dk[1])); cs_alt.append(cos(dk[0], dk[2]))
        wh_tf.append(whc(dk[0], dk[1])); wh_alt.append(whc(dk[0], dk[2]))
    r["fact_delta_cos_allmasked_vs_TFvisible"] = sum(cs_tf) / 4
    r["fact_delta_cos_allmasked_vs_altvisible"] = sum(cs_alt) / 4
    r["fact_delta_whitened_cos_allmasked_vs_TF"] = sum(wh_tf) / 4
    r["fact_delta_whitened_cos_allmasked_vs_alt"] = sum(wh_alt) / 4
    # visible-path share of a generic edit's effect at masked readouts (ungated): ||beta sum_{j vis} a2_ij k_j|| / ||kt_i||
    toks = torch.tensor([rec[0] for rec in R.recs]); pos = torch.tensor([rec[1] for rec in R.recs])
    kt, ktg, k, a2, gm = toy.feats(toks)
    ar = torch.arange(len(pos))
    vis_part = (toy.beta * a2 @ ((1 - gm) * k))[ar, pos]
    r["visible_path_norm_share"] = (vis_part.norm(dim=1) / kt[ar, pos].norm(dim=1)).mean().item()
    out[seed] = r
json.dump(out, open(sys.argv[1], "w"), indent=1)
for s, r in out.items():
    print(s, {k: round(v, 3) for k, v in r.items()})
