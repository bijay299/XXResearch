#!/usr/bin/env python3
"""Test of the Prop.-1 rank rule, fixed before running: r = #{j : mu_j(C_F, C_R) >= tau}, tau in {2, 4}
(the default MSDA basis is raw GEP, shrink 1e-2). Compared against LoRA16/LoRA48/FT from out_main_seed*.json."""
import sys, json, os, torch
sys.path.insert(0, os.path.dirname(__file__))
import c_toy_mswe as T
seed = int(sys.argv[1]); D = T.build(seed); out = {"seed": seed}
_, mu = T.ge_basis(D, 256, "raw")
for tau in (2.0, 4.0):
    r = int((mu >= tau).sum())
    A, _ = T.ge_basis(D, r, "raw")
    fr = T.frontier(D, "fixed", T.KL_CFG, False, A_fixed=A)
    out[f"rule_tau{tau}"] = {"r": r, "frontier": fr}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print({k: (v["r"], {t: (round(x["ho_kl_mean"], 3), round(x["ho_copy_kl"], 3)) if x.get("feasible") else None for t, x in v["frontier"].items()}) for k, v in out.items() if k != "seed"})
