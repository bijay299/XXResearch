#!/usr/bin/env python3
"""Extra arm: LoRA with A initialised at the GEP rows (raw C_F vs C_R) and trainable ('GEinit'), same protocol."""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))
import c_toy_mswe as T
seed = int(sys.argv[1]); D = T.build(seed); out = {"seed": seed}
for r in (16, 48):
    A, _ = T.ge_basis(D, r, "raw")
    out[f"GEinit{r}"] = T.frontier(D, "geinit", T.KL_CFG, False, A_fixed=A)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print({k: {t: (round(v["ho_kl_mean"], 3) if v.get("feasible") else None) for t, v in x.items()} for k, x in out.items() if k != "seed"})
