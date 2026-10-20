import sys, time, json, math, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))  # seed-0 probe; output saved as out_probe_shrink_rank_s0.txt
import c_toy_mswe as T
seed=int(sys.argv[1])
D=T.build(seed)
for which in ("delta","raw"):
  for shrink in (1e-2, 1e-1):
    for rank in (16, 48):
      A,_=T.ge_basis(D,rank,which,shrink=shrink)
      t=time.time(); fr=T.frontier(D,"fixed",T.KL_CFG,False,A_fixed=A)
      print(which,shrink,rank, round(time.time()-t,1), {thr:(round(r["ho_kl_mean"],3) if r.get("feasible") else None) for thr,r in fr.items()}, flush=True)
