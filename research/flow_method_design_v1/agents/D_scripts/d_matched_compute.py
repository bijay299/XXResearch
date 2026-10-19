"""P-D4: native GEM prefix vs uniform 11-of-28 on-policy timesteps at matched forward compute (fresh seeds 20-27).
Forward-equivalents per iteration (F = one student/teacher forward; student fwd+bwd ~ 3F):
  prefix: rollout 11F (no grad; GEM's 11 latents need 11 steps) + 11 latents x (2 teacher + 3) = 66F  -> 23 iters
  uniform11-of-28: rollout ~27F (needs up to step 27) + 55F = 82F -> floor(23*66/82) = 18 iters"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d_window_toy as T

C = T.cases(); out = {}; wins = 0; losses = 0
for cname in ["UNI", "BI_FINE", "BI_COARSE"]:
    for kind in ["SMOOTH", "SHARED"]:
        for lr in [0.05, 0.2]:
            r = {}
            for lab, kw, budget in [("prefix", dict(window="prefix"), 250), ("uniform11", dict(window="uniform11"), 18 * 11)]:
                T.BUDGET = budget
                sc = [T.evaluate(C[cname], T.train(C[cname], kind, seed=s, lr=lr, **kw)[0], n=1000)["score_norm"] for s in range(20, 28)]
                r[lab] = dict(mean=float(np.mean(sc)), se=float(np.std(sc, ddof=1) / np.sqrt(8)))
            d = r["prefix"]["mean"] - r["uniform11"]["mean"]
            wins += d > 0.05; losses += d < -0.05
            out[f"{cname}|{kind}|lr{lr}"] = dict(r, diff_prefix_minus_uniform=float(d))
            print(cname, kind, lr, {k: round(v["mean"], 3) for k, v in r.items()}, "diff %.3f" % d, flush=True)
T.BUDGET = 250
out["strict_wins_uniform"] = int(wins); out["strict_losses_uniform"] = int(losses)
out["P_D4_verdict"] = "SUPPORTED" if wins >= 7 else ("FALSIFIED" if wins <= 4 else "INCONCLUSIVE")
print(out["strict_wins_uniform"], out["strict_losses_uniform"], out["P_D4_verdict"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_matched_compute_output.json"), "w"), indent=1)
