#!/usr/bin/env python3
"""Agent C -> B objection check: on a SHARED-parameter mask predictor, where does the FCH-Mix edit live?

Substrate: Agent A's exact log-linear mask predictor (A_scripts/sharing_check.py, imported read-only):
  logits(c,s,j) = b[j] + U[c,j] + S[subj(c),j] + R[rel(c),j] + sum_{k!=j} W[j,k,s_k]
  prompts F=(X,when) [forget, fact token 1951], R1=(Y,when), R2=(X,where), R3=(Z,when) [answer 'born 1951'].
Arms (same exclusion/I-projection target r_eps of theta0 at the same state, forward KL, + retain KL, L-BFGS):
  TF   : fact-free teacher-forced states (B's TF_ff / alpha=1 limit)
  OP   : fact-free states visited by the student's own native samplers, all 6 configs, 3 DAgger rounds (FCH-Mix
         state measure; A's FFX_OP arm reproduced)
  OP-g : OP with the generic all-masked state's weight capped at its TF_ff weight (the down-weighting B's own sec.6 asks for)
Reports, per lambda: max-over-configs leakage, retain decode-TV, R3 recall (retained fact sharing the token 1951),
the on-policy weight on the all-masked state, and the decomposition of the 1951-logit change at R3's masked states into
parameter groups: shared (b, W[.,.,MASK], R[when]) = a learned prompt-independent token ban vs the rest.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "A_scripts"))
import sharing_check as sc      # noqa: E402  (read-only import)
import tabular_check as T       # noqa: E402

torch.set_default_dtype(torch.float64)
MASK, L, V = T.MASK, T.L, T.V
F51 = T.F51


def excl_rows(P, s):
    Q = P.copy()
    for j in range(L):
        if s[j] == MASK:
            Q[j] = T.excl(Q[j])
    return Q


def decompose(th, th0):
    """1951-logit change at R3's masked TF states, by parameter group (mean over states x masked positions)."""
    d = {k: th[k] - th0[k] for k in th}
    rows = []
    for y in T.SPEC["R3"]:
        for s in sc.tf_states(y):
            for j in range(L):
                if s[j] != MASK:
                    continue
                g = {"b(global)": d["b"][j, F51].item(),
                     "R[when](shared rel)": d["R"][sc.REL["R3"], j, F51].item(),
                     "W[.,.,MASK](shared mask feats)": sum(d["W"][j, k, s[k], F51].item() for k in range(L)
                                                         if k != j and s[k] == MASK),
                     "W[.,.,visible tok]": sum(d["W"][j, k, s[k], F51].item() for k in range(L)
                                               if k != j and s[k] != MASK),
                     "U[R3]+S[Z](R3-specific)": d["U"][sc.PR.index("R3"), j, F51].item() + d["S"][sc.SUBJ["R3"], j, F51].item()}
                rows.append(g)
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["total"] = float(sum(out.values()))
    return out


def main():
    t0 = time.time()
    p_init = sc.init_params()
    bd = sc.base_data()
    theta0 = sc.lbfgs(lambda p: sc.ce_soft(p, bd) + 1e-4 * (sc.flat(p) ** 2).sum(), p_init, iters=500)
    fam0 = {c: sc.model_family(theta0, c) for c in sc.PR}
    base_dec = {c: {cn: T.decode(fam0[c], cfg) for cn, cfg in sc.CFGS.items()} for c in ["R1", "R2", "R3"]}
    base = sc.evaluate(theta0, base_dec)
    ret = []
    for c in ["R1", "R2", "R3"]:
        for y, py in T.SPEC[c].items():
            for s in sc.tf_states(y):
                if MASK in s:
                    ret.append((c, s, fam0[c](s), py))
    ffF = list(T.TF_FF)
    allm = (MASK,) * L
    tgt = lambda s: excl_rows(fam0["F"](s), s)
    prox = lambda p: sc.MU * ((sc.flat(p) - sc.flat(theta0)) ** 2).sum()
    res = {"base": {"max_leak": base["max_leak"], "R3_recall": base["retain_fact_recall"]["R3"]}, "arms": {}}
    for arm in ("TF", "OP", "OP-g"):
        rows = []
        for lam in (3, 10, 30, 100):
            if arm == "TF":
                agg = {s: 1.0 for s in ffF}
                it = [("F", s, tgt(s), w) for s, w in agg.items()]
                th = sc.lbfgs(lambda p: lam * sc.ce_soft(p, it) + sc.ce_soft(p, ret) + prox(p), theta0)
            else:
                agg = {s: 1.0 for s in ffF}
                th = theta0
                for rnd in range(3):
                    for s, w in sc.onpolicy_ff_states(th, fam0["F"]).items():
                        agg[s] = agg.get(s, 0.0) + w
                    use = dict(agg)
                    if arm == "OP-g":       # cap the generic all-masked state at its TF_ff share
                        tot_other = sum(w for s, w in use.items() if s != allm)
                        use[allm] = min(use.get(allm, 0.0), tot_other / max(1, len(use) - 1))
                    it = [("F", s, tgt(s), w) for s, w in use.items() if MASK in s]
                    th = sc.lbfgs(lambda p: lam * sc.ce_soft(p, it) + sc.ce_soft(p, ret) + prox(p), theta0)
            tot = sum(w for _, _, _, w in it)
            ev = sc.evaluate(th, base_dec)
            row = {"lam": lam, "max_leak": ev["max_leak"], "retainTV": ev["retain_decodeTV_mean"],
                   "recall": ev["retain_fact_recall"], "w_allmasked": float(sum(w for _, s, _, w in it if s == allm) / tot),
                   "n_states": len(it), "dz1951_at_R3": decompose(th, theta0)}
            rows.append(row)
            print(arm, lam, "maxleak %.3f TV %.3f R3 %.3f w_allmask %.2f n %d" % (
                row["max_leak"], row["retainTV"], row["recall"]["R3"], row["w_allmasked"], row["n_states"]),
                {k: round(v, 3) for k, v in row["dz1951_at_R3"].items()}, flush=True)
        res["arms"][arm] = rows
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(HERE, "objection_fchmix_tokenban_out.json"), "w"), indent=1, default=float)
    print("base", res["base"], "wall %.1fs" % res["wall_s"])


if __name__ == "__main__":
    main()
