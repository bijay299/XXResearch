#!/usr/bin/env python3
"""Objection B->A: is FFX-OP's config-robust forgetting an in-sample result?

A's toy (A_scripts/sharing_check.py L146/L168) collects on-policy states from the SAME six configs C1..C6 on which
max-leak is evaluated (no held-out config). Here, in B's exact tabular world (tabular_commit_hazard.py, unigram
backoff), FFX-OP is run with (i) A's toy pool C1..C6 and (ii) A's proposed LLaDA pool mapped to L=4
  (128,32,lc,0)->(4,2,lc,0); (64,128,lc,0)->(2,4,lc,0); (128,32,random,0.7)->(4,2,random,0.7); (128,128,lc,1)->(4,4,lc,1)
with A's state mix 1/2 TF fact-free + 1/2 on-policy, A's target (exclusion, eps -> 1e-9) and A's spec E(s)=E_t
(eligible = masked positions of the current block) for on-policy states, vs. all masked positions (what A's toy did).
Evaluation on 41 native configs; leak reported separately for configs inside / outside the training pool.
Also: exact check of A's P2 KL-form (pi_T(F) <= 1-exp(-KL)) for 0<T<1."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/user/XXResearch/research/llada_method_design_v1/agents/B_scripts')
import tabular_commit_hazard as t

A_TOY = [(4, 4, 'low_confidence', 0.0), (4, 4, 'random', 1.0), (2, 2, 'low_confidence', 0.0),
         (2, 4, 'low_confidence', 1.0), (4, 2, 'random', 0.5), (2, 4, 'random', 0.0)]
A_LLADA = [(4, 2, 'low_confidence', 0.0), (2, 4, 'low_confidence', 0.0), (4, 2, 'random', 0.7),
           (4, 4, 'low_confidence', 1.0)]
PAR_FIX = [(2, 2, 'random', 1.0)]
GRID = t.EVAL_CFGS + [(N, b, r, 0.7) for (N, b) in t.EVAL_NB for r in ('low_confidence', 'random')
                      if not (b == 1 and r == 'random')]


def first_block_elig(b):
    """eligible mask per state for block length b: masked positions of the first block that still has a mask."""
    E = np.zeros((t.NS, t.L), bool)
    for idx, s in enumerate(t.STATES):
        for blk in range(t.L // b):
            pos = range(blk * b, (blk + 1) * b)
            if any(s[i] == t.MASK for i in pos):
                for i in pos:
                    E[idx, i] = s[i] == t.MASK
                break
    return E


ELIG = {b: first_block_elig(b) for b in (1, 2, 4)}


def onpolicy_W(Z, cfgs, elig_only):
    W = np.zeros((t.NS, t.L))
    for c in cfgs:
        v = t.run_sampler(Z, c, want_visits=True)['visits']
        d = np.zeros(t.NS)
        for s, m in v.items():
            d[s] += m
        d /= d.sum()
        E = ELIG[c[1]] if elig_only else t.MASKED
        W += d[:, None] * E / np.maximum(E.sum(1), 1)[:, None]
    return W / len(cfgs)


def run_arm(name, Z0, R, tf_ff, cfgs, elig_only, rounds=40, spr=10, eta=100.0):
    Z = Z0.copy()
    Wtf = tf_ff[:, None] * t.MASKED / np.maximum(t.NMASK, 1)[:, None]
    logT = np.log(R + 1e-300)
    for r in range(rounds):
        W = 0.5 * Wtf + 0.5 * onpolicy_W(Z, cfgs, elig_only)
        lam = np.minimum(eta * W, 1.0)[..., None]
        for _ in range(spr):
            Zc = Z - np.log(np.exp(Z - Z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - Z.max(-1, keepdims=True)
            Z = (1 - lam) * Zc + lam * logT
            Z[~t.MASKED] = 0.0
    leaks = {str(c): t.run_sampler(Z, c)['leak'] for c in GRID}
    inpool = [leaks[str(c)] for c in GRID if c in cfgs]
    held = {str(c): leaks[str(c)] for c in GRID if c not in cfgs}
    worst = sorted(held.items(), key=lambda kv: -kv[1])[:4]
    P, P0 = t.softmax(Z), t.softmax(Z0)
    foot = float((t.kl_rows(P0, P) * t.MASKED).sum())
    out = dict(arm=name, pool=[list(c) for c in cfgs], elig_only=elig_only, max_leak_in_pool=max(inpool),
               max_leak_heldout=max(held.values()), mean_leak_heldout=float(np.mean(list(held.values()))),
               worst_heldout=worst, footprint_kl=foot)
    print(json.dumps(out), flush=True)
    return out


def p2_counterexample():
    rows = []
    for nv in (8, 1000, 126000):
        p = np.zeros(nv); p[0] = 0.35; p[1] = 0.30; p[2:] = 0.35 / (nv - 2)   # index 1 = fact, second-ranked
        kl = -np.log(1 - p[1])                       # KL(qhat||p) with qhat = p renormalised without the fact
        for T in (0.5, 0.7, 1.0):
            w = p ** (1 / T); piF = w[1] / w.sum()
            rows.append(dict(V=nv, T=T, KL=kl, A_bound=1 - np.exp(-kl), pi_T_F=piF, violated=bool(piF > 1 - np.exp(-kl) + 1e-12),
                             B_rho_bound=p[1] / p[0]))
    for r in rows:
        print(r)
    return rows


def main():
    Z0 = t.posterior_logits(t.FORGET_T, 0.02, 'unigram', 0.02, 0)
    R = t.iproj_target(Z0, 1e-9)
    tf_ff = t.tf_dist([(t.GT, 1.0)], only_fact_masked=True)
    res = dict(grid=[list(c) for c in GRID])
    res['arms'] = [
        run_arm('FFX-OP A_toy_pool (C1..C6), all masked pos (as A toy)', Z0, R, tf_ff, A_TOY, False),
        run_arm('FFX-OP A_LLaDA_pool, E_t only (A spec)', Z0, R, tf_ff, A_LLADA, True),
        run_arm('FFX-OP A_LLaDA_pool, all masked pos', Z0, R, tf_ff, A_LLADA, False),
        run_arm('FFX-OP A_LLaDA_pool + (2,2,random,1.0), E_t only', Z0, R, tf_ff, A_LLADA + PAR_FIX, True),
    ]
    res['p2_counterexample'] = p2_counterexample()
    json.dump(res, open(sys.argv[1], 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
