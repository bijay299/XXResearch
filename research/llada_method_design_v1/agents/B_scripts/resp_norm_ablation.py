#!/usr/bin/env python3
"""RESP_B check: per-position normalisation of the forget loss, mean over M(s) (SPEC v1.1) vs sum/L (the hazard
integrand's weighting). Same exact tabular world, B's OP_mix pool, rounds 1..20, held-out max leak over B's 33-config grid."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/user/XXResearch/research/llada_method_design_v1/agents/B_scripts')
import tabular_commit_hazard as t

def run(norm, rounds=20, spr=10, eta=100.0, eps=1e-4):
    Z0 = t.posterior_logits(t.FORGET_T, 0.02, 'unigram', 0.02, 0)
    R = t.iproj_target(Z0, eps); logT = np.log(R + 1e-300)
    tf_ff = t.tf_dist([(t.GT, 1.0)], only_fact_masked=True)
    Z = Z0.copy(); hist = []
    for r in range(rounds):
        D = 0.75 * t.onpolicy_dist(Z, t.TRAIN_CFGS) + 0.25 * tf_ff
        den = np.maximum(t.NMASK, 1)[:, None] if norm == 'mean_M' else float(t.L)
        lam = np.minimum(eta * D[:, None] * t.MASKED / den, 1.0)[..., None]
        for _ in range(spr):
            Zc = Z - np.log(np.exp(Z - Z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - Z.max(-1, keepdims=True)
            Z = (1 - lam) * Zc + lam * logT; Z[~t.MASKED] = 0.0
        if r in (0, 1, 3, 7, 19):
            lk = [t.run_sampler(Z, c)['leak'] for c in t.EVAL_CFGS]
            held = [l for l, c in zip(lk, t.EVAL_CFGS) if c not in t.TRAIN_CFGS]
            foot = float((t.kl_rows(t.softmax(Z0), t.softmax(Z)) * t.MASKED).sum())
            hist.append(dict(round=r + 1, max_all=max(lk), max_heldout=max(held), mean_all=float(np.mean(lk)), footprint_kl=foot))
            print(norm, hist[-1], flush=True)
    return hist

out = {n: run(n) for n in ('mean_M', 'sum_over_L')}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
