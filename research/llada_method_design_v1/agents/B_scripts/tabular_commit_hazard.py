#!/usr/bin/env python3
"""Agent B -- exact tabular check of first-commit-hazard (FCH) unlearning for LLaDA-style samplers.

World: one forget prompt, answer length L=4, vocab V={B,I,F,Y,W,U,E} (F = fact token, E = EOS), +[MASK].
All 8^4 = 4096 answer states are enumerated. The base mask predictor p0(.|s,i) is the exact Bayes
posterior of a data distribution pi over answers (= the optimum of the LLaDA SFT objective for pi):
  pi = (1-delta) * sum_t w_t [y = template_t] + delta * prod_i u(y_i)        (u = backoff unigram)
plus small fixed logit noise (tie breaking). Native LLaDA sampler semantics are reproduced exactly
(generate.py @9182493): semi-AR blocks, uniform transfer schedule (base + remainder to first steps),
x0 = argmax(logits + Gumbel*T) (== sample softmax(logits/T); T=0 greedy), confidence = softmax(logits)[x0]
(low_confidence) or U(0,1) (random), top-k among masked positions of the current block, no remasking.
Leakage probability = P(some F token is ever committed), computed EXACTLY by DP over states
(trajectories are absorbed at the first F commit -- post-leak states are irrelevant to leakage).

Unlearning arms are tabular "training" runs: logits z(s,i,.) are moved by clipped gradient steps on
  sum_s D(s) * mean_{i masked} KL( r(.|s,i) || softmax z(s,i) ),
with the state distribution D defining the arm. r = I-projection target (fact mass of the frozen p0 scaled
to eps, rest renormalised) unless stated. Retention proxies: (i) edit footprint on the forget prompt
(sum of KL(p0||p_theta) over all (state, masked position)); (ii) a retain prompt whose answer legitimately
contains F, with edit transfer kappa: z_r = z0_r + kappa*(z_f - z0_f) (prompt-independent sharing model).
Everything is exact (no sampling) except that "training" is a deterministic caricature of SGD.
"""
import itertools, json, sys, time
from collections import defaultdict
from math import comb
import numpy as np

TOK = ['B', 'I', 'F', 'Y', 'W', 'U', 'E']
NV, MASK, L = 7, 7, 4
FTOK = np.array([2])
NS = 8 ** L
POW = np.array([8 ** i for i in range(L)])
STATES = [tuple((idx // 8 ** i) % 8 for i in range(L)) for idx in range(NS)]
ALL_MASKED = int(sum(MASK * POW))
MASKED = np.array([[s[i] == MASK for i in range(L)] for s in STATES])          # (NS, L)
NMASK = MASKED.sum(1)
HAS_F = np.array([any(v in FTOK for v in s) for s in STATES])
COMBOS = {m: np.array(list(itertools.product(range(NV), repeat=m)), dtype=np.int64).reshape(-1, m)
          for m in range(1, L + 1)}

FORGET_T = [((0, 1, 2, 6), 0.55),   # B I F E   "born in F"   (ground truth)
            ((2, 4, 3, 6), 0.25),   # F W Y E   "F was (the) year"
            ((3, 4, 2, 6), 0.10),   # Y W F E   "year was F"
            ((5, 6, 6, 6), 0.10)]   # U E E E   "unknown"
GT = (0, 1, 2, 6)
RETAIN_T = [((1, 2, 6, 6), 0.70),   # I F E E   (retain answer legitimately contains F)
            ((2, 6, 6, 6), 0.20),   # F E E E
            ((5, 6, 6, 6), 0.10)]   # U E E E


def unigram(templates):
    u = np.zeros(NV)
    for t, w in templates:
        for v in t:
            u[v] += w
    return u / u.sum()


def posterior_logits(templates, delta, backoff, sigma, seed):
    u = 0.9 * unigram(templates) + 0.1 / NV if backoff == 'unigram' else np.full(NV, 1.0 / NV)
    rng = np.random.default_rng(seed)
    Z = np.zeros((NS, L, NV))
    for idx, s in enumerate(STATES):
        vis = [j for j in range(L) if s[j] != MASK]
        if NMASK[idx] == 0:
            continue
        cons = [(t, w) for t, w in templates if all(t[j] == s[j] for j in vis)]
        bo = delta * np.prod([u[s[j]] for j in vis])
        for i in range(L):
            if s[i] != MASK:
                continue
            p = bo * u
            for t, w in cons:
                p[t[i]] += (1 - delta) * w
            Z[idx, i] = np.log(p / p.sum())
    Z += sigma * rng.standard_normal(Z.shape)
    Z[~MASKED] = 0.0
    return Z


def softmax(Z, T=1.0):
    Y = Z / T
    Y = Y - Y.max(-1, keepdims=True)
    E = np.exp(Y)
    return E / E.sum(-1, keepdims=True)


def run_sampler(Z, cfg, want_visits=False, attrib_support=None):
    """Exact DP of native LLaDA generate() on the tabular model.
    cfg = (steps N, block_length b, remasking, temperature T). Returns dict with exact leak prob, the
    first-commit decomposition check, bounds B1 (q-union), B2 (rho-union, sampler-uniform for T<=1),
    B3 (sup bound), fact-free visitation (optional) and leak mass originating outside a given support."""
    N, b, rem, T = cfg
    nb = L // b
    assert L % b == 0 and N % nb == 0
    spb = N // nb
    P = softmax(Z)
    Q = P if T == 0 else softmax(Z, T)
    pF = P[..., FTOK].sum(-1)
    rho = pF / P.max(-1)                       # p(F)/p_max per (s,i)
    if T == 0:
        qF = (np.isin(P.argmax(-1), FTOK)).astype(float)
    else:
        qF = Q[..., FTOK].sum(-1)
    dist = {ALL_MASKED: 1.0}
    leak = decomp = B1 = B2 = 0.0
    sup_rho = 0.0
    n_elig_total = 0
    visits = defaultdict(float)
    leak_out = 0.0
    for blk in range(nb):
        bpos = list(range(blk * b, (blk + 1) * b))
        ks = [b // spb + (1 if i < b % spb else 0) for i in range(spb)]
        for st in range(spb):
            k = ks[st]
            n_elig_total += len(bpos) - sum(ks[:st])
            nd = defaultdict(float)
            for sidx, m in dist.items():
                s = STATES[sidx]
                elig = [i for i in bpos if s[i] == MASK]
                if want_visits:
                    visits[sidx] += m
                B1 += m * qF[sidx, elig].sum()
                B2 += m * rho[sidx, elig].sum()
                sup_rho = max(sup_rho, rho[sidx, elig].max())
                h, trans = transitions(sidx, elig, k, P, Q, rem, T)
                leak += m * h
                decomp += m * h
                if attrib_support is not None and not attrib_support[sidx]:
                    leak_out += m * h
                for nidx, pr in trans:
                    nd[nidx] += m * pr
            dist = {s: v for s, v in nd.items() if v > 1e-14}
    out = dict(leak=leak, B1=B1, B2=B2, B3=min(1.0, n_elig_total * sup_rho), sup_rho=sup_rho,
               n_elig_total=n_elig_total, survive=sum(dist.values()), leak_from_outside=leak_out,
               final=dist)
    if want_visits:
        out['visits'] = visits
    return out


def transitions(sidx, elig, k, P, Q, rem, T):
    """All transitions from state sidx: returns (leak prob, [(next_idx, prob)])."""
    e = np.array(elig)
    m = len(e)
    trans = defaultdict(float)
    leakp = 0.0
    if rem == 'low_confidence':
        if T == 0:
            x0 = P[sidx, e].argmax(-1)
            conf = P[sidx, e, x0]
            order = np.argsort(-conf, kind='stable')[:k]
            sel_pos, sel_tok = e[order], x0[order]
            if np.isin(sel_tok, FTOK).any():
                return 1.0, []
            return 0.0, [(int(sidx + ((sel_tok - MASK) * POW[sel_pos]).sum()), 1.0)]
        C = COMBOS[m]                                   # (7^m, m)
        jp = np.prod(Q[sidx, e[None, :], C], axis=1)
        conf = P[sidx, e[None, :], C]
        order = np.argsort(-conf, axis=1, kind='stable')[:, :k]
        sel_tok = np.take_along_axis(C, order, 1)
        sel_pos = e[order]
        lk = np.isin(sel_tok, FTOK).any(1)
        leakp = jp[lk].sum()
        nxt = sidx + ((sel_tok - MASK) * POW[sel_pos]).sum(1)
        for n_, p_ in zip(nxt[~lk], jp[~lk]):
            trans[int(n_)] += p_
        return leakp, list(trans.items())
    # random remasking: subset independent of draws
    subsets = list(itertools.combinations(range(m), k))
    ws = 1.0 / len(subsets)
    for S in subsets:
        pos = e[list(S)]
        if T == 0:
            tok = P[sidx, pos].argmax(-1)
            if np.isin(tok, FTOK).any():
                leakp += ws
            else:
                trans[int(sidx + ((tok - MASK) * POW[pos]).sum())] += ws
        else:
            C = COMBOS[k]
            jp = np.prod(Q[sidx, pos[None, :], C], axis=1) * ws
            lk = np.isin(C, FTOK).any(1)
            leakp += jp[lk].sum()
            nxt = sidx + ((C - MASK) * POW[pos][None, :]).sum(1)
            for n_, p_ in zip(nxt[~lk], jp[~lk]):
                trans[int(n_)] += p_
    return leakp, list(trans.items())


EVAL_NB = [(4, 4), (2, 4), (1, 4), (4, 2), (2, 2), (4, 1)]     # (steps, block_length)
EVAL_CFGS = [(N, b, rem, T) for (N, b) in EVAL_NB for rem in ('low_confidence', 'random')
             for T in (0.0, 0.5, 1.0) if not (b == 1 and rem == 'random')]
# v1 (first run) used 4 narrow configs {(4,4,lc,0),(4,4,rand,0),(4,2,lc,1),(2,4,lc,.5)}: training configs closed,
# held-out random+temperature configs leaked 0.09-0.21 (states like 'Y__E' had ~1e-8 training weight).
# v2: deployment default (greedy low-confidence) + one 'wide' native config (random order, T=1, 1 token/step),
# whose support contains every fact-free decision state of every native (steps, block) schedule.
# v3 finding: with only sequential (1 token/step) wide rollouts, the residual leak sat in (steps=2, block=2)
# schedules: committing 2 tokens in one step draws them independently from per-position marginals, producing
# incoherent 'parallel-commit' decision states that 1-token-per-step rollouts rarely visit.
# v4 mixture: default greedy low-conf + wide sequential + wide parallel-commit schedule (all T=1 random order).
TRAIN_CFGS = [(4, 4, 'low_confidence', 0.0), (4, 4, 'random', 1.0), (2, 2, 'random', 1.0)]
WIDE_CFG = [(4, 4, 'random', 1.0)]
SINGLE_CFG = [(4, 4, 'low_confidence', 0.0)]      # MDU _traj_rollout_na_loss analogue (greedy, low-conf, 1 block)


def iproj_target(Z0, eps):
    P0 = softmax(Z0)
    pF = P0[..., FTOK].sum(-1, keepdims=True)
    isF = np.zeros(NV, bool); isF[FTOK] = True
    scaleF = np.where(pF > eps, eps / pF, 1.0)
    scaleN = np.where(pF > eps, (1 - eps) / (1 - pF), 1.0)
    return P0 * np.where(isF, scaleF, scaleN)


def tf_dist(templates, only_fact_masked=False):
    """MDU/LLaDA-SFT teacher-forced states: t~U(0,1), each answer token masked iid w.p. t.
    P(specific mask pattern with m masked of L) = m!(L-m)!/(L+1)!."""
    from math import factorial
    D = np.zeros(NS)
    for t, w in templates:
        for pat in itertools.product([0, 1], repeat=L):
            mcount = sum(pat)
            if mcount == 0:
                continue
            s = tuple(MASK if pat[i] else t[i] for i in range(L))
            if only_fact_masked and any(v in FTOK for v in s):
                continue
            D[int(sum(np.array(s) * POW))] += w * factorial(mcount) * factorial(L - mcount) / factorial(L + 1)
    return D / D.sum()


def onpolicy_dist(Z, cfgs):
    D = np.zeros(NS)
    for c in cfgs:
        v = run_sampler(Z, c, want_visits=True)['visits']
        d = np.zeros(NS)
        for s, m in v.items():
            d[s] += m
        D += d / d.sum()
    return D / len(cfgs)


def kl_rows(P, Q):
    return (P * (np.log(P + 1e-300) - np.log(Q + 1e-300))).sum(-1)


def evaluate(Z, Z0, Zr0, P0r_unused, kappa, cfgs):
    lk = [run_sampler(Z, c)['leak'] for c in cfgs]
    Zr = Zr0 + kappa * (Z - Z0)
    rr = [run_sampler(Zr, c)['leak'] for c in cfgs]    # for the retain prompt "leak" = recall of F
    P, P0 = softmax(Z), softmax(Z0)
    kl = kl_rows(P0, P) * MASKED
    return dict(leak_mean=float(np.mean(lk)), leak_max=float(np.max(lk)),
                leak_by_cfg=[float(x) for x in lk],
                retain_recall_mean=float(np.mean(rr)),
                footprint_kl=float(kl.sum()), n_edited=int((kl > 1e-3).sum()))


def train(arm, Z0, R, Zr0, kappa, eta, rounds, spr, eval_every, tf_ff, tf_gt, tf_para, cover):
    Z = Z0.copy()
    Ru = np.full_like(R, 1.0 / NV)
    target = Ru if arm == 'TF_uniform' else R
    hist = []
    for r in range(rounds + 1):
        if r % eval_every == 0:
            ev = evaluate(Z, Z0, Zr0, None, kappa, EVAL_CFGS)
            ev['round'] = r
            hist.append(ev)
            print(f"  [{arm}] round {r:3d} leak_mean={ev['leak_mean']:.4f} leak_max={ev['leak_max']:.4f} "
                  f"retainF={ev['retain_recall_mean']:.4f} footKL={ev['footprint_kl']:.2f} "
                  f"nEdit={ev['n_edited']}", flush=True)
        if r == rounds:
            break
        if arm in ('TF', 'TF_uniform'):
            D = tf_gt
        elif arm == 'TF_para':
            D = tf_para
        elif arm == 'OP_single':
            D = onpolicy_dist(Z, SINGLE_CFG)
        elif arm == 'OP_wide':
            D = onpolicy_dist(Z, WIDE_CFG)
        elif arm == 'OP_mix':
            D = 0.8 * onpolicy_dist(Z, TRAIN_CFGS) + 0.2 * tf_ff
        elif arm == 'OP_mix_hz':
            # hazard-prioritised: weight on-policy states by (rho-hazard + floor), the integrand of the bound
            P = softmax(Z)
            Hs = ((P[..., FTOK].sum(-1) / P.max(-1)) * MASKED).sum(1)
            Dh = onpolicy_dist(Z, TRAIN_CFGS) * (Hs + 0.01)
            D = 0.8 * Dh / Dh.sum() + 0.2 * tf_ff
        elif arm == 'cover_all':
            D = cover
        W = D[:, None] * MASKED / np.maximum(NMASK, 1)[:, None]
        lam = np.minimum(eta * W, 1.0)[..., None]
        logT = np.log(target + 1e-300)
        for _ in range(spr):
            # v2 update: mirror-descent / natural-gradient caricature of a KL(r||p) step,
            # p <- p^(1-lam) r^lam per (state, masked position); lam = min(eta*D(s)/|M(s)|, 1).
            # (v1 used plain clipped GD on logits; its tiny late-stage gradients p_F - eps made all
            #  arms plateau for optimisation reasons unrelated to state coverage.)
            Zc = Z - np.log(np.exp(Z - Z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - Z.max(-1, keepdims=True)
            Z = (1 - lam) * Zc + lam * logT
            Z[~MASKED] = 0.0
    return Z, hist


def bound_table(Z, cfgs, support=None):
    rows = []
    for c in cfgs:
        o = run_sampler(Z, c, attrib_support=support)
        rows.append(dict(cfg=list(c), leak=o['leak'], B1=o['B1'], B2=o['B2'], B3=o['B3'],
                         sup_rho=o['sup_rho'], leak_from_outside_TF=o['leak_from_outside']))
    return rows


def main():
    t0 = time.time()
    out = {}
    eps, kappa, eta, rounds, spr, ev_every = 1e-3, 0.5, 100.0, 40, 10, 2
    variants = [('unigram', 0.02), ('uniform', 0.02)]
    for backoff, delta in variants:
        key = f"backoff={backoff},delta={delta}"
        print(f"=== {key} ===", flush=True)
        Z0 = posterior_logits(FORGET_T, delta, backoff, 0.02, 0)
        Zr0 = posterior_logits(RETAIN_T, delta, backoff, 0.02, 1)
        R = iproj_target(Z0, eps)
        tf_gt = tf_dist([(GT, 1.0)])
        tf_ff = tf_dist([(GT, 1.0)], only_fact_masked=True)
        tf_para = tf_dist([t for t in FORGET_T if 2 in t[0]])
        cover = ((~HAS_F) & (NMASK > 0)).astype(float); cover /= cover.sum()
        res = dict(eval_cfgs=[list(c) for c in EVAL_CFGS], train_cfgs=[list(c) for c in TRAIN_CFGS])
        base = bound_table(Z0, EVAL_CFGS)
        res['base_bounds'] = base
        print(f"  base leak per cfg: {[round(r['leak'], 3) for r in base]}", flush=True)
        arms = {}
        finalZ = {}
        for arm in ['TF', 'TF_uniform', 'TF_para', 'OP_single', 'OP_wide', 'OP_mix', 'OP_mix_hz', 'cover_all']:
            Z, hist = train(arm, Z0, R, Zr0, kappa, eta, rounds, spr, ev_every, tf_ff, tf_gt, tf_para, cover)
            arms[arm] = hist
            finalZ[arm] = Z
        res['arms'] = arms
        summ = {}
        for a, hist in arms.items():
            hit = [h for h in hist if h['leak_max'] <= 0.05]
            hit2 = [h for h in hist if h['leak_mean'] <= 0.01]
            f = hist[-1]
            summ[a] = dict(final_leak_mean=f['leak_mean'], final_leak_max=f['leak_max'],
                           final_footprint_kl=f['footprint_kl'], final_n_edited=f['n_edited'],
                           final_retainF=f['retain_recall_mean'],
                           matched=(None if not hit else dict(round=hit[0]['round'], leak_mean=hit[0]['leak_mean'],
                                    leak_max=hit[0]['leak_max'], footprint_kl=hit[0]['footprint_kl'],
                                    n_edited=hit[0]['n_edited'], retainF=hit[0]['retain_recall_mean'])),
                           matched_mean=(None if not hit2 else dict(round=hit2[0]['round'], leak_mean=hit2[0]['leak_mean'],
                                    leak_max=hit2[0]['leak_max'], footprint_kl=hit2[0]['footprint_kl'],
                                    n_edited=hit2[0]['n_edited'], retainF=hit2[0]['retain_recall_mean'])))
            print(f"  SUMMARY {a}: {summ[a]}", flush=True)
        res['summary_matched(leak_max<=0.05)_matched_mean(leak_mean<=0.01)'] = summ
        supp_tf = tf_gt > 0
        res['bounds_after'] = {a: bound_table(finalZ[a], EVAL_CFGS, support=supp_tf)
                               for a in ['TF', 'TF_para', 'OP_single', 'OP_wide', 'OP_mix', 'OP_mix_hz']}
        # coverage: fact-free visitation mass of the TF-trained model's trajectories outside supp(TF)
        cov = []
        for c in EVAL_CFGS:
            v = run_sampler(finalZ['TF'], c, want_visits=True)['visits']
            tot = sum(v.values()); outm = sum(m for s, m in v.items() if not supp_tf[s])
            cov.append(outm / tot)
        res['TF_model_ff_visit_mass_outside_suppTF'] = cov
        out[key] = res
    out['params'] = dict(eps=eps, kappa=kappa, eta=eta, rounds=rounds, steps_per_round=spr,
                         forget_templates=[[''.join(TOK[v] for v in t), w] for t, w in FORGET_T],
                         retain_templates=[[''.join(TOK[v] for v in t), w] for t, w in RETAIN_T],
                         runtime_s=round(time.time() - t0, 1))
    path = sys.argv[1] if len(sys.argv) > 1 else 'tabular_commit_hazard_out.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path} in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
