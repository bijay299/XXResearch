#!/usr/bin/env python3
"""Agent A -- retention at matched forgetting with PARAMETER SHARING (log-linear mask predictor).

Not a neural net: a log-linear (multinomial-logistic) mask predictor
  logits(c,s,j) = U[c,j] + S[subj(c),j] + R[rel(c),j] + sum_{k!=j} W[j,k,s_k]      (s_k in V u {MASK})
shared across prompts F=(X,when) R1=(Y,when) R2=(X,where) R3=(Z,when) and a prompt-masked input 0=(none).
It is in general INCONSISTENT (not derived from one joint), like a real network.
1. Base fit theta0: distil exact conditionals of the tabular joints (tabular_check.py) on all
   teacher-forced states of every listed answer (prompt-masked input distilled to the prompt mixture).
2. Unlearn: min_theta  lam * L_forget + L_retain + mu*||theta-theta0||^2, L-BFGS to convergence,
   L_retain = KL(p_theta0||p_theta) on retain prompts' teacher-forced states.  Methods:
   GA (log p(y*_j) on all TF states), ME (CE to uniform, all TF), MDU tau=1 (MDU code direction
   KL(p_theta||anchor), anchor = p_theta0 with prompt masked, all TF), FFX-TF (CE to the
   token-exclusion target on fact-free TF states), FFX-OP (same target on fact-free states visited by
   the student's own native samplers, 3 DAgger rounds, aggregated) .
3. Sweep lam; report retain distortion at the smallest lam with max-over-configs leakage <= 0.05.
Leakage/decoding = exact LLaDA sampler enumeration from tabular_check.decode.
"""
import itertools, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tabular_check as T

torch.set_default_dtype(torch.float64)
V, L, MASK = T.V, T.L, T.MASK
PR = ['F', 'R1', 'R2', 'R3', '0']
SUBJ = {'F': 0, 'R1': 1, 'R2': 0, 'R3': 2, '0': None}
REL = {'F': 0, 'R1': 0, 'R2': 1, 'R3': 0, '0': None}
CFGS = T.CONFIGS
MU = 0.05
LEAK_TOL = 0.05


def init_params():
    g = torch.Generator().manual_seed(0)
    return {'U': torch.zeros(len(PR), L, V), 'S': torch.zeros(3, L, V), 'R': torch.zeros(2, L, V),
            'W': torch.zeros(L, L, V + 1, V), 'b': torch.zeros(L, V)}


def flat(p): return torch.cat([p[k].reshape(-1) for k in sorted(p)])


def unflat(v, like):
    out, i = {}, 0
    for k in sorted(like):
        n = like[k].numel(); out[k] = v[i:i + n].reshape(like[k].shape); i += n
    return out


def logits_batch(p, prompts, states):
    """prompts: list of prompt names; states: LongTensor (B,L) with MASK=8. returns (B,L,V)."""
    B = states.shape[0]
    ci = torch.tensor([PR.index(c) for c in prompts])
    out = p['b'].unsqueeze(0).expand(B, L, V) + p['U'][ci]
    for n, c in enumerate(prompts):
        if SUBJ[c] is not None:
            out[n] = out[n] + p['S'][SUBJ[c]] + p['R'][REL[c]]
    # pairwise: sum_k!=j W[j,k,s_k]
    Wg = p['W'][:, torch.arange(L).unsqueeze(0).expand(B, L), states]   # (L, B, L(k), V) -> W[j,k,s_k]
    Wg = Wg.permute(1, 0, 2, 3)                                           # (B, j, k, V)
    eye = torch.eye(L, dtype=torch.bool)
    Wg = Wg.masked_fill(eye.unsqueeze(0).unsqueeze(-1), 0.0)
    return out + Wg.sum(2)


def model_family(pnp, c):
    """numpy-cached conditional family of the log-linear model for prompt c."""
    def fn(s):
        with torch.no_grad():
            lg = logits_batch(pnp, [c], torch.tensor([s]))[0]
        P = torch.softmax(lg, -1).numpy().copy()
        for j in range(L):
            if s[j] != MASK:
                P[j] = np.nan
        return P
    return T.Fam(fn, c)


def tf_states(y):
    return [tuple(y[j] if keep else MASK for j, keep in enumerate(m)) for m in itertools.product([0, 1], repeat=L)]


# -------- base-fit data: (prompt, state, soft target rows) with weights
def base_data():
    data = []
    for c in PR:
        J = T.J_anchor if c == '0' else T.J0[c]
        spec = {}
        if c == '0':
            for cc in T.PROMPTS:
                for y, pp in T.SPEC[cc].items():
                    spec[y] = spec.get(y, 0) + pp / len(T.PROMPTS)
        else:
            spec = T.SPEC[c]
        for y, py in spec.items():
            for s in tf_states(y):
                if MASK not in s:
                    continue
                P, _ = T.cond_joint(J, s)
                data.append((c, s, P, py))
    return data


def ce_soft(p, items, direction='forward'):
    """items: list of (prompt, state, target (L,V) with nan rows for visible, weight). mean over masked."""
    prompts = [it[0] for it in items]
    st = torch.tensor([it[1] for it in items])
    tgt = torch.tensor(np.nan_to_num(np.stack([it[2] for it in items]), nan=0.0))
    w = torch.tensor([it[3] for it in items])
    m = (st == MASK).double()
    lp = torch.log_softmax(logits_batch(p, prompts, st), -1)
    if direction == 'forward':      # KL(target || p) up to const = CE(target, p)
        per = -(tgt * lp).sum(-1)
    else:                           # KL(p || target)   (MDU code default direction)
        pp = lp.exp()
        per = (pp * (lp - torch.log(tgt.clamp_min(1e-30)))).sum(-1)
    per = (per * m).sum(1) / m.sum(1).clamp_min(1)
    return (w * per).sum() / w.sum()


def ga_loss(p, items):
    prompts = [it[0] for it in items]
    st = torch.tensor([it[1] for it in items])
    m = (st == MASK).double()
    lp = torch.log_softmax(logits_batch(p, prompts, st), -1)
    y = torch.tensor([T.YSTAR] * len(items))
    ll = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return ((ll * m).sum(1) / m.sum(1).clamp_min(1)).mean()   # minimise log-lik = ascend NLL


def lbfgs(obj, p0, iters=200):
    v = flat(p0).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([v], lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); f = obj(unflat(v, p0)); f.backward(); return f
    opt.step(closure)
    return {k: t.detach() for k, t in unflat(v, p0).items()}


def evaluate(p, base_dec):
    r = {}
    famF = model_family(p, 'F')
    r['leak'] = {cn: T.leak(T.decode(famF, cfg)) for cn, cfg in CFGS.items()}
    r['max_leak'] = max(r['leak'].values())
    fF = {cn: T.decode(famF, cfg) for cn, cfg in CFGS.items()}
    r['F_P(X_kept)'] = float(np.mean([sum(pp for y, pp in d.items() if y[0] == T.X) for d in fF.values()]))
    r['F_junk'] = float(np.mean([T.junk(d) for d in fF.values()]))
    tvs, accs = {}, {}
    key = {'R1': T.F60, 'R2': T.PAR, 'R3': T.F51}
    for c in ['R1', 'R2', 'R3']:
        fam = model_family(p, c)
        decs = {cn: T.decode(fam, cfg) for cn, cfg in CFGS.items()}
        tvs[c] = float(np.mean([T.tv(decs[cn], base_dec[c][cn]) for cn in CFGS]))
        accs[c] = float(np.mean([sum(pp for y, pp in d.items() if key[c] in y) for d in decs.values()]))
    r['retain_decodeTV_mean'] = float(np.mean(list(tvs.values())))
    r['retain_decodeTV'] = tvs
    r['retain_fact_recall'] = accs
    return r


def onpolicy_ff_states(p, ref_famF):
    """fact-free states visited by the student's native samplers (exact visitation, summed over configs)."""
    famF = model_family(p, 'F')
    vis = {}
    for cn, (steps, block, remask, Tt) in CFGS.items():
        # re-run decode but record intermediate states: simple wrapper around T.decode internals
        nb = L // block; spb = steps // nb
        sched = [block // spb + (1 if k < block % spb else 0) for k in range(spb)]
        dist = {(MASK,) * L: 1.0}
        for b in range(nb):
            for i in range(spb):
                k = sched[i]; new = {}
                for s, ps in dist.items():
                    if T.FACT not in s:
                        vis[s] = vis.get(s, 0.0) + ps
                    fam_one = T.Fam(lambda _s, s=s: famF(s), 'x')
                    # one step: reuse decode on a single-step config by temporarily building transitions
                    P = famF(s)
                    elig = [j for j in range(b * block, (b + 1) * block) if s[j] == MASK]
                    samp = {}
                    for j in elig:
                        if Tt == 0:
                            d = np.zeros(V); d[int(np.argmax(P[j]))] = 1.0
                        else:
                            w = np.power(P[j], 1.0 / Tt); d = w / w.sum()
                        samp[j] = d
                    if remask == 'random':
                        subs = list(itertools.combinations(elig, k))
                        for sub in subs:
                            for vals in itertools.product(*[np.nonzero(samp[j] > 1e-6)[0] for j in sub]):
                                pr = np.prod([samp[j][v] for j, v in zip(sub, vals)]) / len(subs)
                                ns = list(s)
                                for j, v in zip(sub, vals): ns[j] = v
                                new[tuple(ns)] = new.get(tuple(ns), 0.0) + ps * pr
                    else:
                        for vals in itertools.product(*[np.nonzero(samp[j] > 1e-6)[0] for j in elig]):
                            pr = np.prod([samp[j][v] for j, v in zip(elig, vals)])
                            conf = [P[j][v] for j, v in zip(elig, vals)]
                            order = sorted(range(len(elig)), key=lambda a: (-conf[a], elig[a]))[:k]
                            ns = list(s)
                            for a in order: ns[elig[a]] = vals[a]
                            new[tuple(ns)] = new.get(tuple(ns), 0.0) + ps * pr
                dist = new
    return {s: w for s, w in vis.items() if w > 1e-4}


def main():
    out = {'mu': MU, 'leak_tol': LEAK_TOL}
    p_init = init_params()
    bd = base_data()
    theta0 = lbfgs(lambda p: ce_soft(p, bd) + 1e-4 * (flat(p) ** 2).sum(), p_init, iters=500)
    fam0 = {c: model_family(theta0, c) for c in PR}
    base_dec = {c: {cn: T.decode(fam0[c], cfg) for cn, cfg in CFGS.items()} for c in ['R1', 'R2', 'R3']}
    out['base'] = evaluate(theta0, base_dec)
    out['base_consistency_violation_F_TF'] = T.consistency_violation(fam0['F'], T.TF_STATES)
    print('base:', json.dumps(out['base'], default=float))

    # retain data: theta0's own conditionals on retain TF states (KL(p0||p))
    ret = []
    for c in ['R1', 'R2', 'R3']:
        for y, py in T.SPEC[c].items():
            for s in tf_states(y):
                if MASK in s:
                    ret.append((c, s, fam0[c](s), py))
    # forget targets (all built from theta0)
    def excl_rows(P, s):
        Q = P.copy()
        for j in range(L):
            if s[j] == MASK:
                Q[j] = T.excl(Q[j])
        return Q
    tfF = [s for s in T.TF_STATES if MASK in s]
    ffF = [s for s in T.TF_FF]
    unif = np.full((L, V), 1.0 / V)
    items = {
        'ME': [('F', s, np.where(np.array(s)[:, None] == MASK, unif, np.nan), 1.0) for s in tfF],
        'MDU_tau1': [('F', s, fam0['0'](s), 1.0) for s in tfF],
        'FFX_TF': [('F', s, excl_rows(fam0['F'](s), s), 1.0) for s in ffF],
        'MDU_tau1_ffTF': [('F', s, fam0['0'](s), 1.0) for s in ffF],
    }
    lams = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    TOLS = [0.30, 0.10, 0.05]
    tgt_fn = {'FFX': lambda s: excl_rows(fam0['F'](s), s), 'MDU1': lambda s: fam0['0'](s)}
    res = {}
    for meth in ['GA', 'ME', 'MDU_tau1', 'MDU_tau1_ffTF', 'MDU_tau1_OP', 'FFX_TF', 'FFX_OP']:
        rows = []
        for lam in lams:
            prox = lambda p: MU * ((flat(p) - flat(theta0)) ** 2).sum()
            if meth == 'GA':
                obj = lambda p: lam * ga_loss(p, [('F', s, None, 1.0) for s in tfF]) + ce_soft(p, ret) + prox(p)
                th = lbfgs(obj, theta0)
            elif meth.endswith('_OP'):
                key = 'FFX' if meth.startswith('FFX') else 'MDU1'
                d = 'forward' if key == 'FFX' else 'reverse'
                agg = {s: 1.0 for s in ffF}
                th = theta0
                for rnd in range(3):
                    for s, w in onpolicy_ff_states(th, fam0['F']).items():
                        agg[s] = agg.get(s, 0.0) + w
                    it = [('F', s, tgt_fn[key](s), w) for s, w in agg.items() if MASK in s]
                    obj = lambda p: lam * ce_soft(p, it, direction=d) + ce_soft(p, ret) + prox(p)
                    th = lbfgs(obj, theta0)
            else:
                d = 'reverse' if meth.startswith('MDU') else 'forward'
                it = items[meth] if meth in items else [('F', s, fam0['0'](s), 1.0) for s in ffF]
                obj = lambda p: lam * ce_soft(p, it, direction=d) + ce_soft(p, ret) + prox(p)
                th = lbfgs(obj, theta0)
            ev = evaluate(th, base_dec); ev['lam'] = lam
            rows.append(ev)
            print(meth, lam, 'maxleak %.3f' % ev['max_leak'], 'retainTV %.3f' % ev['retain_decodeTV_mean'],
                  'Xkept %.2f junk %.2f' % (ev['F_P(X_kept)'], ev['F_junk']),
                  {k: round(v, 2) for k, v in ev['retain_fact_recall'].items()},
                  'leaks', [round(x, 2) for x in ev['leak'].values()], flush=True)
        res[meth] = {'sweep': rows, 'matched': {}}
        for tol in TOLS:
            ok = [r for r in rows if r['max_leak'] <= tol]
            res[meth]['matched'][str(tol)] = ok[0] if ok else None
    out['methods'] = res
    summ = {}
    for m, r in res.items():
        summ[m] = {}
        for tol, e in r['matched'].items():
            summ[m][tol] = ('INFEASIBLE' if e is None else
                            'lam=%g maxleak=%.3f retainTV=%.3f recall(R1,R2,R3)=%s Xkept=%.2f junk=%.2f' % (
                                e['lam'], e['max_leak'], e['retain_decodeTV_mean'],
                                [round(v, 3) for v in e['retain_fact_recall'].values()], e['F_P(X_kept)'], e['F_junk']))
    out['summary_at_matched_forgetting'] = summ
    print(json.dumps(out['summary_at_matched_forgetting'], indent=1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sharing_check_out.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    main()
