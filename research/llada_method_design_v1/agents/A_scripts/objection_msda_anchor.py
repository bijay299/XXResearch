#!/usr/bin/env python3
"""Agent A -> C objection check: MSDA (frozen generalised-eigen row space A, train B only) with
(a) C's specified loss = MDU tau=1 anchor (KL(p||p_theta0(.|m,y_t)), all masked answer positions) and
(b) the same parameterisation with a leak-free exclusion target (FFX/TNFF), on the log-linear
shared-parameter mask predictor of sharing_check.py, whose logits are EXACTLY linear in a key vector:
    logits(c,s,j) = W0 phi(c,s,j) + B A phi(c,s,j),  phi = one-hot features (prompt,j),(subj,j),(rel,j),
    (j,k,s_k),(j)   -> m = 188 "keys", W0 = base fit.
The toy's answers restate the subject (X visible) so the prompt-masked anchor still predicts the fact.
A: top generalised eigenvectors of C_F u = mu C_R u (C_F over forget TF states + theta0 native-rollout
states under all 6 configs; C_R over retain TF states + theta0 rollouts; shrinkage rho=1e-2), r=#{mu>=4}.
Reports max-over-6-configs generated leakage vs retain distortion, sweeping lambda.
"""
import itertools, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sharing_check as sc
import tabular_check as T

torch.set_default_dtype(torch.float64)
V, L, MASK = T.V, T.L, T.MASK
PR = sc.PR
M_DIM = len(PR) * L + 3 * L + 2 * L + L * L * (V + 1) + L


def phi(prompts, states):
    B = states.shape[0]
    out = torch.zeros(B, L, M_DIM)
    for n, c in enumerate(prompts):
        for j in range(L):
            out[n, j, PR.index(c) * L + j] = 1
            if sc.SUBJ[c] is not None:
                out[n, j, len(PR) * L + sc.SUBJ[c] * L + j] = 1
                out[n, j, len(PR) * L + 3 * L + sc.REL[c] * L + j] = 1
            base = len(PR) * L + 5 * L
            for k in range(L):
                if k != j:
                    out[n, j, base + (j * L + k) * (V + 1) + int(states[n, k])] = 1
            out[n, j, base + L * L * (V + 1) + j] = 1
    return out


ORIG = sc.logits_batch
STATE = {}


def wrapped(p, prompts, states):
    if 'Bm' in p:
        lg = ORIG(STATE['theta0'], prompts, states)
        return lg + phi(prompts, states) @ STATE['A'].T @ p['Bm'].T
    return ORIG(p, prompts, states)


sc.logits_batch = wrapped


def visitation(fam, cfgs):
    """all states visited by native samplers (exact), summed over configs (incl. fact-visible)."""
    vis = {}
    for cn, (steps, block, remask, Tt) in cfgs.items():
        nb = L // block; spb = steps // nb
        sched = [block // spb + (1 if k < block % spb else 0) for k in range(spb)]
        dist = {(MASK,) * L: 1.0}
        for b in range(nb):
            for i in range(spb):
                k = sched[i]; new = {}
                for s, ps in dist.items():
                    vis[s] = vis.get(s, 0.0) + ps
                    P = fam(s)
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
    return {s: w for s, w in vis.items() if w > 1e-4 and MASK in s}


def second_moment(items):
    C = torch.zeros(M_DIM, M_DIM); n = 0.0
    for c, s, w in items:
        f = phi([c], torch.tensor([s]))[0]
        for j in range(L):
            if s[j] == MASK:
                C += w * torch.outer(f[j], f[j]); n += w
    return C / n


def main():
    out = {}
    theta0 = sc.lbfgs(lambda p: sc.ce_soft(p, sc.base_data()) + 1e-4 * (sc.flat(p) ** 2).sum(), sc.init_params(), iters=500)
    STATE['theta0'] = theta0
    STATE['A'] = torch.zeros(1, M_DIM)
    fam0 = {c: sc.model_family(theta0, c) for c in PR}
    base_dec = {c: {cn: T.decode(fam0[c], cfg) for cn, cfg in sc.CFGS.items()} for c in ['R1', 'R2', 'R3']}
    # anchor's fact mass on forget fact-free TF states (theta0, prompt masked)
    out['anchor_p_fact_ffTF'] = {str(s): float(fam0['0'](s)[T.FPOS][T.FACT]) for s in T.TF_FF}
    out['cond_p_fact_ffTF'] = {str(s): float(fam0['F'](s)[T.FPOS][T.FACT]) for s in T.TF_FF}
    # state distributions
    visF = visitation(fam0['F'], sc.CFGS)
    tfF = [s for s in T.TF_STATES if MASK in s]
    QF = [('F', s, 1.0) for s in tfF] + [('F', s, w) for s, w in visF.items()]
    QR = []
    for c in ['R1', 'R2', 'R3']:
        for y, py in T.SPEC[c].items():
            QR += [(c, s, py) for s in sc.tf_states(y) if MASK in s]
        QR += [(c, s, w) for s, w in visitation(fam0[c], {'C1': sc.CFGS['C1_greedy_lowconf_s4_b4']}).items()]
    CF = second_moment(QF); CR = second_moment(QR)
    CR = CR + 1e-2 * torch.trace(CR) / M_DIM * torch.eye(M_DIM)
    Lc = torch.linalg.cholesky(CR)
    Li = torch.linalg.inv(Lc)
    Mm = Li @ CF @ Li.T
    mu, U = torch.linalg.eigh((Mm + Mm.T) / 2)
    mu, U = mu.flip(0), U.flip(1)
    r = int((mu >= 4).sum().clamp(1, 32))
    out['mu_top12'] = [round(float(x), 2) for x in mu[:12]]
    out['rank_rule_r'] = r
    A = (Li.T @ U[:, :r]).T
    A = A / A.norm(dim=1, keepdim=True)
    STATE['A'] = A
    # which feature groups do the rows of A load on?
    groups = {'prompt(c,j)': (0, len(PR) * L), 'subj': (len(PR) * L, len(PR) * L + 3 * L),
              'rel': (len(PR) * L + 3 * L, len(PR) * L + 5 * L),
              'pairwise(j,k,s_k)': (len(PR) * L + 5 * L, len(PR) * L + 5 * L + L * L * (V + 1)), 'bias': (M_DIM - L, M_DIM)}
    out['A_row_energy_by_group'] = {g: round(float((A[:, a:b] ** 2).sum() / (A ** 2).sum()), 3) for g, (a, b) in groups.items()}
    print('mu', out['mu_top12'], 'r', r, out['A_row_energy_by_group'], flush=True)
    # retain data and targets
    ret = [(c, s, fam0[c](s), w) for c, s, w in QR]
    anchor_items = [('F', s, fam0['0'](s), w) for _, s, w in QF]

    def excl_rows(P, s):
        Q = P.copy()
        for j in range(L):
            if s[j] == MASK:
                Q[j] = T.excl(Q[j])
        return Q
    ffx_items = [('F', s, excl_rows(fam0['F'](s), s), w) for _, s, w in QF if T.FACT not in s]
    res = {}
    for meth in ['MSDA+MDU_tau1(C as specified)', 'MSDA+FFX_target', 'FULL+MDU_tau1', 'FULL+FFX_target']:
        rows = []
        for lam in [0.3, 1, 3, 10, 30, 100]:
            items, d = (anchor_items, 'reverse') if 'MDU' in meth else (ffx_items, 'forward')
            if meth.startswith('MSDA'):
                p0 = {'Bm': torch.zeros(V, r)}
                obj = lambda p: lam * sc.ce_soft(p, items, direction=d) + sc.ce_soft(p, ret) + sc.MU * (p['Bm'] ** 2).sum()
                th = sc.lbfgs(obj, p0)
            else:
                obj = lambda p: lam * sc.ce_soft(p, items, direction=d) + sc.ce_soft(p, ret) + sc.MU * ((sc.flat(p) - sc.flat(theta0)) ** 2).sum()
                th = sc.lbfgs(obj, theta0)
            ev = sc.evaluate(th, base_dec); ev['lam'] = lam
            famF = sc.model_family(th, 'F')
            ev['fact_mass_at_fact_pos_ffTF_mean'] = float(np.mean([famF(s)[T.FPOS][T.FACT] for s in T.TF_FF]))
            rows.append(ev)
            print(meth, lam, 'maxleak %.3f' % ev['max_leak'], 'leaks', [round(float(x), 2) for x in ev['leak'].values()],
                  'retainTV %.3f' % ev['retain_decodeTV_mean'], 'Xkept %.2f junk %.2f' % (ev['F_P(X_kept)'], ev['F_junk']),
                  'p_fact(ffTF) %.3f' % ev['fact_mass_at_fact_pos_ffTF_mean'], flush=True)
        best = min(rows, key=lambda e: e['max_leak'])
        res[meth] = {'sweep': rows, 'min_max_leak': best['max_leak'], 'at_lam': best['lam'],
                     'first_lam_maxleak<=0.10': next((e['lam'] for e in rows if e['max_leak'] <= 0.10), None)}
    out['methods'] = res
    out['summary'] = {m: {k: v for k, v in r_.items() if k != 'sweep'} for m, r_ in res.items()}
    print(json.dumps(out['summary'], indent=1, default=float))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objection_msda_anchor_out.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    main()
