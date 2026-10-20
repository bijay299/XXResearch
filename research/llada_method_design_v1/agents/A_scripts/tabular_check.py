#!/usr/bin/env python3
"""Agent A -- exact tabular 'consistent MDM' check of unlearning TARGETS under native LLaDA decoding.

Everything is exact (no Monte Carlo): joints over V^L answers (V=8, L=4), conditionals at any
visible subset by marginalisation, and the LLaDA `generate` sampler (generate.py L95-200, pinned
9182493) enumerated exactly: per step x0 ~ softmax(logits/T) independently at every masked position
(argmax, first index, if T=0), confidence = untempered prob of x0 (low_confidence) or uniform
(random), top-k inside the current block committed, never re-masked, uniform k schedule.
Tie-break in top-k: higher confidence, then lower index (torch.topk tie order is unspecified).

A "target model" = model whose conditionals equal the method's target on the states the method
trains on (teacher-forced states of the forget answer y*) and equal the reference elsewhere
(i.e. NO generalisation; the pessimistic/zero-sharing case). The QALL oracle uses the exact
I-projection conditionals on every state where they are defined.

Usage: python tabular_check.py  -> writes tabular_check_out.json and prints a report.
"""
import itertools, json, os, sys, hashlib
import numpy as np

V, L, MASK = 8, 4, 8
TOK = ['EOS', 'X', 'Y', 'born', '1951', '1952', '1960', 'Paris', '_']
EOS, X, Y, BORN, F51, F52, F60, PAR = range(8)
FACT = F51                    # fact token; event A = "1951 appears anywhere in the answer"
YSTAR = (X, BORN, F51, EOS)   # forget answer (teacher-forced ground truth)
FPOS = 2                      # fact position in y*

# ---- reference joint p0(y|c): listed answers + EPS spread uniformly over all V^L answers ----
SPEC = {
    'F':  {(X, BORN, F51, EOS): .70, (BORN, F51, EOS, EOS): .10, (X, BORN, F51, PAR): .06,
           (X, BORN, F52, PAR): .04, (X, BORN, F60, EOS): .03, (BORN, F52, EOS, EOS): .02,
           (X, BORN, EOS, EOS): .03},                         # "When was X born?"  (forget)
    'R1': {(Y, BORN, F60, EOS): .80, (BORN, F60, EOS, EOS): .12, (Y, BORN, F52, EOS): .03,
           (Y, BORN, F51, EOS): .03},                         # "When was Y born?"  (shares template)
    'R2': {(X, BORN, PAR, EOS): .80, (BORN, PAR, EOS, EOS): .10, (X, BORN, F51, PAR): .05,
           (X, BORN, EOS, EOS): .03},                         # "Where was X born?" (same subject)
    'R3': {(BORN, F51, EOS, EOS): .85, (BORN, F52, EOS, EOS): .08, (BORN, F60, EOS, EOS): .05},
                                                              # "When was Z born?"  (same answer token)
}
EPS = 0.02
PROMPTS = list(SPEC)


def make_joint(spec):
    J = np.full((V,) * L, EPS / V ** L)
    for y, p in spec.items():
        J[y] += p
    return J / J.sum()


J0 = {c: make_joint(SPEC[c]) for c in PROMPTS}
J_anchor = sum(J0[c] for c in PROMPTS) / len(PROMPTS)       # prompt-masked p0(.|m, y_V): Bayes mixture
NOFACT = np.ones((V,) * L)
for j in range(L):
    idx = [slice(None)] * L; idx[j] = FACT; NOFACT[tuple(idx)] = 0.0
JQ = {c: J0[c] * NOFACT / (J0[c] * NOFACT).sum() for c in PROMPTS}  # exact I-projection q = p0(.|A^c)


def cond_joint(J, s):
    """Exact conditionals of joint J at state s (tuple, MASK=8). rows of visible positions = nan."""
    idx = tuple(slice(None) if v == MASK else v for v in s)
    sub = J[idx]
    masked = [j for j in range(L) if s[j] == MASK]
    out = np.full((L, V), np.nan)
    tot = sub.sum()
    if tot <= 0:
        return out, 0.0
    for a, j in enumerate(masked):
        axes = tuple(b for b in range(len(masked)) if b != a)
        out[j] = (sub.sum(axis=axes) if axes else sub) / tot
    return out, tot


def fill(s, j, v):
    t = list(s); t[j] = v; return tuple(t)


ALL_STATES = list(itertools.product(range(V + 1), repeat=L))
TF_STATES = [tuple(YSTAR[j] if keep else MASK for j, keep in enumerate(m))
             for m in itertools.product([0, 1], repeat=L)]
TF_FF = [s for s in TF_STATES if s[FPOS] == MASK]          # fact-free teacher-forced states (8)
TF_FV = [s for s in TF_STATES if s[FPOS] != MASK]          # fact-visible teacher-forced states (8)


# ---------------------------------------------------------------- families of conditionals
class Fam:
    """A conditional family: callable s -> (L,V) array (rows for masked positions)."""
    def __init__(self, fn, name):
        self.fn, self.name, self.cache = fn, name, {}

    def __call__(self, s):
        if s not in self.cache:
            self.cache[s] = self.fn(s)
        return self.cache[s]


def ref_family(J):
    return Fam(lambda s: cond_joint(J, s)[0], 'ref')


def excl(p, toks=(FACT,)):
    p = p.copy(); p[list(toks)] = 0.0
    z = p.sum()
    return p / z if z > 0 else np.full(V, 1.0 / V)


def perturbed(base, sigma, seed):
    """Inconsistent family: p~(.|s,j) ∝ p(.|s,j) exp(sigma*eps), eps iid N(0,1) per (s,j,v)."""
    def fn(s):
        P = base(s).copy()
        for j in range(L):
            if s[j] == MASK:
                h = int(hashlib.md5(f'{seed}|{s}|{j}'.encode()).hexdigest()[:8], 16)
                e = np.random.default_rng(h).standard_normal(V)
                w = P[j] * np.exp(sigma * e); P[j] = w / w.sum()
        return P
    return Fam(fn, f'pert{sigma}')


def target_model(ref, target_fn, states, name):
    st = set(states)
    def fn(s):
        return target_fn(s) if s in st else ref(s)
    return Fam(fn, name)


# target constructors (all take the reference family `ref` of the forget prompt)
def t_ga(ref, beta):
    def fn(s):
        P = ref(s).copy()
        for j in range(L):
            if s[j] == MASK:
                w = P[j].copy(); w[YSTAR[j]] *= np.exp(-beta); P[j] = w / w.sum()
        return P
    return fn


def t_mdu(anchor, tau):
    def fn(s):
        A = anchor(s).copy()
        for j in range(L):
            if s[j] == MASK:
                w = np.power(np.clip(A[j], 1e-300, None), tau); A[j] = w / w.sum()
        return A
    return fn


def t_ffx_mf(ref):
    def fn(s):
        P = ref(s).copy()
        for j in range(L):
            if s[j] == MASK:
                P[j] = excl(P[j])
        return P
    return fn


def t_ffx_cf(ref, i=FPOS):
    """counterfactual fill: q(x_j|s) = sum_{v!=f} r(v) p(x_j|s, x_i=v), r = renormalised p(x_i|s)."""
    def fn(s):
        P = ref(s).copy()
        r = excl(P[i])
        for j in range(L):
            if s[j] == MASK and j != i:
                m = sum(r[v] * ref(fill(s, i, v))[j] for v in range(V) if r[v] > 0)
                P[j] = excl(m)
        P[i] = r
        return P
    return fn


NEG_LOG = []


def t_ffx_bayes(ref, i=FPOS):
    """Bayes exclusion: q(x_j|s) = [p(x_j|s) - pi p(x_j|s,x_i=f)]/(1-pi); clip<0 then renorm."""
    def fn(s):
        P = ref(s).copy()
        pi = P[i][FACT]
        Pf = ref(fill(s, i, FACT))
        for j in range(L):
            if s[j] == MASK and j != i:
                raw = (P[j] - pi * Pf[j]) / (1 - pi)
                NEG_LOG.append(float(-raw[raw < 0].sum()))
                P[j] = excl(np.clip(raw, 0, None))
        P[i] = excl(P[i])
        return P
    return fn


def t_exact_q(c):
    return lambda s: cond_joint(JQ[c], s)[0]


# ---------------------------------------------------------------- exact native sampler
CONFIGS = {  # name: (steps, block_length, remasking, temperature)   gen_length = L = 4
    'C1_greedy_lowconf_s4_b4': (4, 4, 'low_confidence', 0.0),
    'C2_T1_random_s4_b4':      (4, 4, 'random', 1.0),
    'C3_greedy_lowconf_s2_b2': (2, 2, 'low_confidence', 0.0),
    'C4_T1_lowconf_s2_b4':     (2, 4, 'low_confidence', 1.0),
    'C5_T0.5_random_s4_b2':    (4, 2, 'random', 0.5),
    'C6_greedy_random_s2_b4':  (2, 4, 'random', 0.0),
}


def decode(fam, cfg, track=False):
    """Exact distribution over final answers. track: also returns fact-commit diagnostics."""
    steps, block, remask, T = cfg
    nb = L // block; spb = steps // nb
    sched = [block // spb + (1 if k < block % spb else 0) for k in range(spb)]
    dist = {((MASK,) * L, None): 1.0}
    for b in range(nb):
        for i in range(spb):
            k = sched[i]; new = {}
            for (s, tag), ps in dist.items():
                P = fam(s)
                elig = [j for j in range(b * block, (b + 1) * block) if s[j] == MASK]
                samp = {}
                for j in elig:
                    if T == 0:
                        d = np.zeros(V); d[int(np.argmax(P[j]))] = 1.0
                    else:
                        w = np.power(np.clip(P[j], 0, None), 1.0 / T); d = w / w.sum()
                    samp[j] = d
                outs = []
                if remask == 'random':
                    subs = list(itertools.combinations(elig, k))
                    for sub in subs:
                        for vals in itertools.product(*[np.nonzero(samp[j])[0] for j in sub]):
                            pr = np.prod([samp[j][v] for j, v in zip(sub, vals)]) / len(subs)
                            outs.append((sub, vals, pr))
                else:
                    for vals in itertools.product(*[np.nonzero(samp[j])[0] for j in elig]):
                        pr = np.prod([samp[j][v] for j, v in zip(elig, vals)])
                        conf = [P[j][v] for j, v in zip(elig, vals)]
                        order = sorted(range(len(elig)), key=lambda a: (-conf[a], elig[a]))[:k]
                        outs.append((tuple(elig[a] for a in order), tuple(vals[a] for a in order), pr))
                for sub, vals, pr in outs:
                    ns = list(s)
                    for j, v in zip(sub, vals):
                        ns[j] = v
                    ns = tuple(ns); ntag = tag
                    if track and tag is None and FACT in vals:
                        nvis = sum(1 for x in s if x != MASK)
                        ntag = ('TF' if s in TF_FF_SET else 'nonTF', nvis)
                    key = (ns, ntag)
                    new[key] = new.get(key, 0.0) + ps * pr
            dist = new
    final = {}
    diag = {}
    for (s, tag), p in dist.items():
        final[s] = final.get(s, 0.0) + p
        if tag is not None:
            diag[str(tag)] = diag.get(str(tag), 0.0) + p
    return (final, diag) if track else final


TF_FF_SET = set(TF_FF)


def leak(final):
    return sum(p for y, p in final.items() if FACT in y)


GRAMMAR = set()
for v in (F51, F52, F60, PAR, EOS):
    GRAMMAR |= {(X, BORN, v, EOS), (X, BORN, v, PAR), (BORN, v, EOS, EOS)}


def junk(final):
    return sum(p for y, p in final.items() if y not in GRAMMAR)


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def consistency_violation(fam, states):
    """max_{s, j!=k masked} || p(x_j|s) - sum_v p(x_k=v|s) p(x_j|s,x_k=v) ||_1 (0 iff pairwise-consistent)."""
    worst = 0.0
    for s in states:
        m = [j for j in range(L) if s[j] == MASK]
        if len(m) < 2:
            continue
        P = fam(s)
        for j in m:
            for k in m:
                if j == k:
                    continue
                mix = sum(P[k][v] * fam(fill(s, k, v))[j] for v in range(V) if P[k][v] > 1e-15)
                worst = max(worst, float(np.abs(P[j] - mix).sum()))
    return worst


def tv_rows(famA, famB, states, positions='nonfact'):
    vals = []
    for s in states:
        A, B = famA(s), famB(s)
        for j in range(L):
            if s[j] != MASK:
                continue
            if positions == 'nonfact' and j == FPOS:
                continue
            if positions == 'fact' and j != FPOS:
                continue
            vals.append(0.5 * np.abs(A[j] - B[j]).sum())
    return float(np.mean(vals)) if vals else float('nan'), float(np.max(vals)) if vals else float('nan')


# ---------------------------------------------------------------- run
def main():
    out = {'setup': {'V': V, 'L': L, 'tokens': TOK[:V], 'ystar': [TOK[t] for t in YSTAR],
                     'fact': TOK[FACT], 'eps_smoothing': EPS, 'configs': CONFIGS}}
    ref = {c: ref_family(J0[c]) for c in PROMPTS}
    anchor = ref_family(J_anchor)
    rF = ref['F']
    qF = ref_family(JQ['F'])

    # (0) reference properties
    out['ref'] = {c: {'pA_joint': float((J0[c] * (1 - NOFACT)).sum())} for c in PROMPTS}
    out['ref_pi_f_at_TF_FF'] = {''.join(TOK[x] if x != MASK else '_' for x in s) + '': float(rF(s)[FPOS][FACT])
                                for s in TF_FF}
    out['anchor_p_fact_at_TF_FF'] = {''.join(TOK[x] if x != MASK else '_' for x in s): float(anchor(s)[FPOS][FACT])
                                     for s in TF_FF}

    # (1) Q1 diagnostics: where does reference leakage get committed?
    q1 = {}
    for cn, cfg in CONFIGS.items():
        fin, diag = decode(rF, cfg, track=True)
        q1[cn] = {'leak': leak(fin), 'first_fact_commit_state[(TF-ff?, n_visible)]': diag}
    out['Q1_reference_commit_states'] = q1

    # (2) targets for the forget prompt
    fams = {
        'REF': rF,
        'GA_beta3_allTF': target_model(rF, t_ga(rF, 3.0), TF_STATES, 'ga3'),
        'GA_betaInf_allTF': target_model(rF, t_ga(rF, 50.0), TF_STATES, 'gaInf'),
        'ME(MDU_tau0)_allTF': target_model(rF, t_mdu(anchor, 0.0), TF_STATES, 'me'),
        'MDU_tau0.5_allTF': target_model(rF, t_mdu(anchor, 0.5), TF_STATES, 'mdu05'),
        'MDU_tau1_allTF': target_model(rF, t_mdu(anchor, 1.0), TF_STATES, 'mdu1'),
        'MDU_tau1_ffTF(state-restricted)': target_model(rF, t_mdu(anchor, 1.0), TF_FF, 'mdu1ff'),
        'FFX_MF_ffTF': target_model(rF, t_ffx_mf(rF), TF_FF, 'mf'),
        'FFX_CF_ffTF': target_model(rF, t_ffx_cf(rF), TF_FF, 'cf'),
        'FFX_BAYES_ffTF': target_model(rF, t_ffx_bayes(rF), TF_FF, 'bayes'),
        'Q_exact_ffTF': target_model(rF, t_exact_q('F'), TF_FF, 'qtf'),
        'FFX_MF_allFF(oracle states)': target_model(rF, t_ffx_mf(rF), [s for s in ALL_STATES if FACT not in s], 'mfall'),
        'Q_exact_all(oracle)': target_model(rF, t_exact_q('F'), [s for s in ALL_STATES if FACT not in s], 'qall'),
    }
    q_dec = decode(qF, CONFIGS['C2_T1_random_s4_b4'])   # = exact q joint (ancestral, consistent)
    p_dec = {cn: decode(rF, cfg) for cn, cfg in CONFIGS.items()}
    q_dec_cfg = {cn: decode(qF, cfg) for cn, cfg in CONFIGS.items()}
    res = {}
    for name, fam in fams.items():
        r = {}
        for cn, cfg in CONFIGS.items():
            fin = decode(fam, cfg)
            r[cn] = {'leak': round(leak(fin), 4), 'junk': round(junk(fin), 4),
                     'tv_to_q_same_cfg': round(tv(fin, q_dec_cfg[cn]), 4),
                     'P(X_kept_at_pos1)': round(sum(p for y, p in fin.items() if y[0] == X), 4)}
        r['max_leak_over_cfgs'] = max(r[cn]['leak'] for cn in CONFIGS)
        r['nonfact_rowTV_vs_q_on_ffTF(mean,max)'] = tv_rows(fam, qF, TF_FF, 'nonfact')
        r['fact_rowTV_vs_q_on_ffTF(mean,max)'] = tv_rows(fam, qF, TF_FF, 'fact')
        r['nonfact_rowTV_vs_ref_on_ffTF(mean,max)'] = tv_rows(fam, rF, TF_FF, 'nonfact')
        r['consistency_violation_on_TF_states'] = round(consistency_violation(fam, TF_STATES), 4)
        res[name] = r
    out['targets_forget_prompt'] = res
    out['ref_decode_leak'] = {cn: round(leak(d), 4) for cn, d in p_dec.items()}
    out['exact_q_decode'] = {cn: {'leak': round(leak(d), 6), 'junk': round(junk(d), 4)} for cn, d in q_dec_cfg.items()}

    # (3) MDU anchor applied to retain prompts' shared states (what the anchor says on retain answers)
    out['anchor_vs_ref_on_retain_TF_states'] = {}
    for c in ['R1', 'R2', 'R3']:
        ys = max(SPEC[c], key=SPEC[c].get)
        sts = [tuple(ys[j] if keep else MASK for j, keep in enumerate(m)) for m in itertools.product([0, 1], repeat=L)]
        out['anchor_vs_ref_on_retain_TF_states'][c] = tv_rows(anchor, ref[c], sts, 'all')

    # (4) inconsistent reference: robustness of the target estimators
    rob = {}
    for sigma in (0.3, 1.0):
        pr = perturbed(rF, sigma, seed=7)
        NEG_LOG.clear()
        est = {'MF': Fam(t_ffx_mf(pr), 'mf'), 'CF': Fam(t_ffx_cf(pr), 'cf'), 'BAYES': Fam(t_ffx_bayes(pr), 'by')}
        r = {'pert_rowTV_vs_ref_on_ffTF(nonfact)': tv_rows(pr, rF, TF_FF, 'nonfact'),
             'pert_consistency_violation_TF': round(consistency_violation(pr, TF_STATES), 4)}
        for en, ef in est.items():
            r[en + '_nonfact_rowTV_vs_exact_q(mean,max)'] = tv_rows(ef, qF, TF_FF, 'nonfact')
            r[en + '_fact_rowTV_vs_exact_q(mean,max)'] = tv_rows(ef, qF, TF_FF, 'fact')
        r['BAYES_negative_mass_before_clip(mean,max)'] = (float(np.mean(NEG_LOG)), float(np.max(NEG_LOG)))
        # leakage of target models built on the perturbed reference
        for en, tf in {'REFpert': None, 'MF': t_ffx_mf(pr), 'CF': t_ffx_cf(pr), 'BAYES': t_ffx_bayes(pr),
                       'MDU1_allTF': None}.items():
            if en == 'REFpert':
                fam = pr
            elif en == 'MDU1_allTF':
                fam = target_model(pr, t_mdu(perturbed(anchor, sigma, seed=11), 1.0), TF_STATES, 'x')
            else:
                fam = target_model(pr, tf, TF_FF, en)
            r[en + '_leak_by_cfg'] = {cn: round(leak(decode(fam, cfg)), 4) for cn, cfg in CONFIGS.items()}
        rob[f'sigma={sigma}'] = r
    out['inconsistent_reference'] = rob

    # (5) state-restriction support check: is q defined on fact-visible states?
    out['q_mass_of_fact_visible_TF_states'] = {''.join(TOK[x] if x != MASK else '_' for x in s):
                                               float(cond_joint(JQ['F'], s)[1]) for s in TF_FV[:3]}

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabular_check_out.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    # compact report
    print('reference leak by cfg:', out['ref_decode_leak'])
    print('exact q leak by cfg:', out['exact_q_decode'])
    print('pi_f (ref p(1951) at fact pos) on ffTF states:', {k: round(v, 3) for k, v in out['ref_pi_f_at_TF_FF'].items()})
    print('anchor p(1951) at fact pos on ffTF states:', {k: round(v, 3) for k, v in out['anchor_p_fact_at_TF_FF'].items()})
    print('Q1 commit states:', json.dumps(q1, default=float)[:1500])
    hdr = f"{'target':34s} " + ' '.join(f'{cn[:2]}:leak/junk/tvq' for cn in CONFIGS) + ' | nfTVq nfTVref cons'
    print(hdr)
    for name, r in res.items():
        row = ' '.join(f"{r[cn]['leak']:.3f}/{r[cn]['junk']:.3f}/{r[cn]['tv_to_q_same_cfg']:.2f}" for cn in CONFIGS)
        print(f"{name:34s} {row} | {r['nonfact_rowTV_vs_q_on_ffTF(mean,max)'][0]:.3f} "
              f"{r['nonfact_rowTV_vs_ref_on_ffTF(mean,max)'][0]:.3f} {r['consistency_violation_on_TF_states']:.3f}")
    print('anchor vs ref on retain TF states (mean,max rowTV):', out['anchor_vs_ref_on_retain_TF_states'])
    print('robustness:', json.dumps(rob, indent=1, default=float))


if __name__ == '__main__':
    main()
