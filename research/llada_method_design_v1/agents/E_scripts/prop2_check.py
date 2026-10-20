"""Agent E — exact tabular check of THEORY.tex Proposition 2 (escape characterisation, zero-sharing class).
Exact DP of the pinned `generate` semantics (Gumbel-max = softmax(z/T) draws; lc = top-k of untempered p(x0) with
lower-position tie-break; rnd = uniform k-subset; uniform k schedule per block incl. k_n = 0 steps; no re-masking).
Checks, for several configs (incl. semi-AR blocks, k_n=0 steps, T=0 and lc):
  (a) P^p(Lambda) <= e_pi(S) for random p in P_S; P^p(sigma_S<inf) == e for every p in P_S (law up to sigma_S);
      P^{p_delta}(Lambda) -> e (sup); P^{rbar}(Lambda) = 0 (inf).
  (b) S = visitation of rbar  =>  e = 0; a p in P_S visits exactly rbar's states.
  (c) S = TF fact-free states, one-token rnd (N=L=b): e >= 1-1/L (m=1), e = 1 (m=2).
  eps>0 variant: sup over p matching r_eps on S equals P^{rbar_eps}(leak before escape or escape).
"""
import itertools, json, math, os, hashlib
import numpy as np

L, V = 4, 5                       # tokens: 0 a, 1 b, 2 e, 3 f, 4 g
MASK = -1
WORLDS = {"m1": dict(y=(0, 3, 1, 2), F=(3,)), "m2": dict(y=(0, 3, 4, 2), F=(3, 4))}


def h(*xs):
    return int(hashlib.md5(repr(xs).encode()).hexdigest()[:8], 16)


def ref_p(s, i, y, seed=0):
    r = np.random.default_rng(h(s, i, seed)); z = 2 * r.normal(size=V); z[y[i]] += 3
    e = np.exp(z - z.max()); return e / e.sum()


def fact_free(s, F):
    return not any(t in F for t in s if t != MASK)


def excl(p, F, eps):
    pF = p[list(F)].sum()
    if pF <= eps: return p.copy()
    q = p.copy(); m = np.zeros(V, bool); m[list(F)] = True
    q[m] *= eps / pF; q[~m] *= (1 - eps) / (1 - pF); return q


def make_models(world, S, eps=0.0, delta=1e-3):
    y, F = world["y"], world["F"]
    def rbar(s, i):
        p = ref_p(s, i, y); return excl(p, F, eps) if fact_free(s, F) else p
    def prand(s, i):
        if s in S: return rbar(s, i)
        r = np.random.default_rng(h(s, i, 99)); z = 2 * r.normal(size=V); e = np.exp(z - z.max()); return e / e.sum()
    def pdelta(s, i, d=delta):
        if s in S: return rbar(s, i)
        if fact_free(s, F):
            z = np.zeros(V); z[F[0]] = 1.0 / d; e = np.exp(z - z.max()); return e / e.sum()
        return ref_p(s, i, y)
    return rbar, prand, pdelta


def schedule(N, b):
    nb = L // b; spb = N // nb; out = []
    for blk in range(nb):
        base, rem = b // spb, b % spb
        for j in range(spb): out.append((blk, base + (1 if j < rem else 0)))
    return out


def tempered(p, T):
    if T == 0:
        q = np.zeros(V); q[int(np.argmax(p))] = 1.0; return q
    with np.errstate(divide="ignore"):
        lz = np.log(p) / T
    lz = lz - lz.max(); e = np.exp(lz); return e / e.sum()


def run(model, cfg, S, F):
    """exact: returns P(leak), P(sigma_S<inf), P(leak before escape), set of visited states with M != empty."""
    N, b, rule, T = cfg
    dist = {((MASK,) * L, False, False, False): 1.0}   # (state, escaped, leaked, leaked_before_escape)
    visited = set()
    for (blk, k) in schedule(N, b):
        new = {}
        for (s, esc, lk, lbe), w in dist.items():
            M = [i for i in range(L) if s[i] == MASK]
            if M: visited.add(s)
            if M and not esc and s not in S: esc = True
            E = [i for i in M if blk * b <= i < (blk + 1) * b]
            if k == 0 or not E:
                key = (s, esc, lk, lbe); new[key] = new.get(key, 0) + w; continue
            P = {i: model(s, i) for i in E}; Q = {i: tempered(P[i], T) for i in E}
            supp = [[v for v in range(V) if Q[i][v] > 0] for i in E]
            for draw in itertools.product(*supp):
                pw = w * np.prod([Q[i][v] for i, v in zip(E, draw)])
                if pw == 0: continue
                if rule == "lc":
                    order = sorted(range(len(E)), key=lambda j: (-P[E[j]][draw[j]], E[j]))
                    sels = [(tuple(order[:k]), 1.0)]
                else:
                    subs = list(itertools.combinations(range(len(E)), k)); sels = [(c, 1.0 / len(subs)) for c in subs]
                for sel, ps in sels:
                    s2 = list(s); hit = False
                    for j in sel:
                        s2[E[j]] = draw[j]; hit |= draw[j] in F
                    key = (tuple(s2), esc, lk or hit, lbe or (hit and not esc))
                    new[key] = new.get(key, 0) + pw * ps
        dist = new
    leak = sum(w for (s, e, l, lb), w in dist.items() if l)
    esc = sum(w for (s, e, l, lb), w in dist.items() if e)
    lbe = sum(w for (s, e, l, lb), w in dist.items() if lb)
    either = sum(w for (s, e, l, lb), w in dist.items() if e or lb)
    return dict(leak=leak, esc=esc, leak_before_esc=lbe, esc_or_lbe=either), visited


def S_TF(world):
    y, F = world["y"], world["F"]; fpos = [j for j in range(L) if y[j] in F]; out = set()
    for vis in itertools.product([0, 1], repeat=L):
        if any(vis[j] for j in fpos): continue
        out.add(tuple(y[j] if vis[j] else MASK for j in range(L)))
    return out


CFGS = [(4, 4, "rnd", 1.0), (4, 4, "rnd", 0.5), (4, 4, "rnd", 0.0), (4, 4, "lc", 0.0), (4, 4, "lc", 1.0),
        (2, 4, "rnd", 1.0), (2, 4, "lc", 0.5), (4, 2, "lc", 1.0), (4, 1, "lc", 1.0),
        (8, 4, "rnd", 1.0),   # k_n = 0 steps at the end
        (8, 2, "lc", 0.0)]    # semi-AR with k_n = 0 steps mid-sequence
OUT = {"schedules": {str(c): schedule(c[0], c[1]) for c in CFGS}}
ok = True
for wname, world in WORLDS.items():
    F = world["F"]; S = S_TF(world)
    for cfg in CFGS:
        rbar, prand, pdelta = make_models(world, S)
        rr, vis_r = run(rbar, cfg, S, F); rp, _ = run(prand, cfg, S, F); rd, _ = run(pdelta, cfg, S, F)
        rd5, _ = run(lambda s, i: pdelta(s, i, 0.2), cfg, S, F)
        e = rr["esc"]
        # (b): S' = visitation of rbar under cfg
        Sv = vis_r; rbar2, prand2, pdelta2 = make_models(world, Sv)
        rr2, _ = run(rbar2, cfg, Sv, F); rp2, vis_p2 = run(prand2, cfg, Sv, F); rd2, _ = run(pdelta2, cfg, Sv, F)
        # eps>0 variant on S_TF
        rbe, _, pde = make_models(world, S, eps=0.05)
        re_, _ = run(rbe, cfg, S, F); rde, _ = run(pde, cfg, S, F)
        row = dict(e=e, leak_rbar=rr["leak"], leak_prand=rp["leak"], esc_prand=rp["esc"], leak_pdelta_1em3=rd["leak"],
                   leak_pdelta_0p2=rd5["leak"], lbe_prand=rp["leak_before_esc"],
                   b_e=rr2["esc"], b_leak_prand=rp2["leak"], b_leak_pdelta=rd2["leak"], b_same_visitation=(vis_p2 == Sv),
                   eps05_sup_pred=re_["esc_or_lbe"], eps05_leak_pdelta=rde["leak"], eps05_e_only=re_["esc"])
        checks = [rp["leak"] <= e + 1e-12, abs(rp["esc"] - e) < 1e-12, abs(rd["leak"] - e) < 1e-9, rr["leak"] < 1e-15,
                  rd5["leak"] <= rd["leak"] + 1e-12, rr2["esc"] < 1e-15, rp2["leak"] < 1e-15, rd2["leak"] < 1e-15,
                  vis_p2 == Sv, abs(rde["leak"] - re_["esc_or_lbe"]) < 1e-9]
        if cfg[:3] == (4, 4, "rnd"):
            checks.append(e >= 1 - 1 / L - 1e-12 if wname == "m1" else abs(e - 1) < 1e-12)
        row["all_checks_pass"] = all(checks); ok &= all(checks)
        OUT[f"{wname} {cfg}"] = row
        print(wname, cfg, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
OUT["ALL_PASS"] = ok
print("ALL_PASS", ok)
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop2_check_out.json"), "w"), indent=1)
