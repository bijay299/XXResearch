#!/usr/bin/env python3
"""Agent C -- exact small check of the Mask-State Discriminant Adapter (MSDA, see ../C_proposal.md) vs plain
FT / LoRA at matched forgetting, plus the separability failure boundary.

NAMING in the output JSONs (historical): 'GEraw{r}' = MSDA default basis (GEP of C_F vs C_R, frozen A);
'MSDA{r}' = MSDA with the Delta-k ("fact-free difference") basis (ablation); 'RandFixed{r}' = random frozen A;
'CF-*' = closed-form MEMIT-form edits (ablations); '-g' = [MASK]-gated; 'MSDA16raw' in sweep = default basis r=16.
Rank-rule arm is in c_rank_rule.py; GEP-initialised trainable LoRA in c_geinit.py.
Usage: c_toy_mswe.py {linear|main|sweep} [--seed S] [--alpha A] --out FILE

Toy (explicit, tiny, no neural training except a closed-form ridge "planting" and <=seconds Adam runs):
  tokens: 12 subjects, 2 relations, 12 objects, [MASK];  sequence = [p0, p1, a0, a1]
  facts  (s, r) -> answer (o1, o2)   (random objects);   copy  [z1, z2] -> answer (z1, z2)
  x_i  = E[tok_i] + P[i]
  h1   = x + softmax(x Wq (x Wk)^T / sqrt(dh) + Bpos) x Wv              (1 bidirectional head, fixed)
  k_i  = relu(A h1_i + b)                                                  (MLP hidden = "key", m dims)
  edit site: W (12 x m) applied at every position; after it one more bidirectional mixing
  with a frozen pattern a2 (from h1):  logits_i = W k_i + beta * sum_j a2_ij W k_j  (read only at [MASK])
  => logits_i = W kt_i, kt_i = k_i + beta * sum_j a2_ij k_j            (exactly linear in W)
  [MASK]-gated edit dW (applied only where tok_j == [MASK]):  dlogits_i = dW ktg_i,
     ktg_i = g_i k_i + beta * sum_j a2_ij g_j k_j.
  W0 is planted by ridge regression on teacher-forced (TF) SFT states + prompt-masked states (target uniform).

Methods (all deployed as a weight change of W only; forgetting matched on TF-state p(true) = THR):
  MSWE      closed form, keys over forget mask-state distribution (TF + alternative-visible "rollout" states),
            retain second moment from [MASK]-position keys of retain-train states; anchor (prompt-masked) targets
  MSWE-g    same, [MASK]-gated
  CF-single ROME/MEMIT-like: same solve but forget keys only from the all-masked state
  CF-visC   same forget keys, covariance from *visible*-position keys of retain text (MEMIT-style text stats)
  CF-iso    same forget keys, isotropic covariance (min-norm edit)
  FT-*      full-matrix Adam on MDU-style loss KL(p || anchor) + retain (rSFT lambda=1, or rKL lambda in {1,10})
  LoRA-r    dW = B A, B=0 init, A random, same losses
Gradient methods are early-stopped at the crossing and linearly interpolated between the two straddling
iterates to hit THR exactly; best LR/lambda chosen by HELD-OUT retain KL (oracle selection, generous to baselines).
"""
import json, math, sys, time, itertools
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

NS, NRL, NO = 12, 2, 12
SUB0, REL0, OBJ0 = 0, NS, NS + NRL
MASK = OBJ0 + NO
V = MASK + 1
THR = 0.15          # matched forgetting: mean p(true) over forget TF states (overridden per call)
THRS = (0.20, 0.15, 0.10)
CHANCE = 1.0 / NO


# ----------------------------------------------------------------------------- model
class Toy:
    def __init__(self, seed, d=32, m=256, beta=0.5, alpha=0.0, partner=None):
        g = torch.Generator().manual_seed(seed)
        self.m, self.beta = m, beta
        E = torch.randn(V, d, generator=g)
        if alpha > 0 and partner:
            E0 = E.clone()
            for f, p in partner.items():
                v = (1 - alpha) * E0[f] + alpha * E0[p]
                E[f] = v / v.norm() * E0[f].norm()
        self.E = E
        self.P = 0.5 * torch.randn(4, d, generator=g)
        dh = 16
        self.Wq = torch.randn(d, dh, generator=g) / math.sqrt(d)
        self.Wk = torch.randn(d, dh, generator=g) / math.sqrt(d)
        self.Wv = torch.randn(d, d, generator=g) / math.sqrt(d)
        self.Wq2 = torch.randn(d, dh, generator=g) / math.sqrt(d)
        self.Wk2 = torch.randn(d, dh, generator=g) / math.sqrt(d)
        self.Bpos = torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                  [1.5, 1.5, 0., .5], [1.5, 1.5, .5, 0.]])
        self.A = 1.5 * torch.randn(m, d, generator=g) / math.sqrt(d)
        self.b = 0.3 * torch.randn(m, generator=g)
        self.gen = g

    def feats(self, toks):
        """toks LongTensor [N,4] -> (kt, ktg) each [N,4,m]"""
        x = self.E[toks] + self.P[None]
        dh = self.Wq.shape[1]
        s = (x @ self.Wq) @ (x @ self.Wk).transpose(1, 2) / math.sqrt(dh) + self.Bpos[None]
        h1 = x + torch.softmax(s, -1) @ (x @ self.Wv)
        k = torch.relu(h1 @ self.A.T + self.b)
        s2 = (h1 @ self.Wq2) @ (h1 @ self.Wk2).transpose(1, 2) / math.sqrt(dh)
        a2 = torch.softmax(s2, -1)
        kt = k + self.beta * a2 @ k
        gm = (toks == MASK).double()[..., None]
        ktg = gm * k + self.beta * a2 @ (gm * k)
        return kt, ktg, k, a2, gm


# ----------------------------------------------------------------------------- data
def make_facts(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    o = torch.randint(0, NO, (NS, NRL, 2), generator=g)
    return o


def base_fact(o, s, r):
    return (SUB0 + s, REL0 + r), (OBJ0 + int(o[s, r, 0]), OBJ0 + int(o[s, r, 1]))


def base_copy(z1, z2):
    return (OBJ0 + z1, OBJ0 + z2), (OBJ0 + z1, OBJ0 + z2)


def states(prompt, ans, alt=0, pm=False, gen=None):
    """readout records (tokens, pos, target_token, is_tf). alt = #alternative visible tokens per pattern (-1 = all)."""
    p = [MASK, MASK] if pm else list(prompt)
    out = [(tuple(p + [MASK, MASK]), 2, ans[0], True), (tuple(p + [MASK, MASK]), 3, ans[1], True),
           (tuple(p + [ans[0], MASK]), 3, ans[1], True), (tuple(p + [MASK, ans[1]]), 2, ans[0], True)]
    if alt:
        objs = [OBJ0 + i for i in range(NO)]
        for vispos, tpos, keep in ((2, 3, ans[0]), (3, 2, ans[1])):
            cands = [v for v in objs if v != keep]
            if alt > 0:
                idx = torch.randperm(len(cands), generator=gen)[:alt]
                cands = [cands[i] for i in idx]
            for v in cands:
                t = p + [MASK, MASK]
                t[vispos] = v
                out.append((tuple(t), tpos, ans[tpos - 2], False))
    return out


def anchor_of(tok):
    return (MASK, MASK) + tuple(tok[2:])


class RecSet:
    """precomputed effective keys for a list of readout records"""
    def __init__(self, toy, recs, group=None):
        self.recs = recs
        toks = torch.tensor([r[0] for r in recs], dtype=torch.long)
        pos = torch.tensor([r[1] for r in recs])
        kt, ktg, _, _, _ = toy.feats(toks)
        ar = torch.arange(len(recs))
        self.kt, self.ktg = kt[ar, pos], ktg[ar, pos]
        atoks = torch.tensor([anchor_of(r[0]) for r in recs], dtype=torch.long)
        akt, aktg, _, _, _ = toy.feats(atoks)
        self.akt, self.aktg = akt[ar, pos], aktg[ar, pos]
        self.y = torch.tensor([r[2] - OBJ0 for r in recs])
        self.tf = torch.tensor([r[3] for r in recs])
        self.group = group


# ----------------------------------------------------------------------------- helpers
def logits(W0, dW, rs, gated):
    return rs.kt @ W0.T + (rs.ktg if gated else rs.kt) @ dW.T


def ptrue(W0, dW, rs, gated, mask=None):
    p = torch.softmax(logits(W0, dW, rs, gated), -1)[torch.arange(len(rs.y)), rs.y]
    return p if mask is None else p[mask]


def kl_p0_p(W0, dW, rs, gated):
    l0 = torch.log_softmax(rs.kt @ W0.T, -1)
    l1 = torch.log_softmax(logits(W0, dW, rs, gated), -1)
    return (l0.exp() * (l0 - l1)).sum(-1)


def generate(toy, W0, dW, gated, bases, mode):
    """2-token native-style decoding: par / l2r / r2l / conf ; returns token-accuracy vs truth"""
    def pred(toks, pos):
        kt, ktg, _, _, _ = toy.feats(torch.tensor(toks, dtype=torch.long))
        ar = torch.arange(len(toks))
        lg = kt[ar, pos] @ W0.T + (ktg if gated else kt)[ar, pos] @ dW.T
        pr = torch.softmax(lg, -1)
        return pr.argmax(-1), pr.max(-1).values
    P = [list(p) + [MASK, MASK] for p, _ in bases]
    truth = torch.tensor([[a[0] - OBJ0, a[1] - OBJ0] for _, a in bases])
    n = len(P)
    if mode == "par":
        x0, _ = pred(P, torch.full((n,), 2)); x1, _ = pred(P, torch.full((n,), 3))
    else:
        if mode in ("l2r", "r2l"):
            first = torch.full((n,), 2 if mode == "l2r" else 3)
        else:
            _, c0 = pred(P, torch.full((n,), 2)); _, c1 = pred(P, torch.full((n,), 3))
            first = torch.where(c0 >= c1, torch.tensor(2), torch.tensor(3))
        xf, _ = pred(P, first)
        Q = [list(t) for t in P]
        for i in range(n):
            Q[i][int(first[i])] = OBJ0 + int(xf[i])
        second = 5 - first
        xs, _ = pred(Q, second)
        x0 = torch.where(first == 2, xf, xs); x1 = torch.where(first == 2, xs, xf)
    gen = torch.stack([x0, x1], 1)
    return (gen == truth).double().mean().item()


# ----------------------------------------------------------------------------- closed form
def solve_cf(R, KX, C, mu):
    M = KX.T @ KX + mu * C
    return torch.linalg.solve(M, KX.T @ R).T   # (NO x m)


def bisect(fun, lo, hi, target, iters=40, increasing=True):
    """find x in [lo,hi] with fun(x)=target, fun monotone"""
    flo, fhi = fun(lo), fun(hi)
    if increasing and not (flo <= target <= fhi):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fun(mid) <= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------- experiment
def build(seed, alpha=0.0):
    o = make_facts(seed)
    forget_facts = [(s, 0) for s in range(4)]
    partner = {SUB0 + s: SUB0 + s + 8 for s in range(4)}
    toy = Toy(seed, alpha=alpha, partner=partner)
    g = torch.Generator().manual_seed(2000 + seed)
    allfacts = [(s, r) for s in range(NS) for r in range(NRL)]
    copies = [(a, b) for a in range(NO) for b in range(NO)]
    perm = torch.randperm(len(copies), generator=g).tolist()
    copy_tr = [copies[i] for i in perm[:72]]
    copy_ho = [copies[i] for i in perm[72:]]
    # ---- planting (pretraining+SFT stand-in): TF states, plus prompt-masked states -> uniform
    plant = []
    for s, r in allfacts:
        pr, an = base_fact(o, s, r)
        plant += [(t, p, y, 1.0) for (t, p, y, _) in states(pr, an)]
        plant += [(t, p, y, 0.0) for (t, p, y, _) in states(pr, an, pm=True)]
    for z1, z2 in copies:
        pr, an = base_copy(z1, z2)
        plant += [(t, p, y, 1.0) for (t, p, y, _) in states(pr, an)]
        plant += [(t, p, y, 0.0) for (t, p, y, _) in states(pr, an, pm=True)]
    toks = torch.tensor([r[0] for r in plant], dtype=torch.long)
    pos = torch.tensor([r[1] for r in plant])
    kt, _, _, _, _ = toy.feats(toks)
    K = kt[torch.arange(len(plant)), pos]
    T = torch.zeros(len(plant), NO)
    for i, (_, _, y, w) in enumerate(plant):
        if w > 0:
            T[i] = 8.0 * (torch.nn.functional.one_hot(torch.tensor(y - OBJ0), NO).double() - 1.0 / NO)
    lam = 1e-3 * (K.pow(2).sum() / K.shape[0])
    W0 = torch.linalg.solve(K.T @ K + lam * torch.eye(toy.m), K.T @ T).T
    # ---- unlearning splits
    F_fit, F_tf = [], []
    for s, r in forget_facts:
        pr, an = base_fact(o, s, r)
        F_fit += states(pr, an, alt=-1)
    R_tr = []
    rtr_facts = [(s, r) for s in range(4, 8) for r in range(NRL)] + [(0, 1), (1, 1)]
    for s, r in rtr_facts:
        pr, an = base_fact(o, s, r)
        R_tr += states(pr, an, alt=3, gen=g)
    for z in copy_tr:
        pr, an = base_copy(*z)
        R_tr += states(pr, an, alt=3, gen=g)
    groups = {
        "subj_nb": [(2, 1), (3, 1)],
        "rel_nb": [(s, 0) for s in range(8, 12)],
        "other": [(s, 1) for s in range(8, 12)],
    }
    HO = {}
    for name, fl in groups.items():
        recs = []
        for s, r in fl:
            pr, an = base_fact(o, s, r)
            recs += states(pr, an, alt=3, gen=g)
        HO[name] = RecSet(toy, recs)
    recs = []
    for z in copy_ho:
        pr, an = base_copy(*z)
        recs += states(pr, an, alt=3, gen=g)
    HO["copy"] = RecSet(toy, recs)
    # visible-position keys of retain-train *text* (MEMIT-style covariance stand-in)
    vis = []
    for s, r in rtr_facts:
        pr, an = base_fact(o, s, r)
        vis.append(tuple(pr) + tuple(an))
    for z in copy_tr:
        pr, an = base_copy(*z)
        vis.append(tuple(pr) + tuple(an))
    vkt, _, _, _, _ = toy.feats(torch.tensor(vis, dtype=torch.long))
    KVIS = vkt.reshape(-1, toy.m)
    D = dict(toy=toy, o=o, W0=W0, F=RecSet(toy, F_fit), Rtr=RecSet(toy, R_tr), HO=HO, KVIS=KVIS,
             forget_bases=[base_fact(o, s, r) for s, r in forget_facts],
             ho_bases={k: [base_fact(o, s, r) for s, r in v] for k, v in groups.items()},
             copy_ho_bases=[base_copy(*z) for z in copy_ho])
    return D


def evaluate(D, dW, gated):
    W0, toy = D["W0"], D["toy"]
    F = D["F"]
    out = {"forget_tf_p": ptrue(W0, dW, F, gated, F.tf).mean().item(),
           "forget_alt_p": ptrue(W0, dW, F, gated, ~F.tf).mean().item()}
    for mode in ("par", "l2r", "r2l", "conf"):
        out[f"forget_gen_{mode}"] = generate(toy, W0, dW, gated, D["forget_bases"], mode)
    kls, dps = [], []
    for name, rs in D["HO"].items():
        kl = kl_p0_p(W0, dW, rs, gated).mean().item()
        p0 = ptrue(W0, torch.zeros_like(dW), rs, gated, rs.tf).mean().item()
        p1 = ptrue(W0, dW, rs, gated, rs.tf).mean().item()
        out[f"ho_{name}_kl"] = kl
        out[f"ho_{name}_dp"] = p1 - p0
        kls.append(kl); dps.append(p1 - p0)
    out["ho_kl_mean"] = sum(kls) / len(kls)
    out["ho_dp_mean"] = sum(dps) / len(dps)
    out["rtr_kl"] = kl_p0_p(W0, dW, D["Rtr"], gated).mean().item()
    ga = [generate(toy, W0, dW, gated, D["ho_bases"][k], "l2r") for k in ("subj_nb", "rel_nb", "other")]
    out["ho_fact_gen_l2r"] = sum(ga) / 3
    out["ho_copy_gen_l2r"] = generate(toy, W0, dW, gated, D["copy_ho_bases"], "l2r")
    out["dW_fro"] = dW.norm().item()
    return out


def run_cf(D, kind, gated):
    W0, F, Rtr = D["W0"], D["F"], D["Rtr"]
    sel = torch.ones(len(F.y), dtype=torch.bool)
    if kind == "single":                                # only all-masked state keys (2 per fact)
        sel = torch.tensor([r[0][2] == MASK and r[0][3] == MASK for r in F.recs])
    KF = (F.ktg if gated else F.kt)[sel]
    R = ((F.akt - F.kt) @ W0.T)[sel]                   # residual to the prompt-masked anchor output
    KRm = Rtr.ktg if gated else Rtr.kt
    m = KF.shape[1]
    if kind in ("ours", "single"):
        C = KRm.T @ KRm / KRm.shape[0]
    elif kind == "visC":
        KV = D["KVIS"]
        C = KV.T @ KV / KV.shape[0]
    elif kind == "iso":
        C = torch.eye(m) * (KRm.pow(2).sum() / KRm.shape[0] / m)
    C = C + 1e-6 * torch.trace(C) / m * torch.eye(m)

    def f(logmu):
        dW = solve_cf(R, KF, C, math.exp(logmu))
        return ptrue(W0, dW, F, gated, F.tf).mean().item()
    lo, hi = -14.0, 10.0
    if f(lo) > THR:
        dW = solve_cf(R, KF, C, math.exp(lo))
        r = evaluate(D, dW, gated); r["feasible"] = False
        return r
    # f increases with mu (more regularisation -> less forgetting); find mu with f=THR
    a, b = lo, hi
    for _ in range(50):
        mid = 0.5 * (a + b)
        if f(mid) < THR:
            a = mid
        else:
            b = mid
    dW = solve_cf(R, KF, C, math.exp(a))
    r = evaluate(D, dW, gated); r["feasible"] = True; r["log_mu"] = a
    return r


def ge_basis(D, rank, which="delta", gated=False, shrink=1e-2, unit=True):
    """MSDA row space: top generalized eigenvectors of (C_x, C_R); rows normalised so A C_R A^T = I.
    which = 'delta': fact-carrying keys k(s)-k(anchor(s)) ; 'raw': forget keys k(s) ; 'rand': random fixed."""
    F, Rtr = D["F"], D["Rtr"]
    KR = Rtr.ktg if gated else Rtr.kt
    m = KR.shape[1]
    CR = KR.T @ KR / KR.shape[0]
    CR = CR + shrink * torch.trace(CR) / m * torch.eye(m)       # shrinkage: C_R is a finite-sample estimate
    if which == "rand":
        g = torch.Generator().manual_seed(5)
        A = torch.randn(rank, m, generator=g) / math.sqrt(m)
        return A, None
    if which == "delta":
        KX = (F.ktg - F.aktg) if gated else (F.kt - F.akt)
    else:
        KX = F.ktg if gated else F.kt
    CX = KX.T @ KX / KX.shape[0]
    L = torch.linalg.cholesky(CR)
    Li = torch.linalg.inv(L)
    M = Li @ CX @ Li.T
    ev, U = torch.linalg.eigh(M)
    idx = torch.argsort(ev, descending=True)[:rank]
    A = (Li.T @ U[:, idx]).T            # rows a with a C_R a^T = 1, a C_X a^T = mu
    if unit:                            # row directions only (scale is learned in B); keeps Adam step sizes comparable
        A = A / A.norm(dim=1, keepdim=True)
    return A, ev[idx]


def run_grad(D, param, retain, lam, lr, gated, rank=4, max_steps=2000, seed=0, A_fixed=None):
    """Adam on dW (ft) | B@A (lora, both trainable) | B@A_fixed (msda / rand-fixed). Records the matched
    point for every THR in THRS (interpolating between the two straddling iterates)."""
    W0, F, Rtr = D["W0"], D["F"], D["Rtr"]
    m = W0.shape[1]
    g = torch.Generator().manual_seed(77 + seed)
    if param == "ft":
        Dm = torch.zeros(NO, m, requires_grad=True); ps = [Dm]
        dfun = lambda: Dm
    elif param == "lora":
        B = torch.zeros(NO, rank, requires_grad=True)
        A = (torch.randn(rank, m, generator=g) / math.sqrt(m)).requires_grad_(True)
        ps = [B, A]; dfun = lambda: B @ A
    elif param == "geinit":                     # LoRA whose A is *initialised* at the GEP rows, then trained
        B = torch.zeros(NO, A_fixed.shape[0], requires_grad=True)
        A = A_fixed.clone().requires_grad_(True)
        ps = [B, A]; dfun = lambda: B @ A
    else:
        Af = A_fixed
        B = torch.zeros(NO, Af.shape[0], requires_grad=True)
        ps = [B]; dfun = lambda: B @ Af
    opt = torch.optim.Adam(ps, lr=lr)
    qa = torch.softmax(F.akt @ W0.T, -1)
    lq = torch.log(qa)
    tfm = Rtr.tf
    p0r = torch.softmax(Rtr.kt @ W0.T, -1)
    prev = torch.zeros(NO, m)
    todo = sorted(THRS, reverse=True)
    found = {}
    fp = lambda X: ptrue(W0, X, F, gated, F.tf).mean().item()
    for step in range(max_steps):
        dW = dfun()
        lf = torch.log_softmax(logits(W0, dW, F, gated), -1)
        loss_f = (lf.exp() * (lf - lq)).sum(-1).mean()            # MDU-style KL(p || anchor), tau = 1
        lr_ = torch.log_softmax(logits(W0, dW, Rtr, gated), -1)
        if retain == "sft":
            loss_r = -lr_[tfm][torch.arange(int(tfm.sum())), Rtr.y[tfm]].mean()
        else:
            loss_r = (p0r * (torch.log(p0r) - lr_)).sum(-1).mean()
        loss = loss_f + lam * loss_r
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cur = dfun().detach().clone()
            fcur = fp(cur)
            while todo and fcur <= todo[0]:
                thr = todo.pop(0)
                a, b = 0.0, 1.0
                for _ in range(40):
                    mid = 0.5 * (a + b)
                    if fp((1 - mid) * prev + mid * cur) > thr:
                        a = mid
                    else:
                        b = mid
                found[thr] = ((1 - b) * prev + b * cur, step + 1)
        if not todo:
            break
        prev = cur
    return found


def run_cf_thr(D, kind, gated, thr):
    global THR
    THR = thr
    return run_cf(D, kind, gated)


def frontier(D, param, configs, gated, rank=4, A_fixed=None):
    """best (lowest held-out KL) matched point per THR over the (retain, lam, lr) grid"""
    best = {}
    for retain, lam in configs:
        for lr in (3e-3, 1e-2, 3e-2, 1e-1):
            found = run_grad(D, param, retain, lam, lr, gated, rank=rank, A_fixed=A_fixed)
            for thr, (dW, steps) in found.items():
                r = evaluate(D, dW, gated)
                r.update(retain=retain, lam=lam, lr=lr, steps=steps, feasible=True)
                if thr not in best or r["ho_kl_mean"] < best[thr]["ho_kl_mean"]:
                    best[thr] = r
    return {thr: best.get(thr, {"feasible": False}) for thr in THRS}


def separability(D, gated, which="delta"):
    F, Rtr = D["F"], D["Rtr"]
    KR = Rtr.ktg if gated else Rtr.kt
    m = KR.shape[1]
    A, mu = ge_basis(D, 16, which, gated)
    return {"mu_top": mu[:4].tolist(), "mu_16": mu[15].item()}


def partner_leak(D, A):
    """whitened overlap of forget-fact and partner-fact (s+8, r=0) all-masked keys inside the MSDA row space"""
    toy, o = D["toy"], D["o"]
    rhos, coss = [], []
    for s in range(4):
        (pf, _), (pp, _) = base_fact(o, s, 0), base_fact(o, s + 8, 0)
        t = torch.tensor([list(pf) + [MASK, MASK], list(pp) + [MASK, MASK]], dtype=torch.long)
        kt, _, _, _, _ = toy.feats(t)
        K = kt[:, 2]
        coss.append((K[0] @ K[1] / K[0].norm() / K[1].norm()).item())
        z = K @ A.T
        rhos.append((z[0] @ z[1] / (z[0] @ z[0])).item())
    return sum(rhos) / 4, sum(coss) / 4


def premodel(D):
    W0 = D["W0"]; z = torch.zeros_like(W0)
    out = {"forget_tf_p0": ptrue(W0, z, D["F"], False, D["F"].tf).mean().item(),
           "forget_alt_p0": ptrue(W0, z, D["F"], False, ~D["F"].tf).mean().item(),
           "forget_gen_l2r0": generate(D["toy"], W0, z, False, D["forget_bases"], "l2r"),
           "forget_gen_par0": generate(D["toy"], W0, z, False, D["forget_bases"], "par"),
           "ho_fact_gen_l2r0": sum(generate(D["toy"], W0, z, False, D["ho_bases"][k], "l2r")
                                   for k in ("subj_nb", "rel_nb", "other")) / 3,
           "ho_copy_gen_l2r0": generate(D["toy"], W0, z, False, D["copy_ho_bases"], "l2r")}
    anc = torch.softmax(D["F"].akt @ W0.T, -1)[torch.arange(len(D["F"].y)), D["F"].y]
    out["anchor_p_true"] = anc[D["F"].tf].mean().item()
    return out


def linear_prop_check(seed=0, m=40, nR=4000):
    """Exact linear-layer check of Proposition 1."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    out = []
    for cond in (1.0, 10.0, 1e2, 1e3):
        spec = torch.logspace(0, -math.log10(cond), m)
        KR = torch.randn(nR, m, generator=g) * spec.sqrt() @ Q.T
        C = KR.T @ KR / nR
        ev = torch.linalg.eigvalsh(C); kap = (ev.max() / ev.min()).item()
        k = torch.randn(m, generator=g); r = torch.randn(NO, generator=g)
        Ci_k = torch.linalg.solve(C, k)
        D_star = (r @ r / (k @ Ci_k)).item()
        dW = torch.outer(r, Ci_k) / (k @ Ci_k)
        D_star_num = torch.trace(dW @ C @ dW.T).item()
        dWe = torch.outer(r, k) / (k @ k)
        D_euc = torch.trace(dWe @ C @ dWe.T).item()
        kant = (kap + 1) ** 2 / (4 * kap)
        KF = torch.randn(5, m, generator=g)
        CF = KF.T @ KF / 5
        mu = torch.linalg.eigvals(torch.linalg.solve(C, CF)).real.max().item()
        worst = min(torch.trace(X @ C @ X.T).item() / torch.trace(X @ CF @ X.T).item()
                    for X in [torch.randn(NO, m, generator=g) for _ in range(200)])
        kin = (torch.randn(m, generator=g) * spec.sqrt()) @ Q.T
        D_in = (r @ r / (kin @ torch.linalg.solve(C, kin))).item()
        out.append(dict(cond=cond, kappa=kap, D_star_formula=D_star, D_star_numeric=D_star_num,
                        D_euclid=D_euc, gain=D_euc / D_star, kantorovich_bound=kant,
                        rayleigh_lb=1 / mu, min_ratio_random_edits=worst,
                        D_star_key_from_retain_dist=D_in, r2=(r @ r).item()))
    return out


KL_CFG = [("kl", 1.0), ("kl", 10.0)]


def main_seed(seed, quick=False):
    D = build(seed)
    row = {"pre": premodel(D), "sep_delta": separability(D, False, "delta"),
           "sep_raw": separability(D, False, "raw")}
    for kind in ("ours", "single", "visC", "iso"):
        row[f"CF-{kind}"] = {thr: run_cf_thr(D, kind, False, thr) for thr in THRS}
    row["CF-ours-g"] = {thr: run_cf_thr(D, "ours", True, thr) for thr in THRS}
    row["FT"] = frontier(D, "ft", KL_CFG, False)
    row["FT-rSFT"] = frontier(D, "ft", [("sft", 1.0)], False)
    for rank in (4, 16):
        row[f"LoRA{rank}"] = frontier(D, "lora", KL_CFG, False, rank=rank)
        A, _ = ge_basis(D, rank, "delta")
        row[f"MSDA{rank}"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    A, _ = ge_basis(D, 48, "delta"); row["MSDA48"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    A, _ = ge_basis(D, 48, "raw"); row["GEraw48"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    row["LoRA48"] = frontier(D, "lora", KL_CFG, False, rank=48)
    A, _ = ge_basis(D, 48, "rand"); row["RandFixed48"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    A, _ = ge_basis(D, 16, "raw"); row["GEraw16"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    A, _ = ge_basis(D, 16, "rand"); row["RandFixed16"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    A, _ = ge_basis(D, 48, "delta", gated=True); row["MSDA48-g"] = frontier(D, "fixed", KL_CFG, True, A_fixed=A)
    A, _ = ge_basis(D, 16, "delta", gated=True); row["MSDA16-g"] = frontier(D, "fixed", KL_CFG, True, A_fixed=A)
    row["FT-g"] = frontier(D, "ft", KL_CFG, True)
    return row


def sweep_seed(seed, alpha):
    D = build(seed, alpha=alpha)
    A, mu = ge_basis(D, 16, "raw")             # default MSDA basis (X = F), r = 16
    rho, cos = partner_leak(D, A)
    rr = {"pre": premodel(D), "rho_in_MSDA_space": rho, "cos_euclid": cos, "mu_top": mu[:4].tolist(),
          "mu_16": mu[15].item()}
    rr["MSDA16raw"] = frontier(D, "fixed", KL_CFG, False, A_fixed=A)
    rr["LoRA16"] = frontier(D, "lora", KL_CFG, False, rank=16)
    rr["FT"] = frontier(D, "ft", KL_CFG, False)
    return rr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["linear", "main", "sweep"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.mode == "linear":
        res = {"linear_prop": linear_prop_check()}
    elif a.mode == "main":
        res = {"seed": a.seed, "THRS": THRS, "main": main_seed(a.seed)}
    else:
        res = {"seed": a.seed, "alpha": a.alpha, "THRS": THRS, "sweep": sweep_seed(a.seed, a.alpha)}
    res["wall_s"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=lambda x: None)
    print(f"done {res['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
