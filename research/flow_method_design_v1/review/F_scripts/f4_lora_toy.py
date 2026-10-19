"""Agent F adversarial toy: zero-init LoRA on layer 1 of f(x)=W2 tanh((W1+BA)x), AdamW torch defaults,
T=23 steps (GEM's budget), erasure loss = mean_e ||f(x_e)-a|| (GEM attract term, L2 not squared).
Methods (all with the SAME retained probe batch R1; lr tuned over a grid for every method):
  adamw        plain AdamW (lr grid = step-norm matching control)
  pre          Euclidean projection of g before AdamW (GPM/OGD-style)
  euclid       Euclidean projection of the actual AdamW step (Adam-NSCL-style)
  vau          D-metric projection of the actual AdamW step (+ reject if g.d1>=0), no gate
  penalty      AdamW on L_e + lam/2 * mean_{R1} ||f-f0||^2   (lam grid)
Regimes: base, rich retained set, stale probes (J only at theta0), beta1=0.99, sketch k=8, near-scalar D proxy.
Reports, per regime and drift budget tau2 (population drift E||f-f0||^2 on 4000 fresh retained inputs):
best erasure progress (L_e(theta0)-L_e(theta_T), held-out erasure inputs) over the grid subject to drift<=tau2,
median over seeds; and per-step diagnostics (sign reversals, ||d1||/||d0|| blow-up, nonlinear share of drift).
Own code; does not import code/vau."""
import numpy as np, itertools, json, sys
din, H, dout, r = 16, 24, 16, 4
T = 23
def setup(seed, Nr):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(H, din)) / np.sqrt(din); W2 = rng.normal(size=(dout, H)) / np.sqrt(H)
    A0 = rng.uniform(-1, 1, size=(r, din)) / np.sqrt(din); B0 = np.zeros((H, r))
    c = rng.normal(size=din); c *= 2.5 / np.linalg.norm(c) * np.sqrt(1)  # concept direction
    Xe = c + 0.3 * rng.normal(size=(64, din)); Xe_test = c + 0.3 * rng.normal(size=(64, din))
    Xpop = rng.normal(size=(4000, din)) * 0.7
    Xr1 = Xpop[:Nr]; Xpop = Xpop[Nr:]
    xa = rng.normal(size=din) * 0.7
    a = W2 @ np.tanh(W1 @ xa)          # anchor output (base model on an 'anchor' input)
    return dict(W1=W1, W2=W2, A0=A0, B0=B0, Xe=Xe, Xet=Xe_test, Xpop=Xpop, Xr1=Xr1, a=a, rng=rng)
def unpack(th):
    B = th[:H * r].reshape(H, r); A = th[H * r:].reshape(r, din); return B, A
def fwd(E, th, X):
    B, A = unpack(th); return np.tanh(X @ (E['W1'] + B @ A).T) @ E['W2'].T
def jac(E, th, X):
    """Jacobian of stacked outputs (N*dout) wrt theta = [vec B (row-major), vec A (row-major)]."""
    B, A = unpack(th); hcur = X @ (E['W1'] + B @ A).T; s = 1 - np.tanh(hcur) ** 2
    rows = []
    for n in range(X.shape[0]):
        x = X[n]; Ax = A @ x
        JB = np.kron(np.eye(H), Ax[None, :])            # H x (H r)
        JA = np.kron(B, x[None, :])                       # H x (r din)
        Jh = np.hstack([JB, JA]); rows.append(E['W2'] @ (s[n][:, None] * Jh))
    return np.vstack(rows)
def erase_loss_grad(E, th, X):
    B, A = unpack(th); hcur = X @ (E['W1'] + B @ A).T; t = np.tanh(hcur); f = t @ E['W2'].T
    dvec = f - E['a']; nrm = np.linalg.norm(dvec, axis=1) + 1e-12; L = nrm.mean()
    gf = dvec / nrm[:, None] / X.shape[0]                 # dL/df
    gh = (gf @ E['W2']) * (1 - t ** 2)                    # N x H
    gW = gh.T @ X                                         # H x din  (dL/dW_eff)
    gB = gW @ A.T; gA = B.T @ gW
    return L, np.r_[gB.ravel(), gA.ravel()]
def pen_grad(E, th, X, f0):
    B, A = unpack(th); hcur = X @ (E['W1'] + B @ A).T; t = np.tanh(hcur); f = t @ E['W2'].T
    gf = (f - f0) / X.shape[0]; gh = (gf @ E['W2']) * (1 - t ** 2); gW = gh.T @ X
    return np.r_[(gW @ A.T).ravel(), (B.T @ gW).ravel()]
def run(E, method, lr, lam=0.0, b1=0.9, stale=False, ksk=None, wd=0.01, diag=None):
    th = np.r_[E['B0'].ravel(), E['A0'].ravel()]; th0 = th.copy(); Pn = th.size
    m = np.zeros(Pn); v = np.zeros(Pn); f0r = fwd(E, th0, E['Xr1'])
    rng = np.random.default_rng(99)
    Sfix = None
    for t in range(1, T + 1):
        _, g = erase_loss_grad(E, th, E['Xe'])
        if method == 'penalty': g = g + lam * pen_grad(E, th, E['Xr1'], f0r)
        if method in ('pre', 'euclid', 'vau'):
            if Sfix is None or not stale:
                J = jac(E, th, E['Xr1'])
                if ksk: J = (rng.normal(size=(ksk, J.shape[0])) / np.sqrt(ksk)) @ J
                Sfix = J
            S = Sfix
        if method == 'pre':
            G = S @ S.T; g = g - S.T @ np.linalg.lstsq(G, S @ g, rcond=None)[0]
        m = b1 * m + (1 - b1) * g; v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - 0.999 ** t); D = 1 / (np.sqrt(vh) + 1e-8)
        d0 = -lr * D * mh - lr * wd * th
        if method == 'euclid':
            G = S @ S.T; d1 = d0 - S.T @ np.linalg.lstsq(G, S @ d0, rcond=None)[0]
            if diag is not None: diag['neg'].append(float(g @ d1 > 0))
            # Adam-NSCL code has no reject rule; we keep the raw Euclid control here
        elif method == 'vau':
            G = S @ (D[:, None] * S.T); mu = 1e-6 * np.trace(G) / G.shape[0] + 1e-300
            d1 = d0 - D * (S.T @ np.linalg.solve(G + mu * np.eye(G.shape[0]), S @ d0))
            if diag is not None:
                diag['neg'].append(float(g @ d1 >= 0)); diag['blow'].append(np.linalg.norm(d1) / np.linalg.norm(d0))
            if g @ d1 >= 0: d1 = 0 * d1
        else:
            d1 = d0
        if diag is not None and method in ('vau', 'euclid'):
            Jc = jac(E, th, E['Xr1']); lin = np.linalg.norm(Jc @ d1)
            ex = np.linalg.norm(fwd(E, th + d1, E['Xr1']) - fwd(E, th, E['Xr1']))
            diag['lin_vs_exact'].append((lin, ex))
        th = th + d1
    Le0, _ = erase_loss_grad(E, th0, E['Xet']); LeT, _ = erase_loss_grad(E, th, E['Xet'])
    drift = np.mean(np.sum((fwd(E, th, E['Xpop']) - fwd(E, th0, E['Xpop'])) ** 2, 1))
    return Le0 - LeT, drift, Le0
LRS = [10 ** e for e in np.linspace(-4, -1.5, 11)]
LAMS = [0.3, 1, 3, 10, 30, 100, 300, 1000]
REGIMES = {
    'base_Nr4': dict(Nr=4), 'rich_Nr16': dict(Nr=16), 'stale_Nr4': dict(Nr=4, stale=True),
    'beta1_0.99_Nr4': dict(Nr=4, b1=0.99), 'sketch8_Nr8': dict(Nr=8, ksk=8),
}
def main():
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    res = {}
    for rname, cfg in REGIMES.items():
        Nr = cfg['Nr']; kw = {k: v for k, v in cfg.items() if k != 'Nr'}
        allpts = {mth: [] for mth in ['adamw', 'pre', 'euclid', 'vau', 'penalty']}
        for seed in seeds:
            E = setup(seed, Nr)
            for mth in ['adamw', 'pre', 'euclid', 'vau']:
                pts = []
                for lr in LRS:
                    pr, dr, L0 = run(E, mth, lr, **kw); pts.append((pr / L0, dr))
                allpts[mth].append(pts)
            pts = []
            for lr, lam in itertools.product(LRS, LAMS):
                k2 = {k: v for k, v in kw.items() if k in ('b1',)}
                pr, dr, L0 = run(E, 'penalty', lr, lam=lam, **k2); pts.append((pr / L0, dr))
            allpts['penalty'].append(pts)
        # reference drift scale: plain AdamW at lr=1e-3 (GEM default)
        ref = np.median([p[LRS.index(min(LRS, key=lambda x: abs(x - 1e-3)))][1] for p in allpts['adamw']])
        out = {'ref_drift_adamw_lr1e-3': ref}
        for frac in [0.01, 0.05, 0.2]:
            tau2 = frac * ref; row = {}
            for mth, per_seed in allpts.items():
                best = [max([p for p, d in pts if d <= tau2], default=0.0) for pts in per_seed]
                row[mth] = (round(float(np.median(best)), 4), [round(b, 3) for b in best])
            out[f'tau2={frac}*ref'] = row
        res[rname] = out
        print(rname, json.dumps(out), flush=True)
    # diagnostics at GEM lr 1e-3, base regime, seed 0
    E = setup(0, 4); dg = {'neg': [], 'blow': [], 'lin_vs_exact': []}
    run(E, 'vau', 1e-3, diag=dg)
    dge = {'neg': [], 'blow': [], 'lin_vs_exact': []}; run(E, 'euclid', 1e-3, diag=dge)
    print('diag_vau: frac steps rejected (g.d1>=0)=', np.mean(dg['neg']), ' max ||d1||/||d0||=', max(dg['blow']),
          ' median ||d1||/||d0||=', np.median(dg['blow']))
    print('diag_vau: per-step retained change on R1 linear vs exact (first 6):',
          [(f'{a:.2e}', f'{b:.2e}') for a, b in dg['lin_vs_exact'][:6]])
    print('diag_euclid: frac steps with positive g.d1 (sign reversal)=', np.mean(dge['neg']))
if __name__ == '__main__':
    main()
