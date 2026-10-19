"""Agent F independent checks of DERIVATION.tex Props 1-3, Lemmas 5-6, Theorem 7, limiting cases.
Own code (does not import code/vau or agents/*). Run: python f1_props.py > f1_props_output.txt"""
import numpy as np, sympy as sp
rng = np.random.default_rng(20261019)
out = {}

def proj_D(S, D, x, mu=0.0):
    G = S @ (D[:, None] * S.T) + mu * np.eye(S.shape[0])
    return x - D * (S.T @ np.linalg.solve(G, S @ x))

def proj_E(S, x):
    return x - S.T @ np.linalg.solve(S @ S.T, S @ x)

# ---------- Prop 1 (pre-projection under Adam) ----------
P, m = 40, 12
J = rng.normal(size=(m, P)); Pi = np.eye(P) - np.linalg.pinv(J) @ J
D = np.exp(rng.uniform(-3, 3, P))
mh = Pi @ rng.normal(size=P)
lhs = np.linalg.norm(J @ (D * mh)); c = (D.max() + D.min()) / 2
bound = np.linalg.norm(J, 2) * (D.max() - D.min()) / 2 * np.linalg.norm(mh)
out['P1b_drift_vs_bound'] = (lhs, bound, lhs <= bound)
# (a) coordinate-aligned null space is D-invariant -> zero
J2 = np.zeros((m, P)); J2[:, :m] = rng.normal(size=(m, m))
Pi2 = np.eye(P) - np.linalg.pinv(J2) @ J2
out['P1a_coord_aligned'] = np.linalg.norm(J2 @ (D * (Pi2 @ rng.normal(size=P))))
# step-1 remark: sign(Pi g) drift independent of projection accuracy
g = rng.normal(size=P)
out['P1_step1_drift_by_residual'] = {}
for r in [1e-2, 1e-5, 1e-8]:
    gp = Pi @ g + r * rng.normal(size=P)
    out['P1_step1_drift_by_residual'][r] = np.linalg.norm(J @ np.sign(gp))

# ---------- Prop 2 (trust region, sign, nonexpansive) ----------
viol = 0; worst_gap = np.inf; nonexp = 0
for _ in range(2000):
    P, k = 10, 3
    S = rng.normal(size=(k, P)); D = np.exp(rng.uniform(-4, 4, P)); g = rng.normal(size=P)
    xs = -proj_D(S, D, D * g); r = np.sqrt(xs @ (xs / D))
    val = -g @ xs
    viol += val < -1e-12
    # random feasible points at the same H radius never beat it
    N = np.linalg.svd(S)[2][k:].T
    z = N @ rng.normal(size=(P - k, 50)); z *= r / np.sqrt(np.sum(z * z / D[:, None], 0))
    worst_gap = min(worst_gap, val - np.max(-g @ z))
    w = rng.normal(size=P); pw = proj_D(S, D, w)
    nonexp += (pw @ (pw / D)) > (w @ (w / D)) * (1 + 1e-10)
out['P2_negative_progress_count'] = int(viol)
out['P2_min(opt - best_random_feasible)'] = worst_gap
out['P2_nonexpansive_violations'] = int(nonexp)
# Prop 2 with weight decay (outside statement): sign can flip
neg_wd = 0
for _ in range(2000):
    S = rng.normal(size=(3, 10)); D = np.exp(rng.uniform(-4, 4, 10)); g = rng.normal(size=10)
    th = rng.normal(size=10) * 5
    d0 = -1e-3 * D * g - 1e-3 * 0.01 * th
    neg_wd += (-g @ proj_D(S, D, d0)) < 0
out['P2_with_wd_negative_frac(theta~N(0,25))'] = neg_wd / 2000

# ---------- Prop 3 exact arithmetic (sympy) ----------
d1, d2 = sp.Rational(1, 100), sp.Integer(1)
for gg in [(1, sp.Rational(1, 2)), (1, -sp.Rational(1, 2))]:
    g1, g2 = sp.Integer(gg[0]), gg[1]
    Sm = sp.Matrix([[1, 1]]); Dm = sp.diag(d1, d2); gv = sp.Matrix([g1, g2])
    PiE = sp.eye(2) - Sm.T * (Sm * Sm.T).inv() * Sm
    PiD = sp.eye(2) - Dm * Sm.T * (Sm * Dm * Sm.T).inv() * Sm
    out[f'P3_g={gg}'] = dict(euclid=(gv.T * PiE * Dm * gv)[0], metric=(gv.T * PiD * Dm * gv)[0],
                              raw=(gv.T * Dm * gv)[0])

# ---------- Counterexample (momentum): explicit instance ----------
S = np.array([[1.0, 1.0, 0.0]]); D = np.array([1.0, 1.0, 1.0])
g = np.array([0.0, 0.0, 1.0]); mh = np.array([0.0, 0.0, -2.0]) * 0.1 + 0.9 * np.array([1, -1, -3.0])
out['CX_momentum_metric_progress'] = -g @ proj_D(S, D, -D * mh)

# ---------- Lemma 5 (ridge) ----------
S = rng.normal(size=(6, 30)); D = np.exp(rng.uniform(-3, 3, 30)); d0 = rng.normal(size=30)
for mu in [1e-6, 1e-2, 1.0]:
    G = S @ (D[:, None] * S.T); res = np.linalg.norm(S @ proj_D(S, D, d0, mu))
    bnd = mu / (np.linalg.eigvalsh(G).min() + mu) * np.linalg.norm(S @ d0)
    out[f'L5_mu={mu}'] = (res, bnd, res <= bnd * (1 + 1e-9))

# ---------- Lemma 6 (sketch residual) ----------
def E_sigma(J, Sig):
    Y = np.linalg.qr(J.T @ Sig.T)[0]
    return np.linalg.norm(J - (J @ Y) @ Y.T, 2)
n, P = 200, 120
U = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :P]; V = np.linalg.qr(rng.normal(size=(P, P)))[0]
for name, sv in [('rank20', np.r_[np.ones(20), np.zeros(P - 20)]),
                 ('decay0.8', 0.8 ** np.arange(P)), ('flat', np.ones(P))]:
    J = U @ np.diag(sv) @ V.T
    for k0, p in [(10, 5), (20, 5), (40, 10)]:
        k = k0 + p
        Es = [E_sigma(J, rng.normal(size=(k, n)) / np.sqrt(k)) for _ in range(200)]
        hmt = (1 + np.sqrt(k0 / (p - 1))) * sv[k0] + np.e * np.sqrt(k) / p * np.sqrt(np.sum(sv[k0:] ** 2))
        out[f'L6_{name}_k0={k0}_p={p}'] = dict(mean_E=float(np.mean(Es)), max_E=float(np.max(Es)),
                                                hmt_bound=float(hmt), holds=bool(np.mean(Es) <= hmt))
# adaptivity: x chosen AFTER Sigma (worst x in null(Sigma J)) -> bound still holds deterministically
J = U @ np.diag(0.8 ** np.arange(P)) @ V.T; Sig = rng.normal(size=(15, n)) / np.sqrt(15)
Nx = np.linalg.svd(Sig @ J)[2][15:].T
worst = np.linalg.svd(J @ Nx, compute_uv=False)[0]
out['L6_worst_adaptive_x_vs_E'] = (worst, E_sigma(J, Sig))

# ---------- Theorem 7 on a nonlinear map, stale J, ridge, sketch ----------
# f(th) = tanh(W2 tanh(W1 th)) style retained map, n outputs
P, h, n = 30, 25, 40
W1 = rng.normal(size=(h, P)) / np.sqrt(P); W2 = rng.normal(size=(n, h)) / np.sqrt(h)
f = lambda th: W2 @ np.tanh(W1 @ th)
Jf = lambda th: W2 @ ((1 - np.tanh(W1 @ th) ** 2)[:, None] * W1)
ok = 0; ratios = []
for trial in range(300):
    ths = rng.normal(size=P) * 0.5; th = ths + rng.normal(size=P) * 0.05
    k = rng.integers(3, 25); Sig = rng.normal(size=(k, n)) / np.sqrt(k)
    Js = Jf(ths); S = Sig @ Js; D = np.exp(rng.uniform(-2, 2, P)); d0 = -0.05 * D * rng.normal(size=P)
    mu = [0.0, 1e-4, 1e-1][trial % 3]
    d1 = proj_D(S, D, d0, mu) if mu > 0 or k <= np.linalg.matrix_rank(S) else proj_D(S, D, d0, 1e-12)
    s = 1.0; hvec = s * d1
    lhs = np.linalg.norm(f(th + hvec) - f(th))
    E = E_sigma(Js, Sig)
    Y = np.linalg.qr(Js.T @ Sig.T)[0]; Vb = np.linalg.qr(Js @ Y)[0]
    kap = 1 / np.linalg.svd(Sig @ Vb, compute_uv=False).min()
    # constants: L_J via finite differences along the segment (empirical sup), beta via 2nd-order remainder sup
    LJ = np.linalg.norm(Jf(th) - Js, 2) / np.linalg.norm(th - ths)
    ts = np.linspace(0, 1, 21)[1:]
    beta = max(2 * np.linalg.norm(f(th + t * hvec) - f(th) - Jf(th) @ (t * hvec)) / np.linalg.norm(t * hvec) ** 2 for t in ts)
    rhs = s * (E * np.linalg.norm(d1) + kap * np.linalg.norm(S @ d1)) + s * LJ * np.linalg.norm(th - ths) * np.linalg.norm(d1) + beta / 2 * s * s * np.linalg.norm(d1) ** 2
    ok += lhs <= rhs * (1 + 1e-9); ratios.append(lhs / rhs)
out['T7_bound_holds'] = f'{ok}/300'; out['T7_tightness_median_lhs/rhs'] = float(np.median(ratios))

# ---------- limiting cases ----------
S = rng.normal(size=(3, 8)); g = rng.normal(size=8); c = 0.37
out['LC_i_D=cI_metric_minus_euclid'] = np.linalg.norm(proj_D(S, c * np.ones(8), c * g) - proj_E(S, c * g))
# (i) with momentum, fixed J: pre == post for D=cI
gs = rng.normal(size=(5, 8)); mE = np.zeros(8); mP = np.zeros(8)
for gi in gs: mE = 0.9 * mE + 0.1 * gi; mP = 0.9 * mP + 0.1 * proj_E(S, gi)
out['LC_i_momentum_pre_vs_post_D=cI'] = np.linalg.norm(proj_E(S, mE) - mP)
# (ii) S=0 with mu = 1e-6 tr(G)/k -> G singular
try:
    S0 = np.zeros((3, 8)); G = S0 @ S0.T; mu = 1e-6 * np.trace(G) / 3
    np.linalg.solve(G + mu * np.eye(3), S0 @ g); out['LC_ii_S=0'] = 'solve ok (no error)'
except np.linalg.LinAlgError as e:
    out['LC_ii_S=0'] = f'LinAlgError: {e} (spec mu=1e-6 tr(G)/k is 0 when S=0)'
# (iii) g in range(S^T) with momentum and wd: progress exactly zero
gr = S.T @ rng.normal(size=3); D = np.exp(rng.uniform(-2, 2, 8))
d0 = -D * rng.normal(size=8) - 0.01 * rng.normal(size=8)
out['LC_iii_progress_g_in_rowspace_any_step'] = -gr @ proj_D(S, D, d0)
# (iv) sketch k < rank J = P: S has nontrivial null space although J has none
Jfull = rng.normal(size=(60, 20)); Sig = rng.normal(size=(5, 60)) / np.sqrt(5)
d1 = proj_D(Sig @ Jfull, np.ones(20), rng.normal(size=20))
out['LC_iv_rankJ=P_sketch_k=5_true_drift/|step|'] = np.linalg.norm(Jfull @ d1) / np.linalg.norm(d1)

for kk, vv in out.items(): print(kk, '=', vv)
