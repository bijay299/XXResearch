"""Agent B numerical checks for GEM target-loss geometry (numpy only; no torch available).

Run (from repo root):
  python3 research/flow_method_design_v1/code/meter.py B_checks -- /tmp/claude-0/venv/bin/python research/flow_method_design_v1/agents/B_scripts/b_checks.py

Exact (eps_n = 0) continuous-norm losses are used on purpose: the all-ones eps_n shift is the CLOSED
artifact (cpr_gem_cpu_prototype); here we study the finite-offset geometry that remains once it is removed.
"""
import json
import numpy as np

rng = np.random.default_rng(0)
OUT = {}


def nrm(x):
    return np.linalg.norm(x, axis=-1)


def gem(v, a, b, eta):
    return np.maximum(nrm(v - a) - eta * nrm(v - b), 0.0)


# ---------------------------------------------------------------- T1 identity at eta = 1
# GEM(eta=1) = relu(2s-L) * L/S,  s=(v-a).e, S=D+ + D-  (L/S = eccentricity of ellipse through v, foci a,b)
d = 17
err = 0.0
for _ in range(2000):
    a, b, v = rng.normal(size=(3, d)) * rng.uniform(0.1, 5)
    L = nrm(b - a); e = (b - a) / L; s = (v - a) @ e; S = nrm(v - a) + nrm(v - b)
    err = max(err, abs(gem(v, a, b, 1.0) - max(2 * s - L, 0) * L / S))
OUT["T1_identity_max_abs_err"] = err

# ---------------------------------------------------------------- T2 orthogonality  <grad D+ - grad D-, grad D+ + grad D-> = 0
mx = 0.0
for _ in range(2000):
    a, b, v = rng.normal(size=(3, d))
    up = (v - a) / nrm(v - a); um = (v - b) / nrm(v - b)
    mx = max(mx, abs((up - um) @ (up + um)))
OUT["T2_orthogonality_max_abs"] = mx


# ---------------------------------------------------------------- flow simulator (full R^d, RK4, adaptive dt)
def grad_fn(kind, eta, a, b, L):
    k = eta * L / (1 + eta)
    e = (b - a) / L

    def g(v):
        dp = nrm(v - a); dm = nrm(v - b)
        up = (v - a) / dp; um = (v - b) / max(dm, 1e-300)
        if kind == "gem":
            return (up - eta * um) if dp - eta * dm > 0 else np.zeros_like(v)
        if kind == "anchor":
            return up
        if kind == "axial":      # relu(L-(1+eta)alpha), alpha=(b-v).e
            return (1 + eta) * e if (L - (1 + eta) * ((b - v) @ e)) > 0 else np.zeros_like(v)
        if kind == "trunc":      # relu(D+ - k)
            return up if dp > k else np.zeros_like(v)
        raise ValueError(kind)
    return g


def flow(kind, eta, eps, d=2, L=1.0, tmax=50.0, perp_dir=None):
    a = np.zeros(d); b = np.zeros(d); b[0] = L
    if perp_dir is None:
        n = np.zeros(d); n[1] = 1.0
    else:
        n = perp_dir
    v = b + eps * n
    g = grad_fn(kind, eta, a, b, L)
    S0 = nrm(v - a) + nrm(v - b)
    t = 0.0; Smax_dev = 0.0; rmax = eps; steps = 0
    while t < tmax and steps < 200000:
        gv = g(v)
        if not gv.any():
            break
        dm = nrm(v - b)
        h = 0.01 * max(dm, 1e-14); h = min(h, 0.01 * L)
        k1 = -g(v); k2 = -g(v + 0.5 * h * k1); k3 = -g(v + 0.5 * h * k2); k4 = -g(v + h * k3)
        v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h; steps += 1
        S = nrm(v - a) + nrm(v - b)
        Smax_dev = max(Smax_dev, abs(S - S0) / S0)
        r = nrm(v[1:])
        rmax = max(rmax, r)
    s_end = v[0]; r_end = nrm(v[1:])
    return dict(s_end=float(s_end), r_end=float(r_end), rmax=float(rmax), S_rel_drift=float(Smax_dev),
                dist_to_a=float(nrm(v - a)), steps=steps, t=float(t))


# ---------------------------------------------------------------- T3 eta=1 amplification law, 2-D reduced flow
def r_pred(eps, L=1.0):
    S0 = eps + np.sqrt(L * L + eps * eps)
    return np.sqrt(S0 * S0 - L * L) / 2


rows = []
for eps in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
    res = flow("gem", 1.0, eps)
    rows.append(dict(eps=eps, **res, r_pred=float(r_pred(eps)), amp=res["r_end"] / eps))
OUT["T3_eta1_perp_offset"] = rows
x = np.log([r["eps"] for r in rows[:4]]); y = np.log([r["r_end"] for r in rows[:4]])
OUT["T3_loglog_slope"] = float(np.polyfit(x, y, 1)[0])

# controls from the same start (eps=1e-4)
OUT["T3_controls_eps1e-4"] = {k: flow(k, 1.0, 1e-4) for k in ["anchor", "axial", "trunc"]}

# ---------------------------------------------------------------- T4 high-d (d=64) random orthogonal offset
d = 64
n = rng.normal(size=d); n[0] = 0; n /= nrm(n)
OUT["T4_d64_eta1_eps1e-4"] = flow("gem", 1.0, 1e-4, d=d, perp_dir=n)

# ---------------------------------------------------------------- T5 offset with axial component: delta = eps*(cos th e + sin th n)
rows = []
for th in [0.0, 0.5, 1.0, 1.5, np.pi / 2, 2.0, 2.8, np.pi - 1e-3]:
    d2 = 2; L = 1.0; eps = 1e-4
    a = np.zeros(2); b = np.array([L, 0.0]); v0 = b + eps * np.array([np.cos(th), np.sin(th)])
    S0 = nrm(v0 - a) + nrm(v0 - b)
    # simulate from custom start
    g = grad_fn("gem", 1.0, a, b, L); v = v0.copy(); t = 0; st = 0
    while st < 100000:
        if not g(v).any():
            break
        h = min(0.01 * max(nrm(v - b), 1e-14), 0.01 * L)
        k1 = -g(v); k2 = -g(v + 0.5 * h * k1); k3 = -g(v + 0.5 * h * k2); k4 = -g(v + h * k3)
        v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); st += 1
    rows.append(dict(theta=float(th), r_end=float(abs(v[1])), r_pred=float(np.sqrt(S0 ** 2 - L ** 2) / 2), s_end=float(v[0])))
OUT["T5_offset_angle"] = rows

# ---------------------------------------------------------------- T6 general eta: zero-set geometry + flow
rows = []
for eta in [0.3, 0.5, 0.8, 0.95]:
    L = 1.0; r = eta * L / (1 - eta ** 2); c0 = -eta ** 2 / (1 - eta ** 2)    # centre on axis (coordinate s), a=0,b=L
    # sample boundary points of the predicted ball and test D+ - eta D- == 0
    th = rng.uniform(0, 2 * np.pi, 200)
    pts = np.stack([c0 + r * np.cos(th), r * np.sin(th)], 1)
    resid = np.max(np.abs(nrm(pts) - eta * nrm(pts - np.array([L, 0.0]))))
    rows.append(dict(eta=eta, radius=r, centre_s=c0, boundary_resid=float(resid),
                     far_extent_behind_a=eta * L / (1 - eta), perp_extent_at_a=eta * L / np.sqrt(1 - eta ** 2),
                     trunc_ball_k=eta * L / (1 + eta)))
OUT["T6_apollonius"] = rows

rows = []
for eta in [0.5, 0.8, 1.0, 1.5]:
    res = flow("gem", eta, 1e-4)
    rows.append(dict(eta=eta, **res, axial_stop_pred=1 / (1 + eta)))
OUT["T6_flow_by_eta_eps1e-4"] = rows

# ---------------------------------------------------------------- T7 asymptotic flatness: eta=1, v=b+r n  => loss ~ L^2/(2r)
rows = []
for r in [1, 10, 100, 1e3, 1e4]:
    L = 1.0; a = np.zeros(2); b = np.array([L, 0.0]); v = b + r * np.array([0, 1.0])
    rows.append(dict(r=r, loss=float(gem(v, a, b, 1.0)), pred=L * L / (2 * r), s=L))
OUT["T7_flat_far_perp_no_progress"] = rows

# ---------------------------------------------------------------- T8 gradient norm / homogeneity / small contrast
rows = []
for L in [1.0, 1e-1, 1e-3, 1e-6]:
    a = np.zeros(2); b = np.array([L, 0]); v = np.array([0.5 * L, 0.3 * L]) * 0 + np.array([0.9 * L, 0.2 * L])
    for eta in [0.5, 1.0]:
        up = (v - a) / nrm(v - a); um = (v - b) / nrm(v - b)
        rows.append(dict(L=L, eta=eta, loss=float(gem(v, a, b, eta)), loss_over_L=float(gem(v, a, b, eta) / L),
                         gradnorm=float(nrm(up - eta * um))))
# anchor-only signal for v at fixed absolute offset 1e-3 from a while a ~ b (a=b): eta=1 gives zero loss
a = np.zeros(2); b = a.copy(); v = np.array([1e-3, 0.0])
rows.append(dict(note="a=b, v offset 1e-3", loss_eta1=float(gem(v, a, b, 1.0)), loss_eta05=float(gem(v, a, b, 0.5))))
OUT["T8_homogeneity_small_contrast"] = rows

# ---------------------------------------------------------------- T9 eta=1 transverse gradient sign map (descent grows |r| iff D- < D+)
rows = []
for (s, r) in [(1.0, 0.1), (0.8, 0.3), (0.6, 0.5), (0.4, 0.5)]:
    v = np.array([s, r]); a = np.zeros(2); b = np.array([1.0, 0])
    up = (v - a) / nrm(v - a); um = (v - b) / nrm(v - b)
    gperp = (up - um)[1]
    rows.append(dict(s=s, r=r, Dplus=float(nrm(v - a)), Dminus=float(nrm(v - b)), grad_perp=float(gperp),
                     descent_moves_away_from_axis=bool(gperp < 0), active=bool(nrm(v - a) > nrm(v - b))))
OUT["T9_transverse_sign"] = rows

print(json.dumps(OUT, indent=1, default=float))
with open(__file__.replace("b_checks.py", "b_checks_output.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)

# ---------------------------------------------------------------- T10 counterexamples outside the conservation theorem
def flow_M(M, eta, eps, L=1.0, th=np.pi / 2):
    a = np.zeros(2); b = np.array([L, 0.0]); v = b + eps * np.array([np.cos(th), np.sin(th)])
    S0 = nrm(v - a) + nrm(v - b); Smax = 0.0; st = 0
    def g(v):
        dp = nrm(v - a); dm = nrm(v - b)
        return -(M @ ((v - a) / dp - eta * (v - b) / dm)) if dp - eta * dm > 0 else np.zeros(2)
    while st < 200000 and g(v).any():
        h = min(0.01 * max(nrm(v - b), 1e-14), 0.01 * L)
        k1 = g(v); k2 = g(v + 0.5 * h * k1); k3 = g(v + 0.5 * h * k2); k4 = g(v + h * k3)
        v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); st += 1
        Smax = max(Smax, abs(nrm(v - a) + nrm(v - b) - S0) / S0)
    return dict(s_end=float(v[0]), r_end=float(abs(v[1])), S_rel_drift_max=float(Smax), r_pred_if_conserved=float(np.sqrt(S0**2 - L**2) / 2))

rows = []
for name, M in [("M=I", np.eye(2)), ("M=5I(time rescale only)", 5 * np.eye(2)),
                ("M=diag(1,100) (stiff transverse)", np.diag([1.0, 100.0])), ("M=diag(100,1) (stiff axial)", np.diag([100.0, 1.0]))]:
    rows.append(dict(M=name, eta=1.0, **flow_M(M, 1.0, 1e-4)))
rows.append(dict(M="M=I", eta=0.5, **flow_M(np.eye(2), 0.5, 1e-2)))
OUT["T10_counterexamples"] = rows
print(json.dumps(OUT["T10_counterexamples"], indent=1))
with open(__file__.replace("b_checks.py", "b_checks_output.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
