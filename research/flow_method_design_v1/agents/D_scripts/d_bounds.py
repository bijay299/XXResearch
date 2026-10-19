"""S1: endpoint error vs local velocity error (Alekseev-Groebner exact vs Gronwall bounds) for rectified flow.
Run from repo root:  python3 research/flow_method_design_v1/code/meter.py D_bounds -- \
   /tmp/claude-0/venv/bin/python research/flow_method_design_v1/agents/D_scripts/d_bounds.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flowtoy import GMM, euler, flux_sigmas

out = {}
s1 = 0.772  # GEM prefix lower sigma (A facts, COMPUTED)

# ---- S1a: unimodal Gaussian N(m, s^2 I): flow-map Jacobian is (s / sigma_tot(sig)) I exactly.
def st(sig, s):
    return np.sqrt((1 - sig) ** 2 * s ** 2 + sig ** 2)

rows = []
for s in [0.3, 0.1, 0.03]:
    g = GMM([[2.0, 0.0]], [1.0], s)
    c = np.array([1.0, 0.0])
    N = 4000
    sig = np.linspace(1, 0, N + 1)
    x0 = np.random.default_rng(0).standard_normal((64, 2))
    ref = euler(lambda x, k, sg: g.v(x, sg), x0, sig)[-1]
    pert = euler(lambda x, k, sg: g.v(x, sg) + (c if sg >= s1 else 0.0), x0, sig)[-1]
    actual = np.linalg.norm(pert - ref, axis=1).mean()
    # exact Alekseev-Groebner (continuous): |c| * int_{s1}^1 s/st(sig) dsig
    q = np.linspace(s1, 1, 20001)
    ag = np.trapezoid(s / st(q, s), q)
    # dv/dx (scalar, isotropic) = d log st / dsig = (sig - (1-sig) s^2) / st^2
    def dv(t): return (t - (1 - t) * s ** 2) / st(t, s) ** 2
    tt = np.linspace(0, 1, 200001)
    cumabs = np.concatenate([[0], np.cumsum(0.5 * (np.abs(dv(tt[1:])) + np.abs(dv(tt[:-1]))) * np.diff(tt))])
    cumsgn = np.concatenate([[0], np.cumsum(0.5 * (dv(tt[1:]) + dv(tt[:-1])) * np.diff(tt))])
    mask = tt >= s1
    gron = np.trapezoid(np.exp(cumabs[mask]), tt[mask])        # Lipschitz (|.|) Gronwall bound
    lognorm = np.trapezoid(np.exp(-cumsgn[mask]), tt[mask])    # one-sided (log-norm) bound
    rows.append(dict(s=s, actual_euler4000=actual, alekseev_exact=ag, lognorm_bound=lognorm,
                     gronwall_lipschitz_bound=gron, gronwall_over_actual=gron / actual,
                     window_length=1 - s1, contraction_at_s1=s / st(s1, s)))
out["S1a_unimodal"] = rows

# ---- S1b: same, discrete GEM/Flux 28-step Euler schedule, perturbation only on prefix steps k=0..10
sig28 = flux_sigmas()
rows = []
for s in [0.3, 0.1, 0.03]:
    g = GMM([[2.0, 0.0]], [1.0], s)
    c = np.array([1.0, 0.0])
    x0 = np.random.default_rng(1).standard_normal((256, 2))
    ref = euler(lambda x, k, sg: g.v(x, sg), x0, sig28)[-1]
    pert = euler(lambda x, k, sg: g.v(x, sg) + (c if k <= 10 else 0.0), x0, sig28)
    shift_mid = np.linalg.norm(pert[11] - euler(lambda x, k, sg: g.v(x, sg), x0, sig28)[11], axis=1).mean()
    shift_end = np.linalg.norm(pert[-1] - ref, axis=1).mean()
    rows.append(dict(s=s, state_shift_after_prefix=shift_mid, endpoint_shift=shift_end,
                     ratio=shift_end / shift_mid, predicted_ratio_s_over_st=s / st(sig28[11], s)))
out["S1b_flux28_prefix"] = rows

# ---- S1c: nonlinear (bimodal) -- first-order (linearised, reference-state) prediction vs actual.
#   Reference-state ("off-policy") linearisation: dx0 ~= sum_k Phi'_{k->end}(x_ref_k) * dsig_k * c
def endpoint_jac_fd(g, x, k, sig, h=1e-5):
    """finite-difference Jacobian of reference flow map from state x at index k to endpoint; x (n,2)."""
    J = np.zeros((x.shape[0], 2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        xp = euler(lambda z, kk, sg: g.v(z, sg), x + e, sig, k0=k)[-1]
        xm = euler(lambda z, kk, sg: g.v(z, sg), x - e, sig, k0=k)[-1]
        J[:, :, j] = (xp - xm) / (2 * h)
    return J

rows = []
for sep, name in [(6.0, "coarse"), (0.6, "fine")]:
    g = GMM([[-sep / 2, 0.0], [sep / 2, 0.0]], [0.5, 0.5], 0.1)
    x0 = np.random.default_rng(2).standard_normal((200, 2))
    traj = euler(lambda x, k, sg: g.v(x, sg), x0, sig28)
    lin = np.zeros_like(x0)
    for k in range(11):
        J = endpoint_jac_fd(g, traj[k + 1], k + 1, sig28)
        lin += np.einsum('nij,j->ni', J, np.array([1.0, 0.0])) * (sig28[k + 1] - sig28[k])
    for cmag in [0.01, 0.1, 0.5, 2.0]:
        c = np.array([cmag, 0.0])
        pert = euler(lambda x, k, sg: g.v(x, sg) + (c if k <= 10 else 0.0), x0, sig28)[-1]
        act = pert - traj[-1]
        pred = cmag * lin
        relerr = np.linalg.norm(act - pred, axis=1).sum() / max(np.linalg.norm(act, axis=1).sum(), 1e-12)
        flips = float(np.mean(g.nearest(pert) != g.nearest(traj[-1])))
        rows.append(dict(case=name, c=cmag, mean_actual_endpoint_shift=float(np.linalg.norm(act, axis=1).mean()),
                         linearised_rel_error=float(relerr), mode_flip_fraction=flips))
out["S1c_bimodal_linearisation"] = rows

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_bounds_output.json")
json.dump(out, open(path, "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
