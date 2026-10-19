"""S2+S3: GEM(eta=1) hinge erasure in a 2D closed-form conditional flow; state coverage / timestep window / parameter
sharing / importance sampling. numpy only. Run from repo root:
  python3 research/flow_method_design_v1/code/meter.py D_window_toy -- \
     /tmp/claude-0/venv/bin/python research/flow_method_design_v1/agents/D_scripts/d_window_toy.py
Toy simplifications (do not transfer to FLUX without test): no guidance (GEM rollout uses 3.0, loss 1.0); batch of
B noises per iteration (GEM: 1); student = target teacher + explicit correction Delta (GEM: LoRA on q/k);
Adam lr 0.05 (GEM 1e-3 for LoRA, not comparable)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flowtoy import GMM, euler, flux_sigmas

SIG = flux_sigmas()            # 29 values, SIG[28] = 0
NS = 28
S = 0.1
BUDGET = 250                   # GEM max_step_budget (timestep evaluations), EG:26


def cases():
    return {
        "UNI": dict(target=GMM([[2, 0]], [1], S),
                    anchor=GMM([[2, 0], [-2, 0], [0, 2], [0, -2]], [1] * 4, S), concept=0),
        "BI_COARSE": dict(target=GMM([[3, 0], [-3, 0]], [1, 1], S),
                          anchor=GMM([[3, 0], [-3, 0], [0, 3], [0, -3]], [1] * 4, S), concept=0),
        "BI_FINE": dict(target=GMM([[0.3, 0], [-0.3, 0]], [1, 1], S),
                        anchor=GMM([[0.3, 0], [-0.3, 0], [0, 2], [0, -2]], [1] * 4, S), concept=0),
    }


class Correction:
    """Delta(x, k) = sum_j phi_j(k) (c_j + W_j x)."""
    def __init__(self, kind, rng):
        self.kind = kind
        if kind == "LOCAL":
            self.Phi = np.eye(NS)                                   # (NS, J)
        elif kind == "SHARED":
            self.Phi = np.ones((NS, 1))
        elif kind == "SMOOTH":
            cen = np.linspace(0, 1, 6)
            self.Phi = np.exp(-0.5 * ((SIG[:NS, None] - cen[None]) / 0.2) ** 2)
        J = self.Phi.shape[1]
        self.c = 1e-3 * rng.standard_normal((J, 2))                 # tiny random init avoids the eps_n artifact at v=b
        self.W = np.zeros((J, 2, 2))
        self.m = [np.zeros_like(self.c), np.zeros_like(self.W)]
        self.v = [np.zeros_like(self.c), np.zeros_like(self.W)]
        self.t = 0

    def __call__(self, x, k):
        ph = self.Phi[k]
        return ph @ self.c + x @ np.einsum('j,jab->ab', ph, self.W).T

    def grads(self, x, k, G):
        ph = self.Phi[k]
        gc = np.outer(ph, G.sum(0))
        gW = ph[:, None, None] * (G.T @ x)[None]
        return gc, gW

    def adam(self, gc, gW, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        out = []
        for i, (p, g) in enumerate([(self.c, gc), (self.W, gW)]):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t); vh = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)


def student_field(case, corr):
    T = case["target"]
    return lambda x, k, sg: T.v(x, sg) + corr(x, k)


def hinge_grad(v, a, b, eta=1.0):
    dp = np.linalg.norm(v - a, axis=1); dm = np.linalg.norm(v - b, axis=1)
    act = (dp - eta * dm) > 0
    up = (v - a) / np.maximum(dp, 1e-12)[:, None]; um = (v - b) / np.maximum(dm, 1e-12)[:, None]
    G = act[:, None] * (up - eta * um)
    return np.maximum(dp - eta * dm, 0), G, np.linalg.norm(b - a, axis=1)


def states(case, corr, mode, rng, B, idx):
    """returns list of (k_eval, X) pairs. idx = list of timestep indices to evaluate.
    GEM pairing quirk (A facts sec.2): latent saved AFTER step k is evaluated at index k / sigma_k."""
    T = case["target"]
    eps = rng.standard_normal((B, 2))
    if mode == "offpolicy_data":
        x0 = T.sample(B, rng)
        return [(k, (1 - SIG[k]) * x0 + SIG[k] * eps) for k in idx]
    kmax = max(idx) + 1
    field = (lambda x, k, sg: T.v(x, sg)) if mode == "teacher_rollout" else student_field(case, corr)
    traj = euler(field, eps, SIG, 0, kmax)          # traj[j] = state after j steps
    return [(k, traj[k + 1]) for k in idx]


def train(case, kind, mode="onpolicy", window="prefix", refresh=1, B=32, seed=0, weights=None, is_q=None, lr=0.05):
    rng = np.random.default_rng(seed)
    corr = Correction(kind, rng)
    A, T = case["anchor"], case["target"]
    nwin = 11 if window in ("prefix", "uniform11") else NS
    budget, it, cache = BUDGET, 0, None
    while budget > 0:
        if window == "prefix":
            idx = list(range(11))
        elif window == "full":
            idx = list(range(NS))
        elif window == "uniform11":
            idx = sorted(rng.choice(NS, 11, replace=False).tolist())
        if is_q is not None:                 # importance-sampled single timestep with weight p/q (unbiased)
            k = int(rng.choice(11, p=is_q)); idx = [k]
        if cache is None or it % refresh == 0 or window == "uniform11":
            cache = states(case, corr, mode, rng, B, idx)
        gc = np.zeros_like(corr.c); gW = np.zeros_like(corr.W)
        for k, X in cache:
            b = T.v(X, SIG[k]); a = A.v(X, SIG[k]); v = b + corr(X, k)
            _, G, _ = hinge_grad(v, a, b)
            w = 1.0 if weights is None else weights[k]
            if is_q is not None:
                w = (1.0 / 11) / is_q[k] * 11      # p/q with p uniform over 11, rescaled to sum-form
            dc, dW = corr.grads(X, k, G * w / (len(cache) if is_q is None else 1) / B)
            gc += dc; gW += dW
        corr.adam(gc, gW, lr=lr)
        budget -= len(cache) if is_q is None else 11
        it += 1
    return corr, it


def evaluate(case, corr, n=2000, seed=123):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, 2))
    A, T = case["anchor"], case["target"]
    traj = euler(student_field(case, corr), eps, SIG)
    tref = euler(lambda x, k, sg: T.v(x, sg), eps, SIG)
    sc = np.mean(A.nearest(traj[-1]) == case["concept"]); sc0 = np.mean(A.nearest(tref[-1]) == case["concept"])
    def leak(ks):
        r = []
        for k in ks:
            X = traj[k]
            r.append(np.linalg.norm(corr(X, k), axis=1).mean() / max(np.linalg.norm(T.v(X, SIG[k]) - A.v(X, SIG[k]), axis=1).mean(), 1e-12))
        return float(np.mean(r))
    def contrast(ks):
        return float(np.mean([np.linalg.norm(T.v(traj[k], SIG[k]) - A.v(traj[k], SIG[k]), axis=1).mean() for k in ks]))
    drift11 = float(np.linalg.norm(traj[11] - tref[11], axis=1).mean())
    return dict(score_norm=float(sc / sc0), raw=float(sc), teacher_raw=float(sc0),
                rho_early=leak(range(0, 11)), rho_late=leak(range(11, NS)),
                contrast_early=contrast(range(0, 11)), contrast_late=contrast(range(11, NS)), state_drift_after_prefix=drift11)


def commitment_sigma(case, n=2000, seed=7, thr=0.9):
    """largest sigma_k at which mean posterior of the finally chosen target mode >= thr (teacher rollout)."""
    T = case["target"]
    if len(T.w) == 1:
        return None
    rng = np.random.default_rng(seed)
    traj = euler(lambda x, k, sg: T.v(x, sg), rng.standard_normal((n, 2)), SIG)
    fin = T.nearest(traj[-1])
    for k in range(NS):
        r, _ = T.post_and_mean(traj[k], SIG[k])
        if np.mean(r[np.arange(n), fin]) >= thr:
            return float(SIG[k])
    return 0.0


def is_analysis(case):
    """S2: per-timestep gradient statistics after 10 native iterations, SHARED student."""
    rng = np.random.default_rng(3)
    corr = Correction("SHARED", rng)
    A, T = case["anchor"], case["target"]
    for _ in range(10):
        cache = states(case, corr, "onpolicy", rng, 32, list(range(11)))
        gc = np.zeros_like(corr.c); gW = np.zeros_like(corr.W)
        for k, X in cache:
            b = T.v(X, SIG[k]); a = A.v(X, SIG[k]); _, G, _ = hinge_grad(b + corr(X, k), a, b)
            dc, dW = corr.grads(X, k, G / 11 / 32); gc += dc; gW += dW
        corr.adam(gc, gW)
    cache = states(case, corr, "onpolicy", rng, 512, list(range(11)))
    g, L, un, lo = [], [], [], []
    for k, X in cache:
        b = T.v(X, SIG[k]); a = A.v(X, SIG[k]); l, G, Lk = hinge_grad(b + corr(X, k), a, b)
        dc, dW = corr.grads(X, k, G / 512)
        g.append(np.concatenate([dc.ravel(), dW.ravel()])); L.append(Lk.mean())
        un.append(np.linalg.norm(G, axis=1).mean()); lo.append(l.mean())
    g = np.array(g); L = np.array(L); gbar = g.mean(0); p = np.full(11, 1 / 11)
    res = dict(sigma=SIG[:11].tolist(), contrast_L=L.tolist(), output_grad_norm=list(map(float,un)), loss=list(map(float,lo)),
               param_grad_norm=np.linalg.norm(g, axis=1).tolist())
    for name, q in [("uniform", p), ("prop_contrast", L / L.sum()),
                    ("prop_gradnorm", np.linalg.norm(g, axis=1) / np.linalg.norm(g, axis=1).sum())]:
        second = sum(q[k] * np.sum((p[k] / q[k] * g[k]) ** 2) for k in range(11))
        var = second - np.sum(gbar ** 2)
        ew = (q[:, None] * g).sum(0)                       # unweighted expectation under q
        cos = float(ew @ gbar / np.linalg.norm(ew) / np.linalg.norm(gbar))
        res[name] = dict(q=q.tolist(), IS_weighted_trace_var=float(var), rel_var=float(var / np.sum(gbar ** 2)),
                         unweighted_expectation_cos_to_true=cos,
                         unweighted_norm_ratio=float(np.linalg.norm(ew) / np.linalg.norm(gbar)))
    return res, L


def main():
    out = {"sigmas": SIG.tolist()}
    C = cases()
    out["commitment_sigma_thr0.9"] = {n: commitment_sigma(c) for n, c in C.items()}
    res = {}
    configs = [("native_prefix_onpolicy", dict(mode="onpolicy", window="prefix")),
               ("full_window_onpolicy", dict(mode="onpolicy", window="full")),
               ("uniform11_of28_onpolicy", dict(mode="onpolicy", window="uniform11")),
               ("prefix_teacher_rollout", dict(mode="teacher_rollout", window="prefix")),
               ("prefix_offpolicy_noised_data", dict(mode="offpolicy_data", window="prefix")),
               ("prefix_onpolicy_stale_R5", dict(mode="onpolicy", window="prefix", refresh=5))]
    for cname, case in C.items():
        for kind in ["LOCAL", "SMOOTH", "SHARED"]:
            for lab, kw in configs:
                rr = []
                for seed in range(3):
                    corr, it = train(case, kind, seed=seed, **kw)
                    e = evaluate(case, corr); e["iters"] = it; rr.append(e)
                res[f"{cname}|{kind}|{lab}"] = dict(
                    score_norm_mean=float(np.mean([r["score_norm"] for r in rr])),
                    score_norm_seeds=[r["score_norm"] for r in rr],
                    rho_early=float(np.mean([r["rho_early"] for r in rr])),
                    rho_late=float(np.mean([r["rho_late"] for r in rr])), iters=rr[0]["iters"],
                    contrast_early=float(np.mean([r["contrast_early"] for r in rr])),
                    contrast_late=float(np.mean([r["contrast_late"] for r in rr])),
                    state_drift_after_prefix=float(np.mean([r["state_drift_after_prefix"] for r in rr])),
                    teacher_raw=rr[0]["teacher_raw"])
                print(cname, kind, lab, res[f"{cname}|{kind}|{lab}"]["score_norm_mean"],
                      [round(x, 3) for x in res[f"{cname}|{kind}|{lab}"]["score_norm_seeds"]],
                      "rho_e %.3f rho_l %.3f L_e %.3f L_l %.3f drift %.3f it %d" % tuple(res[f"{cname}|{kind}|{lab}"][q] for q in
                      ["rho_early", "rho_late", "contrast_early", "contrast_late", "state_drift_after_prefix", "iters"]), flush=True)
    out["S3"] = res
    foc = {}
    for cname in ["UNI", "BI_FINE", "BI_COARSE"]:
        for kind in ["SMOOTH", "SHARED"]:
            for lr in [0.05, 0.2]:
                for lab, kw in configs[:3]:
                    sc = []
                    for seed in range(10, 18):
                        corr, it = train(C[cname], kind, seed=seed, lr=lr, **kw); sc.append(evaluate(C[cname], corr, n=1000)["score_norm"])
                    foc[f"{cname}|{kind}|lr{lr}|{lab}"] = dict(mean=float(np.mean(sc)), se=float(np.std(sc, ddof=1) / np.sqrt(len(sc))))
                    print("FOCUS", cname, kind, lr, lab, round(np.mean(sc), 3), "+-", round(np.std(sc, ddof=1) / np.sqrt(len(sc)), 3), flush=True)
    out["S3_focus_8seeds"] = foc
    # S2: importance sampling / reweighting
    isr, L = is_analysis(C["UNI"])
    out["S2_IS_UNI_shared"] = isr
    q = L / L.sum()
    tr = {}
    for lab, kw in [("deterministic_uniform", {}), ("deterministic_weights_prop_contrast",
                     dict(weights=np.concatenate([q * 11, np.zeros(NS - 11)]))),
                    ("IS_single_t_prop_contrast_weighted", dict(is_q=q)),
                    ("IS_single_t_uniform", dict(is_q=np.full(11, 1 / 11)))]:
        sc = [evaluate(C["UNI"], train(C["UNI"], "SMOOTH", seed=s, **kw)[0])["score_norm"] for s in range(3)]
        tr[lab] = sc
        print("S2 train", lab, sc, flush=True)
    out["S2_train_UNI_SMOOTH"] = tr
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d_window_toy_output.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ["commitment_sigma_thr0.9", "S2_IS_UNI_shared"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
