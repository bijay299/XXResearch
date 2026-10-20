"""Agent O exact/analytic checks (CPU, numpy only, < 2 min, 1 thread).

Check A (tabular, exact): which state distribution makes stranger-swap distillation converge to the intended target
  when the answer has >1 token (masked-diffusion conditionals, two unmasking orders).
Check B (log-linear shared-feature model, exact gradients): twin-differenced forget gradient (operation X) vs the
  plain sum with the same targets, anchors, retain set and forwards (control Z), at matched forgetting.
  Includes the predicted failure cases (object-attribute feature shared by same-object subjects; weak identity).
Output: o_check_out.json next to this file.
"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "o_check_out.json")
PI = np.array([.30, .15, .12, .10, .08, .07, .06, .05, .04, .03])
V, PARIS = 10, 0


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


# ============================================================================ Check A
def check_a():
    """Answer = 2 slots, 3 symbols each (e.g. slot0 in {New, Los, San}, slot1 in {York, Angeles, Jose}).
    Two strangers with different joints. Target T1-mean = trajectory-level mixture M = (P1+P2)/2.
    Arm 'traj': states drawn from each stranger's own samples, soft target = that stranger's conditional
      -> minimiser at state s is E[P_k(.|s) | s] (posterior-weighted).
    Arm 'flat': states drawn independently of k (e.g. from Alice's original answer), target = (P1(.|s)+P2(.|s))/2.
    Arm 'K1': one fixed stranger (T1-sample)."""
    P1 = np.array([[.80, .02, .02], [.02, .05, .01], [.02, .01, .05]])  # mostly (New, York)
    P2 = np.array([[.05, .02, .01], [.02, .80, .02], [.01, .02, .05]])  # mostly (Los, Angeles)
    P1, P2 = P1 / P1.sum(), P2 / P2.sum()
    M = (P1 + P2) / 2

    def conds(J):  # conditionals of a joint at the three masked states
        return {"both": (J.sum(1), J.sum(0)),            # marginals of slot0, slot1 when both masked
                "s0|y1": (J / J.sum(0, keepdims=True)),   # [y0, y1] -> p(y0 | y1)
                "s1|y0": (J / J.sum(1, keepdims=True))}   # [y0, y1] -> p(y1 | y0)

    def gen(c, order):  # joint produced by a sampler that commits one slot then the other (exact)
        G = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                if order == "L2R":
                    G[a, b] = c["both"][0][a] * c["s1|y0"][a, b]
                else:
                    G[a, b] = c["both"][1][b] * c["s0|y1"][a, b]
        return G

    c1, c2, cm = conds(P1), conds(P2), conds(M)
    # traj minimiser = posterior-weighted mixture of conditionals; at 'both' the weights are 1/2 each
    w_y1 = np.stack([P1.sum(0), P2.sum(0)]) / 2  # P(k, y1) up to norm
    w_y0 = np.stack([P1.sum(1), P2.sum(1)]) / 2
    traj = {"both": ((c1["both"][0] + c2["both"][0]) / 2, (c1["both"][1] + c2["both"][1]) / 2),
            "s0|y1": (w_y1[0][None, :] * c1["s0|y1"] + w_y1[1][None, :] * c2["s0|y1"]) / w_y1.sum(0)[None, :],
            "s1|y0": (w_y0[0][:, None] * c1["s1|y0"] + w_y0[1][:, None] * c2["s1|y0"]) / w_y0.sum(0)[:, None]}
    flat = {k: (tuple((a + b) / 2 for a, b in zip(c1[k], c2[k])) if k == "both" else (c1[k] + c2[k]) / 2) for k in c1}
    res = {}
    for name, c in [("traj_states_softtarget", traj), ("flat_states_avgtarget", flat), ("K1_fixed_stranger", c1)]:
        ref = P1 if name.startswith("K1") else M
        for order in ("L2R", "R2L"):
            G = gen(c, order)
            res[f"{name}/{order}/TV_to_target"] = round(float(0.5 * np.abs(G - ref).sum()), 4)
            res[f"{name}/{order}/mass_on_incoherent_offdiag"] = round(float(G.sum() - np.trace(G)), 4)
    res["target_M_offdiag_mass"] = round(float(M.sum() - np.trace(M)), 4)
    return res


# ============================================================================ Check B
class World:
    def __init__(self, seed, s_sh=1.0, s_nm=1.0, s_id=1.0, s_cl=0.0):
        rng = np.random.default_rng(seed)
        self.s_sh, self.s_nm, self.s_id, self.s_cl = s_sh, s_nm, s_id, s_cl
        allp = [(f, l) for f in range(16) for l in range(16)]
        perm = rng.permutation(len(allp))
        self.pairs = [allp[i] for i in perm[:128]]  # 0..63 known, 64..95 N_train, 96..127 N_eval
        while True:
            lives = rng.choice(V, 64, p=PI)
            if (lives == PARIS).sum() >= 14:
                break
        self.lives = lives
        self.born = np.where(rng.random(64) < .3, lives, rng.choice(V, 64, p=PI))
        self.CL = 43 + 128 * 2
        self.D = self.CL + V
        self.rng = rng

    def phi(self, pidx, rel, tmpl):
        f, l = self.pairs[pidx]
        v = np.zeros(self.D)
        v[0] = self.s_sh; v[1 + rel] = self.s_sh; v[3 + tmpl] = self.s_sh
        v[11 + f] = self.s_nm; v[27 + l] = self.s_nm
        v[43 + 2 * pidx + rel] = self.s_id
        if self.s_cl and pidx < 64:
            v[self.CL + self.lives[pidx]] = self.s_cl  # object attribute carried by the subject representation
        return v

    def Phi(self, states):
        return np.stack([self.phi(*s) for s in states])

    def pretrain(self, l2=1e-3):
        st, Y = [], []
        for p in range(64):
            for rel, obj in ((0, self.lives[p]), (1, self.born[p])):
                for k in range(4):
                    st.append((p, rel, 4 * rel + k))
                    y = np.full(V, .1 / V); y[obj] += .9; Y.append(y)
        X, Y = self.Phi(st), np.array(Y)
        D = self.D

        def f(w):
            W = w.reshape(V, D)
            P = softmax(X @ W.T)
            loss = -(Y * np.log(P + 1e-300)).sum(1).mean() + l2 * (W ** 2).sum()
            G = (P - Y).T @ X / len(X) + 2 * l2 * W
            return loss, G.ravel()

        r = minimize(f, np.zeros(V * D), jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-9})
        self.W0 = r.x.reshape(V, D)
        return float(r.fun)


def build(w, rng):
    paris = [p for p in range(64) if w.lives[p] == PARIS]
    other = [p for p in range(64) if w.lives[p] != PARIS]
    rng.shuffle(paris); rng.shuffle(other)
    forget, bobs, r1_train_extra, r1_ho = paris[:4], paris[4:8], paris[8:10], paris[10:]
    r5_train, r5_ho = other[:16], other[16:]
    nus = list(range(64, 68))
    return dict(forget=forget, bobs=bobs, r1_train=bobs + r1_train_extra, r1_ho=r1_ho, r5_train=r5_train,
                r5_ho=r5_ho, nus=nus, n_eval=list(range(96, 128)))


def run_arm(w, S, arm, opt="adam", steps=3000, lr=None, delta=0.05, lam_anchor=1.0, lam_ret=1.0, eval_every=10,
            target="T1"):
    W0 = w.W0
    W = W0.copy()
    P0 = lambda st: softmax(w.Phi(st) @ W0.T)
    vis, ho = (0, 1), (2, 3)
    fst = [(a, 0, t) for a in S["forget"] for t in vis]
    nu_st = [(n, 0, t) for n in S["nus"] for t in vis]
    bob_st = [(b, 0, t) for b in S["bobs"] for t in vis]
    if target == "T1":
        tF = P0(nu_st)  # T1-sample: p0 on the stranger twin at the same state
    else:  # exclusion target (TNFF / UNDIAL gamma->inf): p0(Alice) with Paris -> 1e-3, renormalised
        q = P0(fst).copy(); q[:, PARIS] = 0; q = q / q.sum(1, keepdims=True) * (1 - 1e-3); q[:, PARIS] = 1e-3; tF = q
    Xf, Xn, Xb = w.Phi(fst), w.Phi(nu_st), w.Phi(bob_st)
    # control for the differenced gradient (same template, same state, subject swapped)
    if arm in ("Z", "Zexcl"):
        Xc = np.zeros_like(Xf)
    elif arm == "X_nu":
        Xc = Xn
    elif arm == "X_bob":
        Xc = Xb
    elif arm == "X_mix":
        Xc = 0.5 * (Xn + Xb)
    Xd = Xf - Xc
    anc_st = nu_st + bob_st
    Xa, tA = w.Phi(anc_st), P0(anc_st)
    ret_st = [(p, 0, t) for p in S["r1_train"] + S["r5_train"] for t in vis] + \
             [(p, 1, t) for p in S["r1_train"] + S["r5_train"] + S["forget"] for t in (4, 5)]
    Xr, tR = w.Phi(ret_st), P0(ret_st)
    # evaluation sets
    ev = {"F_vis": fst, "F_ho": [(a, 0, t) for a in S["forget"] for t in ho],
          "Bob_ho": [(p, 0, t) for p in S["r1_ho"] for t in range(4)],
          "Other_ho": [(p, 0, t) for p in S["r5_ho"] for t in range(4)],
          "R3_ho": [(a, 1, t) for a in S["forget"] for t in (6, 7)],
          "Nu_eval": [(n, 0, t) for n in S["n_eval"] for t in range(4)],
          "Nu_twin": [(n, 0, t) for n in S["nus"] for t in range(4)]}
    evX = {k: w.Phi(v) for k, v in ev.items()}
    ev0 = {k: softmax(X @ W0.T) for k, X in evX.items()}
    tFho = P0([(n, 0, t) for n in S["nus"] for t in ho]) if target == "T1" else None
    shared_cols = np.r_[0:11, w.CL:w.D]
    if lr is None:
        lr = 0.02 if opt == "adam" else 2.0
    m = np.zeros_like(W); v = np.zeros_like(W); b1, b2 = .9, .999
    hit = None; hist = []

    def evaluate(W, step):
        r = {"step": step}
        P = {k: softmax(X @ W.T) for k, X in evX.items()}
        r["F_vis_TV"] = float(tv(P["F_vis"], tF).mean())
        r["F_ho_TV"] = float(tv(P["F_ho"], tFho).mean()) if tFho is not None else float("nan")
        r["F_pParis"] = float(P["F_vis"][:, PARIS].mean())
        for k in ("Bob_ho", "Other_ho", "R3_ho", "Nu_eval", "Nu_twin"):
            r[k + "_TV"] = float(tv(P[k], ev0[k]).mean())
        r["Bob_ho_dlogParis"] = float((np.log(P["Bob_ho"][:, PARIS]) - np.log(ev0["Bob_ho"][:, PARIS])).mean())
        r["Nu_eval_dParis"] = float((P["Nu_eval"][:, PARIS] - ev0["Nu_eval"][:, PARIS]).mean())
        r["G_SO"] = r["Bob_ho_TV"] - r["Other_ho_TV"]
        dz = (W - W0)[PARIS]
        tot = evX["Bob_ho"] @ dz
        sh = evX["Bob_ho"][:, shared_cols] @ dz[shared_cols]
        r["Bob_dzParis_total"] = float(tot.mean()); r["Bob_dzParis_sharedcols"] = float(sh.mean())
        return r

    for step in range(steps + 1):
        if step % eval_every == 0:
            r = evaluate(W, step)
            hist.append(r)
            if hit is None and r["F_vis_TV"] <= delta:
                hit = r
        PF = softmax(Xf @ W.T); eF = (PF - tF) / len(Xf)
        PA = softmax(Xa @ W.T); eA = (PA - tA) / len(Xa)
        PR = softmax(Xr @ W.T); eR = (PR - tR) / len(Xr)
        G = eF.T @ Xd + lam_anchor * eA.T @ Xa + lam_ret * eR.T @ Xr
        if opt == "adam":
            m = b1 * m + (1 - b1) * G; v = b2 * v + (1 - b2) * G * G
            mh = m / (1 - b1 ** (step + 1)); vh = v / (1 - b2 ** (step + 1))
            W = W - lr * mh / (np.sqrt(vh) + 1e-8)
        else:
            W = W - lr * G
    return {"matched": hit, "final": hist[-1], "steps_to_match": None if hit is None else hit["step"]}


KEYS = ["steps_to_match", "F_ho_TV", "F_pParis", "Bob_ho_TV", "Bob_ho_dlogParis", "G_SO", "Other_ho_TV", "R3_ho_TV",
        "Nu_eval_TV", "Nu_eval_dParis", "Nu_twin_TV", "Bob_dzParis_total", "Bob_dzParis_sharedcols"]


def summarise(runs):
    out = {}
    for k in KEYS:
        vals = []
        for r in runs:
            if r["matched"] is None:
                continue
            vals.append(r["steps_to_match"] if k == "steps_to_match" else r["matched"][k])
        out[k] = None if not vals else round(float(np.mean(vals)), 4)
    out["n_matched"] = sum(r["matched"] is not None for r in runs)
    out["final_F_vis_TV"] = round(float(np.mean([r["final"]["F_vis_TV"] for r in runs])), 4)
    out["final_Bob_ho_TV"] = round(float(np.mean([r["final"]["Bob_ho_TV"] for r in runs])), 4)
    out["final_Nu_eval_TV"] = round(float(np.mean([r["final"]["Nu_eval_TV"] for r in runs])), 4)
    return out


def check_b(seeds=(0, 1, 2)):
    worlds = {"a_identity_separable": dict(),
              "b_object_attribute_in_subject_rep": dict(s_cl=1.0),
              "c_weak_identity": dict(s_nm=0.3, s_id=0.3)}
    arms = ["Z", "X_nu", "X_bob", "X_mix", "Zexcl"]
    res = {}
    for wname, kw in worlds.items():
        for opt in ("adam", "gd"):
            per = {a: [] for a in arms}
            base = []
            for sd in seeds:
                w = World(sd, **kw); w.pretrain()
                S = build(w, np.random.default_rng(100 + sd))
                P0 = softmax(w.Phi([(a, 0, 0) for a in S["forget"]]) @ w.W0.T)
                base.append(float(P0[:, PARIS].mean()))
                for a in arms:
                    per[a].append(run_arm(w, S, a, opt=opt, target="excl" if a == "Zexcl" else "T1"))
            res[f"{wname}/{opt}"] = {a: summarise(per[a]) for a in arms}
            res[f"{wname}/{opt}"]["p0_Alice_pParis"] = round(float(np.mean(base)), 4)
            print(wname, opt, json.dumps({a: (res[f'{wname}/{opt}'][a]['Bob_ho_TV'],
                                              res[f'{wname}/{opt}'][a]['Nu_eval_TV'],
                                              res[f'{wname}/{opt}'][a]['steps_to_match']) for a in arms}),
                  flush=True)
    return res


if __name__ == "__main__":
    out = {"check_A": check_a()}
    print(json.dumps(out["check_A"], indent=1), flush=True)
    out["check_B"] = check_b()
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)
