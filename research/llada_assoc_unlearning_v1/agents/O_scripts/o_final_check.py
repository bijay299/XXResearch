"""O final check (<30 s): Bob logit change per unit Alice forgetting, linear model z = W phi (exact NTK, any loss),
X (differenced, lambda_T=0, fixed or rotating control) vs Z (plain, +/- twin anchor), GD vs Adam (counterexample)."""
import json, os, numpy as np
rng = np.random.default_rng(0); n, V, P = 64, 10, 0
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))  # exactly orthonormal directions
def run(arm, opt, m, Kpool=16, lam_T=1.0, c=3.0, i=1.0, lr=None, steps=4000):
    e_c, e_m = Q[0] * np.sqrt(c), Q[1] * np.sqrt(m)
    idv = lambda k: Q[2 + k] * np.sqrt(i)
    A = e_c + e_m + idv(0); B = e_c + e_m + idv(1); Nev = e_c + idv(2)
    nus = [e_c + idv(3 + k) for k in range(Kpool)]
    W0 = np.zeros((V, n)); W0[P] = 2.0 * (e_c + e_m) / max(c + m, 1e-9) + 2.0 * idv(0) / i  # Paris-heavy for Alice/Bob
    W0[1:] = 0.3 * rng.standard_normal((V - 1, n)) / np.sqrt(n)
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    t = sm(W0 @ nus[0]) if True else None  # T1-sample target: p0 on a fixed stranger
    W = W0.copy(); mo = np.zeros_like(W); vo = np.zeros_like(W)
    lr = lr or (0.05 if opt == "gd" else 0.01)
    for s in range(steps):
        pA = sm(W @ A)
        if 0.5 * np.abs(pA - t).sum() <= 0.05: break
        nu = nus[s % Kpool] if arm.endswith("rot") else nus[0]
        eA = pA - t
        if arm.startswith("X"): G = np.outer(eA, A - nu)
        else: G = np.outer(eA, A)
        if lam_T and "anc" in arm: G += lam_T * np.outer(sm(W @ nu) - sm(W0 @ nu), nu)
        if opt == "gd": W -= lr * G
        else:
            mo = .9 * mo + .1 * G; vo = .999 * vo + .001 * G * G
            W -= lr * (mo / (1 - .9 ** (s + 1))) / (np.sqrt(vo / (1 - .999 ** (s + 1))) + 1e-8)
    dz = lambda x: float(((W - W0) @ x)[P])
    return dict(steps=s, dzA=round(dz(A), 3), ratio_Bob=round(dz(B) / dz(A), 4), ratio_evalStranger=round(dz(Nev) / dz(A), 4),
                ratio_twin0=round(dz(nus[0]) / dz(A), 4))
out = {}
for m in (0.0, 0.5, 2.0):
    for opt in ("gd", "adam"):
        for arm in ("Z", "Z_anc", "Z_anc_rot", "X", "X_rot", "X_anc"):
            out[f"m={m}/{opt}/{arm}"] = run(arm, opt, m)
# predicted (GD, symmetric kernel): Z first-step K_BA/K_AA ; X (lam_T=0) (K_BA-K_Bnu)/(K_AA-K_Anu) ; X_anc/Z_anc endpoint alpha
c, i = 3.0, 1.0
for m in (0.0, 0.5, 2.0):
    KAA, KBA, KAn, KBn, Knn = c + m + i, c + m, c, c, c + i
    out[f"pred m={m}"] = dict(Z_first=round(KBA / KAA, 4), X=round((KBA - KBn) / (KAA - KAn), 4),
                              anchored_endpoint=round((KBA * Knn - KBn * KAn) / (KAA * Knn - KAn ** 2), 4))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "o_final_check_out.json"), "w"), indent=1)
for k, v in out.items(): print(k, v)
