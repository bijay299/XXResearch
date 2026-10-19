"""Agent F: how often is Euclidean post-projection of an Adam step sign-reversed (Prop 3), as a function of
k/P, anisotropy and the D-g coupling; and how progress compares at matched Euclidean / matched H step norm.
D_mode 'indep': log-uniform D independent of g (coordinator's sign_check); 'adam1': D = 1/|g| (AdamW step 1,
mhat/sqrt(vhat)=sign g); 'adamv': D = 1/sqrt(v) with v an EMA of past squared gradients of a drifting scale."""
import numpy as np
rng = np.random.default_rng(3)
def pe(S, D, g):
    x = D * g; xE = x - S.T @ np.linalg.solve(S @ S.T, S @ x)
    xH = x - D * (S.T @ np.linalg.solve(S @ (D[:, None] * S.T), S @ x))
    return xE, xH
rows = []
for P in [6, 50, 400]:
    for frac in [0.1, 0.33, 0.66, 0.9]:
        k = max(1, int(round(frac * P)))
        for dec in [1, 3]:
            for mode in ['indep', 'adam1', 'adamv']:
                N = 3000 if P <= 50 else 300
                negE = negH = 0; rE = []; rH_E = []; rH_H = []
                for _ in range(N):
                    S = rng.normal(size=(k, P))
                    scale = np.exp(rng.uniform(-dec, dec, P) * np.log(10) / 2)
                    g = rng.normal(size=P) * scale
                    if mode == 'indep': D = np.exp(rng.uniform(-dec, dec, P) * np.log(10) / 2)
                    elif mode == 'adam1': D = 1 / (np.abs(g) + 1e-8)
                    else:
                        v = (g * np.exp(rng.normal(size=P))) ** 2; D = 1 / (np.sqrt(v) + 1e-8)
                    xE, xH = pe(S, D, g)
                    p_E, p_H = g @ xE, g @ xH
                    negE += p_E < 0; negH += p_H < -1e-10
                    # progress per unit Euclidean norm and per unit H-norm (H = D^-1)
                    nE, nH = np.linalg.norm(xE), np.linalg.norm(xH)
                    hE, hH = np.sqrt(xE @ (xE / D)), np.sqrt(xH @ (xH / D))
                    rH_E.append((p_H / nH) / max(p_E / nE, 1e-300) if p_E > 0 else np.nan)
                    rH_H.append((p_H / hH) / max(p_E / hE, 1e-300) if p_E > 0 else np.nan)
                    # best possible per Euclidean norm: projected gradient Pi g
                    pg = g - S.T @ np.linalg.solve(S @ S.T, S @ g)
                    rE.append((p_H / nH) / np.linalg.norm(pg))
                rows.append((P, k, dec, mode, negE / N, negH / N, np.nanmedian(rH_E), np.nanmedian(rH_H),
                             np.nanmean(np.array(rH_E) < 1), np.median(rE)))
print('P k decades Dmode | fracNeg_Euclid fracNeg_metric | median[metric/Euclid progress at equal Euclid norm] '
      '| median[same at equal H norm] | P(metric worse at equal Euclid norm | Euclid>0) | median[metric / projected-SGD(Pi g) at equal Euclid norm]')
for r in rows:
    print(f'{r[0]:4d} {r[1]:4d} {r[2]} {r[3]:6s} | {r[4]:.4f} {r[5]:.4f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]:.3f}')
