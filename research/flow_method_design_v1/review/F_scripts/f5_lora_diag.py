"""Agent F: where does the metric correction put its mass under zero-init LoRA + AdamW, and is the residual
retained change first-order or bilinear/curvature? Uses the f4 toy (own code). lowdim_pop3, Nr=4, lr=1e-3."""
import numpy as np, f4_lora_toy as F
E = F.setup(0, 4, 3)
for method in ['vau', 'euclid']:
    th = np.r_[E['B0'].ravel(), E['A0'].ravel()]; Pn = th.size; m = np.zeros(Pn); v = np.zeros(Pn); nB = F.H * F.r
    print(f'--- {method}: step | ||d1_A||/||d1_B|| | max D_A/median D_B | lin | exact | bilinear part | rest')
    for t in range(1, 9):
        _, g = F.erase_loss_grad(E, th, E['Xe']); S = F.jac(E, th, E['Xr1'])
        m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        D = 1 / (np.sqrt(v / (1 - .999 ** t)) + 1e-8); d0 = -1e-3 * D * m / (1 - .9 ** t) - 1e-5 * th
        if method == 'vau':
            G = S @ (D[:, None] * S.T); d1 = d0 - D * (S.T @ np.linalg.solve(G + 1e-6 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), S @ d0))
        else:
            d1 = d0 - S.T @ np.linalg.lstsq(S @ S.T, S @ d0, rcond=None)[0]
        dB = np.r_[d1[:nB], 0 * d1[nB:]]; dA = np.r_[0 * d1[:nB], d1[nB:]]
        f = lambda x: F.fwd(E, th + x, E['Xr1']) - F.fwd(E, th, E['Xr1'])
        ex = f(d1).ravel(); bil = ex - f(dB).ravel() - f(dA).ravel()
        print(t, f'{np.linalg.norm(d1[nB:]) / max(np.linalg.norm(d1[:nB]), 1e-30):.3g}',
              f'{D[nB:].max() / np.median(D[:nB]):.3g}', f'{np.linalg.norm(S @ d1):.2e}', f'{np.linalg.norm(ex):.2e}',
              f'{np.linalg.norm(bil):.2e}', f'{np.linalg.norm(ex - bil - S @ d1):.2e}')
        th = th + d1
