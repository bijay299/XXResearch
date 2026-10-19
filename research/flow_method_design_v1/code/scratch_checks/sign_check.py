"""Coordinator pre-check (development, not validation): can the Euclidean post-projection of a
preconditioned step have negative first-order progress, while the D-metric projection cannot?"""
import numpy as np
rng = np.random.default_rng(12345)
neg_E = neg_H = 0; worst_E = 0.0; trials = 20000
for _ in range(trials):
    P, m = 6, 2
    J = rng.normal(size=(m, P)); g = rng.normal(size=P)
    D = np.exp(rng.uniform(-3, 3, size=P))           # 0..~2.6 decades each way
    Pi = np.eye(P) - np.linalg.pinv(J) @ J
    stepE = -Pi @ (D * g)
    DJ = D[:, None] * J.T
    stepH = -(D * g - DJ @ np.linalg.solve(J @ DJ, J @ (D * g)))
    pE, pH = -g @ stepE, -g @ stepH
    neg_E += pE < -1e-12; neg_H += pH < -1e-12
    worst_E = min(worst_E, pE / (g @ (D * g)))
print(dict(trials=trials, frac_negative_euclid=neg_E / trials, frac_negative_H=neg_H / trials, worst_euclid_rel=worst_E))
# explicit 2-D counterexample
J = np.array([[1.0, 1.0]]); g = np.array([1.0, -0.5]); D = np.array([0.01, 1.0])
Pi = np.eye(2) - np.linalg.pinv(J) @ J
print("2D: euclid progress", -g @ (-Pi @ (D * g)), "H progress",
      -g @ (-(D * g - (D[:, None] * J.T) @ np.linalg.solve(J @ (D[:, None] * J.T), J @ (D * g)))))
