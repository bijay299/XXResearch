"""Coordinator pre-pivot check 2 (development seeds only).
(1) Zero-init B with q >= r: factor-wise exact constraints force dB = 0 -> LoRA output on ANY input stays 0 (no erasure).
(2) Null-A LoRA (rows of A orthogonal to span(U), A frozen, train B freely): exact probed retention, nonzero target change."""
import numpy as np
rng = np.random.default_rng(4)
d_in, d_out, r, q = 12, 8, 3, 5
U, _ = np.linalg.qr(rng.normal(size=(d_in, q)))           # retained feature span
A = rng.normal(size=(r, d_in)); B = np.zeros((d_out, r))   # peft-style: B = 0
x_t = rng.normal(size=d_in)                                 # target input (generic, not in span U)
P_U = np.eye(d_in) - U @ U.T
AU = A @ U; Z, s, _ = np.linalg.svd(AU, full_matrices=False); Z = Z[:, s > 1e-10 * s.max()]
print("rank(A U) =", Z.shape[1], "of r =", r)
for it in range(50):
    dA, dB = rng.normal(size=A.shape), rng.normal(size=B.shape)
    dA = dA @ P_U; dB = dB @ (np.eye(r) - Z @ Z.T)
    A, B = A + 0.1 * dA, B + 0.1 * dB
print("(1) max |B| after 50 constrained steps:", np.abs(B).max(), "| LoRA output on target:", np.abs(B @ A @ x_t).max())
# (2) null-A construction
A2 = rng.normal(size=(r, d_in)) @ P_U; B2 = np.zeros((d_out, r))
for it in range(50):
    B2 = B2 + 0.1 * rng.normal(size=B2.shape)
Xr = U @ rng.normal(size=(q, 20))
print("(2) null-A: retained output change", np.abs(B2 @ A2 @ Xr).max(), "| target change", np.abs(B2 @ A2 @ x_t).max(),
      "| target residual outside span", np.linalg.norm(P_U @ x_t))
