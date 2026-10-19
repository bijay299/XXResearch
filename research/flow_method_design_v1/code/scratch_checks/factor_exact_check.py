"""Coordinator pre-pivot check (development seeds only): factor-wise feature-null projection of a LoRA
step preserves retained outputs EXACTLY (any step size) through a 2-layer tanh MLP; compare with
first-order (map-linear) projection and A-only projection."""
import numpy as np
rng = np.random.default_rng(3)
d0, d1, d2, r, k = 12, 10, 6, 3, 4
W1, W2 = rng.normal(size=(d1, d0)) / 3, rng.normal(size=(d2, d1)) / 3
A1, B1 = rng.normal(size=(r, d0)), rng.normal(size=(d1, r)) * 0.3   # nonzero B (mid-training)
A2, B2 = rng.normal(size=(r, d1)), rng.normal(size=(d2, r)) * 0.3
Xr = rng.normal(size=(d0, k))                                       # retained inputs (probe)
def fwd(A1, B1, A2, B2, X):
    h = np.tanh((W1 + B1 @ A1) @ X); return (W2 + B2 @ A2) @ h, h
def nullproj(M):            # projector onto null of M^T (right action: dX -> dX P makes dX M = 0)
    Q, _ = np.linalg.qr(M); return np.eye(M.shape[0]) - Q @ Q.T
y0, h0 = fwd(A1, B1, A2, B2, Xr)
for scale in [1e-3, 1e-1, 1.0, 10.0]:
    dA1, dB1, dA2, dB2 = [scale * rng.normal(size=s.shape) for s in (A1, B1, A2, B2)]
    # factor-wise exact: dA X = 0 and dB (A X) = 0, per layer, with X the layer's retained input
    fA1, fB1 = dA1 @ nullproj(Xr), dB1 @ nullproj(A1 @ Xr)
    fA2, fB2 = dA2 @ nullproj(h0), dB2 @ nullproj(A2 @ h0)
    yf, _ = fwd(A1 + fA1, B1 + fB1, A2 + fA2, B2 + fB2, Xr)
    # A-only (naive Adam-NSCL-on-A): dA X = 0, dB free
    ya, _ = fwd(A1 + fA1, B1 + dB1, A2 + fA2, B2 + dB2, Xr)
    print(f"scale {scale:6g}: factor-exact drift {np.abs(yf - y0).max():.2e} | A-only drift {np.abs(ya - y0).max():.2e}"
          f" | raw drift {np.abs(fwd(A1+dA1,B1+dB1,A2+dA2,B2+dB2,Xr)[0]-y0).max():.2e}")
# held-out retained input: drift bounded by its component outside span(Xr) -- show it is nonzero
Xh = rng.normal(size=(d0, 3)); yh0, _ = fwd(A1, B1, A2, B2, Xh)
yh, _ = fwd(A1 + fA1, B1 + fB1, A2 + fA2, B2 + fB2, Xh)
print("held-out drift (not protected):", np.abs(yh - yh0).max())
