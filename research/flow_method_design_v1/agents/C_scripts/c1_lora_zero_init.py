"""C-E1: LoRA W+BA, zero-init B. Checks analytic grads, grad_A==0 at step 1, AdamW vs SGD movement of A at step 2,
and the bilinear cross term DeltaB DeltaA that first-order gates miss. COMPUTED."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import AdamW
rng = np.random.default_rng(0)
din, dout, r, N = 64, 48, 8, 32
W = rng.normal(size=(dout, din))/np.sqrt(din)
X = rng.normal(size=(N, din)); T = rng.normal(size=(N, dout))*0.5   # "erasure target" (mse toward T)
def loss_grads(A, B):
    Y = X@(W+B@A).T; R = Y-T; L = 0.5*np.sum(R*R)/N
    dY = R/N                    # output-space gradient
    gB = dY.T@(X@A.T); gA = B.T@(dY.T@X)
    return L, gA, gB
A0 = rng.normal(size=(r, din))/np.sqrt(din); B0 = np.zeros((dout, r))
# finite difference check on random nonzero B
B1 = rng.normal(size=(dout, r))*0.1
L, gA, gB = loss_grads(A0, B1); h=1e-6
E = np.zeros_like(A0); E[2,5]=h; fdA = (loss_grads(A0+E,B1)[0]-loss_grads(A0-E,B1)[0])/(2*h)
E = np.zeros_like(B1); E[3,1]=h; fdB = (loss_grads(A0,B1+E)[0]-loss_grads(A0,B1-E)[0])/(2*h)
print("grad check A %.3e vs %.3e ; B %.3e vs %.3e" % (gA[2,5], fdA, gB[3,1], fdB))
L, gA, gB = loss_grads(A0, B0); print("step1 |gA|=%.3e (exactly 0), |gB|=%.3e" % (np.abs(gA).max(), np.abs(gB).max()))
for name in ["adamw", "sgd"]:
  for lr in [1e-3, 1e-4]:
    A, B = A0.copy(), B0.copy(); P = A.size+B.size
    opt = AdamW(P, lr=lr, wd=0.01)
    rec=[]
    for t in range(4):
        L, gA, gB = loss_grads(A, B); g = np.concatenate([gA.ravel(), gB.ravel()])
        th = np.concatenate([A.ravel(), B.ravel()])
        d = opt.step(g, th) if name=="adamw" else -lr*g
        dA = d[:A.size].reshape(A.shape); dB = d[A.size:].reshape(B.shape)
        cross = np.linalg.norm(X@(dB@dA).T)/np.sqrt(N); first = np.linalg.norm(X@(dB@A+B@dA).T)/np.sqrt(N)
        rec.append((t+1, np.abs(dA).max(), np.abs(dB).max(), first, cross))
        A, B = A+dA, B+dB
    print(name, "lr", lr)
    for t,a,b,f,c in rec: print("  step %d max|dA|=%.2e max|dB|=%.2e rms first-order out drift=%.3e  cross(dB dA) drift=%.3e" % (t,a,b,f,c))
# eps regime: gradient of A at step2 scales ~ |B| |g|; if << eps Adam update of A is suppressed
for eps in [1e-8, 1e-6, 1e-4]:
    A, B = A0.copy(), B0.copy(); opt = AdamW(A.size+B.size, lr=1e-3, eps=eps)
    for t in range(2):
        L, gA, gB = loss_grads(A, B); g = np.concatenate([gA.ravel(), gB.ravel()]); d = opt.step(g, np.zeros_like(g))
        A += d[:A.size].reshape(A.shape); B += d[A.size:].reshape(B.shape)
    print("eps %.0e: after step2 max|dA_step2|=%.2e (max|gA|=%.2e)" % (eps, np.abs(d[:A.size]).max(), np.abs(gA).max()))
