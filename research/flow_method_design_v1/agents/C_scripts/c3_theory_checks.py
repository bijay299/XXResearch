"""C-E3: numerical checks of the DERIVED statements in C_analysis.md (T1..T4) and counterexamples. COMPUTED."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import AdamW
from c2_order_of_ops import proj_euclid, proj_H
rng = np.random.default_rng(1)
# ---- T1: beta1=0 pre-projection drift identity  J D P g = J (D - dI) P g  for any scalar d; bound lr*||J||*(spread/2)*||Pg||
P, m = 200, 60
J = rng.normal(size=(m,P)); g = rng.normal(size=P)*10**rng.uniform(-1,1,P)
opt = AdamW(P, lr=1e-3, b1=0.0, b2=0.9, eps=1e-8); opt.v = (10**rng.uniform(-1,1,P))**2*0.5; opt.t=50
gp = proj_euclid(J, g); d,_ = opt.peek(gp, np.zeros(P)); D = opt.D(gp)
bc = 1-opt.b2**(opt.t+1); drift = np.linalg.norm(J@d)
dbar = (D.max()+D.min())/2
lhs = 1e-3*np.linalg.norm(J@((D-dbar)*gp)); ub = 1e-3*np.linalg.norm(J,2)*(D.max()-D.min())/2*np.linalg.norm(gp)
print("T1 identity: drift=%.6e  lr||J(D-dbar)Pg||=%.6e  bound=%.3e  (drift<=bound: %s)"%(drift,lhs,ub,drift<=ub+1e-12))
print("   uniform D (spread 0): drift = %.2e"%np.linalg.norm(J@(2.0*gp)) )
# residual-scale invariance (sign amplification): gp + rho*h with h in range(J); step 1 of Adam (v=0,t=0): update = -lr*sign
h = rng.normal(size=P); h = h - proj_euclid(J,h)   # null comp; use range comp instead
hr = J.T@rng.normal(size=m); hr/=np.linalg.norm(hr)
for rho in [1e-2,1e-4,1e-6,1e-8]:
    o = AdamW(P, lr=1e-3); gg = gp/np.linalg.norm(gp)*1.0 + rho*hr
    dd = o.step(gg, np.zeros(P)); print("   residual rho=%.0e : drift ||J step||=%.3e (pre-proj step 1 = lr*sign(Pg+rho*h): O(lr) drift independent of rho; this is T1 non-neutrality, not a residual effect)"%(rho,np.linalg.norm(J@dd)))
# coordinate-subspace exception: J supported on first 20 coords -> null(J) = coordinate subspace -> pre-projection IS neutral
J2 = np.zeros((20,P)); J2[:,:20]=rng.normal(size=(20,20)); gp2 = g - np.linalg.pinv(J2)@(J2@g)   # row space = coords 0..19 -> null(J2)=coordinate subspace
o = AdamW(P, lr=1e-3); o.v=(10**rng.uniform(-1,1,P))**2; o.t=20; o.b1=0.0
d2 = o.step(gp2, np.zeros(P)); print("T1 exception (null(J) coordinate-aligned): drift=%.2e"%np.linalg.norm(J2@d2))
# ---- T2: post-H correction exact for tentative state incl momentum + WD; progress positivity condition
th = rng.normal(size=P); o = AdamW(P, lr=1e-3, wd=0.1)
for _ in range(30): o.step(rng.normal(size=P)*10**rng.uniform(-1,1,P), th)
gt = rng.normal(size=P)*10**rng.uniform(-1,1,P); dd,_ = o.peek(gt, th); Dg = o.D(gt)
dc = proj_H(J, dd, Dg); print("T2 post-H: ||J d_corr||=%.2e (raw %.2e); H-norm ratio %.3f (<=1)"%(np.linalg.norm(J@dc),np.linalg.norm(J@dd),np.sqrt(np.sum(dc*dc/Dg)/np.sum(dd*dd/Dg))))
# ---- T3 feasibility: LoRA layer params r(din+dout) vs retained output dims
for (din,dout,r) in [(3072,3072,16),(3072,12288,16)]:
    Pp = r*(din+dout); print("T3 feasibility: LoRA P=%d params/layer; exact null space of stacked retained Jacobian nonempty only if rank(J_r)<P; rank<=N_r*d_out (e.g. N_r=64,d_out=4096 tokens*chans -> %d)"%(Pp,64*4096))
# generic rank experiment: J_r rows = N*dout, P params -> null dim = max(P - rank,0)
for N in [4,16,64]:
    dout=8; Pp=120; Jr = rng.normal(size=(N*dout,Pp)); print("   N=%d,dout=%d,P=%d: rank=%d null dim=%d"%(N,dout,Pp,np.linalg.matrix_rank(Jr),Pp-np.linalg.matrix_rank(Jr)))
# ---- T4 fresh-batch verification vs batch-constructed projection; population with decaying spectrum
Pp, dout, Npop = 120, 4, 20000
spec = 1.0/(1+np.arange(Pp))**1.0
V = np.linalg.qr(rng.normal(size=(Pp,Pp)))[0]
Z = rng.normal(size=(Npop, dout, Pp))*spec; Jpop = Z@V.T        # per-sample Jacobians (dout x Pp)
def drift2(Dl, idx): return np.mean(np.sum((Jpop[idx]@Dl)**2, axis=1))
g = rng.normal(size=Pp)
n1, n2 = 20, 200
tr = {}
trials=300; gap=[]; cover=0; cert_over=[]; M_est=None
idx_pop = np.arange(Npop)
for t in range(trials):
    b1 = rng.choice(Npop, n1, replace=False); b2 = rng.choice(Npop, n2, replace=False)
    Jb = Jpop[b1].reshape(-1,Pp)                 # n1*dout x Pp  (80 rows < Pp=120)
    step = g - np.linalg.pinv(Jb)@(Jb@g); step/=np.linalg.norm(step)*1.0
    dpop = drift2(step, idx_pop); dtr = drift2(step, b1); d2b = drift2(step, b2)
    a2 = np.sum((Jpop[b2]@step)**2,axis=1); Mb = np.max(np.sum((Jpop@step)**2,axis=1))   # ORACLE per-sample bound M (unknown in practice)
    ub = a2.mean()+Mb*np.sqrt(np.log(20)/(2*n2))
    cover += dpop<=ub; gap.append((dpop,dtr,d2b))
gap=np.array(gap); Mtrue = np.max(np.sum((Jpop@step)**2,axis=1))
print("T4 pop drift^2 / train-batch drift^2 (train drift ~0 by construction): mean pop=%.3e train=%.3e fresh-batch est=%.3e"%tuple(gap.mean(0)))
print("   Hoeffding cert (delta=0.05, ORACLE M; last-step max per-sample a^2=%.4f; slack/mean = %.2f): coverage %.3f (>=0.95 required); median cert/pop = %.1f"%(Mtrue, Mtrue*np.sqrt(np.log(20)/(2*n2))/gap[:,0].mean(), cover/trials, 1.0))
# tightness: empirical-fresh estimate without slack
print("   fresh estimate unbiased: mean est/pop = %.3f"%(gap[:,2].mean()/gap[:,0].mean()))
# rare subgroup counterexample: mass p carries drift huge direction; missing prob (1-p)^n2
p=0.005; n2=200; print("   rare-subgroup counterexample: mass p=%.3f, n2=%d -> P(no member in fresh batch)=%.3f ; certificate silent on it"%(p,n2,(1-p)**n2))
# ---- nonlinear / bilinear counterexample to first-order gate
th0 = np.array([0.0, 0.0]); f = lambda t: t[0]*t[1]; Jf = lambda t: np.array([t[1], t[0]])
dl = np.array([0.3, 0.3]); print("C-X1 f=th1*th2 at origin: J d = %.2e but f(th+d)-f(th)=%.3f (curvature; first-order projection/gate is blind)"%(Jf(th0)@dl, f(th0+dl)-f(th0)))
# LoRA gauge-direction counterexample: dB = B K, dA = -K A  => B dA + dB A = 0 exactly (first-order retention-neutral for ALL inputs)
din,dout,r=40,30,4; A=rng.normal(size=(r,din))/6; B=rng.normal(size=(dout,r))*0.3; X=rng.normal(size=(50,din))
R=rng.normal(size=(r,r)); R/=np.linalg.norm(R,2)
for sc in [0.01,0.03,0.1,0.3]:
    K=sc*R; dB=B@K; dA=-K@A
    fo=np.linalg.norm((dB@A+B@dA)@X.T); ex=np.linalg.norm(((B+dB)@(A+dA)-B@A)@X.T); pred=np.linalg.norm(B@K@K@A@X.T)
    print("C-X2 LoRA gauge step |K|=%.2f: first-order drift %.1e, exact drift %.3e (= |B K^2 A X| = %.3e), rel to |BAX|=%.3f"%(sc,fo,ex,pred,ex/np.linalg.norm(B@A@X.T)))
