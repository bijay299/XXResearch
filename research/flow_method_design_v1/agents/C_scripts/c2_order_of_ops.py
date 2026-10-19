"""C-E2: order of operations under AdamW, linear-in-parameters model (dense J, small P; exact first-order = exact).
Methods: raw, pre-Euclid-projected gradient (GPM-like), post-Euclid projection of actual update (Adam-NSCL-like),
post-H-metric projection of actual update (H=diag(1/D), D = 1/(sqrt(vhat)+eps) from tentative state), plus
stale-J variants. Metrics per run (T steps, beta1=.9): mean per-step first-order retained drift ||J d||,
cumulative erasure progress -g0.sum(d), and progress per Adam-metric norm. COMPUTED."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import AdamW
def make(P, m, dec, rng, stale=0.0):
    c = 10**(rng.uniform(-dec/2, dec/2, P))             # per-parameter sensitivity scale (LoRA factor imbalance / anisotropy)
    J = rng.normal(size=(m, P))*c/np.sqrt(P)             # retained Jacobian: same column scale as gradient scale
    g0 = c*rng.normal(size=P)                            # erasure gradient, same scales
    Js = J + stale*np.linalg.norm(J)/np.sqrt(J.size)*rng.normal(size=J.shape)*c/np.sqrt(P) if stale>0 else J
    return c, J, Js, g0
def proj_euclid(Jc, v):
    return v - Jc.T@np.linalg.solve(Jc@Jc.T, Jc@v)
def proj_H(Jc, v, Dg):   # min ||x-v||_H, H=diag(1/Dg) -> x=v-Dg*J^T (J Dg J^T)^-1 J v
    JD = Jc*Dg; return v - JD.T@np.linalg.solve(JD@Jc.T, Jc@v)
def run(method, c, J, Js, g0, T=60, lr=1e-3, wd=0.0, beta1=0.9, noise=0.3, seed=0, eps=1e-8):
    rng = np.random.default_rng(seed); P = g0.size
    th = np.random.default_rng(99).normal(size=P)*c; opt = AdamW(P, lr=lr, b1=beta1, wd=wd, eps=eps)
    drift=[]; prog=0.0; hn=0.0
    for t in range(T):
        g = g0 + noise*c*rng.normal(size=P)
        if method=="pre_euclid": g = proj_euclid(Js, g)
        if method=="pre_H_prevD": g = proj_H(Js, g, 1.0/(np.sqrt(opt.v/max(1-opt.b2**opt.t,1e-12))+opt.eps) if opt.t>0 else np.ones(P))
        d, st = opt.peek(g, th)
        if method=="post_euclid": d = proj_euclid(Js, d)
        if method=="post_H": d = proj_H(Js, d, opt.D(g))
        opt.commit(st)
        drift.append(np.linalg.norm(J@d)); prog += -g0@d; hn += np.sum(d*d/opt.D(g)); th = th+d
    return np.mean(drift), prog, prog/np.sqrt(hn+1e-300)
if __name__=="__main__":
    P, m = 300, 100
    print("regime: anisotropy decades `dec`, stale-J relative error, WD. Values: mean per-step drift ||J d|| / (raw drift), progress/(raw progress), prog per H-norm / raw")
    for dec in [0, 1, 2, 4]:
      for stale in [0.0, 0.1]:
        for wd in [0.0, 1.0]:
          rows = {}
          for meth in ["raw","pre_euclid","pre_H_prevD","post_euclid","post_H"]:
              acc=[run(meth, *make(P,m,dec,np.random.default_rng(s),stale), seed=s, wd=wd) for s in range(10)]
              rows[meth]=np.mean(acc,axis=0)
          r=rows["raw"]
          print("dec=%d stale=%.1f wd=%.1f | "%(dec,stale,wd) + " | ".join("%s d=%.3f p=%.3f e=%.3f"%(k, v[0]/r[0], v[1]/r[1], v[2]/r[2]) for k,v in rows.items() if k!="raw"))
