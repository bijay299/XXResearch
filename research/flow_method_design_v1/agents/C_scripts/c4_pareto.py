"""C-E4: matched-drift progress. Linear-in-parameters toy. Compare strongest simple alternative (retention penalty
lam/2||J(th-th0)||^2 + AdamW, lam swept) with pre/post projections, with and without stale J, and with verified-step
gate (accept/scale actual step using exact fresh JVP; rollback state). Progress at cumulative drift ||J(th-th0)|| <= tau. COMPUTED."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import AdamW
from c2_order_of_ops import make, proj_euclid, proj_H
def traj(method, c, J, Js, g0, T=80, lr=1e-3, lam=0.0, tau=None, noise=0.3, seed=0, beta1=0.9):
    rng=np.random.default_rng(seed); P=g0.size; th=np.zeros(P); opt=AdamW(P,lr=lr,b1=beta1)
    for t in range(T):
        g=g0+noise*c*rng.normal(size=P)
        if method=="penalty": g=g+lam*(J.T@(J@th))
        if method=="pre_euclid": g=proj_euclid(Js,g)
        d,st=opt.peek(g,th)
        if method in("post_euclid","post_euclid_gate"): d=proj_euclid(Js,d)
        if method in("post_H","post_H_gate"): d=proj_H(Js,d,opt.D(g))
        if method.endswith("_gate"):      # verified step: exact cumulative drift of th+d must be <= tau, else scale (linear => closed form) and rollback state
            cur=J@th; a=J@d; # |cur + s a| <= tau, s in [0,1]; solve quadratic
            if np.linalg.norm(cur+a)>tau:
                qa=a@a; qb=2*cur@a; qc=cur@cur-tau**2
                s=0.0 if qa<1e-300 or qb*qb-4*qa*qc<0 else max(0.0,min(1.0,(-qb+np.sqrt(qb*qb-4*qa*qc))/(2*qa)))
                d=s*d
        opt.commit(st); th=th+d
    return np.linalg.norm(J@th), -g0@th
rows=[]
for dec in [1,3]:
  for stale in [0.0,0.1,0.5,1.0]:
    res={}
    S=8
    for meth in ["penalty","pre_euclid","post_euclid","post_H","post_euclid_gate","post_H_gate"]:
        out=[]
        for s in range(S):
            c,J,Js,g0=make(300,100,dec,np.random.default_rng(s),stale)
            raw=traj("raw",c,J,Js,g0,seed=s)
            tau=0.05*raw[0]
            if meth=="penalty":
                best=0.0   # max progress with drift<=tau over lam grid
                sc=np.linalg.norm(g0)/np.linalg.norm(J,2)**2
                for lam in sc*10**np.linspace(-3,6,28):
                    dr,pg=traj("penalty",c,J,Js,g0,lam=lam,seed=s)
                    if dr<=tau: best=max(best,pg)
                out.append((best/raw[1], 0.0))
            else:
                dr,pg=traj(meth,c,J,Js,g0,tau=tau,seed=s); out.append((pg/raw[1], dr/tau))
        res[meth]=np.mean(out,axis=0)
    print("dec=%d stale=%.1f | tau=5%% of raw cumulative drift | progress/raw, drift/tau: "%(dec,stale)+" | ".join("%s %.3f (%.2f)"%(k,v[0],v[1]) for k,v in res.items()))
