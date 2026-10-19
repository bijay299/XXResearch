import numpy as np
class AdamW:
    """Flat-vector AdamW, torch semantics (decoupled WD: theta *= 1-lr*wd; bias-corrected; eps outside sqrt)."""
    def __init__(s, P, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        s.lr, s.b1, s.b2, s.eps, s.wd = lr, b1, b2, eps, wd
        s.m = np.zeros(P); s.v = np.zeros(P); s.t = 0
    def peek(s, g, theta):
        """actual step Delta (to ADD to theta) and tentative state, no mutation."""
        t = s.t + 1
        m = s.b1*s.m + (1-s.b1)*g; v = s.b2*s.v + (1-s.b2)*g*g
        mh = m/(1-s.b1**t); vh = v/(1-s.b2**t)
        d = -s.lr*mh/(np.sqrt(vh)+s.eps) - s.lr*s.wd*theta
        return d, (m, v, t)
    def commit(s, st): s.m, s.v, s.t = st
    def step(s, g, theta):
        d, st = s.peek(g, theta); s.commit(st); return d
    def D(s, g):  # diagonal map applied to mhat: 1/(sqrt(vhat)+eps) for tentative state
        t = s.t+1; v = s.b2*s.v+(1-s.b2)*g*g
        return 1.0/(np.sqrt(v/(1-s.b2**t))+s.eps)
