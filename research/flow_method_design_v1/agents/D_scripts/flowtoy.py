"""Shared closed-form toy for Agent D: GMM teachers for rectified flow x_s = (1-s) x0 + s eps, v = eps - x0.
numpy only. All velocities are exact posterior-mean velocities of an isotropic GMM data law."""
import numpy as np


def flux_sigmas(n=28, mu=0.630):
    """GEM/Flux schedule: linspace(1, 1/n, n) then exponential time shift (diffusers formula, DERIVED in A facts),
    with terminal 0 appended. Returns array of length n+1."""
    s = np.linspace(1.0, 1.0 / n, n)
    s = np.exp(mu) / (np.exp(mu) + (1.0 / s - 1.0))
    return np.concatenate([s, [0.0]])


class GMM:
    def __init__(self, means, weights, s):
        self.m = np.asarray(means, float)          # (K,d)
        self.w = np.asarray(weights, float) / np.sum(weights)
        self.s = float(s)

    def post_and_mean(self, x, sig):
        """x (n,d); returns responsibilities (n,K) and E[x0|x_sig] (n,d)."""
        st2 = (1 - sig) ** 2 * self.s ** 2 + sig ** 2
        mu = (1 - sig) * self.m                          # (K,d)
        d2 = ((x[:, None, :] - mu[None]) ** 2).sum(-1)   # (n,K)
        logp = np.log(self.w)[None] - 0.5 * d2 / st2
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp); r /= r.sum(1, keepdims=True)
        gain = (1 - sig) * self.s ** 2 / st2
        Ek = self.m[None] + gain * (x[:, None, :] - mu[None])   # (n,K,d)
        return r, (r[..., None] * Ek).sum(1)

    def v(self, x, sig):
        if sig <= 0:
            raise ValueError("v undefined at sigma=0 for posterior-mean form; sampler never evaluates it")
        _, E = self.post_and_mean(x, sig)
        return (x - E) / sig

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w)
        return self.m[k] + self.s * rng.standard_normal((n, self.m.shape[1]))

    def nearest(self, x):
        return np.argmin(((x[:, None, :] - self.m[None]) ** 2).sum(-1), 1)


def euler(field, x, sigmas, k0=0, k1=None):
    """Euler sampler x_{k+1} = x_k + (s_{k+1}-s_k) field(x_k, k, s_k). Returns list of states [x_k0, ..., x_k1]."""
    k1 = len(sigmas) - 1 if k1 is None else k1
    out = [x]
    for k in range(k0, k1):
        x = x + (sigmas[k + 1] - sigmas[k]) * field(x, k, sigmas[k])
        out.append(x)
    return out
