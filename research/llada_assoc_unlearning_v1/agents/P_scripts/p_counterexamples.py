"""Agent P: tiny exact / Monte-Carlo calculations that sharpen counterexamples in P_PROBLEM.md.

C1 absence leak: an attacker who knows the novel-entity reference identifies the forgotten object
   from an unlearned model whose q(o*|s) deviates from the reference (exclusion => 100%).
   Reference ("never-seen entity") per-name object distribution q_nu ~ Dir(alpha*pi) (hallucination spread).
C2 single-fact power: upper percentile of novel-name q(Paris|nu) = minimum detectable leak for ONE subject.
C3 base-rate estimation sample size with Dirichlet-multinomial design effect.
C4 correlated attribute: Bayes posterior (retrain-like reference) vs population prior (novel-entity reference).
C5 prior-mode: greedy-decoding 'leak' of a perfect reference when Paris is the prior mode.
Pure numpy/scipy, seed 0, < 1 min on 1 thread.
"""
import json
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
K = 10
pi = np.array([0.30, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])  # population prior, Paris = index 0
out = {"pi": pi.tolist()}

# ---------- C1 absence-leak identification (Bayes-optimal attacker) ----------
N = 100_000
def log_dirichlet_sample(beta, n):
    # log of Dirichlet draws via log-gamma (avoids underflow for small concentration): G_a = G_{a+1} * U^{1/a}
    g = np.log(rng.gamma(beta + 1.0, size=(n, len(beta)))) + np.log(rng.random((n, len(beta)))) / beta
    return g - np.logaddexp.reduce(g, axis=1, keepdims=True)

def attack_acc(alpha, lam):
    """True object c* ~ pi. Method output q: stranger-like draw p ~ Dir(alpha*pi), then q[c*] *= lam, renormalise
    (lam=0 exclusion; lam=1 exact novel-entity target T1; lam<1 over-suppression; lam>1 residual leak).
    Attacker knows pi, alpha, lam (worst case) and returns argmax_c pi_c f_c(q) (Bayes-optimal)."""
    beta = alpha * pi
    cstar = rng.choice(K, size=N, p=pi)
    lp = log_dirichlet_sample(beta, N)
    if lam == 0.0:
        return 1.0   # q[c*] == 0 exactly while every reference draw has q_c > 0 a.s.
    lq = lp.copy(); lq[np.arange(N), cstar] += np.log(lam)
    lq -= np.logaddexp.reduce(lq, axis=1, keepdims=True)
    q = np.exp(lq)
    from scipy.special import gammaln
    logB = gammaln(beta).sum() - gammaln(beta.sum())
    post = np.empty((N, K))
    for c in range(K):
        qc = q[:, c]
        l1m = np.log1p(-np.minimum(qc, 1 - 1e-16))
        den = np.logaddexp(lq[:, c], np.log(lam) + l1m)          # log(q_c + lam(1-q_c))
        lpc = lq[:, c] - den                                         # log p_c
        l1mpc = np.log(lam) + l1m - den                              # log(1-p_c)
        lpj = lq + (l1mpc - l1m)[:, None]                             # log p_j, j != c
        lpj[:, c] = lpc
        logdir = ((beta - 1) * lpj).sum(1) - logB
        post[:, c] = np.log(pi[c]) + logdir + (K - 1) * np.log(lam) - K * den
    return float((post.argmax(1) == cstar).mean())

c1 = {}
for alpha in (1.0, 5.0, 50.0):
    for lam in (0.0, 0.1, 0.5, 1.0, 2.0, 10.0):
        c1[f"alpha={alpha},lam={lam}"] = attack_acc(alpha, lam)
c1["chance(max pi)"] = float(pi.max())
out["C1_identification_accuracy_bayes"] = c1

# ---------- C2 single-subject minimum detectable leak ----------
c2 = {}
for alpha in (1.0, 5.0, 50.0):
    a, b = alpha * pi[0], alpha * (1 - pi[0])
    c2[f"alpha={alpha}"] = {"q95_of_q(Paris|nu)": float(stats.beta.ppf(0.95, a, b)),
                            "q05": float(stats.beta.ppf(0.05, a, b)),
                            "P(q(Paris|nu) < 1e-3)": float(stats.beta.cdf(1e-3, a, b))}
out["C2_single_subject_reference_spread"] = c2

# ---------- C3 base-rate sample size ----------
c3 = {}
for alpha in (1.0, 5.0, 50.0):
    rho = 1.0 / (1.0 + alpha)          # intra-name correlation of samples (Dirichlet-multinomial)
    for M in (1, 8, 32):
        deff = 1 + (M - 1) * rho
        n_names = pi[0] * (1 - pi[0]) * deff / (0.01 ** 2) / M   # SE(pi_hat)=0.01
        c3[f"alpha={alpha},M={M}"] = int(np.ceil(n_names))
out["C3_novel_names_needed_for_SE_0.01"] = c3

# ---------- C4 correlated attribute ----------
# population: 5% work at 'Louvre'; of those 95% live in Paris; others follow pi renormalised so the overall prior stays ~pi
pL, pPgL = 0.05, 0.95
post = np.full(K, (1 - pPgL) / (K - 1)); post[0] = pPgL
tv = 0.5 * np.abs(post - pi).sum()
out["C4_correlated"] = {"P(Paris|Louvre worker) [retrain/Bayes ref]": pPgL, "P(Paris|stranger) [novel ref]": float(pi[0]),
                        "TV(ref_T0, ref_T1)": float(tv)}

# ---------- C5 prior mode under greedy ----------
out["C5_prior_mode"] = {"greedy Paris rate of perfect stranger-like model (Paris = mode)": 1.0,
                        "T=1 sampling Paris rate": float(pi[0]),
                        "fraction of Dir(5pi) strangers whose argmax is Paris":
                            float((rng.dirichlet(5 * pi, size=N).argmax(1) == 0).mean())}
print(json.dumps(out, indent=1))
json.dump(out, open(__file__.replace(".py", "_out.json"), "w"), indent=1)
