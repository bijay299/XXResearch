"""Agent F: Theorem 8 (gate) checks. Own code.
(a) coverage under adaptive, random number of trials with fresh batches (theorem setting);
(b) same with a REUSED finite verification pool (common implementation shortcut) -> violation;
(c) size of c relative to M at GEM's budget; (d) what a clipped-mean certificate says under heavy tails."""
import numpy as np
rng = np.random.default_rng(7)
out = {}
T, R, delta, M, tau2 = 23, 4, 0.05, 1.0, 0.25
q = 40
def run(fresh, n2, runs=400, pool_n=None):
    bad_runs = 0; accepted = 0
    c = M * np.sqrt(np.log(T * R / delta) / (2 * n2))
    for _ in range(runs):
        Z = rng.normal(size=(5000, q)) / np.sqrt(q)          # retained population features
        pool = Z[rng.choice(5000, n2, replace=False)]
        th = np.zeros(q); bad = False
        for t in range(T):
            prop = rng.normal(size=q); prop *= 0.35 / np.linalg.norm(prop)   # random proposal direction
            for j in range(R):                                  # adaptive number of trials (stop on accept)
                trial = th + prop * 0.5 ** j
                B = Z[rng.choice(5000, n2)] if fresh else pool
                rhat = np.mean(np.minimum((B @ trial) ** 2, M))
                if rhat + c <= tau2:
                    rho = np.mean(np.minimum((Z @ trial) ** 2, M)); accepted += 1
                    bad |= rho > tau2; th = trial; break
        bad_runs += bad
    return bad_runs / runs, accepted / runs, c
for n2 in [50, 200]:
    out[f'fresh_n2={n2}: P(any accepted iterate violates), accepts/run, c'] = run(True, n2)
# uncorrected c (no union bound) with reused pool: c computed for a single test
def run_reuse_nounion(n2, runs=400, fresh=False, union=False):
    bad_runs = 0
    c = M * np.sqrt(np.log((T if union else 1) / delta) / (2 * n2))
    for _ in range(runs):
        Z = rng.normal(size=(5000, q)) / np.sqrt(q); pool = Z[rng.choice(5000, n2, replace=False)]
        th = np.zeros(q); bad = False
        # adversarial-but-innocent proposal: directions built from the pool (as VAU builds from R1)
        Npool = np.linalg.svd(pool)[2][min(n2, q):]
        for t in range(T):
            prop = rng.normal(size=q)
            if len(Npool): prop = Npool.T @ (Npool @ prop)
            prop *= 1.0 / np.linalg.norm(prop)
            trial = th + prop
            B = Z[rng.choice(5000, n2)] if fresh else pool
            if np.mean(np.minimum((B @ trial) ** 2, M)) + c <= tau2:
                bad |= np.mean(np.minimum((Z @ trial) ** 2, M)) > tau2; th = trial
        bad_runs += bad
    return bad_runs / runs
out['reused pool n2=30 (<q), proposals built in null(pool), single-test c: P(violation)'] = run_reuse_nounion(30)
out['same, union-bound c (T trials): P(violation)'] = run_reuse_nounion(30, union=True)
out['same construction, FRESH batch per trial, single-test c: P(violation)'] = run_reuse_nounion(30, fresh=True)
out['same construction, FRESH batch per trial, union c: P(violation)'] = run_reuse_nounion(30, fresh=True, union=True)
# (c) c/M at GEM budget
for n2 in [50, 200, 1000, 5000]:
    out[f'c/M at T=23,R=4,delta=.05,n2={n2}'] = np.sqrt(np.log(T * R / delta) / (2 * n2))
# (d) heavy-tailed per-sample drift (lognormal sigma=2): clipped mean vs true mean at M = tau2 / (c/M) margin
for sig in [1.0, 2.0, 3.0]:
    x = rng.lognormal(0, sig, 2_000_000); x /= np.mean(x)          # true mean drift = 1
    for Mrel in [2, 7, 20]:
        out[f'lognormal sigma={sig}: clipped mean at M={Mrel}x true mean'] = np.mean(np.minimum(x, Mrel))
for k, v in out.items(): print(k, '=', v)
