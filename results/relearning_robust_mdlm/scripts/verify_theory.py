"""
Machine-check every claim in theory/THEORY.md. Exits 0 only if all pass.

Symbolic checks with sympy where a claim is an identity; numeric checks over many random
instances where it is an inequality. The point is to catch a wrong derivation BEFORE any
GPU time is spent on it -- the previous sprint's founding idea died here at 0:08, which is
the cheapest place for an idea to die.
"""
import sys

import numpy as np
import sympy as sp

RESULTS = []


def claim(name, ok, detail=""):
    RESULTS.append({"claim": name, "passed": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return bool(ok)


# ─────────────────────────────────────────────────────────────────────────────
# L1: L(θ − ηg) = L(θ) − η‖g‖² + O(η²), and the M-smooth lower bound on recovery
# ─────────────────────────────────────────────────────────────────────────────
def check_L1_symbolic():
    """Exact second-order expansion in 1-D, symbolically."""
    t, eta = sp.symbols("t eta", real=True)
    L = sp.Function("L")
    g = sp.diff(L(t), t)
    step = L(t - eta * g)
    ser = sp.series(step, eta, 0, 3).removeO()
    # coefficient of eta^1 must be -g^2
    c1 = sp.simplify(sp.expand(ser).coeff(eta, 1))
    return claim("L1_symbolic_first_order_coeff_is_minus_grad_squared",
                 sp.simplify(c1 + g**2) == 0, f"coeff(eta^1) = {c1}")


def check_L1_numeric(n_inst=300, seed=0):
    """Random smooth quadratics: recovery >= eta*||g||^2*(1 - eta*M/2)."""
    rng = np.random.default_rng(seed)
    worst = np.inf
    for _ in range(n_inst):
        d = int(rng.integers(2, 12))
        A = rng.normal(size=(d, d))
        H = A @ A.T / d                      # PSD Hessian
        b = rng.normal(size=d)
        th = rng.normal(size=d)
        Lf = lambda x: 0.5 * x @ H @ x + b @ x
        gf = lambda x: H @ x + b
        g0 = gf(th)
        M = np.linalg.eigvalsh(H).max()
        if M <= 0:
            continue
        eta = float(rng.uniform(1e-3, 1.9 / M))
        lhs = Lf(th) - Lf(th - eta * g0)
        rhs = eta * (g0 @ g0) * (1 - eta * M / 2)
        worst = min(worst, lhs - rhs)
    return claim("L1_numeric_recovery_lower_bound", worst >= -1e-9,
                 f"min(LHS-RHS) over {n_inst} instances = {worst:.3e}")


def check_L1_value_independence(n_inst=200, seed=1):
    """THE point of L1: equal loss VALUE, wildly different one-step recovery.
    Construct pairs at matched L but different ||g|| and show recovery differs."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_inst):
        d = int(rng.integers(2, 8))
        H = np.eye(d)
        # two points with the same quadratic value 0.5*||x||^2 = c but we rescale
        # the gradient by moving along a flat direction of a degenerate Hessian
        H[0, 0] = 1.0
        H[1:, 1:] = np.eye(d - 1) * 1e-6     # near-flat subspace
        c = float(rng.uniform(0.5, 2.0))
        x_steep = np.zeros(d); x_steep[0] = np.sqrt(2 * c / H[0, 0])
        x_flat = np.zeros(d); x_flat[1] = np.sqrt(2 * c / H[1, 1])
        Lf = lambda x: 0.5 * x @ H @ x
        gf = lambda x: H @ x
        assert abs(Lf(x_steep) - Lf(x_flat)) < 1e-6 * max(1, c)
        eta = 1e-2
        r_steep = Lf(x_steep) - Lf(x_steep - eta * gf(x_steep))
        r_flat = Lf(x_flat) - Lf(x_flat - eta * gf(x_flat))
        if r_flat > 0:
            ratios.append(r_steep / r_flat)
    med = float(np.median(ratios))
    return claim("L1_equal_loss_different_recovery", med > 100,
                 f"median recovery ratio steep/flat = {med:.3e} at matched loss")


# ─────────────────────────────────────────────────────────────────────────────
# P2: L(θ) >= L(θ_u) - ||g0||r - (M/2)r^2 inside a ball, and its inversion P2'
# ─────────────────────────────────────────────────────────────────────────────
def check_P2(n_inst=300, seed=2):
    rng = np.random.default_rng(seed)
    worst = np.inf
    for _ in range(n_inst):
        d = int(rng.integers(2, 10))
        A = rng.normal(size=(d, d))
        H = (A + A.T) / 2                     # symmetric, possibly indefinite
        b = rng.normal(size=d)
        th = rng.normal(size=d)
        Lf = lambda x: 0.5 * x @ H @ x + b @ x
        g0 = H @ th + b
        M = np.abs(np.linalg.eigvalsh(H)).max()
        r = float(rng.uniform(1e-3, 2.0))
        # worst case over random directions at exactly radius r
        for _ in range(20):
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            lhs = Lf(th + r * u)
            rhs = Lf(th) - np.linalg.norm(g0) * r - 0.5 * M * r**2
            worst = min(worst, lhs - rhs)
    return claim("P2_ball_lower_bound", worst >= -1e-9,
                 f"min(L(theta) - bound) = {worst:.3e}")


def check_P2_prime(n_inst=500, seed=3):
    """P2': r >= (-||g|| + sqrt(||g||^2 + 2*M*Delta))/M is exactly the inversion of P2."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_inst):
        gn = float(rng.uniform(0, 5))
        M = float(rng.uniform(1e-3, 5))
        D = float(rng.uniform(1e-6, 5))
        r = (-gn + np.sqrt(gn**2 + 2 * M * D)) / M
        # at this r the bound equals exactly -Delta
        drop = gn * r + 0.5 * M * r**2
        if abs(drop - D) > 1e-8 * max(1.0, D):
            bad += 1
    return claim("P2prime_inversion_exact", bad == 0,
                 f"{bad}/{n_inst} instances off by > 1e-8 relative")


def check_P2_regimes(seed=4):
    """The two regimes: r ~ Delta/||g|| when steep, r ~ sqrt(2 Delta/M) when stationary."""
    M, D = 1.0, 0.1
    r_steep = (-50.0 + np.sqrt(50.0**2 + 2 * M * D)) / M
    r_stat = (-0.0 + np.sqrt(0.0**2 + 2 * M * D)) / M
    ok = (abs(r_steep - D / 50.0) < 1e-3 * (D / 50.0) * 10
          and abs(r_stat - np.sqrt(2 * D / M)) < 1e-12
          and r_stat > 100 * r_steep)
    return claim("P2_two_regimes", ok,
                 f"r_steep={r_steep:.3e} (~D/|g|={D/50:.3e})  "
                 f"r_stationary={r_stat:.3e} (~sqrt(2D/M)={np.sqrt(2*D/M):.3e})  "
                 f"ratio={r_stat/r_steep:.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# P3: the spurious attractor. grad of J vanishes at ANY critical point of L,
#     including the memorized MINIMUM. This is the method's failure mode.
# ─────────────────────────────────────────────────────────────────────────────
def check_P3_symbolic():
    x = sp.symbols("x", real=True)
    lam = sp.symbols("lambda_G", positive=True)
    # a loss with a minimum at x=0 (the "memorized" solution) and a local max at x=2
    L = (x**2) * (x - 3)**2
    J = -L + lam * sp.diff(L, x)**2
    dJ = sp.simplify(sp.diff(J, x))
    crit_L = sp.solve(sp.diff(L, x), x)
    vals = {c: sp.simplify(dJ.subs(x, c)) for c in crit_L}
    all_zero = all(v == 0 for v in vals.values())
    mem_min = sp.nsimplify(0)
    return claim("P3_penalty_vanishes_at_every_critical_point_including_memorized_min",
                 all_zero and vals.get(mem_min, None) == 0,
                 f"critical points of L: {crit_L}; dJ there: "
                 f"{[str(v) for v in vals.values()]}")


def check_P3_drift(seed=5):
    """CORRECTED. The first version of this test asserted that a LARGE lambda_G drifts
    back toward memorization, and checked lambda_G = 50. It FAILED: at 50 the dynamics
    are merely unstable, not drifting. The wrong version is preserved in THEORY.md
    alongside this correction.

    What is actually true, and is what this now checks: the drift is governed by WHICH
    BASIN the optimiser starts in, not by the size of lambda_G.
      * from INSIDE the memorized basin, moderate lambda_G (1..20) converges to the
        memorized minimum -- L -> 0, the fact fully returns;
      * from OUTSIDE it (post-unlearning), every lambda_G keeps L high.
    Hence the method is only safe as a SECOND STAGE after standard unlearning.
    """
    L = lambda x: (x**2) * (x - 3.0)**2
    dL = lambda x: 2 * x * (x - 3.0)**2 + 2 * (x**2) * (x - 3.0)
    d2L = lambda x: 2 * (x - 3.0)**2 + 8 * x * (x - 3.0) + 2 * x**2

    def run(lam, x0, eta=1e-4, n=20000):
        x = x0
        for _ in range(n):
            dJ = -dL(x) + lam * 2 * d2L(x) * dL(x)
            if not np.isfinite(dJ):
                return np.nan, np.nan
            x -= eta * dJ
        return x, L(x)

    inside = {lam: run(lam, 0.45)[1] for lam in (1.0, 5.0, 20.0)}
    outside = {lam: run(lam, 1.40)[1] for lam in (1.0, 5.0, 20.0, 50.0)}
    L_target = L(1.5)                                  # the high-loss stationary point
    returned = all(v < 0.05 for v in inside.values())
    stays = all(abs(v - L_target) < 1e-3 for v in outside.values())
    return claim("P3_drift_is_basin_determined_not_lambda_determined",
                 returned and stays,
                 f"from inside memorized basin L_end={{{', '.join(f'{k:g}:{v:.4f}' for k, v in inside.items())}}} "
                 f"(fact RETURNED); from outside L_end={{{', '.join(f'{k:g}:{v:.4f}' for k, v in outside.items())}}} "
                 f"(target {L_target:.4f}, stays high)")


def check_P3b_ascent_unbounded_in_high_dim(seed=7):
    """Why the first term cannot be raw -L. In 1-D, ascent on L lands on a local max,
    which is automatically stationary, so the penalty looks unnecessary. In high
    dimensions raw ascent is UNBOUNDED -- it runs off to infinity instead of finding a
    critical point. So the method's first term must be an anchored/bounded forget loss
    (as MDU/EMDU/NPO all use), not raw ascent. Checked by divergence of raw ascent."""
    rng = np.random.default_rng(seed)
    d = 40
    A = rng.normal(size=(d, d))
    H = (A + A.T) / 2 / np.sqrt(d)
    w = rng.normal(size=d)
    dLf = lambda x: H @ x + (np.sum((x @ w) ** 3) / d) * w
    x = rng.normal(size=d) * 1e-2
    for _ in range(3000):
        g = dLf(x)
        if not np.isfinite(g).all():
            break
        x = x + 2e-3 * g                               # raw ascent
    diverged = not np.isfinite(x).all() or np.linalg.norm(x) > 1e3
    return claim("P3b_raw_ascent_diverges_in_high_dim", diverged,
                 "||theta|| after 3000 raw-ascent steps = %.3e -> first term must be "
                 "an ANCHORED forget loss, not -L"
                 % (np.linalg.norm(x) if np.isfinite(x).all() else float("inf")))


# ─────────────────────────────────────────────────────────────────────────────
# P4: the finite-difference surrogate approximates ||g||, NOT ||g||^2
# ─────────────────────────────────────────────────────────────────────────────
def check_P4_symbolic():
    rho = sp.symbols("rho", positive=True)
    x1, x2 = sp.symbols("x1 x2", real=True)
    L = sp.Rational(1, 2) * (3 * x1**2 + x2**2) + x1 * x2
    g = sp.Matrix([sp.diff(L, x1), sp.diff(L, x2)])
    gn = sp.sqrt((g.T * g)[0, 0])
    u = g / gn
    Lt = L.subs({x1: x1 + rho * u[0], x2: x2 + rho * u[1]})
    ser = sp.series(Lt, rho, 0, 2).removeO()
    c1 = sp.simplify(sp.expand(ser).coeff(rho, 1) - gn)
    return claim("P4_surrogate_first_order_is_grad_NORM_not_norm_squared",
                 sp.simplify(c1) == 0,
                 "coeff(rho^1) == ||g|| exactly (so the surrogate penalises "
                 "||g||, a different penalty from ||g||^2)")


def check_P4_not_interchangeable(n_inst=200, seed=6):
    """||g|| and ||g||^2 share a zero set but have different gradients -- so the two
    penalties are NOT interchangeable in the update, only in their minimiser."""
    rng = np.random.default_rng(seed)
    maxrel = 0.0
    for _ in range(n_inst):
        d = int(rng.integers(2, 8))
        A = rng.normal(size=(d, d))
        H = (A + A.T) / 2
        b = rng.normal(size=d)
        x = rng.normal(size=d)
        g = H @ x + b
        gn = np.linalg.norm(g)
        if gn < 1e-6:
            continue
        grad_sq = 2 * H @ g                  # d/dx ||g||^2
        grad_norm = (H @ g) / gn             # d/dx ||g||
        # they are parallel but differ in magnitude by 2*||g||; equal only if ||g||=1/2
        ratio = np.linalg.norm(grad_sq) / max(np.linalg.norm(grad_norm), 1e-12)
        maxrel = max(maxrel, abs(ratio - 2 * gn) / max(2 * gn, 1e-12))
    return claim("P4_grad_of_norm_vs_norm_squared_differ_by_factor_2g", maxrel < 1e-6,
                 f"max relative deviation from the factor 2||g|| = {maxrel:.3e}")


if __name__ == "__main__":
    ok = True
    for fn in (check_L1_symbolic, check_L1_numeric, check_L1_value_independence,
               check_P2, check_P2_prime, check_P2_regimes,
               check_P3_symbolic, check_P3_drift, check_P3b_ascent_unbounded_in_high_dim,
               check_P4_symbolic, check_P4_not_interchangeable):
        try:
            ok &= fn()
        except Exception as e:
            ok &= claim(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
    import json
    n_pass = sum(r["passed"] for r in RESULTS)
    json.dump({"n_claims": len(RESULTS), "n_passed": n_pass, "all_passed": bool(ok),
               "results": RESULTS},
              open("/data/bijaypandey/anvil/outputs/theory_verification.json", "w"),
              indent=2)
    print(f"\n{n_pass}/{len(RESULTS)} claims pass")
    sys.exit(0 if ok else 1)
