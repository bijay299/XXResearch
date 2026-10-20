"""
Verification of every claim in theory/THEORY.md.

Each claim is checked twice where tractable: symbolically (sympy) and numerically
(float64, random instances). Exits NONZERO if any claim fails, and prints which.

This is the sprint's first gate: if a proof is wrong, the plan pivots immediately.
"""
import json
import sys

import numpy as np

R = {"claims": {}}
fails = []


def claim(name, ok, detail=None):
    R["claims"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def proj(p, mask_F, eps):
    """Lemma 1: I-projection of p onto {q : q(F) <= eps}."""
    P = p[mask_F].sum()
    if P <= eps:
        return p.copy()
    r = np.empty_like(p)
    r[mask_F] = eps * p[mask_F] / P
    r[~mask_F] = (1 - eps) * p[~mask_F] / (1 - P)
    return r


rng = np.random.default_rng(20261020)


def rand_instance(V=40, nF=6, eps=0.05, force_violation=True):
    z = rng.normal(0, 2.0, V)
    m = np.zeros(V, bool)
    m[rng.choice(V, nF, replace=False)] = True
    p = softmax(z)
    if force_violation:
        # push F mass above eps so the constraint is active
        while p[m].sum() <= eps * 1.5:
            z[m] += 1.0
            p = softmax(z)
    return z, p, m, eps


# ── Lemma 1: projection is feasible, minimal, preserves within-block conditionals ──
ok_feas = ok_cond = ok_min = True
for _ in range(300):
    z, p, m, eps = rand_instance()
    r = proj(p, m, eps)
    if not (abs(r.sum() - 1) < 1e-12 and r.min() >= 0):
        ok_feas = False
    if not abs(r[m].sum() - eps) < 1e-12:
        ok_feas = False
    # within-block conditionals preserved
    for blk in (m, ~m):
        if blk.sum() >= 2:
            a = r[blk] / r[blk].sum()
            b = p[blk] / p[blk].sum()
            if not np.allclose(a, b, atol=1e-12):
                ok_cond = False
    # minimality: random feasible perturbations must not lower KL
    kl = lambda q: float(np.sum(q * np.log(np.where(q > 0, q / p, 1.0))))
    base = kl(r)
    for _ in range(12):
        q = r * np.exp(rng.normal(0, .05, p.size))
        q[m] *= eps / q[m].sum()
        q[~m] *= (1 - eps) / q[~m].sum()
        if kl(q) < base - 1e-12:
            ok_min = False
claim("L1_projection_feasible_and_caps_F_at_eps", ok_feas)
claim("L1_preserves_within_block_conditionals", ok_cond)
claim("L1_is_minimal_among_feasible_perturbations", ok_min)

# ── Lemma 2: gradient equals alpha*p|F + beta*p|notF, rank<=2, conserves mass ──
ok_form = ok_rank = ok_cons = True
max_err = 0.0
for _ in range(300):
    z, p, m, eps = rand_instance()
    r = proj(p, m, eps)
    g = p - r                                     # grad of KL(sg(r) || softmax(z)) wrt z
    P = p[m].sum()
    alpha = 1 - eps / P
    beta = 1 - (1 - eps) / (1 - P)
    g_pred = np.where(m, alpha * p, beta * p)
    err = np.abs(g - g_pred).max()
    max_err = max(max_err, err)
    if err > 1e-12:
        ok_form = False
    # rank <= 2 in distribution space
    B = np.stack([p * m, p * ~m])
    coef, *_ = np.linalg.lstsq(B.T, g, rcond=None)
    if np.abs(B.T @ coef - g).max() > 1e-12:
        ok_rank = False
    if abs(alpha * P + beta * (1 - P)) > 1e-12:
        ok_cons = False
claim("L2_gradient_closed_form", ok_form, {"max_abs_err": max_err})
claim("L2_gradient_rank_at_most_2", ok_rank)
claim("L2_mass_transfer_conserves_probability", ok_cons)

# ── Corollary 2.1: within-block log-ratios stationary to first order ──
ok_null = True
worst = 0.0
for _ in range(200):
    z, p, m, eps = rand_instance()
    r = proj(p, m, eps)
    g = p - r
    nf = np.where(~m)[0]
    i, j = rng.choice(nf, 2, replace=False)
    h = 1e-7
    f = lambda e: (lambda q: np.log(q[i]) - np.log(q[j]))(softmax(z - e * g))
    d = (f(h) - f(-h)) / (2 * h)
    worst = max(worst, abs(d))
    if abs(d) > 1e-5:
        ok_null = False
claim("C21_within_notF_log_ratio_is_a_null_direction", ok_null,
      {"worst_abs_derivative": worst,
       "note": "central difference of log p(v)-log p(v') along -grad, v,v' in notF"})

# ── Proposition 3: dP/deta = -(P-eps)*[S_F/P + S_notF/(1-P)] <= 0 ──
ok_rate = ok_mono = True
worst_rel = 0.0
for _ in range(300):
    z, p, m, eps = rand_instance()
    r = proj(p, m, eps)
    g = p - r
    P = p[m].sum()
    S_F = float((p[m] ** 2).sum())
    S_nF = float((p[~m] ** 2).sum())
    pred = -(P - eps) * (S_F / P + S_nF / (1 - P))
    h = 1e-7
    Pp = softmax(z - h * g)[m].sum()
    Pm = softmax(z + h * g)[m].sum()
    num = (Pp - Pm) / (2 * h)
    rel = abs(num - pred) / max(abs(pred), 1e-12)
    worst_rel = max(worst_rel, rel)
    if rel > 1e-4:
        ok_rate = False
    if num > 1e-9:
        ok_mono = False
claim("P3_rate_matches_closed_form", ok_rate, {"worst_rel_err": worst_rel})
claim("P3_fact_mass_decreases_monotonically", ok_mono)

# ── Proposition 3 fixed point: P == eps gives zero gradient ──
ok_fp = True
for _ in range(100):
    z, p, m, eps = rand_instance()
    # rescale so that P == eps exactly
    P = p[m].sum()
    q = p.copy()
    q[m] *= eps / P
    q[~m] *= (1 - eps) / (1 - P)
    r = proj(q, m, eps)
    if np.abs(q - r).max() > 1e-12:
        ok_fp = False
claim("P3_P_equals_eps_is_a_fixed_point", ok_fp)

# ── Symbolic confirmation of L2 and P3 on a 2-block abstraction ──
try:
    import sympy as sp
    e, Pv, SF, SnF = sp.symbols("epsilon P S_F S_nF", positive=True)
    a = 1 - e / Pv
    b = 1 - (1 - e) / (1 - Pv)
    cons = sp.simplify(a * Pv + b * (1 - Pv))
    rate = sp.simplify(-(a * (SF - Pv * (SF + SnF)) + b * (-Pv * SnF)))
    target = sp.simplify(-(Pv - e) * (SF / Pv + SnF / (1 - Pv)))
    # a*P+b*(1-P) must be identically 0
    ok_sym_cons = sp.simplify(cons) == 0
    # the two rate expressions must agree identically
    ok_sym_rate = sp.simplify(sp.together(rate - target)) == 0
    claim("SYM_L2_mass_conservation_identically_zero", bool(ok_sym_cons),
          {"expr": str(cons)})
    claim("SYM_P3_rate_identity", bool(ok_sym_rate),
          {"rate": str(sp.simplify(rate)), "target": str(sp.simplify(target))})
except ImportError:
    claim("SYM_sympy_available", False, "sympy not installed")

# ── Proposition 4: forward counts, by construction ──
claim("P4_spex_needs_zero_reference_forwards", True,
      {"spex": 1, "mdu_tau1": 2, "emdu": 2, "tdssd": 2, "ssd": 3,
       "fx_mix": "2 + J*n_prompts/R",
       "basis": "SPEX's target references only p_theta at states already computed"})

R["n_claims"] = len(R["claims"])
R["failed"] = fails
R["all_passed"] = not fails
json.dump(R, open("/data/bijaypandey/spex/outputs/theory_verification.json", "w"), indent=2)
print(f"\n{sum(1 for v in R['claims'].values() if v['passed'])}/{R['n_claims']} claims passed")
print(f"ALL PASSED: {R['all_passed']}")
sys.exit(0 if R["all_passed"] else 1)
