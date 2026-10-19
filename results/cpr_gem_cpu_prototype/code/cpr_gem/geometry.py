"""Corrected matched-first-order-progress comparison (contract section 5).

All of this is an ANALYTICAL DIAGNOSTIC evaluated at one common detached
state/parameter point.  It is NOT a replacement for native AdamW, and equal
leading drift does NOT imply equal cumulative model behaviour.

Derivation (reproduced so the code and the algebra can be checked against each
other).  Let the target map be ``c(psi)`` with Jacobian ``J_c``, the allowed
map ``r(psi)`` with Jacobian ``J_a``, and ``P`` a fixed SPD preconditioner.
The compatible kernel blocks are DERIVED from the SAME ``J_c, J_a, P`` -- they
are never chosen independently:

    K_cc = J_c P^-1 J_c^T
    K_ac = J_a P^-1 J_c^T

With a common unit contrast ``e`` and a common perpendicular unit ``u``, an
active output-space gradient is written ``g = g_parallel*(e + rho*u)``.  The
preconditioned step is ``Delta_psi = -s * P^-1 J_c^T (e + rho*u)``, so

    e^T J_c Delta_psi = -s*(e^T K_cc e + rho*e^T K_cc u) = -s*k*(1 + rho*gamma_cross)

with ``k = e^T K_cc e`` and ``gamma_cross = (e^T K_cc u)/k``.  Requiring a
COMMON POSITIVE first-order target progress ``p = -e^T J_c Delta_psi`` fixes
``s = p/[k*(1 + rho*gamma_cross)]``, giving the contract's update

    Delta_psi = -p/[k*(1+rho*gamma_cross)] * P^-1 J_c^T (e + rho*u).

The allowed first-order drift is ``J_a Delta_psi = -s*(z_e + rho*z_u)`` with
``z_e = K_ac e`` and ``z_u = K_ac u``.  At an allowed-reference-matching
starting point,

    D_allow^(2) = 1/2 * ||J_a Delta_psi||^2 = p^2/(2*k^2) * F(rho),
    F(rho) = (A + 2*B*rho + C*rho^2)/(1 + rho*gamma_cross)^2,
    A = ||z_e||^2,  B = z_e . z_u,  C = ||z_u||^2.

The ``B`` ALIGNMENT term is essential: sensitivity norms (``A``, ``C``) alone
do not determine the ordering, and a native gradient with ``B*rho < 0`` can
CANCEL drift and beat CPR's ``rho = 0``.
"""

import torch

# ---- frozen positive safety margins and the local-neighbourhood bound ----
MARGIN_G_PARALLEL = 1e-3     # |g_parallel| floor: the arm must really be active along e
MARGIN_K = 1e-6              # k floor: target direction must be reachable
MARGIN_DENOM = 1e-2          # 1 + rho*gamma_cross floor (positive side only)
STEP_NORM_BOUND = 1e-1       # ||Delta_psi|| bound for first-order locality
TRANSVERSE_FLOOR = 1e-9      # below this the gradient is treated as zero-transverse

# status codes
OK = "OK"
POSITIVE_PROGRESS_NOT_FEASIBLE = "POSITIVE_PROGRESS_NOT_FEASIBLE"
REJECT_NEAR_SINGULAR_DENOMINATOR = "REJECT_NEAR_SINGULAR_DENOMINATOR"
REJECT_SMALL_K = "REJECT_SMALL_K"
INACTIVE_OR_WEAK_GRADIENT = "INACTIVE_OR_WEAK_GRADIENT"
REJECT_STEP_TOO_LARGE = "REJECT_STEP_TOO_LARGE"
ZERO_TRANSVERSE = "ZERO_TRANSVERSE"


def kernels(J_c, J_a, P):
    """Compatible kernel blocks derived from one common ``(J_c, J_a, P)``."""
    Pinv = torch.linalg.inv(P)
    K_cc = J_c @ Pinv @ J_c.T
    K_ac = J_a @ Pinv @ J_c.T
    return K_cc, K_ac, Pinv


def transverse_direction(g, e):
    """The ACTUAL common transverse unit direction of gradient ``g``.

    This is the real ``w + epsilon_perp`` taken from the implemented loss's
    numerical gradient -- NOT a silently idealized ``w``.  Returns
    ``(u, transverse_norm)``; ``u`` is ``None`` when the transverse part is
    below the frozen floor.
    """
    g = g.reshape(-1).double()
    e = e.reshape(-1).double()
    perp = g - (g @ e) * e
    n = float(torch.linalg.vector_norm(perp))
    if n <= TRANSVERSE_FLOOR:
        return None, n
    return perp / n, n


def decompose(g, e, u):
    """Write ``g = g_parallel*(e + rho*u)`` in the COMMON ``(e, u)`` frame."""
    g = g.reshape(-1).double()
    e = e.reshape(-1).double()
    g_par = float(g @ e)
    if u is None:
        return g_par, 0.0
    return g_par, (float(g @ u.double()) / g_par if g_par != 0.0 else float("inf"))


def analyse(J_c, J_a, P, e, u, rho, p, g_parallel):
    """Matched-progress diagnostic for ONE arm.

    Returns a dict with the kernel scalars, the feasibility verdict and -- only
    when feasible -- the step, the predicted leading drift and the observed
    drift computed directly from ``J_a @ Delta_psi``.
    """
    J_c, J_a, P = J_c.double(), J_a.double(), P.double()
    e = e.reshape(-1).double()
    K_cc, K_ac, Pinv = kernels(J_c, J_a, P)

    k = float(e @ K_cc @ e)
    if u is None:
        u_vec = torch.zeros_like(e)
        gamma_cross = 0.0
        zero_transverse = True
    else:
        u_vec = u.reshape(-1).double()
        gamma_cross = float(e @ K_cc @ u_vec) / k if abs(k) > 0 else float("inf")
        zero_transverse = False

    z_e = K_ac @ e
    z_u = K_ac @ u_vec
    A = float(z_e @ z_e)
    B = float(z_e @ z_u)
    C = float(z_u @ z_u)
    denom = 1.0 + rho * gamma_cross

    out = {
        "k": k, "gamma_cross": gamma_cross, "rho": rho,
        "g_parallel": g_parallel, "A": A, "B": B, "C": C,
        "denominator_1_plus_rho_gamma": denom,
        "zero_transverse": zero_transverse,
        "margins": {"g_parallel": MARGIN_G_PARALLEL, "k": MARGIN_K,
                    "denominator": MARGIN_DENOM, "step_norm": STEP_NORM_BOUND},
    }

    # ---- feasibility gates: reject rather than repair ----
    if abs(g_parallel) < MARGIN_G_PARALLEL:
        out["status"] = INACTIVE_OR_WEAK_GRADIENT
        out["rejection_reason"] = (
            f"|g_parallel|={abs(g_parallel):.3e} < margin {MARGIN_G_PARALLEL:.1e}")
        return out
    if k < MARGIN_K:
        out["status"] = REJECT_SMALL_K
        out["rejection_reason"] = f"k={k:.3e} < margin {MARGIN_K:.1e}"
        return out
    if denom <= 0.0:
        out["status"] = POSITIVE_PROGRESS_NOT_FEASIBLE
        out["rejection_reason"] = (
            f"1+rho*gamma_cross={denom:.6f} <= 0: a positive first-order target "
            f"progress p>0 would require a negative step scale; matching refused")
        return out
    if denom < MARGIN_DENOM:
        out["status"] = REJECT_NEAR_SINGULAR_DENOMINATOR
        out["rejection_reason"] = (
            f"1+rho*gamma_cross={denom:.3e} < margin {MARGIN_DENOM:.1e}")
        return out

    # ---- feasible: build the matched step ----
    s = p / (k * denom)
    direction = Pinv @ J_c.T @ (e + rho * u_vec)
    delta_psi = -s * direction
    step_norm = float(torch.linalg.vector_norm(delta_psi))
    out["step_scale_s"] = s
    out["step_norm"] = step_norm
    if step_norm > STEP_NORM_BOUND:
        out["status"] = REJECT_STEP_TOO_LARGE
        out["rejection_reason"] = (
            f"||Delta_psi||={step_norm:.3e} > local-neighbourhood bound "
            f"{STEP_NORM_BOUND:.1e}")
        return out

    F_rho = (A + 2 * B * rho + C * rho * rho) / (denom * denom)
    predicted = p * p / (2 * k * k) * F_rho
    drift_vec = J_a @ delta_psi
    observed = 0.5 * float(drift_vec @ drift_vec)
    progress = float(-(e @ (J_c @ delta_psi)))

    out.update({
        "status": OK if not zero_transverse else ZERO_TRANSVERSE,
        "F_rho": F_rho,
        "predicted_D_allow_2": predicted,
        "observed_D_allow_2": observed,
        "drift_abs_error": abs(predicted - observed),
        "drift_rel_error": abs(predicted - observed) / max(abs(predicted), 1e-300),
        "first_order_progress": progress,
        "requested_p": p,
        "progress_abs_error": abs(progress - p),
        "delta_psi": [float(x) for x in delta_psi],
        "alignment_B_over_sqrtAC": (B / ((A * C) ** 0.5)) if A > 0 and C > 0 else None,
    })
    return out
