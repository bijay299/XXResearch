"""Frozen fixture declarations.  No random search, no adaptive sweep.

Every tensor here is either a literal or drawn from a FIXED seed with a FIXED
shape.  Budget caps asserted at import: <=256 declared configurations,
batch size 8, flattened loss dim 65536, shared-map output dim <=8,
parameters <=64, LoRA rank <=2.
"""

import torch

# ---------------- frozen global constants ----------------
SEEDS = (20261018, 20261019, 20261020)
BATCH = 8
D_FLAT = 65536                 # flattened loss dimension
ETA_PRINCIPAL = 0.5            # eta = 1/2 for the principal paired coupling fixtures
ETA_BOUNDARY = (0.0, 1.0, 2.0)  # other fixed values ONLY in declared boundary/domain checks
GAMMA_SCALE = 100.0            # upstream GEM default gamma
N_LATENTS = 5                  # independently supplied; deliberately != BATCH

# numerical tolerances (frozen, disclosed)
RTOL_FP32_LARGE_D = 1e-4       # FP32 accumulation over d=65536
RTOL_FP32_SMALL_D = 1e-6
RTOL_FD_FP64 = 1e-7            # FP64 central differences, h=1e-6
ATOL_ZERO = 1e-12

# Section-5 matched-progress settings
P_PROGRESS = 1e-3              # common positive first-order target progress
NEAR_SINGULAR_G = 1.8266       # frozen scalar giving 1+rho*gamma_cross ~ 1e-3

# nonlinear remainder check: frozen shrinking step scales
NONLINEAR_SCALES = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
NONLINEAR_M_BOUND = 2.0        # declared first-order remainder constant M

# LoRA toy
LORA_RANK = 2
LORA_IN = 4
LORA_OUT = 3


def big_triplet(seed, mode="generic"):
    """A (BATCH, D_FLAT) student/anchor/target triplet from a fixed seed."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(BATCH, D_FLAT, generator=g)
    a = torch.randn(BATCH, D_FLAT, generator=g)
    b = torch.randn(BATCH, D_FLAT, generator=g)
    if mode == "active":
        # push v near a so D_plus is small relative to eta*D_minus -> hinge ACTIVE
        v = a + 0.05 * v
    elif mode == "inactive":
        v = b + 0.05 * v
    return v, a, b


def principal_2d():
    """Principal fixture: fresh d=2, a=(0,0), b=v=(1,0), eta=1/2."""
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[1.0, 0.0]])
    v = b.clone()
    return v, a, b


# ---------------- Section-5 shared-parameter maps (all with P = I) ----------
I2 = torch.eye(2)

SHARED_MAPS = {
    # name: (J_c, J_a, P, expectation)
    "favorable": (I2.clone(), I2.clone(), I2.clone(),
                  "matched-progress identity holds; CPR leading drift LOWER"),
    "adverse_alignment": (I2.clone(), torch.tensor([[1.0, 1.0]]), I2.clone(),
                          "B*rho < 0 cancellation; NATIVE GEM leading drift LOWER"),
    "progress_rejection": (torch.tensor([[1.0, 0.0], [2.0, 1.0]]), I2.clone(), I2.clone(),
                           "native denominator negative -> POSITIVE_PROGRESS_NOT_FEASIBLE"),
    "near_singular": (torch.tensor([[1.0, 0.0], [NEAR_SINGULAR_G, 1.0]]), I2.clone(), I2.clone(),
                      "native denominator ~1e-3 -> REJECT_NEAR_SINGULAR_DENOMINATOR"),
    "no_leading_advantage": (I2.clone(), torch.tensor([[1.0, 0.0]]), I2.clone(),
                             "K_ac u = 0 AND gamma_cross = 0 -> EQUAL leading drift"),
}


def nonlinear_map():
    """One small compatible nonlinear map with EXPLICITLY BOUNDED quadratic
    perturbation, plus a cubic tail so the O(h^2) first-order remainder is a
    real measurement rather than an identity.

    c(psi) = b + J_c psi + 1/2 Q[psi,psi] + 1/6 C3[psi,psi,psi]
    r(psi) = r0 + J_a psi + 1/2 R[psi,psi]
    Output dims 2 (<=8), parameters 2 (<=64).
    """
    g = torch.Generator().manual_seed(SEEDS[1])
    J_c = torch.eye(2, dtype=torch.float64)
    J_a = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64)
    Q = torch.randn(2, 2, 2, generator=g, dtype=torch.float64) * 0.3
    Q = 0.5 * (Q + Q.transpose(1, 2))          # symmetric in the two psi slots
    R = torch.randn(2, 2, 2, generator=g, dtype=torch.float64) * 0.2
    R = 0.5 * (R + R.transpose(1, 2))
    C3 = torch.randn(2, 2, 2, 2, generator=g, dtype=torch.float64) * 0.1
    return J_c, J_a, Q, R, C3


# ---------------- declared configuration enumeration ----------------
def declared_cases():
    """The complete frozen list of case IDs.  Every one MUST appear in the
    ledger and in the results file."""
    cases = []
    # A: source-faithful parity and structure
    for s in SEEDS:
        for mode in ("generic", "active", "inactive"):
            cases.append(f"A1.parity.native.seed{s}.{mode}")
            cases.append(f"A2.autograd_vs_analytic.cpr.seed{s}.{mode}")
            cases.append(f"A2.autograd_vs_analytic.native.seed{s}.{mode}")
            cases.append(f"A2.autograd_vs_analytic.centered.seed{s}.{mode}")
    cases += [
        "A3.teacher_detachment",
        "A4.fp32_promotion",
        "A5.shape_batch_handling",
        "A5.reject_shape_mismatch",
        "A5.reject_1d_input",
        "A6.independent_divisor",
        "A7.scaling_and_reduction",
        "A8.reject_nonfinite_v",
        "A8.reject_nonfinite_teacher",
        "A8.reject_invalid_divisor",
    ]
    # B: finite differences (FP64, away from hinges/cusps)
    for arm in ("native", "cpr", "centered"):
        cases.append(f"B1.finite_difference.{arm}")
    # C: hinge and norm-geometry domain
    cases += [
        "C1.hinge_active", "C2.hinge_inactive", "C3.hinge_exactly_zero",
        "C4.native_shifted_norm_minimum", "C5.v_equals_b", "C6.v_equals_a_idealized",
    ]
    # D: contrast degeneracy and epsilon orientation
    cases += [
        "D1.L_zero", "D2.L_below_tau", "D3.L_at_tau", "D4.L_above_tau",
        "D5.degenerate_finite_fallback",
        "D6.epsilon_parallel", "D7.epsilon_antiparallel", "D8.epsilon_perpendicular",
    ]
    # E: controls
    cases += [
        "E1.centered_matches_native_value_at_v_eq_b",
        "E2.anchor_regression_equals_native_eta0",
        "E3.axial_margin_in_domain", "E3.axial_margin_degenerate_NA",
        "E4.margin_truncated_ideal_kt", "E4.cpr_stopping_point_identity",
        "E5.margin_truncated_matched_kt", "E5.matched_kt_not_bracketed",
    ]
    # F: Section-5 matched-progress geometry (per map, per arm)
    for m in SHARED_MAPS:
        for arm in ("native_gem", "cpr", "centered_smooth_radial"):
            cases.append(f"F.{m}.{arm}")
    # G: nonlinear remainder
    for h in NONLINEAR_SCALES:
        cases.append(f"G1.nonlinear_remainder.h{h:g}")
    # H: LoRA toy
    cases += ["H1.lora_zero_effective_update", "H1.lora_nonzero_jacobian"]
    return cases


DECLARED_CASES = declared_cases()

assert len(DECLARED_CASES) <= 256, f"too many declared configs: {len(DECLARED_CASES)}"
assert len(set(DECLARED_CASES)) == len(DECLARED_CASES), "duplicate case IDs"
assert BATCH == 8 and D_FLAT == 65536
assert LORA_RANK <= 2 and LORA_IN <= 64 and LORA_OUT <= 8
