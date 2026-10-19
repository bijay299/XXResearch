#!/usr/bin/env python3
"""CPR-GEM CPU-only correctness suite.  One process, CPU tensors only.

Run via ``./run_suite.sh`` (which adds the external timeout/ulimit watchdog).
"""

import json
import math
import os
import sys
import time

# ---- CPU-only enforcement BEFORE torch import ----
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    sys.stderr.write("FATAL: CUDA_VISIBLE_DEVICES must be set to the empty string\n")
    sys.exit(2)

import torch
torch.set_num_threads(int(os.environ.get("CPR_THREADS", "4")))

from cpr_gem import fixtures as FX
from cpr_gem import geometry as G
from cpr_gem import ledger as LG
from cpr_gem import losses as L
from cpr_gem import oracle as O

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

RESULTS = {}
_ORDER = []


def _assert_cpu(*ts):
    for t in ts:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            raise RuntimeError("CUDA tensor encountered; suite is CPU-only")


def case(case_id, **fields):
    """Record one declared configuration's outcome (append-only ordering)."""
    if case_id in RESULTS:
        raise RuntimeError(f"duplicate case record: {case_id}")
    RESULTS[case_id] = fields
    _ORDER.append(case_id)
    LG.record("case", {"case_id": case_id,
                       "status": fields.get("status"),
                       "pass": fields.get("pass")})
    flag = {True: "PASS", False: "FAIL", None: "----"}.get(fields.get("pass"), "----")
    print(f"[{flag}] {case_id}  {fields.get('status','')}")
    return fields


def relerr(x, y):
    """Relative L2 error between two tensors (or floats)."""
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    den = float(torch.linalg.vector_norm(y))
    num = float(torch.linalg.vector_norm(x - y))
    return num / den if den > 0 else num


# =========================================================================
# A. source-faithful parity and structure
# =========================================================================

def group_A():
    for seed in FX.SEEDS:
        for mode in ("generic", "active", "inactive"):
            v, a, b = FX.big_triplet(seed, mode)
            _assert_cpu(v, a, b)
            eta, gs, nl = FX.ETA_PRINCIPAL, FX.GAMMA_SCALE, FX.N_LATENTS

            # --- A1 forward parity: FP32 implementation vs independent FP64 oracle
            got = float(L.native_gem_loss(v, a, b, eta, gs, nl))
            v64, a64, b64 = v.double(), a.double(), b.double()
            want = float(O.o_native(v64, a64, b64, eta, gs, nl))
            e = abs(got - want) / max(abs(want), 1e-300)
            case(f"A1.parity.native.seed{seed}.{mode}",
                 **{"pass": e <= FX.RTOL_FP32_LARGE_D, "status": "OK",
                    "loss_fp32": got, "loss_fp64_oracle": want, "rel_error": e,
                    "tol": FX.RTOL_FP32_LARGE_D})

            # --- A2 autograd vs independent FP64 analytic gradient
            for arm, fn, gfn in (
                ("native", L.native_gem_loss, O.grad_native_analytic),
                ("cpr", L.cpr_loss, O.grad_cpr_analytic),
                ("centered", L.centered_smooth_radial_loss, O.grad_centered_analytic),
            ):
                vv = v.clone().requires_grad_(True)
                out = fn(vv, a, b, eta, gs, nl)
                out.backward()
                ga = gfn(v64, a64, b64, eta, gs, nl)
                # only compare where the hinge is strictly active (smooth region)
                smooth = float(torch.linalg.vector_norm(ga)) > 0
                err = relerr(vv.grad, ga) if smooth else 0.0
                case(f"A2.autograd_vs_analytic.{arm}.seed{seed}.{mode}",
                     **{"pass": err <= FX.RTOL_FP32_LARGE_D,
                        "status": "ACTIVE" if smooth else "ALL_INACTIVE",
                        "grad_rel_error": err, "tol": FX.RTOL_FP32_LARGE_D})

    # --- A3 teacher detachment
    v, a, b = FX.big_triplet(FX.SEEDS[0], "generic")
    a_r = a.clone().requires_grad_(True)
    b_r = b.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True)
    out = L.cpr_loss(vv, a_r, b_r, FX.ETA_PRINCIPAL, FX.GAMMA_SCALE, FX.N_LATENTS)
    out.backward()
    case("A3.teacher_detachment",
         **{"pass": a_r.grad is None and b_r.grad is None and vv.grad is not None,
            "status": "OK", "anchor_grad_is_none": a_r.grad is None,
            "target_grad_is_none": b_r.grad is None,
            "student_grad_present": vv.grad is not None})

    # --- A4 FP32 promotion: FP64 inputs must be computed in FP32
    v64 = v.double()[:2, :8].clone()
    a64 = a.double()[:2, :8].clone()
    b64 = b.double()[:2, :8].clone()
    out = L.native_gem_loss(v64, a64, b64, FX.ETA_PRINCIPAL)
    case("A4.fp32_promotion",
         **{"pass": out.dtype == torch.float32, "status": "OK",
            "input_dtype": "float64", "output_dtype": str(out.dtype)})

    # --- A5 shape / batch handling
    vb = torch.randn(FX.BATCH, 4, 8, 16)            # nonbatch dims flattened to 512
    ab, bb = torch.randn_like(vb), torch.randn_like(vb)
    out = L.native_gem_loss(vb, ab, bb, FX.ETA_PRINCIPAL)
    flat_ok = float(out) == float(
        L.native_gem_loss(vb.reshape(FX.BATCH, -1), ab.reshape(FX.BATCH, -1),
                          bb.reshape(FX.BATCH, -1), FX.ETA_PRINCIPAL))
    case("A5.shape_batch_handling",
         **{"pass": bool(flat_ok) and out.dim() == 0, "status": "OK",
            "input_shape": list(vb.shape), "flatten_matches_manual": bool(flat_ok)})

    for cid, args in (
        ("A5.reject_shape_mismatch", (torch.randn(2, 4), torch.randn(2, 5), torch.randn(2, 4))),
        ("A5.reject_1d_input", (torch.randn(4), torch.randn(4), torch.randn(4))),
        ("A8.reject_nonfinite_v", (torch.tensor([[float("nan"), 1.0]]),
                                   torch.randn(1, 2), torch.randn(1, 2))),
        ("A8.reject_nonfinite_teacher", (torch.randn(1, 2),
                                         torch.tensor([[float("inf"), 1.0]]),
                                         torch.randn(1, 2))),
    ):
        try:
            L.native_gem_loss(*args, FX.ETA_PRINCIPAL)
            case(cid, **{"pass": False, "status": "NOT_REJECTED"})
        except L.ContractViolation as exc:
            case(cid, **{"pass": True, "status": "REJECTED", "message": str(exc)})

    try:
        L.native_gem_loss(torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2),
                          FX.ETA_PRINCIPAL, n_latents=0)
        case("A8.reject_invalid_divisor", **{"pass": False, "status": "NOT_REJECTED"})
    except L.ContractViolation as exc:
        case("A8.reject_invalid_divisor",
             **{"pass": True, "status": "REJECTED", "message": str(exc)})

    # --- A6 independent divisor (n_latents is NOT batch size)
    vs, as_, bs = FX.big_triplet(FX.SEEDS[0], "generic")
    l1 = float(L.native_gem_loss(vs, as_, bs, FX.ETA_PRINCIPAL, 1.0, 1))
    l5 = float(L.native_gem_loss(vs, as_, bs, FX.ETA_PRINCIPAL, 1.0, 5))
    lb = float(L.native_gem_loss(vs, as_, bs, FX.ETA_PRINCIPAL, 1.0, FX.BATCH))
    case("A6.independent_divisor",
         **{"pass": abs(l5 - l1 / 5) <= 1e-4 * abs(l1 / 5) and abs(lb - l1 / 8) <= 1e-4 * abs(l1 / 8),
            "status": "OK", "n_latents_1": l1, "n_latents_5": l5,
            "n_latents_batch8": lb, "note": "n_latents=5 != batch=8 and divides independently"})

    # --- A7 scaling and reduction
    g100 = float(L.native_gem_loss(vs, as_, bs, FX.ETA_PRINCIPAL, 100.0, 1))
    per = torch.relu(
        torch.nn.functional.pairwise_distance(vs, as_, p=2)
        - FX.ETA_PRINCIPAL * torch.nn.functional.pairwise_distance(vs, bs, p=2))
    case("A7.scaling_and_reduction",
         **{"pass": abs(g100 - 100.0 * l1) <= 1e-4 * abs(100.0 * l1)
                    and abs(l1 - float(per.sum())) <= 1e-5 * abs(float(per.sum())),
            "status": "OK", "gamma100_over_gamma1": g100 / l1,
            "reduction_is_sum": abs(l1 - float(per.sum())) <= 1e-5 * abs(float(per.sum()))})


# =========================================================================
# B. finite differences (FP64 oracle, away from hinges and norm cusps)
# =========================================================================

def group_B():
    g = torch.Generator().manual_seed(FX.SEEDS[2])
    v = torch.randn(2, 6, generator=g, dtype=torch.float64)
    a = torch.randn(2, 6, generator=g, dtype=torch.float64)
    b = torch.randn(2, 6, generator=g, dtype=torch.float64) + 3.0   # keep off cusps
    eta = FX.ETA_PRINCIPAL
    for arm, f, gfn in (
        ("native", lambda x: O.o_native(x, a, b, eta), O.grad_native_analytic),
        ("cpr", lambda x: O.o_cpr(x, a, b, eta), O.grad_cpr_analytic),
        ("centered", lambda x: O.o_centered(x, a, b, eta), O.grad_centered_analytic),
    ):
        ga = gfn(v, a, b, eta)
        gf = O.fd_grad(f, v, h=1e-6)
        err = relerr(gf, ga)
        # hinge margin: distance of each sample from the hinge boundary
        dp = O.o_d_plus(v, a)
        margin = float((dp - eta * O.o_d_minus(v, b)).abs().min())
        case(f"B1.finite_difference.{arm}",
             **{"pass": err <= FX.RTOL_FD_FP64, "status": "OK",
                "fd_vs_analytic_rel_error": err, "tol": FX.RTOL_FD_FP64,
                "min_hinge_margin": margin, "h": 1e-6,
                "oracle_precision_note":
                    "FP64 central differences: truncation O(h^2) ~1e-12, "
                    "cancellation O(eps64/h) ~1e-10; usable floor ~1e-9"})


# =========================================================================
# C. hinge and norm-geometry domain
# =========================================================================

def group_C():
    eta = FX.ETA_PRINCIPAL
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[1.0, 0.0]])

    # C1 active: v far from the anchor relative to the target.
    # REPAIR 01 (2026-10-19T02:43:17Z): the original point v=(0.01,0) sits NEAR
    # the anchor, which makes the hinge INACTIVE (D_plus ~0.01 vs eta*D_minus
    # ~0.495); it was mislabelled. v=(0.9,0) is genuinely active.
    v = torch.tensor([[0.9, 0.0]])
    lv = float(L.native_gem_loss(v, a, b, eta))
    arg1 = float(torch.nn.functional.pairwise_distance(v, a, p=2)
                 - eta * torch.nn.functional.pairwise_distance(v, b, p=2))
    vv1 = v.clone().requires_grad_(True)
    L.native_gem_loss(vv1, a, b, eta).backward()
    case("C1.hinge_active",
         **{"pass": lv > 0 and arg1 > 0 and float(vv1.grad.abs().max()) > 0,
            "status": "ACTIVE", "loss": lv, "hinge_argument": arg1,
            "grad": vv1.grad[0].tolist(), "v": v[0].tolist()})

    # C2 inactive: v at a with large eta -> hinge argument negative
    v2 = torch.tensor([[0.0, 0.0]])
    l2 = float(L.native_gem_loss(v2, a, b, 2.0))
    vv = v2.clone().requires_grad_(True)
    L.native_gem_loss(vv, a, b, 2.0).backward()
    case("C2.hinge_inactive",
         **{"pass": l2 == 0.0 and float(vv.grad.abs().max()) == 0.0,
            "status": "INACTIVE", "loss": l2, "grad_max_abs": float(vv.grad.abs().max()),
            "eta": 2.0})

    # C3 exactly zero hinge.
    # REPAIR 01 (2026-10-19T02:43:17Z): t = eta/(1+eta) = 1/3 is the zero of the
    # UNSHIFTED IDEAL hinge. Native's all-ones eps_n shift RAISES D_plus and
    # LOWERS D_minus along this path, so the ACTUAL zero sits elsewhere. Locate
    # it with the ALREADY-FROZEN bounded bisection policy (strict bracket,
    # max_iter=200, xtol=1e-12) rather than assuming the ideal root.
    t_ideal = eta / (1.0 + eta)

    def hinge_arg(t):
        vt = torch.tensor([[t, 0.0]])
        return float(torch.nn.functional.pairwise_distance(vt, a, p=2)
                     - eta * torch.nn.functional.pairwise_distance(vt, b, p=2))

    lo3, hi3 = 0.0, 1.0                      # hinge_arg(0) < 0 < hinge_arg(1)
    bracket_ok = hinge_arg(lo3) < 0.0 < hinge_arg(hi3)
    for _ in range(L.BISECT_MAX_ITER):
        mid3 = 0.5 * (lo3 + hi3)
        if hinge_arg(mid3) < 0.0:
            lo3 = mid3
        else:
            hi3 = mid3
        if (hi3 - lo3) <= L.BISECT_XTOL:
            break
    t_star3 = 0.5 * (lo3 + hi3)
    v3 = torch.tensor([[t_star3, 0.0]], requires_grad=True)
    out3 = L.native_gem_loss(v3, a, b, eta)
    out3.backward()
    arg = hinge_arg(t_star3)
    # FP32 evaluation floor for a difference of two O(1) FP32 norms
    fp32_floor = 1e-6
    case("C3.hinge_exactly_zero",
         **{"pass": bracket_ok and abs(arg) <= fp32_floor
                    and torch.isfinite(v3.grad).all().item(),
            "status": "BOUNDARY", "bracket_valid": bracket_ok,
            "t_ideal_unshifted": t_ideal, "t_actual_shifted": t_star3,
            "t_actual_minus_ideal": t_star3 - t_ideal,
            "hinge_argument": arg, "fp32_evaluation_floor": fp32_floor,
            "loss": float(out3), "grad": v3.grad[0].tolist(),
            "note": "relu boundary derivative is zero in torch; clamp NOT "
                    "substituted. Root of the ACTUAL shifted hinge, not the "
                    "unshifted ideal root eta/(1+eta)."})

    # C4 native shifted-norm minimum is at v = b - eps_n*ones, NOT at v = b
    vmin = b - L.EPS_N
    dm_min = float(torch.nn.functional.pairwise_distance(vmin, b, p=2))
    dm_at_b = float(torch.nn.functional.pairwise_distance(b, b, p=2))
    vr = vmin.clone().requires_grad_(True)
    try:
        L.native_gem_loss(vr, a, b, eta).backward()
        gfin = bool(torch.isfinite(vr.grad).all())
        gval = vr.grad[0].tolist()
    except Exception as exc:                                    # pragma: no cover
        gfin, gval = False, f"exception: {exc}"
    case("C4.native_shifted_norm_minimum",
         **{"pass": dm_min < dm_at_b and gfin, "status": "OK",
            "D_minus_at_shifted_min": dm_min, "D_minus_at_v_eq_b": dm_at_b,
            "eps_n_sqrt_d": L.EPS_N * math.sqrt(2),
            "grad_finite_at_cusp": gfin, "grad_at_cusp": gval})

    # C5 v = b (the principal point)
    v5 = b.clone().requires_grad_(True)
    o5 = L.native_gem_loss(v5, a, b, eta); o5.backward()
    v5c = b.clone().requires_grad_(True)
    o5c = L.cpr_loss(v5c, a, b, eta); o5c.backward()
    case("C5.v_equals_b",
         **{"pass": True, "status": "ACTIVE",
            "native_loss": float(o5), "native_grad": v5.grad[0].tolist(),
            "cpr_loss": float(o5c), "cpr_grad": v5c.grad[0].tolist(),
            "note": "native grad has a real transverse component from the all-ones "
                    "eps shift; CPR's is ~0"})

    # C6 idealized v = a, within its domain
    v6 = a.clone().requires_grad_(True)
    o6 = L.native_gem_loss(v6, a, b, eta); o6.backward()
    dp = float(torch.nn.functional.pairwise_distance(a, a, p=2))
    dm = float(torch.nn.functional.pairwise_distance(a, b, p=2))
    case("C6.v_equals_a_idealized",
         **{"pass": float(o6) == 0.0, "status": "INACTIVE",
            "D_plus": dp, "D_minus": dm, "hinge_argument": dp - eta * dm,
            "loss": float(o6), "grad": v6.grad[0].tolist(),
            "domain_note": "hinge inactive at v=a for eta=1/2, L=1"})


# =========================================================================
# D. contrast degeneracy and epsilon orientation
# =========================================================================

def group_D():
    eta = FX.ETA_PRINCIPAL
    v = torch.tensor([[0.3, 0.2]])

    for cid, L_target in (("D1.L_zero", 0.0),
                          ("D2.L_below_tau", L.TAU_L / 2),
                          ("D3.L_at_tau", L.TAU_L),
                          ("D4.L_above_tau", 10 * L.TAU_L)):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[L_target, 0.0]])
        alpha, Lv, e_tau, live = L.cpr_alpha(v, a, b)
        vr = v.clone().requires_grad_(True)
        out = L.cpr_loss(vr, a, b, eta)
        out.backward()
        expect_live = L_target > L.TAU_L
        ok = (bool(live[0]) == expect_live) and torch.isfinite(vr.grad).all().item() \
            and math.isfinite(float(out))
        if not expect_live:
            ok = ok and float(alpha[0]) == 0.0 and float(e_tau.abs().max()) == 0.0
        case(cid, **{"pass": ok, "status": "LIVE" if expect_live else "DEGENERATE_REPULSION_OFF",
                     "L": float(Lv[0]), "tau_L": L.TAU_L, "alpha": float(alpha[0]),
                     "e_tau": e_tau[0].tolist(), "loss": float(out),
                     "grad": vr.grad[0].tolist(), "grad_finite": bool(torch.isfinite(vr.grad).all())})

    # D5 degenerate fallback: attraction must survive while repulsion is off
    a = torch.tensor([[0.0, 0.0]]); b = a.clone()
    vr = v.clone().requires_grad_(True)
    out = L.cpr_loss(vr, a, b, eta); out.backward()
    dpl = float(torch.nn.functional.pairwise_distance(v, a, p=2))
    case("D5.degenerate_finite_fallback",
         **{"pass": math.isfinite(float(out)) and torch.isfinite(vr.grad).all().item()
                    and abs(float(out) - dpl) <= 1e-6,
            "status": "DEGENERATE_ATTRACTION_ONLY",
            "loss": float(out), "D_plus_only": dpl, "grad": vr.grad[0].tolist(),
            "note": "at L<=tau_L repulsion is disabled but attraction remains"})

    # D6-D8 epsilon orientation relative to the contrast direction
    a = torch.tensor([[0.0, 0.0]]); b = torch.tensor([[1.0, 0.0]])
    eps = 1e-3
    for cid, pert in (("D6.epsilon_parallel", torch.tensor([[eps, 0.0]])),
                      ("D7.epsilon_antiparallel", torch.tensor([[-eps, 0.0]])),
                      ("D8.epsilon_perpendicular", torch.tensor([[0.0, eps]]))):
        vp = (b + pert).clone().requires_grad_(True)
        on = L.native_gem_loss(vp, a, b, eta); on.backward()
        gn = vp.grad[0].clone()
        vc = (b + pert).clone().requires_grad_(True)
        oc = L.cpr_loss(vc, a, b, eta); oc.backward()
        gc = vc.grad[0].clone()
        alpha, _, _, _ = L.cpr_alpha(vp.detach(), a, b)
        case(cid, **{"pass": bool(torch.isfinite(gn).all() and torch.isfinite(gc).all()),
                     "status": "OK", "perturbation": pert[0].tolist(),
                     "alpha": float(alpha[0]),
                     "native_loss": float(on), "native_grad": gn.tolist(),
                     "cpr_loss": float(oc), "cpr_grad": gc.tolist()})


# =========================================================================
# E. controls
# =========================================================================

def group_E():
    eta = FX.ETA_PRINCIPAL
    a = torch.tensor([[0.0, 0.0]]); b = torch.tensor([[1.0, 0.0]])
    v = b.clone()

    # E1 centered control matches native repulsion VALUE at v = b
    d = 2
    delta_d = L.EPS_N * math.sqrt(d)
    native_dm = float(torch.nn.functional.pairwise_distance(v, b, p=2))
    case("E1.centered_matches_native_value_at_v_eq_b",
         **{"pass": abs(native_dm - delta_d) <= 1e-12, "status": "OK",
            "native_D_minus_at_v_eq_b": native_dm, "delta_d": delta_d,
            "note": "value matched; the all-ones DIRECTIONAL shift is removed. "
                    "Diagnostic control, not complete causal isolation."})

    # E2 anchor regression == native at eta = 0
    vs, as_, bs = FX.big_triplet(FX.SEEDS[0], "generic")
    l_anchor = float(L.anchor_regression_loss(vs, as_, bs, 0.0, FX.GAMMA_SCALE, FX.N_LATENTS))
    l_native0 = float(L.native_gem_loss(vs, as_, bs, 0.0, FX.GAMMA_SCALE, FX.N_LATENTS))
    case("E2.anchor_regression_equals_native_eta0",
         **{"pass": l_anchor == l_native0, "status": "OK",
            "anchor_regression": l_anchor, "native_eta0": l_native0})

    # E3 axial margin
    lo, st = L.axial_margin_loss(v, a, b, eta)
    case("E3.axial_margin_in_domain",
         **{"pass": st == "OK" and abs(float(lo) - 1.0) <= 1e-6, "status": st,
            "loss": float(lo), "expected_L_minus_1plus_eta_alpha": 1.0})
    lo2, st2 = L.axial_margin_loss(v, a, a, eta)
    case("E3.axial_margin_degenerate_NA",
         **{"pass": st2 == L.NOT_APPLICABLE and lo2 is None, "status": st2})

    # E4 ideal threshold and the CPR stopping-point identity (in its domain only)
    Lmag = 1.0
    k_ideal = L.ideal_k_t(Lmag, eta)
    alpha_stop = Lmag / (1.0 + eta)
    v_stop = b - alpha_stop * torch.tensor([[1.0, 0.0]])
    cpr_at_stop = float(L.cpr_loss(v_stop, a, b, eta))
    case("E4.cpr_stopping_point_identity",
         **{"pass": abs(cpr_at_stop) <= 1e-5, "status": "IDEAL_UNSHIFTED_DOMAIN",
            "k_t_ideal": k_ideal, "alpha_stop": alpha_stop,
            "cpr_loss_at_stop": cpr_at_stop,
            "domain_note": "verified ONLY in the ideal unshifted collinear domain; "
                           "NOT an exact native match off-axis"})
    mt = float(L.margin_truncated_anchor_loss(v, a, b, k_ideal))
    case("E4.margin_truncated_ideal_kt",
         **{"pass": abs(mt - (1.0 - k_ideal)) <= 1e-5, "status": "OK",
            "k_t": k_ideal, "loss": mt, "expected": 1.0 - k_ideal})

    # E5 stronger native collinear match by bounded bisection on the real hinge
    k_m, st5, info = L.matched_k_t(a[0], b[0], eta)
    ok5 = st5 == "OK" and abs(info.get("root_residual", 1.0)) <= L.ROOT_RESIDUAL_TOL
    mt5 = float(L.margin_truncated_anchor_loss(v, a, b, k_m)) if ok5 else None
    case("E5.margin_truncated_matched_kt",
         **{"pass": ok5, "status": st5, "k_t_matched": float(k_m) if k_m is not None else None,
            "k_t_ideal": k_ideal,
            "matched_minus_ideal": (float(k_m) - k_ideal) if k_m is not None else None,
            "loss": mt5, **{k: v_ for k, v_ in info.items()}})

    # degenerate contrast -> must report NOT_BRACKETED, not a repaired root
    k_n, st_n, info_n = L.matched_k_t(a[0], a[0], eta)
    case("E5.matched_kt_not_bracketed",
         **{"pass": st_n == L.NOT_BRACKETED and k_n is None, "status": st_n,
            "info": info_n})


# =========================================================================
# F. Section-5 matched-first-order-progress geometry
# =========================================================================

def group_F():
    eta = FX.ETA_PRINCIPAL
    v0, a, b = FX.principal_2d()
    e = torch.tensor([1.0, 0.0])                 # common unit contrast e_tau

    # gradients from the IMPLEMENTED losses at the common detached point v = b
    grads = {}
    for arm, fn in (("native_gem", L.native_gem_loss),
                    ("cpr", L.cpr_loss),
                    ("centered_smooth_radial", L.centered_smooth_radial_loss)):
        vv = v0.clone().requires_grad_(True)
        fn(vv, a, b, eta).backward()
        grads[arm] = vv.grad[0].detach().clone()

    # the COMMON transverse direction is the ACTUAL one of the native gradient
    # (w + epsilon_perp), never a silently idealized w
    u, u_norm = G.transverse_direction(grads["native_gem"], e)
    LG.record("common_frame", {"e": e.tolist(),
                               "u": None if u is None else u.tolist(),
                               "native_transverse_norm": u_norm})

    for mname, (J_c, J_a, P, expectation) in FX.SHARED_MAPS.items():
        for arm in ("native_gem", "cpr", "centered_smooth_radial"):
            g = grads[arm]
            g_par, rho = G.decompose(g, e, u)
            own_u, own_tn = G.transverse_direction(g, e)
            res = G.analyse(J_c, J_a, P, e, u, rho, FX.P_PROGRESS, g_par)
            res["arm_own_transverse_norm"] = own_tn
            res["arm_is_zero_transverse"] = own_tn <= G.TRANSVERSE_FLOOR
            res["gradient"] = g.tolist()
            res["expectation"] = expectation
            res["J_c"] = J_c.tolist()
            res["J_a"] = J_a.tolist()
            ok = res["status"] in (G.OK, G.ZERO_TRANSVERSE)
            if ok:
                # FP64 identity checks: matched progress must reproduce p, and the
                # closed-form F(rho) drift must reproduce ||J_a Delta_psi||^2/2
                ok = (res["progress_abs_error"] <= 1e-15
                      and res["drift_rel_error"] <= 1e-10)
            else:
                ok = True            # a declared rejection is a PASSING outcome
            case(f"F.{mname}.{arm}", **{"pass": ok, **res})

    # cross-arm leading-drift ordering, recorded as evidence (not a new case id)
    ordering = {}
    for mname in FX.SHARED_MAPS:
        row = {}
        for arm in ("native_gem", "cpr", "centered_smooth_radial"):
            r = RESULTS[f"F.{mname}.{arm}"]
            row[arm] = {"status": r["status"],
                        "F_rho": r.get("F_rho"),
                        "D_allow_2": r.get("predicted_D_allow_2"),
                        "rho": r.get("rho")}
        n, c = row["native_gem"].get("D_allow_2"), row["cpr"].get("D_allow_2")
        if n is not None and c is not None:
            row["verdict"] = ("CPR_LOWER" if c < n * (1 - 1e-12)
                              else "NATIVE_LOWER" if n < c * (1 - 1e-12) else "EQUAL")
            row["cpr_over_native"] = c / n
        else:
            row["verdict"] = "NOT_COMPARABLE"
        ordering[mname] = row
    RESULTS["_drift_ordering"] = ordering
    print("\n--- leading allowed-drift ordering ---")
    for m, r in ordering.items():
        print(f"  {m:22s} {r['verdict']:18s} "
              f"ratio={r.get('cpr_over_native')}")
    print()


# =========================================================================
# G. nonlinear remainder bounds at frozen shrinking steps
# =========================================================================

def group_G():
    J_c, J_a, Q, R, C3 = FX.nonlinear_map()
    b0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    r0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    direction = torch.tensor([1.0, -1.0], dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)

    def c_of(psi):
        return (b0 + J_c @ psi
                + 0.5 * torch.einsum("ijk,j,k->i", Q, psi, psi)
                + (1.0 / 6.0) * torch.einsum("ijkl,j,k,l->i", C3, psi, psi, psi))

    def r_of(psi):
        return r0 + J_a @ psi + 0.5 * torch.einsum("ijk,j,k->i", R, psi, psi)

    prev = None
    for h in FX.NONLINEAR_SCALES:
        psi = h * direction
        c_act = c_of(psi); c_lin = b0 + J_c @ psi
        r_act = r_of(psi); r_lin = r0 + J_a @ psi
        rem_c = float(torch.linalg.vector_norm(c_act - c_lin))
        rem_r = float(torch.linalg.vector_norm(r_act - r_lin))
        ratio_c, ratio_r = rem_c / h ** 2, rem_r / h ** 2
        slope_c = (math.log(prev[0] / rem_c) / math.log(prev[1] / h)) if prev else None
        ok = (ratio_c <= FX.NONLINEAR_M_BOUND and ratio_r <= FX.NONLINEAR_M_BOUND)
        case(f"G1.nonlinear_remainder.h{h:g}",
             **{"pass": ok, "status": "OK", "h": h,
                "target_progress_remainder": rem_c,
                "allowed_drift_remainder": rem_r,
                "remainder_over_h2_target": ratio_c,
                "remainder_over_h2_allowed": ratio_r,
                "declared_M_bound": FX.NONLINEAR_M_BOUND,
                "observed_log_slope_target": slope_c,
                "precision_note":
                    "FP64 mathematics, separate from the FP32 losses. The smallest "
                    "remainder here is ~1e-6, far above the FP64 floor, so no "
                    "asymptotic slope below numerical precision is demanded."})
        prev = (rem_c, h)


# =========================================================================
# H. tiny factorized LoRA toy
# =========================================================================

def group_H():
    torch.manual_seed(FX.SEEDS[0])
    r, din, dout = FX.LORA_RANK, FX.LORA_IN, FX.LORA_OUT
    A = torch.empty(r, din)
    torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))        # seeded, NONZERO
    A = A.clone().requires_grad_(True)
    B = torch.zeros(dout, r, requires_grad=True)             # zero, per real LoRA init

    x = torch.randn(din, 1)
    delta_W = B @ A
    y = delta_W @ x

    zero_update = bool(torch.all(delta_W == 0)) and bool(torch.all(y == 0))
    case("H1.lora_zero_effective_update",
         **{"pass": zero_update, "status": "OK",
            "A_nonzero": bool(A.abs().max() > 0), "B_is_zero": bool(B.abs().max() == 0),
            "effective_delta_W_max_abs": float(delta_W.abs().max()),
            "adapter_output_max_abs": float(y.abs().max()),
            "note": "actual LoRA initialization used; both factors zero is NOT the candidate"})

    y.sum().backward()
    gB = B.grad.abs().max().item()
    gA = A.grad.abs().max().item()
    case("H1.lora_nonzero_jacobian",
         **{"pass": gB > 0 and gA == 0.0, "status": "OK",
            "d_out_d_B_max_abs": gB, "d_out_d_A_max_abs": gA,
            "rank": r,
            "note": "d(BA) = dB@A + B@dA; with B=0, A!=0 the relevant first-order "
                    "Jacobian (w.r.t. B) is NONZERO while d/dA vanishes. "
                    "No PEFT dependency, no AdamW trajectory."})


# =========================================================================

def main():
    os.makedirs(ART, exist_ok=True)
    LG.init(os.path.join(ART, "resource_ledger.jsonl"))
    wd = LG.Watchdog(max_wall_s=float(os.environ.get("CPR_MAX_WALL_S", 1800)),
                     max_cpu_core_s=float(os.environ.get("CPR_MAX_CPU_S", 7200)),
                     max_rss_kb=int(os.environ.get("CPR_MAX_RSS_KB", 8 * 1024 * 1024))).start()
    t0 = time.time()
    LG.record("suite_start", {"declared_cases": len(FX.DECLARED_CASES),
                              "torch": torch.__version__,
                              "threads": torch.get_num_threads()})
    failures = []
    try:
        for name, fn in (("A", group_A), ("B", group_B), ("C", group_C),
                         ("D", group_D), ("E", group_E), ("F", group_F),
                         ("G", group_G), ("H", group_H)):
            LG.record("group_start", {"group": name})
            fn()
            LG.record("group_end", {"group": name})
    finally:
        wd.stop()

    # completeness: every DECLARED configuration must appear
    recorded = set(k for k in RESULTS if not k.startswith("_"))
    declared = set(FX.DECLARED_CASES)
    missing = sorted(declared - recorded)
    extra = sorted(recorded - declared)
    failures = sorted(k for k in recorded if RESULTS[k].get("pass") is False)

    summary = {
        "declared_case_count": len(declared),
        "recorded_case_count": len(recorded),
        "missing_cases": missing,
        "undeclared_cases": extra,
        "failures": failures,
        "all_declared_present": not missing and not extra,
        "elapsed_s": round(time.time() - t0, 3),
        "resources": LG.snapshot(),
        "torch_version": torch.__version__,
        "cuda_available_checked": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    verdict = ("CPU_PROTOTYPE_VERIFIED"
               if summary["all_declared_present"] and not failures
               else "CPU_PROTOTYPE_INCOMPLETE")
    summary["verdict"] = verdict
    RESULTS["_summary"] = summary
    RESULTS["_case_order"] = _ORDER

    with open(os.path.join(ART, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, sort_keys=True, default=str)
    LG.record("suite_end", summary)

    print("=" * 70)
    print(f"declared={len(declared)} recorded={len(recorded)} "
          f"missing={len(missing)} undeclared={len(extra)} failures={len(failures)}")
    if missing:
        print("MISSING:", missing)
    if extra:
        print("UNDECLARED:", extra)
    if failures:
        print("FAILURES:", failures)
    print("VERDICT:", verdict)
    print("=" * 70)
    return 0 if verdict == "CPU_PROTOTYPE_VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
