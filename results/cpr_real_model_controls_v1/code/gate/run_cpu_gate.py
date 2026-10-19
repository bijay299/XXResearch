#!/usr/bin/env python3
"""Model-free entry gate: correctness of all six arms, on CPU, before any GPU work.

Genuine assertions against INDEPENDENT expected values (cprlib.oracle), which
is written from the contract's formulas rather than by calling the production
code. An inapplicable oracle is recorded as INAPPLICABLE and never counted as a
pass.
"""
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

sys.path.insert(0, os.environ.get("CPR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cprlib import ledger as LG
from cprlib import losses as LS
from cprlib import oracle as OR

ROOT = os.environ["CPR_ROOT"]
ETA, GAMMA, NL = 1.0, 100.0, 11      # eta=1, gamma=100, native divisor n_latents=11
SEED = 20261019

R, ORDER = {}, []


def case(cid, **f):
    if cid in R:
        raise RuntimeError(f"duplicate case {cid}")
    R[cid] = f
    ORDER.append(cid)
    tag = {True: "PASS", False: "FAIL", None: "INAPPLICABLE"}.get(f.get("pass"), "----")
    print(f"[{tag}] {cid}  {f.get('status','')}")
    return f


def relerr(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    den = float(torch.linalg.vector_norm(y))
    num = float(torch.linalg.vector_norm(x - y))
    return num / den if den > 0 else num


def triplet(seed, n=4, d=1024, mode="generic"):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    a = torch.randn(n, d, generator=g)
    b = torch.randn(n, d, generator=g)
    if mode == "near_b":
        v = b + 0.02 * v
    elif mode == "near_a":
        v = a + 0.02 * v
    return v, a, b


# ===================================================================== 1
def g1_parity():
    """Source-faithful forward parity for every arm vs the independent FP64 oracle."""
    for mode in ("generic", "near_b", "near_a"):
        v, a, b = triplet(SEED, mode=mode)
        for arm in LS.ARMS:
            try:
                got = float(LS.arm_loss(arm, v, a, b, ETA, GAMMA, NL))
                want = float(OR.value(arm, v, a, b, ETA, GAMMA, NL))
            except LS.AxialDomainViolation as exc:
                case(f"G1.parity.{arm}.{mode}", **{"pass": None, "status": "INAPPLICABLE",
                                                   "reason": str(exc)})
                continue
            e = abs(got - want) / max(abs(want), 1e-300)
            case(f"G1.parity.{arm}.{mode}",
                 **{"pass": e <= 1e-5, "status": "OK", "fp32": got,
                    "fp64_oracle": want, "rel_error": e, "tol": 1e-5})


# ===================================================================== 2
def g2_grad():
    """Autograd vs independent analytic gradient, in applicable smooth regions."""
    for mode in ("generic", "near_b"):
        v0, a, b = triplet(SEED + 1, mode=mode)
        for arm in LS.ARMS:
            vv = v0.clone().requires_grad_(True)
            try:
                LS.arm_loss(arm, vv, a, b, ETA, GAMMA, NL).backward()
            except LS.AxialDomainViolation as exc:
                case(f"G2.grad.{arm}.{mode}", **{"pass": None, "status": "INAPPLICABLE",
                                                 "reason": str(exc)})
                continue
            ga = OR.grad_v(arm, v0, a, b, ETA, GAMMA, NL)
            if float(torch.linalg.vector_norm(ga)) == 0.0:
                # oracle says fully inactive -> ASSERT the implementation agrees
                mx = float(vv.grad.abs().max())
                case(f"G2.grad.{arm}.{mode}",
                     **{"pass": mx == 0.0, "status": "INACTIVE_BOTH_ZERO",
                        "impl_grad_max_abs": mx})
                continue
            err = relerr(vv.grad, ga)
            case(f"G2.grad.{arm}.{mode}",
                 **{"pass": err <= 1e-5, "status": "ACTIVE_SMOOTH",
                    "grad_rel_error": err, "tol": 1e-5})

    # FP64 finite differences, away from hinges and cusps
    g = torch.Generator().manual_seed(SEED + 2)
    v = torch.randn(2, 8, generator=g, dtype=torch.float64)
    a = torch.randn(2, 8, generator=g, dtype=torch.float64)
    b = torch.randn(2, 8, generator=g, dtype=torch.float64) + 4.0
    for arm in LS.ARMS:
        ga = OR.grad_v(arm, v, a, b, ETA, GAMMA, NL)
        gf = OR.fd_grad(lambda x: OR.value(arm, x, a, b, ETA, GAMMA, NL), v, h=1e-6)
        err = relerr(gf, ga)
        case(f"G2.finite_difference.{arm}",
             **{"pass": err <= 1e-7, "status": "OK", "rel_error": err, "tol": 1e-7,
                "h": 1e-6,
                "oracle_precision": "FP64 central differences; usable floor ~1e-9"})


# ===================================================================== 3
def g3_regions():
    """Active, inactive and boundary behaviour, per arm, with expected values."""
    a = torch.tensor([[0.0, 0.0]]); b = torch.tensor([[1.0, 0.0]])

    # --- ACTIVE: v = b. Dplus ~ 1, Dminus ~ 0 -> every hinge strictly positive
    v = b.clone()
    for arm in LS.ARMS:
        vv = v.clone().requires_grad_(True)
        out = LS.arm_loss(arm, vv, a, b, ETA, GAMMA, NL); out.backward()
        want = float(OR.value(arm, v, a, b, ETA, GAMMA, NL))
        case(f"G3.active.{arm}",
             **{"pass": abs(float(out) - want) <= 1e-4 * max(abs(want), 1.0)
                        and float(out) > 0 and torch.isfinite(vv.grad).all().item(),
                "status": "ACTIVE", "loss": float(out), "oracle": want,
                "grad": vv.grad[0].tolist()})

    # --- INACTIVE: v = a, so Dplus ~ 0 while the subtracted term is ~1.
    #     Arms 1,2,3,4 must be exactly 0 with exactly zero gradient.
    #     Arm 5 (anchor_only) has no hinge; arm 6 does not depend on Dplus.
    v = a.clone()
    for arm in ("native_gem", "cpr", "centered_radial", "margin_truncated"):
        vv = v.clone().requires_grad_(True)
        out = LS.arm_loss(arm, vv, a, b, ETA, GAMMA, NL); out.backward()
        case(f"G3.inactive.{arm}",
             **{"pass": float(out) == 0.0 and float(vv.grad.abs().max()) == 0.0,
                "status": "INACTIVE", "loss": float(out),
                "grad_max_abs": float(vv.grad.abs().max()),
                "note": "relu boundary derivative is zero; clamp NOT substituted"})
    for arm in ("anchor_only", "axial_margin"):
        case(f"G3.inactive.{arm}", **{"pass": None, "status": "INAPPLICABLE",
                                      "reason": "no Dplus hinge that this point deactivates"})

    # --- BOUNDARY: native hinge exactly zero on the collinear path.
    #     Dplus(t) - eta*Dminus(t) = 0 with eta=1 -> t = 1/2 up to the eps shift;
    #     locate the ACTUAL shifted root by bisection and assert zero loss AND
    #     exactly zero gradient there.
    def harg(t):
        vt = torch.tensor([[t, 0.0]])
        gg = LS.geometry(vt, a, b, NL)
        return float(gg["d_plus"] - ETA * gg["d_minus"])
    lo, hi = 0.0, 1.0
    bracket = harg(lo) < 0.0 < harg(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if harg(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12:
            break
    tstar = 0.5 * (lo + hi)
    vv = torch.tensor([[tstar, 0.0]], requires_grad=True)
    out = LS.arm_loss("native_gem", vv, a, b, ETA, GAMMA, NL); out.backward()
    case("G3.boundary.native_gem",
         **{"pass": bracket and float(out) == 0.0 and float(vv.grad.abs().max()) == 0.0,
            "status": "BOUNDARY", "bracket_valid": bracket, "t_star": tstar,
            "hinge_argument": harg(tstar), "loss": float(out),
            "grad_max_abs": float(vv.grad.abs().max()),
            "t_ideal_unshifted": 0.5})


# ===================================================================== 4
def g4_reduction():
    """Reduction and scaling ORDER: (gamma * sum) / n_latents, divisor independent."""
    v, a, b = triplet(SEED + 3)
    g = LS.geometry(v, a, b, NL)
    per = LS.per_sample_loss("native_gem", g, ETA)
    manual = float((GAMMA * per.sum()) / NL)
    got = float(LS.arm_loss("native_gem", v, a, b, ETA, GAMMA, NL))
    # divisor is independent of batch size (batch=4, n_latents=11)
    l1 = float(LS.arm_loss("native_gem", v, a, b, ETA, GAMMA, 1))
    l11 = float(LS.arm_loss("native_gem", v, a, b, ETA, GAMMA, 11))
    lbatch = float(LS.arm_loss("native_gem", v, a, b, ETA, GAMMA, v.shape[0]))
    case("G4.reduction_order",
         **{"pass": got == manual, "status": "OK", "impl": got, "manual": manual})
    case("G4.independent_divisor",
         **{"pass": abs(l11 - l1 / 11) <= 1e-9 * abs(l1) and abs(lbatch - l1 / 4) <= 1e-9 * abs(l1),
            "status": "OK", "batch": v.shape[0], "n_latents": NL,
            "n1": l1, "n11": l11, "n_batch4": lbatch,
            "note": "n_latents=11 != batch=4; divisor supplied independently"})
    g100 = float(LS.arm_loss("native_gem", v, a, b, ETA, 100.0, NL))
    g1 = float(LS.arm_loss("native_gem", v, a, b, ETA, 1.0, NL))
    case("G4.gamma_scaling",
         **{"pass": abs(g100 - 100.0 * g1) <= 1e-9 * abs(100.0 * g1), "status": "OK",
            "ratio": g100 / g1})


# ===================================================================== 5
def g5_detach_dtype():
    """Teacher detachment and FP32 promotion, for every arm."""
    v0, a0, b0 = triplet(SEED + 4)
    for arm in LS.ARMS:
        ar = a0.clone().requires_grad_(True)
        br = b0.clone().requires_grad_(True)
        vv = v0.clone().requires_grad_(True)
        LS.arm_loss(arm, vv, ar, br, ETA, GAMMA, NL).backward()
        case(f"G5.teacher_detachment.{arm}",
             **{"pass": ar.grad is None and br.grad is None and vv.grad is not None,
                "status": "OK", "anchor_grad_none": ar.grad is None,
                "target_grad_none": br.grad is None})
    # FP32 promotion from FP64 and from BF16 inputs
    for dt in (torch.float64, torch.bfloat16):
        v = v0[:2, :16].to(dt); a = a0[:2, :16].to(dt); b = b0[:2, :16].to(dt)
        out = LS.arm_loss("native_gem", v, a, b, ETA, GAMMA, NL)
        case(f"G5.fp32_promotion.{str(dt).split('.')[-1]}",
             **{"pass": out.dtype == torch.float32, "status": "OK",
                "input_dtype": str(dt), "output_dtype": str(out.dtype)})


# ===================================================================== 6
def g6_small_contrast():
    """Finite gradients and safe small-contrast handling (no eager 0/0)."""
    v = torch.tensor([[0.3, 0.2]])
    for name, Lt in (("L_zero", 0.0), ("L_half_tau", LS.TAU_L / 2),
                     ("L_at_tau", LS.TAU_L), ("L_above_tau", 10 * LS.TAU_L)):
        a = torch.tensor([[0.0, 0.0]]); b = torch.tensor([[Lt, 0.0]])
        expect_live = Lt > LS.TAU_L
        gg = LS.geometry(v, a, b, NL)
        ok_mask = bool(gg["live"][0]) == expect_live
        if not expect_live:
            ok_mask = ok_mask and float(gg["alpha"][0]) == 0.0 and float(gg["e"].abs().max()) == 0.0
        # CPR: repulsion disabled, attraction retained
        vv = v.clone().requires_grad_(True)
        out = LS.arm_loss("cpr", vv, a, b, ETA, GAMMA, NL); out.backward()
        finite = bool(torch.isfinite(vv.grad).all()) and math.isfinite(float(out))
        if not expect_live:
            dplus_only = float(LS.reduce_native(gg["d_plus"], GAMMA, NL))
            ok_mask = ok_mask and abs(float(out) - dplus_only) <= 1e-4
        case(f"G6.small_contrast.cpr.{name}",
             **{"pass": ok_mask and finite,
                "status": "LIVE" if expect_live else "DEGENERATE_REPULSION_OFF_ATTRACTION_KEPT",
                "L": float(gg["L"][0]), "alpha": float(gg["alpha"][0]),
                "e": gg["e"][0].tolist(), "loss": float(out),
                "grad": vv.grad[0].tolist(), "finite": finite})
    # no eager 0/0 anywhere: L==0 exactly must not produce NaN in any arm
    a = torch.tensor([[0.5, 0.5]]); b = a.clone()
    for arm in LS.ARMS:
        vv = v.clone().requires_grad_(True)
        try:
            out = LS.arm_loss(arm, vv, a, b, ETA, GAMMA, NL); out.backward()
            ok = math.isfinite(float(out)) and bool(torch.isfinite(vv.grad).all())
            case(f"G6.no_nan.{arm}", **{"pass": ok, "status": "FINITE",
                                        "loss": float(out), "grad": vv.grad[0].tolist()})
        except LS.AxialDomainViolation as exc:
            case(f"G6.no_nan.{arm}",
                 **{"pass": True, "status": "AXIAL_DOMAIN_VIOLATION_RAISED",
                    "reason": str(exc),
                    "note": "explicit recorded failure, not a silent skip or fallback"})


# ===================================================================== 7
def g7_axial_domain():
    """Explicit axial-domain handling: raises in-domain-violation, works in domain."""
    a = torch.tensor([[0.0, 0.0]]); b = torch.tensor([[1.0, 0.0]]); v = b.clone()
    out = float(LS.arm_loss("axial_margin", v, a, b, ETA, GAMMA, NL))
    want = float(OR.value("axial_margin", v, a, b, ETA, GAMMA, NL))
    case("G7.axial_in_domain",
         **{"pass": abs(out - want) <= 1e-4 * max(abs(want), 1.0), "status": "OK",
            "loss": out, "oracle": want})
    raised = False
    try:
        LS.arm_loss("axial_margin", v, a, a, ETA, GAMMA, NL)
    except LS.AxialDomainViolation:
        raised = True
    case("G7.axial_domain_violation_raises",
         **{"pass": raised, "status": "RAISED" if raised else "NOT_RAISED",
            "note": "a production violation is an explicitly recorded ARM FAILURE; "
                    "never silently skipped and no fallback invented"})
    # mixed batch: one degenerate sample must still raise
    vm = torch.tensor([[1.0, 0.0], [0.3, 0.3]])
    am = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    bm = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    raised = False
    try:
        LS.arm_loss("axial_margin", vm, am, bm, ETA, GAMMA, NL)
    except LS.AxialDomainViolation:
        raised = True
    case("G7.axial_mixed_batch_raises", **{"pass": raised, "status": "RAISED" if raised else "NOT_RAISED"})


# ===================================================================== 8
def g8_rejection():
    """Invalid shapes / nonfinite / invalid divisor are rejected, not repaired."""
    cases = {
        "shape_mismatch": (torch.randn(2, 4), torch.randn(2, 5), torch.randn(2, 4), 11),
        "one_dim": (torch.randn(4), torch.randn(4), torch.randn(4), 11),
        "nonfinite_student": (torch.tensor([[float("nan"), 1.0]]), torch.randn(1, 2), torch.randn(1, 2), 11),
        "nonfinite_teacher": (torch.randn(1, 2), torch.tensor([[float("inf"), 1.0]]), torch.randn(1, 2), 11),
        "zero_divisor": (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2), 0),
    }
    for name, (v, a, b, nl) in cases.items():
        try:
            LS.arm_loss("native_gem", v, a, b, ETA, GAMMA, nl)
            case(f"G8.reject.{name}", **{"pass": False, "status": "NOT_REJECTED"})
        except LS.ContractViolation as exc:
            case(f"G8.reject.{name}", **{"pass": True, "status": "REJECTED", "message": str(exc)})


def main():
    LG.init("cpu_entry_gate", devices=())
    wd = LG.Watchdog(max_wall_s=1800).start()
    LG.record("gate_start", {"arms": list(LS.ARMS), "eta": ETA, "gamma": GAMMA,
                             "n_latents": NL, "seed": SEED})
    try:
        for fn in (g1_parity, g2_grad, g3_regions, g4_reduction,
                   g5_detach_dtype, g6_small_contrast, g7_axial_domain, g8_rejection):
            fn()
    finally:
        wd.stop()

    fails = [k for k in ORDER if R[k].get("pass") is False]
    inapp = [k for k in ORDER if R[k].get("pass") is None]
    passes = [k for k in ORDER if R[k].get("pass") is True]
    summary = {"total": len(ORDER), "passed": len(passes), "failed": len(fails),
               "inapplicable": len(inapp), "failures": fails, "inapplicable_cases": inapp,
               "verdict": "CPU_GATE_PASS" if not fails else "CPU_GATE_FAIL",
               "note": "INAPPLICABLE cases are recorded as such and are NOT counted as passes"}
    R["_summary"] = summary
    R["_order"] = ORDER
    out = os.path.join(ROOT, "artifacts", "cpu_gate_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=2, sort_keys=True, default=str)
    LG.record("gate_end", summary)
    LG.close(bucket="entry")
    print("=" * 72)
    print(f"total={len(ORDER)} passed={len(passes)} failed={len(fails)} inapplicable={len(inapp)}")
    if fails:
        print("FAILURES:", fails)
    print("VERDICT:", summary["verdict"])
    print("=" * 72)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
