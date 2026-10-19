"""Numerical gate: independent FP64 analytic references, prospective criteria, synthetic full-dimension cases, and the
read-only real-operand probe used inside the actual native GPU trainer.

Criteria (frozen before final validation; see protocol.TOL):
  loss:      |l - r| <= atol + rtol*|r|                 (atol 1e-6, rtol 1e-5)
  grad RMS:  rms(g - r)   <= 1e-7 + 1e-4 * rms(r)
  grad max:  max|g - r|   <= 1e-7 + 1e-4 * max|r|
  plus: equal shapes, finite values, EXACT zero loss/gradient when the FP64 reference is clearly inactive.
A mismatch whose FP64 pre-ReLU margin lies inside the FP32 rounding band is classified BRANCH_ROUNDING_AMBIGUITY (reported,
not a pass-by-silence); any other mismatch is FORMULA_OR_WIRING_FAILURE.
"""
import json
import math
import os

import numpy as np
import torch

from cprx import losses as LS
from cprx import protocol as P
from cprx.flux_run import rng_digest

EPS, TAU = 1e-6, 1e-6
ETA, GAMMA = P.SETTINGS["eta"], P.SETTINGS["gamma"]
AMBIG_REL = 1e-5     # FP32 branch-rounding band: |margin| <= AMBIG_REL * (|D+| + |eta*term|)


# ----------------------------------------------------------------------------- independent FP64 analytic references
def ref64(arm, v, a, b, eta=ETA):
    """v,a,b: 1-D numpy float64. Returns dict(x, loss, grad, active, band). Closed-form gradients (no autograd)."""
    v, a, b = (np.asarray(t, dtype=np.float64).reshape(-1) for t in (v, a, b))
    d = v.size
    up = v - a + EPS
    dp = math.sqrt(float(np.dot(up, up)))
    um = v - b + EPS
    dm = math.sqrt(float(np.dot(um, um)))
    c = b - a
    L = math.sqrt(float(np.dot(c, c)))
    e = c / L if L > TAU else np.zeros_like(c)
    alpha = float(np.dot(b - v, e))
    gp = up / dp
    if arm == "native_gem":
        x, term, g = dp - eta * dm, eta * dm, gp - eta * (um / dm)
    elif arm == "cpr":
        x, term, g = dp - eta * alpha, abs(eta * alpha), gp + eta * e
    elif arm == "centered_radial":
        w = v - b
        rad = math.sqrt(float(np.dot(w, w)) + EPS ** 2 * d)
        x, term, g = dp - eta * rad, eta * rad, gp - eta * (w / rad)
    elif arm == "truncated_anchor":
        k = eta * L / (1.0 + eta)
        x, term, g = dp - k, k, gp
    elif arm == "anchor_only":
        x, term, g = dp, 0.0, gp
    elif arm == "axial_margin":
        if L <= TAU:
            return {"domain_failure": True, "L": L}
        x, term, g = L - (1.0 + eta) * alpha, (1.0 + eta) * abs(alpha), (1.0 + eta) * e
    else:
        raise KeyError(arm)
    relu = arm != "anchor_only"
    active = (x > 0) if relu else True
    band = AMBIG_REL * (abs(dp) + abs(term) + L)
    return {"domain_failure": False, "x": x, "loss": max(x, 0.0) if relu else x, "grad": g if active else np.zeros_like(g),
            "active": bool(active), "band": band, "L": L, "d_plus": dp, "d_minus": dm, "alpha": alpha, "relu": relu}


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def compare(loss32, grad32, ref):
    """Apply the frozen criteria to ONE sample. grad32: array-like (d,)."""
    g = np.asarray(grad32, dtype=np.float64).reshape(-1)
    r = ref["grad"]
    out = {"shape_ok": g.shape == r.shape, "finite": bool(np.isfinite(g).all() and math.isfinite(float(loss32)))}
    if not (out["shape_ok"] and out["finite"]):
        return dict(out, ok=False, classification="FORMULA_OR_WIRING_FAILURE")
    tl, tr, tm = P.TOL["loss"], P.TOL["grad_rms"], P.TOL["grad_max"]
    out["loss_err"] = abs(float(loss32) - ref["loss"])
    out["loss_ok"] = out["loss_err"] <= tl["atol"] + tl["rtol"] * abs(ref["loss"])
    err = g - r
    out["rms_err"], out["rms_ref"] = _rms(err), _rms(r)
    out["rms_ok"] = out["rms_err"] <= tr["atol"] + tr["rtol"] * out["rms_ref"]
    out["max_err"], out["max_ref"] = float(np.abs(err).max()), float(np.abs(r).max())
    out["max_ok"] = out["max_err"] <= tm["atol"] + tm["rtol"] * out["max_ref"]
    clearly_inactive = ref["relu"] and ref["x"] < -ref["band"]
    out["clearly_inactive"] = bool(clearly_inactive)
    out["inactive_exact_zero"] = (float(loss32) == 0.0 and not np.any(g)) if clearly_inactive else None
    ok = out["loss_ok"] and out["rms_ok"] and out["max_ok"] and (out["inactive_exact_zero"] in (None, True))
    out["ok"] = bool(ok)
    out["margin"], out["band"] = ref["x"], ref["band"]
    if ok:
        out["classification"] = "PASS"
    elif ref["relu"] and abs(ref["x"]) <= ref["band"]:
        out["classification"] = "BRANCH_ROUNDING_AMBIGUITY"
    else:
        out["classification"] = "FORMULA_OR_WIRING_FAILURE"
    return out


# ----------------------------------------------------------------------------- torch evaluation of the hooked formulas
def torch_eval(fn, v, a, b, device):
    """fn(v,a,b)->per-sample. Returns (per_sample np (n,), grad np (n,d)) in FP32 on `device`. Mirrors trainer reduction."""
    vt = torch.as_tensor(v, dtype=torch.float32, device=device).clone().requires_grad_(True)
    at = torch.as_tensor(a, dtype=torch.float32, device=device).clone()
    bt = torch.as_tensor(b, dtype=torch.float32, device=device).clone()
    ps = fn(vt, at, bt)
    g, = torch.autograd.grad(ps.sum(), vt)
    return ps.detach().double().cpu().numpy(), g.detach().double().cpu().numpy()


# ----------------------------------------------------------------------------- synthetic full-dimension cases
def synth_cases(seed=20261019, d=P.SYNTH_D):
    rg = np.random.default_rng(seed)
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    a = f32(rg.standard_normal(d))
    b = f32(a + rg.standard_normal(d) * 0.7)
    z = f32(rg.standard_normal(d))
    mid = f32(0.5 * (a + b) + 0.2 * rg.standard_normal(d))
    cases = {
        "generic_midpoint": (mid, a, b),
        "student_equals_target_bitwise": (b.copy(), a, b),
        "student_equals_anchor_bitwise": (a.copy(), a, b),
        "near_target_cancellation_1e-6": (f32(b + 1e-6 * z), a, b),
        "near_anchor_1e-6": (f32(a + 1e-6 * z), a, b),
        "anchor_side_inactive_for_hinge_arms": (f32(a + 0.1 * (b - a)), a, b),
        "beyond_target_active": (f32(b + 0.5 * (b - a)), a, b),
    }
    for name, scale in (("small_contrast_1e-3", 1e-3), ("small_contrast_1e-5", 1e-5), ("small_contrast_1e-8", 1e-8),
                        ("below_tau_contrast_2e-9", 2e-9)):
        bb = f32(a + scale * z)
        cases[name] = (f32(a + 0.01 * rg.standard_normal(d)), a, bb)
    cases["zero_contrast_a_equals_b"] = (f32(a + 0.01 * rg.standard_normal(d)), a, a.copy())
    cases["zero_contrast_student_equals_both"] = (a.copy(), a, a.copy())
    return cases


def relu_boundary_scalar_tests(device):
    out = {}
    for val in (0.0, 1e-30, -1e-30, 1.0):
        x = torch.tensor([val], device=device, dtype=torch.float32, requires_grad=True)
        g, = torch.autograd.grad(torch.relu(x).sum(), x)
        out[repr(val)] = float(g)
    out["convention_grad_at_zero_is_zero"] = out["0.0"] == 0.0 and out["1e-30"] == 1.0 and out["-1e-30"] == 0.0
    return out


def run_synthetic(device, d=P.SYNTH_D, seed=20261019):
    """All six arms x all cases on `device` (FP32), vs FP64 analytic references. Returns report dict."""
    rep = {"device": str(device), "d": d, "seed": seed, "criteria": P.TOL, "cases": {}, "batch": {}, "failures": [],
           "ambiguities": [], "domain_checks": {}, "boundary": {}}
    cases = synth_cases(seed, d)
    for cname, (v, a, b) in cases.items():
        rep["cases"][cname] = {}
        for arm in P.ARMS:
            ref = ref64(arm, v, a, b)
            if ref.get("domain_failure"):
                try:
                    torch_eval(lambda x, y, z: LS.hooked(arm, x, y, z, ETA), v[None], a[None], b[None], device)
                    ok = False
                except LS.AxialDomainViolation:
                    ok = True
                rep["domain_checks"][f"{cname}/{arm}"] = {"raises_AxialDomainViolation": ok, "L": ref["L"]}
                if not ok:
                    rep["failures"].append(f"{cname}/{arm}: expected AxialDomainViolation")
                continue
            ps, g = torch_eval(lambda x, y, z, arm=arm: LS.hooked(arm, x, y, z, ETA), v[None], a[None], b[None], device)
            c = compare(ps[0], g[0], ref)
            c.update({"active_ref": ref["active"], "loss_ref": ref["loss"], "loss32": float(ps[0])})
            rep["cases"][cname][arm] = c
            if c["classification"] == "FORMULA_OR_WIRING_FAILURE":
                rep["failures"].append(f"{cname}/{arm}")
            elif c["classification"] == "BRANCH_ROUNDING_AMBIGUITY":
                rep["ambiguities"].append(f"{cname}/{arm}")
        # native arm: also compare the trainer's own source block (pairwise_distance form) via torch_eval
    # batch reduction check (n=4, per-sample + sum order)
    names = ["generic_midpoint", "near_target_cancellation_1e-6", "anchor_side_inactive_for_hinge_arms", "beyond_target_active"]
    V = np.stack([cases[n][0] for n in names]); A = np.stack([cases[n][1] for n in names]); B = np.stack([cases[n][2] for n in names])
    for arm in P.ARMS:
        ps, g = torch_eval(lambda x, y, z, arm=arm: LS.hooked(arm, x, y, z, ETA), V, A, B, device)
        res = []
        for i in range(4):
            ref = ref64(arm, V[i], A[i], B[i])
            res.append(compare(ps[i], g[i], ref)["classification"])
        rep["batch"][arm] = res
        rep["failures"] += [f"batch/{arm}/{i}" for i, r in enumerate(res) if r == "FORMULA_OR_WIRING_FAILURE"]
    # ReLU boundary convention + exact per-arm boundary evidence
    rep["relu_scalar"] = relu_boundary_scalar_tests(device)
    if not rep["relu_scalar"]["convention_grad_at_zero_is_zero"]:
        rep["failures"].append("relu_scalar_boundary_convention")
    for arm in P.ARMS:
        if arm == "axial_margin":
            rep["boundary"][arm] = "not_claimed (zero contrast is a domain failure)"
            continue
        v, a, b = cases["zero_contrast_student_equals_both"]
        ps, g = torch_eval(lambda x, y, z, arm=arm: LS.hooked(arm, x, y, z, ETA), v[None], a[None], b[None], device)
        exact = bool(ps[0] == 0.0 and not np.any(g))
        rep["boundary"][arm] = "exact_zero_loss_and_grad_observed" if exact else "not_claimed_boundary_not_exactly_zero"
    # the native arm: a==b gives bitwise D+==D- -> exact boundary
    v, a, b = cases["zero_contrast_a_equals_b"]
    ps, g = torch_eval(lambda x, y, z: LS.hooked("native_gem", x, y, z, ETA), v[None], a[None], b[None], device)
    rep["boundary"]["native_gem_a_equals_b"] = ("exact_zero_loss_and_grad_observed" if (ps[0] == 0.0 and not np.any(g))
                                                else f"not_zero loss={float(ps[0])}")
    rep["n_checks"] = sum(len(v) for v in rep["cases"].values()) + 4 * len(P.ARMS)
    rep["pass"] = not rep["failures"]
    return rep


# ----------------------------------------------------------------------------- real-operand probe (inside the trainer)
def _grad_snapshot(wrapper):
    return {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None}


class Probe:
    """Config.cpr_probe_hook: read-only, uses autograd.grad(retain_graph=True) only."""

    def __init__(self, arm, indices, out_dir, eta=ETA, gamma=GAMMA):
        self.arm, self.indices, self.out_dir, self.eta, self.gamma = arm, set(indices), out_dir, eta, gamma
        self.records, self.violations = [], []
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, processed_index, global_step, batch_start, n_latents, pred, anchor, target, target_loss, loss,
                 native_block, config, wrapper):
        if processed_index not in self.indices:
            return
        arm, eta = self.arm, self.eta
        dev = pred.device
        # ---- invariants: snapshots
        rng0, ops0 = rng_digest(), tuple(t.detach().clone() for t in (pred, anchor, target))
        gb0 = _grad_snapshot(wrapper)
        was_training = wrapper.training
        rec = {"processed_index": int(processed_index), "optimizer_updates_before": int(global_step), "arm": arm,
               "dtype": [str(t.dtype) for t in (pred, anchor, target)], "device": str(dev),
               "shape": list(pred.shape), "student_is_fp32": pred.dtype == torch.float32,
               "teachers_detached": (not anchor.requires_grad) and (not target.requires_grad) and anchor.grad_fn is None
                                    and target.grad_fn is None, "student_requires_grad": bool(pred.requires_grad),
               "n_latents": int(n_latents), "wrapper_training_mode": bool(was_training)}
        with torch.no_grad():
            a_, b_, v_ = anchor.double(), target.double(), pred.double()
            rec["norm_v_minus_b"] = float(torch.linalg.vector_norm(v_ - b_))
            rec["norm_v_minus_a"] = float(torch.linalg.vector_norm(v_ - a_))
            rec["norm_b_minus_a"] = float(torch.linalg.vector_norm(b_ - a_))
            rec["student_equals_target_bitwise"] = bool(torch.equal(pred.detach(), target))
            rec["student_equals_anchor_bitwise"] = bool(torch.equal(pred.detach(), anchor))
        # ---- actual training graph
        g_graph, = torch.autograd.grad(target_loss, pred, retain_graph=True)
        g_loss, = torch.autograd.grad(loss, pred, retain_graph=True)
        tl, ls = float(target_loss.detach()), float(loss.detach())
        rec["target_loss_graph"], rec["loss_graph"] = tl, ls
        # same-device recomputation of the native expression (gamma*loss)/n_latents; CUDA scalar division is a
        # reciprocal-multiply, so a CPU-side recomputation can differ by 1 ulp and is NOT the right comparator
        rec["reduction_bitwise"] = bool(torch.equal((self.gamma * target_loss.detach()) / n_latents, loss.detach()))
        gl = g_loss.detach().double().cpu().numpy().reshape(-1)
        gs = (g_graph.detach().double() * (self.gamma / n_latents)).cpu().numpy().reshape(-1)
        e = gl - gs
        rms_e, max_e = float(np.sqrt(np.mean(e * e))), float(np.max(np.abs(e)))
        rms_r, max_r = float(np.sqrt(np.mean(gs * gs))), float(np.max(np.abs(gs)))
        rec["grad_scale_check"] = {"rms_err": rms_e, "rms_ref": rms_r, "max_err": max_e, "max_ref": max_r,
                                   "elementwise_allclose_rtol1e-6": bool(torch.allclose(g_loss, g_graph * (self.gamma / n_latents), rtol=1e-6, atol=0.0))}
        rec["grad_scale_consistent"] = bool(rms_e <= P.TOL["grad_rms"]["atol"] + P.TOL["grad_rms"]["rtol"] * rms_r
                                            and max_e <= P.TOL["grad_max"]["atol"] + P.TOL["grad_max"]["rtol"] * max_r)
        # ---- clones: native source, hooked formulas (all arms), FP64 references (all arms)
        def clone_eval(fn):
            p = pred.detach().clone().requires_grad_(True)
            out = fn(p, anchor.detach().clone(), target.detach().clone())
            g, = torch.autograd.grad(out.sum(), p)
            return out.detach(), g.detach()
        ns_ps, ns_g = clone_eval(lambda p, a, b: native_block(p, a, b, eta))
        rec["native_source_loss"] = float(ns_ps.sum())
        v64, a64, b64 = (t.detach().double().cpu().numpy()[0] for t in (pred, anchor, target))
        arms = {}
        for k in P.ARMS:
            ref = ref64(k, v64, a64, b64, eta)
            if ref.get("domain_failure"):
                arms[k] = {"domain_failure": True, "L": ref["L"]}
                continue
            try:
                hp, hg = clone_eval(lambda p, a, b, k=k: LS.hooked(k, p, a, b, eta))
            except LS.AxialDomainViolation:
                arms[k] = {"unexpected_domain_violation": True}
                continue
            c = compare(float(hp.sum()), hg[0].double().cpu().numpy(), ref)
            c.update({"x_fp64": ref["x"], "active_fp64": ref["active"], "loss_ref": ref["loss"]})
            arms[k] = c
        rec["hooked_vs_fp64_all_arms"] = arms
        own_ref = ref64(arm, v64, a64, b64, eta)
        if own_ref.get("domain_failure"):
            rec["actual_graph_vs_fp64"] = {"domain_failure": True}
        else:
            rec["actual_graph_vs_fp64"] = compare(tl, g_graph[0].double().cpu().numpy(), own_ref)
            rec["actual_graph_vs_fp64"].update({"x_fp64": own_ref["x"], "active_fp64": own_ref["active"]})
        # identical-clone agreement: trainer's graph vs hooked/native source on the same operands, same device & dtype
        ref_fn = (lambda p, a, b: LS.hooked(arm, p, a, b, eta)) if not own_ref.get("domain_failure") else None
        agree = {}
        if ref_fn is not None:
            hp, hg = clone_eval(ref_fn)
            agree["graph_vs_hooked"] = _fp32_agree(tl, g_graph, float(hp.sum()), hg)
        agree["native_source_vs_hooked_native_reference"] = _fp32_agree(
            float(ns_ps.sum()), ns_g, *( lambda r: (float(r[0].sum()), r[1]) )(clone_eval(lambda p, a, b: LS.reference_native(p, a, b, eta))))
        if arm == "native_gem":
            agree["graph_vs_native_source"] = _fp32_agree(tl, g_graph, float(ns_ps.sum()), ns_g)
            agree["graph_vs_native_source"]["bitwise_grad"] = bool(torch.equal(g_graph, ns_g))
        rec["fp32_clone_agreement"] = agree
        # ---- post-conditions
        gb1 = _grad_snapshot(wrapper)
        rec["param_grad_buffers_unchanged"] = (gb0.keys() == gb1.keys()) and all(torch.equal(gb0[k], gb1[k]) for k in gb0)
        rec["n_param_grad_buffers"] = len(gb0)
        rec["operands_unchanged"] = all(torch.equal(x, y.detach()) for x, y in zip(ops0, (pred, anchor, target)))
        rec["rng_unchanged"] = rng_digest() == rng0
        rec["training_mode_unchanged"] = wrapper.training == was_training
        rec["probe_ok"] = bool(
            rec["student_is_fp32"] and rec["teachers_detached"] and rec["reduction_bitwise"] and rec["grad_scale_consistent"]
            and rec["param_grad_buffers_unchanged"] and rec["operands_unchanged"] and rec["rng_unchanged"]
            and rec["training_mode_unchanged"]
            and all(v.get("ok", False) or v.get("classification") == "BRANCH_ROUNDING_AMBIGUITY" or v.get("domain_failure")
                    for v in arms.values())
            and (rec["actual_graph_vs_fp64"].get("domain_failure") or rec["actual_graph_vs_fp64"]["classification"] != "FORMULA_OR_WIRING_FAILURE")
            and all(v["ok"] for v in agree.values()))
        torch.save({"pred": pred.detach().float().cpu(), "anchor": anchor.detach().float().cpu(),
                    "target": target.detach().float().cpu(), "processed_index": processed_index, "arm": arm},
                   os.path.join(self.out_dir, f"operands_{arm}_{processed_index}.pt"))
        self.records.append(rec)


def _fp32_agree(l_a, g_a, l_b, g_b):
    """Same-device FP32 agreement of two (loss, grad) pairs under the frozen criteria (reference := b)."""
    ga = g_a.detach().double().cpu().numpy().reshape(-1)
    gb = g_b.detach().double().cpu().numpy().reshape(-1)
    ref = {"grad": gb, "loss": float(l_b), "relu": False, "x": 0.0, "band": 0.0}
    c = compare(l_a, ga, ref)
    return {k: c[k] for k in ("ok", "loss_err", "rms_err", "rms_ref", "max_err", "max_ref", "loss_ok", "rms_ok", "max_ok")}
