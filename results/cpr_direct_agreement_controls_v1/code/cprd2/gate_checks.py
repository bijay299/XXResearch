"""Reusable gate comparisons (used on synthetic fixtures AND on captured native triplets)."""
import torch
import torch.nn.functional as F

from cprd2 import losses as LS
from cprd2.gate_core import (ETA, GAMMA, N_LATENTS, GateFailure, RecordingHook, close, away_from_boundary,
                             native_block_lines, ref64_full, run_trainer)


def exec_pristine_block(src, V, A, B):
    """Execute the pinned source's own native loss lines on cloned operands (reference D)."""
    lines = native_block_lines(src)
    pred = V.clone().float().requires_grad_(True)
    ns = {"torch": torch, "pred_flat": pred, "pos_flat": A.clone().float(), "neg_flat": B.clone().float(),
          "config": type("C", (), {"eta": ETA, "gamma": GAMMA})(), "n_latents": N_LATENTS}
    exec(lines["dist_pos"], ns)
    exec(lines["dist_neg"], ns)
    per_sample_expr = lines["target_loss"].split("=", 1)[1].strip()
    assert per_sample_expr.endswith(".sum()")
    ps = eval(per_sample_expr[:-len(".sum()")], ns)
    exec(lines["target_loss"], ns)
    ns["loss"] = ns["target_loss"]
    exec(lines["scale"], ns)
    g, = torch.autograd.grad(ns["loss"], pred)
    return {"ps": ps.detach(), "grad": g, "scalar": float(ns["loss"].detach()), "target_loss": float(ns["target_loss"].detach())}


def _rows(V, A, B, i):
    return V[i:i + 1], A[i:i + 1], B[i:i + 1]


def native_three_way(R, name, V, A, B, patched, pristine, pristine_src, fp64_only_away=False):
    """native_gem: (A) patched trainer, no hook; (B) patched trainer, hook-enabled native; (C) pristine trainer;
    (D) pristine source lines exec'd directly. Identical cloned FP32 operands."""
    n = V.shape[0]
    snap = tuple(t.clone() for t in (V, A, B))

    def three(rows):
        v, a, b = rows
        k = v.shape[0]
        hk = RecordingHook(LS.make_dispatch("native_gem", ETA))
        ra = run_trainer(patched, v, a, b, k)
        rb = run_trainer(patched, v, a, b, k, hook=hk)
        rc = run_trainer(pristine, v, a, b, k)
        rd = exec_pristine_block(pristine_src, v, a, b)
        return ra, rb, rc, rd, hk

    ra, rb, rc, rd, hk = three((V, A, B))
    # per-sample = target_loss of singleton runs (n=1 -> sum is the single sample)
    ps = {"A": [], "B": [], "C": [], "D": []}
    sing_grad_ok = True
    for i in range(n):
        sa, sb, sc, sd, _ = three(_rows(V, A, B, i))
        ps["A"].append(sa["target_loss"][0]); ps["B"].append(sb["target_loss"][0]); ps["C"].append(sc["target_loss"][0])
        ps["D"].append(float(sd["ps"][0]))
    out, bit = {}, {}
    for tag, r in (("B", rb), ("C", rc)):
        ok_s, e_s = close(r["loss"][0], ra["loss"][0], "loss")
        ok_g, e_g = close(r["grad"], ra["grad"], "grad")
        out[f"{tag}_vs_A"] = {"scalar_ok": ok_s, "scalar_err": e_s, "grad_ok": ok_g, "grad_err": e_g,
                              "bitwise_scalar": r["loss"][0] == ra["loss"][0], "bitwise_grad": torch.equal(r["grad"], ra["grad"])}
    ok_s, e_s = close(rd["scalar"], ra["loss"][0], "loss")
    ok_g, e_g = close(rd["grad"], ra["grad"], "grad")
    out["D_vs_A"] = {"scalar_ok": ok_s, "scalar_err": e_s, "grad_ok": ok_g, "grad_err": e_g,
                     "bitwise_scalar": rd["scalar"] == ra["loss"][0], "bitwise_grad": torch.equal(rd["grad"], ra["grad"])}
    for tag in ("B", "C", "D"):
        ok, e = close(torch.tensor(ps[tag]), torch.tensor(ps["A"]), "loss")
        out[f"per_sample_{tag}_vs_A"] = {"ok": ok, "err": e, "bitwise": ps[tag] == ps["A"]}
    ok, e = close(torch.tensor(ps["D"]), rd["ps"], "loss")
    out["per_sample_D_singleton_vs_batch"] = {"ok": ok, "err": e}
    # FP64 independent reference
    r64 = ref64_full("native_gem", V, A, B)
    away = bool(away_from_boundary(r64).all())
    ok_ps, e_ps = close(torch.tensor(ps["A"]), r64["ps"], "loss")
    ok_sc, e_sc = close(ra["loss"][0], r64["scalar"], "loss")
    ok_gr, e_gr = close(ra["grad"], r64["grad"], "grad")
    out["fp64_ref"] = {"away_from_boundary_all_rows": away, "per_sample_ok": ok_ps, "per_sample_err": e_ps,
                       "scalar_ok": ok_sc, "scalar_err": e_sc, "grad_ok": ok_gr, "grad_err": e_gr,
                       "compared": True}
    if fp64_only_away and not away:
        out["fp64_ref"].update({"compared": False, "skipped_near_boundary": True})
        ok_ps = ok_sc = ok_gr = True
    protocol_ok = (hk.calls == 1 and all(hk.returns_none) and not hk.violations)
    out["hook_protocol"] = {"calls": hk.calls, "all_return_none": all(hk.returns_none), "violations": hk.violations,
                            "arg_dtypes": hk.arg_dtypes[0]}
    unchanged = all(torch.equal(x, y) for x, y in zip(snap, (V, A, B)))
    out["inputs_unchanged"] = unchanged
    # divisor/reduction: bitwise (gamma*sum)/n_latents
    tl = torch.tensor(ra["target_loss"][0], dtype=torch.float32)
    out["reduction_bitwise"] = float((GAMMA * tl) / N_LATENTS) == ra["loss"][0]
    out["finite_equal_shape"] = bool(torch.isfinite(ra["grad"]).all() and ra["grad"].shape == V.shape)
    ok_all = (all(v["scalar_ok"] and v["grad_ok"] for k, v in out.items() if k.endswith("_vs_A") and "per_sample" not in k)
              and all(out[f"per_sample_{t}_vs_A"]["ok"] for t in ("B", "C", "D")) and out["per_sample_D_singleton_vs_batch"]["ok"]
              and ok_ps and ok_sc and ok_gr and protocol_ok and unchanged and out["reduction_bitwise"] and out["finite_equal_shape"])
    return bool(ok_all), out


def hooked_arm(R, name, arm, V, A, B, patched, only_away=False):
    """Patched trainer with the arm's hook vs the independent FP64 reference; hook protocol checks."""
    snap = tuple(t.clone() for t in (V, A, B))
    hk = RecordingHook(LS.make_dispatch(arm, ETA))
    r = run_trainer(patched, V, A, B, V.shape[0], hook=hk)
    ref = ref64_full(arm, V, A, B)
    away = away_from_boundary(ref)
    out = {"rows": int(V.shape[0]), "rows_away_from_boundary": int(away.sum()), "hinge": ref["hinge"]}
    native = arm == "native_gem"
    if only_away and not bool(away.all()):
        out["skipped_near_boundary"] = True
        out["compared"] = False
        ps_ok = sc_ok = gr_ok = True
    else:
        ps_ok, e_ps = (True, 0.0) if native else close(hk.ps[0], ref["ps"], "loss")   # native: per-sample via three-way
        sc_ok, e_sc = close(r["loss"][0], ref["scalar"], "loss")
        gr_ok, e_gr = close(r["grad"], ref["grad"], "grad")
        out.update({"compared": True, "per_sample_ok": ps_ok, "per_sample_err": e_ps, "scalar_ok": sc_ok,
                    "scalar_err": e_sc, "grad_ok": gr_ok, "grad_err": e_gr})
    tl = torch.tensor(r["target_loss"][0], dtype=torch.float32)
    red = float((GAMMA * tl) / N_LATENTS) == r["loss"][0] and (native or r["target_loss"][0] == float(hk.ps[0].sum()))
    prot = hk.calls == 1 and (all(hk.returns_none) if native else not any(hk.returns_none)) and not hk.violations
    unchanged = all(torch.equal(x, y) for x, y in zip(snap, (V, A, B)))
    fin = bool(torch.isfinite(r["grad"]).all() and r["grad"].shape == V.shape and (native or torch.isfinite(hk.ps[0]).all()))
    out.update({"reduction_bitwise": red, "hook_protocol_ok": prot, "hook_violations": hk.violations,
                "inputs_unchanged": unchanged, "finite_equal_shape": fin,
                "arg_dtypes": hk.arg_dtypes[0], "eta_in_hook": hk.inner.eta})
    return bool(ps_ok and sc_ok and gr_ok and red and prot and unchanged and fin), out
