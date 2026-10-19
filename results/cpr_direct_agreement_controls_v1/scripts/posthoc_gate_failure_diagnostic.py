#!/usr/bin/env python3
"""POST-HOC DIAGNOSTIC (not a gate, not part of the freeze; written after the replay gate FAILED, no repair cycle remaining).
Read-only analysis of the two captured real triplets, native_gem arm only. It changes no gate outcome and continues no study work.
Question: is the failed `fp64_ref.grad_ok` a native-vs-reference formula disagreement or FP32 precision at dimension 65,536?"""
import json
import os

import torch

from cprd2.common import ROOT, dump, utc
from cprd2.gate_core import (ETA, GAMMA, N_LATENTS, TOL, load_patched_module, load_pristine_module, native_block_lines, ref64_full,
                             run_trainer)


def native_block_fp64(src, V, A, B):
    L = native_block_lines(src)
    pred = V.double().clone().requires_grad_(True)
    ns = {"torch": torch, "pred_flat": pred, "pos_flat": A.double(), "neg_flat": B.double(),
          "config": type("C", (), {"eta": ETA, "gamma": GAMMA})(), "n_latents": N_LATENTS}
    exec(L["dist_pos"], ns); exec(L["dist_neg"], ns); exec(L["target_loss"], ns); ns["loss"] = ns["target_loss"]; exec(L["scale"], ns)
    g, = torch.autograd.grad(ns["loss"], pred)
    return float(ns["loss"].detach()), g


def main():
    cap = torch.load(os.path.join(ROOT, "artifacts", "gate", "captured_triplets.pt"), map_location="cpu")
    patched = load_patched_module()
    _, psrc = load_pristine_module()
    out = {"utc": utc(), "kind": "POST_HOC_DIAGNOSTIC_NOT_A_GATE", "tolerance": TOL["grad"], "rows": []}
    for c in cap:
        V, A, B = (c[k].float() for k in ("pred", "anchor", "target"))
        r32 = run_trainer(patched, V, A, B, 1)
        ref = ref64_full("native_gem", V, A, B)
        l64, g64n = native_block_fp64(psrc, V, A, B)
        g32, g64 = r32["grad"].double(), ref["grad"].double()
        thr = TOL["grad"]["atol"] + TOL["grad"]["rtol"] * g64.abs()
        d = (g32 - g64).abs()
        viol = d > thr
        out["rows"].append({
            "processed_index": c["processed_index"], "dim": int(V.numel()),
            "student_equals_target_teacher_bitwise": bool(torch.equal(V, B)),
            "dminus_fp64": float((V.double() - B.double() + 1e-6).norm()), "dplus_fp64": float((V.double() - A.double() + 1e-6).norm()),
            "loss_fp32": r32["loss"][0], "loss_fp64_ref": ref["scalar"], "loss_rel_err": abs(r32["loss"][0] - ref["scalar"]) / abs(ref["scalar"]),
            "native_block_run_in_fp64_vs_independent_fp64_ref": {"loss_abs_diff": abs(l64 - ref["scalar"]),
                                                                  "grad_max_abs_diff": float((g64n - g64).abs().max())},
            "grad_fp32_vs_fp64": {"max_abs_diff": float(d.max()), "max_abs_g": float(g64.abs().max()),
                                  "median_abs_g": float(g64.abs().median()), "n_violating_elements": int(viol.sum()),
                                  "n_elements": int(viol.numel()), "max_abs_g_among_violators": float(g64.abs()[viol].max()) if viol.any() else None,
                                  "max_diff_over_max_g": float(d.max() / g64.abs().max())}})
    dump(out, os.path.join(ROOT, "artifacts", "analysis", "posthoc_gate_failure_diagnostic.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
