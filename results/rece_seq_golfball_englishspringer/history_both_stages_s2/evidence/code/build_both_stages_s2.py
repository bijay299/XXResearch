#!/usr/bin/env python3
"""CPU-only fp64 solve-and-build for S2_HISTORY_BOTH_STAGES: one Stage-2
refinement update applied to the already-improved S2_STAGE1_HISTORY
checkpoint. Same three-term objective and fp64 Cholesky methodology as
history_preservation_confirmation/run_build_arm_seq.py, with W taken
from S2_STAGE1_HISTORY (not S2_Stage1), X the freshly recovered
S2_BOTH_STAGES embedding (not the old, now-invalid E_S2_newtarget), and
G/R the REUSED existing guide_S2.pt / retain_S2_historical.pt (loaded
and hash-verified, never re-encoded).

Unlike the prior script this assignment corrects: before saving, this
script independently clones the intended final CPU state (the dict that
will be written) and, after reload, asserts EXACT equality of every
key/shape/dtype/value against that specific clone -- not merely against
a dynamically computed "changed vs. start" complement.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

torch.set_num_threads(8)

LAMB = 0.1
ERASE = 1.0
PRESERVE = 0.1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frob(x):
    return x.norm(p="fro").item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_ckpt", required=True, help="S2_STAGE1_HISTORY.pt")
    ap.add_argument("--w_ckpt_sha256", required=True)
    ap.add_argument("--x_path", required=True, help="E_S2_BOTH_STAGES_newtarget.pt")
    ap.add_argument("--x_sha256", required=True)
    ap.add_argument("--guide_path", required=True)
    ap.add_argument("--guide_sha256", required=True)
    ap.add_argument("--retain_path", required=True)
    ap.add_argument("--retain_sha256", required=True)
    ap.add_argument("--expected_32_keys_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    assert torch.cuda.is_available() is False, "Refusing: CUDA visible during the CPU-only solve step."

    result = {"endpoint": "S2_HISTORY_BOTH_STAGES", "stop_reason": None}

    # ---- hash-verify every input before touching it ----
    checks = {
        "w_ckpt": (args.w_ckpt, args.w_ckpt_sha256),
        "x": (args.x_path, args.x_sha256),
        "guide": (args.guide_path, args.guide_sha256),
        "retain": (args.retain_path, args.retain_sha256),
    }
    hash_report = {}
    all_ok = True
    for name, (p, expected) in checks.items():
        actual = sha256_file(Path(p))
        hash_report[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        all_ok = all_ok and (actual == expected)
    result["input_hash_verification"] = hash_report
    if not all_ok:
        result["stop_reason"] = "SCOPED STOP: one or more input hashes mismatched."
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".build_log.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    X = torch.load(args.x_path, map_location="cpu")[0].to(torch.float64)  # [77,768]
    G = torch.load(args.guide_path, map_location="cpu").to(torch.float64)  # [77,768]
    R = torch.load(args.retain_path, map_location="cpu").to(torch.float64)  # [77,768]
    assert torch.isfinite(X).all() and torch.isfinite(G).all() and torch.isfinite(R).all()
    result["operand_norms"] = {"||X||_F (E_newtarget)": frob(X), "||G||_F (guide)": frob(G), "||R||_F": frob(R)}

    expected_32_keys = set(json.loads(Path(args.expected_32_keys_json).read_text()))
    assert len(expected_32_keys) == 32

    H = LAMB * torch.eye(768, dtype=torch.float64) + ERASE * (X.T @ X) + PRESERVE * (R.T @ R)
    sym_err = (H - H.T).abs().max().item()
    eigvals = torch.linalg.eigvalsh((H + H.T) / 2)
    lam_min, lam_max = eigvals[0].item(), eigvals[-1].item()
    cond = lam_max / lam_min if lam_min > 0 else float("inf")
    H_finite = bool(torch.isfinite(H).all())
    try:
        L = torch.linalg.cholesky(H)
        chol_ok, chol_err = True, None
    except Exception as e:  # noqa: BLE001
        chol_ok, chol_err, L = False, str(e), None

    result["H_diagnostics"] = {"finite": H_finite, "symmetry_error": sym_err, "min_eig": lam_min,
                                "max_eig": lam_max, "cond": cond, "chol_ok": chol_ok, "chol_err": chol_err}
    print(f"[{time.time()-t0:.1f}s] H: finite={H_finite} sym_err={sym_err:.3e} "
          f"min_eig={lam_min:.6e} max_eig={lam_max:.6e} cond={cond:.6e} chol_ok={chol_ok}", flush=True)

    if not chol_ok:
        result["stop_reason"] = "SCOPED STOP: Cholesky factorization of H failed."
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".build_log.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    GtX = G.T @ X
    RtR = R.T @ R
    W_sd = torch.load(args.w_ckpt, map_location="cpu")

    layer_rows = []
    new_layers_fp32 = {}
    for key in sorted(expected_32_keys):
        W = W_sd[key].detach().to(torch.float64)
        assert torch.isfinite(W).all(), f"{key}: non-finite W"

        N = LAMB * W + ERASE * (W @ GtX) + PRESERVE * (W @ RtR)
        Vstar_T = torch.cholesky_solve(N.T, L)
        Vstar = Vstar_T.T
        assert torch.isfinite(Vstar).all(), f"{key}: non-finite V*"

        def backward_error(V):
            resid = V @ H - N
            denom = frob(V) * lam_max + frob(N)
            return {"numerator": frob(resid), "denominator": denom,
                    "normalized": (frob(resid) / denom) if denom > 1e-300 else None}

        def J(V):
            reg = LAMB * (V - W).norm(p="fro").item() ** 2
            erase_t = ERASE * (V @ X.T - W @ G.T).norm(p="fro").item() ** 2
            preserve_t = PRESERVE * (V @ R.T - W @ R.T).norm(p="fro").item() ** 2
            return reg, erase_t, preserve_t, reg + erase_t + preserve_t

        J_W = J(W)
        J_star = J(Vstar)
        bw_star = backward_error(Vstar)
        bw_W = backward_error(W)

        diff = W - Vstar
        quad_gap = torch.trace(diff @ H @ diff.T).item()
        excess_abs = J_W[3] - J_star[3]
        identity_tol = 1e-8 * max(1.0, abs(J_W[3]), abs(J_star[3]))
        identity_err = abs(excess_abs - quad_gap)

        Vstar_fp32 = Vstar.to(torch.float32)
        Vstar_fp32_roundtrip = Vstar_fp32.to(torch.float64)
        J_round = J(Vstar_fp32_roundtrip)
        bw_star_fp32 = backward_error(Vstar_fp32_roundtrip)

        new_layers_fp32[key] = Vstar_fp32

        layer_rows.append({
            "key": key, "d_out": W.shape[0],
            "J_at_W": {"reg": J_W[0], "erase": J_W[1], "preserve": J_W[2], "total": J_W[3]},
            "J_at_V_star_fp64": {"reg": J_star[0], "erase": J_star[1], "preserve": J_star[2], "total": J_star[3]},
            "J_at_V_star_fp32_roundtrip": {"reg": J_round[0], "erase": J_round[1], "preserve": J_round[2], "total": J_round[3]},
            "backward_error_V_star_fp64": bw_star,
            "backward_error_V_star_fp32_roundtrip": bw_star_fp32,
            "backward_error_W_candidate": bw_W,
            "excess_objective_W_minus_star_abs": excess_abs,
            "quadratic_gap_tr_diff_H_diffT": quad_gap,
            "quadratic_gap_identity_abs_error": identity_err,
            "quadratic_gap_identity_tol": identity_tol,
            "quadratic_gap_identity_pass": identity_err <= identity_tol,
            "reference_backward_error_tol": 1e-10,
            "reference_backward_error_pass": (bw_star["normalized"] is not None and bw_star["normalized"] <= 1e-10),
            "V_star_fp32_minus_fp64_roundtrip_fro": frob(Vstar_fp32.to(torch.float64) - Vstar),
        })
        print(f"[{time.time()-t0:.1f}s] {key}: ref_bw={bw_star['normalized']:.3e} "
              f"identity_pass={layer_rows[-1]['quadratic_gap_identity_pass']}", flush=True)

    ref_valid = all(r["reference_backward_error_pass"] for r in layer_rows)
    identity_valid = all(r["quadratic_gap_identity_pass"] for r in layer_rows)
    result["n_layers"] = len(layer_rows)
    result["reference_solve_fully_valid"] = ref_valid
    result["quadratic_gap_identity_fully_valid"] = identity_valid
    result["layers"] = layer_rows

    if not (ref_valid and identity_valid):
        result["stop_reason"] = "SCOPED STOP: reference solve or quadratic-gap identity failed its frozen tolerance for at least one layer."
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".build_log.json").write_text(json.dumps({k: v for k, v in result.items()}, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
        return

    # ---- assemble the intended final CPU state and CLONE it independently before saving ----
    intended_final_state = {}
    for k, v in W_sd.items():
        if k in new_layers_fp32:
            intended_final_state[k] = new_layers_fp32[k].detach().clone().cpu()
        else:
            intended_final_state[k] = v.detach().clone().cpu()
    # independent clone: separate tensor objects/storage, used ONLY for the post-reload equality check
    intended_final_state_snapshot = {k: v.clone() for k, v in intended_final_state.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(intended_final_state, out_path)
    out_sha = sha256_file(out_path)
    result["out_checkpoint_sha256"] = out_sha
    print(f"[{time.time()-t0:.1f}s] Saved {out_path}, sha256={out_sha}", flush=True)

    # ---- reload and assert EXACT equality to the independently-cloned intended final state ----
    reloaded = torch.load(out_path, map_location="cpu")
    assert set(reloaded.keys()) == set(intended_final_state_snapshot.keys()), "key set mismatch vs. intended final state"
    key_shape_dtype_value_ok = True
    mismatches = []
    for k in intended_final_state_snapshot:
        a, b = reloaded[k], intended_final_state_snapshot[k]
        ok = (a.shape == b.shape) and (a.dtype == b.dtype) and torch.equal(a, b)
        if not ok:
            key_shape_dtype_value_ok = False
            mismatches.append(k)

    # changed-vs-W_sd (the explicit whitelist check, independent of the snapshot-equality check above)
    changed_set = set(k for k in reloaded if not torch.equal(reloaded[k], W_sd[k]))
    solved_exact_fp32_cast = all(torch.equal(reloaded[k], new_layers_fp32[k]) for k in expected_32_keys)
    unchanged_654_exact = all(torch.equal(reloaded[k], W_sd[k]) for k in reloaded if k not in expected_32_keys)
    all_finite = all(torch.isfinite(reloaded[k]).all().item() for k in reloaded)

    result["reload_verification"] = {
        "n_total_tensors": len(reloaded),
        "exact_equality_vs_independently_cloned_intended_final_state": {
            "all_keys_match": key_shape_dtype_value_ok, "mismatched_keys": mismatches,
        },
        "changed_set_equals_expected_32_whitelist_exactly": changed_set == expected_32_keys,
        "extra_changed_keys_vs_whitelist": sorted(changed_set - expected_32_keys),
        "missing_changed_keys_vs_whitelist": sorted(expected_32_keys - changed_set),
        "solved_32_keys_exact_fp32_cast_match": solved_exact_fp32_cast,
        "other_654_tensors_byte_identical_to_S2_STAGE1_HISTORY_input": unchanged_654_exact,
        "all_686_tensors_finite": all_finite,
    }
    print(f"[{time.time()-t0:.1f}s] Reload verification: {result['reload_verification']}", flush=True)
    assert key_shape_dtype_value_ok, f"reload did not exactly match the intended final state: {mismatches}"
    assert changed_set == expected_32_keys and solved_exact_fp32_cast and unchanged_654_exact and all_finite, \
        "reload verification failed"

    result["total_seconds"] = time.time() - t0
    log_path = out_path.with_suffix(".build_log.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"DONE in {result['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
