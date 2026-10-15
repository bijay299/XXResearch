#!/usr/bin/env python3
"""CPU-only fp64 solve-and-build for one (sequence, arm) cell of the new
history-preservation-confirmation experiment. Same three-term objective
and fp64 Cholesky methodology as preservation_content_control/
run_build_arm.py, generalized: W is the sequence's own untouched Stage-1
weight (not AB_UCE), X is the freshly recovered new-target embedding (not
the reused E_S), R is either blank or the sequence's historical-concept
retain text. No external numerically-accurate comparator checkpoint
exists for these new sequences, so the quadratic-gap self-consistency
check uses the untouched Stage-1 weight W itself as the diagnostic
candidate (always available, requires no extra artifact) -- this is a
pure correctness check on H/N assembly and the Cholesky solve, not a
comparison between preservation choices.
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

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_preservation_confirmation")

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
    ap.add_argument("--seq", required=True, choices=["S1", "S2"])
    ap.add_argument("--arm", required=True, choices=["BLANK", "HISTORY"])
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--stage1_sha256", required=True)
    ap.add_argument("--recover_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_32_keys_json", required=True,
                     help="JSON file with a list of the 32 expected changed cross-attn K/V keys")
    args = ap.parse_args()
    seq, arm = args.seq, args.arm

    t0 = time.time()
    assert torch.cuda.is_available() is False, "Refusing: CUDA visible during the CPU-only solve step."

    result = {"seq": seq, "arm": arm, "stop_reason": None}

    stage1_path = Path(args.stage1_ckpt)
    actual_stage1_sha = sha256_file(stage1_path)
    result["stage1_hash_recheck"] = {"expected": args.stage1_sha256, "actual": actual_stage1_sha}
    if actual_stage1_sha != args.stage1_sha256:
        result["stop_reason"] = "SCOPED STOP: Stage-1 checkpoint hash mismatch."
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / f"RESULT_{seq}_{arm}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    recover_dir = Path(args.recover_dir)
    recover_result = json.loads((recover_dir / f"RECOVER_ENCODE_{seq}_RESULT.json").read_text())
    assert recover_result["stage1_ckpt_sha256"] == args.stage1_sha256, "recover/encode step used a different Stage-1 checkpoint"

    X = torch.load(recover_dir / f"E_{seq}_newtarget.pt", map_location="cpu")[0].to(torch.float64)  # [77,768]
    G = torch.load(recover_dir / f"guide_{seq}.pt", map_location="cpu").to(torch.float64)            # [77,768]
    retain_label = "blank" if arm == "BLANK" else "historical"
    retain_info = recover_result["retain_encoding"][retain_label]
    assert retain_info["rows_equal_bit_identical"] is True
    R = torch.load(retain_info["path"], map_location="cpu").to(torch.float64)

    assert torch.isfinite(X).all() and torch.isfinite(G).all() and torch.isfinite(R).all()
    result["operand_norms"] = {"||X||_F (E_newtarget)": frob(X), "||G||_F (guide)": frob(G), "||R||_F": frob(R)}
    result["retain_text"] = retain_info["text"]

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
    print(f"[{seq}/{arm} {time.time()-t0:.1f}s] H: finite={H_finite} sym_err={sym_err:.3e} "
          f"min_eig={lam_min:.6e} max_eig={lam_max:.6e} cond={cond:.6e} chol_ok={chol_ok}", flush=True)

    if not chol_ok:
        result["stop_reason"] = "SCOPED STOP: Cholesky factorization of H failed."
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / f"RESULT_{seq}_{arm}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    GtX = G.T @ X
    RtR = R.T @ R
    W_sd = torch.load(stage1_path, map_location="cpu")

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
        print(f"[{seq}/{arm} {time.time()-t0:.1f}s] {key}: ref_bw={bw_star['normalized']:.3e} "
              f"identity_pass={layer_rows[-1]['quadratic_gap_identity_pass']}", flush=True)

    ref_valid = all(r["reference_backward_error_pass"] for r in layer_rows)
    identity_valid = all(r["quadratic_gap_identity_pass"] for r in layer_rows)
    result["n_layers"] = len(layer_rows)
    result["reference_solve_fully_valid"] = ref_valid
    result["quadratic_gap_identity_fully_valid"] = identity_valid
    result["layers"] = layer_rows

    if not (ref_valid and identity_valid):
        result["stop_reason"] = "SCOPED STOP: reference solve or quadratic-gap identity failed its frozen tolerance for at least one layer."
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / f"RESULT_{seq}_{arm}.json").write_text(json.dumps({k: v for k, v in result.items()}, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
        return

    out_sd = {k: (v.clone() if k not in new_layers_fp32 else new_layers_fp32[k]) for k, v in W_sd.items()}
    out_name = f"{seq}_{arm}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.pt"
    torch.save(out_sd, out_path)
    out_sha = sha256_file(out_path)
    result["out_checkpoint_name"] = out_name
    result["out_checkpoint_path"] = str(out_path)
    result["out_checkpoint_sha256"] = out_sha
    print(f"[{seq}/{arm} {time.time()-t0:.1f}s] Saved {out_path}, sha256={out_sha}", flush=True)

    reloaded = torch.load(out_path, map_location="cpu")
    assert set(reloaded.keys()) == set(W_sd.keys())
    changed_keys = [k for k in reloaded if not torch.equal(reloaded[k], W_sd[k])]
    changed_set = set(changed_keys)
    all_finite = all(torch.isfinite(reloaded[k]).all().item() for k in reloaded)
    solved_exact = all(torch.equal(reloaded[k], new_layers_fp32[k]) for k in expected_32_keys)
    unchanged_exact = all(torch.equal(reloaded[k], W_sd[k]) for k in reloaded if k not in expected_32_keys)

    result["reload_verification"] = {
        "n_total_tensors": len(reloaded), "n_changed_vs_stage1": len(changed_keys),
        "changed_set_equals_expected_32": changed_set == expected_32_keys,
        "extra_changed_keys": sorted(changed_set - expected_32_keys),
        "missing_changed_keys": sorted(expected_32_keys - changed_set),
        "all_tensors_finite": all_finite,
        "solved_32_keys_exact_fp32_cast_match": solved_exact,
        "other_654_tensors_byte_identical_to_stage1": unchanged_exact,
    }
    print(f"[{seq}/{arm} {time.time()-t0:.1f}s] Reload verification: {result['reload_verification']}", flush=True)
    assert changed_set == expected_32_keys and all_finite and solved_exact and unchanged_exact

    result["total_elapsed_seconds"] = time.time() - t0
    (out_dir / f"RESULT_{seq}_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{seq}/{arm}] DONE in {result['total_elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
