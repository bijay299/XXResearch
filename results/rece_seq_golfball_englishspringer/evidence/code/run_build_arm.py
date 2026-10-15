#!/usr/bin/env python3
"""CPU-only solve-and-build for one preservation-content arm (GOLF or HORN).
Minimal generalization of solver_only_intervention/run_solver_only_build.py:
same three-term objective, same fp64 Cholesky methodology, only the retain
operand R changes (native-encoded 'golf ball' or 'French horn' instead of
blank). W is the untouched AB_UCE weight (same W as AB_STABLE_SOLVE used).
X is the saved native E_S (not recovered again). Quadratic-gap identity
uses the saved AB_STABLE_SOLVE checkpoint as the "candidate" scored under
THIS arm's own (golf- or horn-specific) objective -- a different, new
diagnostic from the already-closed AtoB-vs-AB_STABLE_SOLVE comparison.
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

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
OUT_DIR = ROOT / "preservation_content_control"
VERIF_LOG_PATH = ROOT / "embedding_cross_swap" / "VERIFICATION_LOG.json"

LAMB = 0.1
ERASE = 1.0
PRESERVE = 0.1

W_PATH = ROOT / "refinement_ablation" / "edits" / "AB_UCE.pt"
X_PATH = ROOT / "embedding_cross_swap" / "edits" / "E_S.pt"
G_PATH = ROOT / "solver_only_intervention" / "native_guide_embedding.pt"
COMPARATOR_PATH = ROOT / "solver_only_intervention" / "edits" / "AB_STABLE_SOLVE.pt"

EXPECTED_HASHES = {
    W_PATH: "3a9d86de1339f9c516399fde1a9ab3c6b4ff8851062fb371712d473e7862b550",
    X_PATH: "e6ea282ad27441aacf47b2824c693166b59d669eb6ae9520978f0194af89bdb8",
    COMPARATOR_PATH: "3b491531a71c86c3b97dad0a9a50cd52112e82334951c5e041c49278366e8829",
}

ARM_RETAIN_TEXT = {"GOLF": "golf ball", "HORN": "French horn"}


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
    ap.add_argument("--arm", required=True, choices=["GOLF", "HORN"])
    args = ap.parse_args()
    arm = args.arm

    t0 = time.time()
    assert torch.cuda.is_available() is False, "Refusing: CUDA visible during the CPU-only solve step."

    result = {"arm": arm, "retain_text": ARM_RETAIN_TEXT[arm], "stop_reason": None}

    recheck = {}
    mismatches = []
    for p, expected in EXPECTED_HASHES.items():
        actual = sha256_file(p)
        recheck[str(p)] = actual
        if actual != expected:
            mismatches.append({"path": str(p), "expected": expected, "actual": actual})
    result["input_hash_recheck"] = recheck
    if mismatches:
        result["stop_reason"] = "SCOPED STOP: input hash mismatch."
        result["mismatches"] = mismatches
        (OUT_DIR / f"RESULT_{arm}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    prov = json.loads((OUT_DIR / "RETAIN_ENCODING_PROVENANCE.json").read_text())
    assert prov["gate_pass"] is True, "Refusing: retain-encoding provenance gate did not pass."

    expected_32_keys = set(json.loads(VERIF_LOG_PATH.read_text())["expected_32_cross_attn_kv_keys"])
    assert len(expected_32_keys) == 32

    X = torch.load(X_PATH, map_location="cpu")[0].to(torch.float64)   # [77,768]
    G = torch.load(G_PATH, map_location="cpu").to(torch.float64)       # [77,768]
    R0 = torch.load(OUT_DIR / f"native_retain_{arm}_row0.pt", map_location="cpu").to(torch.float64)
    R1 = torch.load(OUT_DIR / f"native_retain_{arm}_row1.pt", map_location="cpu").to(torch.float64)
    assert torch.equal(R0, R1), f"{arm}: native retain rows differ -- this build assumed equality verified by the provenance check."
    R = R0

    assert torch.isfinite(X).all() and torch.isfinite(G).all() and torch.isfinite(R).all()

    result["operand_norms"] = {
        "||X||_F (E_S)": frob(X), "||G||_F (guide)": frob(G), f"||R||_F ({arm})": frob(R),
    }

    H = LAMB * torch.eye(768, dtype=torch.float64) + ERASE * (X.T @ X) + PRESERVE * (R.T @ R)
    sym_err = (H - H.T).abs().max().item()
    eigvals = torch.linalg.eigvalsh((H + H.T) / 2)
    lam_min, lam_max = eigvals[0].item(), eigvals[-1].item()
    cond = lam_max / lam_min if lam_min > 0 else float("inf")
    H_finite = bool(torch.isfinite(H).all())
    try:
        L = torch.linalg.cholesky(H)
        chol_ok = True
        chol_err = None
    except Exception as e:  # noqa: BLE001
        chol_ok = False
        chol_err = str(e)
        L = None

    result["H_diagnostics"] = {
        "finite": H_finite, "symmetry_error": sym_err, "min_eig": lam_min, "max_eig": lam_max,
        "cond": cond, "chol_ok": chol_ok, "chol_err": chol_err,
    }
    print(f"[{arm} {time.time()-t0:.1f}s] H: finite={H_finite} sym_err={sym_err:.3e} min_eig={lam_min:.6e} "
          f"max_eig={lam_max:.6e} cond={cond:.6e} chol_ok={chol_ok}", flush=True)

    if not chol_ok:
        result["stop_reason"] = "SCOPED STOP: Cholesky factorization of H failed."
        (OUT_DIR / f"RESULT_{arm}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    GtX = G.T @ X
    RtR = R.T @ R

    W_sd = torch.load(W_PATH, map_location="cpu")
    comparator_sd = torch.load(COMPARATOR_PATH, map_location="cpu")  # AB_STABLE_SOLVE, scored under THIS arm's objective
    assert set(W_sd.keys()) == set(comparator_sd.keys())

    layer_rows = []
    new_layers_fp32 = {}
    for key in sorted(expected_32_keys):
        W = W_sd[key].detach().to(torch.float64)
        V_candidate = comparator_sd[key].detach().to(torch.float64)  # AB_STABLE_SOLVE's own solved layer
        assert torch.isfinite(W).all(), f"{key}: non-finite W"

        N = LAMB * W + ERASE * (W @ GtX) + PRESERVE * (W @ RtR)
        Vstar_T = torch.cholesky_solve(N.T, L)
        Vstar = Vstar_T.T
        assert torch.isfinite(Vstar).all(), f"{key}: non-finite V*"

        normH2 = lam_max

        def backward_error(V):
            resid = V @ H - N
            denom = frob(V) * normH2 + frob(N)
            return {"numerator": frob(resid), "denominator": denom,
                    "normalized": (frob(resid) / denom) if denom > 1e-300 else None}

        def J(V):
            reg = LAMB * (V - W).norm(p="fro").item() ** 2
            erase_t = ERASE * (V @ X.T - W @ G.T).norm(p="fro").item() ** 2
            preserve_t = PRESERVE * (V @ R.T - W @ R.T).norm(p="fro").item() ** 2
            return reg, erase_t, preserve_t, reg + erase_t + preserve_t

        J_W = J(W)
        J_candidate = J(V_candidate)
        J_star = J(Vstar)
        bw_star = backward_error(Vstar)
        bw_candidate = backward_error(V_candidate)

        diff = V_candidate - Vstar
        quad_gap = torch.trace(diff @ H @ diff.T).item()
        excess_abs = J_candidate[3] - J_star[3]
        identity_tol = 1e-8 * max(1.0, abs(J_candidate[3]), abs(J_star[3]))
        identity_err = abs(excess_abs - quad_gap)

        Vstar_fp32 = Vstar.to(torch.float32)
        Vstar_fp32_roundtrip = Vstar_fp32.to(torch.float64)
        J_round = J(Vstar_fp32_roundtrip)
        bw_star_fp32 = backward_error(Vstar_fp32_roundtrip)

        new_layers_fp32[key] = Vstar_fp32

        layer_rows.append({
            "key": key, "d_out": W.shape[0],
            "J_at_W": {"reg": J_W[0], "erase": J_W[1], "preserve": J_W[2], "total": J_W[3]},
            "J_at_V_candidate_AB_STABLE_SOLVE": {"reg": J_candidate[0], "erase": J_candidate[1], "preserve": J_candidate[2], "total": J_candidate[3]},
            "J_at_V_star_fp64": {"reg": J_star[0], "erase": J_star[1], "preserve": J_star[2], "total": J_star[3]},
            "J_at_V_star_fp32_roundtrip": {"reg": J_round[0], "erase": J_round[1], "preserve": J_round[2], "total": J_round[3]},
            "backward_error_V_star_fp64": bw_star,
            "backward_error_V_star_fp32_roundtrip": bw_star_fp32,
            "backward_error_V_candidate": bw_candidate,
            "excess_objective_candidate_minus_star_abs": excess_abs,
            "excess_objective_candidate_minus_star_rel": excess_abs / max(1e-30, abs(J_star[3])),
            "quadratic_gap_tr_diff_H_diffT": quad_gap,
            "quadratic_gap_identity_abs_error": identity_err,
            "quadratic_gap_identity_tol": identity_tol,
            "quadratic_gap_identity_pass": identity_err <= identity_tol,
            "reference_backward_error_tol": 1e-10,
            "reference_backward_error_pass": (bw_star["normalized"] is not None and bw_star["normalized"] <= 1e-10),
            "diff_V_candidate_minus_V_star_fro": frob(diff),
            "diff_rel_to_V_star": frob(diff) / max(1e-300, frob(Vstar)),
            "V_star_fp32_minus_fp64_roundtrip_fro": frob(Vstar_fp32.to(torch.float64) - Vstar),
        })
        print(f"[{arm} {time.time()-t0:.1f}s] {key}: ref_bw={bw_star['normalized']:.3e} "
              f"excess_vs_AB_STABLE_SOLVE_rel={layer_rows[-1]['excess_objective_candidate_minus_star_rel']:.4f} "
              f"identity_pass={layer_rows[-1]['quadratic_gap_identity_pass']}", flush=True)

    ref_valid = all(r["reference_backward_error_pass"] for r in layer_rows)
    identity_valid = all(r["quadratic_gap_identity_pass"] for r in layer_rows)
    result["n_layers"] = len(layer_rows)
    result["reference_solve_fully_valid"] = ref_valid
    result["quadratic_gap_identity_fully_valid"] = identity_valid
    result["layers"] = layer_rows

    if not (ref_valid and identity_valid):
        result["stop_reason"] = "SCOPED STOP: reference solve or quadratic-gap identity failed its frozen tolerance for at least one layer."
        (OUT_DIR / f"RESULT_{arm}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
        return

    out_sd = {k: (v.clone() if k not in new_layers_fp32 else new_layers_fp32[k]) for k, v in W_sd.items()}
    out_name = f"AB_RETAIN_{arm}"
    out_path = OUT_DIR / "edits" / f"{out_name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, out_path)
    out_sha = sha256_file(out_path)
    result["out_checkpoint_name"] = out_name
    result["out_checkpoint_path"] = str(out_path)
    result["out_checkpoint_sha256"] = out_sha
    print(f"[{arm} {time.time()-t0:.1f}s] Saved {out_path}, sha256={out_sha}", flush=True)

    reloaded = torch.load(out_path, map_location="cpu")
    assert set(reloaded.keys()) == set(W_sd.keys())
    changed_keys = [k for k in reloaded if not torch.equal(reloaded[k], W_sd[k])]
    changed_set = set(changed_keys)
    all_finite = all(torch.isfinite(reloaded[k]).all().item() for k in reloaded)
    solved_exact = all(torch.equal(reloaded[k], new_layers_fp32[k]) for k in expected_32_keys)
    unchanged_exact = all(torch.equal(reloaded[k], W_sd[k]) for k in reloaded if k not in expected_32_keys)

    result["reload_verification"] = {
        "n_total_tensors": len(reloaded),
        "n_changed_vs_AB_UCE": len(changed_keys),
        "changed_set_equals_expected_32": changed_set == expected_32_keys,
        "extra_changed_keys": sorted(changed_set - expected_32_keys),
        "missing_changed_keys": sorted(expected_32_keys - changed_set),
        "all_tensors_finite": all_finite,
        "solved_32_keys_exact_fp32_cast_match": solved_exact,
        "other_654_tensors_byte_identical_to_AB_UCE": unchanged_exact,
    }
    print(f"[{arm} {time.time()-t0:.1f}s] Reload verification: {result['reload_verification']}", flush=True)

    assert changed_set == expected_32_keys and all_finite and solved_exact and unchanged_exact, "Reload verification failed."

    result["total_elapsed_seconds"] = time.time() - t0
    (OUT_DIR / f"RESULT_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] DONE in {result['total_elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
