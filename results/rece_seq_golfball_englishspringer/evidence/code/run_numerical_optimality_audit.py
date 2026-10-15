#!/usr/bin/env python3
"""CPU-only numerical-optimality audit of edit_model_adversarial's closed-form
UCE solve, for all four embedding_cross_swap cells and all 32 edited
cross-attention K/V layers. See FROZEN_MANIFEST.json (written before this
script ran) for the full equations, scope, and frozen tolerances. This
script performs NO GPU allocation, NO image generation, NO training, NO
embedding recovery -- it only loads already-saved tensors and solves a
768x768 linear system per cell.
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # belt-and-suspenders: never allocate CUDA

import torch

torch.set_num_threads(8)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # rece_seq_golfball_englishspringer/ (this script lives in numerical_optimality_audit/ under it)
OUT_DIR = HERE
MANIFEST_PATH = OUT_DIR / "FROZEN_MANIFEST.json"

LAMB = 0.1
ERASE = 1.0
PRESERVE = 0.1

EXPECTED_32_KEYS_FILE = ROOT / "embedding_cross_swap" / "VERIFICATION_LOG.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_embedding_token_by_dim(path: Path) -> torch.Tensor:
    """Loads a saved [1,77,768] embedding tensor, returns [77,768] fp64."""
    t = torch.load(path, map_location="cpu")
    assert t.dtype == torch.float32, f"{path}: expected float32, got {t.dtype}"
    assert tuple(t.shape) == (1, 77, 768), f"{path}: expected shape (1,77,768), got {tuple(t.shape)}"
    assert torch.isfinite(t).all(), f"{path}: contains non-finite values"
    return t[0].to(torch.float64)


def main():
    t_start = time.time()
    manifest = json.loads(MANIFEST_PATH.read_text())
    result = {
        "manifest_sha256_at_run_time": sha256_file(MANIFEST_PATH),
        "cpu_threads": torch.get_num_threads(),
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available_per_torch": torch.cuda.is_available(),
        "stop_reason": None,
    }
    assert result["cuda_available_per_torch"] is False, "Refusing to proceed: CUDA is visible to torch."

    # --- re-verify every input hash against the frozen manifest, live, right now ---
    expected = manifest["input_hashes_sha256_verified_2026-10-14"]
    input_paths = {
        "W_D (refinement_ablation/edits/B_UCE.pt)": ROOT / "refinement_ablation/edits/B_UCE.pt",
        "W_S (refinement_ablation/edits/AB_UCE.pt)": ROOT / "refinement_ablation/edits/AB_UCE.pt",
        "native final B (edits/B_english_springer.pt)": ROOT / "edits/B_english_springer.pt",
        "native final AtoB (edits/AtoB_english_springer.pt)": ROOT / "edits/AtoB_english_springer.pt",
        "crossed final D_WITH_SEQ_EMB (embedding_cross_swap/edits/D_WITH_SEQ_EMB.pt)": ROOT / "embedding_cross_swap/edits/D_WITH_SEQ_EMB.pt",
        "crossed final S_WITH_DIRECT_EMB (embedding_cross_swap/edits/S_WITH_DIRECT_EMB.pt)": ROOT / "embedding_cross_swap/edits/S_WITH_DIRECT_EMB.pt",
        "E_D (embedding_cross_swap/edits/E_D.pt)": ROOT / "embedding_cross_swap/edits/E_D.pt",
        "E_S (embedding_cross_swap/edits/E_S.pt)": ROOT / "embedding_cross_swap/edits/E_S.pt",
        "guide_embedding (embedding_cross_swap/edits/guide_embedding.pt)": ROOT / "embedding_cross_swap/edits/guide_embedding.pt",
        "retain_embedding (embedding_cross_swap/edits/retain_embedding.pt)": ROOT / "embedding_cross_swap/edits/retain_embedding.pt",
    }
    recheck = {}
    mismatches = []
    for key, path in input_paths.items():
        if not path.exists():
            mismatches.append({"key": key, "path": str(path), "issue": "MISSING"})
            continue
        actual = sha256_file(path)
        recheck[key] = actual
        if actual != expected[key]:
            mismatches.append({"key": key, "path": str(path), "expected": expected[key], "actual": actual})
    result["input_verification_recheck"] = recheck
    if mismatches:
        result["stop_reason"] = "SCOPED STOP: input hash/presence mismatch, per task instruction no reconstruction or recovery is performed."
        result["mismatches"] = mismatches
        (OUT_DIR / "RESULT.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    print(f"[{time.time()-t_start:.1f}s] All 10 inputs hash-verified against frozen manifest.", flush=True)

    expected_32_keys = set(json.loads(EXPECTED_32_KEYS_FILE.read_text())["expected_32_cross_attn_kv_keys"])
    assert len(expected_32_keys) == 32

    # --- load the two shared, cell-independent embeddings ---
    G = load_embedding_token_by_dim(input_paths["guide_embedding (embedding_cross_swap/edits/guide_embedding.pt)"])
    R = load_embedding_token_by_dim(input_paths["retain_embedding (embedding_cross_swap/edits/retain_embedding.pt)"])
    E_D = load_embedding_token_by_dim(input_paths["E_D (embedding_cross_swap/edits/E_D.pt)"])
    E_S = load_embedding_token_by_dim(input_paths["E_S (embedding_cross_swap/edits/E_S.pt)"])
    print(f"[{time.time()-t_start:.1f}s] G,R,E_D,E_S loaded, cast to fp64, shapes {G.shape} {R.shape} {E_D.shape} {E_S.shape}", flush=True)

    RtR = R.T @ R  # [768,768], fp64

    def build_H_and_cache(X: torch.Tensor):
        H = LAMB * torch.eye(768, dtype=torch.float64) + ERASE * (X.T @ X) + PRESERVE * RtR
        sym_err = (H - H.T).abs().max().item()
        eigvals = torch.linalg.eigvalsh((H + H.T) / 2)  # symmetrize for the eig solver only; H itself is used unmodified in the solve
        lam_min, lam_max = eigvals[0].item(), eigvals[-1].item()
        cond = lam_max / lam_min if lam_min > 0 else float("inf")
        chol_ok = True
        chol_err = None
        try:
            L = torch.linalg.cholesky(H)
        except Exception as e:  # noqa: BLE001
            chol_ok = False
            chol_err = str(e)
            L = None
        GtX = G.T @ X  # [768,768]
        return {
            "H": H, "L": L, "chol_ok": chol_ok, "chol_err": chol_err,
            "symmetry_error": sym_err, "min_eig": lam_min, "max_eig": lam_max,
            "cond": cond, "GtX": GtX, "finite": bool(torch.isfinite(H).all()),
        }

    H_cache = {"E_D": build_H_and_cache(E_D), "E_S": build_H_and_cache(E_S)}
    for emb_name, c in H_cache.items():
        print(f"[{time.time()-t_start:.1f}s] H[{emb_name}]: finite={c['finite']} sym_err={c['symmetry_error']:.3e} "
              f"min_eig={c['min_eig']:.6e} max_eig={c['max_eig']:.6e} cond={c['cond']:.6e} chol_ok={c['chol_ok']}", flush=True)

    cells = {
        "W_D,E_D (B)": {"W_path": input_paths["W_D (refinement_ablation/edits/B_UCE.pt)"],
                        "V_path": input_paths["native final B (edits/B_english_springer.pt)"],
                        "emb_name": "E_D", "X": E_D},
        "W_D,E_S (D_WITH_SEQ_EMB)": {"W_path": input_paths["W_D (refinement_ablation/edits/B_UCE.pt)"],
                                      "V_path": input_paths["crossed final D_WITH_SEQ_EMB (embedding_cross_swap/edits/D_WITH_SEQ_EMB.pt)"],
                                      "emb_name": "E_S", "X": E_S},
        "W_S,E_D (S_WITH_DIRECT_EMB)": {"W_path": input_paths["W_S (refinement_ablation/edits/AB_UCE.pt)"],
                                         "V_path": input_paths["crossed final S_WITH_DIRECT_EMB (embedding_cross_swap/edits/S_WITH_DIRECT_EMB.pt)"],
                                         "emb_name": "E_D", "X": E_D},
        "W_S,E_S (AtoB)": {"W_path": input_paths["W_S (refinement_ablation/edits/AB_UCE.pt)"],
                            "V_path": input_paths["native final AtoB (edits/AtoB_english_springer.pt)"],
                            "emb_name": "E_S", "X": E_S},
    }

    cell_results = {}
    for cell_name, spec in cells.items():
        t_cell = time.time()
        W_sd = torch.load(spec["W_path"], map_location="cpu")
        V_sd = torch.load(spec["V_path"], map_location="cpu")
        assert set(W_sd.keys()) == set(V_sd.keys()), f"{cell_name}: key set mismatch between W and V checkpoints"
        Hc = H_cache[spec["emb_name"]]
        H, GtX = Hc["H"], Hc["GtX"]

        layer_rows = []
        for key in sorted(expected_32_keys):
            W = W_sd[key].detach().to(torch.float64)
            Vn = V_sd[key].detach().to(torch.float64)
            assert torch.isfinite(W).all() and torch.isfinite(Vn).all(), f"{cell_name}/{key}: non-finite weight"
            assert W.shape == Vn.shape, f"{cell_name}/{key}: shape mismatch {W.shape} vs {Vn.shape}"

            N = LAMB * W + ERASE * (W @ GtX) + PRESERVE * (W @ RtR)  # [d_out,768]

            if Hc["chol_ok"]:
                # Solve H V*^T = N^T via the cached Cholesky factor of H.
                Vstar_T = torch.cholesky_solve(N.T, Hc["L"])
                Vstar = Vstar_T.T
                solve_method = "cholesky"
            else:
                Vstar = None
                solve_method = "FAILED"

            row = {"key": key, "d_out": W.shape[0], "solve_method": solve_method}

            def frob(x):
                return x.norm(p="fro").item()

            X = spec["X"]

            def J(V):
                reg = LAMB * (V - W).norm(p="fro").item() ** 2
                erase_t = ERASE * (V @ X.T - W @ G.T).norm(p="fro").item() ** 2
                preserve_t = PRESERVE * (V @ R.T - W @ R.T).norm(p="fro").item() ** 2
                return reg, erase_t, preserve_t, reg + erase_t + preserve_t

            J_W = J(W)
            J_Vn = J(Vn)
            row["J_at_W"] = {"reg": J_W[0], "erase": J_W[1], "preserve": J_W[2], "total": J_W[3]}
            row["J_at_V_native"] = {"reg": J_Vn[0], "erase": J_Vn[1], "preserve": J_Vn[2], "total": J_Vn[3]}

            normH2 = Hc["max_eig"]  # spectral norm of symmetric PSD H

            def backward_error(V):
                resid = V @ H - N
                denom = frob(V) * normH2 + frob(N)
                if denom < 1e-300:
                    return {"numerator": frob(resid), "denominator": denom, "normalized": None, "undefined": True}
                return {"numerator": frob(resid), "denominator": denom, "normalized": frob(resid) / denom, "undefined": False}

            row["backward_error_V_native"] = backward_error(Vn)

            if Vstar is not None:
                J_Vs = J(Vstar)
                row["J_at_V_star"] = {"reg": J_Vs[0], "erase": J_Vs[1], "preserve": J_Vs[2], "total": J_Vs[3]}
                row["backward_error_V_star"] = backward_error(Vstar)

                diff = Vn - Vstar
                row["diff_V_native_minus_V_star_fro"] = frob(diff)
                normVstar = frob(Vstar)
                normDisp = frob(Vn - W)
                row["diff_rel_to_V_star"] = (frob(diff) / normVstar) if normVstar > 1e-300 else None
                row["diff_rel_to_native_edit_displacement"] = (frob(diff) / normDisp) if normDisp > 1e-300 else None
                row["native_edit_displacement_fro"] = normDisp

                excess_abs = J_Vn[3] - J_Vs[3]
                excess_rel = excess_abs / max(1e-30, abs(J_Vs[3]))
                row["excess_objective_abs"] = excess_abs
                row["excess_objective_rel"] = excess_rel

                quad_gap = torch.trace(diff @ H @ diff.T).item()
                row["quadratic_gap_tr_diff_H_diffT"] = quad_gap
                identity_tol = 1e-8 * max(1.0, abs(J_Vn[3]), abs(J_Vs[3]))
                identity_err = abs(excess_abs - quad_gap)
                row["quadratic_gap_identity_abs_error"] = identity_err
                row["quadratic_gap_identity_tol"] = identity_tol
                row["quadratic_gap_identity_pass"] = identity_err <= identity_tol

                Vstar_fp32_roundtrip = Vstar.to(torch.float32).to(torch.float64)
                J_round = J(Vstar_fp32_roundtrip)
                row["J_at_V_star_fp32_roundtrip"] = {"reg": J_round[0], "erase": J_round[1], "preserve": J_round[2], "total": J_round[3]}
                row["output_rounding_scale_abs"] = J_round[3] - J_Vs[3]

                row["reference_backward_error_tol"] = 1e-10
                row["reference_backward_error_pass"] = (
                    row["backward_error_V_star"]["normalized"] is not None
                    and row["backward_error_V_star"]["normalized"] <= 1e-10
                )
            else:
                row["reference_solve_failed"] = True

            layer_rows.append(row)

        valid_rows = [r for r in layer_rows if r.get("solve_method") == "cholesky"]
        ref_valid = all(r.get("reference_backward_error_pass") for r in valid_rows) and len(valid_rows) == 32
        identity_valid = all(r.get("quadratic_gap_identity_pass") for r in valid_rows)

        def agg_total_J(field):
            return sum(r[field]["total"] for r in layer_rows)

        def agg_quad(values):
            return sum(v ** 2 for v in values) ** 0.5

        backward_norm_native = [r["backward_error_V_native"]["normalized"] for r in layer_rows if not r["backward_error_V_native"]["undefined"]]
        backward_norm_star = [r["backward_error_V_star"]["normalized"] for r in valid_rows if r.get("backward_error_V_star") and not r["backward_error_V_star"]["undefined"]]
        rel_diffs = [r["diff_rel_to_V_star"] for r in valid_rows if r.get("diff_rel_to_V_star") is not None]

        cell_results[cell_name] = {
            "W_path": str(spec["W_path"]), "V_path": str(spec["V_path"]), "embedding_used": spec["emb_name"],
            "H_diagnostics": {k: Hc[k] for k in ("finite", "symmetry_error", "min_eig", "max_eig", "cond", "chol_ok", "chol_err")},
            "n_layers": len(layer_rows),
            "n_layers_solved": len(valid_rows),
            "reference_solve_fully_valid": ref_valid,
            "quadratic_gap_identity_fully_valid": identity_valid,
            "aggregate_J_total": {
                "at_W": agg_total_J("J_at_W"),
                "at_V_native": agg_total_J("J_at_V_native"),
                "at_V_star": (sum(r["J_at_V_star"]["total"] for r in valid_rows) if valid_rows else None),
            },
            "aggregate_native_edit_displacement_fro": agg_quad([r["native_edit_displacement_fro"] for r in valid_rows]) if valid_rows else None,
            "aggregate_diff_V_native_minus_V_star_fro": agg_quad([r["diff_V_native_minus_V_star_fro"] for r in valid_rows]) if valid_rows else None,
            "median_backward_error_normalized_native": sorted(backward_norm_native)[len(backward_norm_native) // 2] if backward_norm_native else None,
            "max_backward_error_normalized_native": max(backward_norm_native) if backward_norm_native else None,
            "median_backward_error_normalized_star": sorted(backward_norm_star)[len(backward_norm_star) // 2] if backward_norm_star else None,
            "max_backward_error_normalized_star": max(backward_norm_star) if backward_norm_star else None,
            "median_diff_rel_to_V_star": sorted(rel_diffs)[len(rel_diffs) // 2] if rel_diffs else None,
            "max_diff_rel_to_V_star": max(rel_diffs) if rel_diffs else None,
            "elapsed_seconds": time.time() - t_cell,
            "layers": layer_rows,
        }
        print(f"[{time.time()-t_start:.1f}s] cell {cell_name}: ref_valid={ref_valid} identity_valid={identity_valid} "
              f"max_backward_err_native={cell_results[cell_name]['max_backward_error_normalized_native']:.3e} "
              f"median_diff_rel={cell_results[cell_name]['median_diff_rel_to_V_star']}", flush=True)
        del W_sd, V_sd

    result["cells"] = cell_results
    result["total_elapsed_seconds"] = time.time() - t_start
    (OUT_DIR / "RESULT.json").write_text(json.dumps(result, indent=2))
    print(f"DONE in {result['total_elapsed_seconds']:.1f}s. Wrote {OUT_DIR / 'RESULT.json'}", flush=True)


if __name__ == "__main__":
    main()
