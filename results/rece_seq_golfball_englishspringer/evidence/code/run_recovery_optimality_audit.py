#!/usr/bin/env python3
"""recovery_optimality_audit: does the saved native embedding-recovery
output (E_D / E_S, produced by close_form_emb_regzero on GPU in fp32) solve
its own stated closed-form objective accurately?

CPU-only. No diffusers import, no pipeline instantiation, no text encoding,
no GPU. All three weight sets (W_D, W_S, M0) are read directly as raw
state_dict tensors; the 32 audited K/V layers are identified structurally
(the key set that differs between M0 and W_D), and the 32 layers' .weight
tensors are summed directly -- order of summation is immaterial to A, B
(both are permutation-invariant sums over l), so no module traversal is
needed to match uce.py's own layer-collection order.

Equations, tolerances and scope are frozen in FROZEN_MANIFEST.json
(written before this script ran).
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

t_script_start = time.time()

# ---------------------------------------------------------------------------
# CPU lockdown -- before importing torch
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

assert not torch.cuda.is_available(), "CUDA must be unavailable for this CPU-only audit"
torch.set_num_threads(8)

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent
DIAG_DIR = HERE / "diagnostics"
DIAG_DIR.mkdir(exist_ok=True)

MANIFEST = json.loads((HERE / "FROZEN_MANIFEST.json").read_text())

W_D_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "B_UCE.pt"
W_S_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "AB_UCE.pt"
E_D_PATH = SEQ_DIR / "embedding_cross_swap" / "edits" / "E_D.pt"
E_S_PATH = SEQ_DIR / "embedding_cross_swap" / "edits" / "E_S.pt"
C_PATH = SEQ_DIR / "embedding_cross_swap" / "edits" / "concept_embedding.pt"
M0_PATH = SEQ_DIR / "embedding_cross_swap" / "edits" / "M0_unet_state_dict.pt"

SD14_SNAPSHOT = Path(MANIFEST["inputs"]["M0"]["pinned_sd14_snapshot"])
SD14_UNET_SAFETENSORS = SD14_SNAPSHOT / "unet" / "diffusion_pytorch_model.safetensors"

R = MANIFEST["equations"]["r"]
BWD_TOL = MANIFEST["tolerances"]["reference_normalized_backward_error_max"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fro(t):
    return t.norm().item()


result = {
    "cpu_lockdown": {
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_is_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }
}

# ---------------------------------------------------------------------------
# Step 1: verify the five supplied hashes (live, not reused)
# ---------------------------------------------------------------------------
print("=== Step 1: hash verification ===", flush=True)
hash_checks = {}
for name, path, expected in [
    ("W_D", W_D_PATH, MANIFEST["inputs"]["W_D"]["sha256_supplied"]),
    ("W_S", W_S_PATH, MANIFEST["inputs"]["W_S"]["sha256_supplied"]),
    ("E_D", E_D_PATH, MANIFEST["inputs"]["E_D"]["sha256_supplied"]),
    ("E_S", E_S_PATH, MANIFEST["inputs"]["E_S"]["sha256_supplied"]),
    ("concept_embedding_C", C_PATH, MANIFEST["inputs"]["concept_embedding_C"]["sha256_supplied"]),
]:
    actual = sha256_file(path)
    ok = actual == expected
    hash_checks[name] = {"path": str(path), "expected": expected, "actual": actual, "match": ok}
    print(f"  {name}: {'MATCH' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit(f"STOP: {name} sha256 mismatch -- scoped stop per manifest, no substitution")

# M0 has no supplied hash to verify against; record its current hash
m0_sha = sha256_file(M0_PATH)
hash_checks["M0"] = {"path": str(M0_PATH), "expected": None, "actual": m0_sha,
                      "match": None, "note": "no supplied hash; recorded, not verified"}
result["hash_verification"] = hash_checks

# ---------------------------------------------------------------------------
# Step 2: M0 provenance -- compare its 32 audited projections against the
# pinned SD1.4 snapshot read directly from safetensors (no pipeline).
# ---------------------------------------------------------------------------
print("=== Step 2: M0 provenance check (32 projections vs pinned snapshot) ===", flush=True)
from safetensors.torch import load_file as load_safetensors  # noqa: E402

assert SD14_UNET_SAFETENSORS.is_file(), f"pinned snapshot UNet weights not found: {SD14_UNET_SAFETENSORS}"
snapshot_sha = sha256_file(SD14_UNET_SAFETENSORS)

W_D = torch.load(W_D_PATH, map_location="cpu")
W_S = torch.load(W_S_PATH, map_location="cpu")
M0 = torch.load(M0_PATH, map_location="cpu")

# structural identification of the 32 audited K/V layers: the key set that
# differs between M0 and W_D (cross-checked against W_S)
changed_vs_wd = sorted(k for k in M0 if not torch.equal(M0[k], W_D[k]))
changed_vs_ws = sorted(k for k in M0 if not torch.equal(M0[k], W_S[k]))
assert len(changed_vs_wd) == 32, f"expected 32 changed keys vs W_D, got {len(changed_vs_wd)}"
assert len(changed_vs_ws) == 32, f"expected 32 changed keys vs W_S, got {len(changed_vs_ws)}"
assert all(k.endswith("attn2.to_k.weight") or k.endswith("attn2.to_v.weight") for k in changed_vs_wd)
assert set(changed_vs_wd) == set(changed_vs_ws), "W_D and W_S edit a different set of 32 keys than each other"
KEYS_32 = changed_vs_wd
n_to_v = sum(1 for k in KEYS_32 if k.endswith("to_v.weight"))
n_to_k = sum(1 for k in KEYS_32 if k.endswith("to_k.weight"))

snapshot_sd = load_safetensors(str(SD14_UNET_SAFETENSORS))
m0_vs_snapshot = {}
all_match = True
for k in KEYS_32:
    snap_key = k  # diffusers state_dict keys are identical between pipeline.unet.state_dict() and the raw safetensors file
    if snap_key not in snapshot_sd:
        m0_vs_snapshot[k] = {"present_in_snapshot": False}
        all_match = False
        continue
    a = M0[k].double()
    b = snapshot_sd[snap_key].double()
    same_shape = a.shape == b.shape
    max_abs = (a - b).abs().max().item() if same_shape else None
    exact = same_shape and torch.equal(M0[k], snapshot_sd[snap_key])
    m0_vs_snapshot[k] = {"present_in_snapshot": True, "same_shape": same_shape,
                          "max_abs_diff": max_abs, "bit_identical": exact}
    all_match = all_match and exact

# bias assumption: structural check, not a whitelist -- no corresponding .bias key exists
bias_keys_present = [k.replace(".weight", ".bias") for k in KEYS_32 if k.replace(".weight", ".bias") in M0]

result["m0_provenance"] = {
    "pinned_sd14_snapshot": str(SD14_SNAPSHOT),
    "pinned_sd14_revision": MANIFEST["inputs"]["M0"]["pinned_sd14_revision"],
    "snapshot_unet_safetensors_sha256": snapshot_sha,
    "M0_unet_state_dict_sha256_current": m0_sha,
    "M0_unet_state_dict_sha256_previously_published": False,
    "note": "M0_unet_state_dict.pt was saved by embedding_cross_swap/run_embedding_cross_swap.py (line ~212, ~283) as the untouched .state_dict() of a freshly-loaded StableDiffusionPipeline.from_pretrained(SD14) UNet, detached/cloned/cpu'd, before any edit was applied to that pipeline object. This audit independently re-derives the same 32 projections directly from the pinned snapshot's own safetensors file (no pipeline, no inference) and compares them bit-for-bit against the saved M0_unet_state_dict.pt.",
    "n_audited_keys": len(KEYS_32),
    "n_to_v": n_to_v,
    "n_to_k": n_to_k,
    "all_32_projections_bit_identical_to_pinned_snapshot": all_match,
    "per_key": m0_vs_snapshot,
    "bias_keys_present_for_audited_layers": bias_keys_present,
    "bias_assumption_confirmed_no_bias": len(bias_keys_present) == 0,
}
print(f"  32 audited keys ({n_to_v} to_v + {n_to_k} to_k); M0 vs pinned snapshot: "
      f"{'all bit-identical' if all_match else 'DISAGREEMENT FOUND'}", flush=True)
print(f"  bias keys present for audited layers: {bias_keys_present} (expect none)", flush=True)

if not all_match:
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2))
    raise SystemExit("STOP: M0 provenance disagreement vs pinned snapshot -- reference validity unresolved")

# ---------------------------------------------------------------------------
# Step 3: shape / finiteness / layer-correspondence checks
# ---------------------------------------------------------------------------
print("=== Step 3: shape / finiteness / layer-correspondence checks ===", flush=True)
C = torch.load(C_PATH, map_location="cpu")
if C.dim() == 3:
    C = C[0]
E_D_native = torch.load(E_D_PATH, map_location="cpu")
E_S_native = torch.load(E_S_PATH, map_location="cpu")
if E_D_native.dim() == 3:
    E_D_native = E_D_native[0]
if E_S_native.dim() == 3:
    E_S_native = E_S_native[0]

shape_checks = {
    "C": list(C.shape), "E_D_native": list(E_D_native.shape), "E_S_native": list(E_S_native.shape),
}
assert C.shape == (77, 768)
assert E_D_native.shape == (77, 768)
assert E_S_native.shape == (77, 768)

finiteness = {
    "C": bool(torch.isfinite(C).all()),
    "E_D_native": bool(torch.isfinite(E_D_native).all()),
    "E_S_native": bool(torch.isfinite(E_S_native).all()),
    "M0_all_32": bool(all(torch.isfinite(M0[k]).all() for k in KEYS_32)),
    "W_D_all_32": bool(all(torch.isfinite(W_D[k]).all() for k in KEYS_32)),
    "W_S_all_32": bool(all(torch.isfinite(W_S[k]).all() for k in KEYS_32)),
}
layer_correspondence = {
    "W_D_vs_M0_shapes_match": bool(all(W_D[k].shape == M0[k].shape for k in KEYS_32)),
    "W_S_vs_M0_shapes_match": bool(all(W_S[k].shape == M0[k].shape for k in KEYS_32)),
    "per_layer_dout": {k: list(M0[k].shape) for k in KEYS_32},
}
assert finiteness["C"] and finiteness["E_D_native"] and finiteness["E_S_native"]
assert finiteness["M0_all_32"] and finiteness["W_D_all_32"] and finiteness["W_S_all_32"]
assert layer_correspondence["W_D_vs_M0_shapes_match"] and layer_correspondence["W_S_vs_M0_shapes_match"]

result["shape_finiteness_layer_checks"] = {
    "shapes": shape_checks, "finiteness": finiteness, "layer_correspondence": layer_correspondence,
}
print("  all shape/finiteness/correspondence checks passed", flush=True)

# ---------------------------------------------------------------------------
# Step 4+5: build A, B, D per branch; fp64 Cholesky solve; validate; diagnose
# ---------------------------------------------------------------------------
C64 = C.double()


def build_and_solve(branch_name, W_state, M0_state, keys):
    I768 = torch.eye(768, dtype=torch.float64)
    A = R * I768.clone()
    B = torch.zeros(768, 768, dtype=torch.float64)
    for k in keys:
        Wl = W_state[k].double()
        W0l = M0_state[k].double()
        A += Wl.T @ Wl
        B += Wl.T @ W0l
    D = B @ C64.T  # [768,77]

    eigvals = torch.linalg.eigvalsh(A)
    min_eig, max_eig = eigvals[0].item(), eigvals[-1].item()
    cond = max_eig / min_eig
    norm_A2 = max_eig  # spectral norm of SPD matrix

    L = torch.linalg.cholesky(A)  # raises if not PD
    Yt = torch.cholesky_solve(D, L)  # [768,77], solves A x = D
    Y_star = Yt.T.contiguous()  # [77,768]

    resid = A @ Yt - D
    bwd_err = fro(resid) / (norm_A2 * fro(Y_star) + fro(D))

    return {
        "A": A, "B": B, "D": D, "L": L, "Y_star": Y_star,
        "min_eig": min_eig, "max_eig": max_eig, "cond": cond, "norm_A2": norm_A2,
        "backward_error": bwd_err,
    }


def objective(Y, W_state, M0_state, keys, C64_):
    term1 = R * (Y.norm() ** 2)
    term2 = torch.zeros((), dtype=torch.float64)
    CT = C64_.T
    for k in keys:
        Wl = W_state[k].double()
        W0l = M0_state[k].double()
        diff = Wl @ Y.T - W0l @ CT
        term2 = term2 + diff.norm() ** 2
    return (term1 + term2).item()


def cosine_angle(a, b):
    na, nb = fro(a), fro(b)
    c = (a * b).sum().item() / (na * nb)
    c = max(-1.0, min(1.0, c))
    return c, math.degrees(math.acos(c))


print("=== Step 4: fp64 reference solve, per branch ===", flush=True)
branches = {}
for branch_name, W_state, E_native in [("D", W_D, E_D_native), ("S", W_S, E_S_native)]:
    t0 = time.time()
    sol = build_and_solve(branch_name, W_state, M0, KEYS_32)
    Y_star = sol["Y_star"]
    E64 = E_native.double()

    # quadratic-gap identity
    F_E = objective(E64, W_state, M0, KEYS_32, C64)
    F_Ystar = objective(Y_star, W_state, M0, KEYS_32, C64)
    diff_mat = E64 - Y_star
    quad_lhs = F_E - F_Ystar
    quad_rhs = torch.trace(diff_mat @ sol["A"] @ diff_mat.T).item()
    quad_gap_discrepancy = abs(quad_lhs - quad_rhs)
    quad_gap_tol = 1e-8 * max(1.0, abs(F_E), abs(F_Ystar))

    # native backward error (E instead of Y*)
    resid_native = sol["A"] @ E64.T - sol["D"]
    native_bwd_err = fro(resid_native) / (sol["norm_A2"] * fro(E64) + fro(sol["D"]))

    abs_discrepancy = fro(E64 - Y_star)
    rel_discrepancy = abs_discrepancy / fro(Y_star)
    mag_native, mag_ystar = fro(E64), fro(Y_star)
    cos, angle_deg = cosine_angle(E64, Y_star)

    per_token = []
    for t in range(77):
        et, yt = E64[t], Y_star[t]
        d = fro(et - yt)
        tc, ta = cosine_angle(et.unsqueeze(0), yt.unsqueeze(0)) if fro(et) > 0 and fro(yt) > 0 else (None, None)
        per_token.append({"token": t, "abs_diff": d, "native_norm": fro(et), "ystar_norm": fro(yt),
                           "cosine": tc, "angle_deg": ta})

    # fp32-cast reference: round Y* to fp32 and back, as an output-rounding floor
    Y_star_fp32cast = Y_star.float().double()
    resid_cast = sol["A"] @ Y_star_fp32cast.T - sol["D"]
    cast_bwd_err = fro(resid_cast) / (sol["norm_A2"] * fro(Y_star_fp32cast) + fro(sol["D"]))
    F_cast = objective(Y_star_fp32cast, W_state, M0, KEYS_32, C64)
    cast_obj_gap = abs(F_cast - F_Ystar)
    cast_vs_native_abs_diff = fro(Y_star_fp32cast - Y_star)

    branch_result = {
        "n_layers": len(KEYS_32),
        "min_eig": sol["min_eig"], "max_eig": sol["max_eig"], "cond_A": sol["cond"],
        "reference_backward_error": sol["backward_error"],
        "reference_backward_error_within_tol": sol["backward_error"] <= BWD_TOL,
        "F_native_E": F_E, "F_Y_star": F_Ystar,
        "quadratic_gap_lhs_F(E)_minus_F(Ystar)": quad_lhs,
        "quadratic_gap_rhs_trace_identity": quad_rhs,
        "quadratic_gap_discrepancy": quad_gap_discrepancy,
        "quadratic_gap_tolerance": quad_gap_tol,
        "quadratic_gap_within_tol": quad_gap_discrepancy <= quad_gap_tol,
        "native_backward_error": native_bwd_err,
        "abs_forward_discrepancy_native_vs_Ystar": abs_discrepancy,
        "rel_forward_discrepancy_native_vs_Ystar": rel_discrepancy,
        "magnitude_native_E": mag_native, "magnitude_Y_star": mag_ystar,
        "magnitude_ratio_native_over_Ystar": mag_native / mag_ystar,
        "cosine_native_vs_Ystar": cos, "angle_deg_native_vs_Ystar": angle_deg,
        "fp32_cast_reference": {
            "note": "Y* rounded to fp32 and back to fp64 -- an output-rounding floor, NOT held to the fp64 1e-10 backward-error tolerance.",
            "backward_error": cast_bwd_err,
            "objective_gap_vs_Ystar": cast_obj_gap,
            "abs_diff_vs_Ystar": cast_vs_native_abs_diff,
        },
        "per_token": per_token,
        "elapsed_seconds": time.time() - t0,
    }
    branches[branch_name] = {"sol": sol, "E64": E64, "result": branch_result}
    print(f"  branch {branch_name}: cond(A)={sol['cond']:.6e}, ref_bwd_err={sol['backward_error']:.3e}, "
          f"native_bwd_err={native_bwd_err:.3e}, rel_fwd_discrepancy={rel_discrepancy:.4%}, "
          f"quad_gap_ok={quad_gap_discrepancy <= quad_gap_tol}", flush=True)

result["branches"] = {k: v["result"] for k, v in branches.items()}

reference_valid = all(
    v["result"]["reference_backward_error_within_tol"] and v["result"]["quadratic_gap_within_tol"]
    for v in branches.values()
)
result["reference_solve_fully_valid"] = reference_valid

if not reference_valid:
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2))
    raise SystemExit("STOP: reference-solve validation failed for at least one branch -- unresolved, no tolerance relaxation")

# ---------------------------------------------------------------------------
# Step 6: direct-vs-sequential contrast, before and after accurate recovery
# ---------------------------------------------------------------------------
print("=== Step 6: direct-vs-sequential embedding contrast (native vs accurate) ===", flush=True)
E_D64, E_S64 = branches["D"]["E64"], branches["S"]["E64"]
YD, YS = branches["D"]["sol"]["Y_star"], branches["S"]["sol"]["Y_star"]

native_cos, native_angle = cosine_angle(E_D64, E_S64)
accurate_cos, accurate_angle = cosine_angle(YD, YS)

contrast = {
    "native_norm_D": fro(E_D64), "native_norm_S": fro(E_S64),
    "native_norm_ratio_D_over_S": fro(E_D64) / fro(E_S64),
    "native_cosine_D_vs_S": native_cos, "native_angle_deg_D_vs_S": native_angle,
    "accurate_norm_D": fro(YD), "accurate_norm_S": fro(YS),
    "accurate_norm_ratio_D_over_S": fro(YD) / fro(YS),
    "accurate_cosine_D_vs_S": accurate_cos, "accurate_angle_deg_D_vs_S": accurate_angle,
    "norm_ratio_shift": (fro(YD) / fro(YS)) - (fro(E_D64) / fro(E_S64)),
    "angle_shift_deg": accurate_angle - native_angle,
}
result["direct_vs_sequential_contrast"] = contrast
print(f"  native:   ||E_D||/||E_S||={contrast['native_norm_ratio_D_over_S']:.4f}, angle={native_angle:.3f} deg", flush=True)
print(f"  accurate: ||Y*_D||/||Y*_S||={contrast['accurate_norm_ratio_D_over_S']:.4f}, angle={accurate_angle:.3f} deg", flush=True)

# ---------------------------------------------------------------------------
# Step 7: save diagnostic tensors locally (never applied to any model)
# ---------------------------------------------------------------------------
print("=== Step 7: saving diagnostic reference tensors (not applied to any model) ===", flush=True)
diag_hashes = {}
for name, tensor in [
    ("Y_STAR_D_fp64", YD), ("Y_STAR_S_fp64", YS),
    ("Y_STAR_D_fp32cast", YD.float()), ("Y_STAR_S_fp32cast", YS.float()),
]:
    p = DIAG_DIR / f"{name}.pt"
    torch.save(tensor.clone(), p)
    diag_hashes[name] = {"path": str(p), "sha256": sha256_file(p), "shape": list(tensor.shape),
                          "dtype": str(tensor.dtype)}
result["diagnostic_tensors_saved"] = diag_hashes
result["diagnostic_tensors_authorized_use"] = (
    "Diagnostic reference solutions only. NOT applied to any model, NOT used to build or modify any "
    "UNet checkpoint, NOT used to generate any image in this task."
)

result["timing"] = {"total_script_seconds": time.time() - t_script_start}
result["rece_source_pin"] = MANIFEST["rece_source"]

(HERE / "RESULT.json").write_text(json.dumps(result, indent=2))
print(f"=== DONE in {result['timing']['total_script_seconds']:.1f}s -- RESULT.json written ===", flush=True)
