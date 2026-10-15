#!/usr/bin/env python3
"""CPU-only diagnostics for the embedding_cross_swap 2x2 design. Reads
already-saved tensors (no GPU, no new model computation, no images):
M0's pristine UNet weights, the two row working checkpoints (W_D, W_S),
the four cells' final checkpoints (two existing, two newly built), and
the saved E_D/E_S/concept/guide/retain embeddings. All cross-attn to_k/to_v
layers in this UNet have bias=False, so a layer's projection of an
embedding e is simply e @ W.T -- no live model/module needed, just the
saved weight tensors.

Formulas match FROZEN_MANIFEST.json's diagnostic_formulas block exactly.
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SEQ_DIR = HERE.parent

W_D_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "B_UCE.pt"
W_S_PATH = SEQ_DIR / "refinement_ablation" / "edits" / "AB_UCE.pt"
CELL_PATHS = {
    ("W_D", "E_D"): SEQ_DIR / "edits" / "B_english_springer.pt",
    ("W_S", "E_S"): SEQ_DIR / "edits" / "AtoB_english_springer.pt",
    ("W_D", "E_S"): HERE / "edits" / "D_WITH_SEQ_EMB.pt",
    ("W_S", "E_D"): HERE / "edits" / "S_WITH_DIRECT_EMB.pt",
}
ROW_OLD_PATHS = {"W_D": W_D_PATH, "W_S": W_S_PATH}

EPS = 1e-12


def load_state(p):
    sd = torch.load(p, map_location="cpu")
    return {k: v.float() for k, v in sd.items()}


def frob_norm(t):
    return t.float().norm().item()


def project(embedding_2d, weight):
    """embedding_2d: [seq_len, in_dim]; weight: [out_dim, in_dim] (nn.Linear,
    bias=False). Returns [seq_len, out_dim]."""
    return embedding_2d.float() @ weight.float().T


def main():
    m0_state = load_state(HERE / "edits" / "M0_unet_state_dict.pt")
    w_old = {"W_D": load_state(W_D_PATH), "W_S": load_state(W_S_PATH)}
    cells = {k: load_state(p) for k, p in CELL_PATHS.items()}

    E_D = torch.load(HERE / "edits" / "E_D.pt", map_location="cpu")[0].float()   # [77,768]
    E_S = torch.load(HERE / "edits" / "E_S.pt", map_location="cpu")[0].float()
    concept_embedding = torch.load(HERE / "edits" / "concept_embedding.pt", map_location="cpu")
    if concept_embedding.dim() == 3:
        concept_embedding = concept_embedding[0]
    concept_embedding = concept_embedding.float()
    guide_embedding = torch.load(HERE / "edits" / "guide_embedding.pt", map_location="cpu")[0].float()
    retain_embedding = torch.load(HERE / "edits" / "retain_embedding.pt", map_location="cpu")[0].float()

    EMB = {"E_D": E_D, "E_S": E_S}
    # LAYER_KEYS = the 32 cross-attn to_k/to_v keys that UCE edits ever touch,
    # derived the same way the GPU script did: where W_D differs from pristine M0.
    LAYER_KEYS = sorted(k for k in m0_state if not torch.equal(m0_state[k], w_old["W_D"][k]))
    assert len(LAYER_KEYS) == 32, f"expected 32 cross-attn K/V layer keys, got {len(LAYER_KEYS)}"
    assert all(("attn2.to_k" in k or "attn2.to_v" in k) for k in LAYER_KEYS)

    out = {}

    # --- embedding magnitudes, relative difference, cosine ---
    nD, nS = frob_norm(E_D), frob_norm(E_S)
    diff_norm = (E_D - E_S).float().norm().item()
    cos = (E_D.flatten() @ E_S.flatten()).item() / max(nD * nS, EPS)
    out["embedding_comparison"] = {
        "E_D_norm": nD, "E_S_norm": nS,
        "difference_norm": diff_norm,
        "relative_difference_pct_vs_E_D": 100 * diff_norm / max(nD, EPS),
        "cosine": cos,
        "note": "E_D and E_S differ far more in MAGNITUDE here (norms %.2f vs %.2f, ratio %.2fx) than the M0-vs-A reference comparison in the prior task did (which held magnitude nearly fixed and varied direction only by ~12.7deg)." % (nD, nS, nS / max(nD, EPS)),
    }

    # --- per-cell: edit displacement, recovery-fit residual, erase-fit residual, retain-projection drift ---
    per_cell = {}
    for (row, col), final_state in cells.items():
        old_state = w_old[row]
        e_col = EMB[col]

        # edit displacement
        disp_per_layer = {k: (final_state[k] - old_state[k]).norm().item() for k in LAYER_KEYS}
        disp_total = sum(v * v for v in disp_per_layer.values()) ** 0.5

        # recovery-fit residual: old_row_l(E_col) vs M0_l(concept_embedding)
        rec_res_per_layer = {}
        for k in LAYER_KEYS:
            lhs = project(e_col, old_state[k])          # [77, out_dim]
            rhs = project(concept_embedding, m0_state[k])
            rec_res_per_layer[k] = (lhs - rhs).norm().item()
        rec_res_total = sum(v * v for v in rec_res_per_layer.values()) ** 0.5

        # erase-fit residual: new_l(E_col) vs old_row_l(guide_embedding)
        erase_res_per_layer = {}
        for k in LAYER_KEYS:
            lhs = project(e_col, final_state[k])
            rhs = project(guide_embedding, old_state[k])
            erase_res_per_layer[k] = (lhs - rhs).norm().item()
        erase_res_total = sum(v * v for v in erase_res_per_layer.values()) ** 0.5

        # retain-projection drift: new_l(retain) vs old_row_l(retain)
        retain_drift_per_layer = {}
        for k in LAYER_KEYS:
            lhs = project(retain_embedding, final_state[k])
            rhs = project(retain_embedding, old_state[k])
            retain_drift_per_layer[k] = (lhs - rhs).norm().item()
        retain_drift_total = sum(v * v for v in retain_drift_per_layer.values()) ** 0.5

        per_cell[f"{row},{col}"] = {
            "status": "existing, reused" if (row, col) in (("W_D", "E_D"), ("W_S", "E_S")) else "new",
            "edit_displacement_total": disp_total,
            "recovery_fit_residual_total": rec_res_total,
            "erase_fit_residual_total": erase_res_total,
            "retain_projection_drift_total": retain_drift_total,
        }

    out["per_cell_2x2"] = per_cell
    out["per_cell_2x2_layout"] = {
        "rows": ["W_D (direct)", "W_S (sequential)"],
        "columns": ["E_D (direct-recovered)", "E_S (sequential-recovered)"],
        "diagonal_cells_existing": ["W_D,E_D (= B)", "W_S,E_S (= A->B)"],
        "off_diagonal_cells_new": ["W_D,E_S (= D_WITH_SEQ_EMB)", "W_S,E_D (= S_WITH_DIRECT_EMB)"],
    }

    # full per-layer detail, saved separately (verbose)
    per_layer_detail = {}
    for (row, col), final_state in cells.items():
        old_state = w_old[row]
        e_col = EMB[col]
        per_layer_detail[f"{row},{col}"] = {
            "edit_displacement": {k: (final_state[k] - old_state[k]).norm().item() for k in LAYER_KEYS},
        }
    (HERE / "PER_LAYER_DISPLACEMENT.json").write_text(json.dumps(per_layer_detail, indent=2))

    (HERE / "DIAGNOSTICS.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
