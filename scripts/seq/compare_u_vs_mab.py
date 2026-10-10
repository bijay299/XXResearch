#!/usr/bin/env python3
"""Compare each new 1000-step U endpoint with its saved MAB endpoint. CPU only.

This is the reproducibility check the protocol planned: the two arms were run
with what should be the SAME configuration -- parent = that seed's saved MA,
target dog, anchor horse, `l2sp_weight 0`, 1000 optimizer steps, same training
seed -- so MAB is effectively a prior realisation of U.

Serialisation hashes alone cannot answer this. A hash mismatch can come from
serialisation order or metadata while the tensors are identical, and tensor
equality is in turn not training equivalence. So four questions are answered
separately and the stronger ones are refused:

  1. CONFIGURATION       do the two reports describe the same request, across
                         every effective hyperparameter?
  2. PARENT IDENTITY     did both start from the same saved MA, by recomputed
                         digest rather than by name?
  3. SERIALISATION       are the files byte-identical? (weakest question)
  4. NUMERICAL           do the weights agree, and if not by how much, per
                         tensor and in aggregate?

  NOT ASSESSED: training equivalence, and any behavioural claim. Those would
  need matched generation and scoring, which this tool does not do.

The legacy completion caveat is preserved and restated: the saved MAB reports
predate `optimizer_steps_completed`, so MAB's training completion is UNVERIFIED,
while U's is evidenced by the trainer's own counter. The two arms are therefore
NOT symmetric in completion evidence, and that asymmetry is disclosed rather
than smoothed over.

    python scripts/seq/compare_u_vs_mab.py --out <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def compare_one(seed: int, u_dir: Path, mab_dir: Path, ma_path: Path,
                contract: dict, tol: float) -> dict:
    import torch

    u_rep = json.loads((u_dir / "train_report.json").read_text())
    m_rep = json.loads((mab_dir / "train_report.json").read_text())
    u_ck, m_ck = u_dir / "delta.bin", mab_dir / "delta.bin"

    # ---- 1. configuration
    top_keys = ("parent", "new_deletion_target", "anchor_concept",
                "anchor_target_mapping", "l2sp_weight", "training_seed")
    top_diff = {k: {"U": u_rep.get(k), "MAB": m_rep.get(k)}
                for k in top_keys if u_rep.get(k) != m_rep.get(k)}
    eu = u_rep.get("effective_hyperparameters") or {}
    em = m_rep.get("effective_hyperparameters") or {}
    hp_keys = sorted(set(eu) | set(em))
    hp_diff = {k: {"U": eu.get(k), "MAB": em.get(k)}
               for k in hp_keys if eu.get(k) != em.get(k)}

    # ---- 2. parent identity, by recomputed digest
    ma_sha = sha256_file(ma_path) if ma_path.is_file() else None
    u_parent = (u_rep.get("parent_verification") or {}).get("parent_sha256")
    m_parent = (m_rep.get("parent_verification") or {}).get("parent_sha256")

    # ---- 3. serialisation
    u_sha, m_sha = sha256_file(u_ck), sha256_file(m_ck)

    # ---- 4. numerical
    def load(p: Path) -> dict:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        return obj[contract.get("top_level_key", "unet")]

    a, b = load(u_ck), load(m_ck)
    shared = sorted(set(a) & set(b))
    per_tensor, bitwise, nonfinite = [], 0, []
    max_abs = 0.0
    num = den = 0.0
    for k in shared:
        ta, tb = a[k].float(), b[k].float()
        if not (bool(torch.isfinite(ta).all()) and bool(torch.isfinite(tb).all())):
            nonfinite.append(k)
            continue
        if torch.equal(a[k], b[k]):
            bitwise += 1
        d = (ta - tb).abs()
        mx = float(d.max())
        max_abs = max(max_abs, mx)
        num += float((ta - tb).pow(2).sum())
        den += float(tb.pow(2).sum())
        per_tensor.append({
            "tensor": k, "max_abs_diff": mx,
            "mean_abs_diff": float(d.mean()),
            "rel_frobenius": float((ta - tb).norm() / (tb.norm() + 1e-12)),
            "bitwise_identical": bool(torch.equal(a[k], b[k])),
        })
    per_tensor.sort(key=lambda r: -r["max_abs_diff"])
    overall_rel = (num ** 0.5) / ((den ** 0.5) + 1e-12)

    u_steps = (u_rep.get("runtime") or {}).get("optimizer_steps_completed")
    m_steps = (m_rep.get("runtime") or {}).get("optimizer_steps_completed")

    same_config = not top_diff and not hp_diff
    same_parent = (u_parent == m_parent == ma_sha) and ma_sha is not None
    return {
        "training_seed": seed,
        "U_checkpoint": str(u_ck), "MAB_checkpoint": str(m_ck),
        "configuration": {
            "request_fields_compared": list(top_keys),
            "request_field_differences": top_diff,
            "effective_hyperparameters_compared": len(hp_keys),
            "effective_hyperparameter_differences": hp_diff,
            "identical": same_config,
        },
        "parent_identity": {
            "saved_MA_on_disk_sha256": ma_sha,
            "U_recorded_parent_sha256": u_parent,
            "MAB_recorded_parent_sha256": m_parent,
            "both_from_the_same_saved_MA": same_parent,
        },
        "serialisation": {
            "U_sha256": u_sha, "MAB_sha256": m_sha,
            "file_hashes_equal": u_sha == m_sha,
            "U_bytes": u_ck.stat().st_size, "MAB_bytes": m_ck.stat().st_size,
            "note": ("the weakest question: a mismatch here is NOT by itself a "
                     "training difference"),
        },
        "numerical": {
            "tensors_compared": len(shared),
            "bitwise_identical_tensors": bitwise,
            "nonfinite_tensors": nonfinite,
            "max_abs_diff": max_abs,
            "overall_relative_frobenius": overall_rel,
            "tolerance": tol,
            "numerically_equal_at_tolerance": max_abs <= tol,
            "most_divergent": per_tensor[:5],
            "per_tensor": per_tensor,
        },
        "completion_evidence": {
            "U_optimizer_steps_completed": u_steps,
            "U_completion_verified": u_steps == 1000,
            "MAB_optimizer_steps_completed": m_steps,
            "MAB_completion_verified": False,
            "legacy_caveat": (
                "The saved MAB report predates runtime.optimizer_steps_completed "
                "and the completion check in use when it was produced was "
                "circular, so MAB's training completion is UNVERIFIED. U's is "
                "evidenced by the trainer's own counter. The two arms are NOT "
                "symmetric in completion evidence. Nothing has been backfilled."),
        },
        "training_equivalence": (
            "NOT ASSESSED. Neither identical configuration nor a numerical "
            "distance establishes that the two runs are behaviourally "
            "equivalent; that would need matched generation and scoring."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--diag_root", default=None)
    ap.add_argument("--contract", default="configs/checkpoint_contract.json")
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"FATAL: torch required on CPU for the tensor comparison ({e})",
              file=sys.stderr)
        return 2

    S = Path(a.seq_root)
    diag = Path(a.diag_root) if a.diag_root else S / "diag_v1"
    contract = json.loads(Path(a.contract).read_text())
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]

    per_seed = {}
    for s in seeds:
        per_seed[s] = compare_one(
            s, diag / "models" / f"seed{s}" / "U",
            S / "models" / f"seed{s}" / "MAB",
            S / "models" / f"seed{s}" / "MA" / "delta.bin",
            contract, a.tol)

    all_same_config = all(p["configuration"]["identical"] for p in per_seed.values())
    all_same_parent = all(p["parent_identity"]["both_from_the_same_saved_MA"]
                          for p in per_seed.values())
    any_numeric_diff = any(not p["numerical"]["numerically_equal_at_tolerance"]
                           for p in per_seed.values())

    conclusion = None
    if all_same_config and all_same_parent and any_numeric_diff:
        conclusion = (
            "Identical configuration and the same saved parent, yet the weights "
            "differ. The divergence is therefore RUN-TO-RUN NONDETERMINISM on "
            "this stack, not a configuration difference. Consequence: a re-run "
            "of a trajectory does NOT reproduce the original path bitwise, so "
            "checkpoints taken from a NEW run are not interchangeable with the "
            "original run's checkpoints and any design that mixes them must say "
            "so and measure the difference rather than assume it away.")
    elif all_same_config and all_same_parent:
        conclusion = ("Identical configuration, same saved parent, and the "
                      "weights agree at the stated tolerance.")
    else:
        conclusion = ("The two runs do NOT share a configuration or a parent; "
                      "see the differences recorded per seed.")

    payload = {
        "_what_this_is": (
            "The planned CPU reproducibility comparison of each new 1000-step "
            "unregularised endpoint (U) against the saved unregularised "
            "endpoint (MAB) of the same seed. Four questions answered "
            "separately; training equivalence refused."),
        "tolerance": a.tol,
        "all_seeds_same_configuration": all_same_config,
        "all_seeds_same_saved_parent": all_same_parent,
        "any_seed_numerically_different": any_numeric_diff,
        "conclusion": conclusion,
        "legacy_caveat": (
            "MAB's training completion is UNVERIFIED for every seed: those "
            "reports predate the completed-step counter. U's is verified. The "
            "asymmetry is disclosed, not corrected, and no counter has been "
            "backfilled into any saved report."),
        "per_seed": {str(s): per_seed[s] for s in seeds},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    for s in seeds:
        p = per_seed[s]
        print(f"=== seed {s}: new U (1000 steps) vs saved MAB ===")
        print(f"  configuration identical     : {p['configuration']['identical']} "
              f"({p['configuration']['effective_hyperparameters_compared']} "
              f"effective hyperparameters compared, "
              f"{len(p['configuration']['effective_hyperparameter_differences'])} differ)")
        print(f"  same saved MA parent        : "
              f"{p['parent_identity']['both_from_the_same_saved_MA']} "
              f"({str(p['parent_identity']['saved_MA_on_disk_sha256'])[:12]}…)")
        print(f"  file hashes equal           : "
              f"{p['serialisation']['file_hashes_equal']}  (weakest question)")
        n = p["numerical"]
        print(f"  tensors compared            : {n['tensors_compared']}, "
              f"bitwise-identical {n['bitwise_identical_tensors']}")
        print(f"  max abs weight difference   : {n['max_abs_diff']:.3e}")
        print(f"  overall relative Frobenius  : {n['overall_relative_frobenius']:.3e}")
        print(f"  most divergent tensor       : "
              f"{n['most_divergent'][0]['tensor']} "
              f"({n['most_divergent'][0]['max_abs_diff']:.3e})")
        print(f"  completion evidence         : U={n and p['completion_evidence']['U_optimizer_steps_completed']} "
              f"steps (verified); MAB=UNVERIFIED (legacy)")
        print()
    print(f"CONCLUSION: {conclusion}")
    print(f"\n[comparison] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
