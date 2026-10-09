#!/usr/bin/env python3
"""Normalised parameter movement over the edited (kv-xattn) tensors.

For each checkpoint X and reference R, over the concatenated trainable tensors:

    relative_l2(X, R) = || vec(X) - vec(R) ||_2  /  || vec(R) ||_2

References are MA (the shared parent of all four second-request children) and M0
(the untouched backbone). M0's values are read from the base UNet itself, since
`delta.bin` stores ABSOLUTE updated values for exactly these tensors.

This is a DIAGNOSTIC ASSOCIATION. Larger movement away from MA alongside larger
historical recovery is a description of the two quantities, not evidence that
weight movement causes recovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from diffusers import UNet2DConditionModel

    def load_delta(p: Path):
        b = torch.load(p, map_location="cpu", weights_only=False)
        return b["unet"] if isinstance(b, dict) and "unet" in b else b

    root = Path(args.models_root)
    names = ["MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]
    deltas = {}
    for n in names:
        p = root / n / "delta.bin"
        if p.exists():
            deltas[n] = load_delta(p)
        else:
            print(f"  missing: {p}")

    if "MA" not in deltas:
        print("MA missing; cannot compute movement")
        return 1

    unet = UNet2DConditionModel.from_pretrained(args.base_model_dir, subfolder="unet")
    base = {k: v.detach().cpu().float() for k, v in unet.named_parameters()
            if k in deltas["MA"]}
    del unet

    def flat(d):
        return torch.cat([d[k].float().flatten() for k in sorted(d)])

    refs = {"M0": flat(base), "MA": flat(deltas["MA"])}

    out = {
        "metric": "relative_l2 = ||vec(X) - vec(R)||_2 / ||vec(R)||_2 over kv-xattn tensors",
        "caveat": ("diagnostic association only; not evidence that weight movement "
                   "causes historical recovery"),
        "n_tensors": len(deltas["MA"]),
        "n_elements": int(sum(v.numel() for v in deltas["MA"].values())),
        "checkpoints": {},
    }
    for n, d in deltas.items():
        v = flat(d)
        row = {}
        for rname, rvec in refs.items():
            row[f"relative_l2_vs_{rname}"] = round(
                float(torch.norm(v - rvec) / torch.norm(rvec)), 6)
            row[f"max_abs_diff_vs_{rname}"] = round(float((v - rvec).abs().max()), 8)
            row[f"mean_abs_diff_vs_{rname}"] = round(float((v - rvec).abs().mean()), 10)
        out["checkpoints"][n] = row
        print(f"  {n:8s} vs M0: {row['relative_l2_vs_M0']:.6f}   "
              f"vs MA: {row['relative_l2_vs_MA']:.6f}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
