#!/usr/bin/env python3
"""Verify a ConAbl `delta.bin` saves and reloads correctly.

CUIG's CustomDiffusionPipeline.save_pretrained(parameter_group="kv-xattn")
writes {"unet": {param_name: tensor}} containing only the cross-attention
key/value projections (`attn2.to_k`, `attn2.to_v`). This script checks that:

  * the file loads,
  * it contains the expected wrapper and only kv-xattn keys,
  * every key exists in a freshly constructed UNet with a matching shape,
  * loading it into a fresh UNet produces no unexpected keys,
  * the weights actually differ from the base UNet (i.e. training moved them),
  * all values are finite.

Run on CPU: no GPU is required, so this never competes for a shared device.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from diffusers import UNet2DConditionModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True, help="path to delta.bin")
    ap.add_argument("--base_model_dir", required=True, help="diffusers pipeline dir")
    ap.add_argument("--out", required=True, help="where to write the JSON report")
    args = ap.parse_args()

    report: dict = {"delta_path": args.delta, "base_model_dir": args.base_model_dir}

    # --- load the delta -----------------------------------------------------
    blob = torch.load(args.delta, map_location="cpu", weights_only=False)
    report["top_level_keys"] = sorted(blob.keys())
    sd = blob["unet"] if "unet" in blob else blob
    report["num_tensors"] = len(sd)

    kv_only = all(("attn2.to_k" in k) or ("attn2.to_v" in k) for k in sd)
    report["all_keys_are_kv_xattn"] = kv_only
    report["sample_keys"] = sorted(sd.keys())[:4]
    report["all_values_finite"] = all(torch.isfinite(v).all().item() for v in sd.values())
    report["total_params"] = int(sum(v.numel() for v in sd.values()))

    # --- compare against a freshly loaded base UNet -------------------------
    unet = UNet2DConditionModel.from_pretrained(args.base_model_dir, subfolder="unet")
    base = dict(unet.named_parameters())

    missing_in_unet = [k for k in sd if k not in base]
    shape_mismatch = [k for k in sd if k in base and tuple(base[k].shape) != tuple(sd[k].shape)]
    report["keys_absent_from_unet"] = missing_in_unet
    report["shape_mismatches"] = shape_mismatch

    # How many of the saved tensors actually changed relative to the base model?
    changed = 0
    max_abs_delta = 0.0
    for k, v in sd.items():
        if k in base:
            d = (base[k].detach().cpu().float() - v.float()).abs().max().item()
            max_abs_delta = max(max_abs_delta, d)
            if d > 0:
                changed += 1
    report["tensors_differing_from_base"] = changed
    report["max_abs_delta_vs_base"] = max_abs_delta

    # --- actually load it, the way sample.py does ---------------------------
    res = unet.load_state_dict(sd, strict=False)
    report["load_missing_keys"] = len(res.missing_keys)
    report["load_unexpected_keys"] = len(res.unexpected_keys)
    report["unexpected_key_names"] = list(res.unexpected_keys)[:5]

    report["PASS"] = bool(
        kv_only
        and report["all_values_finite"]
        and not missing_in_unet
        and not shape_mismatch
        and report["load_unexpected_keys"] == 0
        and changed > 0
    )

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
