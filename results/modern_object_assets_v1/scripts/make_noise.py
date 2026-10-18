#!/usr/bin/env python3
"""Precompute, hash and persist the per-case initial noise latents (CPU only).

Mirrors diffusers' FluxPipeline.prepare_latents exactly for batch_size=1,
512x512, bfloat16:

    height = width = 2 * (512 // (vae_scale_factor=8 * 2)) = 64
    shape  = (1, 16, 64, 64)
    latents = randn_tensor(shape, generator, device='cpu', dtype=bfloat16)
    latents = FluxPipeline._pack_latents(latents, 1, 16, 64, 64)  -> [1, 1024, 64]

Both the M0 and the edited run load these exact tensors from disk and pass them
as `latents=`, so the two conditions provably share identical initial noise.
run_gen.py additionally asserts that the native pipe.prepare_latents call with
the same seed reproduces the identical tensor.
"""
import json
import os
import sys

import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402

VAE_SCALE_FACTOR = 8
NUM_CHANNELS_LATENTS = 16          # transformer.config.in_channels (64) // 4
IMAGE_SIZE = 512
DTYPE = torch.bfloat16


def build_latents(seed: int):
    h = 2 * (IMAGE_SIZE // (VAE_SCALE_FACTOR * 2))
    w = 2 * (IMAGE_SIZE // (VAE_SCALE_FACTOR * 2))
    shape = (1, NUM_CHANNELS_LATENTS, h, w)
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = randn_tensor(shape, generator=g, device=torch.device("cpu"), dtype=DTYPE)
    packed = FluxPipeline._pack_latents(lat, 1, NUM_CHANNELS_LATENTS, h, w)
    return packed, shape, (h, w)


def main():
    out = os.path.join(C.ROOT, "gen", "noise")
    os.makedirs(out, exist_ok=True)
    cases = C.load_cases()["cases"]

    index = []
    seen = set()
    for c in cases:
        seed = c["eval_seed"]
        assert seed not in seen, f"duplicate eval seed {seed}"
        seen.add(seed)
        packed, shape, hw = build_latents(seed)
        p = os.path.join(out, f"{c['case_id']}_seed{seed}_noise.pt")
        torch.save(packed, p)
        rec = {
            "case_id": c["case_id"],
            "eval_seed": seed,
            "prompt": c["prompt"],
            "class": c["class"],
            "category": c["category"],
            "unpacked_shape": list(shape),
            "packed_shape": list(packed.shape),
            "dtype": str(packed.dtype),
            "tensor_sha256": C.tensor_sha256(packed),
            "file": os.path.relpath(p, C.ROOT),
            "file_sha256": C.file_sha256(p),
        }
        index.append(rec)
        print(f"{c['case_id']} seed={seed} packed={list(packed.shape)} "
              f"sha256={rec['tensor_sha256'][:16]}...")

    hashes = [r["tensor_sha256"] for r in index]
    assert len(set(hashes)) == len(hashes), "initial-noise tensors are not distinct"

    C.dump_json({
        "note": ("Per-case initial noise. Identical tensors are loaded by BOTH "
                 "the M0 and the edited condition; matching seeds alone are not "
                 "treated as proof of pairing."),
        "recipe": {
            "mirrors": "diffusers FluxPipeline.prepare_latents (batch=1, 512x512, bfloat16)",
            "vae_scale_factor": VAE_SCALE_FACTOR,
            "num_channels_latents": NUM_CHANNELS_LATENTS,
            "unpacked_shape": [1, NUM_CHANNELS_LATENTS, 64, 64],
            "packed_shape": [1, 1024, 64],
            "dtype": "torch.bfloat16",
            "generator": "torch.Generator(device='cpu').manual_seed(eval_seed)",
        },
        "n_cases": len(index),
        "all_tensor_hashes_distinct": True,
        "cases": index,
    }, os.path.join(C.ROOT, "manifests", "initial_noise_index.json"))
    print(f"\nwrote {len(index)} noise tensors + manifests/initial_noise_index.json")


if __name__ == "__main__":
    main()
