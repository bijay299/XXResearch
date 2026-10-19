"""Shared image-noise definition: identical packed FLUX initial latents for every condition of a case."""
import hashlib

import torch

NOISE_SHAPE = (1, 16, 64, 64)       # unpacked FLUX latent for 512x512 (vae scale 8, 16 channels)


def pack(lat):
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return lat.reshape(b, (h // 2) * (w // 2), c * 4)


def packed_noise(seed):
    g = torch.Generator("cpu").manual_seed(int(seed))
    return pack(torch.randn(NOISE_SHAPE, generator=g, dtype=torch.float32)).contiguous()


def noise_hash(seed):
    return hashlib.sha256(packed_noise(seed).numpy().tobytes()).hexdigest()
