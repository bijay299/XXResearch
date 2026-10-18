#!/usr/bin/env python3
"""Shared helpers for modern_object_assets_v1.

Nothing here changes GEM's training mathematics, optimizer, target/anchor
construction, trainable subset or sampling method. These are provenance,
digest and path helpers only.
"""
import hashlib
import json
import os

import torch

SNAPSHOT = ("/data/bijaypandey/cuig_pilot/hf_home/hub/"
            "models--black-forest-labs--FLUX.1-dev/snapshots/"
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
BASE_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
GEM_COMMIT = "c328fc6bec687efe60223498d1b6bed2e5fbcffd"
ROOT = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1"


def pin_base_model_path():
    """PATCH P2 (mechanical, path-only).

    GEM resolves the FLUX backbone through
    MODEL_CONFIGS['flux']['model_name_or_path'] == 'black-forest-labs/FLUX.1-dev',
    which would be resolved against the Hub's *current* `main`. Repointing it at
    the already-cached, verified-pristine snapshot directory for the pinned
    revision makes the run revision-exact and offline-safe, and keeps
    create_pipeline's `base_model_name == config.model_name_or_path` identity
    (so no spurious component overwrite happens for the M0 load, while an
    edited-checkpoint load still correctly takes the overwrite path).

    Returns the previous value.
    """
    from gem.configs.pipeline_config import MODEL_CONFIGS
    prev = MODEL_CONFIGS["flux"]["model_name_or_path"]
    MODEL_CONFIGS["flux"]["model_name_or_path"] = SNAPSHOT
    return prev


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 over the raw little-endian bytes of a tensor's values."""
    t = t.detach().to("cpu").contiguous().flatten()
    u8 = t.view(torch.uint8)
    return hashlib.sha256(memoryview(u8.numpy())).hexdigest()


def state_dict_digest(sd: dict) -> dict:
    """Per-key {shape, dtype, sha256} plus an aggregate digest.

    This is the comparison unit for keys, shapes, dtypes and tensor values.
    """
    per = {}
    for k in sorted(sd.keys()):
        v = sd[k]
        if not torch.is_tensor(v):
            per[k] = {"non_tensor": repr(type(v))}
            continue
        per[k] = {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "numel": int(v.numel()),
            "sha256": tensor_sha256(v),
        }
    blob = json.dumps(per, sort_keys=True).encode()
    return {
        "n_keys": len(per),
        "aggregate_sha256": hashlib.sha256(blob).hexdigest(),
        "keys": per,
    }


def compare_digests(a: dict, b: dict, label_a="A", label_b="B") -> dict:
    """Exact comparison of two state_dict digests."""
    ka, kb = set(a["keys"]), set(b["keys"])
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    shared = sorted(ka & kb)

    shape_mismatch, dtype_mismatch, value_mismatch = [], [], []
    for k in shared:
        ra, rb = a["keys"][k], b["keys"][k]
        if ra.get("shape") != rb.get("shape"):
            shape_mismatch.append(k)
        if ra.get("dtype") != rb.get("dtype"):
            dtype_mismatch.append(k)
        if ra.get("sha256") != rb.get("sha256"):
            value_mismatch.append(k)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "aggregate_sha256_a": a["aggregate_sha256"],
        "aggregate_sha256_b": b["aggregate_sha256"],
        "aggregate_identical": a["aggregate_sha256"] == b["aggregate_sha256"],
        "n_keys_a": a["n_keys"],
        "n_keys_b": b["n_keys"],
        "keys_only_in_a": only_a,
        "keys_only_in_b": only_b,
        "n_shared_keys": len(shared),
        "shape_mismatches": shape_mismatch,
        "dtype_mismatches": dtype_mismatch,
        "n_value_mismatches": len(value_mismatch),
        "value_mismatch_keys": value_mismatch,
        "bitwise_identical": (not only_a and not only_b and not shape_mismatch
                              and not dtype_mismatch and not value_mismatch),
    }


def finite_report(sd: dict, sample_keys=None) -> dict:
    """Confirm trained/reloaded tensors are finite; report min/max/absmean."""
    bad = []
    stats = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        vf = v.detach().to("cpu")
        ok = bool(torch.isfinite(vf).all())
        if not ok:
            bad.append(k)
        if sample_keys and k in sample_keys:
            f = vf.float()
            stats[k] = {"min": float(f.min()), "max": float(f.max()),
                        "absmean": float(f.abs().mean())}
    return {"n_tensors": sum(1 for v in sd.values() if torch.is_tensor(v)),
            "all_finite": not bad, "non_finite_keys": bad, "sampled_stats": stats}


def file_sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=str)
    os.replace(tmp, path)
    return path


def load_cases():
    with open(os.path.join(ROOT, "manifests", "CASES.json")) as fh:
        return json.load(fh)


def load_prod_config():
    with open(os.path.join(ROOT, "configs", "production_config.json")) as fh:
        return json.load(fh)
