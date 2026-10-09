#!/usr/bin/env python3
"""Generate the frozen 280-image evaluation set for one checkpoint.

Identical generation settings at every checkpoint (manifest-enforced): same
prompt/seed pairs, scheduler, guidance, precision, 512x512, 30 steps. The
per-image latent seed is derived by a fixed documented rule from (prompt_id,
gen_seed), so images are reproducible and matched across checkpoints while
different prompts do not share identical initial noise.

Writes an image manifest with a SHA-256 per image. Missing or failed
generations are recorded explicitly and cause a non-zero exit -- they are never
silently dropped.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode()) & 0xFFFFFFFF


def image_seed_for(prompt_id: str, gen_seed: int) -> int:
    """Fixed, documented derivation. Same for every checkpoint."""
    return (gen_seed * 1_000_003 + stable_hash(prompt_id)) % (2**31 - 1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--unet_ckpt", default=None, help="delta.bin; omit for M0")
    ap.add_argument("--checkpoint_name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap image count")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline

    man = json.loads(Path(args.manifest).read_text())
    gs = man["generation_settings"]
    records = man["records"]
    if args.limit:
        records = records[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = lambda images, **kw: (images, [False] * len(images))

    load_info = {"unet_ckpt": args.unet_ckpt, "applied": False}
    if args.unet_ckpt:
        ck = Path(args.unet_ckpt)
        if not ck.exists():
            print(f"FATAL: checkpoint missing: {ck}", file=sys.stderr)
            return 2
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        sd = blob["unet"] if isinstance(blob, dict) and "unet" in blob else blob
        sd = {k: v.to(dtype) for k, v in sd.items()}
        res = pipe.unet.load_state_dict(sd, strict=False)
        if len(res.unexpected_keys) != 0:
            print(f"FATAL: unexpected keys loading {ck}: {res.unexpected_keys[:5]}",
                  file=sys.stderr)
            return 2
        load_info.update({"applied": True, "tensors": len(sd),
                          "sha256": sha256_file(ck),
                          "unexpected_keys": len(res.unexpected_keys),
                          "missing_keys": len(res.missing_keys)})
        print(f"[load] {ck.name}: {len(sd)} tensors, 0 unexpected")

    rows, failures = [], []
    t0 = time.time()
    for i, rec in enumerate(records):
        path = out_dir / rec["image_name"]
        iseed = image_seed_for(rec["prompt_id"], rec["gen_seed"])
        if path.exists():
            rows.append({**rec, "image_path": str(path), "image_seed": iseed,
                         "sha256": sha256_file(path), "status": "reused"})
            continue
        try:
            g = torch.Generator(device=device).manual_seed(iseed)
            img = pipe(prompt=rec["prompt"], width=gs["resolution"], height=gs["resolution"],
                       num_inference_steps=gs["num_inference_steps"],
                       guidance_scale=gs["guidance_scale"], generator=g).images[0]
            img.save(path, quality=95)
            rows.append({**rec, "image_path": str(path), "image_seed": iseed,
                         "sha256": sha256_file(path), "status": "generated"})
        except Exception as exc:  # recorded, never silently skipped
            failures.append({**rec, "error": repr(exc)})
            print(f"[FAIL] {rec['image_name']}: {exc}", file=sys.stderr)
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(records)}  {el:.0f}s  "
                  f"({el/(i+1):.2f}s/img, eta {(len(records)-i-1)*el/(i+1):.0f}s)", flush=True)

    elapsed = time.time() - t0
    report = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "checkpoint": args.checkpoint_name,
        "base_model_dir": args.base_model_dir,
        "checkpoint_load": load_info,
        "manifest_sha256": man["manifest_sha256"],
        "generation_settings": gs,
        "image_seed_rule": "(gen_seed * 1000003 + crc32(prompt_id)) % (2**31 - 1)",
        "expected_images": len(records),
        "generated_or_reused": len(rows),
        "failures": failures,
        "complete": len(rows) == len(records) and not failures,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(1, len(records)), 3),
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20)
                            if device.type == "cuda" else None,
        "gpu": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "name": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "images": rows,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n[{args.checkpoint_name}] {len(rows)}/{len(records)} images in "
          f"{elapsed:.0f}s ({elapsed/max(1,len(records)):.2f}s/img) -> {args.report}")
    if not report["complete"]:
        print(f"INCOMPLETE: {len(failures)} failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
