#!/usr/bin/env python3
"""No-update reload check.

Builds a NO-OP delta containing M0's own kv-xattn tensors and pushes it through
exactly the same `load_state_dict(..., strict=False)` path a real `delta.bin`
takes, then regenerates a sample of the frozen evaluation images and compares
them to the images M0 produced with no checkpoint loaded at all.

If the loading path itself perturbed outputs, every "change after editing" would
be partly an artefact of loading. Byte-identical images (or detections identical
within the numerical behaviour of this setup) rule that out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while c := fh.read(1 << 20):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--m0_report", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_eval_images import image_seed_for  # type: ignore

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- build the no-op delta from M0's own trainable tensors ---------------
    unet = UNet2DConditionModel.from_pretrained(args.base_model_dir, subfolder="unet")
    noop = {k: v.detach().cpu().clone() for k, v in unet.named_parameters()
            if "attn2.to_k" in k or "attn2.to_v" in k}
    del unet
    noop_path = out / "noop_delta.bin"
    torch.save({"unet": noop}, noop_path)
    print(f"no-op delta: {len(noop)} tensors -> {noop_path}")

    man = json.loads(Path(args.manifest).read_text())
    gs = man["generation_settings"]
    m0 = json.loads(Path(args.m0_report).read_text())
    by_name = {r["image_name"]: r for r in m0["images"]}
    sample = man["records"][:: max(1, len(man["records"]) // args.n)][: args.n]

    device = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model_dir,
                                                   torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = lambda images, **kw: (images, [False] * len(images))

    blob = torch.load(noop_path, map_location="cpu", weights_only=False)["unet"]
    sd = {k: v.to(torch.float16) for k, v in blob.items()}
    res = pipe.unet.load_state_dict(sd, strict=False)
    assert len(res.unexpected_keys) == 0, res.unexpected_keys

    rows = []
    for rec in sample:
        iseed = image_seed_for(rec["prompt_id"], rec["gen_seed"])
        g = torch.Generator(device=device).manual_seed(iseed)
        img = pipe(prompt=rec["prompt"], width=gs["resolution"], height=gs["resolution"],
                   num_inference_steps=gs["num_inference_steps"],
                   guidance_scale=gs["guidance_scale"], generator=g).images[0]
        p = out / rec["image_name"]
        img.save(p, quality=95)
        orig = by_name.get(rec["image_name"], {})
        rows.append({"image_name": rec["image_name"], "prompt_id": rec["prompt_id"],
                     "sha256_noop_reload": sha(p),
                     "sha256_m0_direct": orig.get("sha256"),
                     "identical": sha(p) == orig.get("sha256")})

    n_id = sum(r["identical"] for r in rows)
    report = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "check": "no-update reload preserves outputs",
        "method": ("a no-op delta holding M0's own kv-xattn tensors is loaded through the "
                   "same strict=False path a real delta.bin uses, then the same "
                   "prompt/seed pairs are regenerated"),
        "n_sampled": len(rows),
        "n_byte_identical": n_id,
        "all_identical": n_id == len(rows),
        "noop_delta_tensors": len(noop),
        "images": rows,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"byte-identical: {n_id}/{len(rows)}")
    print(f"report: {args.report}")
    return 0 if n_id == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
