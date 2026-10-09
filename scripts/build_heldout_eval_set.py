#!/usr/bin/env python3
"""Build a labelled held-out image set for validating the UnlearnCanvas classifiers.

Source: the official UnlearnCanvas dataset on HuggingFace
(`OPTML-Group/UnlearnCanvas`), which *is* reachable from this host even though
the Google Drive checkpoint folders are not. Each row carries an image plus a
caption of the form:

    "A {Object} image in {Style} style"        (or "An {Object} ..." )

so ground-truth style and object labels are recoverable exactly. These are real
benchmark images, never produced by our pipeline and never used for unlearning,
which makes them a legitimate held-out probe for two questions a classifier
checkpoint must answer before it can be trusted:

  1. is its output class ordering the ordering `evaluate.py` assumes?
  2. is its accuracy high enough to be the benchmark's evaluator at all?

Images are decoded, converted to RGB and downscaled (default 512 px on the long
edge) purely to keep the set small; the classifier transform resizes to 224×224
anyway, so this does not change what is being measured in any material way. The
downscale factor is recorded in the manifest.

Usage:
    python scripts/build_heldout_eval_set.py --shards 6 --per-class 4
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CAPTION_RE = re.compile(r"^\s*An?\s+(?P<obj>.+?)\s+image\s+in\s+(?P<style>.+?)\s+style\s*$",
                        re.IGNORECASE)

REPO_ID = "OPTML-Group/UnlearnCanvas"
SHARD_FMT_URL = "https://huggingface.co/api/datasets/OPTML-Group/UnlearnCanvas"


def list_shards() -> list[str]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/bijaypandey/cuig_pilot/heldout_eval_set")
    ap.add_argument("--shards", type=int, default=6,
                    help="number of parquet shards to pull (each ~0.5 GB)")
    ap.add_argument("--per-class", type=int, default=4,
                    help="max images per (style, object) pair")
    ap.add_argument("--max-side", type=int, default=512,
                    help="downscale long edge to this many pixels")
    ap.add_argument("--cache", default="/data/bijaypandey/cuig_pilot/uc_dataset_probe")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from PIL import Image
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print("       pip install pyarrow huggingface_hub pillow", file=sys.stderr)
        return 2

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Shards are ordered by style, so taking the first N would cover only one or
    # two styles. Spread the selection evenly across the whole list instead, so
    # the held-out set probes many style indices rather than a contiguous block.
    all_shards = list_shards()
    n = min(args.shards, len(all_shards))
    step = max(1, len(all_shards) // n)
    shards = [all_shards[i * step] for i in range(n)]
    print(f"using {len(shards)} shard(s) of {len(all_shards)} (stride {step} for style spread)")

    kept: list[dict] = []
    per_pair: Counter = Counter()
    unparsed = 0

    for shard in shards:
        print(f"  fetching {shard}")
        path = hf_hub_download(REPO_ID, repo_type="dataset", filename=shard,
                               local_dir=args.cache)
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            texts = table.column("text").to_pylist()
            images = table.column("image").to_pylist()
            for caption, blob in zip(texts, images):
                m = CAPTION_RE.match(caption or "")
                if not m:
                    unparsed += 1
                    continue
                style = m.group("style").strip().replace(" ", "_")
                obj = m.group("obj").strip().replace(" ", "_")
                key = (style, obj)
                if per_pair[key] >= args.per_class:
                    continue

                raw = blob["bytes"] if isinstance(blob, dict) else blob
                if raw is None:
                    continue
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                if max(im.size) > args.max_side:
                    scale = args.max_side / max(im.size)
                    im = im.resize((round(im.width * scale), round(im.height * scale)),
                                   Image.LANCZOS)

                idx = per_pair[key]
                fname = f"{style}__{obj}__{idx:02d}.jpg"
                im.save(img_dir / fname, quality=95)
                per_pair[key] += 1
                kept.append({"file": fname, "style": style, "object": obj,
                             "caption": caption})

    by_style: dict[str, int] = defaultdict(int)
    by_object: dict[str, int] = defaultdict(int)
    for r in kept:
        by_style[r["style"]] += 1
        by_object[r["object"]] += 1

    manifest = {
        "description": "Labelled held-out images for UnlearnCanvas classifier validation",
        "source": {"hf_dataset": REPO_ID, "shards_used": shards},
        "label_source": "parsed from the dataset's own caption field",
        "caption_regex": CAPTION_RE.pattern,
        "never_used_for": ["unlearning", "anchor generation", "any training"],
        "processing": {"converted_to": "RGB", "max_side_px": args.max_side,
                       "note": "downscale only reduces storage; evaluate.py resizes to 224x224"},
        "counts": {"images": len(kept), "styles": len(by_style), "objects": len(by_object),
                   "unparsed_captions": unparsed},
        "images_per_style": dict(sorted(by_style.items())),
        "images_per_object": dict(sorted(by_object.items())),
        "records": kept,
    }
    (out / "labels.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {len(kept)} images to {img_dir}")
    print(f"      {len(by_style)} distinct styles, {len(by_object)} distinct objects")
    print(f"      manifest: {out / 'labels.json'}")
    if unparsed:
        print(f"      {unparsed} caption(s) did not parse and were skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
