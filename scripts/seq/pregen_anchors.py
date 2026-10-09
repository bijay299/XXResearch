#!/usr/bin/env python3
"""Pre-generate a ConAbl anchor cache from the untouched M0 backbone.

Calls CUIG's own `generate_anchor_images_if_needed`, so the cache is produced by
the exact upstream code path that `train_conabl.py` would use, with a fixed seed.

Why pre-generate instead of letting the first training run do it:

  * every arm must see an *identical* cache. MAB and MAB_L2 differ only in the
    L2 penalty, so they must consume the same anchor images in the same order.
  * if the cache is created inline, the run that creates it consumes RNG during
    generation and the run that reuses it does not, so the two would draw
    different training batches from the same seed. Pre-generating puts every arm
    in the "cache already present" state, which makes the seed actually
    reproducible across arms.

Usage:
    python scripts/seq/pregen_anchors.py --anchor Horses --target cat
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuig_root", default=os.environ.get(
        "REPO_ROOT", "/data/bijaypandey/cuig_pilot/CUIG"))
    ap.add_argument("--base_model_dir", default=os.environ.get("SEQ_BASE_MODEL"))
    ap.add_argument("--anchor", required=True, help="anchor concept folder, e.g. Horses")
    ap.add_argument("--anchor_name", required=True, help="anchor word, e.g. horse")
    ap.add_argument("--target", required=True, help="target word, e.g. cat")
    ap.add_argument("--out", required=True, help="anchor cache directory")
    ap.add_argument("--prompts", required=True, help="anchor prompt txt file")
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sample_batch_size", type=int, default=4)
    args_cli = ap.parse_args()

    cuig = Path(args_cli.cuig_root)
    conabl = cuig / "UnlearningMethods" / "ConAbl"
    sys.path.insert(0, str(conabl))
    sys.path.insert(0, str(cuig))

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    from src.data import generate_anchor_images_if_needed  # type: ignore
    from src.utils import setup_concepts_list  # type: ignore

    accelerator = Accelerator()
    set_seed(args_cli.seed)

    # Mirror exactly the fields train_conabl.py would have populated by this point.
    args = SimpleNamespace(
        anchor_target_concepts=f"{args_cli.anchor_name}+{args_cli.target}",
        anchor_dataset_dirs=args_cli.out,
        anchor_prompt_paths=args_cli.prompts,
        concept_configs=None,
        concept_type="object",
        instance_anchor_prompts_file=None,
        instance_anchor_image_paths_file=None,
        previously_unlearned=None,
        num_anchor_images=args_cli.num_images,
        num_anchor_prompts=args_cli.num_prompts,
        prompt_gen_model="openai",          # never reached: prompt file has 200 lines
        base_model_dir=args_cli.base_model_dir,
        hf_revision=None,
        prior_generation_precision="fp16",
        resolution=512,
        sample_batch_size=args_cli.sample_batch_size,
        seed=args_cli.seed,
    )

    setup_concepts_list(args)
    print(f"target_concepts={args.target_concepts} anchor_concepts={args.anchor_concepts}")

    generate_anchor_images_if_needed(args, accelerator, logging.getLogger(__name__))

    out = Path(args_cli.out)
    images = sorted((out / "images").glob("*.jpg"))
    prompts_txt = (out / "prompts.txt").read_text().splitlines() if (out / "prompts.txt").exists() else []
    meta = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "anchor": args_cli.anchor,
        "anchor_name": args_cli.anchor_name,
        "target_used_for_exclusion_string": args_cli.target,
        "backbone_M0": args_cli.base_model_dir,
        "anchor_seed": args_cli.seed,
        "num_images": len(images),
        "num_prompt_lines": len(prompts_txt),
        "prompt_file": args_cli.prompts,
        "generation": {"num_inference_steps": 100, "guidance_scale": 6.0,
                       "scheduler": "DPMSolverMultistep", "resolution": 512,
                       "source": "CUIG src/data/_generate_images_from_prompts"},
        "cache_dir": str(out),
    }
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if len(images) < args_cli.num_images:
        print(f"ERROR: only {len(images)}/{args_cli.num_images} anchor images", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
