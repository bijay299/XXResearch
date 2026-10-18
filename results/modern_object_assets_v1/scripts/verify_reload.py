#!/usr/bin/env python3
"""Save/reload verification for a GEM checkpoint.

Exercises GEM's own documented load path -- PipelineFactory.create_pipeline with
model_name_or_path pointing at the checkpoint directory, which triggers
overwrite_pipeline_components_from_directory against the pinned base -- and then
compares the RELOADED transformer state against the canonical post-merge state
that was digested immediately before the native save.

Runs on CPU on purpose: it is a byte-level comparison, so it consumes no GPU
allocation and decodes no images.

Usage:
  verify_reload.py --run-dir DIR [--out-name reload_check.json]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-name", default="reload_check.json")
    a = ap.parse_args()

    ev = os.path.join(a.run_dir, "evidence")
    pm_path = os.path.join(ev, "digest_postmerge_transformer.json")
    with open(pm_path) as fh:
        postmerge = json.load(fh)

    prev = C.pin_base_model_path()
    print(f"[P2] base pinned: {prev!r} -> {C.SNAPSHOT!r}")

    from dataclasses import replace
    from gem.configs.pipeline_config import PipelineConfig
    from gem.wrappers.factory import PipelineFactory

    pipe_config = PipelineConfig.default_for_model_type(model_type="flux")
    pipe_config = replace(pipe_config, model_dtype="bfloat16",
                          model_name_or_path=a.run_dir)
    print("[reload] create_pipeline via the native documented path (CPU) ...")
    t0 = time.time()
    pipe, modified = PipelineFactory.create_pipeline(
        pipe_config, return_local_component_names=True)
    load_s = time.time() - t0
    print(f"[reload] loaded in {load_s:.1f}s; overwritten components: {modified}")

    sd = pipe.transformer.state_dict()
    print("[reload] digesting reloaded transformer ...")
    t0 = time.time()
    reloaded = C.state_dict_digest(sd)
    print(f"[reload] {reloaded['n_keys']} keys, agg={reloaded['aggregate_sha256'][:16]}..., "
          f"{time.time()-t0:.1f}s")

    cmp_ = C.compare_digests(postmerge, reloaded,
                             "postmerge_before_save", "reloaded_from_disk")
    fin = C.finite_report(sd)

    out = {
        "run_dir": a.run_dir,
        "base_revision": C.BASE_REVISION,
        "base_snapshot": C.SNAPSHOT,
        "gem_commit": C.GEM_COMMIT,
        "load_seconds": round(load_s, 1),
        "overwritten_components": modified,
        "overwrite_included_transformer": "transformer" in (modified or []),
        "comparison_postmerge_vs_reloaded": cmp_,
        "reloaded_finite": fin,
        "verdict": ("SAVE_RELOAD_BITWISE_IDENTICAL" if cmp_["bitwise_identical"]
                    else "SAVE_RELOAD_MISMATCH"),
        "scope_note": (
            "This compares the canonical POST-MERGE transformer state captured "
            "immediately before GEM's native save against the state reloaded "
            "from disk. It is NOT a claim that an unmerged LoRA adapter and the "
            "merged weights are bit-identical."),
    }
    C.dump_json(out, os.path.join(ev, a.out_name))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("comparison_postmerge_vs_reloaded", "reloaded_finite")},
                     indent=2))
    print("bitwise_identical:", cmp_["bitwise_identical"],
          "| value_mismatches:", cmp_["n_value_mismatches"],
          "| all_finite:", fin["all_finite"])
    sys.exit(0 if cmp_["bitwise_identical"] and fin["all_finite"] else 4)


if __name__ == "__main__":
    main()
