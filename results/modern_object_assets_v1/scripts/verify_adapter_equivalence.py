#!/usr/bin/env python3
"""Measure whether the published pre-merge LoRA adapter reconstructs the
published merged endpoint from the pinned base.

Rebuilds the pristine FLUX.1-dev pipeline wrapper at the pinned revision (CPU),
attaches the published adapter with PEFT, merges it exactly the way GEM's own
save path does (merge_and_unload_all_adapters), and compares the resulting
transformer state against the digest of the native merged checkpoint.

The result is reported as measured -- identical or not. No bit-identity between
the unmerged adapter and the merged weights is assumed anywhere.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()

    ev = os.path.join(a.run_dir, "evidence")
    with open(os.path.join(ev, "digest_postmerge_transformer.json")) as fh:
        merged = json.load(fh)

    C.pin_base_model_path()
    from peft import PeftModel
    from gem.configs.pipeline_config import PipelineConfig
    from gem.wrappers.factory import PipelineFactory
    from gem.wrappers.custom_adapter_utils import merge_and_unload_all_adapters

    pipe_config = PipelineConfig.default_for_model_type(model_type="flux")
    pipe_config = replace(pipe_config, model_dtype="bfloat16")
    assert pipe_config.model_name_or_path == C.SNAPSHOT

    print("[adapter] loading pristine base pipeline (CPU) ...")
    t0 = time.time()
    pipe, modified = PipelineFactory.create_pipeline(
        pipe_config, return_local_component_names=True)
    assert not modified, f"pristine load must overwrite nothing, got {modified}"
    wrapper = PipelineFactory.wrap_pipeline(pipe)
    print(f"[adapter] loaded in {time.time()-t0:.1f}s")

    adapter_dir = os.path.join(a.run_dir, "lora_adapter")
    print(f"[adapter] attaching published adapter from {adapter_dir} ...")
    peft_model = PeftModel.from_pretrained(wrapper, adapter_dir, is_trainable=False)
    print("[adapter] merging via GEM's own merge_and_unload_all_adapters ...")
    remerged = merge_and_unload_all_adapters(peft_model)

    print("[adapter] digesting re-merged transformer ...")
    d = C.state_dict_digest(remerged.pipe.transformer.state_dict())
    cmp_ = C.compare_digests(merged, d, "native_merged_checkpoint",
                             "base_plus_published_adapter_remerged")

    out = {
        "question": ("Does the published pre-merge LoRA adapter, re-applied to "
                     "the pinned base and merged the same way, reproduce the "
                     "published merged endpoint?"),
        "base_revision": C.BASE_REVISION,
        "adapter_dir": os.path.relpath(adapter_dir, C.ROOT),
        "comparison": cmp_,
        "measured_result": ("BITWISE_IDENTICAL" if cmp_["bitwise_identical"]
                            else "NOT_BITWISE_IDENTICAL"),
        "n_differing_tensors": cmp_["n_value_mismatches"],
        "interpretation": (
            "If BITWISE_IDENTICAL, the ~29 MiB adapter is a sufficient "
            "substitute for the ~23 GiB merged transformer when combined with "
            "the pinned gated base. If not, only the full native merged "
            "component reproduces the exact published endpoint. This is a "
            "measurement, not an assumption."),
    }
    C.dump_json(out, os.path.join(ev, "adapter_equivalence.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "comparison"}, indent=2))
    print("bitwise_identical:", cmp_["bitwise_identical"],
          "| differing tensors:", cmp_["n_value_mismatches"])


if __name__ == "__main__":
    main()
