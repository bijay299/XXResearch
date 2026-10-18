#!/usr/bin/env python3
"""Run one GEM erasure training on FLUX.1-dev for the frozen configuration.

The native GEM code path does all of the training, merging and saving. This
driver only:
  * pins the backbone to the exact cached revision (path-only, patch P2)
  * builds the frozen configuration
  * OBSERVES: digests the pristine M0 transformer, digests the canonical
    post-merge transformer immediately before the native save, counts optimizer
    updates and records gradient norms / losses
  * serialises the pre-merge native LoRA adapter alongside the native merged
    checkpoint (serialisation only)

It does not alter training mathematics, the optimizer, target/anchor
construction, the trainable subset or the sampling method.

Usage:
  run_train.py --mode smoke|production --run-dir DIR
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["smoke", "production"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-step-budget", type=int, default=None,
                    help="smoke only; production always uses the frozen 250")
    a = ap.parse_args()

    os.makedirs(a.run_dir, exist_ok=True)
    ev = os.path.join(a.run_dir, "evidence")
    os.makedirs(ev, exist_ok=True)

    prod = C.load_prod_config()

    # ---- patch P2: pin the backbone to the exact cached revision ----
    prev = C.pin_base_model_path()
    print(f"[P2] MODEL_CONFIGS['flux']['model_name_or_path']: {prev!r} -> {C.SNAPSHOT!r}")

    from peft import PeftModel
    from gem.configs import AdaptationConfig
    from gem.configs.pipeline_config import PipelineConfig
    from gem.operators.registry import get_config_and_operator
    from gem.wrappers.base_pipeline_wrapper import BasePipelineWrapper
    import gem.train as gem_train

    # ---------------- frozen configuration ----------------
    pipe_config = PipelineConfig.default_for_model_type(model_type="flux")
    pipe_config = replace(pipe_config, model_dtype="bfloat16")      # mandated deviation
    assert pipe_config.num_inference_steps == 28, pipe_config
    assert pipe_config.inference_guidance_scale == 3.5, pipe_config
    assert pipe_config.image_size == 512, pipe_config
    assert pipe_config.model_name_or_path == C.SNAPSHOT

    operator_config, operator = get_config_and_operator("gem")
    # native default adaptation config (lora / qk_dual) -- taken, not overridden
    assert operator_config.adaptation_config.adapter_mode == "lora"
    assert operator_config.adaptation_config.subset_name == "qk_dual"

    if a.mode == "production":
        train_seed = prod["seeds"]["production_train_seed"]
        budget = prod["training"]["max_step_budget"]
        assert budget == 250
    else:
        train_seed = prod["seeds"]["smoke_train_seed"]
        budget = a.max_step_budget if a.max_step_budget is not None else 2

    operator_config = replace(
        operator_config,
        targets=["teapot"],                      # assignment target
        # anchors: left at the pinned source default [EMPTY_CONCEPT] == ['']
        train_seed=train_seed,
        val_seed=prod["seeds"]["val_seed"],
        max_step_budget=budget,
        max_parallel_timesteps=1,                # mandated deviation (source default 6)
        use_wandb=False,
    )
    assert operator_config.anchors == [""], operator_config.anchors
    assert operator_config.max_parallel_timesteps == 1

    C.dump_json({
        "mode": a.mode,
        "pipe_config": asdict(pipe_config),
        "operator_config": asdict(operator_config),
        "gem_commit": C.GEM_COMMIT,
        "base_revision": C.BASE_REVISION,
        "base_snapshot": C.SNAPSHOT,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, os.path.join(ev, "resolved_config.json"))

    # ---------------- observation hooks ----------------
    counters = {"optimizer_updates": 0, "grad_norms": [], "losses": [],
                "m0_digest": None, "postmerge_digest": None}

    import torch.optim as optim
    _orig_step = optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        tot = 0.0
        n_with_grad = 0
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    tot += float(p.grad.detach().float().norm(2) ** 2)
                    n_with_grad += 1
        counters["optimizer_updates"] += 1
        counters["grad_norms"].append(
            {"update": counters["optimizer_updates"],
             "global_grad_l2": tot ** 0.5,
             "n_params_with_grad": n_with_grad})
        return _orig_step(self, *args, **kwargs)

    optim.AdamW.step = counted_step

    # pristine M0 digest, captured before any adaptation touches the model
    _orig_prepare = gem_train.PipelineFactory.prepare_adaptation

    def prepare_hook(adaptation_config, wrapper, adapter_name="default"):
        print("[obs] digesting pristine M0 transformer state ...")
        t0 = time.time()
        d = C.state_dict_digest(wrapper.pipe.transformer.state_dict())
        print(f"[obs] M0 digest: {d['n_keys']} keys, agg={d['aggregate_sha256'][:16]}..., "
              f"{time.time()-t0:.1f}s")
        C.dump_json(d, os.path.join(ev, "digest_M0_transformer.json"))
        counters["m0_digest"] = d["aggregate_sha256"]
        # also record pristine finiteness + a few sampled target weights
        return _orig_prepare(adaptation_config, wrapper, adapter_name)

    gem_train.PipelineFactory.prepare_adaptation = prepare_hook

    # pre-merge native LoRA adapter serialisation
    _orig_merge = gem_train.merge_and_unload_all_adapters

    def merge_hook(wrapper):
        if isinstance(wrapper, PeftModel):
            ad = os.path.join(a.run_dir, "lora_adapter")
            print(f"[obs] saving pre-merge native LoRA adapter -> {ad}")
            wrapper.save_pretrained(ad)
            # record the trained LoRA tensors themselves
            lsd = {k: v for k, v in wrapper.state_dict().items() if "lora_" in k}
            C.dump_json(C.state_dict_digest(lsd),
                        os.path.join(ev, "digest_lora_adapter.json"))
            nz = {k: float(v.detach().float().abs().sum()) for k, v in lsd.items()}
            C.dump_json({"n_lora_tensors": len(lsd),
                         "n_nonzero_tensors": sum(1 for v in nz.values() if v > 0),
                         "abs_sum_by_tensor": nz},
                        os.path.join(ev, "lora_nonzero_report.json"))
        return _orig_merge(wrapper)

    gem_train.merge_and_unload_all_adapters = merge_hook

    # canonical post-merge state, captured immediately before the native save
    _orig_save = BasePipelineWrapper.save_pipeline

    def save_hook(self, path, *args, **kwargs):
        print("[obs] digesting canonical POST-MERGE transformer state "
              "immediately before the native save ...")
        t0 = time.time()
        sd = self.pipe.transformer.state_dict()
        d = C.state_dict_digest(sd)
        print(f"[obs] post-merge digest: {d['n_keys']} keys, "
              f"agg={d['aggregate_sha256'][:16]}..., {time.time()-t0:.1f}s")
        C.dump_json(d, os.path.join(ev, "digest_postmerge_transformer.json"))
        counters["postmerge_digest"] = d["aggregate_sha256"]
        fr = C.finite_report(sd, sample_keys=set(list(d["keys"].keys())[:5]))
        C.dump_json(fr, os.path.join(ev, "finite_postmerge.json"))
        print(f"[obs] post-merge all_finite={fr['all_finite']}")
        return _orig_save(self, path, *args, **kwargs)

    BasePipelineWrapper.save_pipeline = save_hook

    # ---------------- run the native operator ----------------
    t_start = time.time()
    new_pipe_config = gem_train.run_operator(
        operator, operator_config, pipe_config,
        run_id=os.path.basename(a.run_dir), run_save_path=a.run_dir)
    wall = time.time() - t_start

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    counters["wall_seconds"] = round(wall, 2)
    counters["peak_gpu_gib_driver_view"] = round(peak, 2)
    counters["resolved_run_save_path"] = new_pipe_config.model_name_or_path

    # compare M0 vs post-merge: intended weights must have changed
    try:
        with open(os.path.join(ev, "digest_M0_transformer.json")) as fh:
            dm0 = json.load(fh)
        with open(os.path.join(ev, "digest_postmerge_transformer.json")) as fh:
            dpm = json.load(fh)
        cmp_ = C.compare_digests(dm0, dpm, "M0_pristine", "postmerge_before_save")
        targets = [m for m in C.load_prod_config()["adaptation"].get("_", [])]
        cmp_["interpretation"] = (
            "value_mismatch_keys are the transformer tensors that training "
            "actually changed relative to pristine M0.")
        C.dump_json(cmp_, os.path.join(ev, "compare_M0_vs_postmerge.json"))
        print(f"[obs] changed tensors vs M0: {cmp_['n_value_mismatches']} "
              f"(expected 76 qk_dual weights)")
        counters["n_changed_vs_m0"] = cmp_["n_value_mismatches"]
        counters["changed_keys_vs_m0"] = cmp_["value_mismatch_keys"]
    except Exception as e:  # pragma: no cover
        print("[obs] M0/post-merge comparison failed:", e)

    C.dump_json(counters, os.path.join(ev, "run_counters.json"))
    print("[obs] optimizer updates:", counters["optimizer_updates"])
    print("[obs] wrote", os.path.join(ev, "run_counters.json"))


if __name__ == "__main__":
    main()
