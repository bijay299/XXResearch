#!/usr/bin/env python3
"""CPU half of the native GEM + PEFT bring-up gate. No GPU allocation.

The pinned factory never moves the pipeline to a device, so everything except
the training loop itself can be verified here for free: native PEFT adaptation,
the qk_dual subset, frozen base, LoRA initialisation, adapter dtype, and
adapter save/reload. Binding assertions.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.environ["CPRD_ROOT"])
sys.path.insert(0, os.path.join(os.environ["CPRD_ROOT"], "pydeps"))
sys.path.insert(0, os.path.join(os.environ["CPRD_ROOT"], "GEM_src"))
import torch

ROOT = os.environ["CPRD_ROOT"]
R, FAILS = {}, []


def chk(name, ok, **f):
    R[name] = {"pass": bool(ok), **f}
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {json.dumps(f, default=str)[:200]}")
    return ok


def adapter_state(w):
    return {n: p.detach().to(torch.float32).cpu().clone()
            for n, p in w.named_parameters() if "lora_" in n and p.requires_grad}


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


def main():
    t0 = time.time()
    from gem.configs.pipeline_config import PipelineConfig
    from gem.configs import AdaptationConfig
    from gem.wrappers.factory import PipelineFactory
    from gem.operators.erasure.erasure_gem import GEMConfig
    from peft import PeftModel

    # Pass the pinned local snapshot explicitly so base_model_name ==
    # config.model_name_or_path and GEM's "delta checkpoint" overwrite step is
    # correctly skipped (it would otherwise try to list the gated hub repo).
    FLUX_PATH = os.environ["CPRD_BASE_MODEL_PATH__FLUX"]
    pc = PipelineConfig(_allow_init=True, model_type="flux", model_name_or_path=FLUX_PATH)
    chk("pipeline_config", pc.num_inference_steps == 28 and pc.image_size == 512
        and abs(pc.inference_guidance_scale - 3.5) < 1e-9
        and pc.model_name_or_path == FLUX_PATH,
        steps=pc.num_inference_steps, guidance=pc.inference_guidance_scale,
        image_size=pc.image_size, model=pc.model_name_or_path,
        config_dtype=pc.model_dtype,
        note="config dtype is float16; the contract freezes BF16 and we pass "
             "model_dtype=torch.bfloat16 explicitly (recorded deviation)")

    gc = GEMConfig(targets=["stop sign"])
    chk("native_gem_defaults",
        gc.learning_rate == 1e-3 and gc.eta == 1.0 and gc.gamma == 100.0
        and gc.max_step_budget == 250 and gc.partial_denoising_end_at_timestep == 10
        and gc.partial_denoising_mode == "student"
        and gc.partial_denoising_guidance_scale == 3.0 and gc.anchors == [""],
        lr=gc.learning_rate, eta=gc.eta, gamma=gc.gamma, budget=gc.max_step_budget,
        end_at=gc.partial_denoising_end_at_timestep, mode=gc.partial_denoising_mode,
        sampler_guidance=gc.partial_denoising_guidance_scale,
        empty_concept=repr(gc.anchors[0]), internal_steps_fallback=pc.num_inference_steps)

    t = time.time()
    pipe = PipelineFactory.create_pipeline(pc, model_dtype=torch.bfloat16)
    wrapper = PipelineFactory.wrap_pipeline(pipe)      # native wrapper
    build_s = time.time() - t
    # the pinned revision must be the one resolved from the pristine local cache
    import glob
    snaps = glob.glob(os.path.join(os.environ["HF_HOME"], "hub",
                                   "models--black-forest-labs--FLUX.1-dev", "snapshots", "*"))
    chk("flux_revision_pinned",
        len(snaps) == 1 and os.path.basename(snaps[0]) == "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        resolved=[os.path.basename(s) for s in snaps], build_seconds=round(build_s, 1))

    ad = AdaptationConfig(adapter_mode="lora", subset_name="qk_dual")
    chk("lora_config", ad.lora_config == {"r": 16, "lora_alpha": 16,
                                          "lora_dropout": 0.1, "bias": "none"},
        **ad.lora_config, subset=ad.subset_name)

    wrapper = PipelineFactory.prepare_adaptation(ad, wrapper)
    chk("native_peft_model", isinstance(wrapper, PeftModel), type=type(wrapper).__name__)

    names = [n for n, p in wrapper.named_parameters() if "lora_" in n and p.requires_grad]
    mods = sorted({n.split(".lora_")[0] for n in names})
    leak = [m for m in mods if "single_transformer_blocks" in m]
    ends = {m.rsplit(".", 1)[-1] for m in mods}
    base_trainable = [n for n, p in wrapper.named_parameters()
                      if p.requires_grad and "lora_" not in n]
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    chk("qk_dual_subset", len(mods) == 76 and not leak
        and ends <= {"to_q", "to_k", "add_q_proj", "add_k_proj"},
        n_modules=len(mods), single_block_leak=len(leak), endswith=sorted(ends),
        example=mods[0] if mods else None)
    chk("base_frozen", not base_trainable,
        n_base_trainable=len(base_trainable), trainable_params=trainable)

    A = [p for n, p in wrapper.named_parameters() if "lora_A" in n and p.requires_grad]
    B = [p for n, p in wrapper.named_parameters() if "lora_B" in n and p.requires_grad]
    chk("lora_init_and_dtype",
        len(A) == len(B) == 76
        and all(float(p.abs().max()) > 0 for p in A)
        and all(float(p.abs().max()) == 0 for p in B)
        and all(p.dtype == torch.float32 for p in A + B),
        n_A=len(A), n_B=len(B),
        A_all_nonzero=all(float(p.abs().max()) > 0 for p in A),
        B_all_zero=all(float(p.abs().max()) == 0 for p in B),
        adapter_dtype=str(A[0].dtype),
        note="native cast_training_params upcasts trainable params to FP32 on a "
             "BF16 backbone")

    base_dtypes = {str(p.dtype) for n, p in wrapper.named_parameters()
                   if "lora_" not in n}
    chk("backbone_bf16", base_dtypes == {"torch.bfloat16"}, base_param_dtypes=sorted(base_dtypes))

    # ---- adapter save / reload (effective-weight restoration) ----
    sd = adapter_state(wrapper); h0 = state_hash(sd)
    ck = os.path.join(ROOT, "ckpt", "bringup_cpu_adapter.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({"adapter": sd, "hash": h0}, ck)
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in sd:
                p.add_(torch.randn_like(p) * 0.01)
    hp = state_hash(adapter_state(wrapper))
    blob = torch.load(ck, map_location="cpu")
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in blob["adapter"]:
                p.copy_(blob["adapter"][n].to(p.device, p.dtype))
    chk("adapter_save_reload", state_hash(adapter_state(wrapper)) == h0 and hp != h0,
        restored=True, perturbation_detectable=hp != h0,
        size_mib=round(os.path.getsize(ck) / 2**20, 2),
        note="effective-weight restoration of native PEFT adapter state; "
             "merged/unmerged BF16 bit equality is NOT assumed")

    # ---- dispatch wiring: hook attaches to the config and returns None natively
    from cprd import losses as LS
    gc.cpr_loss_hook = LS.make_dispatch("native_gem", eta=1.0)
    v = torch.randn(2, 16); a = torch.randn(2, 16); b = torch.randn(2, 16)
    chk("dispatch_native_returns_none", gc.cpr_loss_hook(v, a, b) is None,
        note="native arm -> trainer executes its ORIGINAL loss block")
    gc.cpr_loss_hook = LS.make_dispatch("cpr", eta=1.0)
    out = gc.cpr_loss_hook(v, a, b)
    chk("dispatch_candidate_returns_tensor",
        out is not None and out.shape == (2,) and out.dtype == torch.float32,
        shape=list(out.shape), dtype=str(out.dtype))

    R["_summary"] = {"failures": FAILS, "elapsed_s": round(time.time() - t0, 1),
                     "verdict": "BRINGUP_CPU_PASS" if not FAILS else "BRINGUP_CPU_FAIL"}
    json.dump(R, open(os.path.join(ROOT, "artifacts", "bringup_cpu.json"), "w"),
              indent=2, sort_keys=True, default=str)
    print("VERDICT:", R["_summary"]["verdict"])
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
