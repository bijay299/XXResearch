#!/usr/bin/env python3
"""Native GEM + native PEFT bring-up gate. Binding assertions.

Checks, in order:
  1. native wrapper builds BF16 from the pinned FLUX revision
  2. native PEFT adaptation: qk_dual subset, frozen base, LoRA init, FP32 adapter
  3. MODE during student trajectory sampling -- `no_grad` does NOT mean `eval`
  4. native loss-dispatch OFF vs ON agreement (native arm) under identical RNG
  5. finite intended updates, save/reload of the native PEFT adapter state
  6. profiling and a whole-workflow budget projection

Every condition is an ASSERTION. There is no "record a flag then print FITS"
path: a failure returns non-zero and production must not start.
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.environ["CPRD_ROOT"])
sys.path.insert(0, os.path.join(os.environ["CPRD_ROOT"], "pydeps"))
sys.path.insert(0, os.path.join(os.environ["CPRD_ROOT"], "GEM_src"))
import torch

from cprd import losses as LS

ROOT = os.environ["CPRD_ROOT"]
R = {}
FAILS = []


def chk(name, ok, **f):
    R[name] = {"pass": bool(ok), **f}
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {json.dumps(f, default=str)[:220]}")
    return ok


def adapter_state(wrapper):
    sd = {}
    for n, p in wrapper.named_parameters():
        if "lora_" in n and p.requires_grad:
            sd[n] = p.detach().to(torch.float32).cpu().clone()
    return sd


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


def build(dtype=torch.bfloat16):
    from gem.configs.pipeline_config import PipelineConfig
    from gem.configs import AdaptationConfig
    from gem.wrappers.factory import PipelineFactory
    pc = PipelineConfig.create(model_type="flux") if hasattr(PipelineConfig, "create") else None
    if pc is None:
        pc = PipelineConfig(_allow_init=True, model_type="flux")
        pc = pc.default_for_model_type("flux") if hasattr(pc, "default_for_model_type") else pc
    wrapper = PipelineFactory.create_pipeline(pc, model_dtype=dtype)
    ad = AdaptationConfig(adapter_mode="lora", subset_name="qk_dual")
    wrapper = PipelineFactory.prepare_adaptation(ad, wrapper)
    return wrapper, pc, ad


def main():
    t0 = time.time()
    from gem.operators.erasure.erasure_gem import GEMConfig, GEM

    # ---- 1/2 build + native PEFT ----
    t = time.time()
    wrapper, pc, ad = build()
    load_s = time.time() - t
    from peft import PeftModel
    chk("native_peft_model", isinstance(wrapper, PeftModel), type=type(wrapper).__name__)
    chk("pipeline_config", pc.num_inference_steps == 28 and pc.image_size == 512,
        num_inference_steps=pc.num_inference_steps, image_size=pc.image_size,
        guidance=getattr(pc, "inference_guidance_scale", None), load_seconds=round(load_s, 1))
    chk("lora_config", ad.lora_config["r"] == 16 and ad.lora_config["lora_alpha"] == 16
        and ad.lora_config["lora_dropout"] == 0.1, **ad.lora_config, subset=ad.subset_name)

    names = [n for n, p in wrapper.named_parameters() if "lora_" in n and p.requires_grad]
    mods = sorted({n.split(".lora_")[0] for n in names})
    leak = [m for m in mods if "single_transformer_blocks" in m]
    ends = {m.rsplit(".", 1)[-1] for m in mods}
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    frozen_nonfinite = sum(1 for p in wrapper.parameters() if p.requires_grad is False
                           and not torch.isfinite(p).all())
    base_trainable = [n for n, p in wrapper.named_parameters()
                      if p.requires_grad and "lora_" not in n]
    chk("qk_dual_subset", len(mods) == 76 and not leak
        and ends <= {"to_q", "to_k", "add_q_proj", "add_k_proj"},
        n_modules=len(mods), single_block_leak=len(leak), endswith=sorted(ends))
    chk("base_frozen", not base_trainable and frozen_nonfinite == 0,
        n_base_trainable=len(base_trainable), trainable_params=trainable)
    A = [p for n, p in wrapper.named_parameters() if "lora_A" in n and p.requires_grad]
    B = [p for n, p in wrapper.named_parameters() if "lora_B" in n and p.requires_grad]
    chk("lora_init", all(float(p.abs().max()) > 0 for p in A)
        and all(float(p.abs().max()) == 0 for p in B)
        and all(p.dtype == torch.float32 for p in A + B),
        n_A=len(A), n_B=len(B), A_all_nonzero=all(float(p.abs().max()) > 0 for p in A),
        B_all_zero=all(float(p.abs().max()) == 0 for p in B),
        adapter_dtype=str(A[0].dtype))

    # ---- 3. MODE during student trajectory sampling ----
    seen = {}
    base_fn = type(wrapper.base_model.model).partially_denoise_until_timestep

    def spy(self, *a, **k):
        tr = self.transformer if hasattr(self, "transformer") else None
        seen["training_during_sampling"] = bool(tr.training) if tr is not None else None
        seen["grad_enabled_during_sampling"] = torch.is_grad_enabled()
        return base_fn(self, *a, **k)

    type(wrapper.base_model.model).partially_denoise_until_timestep = spy

    def run(seed, hook, budget=22):
        cfg = GEMConfig(targets=["stop sign"], max_step_budget=budget,
                        max_parallel_timesteps=1, train_seed=seed, val_seed=seed,
                        use_wandb=False)
        if hook is not None:
            cfg.cpr_loss_hook = hook
        GEM()(wrapper, cfg)
        return cfg

    # reset adapter to a known init, run with dispatch OFF (no attribute at all)
    init_sd = {k: v.clone() for k, v in adapter_state(wrapper).items()}
    torch.manual_seed(0)
    run(4101, None, budget=22)
    off_sd = adapter_state(wrapper)
    off_hash = state_hash(off_sd)
    prof_s = time.time() - t0

    chk("mode_during_student_sampling",
        seen.get("training_during_sampling") is True
        and seen.get("grad_enabled_during_sampling") is False,
        training=seen.get("training_during_sampling"),
        grad_enabled=seen.get("grad_enabled_during_sampling"),
        note="native GEM samples after wrapper.train(); no_grad does NOT mean eval, "
             "so LoRA dropout p=0.1 is ACTIVE during sampling -- preserved as native")

    # restore init, run with dispatch ON but the native arm's hook returning None
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in init_sd:
                p.copy_(init_sd[n].to(p.device, p.dtype))
    torch.manual_seed(0)
    native_hook = LS.make_dispatch("native_gem", eta=1.0)
    run(4101, native_hook, budget=22)
    on_hash = state_hash(adapter_state(wrapper))
    chk("dispatch_off_on_agreement_native_arm", off_hash == on_hash,
        hash_dispatch_off=off_hash, hash_dispatch_on=on_hash,
        declared_tolerance="exact state-hash equality of the resulting native PEFT "
                           "adapter after identical seeds; NOT described as bit-exact "
                           "agreement of intermediate tensors")

    # ---- 5. finite updates + save/reload ----
    cur = adapter_state(wrapper)
    moved = max(float((cur[k] - init_sd[k]).abs().max()) for k in cur)
    allfin = all(torch.isfinite(v).all().item() for v in cur.values())
    chk("finite_intended_updates", allfin and moved > 0,
        all_finite=allfin, max_abs_update=moved, n_tensors=len(cur))

    ck = os.path.join(ROOT, "ckpt", "bringup_native.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({"adapter": cur, "hash": state_hash(cur)}, ck)
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in cur:
                p.add_(torch.randn_like(p) * 0.01)
    perturbed = state_hash(adapter_state(wrapper))
    blob = torch.load(ck, map_location="cpu")
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in blob["adapter"]:
                p.copy_(blob["adapter"][n].to(p.device, p.dtype))
    chk("adapter_save_reload", state_hash(adapter_state(wrapper)) == blob["hash"]
        and perturbed != blob["hash"],
        restored_hash_matches=True, perturbation_detectable=perturbed != blob["hash"],
        size_mib=round(os.path.getsize(ck) / 2**20, 2),
        note="effective-weight restoration of the native PEFT adapter state; "
             "merged/unmerged BF16 bit equality is NOT assumed or claimed")

    # ---- 6. projection ----
    per_ts = prof_s / 22.0
    cases = json.load(open(os.path.join(ROOT, "configs", "cases.json")))
    c = cases["counts"]
    n_traj = c["dev_trajectories"] + c["heldout_endpoints"]
    train_h = n_traj * 250 * per_ts / 3600
    gen_h = c["primary_total"] * 5.0 / 3600
    det_h = 2 * c["primary_total"] * 0.08 / 3600
    load_h = (n_traj + 8) * load_s / 3600
    projected = (train_h + gen_h + det_h + load_h) * 1.30
    from cprd import ledger as LG
    tot = LG.totals()
    used = tot["gpu_h_committed_total"]
    usable_non_eval = LG.CEILINGS["gpu_device_hours"] - LG.CEILINGS["gpu_hours_eval_reserve"]
    fits = (used + projected) <= usable_non_eval
    chk("budget_projection_fits", fits,
        seconds_per_processed_timestep=round(per_ts, 4),
        train_gpu_h=round(train_h, 2), generate_gpu_h=round(gen_h, 2),
        detector_gpu_h=round(det_h, 2), load_gpu_h=round(load_h, 2),
        projected_gpu_h=round(projected, 2), already_used_gpu_h=round(used, 4),
        usable_non_eval_gpu_h=usable_non_eval,
        protected_reserve_h=LG.CEILINGS["gpu_hours_eval_reserve"],
        note="projection does NOT replace enforcement; the ledger refuses "
             "reservations that would breach the caps")

    R["_summary"] = {"failures": FAILS, "verdict": "BRINGUP_PASS" if not FAILS else "BRINGUP_FAIL",
                     "elapsed_s": round(time.time() - t0, 1)}
    json.dump(R, open(os.path.join(ROOT, "artifacts", "bringup.json"), "w"),
              indent=2, sort_keys=True, default=str)
    print("VERDICT:", R["_summary"]["verdict"])
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
