#!/usr/bin/env python3
"""GPU half of the native GEM + PEFT bring-up gate. Binding assertions.

Everything verifiable without a device was already asserted by
`bringup_cpu.py`. This run covers only what needs the GPU:

  * mode during student trajectory sampling (`no_grad` is NOT `eval`)
  * loss-dispatch OFF vs ON agreement for the native arm under identical RNG
  * finite intended updates through the real native loop
  * profiling and the whole-workflow budget projection

Memory note (recorded, not hidden): `transformer.enable_gradient_checkpointing()`
is enabled before training. This is a memory/compute trade -- activations are
recomputed -- and is NOT a train/eval change, an optimizer change or a sampler
change. PyTorch's checkpoint preserves RNG state across recompute, so dropout
masks are reproduced.
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

ROOT = os.environ["CPRD_ROOT"]
R, FAILS = {}, []
PROFILE_BUDGET = 22          # processed timesteps for the timing/agreement probe


def chk(name, ok, **f):
    R[name] = {"pass": bool(ok), **f}
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {json.dumps(f, default=str)[:260]}", flush=True)
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
    from gem.operators.erasure.erasure_gem import GEMConfig, GEM
    from cprd import losses as LS

    FLUX_PATH = os.environ["CPRD_BASE_MODEL_PATH__FLUX"]
    pc = PipelineConfig(_allow_init=True, model_type="flux", model_name_or_path=FLUX_PATH)
    t = time.time()
    pipe = PipelineFactory.create_pipeline(pc, model_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    wrapper = PipelineFactory.wrap_pipeline(pipe)
    wrapper = PipelineFactory.prepare_adaptation(
        AdaptationConfig(adapter_mode="lora", subset_name="qk_dual"), wrapper)
    load_s = time.time() - t
    pipe.transformer.enable_gradient_checkpointing()
    torch.cuda.reset_peak_memory_stats()
    chk("gpu_build", True, load_seconds=round(load_s, 1),
        device=str(next(pipe.transformer.parameters()).device),
        gradient_checkpointing=True,
        resident_gib=round(torch.cuda.memory_allocated() / 2**30, 2))

    # ---- spy on the mode used during student trajectory sampling ----
    seen = {}
    inner = wrapper.base_model.model if hasattr(wrapper, "base_model") else wrapper
    cls = type(inner)
    base_fn = cls.partially_denoise_until_timestep

    def spy(self, *a, **k):
        seen["transformer_training"] = bool(self.transformer.training)
        seen["grad_enabled"] = torch.is_grad_enabled()
        seen["dropout_p"] = next((m.p for m in self.transformer.modules()
                                  if m.__class__.__name__ == "Dropout"), None)
        return base_fn(self, *a, **k)

    cls.partially_denoise_until_timestep = spy

    def run(seed, hook, budget=PROFILE_BUDGET):
        cfg = GEMConfig(targets=["stop sign"], max_step_budget=budget,
                        max_parallel_timesteps=1, train_seed=seed, val_seed=seed,
                        use_wandb=False)
        if hook is not None:
            cfg.cpr_loss_hook = hook
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        GEM()(wrapper, cfg)

    init_sd = {k: v.clone() for k, v in adapter_state(wrapper).items()}

    # ---------- dispatch OFF: no hook attribute at all (original code path) ----------
    t = time.time()
    run(4101, None)
    off_s = time.time() - t
    off_hash = state_hash(adapter_state(wrapper))
    peak = torch.cuda.max_memory_allocated() / 2**30

    chk("mode_during_student_sampling",
        seen.get("transformer_training") is True and seen.get("grad_enabled") is False,
        transformer_training=seen.get("transformer_training"),
        grad_enabled=seen.get("grad_enabled"), dropout_p=seen.get("dropout_p"),
        note="native GEM samples the student trajectory after wrapper.train(); "
             "torch.no_grad() does NOT put the module in eval, so native LoRA "
             "dropout (p=0.1) is ACTIVE during sampling. Preserved as native.")

    # ---------- dispatch ON, native arm (hook returns None) ----------
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in init_sd:
                p.copy_(init_sd[n].to(p.device, p.dtype))
    t = time.time()
    run(4101, LS.make_dispatch("native_gem", eta=1.0))
    on_s = time.time() - t
    on_hash = state_hash(adapter_state(wrapper))

    chk("dispatch_off_on_agreement_native_arm", off_hash == on_hash,
        hash_dispatch_off=off_hash, hash_dispatch_on=on_hash,
        seconds_off=round(off_s, 1), seconds_on=round(on_s, 1),
        declared_tolerance="EXACT equality of the resulting native PEFT adapter "
                           "state hash after identical seeds and identical budget. "
                           "This is agreement of the trained adapter state, NOT a "
                           "claim of bit-exact agreement of intermediate tensors.")

    cur = adapter_state(wrapper)
    moved = max(float((cur[k] - init_sd[k]).abs().max()) for k in cur)
    allfin = all(torch.isfinite(v).all().item() for v in cur.values())
    chk("finite_intended_updates", allfin and moved > 0,
        all_finite=allfin, max_abs_update=moved,
        peak_gpu_gib=round(peak, 2))

    # ---------- candidate arm actually drives the native loop ----------
    with torch.no_grad():
        for n, p in wrapper.named_parameters():
            if n in init_sd:
                p.copy_(init_sd[n].to(p.device, p.dtype))
    run(4101, LS.make_dispatch("cpr", eta=1.0))
    cpr_hash = state_hash(adapter_state(wrapper))
    chk("candidate_arm_changes_result", cpr_hash != off_hash,
        hash_cpr=cpr_hash, hash_native=off_hash,
        note="the CPR dispatch produces a different trained adapter than the "
             "native block under identical seeds, confirming the hook is live")

    # ---------- profile + projection ----------
    per_ts = ((off_s + on_s) / 2.0) / PROFILE_BUDGET
    cases = json.load(open(os.path.join(ROOT, "configs", "cases.json")))
    c = cases["counts"]
    n_traj = c["dev_trajectories"] + c["heldout_endpoints"]
    train_h = n_traj * 250 * per_ts / 3600
    gen_h = c["primary_total"] * 5.0 / 3600
    det_h = 2 * c["primary_total"] * 0.08 / 3600
    load_h = (n_traj + 10) * load_s / 3600
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
        note="a projection does NOT replace enforcement; the ledger refuses any "
             "reservation that would breach a cap or the protected reserve")

    R["_summary"] = {"failures": FAILS, "elapsed_s": round(time.time() - t0, 1),
                     "verdict": "BRINGUP_GPU_PASS" if not FAILS else "BRINGUP_GPU_FAIL"}
    json.dump(R, open(os.path.join(ROOT, "artifacts", "bringup_gpu.json"), "w"),
              indent=2, sort_keys=True, default=str)
    print("VERDICT:", R["_summary"]["verdict"])
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
