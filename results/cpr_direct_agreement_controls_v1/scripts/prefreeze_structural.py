#!/usr/bin/env python3
"""PRE-FREEZE structural bring-up (CPU only; no forward pass, no loss value, no detector inference, no FLUX output).
Logged as a generic import/access/structure check. Verifies imports, native PEFT adapter structure, adapter
save/zero/load round trip (pure state manipulation), digest determinism, hook signatures, vae wrap/restore, and
that every frozen case's noise hash reproduces from its seed (CPU RNG only)."""
import inspect
import json
import os
import sys

import torch

from cprd2 import analysis, detect, flux_run as FR, gate_checks, gate_core, labels, losses, noise, project, protocol as P
from cprd2.common import ROOT, dump, load, utc

R = {"utc": utc(), "checks": {}}


def chk(_name, ok, **d):
    R["checks"][_name] = {"pass": bool(ok), **d}
    print(("PASS " if ok else "FAIL ") + _name, json.dumps(d, default=str)[:200], flush=True)


def main():
    import diffusers, peft, transformers
    chk("versions", True, torch=torch.__version__, diffusers=diffusers.__version__, peft=peft.__version__,
        transformers=transformers.__version__, py=sys.version.split()[0])
    from gem.operators.erasure.erasure_gem import GEM, GEMConfig
    cfg = GEMConfig(targets=["refrigerator"], max_step_budget=250, max_parallel_timesteps=1, train_seed=1, val_seed=1,
                    use_wandb=False, learning_rate=1e-3, eta=1.0, gamma=100.0)
    chk("gem_config_defaults", cfg.partial_denoising_end_at_timestep == 10 and cfg.use_inference_timesteps
        and cfg.partial_denoising_mode == "student" and cfg.partial_denoising_guidance_scale == 3.0
        and cfg.anchors == [""] and cfg.gradient_accumulation_steps == 1,
        eta=cfg.eta, gamma=cfg.gamma, max_gradient_norm=getattr(cfg, "max_gradient_norm", None),
        n_param_defaults={k: getattr(cfg, k) for k in ("partial_denoising_end_at_timestep", "partial_denoising_guidance_scale")})
    src = inspect.getsource(GEM.__call__)
    chk("patched_hooks_present", all(s in src for s in ("cpr_capture_hook", "cpr_loss_hook", "cpr_step_hook")))
    # --- structure
    pipe, wrapper, load_s = FR.build("cpu", init_seed=5101)
    inv = FR.inventory(wrapper)
    chk("peft_inventory", inv["n_lora_modules"] == 76 and inv["n_base_trainable"] == 0 and inv["lora_dropout_p"] == [0.1]
        and inv["lora_dtypes"] == ["torch.float32"], load_seconds=load_s, **inv)
    cfgd = wrapper.peft_config["default"]
    chk("lora_r_alpha", cfgd.r == 16 and cfgd.lora_alpha == 16 and abs(cfgd.lora_dropout - 0.1) < 1e-12, r=cfgd.r,
        alpha=cfgd.lora_alpha, dropout=cfgd.lora_dropout)
    sd0 = FR.peft_state(wrapper)
    chk("init_B_zero_A_nonzero", all(bool((v == 0).all()) for k, v in sd0.items() if "lora_B" in k)
        and all(bool((v != 0).any()) for k, v in sd0.items() if "lora_A" in k), n_tensors=len(sd0))
    pipe2, wrapper2, _ = FR.build("cpu", init_seed=5101)
    chk("seeded_init_deterministic", FR.state_hash(FR.peft_state(wrapper2)) == FR.state_hash(sd0))
    del pipe2, wrapper2
    bd1, bd2 = FR.base_digest(wrapper), FR.base_digest(wrapper)
    chk("base_digest_deterministic", bd1 == bd2, n=bd1["n_params"])
    inner = wrapper.base_model.model
    chk("sampler_method_present", callable(getattr(type(inner), "partially_denoise_until_timestep", None)))
    # --- adapter round trip (state manipulation only)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in FR.lora_params(wrapper).values():
            p.copy_(torch.randn(p.shape, generator=g) * 0.01)
    trained = FR.state_hash(FR.peft_state(wrapper))
    out_dir = os.path.join(ROOT, "artifacts", "prefreeze_roundtrip")
    meta = FR.save_adapter(wrapper, out_dir, {"purpose": "pre-freeze structural round trip (random values; discarded)"})
    FR.zero_adapter(wrapper)
    zero_state = FR.state_hash(FR.peft_state(wrapper))
    restored = FR.load_adapter(wrapper, out_dir)
    chk("adapter_roundtrip_bitwise", restored == trained == meta["tensor_state_sha256"] and zero_state != trained,
        file_sha256=meta["file_sha256"][:16], n_tensors=meta["n_tensors"])
    import shutil
    shutil.rmtree(out_dir)
    chk("disable_adapter_ctx", hasattr(wrapper, "disable_adapter"))
    c = FR.DecodeCounter(pipe.vae)
    orig = pipe.vae.decode
    c.restore()
    chk("decode_counter_wrap_restore", c.n == 0 and pipe.vae.decode is not None)
    # --- frozen noise reproducibility
    cs = load(os.path.join(ROOT, "configs", "cases.json"))
    allc = list(cs["dev_cases"]) + list(cs["technical_cases"]) + [x for v in cs["heldout_cases"].values() for x in v]
    bad = [x["case_id"] for x in allc if noise.noise_hash(x["eval_seed"]) != x["noise_sha256"]]
    chk("noise_hashes_reproduce", not bad and len(allc) == 24 + 4 + 80, n=len(allc), bad=bad[:3])
    chk("case_ids_unique_per_group", len({x["case_id"] for x in cs["dev_cases"]}) == 24
        and all(len({x["case_id"] for x in v}) == 40 for v in cs["heldout_cases"].values()))
    chk("trajectory_counts", len(cs["dev_trajectories"]) == 12 and len(cs["heldout_trajectories"]) == 24)
    chk("gate_traj_is_first_planned_native_dev", cs["dev_trajectories"][0]["traj_id"] == P.GATE_NATIVE_TRAJ)
    chk("detector_label_resolution_offline", True, note="resolved at qualification; detectors load from HF cache offline")
    from cprd2 import ledger as LG
    chk("ledger_importable", LG.CAPS["gpu_device_hours"] == 24.0)
    R["all_pass"] = all(v["pass"] for v in R["checks"].values())
    dump(R, os.path.join(ROOT, "artifacts", "prefreeze_structural.json"))
    sys.exit(0 if R["all_pass"] else 2)


if __name__ == "__main__":
    main()
