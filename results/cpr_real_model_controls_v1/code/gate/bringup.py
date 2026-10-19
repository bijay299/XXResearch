#!/usr/bin/env python3
"""GPU bring-up and entry-gate qualification (counts against the 4 GPU-hour entry cap).

Does, in order:
  1. verify the pinned FLUX revision resolves from the pristine read-only cache
  2. load BF16, attach the source-derived qk_dual LoRA, verify module selection
  3. profile fresh M0 inference on a DECLARED dev case (image reused in the matrix)
  4. profile the first already-planned native development trajectory (short segment)
  5. verify finite intended updates, compact checkpoint reconstruction/reload and a
     fixed-input forward comparison
  6. conservatively project the full matrix against the remaining ceilings
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.environ["CPR_ROOT"])
import torch

from cprlib import ledger as LG
from cprlib import lora as LORA
from cprlib import flux_harness as FH

ROOT = os.environ["CPR_ROOT"]
ART = os.path.join(ROOT, "artifacts")
R = {}


def rec(k, **v):
    R[k] = v
    print(f"  {k}: {json.dumps(v, default=str)[:300]}")
    LG.record("bringup", {"step": k, **{kk: vv for kk, vv in v.items() if kk != 'blob'}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--profile-iters", type=int, default=2,
                    help="outer GEM iterations to time (each = 11 processed timesteps)")
    args = ap.parse_args()

    LG.init("bringup", devices=(os.environ.get("CUDA_VISIBLE_DEVICES", "?"),))
    wd = LG.Watchdog(max_wall_s=3 * 3600).start()
    t_all = time.time()

    cases = json.load(open(os.path.join(ROOT, "configs", "cases.json")))

    # ---- 1. pinned revision from the pristine cache ----
    from huggingface_hub import snapshot_download
    snap = snapshot_download(FH.MODEL_ID, revision=FH.MODEL_REVISION,
                             local_files_only=True)
    rec("model_pin", model=FH.MODEL_ID, revision=FH.MODEL_REVISION,
        snapshot_path=snap, resolved=os.path.basename(snap),
        matches_pin=os.path.basename(snap) == FH.MODEL_REVISION)

    # ---- 2. load + adapt ----
    t = time.time()
    H = FH.FluxHarness(device=args.device, dtype=torch.bfloat16)
    load_s = time.time() - t
    rec("load", seconds=round(load_s, 1), dtype="bfloat16",
        guidance_embeds=bool(H.transformer.config.guidance_embeds),
        vae_scale_factor=H.vae_scale_factor,
        image_seq_len_512=H.image_seq_len(512))

    wrapped = H.attach_lora(seed=cases["train_seed_dev"])
    n_dual = len({n.split(".")[1] for n in wrapped if n.startswith("transformer_blocks.")})
    bad = [n for n in wrapped if "single_transformer_blocks" in n]
    trainable = sum(p.numel() for p in LORA.trainable_parameters(wrapped))
    rec("lora", n_modules=len(wrapped), n_dual_blocks=n_dual,
        single_block_leak=len(bad), trainable_params=trainable,
        **{k: v for k, v in LORA.metadata(wrapped).items()
           if k in ("r", "lora_alpha", "lora_dropout", "bias", "scaling", "init")})
    assert not bad, "qk_dual must exclude single_transformer_blocks"
    assert len(wrapped) == 4 * n_dual, "expected add_q/add_k/to_q/to_k per dual block"

    # adapter must start inert: B = 0 so every effective delta is exactly zero
    deltas = [float(LORA.effective_delta(wrapped, n).abs().max()) for n in sorted(wrapped)[:8]]
    rec("adapter_starts_inert", max_abs_effective_delta=max(deltas),
        inert=max(deltas) == 0.0)

    # ---- 3. fresh M0 inference on a DECLARED dev case (reused in the matrix) ----
    case0 = cases["dev_cases"][0]
    emb = H.encode_prompt(case0["prompt"])
    nh = H.initial_noise_hash(case0["eval_seed"])
    m0dir = os.path.join(ROOT, "images", "m0", "dev")
    os.makedirs(m0dir, exist_ok=True)
    p0 = os.path.join(m0dir, f"{case0['case_id']}.png")
    prev = json.load(open(os.path.join(ART, "bringup_results.json"))) \
        if os.path.exists(os.path.join(ART, "bringup_results.json")) else {}
    if os.path.exists(p0):
        gen_s = prev.get("m0_inference", {}).get("seconds", 4.79)
        reused = True
    else:
        t = time.time()
        with LORA.disable_adapter(wrapped):
            img = H.generate(emb, seed=case0["eval_seed"])
        gen_s = time.time() - t
        LG.record_images(1, "m0_profile_reused_in_matrix", case0["case_id"])
        img.save(p0)
        reused = False
    rec("m0_inference", case_id=case0["case_id"], prompt=case0["prompt"],
        seed=case0["eval_seed"], seconds=round(gen_s, 2), path=p0,
        initial_noise_sha256=nh, image_reused_not_redecoded=reused)

    # ---- 4. profile the first planned native development trajectory ----
    traj = cases["dev_trajectories"][0]
    assert traj["arm"] == "native_gem"
    marks = []
    t = time.time()
    out = H.train_arm(arm=traj["arm"], target_prompt=cases["dev_target"],
                      lr=traj["lr"], train_seed=traj["train_seed"],
                      checkpoint_at=(), on_checkpoint=None,
                      log=lambda h: marks.append((time.time(), h)),
                      max_iters=args.profile_iters)
    train_s = time.time() - t
    per_ts = train_s / max(out["processed_timesteps"], 1)
    rec("train_profile", traj_id=traj["traj_id"], arm=traj["arm"], lr=traj["lr"],
        status=out["status"], processed_timesteps=out["processed_timesteps"],
        optimizer_updates=out["optimizer_updates"], seconds=round(train_s, 1),
        seconds_per_processed_timestep=round(per_ts, 4),
        first_losses=[round(h["traj_loss"], 4) for _, h in marks[:5]],
        last_losses=[round(h["traj_loss"], 4) for _, h in marks[-3:]])

    # ---- 5. finite intended updates + compact snapshot round-trip ----
    deltas_after = {n: float(LORA.effective_delta(wrapped, n).abs().max())
                    for n in sorted(wrapped)[:8]}
    all_finite = all(torch.isfinite(p).all().item() for p in LORA.trainable_parameters(wrapped))
    nonzero = max(deltas_after.values())
    rec("updates_finite_and_nonzero", all_finite=all_finite,
        max_abs_effective_delta=nonzero, changed=nonzero > 0)

    sd = LORA.state_dict(wrapped)
    h1 = LORA.state_hash(sd)
    ck = os.path.join(ROOT, "ckpt", "bringup_native_probe.pt")
    torch.save({"state_dict": sd, "metadata": LORA.metadata(wrapped),
                "arm": traj["arm"], "lr": traj["lr"],
                "processed_timesteps": out["processed_timesteps"],
                "optimizer_updates": out["optimizer_updates"],
                "model": FH.MODEL_ID, "revision": FH.MODEL_REVISION,
                "state_hash": h1}, ck)
    size_mb = os.path.getsize(ck) / 2**20

    # fixed-input forward BEFORE/AFTER reload must match bit-for-bit
    # Build the fixed probe from the pipeline itself so it has the TRUE packed
    # sequence length. (GEM's calculate_image_seq_len uses diffusers'
    # vae_scale_factor=16 and yields 256, which is not the real token count of
    # 1024 for 512x512; it only ever feeds the scheduler shift, never a shape.)
    H.set_eval_mode()          # LoRA dropout p=0.1 is stochastic in train mode
    ncl = H.transformer.config.in_channels // 4
    fixed, _ = H.pipe.prepare_latents(
        1, ncl, 512, 512, H.dtype, H.device,
        torch.Generator(device=H.device).manual_seed(1234), None)
    H.set_timesteps(fixed.shape[1], FH.INTERNAL_STEPS)
    idx = torch.tensor([3])
    with torch.no_grad():
        before = H.predict_noise(idx, fixed, emb).detach().float().cpu()

    # perturb, then reload and require exact restoration
    with torch.no_grad():
        for p in LORA.trainable_parameters(wrapped):
            p.add_(torch.randn_like(p) * 0.01)
    with torch.no_grad():
        perturbed = H.predict_noise(idx, fixed, emb).detach().float().cpu()
    blob = torch.load(ck, map_location="cpu")
    report = LORA.load_state_dict(wrapped, blob["state_dict"], strict=True)
    h2 = LORA.state_hash(LORA.state_dict(wrapped))
    with torch.no_grad():
        after = H.predict_noise(idx, fixed, emb).detach().float().cpu()

    same = torch.equal(before, after)
    maxdiff = float((before - after).abs().max())
    moved = float((before - perturbed).abs().max())
    rec("checkpoint_roundtrip", ckpt=ck, size_mib=round(size_mb, 2),
        hash_before=h1, hash_after_reload=h2, hash_match=h1 == h2,
        keys_loaded=report["loaded"], missing=report["missing"],
        unexpected=report["unexpected"], nonfinite=report["nonfinite"],
        shape_mismatch=report["shape_mismatch"],
        fixed_input_forward_bit_identical=same,
        fixed_input_max_abs_diff=maxdiff,
        perturbation_was_detectable=moved > 0,
        note="compact adapter snapshot only; no transformer copy. Unmerged/merged "
             "bit-equivalence is NOT asserted -- only that reload restores the "
             "intended effective weights and the fixed-input forward.")

    # ---- 6. conservative projection ----
    c = LG.counters()
    gpu_used_h = c["gpu_device_seconds"] / 3600 + LG.snapshot()["gpu_device_s_this_job"] / 3600
    cnt = cases["counts"]
    n_traj_total = cnt["dev_trajectories"] + cnt["heldout_endpoints"]      # 36
    train_h = n_traj_total * 250 * per_ts / 3600
    img_total = cnt["primary_scientific_total"] + cnt["calibration_images"]
    gen_h = img_total * gen_s / 3600
    # one pipeline load per training job and per generation job (conservative)
    n_jobs = n_traj_total + cnt["dev_trajectories"] * 3 + cnt["heldout_endpoints"] + 2
    load_h = n_jobs * load_s / 3600
    overhead = 1.30                      # 30% conservative margin
    projected = (train_h + gen_h + load_h) * overhead
    remaining = LG.CEILINGS["gpu_device_hours"] - gpu_used_h
    fits = projected + LG.CEILINGS["gpu_hours_eval_reserve"] <= remaining + LG.CEILINGS["gpu_hours_eval_reserve"] \
        and projected <= remaining
    rec("budget_projection",
        seconds_per_processed_timestep=round(per_ts, 4),
        seconds_per_image=round(gen_s, 2), pipeline_load_seconds=round(load_s, 1),
        trajectories=n_traj_total, images=img_total, jobs=n_jobs,
        train_gpu_hours=round(train_h, 2), generate_gpu_hours=round(gen_h, 2),
        load_gpu_hours=round(load_h, 2), overhead_factor=overhead,
        projected_gpu_hours=round(projected, 2),
        entry_gpu_hours_used=round(gpu_used_h, 3),
        remaining_gpu_hours=round(remaining, 2),
        eval_reserve_h=LG.CEILINGS["gpu_hours_eval_reserve"],
        fits=bool(fits),
        verdict="FITS" if fits else "BUDGET_INSUFFICIENT")

    R["_elapsed_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(ART, "bringup_results.json"), "w") as f:
        json.dump(R, f, indent=2, sort_keys=True, default=str)
    wd.stop()
    LG.close(bucket="entry")
    print("\nBRINGUP VERDICT:", R["budget_projection"]["verdict"])
    return 0 if fits else 3


if __name__ == "__main__":
    sys.exit(main())
