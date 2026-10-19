#!/usr/bin/env python3
"""One process = one independent training trajectory (train phase) OR one generation set (gen phase).

train phase: fresh pristine FLUX -> native PEFT qk_dual LoRA (seeded init) -> real native GEM trainer with the
             loss-dispatch hook -> intermediate/terminal canonical adapters (FP32 safetensors) + counters.
             `--gate` additionally captures the predeclared native forward triplets and asserts the instrumentation
             checks on the first planned native development trajectory.
gen phase:   fresh pristine FLUX + adapter structure; each canonical adapter is zeroed, loaded from its saved file and
             hash-verified, then images are generated from the frozen shared noise.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import torch

from cprd2 import flux_run as FR
from cprd2 import losses as LS
from cprd2 import protocol as P
from cprd2.common import ROOT, dump, load, sha256_file, utc

A = os.path.join(ROOT, "artifacts")


def traj_spec(traj_id):
    c = load(os.path.join(ROOT, "configs", "cases.json"))
    for t in c["dev_trajectories"]:
        if t["traj_id"] == traj_id:
            return dict(t, stage="dev", budget=P.CAP)
    lock_p = os.path.join(ROOT, "configs", "selection_lock.json")
    for t in c["heldout_trajectories"]:
        if t["traj_id"] == traj_id:
            lock = load(lock_p)["selected"][t["arm"]]
            return dict(t, stage="ho", lr=P.SETTINGS["learning_rates"][lock["lr_name"]], lr_name=lock["lr_name"],
                        budget=int(lock["states"]), selection=lock)
    raise KeyError(traj_id)


def mem_gib():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None


# ------------------------------------------------------------------------------------------------ train phase
def train(args):
    from gem.operators.erasure.erasure_gem import GEM, GEMConfig
    from gem.utils.seed import seed_everything
    spec = traj_spec(args.traj_id)
    out = os.path.join(A, "traj", args.traj_id)
    os.makedirs(out, exist_ok=True)
    adir = os.path.join(A, "adapters", args.traj_id)
    gate = args.gate
    rec = {"traj_id": args.traj_id, "spec": {k: v for k, v in spec.items()}, "started_utc": utc(), "gate_mode": gate,
           "job": os.environ.get("CPRD2_JOB_ID")}
    t_all = time.time()
    pipe, wrapper, load_s = FR.build("cuda", init_seed=spec["train_seed"])
    counter = FR.DecodeCounter(pipe.vae)
    inner = wrapper.base_model.model if hasattr(wrapper, "base_model") else wrapper
    inv = FR.inventory(wrapper)
    init_sd = FR.peft_state(wrapper)
    rec.update({"load_seconds": load_s, "inventory": inv, "init_adapter_sha256": FR.state_hash(init_sd),
                "init_B_all_zero": all(bool((v == 0).all()) for k, v in init_sd.items() if "lora_B" in k),
                "init_A_nonzero": all(bool((v != 0).any()) for k, v in init_sd.items() if "lora_A" in k),
                "scheduler_config_sha256": hashlib.sha256(json.dumps(pipe.scheduler.config, sort_keys=True, default=str).encode()).hexdigest(),
                "rng_at_start": FR.rng_digest(), "peft_config": str(wrapper.peft_config)})
    gate_checks = {}
    if gate:
        rec["base_digest_before"] = FR.base_digest(wrapper)

    # ---- hooks
    states = []                                       # per processed state: loss, target_loss, hinge activity
    sampling = []                                     # per window: modes during student sampling
    updates = []                                      # per optimizer update
    ckpt_meta = {}
    cap_store, cap_rng_viol, cap_digests = [], [], []
    base_hook = LS.make_dispatch(spec["arm"], P.SETTINGS["eta"])
    activity = []

    def loss_hook(pred, pos, neg):
        ps = base_hook(pred, pos, neg)
        if ps is not None:
            activity.append(float((ps.detach() > 0).float().mean()))
        return ps

    loss_hook.arm, loss_hook.eta = base_hook.arm, base_hook.eta

    def capture_hook(processed_index, global_step, batch_start, pred, anchor, target):
        if processed_index in P.CAPTURE_PROCESSED_INDICES:
            r0 = FR.rng_digest()
            cap_store.append({"processed_index": int(processed_index), "global_step": int(global_step),
                              "batch_start": int(batch_start), "pred": pred.float().cpu().contiguous(),
                              "anchor": anchor.float().cpu().contiguous(), "target": target.float().cpu().contiguous()})
            if FR.rng_digest() != r0:
                cap_rng_viol.append(processed_index)

    thresholds = [("pct25", 62.5), ("pct50", 125.0)]
    saved = set()

    def save_ckpt(name, processed, global_step):
        meta = FR.save_adapter(wrapper, os.path.join(adir, name),
                               {"traj_id": args.traj_id, "checkpoint": name, "arm": spec["arm"], "lr": spec["lr"],
                                "train_seed": spec["train_seed"], "target": spec["target"],
                                "processed_states": int(processed), "optimizer_updates": int(global_step) + 1,
                                "init_adapter_sha256": rec["init_adapter_sha256"]})
        ckpt_meta[name] = {k: meta[k] for k in ("tensor_state_sha256", "file_sha256", "processed_states", "optimizer_updates")}

    def step_hook(global_step, processed_timesteps, step_budget_left, wrapper, optimizer):
        sd = FR.lora_params(wrapper)
        finite = all(bool(torch.isfinite(p).all()) for p in sd.values())
        st = [s["step"] for s in optimizer.state.values()] if optimizer.state else []
        steps = sorted({int(float(x)) for x in st})
        updates.append({"global_step": int(global_step), "processed": int(processed_timesteps),
                        "budget_left": int(step_budget_left), "finite": finite, "optimizer_step_counts": steps,
                        "n_state_tensors": len(optimizer.state), "n_param_groups": len(optimizer.param_groups),
                        "lr": optimizer.param_groups[0]["lr"]})
        for name, thr in thresholds:
            if name not in saved and processed_timesteps >= thr and spec["stage"] == "dev":
                saved.add(name)
                save_ckpt(name, processed_timesteps, global_step)
        if step_budget_left <= 0:
            save_ckpt("terminal", processed_timesteps, global_step)

    # sampling-mode spy (instrumentation only; calls through)
    cls = type(inner)
    orig_pd = cls.partially_denoise_until_timestep

    def spy(self, *a, **k):
        tr = self.transformer
        sampling.append({"transformer_training": bool(tr.training), "grad_enabled": torch.is_grad_enabled(),
                         "lora_dropout_modules_training": all(m.training for n, m in tr.named_modules() if "lora_dropout" in n),
                         # REPAIR-2: count only actual nn.Dropout instances inside lora_dropout containers (the old count
                         # also matched the ModuleDict container name, giving 2 per layer = 152)
                         "n_lora_dropout_modules": sum(1 for n, m in tr.named_modules()
                                                       if "lora_dropout" in n and isinstance(m, torch.nn.Dropout)),
                         # REPAIR-2: diagnostic only: the old flag read diffusers' PeftAdapterMixin.disable_adapters METHOD (always truthy)
                         "n_lora_layers_adapter_disabled": sum(1 for _, m in tr.named_modules()
                                                               if hasattr(m, "lora_A") and getattr(m, "disable_adapters", False) is True)})
        return orig_pd(self, *a, **k)

    cls.partially_denoise_until_timestep = spy

    cfg = GEMConfig(targets=[spec["target"]], max_step_budget=int(spec["budget"]), max_parallel_timesteps=1,
                    train_seed=int(spec["train_seed"]), val_seed=int(spec["train_seed"]), use_wandb=False,
                    learning_rate=float(spec["lr"]), eta=P.SETTINGS["eta"], gamma=P.SETTINGS["gamma"])
    cfg.cpr_loss_hook = loss_hook
    cfg.cpr_step_hook = step_hook
    if gate:
        cfg.cpr_capture_hook = capture_hook
    rec["native_config"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                            for k, v in vars(cfg).items() if not k.startswith("_")}
    op = GEM()
    orig_log = op.log_loss_and_metrics_to_wandb

    def log_spy(loss, step, metrics=None, **k):
        if metrics is not None:
            states.append({"loss": float(loss), "target_loss": float(metrics["target_loss"])})
        return orig_log(loss=loss, step=step, metrics=metrics, **k)

    op.log_loss_and_metrics_to_wandb = log_spy
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    op(wrapper, cfg)
    train_s = time.time() - t0
    cls.partially_denoise_until_timestep = orig_pd

    final_sd = FR.peft_state(wrapper)
    rec.update({"train_seconds": round(train_s, 1), "seconds_per_state": round(train_s / max(1, len(states)), 3),
                "peak_gpu_gib": mem_gib(), "processed_states": len(states), "optimizer_updates": len(updates),
                "updates": updates, "final_adapter_sha256": FR.state_hash(final_sd),
                "checkpoints": ckpt_meta, "preview_decode_skips": getattr(inner, "_cpr_preview_skips", None),
                "decodes_during_training": counter.n, "loss_hook_calls": base_hook.calls,
                "rng_at_end": FR.rng_digest(), "sampling_windows": len(sampling),
                "hinge_active_fraction_mean": (sum(activity) / len(activity)) if activity else None,
                "all_updates_finite": all(u["finite"] for u in updates),
                "max_abs_adapter_move": max(float((final_sd[k] - init_sd[k]).abs().max()) for k in final_sd)})
    if counter.n:                                     # unexpected decodes during training are charged as technical images
        FR.log_decodes(os.environ.get("CPRD2_JOB_ID", args.traj_id + ".train"), "technical", counter.n)
    with open(os.path.join(out, "states.jsonl"), "w") as f:
        for i, s in enumerate(states):
            f.write(json.dumps(dict(s, processed_index=i)) + "\n")
    expected_updates = {66: 6, 132: 12, 250: 23}.get(int(spec["budget"]))
    rec["expected_updates"] = expected_updates
    rec["window_behaviour_confirmed"] = (len(states) == int(spec["budget"]) and (expected_updates is None or len(updates) == expected_updates))

    if gate:
        after = FR.base_digest(wrapper)
        rec["base_digest_after"] = after
        def gc_(name, ok, **d):
            gate_checks[name] = {"pass": bool(ok), **d}
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(d, default=str)[:260]}", flush=True)
        gc_("peft_inventory", inv["n_lora_modules"] == 76 and inv["n_base_trainable"] == 0 and inv["lora_dropout_p"] == [0.1]
            and inv["lora_dtypes"] == ["torch.float32"], **inv)
        gc_("adapter_init_fresh", rec["init_B_all_zero"] and rec["init_A_nonzero"])
        gc_("frozen_base_unchanged", rec["base_digest_before"] == after, before=rec["base_digest_before"]["sha256"][:16],
            after=after["sha256"][:16])
        gc_("student_sampling_train_mode_no_grad",
            len(sampling) > 0 and all(s["transformer_training"] and not s["grad_enabled"] and s["lora_dropout_modules_training"]
                                      and s["n_lora_dropout_modules"] == 76 for s in sampling),
            windows=len(sampling), first=sampling[0] if sampling else None,
            note="torch.no_grad() is not eval(): native LoRA dropout p=0.1 is ACTIVE while sampling the student trajectory")
        gc_("fresh_optimizer_each_call", bool(updates) and updates[0]["optimizer_step_counts"] == [1]
            and updates[-1]["optimizer_step_counts"] == [len(updates)] and updates[0]["n_param_groups"] == 1,
            first=updates[0]["optimizer_step_counts"], last=updates[-1]["optimizer_step_counts"],
            n_updates=len(updates))
        gc_("finite_intended_updates", rec["all_updates_finite"] and rec["max_abs_adapter_move"] > 0,
            max_abs_adapter_move=rec["max_abs_adapter_move"])
        gc_("capture_hook_read_only_and_predeclared",
            [c["processed_index"] for c in cap_store] == P.CAPTURE_PROCESSED_INDICES and not cap_rng_viol,
            captured=[c["processed_index"] for c in cap_store], rng_violations=cap_rng_viol)
        gc_("native_window_budget_behaviour", rec["window_behaviour_confirmed"], states=len(states), updates=len(updates),
            expected_updates=expected_updates)
        gc_("preview_decode_skipped_and_counted", counter.n == 0 and rec["preview_decode_skips"] == len(sampling),
            decodes=counter.n, skips=rec["preview_decode_skips"], windows=len(sampling))
        gc_("loss_hook_calls_equal_states", base_hook.calls == len(states), calls=base_hook.calls, states=len(states))
        # adapter restoration from the saved terminal canonical adapter
        term = os.path.join(adir, "terminal")
        trained_hash = FR.state_hash(FR.peft_state(wrapper))
        FR.zero_adapter(wrapper)
        zero_hash = FR.state_hash(FR.peft_state(wrapper))
        restored = FR.load_adapter(wrapper, term)
        gc_("adapter_restoration_from_saved_terminal", restored == trained_hash and zero_hash != trained_hash,
            trained=trained_hash[:16], restored=restored[:16])
        os.makedirs(os.path.join(A, "gate"), exist_ok=True)
        torch.save(cap_store, os.path.join(A, "gate", "captured_triplets.pt"))
        # same digest definition as gate_core.digest_tensors(pred, anchor, target)
        from cprd2.gate_core import digest_tensors
        cap_digests = [digest_tensors(c["pred"], c["anchor"], c["target"]) for c in cap_store]
        dump({"checks": gate_checks, "all_pass": all(v["pass"] for v in gate_checks.values()),
              "capture_digests": cap_digests, "capture_file_sha256": sha256_file(os.path.join(A, "gate", "captured_triplets.pt")),
              "traj_id": args.traj_id, "captured_indices": [c["processed_index"] for c in cap_store]},
             os.path.join(A, "gate", "gpu_instrumentation.json"))
        rec["gate_all_pass"] = all(v["pass"] for v in gate_checks.values())
    rec["finished_utc"] = utc()
    rec["total_seconds"] = round(time.time() - t_all, 1)
    dump(rec, os.path.join(out, "train_record.json"))
    if gate and not rec["gate_all_pass"]:
        print("GPU_GATE_FAIL")
        sys.exit(3)
    print("TRAIN_DONE", args.traj_id)


# ------------------------------------------------------------------------------------------------ gen phase
def gen(args):
    sets = load(args.plan)                      # list of {adapter_dir|null, label, cases:[case dicts], image_dir}
    job = os.environ.get("CPRD2_JOB_ID", args.traj_id)
    pipe, wrapper, load_s = FR.build("cuda", init_seed=args.init_seed)
    counter = FR.DecodeCounter(pipe.vae)
    res = {"job": job, "load_seconds": load_s, "sets": [], "started_utc": utc()}
    n_images = 0
    img_by_class = {"primary": 0, "technical": 0}
    for s in sets:
        adapter_on = s["adapter_dir"] is not None
        rh = FR.load_adapter(wrapper, os.path.join(ROOT, s["adapter_dir"])) if adapter_on else None
        if not adapter_on:
            FR.zero_adapter(wrapper)             # M0 additionally runs under disable_adapter
        before = counter.n
        recs = []
        for c in s["cases"]:
            path = os.path.join(ROOT, s["image_dir"], c["case_id"] + ".png")
            if os.path.exists(path) and not args.overwrite:
                raise RuntimeError(f"refusing to regenerate existing image {path}")
            recs.append(generate_one(pipe, wrapper, c, adapter_on, path))
            n_images += 1
            img_by_class[s["image_class"]] += 1
            FR.log_decodes(job, s["image_class"], 1)     # incremental: a crash mid-job still charges generated images
        res["sets"].append({"label": s["label"], "adapter_dir": s["adapter_dir"], "restored_state_sha256": rh,
                            "n_images": len(recs), "decodes": counter.n - before, "images": recs})
    unexpected = counter.n - n_images
    if unexpected > 0:
        FR.log_decodes(job, "technical", unexpected)
    res.update({"n_images": n_images, "decodes_total": counter.n, "unexpected_decodes": unexpected, "images_by_class": img_by_class,
                "peak_gpu_gib": mem_gib(), "finished_utc": utc()})
    dump(res, os.path.join(A, "gen_records", job + ".json"))
    print("GEN_DONE", job, n_images)


def generate_one(pipe, wrapper, case, adapter_on, path):
    return FR.generate_case(pipe, wrapper, case, adapter_on, "cuda", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "gen"])
    ap.add_argument("--traj-id")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--plan")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    (train if a.phase == "train" else gen)(a)
