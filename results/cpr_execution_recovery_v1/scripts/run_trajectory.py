#!/usr/bin/env python3
"""One process = one independent training trajectory (train phase) OR one generation set (gen phase).

train phase: fresh pristine FLUX -> native PEFT qk_dual LoRA (seeded init) -> real native GEM trainer with the loss-dispatch
             hook -> canonical adapters (FP32 safetensors) + counters. A spec JSON (`--spec`) carries arm/lr/seed/budget/
             checkpoints; `probe_indices` (engineering/validation) additionally installs the READ-ONLY probe hook that
             evaluates the trainer's live graph against native-source / hooked / FP64 references on cloned operands.
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

import psutil
import torch

from cprx import flux_run as FR
from cprx import losses as LS
from cprx import protocol as P
from cprx.common import ROOT, dump, load, sha256_file, utc

A = os.path.join(ROOT, "artifacts")


def traj_spec(path):
    return load(path)


def mem_gib():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None


# ------------------------------------------------------------------------------------------------ train phase
def train(args):
    from gem.operators.erasure.erasure_gem import GEM, GEMConfig
    from cprx import gate as GT
    spec = traj_spec(args.spec)
    tid = spec["traj_id"]
    out = os.path.join(A, "traj", tid)
    os.makedirs(out, exist_ok=True)
    adir = os.path.join(A, "adapters", tid)
    probe_idx = list(spec.get("probe_indices") or [])
    rec = {"traj_id": tid, "spec": {k: v for k, v in spec.items()}, "started_utc": utc(), "job": os.environ.get("CPRX_JOB_ID"),
           "spec_file_sha256": sha256_file(args.spec)}
    t_all = time.time()
    pipe, wrapper, load_s = FR.build("cuda", init_seed=spec["train_seed"])
    counter = FR.DecodeCounter(pipe.vae)
    inner = wrapper.base_model.model if hasattr(wrapper, "base_model") else wrapper
    inv = FR.inventory(wrapper)
    init_sd = FR.peft_state(wrapper)
    lora_ids = {id(p) for p in FR.lora_params(wrapper).values()}
    rec.update({"load_seconds": load_s, "inventory": inv, "init_adapter_sha256": FR.state_hash(init_sd),
                "init_B_all_zero": all(bool((v == 0).all()) for k, v in init_sd.items() if "lora_B" in k),
                "init_A_nonzero": all(bool((v != 0).any()) for k, v in init_sd.items() if "lora_A" in k),
                "scheduler_config_sha256": hashlib.sha256(json.dumps(pipe.scheduler.config, sort_keys=True, default=str).encode()).hexdigest(),
                "rng_at_start": FR.rng_digest(), "peft_config": str(wrapper.peft_config),
                "host_rss_gib_after_load": round(psutil.Process().memory_info().rss / 2**30, 2)})
    if probe_idx:
        rec["base_digest_before"] = FR.base_digest(wrapper)

    states, sampling, updates, ckpt_meta, activity = [], [], [], {}, []
    base_hook = LS.make_dispatch(spec["arm"], P.SETTINGS["eta"])

    def loss_hook(pred, pos, neg):
        ps = base_hook(pred, pos, neg)
        if ps is not None:
            activity.append(float((ps.detach() > 0).float().mean()))
        return ps

    loss_hook.arm, loss_hook.eta = base_hook.arm, base_hook.eta
    probe = GT.Probe(spec["arm"], probe_idx, os.path.join(A, "gate", tid)) if probe_idx else None
    saved = set()
    ckpts = spec.get("checkpoints") or []                      # [[name, processed_threshold], ...] (terminal is implicit)

    def save_ckpt(name, processed, global_step):
        meta = FR.save_adapter(wrapper, os.path.join(adir, name),
                               {"traj_id": tid, "checkpoint": name, "arm": spec["arm"], "lr": spec["lr"],
                                "train_seed": spec["train_seed"], "target": spec["target"],
                                "processed_states": int(processed), "optimizer_updates": int(global_step) + 1,
                                "init_adapter_sha256": rec["init_adapter_sha256"]})
        ckpt_meta[name] = {k: meta[k] for k in ("tensor_state_sha256", "file_sha256", "processed_states", "optimizer_updates")}

    def step_hook(global_step, processed_timesteps, step_budget_left, wrapper, optimizer):
        sd = FR.lora_params(wrapper)
        finite = all(bool(torch.isfinite(p).all()) for p in sd.values())
        st = [s["step"] for s in optimizer.state.values()] if optimizer.state else []
        u = {"global_step": int(global_step), "processed": int(processed_timesteps), "budget_left": int(step_budget_left),
             "finite": finite, "optimizer_step_counts": sorted({int(float(x)) for x in st}),
             "n_state_tensors": len(optimizer.state), "n_param_groups": len(optimizer.param_groups),
             "lr": optimizer.param_groups[0]["lr"]}
        if not updates:
            ps_ = [p for g in optimizer.param_groups for p in g["params"]]
            req = [p for p in ps_ if p.requires_grad]
            u.update({"optimizer_n_params_in_group": len(ps_),
                      "optimizer_n_requires_grad": len(req),
                      "optimizer_requires_grad_are_exactly_lora": {id(p) for p in req} == lora_ids,
                      "optimizer_n_frozen_in_group_without_grad": len(ps_) - len(req),
                      "optimizer_trainable_param_count": int(sum(p.numel() for p in req)),
                      "optimizer_class": type(optimizer).__name__,
                      "optimizer_defaults": {k: str(v) for k, v in optimizer.defaults.items()}})
        updates.append(u)
        for name, thr in ckpts:
            if name not in saved and processed_timesteps >= thr:
                saved.add(name)
                save_ckpt(name, processed_timesteps, global_step)
        if step_budget_left <= 0 and spec.get("save_terminal", True):
            save_ckpt("terminal", processed_timesteps, global_step)

    cls = type(inner)
    orig_pd = cls.partially_denoise_until_timestep

    def spy(self, *a, **k):
        tr = self.transformer
        sampling.append({"transformer_training": bool(tr.training), "grad_enabled": torch.is_grad_enabled(),
                         "lora_dropout_modules_training": all(m.training for n, m in tr.named_modules() if "lora_dropout" in n),
                         "n_lora_dropout_modules": sum(1 for n, m in tr.named_modules()
                                                       if "lora_dropout" in n and isinstance(m, torch.nn.Dropout))})
        return orig_pd(self, *a, **k)

    cls.partially_denoise_until_timestep = spy
    cfg = GEMConfig(targets=[spec["target"]], max_step_budget=int(spec["budget"]), max_parallel_timesteps=1,
                    train_seed=int(spec["train_seed"]), val_seed=int(spec["train_seed"]), use_wandb=False,
                    learning_rate=float(spec["lr"]), eta=P.SETTINGS["eta"], gamma=P.SETTINGS["gamma"])
    cfg.cpr_loss_hook = loss_hook
    cfg.cpr_step_hook = step_hook
    if probe is not None:
        cfg.cpr_probe_hook = probe
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
    try:
        op(wrapper, cfg)
    except LS.AxialDomainViolation as ex_:
        rec.update({"status": "domain_failure", "error": repr(ex_)[:300], "states_before_failure": len(states),
                    "optimizer_updates": len(updates), "finished_utc": utc()})
        dump(rec, os.path.join(out, "train_record.json"))
        print("DOMAIN_FAILURE", tid, repr(ex_)[:200])
        sys.exit(7)
    train_s = time.time() - t0
    cls.partially_denoise_until_timestep = orig_pd

    final_sd = FR.peft_state(wrapper)
    expected_updates = {66: 6, 132: 12, 250: 23, 44: 4}.get(int(spec["budget"]))
    rec.update({"train_seconds": round(train_s, 1), "seconds_per_state": round(train_s / max(1, len(states)), 3),
                "peak_gpu_gib": mem_gib(), "processed_states": len(states), "optimizer_updates": len(updates),
                "updates": updates, "final_adapter_sha256": FR.state_hash(final_sd), "checkpoints": ckpt_meta,
                "preview_decode_skips": getattr(inner, "_cpr_preview_skips", None),
                "decodes_during_training": counter.n, "loss_hook_calls": base_hook.calls, "rng_at_end": FR.rng_digest(),
                "sampling_windows": len(sampling), "hinge_active_fraction_mean": (sum(activity) / len(activity)) if activity else None,
                "all_updates_finite": all(u["finite"] for u in updates),
                "max_abs_adapter_move": max(float((final_sd[k] - init_sd[k]).abs().max()) for k in final_sd),
                "host_rss_gib_peak_self": round(psutil.Process().memory_info().rss / 2**30, 2),
                "expected_updates": expected_updates})
    rec["window_behaviour_confirmed"] = (len(states) == int(spec["budget"]) and (expected_updates is None or len(updates) == expected_updates))
    if counter.n:
        FR.log_decodes(os.environ.get("CPRX_JOB_ID", tid + ".train"), "technical", counter.n)
    with open(os.path.join(out, "states.jsonl"), "w") as f:
        for i, s in enumerate(states):
            f.write(json.dumps(dict(s, processed_index=i)) + "\n")

    if probe_idx:
        checks = {}

        def gc_(name, ok, **d):
            checks[name] = {"pass": bool(ok), **d}
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(d, default=str)[:260]}", flush=True)
        after = FR.base_digest(wrapper)
        rec["base_digest_after"] = after
        gc_("peft_inventory", inv["n_lora_modules"] == 76 and inv["n_lora_tensors"] == 152 and inv["n_base_trainable"] == 0
            and inv["trainable_params"] == 7471104 and inv["lora_dropout_p"] == [0.1] and inv["lora_dtypes"] == ["torch.float32"], **inv)
        gc_("adapter_init_fresh", rec["init_B_all_zero"] and rec["init_A_nonzero"])
        gc_("frozen_base_unchanged", rec["base_digest_before"] == after, note="aggregate float64 sum/sumsq/absmax per tensor, "
            "NOT a hash of every weight byte")
        gc_("student_sampling_train_mode_no_grad", len(sampling) > 0 and all(
            s["transformer_training"] and not s["grad_enabled"] and s["lora_dropout_modules_training"]
            and s["n_lora_dropout_modules"] == 76 for s in sampling), windows=len(sampling), first=sampling[0] if sampling else None)
        gc_("frozen_base_cannot_be_updated_by_optimizer_fresh_optimizer_each_call", bool(updates)
            and updates[0].get("optimizer_requires_grad_are_exactly_lora") and updates[0]["optimizer_n_requires_grad"] == 152
            and updates[0]["optimizer_n_frozen_in_group_without_grad"] >= 0 and updates[0]["n_state_tensors"] == 152
            and updates[0]["optimizer_trainable_param_count"] == 7471104
            and updates[0]["optimizer_step_counts"] == [1] and updates[-1]["optimizer_step_counts"] == [len(updates)],
            first=updates[0] if updates else None)
        gc_("finite_intended_updates", rec["all_updates_finite"] and rec["max_abs_adapter_move"] > 0,
            max_abs_adapter_move=rec["max_abs_adapter_move"])
        gc_("native_window_budget_behaviour", rec["window_behaviour_confirmed"], states=len(states), updates=len(updates),
            expected_updates=expected_updates)
        gc_("preview_decode_skipped_and_counted", counter.n == 0 and rec["preview_decode_skips"] == len(sampling),
            decodes=counter.n, skips=rec["preview_decode_skips"], windows=len(sampling))
        gc_("loss_hook_calls_equal_states", base_hook.calls == len(states), calls=base_hook.calls, states=len(states))
        got_idx = [r["processed_index"] for r in probe.records]
        gc_("probe_coverage_predeclared", got_idx == sorted(set(probe_idx)) and all(i < len(states) for i in probe_idx),
            requested=probe_idx, captured=got_idx, updates_before=[r["optimizer_updates_before"] for r in probe.records])
        for r in probe.records:
            gc_(f"probe_index_{r['processed_index']}_ok", r["probe_ok"], updates_before=r["optimizer_updates_before"],
                student_equals_target_bitwise=r["student_equals_target_bitwise"], norm_v_minus_b=r["norm_v_minus_b"],
                graph_vs_fp64=r["actual_graph_vs_fp64"].get("classification"),
                agreement={k: v["ok"] for k, v in r["fp32_clone_agreement"].items()},
                hooked_all_arms={k: v.get("classification", "domain") for k, v in r["hooked_vs_fp64_all_arms"].items()})
        final_dir = os.path.join(adir, "terminal")
        if os.path.isdir(final_dir):
            trained_hash = FR.state_hash(FR.peft_state(wrapper))
            FR.zero_adapter(wrapper)
            zero_hash = FR.state_hash(FR.peft_state(wrapper))
            restored = FR.load_adapter(wrapper, final_dir)
            gc_("adapter_restoration_from_saved_terminal", restored == trained_hash and zero_hash != trained_hash,
                trained=trained_hash[:16], restored=restored[:16])
        allp = all(v["pass"] for v in checks.values())
        dump({"traj_id": tid, "arm": spec["arm"], "checks": checks, "all_pass": allp, "probe_records": probe.records,
              "coverage_note": ("Captures are at the predeclared processed-state indices only; index 0 precedes any optimizer update "
                                "(adapter B=0, student==teacher bitwise); later indices follow >=1 update, but an update does not "
                                "prove BF16 student outputs differ from the teacher (see student_equals_target_bitwise per capture).")},
             os.path.join(A, "gate", tid, "gate_report.json"))
        rec["gate_all_pass"] = allp
    rec["finished_utc"] = utc()
    rec["total_seconds"] = round(time.time() - t_all, 1)
    dump(rec, os.path.join(out, "train_record.json"))
    if probe_idx and not rec["gate_all_pass"]:
        print("GPU_GATE_FAIL")
        sys.exit(3)
    print("TRAIN_DONE", tid)


# ------------------------------------------------------------------------------------------------ gen phase
def gen(args):
    sets = load(args.plan)                      # list of {adapter_dir|null, label, cases:[case dicts], image_dir}
    job = os.environ.get("CPRX_JOB_ID", args.traj_id)
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
    ap.add_argument("--spec")
    ap.add_argument("--traj-id")
    ap.add_argument("--plan")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    (train if a.phase == "train" else gen)(a)
