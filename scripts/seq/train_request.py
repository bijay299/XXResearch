#!/usr/bin/env python3
"""Run ONE ConAbl deletion request, with parent-checkpoint and L2-SP verification.

Wraps CUIG's native `train_conabl.py` (no fork, no patch to the upstream tree):
`--unet_ckpt` continuation, `kv-xattn` parameter group, upstream object
mappings. What this wrapper adds is verification that upstream does not do.

1. PARENT CHECKPOINT (before the child update)
   Upstream prints a warning and *silently falls back to the base model* when
   `--unet_ckpt` does not exist:

       if not os.path.exists(args.unet_ckpt):
           print("UNet checkpoint not found ... Using default UNet from pipeline")

   For a sequential experiment that would silently turn MAB into a one-step
   deletion from M0 and invalidate the whole comparison. We therefore require
   the file up front, load it, and check that every saved trainable tensor is
   byte-equal to the parent's after loading, that key coverage is exactly the
   expected kv-xattn set, and that there are no unexpected keys.

   These `delta.bin` files hold ABSOLUTE updated values for the trainable
   tensors, not additive offsets, so equality (not summation) is the right test.

2. L2-SP REFERENCE
   `Regularizers/Weight/l2sp.py` captures its reference on first call, from the
   model's `requires_grad` parameters. Because that call happens before the
   first optimizer step and after `--unet_ckpt` has been loaded, the reference
   *should* be exactly the parent (MA). We assert it rather than assume it, and
   record the max abs deviation.

3. LOSS SEPARATION
   The raw L2-SP term is recorded per step so the weighted regularisation loss
   and the unlearning loss can be reported separately.

Run under: accelerate launch --config_file <gpu_ids: all cfg> this_script.py ...
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

EXPECTED_SUBSTR = ("attn2.to_k", "attn2.to_v")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_delta(path: Path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["unet"] if isinstance(blob, dict) and "unet" in blob else blob


def verify_parent(parent_ckpt: Path, base_model_dir: str, report: dict) -> dict:
    """Fail loudly rather than let upstream fall back to the base model."""
    import torch
    from diffusers import UNet2DConditionModel

    if not parent_ckpt.exists():
        raise SystemExit(f"FATAL: parent checkpoint missing: {parent_ckpt}\n"
                         "Upstream would silently fall back to the base model, which "
                         "would invalidate the sequential comparison.")

    sd = load_delta(parent_ckpt)
    unet = UNet2DConditionModel.from_pretrained(base_model_dir, subfolder="unet")
    named = dict(unet.named_parameters())

    expected_keys = {n for n in named if any(s in n for s in EXPECTED_SUBSTR)}
    saved_keys = set(sd.keys())

    missing_from_ckpt = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    absent_from_unet = sorted(k for k in saved_keys if k not in named)

    res = unet.load_state_dict(sd, strict=False)

    # After loading, every saved tensor must equal the parent's tensor exactly.
    reloaded = dict(unet.named_parameters())
    max_dev = 0.0
    mismatched = []
    for k, v in sd.items():
        if k in reloaded:
            d = (reloaded[k].detach().cpu().float() - v.float()).abs().max().item()
            max_dev = max(max_dev, d)
            if d != 0.0:
                mismatched.append(k)

    info = {
        "parent_ckpt": str(parent_ckpt),
        "parent_sha256": sha256_file(parent_ckpt),
        "parent_bytes": parent_ckpt.stat().st_size,
        "saved_tensors": len(sd),
        "expected_kv_xattn_tensors": len(expected_keys),
        "keys_missing_from_ckpt": missing_from_ckpt,
        "unexpected_keys_in_ckpt": unexpected,
        "keys_absent_from_unet": absent_from_unet,
        "load_unexpected_keys": len(res.unexpected_keys),
        "tensors_mismatched_after_load": mismatched,
        "max_abs_deviation_after_load": max_dev,
        "semantics": "delta.bin holds ABSOLUTE updated values for trainable tensors",
    }
    report["parent_verification"] = info

    ok = (not missing_from_ckpt and not unexpected and not absent_from_unet
          and not mismatched and len(res.unexpected_keys) == 0)
    info["PASS"] = ok
    if not ok:
        raise SystemExit(f"FATAL: parent checkpoint verification failed:\n"
                         f"{json.dumps(info, indent=2)}")
    print(f"[verify] parent OK: {len(sd)} kv-xattn tensors, max_dev={max_dev}, "
          f"sha256={info['parent_sha256'][:16]}...")
    del unet
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="checkpoint name, e.g. MAB_L2")
    ap.add_argument("--parent_name", default="M0")
    ap.add_argument("--anchor_name", required=True, help="e.g. horse")
    ap.add_argument("--target", required=True, help="e.g. dog")
    ap.add_argument("--anchor_dataset_dir", required=True)
    ap.add_argument("--anchor_prompt_path", required=True)
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--unet_ckpt", default=None, help="parent delta.bin; omit for first request")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--num_anchor_images", type=int, default=200)
    ap.add_argument("--num_anchor_prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--l2sp_weight", type=float, default=0.0)
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="save an intermediate dump every N optimizer steps "
                         "(0 = off). Uses upstream's own periodic save, so a "
                         "dump has exactly the same structure as delta.bin and "
                         "validates against the same checkpoint contract. "
                         "Required by the matched-effectiveness diagnostic, "
                         "which scans every dump rather than bisecting.")
    ap.add_argument("--report", required=True, help="where to write the run report JSON")
    args = ap.parse_args()

    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "checkpoint": args.name,
        "parent": args.parent_name,
        "new_deletion_target": args.target,
        "anchor_concept": args.anchor_name,
        "anchor_target_mapping": f"{args.anchor_name}+{args.target}",
        "l2sp_weight": args.l2sp_weight,
        "training_seed": args.seed,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "uuid": None},
    }
    try:
        import subprocess
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        report["gpu"]["uuid"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader", "-i", cvd],
            text=True).strip()
    except Exception:
        pass

    # ---- 1. parent checkpoint verification -------------------------------
    parent_sd = None
    if args.unet_ckpt:
        parent_sd = verify_parent(Path(args.unet_ckpt), args.base_model_dir, report)
    else:
        report["parent_verification"] = {"note": "first request; starts from untouched M0",
                                         "PASS": True}

    # ---- 2. build upstream argv ------------------------------------------
    cuig_root = Path(os.environ["REPO_ROOT"])
    conabl_dir = cuig_root / "UnlearningMethods" / "ConAbl"
    os.chdir(conabl_dir)
    sys.path.insert(0, str(conabl_dir))
    sys.path.insert(0, str(cuig_root))

    argv = [
        "train_conabl.py",
        "--anchor_target_concepts", f"{args.anchor_name}+{args.target}",
        "--concept_type", "object",
        "--output_dir", str(out_dir),
        "--base_model_dir", args.base_model_dir,
        "--iterations", str(args.iterations),
        "--epochs", str(args.epochs),
        "--num_anchor_images", str(args.num_anchor_images),
        "--num_anchor_prompts", str(args.num_anchor_prompts),
        "--anchor_dataset_dirs", args.anchor_dataset_dir,
        "--anchor_prompt_paths", args.anchor_prompt_path,
        "--seed", str(args.seed),
        "--scale_lr", "--hflip", "--noaug",
        "--enable_xformers_memory_efficient_attention",
        "--overwrite_existing_ckpt",
    ]
    if args.unet_ckpt:
        argv += ["--unet_ckpt", args.unet_ckpt]
    if args.l2sp_weight > 0:
        argv += ["--l2sp_weight", str(args.l2sp_weight)]
    if args.checkpoint_every > 0:
        # Upstream's own periodic save, inside update_progress_and_checkpoint:
        # it writes output_dir/delta-<completed_steps> via the SAME
        # save_pretrained(parameter_group=...) call that writes the final
        # delta.bin. Same structure, same contract, no custom dump path.
        argv += ["--turn_on_checkpointing",
                 "--checkpointing_steps", str(args.checkpoint_every)]
    sys.argv = argv
    report["upstream_argv"] = argv

    import train_conabl  # type: ignore
    from src.args import parse_args  # type: ignore

    up_args = parse_args()

    # ---- 3. instrument L2-SP ---------------------------------------------
    import Regularizers.Weight.l2sp as l2sp_mod  # type: ignore
    orig_l2sp = l2sp_mod.calculate_l2sp_loss
    l2_trace: list[dict] = []
    l2_state = {"captured": False, "ref_check": None}

    def instrumented_l2sp(model, original_params):
        first = not original_params
        val = orig_l2sp(model, original_params)
        if first and original_params:
            # Reference was just captured. It must equal the parent (MA).
            max_dev, n_checked, missing = 0.0, 0, []
            if parent_sd is not None:
                for k, ref in original_params.items():
                    if k in parent_sd:
                        d = (ref.detach().cpu().float()
                             - parent_sd[k].float()).abs().max().item()
                        max_dev = max(max_dev, d)
                        n_checked += 1
                    else:
                        missing.append(k)
            l2_state["ref_check"] = {
                "reference_params": len(original_params),
                "total_elements": int(sum(p.numel() for p in original_params.values())),
                "checked_against_parent": n_checked,
                "keys_not_in_parent": missing,
                "max_abs_deviation_from_parent": max_dev,
                "PASS": (parent_sd is not None and n_checked == len(original_params)
                         and not missing and max_dev == 0.0),
            }
            l2_state["captured"] = True
            print(f"[verify] L2-SP reference == parent: {l2_state['ref_check']}")
            if not l2_state["ref_check"]["PASS"]:
                raise SystemExit("FATAL: L2-SP reference does not equal the parent "
                                 f"checkpoint:\n{json.dumps(l2_state['ref_check'], indent=2)}")
        try:
            l2_trace.append({"raw_l2sp": float(val)})
        except Exception:
            pass
        return val

    l2sp_mod.calculate_l2sp_loss = instrumented_l2sp
    train_conabl.calculate_l2sp_loss = instrumented_l2sp

    # ---- 3b. instrument the ACTUAL optimizer-step counter -----------------
    # Upstream increments its own completed-step count inside
    # update_progress_and_checkpoint, which it calls only when gradients are
    # synced, and returns the updated value. Wrapping that captures the
    # trainer's authoritative count of real optimizer updates.
    #
    # This exists because the previous completion check was circular: the report
    # stored seconds_per_optimizer_step = train_seconds / iterations_requested,
    # and the validator then divided those two quantities back, recovering
    # iterations_requested by construction and comparing the request against
    # itself. A completed-step count must come from the optimizer loop, never
    # from the request, a timestamp, epoch capacity or the L2-SP trace.
    orig_update = train_conabl.update_progress_and_checkpoint
    step_state = {"completed": 0, "calls": 0}

    def instrumented_update(*a, **kw):
        n = orig_update(*a, **kw)
        step_state["calls"] += 1
        try:
            step_state["completed"] = max(step_state["completed"], int(n))
        except (TypeError, ValueError):
            pass
        return n

    train_conabl.update_progress_and_checkpoint = instrumented_update

    # ---- 4. train ---------------------------------------------------------
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    train_conabl.main(up_args)
    train_seconds = time.time() - t0
    steps_completed = int(step_state["completed"])
    print(f"[verify] optimizer steps completed (upstream counter): "
          f"{steps_completed} over {step_state['calls']} sync points")
    if steps_completed != int(up_args.iterations):
        print(f"[warn] completed {steps_completed} optimizer steps but "
              f"{up_args.iterations} were requested; the report will record the "
              f"actual count and validation will reject it as incomplete",
              file=sys.stderr)

    # ---- 5. record effective hyperparameters ------------------------------
    # NOTE: upstream's setup_training_configuration MUTATES args.learning_rate in
    # place when --scale_lr is set, so by this point up_args.learning_rate is
    # ALREADY the scaled value. Re-applying the formula here would double-scale.
    n_proc = 1  # accelerate single_gpu config: num_processes = 1
    LR_CLI_DEFAULT = 2.0e-06
    eff_lr = up_args.learning_rate            # post-scaling, as actually used
    report["effective_hyperparameters"] = {
        "learning_rate_cli_default_before_scaling": LR_CLI_DEFAULT,
        "scale_lr": up_args.scale_lr,
        "effective_learning_rate": eff_lr,
        "effective_lr_cross_check": LR_CLI_DEFAULT * up_args.gradient_accumulation_steps
                                    * up_args.anchor_batch_size * n_proc,
        "scale_lr_formula": "lr * grad_accum * anchor_batch_size * num_processes",
        "anchor_batch_size": up_args.anchor_batch_size,
        "gradient_accumulation_steps": up_args.gradient_accumulation_steps,
        "num_processes": n_proc,
        "optimizer": "AdamW (torch.optim.AdamW; use_8bit_adam=False)",
        "adam_beta1": up_args.adam_beta1, "adam_beta2": up_args.adam_beta2,
        "adam_weight_decay": up_args.adam_weight_decay, "adam_epsilon": up_args.adam_epsilon,
        "max_grad_norm": up_args.max_grad_norm,
        "lr_scheduler": up_args.lr_scheduler, "lr_warmup_steps": up_args.lr_warmup_steps,
        "precision": "fp32 (accelerate mixed_precision: 'no')",
        "parameter_group": up_args.parameter_group,
        "iterations_requested": up_args.iterations,
        "epochs_cap": up_args.epochs,
        "steps_per_epoch": args.num_anchor_images // up_args.anchor_batch_size,
        "epoch_capacity_steps": up_args.epochs * (args.num_anchor_images // up_args.anchor_batch_size),
        "with_anchor_preservation": up_args.with_anchor_preservation,
        "l1sp_weight": up_args.l1sp_weight, "l2sp_weight": up_args.l2sp_weight,
        "with_gradient_projection": up_args.with_gradient_projection,
        "with_selft": up_args.with_selft,
        "eval_interval": up_args.eval_interval,
        "seed": up_args.seed, "hflip": up_args.hflip, "noaug": up_args.noaug,
        "resolution": up_args.resolution,
    }

    # ---- 6. verify the produced checkpoint --------------------------------
    delta = out_dir / "delta.bin"
    if not delta.exists():
        raise SystemExit(f"FATAL: training produced no checkpoint at {delta}")
    child_sd = load_delta(delta)
    moved = n_same = 0
    max_move = 0.0
    if parent_sd is not None:
        for k, v in child_sd.items():
            if k in parent_sd:
                d = (v.float() - parent_sd[k].float()).abs().max().item()
                max_move = max(max_move, d)
                if d > 0:
                    moved += 1
                else:
                    n_same += 1
    report["child_checkpoint"] = {
        "path": str(delta), "sha256": sha256_file(delta),
        "bytes": delta.stat().st_size, "tensors": len(child_sd),
        "all_finite": bool(all(__import__("torch").isfinite(v).all().item()
                               for v in child_sd.values())),
        "tensors_moved_vs_parent": moved,
        "tensors_identical_to_parent": n_same,
        "max_abs_move_vs_parent": max_move,
    }
    report["l2sp_reference_check"] = l2_state["ref_check"]
    report["l2sp_trace"] = {
        "steps_recorded": len(l2_trace),
        "raw_first": l2_trace[0]["raw_l2sp"] if l2_trace else None,
        "raw_last": l2_trace[-1]["raw_l2sp"] if l2_trace else None,
        "weighted_last": (l2_trace[-1]["raw_l2sp"] * args.l2sp_weight) if l2_trace else None,
        "note": "weighted regularisation loss = raw_l2sp * l2sp_weight",
    }
    report["runtime"] = {
        "train_seconds": round(train_seconds, 1),
        # ACTUAL completed optimizer updates, from upstream's own counter. This
        # is the only field that evidences training completion.
        "optimizer_steps_completed": steps_completed,
        "optimizer_steps_source": ("upstream update_progress_and_checkpoint "
                                   "return value, captured per gradient sync"),
        "optimizer_sync_points": step_state["calls"],
        # DERIVED from train_seconds and the COMPLETED step count -- a
        # throughput figure only. It is not evidence of completion: dividing
        # train_seconds by it merely returns the denominator used here.
        "seconds_per_optimizer_step": round(
            train_seconds / max(1, steps_completed), 3),
        "seconds_per_optimizer_step_is_derived": True,
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_gpu_mem_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
    }
    # ---- intermediate dumps: inventoried by CONTENT, and the schedule checked.
    #
    # Each dump is evidence in its own right (the diagnostic selects one of them
    # as an arm), so it is recorded with its own digest and the expected
    # schedule is verified rather than assumed. A missing dump is reported here,
    # not discovered later by an analysis that silently scores fewer arms.
    if args.checkpoint_every > 0:
        want_steps = list(range(args.checkpoint_every,
                                steps_completed + 1, args.checkpoint_every))
        dumps, missing = [], []
        for st in want_steps:
            p = out_dir / f"delta-{st}"
            if not p.is_file() or p.stat().st_size == 0:
                missing.append(st)
                continue
            dumps.append({"step": st, "path": str(p), "sha256": sha256_file(p),
                          "bytes": p.stat().st_size})
        stray = sorted(q.name for q in out_dir.glob("delta-*")
                       if q.is_file() and q.name not in
                       {f"delta-{d['step']}" for d in dumps})
        digests = [d["sha256"] for d in dumps]
        report["dumps"] = {
            "checkpoint_every": args.checkpoint_every,
            "expected_steps": want_steps,
            "n_expected": len(want_steps),
            "n_present": len(dumps),
            "missing_steps": missing,
            "unexpected_files": stray,
            "all_digests_distinct": len(set(digests)) == len(digests),
            "complete": not missing and len(dumps) == len(want_steps),
            "source": ("upstream update_progress_and_checkpoint periodic save; "
                       "same save_pretrained(parameter_group) call as delta.bin"),
            "dumps": dumps,
        }
        print(f"[verify] dumps: {len(dumps)}/{len(want_steps)} present, "
              f"distinct digests={report['dumps']['all_digests_distinct']}"
              + (f", MISSING {missing}" if missing else ""))
        if not report["dumps"]["complete"]:
            print(f"[warn] intermediate dump schedule incomplete: missing "
                  f"{missing}; the report records this and validation will "
                  f"reject the trajectory", file=sys.stderr)

    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n[report] {args.report}")
    print(json.dumps({k: report[k] for k in ("checkpoint", "parent", "runtime",
                                             "child_checkpoint")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
