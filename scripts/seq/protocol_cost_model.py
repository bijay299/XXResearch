#!/usr/bin/env python3
"""Resource estimate for the bounded matched-effectiveness diagnostic.

The run matrix is the one specified for review: dog branch only, two training
seeds, a FULL fixed 100-step scan (no bisection), reusing MA17/MA29 and the
existing L2=25000 final children. No lower-L2 stage and no sandwich coefficient
search are included.

Throughput constants are MEASURED on this host, read from the completed pilot
reports rather than assumed. Everything else is arithmetic over the run matrix.

GPU-hours and wall time are reported separately: four A100s are present but
shared without a scheduler, so wall time depends on availability this estimate
cannot predict.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

DELTA_BYTES = 76686190
IMG_KB = 98.0
GIB = 1024 ** 3


def measured(seq_root: Path) -> dict:
    """Read measured throughput from the completed pilot runs."""
    steps, gens, det = [], [], []
    for s in ("seed17", "seed29"):
        for c in ("MA", "MAB", "MAB_L2", "MAC", "MAC_L2"):
            p = seq_root / "models" / s / c / "train_report.json"
            if p.exists():
                steps.append(json.loads(p.read_text())["runtime"]
                             ["seconds_per_optimizer_step"])
    for root in (seq_root / "eval", seq_root / "eval_seed29"):
        for c in ("M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"):
            p = root / c / "image_report.json"
            if p.exists():
                gens.append(json.loads(p.read_text())["seconds_per_image"])
            # Detection cost is the gap between the generation and detection
            # reports. M0 is excluded: its images were generated in a separate
            # earlier session and reused, so that gap is not a detection time.
            if c == "M0":
                continue
            g, d = root / c / "image_report.json", root / c / "detect_report.json"
            if g.exists() and d.exists():
                t0 = datetime.fromisoformat(json.loads(g.read_text())["finished_utc"])
                t1 = datetime.fromisoformat(json.loads(d.read_text())["finished_utc"])
                n = json.loads(d.read_text())["images_scored"]
                dt = (t1 - t0).total_seconds()
                if 0 < dt < 600:
                    det.append(dt / n)
    return {
        "s_per_optimizer_step": round(sum(steps) / len(steps), 4),
        "s_per_generated_image": round(sum(gens) / len(gens), 4),
        "s_per_detected_image": round(sum(det) / len(det), 4),
        "n_training_runs_sampled": len(steps),
        "n_generation_reports_sampled": len(gens),
        "n_detection_reports_sampled": len(det),
        "source": "the 10 completed pilot training runs and the pilot checkpoint "
                  "evaluations at a9de625, on A100-SXM4-40GB",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--out", default="results/audit_v1/protocol_cost_model.json")
    args = ap.parse_args()

    m = measured(Path(args.seq_root))
    s_step = m["s_per_optimizer_step"]
    s_img = m["s_per_generated_image"] + m["s_per_detected_image"]

    # ---- run matrix (fixed by the review brief) ---------------------------
    seeds, steps = 2, 1000
    dump_every, n_dumps = 100, 10            # steps 100..1000, full fixed scan

    dev_prompts_total = 20                   # dog branch only, balanced lit/par
    dev_gen_seeds = 4
    dev_images_per_ckpt = dev_prompts_total * dev_gen_seeds      # 80
    dev_ckpts = seeds * (1 + 1 + n_dumps)    # per seed: MA + L2 + ten dumps = 12
    dev_images = dev_ckpts * dev_images_per_ckpt                 # 1920

    test_prompts_per_cat = 20
    test_cats, test_gen_seeds = 7, 4
    test_images_per_ckpt = test_prompts_per_cat * test_gen_seeds * test_cats  # 560
    test_ckpts = seeds * 3                   # per seed: MA, L2, selected-U
    test_images = test_ckpts * test_images_per_ckpt              # 3360

    n_train_invocations = seeds                                  # 2
    n_gen_invocations = dev_ckpts + test_ckpts                   # 30
    n_det_invocations = dev_ckpts + test_ckpts                   # 30
    n_invocations = n_train_invocations + n_gen_invocations + n_det_invocations  # 62
    setup_s_each = 25.0
    contingency = 0.25

    train_s = seeds * steps * s_step
    dev_s = dev_images * s_img
    test_s = test_images * s_img
    setup_s = n_invocations * setup_s_each
    core_s = train_s + dev_s + test_s + setup_s
    total_s = core_s * (1 + contingency)
    h = lambda x: round(x / 3600, 3)

    storage = {
        "intermediate_dumps": {
            "n": seeds * n_dumps,
            "bytes_each": DELTA_BYTES,
            "total_bytes": seeds * n_dumps * DELTA_BYTES,
            "total_gib": round(seeds * n_dumps * DELTA_BYTES / GIB, 2),
            "note": f"{seeds * n_dumps} dumps x {DELTA_BYTES:,} bytes; this is the "
                    f"figure an earlier version of this model got wrong",
        },
        "images_gib": round((dev_images + test_images) * IMG_KB * 1024 / GIB, 2),
        "peak_total_gib": round(
            (seeds * n_dumps * DELTA_BYTES
             + (dev_images + test_images) * IMG_KB * 1024) / GIB, 2),
        "after_pruning_unselected_dumps_gib": round(
            (seeds * DELTA_BYTES
             + (dev_images + test_images) * IMG_KB * 1024) / GIB, 2),
        "excludes": "optimizer/RNG state (not saved by the current trainer) and "
                    "any logs",
    }

    out = {
        "plan": "BOUNDED matched-effectiveness diagnostic, dog branch, 2 training "
                "seeds, full fixed 100-step scan",
        "status": "PROPOSED FOR REVIEW. Not approved, not authorised to run.",
        "measured_throughput": m,
        "reused_without_new_compute": [
            "MA17 and MA29 parents (existing)",
            "MAB_L2 at L2=25000 for both seeds (existing final children)",
        ],
        "explicitly_excluded": [
            "any lower-L2 / alternative-coefficient stage",
            "any sandwich-branch coefficient search",
            "a new M0 evaluation on the frozen test set",
            "an unmatched full-step (MAB) reference arm on the frozen test set",
        ],
        "run_matrix": {
            "new_unregularised_dog_trajectories": seeds,
            "optimizer_steps_each": steps,
            "dumps_saved_and_scored_each": n_dumps,
            "dump_schedule": f"steps {dump_every}..{steps} every {dump_every} "
                             f"(full fixed scan, no bisection)",
            "development": {
                "prompts": dev_prompts_total,
                "balance": "balanced literal/paraphrase",
                "gen_seeds": dev_gen_seeds,
                "images_per_checkpoint": dev_images_per_ckpt,
                "checkpoints_scored": dev_ckpts,
                "composition": "per seed: MA + L2 endpoint + 10 dumps = 12",
                "images_total": dev_images,
            },
            "frozen_test": {
                "prompts_per_category": test_prompts_per_cat,
                "categories": test_cats,
                "gen_seeds": test_gen_seeds,
                "images_per_checkpoint": test_images_per_ckpt,
                "checkpoints_scored": test_ckpts,
                "composition": "per seed: MA, L2, selected-U = 3",
                "images_total": test_images,
            },
            "images_total": dev_images + test_images,
        },
        "gpu_hours": {
            "training": h(train_s),
            "development_generation_and_detection": h(dev_s),
            "frozen_test_generation_and_detection": h(test_s),
            "generation_and_detection_combined": h(dev_s + test_s),
            "process_setup": h(setup_s),
            "subtotal": h(core_s),
            "total_with_25pct_contingency": h(total_s),
        },
        "assumptions": {
            "process_setup": {
                "n_invocations": n_invocations,
                "breakdown": f"{n_train_invocations} training + "
                             f"{n_gen_invocations} generation + "
                             f"{n_det_invocations} detection",
                "seconds_each": setup_s_each,
                "DISCLOSURE": (
                    "This 0.431 GPU-h setup term may be PARTLY DOUBLE-COUNTED. "
                    "The measured s_per_generated_image is elapsed/n_images from "
                    "the pilot reports, so it already amortises that run's "
                    "pipeline construction and checkpoint load over its images. "
                    "Adding a separate 25 s per invocation therefore charges some "
                    "setup twice. It is kept as a deliberate upper bound because "
                    "the new runs have many SMALL evaluations (80 images) where "
                    "per-invocation setup is a much larger share than in the "
                    "pilot's 280-image runs. Reconcile against the actual "
                    "implementation before the figure is relied on: if dev "
                    "checkpoints are scored within one process, most of this term "
                    "disappears and the subtotal falls toward "
                    f"{h(core_s - setup_s):.3f} GPU-h."),
            },
            "contingency": {
                "fraction": contingency,
                "basis": "one re-run of a failed stage; no allowance for a change "
                         "of design",
            },
            "not_included": [
                "human annotation labour (no GPU)",
                "CPU analysis time",
                "any re-run caused by a failed match (see the stop rule)",
            ],
        },
        "wall_time_hours": {
            "note": "NOT GPU-hours. The GPUs are shared without a scheduler and "
                    "availability is not guaranteed; only independently verified "
                    "idle devices would be used.",
            "two_trajectories_in_parallel_then_serial_scoring": round(
                (train_s / 2 + dev_s + test_s + setup_s) / 3600, 2),
            "scoring_split_across_two_idle_gpus": round(
                (train_s / 2 + (dev_s + test_s) / 2 + setup_s) / 3600, 2),
        },
        "storage": storage,
        "ceiling_gpu_hours_proposed_not_approved": 4.0,
        "within_proposed_ceiling": h(total_s) <= 4.0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    g = out["gpu_hours"]
    print(f"measured: step={s_step}s  gen={m['s_per_generated_image']}s  "
          f"det={m['s_per_detected_image']}s  (gen+det={s_img:.4f}s/img)")
    print(f"\nrun matrix: {seeds} trajectories x {steps} steps, {n_dumps} dumps each")
    print(f"  development : {dev_ckpts} checkpoints x {dev_images_per_ckpt} = "
          f"{dev_images} images")
    print(f"  frozen test : {test_ckpts} checkpoints x {test_images_per_ckpt} = "
          f"{test_images} images")
    print(f"  images total: {dev_images + test_images}")
    print(f"\ntraining {g['training']:.3f} + gen/det "
          f"{g['generation_and_detection_combined']:.3f} + setup "
          f"{g['process_setup']:.3f} = {g['subtotal']:.3f} GPU-h")
    print(f"with {int(contingency*100)}% contingency: "
          f"{g['total_with_25pct_contingency']:.3f} GPU-h "
          f"({'within' if out['within_proposed_ceiling'] else 'OVER'} the "
          f"proposed 4 GPU-h ceiling)")
    print(f"  without the possibly-double-counted setup term: "
          f"{h(core_s - setup_s):.3f} GPU-h subtotal")
    print(f"\nstorage: {storage['intermediate_dumps']['n']} dumps = "
          f"{storage['intermediate_dumps']['total_gib']} GiB, images "
          f"{storage['images_gib']} GiB, peak {storage['peak_total_gib']} GiB")
    print(f"wall time: ~{out['wall_time_hours']['scoring_split_across_two_idle_gpus']} h "
          f"on two idle GPUs (not GPU-hours)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
