#!/usr/bin/env python3
"""Resource estimate for the proposed matched-effectiveness diagnostic.

Throughput constants are MEASURED on this host, read from the existing run
reports rather than assumed. Everything else is arithmetic over the run matrix,
so the estimate can be re-derived and audited.

GPU-hours and wall time are reported separately: four A100s are present but
shared without a scheduler, so wall time depends on availability this estimate
cannot predict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DELTA_MIB = 76686190 / 1024 / 1024
IMG_KB = 98.0


def measured(seq_root: Path) -> dict:
    """Read measured throughput from the completed pilot runs."""
    steps, gens = [], []
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
    # Detection: elapsed between the generation and detection reports, per image.
    det = []
    for root in (seq_root / "eval", seq_root / "eval_seed29"):
        for c in ("MA", "MAB", "MAB_L2", "MAC", "MAC_L2"):
            g, d = root / c / "image_report.json", root / c / "detect_report.json"
            if g.exists() and d.exists():
                from datetime import datetime
                t0 = datetime.fromisoformat(json.loads(g.read_text())["finished_utc"])
                t1 = datetime.fromisoformat(json.loads(d.read_text())["finished_utc"])
                n = json.loads(d.read_text())["images_scored"]
                if 0 < (t1 - t0).total_seconds() < 600:
                    det.append((t1 - t0).total_seconds() / n)
    return {
        "s_per_optimizer_step": round(sum(steps) / len(steps), 4),
        "s_per_generated_image": round(sum(gens) / len(gens), 4),
        "s_per_detected_image": round(sum(det) / len(det), 4) if det else 0.08,
        "n_runs_sampled": len(steps),
        "n_gen_sampled": len(gens),
        "n_det_sampled": len(det),
    }


def plan(m: dict, *, seeds: int, branches: int, steps: int, dumps_evaluated: int,
         dev_images: int, final_images: int, final_checkpoints: int,
         reference_dev_evals: int, process_setup_s: float, contingency: float,
         label: str) -> dict:
    tr_s = seeds * branches * steps * m["s_per_optimizer_step"]

    n_dev_evals = seeds * branches * dumps_evaluated + reference_dev_evals
    dev_imgs = n_dev_evals * dev_images
    dev_s = dev_imgs * (m["s_per_generated_image"] + m["s_per_detected_image"])

    fin_imgs = final_checkpoints * final_images
    fin_s = fin_imgs * (m["s_per_generated_image"] + m["s_per_detected_image"])

    n_proc = seeds * branches + n_dev_evals + final_checkpoints
    setup_s = n_proc * process_setup_s

    core_s = tr_s + dev_s + fin_s + setup_s
    total_s = core_s * (1 + contingency)

    peak_dumps = seeds * branches * (steps // 100)
    storage_gib = (peak_dumps * DELTA_MIB / 1024
                   + (dev_imgs + fin_imgs) * IMG_KB / 1024 / 1024)

    h = lambda s: round(s / 3600, 3)
    return {
        "label": label,
        "run_matrix": {
            "training_seeds": seeds, "branches": branches,
            "new_trajectories": seeds * branches, "steps_each": steps,
            "dumps_saved_each": steps // 100,
            "dumps_evaluated_each": dumps_evaluated,
            "dev_checkpoint_evaluations": n_dev_evals,
            "final_checkpoint_evaluations": final_checkpoints,
        },
        "gpu_hours": {
            "training": h(tr_s),
            "development_selection": h(dev_s),
            "final_confirmatory_evaluation": h(fin_s),
            "process_setup": h(setup_s),
            "subtotal": h(core_s),
            f"total_with_{int(contingency*100)}pct_contingency": h(total_s),
        },
        "images": {"development": dev_imgs, "final": fin_imgs,
                   "total": dev_imgs + fin_imgs},
        "storage_gib": {
            "peak_intermediate_checkpoints": round(peak_dumps * DELTA_MIB / 1024, 2),
            "images": round((dev_imgs + fin_imgs) * IMG_KB / 1024 / 1024, 2),
            "peak_total": round(storage_gib, 2),
            "after_pruning_unmatched_dumps": round(
                (seeds * branches * DELTA_MIB / 1024)
                + (dev_imgs + fin_imgs) * IMG_KB / 1024 / 1024, 2),
        },
        "wall_time_hours_note": {
            "on_4_idle_gpus": round(
                (tr_s / min(4, seeds * branches) + dev_s / 2 + fin_s / 2
                 + setup_s) / 3600, 2),
            "on_2_idle_gpus": round(
                (tr_s / 2 + dev_s / 2 + fin_s / 2 + setup_s) / 3600, 2),
            "caveat": ("wall time is NOT GPU-hours; the GPUs are shared without a "
                       "scheduler and availability is not guaranteed"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--out", default="results/audit_v1/protocol_cost_model.json")
    args = ap.parse_args()

    m = measured(Path(args.seq_root))
    assumptions = {
        "throughput_source": "measured on this host from the 10 pilot training runs "
                             "and 11 pilot checkpoint evaluations",
        "process_setup_s": 25.0,
        "process_setup_basis": "SD-1.5 pipeline construction and checkpoint load per "
                               "invocation; an assumption, not measured in isolation, "
                               "because the pilot's seconds_per_image already "
                               "amortises setup over 280 images",
        "contingency": 0.25,
        "contingency_basis": "one re-run of a failed stage plus bisection overshoot",
        "dev_images_per_checkpoint": 80,
        "dev_composition": "new target 10 prompts x 4 gen seeds = 40 (2.5 pp "
                           "resolution, the matching quantity); cat 10 x 2 = 20; "
                           "bird 10 x 2 = 20",
        "dumps_evaluated_basis": "bisection over 10 saved dumps, assuming suppression "
                                 "is monotone in steps; monotonicity is checked at the "
                                 "evaluated points and a full scan is the fallback",
        "final_images_per_checkpoint": 280,
        "final_composition": "7 categories x 10 NEW prompts x 4 gen seeds, newly "
                             "frozen; the pilot set is exploratory from here on",
    }

    plans = [
        plan(m, seeds=2, branches=1, steps=1000, dumps_evaluated=4, dev_images=80,
             final_images=280, final_checkpoints=7, reference_dev_evals=6,
             process_setup_s=25.0, contingency=0.25,
             label="CORE — dog branch only, 2 training seeds. The sandwich branch "
                   "needs no new compute: it is declared infeasible for matched "
                   "comparison at L2-SP 25000 from existing evidence (+7.5 pp "
                   "suppression, below the meaningful region)"),
        plan(m, seeds=2, branches=2, steps=1000, dumps_evaluated=4, dev_images=80,
             final_images=280, final_checkpoints=11, reference_dev_evals=6,
             process_setup_s=25.0, contingency=0.25,
             label="EXTENDED — both branches confirmatory. NOT a drop-in option: it "
                   "presupposes a sandwich L2-SP coefficient already inside the "
                   "meaningful region, and the coefficient search that would find "
                   "one is NOT costed here"),
        plan(m, seeds=1, branches=1, steps=1000, dumps_evaluated=4, dev_images=80,
             final_images=280, final_checkpoints=4, reference_dev_evals=3,
             process_setup_s=25.0, contingency=0.25,
             label="FALLBACK — dog branch, 1 training seed (direction check only, "
                   "no cross-seed agreement possible)"),
    ]

    out = {"measured_throughput": m, "assumptions": assumptions, "plans": plans,
           "ceiling_gpu_hours": 4.0}
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("measured:", json.dumps(m))
    print()
    for p in plans:
        g = p["gpu_hours"]
        print(f"{p['label']}")
        print(f"  train {g['training']:.3f} + dev {g['development_selection']:.3f} "
              f"+ final {g['final_confirmatory_evaluation']:.3f} "
              f"+ setup {g['process_setup']:.3f} = {g['subtotal']:.3f} GPU-h")
        print(f"  with 25% contingency: {g['total_with_25pct_contingency']:.2f} GPU-h "
              f"({'WITHIN' if g['total_with_25pct_contingency'] <= 4 else 'OVER'} "
              f"the 4 GPU-h ceiling)")
        print(f"  images {p['images']['total']}, peak storage "
              f"{p['storage_gib']['peak_total']} GiB, wall ~"
              f"{p['wall_time_hours_note']['on_2_idle_gpus']} h on 2 idle GPUs")
        print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
