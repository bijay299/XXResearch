#!/usr/bin/env python3
"""A conservative, measured cost model for the matched-effectiveness diagnostic.

Why this exists. The runner previously carried hand-written per-stage estimates
(0.45 GPU-h per trajectory, 0.03 per development slot, 0.18 per test slot) and a
separately hand-written 1.008 GPU-h test reserve. Those two were inconsistent:
six test slots at 0.18 is 1.08 GPU-h, which the 1.008 reserve could not cover.
An estimate is now also a RUNTIME BOUND, so an under-estimate would kill
legitimate work, and an under-provisioned reserve would strand the confirmatory
stage after selection had already been paid for.

So every figure is derived here from what the completed run actually measured,
with explicit conservatism, and the reserve is defined as the SUM of the test
stage's own admitted estimates rather than guessed independently.

Conservatism, stated rather than implied:
  * per-image generation uses the MAXIMUM per-image time observed, not the mean;
  * fixed per-slot overhead uses the MAXIMUM observed, not the mean;
  * a margin (default 25%) is applied on top of that;
  * where a quantity was not timed at all (detection), the previously measured
    throughput is used and the fact that it is not re-measured is recorded.

CPU only. Reads the ledger and the evaluation reports; writes one JSON.

    python scripts/seq/diag_cost_model.py --diag_root <diag> --out <path>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

H = 3600.0
# From results/audit_v1/protocol_cost_model.json, measured at the pilot. The
# detector is not timed per slot in detect_report.json, so this is carried
# forward rather than re-measured here, and that is recorded in the output.
S_PER_DETECTED_IMAGE = 0.0718


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag_root", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--dev_images", type=int, default=80)
    ap.add_argument("--test_images", type=int, default=560)
    ap.add_argument("--test_slots", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    diag = Path(a.diag_root)
    ledger = Path(a.ledger) if a.ledger else diag / "gpu_budget.json"
    led = json.loads(ledger.read_text())
    entries = led["entries"]

    dev = [e for e in entries if e["stage"].startswith("eval:eval_dev")]
    train = [e for e in entries if e["stage"].startswith("train:")]
    if not dev or not train:
        raise SystemExit("FATAL: the ledger holds no completed development or "
                         "training stages to model from")

    # ---- generation, per image, from the generators' own reports
    gen_per_image, gen_elapsed = [], []
    for d in sorted((diag / "eval_dev").iterdir()):
        ir = json.loads((d / "image_report.json").read_text())
        n = int(ir["expected_images"])
        gen_elapsed.append(ir["elapsed_seconds"])
        gen_per_image.append(ir["elapsed_seconds"] / n)
    gen_max = max(gen_per_image)
    gen_mean = statistics.mean(gen_per_image)

    # ---- fixed per-slot overhead: slot wall MINUS generation MINUS detection
    overheads = []
    for e in dev:
        # pair each charged slot with its own generation report
        name = e["stage"].rsplit(":", 1)[1]
        ir = json.loads((diag / "eval_dev" / name / "image_report.json").read_text())
        det = int(ir["expected_images"]) * S_PER_DETECTED_IMAGE
        overheads.append(e["wall_seconds"] - ir["elapsed_seconds"] - det)
    ov_max = max(overheads)
    ov_mean = statistics.mean(overheads)

    def slot_seconds(n_images: int, conservative: bool = True) -> float:
        g = gen_max if conservative else gen_mean
        o = ov_max if conservative else ov_mean
        return o + n_images * (g + S_PER_DETECTED_IMAGE)

    m = 1.0 + a.margin
    dev_slot = slot_seconds(a.dev_images) * m / H
    test_slot = slot_seconds(a.test_images) * m / H
    train_wall_max = max(e["wall_seconds"] for e in train)
    train_est = train_wall_max * m / H

    reserve = round(a.test_slots * test_slot, 6)

    model = {
        "_what_this_is": (
            "Conservative per-stage GPU-hour estimates for the matched-"
            "effectiveness diagnostic, derived from the measured run. Each "
            "estimate is BOTH an admission amount and a runtime bound, so it is "
            "deliberately above the observed cost."),
        "_derived_from": {
            "ledger": str(ledger),
            "completed_development_slots": len(dev),
            "completed_training_stages": len(train),
        },
        "conservatism": {
            "per_image_generation": "MAXIMUM observed, not mean",
            "per_slot_overhead": "MAXIMUM observed, not mean",
            "margin_applied": a.margin,
            "detection_throughput": (
                f"{S_PER_DETECTED_IMAGE} s/image carried forward from "
                f"results/audit_v1/protocol_cost_model.json; detect_report.json "
                f"does not time the detector, so this figure is NOT re-measured "
                f"here and is a carried assumption"),
        },
        "measured": {
            "generation_s_per_image_mean": round(gen_mean, 4),
            "generation_s_per_image_max": round(gen_max, 4),
            "per_slot_overhead_s_mean": round(ov_mean, 2),
            "per_slot_overhead_s_max": round(ov_max, 2),
            "training_wall_seconds_max": round(train_wall_max, 1),
            "development_slot_wall_seconds_observed_max":
                round(max(e["wall_seconds"] for e in dev), 1),
        },
        "estimates_gpu_hours": {
            "train_trajectory": round(train_est, 6),
            "development_slot": round(dev_slot, 6),
            "frozen_test_slot": round(test_slot, 6),
        },
        "runtime_bounds_seconds_single_gpu": {
            "train_trajectory": round(train_est * H, 1),
            "development_slot": round(dev_slot * H, 1),
            "frozen_test_slot": round(test_slot * H, 1),
        },
        "frozen_test_reserve_gpu_hours": reserve,
        "reserve_reconciliation": {
            "rule": ("the reserve IS the sum of the test stage's own admitted "
                     "estimates, so the two can never disagree"),
            "test_slots": a.test_slots,
            "per_slot_estimate": round(test_slot, 6),
            "sum": reserve,
            "previous_hand_written_reserve": 1.008,
            "previous_runner_per_slot_estimate": 0.18,
            "previous_inconsistency": (
                "6 x 0.18 = 1.08 GPU-h of admissions against a 1.008 GPU-h "
                "reserve: the reserve could not cover the stage it existed to "
                "protect, and the shortfall would have appeared only after "
                "selection had already been paid for"),
        },
        "observed_vs_estimate": {
            "development_slot_observed_max_gpu_hours":
                round(max(e["gpu_hours"] for e in dev), 6),
            "development_slot_estimate_gpu_hours": round(dev_slot, 6),
            "headroom_factor": round(
                dev_slot / max(e["gpu_hours"] for e in dev), 3),
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2) + "\n")

    print(f"measured: generation {gen_mean:.4f} s/img mean, {gen_max:.4f} max; "
          f"per-slot overhead {ov_mean:.1f}s mean, {ov_max:.1f}s max")
    print(f"margin  : {a.margin:.0%}")
    print(f"estimates (GPU-h): train {train_est:.4f}  dev-slot {dev_slot:.4f}  "
          f"test-slot {test_slot:.4f}")
    print(f"runtime bounds (s): train {train_est*H:.0f}  dev-slot {dev_slot*H:.0f}  "
          f"test-slot {test_slot*H:.0f}")
    print(f"dev-slot headroom over the worst observed slot: "
          f"{model['observed_vs_estimate']['headroom_factor']}x")
    print(f"frozen-test reserve = {a.test_slots} x {test_slot:.4f} = {reserve:.4f} GPU-h "
          f"(was a hand-written 1.008 against 6x0.18=1.08 of admissions)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
