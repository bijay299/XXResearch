#!/usr/bin/env python3
"""Re-derive every number in docs/research/FEASIBILITY_BRIEF_2026-10-11.md. CPU only.

The brief quotes measured unit costs from the Amendment 01 ledger, interval
widths from the committed analysis, and arithmetic built on both. This script
recomputes each from the committed source and asserts the brief's figure, so a
reviewer does not have to take the table on trust. It reads only committed
files; it runs no model and touches no GPU.

    python scripts/seq/check_feasibility_brief_numbers.py
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

LEDGER = "results/audit_v1/amendment01_evidence/gpu_usage_record.json"
ANALYSIS = "results/audit_v1/amendment01_evidence/analysis_test.json"
BRIEF = "docs/research/FEASIBILITY_BRIEF_2026-10-11.md"

ok = fail = 0


def check(name: str, got, want, tol=0.0) -> None:
    global ok, fail
    good = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    print(f"{'ok  ' if good else 'FAIL'} {name}: got {got!r} want {want!r}"
          + (f" (+-{tol})" if tol else ""))
    ok, fail = (ok + 1, fail) if good else (ok, fail + 1)


def main() -> int:
    led = json.load(open(LEDGER))["entries"]
    brief = pathlib.Path(BRIEF).read_text()

    # --- measured unit costs -------------------------------------------------
    test_slots = [e["wall_seconds"] for e in led if e["stage"].startswith("eval:eval_test:")]
    dev_slots = [e["wall_seconds"] for e in led if e["stage"].startswith("eval:eval_dev:")]
    check("frozen-test slots counted", len(test_slots), 6)
    check("test slot seconds min", min(test_slots), 609)
    check("test slot seconds max", max(test_slots), 625)
    check("s/image on 560-image slots", statistics.mean(test_slots) / 560, 1.10, 0.01)
    check("s/image on 80-image slots", statistics.mean(dev_slots) / 80, 1.36, 0.02)

    train = [e["wall_seconds"] for e in led if e["stage"].startswith("train:")]
    check("ConAbl training stages recorded", len(train), 4)
    check("ConAbl 1000-step wall seconds", max(train), 842)
    check("ConAbl s/step at 1000 steps", max(train) / 1000, 0.84, 0.005)
    total_dev_h = sum(e["wall_seconds"] * e["gpus_occupied"] for e in led) / 3600
    check("ledger total device-hours", round(total_dev_h, 4), 2.9014, 1e-4)
    check("ledger entries", len(led), 56)

    # --- precision, from the committed analysis ------------------------------
    est = json.load(open(ANALYSIS))["per_seed"]["17"]["estimates"]
    widths = {k: v["hi95"] - v["lo95"] for k, v in est.items()
              if isinstance(v, dict) and "lo95" in v}
    def med(pat):
        w = [x for k, x in widths.items() if re.fullmatch(pat, k)]
        assert w, pat
        return statistics.median(w)
    contrast = med(r"contrast\[[a-z]+\|0\.3\]")
    marg = med(r"D\[[a-z]+\|0\.3\]\|L2")
    check("median paired-contrast width at 80 images/category (pp)", contrast, 10.0)
    check("median marginal-rate width (pp)", marg, 22.5)
    half = contrast / 2
    check("contrast half-width (pp)", half, 5.0)
    inter_half = half * 2 ** 0.5
    check("interaction half-width at n=80 (pp)", round(inter_half, 1), 7.1, 0.05)
    check("I MATERIAL needs |point| >= (pp)", round(10 + inter_half), 17)
    check("I NULL(10) needs |point| <= (pp)", round(10 - inter_half), 3)
    for target, want in ((5.0, 160), (3.5, 330)):
        n = 80 * (inter_half / target) ** 2
        check(f"images/cell for interaction half-width {target} pp", round(n, -1), want, 10)

    # --- design counts -------------------------------------------------------
    ca = {"ConAbl": 4 + 16, "SPEED": 2 + 8, "SPEED+AIC": 2 + 8, "LACU": 4 + 16}
    nat = {"LACU": 2 + 4, "SPEED": 1 + 2, "SPEED+AIC": 1 + 2, "ConAbl": 0}
    check("CA ConAbl checkpoints", ca["ConAbl"], 20)
    check("CA SPEED checkpoints (seed factor degenerate)", ca["SPEED"], 10)
    check("NAT LACU checkpoints", nat["LACU"], 6)
    check("total checkpoints", sum(ca.values()) + sum(nat.values()), 72)

    # --- costed plan ---------------------------------------------------------
    s_per_step = max(train) / 1000
    conabl_h = ca["ConAbl"] * 1000 * s_per_step / 3600
    check("CA ConAbl training device-hours at 1000 iterations", round(conabl_h, 1), 4.7, 0.05)
    check("same at upstream 2000 iterations", round(2 * conabl_h, 1), 9.4, 0.1)
    check("ConAbl request-1 retrain cost if reuse fails (device-h)",
          round(2 * 1000 * s_per_step / 3600, 2), 0.47, 0.05)

    # LACU: passes per optimizer step is invariant to the batch/accum split
    eff = 108
    per_step = eff * 2 * 20
    check("LACU UNet passes per optimizer step at effective batch 108", per_step, 4320)
    lo_passes, hi_passes = per_step * 250, per_step * 350
    check("LACU passes per request, 250 steps (M)", round(lo_passes / 1e6, 2), 1.08, 0.01)
    check("LACU passes per request, 350 steps (M)", round(hi_passes / 1e6, 2), 1.51, 0.01)
    # one of our images = 30 sampler steps x 2 (CFG) = 60 passes ~ 1.0 s at batch 1
    img_eq_lo, img_eq_hi = lo_passes / 60, hi_passes / 60
    check("LACU image-equivalents per request, low (k)", round(img_eq_lo / 1000, 1), 18.0, 0.1)
    check("LACU image-equivalents per request, high (k)", round(img_eq_hi / 1000, 1), 25.2, 0.1)
    check("LACU device-hours at batch-1 throughput, low", round(img_eq_lo / 3600, 1), 5.0, 0.1)
    check("LACU device-hours at batch-1 throughput, high", round(img_eq_hi / 3600, 1), 7.0, 0.1)
    check("LACU device-hours with a 4x batched speed-up (low end)",
          round(img_eq_lo / 3600 / 4, 2), 1.25, 0.01)
    check("LACU device-hours with a 2x batched speed-up (high end)",
          round(img_eq_hi / 3600 / 2, 1), 3.5, 0.1)
    check("mapping rescore passes per stage (k)", 100 * 10 * 32 / 1000, 32.0)

    lacu_reqs = ca["LACU"] + nat["LACU"]
    check("LACU requests", lacu_reqs, 26)
    check("LACU training device-hours, low", round(lacu_reqs * 1.5), 39)
    check("LACU training device-hours, high", round(lacu_reqs * 4.0), 104)

    # evaluation, at 160 images per category-cell
    per_img = statistics.mean(test_slots) / 560
    ckpts = sum(ca.values()) + sum(nat.values())
    test_h = ckpts * 8 * 160 * per_img / 3600
    acc_h = ckpts * 3 * 160 * per_img / 3600
    r_h = ckpts * 160 * per_img / 3600
    check("TEST device-hours (8 categories x 160)", round(test_h), 28, 1)
    check("ACCEPT device-hours (3 x 160)", round(acc_h), 11, 1)
    check("R device-hours (1 x 160)", round(r_h, 1), 3.6, 0.3)
    check("per-checkpoint TEST device-hours", round(8 * 160 * per_img / 3600, 2), 0.39, 0.01)

    # --- the brief must actually quote these ---------------------------------
    for s in ["1.10 s/image", "0.84 s/step", "**4,320**", "1.5 – 4.0 device-hours", "1.2 – 3.5 device-hours",
              "**10.0 pp**", "≈ **7 pp**", "≈ **160**", "**72**", "≈ 90 – 160 device-hours",
              "2.9014 device-hours", "204"]:
        check(f"brief quotes {s!r}", s in brief, True)

    print(f"\n{ok} passed / {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
