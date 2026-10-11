#!/usr/bin/env python3
"""Re-derive every number in docs/research/FEASIBILITY_BRIEF_2026-10-11.md. CPU only.

Design-dependent figures come from screen_measurement_inventory.py, so the brief,
the inventory and this checker cannot drift apart.

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

    # --- design counts and workload, from the inventory module -------------
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inv_mod", "scripts/seq/screen_measurement_inventory.py")
    inv_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv_mod)
    inv = inv_mod.inventory()

    check("arms in family CA", len(inv_mod.CA_ARMS), 4)
    check("arms in family NAT (ConAbl shares its CA cells)", len(inv_mod.NAT_ARMS), 3)
    check("edit/training seeds per arm", inv_mod.SEEDS, 2)
    check("request-1 checkpoints", inv["n_r1"], 22)
    check("request-2 checkpoints", inv["n_r2"], 76)
    check("SELECTED checkpoints (M0 and candidates excluded)", inv["n_selected"], 98)
    check("calibration candidates / intermediate dumps", inv["n_candidates"], 486)
    check("measured concepts on TEST", len(inv_mod.CONCEPTS_TEST), 9)
    check("flower is NOT a measured concept", "flower" in inv_mod.CONCEPTS_TEST, False)
    check("disjoint anchor is COCO-measurable", inv_mod.DISJOINT_ANCHOR, "potted plant")
    check("total images across every enumerated slot", inv["images"], 220_160)
    check("evaluation device-hours", round(inv["eval_h"], 1), 67.4, 0.1)
    t = inv["train"]
    check("ConAbl training device-hours (measured rate, 1000 steps)",
          round(t["ConAbl"], 2), 4.68, 0.01)
    check("same at upstream 2000 iterations", round(2 * t["ConAbl"], 1), 9.4, 0.1)
    check("ConAbl request-1 retrain cost if all-fresh is chosen prospectively",
          round(2 * inv_mod.CONABL_STEPS * inv["rates"]["conabl_s_per_step"] / 3600, 2), 0.47, 0.01)
    check("SPEED + SPEED+AIC edits (4 dial points x 52 selected)", t["n_speed_edits"], 208)
    check("LACU requests", t["n_lacu_requests"], 26)
    check("LACU device-hours, low", round(t["LACU_lo"]), 43)
    check("LACU device-hours, high", round(t["LACU_hi"]), 108)
    check("TOTAL device-hours, low", round(inv["total_lo"]), 119)
    check("TOTAL device-hours, high", round(inv["total_hi"]), 184)

    # --- contrasts ----------------------------------------------------------
    cs = inv_mod.contrasts()
    check("CA cells per arm per edit seed", len(inv_mod.CELLS), 8)
    for a, b in (("S", "P"), ("S", "I"), ("P", "I"), ("S", "A2_identity"), ("P", "A2_identity")):
        check(f"<{a},{b}> = 0", round(inv_mod.dot(cs[a], cs[b]), 9), 0.0)
    ref = [1.0 if x else 0.0 for x in inv_mod.REF_IS_HORSE]
    for name in ("S", "P", "I"):
        check(f"request-1 reference loading of {name} cancels",
              round(inv_mod.dot(cs[name], ref), 9), 0.0)
    check("S coefficients", cs["S"], [0.25, -0.25] * 4)
    check("I coefficients", cs["I"], [0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5])
    check("four prespecified simple effects present",
          sorted(k for k in cs if "|" in k),
          ["P|disjoint", "P|shared", "S|far", "S|near"])

    # --- precision, as observed-interval arithmetic --------------------------
    h20 = half * (10 / inv_mod.SCENES["TEST"]) ** 0.5
    check("per-cell half-width scaled to 20 scenes (pp)", round(h20, 2), 3.54, 0.01)
    sens = inv_mod.precision(h20)
    check("I half-width at rho=0 (pp)", round(sens[0.0]["I"], 2), 5.00, 0.01)
    check("I half-width at rho=+0.5 (pp)", round(sens[0.5]["I"], 2), 3.54, 0.01)
    check("S half-width at rho=0 (pp)", round(sens[0.0]["S"], 2), 2.50, 0.01)
    check("reading rule: MATERIAL+ needs observed point > 10 + h",
          round(10 + sens[0.0]["I"], 2), 15.00, 0.01)
    check("reading rule: NULL(10) needs |observed point| <= 10 - h",
          round(10 - sens[0.0]["I"], 2), 5.00, 0.01)

    # --- the smoke calibration ---------------------------------------------
    sm = inv_mod.smoke_cost()
    check("UNet passes per optimizer step at effective batch 108",
          inv_mod.passes_per_optimizer_step(), 4320)
    check("smoke optimizer steps", sm["optimizer_steps"], 136)
    check("smoke UNet passes", sm["unet_passes"], 587_520)
    check("smoke image-equivalents", round(sm["image_equivalents"]), 9792)
    check("smoke device-hours, low", round(sm["device_hours"][0], 2), 0.75, 0.01)
    check("smoke device-hours, high", round(sm["device_hours"][1], 2), 3.00, 0.01)

    # --- the brief must actually quote these ---------------------------------
    rev6 = pathlib.Path("docs/research/REVISION_6_CHANGES.md").read_text()
    smoke = pathlib.Path("docs/research/LACU_SMOKE_CALIBRATION.md").read_text()
    for q in ["1.1027 s/image", "0.8420 s/step", "**98**", "**486**", "**220,160**",
              "**67.4**", "**4.68**", "**208 edits**", "**43 – 108**",
              "119 – 184 device-hours", "`potted plant`", "`clock`", "0.47",
              "6.12", "2.24", "`ALREADY-SOLVES-CORE", "AIC-NOOP", "AIC-current"]:
        check(f"brief quotes {q!r}", q in brief, True)
    for q in ["98", "AIC-NOOP", "potted plant", "ANCHOR-UNMEASURED", "0.47 device-hours"]:
        check(f"revision-6 list quotes {q!r}", q in rev6, True)
    for q in ["136 optimizer steps", "0.75 – 3.0 device-hours", "candidates[selected − 1]",
              "`apple`", "every configuration OOMs"]:
        check(f"smoke proposal quotes {q!r}", q in smoke, True)
    # The retracted claims must appear ONLY inside the correction table, each
    # paired with its retraction -- so assert the retractions, not absence.
    for retraction in [
        '"training-free and deterministic"',          # the SPEED claim
        "**Wrong.** The editor has four randomness sources",
        "**98**",                                      # replaces 72
        "**Wrong twice:**",                            # the reuse claim
        "**Not measurable.**",                         # the flower claim
        "**Mis-stated.**",                             # the 17/3 pp claim
        "**Under-specified and degenerate in the native setting.**",
    ]:
        check(f"brief carries the retraction {retraction!r}", retraction in brief, True)
    check("brief no longer presents 72 as the count",
          "= 72" in brief or "**72** checkpoints**" in brief, False)

    print(f"\n{ok} passed / {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
