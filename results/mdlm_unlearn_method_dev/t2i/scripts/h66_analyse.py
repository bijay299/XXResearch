"""H66 analysis: FRP (our port of a PUBLISHED method) against the frozen attack grid.

Written BEFORE any FRP attack number exists, so PREREG_H66's declared bands are executed rather
than recalled. FRP is THEIR method: if it wins, the report says a published method beat everything
we built.

PREREG_H66 S5 bands, verbatim in intent:
  retain >= 0.85 and max-over-valid H <= 0.40  -> the mandate's bar met BY A PUBLISHED METHOD;
                                                  PREREG_H40 refuted by someone else's work
  retain >= 0.85 and 0.40 < H <= 0.50          -> real frontier point short of H40's threshold
  H -> 1.000 like every other arm               -> the adversary defeats FRP too; a STRONGER
                                                  version of the phase's negative result
  no rho <= 20% erases to 0.400                 -> the PORT does not transfer; a port/domain
                                                  limitation, NOT evidence against the paper
  utility collapses at the rho needed to erase  -> inseparability reappears in pruning as well as
                                                  in noise (a second route to PREREG_H51 S7)

Run: python scripts/h66_analyse.py
"""
import glob, json, os

OUT = "/data/bijaypandey/t2i_unlearn/outputs"
FROZEN = {("0", "1e-5"), ("1", "1e-5"), ("6", "1e-5"),
          ("0", "5e-5"), ("1", "5e-5"), ("0", "1e-4")}
ERASE_THRESHOLD = 0.400      # AdvUnlearn's church depth; PREREG_H66 S4
BAR_RETAIN, BAR_H40, BAR_H47 = 0.85, 0.40, 0.50
# the H65 comparators, measured under the IDENTICAL frozen grid, so this is budget-matched
BASELINE = {"church": {"retain8": 0.992, "max_valid_H": 1.000},
            "parachute": {"retain8": 1.000, "max_valid_H": 1.000}}


def ladder(concept):
    rows = []
    for f in sorted(glob.glob(f"{OUT}/gen_frp_{concept}_r*_l*.json")):
        r = json.load(open(f))
        m = r.get("active_target_metrics") or {}
        if not m:
            continue
        b = os.path.basename(f)
        rows.append({"arm": r["label"], "file": b,
                     "target_suppression": m["target_suppression_acc"],
                     "utility_all_nontarget": m["utility_all_nontarget"],
                     "retain8": m["legacy_retain8"],
                     "erases": m["target_suppression_acc"] <= ERASE_THRESHOLD})
    return rows


def attacks(concept):
    out = []
    for f in sorted(glob.glob(f"{OUT}/atk_H66_frp_{concept}_sd*_lr*.json")):
        d = json.load(open(f))
        h = d["hparams"]
        sd = str(h["seed"])
        lr = f"{h['lr']:.0e}".replace("e-05", "e-5").replace("e-04", "e-4")
        pau = d.get("post_attack_utility", {})
        out.append({"seed": sd, "lr": lr, "in_frozen_grid": (sd, lr) in FROZEN,
                    "valid": bool(d["attack_validity"]["valid"]),
                    "T_rise": d["attack_validity"]["T_rise"],
                    "H_before": d["recovery"]["H_before"],
                    "H_after": d["recovery"]["H_after"],
                    "pau_after": (pau.get("retained_capability") or {}).get("after")})
    return out


def main():
    res = {"study": "h66_analysis", "prereg": "docs/PREREG_H66_frp_port.md",
           "method": "FRP (Forget-Retain Pruning), arXiv 2608.25429 -- OUR PORT of a PUBLISHED "
                     "method. Not the authors' code; not a reproduction of their numbers.",
           "budget_matched_to": "PROTOCOL_FROZEN_v1, the same six-point grid H65 measured the "
                                "official AdvUnlearn baselines under",
           "per_concept": {}}

    for c in ("church", "parachute"):
        lad = ladder(c)
        sel_file = f"{OUT}/h66_selected_{c}.txt"
        selected = open(sel_file).read().strip() if os.path.exists(sel_file) else None
        atk = attacks(c)
        fz = [a for a in atk if a["in_frozen_grid"]]
        val = [a for a in fz if a["valid"]]
        mv = max((a["H_after"] for a in val), default=None)
        ung = max((a["H_after"] for a in fz), default=None)
        srow = next((x for x in lad if x["arm"] == selected), None)
        retain = srow["retain8"] if srow else None

        band = None
        if not lad:
            band = "INCOMPLETE: no rho ladder measured yet"
        elif selected is None:
            band = (f"NO rho on the ladder reaches target <= {ERASE_THRESHOLD}. Per PREREG_H66 S5 "
                    f"this is the PORT failing to erase in this domain, NOT evidence against the "
                    f"published method, and no robustness number is claimed.")
        elif not val:
            band = (f"INCOMPLETE: {selected} selected but no gate-passing attack yet "
                    f"({len(fz)}/{len(FROZEN)} frozen points run)")
        elif len(fz) < len(FROZEN):
            band = (f"INCOMPLETE: {len(fz)}/{len(FROZEN)} frozen points run; max-over-valid is "
                    f"monotone in attacks run so a partial grid can only UNDERSTATE recovery")
        elif retain is not None and retain >= BAR_RETAIN and mv <= BAR_H40:
            band = ("★ THE MANDATE'S BAR, MET BY A PUBLISHED METHOD. retain >= 0.85 with "
                    "max-over-valid H <= 0.40. PREREG_H40 is refuted by someone else's work, and "
                    "FRP beats every arm we built. Report it as theirs.")
        elif retain is not None and retain >= BAR_RETAIN and mv <= BAR_H47:
            band = ("real frontier point: retain >= 0.85 with H <= 0.50 but > 0.40 -- clears H47's "
                    "bar, does NOT refute H40 (ERRATUM_r2 C5: the two thresholds are not "
                    "interchangeable)")
        elif mv is not None and mv >= 0.933:
            band = ("FRP is fully reversed like every other arm -> the relearning adversary defeats "
                    "a method DESIGNED for relearning robustness. A stronger version of the "
                    "phase's negative result, not a weaker one.")
        elif retain is not None and retain < 0.60:
            band = ("utility collapsed at the rho needed to erase -> inseparability reappears in "
                    "PRUNING as well as in noise, a second independent route to PREREG_H51 S7's "
                    "provisional mechanism claim")
        else:
            band = (f"intermediate: retain {retain}, max-valid H {mv} -- matches none of the "
                    f"declared bands cleanly; report as measured, do not reframe")

        res["per_concept"][c] = {
            "rho_ladder": lad,
            "paper_setting_rho3": next((x for x in lad if "_r3_l05" in x["file"]), None),
            "selected_arm": selected, "selected_retain8": retain,
            "frozen_grid": {"n_run": len(fz), "n_declared": len(FROZEN), "n_valid": len(val),
                            "max_over_valid_H": mv, "max_ungated_H": ung},
            "baseline_same_grid": BASELINE[c],
            "beats_baseline": (None if (mv is None or retain is None) else
                               bool(mv < BASELINE[c]["max_valid_H"]
                                    and retain >= BAR_RETAIN)),
            "declared_band": band, "attacks": atk}

    json.dump(res, open(f"{OUT}/h66_analysis.json", "w"), indent=1)

    print("=== H66: FRP port (a PUBLISHED method) under the frozen grid ===")
    for c, b in res["per_concept"].items():
        print(f"\n--- {c}")
        if b["rho_ladder"]:
            print(f"  {'arm':30s} {'target':>8s} {'utility':>8s} {'retain8':>8s}  erases<={ERASE_THRESHOLD}")
            for x in b["rho_ladder"]:
                print(f"  {x['arm']:30s} {x['target_suppression']:8.4f} "
                      f"{x['utility_all_nontarget']:8.4f} {x['retain8']:8.4f}  {x['erases']}")
        else:
            print("  (no ladder yet)")
        print(f"  selected: {b['selected_arm']}  retain8={b['selected_retain8']}")
        f = b["frozen_grid"]
        print(f"  frozen grid: {f['n_run']}/{f['n_declared']} run, {f['n_valid']} valid, "
              f"max-over-valid H={f['max_over_valid_H']}, ungated={f['max_ungated_H']}")
        print(f"  baseline (same grid): AdvUnlearn retain {b['baseline_same_grid']['retain8']}, "
              f"max-valid H {b['baseline_same_grid']['max_valid_H']}")
        print(f"  BAND: {b['declared_band']}")
    print(f"\nwrote {OUT}/h66_analysis.json")


if __name__ == "__main__":
    main()
