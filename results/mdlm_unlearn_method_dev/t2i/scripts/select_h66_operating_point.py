"""Apply PREREG_H66 S4's operating-point rule to the written-out stage-1 numbers.

The rule, fixed before any FRP number existed:

    the smallest rho whose TARGET SUPPRESSION reaches <= 0.400 -- the erasure depth of the official
    AdvUnlearn church encoder -- with UTILITY NOT CONSULTED in the selection.

This lives in its own script, applied to JSONs on disk, so the rule cannot be quietly adjusted
while looking at the arms. If no rho on the declared ladder reaches 0.400, it writes no selection
and says so: that outcome is "the port does not erase", which PREREG_H66 S5 pre-declares as a port
limitation and NOT as evidence against the paper.

Run: python scripts/select_h66_operating_point.py --concept church
"""
import argparse, glob, json, os, re

OUT = "/data/bijaypandey/t2i_unlearn/outputs"
THRESHOLD = 0.400          # AdvUnlearn church erasure depth; declared in PREREG_H66 S4
LADDER = [1, 3, 5, 10, 20]  # rho percent, declared in PREREG_H66 S4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{OUT}/gen_frp_{a.concept}_r*_l05.json")):
        r = json.load(open(f))
        m = r.get("active_target_metrics") or {}
        mm = re.search(r"_r(\d+)_l05", os.path.basename(f))
        if not mm or not m:
            continue
        rows.append({"rho_pct": int(mm.group(1)), "arm": r["label"],
                     "target_suppression": m["target_suppression_acc"],
                     "utility_all_nontarget": m["utility_all_nontarget"],
                     "legacy_retain8": m["legacy_retain8"]})
    rows.sort(key=lambda x: x["rho_pct"])

    print(f"=== H66 stage-1 ladder, {a.concept} (lambda = 0.05 primary) ===")
    print(f"{'rho%':>5s} {'target':>8s} {'utility':>8s} {'retain8':>8s}   rule: target <= {THRESHOLD}")
    for x in rows:
        ok = "<-- qualifies" if x["target_suppression"] <= THRESHOLD else ""
        print(f"{x['rho_pct']:5d} {x['target_suppression']:8.4f} "
              f"{x['utility_all_nontarget']:8.4f} {x['legacy_retain8']:8.4f}   {ok}")

    missing = [p for p in LADDER if p not in [x["rho_pct"] for x in rows]]
    qual = [x for x in rows if x["target_suppression"] <= THRESHOLD]
    sel = qual[0] if qual else None

    rec = {"study": "h66_operating_point", "concept": a.concept,
           "rule": f"smallest rho with target_suppression <= {THRESHOLD}; utility NOT consulted",
           "rule_source": "PREREG_H66 S4, declared before any FRP number existed",
           "ladder_declared": LADDER, "ladder_missing": missing,
           "rows": rows, "selected": sel,
           "paper_setting_row": next((x for x in rows if x["rho_pct"] == 3), None)}
    if sel is None:
        rec["verdict"] = ("NO rho on the declared ladder reaches the erasure threshold. Per "
                          "PREREG_H66 S5 this is reported as the PORT failing to erase in this "
                          "domain, not as evidence against the published method, and no "
                          "robustness number is claimed.")
        print(f"\nNO SELECTION: {rec['verdict']}")
    else:
        rec["verdict"] = (f"selected {sel['arm']} at rho={sel['rho_pct']}% "
                          f"(target {sel['target_suppression']:.4f} <= {THRESHOLD})")
        with open(f"{OUT}/h66_selected_{a.concept}.txt", "w") as fh:
            fh.write(sel["arm"])
        print(f"\nSELECTED: {rec['verdict']}")
        print(f"  -> wrote {OUT}/h66_selected_{a.concept}.txt")
    if missing:
        print(f"  WARNING: ladder incomplete, missing rho% {missing} -- selection is provisional")
    json.dump(rec, open(f"{OUT}/h66_operating_point_{a.concept}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
