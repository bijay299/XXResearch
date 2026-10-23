"""H64 analysis: apply PREREG_H64's declared kill rules to the files on disk.

Written BEFORE the attack numbers exist so the rules are executed, not recalled:

  K1  (S4a) report max-over-valid on the FROZEN 6 points -- the only figures comparable to the
      H65 baselines -- AND on all 9 including the K1 extension. The 9-point figure is the
      stricter test, since max-over-valid is monotone in attacks run.
  K2  retain8 < 0.60 => NOT A CANDIDATE regardless of any robustness number.
  K3  a positive requires BOTH concepts.
  S3  the declared outcome bands: (retain >= 0.85 and H <= 0.50) = H64 true; (retain >= 0.85,
      H > 0.80) = preserves utility, buys nothing; (retain < 0.60) = inherits forget_fisher's
      collapse; (H <= 0.50 but retain < 0.60) = same trade as every previous arm.

Also reports post-attack utility (issue 7) and the gated/ungated pair (issue 6), both of which are
now required on every arm.

Run: python scripts/h64_analyse.py
"""
import glob, json, os

OUT = "/data/bijaypandey/t2i_unlearn/outputs"
FROZEN = {("0", "1e-5"), ("1", "1e-5"), ("6", "1e-5"),
          ("0", "5e-5"), ("1", "5e-5"), ("0", "1e-4")}
K2_FLOOR = 0.60
BAR_RETAIN, BAR_H = 0.85, 0.50


def load(concept):
    arm = f"h64v2_frag_a0.3_{concept}"
    g = f"{OUT}/gen_{arm}.json"
    util = None
    if os.path.exists(g):
        r = json.load(open(g))
        m = r.get("active_target_metrics") or {}
        util = {"target_suppression": m.get("target_suppression_acc"),
                "utility_all_nontarget": m.get("utility_all_nontarget"),
                "retain8": m.get("legacy_retain8") if m else r.get("retain_mean_acc_LEGACY")}
    atks = []
    for f in sorted(glob.glob(f"{OUT}/atk_H64v2_*_{concept}_sd*_lr*.json")):
        d = json.load(open(f))
        h = d["hparams"]
        b = os.path.basename(f)
        tag = "frozen" if "_frozen_" in b else "k1ext"
        sd, lr = str(h["seed"]), f"{h['lr']:.0e}".replace("e-05", "e-5").replace("e-04", "e-4")
        pau = d.get("post_attack_utility", {})
        atks.append({"file": b, "tag": tag, "seed": sd, "lr": lr,
                     "in_frozen_grid": (sd, lr) in FROZEN,
                     "valid": bool(d["attack_validity"]["valid"]),
                     "T_rise": d["attack_validity"]["T_rise"],
                     "H_after": d["recovery"]["H_after"],
                     "H_before": d["recovery"]["H_before"],
                     "pau_retained_after": (pau.get("retained_capability") or {}).get("after"),
                     "pau_adherence_delta": (pau.get("prompt_adherence") or {}).get("delta")})
    return arm, util, atks


def mx(rows, only_valid=True):
    s = [r for r in rows if r["valid"]] if only_valid else rows
    return (max(r["H_after"] for r in s), len(s)) if s else (None, 0)


def main():
    res = {"study": "h64_analysis", "prereg": "docs/PREREG_H64_frag_allocation.md",
           "recipe": "pools_v2", "per_concept": {}}
    for c in ("church", "parachute"):
        arm, util, atks = load(c)
        fz = [a for a in atks if a["in_frozen_grid"]]
        mv_fz, n_fz = mx(fz)
        mv_all, n_all = mx(atks)
        ung_fz, _ = mx(fz, only_valid=False)
        ung_all, _ = mx(atks, only_valid=False)
        retain = (util or {}).get("retain8")
        band = None
        if retain is not None and mv_all is not None:
            if retain < K2_FLOOR:
                band = ("retain < 0.60 -> inherits forget_fisher's collapse; NOT A CANDIDATE per "
                        "K2 regardless of the robustness number")
            elif retain >= BAR_RETAIN and mv_all <= BAR_H:
                band = ("retain >= 0.85 and max-valid H <= 0.50 -> H64 TRUE on this concept. K3 "
                        "requires the other concept before any claim.")
            elif retain >= BAR_RETAIN and mv_all > 0.80:
                band = "retain >= 0.85 but H > 0.80 -> preserves utility, buys no robustness"
            else:
                band = (f"intermediate: retain {retain:.4f}, max-valid H {mv_all:.4f} -- neither "
                        f"the declared positive nor a clean collapse")
        res["per_concept"][c] = {
            "arm": arm, "utility": util,
            "k2_floor": K2_FLOOR,
            "k2_pass": (retain is not None and retain >= K2_FLOOR),
            "frozen_grid": {"n_attacks": len(fz), "n_valid": n_fz,
                            "max_over_valid_H": mv_fz, "max_ungated_H": ung_fz},
            "all_nine": {"n_attacks": len(atks), "n_valid": n_all,
                         "max_over_valid_H": mv_all, "max_ungated_H": ung_all},
            "declared_band": band, "attacks": atks}

    both = [res["per_concept"][c] for c in ("church", "parachute")]
    # INCOMPLETE must never read as a measured negative. A missing utility number or an empty
    # attack set is "not measured yet", not "failed its bar".
    incomplete = [c for c, b in res["per_concept"].items()
                  if (b["utility"] or {}).get("retain8") is None
                  or b["all_nine"]["n_attacks"] < len(FROZEN)]
    k3 = (not incomplete) and all(b["k2_pass"] for b in both) and all(
        (b["all_nine"]["max_over_valid_H"] is not None
         and b["all_nine"]["max_over_valid_H"] <= BAR_H) for b in both)
    res["verdict"] = {
        "status": "INCOMPLETE" if incomplete else "COMPLETE",
        "incomplete_concepts": incomplete,
        "k3_both_concepts_positive": None if incomplete else bool(k3),
        "statement": (
            f"INCOMPLETE -- {incomplete} still missing a utility number or the full frozen grid. "
            f"NO VERDICT. This is not a negative result; it is an unfinished measurement."
            if incomplete else
            ("H64 POSITIVE on both concepts under its own declared bar -- report as an "
             "extension of a PUBLISHED principle (2608.25429) to noise allocation and to "
             "diffusion, never as a new mechanism."
             if k3 else
             "H64 does NOT meet its declared bar on both concepts. Per PREREG_H64 S3 this "
             "is reported as the declared outcome it matches, not reframed.")),
        "comparability_warning": ("built on pools recipe v2; NOT comparable to the v1 retain_fisher "
                                  "or forget_fisher arms that motivated it (FINDING38 S2-3). A "
                                  "matched v2 rebuild would be needed for that comparison and has "
                                  "not been run.")}
    json.dump(res, open(f"{OUT}/h64_analysis.json", "w"), indent=1)

    print("=== H64 (frag allocation, pools v2) ===")
    for c, b in res["per_concept"].items():
        u = b["utility"] or {}
        print(f"\n--- {c}: {b['arm']}")
        print(f"  target suppression : {u.get('target_suppression')}")
        print(f"  retain8            : {u.get('retain8')}   K2 pass (>= {K2_FLOOR}): {b['k2_pass']}")
        print(f"  utility_all_nontgt : {u.get('utility_all_nontarget')}")
        fz, al = b["frozen_grid"], b["all_nine"]
        print(f"  FROZEN 6  : {fz['n_valid']}/{fz['n_attacks']} valid, "
              f"max-over-valid H={fz['max_over_valid_H']}, ungated={fz['max_ungated_H']}")
        print(f"  ALL 9     : {al['n_valid']}/{al['n_attacks']} valid, "
              f"max-over-valid H={al['max_over_valid_H']}, ungated={al['max_ungated_H']}")
        print(f"  BAND: {b['declared_band']}")
        for a in b["attacks"]:
            print(f"    {a['tag']:6s} sd{a['seed']} lr{a['lr']:5s} valid={str(a['valid']):5s} "
                  f"Trise={a['T_rise']:+.3f} H {a['H_before']:.3f}->{a['H_after']:.3f} "
                  f"pau_after={a['pau_retained_after']}")
    print(f"\nVERDICT: {res['verdict']['statement']}")
    print(f"NOTE: {res['verdict']['comparability_warning']}")
    print(f"\nwrote {OUT}/h64_analysis.json")


if __name__ == "__main__":
    main()
