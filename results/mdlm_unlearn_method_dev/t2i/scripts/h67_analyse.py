"""H67 analysis: does FRAG predict our measured post-attack recovery?

Applies PREREG_H67 S4-S5 exactly as declared, to files on disk:

  outcome      max-over-valid H after attack, post-dating the critical evaluation fix
  Population A all eligible arms (UNet genuinely modified, >= 1 gate-passing attack)
  Population B "healthy" only, legacy_retain8 >= 0.60 -- the PRIMARY, mirroring the paper's own
               protocol of correlating over healthy checkpoints with collapsed/noise controls out
  comparators  global l2, and A_f / A_r separately
  rule         report rho with n and a bootstrap CI; MAKE NO TRANSFER CLAIM IN EITHER DIRECTION
               unless the CI excludes zero

Run: python scripts/h67_analyse.py --concept church
"""
import argparse, glob, json, os
import numpy as np

OUT = "/data/bijaypandey/t2i_unlearn/outputs"
N_BOOT = 10000
HEALTHY = 0.60                    # PREREG_H67 S4, matching PREREG_H51 K2's floor
POST_FIX_PREFIXES = ("H59", "H60", "H61", "H62", "H63", "H65", "H64v2", "H66", "h52", "h53", "h54")


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        # average ties
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def boot_ci(x, y, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = len(x)
    if k < 4:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        if len(np.unique(x[i])) < 2 or len(np.unique(y[i])) < 2:
            continue
        vals.append(spearman(x[i], y[i]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def arm_recovery():
    """max-over-VALID H per arm, from post-fix attack records keyed by the arm's unet path."""
    best = {}
    for f in glob.glob(f"{OUT}/atk_*.json"):
        b = os.path.basename(f)
        if not any(b[4:].startswith(p) for p in POST_FIX_PREFIXES):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        h = r.get("hparams") or {}
        u = h.get("unet")
        if not u or not r.get("attack_validity", {}).get("valid"):
            continue
        arm = os.path.basename(os.path.dirname(u))
        hv = r.get("recovery", {}).get("H_after")
        if hv is None:
            continue
        cur = best.get(arm)
        best[arm] = {"max_valid_H": max(hv, cur["max_valid_H"]) if cur else hv,
                     "n_valid": (cur["n_valid"] + 1) if cur else 1}
    return best


def arm_utility():
    out = {}
    for f in glob.glob(f"{OUT}/gen_*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        lab = r.get("label")
        rt = r.get("retain_mean_acc_LEGACY")
        if lab and rt is not None:
            out[lab] = rt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    a = ap.parse_args()

    fr = json.load(open(f"{OUT}/frag_scores_{a.concept}.json"))
    scored = {r["arm"]: r for r in fr["arms"] if "FRAG" in r}
    excluded_identical = [r["arm"] for r in fr["arms"] if r.get("unet_identical_to_base")]
    rec_map, util = arm_recovery(), arm_utility()

    rows = []
    for arm, s in sorted(scored.items()):
        if arm not in rec_map:
            continue
        rows.append({"arm": arm, "FRAG": s["FRAG"], "A_f": s["A_f"], "A_r": s["A_r"],
                     "global_l2": s["global_l2"],
                     "max_valid_H": rec_map[arm]["max_valid_H"],
                     "n_valid_attacks": rec_map[arm]["n_valid"],
                     "retain8": util.get(arm)})
    A = rows
    B = [r for r in rows if r["retain8"] is not None and r["retain8"] >= HEALTHY]

    def block(pop, name):
        if len(pop) < 4:
            return {"population": name, "n": len(pop),
                    "verdict": "n < 4: no correlation computed"}
        y = [r["max_valid_H"] for r in pop]
        o = {}
        for pred in ("FRAG", "A_f", "A_r", "global_l2"):
            x = [r[pred] for r in pop]
            rho = spearman(x, y)
            lo, hi = boot_ci(x, y)
            excl = (not np.isnan(lo)) and (lo * hi > 0)
            o[pred] = {"spearman": rho, "ci95": [lo, hi], "ci_excludes_zero": bool(excl)}
        o.update({"population": name, "n": len(pop),
                  "arms": [r["arm"] for r in pop]})
        f = o["FRAG"]
        o["verdict"] = (
            ("TRANSFERS: FRAG's CI excludes zero and is negative"
             if f["spearman"] < 0 else
             "DOES NOT TRANSFER: FRAG's CI excludes zero but the sign is POSITIVE, the opposite "
             "of the published direction")
            if f["ci_excludes_zero"] else
            f"INCONCLUSIVE at n={len(pop)}: FRAG's bootstrap CI "
            f"[{f['ci95'][0]:.2f}, {f['ci95'][1]:.2f}] includes zero. Per PREREG_H67 S5 no "
            f"transfer claim is made in either direction. A point estimate near the paper's "
            f"-0.78 would be CONSISTENT WITH transfer, not evidence of it.")
        return o

    res = {"study": "h67_frag_transfer", "concept": a.concept,
           "prereg": "docs/PREREG_H67_frag_predictor_transfer.md",
           "outcome": "max-over-valid H after attack (post-fix records only)",
           "published_reference": {"FRAG_pooled_healthy": -0.78, "global_l2_pooled_healthy": -0.36,
                                   "note": "their numbers are vs dES on LLM text benchmarks; ours "
                                           "are vs max-valid H on SD v1.4. Not the same outcome "
                                           "variable, so magnitudes are not directly comparable."},
           "excluded_unet_identical": excluded_identical,
           "n_excluded_unet_identical": len(excluded_identical),
           "exclusion_note": "arms whose UNet is bit-identical to base (text-encoder erasures and "
                             "exact alpha=0 no-ops) have D == 0, so FRAG is UNDEFINED. This "
                             "removes the best-utility arms in the phase -- a real scope "
                             "limitation of the predictor, reported as a finding.",
           "rows": rows,
           "population_A_all": block(A, "A: all eligible"),
           "population_B_healthy_PRIMARY": block(B, f"B: healthy (retain8 >= {HEALTHY})")}
    json.dump(res, open(f"{OUT}/h67_frag_transfer_{a.concept}.json", "w"), indent=1)

    print(f"=== H67: FRAG transfer test, {a.concept} ===")
    print(f"excluded (UNet bit-identical, FRAG undefined): {len(excluded_identical)} arms")
    print(f"\n{'arm':34s} {'FRAG':>8s} {'A_f':>7s} {'A_r':>7s} {'l2':>10s} {'maxH':>6s} {'ret8':>6s} {'n':>2s}")
    for r in rows:
        print(f"{r['arm']:34s} {r['FRAG']:+8.4f} {r['A_f']:7.4f} {r['A_r']:7.4f} "
              f"{r['global_l2']:10.3e} {r['max_valid_H']:6.3f} "
              f"{(r['retain8'] if r['retain8'] is not None else float('nan')):6.3f} "
              f"{r['n_valid_attacks']:2d}")
    for key in ("population_A_all", "population_B_healthy_PRIMARY"):
        b = res[key]
        print(f"\n--- {b['population']} (n={b['n']}) ---")
        if b["n"] >= 4:
            for pred in ("FRAG", "A_f", "A_r", "global_l2"):
                v = b[pred]
                print(f"  {pred:10s} rho={v['spearman']:+.3f}  CI95=[{v['ci95'][0]:+.3f}, "
                      f"{v['ci95'][1]:+.3f}]  excludes0={v['ci_excludes_zero']}")
        print(f"  VERDICT: {b['verdict']}")
    print(f"\nwrote {OUT}/h67_frag_transfer_{a.concept}.json")


if __name__ == "__main__":
    main()
