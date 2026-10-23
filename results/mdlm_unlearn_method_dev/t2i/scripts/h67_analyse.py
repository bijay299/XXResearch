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
# CORRECTED before any correlation was read. The first version filtered attack records by a LABEL
# PREFIX heuristic for "post-fix", which wrongly discarded the 65 pre-fix UNet-ONLY runs that the
# critical text-encoder evaluation bug never touched. run_inventory.json already carries the
# authoritative per-run classification built in `ecfe8ad`, so it is used directly instead:
#   VOID_te_omitted_post_attack (136)  -> excluded
#   valid_unaffected_unet_only  (65)   -> ELIGIBLE (the bug could not affect a UNet-only run)
#   valid_post_te_fix           (33)   -> ELIGIBLE
# The n=1 first run produced no correlation, so nothing was seen before this correction.
INVENTORY = "run_inventory.json"
VALID_STATUSES = ("valid_unaffected_unet_only", "valid_post_te_fix")
# Identity-unresolved arms are excluded on PROVENANCE grounds, not on their numbers:
# adapters/uce_church/unet.pt was OVERWRITTEN between a retain-0.350 and a retain-0.850 build
# (FINDING35 S5), so a FRAG score computed on today's file cannot be paired with recovery numbers
# that came partly from a different checkpoint.
IDENTITY_UNRESOLVED = {"uce_church"}


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
    """max-over-VALID H per arm, from the inventory's own eligibility classification, plus any
    attack records written after the inventory was built (H64v2 onward)."""
    best = {}

    def add(arm, hv):
        cur = best.get(arm)
        best[arm] = {"max_valid_H": max(hv, cur["max_valid_H"]) if cur else hv,
                     "n_valid": (cur["n_valid"] + 1) if cur else 1}

    seen = set()
    for r in json.load(open(f"{OUT}/{INVENTORY}")):
        seen.add(r["file"])
        if r["post_attack_eval_status"] not in VALID_STATUSES:
            continue
        if not r["unet_path"] or not r["gate_valid"] or r["H_after"] is None:
            continue
        add(os.path.basename(os.path.dirname(r["unet_path"])), r["H_after"])
    # records newer than the inventory: eligible by construction (written post-fix)
    for f in glob.glob(f"{OUT}/atk_*.json"):
        if os.path.basename(f) in seen:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        u = (r.get("hparams") or {}).get("unet")
        if not u or not r.get("attack_validity", {}).get("valid"):
            continue
        hv = r.get("recovery", {}).get("H_after")
        if hv is not None:
            add(os.path.basename(os.path.dirname(u)), hv)
    return best


def arm_utility():
    out = {}
    for f in glob.glob(f"{OUT}/gen_*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        lab = r.get("label")
        # the issue-4 repair RENAMED this field; files written before it use the old name, so both
        # are accepted. Same quantity (the historical retain8 denominator) under both names.
        rt = r.get("retain_mean_acc_LEGACY")
        if rt is None:
            rt = r.get("retain_mean_acc")
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

    rows, dropped = [], {"no_valid_attack": [], "identity_unresolved": []}
    for arm, s in sorted(scored.items()):
        if arm in IDENTITY_UNRESOLVED:
            dropped["identity_unresolved"].append(arm)
            continue
        if arm not in rec_map:
            dropped["no_valid_attack"].append(arm)
            continue
        rows.append({"arm": arm, "FRAG": s["FRAG"], "A_f": s["A_f"], "A_r": s["A_r"],
                     "global_l2": s["global_l2"],
                     "max_valid_H": rec_map[arm]["max_valid_H"],
                     "n_valid_attacks": rec_map[arm]["n_valid"],
                     "retain8": util.get(arm)})
    # the binding constraint: arms with a valid recovery number but NO surviving checkpoint
    no_ckpt = sorted(a for a in rec_map if a not in scored)
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
           "dropped": dropped,
           "arms_with_recovery_but_no_surviving_checkpoint": no_ckpt,
           "n_arms_with_recovery_but_no_checkpoint": len(no_ckpt),
           "binding_constraint": (
               "FRAG needs the arm's WEIGHTS; a recovery number alone is not enough. "
               f"{len(no_ckpt)} arms have a valid max-over-valid H but their erased checkpoint was "
               "deleted for storage, so they cannot be scored. This is the provenance failure the "
               "audit's issue 9 warned about -- 'do not delete the only checkpoint needed to "
               "verify a headline result' -- and it is now the reason this study cannot be done at "
               "the pre-declared standard. Rebuilding is NOT a remedy: construction was "
               "nondeterministic until 08742c1, and a rebuild is a NEW arm whose weights must not "
               "be paired with the old arm's measured recovery."),
           "exclusion_note": "arms whose UNet is bit-identical to base (text-encoder erasures and "
                             "exact alpha=0 no-ops) have D == 0, so FRAG is UNDEFINED. This "
                             "removes the best-utility arms in the phase -- a real scope "
                             "limitation of the predictor, reported as a finding.",
           "rows": rows,
           "population_A_all": block(A, "A: all eligible"),
           "population_B_healthy_PRIMARY": block(B, f"B: healthy (retain8 >= {HEALTHY})")}
    json.dump(res, open(f"{OUT}/h67_frag_transfer_{a.concept}.json", "w"), indent=1)

    print(f"=== H67: FRAG transfer test, {a.concept} ===")
    print(f"scored arms (surviving checkpoint)      : {len(scored)}")
    print(f"excluded, UNet bit-identical (undefined): {len(excluded_identical)}")
    print(f"excluded, identity unresolved           : {dropped['identity_unresolved']}")
    print(f"scored but no valid attack              : {dropped['no_valid_attack']}")
    print(f"HAVE recovery but checkpoint DELETED    : {len(no_ckpt)} arms  <-- binding constraint")
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
