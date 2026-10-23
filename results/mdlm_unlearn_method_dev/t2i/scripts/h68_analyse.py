"""H68: apply FINDING39 S4's pre-declared decision rule, and COMPUTE what the H64/H68 comparison
is and is not matched on rather than asserting it.

FINDING39 S4, declared before H68 ran:
  retain near 0.85  -> the v2 pools are innocent and the `frag` ALLOCATION is the cause; H64
                       cleanly false
  retain collapses  -> the POOL REPAIR is the cause, and every published damage+distil retain
                       number in this phase is an in-sample artifact, extending FINDING38 S3's
                       withdrawal from the cross-family ranking to the within-family alpha-ladder

The matching audit matters because a conclusion about ALLOCATION requires the two arms to differ
only in allocation. They are matched on alpha, beta, teacher, steps, pool recipe and realised mask
MEAN -- but NOT necessarily on realised perturbation NORM, which is what actually perturbs the
weights. That is computed here and reported with the verdict, not left implicit.

Run: python scripts/h68_analyse.py
"""
import json, os

OUT = "/data/bijaypandey/t2i_unlearn/outputs"
NEAR_INTACT = 0.80      # "lands near 0.85" -- read generously toward the costlier branch
COLLAPSE = 0.30
V1_REFERENCE = {"h47_adv_a0.3": 0.8667, "h51_ff_a0.3": 0.0917}   # v1 pools, partly IN-SAMPLE


def util(label):
    p = f"{OUT}/gen_{label}.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    m = r.get("active_target_metrics") or {}
    return {"target": m.get("target_suppression_acc"),
            "retain8": m.get("legacy_retain8", r.get("retain_mean_acc_LEGACY")),
            "utility_all_nontarget": m.get("utility_all_nontarget")}


def mask(label):
    p = f"{OUT}/maskaudit_{label}.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {"protect": r["protect"], "alpha": r["alpha"], "beta": r["beta"],
            "mask_mean_post_clip": r["mask_mean_post_clip"],
            "clipped_fraction": r["clipped_fraction"],
            "realised_perturbation_norm": r["realised_perturbation_norm"],
            "weight_norm": r.get("weight_norm")}


def main():
    res = {"study": "h68_analysis", "rule_source": "FINDING39 S4 (declared before H68 ran)",
           "per_concept": {}}
    for c in ("church", "parachute"):
        h64, h68 = f"h64v2_frag_a0.3_{c}", f"h68v2_rf_a0.3_{c}"
        u64, u68 = util(h64), util(h68)
        m64, m68 = mask(h64), mask(h68)

        matched, unmatched = [], []
        if m64 and m68:
            for k in ("alpha", "beta"):
                (matched if m64[k] == m68[k] else unmatched).append(f"{k}: {m64[k]} vs {m68[k]}")
            mm = abs(m64["mask_mean_post_clip"] - m68["mask_mean_post_clip"])
            (matched if mm < 1e-6 else unmatched).append(
                f"realised mask MEAN: {m64['mask_mean_post_clip']:.4f} vs "
                f"{m68['mask_mean_post_clip']:.4f}")
            r64, r68 = m64["realised_perturbation_norm"], m68["realised_perturbation_norm"]
            ratio = r64 / r68 if r68 else None
            unmatched.append(f"realised perturbation NORM: {r64:.3f} vs {r68:.3f} "
                             f"(ratio {ratio:.3f}x)" if ratio else "norm unavailable")
        matched += ["teacher (AdvUnlearn TE)", "distil steps (6000)", "lr (1e-5)",
                    "pool recipe (v2)", "seed (0)"]

        verdict = None
        if u68 is None or u68["retain8"] is None:
            verdict = "INCOMPLETE: H68 utility not measured yet. No verdict."
        elif u68["retain8"] >= NEAR_INTACT:
            verdict = (f"BRANCH 1 — the v2 pools are INNOCENT. retain_fisher on v2 pools holds "
                       f"retain8 at {u68['retain8']:.4f}, so the pool repair did not cause "
                       f"H64's collapse. The `frag` ALLOCATION is the cause and FINDING39's "
                       f"pre-declared reading stands. The phase's published damage+distil retain "
                       f"numbers are NOT shown to be in-sample artifacts by this control.")
        elif u68["retain8"] <= COLLAPSE:
            verdict = (f"BRANCH 2 — the POOL REPAIR is the cause. retain_fisher collapses to "
                       f"{u68['retain8']:.4f} on v2 pools, against {V1_REFERENCE['h47_adv_a0.3']} "
                       f"on v1. The v1 number was held up by in-sample evaluation prompts, so "
                       f"EVERY published damage+distil retain figure in this phase is an "
                       f"in-sample artifact, and FINDING38 S3's withdrawal must extend from the "
                       f"cross-family ranking to the within-family alpha-ladder. This is the "
                       f"costlier branch and it is reported as such.")
        else:
            verdict = (f"BRANCH 3 (neither declared branch) — retain8 {u68['retain8']:.4f} sits "
                       f"between {COLLAPSE} and {NEAR_INTACT}. Both the allocation and the pool "
                       f"repair plausibly contribute and this control does not separate them. "
                       f"Reported as not separating them; no attribution claimed.")

        res["per_concept"][c] = {
            "h64_frag": u64, "h68_retain_fisher": u68,
            "h64_mask": m64, "h68_mask": m68,
            "matched_on": matched, "NOT_matched_on": unmatched,
            "v1_reference_partly_in_sample": V1_REFERENCE,
            "verdict": verdict}

    json.dump(res, open(f"{OUT}/h68_analysis.json", "w"), indent=1)
    print("=== H68: does the frag ALLOCATION or the POOL REPAIR explain H64's collapse? ===")
    for c, b in res["per_concept"].items():
        print(f"\n--- {c}")
        for nm, d in (("H64 frag        ", b["h64_frag"]), ("H68 retain_fisher", b["h68_retain_fisher"])):
            if d:
                print(f"  {nm}: target={d['target']}  retain8={d['retain8']}  "
                      f"utility={d['utility_all_nontarget']}")
            else:
                print(f"  {nm}: not measured")
        print(f"  v1 reference (partly IN-SAMPLE): retain_fisher 0.8667, forget_fisher 0.0917")
        print(f"  matched on      : {', '.join(b['matched_on'])}")
        print(f"  NOT matched on  : {'; '.join(b['NOT_matched_on'])}")
        print(f"  VERDICT: {b['verdict']}")
    print(f"\nwrote {OUT}/h68_analysis.json")


if __name__ == "__main__":
    main()
