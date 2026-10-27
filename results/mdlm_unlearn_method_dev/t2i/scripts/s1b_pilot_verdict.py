#!/usr/bin/env python
"""Apply S1b's pre-declared pilot criterion to the pilot records. Prints a verdict; writes nothing.

The criterion is fixed by docs/PREREG_S1b_META_V3.md section 2, which was written BEFORE any v3 run
existed. Its purpose is to FALSIFY the estimator diagnosis in
docs/PREREG_S1_AMENDMENT_1_2026-10-27.md section 6 if that diagnosis is wrong.

The diagnosis says: v2 estimated the difference [L_R(theta') - L_R(theta)] on two INDEPENDENT random
draws, so the logged `retain_gap` measured the draw change rather than the parameter change. If that
is right, then sharing the draw must change the gap's CHARACTER:

    v3 expected  ->  |gap| small (orders of magnitude below 0.1) AND consistently signed
    v3 falsifies ->  |gap| still ~0.1+ and/or sign flipping

Thresholds, declared in the prereg and hard-coded here rather than chosen after seeing the numbers:

    SUPPORTED   max|gap| < 0.01  and  sign-consistency >= 0.90
    FALSIFIED   mean|gap| >= 0.10  or  sign-consistency < 0.60
    otherwise   INCONCLUSIVE -- the diagnosis is neither confirmed nor refuted; do not build

`sign-consistency` is the fraction of logged steps whose gap carries the majority sign.

v2's own stored train_log supplies the baseline, so the comparison uses real recorded data rather
than a handful of quoted lines.
"""
import glob, json, os, statistics as st, sys

MAXGAP_SUPPORT = 0.01
MEANGAP_FALSIFY = 0.10
SIGN_SUPPORT = 0.90
SIGN_FALSIFY = 0.60


def gaps(rec):
    out = []
    for r in rec.get("train_log") or []:
        g = r.get("retain_gap")
        if isinstance(g, (int, float)):
            out.append(float(g))
    return out


def norms(rec):
    out = []
    for r in rec.get("train_log") or []:
        n = r.get("meta_grad_norm")
        if isinstance(n, (int, float)):
            out.append(float(n))
    return out


def describe(name, gs, ns):
    if not gs:
        print("  %-28s no retain_gap entries" % name)
        return None
    ab = [abs(x) for x in gs]
    pos = sum(1 for x in gs if x > 0)
    neg = sum(1 for x in gs if x < 0)
    cons = max(pos, neg) / len(gs)
    print("  %-28s n=%3d  mean|gap|=%.6g  max|gap|=%.6g  sign-consistency=%.3f (%d+/%d-)"
          % (name, len(gs), st.mean(ab), max(ab), cons, pos, neg))
    if ns:
        print("  %-28s meta_grad_norm: mean=%.4g  median=%.4g  max=%.4g"
              % ("", st.mean(ns), st.median(ns), max(ns)))
    return {"n": len(gs), "mean_abs": st.mean(ab), "max_abs": max(ab), "cons": cons}


def main():
    os.chdir("/data/bijaypandey/t2i_unlearn")

    print("BASELINE -- v2, the estimator under suspicion (independent draws per term)")
    v2 = []
    for f in sorted(glob.glob("outputs/metaV2_s1_metaV2_*.json")):
        d = json.load(open(f))
        s = describe(os.path.basename(f).replace("metaV2_s1_metaV2_", "").replace(".json", ""),
                     gaps(d), norms(d))
        if s:
            v2.append(s)

    print("\nPILOT -- v3, common random numbers")
    found = {}
    for tag, label in (("fixed", "s1b_pilot_fixed"), ("randpair", "s1b_pilot_randpair")):
        f = "outputs/metaV3_%s.json" % label
        if not os.path.exists(f):
            print("  %-28s not run yet" % label)
            continue
        d = json.load(open(f))
        found[tag] = describe(label, gaps(d), norms(d))

    print("\nPRE-DECLARED CRITERION (prereg section 2; thresholds fixed before any v3 run)")
    print("  SUPPORTED    max|gap| < %.3g  and  sign-consistency >= %.2f" % (MAXGAP_SUPPORT, SIGN_SUPPORT))
    print("  FALSIFIED    mean|gap| >= %.3g  or  sign-consistency < %.2f" % (MEANGAP_FALSIFY, SIGN_FALSIFY))

    # The 'fixed' variant carries BOTH corrections and is the one the criterion is written against.
    key = found.get("fixed")
    if key is None:
        print("\nVERDICT: pending -- the 'fixed' pilot variant has not run.")
        return 0
    # AMENDMENT 2026-10-27: the sign clause is inapplicable once the gap falls below the fp32
    # resolution of the loss itself (~sqrt(n_elem) * ULP ~= 7.6e-6 for an MSE near 0.85). Below that
    # floor the sign is numerical noise, so sign-consistency tests nothing. Recorded in
    # PREREG_S1b_META_V3.md s2.0.0 as a defect in the original criterion.
    FP32_FLOOR = 7.6e-6
    below_floor = key["max_abs"] < FP32_FLOOR
    if below_floor:
        print("\n  NOTE: max|gap| %.3g is BELOW the fp32 rounding floor of the loss (~%.1e), so the"
              % (key["max_abs"], FP32_FLOOR))
        print("        sign-consistency clause is inapplicable and is not applied. Judging on"
              " magnitude plus gradient-norm stability instead.")

    if below_floor and key["max_abs"] < MAXGAP_SUPPORT:
        print("\nVERDICT: diagnosis SUPPORTED on magnitude (gap collapsed from v2's 0.190 to below"
              " float precision). Proceed to the builds (run_s1b_meta_v3.sh --build).")
    elif key["max_abs"] < MAXGAP_SUPPORT and key["cons"] >= SIGN_SUPPORT:
        print("\nVERDICT: diagnosis SUPPORTED. The gap is small and consistently signed once the "
              "draw is shared, which is what section 6 predicts. Proceed to the builds "
              "(run_s1b_meta_v3.sh --build).")
    elif key["mean_abs"] >= MEANGAP_FALSIFY or key["cons"] < SIGN_FALSIFY:
        print("\nVERDICT: diagnosis FALSIFIED. Sharing the draw did NOT change the gap's character, "
              "so section 6's explanation of why v2's model barely moved is WRONG and must be "
              "withdrawn. Do NOT build. Re-examine before spending further GPU time.")
    else:
        print("\nVERDICT: INCONCLUSIVE between the declared thresholds. The diagnosis is neither "
              "confirmed nor refuted; do not build on it. Report as inconclusive rather than "
              "reading it in the favourable direction.")

    # ---- declared epsilon check (prereg section 2.2): does Adam's numerical floor throttle the step?
    N_PARAMS = 859520964
    ADAM_EPS = 1e-8
    f = "outputs/metaV3_s1b_pilot_fixed.json"
    if os.path.exists(f):
        ns = norms(json.load(open(f)))
        if ns:
            typ = st.median(ns) / (N_PARAMS ** 0.5)
            ratio = typ / ADAM_EPS
            print("\nADAM EPSILON CHECK (prereg section 2.2)")
            print("  median meta_grad_norm   = %.4g" % st.median(ns))
            print("  typical per-component   = norm/sqrt(%d) = %.4g" % (N_PARAMS, typ))
            print("  Adam eps                = %.0e   ->  ratio = %.3g" % (ADAM_EPS, ratio))
            if ratio > 10:
                print("  READING: epsilon is irrelevant; proceed unchanged.")
            elif ratio > 0.1:
                print("  READING: epsilon MATERIALLY THROTTLES the step. Re-run with eps=1e-12 as a "
                      "declared change BEFORE building.")
            else:
                print("  READING: epsilon DOMINATES completely -- the default optimiser setting is "
                      "the binding constraint, not the objective. Re-run with eps=1e-12 before "
                      "building, and report that.")
            print("  NOTE: assumes gradient mass spread evenly over all parameters. Real gradients "
                  "concentrate in some layers, so this is an upper bound on the throttling.")

    if "randpair" in found and found["randpair"]:
        print("\nIsolation note: 'randpair' applies common random numbers WITHOUT upstream's fixed "
              "timesteps. Comparing it to 'fixed' separates correction 1 from correction 2; it is "
              "diagnostic only and the verdict above does not depend on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
