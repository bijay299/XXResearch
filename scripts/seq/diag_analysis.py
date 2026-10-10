#!/usr/bin/env python3
"""Analysis for the matched-effectiveness diagnostic. CPU only, no GPU.

Deliberately a SEPARATE step from run_diagnostic.sh, so no confirmatory number
is produced by the process that spends GPU time.

Everything here is fixed in advance and read from frozen artifacts rather than
chosen at runtime:

  * the resampling grouping comes from
    results/audit_v1/draft_manifests/bootstrap_grouping.json -- the scene
    cluster (set, category, prompt_index), carrying BOTH family wordings and
    ALL FOUR generation seeds together, with CATEGORY as the stratum, 10 000
    draws, 95% percentile intervals and RNG seed 2026100901. Its identity is
    re-verified before use;
  * contrasts are paired at identical (prompt_id, gen_seed), and ONE resampled
    cluster list indexes BOTH arms and every endpoint within a draw;
  * TRAINING SEEDS ARE ANALYSED SEPARATELY AND NEVER POOLED. Two runs support a
    direction check only;
  * every retained category is reported SEPARATELY and never averaged; dog is
    the matching check; sandwich is a non-target; literal and paraphrase are
    reported separately; thresholds 0.3/0.5/0.7 with 0.5 primary.

The primary endpoint is the paired bird contrast D_bird(L2) - D_bird(U*) on the
frozen test set at t=0.5, where U* is the development-selected matched dump.

Three separate test-set checks, never collapsed into one
--------------------------------------------------------
Protocol §5 requires "the SAME gates and the same match" to be re-checked on
TEST. That is three distinct questions and they are reported as three:

  (a) TOLERANCE AGREEMENT   do the two arms delete the same amount here
                            (mismatch <= 5 pp)?
  (b) ELIGIBILITY           does EACH arm, on its own, sit inside the declared
                            partial-suppression regime here (>= 30 pp target
                            suppression vs its own MA AND <= 60% residue)?
  (c) PROTOCOL VALIDITY     (a) AND (b). Only this licenses the word "matched"
                            and the §7 decision table.

(a) without (b) is agreement outside the authorised regime -- two arms that both
under-deleted can agree with each other perfectly. Where (c) fails the retention
contrast is retained as a DESCRIPTIVE measurement and every protocol-level claim
is WITHDRAWN for that seed; withheld is not disproved.

A note on the parent-referenced estimand. Both arms of a seed share that seed's
own MA, so the contrast of parent-referenced changes is ALGEBRAICALLY IDENTICAL
to the direct contrast:
    [D(L2) - D(MA)] - [D(U*) - D(MA)] = D(L2) - D(U*)
Per-arm changes from MA are therefore reported for interpretation, and the fact
that the contrast coincides is stated rather than presented as a second result.

Detector output is a PROXY. Nothing here is a final scientific conclusion: the
blinded human annotation is a prerequisite and is not an input to this script.

    python scripts/seq/diag_analysis.py --diag_root /data/.../diag_v1
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

THRESHOLDS = ("0.3", "0.5", "0.7")
PRIMARY_THR = "0.5"
PRIMARY_CATEGORY = "bird"
RETAINED = ("bird", "horse", "chair", "bicycle")
MATCHING_CHECK = "dog"
NON_TARGET = "sandwich"
HISTORICAL = "cat"
MATERIAL_PP = 10.0          # prespecified material effect size
EQUIV_LO, EQUIV_HI = -10.0, 10.0
# The matching tolerance the selection used. Re-applied HERE, on the frozen test
# set, because matching on the development set does not guarantee matching on
# the test set -- and a retention comparison between arms that are not matched
# on the set being reported is not a matched comparison.
MATCH_TOLERANCE_PP = 5.0
# The REFERENCE GATES of the protocol, MATCHED_EFFECTIVENESS_PROTOCOL.md §5.
# They define the partial-suppression regime the comparison was prespecified to
# live in, and §5 requires "the SAME gates and the same match" to be re-checked
# on TEST -- not the tolerance alone. Both arms must clear both gates: a
# tolerance agreement between two arms that each failed the suppression gate is
# agreement outside the declared regime, which is not a matched comparison.
GATE_SUPPRESSION_PP = 30.0   # >= 30 pp target suppression vs that seed's own MA
GATE_RESIDUE_PCT = 60.0      # <= 60% target residue


def verify_grouping(p: Path) -> dict:
    d = json.loads(p.read_text())
    body = {k: v for k, v in d.items() if k != "grouping_sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if actual != d["grouping_sha256"]:
        raise SystemExit(f"FATAL: bootstrap grouping digest {actual[:12]}… != "
                         f"stored {d['grouping_sha256'][:12]}…: the frozen "
                         f"grouping was altered. Refusing to analyse.")
    return d


def load_arm(eval_root: Path, name: str) -> dict[tuple[str, int], dict]:
    det = eval_root / name / "detections.jsonl"
    if not det.is_file():
        raise SystemExit(f"FATAL: {det} absent")
    rows = {}
    for line in det.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rows[(o["prompt_id"], int(o["gen_seed"]))] = o
    return rows


def rate(rows: list[dict], thr: str) -> float:
    """Detection rate in PERCENT over the given rows."""
    return 100.0 * sum(1 for r in rows if r["hit"][thr]) / len(rows) if rows else float("nan")


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


class Endpoints:
    """Every endpoint, computed from one resampled list of scene clusters."""

    def __init__(self, grouping_set: dict, arms: dict[str, dict]):
        self.arms = arms
        self.clusters = grouping_set["clusters"]
        # stratum -> list of cluster indices
        self.by_stratum: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.clusters):
            self.by_stratum[c["stratum"]].append(i)
        # cluster index -> the (prompt_id, gen_seed) keys it owns, split by family
        self.keys: list[dict] = []
        gen_seeds = grouping_set["gen_seeds"]
        any_arm = next(iter(arms.values()))
        for c in self.clusters:
            allk, byfam = [], defaultdict(list)
            for pid in c["prompt_ids"]:
                for gs in gen_seeds:
                    k = (pid, int(gs))
                    if k not in any_arm:
                        raise SystemExit(f"FATAL: {k} missing from the scored "
                                         f"rows; the evaluation does not cover "
                                         f"the frozen grouping")
                    allk.append(k)
                    byfam[any_arm[k]["prompt_family"]].append(k)
            self.keys.append({"all": allk, "by_family": dict(byfam),
                              "stratum": c["stratum"]})

    def rows_for(self, arm: str, idxs: list[int], stratum: str | None,
                 family: str | None) -> list[dict]:
        a = self.arms[arm]
        out = []
        for i in idxs:
            k = self.keys[i]
            if stratum is not None and k["stratum"] != stratum:
                continue
            ks = k["by_family"].get(family, []) if family else k["all"]
            out.extend(a[x] for x in ks)
        return out

    def compute(self, idxs: list[int]) -> dict:
        """All endpoints for one draw (or for the point estimate)."""
        out: dict[str, float] = {}
        cats = sorted(self.by_stratum)
        for thr in THRESHOLDS:
            for cat in cats:
                for fam in (None, "literal", "paraphrase"):
                    tag = f"{cat}|{thr}" + (f"|{fam}" if fam else "")
                    r = {}
                    for arm in self.arms:
                        r[arm] = rate(self.rows_for(arm, idxs, cat, fam), thr)
                    out[f"D[{tag}]|L2"] = r["L2"]
                    out[f"D[{tag}]|U"] = r["U"]
                    out[f"D[{tag}]|MA"] = r["MA"]
                    # the paired contrast: L2 minus the matched unregularised arm
                    out[f"contrast[{tag}]"] = r["L2"] - r["U"]
                    # per-arm change from that seed's own MA
                    out[f"vsMA[{tag}]|L2"] = r["L2"] - r["MA"]
                    out[f"vsMA[{tag}]|U"] = r["U"] - r["MA"]
        return out


def analyse_seed(seed: int, sel_step: int, eval_root: Path,
                 grouping_set: dict, n_draws: int, rng_seed: int) -> dict:
    arms = {
        "MA": load_arm(eval_root, f"seed{seed}_MA"),
        "L2": load_arm(eval_root, f"seed{seed}_MAB_L2"),
        "U":  load_arm(eval_root, f"seed{seed}_U_step{sel_step}"),
    }
    E = Endpoints(grouping_set, arms)
    all_idx = list(range(len(E.clusters)))
    point = E.compute(all_idx)

    try:
        import numpy as np
    except Exception as e:
        raise SystemExit(f"FATAL: numpy required for the bootstrap ({e})")
    rng = np.random.default_rng(rng_seed)
    strata = sorted(E.by_stratum)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_draws):
        idxs: list[int] = []
        for s in strata:                      # sorted category order, as frozen
            pool = E.by_stratum[s]
            picks = rng.integers(0, len(pool), size=len(pool))
            idxs.extend(pool[int(j)] for j in picks)
        d = E.compute(idxs)
        for k, v in d.items():
            draws[k].append(v)

    ci = {k: {"point": point[k],
              "lo95": percentile(v, 0.025),
              "hi95": percentile(v, 0.975)}
          for k, v in draws.items()}
    return {"training_seed": seed, "selected_U_step": sel_step,
            "n_clusters": len(E.clusters),
            "clusters_per_stratum": {s: len(E.by_stratum[s]) for s in strata},
            "estimates": ci}


def arm_eligibility(arm: str, suppression_pp: float,
                    residue_pct: float) -> dict:
    """Re-apply the §5 reference gates to ONE arm on the set being reported.

    Eligibility is a property of each arm on its own. It is NOT the matching
    tolerance, and the two must never be collapsed: the tolerance asks whether
    the arms deleted the SAME amount, the gates ask whether that amount is
    inside the declared partial-suppression regime at all.
    """
    g_sup = suppression_pp >= GATE_SUPPRESSION_PP - 1e-9
    g_res = residue_pct <= GATE_RESIDUE_PCT + 1e-9
    return {
        "arm": arm,
        "target_suppression_pp_vs_own_MA": round(suppression_pp, 4),
        "target_residue_pct": round(residue_pct, 4),
        "gate_suppression_ge_30pp": g_sup,
        "gate_residue_le_60pct": g_res,
        "eligible": bool(g_sup and g_res),
        "failed_gates": [n for n, ok in (("suppression_ge_30pp", g_sup),
                                         ("residue_le_60pct", g_res)) if not ok],
    }


def read_decision(ci: dict, tag: str) -> dict:
    """Map one interval onto the prespecified rows of the decision table.

    The rows OVERLAP and are reported as such. An interval wholly inside
    [-10, +10] both rules out a 10 pp benefit and supports the equivalence
    statement; testing "upper bound < +10" first and stopping would hide the
    equivalence row in exactly the case where it applies. So every row that
    holds is reported, most specific first.
    """
    c = ci[f"contrast[{tag}]"]
    lo, hi = c["lo95"], c["hi95"]
    rows = []
    within = EQUIV_LO <= lo and hi <= EQUIV_HI
    if within:
        rows.append(f"interval wholly within [{EQUIV_LO:.0f}, {EQUIV_HI:.0f}]")
    if hi < MATERIAL_PP:
        rows.append(f"upper bound {hi:+.1f} < +{MATERIAL_PP:.0f}")
    if lo > MATERIAL_PP:
        rows.append(f"lower bound {lo:+.1f} > +{MATERIAL_PP:.0f}")
    if not rows:
        rows.append("the interval spans the declared margin")
    return {"point": c["point"], "lo95": lo, "hi95": hi,
            "interval_within_declared_margin": within,
            "interval_upper_below_material": hi < MATERIAL_PP,
            "interval_lower_above_material": lo > MATERIAL_PP,
            "interval_spans_material_margin": not (hi < MATERIAL_PP
                                                   or lo > MATERIAL_PP),
            "interval_arithmetic": "; ".join(rows)}


def protocol_reading(arith: dict, valid: bool, failures: list[str]) -> dict:
    """Attach the §7 decision table to an interval -- ONLY if §5 was satisfied.

    Where §5 was not satisfied on the set being reported, the interval survives
    as a DESCRIPTIVE result and every protocol-level claim is withheld. §7's last
    row states this for a failed MATCH -- "the comparison is rejected, not the
    hypothesis" -- and a failed REFERENCE GATE is the same kind of failure of the
    declared test conditions: §10 makes a reference-gate failure a
    stop-and-report condition in its own right. Either way no equivalence
    statement and no exclusion of a material benefit is available.
    """
    if valid:
        rows = []
        if arith["interval_within_declared_margin"]:
            rows.append(f"interval wholly within [{EQUIV_LO:.0f}, {EQUIV_HI:.0f}]: "
                        f"PRACTICAL EQUIVALENCE at the declared margin")
        if arith["interval_upper_below_material"]:
            rows.append(f"upper bound {arith['hi95']:+.1f} < +{MATERIAL_PP:.0f}: a "
                        f"benefit of {MATERIAL_PP:.0f} pp or more is RULED OUT, "
                        f"conditionally")
        if arith["interval_lower_above_material"]:
            rows.append(f"lower bound {arith['lo95']:+.1f} > +{MATERIAL_PP:.0f}: a "
                        f"MATERIAL benefit carries forward")
        if not rows:
            rows.append("INCONCLUSIVE: the interval spans the declared margin; "
                        "do not read the point estimate as a result")
        return {
            **arith,
            "practical_equivalence": arith["interval_within_declared_margin"],
            "material_benefit": arith["interval_lower_above_material"],
            "benefit_of_10pp_ruled_out": arith["interval_upper_below_material"],
            "inconclusive": arith["interval_spans_material_margin"],
            "protocol_level_claim_available": True,
            "withdrawn_claims": [],
            "reading": "; ".join(rows),
        }
    return {
        **arith,
        # Every protocol-level claim is FALSE here -- withheld, not disproved.
        "practical_equivalence": False,
        "material_benefit": False,
        "benefit_of_10pp_ruled_out": False,
        "inconclusive": True,
        "protocol_level_claim_available": False,
        "withdrawn_claims": ["practical_equivalence_at_the_declared_margin",
                             "exclusion_of_a_10pp_material_benefit",
                             "material_benefit_carries_forward"],
        "reading": (
            "DESCRIPTIVE ONLY -- no protocol-level reading is available. The "
            f"prespecified test conditions of §5 were not met on this set "
            f"({', '.join(failures)}), so the §7 decision table does not apply. "
            f"§7's last row rejects the COMPARISON rather than the hypothesis "
            f"when the match fails, and §10 makes a reference-gate failure a "
            f"stop-and-report condition; this is read the same way. The interval "
            f"({arith['point']:+.1f} pp [{arith['lo95']:+.1f}, "
            f"{arith['hi95']:+.1f}]) stands as a description of what was "
            f"measured between these two checkpoints and supports no "
            f"equivalence statement and no exclusion of a material benefit. "
            f"Arithmetically: {arith['interval_arithmetic']}."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag_root", required=True)
    ap.add_argument("--grouping",
                    default="results/audit_v1/draft_manifests/bootstrap_grouping.json")
    ap.add_argument("--set", default="test", choices=["test", "dev"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diag = Path(args.diag_root)
    grouping = verify_grouping(Path(args.grouping))
    gset = grouping["sets"][args.set]
    n_draws = grouping["n_draws"]
    rng_seed = grouping["rng"]["bootstrap_rng_seed"]
    eval_root = diag / ("eval_test" if args.set == "test" else "eval_dev")

    sel_p = diag / "selection.json"
    if not sel_p.is_file():
        raise SystemExit(f"FATAL: {sel_p} absent; selection has not been decided")
    sel = json.loads(sel_p.read_text())
    if not sel.get("proceed_to_frozen_test"):
        raise SystemExit("FATAL: development matching did not succeed for both "
                         "seeds, so there is no confirmatory analysis to run. "
                         "The selection record is the result.")
    seeds = sel["seeds"]

    print(f"grouping identity   : {grouping['grouping_sha256'][:16]}… verified")
    print(f"resampling unit     : {grouping['resampling_unit']['definition']}")
    print(f"stratum             : {grouping['stratification']['stratum']}")
    print(f"draws / interval    : {n_draws} / 95% percentile")
    print(f"bootstrap RNG seed  : {rng_seed}")
    print(f"set                 : {args.set} ({gset['n_rows']} rows, "
          f"{gset['n_clusters']} clusters)")
    print("training seeds are analysed SEPARATELY and never pooled\n")

    per_seed = {}
    for s in seeds:
        step = sel["per_seed"][str(s)]["selected"]["step"]
        print(f"--- seed {s}: U* = dump at step {step} "
              f"(bootstrapping {n_draws} draws) ---", flush=True)
        per_seed[s] = analyse_seed(s, step, eval_root, gset, n_draws, rng_seed)

    primary_tag = f"{PRIMARY_CATEGORY}|{PRIMARY_THR}"

    # ---- the matching check, RE-REPORTED ON TEST. No reselection.
    #
    # Both arms of a seed share that seed's own MA, so the on-test difference in
    # target deletion is exactly -contrast[dog], and its interval comes free from
    # the same bootstrap. Selection matched on the DEVELOPMENT set; whether the
    # arms are still matched on the FROZEN TEST set is a separate question, and
    # it is the one that licenses calling the retention contrast "matched".
    match_tag = f"{MATCHING_CHECK}|{PRIMARY_THR}"
    test_matching, test_eligibility, test_validity = {}, {}, {}
    for s in seeds:
        est = per_seed[s]["estimates"]
        c = est[f"contrast[{match_tag}]"]
        l2 = est[f"vsMA[{match_tag}]|L2"]
        u = est[f"vsMA[{match_tag}]|U"]
        mismatch = abs(c["point"])
        ok = mismatch <= MATCH_TOLERANCE_PP + 1e-9
        dev_mismatch = sel["per_seed"][str(s)]["selected"]["mismatch_vs_L2_pp"]

        # ---- (a) TOLERANCE AGREEMENT: did the two arms delete the SAME amount?
        test_matching[str(s)] = {
            "_this_checks": ("ONLY whether the two arms agree in target "
                             "deletion. It says nothing about whether that "
                             "deletion level is inside the declared regime; "
                             "that is the separate eligibility check."),
            "target": MATCHING_CHECK, "threshold": PRIMARY_THR,
            "L2_target_deletion_pp_vs_MA": -l2["point"],
            "U_target_deletion_pp_vs_MA": -u["point"],
            "mismatch_pp": round(mismatch, 4),
            "mismatch_interval_pp": [c["lo95"], c["hi95"]],
            "which_arm_deleted_more": ("U" if c["point"] > 0 else
                                       "L2" if c["point"] < 0 else "neither"),
            "tolerance_pp": MATCH_TOLERANCE_PP,
            "tolerance_agreement_on_test": ok,
            "matched_on_test": ok,   # retained key name; tolerance only
            "development_mismatch_pp": dev_mismatch,
            "status": ("TOLERANCE_AGREEMENT_ON_TEST" if ok else
                       "TOLERANCE_FAILED_ON_TEST"),
            "consequence": (
                "the two arms agree in target deletion on the frozen test set "
                "within the tolerance selection used. This is necessary but "
                "NOT sufficient: the reference gates must also hold on this "
                "set for the comparison to be a matched comparison in the "
                "prespecified regime."
                if ok else
                f"the arms differ by {mismatch:.2f} pp in target deletion ON "
                f"THE TEST SET, beyond the {MATCH_TOLERANCE_PP} pp tolerance. "
                f"Agreement held on the development set "
                f"({dev_mismatch} pp) but does not hold here, so the retention "
                f"contrast for this seed is INCONCLUSIVE as a matched "
                f"comparison. No reselection is performed: the selected "
                f"checkpoint stands and this failure is reported as the result "
                f"for this seed."),
        }

        # ---- (b) ELIGIBILITY: is each arm inside the declared regime at all?
        #
        # §5 gates the REFERENCE and the CANDIDATE, and §5's reporting rule
        # requires the SAME gates re-checked on TEST. Deletion measured on the
        # test prompts is a different quantity from deletion measured on the
        # development prompts, so passing on DEV does not carry over.
        sup_l2, sup_u = -l2["point"], -u["point"]
        res_l2 = est[f"D[{match_tag}]|L2"]["point"]
        res_u = est[f"D[{match_tag}]|U"]["point"]
        e_l2 = arm_eligibility("L2", sup_l2, res_l2)      # the reference arm
        e_u = arm_eligibility("U", sup_u, res_u)           # the candidate arm
        both_eligible = e_l2["eligible"] and e_u["eligible"]
        test_eligibility[str(s)] = {
            "_this_checks": ("whether EACH arm independently clears the §5 "
                             "reference gates on the set being reported: "
                             f">= {GATE_SUPPRESSION_PP:.0f} pp target "
                             f"suppression vs that seed's own MA AND "
                             f"<= {GATE_RESIDUE_PCT:.0f}% target residue."),
            "target": MATCHING_CHECK, "threshold": PRIMARY_THR,
            "gate_suppression_pp": GATE_SUPPRESSION_PP,
            "gate_residue_pct": GATE_RESIDUE_PCT,
            "reference_arm_L2": e_l2,
            "candidate_arm_U": e_u,
            "both_arms_eligible": both_eligible,
            "status": ("BOTH_ARMS_ELIGIBLE_ON_TEST" if both_eligible else
                       "NOT_ELIGIBLE_ON_TEST"),
            "development_note": (
                "Eligibility was satisfied on the DEVELOPMENT set at selection "
                "time; it is re-checked here because the test prompts are a "
                "different measurement. A DEV pass does not transfer."),
            "consequence": (
                "both arms sit inside the declared partial-suppression regime "
                "on this set"
                if both_eligible else
                "at least one arm is OUTSIDE the declared partial-suppression "
                "regime on this set, so there is no prespecified regime in "
                "which to read the retention contrast: the arms may agree with "
                "each other while both having deleted too little to be the "
                "comparison that was authorised."),
        }

        # ---- (c) OVERALL PROTOCOL VALIDITY: (a) AND (b), stated separately
        failures: list[str] = []
        if not ok:
            failures.append(f"tolerance: mismatch {mismatch:.2f} pp > "
                            f"{MATCH_TOLERANCE_PP:.0f} pp")
        for e in (e_l2, e_u):
            if not e["eligible"]:
                failures.append(
                    f"eligibility ({e['arm']}): suppression "
                    f"{e['target_suppression_pp_vs_own_MA']:.2f} pp, residue "
                    f"{e['target_residue_pct']:.2f}% -> failed "
                    f"{', '.join(e['failed_gates'])}")
        test_validity[str(s)] = {
            "tolerance_agreement_on_test": ok,
            "both_arms_eligible_on_test": both_eligible,
            "valid_matched_comparison": bool(ok and both_eligible),
            "failed_conditions": failures,
            "status": ("VALID_MATCHED_COMPARISON" if ok and both_eligible
                       else "NOT_A_VALID_MATCHED_COMPARISON"),
            "consequence": (
                "the prespecified test conditions hold on this set, so the §7 "
                "decision table applies to this seed's retention contrast"
                if ok and both_eligible else
                "the prespecified test conditions do NOT hold on this set. The "
                "retention contrast is retained as a DESCRIPTIVE result and "
                "every protocol-level reading is withdrawn for this seed: the "
                "comparison is rejected, not the hypothesis (§7 last row for a "
                "failed match; §10 for a failed reference gate). No "
                "reselection, no widened tolerance, no retraining."),
        }

    all_matched_on_test = all(v["matched_on_test"] for v in test_matching.values())
    all_eligible_on_test = all(v["both_arms_eligible"]
                               for v in test_eligibility.values())
    any_valid = any(v["valid_matched_comparison"] for v in test_validity.values())
    all_valid = all(v["valid_matched_comparison"] for v in test_validity.values())
    payload = {
        "_what_this_is": (
            "The matched-effectiveness diagnostic's detector-based estimates. "
            "PROVISIONAL: the blinded human annotation is a prerequisite for any "
            "final scientific conclusion and is not an input here."),
        "primary_endpoint": (
            f"paired contrast D_{PRIMARY_CATEGORY}(L2) - D_{PRIMARY_CATEGORY}(U*) "
            f"on the frozen test set at t={PRIMARY_THR}"),
        "grouping_sha256": grouping["grouping_sha256"],
        "bootstrap": {"n_draws": n_draws, "rng_seed": rng_seed,
                      "interval": "95% percentile",
                      "unit": grouping["resampling_unit"]["definition"],
                      "stratum": grouping["stratification"]["stratum"]},
        "selection": {str(s): {"step": sel["per_seed"][str(s)]["selected"]["step"],
                               "mismatch_vs_L2_pp":
                                   sel["per_seed"][str(s)]["selected"]["mismatch_vs_L2_pp"]}
                      for s in seeds},
        "seeds_pooled": False,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "test_conditions_are_three_separate_checks": (
            "(a) test_matching_check -- TOLERANCE AGREEMENT between the arms; "
            "(b) test_eligibility_check -- each arm's own §5 REFERENCE GATES; "
            "(c) test_protocol_validity -- (a) AND (b), which is what licenses "
            "the word 'matched' and the §7 decision table. These are reported "
            "separately and must not be collapsed into one another."),
        "test_matching_check": test_matching,
        "test_eligibility_check": test_eligibility,
        "test_protocol_validity": test_validity,
        "matched_on_test_all_seeds": all_matched_on_test,
        "eligible_on_test_all_seeds": all_eligible_on_test,
        "valid_matched_comparison_all_seeds": all_valid,
        "valid_matched_comparison_any_seed": any_valid,
        "matched_comparison_status": (
            "VALID matched comparison on the frozen test set for every seed: "
            "both arms clear the §5 reference gates and agree within tolerance"
            if all_valid else
            "NOT A VALID MATCHED COMPARISON on the frozen test set for " +
            ("at least one seed" if any_valid else "ANY seed") + ". The "
            "prespecified test conditions of §5 -- the reference gates AND the "
            "match tolerance -- were not all met. The retention contrasts below "
            "are retained as DESCRIPTIVE results; they do NOT license a "
            "matched-effectiveness claim, a practical-equivalence statement or "
            "the exclusion of a material benefit for any seed whose conditions "
            "failed. The comparison is rejected, not the hypothesis."),
        "primary_reading_per_seed": {
            str(s): {**protocol_reading(
                         read_decision(per_seed[s]["estimates"], primary_tag),
                         test_validity[str(s)]["valid_matched_comparison"],
                         test_validity[str(s)]["failed_conditions"]),
                     "matched_on_test": test_matching[str(s)]["matched_on_test"],
                     "tolerance_agreement_on_test":
                         test_matching[str(s)]["tolerance_agreement_on_test"],
                     "both_arms_eligible_on_test":
                         test_eligibility[str(s)]["both_arms_eligible"],
                     "valid_matched_comparison":
                         test_validity[str(s)]["valid_matched_comparison"],
                     "failed_conditions":
                         test_validity[str(s)]["failed_conditions"],
                     "matched_comparison": (
                         "VALID: the arms clear the reference gates and agree "
                         "within tolerance on this set"
                         if test_validity[str(s)]["valid_matched_comparison"] else
                         "NOT A VALID MATCHED COMPARISON on this set: " +
                         "; ".join(test_validity[str(s)]["failed_conditions"]))}
            for s in seeds},
        "parent_referenced_note": (
            "Both arms of a seed share that seed's own MA, so the contrast of "
            "parent-referenced changes is algebraically identical to the direct "
            "contrast. Per-arm changes from MA are reported for interpretation; "
            "the coincidence is stated, not presented as a second result."),
        "interpretation_rules": [
            "An interval including zero is NOT evidence of no effect.",
            "A difference in significance labels between arms or seeds is NOT a "
            "significant difference.",
            "No multiplicity correction is applied, so each interval is "
            "DESCRIPTIVE, not a test.",
            "Only an interval wholly within [-10, +10] supports an equivalence "
            "statement, and only at that margin.",
            "Two training seeds support a DIRECTION CHECK only; they are never "
            "pooled and support no population claim.",
            "Intervals are conditional on this base model, these concepts, these "
            "anchors, these prompts and this detector.",
            "Residual dependence from a template shared ACROSS scene clusters is "
            "not absorbed by this bootstrap, and its direction and size are not "
            "established.",
            "The detector is a PROXY. Final conclusions require the blinded human "
            "annotation.",
            "Matching was established on the DEVELOPMENT set and is re-checked "
            "here on the TEST set. Where the on-test check fails, the retention "
            "contrast for that seed is INCONCLUSIVE as a matched comparison, "
            "and no reselection is permitted.",
            "The on-test re-check is the SAME GATES AND THE SAME MATCH, not the "
            "match alone: each arm must independently clear >= 30 pp target "
            "suppression and <= 60% target residue on this set, AND the two "
            "arms must agree within 5 pp. Tolerance agreement between two "
            "arms that both under-deleted is agreement outside the declared "
            "regime and is not a matched comparison.",
            "Where the prespecified test conditions fail, the contrast is "
            "retained as a DESCRIPTIVE measurement and every protocol-level "
            "claim -- practical equivalence at the declared margin, and the "
            "exclusion of a 10 pp material benefit -- is WITHDRAWN for that "
            "seed. Withheld is not disproved: nothing here shows the arms "
            "differ, only that this run cannot say they do not.",
            "A failure of the test conditions is a statement about THIS "
            "selected pair in THIS pilot. It is not evidence of a general "
            "defect in matched-effectiveness designs, and its causes are "
            "unresolved here.",
        ],
        "analysed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out = Path(args.out) if args.out else diag / f"analysis_{args.set}.json"
    # Never overwrite a previous analysis record: it is evidence of what was
    # reported and when.
    if out.exists():
        prev = json.loads(out.read_text())
        stamp = (prev.get("analysed_utc") or "unknown").replace(":", "").replace("-", "")[:15]
        keep = out.with_name(f"{out.stem}.{stamp}.superseded.json")
        n = 0
        while keep.exists():
            n += 1
            keep = out.with_name(f"{out.stem}.{stamp}.superseded.{n}.json")
        keep.write_text(json.dumps(prev, indent=2) + "\n")
        payload["supersedes"] = {"preserved_at": str(keep),
                                 "preserved_analysed_utc": prev.get("analysed_utc")}
        print(f"[preserved] previous analysis record -> {keep}")
    out.write_text(json.dumps(payload, indent=2) + "\n")

    # ---- human-readable summary
    print(f"\n=== THE SAME GATES AND THE SAME MATCH, RE-CHECKED ON THE "
          f"FROZEN TEST SET ===")
    print(f"gates: >= {GATE_SUPPRESSION_PP:.0f} pp {MATCHING_CHECK} suppression "
          f"vs own MA AND <= {GATE_RESIDUE_PCT:.0f}% residue, EACH ARM; "
          f"tolerance: mismatch <= {MATCH_TOLERANCE_PP:.0f} pp")
    for s in seeds:
        m, el, v = test_matching[str(s)], test_eligibility[str(s)], test_validity[str(s)]
        print(f"\nseed {s}:")
        for e in (el["reference_arm_L2"], el["candidate_arm_U"]):
            label = "L2 (reference)" if e["arm"] == "L2" else "U* (candidate)"
            print(f"  {label:<16} suppression "
                  f"{e['target_suppression_pp_vs_own_MA']:6.2f} pp "
                  f"[{'PASS' if e['gate_suppression_ge_30pp'] else 'FAIL'}]   "
                  f"residue {e['target_residue_pct']:6.2f}% "
                  f"[{'PASS' if e['gate_residue_le_60pct'] else 'FAIL'}]   -> "
                  f"{'ELIGIBLE' if e['eligible'] else 'NOT ELIGIBLE'}")
        print(f"  tolerance        mismatch {m['mismatch_pp']:6.2f} pp "
              f"(development: {m['development_mismatch_pp']} pp) -> "
              f"{m['status']}")
        print(f"  OVERALL          {v['status']}")
        for f in v["failed_conditions"]:
            print(f"                   failed: {f}")
        if not v["valid_matched_comparison"]:
            print(f"                   {v['consequence']}")
    if not any_valid:
        print(f"\n  NO SEED supplies a valid matched comparison in the "
              f"prespecified regime.")

    print(f"\n=== PRIMARY: paired {PRIMARY_CATEGORY} contrast at t={PRIMARY_THR} "
          f"(L2 minus matched-U), per seed ===")
    for s in seeds:
        r = payload["primary_reading_per_seed"][str(s)]
        print(f"  seed {s}: {r['point']:+.1f} pp  [{r['lo95']:+.1f}, {r['hi95']:+.1f}]"
              f"   {'PROTOCOL-LEVEL' if r['protocol_level_claim_available'] else 'DESCRIPTIVE ONLY'}")
        print(f"           {r['reading']}")
        if r["withdrawn_claims"]:
            print(f"           withdrawn: {', '.join(r['withdrawn_claims'])}")
    dirs = {s: payload["primary_reading_per_seed"][str(s)]["point"] for s in seeds}
    if len(seeds) == 2 and (dirs[seeds[0]] > 0) != (dirs[seeds[1]] > 0):
        print("  DIRECTIONS DISAGREE between seeds -> UNRESOLVED. Do not report "
              "the agreeing seed.")

    print(f"\n=== every category separately, never averaged (t={PRIMARY_THR}) ===")
    for s in seeds:
        ci = per_seed[s]["estimates"]
        print(f"  seed {s}:")
        for cat in sorted(per_seed[s]["clusters_per_stratum"]):
            tag = f"{cat}|{PRIMARY_THR}"
            c = ci[f"contrast[{tag}]"]
            role = ("PRIMARY" if cat == PRIMARY_CATEGORY else
                    "matching check" if cat == MATCHING_CHECK else
                    "non-target" if cat == NON_TARGET else
                    "historical" if cat == HISTORICAL else
                    "retained" if cat in RETAINED else "")
            print(f"    {cat:<9} L2={ci[f'D[{tag}]|L2']['point']:5.1f}%  "
                  f"U*={ci[f'D[{tag}]|U']['point']:5.1f}%  "
                  f"MA={ci[f'D[{tag}]|MA']['point']:5.1f}%  "
                  f"contrast {c['point']:+6.1f} "
                  f"[{c['lo95']:+6.1f}, {c['hi95']:+6.1f}]  {role}")

    print(f"\n=== literal vs paraphrase, reported separately (t={PRIMARY_THR}) ===")
    for s in seeds:
        ci = per_seed[s]["estimates"]
        print(f"  seed {s}:")
        for fam in ("literal", "paraphrase"):
            tag = f"{PRIMARY_CATEGORY}|{PRIMARY_THR}|{fam}"
            c = ci[f"contrast[{tag}]"]
            print(f"    {PRIMARY_CATEGORY} {fam:<11} contrast {c['point']:+6.1f} "
                  f"[{c['lo95']:+6.1f}, {c['hi95']:+6.1f}]")

    print(f"\n=== matching check on TEST: {MATCHING_CHECK} "
          f"(the match was fixed on DEV and is NOT reselected here) ===")
    for s in seeds:
        ci = per_seed[s]["estimates"]
        tag = f"{MATCHING_CHECK}|{PRIMARY_THR}"
        c = ci[f"contrast[{tag}]"]
        print(f"  seed {s}: residue L2={ci[f'D[{tag}]|L2']['point']:.1f}%  "
              f"U*={ci[f'D[{tag}]|U']['point']:.1f}%  "
              f"achieved difference {c['point']:+.1f} pp "
              f"[{c['lo95']:+.1f}, {c['hi95']:+.1f}]")

    print(f"\n[analysis] {out}")
    print("\nPROVISIONAL. The detector is a proxy; the blinded human annotation "
          "is a prerequisite for any final scientific conclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
