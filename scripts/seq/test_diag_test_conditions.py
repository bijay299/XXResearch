#!/usr/bin/env python3
"""Focused CPU regression for the ON-TEST test conditions of protocol §5.

Why this file exists
--------------------
`diag_analysis.py` originally re-checked only the MATCH TOLERANCE on the frozen
test set. Protocol §5 requires "the SAME gates and the same match": each arm
must independently clear the reference gates (>= 30 pp target suppression vs its
own MA AND <= 60% target residue) on the set being reported, AND the two arms
must agree within 5 pp. With only the tolerance checked, a pair of arms that had
both under-deleted could agree with each other perfectly and be reported as a
practical-equivalence result — which is what happened in Amendment 01.

The two cases that matter are therefore:

  CASE A  arms CLOSE but BOTH UNDER-SUPPRESSED. Tolerance passes, eligibility
          fails, so there is no valid matched comparison and every
          protocol-level claim must be withdrawn — while the interval itself
          survives as a descriptive result.

  CASE B  a VALID POSITIVE. Both arms inside the regime and within tolerance, so
          the §7 decision table does apply and the readings are available.

Plus the gate boundaries, and a source-bound check against the committed
Amendment 01 analysis record.

No GPU, no model, no network.

    python scripts/seq/test_diag_test_conditions.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_diag_analysis as H        # noqa: E402  (build/run harness reuse)

ANALYSIS_RECORD = (REPO / "results" / "audit_v1" / "amendment01_evidence"
                   / "analysis_test.json")

PASS: list[str] = []
FAIL: list[str] = []


def check(desc: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(desc)
    print(f"  {'ok  ' if cond else 'FAIL'}  {desc}" + ("" if cond else f"  <- {detail}"))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "diag_analysis", REPO / "scripts" / "seq" / "diag_analysis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Every arm shares one non-target baseline so only dog and bird move.
def rates(ma_dog: float, arm_dog: float, l2_bird: float, u_bird: float) -> dict:
    base = {c: 0.9 for c in ("cat", "dog", "sandwich", "horse", "bird",
                             "chair", "bicycle")}
    return {
        "MA": {**base, "dog": ma_dog},
        "L2": {**base, "dog": arm_dog, "bird": l2_bird},
        "U":  {**base, "dog": arm_dog, "bird": u_bird},
    }


def main() -> int:
    m = load_module()

    print("\nthe gate constants are the protocol's, not invented here")
    check("suppression gate is 30 pp", m.GATE_SUPPRESSION_PP == 30.0,
          str(m.GATE_SUPPRESSION_PP))
    check("residue gate is 60%", m.GATE_RESIDUE_PCT == 60.0,
          str(m.GATE_RESIDUE_PCT))
    check("match tolerance is 5 pp", m.MATCH_TOLERANCE_PP == 5.0,
          str(m.MATCH_TOLERANCE_PP))

    print("\narm_eligibility: both gates, and the boundaries are inclusive")
    e = m.arm_eligibility("L2", 30.0, 60.0)
    check("exactly on both gates is ELIGIBLE", e["eligible"], json.dumps(e))
    check("29.99 pp suppression fails the suppression gate",
          not m.arm_eligibility("U", 29.99, 50.0)["gate_suppression_ge_30pp"])
    check("60.01% residue fails the residue gate",
          not m.arm_eligibility("U", 40.0, 60.01)["gate_residue_le_60pct"])
    under = m.arm_eligibility("U", 16.25, 52.5)        # the real seed-17 U* arm
    check("an under-suppressed arm that clears residue is NOT eligible",
          not under["eligible"] and under["gate_residue_le_60pct"]
          and not under["gate_suppression_ge_30pp"], json.dumps(under))
    check("the failing gate is named",
          under["failed_gates"] == ["suppression_ge_30pp"],
          str(under["failed_gates"]))

    print("\nprotocol_reading withholds every claim when §5 was not satisfied")
    arith = m.read_decision({"contrast[bird|0.5]": {"point": 1.25, "lo95": -2.5,
                                                    "hi95": 5.0}}, "bird|0.5")
    check("the arithmetic layer carries no protocol claim",
          "practical_equivalence" not in arith
          and arith["interval_within_declared_margin"] is True, json.dumps(arith))
    good = m.protocol_reading(arith, True, [])
    check("a VALID comparison gets the equivalence reading",
          good["practical_equivalence"] and good["benefit_of_10pp_ruled_out"]
          and good["protocol_level_claim_available"]
          and "PRACTICAL EQUIVALENCE" in good["reading"], good["reading"])
    bad = m.protocol_reading(arith, False, ["eligibility (L2): failed"])
    check("an INVALID comparison gets no equivalence claim",
          bad["practical_equivalence"] is False
          and bad["benefit_of_10pp_ruled_out"] is False
          and bad["material_benefit"] is False
          and bad["protocol_level_claim_available"] is False, json.dumps(bad))
    check("and the withdrawal is recorded explicitly, not implied",
          set(bad["withdrawn_claims"]) == {
              "practical_equivalence_at_the_declared_margin",
              "exclusion_of_a_10pp_material_benefit",
              "material_benefit_carries_forward"},
          str(bad["withdrawn_claims"]))
    check("the interval survives as a descriptive result",
          bad["point"] == 1.25 and bad["lo95"] == -2.5 and bad["hi95"] == 5.0
          and "DESCRIPTIVE ONLY" in bad["reading"], bad["reading"])
    check("withheld is not presented as disproved",
          "rejects the COMPARISON rather than the hypothesis" in bad["reading"],
          bad["reading"])

    tmp = Path(tempfile.mkdtemp(prefix="testconditions_"))
    try:
        # ---- CASE A: close, but both arms under-suppressed -----------------
        # MA dog 50% -> both arms 30%: suppression 20 pp each (FAILS the 30 pp
        # gate), residue 30% each (clears the 60% gate), mismatch 0.00 pp
        # (clears the 5 pp tolerance). Bird identical in both arms, so the bird
        # interval is exactly [0, 0] -- the single most seductive case, because
        # the tolerance-only code read it as practical equivalence.
        print("\nCASE A: arms close, BOTH under-suppressed")
        rootA = H.build(tmp / "A", rates(0.5, 0.3, 0.6, 0.6))
        rc, out = H.run(rootA)
        check("analysis still exits 0 and reports, rather than crashing",
              rc == 0, out.strip()[-300:])
        resA = json.loads((rootA / "analysis_test.json").read_text())
        mt = resA["test_matching_check"]["17"]
        el = resA["test_eligibility_check"]["17"]
        va = resA["test_protocol_validity"]["17"]
        check("tolerance agreement PASSES (mismatch 0.00 pp)",
              mt["tolerance_agreement_on_test"] and mt["mismatch_pp"] == 0.0,
              json.dumps(mt["mismatch_pp"]))
        check("suppression is 20.00 pp on both arms",
              el["reference_arm_L2"]["target_suppression_pp_vs_own_MA"] == 20.0
              and el["candidate_arm_U"]["target_suppression_pp_vs_own_MA"] == 20.0,
              json.dumps(el))
        check("residue clears its gate on both arms",
              el["reference_arm_L2"]["gate_residue_le_60pct"]
              and el["candidate_arm_U"]["gate_residue_le_60pct"])
        check("ELIGIBILITY FAILS for both arms",
              not el["both_arms_eligible"]
              and not el["reference_arm_L2"]["eligible"]
              and not el["candidate_arm_U"]["eligible"], el["status"])
        check("overall: NOT a valid matched comparison",
              not va["valid_matched_comparison"], va["status"])
        check("the failure names eligibility, not tolerance",
              va["failed_conditions"]
              and all("eligibility" in f for f in va["failed_conditions"]),
              str(va["failed_conditions"]))
        rA = resA["primary_reading_per_seed"]["17"]
        check("the bird interval is exactly [0.0, 0.0] and is still reported",
              rA["point"] == 0.0 and rA["lo95"] == 0.0 and rA["hi95"] == 0.0,
              f"{rA['point']} [{rA['lo95']}, {rA['hi95']}]")
        check("a zero-width interval inside the margin is NOT called equivalence",
              rA["practical_equivalence"] is False
              and "PRACTICAL EQUIVALENCE" not in rA["reading"], rA["reading"])
        check("and a 10 pp benefit is NOT reported as ruled out",
              rA["benefit_of_10pp_ruled_out"] is False, rA["reading"])
        check("the arithmetic fact is still available, separately",
              rA["interval_within_declared_margin"] is True
              and rA["interval_upper_below_material"] is True,
              json.dumps({k: rA[k] for k in rA if k.startswith("interval_")}))
        check("no seed is reported as a valid matched comparison",
              resA["valid_matched_comparison_any_seed"] is False
              and resA["valid_matched_comparison_all_seeds"] is False)
        check("the top-level status says ANY seed, not just one",
              "ANY seed" in resA["matched_comparison_status"],
              resA["matched_comparison_status"][:160])
        check("the three checks are documented as separate",
              "must not be collapsed" in
              resA["test_conditions_are_three_separate_checks"])
        check("the output warns against generalising the failure",
              any("general defect" in r for r in resA["interpretation_rules"]))

        # ---- CASE B: a valid positive --------------------------------------
        # MA dog 90% -> both arms 40%: suppression 50 pp (PASS), residue 40%
        # (PASS), mismatch 0.00 pp (PASS). Bird 80% vs 50% -> +30 pp.
        print("\nCASE B: a VALID positive case")
        rootB = H.build(tmp / "B", rates(0.9, 0.4, 0.8, 0.5))
        rc, out = H.run(rootB)
        check("analysis exits 0", rc == 0, out.strip()[-300:])
        resB = json.loads((rootB / "analysis_test.json").read_text())
        elB = resB["test_eligibility_check"]["17"]
        vaB = resB["test_protocol_validity"]["17"]
        check("suppression is 50.00 pp on both arms, residue 40.00%",
              elB["reference_arm_L2"]["target_suppression_pp_vs_own_MA"] == 50.0
              and elB["candidate_arm_U"]["target_residue_pct"] == 40.0,
              json.dumps(elB))
        check("BOTH arms are eligible",
              elB["both_arms_eligible"], elB["status"])
        check("tolerance agreement passes",
              resB["test_matching_check"]["17"]["tolerance_agreement_on_test"])
        check("overall: VALID matched comparison, no failed conditions",
              vaB["valid_matched_comparison"] and vaB["failed_conditions"] == [],
              json.dumps(vaB))
        check("every seed is valid, so the top-level status says so",
              resB["valid_matched_comparison_all_seeds"] is True
              and "VALID matched comparison" in resB["matched_comparison_status"])
        rB = resB["primary_reading_per_seed"]["17"]
        check("the §7 decision table IS applied here",
              rB["protocol_level_claim_available"] is True
              and rB["withdrawn_claims"] == [], rB["reading"])
        check("the +30 pp reading agrees with its own bounds",
              rB["material_benefit"] == (rB["lo95"] > 10.0)
              and ("MATERIAL" in rB["reading"]) == (rB["lo95"] > 10.0),
              f"{rB['point']:+.1f} [{rB['lo95']:+.1f}, {rB['hi95']:+.1f}] "
              f"-> {rB['reading']}")
        check("a valid comparison carries no DESCRIPTIVE-ONLY disclaimer",
              "DESCRIPTIVE ONLY" not in rB["reading"], rB["reading"])

        # ---- CASE C: the residue gate fails on its own ---------------------
        # MA dog 100% -> both arms 70%: suppression 30 pp (exactly ON the gate,
        # PASS), residue 70% (FAILS). Confirms the two gates are independent and
        # that the suppression boundary is inclusive end to end.
        print("\nCASE C: suppression exactly on the gate, residue over it")
        rootC = H.build(tmp / "C", rates(1.0, 0.7, 0.6, 0.6))
        rc, out = H.run(rootC)
        check("analysis exits 0", rc == 0, out.strip()[-300:])
        resC = json.loads((rootC / "analysis_test.json").read_text())
        elC = resC["test_eligibility_check"]["17"]
        check("suppression of exactly 30.00 pp PASSES",
              elC["reference_arm_L2"]["gate_suppression_ge_30pp"]
              and elC["reference_arm_L2"]["target_suppression_pp_vs_own_MA"] == 30.0,
              json.dumps(elC["reference_arm_L2"]))
        check("70.00% residue FAILS, so the arm is not eligible",
              not elC["reference_arm_L2"]["gate_residue_le_60pct"]
              and not elC["both_arms_eligible"])
        check("the failure names the residue gate specifically",
              elC["reference_arm_L2"]["failed_gates"] == ["residue_le_60pct"],
              str(elC["reference_arm_L2"]["failed_gates"]))
        check("tolerance still passes, and is reported separately",
              resC["test_matching_check"]["17"]["tolerance_agreement_on_test"])
        check("overall is invalid despite the tolerance passing",
              not resC["test_protocol_validity"]["17"]["valid_matched_comparison"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- source-bound: the committed Amendment 01 record -------------------
    print("\nsource-bound: the committed Amendment 01 analysis record")
    if not ANALYSIS_RECORD.is_file():
        check(f"{ANALYSIS_RECORD.relative_to(REPO)} exists", False, "absent")
    else:
        rec = json.loads(ANALYSIS_RECORD.read_text())
        check("the record carries the eligibility check",
              "test_eligibility_check" in rec and "test_protocol_validity" in rec,
              str(sorted(rec)[:6]))
        expect = {"17": (17.5, 16.25, True), "29": (22.5, 28.75, False)}
        for sd, (sup_l2, sup_u, tol) in expect.items():
            el = rec["test_eligibility_check"][sd]
            mt = rec["test_matching_check"][sd]
            va = rec["test_protocol_validity"][sd]
            check(f"seed {sd}: L2 suppression {sup_l2} pp, below the 30 pp gate",
                  el["reference_arm_L2"]["target_suppression_pp_vs_own_MA"] == sup_l2
                  and not el["reference_arm_L2"]["gate_suppression_ge_30pp"],
                  json.dumps(el["reference_arm_L2"]))
            check(f"seed {sd}: U* suppression {sup_u} pp, below the 30 pp gate",
                  el["candidate_arm_U"]["target_suppression_pp_vs_own_MA"] == sup_u
                  and not el["candidate_arm_U"]["gate_suppression_ge_30pp"],
                  json.dumps(el["candidate_arm_U"]))
            check(f"seed {sd}: tolerance agreement is "
                  f"{'PASS' if tol else 'FAIL'}, reported separately",
                  mt["tolerance_agreement_on_test"] is tol,
                  json.dumps(mt["mismatch_pp"]))
            check(f"seed {sd}: NOT a valid matched comparison",
                  not va["valid_matched_comparison"], va["status"])
            r = rec["primary_reading_per_seed"][sd]
            check(f"seed {sd}: no practical-equivalence claim stands",
                  r["practical_equivalence"] is False
                  and r["benefit_of_10pp_ruled_out"] is False
                  and r["protocol_level_claim_available"] is False, r["reading"])
            check(f"seed {sd}: the bird interval is retained as descriptive",
                  all(k in r for k in ("point", "lo95", "hi95"))
                  and "DESCRIPTIVE ONLY" in r["reading"],
                  f"{r['point']} [{r['lo95']}, {r['hi95']}]")
        check("NEITHER seed supplies a valid matched comparison",
              rec["valid_matched_comparison_any_seed"] is False,
              rec["matched_comparison_status"][:160])
        check("the record is still labelled provisional on human annotation",
              "PROVISIONAL" in rec["_what_this_is"])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL ON-TEST CONDITION CHECKS PASSED (CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
