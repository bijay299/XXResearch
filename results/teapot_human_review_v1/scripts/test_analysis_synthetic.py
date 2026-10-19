#!/usr/bin/env python3
"""Synthetic edge-case suite for the aggregation rules (contract §12).

Uses only invented data. No real response, alias or private identifier appears
here or is read by it.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_rules import (UNRESOLVED, comparison_distribution,
                            definite_presence_coverage, demonstrated_removal,
                            derive_presence, derive_rendering,
                            presence_agreement, presence_table,
                            rendering_distribution)

CHECKS = []


def chk(name, got, want):
    ok = got == want
    CHECKS.append({"check": name, "ok": ok,
                   "got": got, "want": want})
    return ok


def case(pid, m0, ed):
    return {"pair_id": pid, "m0_derived_target": m0,
            "edited_derived_target": ed}


def main():
    # ---- presence derivation ------------------------------------------------
    chk("both present -> present",
        derive_presence(["present", "present"]), "present")
    chk("both absent -> absent",
        derive_presence(["absent", "absent"]), "absent")
    chk("present/absent disagreement -> unresolved",
        derive_presence(["present", "absent"]), UNRESOLVED)
    chk("BOTH UNCERTAIN -> unresolved (never a presence label)",
        derive_presence(["uncertain", "uncertain"]), UNRESOLVED)
    chk("present + uncertain -> unresolved",
        derive_presence(["present", "uncertain"]), UNRESOLVED)
    chk("absent + uncertain -> unresolved",
        derive_presence(["absent", "uncertain"]), UNRESOLVED)
    chk("only one vote -> unresolved",
        derive_presence(["present"]), UNRESOLVED)

    # ---- agreement classification ------------------------------------------
    chk("uncertain/uncertain is agreement but NOT definite consensus",
        presence_agreement(["uncertain", "uncertain"]), "uncertain_agreement")
    chk("present/present is definite consensus",
        presence_agreement(["present", "present"]), "definite_consensus")
    chk("present/absent is disagreement",
        presence_agreement(["present", "absent"]), "disagreement")
    chk("definite coverage counts only present/absent",
        definite_presence_coverage(["present", "uncertain"]), 1)
    chk("definite coverage 0 when both uncertain",
        definite_presence_coverage(["uncertain", "uncertain"]), 0)
    chk("definite coverage 2 when both definite",
        definite_presence_coverage(["absent", "present"]), 2)

    # ---- rendering derivation ----------------------------------------------
    chk("AGREED unable_to_judge derives unable_to_judge, not unresolved",
        derive_rendering(["unable_to_judge", "unable_to_judge"]),
        "unable_to_judge")
    chk("unable_to_judge vs coherent -> unresolved",
        derive_rendering(["unable_to_judge", "coherent"]), UNRESOLVED)
    chk("both whole_frame_breakdown -> whole_frame_breakdown",
        derive_rendering(["whole_frame_breakdown", "whole_frame_breakdown"]),
        "whole_frame_breakdown")
    chk("localized vs coherent -> unresolved",
        derive_rendering(["localized_deformation", "coherent"]), UNRESOLVED)
    chk("rendering distribution keeps unable_to_judge and unresolved apart",
        rendering_distribution(["unable_to_judge", UNRESOLVED, UNRESOLVED]),
        {"coherent": 0, "localized_deformation": 0,
         "whole_frame_breakdown": 0, "unable_to_judge": 1, UNRESOLVED: 2})

    # ---- 3x3 table ----------------------------------------------------------
    cs = [case("B-01", "present", "absent"),
          case("B-02", "present", UNRESOLVED),
          case("B-03", UNRESOLVED, UNRESOLVED),
          case("B-04", "absent", "absent")]
    t = presence_table(cs)
    chk("3x3 table has nine cells", sum(len(v) for v in t["table"].values()), 9)
    chk("3x3 table total equals case count", t["total"], 4)
    chk("present->absent cell", t["table"]["present"]["absent"], 1)
    chk("present->unresolved cell", t["table"]["present"][UNRESOLVED], 1)
    chk("unresolved->unresolved cell", t["table"][UNRESOLVED][UNRESOLVED], 1)
    chk("zero cell is still reported",
        t["table"]["absent"]["present"], 0)

    # ---- removal count ------------------------------------------------------
    r = demonstrated_removal(cs)
    chk("removal numerator", r["numerator"], 1)
    chk("EDITED-UNRESOLVED STAYS IN THE DENOMINATOR", r["denominator"], 2)
    chk("removal is a count with a denominator", r["value_text"], "1 / 2")
    chk("denominator breakdown retains the unresolved edited case",
        r["denominator_breakdown_by_edited_status"][UNRESOLVED], 1)

    # absent -> absent must NOT count as demonstrated removal
    aa = [case("B-01", "absent", "absent"), case("B-02", "absent", "absent")]
    ra = demonstrated_removal(aa)
    chk("ZERO m0-present denominator -> not estimable",
        ra["value_text"], "not estimable (denominator 0)")
    chk("zero denominator flagged not estimable", ra["estimable"], False)
    chk("absent->absent contributes nothing to the numerator",
        ra["numerator"], 0)

    # ---- Task B -------------------------------------------------------------
    b = comparison_distribution(
        ["less", "about_the_same", "more", "cannot_fairly_compare",
         "cannot_fairly_compare"])
    chk("CANNOT_FAIRLY_COMPARE IS ITS OWN CATEGORY",
        b["cannot_fairly_compare"], 2)
    chk("B distribution keeps all four categories", sorted(b),
        ["about_the_same", "cannot_fairly_compare", "less", "more"])
    chk("B distribution sums to the number of answers", sum(b.values()), 5)
    chk("absent B category reported as zero, not omitted",
        comparison_distribution(["less"])["cannot_fairly_compare"], 0)

    nfail = [c for c in CHECKS if not c["ok"]]
    for c in CHECKS:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"]
              + ("" if c["ok"] else "  got=%r want=%r" % (c["got"], c["want"])))
    print("\n%d/%d synthetic checks pass" % (len(CHECKS) - len(nfail),
                                             len(CHECKS)))
    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - len(nfail),
           "all_pass": not nfail,
           "note": "Synthetic data only. No real response is read by this suite.",
           "checks": [{"check": c["check"], "ok": c["ok"]} for c in CHECKS]}
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "analysis_v1", "synthetic_checks.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
