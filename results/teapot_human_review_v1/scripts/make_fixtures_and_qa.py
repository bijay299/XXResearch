#!/usr/bin/env python3
"""Build explicitly synthetic QA fixtures and run the mechanical validator suite.

EVERY fixture is synthetic and machine-generated. No fixture contains, claims to
contain, or may be used as, a human judgement about any image. Fixture choices
are filled from a fixed rotation, not from looking at images. Fixtures live
under qa/fixtures/ and must never be placed in a reviewer bundle or a
real-response directory.
"""
import json
import os
import subprocess
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
FX = os.path.join(ROOT, "qa", "fixtures")
VAL = os.path.join(ROOT, "scripts", "validate_export.py")

PROTOCOL_ID = "TEAPOT-HR-PROTO-1.0.0"
SCHEMA_ID = "THR-EXPORT-SCHEMA-1.0.0"
SYNTH = ("SYNTHETIC QA FIXTURE -- machine-generated, not a human judgement, "
         "not reviewer evidence. Choices follow a fixed rotation and encode no "
         "assessment of any image.")

TARGETS = ["present", "absent", "uncertain"]
RENDERS = ["coherent", "localized_deformation", "whole_frame_breakdown", "unable_to_judge"]
CMPS = ["less", "about_the_same", "more", "cannot_fairly_compare"]


def a_ids():
    return list(json.load(open(os.path.join(ROOT, "protocol", "taskA_items.json")))["item_ids"])


def b_ids():
    p = os.path.join(ROOT, "private", "taskB_package", "items.json")
    return list(json.load(open(p))["item_ids"])


def base(stage, kind, alias):
    pkg = "THR-V1-A" if stage == "A" else "THR-V1-B"
    exp = ({"declared": True,
            "prior_access_to_source_pairings": False,
            "prior_access_to_condition_labels": False,
            "image_specific_coaching_before_task_a": False,
            "other_exposure_note": ""} if stage == "A" else
           {"declared": True,
            "task_a_final_exported_before_task_b": True,
            "prior_access_to_condition_labels": False,
            "image_specific_coaching_before_task_a": False,
            "reference_identity_disclosed_in_this_stage": True,
            "other_exposure_note": ""})
    return {"_fixture_warning": SYNTH, "package_id": pkg, "protocol_id": PROTOCOL_ID,
            "schema_id": SCHEMA_ID, "stage": stage, "export_kind": kind,
            "reviewer_alias": alias, "order_variant": "V1", "exposure": exp,
            "exported_iso": "2026-10-18T00:00:00Z"}


def a_resp(ids, n=None):
    out = []
    for i, x in enumerate(ids if n is None else ids[:n]):
        out.append({"item_id": x, "target": TARGETS[i % 3],
                    "rendering": RENDERS[i % 4], "reason": ""})
    return out


def b_resp(ids, n=None):
    return [{"pair_id": x, "comparison": CMPS[i % 4], "reason": ""}
            for i, x in enumerate(ids if n is None else ids[:n])]


def write(name, obj):
    p = os.path.join(FX, name)
    if isinstance(obj, str):
        open(p, "w").write(obj)
    else:
        json.dump(obj, open(p, "w"), indent=1)
    return p


def main():
    os.makedirs(FX, exist_ok=True)
    A, B = a_ids(), b_ids()
    cases = []

    def case(name, obj, stage, expect_valid, why):
        p = write(name, obj)
        cases.append({"fixture": name, "stage": stage,
                      "expect_valid": expect_valid, "why": why, "path": p})

    # ---- should PASS ----
    d = base("A", "final", "synthetic-alpha"); d["responses"] = a_resp(A)
    d["completion"] = {"answered_items": 40, "unanswered_items": 0, "self_reported": True}
    case("A_final_valid.json", d, "A", True, "complete 40/40 final")

    d = base("A", "final", "synthetic-beta")
    d["responses"] = [{"item_id": x, "target": "uncertain",
                       "rendering": "unable_to_judge", "reason": ""} for x in A]
    case("A_final_all_uncertain.json", d, "A", True,
         "uncertain / unable_to_judge are valid COMPLETED answers")

    d = base("A", "draft", "synthetic-gamma"); d["responses"] = a_resp(A, 7)
    case("A_draft_partial.json", d, "A", True, "drafts may be partial")

    d = base("B", "final", "synthetic-alpha"); d["responses"] = b_resp(B)
    case("B_final_valid.json", d, "B", True, "complete 20/20 final")

    # ---- should FAIL ----
    d = base("A", "final", "synthetic-delta"); d["responses"] = a_resp(A, 39)
    d["completion"] = {"answered_items": 40, "unanswered_items": 0, "self_reported": True}
    case("A_final_missing_one.json", d, "A", False,
         "partial final with an inflated self-reported count")

    d = base("A", "final", "synthetic-eps"); d["responses"] = []
    case("A_final_empty.json", d, "A", False, "empty final must fail")

    d = base("A", "final", "synthetic-zeta"); r = a_resp(A); r.append(dict(r[0]))
    d["responses"] = r
    case("A_final_duplicate_id.json", d, "A", False, "duplicate ID")

    d = base("A", "final", "synthetic-eta"); r = a_resp(A)
    r.append({"item_id": "A-999", "target": "present", "rendering": "coherent"})
    d["responses"] = r
    case("A_final_extra_id.json", d, "A", False, "unexpected/extra ID")

    d = base("A", "final", "synthetic-theta"); r = a_resp(A)
    r[3]["target"] = "definitely_yes"
    d["responses"] = r
    case("A_final_bad_enum.json", d, "A", False, "invalid enum value")

    d = base("A", "final", "synthetic-iota"); r = a_resp(A)
    r[5]["rendering"] = None
    d["responses"] = r
    case("A_final_missing_answer.json", d, "A", False, "one answer missing")

    d = base("A", "final", "synthetic-kappa"); d["responses"] = a_resp(A)
    del d["exposure"]["prior_access_to_source_pairings"]
    case("A_final_missing_exposure.json", d, "A", False,
         "required exposure declaration absent")

    d = base("A", "final", "synthetic-lambda"); d["responses"] = a_resp(A)
    d["exposure"]["prior_access_to_source_pairings"] = "no"
    case("A_final_nonbool_exposure.json", d, "A", False,
         "exposure declaration not an explicit boolean")

    d = base("A", "final", "synthetic-mu"); d["responses"] = a_resp(A)
    d["package_id"] = "SOME-OTHER-PACKAGE"
    case("A_final_wrong_package.json", d, "A", False, "untrusted package_id")

    d = base("B", "final", "synthetic-nu"); d["responses"] = b_resp(B)
    d["stage"] = "A"
    case("B_final_wrong_stage.json", d, "A", False, "stage/content mismatch")

    d = base("A", "final", "synthetic-xi"); d["responses"] = a_resp(A)
    d["schema_id"] = "SOME-OLD-SCHEMA-0.1"
    case("A_final_wrong_schema.json", d, "A", False, "untrusted schema_id")

    d = base("B", "final", "synthetic-omicron"); d["responses"] = b_resp(B, 19)
    case("B_final_missing_one.json", d, "B", False, "partial B final")

    write("A_malformed.json", "{ this is not valid json ,,, ")
    cases.append({"fixture": "A_malformed.json", "stage": "A", "expect_valid": False,
                  "why": "malformed JSON",
                  "path": os.path.join(FX, "A_malformed.json")})

    # ---- run the validator ----
    results = []
    for c in cases:
        out = subprocess.run([sys.executable, VAL, c["path"],
                              "--expect-stage", c["stage"]],
                             capture_output=True, text=True)
        try:
            rep = json.loads(out.stdout)
            got = rep["results"][0]["valid"]
            errs = rep["results"][0]["errors"]
        except Exception:
            got, errs = None, ["validator did not return JSON"]
        ok = (got == c["expect_valid"])
        results.append({"fixture": c["fixture"], "expect_valid": c["expect_valid"],
                        "got_valid": got, "pass": ok, "why": c["why"],
                        "validator_errors": errs})

    # ---- cohort / duplicate audit ----
    dupdir = os.path.join(FX, "dup")
    os.makedirs(dupdir, exist_ok=True)
    src = os.path.join(FX, "A_final_valid.json")
    import shutil
    shutil.copyfile(src, os.path.join(dupdir, "copy_one.json"))
    shutil.copyfile(src, os.path.join(dupdir, "copy_two.json"))
    out = subprocess.run([sys.executable, VAL, "--cohort",
                          os.path.join(dupdir, "copy_one.json"),
                          os.path.join(dupdir, "copy_two.json")],
                         capture_output=True, text=True)
    coh = json.loads(out.stdout)
    coh_ok = any(f["kind"] == "identical_file_content" for f in coh["flags"]) and \
             any(f["kind"] == "repeated_alias" for f in coh["flags"]) and \
             coh["distinct_file_hashes"] == 1
    results.append({"fixture": "cohort_duplicate_audit",
                    "expect_valid": "flagged", "got_valid": "flagged" if coh_ok else "not flagged",
                    "pass": coh_ok,
                    "why": "two identical files / repeated alias must NOT count as two reviewers",
                    "validator_errors": []})

    n_pass = sum(1 for r in results if r["pass"])
    report = {
        "suite": "mechanical validator QA",
        "schema_id": SCHEMA_ID, "protocol_id": PROTOCOL_ID,
        "fixtures_are_synthetic": True,
        "fixture_warning": SYNTH,
        "contains_human_judgements": False,
        "contains_answer_keys": False,
        "n_cases": len(results), "n_pass": n_pass,
        "n_fail": len(results) - n_pass,
        "all_pass": n_pass == len(results),
        "cases": results,
    }
    json.dump(report, open(os.path.join(ROOT, "qa", "validator_qa_report.json"), "w"),
              indent=1)
    for r in results:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['fixture']:34} "
              f"expect={r['expect_valid']} got={r['got_valid']}")
    print(f"\n{n_pass}/{len(results)} mechanical cases pass")
    sys.exit(0 if report["all_pass"] else 3)


if __name__ == "__main__":
    main()
