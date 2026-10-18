#!/usr/bin/env python3
"""CPU-only synthetic regression tests for the companion validator (r2).

Every fixture is machine-generated and clearly synthetic. None is a human
judgement, an image assessment or a reviewer response, and none may be used as
such. No browser, no model, no GPU, no network.

This exercises FILE STRUCTURE handling only. It is not browser validation and
not scientific validation; the questionnaire itself is unchanged, so the
existing 150/150 browser evidence stands as the host-performed browser record.

Usage: test_validator_r2.py [--before OLD_VALIDATOR.py] [--out REPORT.json]
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
VALIDATOR = os.path.join(ROOT, "scripts", "validate_export.py")
ITEMS = os.path.join(ROOT, "protocol", "taskA_items.json")
SYNTH = ("SYNTHETIC TEST FIXTURE -- machine-generated, not a human judgement, "
         "not a reviewer response, not an image assessment.")

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          f"{'' if ok else '  <- ' + str(detail)[:160]}")
    return bool(ok)


def run(validator, files, stage="A", cohort=False):
    cmd = [sys.executable, validator]
    if cohort:
        cmd.append("--cohort")
    cmd += files
    if not cohort:
        cmd += ["--expect-stage", stage]
    cmd += ["--items", ITEMS]
    p = subprocess.run(cmd, capture_output=True, text=True)
    crashed = "Traceback (most recent call last)" in p.stderr
    try:
        out = json.loads(p.stdout)
    except Exception:
        out = None
    return {"crashed": crashed, "json": out, "stderr": p.stderr[-400:],
            "returncode": p.returncode}


def ids():
    return json.load(open(ITEMS))["item_ids"]


def base(kind="draft", n=None, stage="A", alias="synthetic-r2"):
    I = ids()
    exp = {"declared": True,
           "prior_access_to_source_pairings": False,
           "prior_access_to_condition_labels": False,
           "image_specific_coaching_before_task_a": False,
           "other_exposure_note": ""}
    T = ["present", "absent", "uncertain"]
    R = ["coherent", "localized_deformation",
         "whole_frame_breakdown", "unable_to_judge"]
    rs = []
    for i, x in enumerate(I):
        if n is not None and i >= n:
            rs.append({"item_id": x, "target": None, "rendering": None, "reason": ""})
        else:
            rs.append({"item_id": x, "target": T[i % 3],
                       "rendering": R[i % 4], "reason": ""})
    return {"_fixture_warning": SYNTH,
            "package_id": "THR-V1-A", "protocol_id": "TEAPOT-HR-PROTO-1.0.0",
            "schema_id": "THR-EXPORT-SCHEMA-1.1.0", "app_version": "1.1.0",
            "stage": stage, "export_kind": kind, "reviewer_alias": alias,
            "order_variant": "V1", "exposure": exp,
            "exported_iso": "2026-10-18T00:00:00Z",
            "item_count_expected": len(I), "responses": rs,
            "completion": {"answered_items": 999, "unanswered_items": 0,
                           "self_reported": True}}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=os.path.join(ROOT, "maint",
                                                     "validate_export.before.py"))
    ap.add_argument("--out", default=os.path.join(ROOT, "maint",
                                                  "validator_r2_report.json"))
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="valr2_", dir=os.path.join(ROOT, "maint"))
    I = ids()

    def W(name, obj):
        p = os.path.join(tmp, name)
        with open(p, "w") as fh:
            fh.write(obj) if isinstance(obj, str) else json.dump(obj, fh, indent=1)
        return p

    # ---------------- A. valid fixtures must stay accepted ----------------
    valid = {
        "valid_draft_partial.json": base("draft", n=7),
        "valid_final_complete.json": base("final"),
    }
    d = base("final")
    for r in d["responses"]:
        r["target"], r["rendering"] = "uncertain", "unable_to_judge"
    valid["valid_final_all_uncertain.json"] = d
    valid_paths = {k: W(k, v) for k, v in valid.items()}

    for k, p in valid_paths.items():
        r = run(VALIDATOR, [p])
        ok = (not r["crashed"] and r["json"]
              and r["json"]["results"][0]["valid"] is True)
        check(f"valid fixture still accepted: {k}", ok,
              (r["json"]["results"][0]["errors"] if r["json"] else r["stderr"]))

    check("uncertain/unable_to_judge counts as a COMPLETE final",
          run(VALIDATOR, [valid_paths["valid_final_all_uncertain.json"]])
          ["json"]["results"][0]["recomputed_complete_answers"] == len(I))

    # supplied counts never trusted
    rr = run(VALIDATOR, [valid_paths["valid_final_complete.json"]])["json"]["results"][0]
    check("supplied completion count is not trusted",
          rr["supplied_answered_items"] == 999
          and rr["recomputed_complete_answers"] == len(I)
          and rr["supplied_count_trusted"] is False)

    # ---------------- B. before/after acceptance parity ----------------
    parity = {}
    if os.path.isfile(a.before):
        for k, p in valid_paths.items():
            old = run(a.before, [p])
            new = run(VALIDATOR, [p])
            o = old["json"]["results"][0] if old["json"] else {}
            n = new["json"]["results"][0] if new["json"] else {}
            same = (o.get("valid") == n.get("valid")
                    and o.get("recomputed_complete_answers")
                    == n.get("recomputed_complete_answers")
                    and sorted(o.get("recomputed_missing_ids") or [])
                    == sorted(n.get("recomputed_missing_ids") or []))
            parity[k] = {"old_valid": o.get("valid"), "new_valid": n.get("valid"),
                         "identical_acceptance": same}
            check(f"acceptance semantics unchanged before/after: {k}", same, parity[k])
    else:
        check("pre-repair validator available for parity comparison", False,
              "missing " + a.before)

    # ---------------- C. known crash classes ----------------
    crashers = {}
    d = base(); d["schema_id"] = []
    crashers["schema_id=[]"] = d
    d = base(); d["completion"] = "bad"
    crashers["completion='bad'"] = d
    d = base(); d["exposure"]["other_exposure_note"] = 7
    crashers["exposure.other_exposure_note=7"] = d
    d = base(); d["responses"][0]["item_id"] = []
    crashers["response.item_id=[]"] = d
    d = base(); d["responses"][0]["target"] = []
    crashers["response.target=[]"] = d
    d = base(); d["export_kind"] = []
    crashers["export_kind=[]"] = d
    d = base(); d["responses"][0]["rendering"] = {}
    crashers["response.rendering={}"] = d
    d = base(); d["reviewer_alias"] = []
    crashers["reviewer_alias=[]"] = d
    d = base(); d["responses"].append(
        {"item_id": {"x": 1}, "target": "present", "rendering": "coherent"})
    crashers["mixed unknown-ID types"] = d
    d = base(stage="B"); d["responses"] = [
        {"pair_id": [], "comparison": "less"}]
    crashers["B: pair_id=[]"] = d
    d = base(stage="B"); d["responses"] = [
        {"pair_id": "B-01", "comparison": {}}]
    crashers["B: comparison={}"] = d
    d = base(); d["item_count_expected"] = "forty"
    crashers["item_count_expected='forty'"] = d
    d = base(); d["exposure"] = "not an object"
    crashers["exposure not an object"] = d
    d = base(); d["responses"] = "not a list"
    crashers["responses not a list"] = d
    d = base(); d["responses"][0]["reason"] = []
    crashers["reason=[]"] = d

    for label, obj in crashers.items():
        stage = "B" if label.startswith("B:") else "A"
        p = W("crash_" + str(abs(hash(label))) + ".json", obj)
        r = run(VALIDATOR, [p], stage=stage)
        ok = (not r["crashed"] and r["json"]
              and r["json"]["results"][0]["valid"] is False
              and r["json"]["results"][0]["errors"])
        check(f"structured invalid, no traceback: {label}", ok,
              r["stderr"] if r["crashed"] else "no errors recorded")

    # ---------------- D. malformed / non-object input ----------------
    for label, body in (("malformed JSON", "{ nope ,,,"),
                        ("top-level list", "[1,2,3]"),
                        ("top-level string", '"hello"'),
                        ("empty file", "")):
        p = W("mal_" + str(abs(hash(label))) + ".json", body)
        r = run(VALIDATOR, [p])
        ok = (not r["crashed"] and r["json"]
              and r["json"]["results"][0]["valid"] is False)
        check(f"handled without crashing: {label}", ok, r["stderr"])

    # ---------------- E. frozen rule set still enforced ----------------
    rules = {}
    d = base("final", n=39); rules["partial final rejected"] = ("A", d)
    d = base("final"); d["responses"] = []
    rules["empty final rejected"] = ("A", d)
    d = base(); d["responses"].append(dict(d["responses"][0]))
    rules["duplicate ID rejected"] = ("A", d)
    d = base(); d["responses"].append(
        {"item_id": "A-999", "target": "present", "rendering": "coherent"})
    rules["unknown ID rejected"] = ("A", d)
    d = base(); d["responses"][1]["target"] = "definitely"
    rules["invalid enum rejected"] = ("A", d)
    d = base(); del d["exposure"]["prior_access_to_source_pairings"]
    rules["missing declaration rejected"] = ("A", d)
    d = base(); d["exposure"]["prior_access_to_condition_labels"] = "no"
    rules["non-boolean declaration rejected"] = ("A", d)
    d = base(); d["package_id"] = "THR-V1-B"
    rules["wrong package rejected"] = ("A", d)
    d = base(); d["protocol_id"] = "OTHER"
    rules["wrong protocol rejected"] = ("A", d)
    d = base(stage="B")
    rules["wrong stage rejected"] = ("A", d)
    d = base(); d["schema_id"] = "THR-EXPORT-SCHEMA-1.0.0"
    rules["legacy schema rejected, not migrated"] = ("A", d)

    for label, (stage, obj) in rules.items():
        p = W("rule_" + str(abs(hash(label))) + ".json", obj)
        r = run(VALIDATOR, [p], stage=stage)
        ok = (not r["crashed"] and r["json"]
              and r["json"]["results"][0]["valid"] is False)
        check(label, ok, r["stderr"] if r["crashed"] else "unexpectedly valid")

    rleg = run(VALIDATOR, [W("legacy.json", rules[
        "legacy schema rejected, not migrated"][1])])
    errs = " ".join(rleg["json"]["results"][0]["errors"])
    check("legacy rejection says 'unsupported legacy', never relabelled",
          "unsupported legacy" in errs and "does not migrate" in errs, errs[:120])

    # ---------------- F. mixed batch + cohort ----------------
    batch = [valid_paths["valid_final_complete.json"],
             W("batch_bad1.json", {"schema_id": [], "stage": "A"}),
             valid_paths["valid_draft_partial.json"],
             W("batch_bad2.json", "{ broken"),
             W("batch_bad3.json", crashers["mixed unknown-ID types"])]
    rb = run(VALIDATOR, batch)
    ok = (not rb["crashed"] and rb["json"]
          and len(rb["json"]["results"]) == len(batch))
    check("mixed batch: every file receives a result, no traceback", ok,
          rb["stderr"] if rb["crashed"] else
          (len(rb["json"]["results"]) if rb["json"] else "no json"))
    if ok:
        check("mixed batch: failures reported, valid files still accepted",
              rb["json"]["n_valid"] == 2 and rb["json"]["n_invalid"] == 3,
              f"valid={rb['json']['n_valid']} invalid={rb['json']['n_invalid']}")

    coh_files = [valid_paths["valid_final_complete.json"],
                 W("coh_dup.json", base("final")),
                 W("coh_badalias.json", crashers["reviewer_alias=[]"]),
                 W("coh_broken.json", "{ nope")]
    rc = run(VALIDATOR, coh_files, cohort=True)
    ok = (not rc["crashed"] and rc["json"]
          and len(rc["json"]["entries"]) == len(coh_files))
    check("cohort: all files processed, no traceback on bad alias/JSON", ok,
          rc["stderr"] if rc["crashed"] else "entry count mismatch")
    if ok:
        flags = rc["json"]["flags"]
        check("cohort flags identical content / repeated alias",
              any(f["kind"] == "identical_file_content" for f in flags)
              and any(f["kind"] == "repeated_alias" for f in flags), flags)
        check("cohort states it cannot prove distinct humans",
              "does not prove distinct humans" in rc["json"]["independence_note"])

    # ---------------- G. inputs never modified ----------------
    allp = sorted(os.path.join(tmp, f) for f in os.listdir(tmp)
                  if f.endswith(".json"))
    before = {p: sha(p) for p in allp}
    run(VALIDATOR, allp)
    run(VALIDATOR, allp, cohort=True)
    after = {p: sha(p) for p in allp}
    check("validator never modifies an input file", before == after,
          [os.path.basename(k) for k in before if before[k] != after[k]])

    n_pass = sum(1 for r in results if r["pass"])
    rep = {
        "suite": "companion Python validator regression (r2)",
        "scope": ("FILE STRUCTURE handling only. CPU-only. Not browser "
                  "validation and not scientific validation. The questionnaire "
                  "is unchanged, so the existing 150/150 browser evidence "
                  "remains the host-performed browser record."),
        "fixtures_are_synthetic": True,
        "fixture_warning": SYNTH,
        "contains_human_judgements": False,
        "contains_reviewer_responses": False,
        "real_participant_files_used": 0,
        "browser_used": False, "gpu_seconds": 0.0, "model_work": 0,
        "validator_revision": "validator-r2",
        "acceptance_parity_vs_pre_repair": parity,
        "n_checks": len(results), "n_pass": n_pass,
        "n_fail": len(results) - n_pass,
        "all_pass": n_pass == len(results),
        "checks": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{n_pass}/{len(results)} checks pass  ->  {a.out}")
    sys.exit(0 if rep["all_pass"] else 1)


if __name__ == "__main__":
    main()
