#!/usr/bin/env python3
"""Arithmetic, join, input-integrity and reproducibility checks on analysis_v1.

Re-runs the analysis into a throwaway directory and compares: every public
table byte-for-byte, and summary.json with only the wall-clock field removed.
Then asserts the table totals and join properties the contract requires.
"""
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
OUT = os.path.join(ROOT, "analysis_v1")
TAB = os.path.join(OUT, "tables")
SCRATCH = os.path.join(ROOT, "private", "analysis_work", "repro")

CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    return ok


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    s = json.load(open(os.path.join(OUT, "summary.json")))

    # ---- input integrity ---------------------------------------------------
    ig = s["integrity"]
    chk("input SHA256 identical before and after analysis",
        ig["input_sha256_before"] == ig["input_sha256_after"]
        and ig["inputs_unchanged_by_analysis"])
    chk("exactly four allowlisted inputs", len(ig["input_sha256_before"]) == 4,
        len(ig["input_sha256_before"]))

    # ---- Task-A table ------------------------------------------------------
    a = rows(os.path.join(TAB, "taskA_responses.csv"))
    chk("Task-A table has 40 rows", len(a) == 40, len(a))
    chk("Task-A item IDs are the 40 public IDs, each once",
        sorted(r["item_id"] for r in a) == ["A-%03d" % i for i in range(1, 41)])
    dt = {}
    for r in a:
        dt[r["derived_target"]] = dt.get(r["derived_target"], 0) + 1
    chk("derived Task-A presence labels sum to 40", sum(dt.values()) == 40, dt)
    chk("derived presence matches summary",
        dt == {k: v for k, v in
               s["taskA"]["derived_presence_counts_over_40_images"].items()
               if v}, dt)
    cov = s["taskA"]["definite_target_vote_coverage_per_image"]
    chk("definite-vote coverage sums to 40", sum(cov.values()) == 40, cov)
    pa = s["taskA"]["presence_agreement"]["all_40_images"]
    chk("presence agreement classes sum to 40",
        pa["definite_consensus"] + pa["uncertain_agreement"]
        + pa["disagreement"] == 40, pa)
    chk("m0 + edited presence denominators sum to 40",
        s["taskA"]["presence_agreement"]["m0_images"]["n"]
        + s["taskA"]["presence_agreement"]["edited_images"]["n"] == 40)
    chk("rendering agreement classes sum to 40",
        s["taskA"]["rendering_agreement"]["all_40_images"]["agreement"]
        + s["taskA"]["rendering_agreement"]["all_40_images"]["disagreement"]
        == 40)
    for rid, d in s["taskA"]["raw_presence_per_reviewer"].items():
        chk("raw Task-A presence votes sum to 40 for " + rid,
            sum(d.values()) == 40, d)
    for rid, d in s["taskA"]["raw_rendering_per_reviewer"].items():
        chk("raw Task-A rendering votes sum to 40 for " + rid,
            sum(d.values()) == 40, d)

    # ---- 3x3 table ---------------------------------------------------------
    pt = s["presence_table_m0_by_edited"]
    t, ax = pt["table"], pt["axes"]
    chk("3x3 presence table has nine cells",
        sum(len(t[m]) for m in ax) == 9)
    chk("3x3 presence table sums to 20",
        sum(t[m][e] for m in ax for e in ax) == 20, pt["total"])
    pcsv = rows(os.path.join(TAB, "presence_table_m0_by_edited.csv"))
    chk("published 3x3 CSV row totals agree with the cells",
        all(int(r["row_total"]) == sum(int(r[e]) for e in ax)
            for r in pcsv if r["m0_derived \\ edited_derived"] in ax))
    chk("published 3x3 CSV grand total is 20",
        any(r["m0_derived \\ edited_derived"] == "column_total"
            and int(r["row_total"]) == 20 for r in pcsv))

    # ---- paired ledger and the removal count -------------------------------
    led = rows(os.path.join(TAB, "paired_case_ledger.csv"))
    chk("paired ledger has 20 rows", len(led) == 20, len(led))
    chk("paired ledger covers B-01..B-20 exactly once",
        sorted(r["pair_id"] for r in led) == ["B-%02d" % i
                                              for i in range(1, 21)])
    rm = s["demonstrated_removal"]
    denom_rows = [r for r in led if r["counts_in_removal_denominator"] == "yes"]
    num_rows = [r for r in led if r["counts_in_removal_numerator"] == "yes"]
    chk("ledger denominator flags match the reported denominator",
        len(denom_rows) == rm["denominator"], len(denom_rows))
    chk("ledger numerator flags match the reported numerator",
        len(num_rows) == rm["numerator"], len(num_rows))
    chk("numerator is a subset of the denominator",
        set(r["pair_id"] for r in num_rows)
        <= set(r["pair_id"] for r in denom_rows))
    chk("every denominator row has m0 present",
        all(r["m0_derived_target"] == "present" for r in denom_rows))
    chk("every numerator row is m0 present -> edited absent",
        all(r["m0_derived_target"] == "present"
            and r["edited_derived_target"] == "absent" for r in num_rows))
    chk("NO absent->absent case is counted as demonstrated removal",
        not any(r["m0_derived_target"] == "absent"
                and r["counts_in_removal_numerator"] == "yes" for r in led))
    chk("denominator RETAINS cases whose edited status is unresolved",
        rm["denominator_breakdown_by_edited_status"].get("unresolved", 0)
        == sum(1 for r in denom_rows
               if r["edited_derived_target"] == "unresolved"))
    chk("denominator breakdown sums to the denominator",
        sum(rm["denominator_breakdown_by_edited_status"].values())
        == rm["denominator"])
    chk("removal is reported as a count with a denominator, not a bare rate",
        "/" in rm["value_text"] or "not estimable" in rm["value_text"],
        rm["value_text"])
    chk("removal numerator equals the 3x3 present->absent cell",
        rm["numerator"] == t["present"]["absent"])

    # ---- absence rendering, reported twice ---------------------------------
    ar = s["target_absence_rendering"]
    n_abs = sum(1 for r in led if r["edited_derived_target"] == "absent")
    chk("edited-absent subset size matches the ledger",
        ar["all_edited_absent_cases"]["n"] == n_abs, n_abs)
    for k, v in ar.items():
        chk("absence-rendering counts sum to the subset size for " + k,
            sum(v["edited_image_rendering"].values()) == v["n"])
        chk("absence-rendering reports unable_to_judge separately from "
            "unresolved for " + k,
            "unable_to_judge" in v["edited_image_rendering"]
            and "unresolved" in v["edited_image_rendering"])
    chk("qualified subset is no larger than the edited-absent subset",
        ar["qualified_m0_present_to_edited_absent"]["n"]
        <= ar["all_edited_absent_cases"]["n"])

    # ---- Task B ------------------------------------------------------------
    b = rows(os.path.join(TAB, "taskB_responses.csv"))
    chk("Task-B table has 20 rows", len(b) == 20, len(b))
    for rid, d in s["taskB"]["distributions"].items():
        chk("Task-B distribution sums to 20 for " + rid,
            sum(d.values()) == 20, d)
        chk("all four Task-B categories reported for " + rid,
            sorted(d) == ["about_the_same", "cannot_fairly_compare", "less",
                          "more"])
    bd = rows(os.path.join(TAB, "taskB_distributions.csv"))
    chk("published Task-B distribution CSV totals are 20",
        all(int(r["total"]) == 20 for r in bd))
    chk("published Task-B CSV totals equal the sum of the four categories",
        all(int(r["total"]) == sum(int(r[c]) for c in
            ("less", "about_the_same", "more", "cannot_fairly_compare"))
            for r in bd))
    agree = sum(1 for r in b if r["comparison_agreement"] == "agreement")
    chk("Task-B exact agreement count matches the table",
        agree == s["taskB"]["exact_per_pair_agreement"]["agreeing_pairs"],
        agree)
    chk("ledger and Task-B table give the same per-pair votes",
        {r["pair_id"]: (r["R1_comparison"], r["R2_comparison"]) for r in b}
        == {r["pair_id"]: (r["R1_comparison"], r["R2_comparison"])
            for r in led})
    chk("no Task-B consensus label is published",
        not any("consensus" in c.lower() for c in b[0].keys()))

    # ---- strata ------------------------------------------------------------
    st = rows(os.path.join(TAB, "descriptive_strata.csv"))
    for kind in ("by_class", "by_category"):
        n = sum(int(r["n_cases"]) for r in st if r["stratum_kind"] == kind)
        chk("%s strata case counts sum to 20" % kind, n == 20, n)
    chk("strata numerators never exceed their denominators",
        all(int(r["removal_numerator"]) <= int(r["removal_denominator"])
            for r in st))

    # ---- reproducibility ---------------------------------------------------
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(SCRATCH)
    before = {f: sha(os.path.join(TAB, f)) for f in sorted(os.listdir(TAB))}
    shutil.copytree(TAB, os.path.join(SCRATCH, "tables"))
    shutil.copyfile(os.path.join(OUT, "summary.json"),
                    os.path.join(SCRATCH, "summary.json"))
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "scripts", "analyze_v1.py"),
                        "--write"], capture_output=True, text=True)
    chk("second analysis run exits cleanly", r.returncode == 0,
        r.returncode)
    after = {f: sha(os.path.join(TAB, f)) for f in sorted(os.listdir(TAB))}
    chk("all public tables are byte-identical across two runs",
        before == after,
        ",".join(sorted(k for k in set(before) | set(after)
                        if before.get(k) != after.get(k))) or "no diff")
    s2 = json.load(open(os.path.join(OUT, "summary.json")))
    a1, a2 = dict(s), dict(s2)
    a1.pop("generated_iso", None)
    a2.pop("generated_iso", None)
    chk("summary.json is identical across two runs apart from its timestamp",
        json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True))
    shutil.rmtree(SCRATCH)

    # ---- statistics exclusions actually honoured ---------------------------
    # Scan the reported content only. The `statistics_excluded` list and its
    # note name these terms on purpose, to say they are NOT reported, so they
    # are excluded from the scan rather than scrubbed out of a flat string.
    scanned = {k: v for k, v in s.items()
               if k not in ("statistics_excluded", "statistics_note")}
    blob = json.dumps(scanned).lower()
    banned = ["p_value", "p-value", "confidence_interval", "confidence "
              "interval", "kappa", "bootstrap", "significant", "significance",
              "state of the art", "sota"]
    found = [w for w in banned if w in blob]
    chk("no test statistic, interval, kappa or significance claim is reported",
        not found, ",".join(found) or "none")
    chk("the statistics exclusion list is still published",
        len(s["statistics_excluded"]) >= 8 and bool(s["statistics_note"]))

    ok = all(c["ok"] for c in CHECKS)
    rep = {
        "generated_iso": datetime.datetime.now().astimezone().isoformat(),
        "what_this_is": ("Host-performed arithmetic, join, input-integrity and "
                         "reproducibility verification of analysis_v1."),
        "n_checks": len(CHECKS), "n_passed": sum(1 for c in CHECKS if c["ok"]),
        "checks": CHECKS,
        "verdict": "ANALYSIS_CHECKS_PASS" if ok else "ANALYSIS_CHECK_FAILED",
    }
    with open(os.path.join(OUT, "verification.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    for c in CHECKS:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"]
              + (("  [" + c["detail"] + "]") if c["detail"] else ""))
    print("\n%d/%d  %s" % (rep["n_passed"], rep["n_checks"], rep["verdict"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
