#!/usr/bin/env python3
"""Validator for THR-EXPORT-SCHEMA-1.0.0 reviewer exports.

Enforces the trusted package / protocol / schema / stage / ID sets by default.
Recomputes completion from the raw choices and never trusts a supplied
completion flag or count.

It validates FILE STRUCTURE ONLY. It makes no judgement about image content,
applies no automated scoring, and cannot establish that two files came from two
distinct, independent humans.

Usage:
  validate_export.py FILE [FILE ...] [--expect-stage A|B] [--json OUT]
  validate_export.py --cohort FILE [FILE ...]        # duplicate/alias audit
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"

PROTOCOL_ID = "TEAPOT-HR-PROTO-1.0.0"
SCHEMA_ID = "THR-EXPORT-SCHEMA-1.0.0"
PACKAGES = {"A": "THR-V1-A", "B": "THR-V1-B"}

TARGET_ENUM = {"present", "absent", "uncertain"}
RENDER_ENUM = {"coherent", "localized_deformation",
               "whole_frame_breakdown", "unable_to_judge"}
CMP_ENUM = {"less", "about_the_same", "more", "cannot_fairly_compare"}
KIND_ENUM = {"draft", "final"}

A_EXPOSURE_KEYS = ["prior_access_to_source_pairings",
                   "prior_access_to_condition_labels",
                   "image_specific_coaching_before_task_a"]
B_EXPOSURE_KEYS = ["task_a_final_exported_before_task_b",
                   "prior_access_to_condition_labels",
                   "image_specific_coaching_before_task_a"]


def expected_ids(stage):
    if stage == "A":
        p = os.path.join(ROOT, "protocol", "taskA_items.json")
        return list(json.load(open(p))["item_ids"])
    p = os.path.join(ROOT, "private", "taskB_package", "items.json")
    if os.path.isfile(p):
        return list(json.load(open(p))["item_ids"])
    return [f"B-{i:02d}" for i in range(1, 21)]


def validate(path, expect_stage=None):
    res = {"file": os.path.basename(path), "errors": [], "warnings": [],
           "valid": False}
    try:
        raw = open(path, "rb").read()
        res["file_sha256"] = hashlib.sha256(raw).hexdigest()
        res["bytes"] = len(raw)
    except OSError as e:
        res["errors"].append(f"unreadable: {e}")
        return res
    try:
        d = json.loads(raw.decode("utf-8"))
    except Exception as e:
        res["errors"].append(f"malformed JSON: {type(e).__name__}: {e}")
        return res
    if not isinstance(d, dict):
        res["errors"].append("top level is not a JSON object")
        return res

    stage = d.get("stage")
    res["stage"] = stage
    res["export_kind"] = d.get("export_kind")
    res["reviewer_alias"] = d.get("reviewer_alias")
    res["order_variant"] = d.get("order_variant")

    if stage not in ("A", "B"):
        res["errors"].append(f"stage must be 'A' or 'B', got {stage!r}")
        return res
    if expect_stage and stage != expect_stage:
        res["errors"].append(f"wrong stage: expected {expect_stage}, got {stage}")
        return res
    if d.get("package_id") != PACKAGES[stage]:
        res["errors"].append(
            f"untrusted package_id {d.get('package_id')!r}; expected {PACKAGES[stage]!r}")
    if d.get("protocol_id") != PROTOCOL_ID:
        res["errors"].append(f"untrusted protocol_id {d.get('protocol_id')!r}")
    if d.get("schema_id") != SCHEMA_ID:
        res["errors"].append(f"untrusted schema_id {d.get('schema_id')!r}")
    kind = d.get("export_kind")
    if kind not in KIND_ENUM:
        res["errors"].append(f"export_kind must be draft|final, got {kind!r}")
    if not isinstance(d.get("reviewer_alias"), str) or not d["reviewer_alias"].strip():
        res["errors"].append("reviewer_alias missing or empty")

    # ---- exposure declarations must be explicit ----
    exp = d.get("exposure")
    keys = A_EXPOSURE_KEYS if stage == "A" else B_EXPOSURE_KEYS
    if not isinstance(exp, dict):
        res["errors"].append("exposure block missing")
    else:
        if exp.get("declared") is not True:
            res["errors"].append("exposure.declared must be true")
        for k in keys:
            if k not in exp:
                res["errors"].append(f"exposure.{k} missing")
            elif not isinstance(exp[k], bool):
                res["errors"].append(f"exposure.{k} must be an explicit boolean")
        res["exposure"] = {k: exp.get(k) for k in keys}
        res["exposure_other_note_present"] = bool((exp.get("other_exposure_note") or "").strip())

    # ---- responses ----
    exp_ids = expected_ids(stage)
    idkey = "item_id" if stage == "A" else "pair_id"
    rs = d.get("responses")
    if not isinstance(rs, list):
        res["errors"].append("responses must be a list")
        rs = []

    seen, dup, unknown = set(), [], []
    complete_ids = set()
    bad_enum = []
    for i, r in enumerate(rs):
        if not isinstance(r, dict):
            res["errors"].append(f"responses[{i}] is not an object")
            continue
        rid = r.get(idkey)
        if rid in seen:
            dup.append(rid)
        seen.add(rid)
        if rid not in exp_ids:
            unknown.append(rid)
            continue
        if stage == "A":
            t, g = r.get("target"), r.get("rendering")
            if t is not None and t not in TARGET_ENUM:
                bad_enum.append(f"{rid}.target={t!r}")
            if g is not None and g not in RENDER_ENUM:
                bad_enum.append(f"{rid}.rendering={g!r}")
            # 'uncertain' / 'unable_to_judge' ARE completed answers
            if t in TARGET_ENUM and g in RENDER_ENUM:
                complete_ids.add(rid)
        else:
            c = r.get("comparison")
            if c is not None and c not in CMP_ENUM:
                bad_enum.append(f"{rid}.comparison={c!r}")
            if c in CMP_ENUM:
                complete_ids.add(rid)
        if r.get("reason") is not None and not isinstance(r.get("reason"), str):
            res["errors"].append(f"{rid}.reason must be a string if present")

    if dup:
        res["errors"].append(f"duplicate IDs: {sorted(set(dup))}")
    if unknown:
        res["errors"].append(f"unexpected/extra IDs: {sorted(set(x for x in unknown))}")
    if bad_enum:
        res["errors"].append(f"invalid enum values: {sorted(bad_enum)}")

    missing = sorted(set(exp_ids) - seen)
    # ---- recomputed completion; supplied counts are never trusted ----
    res["expected_items"] = len(exp_ids)
    res["recomputed_complete_answers"] = len(complete_ids)
    res["recomputed_missing_ids"] = sorted(set(exp_ids) - complete_ids)
    supplied = (d.get("completion") or {}).get("answered_items")
    res["supplied_answered_items"] = supplied
    res["supplied_count_trusted"] = False
    if supplied is not None and supplied != len(complete_ids):
        res["warnings"].append(
            f"supplied completion ({supplied}) disagrees with recomputed "
            f"({len(complete_ids)}); the recomputed value governs")

    if kind == "final":
        if missing:
            res["errors"].append(f"FINAL is missing {len(missing)} expected ID(s)")
        if len(complete_ids) != len(exp_ids):
            res["errors"].append(
                f"FINAL requires all {len(exp_ids)} items fully answered; "
                f"{len(complete_ids)} are complete")
        if not rs:
            res["errors"].append("FINAL has no responses (empty final)")
    else:
        res["note_draft"] = "drafts may be partial"

    res["valid"] = not res["errors"]
    return res


def cohort(paths):
    """Duplicate / alias audit. Identical content or a repeated alias is NOT
    evidence of a second independent reviewer."""
    out = {"files": len(paths), "entries": [], "flags": []}
    by_hash, by_alias = {}, {}
    for p in paths:
        r = validate(p)
        out["entries"].append({k: r.get(k) for k in
                               ("file", "stage", "export_kind", "reviewer_alias",
                                "file_sha256", "valid", "recomputed_complete_answers")})
        by_hash.setdefault(r.get("file_sha256"), []).append(r["file"])
        a = (r.get("reviewer_alias") or "").strip().lower()
        by_alias.setdefault(a, []).append(r["file"])
    for h, fs in by_hash.items():
        if len(fs) > 1:
            out["flags"].append({"kind": "identical_file_content", "files": fs,
                                 "meaning": "the same file submitted more than once; "
                                            "counts as ONE return, not several"})
    for a, fs in by_alias.items():
        if len(fs) > 1:
            out["flags"].append({"kind": "repeated_alias", "alias": a, "files": fs,
                                 "meaning": "same alias on multiple files; not "
                                            "evidence of distinct reviewers"})
    out["distinct_aliases"] = len(by_alias)
    out["distinct_file_hashes"] = len(by_hash)
    out["independence_note"] = (
        "An alias, a file count or a declaration alone does not prove distinct "
        "humans or independence. This audit counts files, not people.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expect-stage", choices=["A", "B"])
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    if a.cohort:
        rep = cohort(a.files)
    else:
        rep = {"schema_id": SCHEMA_ID, "protocol_id": PROTOCOL_ID,
               "results": [validate(f, a.expect_stage) for f in a.files]}
        rep["n_valid"] = sum(1 for r in rep["results"] if r["valid"])
        rep["n_invalid"] = len(rep["results"]) - rep["n_valid"]

    txt = json.dumps(rep, indent=1)
    if a.json:
        open(a.json, "w").write(txt)
    print(txt)
    bad = rep.get("n_invalid", 0)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
