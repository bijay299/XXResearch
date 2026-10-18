#!/usr/bin/env python3
"""Validator for THR-EXPORT-SCHEMA-1.1.0 reviewer exports.

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

PROTOCOL_ID = "TEAPOT-HR-PROTO-1.0.0"
SCHEMA_ID = "THR-EXPORT-SCHEMA-1.1.0"
LEGACY_SCHEMAS = {"THR-EXPORT-SCHEMA-1.0.0"}
APP_VERSION = "1.1.0"
# Maintenance revision of this validator only. The questionnaire bundle, its
# app version and the export schema are NOT changed by this.
VALIDATOR_REVISION = "validator-r2 (AMENDMENT_02 type-guard maintenance)"

_HERE = os.path.dirname(os.path.abspath(__file__))

# Candidate locations for the trusted Task-A item list, searched in order.
# No absolute host path is baked in: the published command works from the
# results directory, from inside an extracted reviewer bundle, or with an
# explicit --items argument.
_ITEM_CANDIDATES = [
    os.path.join(_HERE, os.pardir, "protocol", "taskA_items.json"),
    os.path.join(_HERE, "taskA_items.json"),
    os.path.join(os.getcwd(), "protocol", "taskA_items.json"),
    os.path.join(os.getcwd(), "taskA_items.json"),
    os.path.join(os.getcwd(), "items.json"),
    os.path.join(_HERE, os.pardir, "items.json"),
]
_ITEMS_OVERRIDE = None
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


def _is_str(v):
    return isinstance(v, str)


def _hashable_str(v):
    """Return v when it is a string, else None.

    Every membership test, set insertion, sort key and .strip()/.get() call
    below is guarded through helpers like this one. Malformed input must yield a
    structured invalid result, never a traceback and never partial processing.
    """
    return v if isinstance(v, str) else None


def _typename(v):
    return type(v).__name__


def _find_items():
    if _ITEMS_OVERRIDE:
        return _ITEMS_OVERRIDE
    for c in _ITEM_CANDIDATES:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            try:
                if "item_ids" in json.load(open(c)):
                    return c
            except Exception:
                continue
    return None


def expected_ids(stage):
    """Trusted expected ID set.

    Stage A is read from the bundle's own item list. Stage B falls back to the
    canonical B-01..B-20 names: the private Task-B package is deliberately NOT
    required to be present for this validator to run."""
    if stage == "A":
        p = _find_items()
        if p:
            return list(json.load(open(p))["item_ids"])
        return [f"A-{i:03d}" for i in range(1, 41)]
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
    sid = d.get("schema_id")
    if not _is_str(sid):
        res["errors"].append(
            f"schema_id must be a string, got {_typename(sid)}")
    elif sid in LEGACY_SCHEMAS:
        res["errors"].append(
            f"unsupported legacy export version {sid!r}; this validator reads "
            f"{SCHEMA_ID} only and does not migrate older files")
    elif sid != SCHEMA_ID:
        res["errors"].append(f"untrusted schema_id {sid!r}")
    kind = d.get("export_kind")
    if not _is_str(kind):
        res["errors"].append(
            f"export_kind must be a string, got {_typename(kind)}")
        kind = None
    elif kind not in KIND_ENUM:
        res["errors"].append(f"export_kind must be draft|final, got {kind!r}")
        kind = None
    if not isinstance(d.get("reviewer_alias"), str) or not d["reviewer_alias"].strip():
        res["errors"].append("reviewer_alias missing or empty")
    ov = d.get("order_variant")
    if not isinstance(ov, str) or ov not in {"V1", "V2", "V3", "ALIAS"}:
        res["errors"].append(f"order_variant must be V1|V2|V3|ALIAS, got {ov!r}")
    if "item_count_expected" in d and not isinstance(d["item_count_expected"], int):
        res["errors"].append("item_count_expected must be an integer if present")
    comp = d.get("completion")
    if comp is not None and not isinstance(comp, dict):
        res["errors"].append("completion must be an object if present")
    elif isinstance(comp, dict) and "answered_items" in comp \
            and not isinstance(comp["answered_items"], int):
        res["errors"].append("completion.answered_items must be an integer if present")

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
        if "other_exposure_note" in exp and not isinstance(
                exp["other_exposure_note"], str):
            res["errors"].append("exposure.other_exposure_note must be a string")
        res["exposure"] = {k: exp.get(k) for k in keys}
        _note = exp.get("other_exposure_note")
        res["exposure_other_note_present"] = bool(
            _note.strip()) if _is_str(_note) else False

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
        rid = _hashable_str(r.get(idkey))
        if rid is None:
            res["errors"].append(
                f"responses[{i}].{idkey} must be a string, got "
                f"{_typename(r.get(idkey))}")
            continue
        if rid in seen:
            dup.append(rid)
        seen.add(rid)
        if rid not in exp_ids:
            unknown.append(rid)
            continue
        if stage == "A":
            t, g = r.get("target"), r.get("rendering")
            if t is not None and not _is_str(t):
                bad_enum.append(f"{rid}.target has type {_typename(t)}")
                t = None
            elif t is not None and t not in TARGET_ENUM:
                bad_enum.append(f"{rid}.target={t!r}")
                t = None
            if g is not None and not _is_str(g):
                bad_enum.append(f"{rid}.rendering has type {_typename(g)}")
                g = None
            elif g is not None and g not in RENDER_ENUM:
                bad_enum.append(f"{rid}.rendering={g!r}")
                g = None
            # 'uncertain' / 'unable_to_judge' ARE completed answers
            if t in TARGET_ENUM and g in RENDER_ENUM:
                complete_ids.add(rid)
        else:
            c = r.get("comparison")
            if c is not None and not _is_str(c):
                bad_enum.append(f"{rid}.comparison has type {_typename(c)}")
                c = None
            elif c is not None and c not in CMP_ENUM:
                bad_enum.append(f"{rid}.comparison={c!r}")
                c = None
            if c in CMP_ENUM:
                complete_ids.add(rid)
        if r.get("reason") is not None and not isinstance(r.get("reason"), str):
            res["errors"].append(f"{rid}.reason must be a string if present")

    if dup:
        res["errors"].append(f"duplicate IDs: {sorted({x for x in dup if _is_str(x)})}")
    if unknown:
        res["errors"].append(
            "unexpected/extra IDs: "
            f"{sorted({x for x in unknown if _is_str(x)})}")
    if bad_enum:
        res["errors"].append(f"invalid enum values: {sorted(bad_enum)}")

    missing = sorted(set(exp_ids) - seen)
    # ---- recomputed completion; supplied counts are never trusted ----
    res["expected_items"] = len(exp_ids)
    res["recomputed_complete_answers"] = len(complete_ids)
    res["recomputed_missing_ids"] = sorted(set(exp_ids) - complete_ids)
    _comp = d.get("completion")
    supplied = _comp.get("answered_items") if isinstance(_comp, dict) else None
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
        _a = r.get("reviewer_alias")
        a = _a.strip().lower() if _is_str(_a) else f"<non-string:{_typename(_a)}>"
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
    ap.add_argument("--items", help="path to the trusted taskA_items.json "
                                    "(otherwise auto-discovered)")
    a = ap.parse_args()
    global _ITEMS_OVERRIDE
    if a.items:
        _ITEMS_OVERRIDE = a.items

    if a.cohort:
        rep = cohort(a.files)
    else:
        rep = {"schema_id": SCHEMA_ID, "protocol_id": PROTOCOL_ID,
               "validator_revision": VALIDATOR_REVISION,
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
