#!/usr/bin/env python3
"""Validates reviewer response files against msv1-response-schema v1.0.0.

Pure stdlib, no network. Run on returned JSON before any analysis:

    python3 validate_responses.py returned/*.json

Reports, per file: schema conformance, stage/package/protocol agreement, duplicate or
unknown neutral ids, and missing-answer counts. It never repairs, imputes or rewrites a
file; it only reports.
"""
import json, sys, re, argparse, os

SCHEMA_ID = "msv1-response-schema"
SCHEMA_VERSION = "1.0.0"
PRESENCE = {"present", "absent", "uncertain", None}
QUALITY = {"coherent", "severe_rendering_failure", "unable_to_judge", None}
DEFORM = {"less", "about_same", "more", "cannot_compare", None}
ALIAS_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
TOP_REQ = ["schema_id", "schema_version", "package_id", "protocol_id", "stage",
           "export_kind", "reviewer_alias", "order_variant", "prior_exposure",
           "exported_utc", "n_items", "n_answered", "n_missing", "responses"]


def validate(path, expect_stage=None, expect_package=None, expect_ids=None):
    errs, warns = [], []
    try:
        d = json.load(open(path))
    except Exception as e:
        return {"file": os.path.basename(path), "ok": False,
                "errors": [f"not parseable JSON: {e}"], "warnings": []}

    for k in TOP_REQ:
        if k not in d:
            errs.append(f"missing required field '{k}'")
    if errs:
        return {"file": os.path.basename(path), "ok": False, "errors": errs, "warnings": warns}

    if d["schema_id"] != SCHEMA_ID:
        errs.append(f"schema_id {d['schema_id']!r} != {SCHEMA_ID!r}")
    if d["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {d['schema_version']!r} != {SCHEMA_VERSION!r}")
    if d["stage"] not in ("A", "B"):
        errs.append(f"stage {d['stage']!r} invalid")
    if expect_stage and d["stage"] != expect_stage:
        errs.append(f"stage {d['stage']!r} != expected {expect_stage!r}")
    if expect_package and d["package_id"] != expect_package:
        errs.append(f"package_id {d['package_id']!r} != expected {expect_package!r}")
    if d["export_kind"] not in ("draft", "final"):
        errs.append(f"export_kind {d['export_kind']!r} invalid")
    if d["export_kind"] == "draft":
        warns.append("DRAFT file: not admissible to the primary analysis")
    if not ALIAS_RE.match(str(d["reviewer_alias"])):
        errs.append("reviewer_alias fails the alias pattern")
    if re.search(r"[@\s]", str(d["reviewer_alias"])):
        errs.append("reviewer_alias looks identifying (contains whitespace or '@')")
    if d["order_variant"] not in ("A", "B", "C"):
        errs.append(f"order_variant {d['order_variant']!r} invalid")
    if d["prior_exposure"] not in ("none", "some", "involved"):
        errs.append(f"prior_exposure {d['prior_exposure']!r} invalid")
    if not isinstance(d["responses"], list):
        errs.append("responses is not a list")
        return {"file": os.path.basename(path), "ok": False, "errors": errs, "warnings": warns}

    seen, answered, missing = set(), 0, 0
    for i, r in enumerate(d["responses"]):
        tag = f"responses[{i}]"
        for k in ("neutral_id", "target", "position", "answered"):
            if k not in r:
                errs.append(f"{tag}: missing '{k}'")
        if "neutral_id" not in r:
            continue
        nid = r["neutral_id"]
        if nid in seen:
            errs.append(f"{tag}: duplicate neutral_id {nid!r}")
        seen.add(nid)
        if expect_ids is not None and nid not in expect_ids:
            errs.append(f"{tag}: unknown neutral_id {nid!r}")
        if d["stage"] == "A":
            if r.get("target_presence") not in PRESENCE:
                errs.append(f"{tag}: target_presence {r.get('target_presence')!r} invalid")
            if r.get("scene_quality") not in QUALITY:
                errs.append(f"{tag}: scene_quality {r.get('scene_quality')!r} invalid")
            ok = bool(r.get("target_presence")) and bool(r.get("scene_quality"))
        else:
            if r.get("deformation_vs_reference") not in DEFORM:
                errs.append(f"{tag}: deformation_vs_reference invalid")
            ok = bool(r.get("deformation_vs_reference"))
        if ok:
            answered += 1
        else:
            missing += 1
        if bool(r.get("answered")) != ok:
            warns.append(f"{tag}: 'answered' flag disagrees with the content")

    if expect_ids is not None:
        absent = set(expect_ids) - seen
        if absent:
            warns.append(f"{len(absent)} package item(s) have no response row")
    if d["n_answered"] != answered:
        warns.append(f"n_answered {d['n_answered']} != recomputed {answered}")
    if d["n_missing"] != missing:
        warns.append(f"n_missing {d['n_missing']} != recomputed {missing}")

    return {"file": os.path.basename(path), "ok": not errs,
            "stage": d["stage"], "alias": d["reviewer_alias"],
            "export_kind": d["export_kind"], "prior_exposure": d["prior_exposure"],
            "order_variant": d["order_variant"],
            "n_rows": len(d["responses"]), "answered": answered, "missing": missing,
            "errors": errs, "warnings": warns}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--package", default=None)
    ap.add_argument("--ids", default=None, help="JSON list of valid neutral ids")
    a = ap.parse_args()
    ids = set(json.load(open(a.ids))) if a.ids else None
    bad = 0
    for f in a.files:
        r = validate(f, a.stage, a.package, ids)
        bad += (not r["ok"])
        print(f"\n== {r['file']} == {'OK' if r['ok'] else 'INVALID'}")
        for k in ("stage", "alias", "export_kind", "prior_exposure", "order_variant",
                  "n_rows", "answered", "missing"):
            if k in r:
                print(f"   {k}: {r[k]}")
        for e in r["errors"]:
            print(f"   ERROR   {e}")
        for w in r["warnings"]:
            print(f"   warning {w}")
    print(f"\n{len(a.files) - bad}/{len(a.files)} file(s) valid")
    sys.exit(1 if bad else 0)
