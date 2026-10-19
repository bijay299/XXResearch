#!/usr/bin/env python3
"""Mechanical before/after preservation check for the analysis amendment.

Hashes, but never parses or aggregates, the private response copies. Snapshots
the three questionnaire releases and the source asset release from the GitHub
API (download counters excluded), plus every file that must not change.

Usage:
  preservation_check.py --mode before --out <path>
  preservation_check.py --mode after  --out <path> --baseline <path>

No alias, response value, free-text field or crosswalk is read or written.
Release/asset descriptions are recorded as digests rather than text so no
private string can enter the record through them.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
PRIV = os.path.join(ROOT, "private")
BACKUP = "/data/bijaypandey/cuig_pilot/_teapot_human_review_v1_private_backup"
REPO = "/data/bijaypandey/xxresearch_standalone"
PKG = "results/teapot_human_review_v1"
GH = "https://api.github.com/repos/bijay299/XXResearch/releases/tags/"
BASE = "5b252193aac601f9653132cec370b93850dbd549"  # reviewed publication

# Files whose bytes must be identical before and after this amendment.
FROZEN_REPO_FILES = [
    PKG + "/analysis_v1/summary.json",
    PKG + "/analysis_v1/ANALYSIS_CONTRACT.md",
    PKG + "/analysis_v1/synthetic_checks.json",
    PKG + "/analysis_v1/verification.json",
    PKG + "/analysis_v1/public_audit.json",
    PKG + "/analysis_v1/mapping_verification.json",
    PKG + "/analysis_v1/tables/taskA_responses.csv",
    PKG + "/analysis_v1/tables/taskB_responses.csv",
    PKG + "/analysis_v1/tables/taskB_distributions.csv",
    PKG + "/analysis_v1/tables/presence_table_m0_by_edited.csv",
    PKG + "/analysis_v1/tables/paired_case_ledger.csv",
    PKG + "/analysis_v1/tables/target_absence_rendering.csv",
    PKG + "/analysis_v1/tables/descriptive_strata.csv",
    PKG + "/scripts/analysis_rules.py",
    PKG + "/scripts/analyze_v1.py",
    PKG + "/scripts/test_analysis_synthetic.py",
    PKG + "/scripts/verify_analysis.py",
    PKG + "/scripts/verify_mappings_for_analysis.py",
    PKG + "/scripts/audit_analysis_public.py",
    PKG + "/protocol/PROTOCOL.md",
    PKG + "/protocol/COMMITMENTS.json",
    PKG + "/protocol/taskA_items.json",
    PKG + "/protocol/SCHEMA.md",
    PKG + "/protocol/INSTRUCTIONS_TASK_A.md",
    PKG + "/protocol/INSTRUCTIONS_TASK_B.md",
    PKG + "/MASKING_AND_LIMITATIONS.md",
    PKG + "/MANIFEST_SHA256.md",          # package-level manifest: historical
    PKG + "/PREPARATION_ACCOUNTING.md",
    PKG + "/PRIVATE_RECOVERY.md",
    PKG + "/TASKA_ZIP_IDENTITY.json",
    PKG + "/TASKB_ZIP_IDENTITY.json",
    PKG + "/AMENDMENT_01_TASKA_PATCH01.md",
    PKG + "/AMENDMENT_02_COORDINATOR_CLEANUP.md",
    PKG + "/AMENDMENT_03_TASKA_CLOSURE_AND_TASKB_RELEASE.md",
    PKG + "/scripts/validate_export.py",
    PKG + "/scripts/freeze_ids.py",
]

RELEASE_TAGS = ["teapot-human-review-v1-taskA",
                "teapot-human-review-v1-taskA-patch01",
                "teapot-human-review-v1-taskB"]
SOURCE_TAG = "modern-object-assets-v1"

CLOSED_DIRS = ["results/modern_object_assets_v1", "results/measurement_study_v1"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256((s or "").encode()).hexdigest()


def api(tag):
    req = urllib.request.Request(
        GH + tag, headers={"User-Agent": "thr-preserve",
                           "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def release_identity(tag):
    """Release identity EXCLUDING download counters."""
    r = api(tag)
    assets = []
    for a in sorted(r["assets"], key=lambda x: x["name"]):
        assets.append({
            "id": a["id"], "name": a["name"], "size": a["size"],
            "digest": a.get("digest"),
            "created_at": a["created_at"], "updated_at": a["updated_at"],
            "content_type": a.get("content_type"),
            "state": a.get("state"),
        })
    return {
        "tag_name": r["tag_name"],
        "target_commitish": r.get("target_commitish"),
        "release_id": r["id"],
        "created_at": r.get("created_at"),
        "published_at": r.get("published_at"),
        "draft": r.get("draft"), "prerelease": r.get("prerelease"),
        "name_sha256": sha_text(r.get("name")),
        "body_sha256": sha_text(r.get("body")),
        "body_bytes": len((r.get("body") or "").encode()),
        "n_assets": len(assets),
        "assets": assets,
        "download_counts_excluded": True,
    }


def git_tree(commit, prefix):
    out = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", commit,
                          "--", prefix], capture_output=True, text=True).stdout
    return {ln.split("\t", 1)[1]: ln.split()[2]
            for ln in out.strip().splitlines() if "\t" in ln}


def snapshot():
    snap = {"taken_iso": datetime.datetime.now().astimezone().isoformat()}

    # four original exports + private copies and backup -- HASHED, NOT READ
    priv = {}
    rd = os.path.join(PRIV, "responses_v1")
    if os.path.isdir(rd):
        for fn in sorted(os.listdir(rd)):
            if fn.endswith(".json"):
                priv["responses_v1/" + fn] = sha_file(os.path.join(rd, fn))
    else:
        priv = None
    bak = {}
    bd = os.path.join(BACKUP, "responses_v1")
    if os.path.isdir(bd):
        for fn in sorted(os.listdir(bd)):
            if fn.endswith(".json"):
                bak["responses_v1/" + fn] = sha_file(os.path.join(bd, fn))
    else:
        bak = None
    for name, extra in (("id_mapping_private.json", PRIV),
                        ("salt.bin", PRIV),
                        ("analysis_input_manifest.json", PRIV)):
        p = os.path.join(extra, name)
        if priv is not None and os.path.isfile(p):
            priv[name] = sha_file(p)
    snap["private_sha256"] = priv
    snap["private_backup_sha256"] = bak
    snap["private_available"] = priv is not None
    snap["private_note"] = ("Private copies are HASHED ONLY. No response value, "
                            "alias or free-text field was read or aggregated.")

    snap["repo_files_sha256"] = {
        f: (sha_file(os.path.join(REPO, f))
            if os.path.isfile(os.path.join(REPO, f)) else None)
        for f in FROZEN_REPO_FILES}

    # Direct proof against the reviewed publication: compare every frozen
    # file's current bytes with its blob at the given commit. This does not
    # depend on when a baseline snapshot was taken.
    snap["frozen_vs_commit"] = {}
    for f in FROZEN_REPO_FILES:
        r = subprocess.run(["git", "-C", REPO, "show", "%s:%s" % (BASE, f)],
                           capture_output=True)
        blob = hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
        snap["frozen_vs_commit"][f] = blob
    snap["frozen_vs_commit_base"] = BASE

    snap["releases"] = {t: release_identity(t) for t in RELEASE_TAGS}
    snap["source_release"] = release_identity(SOURCE_TAG)
    snap["closed_work_trees"] = {d: git_tree("HEAD", d) for d in CLOSED_DIRS}
    snap["head"] = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    return snap


def compare(before, after):
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    for key, label in (("private_sha256", "private copies (4 exports + mapping "
                                          "+ salt + input manifest)"),
                       ("private_backup_sha256", "private backup copies")):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            chk(label + " unchanged", False,
                "SKIPPED -- private material unavailable on this host")
        else:
            chk(label + " unchanged", b == a,
                "%d files" % len(a) if b == a else "DIFFERS")

    bf, af = before["repo_files_sha256"], after["repo_files_sha256"]
    common = sorted(set(bf) & set(af))
    diff = sorted(k for k in common if bf[k] != af[k])
    chk("frozen repo files unchanged across the amendment (snapshot "
        "before vs after)", not diff,
        ",".join(diff) or "%d files compared" % len(common))

    fvc = after.get("frozen_vs_commit", {})
    bad = sorted(k for k, v in fvc.items() if v != af.get(k))
    chk("every frozen repo file byte-identical to its blob at the reviewed "
        "publication %s" % after.get("frozen_vs_commit_base", "?"),
        not bad, ",".join(bad) or "%d files" % len(fvc))

    for t in RELEASE_TAGS:
        b, a = before["releases"][t], after["releases"][t]
        chk("release %s identical excluding download counters" % t, b == a,
            "assets=%d target=%s" % (a["n_assets"], a["target_commitish"]))
    b, a = before["source_release"], after["source_release"]
    chk("source asset release %s identical excluding download counters"
        % SOURCE_TAG, b == a, "assets=%d" % a["n_assets"])
    chk("source asset release still carries 31 assets",
        a["n_assets"] == 31, a["n_assets"])

    for d in CLOSED_DIRS:
        chk("closed work unchanged: " + d,
            before["closed_work_trees"][d] == after["closed_work_trees"][d],
            "%d tracked files" % len(after["closed_work_trees"][d]))

    ok = all(c["ok"] for c in checks)
    return {"n_checks": len(checks), "n_passed": sum(1 for c in checks
                                                     if c["ok"]),
            "checks": checks,
            "verdict": "PRESERVATION_VERIFIED" if ok
                       else "PRESERVATION_PROBLEM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["before", "after"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline")
    a = ap.parse_args()

    snap = snapshot()
    if a.mode == "before":
        with open(a.out, "w") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
        print("baseline written: %s" % a.out)
        print("private available: %s" % snap["private_available"])
        for t in RELEASE_TAGS + [SOURCE_TAG]:
            r = snap["releases"][t] if t in snap["releases"] \
                else snap["source_release"]
            print("  %-40s assets=%-3d target=%s"
                  % (t, r["n_assets"], r["target_commitish"]))
        return 0

    before = json.load(open(a.baseline))
    rep = compare(before, snap)
    rep["generated_iso"] = datetime.datetime.now().astimezone().isoformat()
    rep["what_this_is"] = (
        "Mechanical before/after preservation check for the documentation, "
        "provenance and evidence-index amendment. Hashing and GitHub API "
        "metadata only -- no response value, alias or free-text field was read "
        "or aggregated, and no numerical analysis was re-run. Host-performed, "
        "not an independent attestation.")
    rep["baseline_head"] = before.get("head")
    rep["after_head"] = snap.get("head")
    rep["download_counters_excluded"] = True
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True)
    for c in rep["checks"]:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"]
              + (("  [" + c["detail"] + "]") if c["detail"] else ""))
    print("\n%d/%d  %s" % (rep["n_passed"], rep["n_checks"], rep["verdict"]))
    return 0 if rep["verdict"] == "PRESERVATION_VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
