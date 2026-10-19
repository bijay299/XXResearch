#!/usr/bin/env python3
"""Write analysis_v1/provenance.json and analysis_v1/MANIFEST_SHA256.md.

Run after the analysis, verification and audit so the manifest covers the final
bytes. The manifest deliberately excludes itself.
"""
import datetime
import hashlib
import json
import os
import subprocess

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
OUT = os.path.join(ROOT, "analysis_v1")
SCRIPTS = os.path.join(ROOT, "scripts")
REPO = "/data/bijaypandey/xxresearch_standalone"

CODE = ("analysis_rules.py", "analyze_v1.py", "test_analysis_synthetic.py",
        "verify_analysis.py", "verify_mappings_for_analysis.py",
        "audit_analysis_public.py", "finalize_analysis.py")
MANIFEST = "MANIFEST_SHA256.md"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git"] + list(a), cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def main():
    summ = json.load(open(os.path.join(OUT, "summary.json")))
    ver = json.load(open(os.path.join(OUT, "verification.json")))
    syn = json.load(open(os.path.join(OUT, "synthetic_checks.json")))
    mapv = json.load(open(os.path.join(OUT, "mapping_verification.json")))
    aud = json.load(open(os.path.join(OUT, "public_audit.json")))

    prov = {
        "analysis_id": summ["analysis_id"],
        "generated_iso": datetime.datetime.now().astimezone().isoformat(),
        "scope": summ["scope"],
        "chronology_disclosure": (
            "The four answer files were already open in the working context "
            "when the analysis contract was written: they arrived attached to "
            "the request. The contract was frozen BEFORE any aggregation, "
            "table or count was computed, but NOT before the raw rows could be "
            "seen. See ANALYSIS_CONTRACT.md section 0. This is recorded as "
            "implementation of existing frozen rules plus clarification of "
            "their shorthand -- not a new retrospective preregistration."),
        "inputs": {
            "n_allowlisted": 4,
            "keyed_by": "neutral reviewer ID and stage",
            "sha256": summ["integrity"]["input_sha256_before"],
            "unchanged_by_analysis":
                summ["integrity"]["inputs_unchanged_by_analysis"],
            "originals_modified": False,
            "note": ("Original filenames embed reviewer aliases and are "
                     "withheld. Private read-only copies (mode 0400) and a "
                     "private SHA256 input manifest were made, with a "
                     "host-verified private backup. The user's originals were "
                     "not modified."),
            "backup_verification": ("host-performed, not independent "
                                    "attestation"),
        },
        "validator": {"revision": summ["validator_revision"],
                      "all_four_valid": True, "errors": 0, "warnings": 0},
        "code_sha256": {f: sha(os.path.join(SCRIPTS, f))
                        for f in CODE if os.path.exists(
                            os.path.join(SCRIPTS, f))},
        "results": {
            "synthetic_checks": "%d/%d" % (syn["n_passed"], syn["n_checks"]),
            "analysis_checks": "%d/%d %s" % (ver["n_passed"], ver["n_checks"],
                                             ver["verdict"]),
            "mapping_verification": "%d/%d %s" % (
                mapv["n_passed"], mapv["n_checks"], mapv["verdict"]),
            "public_audit": "%d findings %s" % (aud["n_findings"],
                                                aud["verdict"]),
            "reproducibility": ("public tables byte-identical across two runs; "
                                "summary.json identical apart from timestamp"),
        },
        "principal_counts": {
            "cases": 20, "taskA_judgments": 80, "taskB_judgments": 40,
            "reviewers_planned": 3, "reviewers_actual": 2,
            "demonstrated_removal": summ["demonstrated_removal"]["value_text"],
            "presence_table_total":
                summ["presence_table_m0_by_edited"]["total"],
            "taskB_exact_agreement": "%d/%d" % (
                summ["taskB"]["exact_per_pair_agreement"]["agreeing_pairs"],
                summ["taskB"]["exact_per_pair_agreement"]["denominator"]),
        },
        "preserved_unchanged": [
            "all four original responses (byte-verified before and after)",
            "private mappings, salt and both published commitments "
            "(re-verified, mappings NOT regenerated)",
            "Task-A item IDs, Task-B pair IDs, role assignments",
            "both Task-A releases and the Task-B release: assets and tag "
            "targets",
            "both questionnaire viewers (neither rebuilt)",
            "the asset-build findings (not updated)",
            "all closed-project files",
        ],
        "not_done": [
            "no new experiment, model run, training or GPU allocation",
            "no image generation", "no new labels",
            "no automated or AI judging / adjudication",
            "no re-review, relabelling or outcome-chosen exclusion",
            "no participant contact", "no revised judgments",
            "no response replacement", "no reopening of closed work",
            "no hypothesis test, interval, bootstrap, kappa or significance "
            "claim",
            "no follow-up study executed",
        ],
        "verification_nature": ("All checks here are HOST-PERFORMED: this "
                                "machine ran them on its own outputs. Not an "
                                "independent attestation."),
        "public_reproducibility_limit": (
            "The case-level m0/edited join runs through the PRIVATE mapping. "
            "Published files allow rechecking every table's arithmetic, "
            "totals and joins, but not independent reproduction of the "
            "condition assignment."),
        "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "head_at_write": git("rev-parse", "HEAD")},
    }
    with open(os.path.join(OUT, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    # ---- manifest over the published analysis files ------------------------
    rows = []
    for dp, _, fns in os.walk(OUT):
        for f in sorted(fns):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, OUT)
            if rel == MANIFEST:
                continue
            rows.append((rel, os.path.getsize(p), sha(p)))
    for f in sorted(prov["code_sha256"]):
        p = os.path.join(SCRIPTS, f)
        rows.append(("../scripts/" + f, os.path.getsize(p), sha(p)))
    rows.sort()

    lines = [
        "# analysis_v1 — file manifest (SHA256)",
        "",
        "Covers every published file of this analysis, plus the analysis code "
        "under `../scripts/`. This manifest excludes itself.",
        "",
        "| file | bytes | sha256 |",
        "|---|---|---|",
    ]
    for rel, n, h in rows:
        lines.append("| `%s` | %d | `%s` |" % (rel, n, h))
    lines += ["", "%d files. Generated %s." % (len(rows),
                                               prov["generated_iso"]), ""]
    with open(os.path.join(OUT, MANIFEST), "w") as fh:
        fh.write("\n".join(lines))

    print("provenance.json and %s written over %d files" % (MANIFEST,
                                                            len(rows)))
    print(json.dumps(prov["results"], indent=1))
    print(json.dumps(prov["principal_counts"], indent=1))


if __name__ == "__main__":
    main()
