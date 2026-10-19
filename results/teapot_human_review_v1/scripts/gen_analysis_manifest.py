#!/usr/bin/env python3
"""Regenerate analysis_v1/MANIFEST_SHA256.md with EXPLICIT exclusions.

Mechanical hashing only. Writes no provenance and touches no numerical
artifact -- deliberately separate from finalize_analysis.py, which rewrites
provenance.json and must not run after the amendment edited it.

No circular coverage is claimed: the manifest excludes itself, the top-level
evidence index, and anything written after it.
"""
import datetime
import hashlib
import os

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
OUT = os.path.join(ROOT, "analysis_v1")
SCRIPTS = os.path.join(ROOT, "scripts")
MANIFEST = "MANIFEST_SHA256.md"

CODE = ("analysis_rules.py", "analyze_v1.py", "test_analysis_synthetic.py",
        "verify_analysis.py", "verify_mappings_for_analysis.py",
        "audit_analysis_public.py", "finalize_analysis.py",
        "gen_analysis_manifest.py", "preservation_check.py",
        "audit_amendment_changeset.py", "gen_evidence_index.py")

# Paths deliberately NOT covered, each with the reason stated in the output.
EXCLUSIONS = [
    (MANIFEST, "this manifest itself -- a file cannot contain its own digest"),
    ("../EVIDENCE_INDEX.md",
     "the top-level evidence index, which is generated FROM the payload "
     "commit that contains this manifest and published in a child commit; it "
     "pins this manifest, not the reverse"),
    ("amendment_public_audit.json",
     "the amendment's change-set privacy scan, which is run AFTER this "
     "manifest precisely so that it covers these final manifest bytes; it "
     "states its own coverage separately"),
    ("<any later verification output>",
     "anything written after this manifest, including any post-index "
     "verification report, which states its own coverage separately"),
]
SKIP_FILES = {"MANIFEST_SHA256.md", "amendment_public_audit.json"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for dp, _, fns in os.walk(OUT):
        for f in sorted(fns):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, OUT)
            if rel in SKIP_FILES:
                continue
            rows.append((rel, os.path.getsize(p), sha(p)))
    for f in sorted(CODE):
        p = os.path.join(SCRIPTS, f)
        if os.path.isfile(p):
            rows.append(("../scripts/" + f, os.path.getsize(p), sha(p)))
    rows.sort()

    now = datetime.datetime.now().astimezone().isoformat()
    L = ["# analysis_v1 — file manifest (SHA256)", "",
         "Mechanical digests of every published file of this analysis, plus "
         "the analysis and amendment code under `../scripts/`.", "",
         "## Coverage and exclusions — stated, not implied", "",
         "Covers the **%d** files listed below. Explicitly **not** covered:"
         % len(rows), ""]
    for path, why in EXCLUSIONS:
        L.append("* `%s` — %s" % (path, why))
    L += ["",
          "**No circular coverage is claimed.** Nothing in this table hashes "
          "itself, and nothing here depends on bytes created after this file.",
          "", "| file | bytes | sha256 |", "|---|---|---|"]
    for rel, n, h in rows:
        L.append("| `%s` | %d | `%s` |" % (rel, n, h))
    L += ["", "%d files. Generated %s." % (len(rows), now), ""]
    with open(os.path.join(OUT, MANIFEST), "w") as fh:
        fh.write("\n".join(L))
    print("wrote %s over %d files, %d explicit exclusions"
          % (MANIFEST, len(rows), len(EXCLUSIONS)))
    for rel, n, h in rows:
        print("  %-52s %9d" % (rel, n))


if __name__ == "__main__":
    main()
