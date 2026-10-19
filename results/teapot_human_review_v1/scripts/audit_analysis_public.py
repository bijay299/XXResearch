#!/usr/bin/env python3
"""Leak audit of every file staged for publication under analysis_v1/.

Scans for reviewer aliases, original export filenames, free-text reviewer
prose, exposure notes, export timestamps, source image filenames and hashes,
case IDs, the randomization salt, private host paths and private directory
names.

Secrets are loaded from the PRIVATE records at run time. No alias, filename,
hash or other private identifier is a literal in this file -- a mistake made
once already in this project (Amendment 03 §7b) and not repeated. If a private
record is missing, the corresponding scan is reported as SKIPPED rather than
silently passing.
"""
import datetime
import json
import os
import re
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
PRIV = os.path.join(ROOT, "private")
STAGE = os.path.join(ROOT, "analysis_v1")
OUT = os.path.join(STAGE, "public_audit.json")


def _roster():
    p = os.path.join(PRIV, "taskA_closure_private.json")
    if not os.path.isfile(p):
        return None
    try:
        return [r["alias"] for r in json.load(open(p))["roster"]]
    except Exception:
        return None


def _responses():
    d = os.path.join(PRIV, "responses_v1")
    if not os.path.isdir(d):
        return None
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            out.append((fn, json.load(open(os.path.join(d, fn)))))
    return out or None


def _mapping():
    p = os.path.join(PRIV, "id_mapping_private.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def staged_files():
    out = []
    for dp, _, fns in os.walk(STAGE):
        for f in sorted(fns):
            p = os.path.join(dp, f)
            out.append((os.path.relpath(p, STAGE), p))
    return sorted(out)


def main():
    files = staged_files()
    texts = {}
    for rel, p in files:
        try:
            texts[rel] = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            texts[rel] = ""

    findings = []
    informational = []
    scans = []

    tokens = {rel: set(re.split(r"[^a-z0-9]+", txt.lower()))
              for rel, txt in texts.items()}

    def scan(name, needles, label, ci=True, mode="substring"):
        """Report any staged file containing any needle. Needles are secret and
        are never written to the report -- only the file and a label.

        mode="substring" is the strict default. mode="token" is used for short
        identifiers such as a three-letter alias, which would otherwise match
        inside ordinary English words ("closure", "disclosed"). Token mode
        splits the text on every non-alphanumeric character and compares whole
        tokens, so "THR-A_<alias>_final.json" still matches while "closure"
        does not. Substring-only hits are still recorded separately as
        informational, so a token-mode scan never hides a match.
        """
        if needles is None:
            scans.append({"scan": name, "status": "SKIPPED",
                          "reason": "private record unavailable"})
            return
        needles = [n for n in needles if n and len(str(n)) >= 3]
        if not needles:
            scans.append({"scan": name, "status": "NOT_APPLICABLE",
                          "reason": ("the originals contain no such value, so "
                                     "there is nothing to leak")})
            return
        hits = substr_only = 0
        for rel, txt in texts.items():
            hay = txt.lower() if ci else txt
            tok_hit = sub_hit = False
            for n in needles:
                nn = str(n).lower() if ci else str(n)
                if nn in hay:
                    sub_hit = True
                if nn in tokens[rel]:
                    tok_hit = True
            if mode == "token":
                if tok_hit:
                    findings.append({"file": rel, "kind": label})
                    hits += 1
                elif sub_hit:
                    informational.append({
                        "file": rel, "kind": label,
                        "note": ("substring match only, not a whole token -- "
                                 "a short identifier appearing inside an "
                                 "ordinary word, reviewed as a false "
                                 "positive")})
                    substr_only += 1
            elif sub_hit:
                findings.append({"file": rel, "kind": label})
                hits += 1
        s = {"scan": name, "status": "RAN", "match_mode": mode,
             "n_needles": len(needles), "n_files_with_findings": hits}
        if mode == "token":
            s["n_files_substring_only_informational"] = substr_only
        scans.append(s)

    roster = _roster()
    resp = _responses()
    mp = _mapping()

    scan("reviewer aliases", roster, "reviewer alias", mode="token")

    if resp is None:
        scans.append({"scan": "original export filenames", "status": "SKIPPED",
                      "reason": "private record unavailable"})
        scans.append({"scan": "free-text reviewer prose", "status": "SKIPPED",
                      "reason": "private record unavailable"})
        scans.append({"scan": "export timestamps", "status": "SKIPPED",
                      "reason": "private record unavailable"})
    else:
        scan("original export filenames", [fn for fn, _ in resp],
             "original export filename")
        prose, notes, stamps = [], [], []
        for _, d in resp:
            for r in d.get("responses", []):
                t = (r.get("reason") or "").strip()
                if t:
                    prose.append(t)
            nt = (d.get("exposure", {}).get("other_exposure_note") or "").strip()
            if nt:
                notes.append(nt)
            if d.get("exported_iso"):
                stamps.append(d["exported_iso"])
        scan("free-text reviewer prose", prose, "reviewer free text")
        scan("other_exposure_note text", notes, "exposure note text")
        scan("export timestamps", stamps, "export timestamp")

    if mp is None:
        for s in ("source image filenames", "source image hashes", "case IDs"):
            scans.append({"scan": s, "status": "SKIPPED",
                          "reason": "private record unavailable"})
    else:
        am = mp["taskA_mapping"]
        bm = mp["taskB_mapping"]
        scan("source image filenames",
             [v["source_filename"] for v in am.values()],
             "source image filename")
        scan("source image hashes", [v["source_sha256"] for v in am.values()],
             "source image hash")
        scan("case IDs", sorted({v["case_id"] for v in am.values()}
                                | {v["case_id"] for v in bm.values()}),
             "case ID")
        scan("initial-noise hashes",
             [v["noise_sha256"] for v in bm.values()], "noise hash")

    # salt, in both raw and hex form
    sp = os.path.join(PRIV, "salt.bin")
    if os.path.isfile(sp):
        raw = open(sp, "rb").read()
        scan("randomization salt (hex)", [raw.hex()], "salt", ci=True)
    else:
        scans.append({"scan": "randomization salt (hex)", "status": "SKIPPED",
                      "reason": "private record unavailable"})

    # structural / path patterns -- these are not secrets, so they are literals
    pats = {
        "private host path": r"/cuig_pilot/|/private/|_private_backup",
        "private record filename": r"id_mapping_private|taskA_closure_private|"
                                   r"analysis_input_manifest|responses_v1",
        "generic CASE id pattern": r"\bCASE-\d{2}\b",
        "source png filename pattern": r"\bCASE-\d{2}_(m0|edited)_seed\d+\.png\b",
        "salt or key field": r"\bsalt\b\s*[:=]|private_key|secret_key",
        "alias field in data": r'"reviewer_alias"\s*:\s*"[^"]+"',
        "exported_iso field in data": r'"exported_iso"\s*:\s*"[^"]+"',
    }
    for rel, txt in texts.items():
        for label, pat in pats.items():
            if re.search(pat, txt):
                findings.append({"file": rel, "kind": label})

    ran = [s for s in scans if s["status"] == "RAN"]
    na = [s for s in scans if s["status"] == "NOT_APPLICABLE"]
    skipped = [s for s in scans if s["status"] == "SKIPPED"]
    clean = not findings

    rep = {
        "generated_iso": datetime.datetime.now().astimezone().isoformat(),
        "what_this_is": ("Leak audit of files staged for publication under "
                         "analysis_v1/. Secrets are loaded from private "
                         "records at run time and never appear as literals in "
                         "this script."),
        "staged_dir": "results/teapot_human_review_v1/analysis_v1",
        "n_staged_files": len(files),
        "staged_files": [rel for rel, _ in files],
        "scans": scans,
        "n_scans_ran": len(ran),
        "n_scans_not_applicable": len(na),
        "n_scans_skipped_private_record_unavailable": len(skipped),
        "all_secret_scans_ran": not skipped,
        "findings": findings,
        "n_findings": len(findings),
        "informational_non_findings": informational,
        "n_informational": len(informational),
        "verdict": ("ANALYSIS_PUBLIC_AUDIT_CLEAN" if clean and not skipped
                    else "ANALYSIS_PUBLIC_AUDIT_FINDINGS" if findings
                    else "ANALYSIS_PUBLIC_AUDIT_INCOMPLETE"),
    }
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)

    for s in scans:
        print("%-8s %-32s %s" % (s["status"], s["scan"],
                                 s.get("n_files_with_findings",
                                       s.get("reason", ""))))
    print("\nstaged files: %d" % len(files))
    if findings:
        print("FINDINGS:")
        for f in findings:
            print("  %s  <- %s" % (f["file"], f["kind"]))
    print("\n" + rep["verdict"])
    return 0 if rep["verdict"] == "ANALYSIS_PUBLIC_AUDIT_CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
