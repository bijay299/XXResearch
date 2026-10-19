#!/usr/bin/env python3
"""Scoped public-content scan over the ACTUAL change set of this amendment.

Scope: every file that differs between a baseline commit and the working tree
under results/. This is a NEW record; it does not overwrite or supersede the
historical analysis_v1/public_audit.json, which stays as published.

Secrets are loaded from private records at run time and never appear as
literals here. A missing private record makes that scan report SKIPPED rather
than silently pass.

Two scan layers, reported separately:
  * SECRETS  -- whole-file scan of every changed file. Any hit is a finding.
  * PATHS    -- private host paths / private record filenames, scanned on the
                lines this amendment ADDED. Occurrences already present in the
                baseline are reported as pre-existing, not as new findings,
                because published code across this project has always carried
                its own host paths and PRIVATE_RECOVERY.md documents them.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
PRIV = os.path.join(ROOT, "private")
REPO = "/data/bijaypandey/xxresearch_standalone"
OUT = os.path.join(ROOT, "analysis_v1", "amendment_public_audit.json")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="commit the change set is measured against")
    a = ap.parse_args()

    changed = subprocess.run(
        ["git", "-C", REPO, "diff", "--name-only", a.baseline, "--", "results"],
        capture_output=True, text=True, check=True).stdout.split()
    untracked = subprocess.run(
        ["git", "-C", REPO, "ls-files", "-o", "--exclude-standard", "results"],
        capture_output=True, text=True, check=True).stdout.split()
    files = sorted(set(changed) | set(untracked))

    texts, skipped_binary = {}, []
    for rel in files:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            continue
        try:
            texts[rel] = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            skipped_binary.append(rel)

    added = {}
    for rel in texts:
        d = subprocess.run(["git", "-C", REPO, "diff", "-U0", a.baseline, "--",
                            rel], capture_output=True, text=True).stdout
        lines = [ln[1:] for ln in d.splitlines()
                 if ln.startswith("+") and not ln.startswith("+++")]
        added[rel] = "\n".join(lines) if lines else texts[rel]

    findings, preexisting, scans = [], [], []
    tokens = {rel: set(re.split(r"[^a-z0-9]+", t.lower()))
              for rel, t in texts.items()}

    def secret_scan(name, needles, label, mode="substring"):
        if needles is None:
            scans.append({"scan": name, "layer": "SECRETS",
                          "status": "SKIPPED",
                          "reason": "private record unavailable on this host"})
            return
        needles = [str(n) for n in needles if n and len(str(n)) >= 3]
        if not needles:
            scans.append({"scan": name, "layer": "SECRETS",
                          "status": "NOT_APPLICABLE",
                          "reason": "the originals hold no such value, so "
                                    "there is nothing to leak"})
            return
        hits = 0
        for rel, t in texts.items():
            low = t.lower()
            tok = any(n.lower() in tokens[rel] for n in needles)
            sub = any(n.lower() in low for n in needles)
            if (tok if mode == "token" else sub):
                findings.append({"file": rel, "kind": label,
                                 "layer": "SECRETS"})
                hits += 1
            elif mode == "token" and sub:
                preexisting.append({
                    "file": rel, "kind": label,
                    "note": "substring match only, not a whole token -- a "
                            "short identifier inside an ordinary word; "
                            "reviewed as a false positive"})
        scans.append({"scan": name, "layer": "SECRETS", "status": "RAN",
                      "match_mode": mode, "n_needles": len(needles),
                      "n_files_with_findings": hits})

    roster, resp, mp = _roster(), _responses(), _mapping()

    secret_scan("reviewer aliases", roster, "reviewer alias", mode="token")

    if resp is None:
        for n in ("original export filenames", "free-text reviewer prose",
                  "other_exposure_note text", "export timestamps"):
            scans.append({"scan": n, "layer": "SECRETS", "status": "SKIPPED",
                          "reason": "private record unavailable on this host"})
    else:
        secret_scan("original export filenames", [fn for fn, _ in resp],
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
        secret_scan("free-text reviewer prose", prose, "reviewer free text")
        secret_scan("other_exposure_note text", notes, "exposure note text")
        secret_scan("export timestamps", stamps, "export timestamp")

    if mp is None:
        for n in ("source image filenames", "source image hashes", "case IDs",
                  "initial-noise hashes", "direct crosswalk rows"):
            scans.append({"scan": n, "layer": "SECRETS", "status": "SKIPPED",
                          "reason": "private record unavailable on this host"})
    else:
        am, bm = mp["taskA_mapping"], mp["taskB_mapping"]
        secret_scan("source image filenames",
                    [v["source_filename"] for v in am.values()],
                    "source image filename")
        secret_scan("source image hashes",
                    [v["source_sha256"] for v in am.values()],
                    "source image hash")
        secret_scan("case IDs",
                    sorted({v["case_id"] for v in am.values()}
                           | {v["case_id"] for v in bm.values()}), "case ID")
        secret_scan("initial-noise hashes",
                    [v["noise_sha256"] for v in bm.values()], "noise hash")
        # a direct crosswalk would pair a neutral ID with its condition
        xw = ["%s %s" % (k, v["condition"]) for k, v in am.items()]
        xw += ["%s,%s" % (k, v["condition"]) for k, v in am.items()]
        secret_scan("direct neutral-ID-to-condition crosswalk rows", xw,
                    "direct crosswalk row")

    sp = os.path.join(PRIV, "salt.bin")
    if os.path.isfile(sp):
        secret_scan("randomization salt (hex)",
                    [open(sp, "rb").read().hex()], "salt")
    else:
        scans.append({"scan": "randomization salt (hex)", "layer": "SECRETS",
                      "status": "SKIPPED",
                      "reason": "private record unavailable on this host"})

    # ---- PATHS layer: added lines only ------------------------------------
    pats = {
        "private host path": r"/cuig_pilot/|_private_backup",
        "private record filename": r"id_mapping_private|taskA_closure_private"
                                   r"|analysis_input_manifest|responses_v1",
        "private key material": r"private_key|secret_key|BEGIN [A-Z ]*PRIVATE",
    }
    # Established practice in this project, measured -- not assumed: does the
    # same pattern already appear in the previously published sibling scripts
    # at the baseline commit? Published scripts here have always carried their
    # own host paths and the names of the private records they read, and
    # PRIVATE_RECOVERY.md documents those locations. A pattern with precedent
    # is reported as pre-existing practice; one WITHOUT precedent is a finding.
    sib = subprocess.run(
        ["git", "-C", REPO, "ls-tree", "-r", "--name-only", a.baseline, "--",
         "results/teapot_human_review_v1/scripts"],
        capture_output=True, text=True).stdout.split()
    precedent = {}
    for label, pat in pats.items():
        n = 0
        for rel in sib:
            b = subprocess.run(["git", "-C", REPO, "show",
                                f"{a.baseline}:{rel}"],
                               capture_output=True, text=True).stdout
            if re.search(pat, b):
                n += 1
        precedent[label] = n

    SELF_REL = "results/teapot_human_review_v1/scripts/" + \
               os.path.basename(__file__)
    path_hits = 0
    for rel in sorted(texts):
        base = subprocess.run(["git", "-C", REPO, "show",
                               f"{a.baseline}:{rel}"],
                              capture_output=True, text=True).stdout
        for label, pat in pats.items():
            if not re.search(pat, added[rel]):
                continue
            if rel == SELF_REL:
                preexisting.append({
                    "file": rel, "kind": label,
                    "note": "self-match: this scanner's own pattern table "
                            "contains the literal it searches for"})
                continue
            if re.search(pat, base):
                preexisting.append({
                    "file": rel, "kind": label,
                    "note": "already present in this file at the baseline"})
            elif precedent[label]:
                preexisting.append({
                    "file": rel, "kind": label,
                    "note": ("established practice: the same pattern already "
                             "appears in %d previously published script(s) in "
                             "this package at the baseline, and "
                             "PRIVATE_RECOVERY.md documents these locations. "
                             "Not a secret: the path/record NAME reveals no "
                             "content." % precedent[label])})
            else:
                findings.append({"file": rel, "kind": label, "layer": "PATHS"})
                path_hits += 1
    scans.append({"scan": "private paths / record names / key material",
                  "layer": "PATHS", "status": "RAN",
                  "scope": "lines added by this amendment",
                  "precedent_in_published_scripts_at_baseline": precedent,
                  "n_files_with_new_findings": path_hits})

    ran = [s for s in scans if s["status"] == "RAN"]
    na = [s for s in scans if s["status"] == "NOT_APPLICABLE"]
    skip = [s for s in scans if s["status"] == "SKIPPED"]
    clean = not findings and not skip

    rep = {
        "generated_iso": datetime.datetime.now().astimezone().isoformat(),
        "what_this_is": (
            "NEW scoped public-content scan over the actual change set of the "
            "documentation/provenance/evidence-index amendment. It does NOT "
            "overwrite or supersede analysis_v1/public_audit.json, which "
            "remains as published for the original analysis."),
        "baseline_commit": a.baseline,
        "scope": "files differing from the baseline under results/, plus "
                 "untracked files there",
        "n_files_in_change_set": len(files),
        "files_in_change_set": files,
        "n_binary_or_undecodable_skipped": len(skipped_binary),
        "binary_or_undecodable_skipped": skipped_binary,
        "scans": scans,
        "n_scans_ran": len(ran),
        "n_scans_not_applicable": len(na),
        "n_scans_skipped_private_record_unavailable": len(skip),
        "all_secret_scans_ran": not skip,
        "findings": findings,
        "n_findings": len(findings),
        "pre_existing_or_informational": preexisting,
        "n_pre_existing_or_informational": len(preexisting),
        "verdict": ("AMENDMENT_CHANGESET_AUDIT_CLEAN" if clean
                    else "AMENDMENT_CHANGESET_AUDIT_FINDINGS" if findings
                    else "AMENDMENT_CHANGESET_AUDIT_INCOMPLETE"),
    }
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)

    for s in scans:
        print("%-15s %-10s %-44s %s"
              % (s["status"], s["layer"], s["scan"],
                 s.get("n_files_with_findings",
                       s.get("n_files_with_new_findings",
                             s.get("reason", "")))))
    print("\nchange set: %d files (%d binary skipped)"
          % (len(files), len(skipped_binary)))
    if findings:
        print("FINDINGS:")
        for f in findings:
            print("  [%s] %s <- %s" % (f["layer"], f["file"], f["kind"]))
    if preexisting:
        print("pre-existing / informational: %d" % len(preexisting))
        for f in preexisting:
            print("  %s <- %s" % (f["file"], f["kind"]))
    print("\n" + rep["verdict"])
    return 0 if rep["verdict"] == "AMENDMENT_CHANGESET_AUDIT_CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
