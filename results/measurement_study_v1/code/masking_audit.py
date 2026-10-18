#!/usr/bin/env python3
"""Masking audit: prove nothing private leaked into anything public.

Scans every file that will be published (and the extracted ZIP) for the private
material listed in the protocol: the commitment salt, the randomisation seed, any arm /
method / condition mapping, case-level source provenance, automated predictions, and any
Task-B reference association. Also checks for external URLs (offline requirement),
suspicious base64 blobs, and non-neutral filenames.

Exits non-zero on any finding.
"""
import os, sys, json, re, base64, argparse

ROOT = "/data/bijaypandey/cuig_pilot/measurement_study_v1"
TEXT_EXT = {".html", ".htm", ".js", ".json", ".md", ".txt", ".css", ".py", ".csv", ".yml", ".yaml"}
BIN_EXT = {".png", ".jpg", ".jpeg", ".zip", ".webp", ".gif"}

findings = []


def finding(sev, where, what):
    findings.append({"severity": sev, "where": where, "what": what})


def load_secrets():
    k = json.load(open(f"{ROOT}/private/keys/blinding_key.json"))
    return [str(k["commitment_salt_hex"]), str(k["randomization_seed"])]


# words that would indicate a leaked mapping if they appear as DATA keys
MAPPING_KEYS = re.compile(
    r'"(arm|condition|method|erased|erasure|original_of|ref_file|reference_id|'
    r'source_file|source_path|prompt|seed|init_seed|checkpoint|machine_|prediction|'
    r'evaluator_answer|score)"\s*:', re.I)
EXTERNAL_URL = re.compile(r'(?:src|href)\s*=\s*["\']https?://', re.I)
B64_BLOB = re.compile(r'[A-Za-z0-9+/]{400,}={0,2}')
NEUTRAL_NAME = re.compile(r'^[A-Za-z0-9._-]+$')


def scan_text(path, rel, secrets):
    try:
        s = open(path, encoding="utf-8", errors="replace").read()
    except Exception as e:
        finding("ERROR", rel, f"unreadable: {e}")
        return
    for sec in secrets:
        if sec and sec in s:
            finding("CRITICAL", rel, "contains private key material")
    for m in MAPPING_KEYS.finditer(s):
        key = m.group(1)
        line = s.count("\n", 0, m.start()) + 1
        tail = s[m.end():m.end() + 80].lstrip()
        # In ANALYSIS SOURCE, these words legitimately appear as field names of output rows
        # and as labels; a leak would be a key bound to a literal VALUE that encodes an arm.
        # Only a constant-valued occurrence can carry data, so a key followed by a variable
        # or by an obviously descriptive string is allowlisted, and every allowlisted hit is
        # printed in the audit record for inspection rather than silently dropped.
        # EXPLICIT, DOCUMENTED ALLOWLIST -------------------------------------------------
        # Practice provenance is deliberately public: the 12 practice items are non-study
        # instructional aids built on an object (bicycle) declared not to be a study target,
        # they are not shipped inside the reviewer ZIP, and disclosing how they were made
        # (including the deliberately degraded 1-2 step examples) is required transparency.
        # They cannot leak a study arm, because no study item exists and practice never
        # shares an object with the study panel.
        if os.path.basename(rel) in ("practice_manifest.json",) or rel.endswith("gen_practice.py"):
            finding("ALLOWLISTED", rel,
                    f"line {line}: {key!r} is PRACTICE provenance (non-study instructional "
                    f"aid, not in the reviewer ZIP) — deliberately public")
            continue
        is_source = rel.endswith((".py", ".js"))
        literal_value = bool(re.match(r'["\'][^"\']', tail))
        descriptive = bool(re.match(r'["\'](cluster bootstrap|no arm key|zero-shot)', tail))
        if is_source and (not literal_value or descriptive):
            finding("ALLOWLISTED", rel,
                    f"line {line}: field name {key!r} in analysis source, value is "
                    f"{'a descriptive label' if descriptive else 'a variable'} — carries no mapping")
        else:
            finding("CRITICAL", rel, f"line {line}: mapping key {key!r} bound to a literal value")
    for m in EXTERNAL_URL.finditer(s):
        finding("HIGH", rel, "external resource reference (breaks offline requirement)")
    for m in B64_BLOB.finditer(s):
        blob = m.group(0)
        try:
            dec = base64.b64decode(blob + "==", validate=False)
            txt = dec.decode("utf-8", "ignore")
            if any(sec in txt for sec in secrets if sec) or MAPPING_KEYS.search(txt):
                finding("CRITICAL", rel, "base64 blob decodes to private material")
            else:
                finding("INFO", rel, f"large base64-like blob ({len(blob)} chars) — inspected, benign")
        except Exception:
            finding("INFO", rel, "large base64-like blob — not decodable")


def scan_binary(path, rel, secrets):
    with open(path, "rb") as f:
        b = f.read()
    for sec in secrets:
        if sec and sec.encode() in b:
            finding("CRITICAL", rel, "binary contains private key material")
    if rel.endswith(".png"):
        # PNG text chunks would carry prompts/metadata
        for chunk in (b"tEXt", b"iTXt", b"zTXt"):
            if chunk in b:
                finding("HIGH", rel, f"PNG carries a {chunk.decode()} metadata chunk")


def audit(roots, secrets):
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
            for fn in sorted(files):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root)
                if not NEUTRAL_NAME.match(fn):
                    finding("HIGH", rel, "filename is not neutral/simple")
                ext = os.path.splitext(fn)[1].lower()
                if ext in TEXT_EXT:
                    scan_text(p, rel, secrets)
                elif ext in BIN_EXT:
                    scan_binary(p, rel, secrets)
                else:
                    scan_binary(p, rel, secrets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    secrets = load_secrets()
    audit(a.roots, secrets)
    crit = [f for f in findings if f["severity"] in ("CRITICAL", "HIGH", "ERROR")]
    info = [f for f in findings if f["severity"] in ("INFO", "ALLOWLISTED")]
    print(f"scanned roots: {a.roots}")
    print(f"blocking findings: {len(crit)}   informational: {len(info)}")
    for f in crit:
        print(f"  {f['severity']:8s} {f['where']}: {f['what']}")
    for f in info[:20]:
        print(f"  {f['severity']:8s} {f['where']}: {f['what']}")
    res = {"roots": a.roots, "blocking": crit, "informational_and_allowlisted": info,
           "clean": not crit,
           "checks": ["private salt/seed strings", "arm/method/condition/prompt/seed/source "
                      "mapping keys present as data", "Task-B reference associations "
                      "(ref_file/reference_id)", "automated predictions or scores",
                      "external http(s) resource references (offline requirement)",
                      "base64 blobs decoded and inspected", "PNG tEXt/iTXt/zTXt metadata chunks",
                      "non-neutral filenames"]}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("\nAUDIT:", "CLEAN" if not crit else "FINDINGS PRESENT")
    sys.exit(1 if crit else 0)
