#!/usr/bin/env python3
"""Audit everything destined for publication for leaked associations.

Scans file contents, filenames, code, comments, JSON keys and PNG container
metadata for: condition labels, case IDs, seeds, prompts, source filenames,
Task-B pair associations, answer keys, or the private salt/mapping.

Also checks the actual ZIP members. Generic shared renderer code with no
associations is acceptable and is not flagged.
"""
import json
import os
import re
import struct
import sys
import zipfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
SRCMAN = "/data/bijaypandey/cuig_pilot/modern_object_assets_v1/manifests"

# association-bearing tokens that must never appear in public artifacts
PATTERNS = [
    ("source filename", re.compile(r"CASE-\d{2}_(m0|edited)_seed\d+\.png")),
    ("case id", re.compile(r"\bCASE-\d{2}\b")),
    ("eval seed", re.compile(r"\bseed ?7\d{3}\b", re.I)),
    ("condition key", re.compile(r'"condition"\s*:\s*"(m0|edited)"')),
    ("pair association", re.compile(r'"(reference_filename|candidate_filename|'
                                    r'reference_sha256|candidate_sha256)"')),
    ("private mapping key", re.compile(r'"(taskA_mapping|taskB_mapping|'
                                       r'source_filename|source_sha256)"')),
    ("answer key", re.compile(r'"(answer_key|ground_truth|gold|label_key)"')),
]
PROMPT_WORDS = None  # loaded from the frozen cases


def load_prompts():
    p = os.path.join(SRCMAN, "CASES.json")
    return [c["prompt"] for c in json.load(open(p))["cases"]]


def scan_text(path, rel):
    hits = []
    try:
        data = open(path, "rb").read()
    except OSError:
        return hits
    try:
        txt = data.decode("utf-8")
    except UnicodeDecodeError:
        return hits          # binary; handled separately
    # Field NAMES appearing in source code are not associations; the published
    # build scripts legitimately declare keys like "source_filename" while
    # carrying no mapping data. Structural key patterns are therefore checked
    # only in data files; value-level patterns are checked everywhere.
    is_code = rel.endswith((".py", ".js", ".css", ".html"))
    KEYNAME_KINDS = {"private mapping key", "pair association", "answer key"}
    for label, rx in PATTERNS:
        if is_code and label in KEYNAME_KINDS:
            continue
        for m in rx.finditer(txt):
            hits.append({"file": rel, "kind": label, "match": m.group(0)[:80]})
    for pr in PROMPT_WORDS:
        # The bare target term ("a teapot") is disclosed to reviewers by design
        # and is not case-identifying. Only distinctive prompts are leaks.
        if len(pr) >= 15 and pr.lower() in txt.lower():
            hits.append({"file": rel, "kind": "verbatim case prompt",
                         "match": pr[:60]})
    return hits


def png_meta(path):
    """Return any textual/metadata chunk types present."""
    found = []
    with open(path, "rb") as fh:
        if fh.read(8) != b"\x89PNG\r\n\x1a\n":
            return ["not-a-png"]
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            ln, typ = struct.unpack(">I4s", hdr)
            t = typ.decode("latin1")
            fh.read(ln); fh.read(4)
            if t in ("tEXt", "iTXt", "zTXt", "eXIf", "tIME"):
                found.append(t)
            if t == "IEND":
                break
    return found


def main():
    global PROMPT_WORDS
    PROMPT_WORDS = load_prompts()

    targets = []
    pubdir = os.path.join(ROOT, "publish", "results_tree")
    zpath = os.path.join(ROOT, "publish", "teapot-human-review-v1-taskA-patch01.zip")

    if os.path.isdir(pubdir):
        for dp, _, fns in os.walk(pubdir):
            for f in fns:
                p = os.path.join(dp, f)
                targets.append((p, os.path.relpath(p, pubdir)))

    hits = []
    for p, rel in targets:
        hits += scan_text(p, rel)

    # filenames themselves
    name_hits = []
    for _, rel in targets:
        for label, rx in PATTERNS:
            if rx.search(rel):
                name_hits.append({"file": rel, "kind": "filename:" + label})

    # ---- ZIP members ----
    zip_report = {}
    if os.path.isfile(zpath):
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            zip_report["n_members"] = len(names)
            zip_report["png_members"] = sum(1 for n in names if n.endswith(".png"))
            bad_names = [n for n in names
                         for label, rx in PATTERNS if rx.search(n)]
            zip_report["member_name_leaks"] = sorted(set(bad_names))
            zhits, meta = [], {}
            for n in names:
                if n.endswith("/"):
                    continue
                data = z.read(n)
                if n.endswith(".png"):
                    tmp = os.path.join(ROOT, "qa", "_m.png")
                    open(tmp, "wb").write(data)
                    m = png_meta(tmp)
                    if m:
                        meta[n] = m
                    os.remove(tmp)
                    continue
                try:
                    txt = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                zis_code = n.endswith((".py", ".js", ".css", ".html"))
                for label, rx in PATTERNS:
                    if zis_code and label in {"private mapping key",
                                              "pair association", "answer key"}:
                        continue
                    for mm in rx.finditer(txt):
                        zhits.append({"member": n, "kind": label,
                                      "match": mm.group(0)[:80]})
                for pr in PROMPT_WORDS:
                    if len(pr) >= 15 and pr.lower() in txt.lower():
                        zhits.append({"member": n, "kind": "verbatim case prompt",
                                      "match": pr[:60]})
                # no synthetic test export may ship inside the reviewer bundle
                if re.search(r'"_fixture"|SYNTHETIC QA FIXTURE', txt):
                    zhits.append({"member": n, "kind": "synthetic test export"})
                # nor any completed response set
                if '"responses"' in txt and '"export_kind"' in txt:
                    zhits.append({"member": n, "kind": "completed export in bundle"})
                if re.search(r'B-\d{2}_(reference|candidate)', txt):
                    zhits.append({"member": n, "kind": "Task-B association"})
            zip_report["content_leaks"] = zhits
            zip_report["png_metadata_chunks"] = meta
            zip_report["png_containers_clean"] = not meta
            zip_report["taskB_members"] = [n for n in names
                                           if re.search(r"\bB-\d{2}\b", n)]
            zip_report["item_ids_neutral"] = all(
                re.fullmatch(r"A-\d{3}", os.path.basename(n)[:-4]) or
                not n.endswith(".png")
                for n in names)

    # ---- salt / private material must not be reachable ----
    salt_leak = []
    salt_path = os.path.join(ROOT, "private", "salt.bin")
    if os.path.isfile(salt_path):
        salt_hex = open(salt_path, "rb").read().hex()
        for p, rel in targets:
            try:
                if salt_hex in open(p, "r", errors="ignore").read():
                    salt_leak.append(rel)
            except OSError:
                pass

    # ---- staged preview assets + the LIVE release asset list ----
    # A previous run published two item-view screenshots that each showed a
    # real research image beside its neutral ID. The audit used to look only at
    # the results tree and the ZIP, so it could not see them. It now checks the
    # staged preview directory and what is actually live on each release.
    ALLOWED_PREVIEWS = {
        "preview_A1_instructions.png", "preview_A3_mobile_instructions.png",
        "preview_A2_item_view_synthetic.png", "preview_A4_mobile_item_synthetic.png",
    }
    preview_report = {}
    stage_dir = os.path.join(ROOT, "publish", "preview_assets")
    if os.path.isdir(stage_dir):
        staged = sorted(os.listdir(stage_dir))
        preview_report["staged"] = staged
        preview_report["staged_disallowed"] = sorted(set(staged) - ALLOWED_PREVIEWS)
    live = {}
    try:
        import urllib.request
        for tag in ("teapot-human-review-v1-taskA",
                    "teapot-human-review-v1-taskA-patch01"):
            rq = urllib.request.Request(
                f"https://api.github.com/repos/bijay299/XXResearch/releases/tags/{tag}",
                headers={"User-Agent": "thr-audit",
                         "Accept": "application/vnd.github+json"})
            rel = json.loads(urllib.request.urlopen(rq, timeout=60).read())
            names = sorted(a["name"] for a in rel["assets"])
            bad_live = [n for n in names
                        if n.endswith(".png") and n not in ALLOWED_PREVIEWS]
            live[tag] = {"assets": names, "disallowed_preview_assets": bad_live}
    except Exception as e:
        live["error"] = str(e)
    preview_report["live_releases"] = live
    preview_clean = (not preview_report.get("staged_disallowed")
                     and all(not v.get("disallowed_preview_assets")
                             for k, v in live.items() if isinstance(v, dict)))
    preview_report["clean"] = preview_clean

    report = {
        "scanned_public_files": len(targets),
        "preview_and_release_assets": preview_report,
        "content_leaks": hits,
        "filename_leaks": name_hits,
        "salt_appears_in_public_files": salt_leak,
        "zip": zip_report,
        "clean": (preview_clean and not hits and not name_hits and not salt_leak
                  and not zip_report.get("content_leaks")
                  and not zip_report.get("member_name_leaks")
                  and not zip_report.get("taskB_members")
                  and zip_report.get("png_containers_clean", True)),
        "limitations": [
            "The 40 images and their condition/pair provenance are ALREADY PUBLIC "
            "in the modern-object-assets-v1 release. Neutral IDs reduce accidental "
            "disclosure during review; they cannot prevent a determined reverse "
            "match and do not guarantee blinding.",
            "Public integrity hashes are not secret. Anyone can hash the public "
            "release images and match them to the published Task-A file hashes, "
            "recovering the mapping. They are published for verifiability, with "
            "that trade-off accepted and disclosed.",
            "Nothing here makes the already-public source provenance private.",
            "Task B deliberately discloses reference identity and is only "
            "partially masked.",
            "Self-declared exposure answers cannot be verified from the artifacts.",
        ],
    }
    os.makedirs(os.path.join(ROOT, "verify"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "verify", "masking_audit.json"), "w"),
              indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "limitations"},
                     indent=1)[:2600])
    sys.exit(0 if report["clean"] else 5)


if __name__ == "__main__":
    main()
