#!/usr/bin/env python3
"""Assemble the public results tree and the versioned Task-A ZIP.

Publishes ONLY Task-A material. No Task-B package, no pair associations, no
private previews, no mapping, no salt.
"""
import hashlib
import json
import os
import shutil
import sys
import zipfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
OUT = os.path.join(ROOT, "publish", "results_tree")
ZIPNAME = "teapot-human-review-v1-taskA-patch01.zip"
ZPATH = os.path.join(ROOT, "publish", ZIPNAME)
ZROOT = "teapot-human-review-v1-taskA-patch01"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    for d in ("protocol", "scripts", "qa", "verify", "ledger"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    # ---- the reviewer ZIP (built first so it can be hashed into the manifest) ----
    src = os.path.join(ROOT, "taskA")
    files = ["index.html", "app.js", "items.js", "style.css", "items.json"]
    imgs = sorted(f for f in os.listdir(os.path.join(src, "images"))
                  if f.endswith(".png"))
    assert len(imgs) == 40, f"expected 40 images, found {len(imgs)}"

    if os.path.exists(ZPATH):
        os.remove(ZPATH)

    # Deterministic archive: fixed member timestamps and fixed external attrs,
    # so rebuilding from the same inputs reproduces the same sha256.
    FIXED_DATE = (2026, 10, 18, 0, 0, 0)

    def add(z, data, arcname):
        zi = zipfile.ZipInfo(arcname, date_time=FIXED_DATE)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, data)

    with zipfile.ZipFile(ZPATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            add(z, open(os.path.join(src, f), "rb").read(), f"{ZROOT}/{f}")
        add(z, open(os.path.join(ROOT, "protocol",
                                 "INSTRUCTIONS_TASK_A.md"), "rb").read(),
            f"{ZROOT}/INSTRUCTIONS.md")
        for im in imgs:
            # stored verbatim; PNG bytes are byte-identical to the published source
            add(z, open(os.path.join(src, "images", im), "rb").read(),
                f"{ZROOT}/images/{im}")
        add(z, (
                   "Task A reviewer bundle\n"
                   "======================\n\n"
                   "Extract this folder fully, then open index.html in a browser.\n"
                   "It works entirely offline: no account, no upload, no tracking.\n"
                   "Your answers stay in your browser until you export a file.\n\n"
                   "See INSTRUCTIONS.md for what to do.\n\n"
                   "package THR-V1-A | protocol TEAPOT-HR-PROTO-1.0.0 | "
                   "schema THR-EXPORT-SCHEMA-1.1.0 | app 1.1.0 patch01 | 40 images\n").encode(),
            f"{ZROOT}/README.txt")

    zsha, zbytes = sha256_file(ZPATH), os.path.getsize(ZPATH)

    # ---- copy public documents ----
    def cp(srcp, rel):
        d = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(srcp, d)

    # hand-written top-level documents live in docs/ so that rebuilding the
    # generated tree cannot destroy them
    for f in ("README.md", "MASKING_AND_LIMITATIONS.md",
              "PREPARATION_ACCOUNTING.md", "PRIVATE_RECOVERY.md",
              "AMENDMENT_01_TASKA_PATCH01.md"):
        cp(os.path.join(ROOT, "docs", f), f)
    for f in ("PROTOCOL.md", "SCHEMA.md", "INSTRUCTIONS_TASK_A.md",
              "taskA_items.json", "COMMITMENTS.json"):
        cp(os.path.join(ROOT, "protocol", f), f"protocol/{f}")
    for f in ("validate_export.py", "verify_source.py", "build_taskA.py",
              "build_taskB.py", "private_backup.py",
              "masking_audit.py", "browser_qa.py", "make_fixtures_and_qa.py",
              "build_publish.py", "freeze_ids.py", "qa_patch01.py",
              "verify_published.py"):
        p = os.path.join(ROOT, "scripts", f)
        if os.path.isfile(p):
            cp(p, f"scripts/{f}")
    cp(os.path.join(ROOT, "ledger", "prep_ledger.py"), "scripts/prep_ledger.py")
    for f in ("validator_qa_report.json", "browser_qa_report.json",
              "qa_patch01_report.json"):
        cp(os.path.join(ROOT, "qa", f), f"qa/{f}")
    cp(os.path.join(ROOT, "verify", "taskB_completeness_summary.json"),
       "verify/taskB_completeness_summary.json")

    # Source-integrity and Task-A completeness are published as SUMMARIES.
    # Their per-file rows pair (case, condition) with a sha256 on one side and a
    # neutral item ID with a sha256 on the other; publishing both row sets side
    # by side would let anyone join on the hash and read off the entire mapping
    # with no effort at all. The verdicts and counts are what a reader needs.
    sv = json.load(open(os.path.join(ROOT, "verify", "source_verification.json")))
    sv_pub = {k: v for k, v in sv.items() if k != "rows"}
    sv_pub["rows"] = "REDACTED -- per-file (case, condition) -> sha256 rows are held privately"
    sv_pub["local_archive"] = {k: v for k, v in sv["local_archive"].items()
                               if k in ("bytes", "sha256", "bytes_match", "sha256_match")}
    sv_pub["redaction_note"] = (
        "Row detail withheld so that this file cannot be joined against the "
        "Task-A per-item hashes to recover the neutral-ID mapping. The images "
        "themselves remain public in the modern-object-assets-v1 release.")
    json.dump(sv_pub, open(os.path.join(OUT, "verify", "source_verification_summary.json"), "w"), indent=1)

    ta = json.load(open(os.path.join(ROOT, "verify", "taskA_completeness.json")))
    ta_pub = {k: v for k, v in ta.items() if k != "rows"}
    ta_pub["rows"] = "REDACTED -- per-item sha256 rows are held privately"
    ta_pub["redaction_note"] = (
        "Per-item hashes withheld for the same reason. The published ZIP has a "
        "single sha256 that anyone can verify, and the member inventory is "
        "published in full.")
    json.dump(ta_pub, open(os.path.join(OUT, "verify", "taskA_completeness_summary.json"), "w"), indent=1)
    # The repository gitignores *.png by standing policy, so the UI previews
    # are published as release assets rather than committed. They are staged
    # here for upload, not copied into the results tree.
    stage = os.path.join(ROOT, "publish", "preview_assets")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    # EXPLICIT ALLOWLIST. Staging "everything in the screenshots directory"
    # once published two item-view screenshots that each showed a real research
    # image beside its neutral ID. Only these four files may ever be published:
    # two instructions pages (no research image at all) and two item-view
    # previews rendered against a clearly synthetic placeholder.
    PUBLISHABLE_PREVIEWS = [
        "A1_instructions.png",
        "A3_mobile_instructions.png",
        "A2_item_view_synthetic.png",
        "A4_mobile_item_synthetic.png",
    ]
    for f in PUBLISHABLE_PREVIEWS:
        src_p = os.path.join(ROOT, "qa", "screenshots", f)
        if not os.path.isfile(src_p):
            raise SystemExit(f"publishable preview missing: {f}")
        shutil.copyfile(src_p, os.path.join(stage, "preview_" + f))
    extra = [f for f in sorted(os.listdir(os.path.join(ROOT, "qa", "screenshots")))
             if f not in PUBLISHABLE_PREVIEWS]
    if extra:
        print(f"[publish] NOT staging {len(extra)} non-allowlisted screenshot(s): {extra}")
    cp(os.path.join(ROOT, "ledger", "prep_ledger.jsonl"), "ledger/prep_ledger.jsonl")
    cp(os.path.join(ROOT, "ledger", "clock.json"), "ledger/clock.json")

    # the Task-A ZIP is a release asset, not a Git file; record its identity
    json.dump({
        "release_asset": ZIPNAME,
        "supersedes": "teapot-human-review-v1-taskA.zip (original, preserved)",
        "app_version": "1.1.0",
        "schema_id": "THR-EXPORT-SCHEMA-1.1.0",
        "bytes": zbytes, "sha256": zsha,
        "zip_root": ZROOT,
        "n_members": 40 + len(files) + 2,
        "n_images": 40,
        "contains_task_b": False,
        "contains_answer_key": False,
        "contains_mapping_or_salt": False,
    }, open(os.path.join(OUT, "TASKA_ZIP_IDENTITY.json"), "w"), indent=1)

    print(f"zip: {ZIPNAME}  {zbytes:,} bytes  sha256={zsha}")
    print(f"public tree: {OUT}")
    print(f"files: {sum(len(f) for _,_,f in os.walk(OUT))}")


if __name__ == "__main__":
    main()
