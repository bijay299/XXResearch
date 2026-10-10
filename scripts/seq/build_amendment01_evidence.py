#!/usr/bin/env python3
"""Complete the Amendment 01 evidence archive. CPU only.

What this adds to the bundle that was published at commit 7e01fbd
-----------------------------------------------------------------
1. ``unattended.log`` -- listed in the published MANIFEST but ABSENT from the
   commit, because ``.gitignore`` excludes ``*.log``. It is supplied here and
   its sha256 is checked against the identity the MANIFEST already recorded, so
   the file is proved to be the one that was listed rather than a later
   substitute. If it cannot be found or its identity does not match, that is
   written into the manifest as an explicit access limitation instead of being
   passed over.

2. The RAW DETECTOR ROWS for the development scan and the frozen test, plus
   their generation and detection reports, as two gzipped text archives. With
   these, selection and every interval can be re-derived WITHOUT the bulk
   images (413 MB test + 179 MB dev) and WITHOUT the weights.

3. The CORRECTED analysis, recorded as a distinct version alongside -- never
   over -- the two previous analysis records. Nothing is overwritten.

Deliberate exclusions, unchanged: model weights, generated images, and the
annotation arm keys. No key, and nothing derived from one, enters this bundle.

    python scripts/seq/build_amendment01_evidence.py \
        --diag_root /data/.../diag_v2_early_grid \
        --out results/audit_v1/amendment01_evidence
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

SLOT_FILES = ("detections.jsonl", "image_report.json", "detect_report.json",
              "_stage_complete.json")


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag_root", required=True)
    ap.add_argument("--seq_root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    diag, S, out = Path(a.diag_root), Path(a.seq_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    # The identities the PUBLISHED manifest recorded, used to CHECK the files we
    # supply rather than to describe them. Once preserved, the published copy --
    # not whatever this script last wrote -- stays the baseline, so rebuilding
    # the bundle cannot quietly re-baseline it against itself.
    pub_p = out / "MANIFEST.published_at_7e01fbd.json"
    cur_p = out / "MANIFEST.json"
    prev_p = pub_p if pub_p.is_file() else cur_p
    prev = json.loads(prev_p.read_text()) if prev_p.is_file() else {}
    recorded: dict[str, dict] = prev.get("files") or {}
    baseline = ("the manifest published at 7e01fbd" if prev_p == pub_p
                else "the manifest found in place at build time")

    files: dict[str, dict] = {}
    limitations: list[dict] = []

    # Payloads that are DELIBERATELY a new version of a previously published
    # file. A changed digest here is a recorded supersession, not a gap -- the
    # earlier bytes are preserved under their own name in the same bundle.
    superseding = {"analysis_test.json":
                   "analysis_test.20261010T205657.superseded.json"}

    def take(src: Path, name: str, note: str) -> None:
        entry: dict = {"source": str(src), "note": note}
        if not src.is_file():
            entry.update({"status": "ABSENT"})
            files[name] = entry
            limitations.append({
                "file": name, "expected_source": str(src),
                "limitation": "not present at the recorded source path",
                "consequence": ("this payload is not in the bundle and cannot "
                                "be independently hash-checked"),
            })
            return
        dst = out / name
        dst.write_bytes(src.read_bytes())
        digest, size = sha256_file(dst), dst.stat().st_size
        entry.update({"sha256": digest, "bytes": size})
        was = recorded.get(name)
        if was and "sha256" in was:
            same = (was["sha256"] == digest and was.get("bytes") == size)
            entry["matches_previously_recorded_identity"] = same
            entry["previously_recorded"] = {"sha256": was["sha256"],
                                            "bytes": was.get("bytes")}
            if not same and name in superseding:
                entry["matches_previously_recorded_identity"] = None
                entry["supersedes_published_version"] = {
                    "published_at_7e01fbd": was,
                    "those_bytes_preserved_in_this_bundle_as": superseding[name],
                    "why": ("a NEW analysis version, written by the repaired "
                            "diag_analysis.py. The published version is kept "
                            "verbatim beside it; nothing was overwritten."),
                }
            elif not same:
                limitations.append({
                    "file": name,
                    "limitation": ("the supplied bytes do NOT match the "
                                   "identity the published manifest recorded"),
                    "recorded": was, "supplied": {"sha256": digest, "bytes": size},
                    "consequence": ("treat as a DIFFERENT artifact; the "
                                    "published listing is not evidence for it"),
                })
        files[name] = entry

    # ---- decision, bridge, binding and accounting records -----------------
    take(diag / "STATUS.json", "STATUS.json",
         "supervisor state machine and its full history")
    take(diag / "selection.json", "selection.json",
         "the DEVELOPMENT-set selection decision, with gate outcomes and the "
         "sha256 of every development input it read")
    take(diag / "bridge_check.json", "bridge_check.json",
         "the step-100 bridge between the original run and the rerun")
    take(diag / "reference_bindings.json", "reference_bindings.json",
         "the four-way binding of the reused MA and L2 reference evaluations")
    take(S / "matched_effectiveness_gpu_budget.json", "gpu_usage_record.json",
         "the experiment's device-hour ledger, all prior spend carried forward")

    # ---- the analysis chain: every version, none overwritten --------------
    take(diag / "analysis_test.json", "analysis_test.json",
         "CURRENT analysis (v3): re-checks the SAME §5 GATES and the match on "
         "TEST. Eligibility fails on both seeds; tolerance additionally fails "
         "on seed 29; no seed supplies a valid matched comparison")
    for p in sorted(diag.glob("analysis_test.*.superseded*.json")):
        take(p, p.name, "PRESERVED earlier analysis record, not overwritten")

    # ---- the supervisor log: the file the published manifest listed --------
    log = diag / "logs" / "unattended.log"
    take(log, "unattended.log",
         "the unattended supervisor log, verbatim. Listed in the manifest "
         "published at 7e01fbd but absent from that commit because "
         ".gitignore excludes '*.log'; supplied here and force-added to git")
    if files["unattended.log"].get("matches_previously_recorded_identity") is True:
        files["unattended.log"]["resolves"] = (
            "the archive gap reported at 7e01fbd: the bytes supplied are the "
            "ones the published manifest already identified")

    # ---- raw detector rows + reports, per set, text only ------------------
    def archive(set_dir: Path, tar_name: str, index_name: str,
                what: str) -> None:
        if not set_dir.is_dir():
            limitations.append({
                "file": tar_name, "expected_source": str(set_dir),
                "limitation": "evaluation directory absent",
                "consequence": f"{what} cannot be re-derived from this bundle",
            })
            return
        tar_p, slots, n_rows = out / tar_name, [], 0

        def flat(ti: tarfile.TarInfo) -> tarfile.TarInfo:
            # Normalise everything that is not content, so rebuilding the
            # archive from the same bytes yields the same digest.
            ti.mtime, ti.uid, ti.gid = 0, 0, 0
            ti.uname = ti.gname = ""
            ti.mode = 0o644
            return ti

        # gzip stamps an mtime into its header; pin it to 0 for the same reason.
        with gzip.GzipFile(filename="", mode="wb", mtime=0,
                           fileobj=tar_p.open("wb")) as gz, \
             tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tf:
            for d in sorted(set_dir.iterdir()):
                if not d.is_dir():
                    continue
                entry: dict = {"slot": d.name}
                for fn in SLOT_FILES:
                    f = d / fn
                    if f.is_file():
                        tf.add(f, arcname=f"{d.name}/{fn}", filter=flat)
                        entry[fn] = {"sha256": sha256_file(f),
                                     "bytes": f.stat().st_size}
                    else:
                        entry[fn] = {"status": "ABSENT"}
                det = d / "detections.jsonl"
                if det.is_file():
                    k = sum(1 for ln in det.read_text().splitlines() if ln.strip())
                    entry["detector_rows"] = k
                    n_rows += k
                ir = d / "image_report.json"
                if ir.is_file():
                    j = json.loads(ir.read_text())
                    entry["generating_checkpoint_sha256"] = (
                        (j.get("checkpoint_load") or {}).get("sha256"))
                    entry["manifest_sha256"] = j.get("manifest_sha256")
                    entry["n_generated"] = j.get("n_generated")
                    entry["n_reused"] = j.get("n_reused")
                    entry["generation_complete"] = j.get("complete")
                slots.append(entry)
        (out / index_name).write_text(json.dumps({
            "_what_this_is": what,
            "archive": tar_name,
            "contents": ("per-slot detections.jsonl (one row per image, the raw "
                         "detector output), image_report.json (per-image "
                         "generation record and digests), detect_report.json and "
                         "the stage-validation marker. NO image files, NO weights."),
            "n_slots": len(slots), "total_detector_rows": n_rows,
            "slots": slots}, indent=2) + "\n")
        for f in (tar_p, out / index_name):
            files[f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size,
                             "source": str(set_dir),
                             "note": f"{what} ({n_rows} detector rows over "
                                     f"{len(slots)} slots)"}

    archive(diag / "eval_dev", "development_slots.tar.gz",
            "development_slots_index.json",
            "the DEVELOPMENT scan: the 20 candidate dumps (10 early steps x 2 "
            "seeds) that selection scored")
    archive(diag / "eval_test", "frozen_test_slots.tar.gz",
            "frozen_test_slots_index.json",
            "the FROZEN TEST evaluation: 6 slots (MA, L2, U*=step 70 per seed), "
            "evaluated once")

    # ---- device-hour reconciliation, recomputed not re-engineered ---------
    led_p = out / "gpu_usage_record.json"
    budget: dict = {}
    if led_p.is_file():
        led = json.loads(led_p.read_text())
        es = led.get("entries") or []
        carried = [e for e in es if e.get("carried_forward_from")]
        new = [e for e in es if not e.get("carried_forward_from")]
        wall = lambda rows: sum(r.get("wall_seconds", 0) for r in rows) / 3600.0
        budget = {
            "_what_this_is": ("A RECOMPUTATION of the existing ledger, for "
                              "cross-checking only. No entry was added, "
                              "removed, re-weighted or re-dated."),
            "n_entries_total": len(es),
            "n_entries_carried_forward": len(carried),
            "n_entries_this_amendment": len(new),
            "device_hours_total_from_wall_seconds": round(wall(es), 7),
            "device_hours_this_amendment_from_wall_seconds": round(wall(new), 7),
            "device_hours_carried_forward_from_wall_seconds":
                round(wall(carried), 7),
            "ledger_stored_spent_gpu_hours": led.get("spent_gpu_hours"),
            "rounding_note": (
                "the ledger's stored total is the sum of per-entry gpu_hours, "
                "each rounded at write time; recomputing from wall_seconds "
                "gives a value ~2e-6 h (about 8 ms) lower. Both are recorded; "
                "neither was altered."),
            "all_entries_ok": all(e.get("outcome") == "ok" for e in es),
            "n_open_reservations": len(led.get("open") or []),
            "ceiling": ("RETIRED (PI, 2026-10-10). The ledger remains a record; "
                        "it caps nothing. Compute is gated by verified GPU "
                        "availability instead."),
        }
        (out / "budget_reconciliation.json").write_text(
            json.dumps(budget, indent=2) + "\n")
        f = out / "budget_reconciliation.json"
        files[f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size,
                         "note": "recomputed cross-check of the existing ledger"}

    manifest = {
        "_what_this_is": (
            "Amendment 01 evidence: the decision records, the bridge, the "
            "reference bindings, every analysis version, the supervisor log, "
            "the GPU usage record, and the RAW per-image detector rows and "
            "stage reports for both the development scan and the frozen test. "
            "Bulk weights and images stay private; the annotation arm keys are "
            "NOT here."),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit_at_build": commit,
        "execution_commit": prev.get("execution_commit",
                                     "0cd9dc611b737e32774c631930ddf5a4199edc27"),
        "output_root": str(diag),
        "outcome": (
            "DEVELOPMENT matches were found at step 70 on both seeds. On the "
            "FROZEN TEST set the prespecified §5 conditions FAIL: both arms of "
            "both seeds fall below the 30 pp suppression gate, and seed 29 "
            "additionally exceeds the 5 pp match tolerance (6.25 pp). NEITHER "
            "seed supplies a valid matched comparison in the prespecified "
            "regime, so the matched-retention question is INCONCLUSIVE there. "
            "The bird contrasts and intervals are retained as descriptive "
            "results. Human labels remain pending."),
        "reproducibility": (
            "selection and every interval can be re-derived from this bundle "
            "alone: development_slots.tar.gz and frozen_test_slots.tar.gz carry "
            "the raw per-image detector rows, and the frozen bootstrap grouping "
            "is committed at results/audit_v1/draft_manifests/."),
        "deliberate_exclusions": {
            "model_weights": ("delta.bin and the 20 early dumps are NOT "
                              "included; their sha256 digests are in "
                              "selection.json and bridge_check.json"),
            "generated_images": ("~592 MB (413 MB frozen test + 179 MB "
                                 "development) are NOT included; per-image "
                                 "digests are in each slot's image_report.json "
                                 "inside the archives"),
            "annotation_arm_key": ("NOT included and not referenced. Both the "
                                   "supplementary 228-item packet's key and the "
                                   "prescribed 180-item paired packet's key stay "
                                   "closed until labels are frozen; nothing from "
                                   "either packet appears in this bundle."),
        },
        "access_limitations": limitations or [
            {"none": "every listed payload is present and hash-checked"}],
        "budget_reconciliation": budget,
        "provisional": ("Detector output is a PROXY. No final scientific "
                        "conclusion follows until the blinded human annotation "
                        "is complete; no labels exist yet."),
        "files": files,
    }
    if prev and not pub_p.is_file():
        pub_p.write_text(json.dumps(prev, indent=2) + "\n")
    if pub_p.is_file():
        manifest["supersedes"] = {
            "preserved_at": pub_p.name,
            "note": ("the manifest published at 7e01fbd, kept verbatim so the "
                     "listing that named an absent unattended.log stays on the "
                     "record"),
        }
        manifest["identity_baseline"] = baseline
        manifest["files"][pub_p.name] = {
            "sha256": sha256_file(pub_p), "bytes": pub_p.stat().st_size,
            "note": "PRESERVED previous manifest, not overwritten"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("=== Amendment 01 evidence archive ===")
    for name in sorted(files):
        e = files[name]
        if e.get("status") == "ABSENT":
            print(f"  ABSENT  {name}")
        else:
            flag = e.get("matches_previously_recorded_identity")
            mark = ("  [supersedes published version]"
                    if "supersedes_published_version" in e else
                    "" if flag is None else
                    "  [identity OK]" if flag else "  [IDENTITY MISMATCH]")
            print(f"  {e['bytes']:>10}  {e['sha256'][:12]}…  {name}{mark}")
    print(f"  access limitations: {len(limitations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
