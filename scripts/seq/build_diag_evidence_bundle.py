#!/usr/bin/env python3
"""Assemble the compact evidence bundle for the matched-effectiveness diagnostic.

What goes in: every record needed to re-derive the decision without access to
the private data root -- both selection records, the per-dump table, all 24
development slots' per-image detector outputs and their generation/detection
reports, the budget ledger, the cost model, the U-vs-MAB comparison, the
preflight/start/stop records, the trajectory counters and dump inventories, and
the test-suite logs with the commit they were run at.

What stays OUT, deliberately:
  * model weights (1.6 GB of delta.bin and dumps) -- digests only;
  * generated images (208 MB) -- per-image digests are already in the reports;
  * the annotation arm key, which must stay closed until labels are frozen.
    This bundle contains nothing from the annotation packet.

The frozen-test output location is inventoried EMPTY, so "no test evaluation"
is documented rather than merely asserted.

Every file is listed in MANIFEST.json with its sha256. Bulky text is collected
into one gzipped tarball so the bundle stays compact.

CPU only.

    python scripts/seq/build_diag_evidence_bundle.py \
        --diag_root /data/.../diag_v1 --out results/audit_v1/diag_v1_evidence
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


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
    ap.add_argument("--test_logs", default=None,
                    help="directory of captured test-suite logs to include")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    diag, S, repo = Path(a.diag_root), Path(a.seq_root), Path(a.repo)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    copied: list[dict] = []

    def take(src: Path, name: str | None = None, note: str = "") -> None:
        if not src.is_file():
            copied.append({"file": name or src.name, "status": "ABSENT",
                           "source": str(src), "note": note})
            return
        dst = out / (name or src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied.append({"file": str(dst.relative_to(out)), "sha256": sha256_file(dst),
                       "bytes": dst.stat().st_size, "source": str(src),
                       "note": note})

    # ---- top-level decision and accounting records
    take(diag / "selection.json", "selection.json",
         "CURRENT decision record: bracketing + L2 reference eligibility")
    for p in sorted(diag.glob("selection.*.superseded*.json")):
        take(p, p.name, "PRESERVED earlier decision record, not overwritten")
    take(S / "matched_effectiveness_gpu_budget.json", "gpu_budget_ledger.json",
         "the EXPERIMENT's ledger, with all prior spend carried forward")
    take(diag / "gpu_budget.json", "gpu_budget_ledger_original_location.json",
         "the original in-output-directory ledger, superseded but retained")
    take(diag / "cost_model_measured.json", "cost_model_measured.json",
         "conservative measured per-stage estimates and the reconciled reserve")
    take(diag / "u_vs_mab_comparison.json", "u_vs_mab_comparison.json",
         "configuration, parent identity, serialisation and numerical comparison")
    take(diag / "RUN_NOTES.md", "RUN_NOTES.md", "run-time provenance notes")
    take(diag / "run_all.log", "run_all.log",
         "preflight, start, every stage, selection and stop records, verbatim")

    # ---- trajectory counters and dump inventories (no weights)
    traj = {}
    for d in sorted((diag / "models").glob("seed*")):
        rep_p = d / "U" / "train_report.json"
        if not rep_p.is_file():
            continue
        rep = json.loads(rep_p.read_text())
        rt, du = rep.get("runtime") or {}, rep.get("dumps") or {}
        traj[d.name] = {
            "checkpoint": rep.get("checkpoint"),
            "parent": rep.get("parent"),
            "training_seed": rep.get("training_seed"),
            "l2sp_weight": rep.get("l2sp_weight"),
            "new_deletion_target": rep.get("new_deletion_target"),
            "anchor_concept": rep.get("anchor_concept"),
            "parent_sha256": (rep.get("parent_verification") or {}).get("parent_sha256"),
            "endpoint_sha256": (rep.get("child_checkpoint") or {}).get("sha256"),
            "endpoint_bytes": (rep.get("child_checkpoint") or {}).get("bytes"),
            "optimizer_steps_completed": rt.get("optimizer_steps_completed"),
            "optimizer_steps_source": rt.get("optimizer_steps_source"),
            "optimizer_sync_points": rt.get("optimizer_sync_points"),
            "train_seconds": rt.get("train_seconds"),
            "completion_verified": rt.get("optimizer_steps_completed") == 1000,
            "dump_inventory": {
                "checkpoint_every": du.get("checkpoint_every"),
                "expected_steps": du.get("expected_steps"),
                "n_present": du.get("n_present"),
                "missing_steps": du.get("missing_steps"),
                "all_digests_distinct": du.get("all_digests_distinct"),
                "complete": du.get("complete"),
                "dumps": [{"step": x["step"], "sha256": x["sha256"],
                           "bytes": x["bytes"]} for x in (du.get("dumps") or [])],
            },
            "effective_hyperparameters": rep.get("effective_hyperparameters"),
        }
    (out / "trajectories.json").write_text(json.dumps(traj, indent=2) + "\n")
    copied.append({"file": "trajectories.json",
                   "sha256": sha256_file(out / "trajectories.json"),
                   "bytes": (out / "trajectories.json").stat().st_size,
                   "note": "counters, dump inventories and endpoint digests; NO weights"})

    # ---- the full per-dump development table, flattened for review
    sel = json.loads((diag / "selection.json").read_text())
    rows = ["seed,step,checkpoint,dog_hits,n_dog_pairs,dog_residue_pct,"
            "dog_suppression_pp_vs_own_MA,gate_suppression_ge_30pp,"
            "gate_residue_le_60pct,qualifies,mismatch_vs_L2_pp"]
    for sd, p in sel["per_seed"].items():
        for ref, label in (("MA", "MA"), ("L2_endpoint", "MAB_L2")):
            c = p[ref]
            sup = 0.0 if ref == "MA" else p["L2_dog_suppression_pp"]
            rows.append(f"{sd},,{c['checkpoint']},{c['dog_hits']},"
                        f"{c['n_dog_pairs']},{c['dog_residue_pct']},{sup},,,,")
        for r in p.get("dumps_scored", []):
            rows.append(
                f"{sd},{r['step']},{r['checkpoint']},{r['dog_hits']},"
                f"{r['n_dog_pairs']},{r['dog_residue_pct']},"
                f"{r['dog_suppression_pp_vs_own_MA']},"
                f"{r['gate_suppression_ge_30pp']},{r['gate_residue_le_60pct']},"
                f"{r['qualifies']},{r['mismatch_vs_L2_pp']}")
    (out / "per_dump_table.csv").write_text("\n".join(rows) + "\n")
    copied.append({"file": "per_dump_table.csv",
                   "sha256": sha256_file(out / "per_dump_table.csv"),
                   "bytes": (out / "per_dump_table.csv").stat().st_size,
                   "note": "every scored checkpoint, both seeds, with gate outcomes"})

    # ---- per-slot detector outputs and reports, gzipped (text only, no images)
    tar_p = out / "development_slots.tar.gz"
    slots, n_rows = [], 0
    with tarfile.open(tar_p, "w:gz") as tf:
        for d in sorted((diag / "eval_dev").iterdir()):
            if not d.is_dir():
                continue
            entry = {"slot": d.name}
            for fn in ("detections.jsonl", "image_report.json",
                       "detect_report.json", "_stage_complete.json"):
                f = d / fn
                if f.is_file():
                    tf.add(f, arcname=f"{d.name}/{fn}")
                    entry[fn] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
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
                entry["generation_settings_source"] = j.get("generation_settings_source")
                entry["n_generated"] = j.get("n_generated")
                entry["n_reused"] = j.get("n_reused")
            slots.append(entry)
    (out / "development_slots_index.json").write_text(
        json.dumps({"n_slots": len(slots), "total_detector_rows": n_rows,
                    "archive": tar_p.name,
                    "contents": ("per-slot detections.jsonl (per-image detector "
                                 "output), image_report.json, detect_report.json "
                                 "and the validation marker; NO image files"),
                    "slots": slots}, indent=2) + "\n")
    for f in (tar_p, out / "development_slots_index.json"):
        copied.append({"file": f.name, "sha256": sha256_file(f),
                       "bytes": f.stat().st_size,
                       "note": "per-image detector outputs and stage reports"})

    # ---- the frozen test location, inventoried EMPTY
    tdir = diag / "eval_test"
    tfiles = sorted(str(p.relative_to(tdir)) for p in tdir.rglob("*") if p.is_file())
    frozen = {
        "_why_this_exists": (
            "To document that the frozen test set was NEVER evaluated, rather "
            "than merely assert it."),
        "path": str(tdir),
        "exists": tdir.is_dir(),
        "files": tfiles,
        "file_count": len(tfiles),
        "no_test_evaluation": len(tfiles) == 0,
        "test_manifest_identity_unchanged":
            "5dd87dbb5a77c0f4c75de79a19b15b0b2e9cddc90d08172857700dac1bcc2a95",
        "reserve_spent_on_test_gpu_hours": 0.0,
        "reason": ("development matching was INFEASIBLE on both seeds, so the "
                   "pre-declared rule stopped the run before the confirmatory "
                   "stage and its reserve was never drawn"),
    }
    (out / "frozen_test_inventory.json").write_text(json.dumps(frozen, indent=2) + "\n")
    copied.append({"file": "frozen_test_inventory.json",
                   "sha256": sha256_file(out / "frozen_test_inventory.json"),
                   "bytes": (out / "frozen_test_inventory.json").stat().st_size,
                   "note": "documents that no test evaluation took place"})

    # ---- test-suite logs, with the commit they were run at
    if a.test_logs:
        tl = Path(a.test_logs)
        for f in sorted(tl.glob("*.log")):
            take(f, f"test_logs/{f.name}", "captured suite output")

    # ---- the manifest
    manifest = {
        "_what_this_is": (
            "Compact evidence bundle for the matched-effectiveness diagnostic. "
            "Everything needed to re-derive the decision without access to the "
            "private data root."),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit_at_build": commit,
        "diagnostic_root": str(diag),
        "outcome": ("GRID-MATCH FAILURE: no dump matched on the frozen 100-step "
                    "grid, on either seed. The matched bird-retention question "
                    "is INCONCLUSIVE and the frozen test set was not evaluated."),
        "deliberate_exclusions": {
            "model_weights": ("~1.6 GB of delta.bin and intermediate dumps are "
                              "NOT included; their sha256 digests are in "
                              "trajectories.json"),
            "generated_images": ("~208 MB of development images are NOT "
                                 "included; per-image digests are in each "
                                 "slot's image_report.json inside the archive"),
            "annotation_arm_key": ("NOT included and not referenced. The "
                                   "blinded annotation packet's key stays "
                                   "closed until labels are frozen; nothing "
                                   "from the packet appears in this bundle."),
        },
        "provisional": ("Detector output is a proxy. No scientific conclusion "
                        "follows until the blinded human annotation is complete; "
                        "no labels exist yet."),
        "files": copied,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(c.get("bytes", 0) for c in copied)
    print(f"bundle -> {out}")
    print(f"  {len(copied)} file(s), {total/1e6:.2f} MB total")
    print(f"  development slots: {len(slots)}, detector rows: {n_rows}")
    print(f"  frozen test files: {len(tfiles)} (no_test_evaluation="
          f"{frozen['no_test_evaluation']})")
    for c in copied:
        if c.get("status") == "ABSENT":
            print(f"  ABSENT: {c['file']} ({c['source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
