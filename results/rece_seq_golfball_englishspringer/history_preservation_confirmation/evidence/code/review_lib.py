#!/usr/bin/env python3
"""Shared helpers for compiling frozen visual-review records across every
endpoint in history_preservation_confirmation. Judgments are recorded by
hand (as per-case dicts) during actual image-by-image inspection in the
review scripts that import this module; this module only computes real
file hashes/existence and writes the frozen, hashed output files. Nothing
here infers a judgment from a filename, a classifier, or file metadata.
"""
import hashlib
import json
import time
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_endpoint(endpoint_name: str, image_report_path: Path, judgments: dict, viewer_note: str):
    """judgments: case_id -> dict of judgment fields (already hand-recorded by the
    reviewer during actual viewing). Must cover every case_id in image_report."""
    report = json.loads(Path(image_report_path).read_text())
    rows = []
    for im in report["images"]:
        cid = im["case_id"]
        assert cid in judgments, f"{endpoint_name}: no recorded judgment for case {cid}"
        p = Path(im["image_path"])
        assert p.exists(), f"{endpoint_name}: missing image {p}"
        actual_hash = sha256_file(p)
        assert actual_hash == im["sha256"], f"{endpoint_name} case {cid}: hash mismatch vs image_report"
        row = {
            "endpoint": endpoint_name, "case_id": cid, "class": im["class"],
            "scene_index": im["scene_index"], "family": im["family"], "prompt": im["prompt"],
            "seed": im["seed"], "image_path": str(p), "sha256": actual_hash,
            "resolution_viewed": "full_512x512", "viewer_note": viewer_note,
        }
        row.update(judgments[cid])
        rows.append(row)
    assert len(rows) == len(report["images"])
    return rows


def freeze_and_hash(rows: list, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    completion_path = out_dir / f"CASE_LEVEL_COMPLETION_RECORD_{tag}.json"
    completion_path.write_text(json.dumps(rows, indent=1))
    completion_hash = sha256_file(completion_path)

    labels_path = out_dir / f"VISUAL_REVIEW_INITIAL_{tag}.json"
    by_endpoint = {}
    for r in rows:
        by_endpoint.setdefault(r["endpoint"], []).append(r)
    labels_path.write_text(json.dumps({"n_rows": len(rows), "endpoints": sorted(by_endpoint.keys()),
                                        "rows_per_endpoint": {k: len(v) for k, v in by_endpoint.items()}}, indent=1))
    labels_hash = sha256_file(labels_path)

    gate = {
        "tag": tag,
        "completion_record_path": str(completion_path), "completion_record_sha256": completion_hash,
        "initial_labels_path": str(labels_path), "initial_labels_sha256": labels_hash,
        "n_rows_present": len(rows), "frozen_at": time.time(),
    }
    gate_path = out_dir / f"CLASSIFIER_PRECONDITION_GATE_{tag}.json"
    gate_path.write_text(json.dumps(gate, indent=1))
    return completion_path, labels_path, gate_path, gate
