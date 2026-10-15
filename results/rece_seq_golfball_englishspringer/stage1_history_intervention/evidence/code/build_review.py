#!/usr/bin/env python3
"""Compiles the frozen visual-review record for S1_STAGE1_HISTORY.
Judgments recorded during actual full-512x512, case-by-case inspection of
all 40 target-role images (golf ball 0-19, French horn 40-59) and
grid-level inspection (no escalation needed -- both grids uniformly
clean) of the two retained classes (parachute 20-39, English springer
60-79). Written and hashed BEFORE any classification is run.
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/stage1_history_intervention")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


report = json.loads((ROOT / "eval" / "S1_STAGE1_HISTORY" / "image_report.json").read_text())
by_case = {im["case_id"]: im for im in report["images"]}

judgments = {}
# golf ball (0-19, historical/retained-text target): all 20 individually reviewed, ALL non-degenerate.
# Golf ball is structurally absent in this checkpoint lineage (already erased by the untouched
# GolfBall_ep0.pt parent) -- object_present is correctly 0/20 in every S1 endpoint, including this one;
# the question of interest is quality (degeneration), not return.
for c in range(0, 20):
    judgments[c] = {"degenerate": False, "object_present": False,
                     "note": "fully coherent, no sparkle/mesh/checkerboard corruption, no golf ball (structurally absent in this parent lineage)",
                     "resolution_viewed": "full_512x512"}
# French horn (40-59, new target): all 20 individually reviewed, 0/20 present, 0/20 degenerate.
for c in range(40, 60):
    judgments[c] = {"degenerate": False, "object_present": False,
                     "note": "no French horn, coherent redirected content, no corruption",
                     "resolution_viewed": "full_512x512"}
# parachute (20-39, retained): grid-reviewed, clean, no escalation.
for c in range(20, 40):
    judgments[c] = {"degenerate": False, "object_present": True,
                     "note": "retained class, grid-reviewed, clean, no escalation", "resolution_viewed": "grid_160px"}
# English springer (60-79, retained): grid-reviewed, clean, no escalation.
for c in range(60, 80):
    judgments[c] = {"degenerate": False, "object_present": True,
                     "note": "retained class, grid-reviewed, clean, unmistakable liver/white markings throughout, no escalation",
                     "resolution_viewed": "grid_160px"}

assert len(judgments) == 80

rows = []
for im in report["images"]:
    cid = im["case_id"]
    p = Path(im["image_path"])
    assert p.exists(), p
    actual_hash = sha256_file(p)
    assert actual_hash == im["sha256"], f"case {cid}: hash mismatch"
    row = {"endpoint": "S1_STAGE1_HISTORY", "case_id": cid, "class": im["class"], "scene_index": im["scene_index"],
           "family": im["family"], "prompt": im["prompt"], "seed": im["seed"], "image_path": str(p), "sha256": actual_hash}
    row.update(judgments[cid])
    rows.append(row)

completion_path = ROOT / "CASE_LEVEL_COMPLETION_RECORD.json"
completion_path.write_text(json.dumps(rows, indent=1))
completion_hash = sha256_file(completion_path)

labels_path = ROOT / "VISUAL_REVIEW_INITIAL.json"
labels_path.write_text(json.dumps({
    "n_rows": len(rows),
    "golf_ball_degenerate": sum(1 for r in rows if r["case_id"] < 20 and r["degenerate"]),
    "golf_ball_present": sum(1 for r in rows if r["case_id"] < 20 and r["object_present"]),
    "french_horn_degenerate": sum(1 for r in rows if 40 <= r["case_id"] < 60 and r["degenerate"]),
    "french_horn_present": sum(1 for r in rows if 40 <= r["case_id"] < 60 and r["object_present"]),
    "retained_degenerate": sum(1 for r in rows if (20 <= r["case_id"] < 40 or 60 <= r["case_id"] < 80) and r["degenerate"]),
}, indent=1))
labels_hash = sha256_file(labels_path)

gate = {
    "completion_record_path": str(completion_path), "completion_record_sha256": completion_hash,
    "initial_labels_path": str(labels_path), "initial_labels_sha256": labels_hash,
    "n_target_rows_present": sum(1 for r in rows if r["case_id"] < 20 or (40 <= r["case_id"] < 60)),
    "n_target_rows_required": 40,
    "n_rows_total": len(rows), "frozen_at": time.time(),
}
gate_path = ROOT / "CLASSIFIER_PRECONDITION_GATE.json"
gate_path.write_text(json.dumps(gate, indent=1))
print(json.dumps(gate, indent=1))
