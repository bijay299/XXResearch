#!/usr/bin/env python3
"""The actual classification entry point for S2_HISTORY_BOTH_STAGES.

This is a single executed wrapper: it validates every precondition in
this process, in this function call, BEFORE classification runs, and
classification is invoked only if every check passes -- in the same
guarded execution path, not as a separate after-the-fact script. This
directly fixes the gap disclosed in
closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md item 4: the prior endpoint's
"guard" was a standalone script run after classification had already
completed, re-verifying hashes retroactively. It could not have refused
classification. This one can, and does, if any check fails.

Checks performed, in order, before classification is permitted to run:
  1. The frozen gate file's own recorded hashes match the actual current
     bytes of CASE_LEVEL_COMPLETION_RECORD.json and VISUAL_REVIEW_INITIAL.json
     (detects any post-freeze tampering).
  2. The completion record contains exactly the required 80 case IDs
     (0-79), each with a completed judgment (not a placeholder).
  3. Every image file referenced in the completion record exists and its
     sha256 matches the recorded hash (re-verified now, not trusted from
     the freeze time).
  4. The image_report.json (classification's own input) references the
     same 80 case IDs and the same image hashes as the completion record.

On any failure: classification is refused, the failure is logged, and
the process exits nonzero. On success: the precondition-pass is logged
with a timestamp, classification is invoked in this same process, and
the completion is logged with a timestamp.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_both_stages_s2")
ENDPOINT = "S2_HISTORY_BOTH_STAGES"
EVAL_DIR = ROOT / "eval" / ENDPOINT
VENV_PY = "/data/bijaypandey/cuig_pilot/venv-cuig/bin/python3"
CLASSIFY_SCRIPT = "/data/bijaypandey/cuig_pilot/rece_golfball_confirmation/classify_confirmation.py"

LOG_PATH = ROOT / "CLASSIFIER_GUARD_INVOCATION_LOG.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse(log: dict, reason: str) -> None:
    log["guard_passed"] = False
    log["refused_reason"] = reason
    log["refused_at"] = time.time()
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"REFUSED: {reason}", file=sys.stderr)
    sys.exit(1)


def main():
    log = {"endpoint": ENDPOINT, "check_started_at": time.time(), "checks": {}}

    gate_path = ROOT / "CLASSIFIER_PRECONDITION_GATE.json"
    completion_path = ROOT / "CASE_LEVEL_COMPLETION_RECORD.json"
    labels_path = ROOT / "VISUAL_REVIEW_INITIAL.json"
    if not (gate_path.is_file() and completion_path.is_file() and labels_path.is_file()):
        refuse(log, "one or more required frozen files are missing")

    gate = json.loads(gate_path.read_text())

    # Check 1: frozen-file hashes unchanged since freeze
    actual_completion_hash = sha256_file(completion_path)
    actual_labels_hash = sha256_file(labels_path)
    check1 = {
        "completion_record_hash_match": actual_completion_hash == gate["completion_record_sha256"],
        "initial_labels_hash_match": actual_labels_hash == gate["initial_labels_sha256"],
    }
    log["checks"]["1_frozen_hashes_unchanged"] = check1
    if not all(check1.values()):
        refuse(log, "frozen completion-record or labels file has changed since the freeze")

    # Check 2: exactly the required 80 case IDs, each with a completed judgment
    rows = json.loads(completion_path.read_text())
    case_ids = sorted(r["case_id"] for r in rows)
    expected_ids = list(range(80))
    required_fields_present = all(
        ("degenerate" in r and "object_present" in r and "note" in r and r["note"]) for r in rows
    )
    check2 = {
        "n_rows": len(rows),
        "case_ids_match_0_to_79": case_ids == expected_ids,
        "every_row_has_completed_judgment_fields": required_fields_present,
    }
    log["checks"]["2_case_id_set_and_completion"] = check2
    if not (check2["case_ids_match_0_to_79"] and check2["every_row_has_completed_judgment_fields"]):
        refuse(log, "case-ID set is not exactly 0-79 with completed judgments, or rows are placeholders")

    # Check 3: every referenced image exists and its hash matches the recorded hash, right now
    mismatches = []
    for r in rows:
        p = Path(r["image_path"])
        if not p.is_file():
            mismatches.append({"case_id": r["case_id"], "error": "missing file"})
            continue
        actual = sha256_file(p)
        if actual != r["sha256"]:
            mismatches.append({"case_id": r["case_id"], "error": "hash mismatch",
                                "expected": r["sha256"], "actual": actual})
    check3 = {"n_checked": len(rows), "n_mismatches": len(mismatches), "mismatches": mismatches}
    log["checks"]["3_image_files_match_recorded_hashes"] = check3
    if mismatches:
        refuse(log, f"{len(mismatches)} image file(s) missing or hash-mismatched vs. the completion record")

    # Check 4: image_report.json (classify's own input) agrees with the completion record
    image_report_path = EVAL_DIR / "image_report.json"
    if not image_report_path.is_file():
        refuse(log, "image_report.json (classification input) is missing")
    image_report = json.loads(image_report_path.read_text())
    ir_by_case = {im["case_id"]: im for im in image_report["images"]}
    completion_by_case = {r["case_id"]: r for r in rows}
    agree = (set(ir_by_case) == set(completion_by_case)) and all(
        ir_by_case[cid]["sha256"] == completion_by_case[cid]["sha256"] for cid in ir_by_case
    )
    check4 = {"image_report_case_ids_match": set(ir_by_case) == set(completion_by_case),
              "image_report_hashes_match_completion_record": agree}
    log["checks"]["4_image_report_agrees_with_completion_record"] = check4
    if not agree:
        refuse(log, "image_report.json's case IDs/hashes do not agree with the frozen completion record")

    # ---- all checks passed: log the pass, then invoke classification in this same process ----
    log["guard_passed"] = True
    log["precondition_check_passed_at"] = time.time()
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print("GUARD PASSED -- invoking classification now, in this same guarded process.", flush=True)

    classify_out = EVAL_DIR / "classifications.jsonl"
    classify_report = EVAL_DIR / "classify_report.json"
    invocation_started_at = time.time()
    result = subprocess.run([
        VENV_PY, CLASSIFY_SCRIPT,
        "--image_report", str(image_report_path),
        "--checkpoint_name", ENDPOINT,
        "--out", str(classify_out),
        "--report", str(classify_report),
        "--device", "cuda:0",
    ], capture_output=True, text=True)
    invocation_ended_at = time.time()

    log["classification_invocation"] = {
        "started_at": invocation_started_at, "ended_at": invocation_ended_at,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:],
    }
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    classify_report_data = json.loads(classify_report.read_text())
    log["classification_complete"] = classify_report_data.get("complete", False)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"DONE: classification complete={log['classification_complete']}")


if __name__ == "__main__":
    main()
