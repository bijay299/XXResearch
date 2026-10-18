#!/usr/bin/env python3
"""Append-only preparation ledger for teapot_human_review_v1.

This package involves ZERO GPU allocation, zero training, zero model inference
and zero new model-generated images. The ledger therefore records preparation
steps, failures and imported-image counts -- not GPU seconds.

Append-only JSONL, flock-serialised. Records are never edited or deleted;
corrections are new records.
"""
import argparse
import fcntl
import json
import os
import socket
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(ROOT, "prep_ledger.jsonl")
CLOCK = os.path.join(ROOT, "clock.json")

DEADLINE_HOURS = 48
STORAGE_GIB = 10.0
MAX_WORKERS = 8
GPU_ALLOWED = 0
NEW_MODEL_IMAGES_ALLOWED = 0


def _append(rec):
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat())
    rec.setdefault("host", socket.gethostname())
    line = json.dumps(rec, sort_keys=True) + "\n"
    fd = os.open(LEDGER, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return rec


def read_all():
    if not os.path.exists(LEDGER):
        return []
    out = []
    with open(LEDGER) as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return out


def dir_gib(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            p = os.path.join(dp, f)
            try:
                if not os.path.islink(p):
                    total += os.lstat(p).st_blocks * 512
            except OSError:
                pass
    return total / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--start-iso", required=True)

    p = sub.add_parser("step")
    p.add_argument("--kind", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--outcome", default="ok")
    p.add_argument("--imported-images", type=int, default=0)
    p.add_argument("--new-model-images", type=int, default=0)
    p.add_argument("--gpu-seconds", type=float, default=0.0)
    p.add_argument("--note", default="")

    sub.add_parser("report")

    a = ap.parse_args()

    if a.cmd == "init":
        if os.path.exists(CLOCK):
            print("clock already recorded; refusing to reset:")
            print(json.dumps(json.load(open(CLOCK)), indent=2))
            return
        start = datetime.fromisoformat(a.start_iso)
        rec = {
            "start_iso": start.isoformat(),
            "deadline_iso": (start + timedelta(hours=DEADLINE_HOURS)).isoformat(),
            "deadline_hours": DEADLINE_HOURS,
            "limits": {
                "storage_gib": STORAGE_GIB,
                "max_worker_threads": MAX_WORKERS,
                "gpu_allocation": GPU_ALLOWED,
                "new_model_generated_images": NEW_MODEL_IMAGES_ALLOWED,
                "automated_judging": "forbidden",
                "paid_services_or_hosting": "forbidden",
            },
        }
        json.dump(rec, open(CLOCK, "w"), indent=2)
        _append({"type": "clock_init", **rec})
        print(json.dumps(rec, indent=2))

    elif a.cmd == "step":
        rec = _append({"type": "step", "kind": a.kind, "label": a.label,
                       "outcome": a.outcome,
                       "imported_images": a.imported_images,
                       "new_model_images": a.new_model_images,
                       "gpu_seconds": a.gpu_seconds,
                       "note": a.note})
        print(json.dumps({k: rec[k] for k in ("label", "outcome", "imported_images")}))

    elif a.cmd == "report":
        recs = read_all()
        steps = [r for r in recs if r.get("type") == "step"]
        clock = json.load(open(CLOCK))
        start = datetime.fromisoformat(clock["start_iso"])
        now = datetime.now(timezone.utc).astimezone(start.tzinfo)
        failures = [r for r in steps if r["outcome"] != "ok"]
        print(json.dumps({
            "clock": {
                "start_iso": clock["start_iso"],
                "deadline_iso": clock["deadline_iso"],
                "elapsed_hours": round((now - start).total_seconds() / 3600, 3),
                "remaining_hours": round(
                    (datetime.fromisoformat(clock["deadline_iso"]) - now).total_seconds() / 3600, 3),
            },
            "steps_recorded": len(steps),
            "failures_recorded": len(failures),
            "failure_labels": [r["label"] for r in failures],
            "imported_images_total": sum(r["imported_images"] for r in steps),
            "new_model_generated_images_total": sum(r["new_model_images"] for r in steps),
            "gpu_seconds_total": sum(r["gpu_seconds"] for r in steps),
            "project_storage_gib": round(dir_gib(os.path.dirname(ROOT)), 3),
            "limits": clock["limits"],
        }, indent=2))


if __name__ == "__main__":
    main()
