#!/usr/bin/env python3
"""Concurrency-safe, APPEND-ONLY job/image ledger for measurement_study_v1.

Deliberately different from the previous sprint's ledger, whose unlocked
read-modify-write through a shared temp file could lose concurrent updates:

  * one JSONL file, opened O_APPEND, one record per line;
  * every append is wrapped in an exclusive fcntl.flock on the same fd;
  * records are never rewritten, so a reader can only ever see whole lines;
  * summaries are computed by re-reading, never by mutating.

Records ACTUAL outcomes, not requested counts: `images_decoded` is written by the
producing job from what it actually wrote to disk, and failed/partial attempts are
recorded with the images they really produced.
"""
import json, os, sys, time, fcntl, datetime, argparse

ROOT = "/data/bijaypandey/cuig_pilot/measurement_study_v1"
LEDGER = f"{ROOT}/ledgers/ledger.jsonl"
START = "2026-10-18T04:42:35Z"
CAPS = {"gpu_hours": 12.0, "images_total": 240,
        "images_study": 180, "images_practice": 12, "images_smoke_recovery": 48,
        "elapsed_hours": 72.0}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    rec = {"at_utc": _now(), **rec}
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


def job(name, kind, gpus, seconds, images_decoded=0, ok=True, note="", bucket=None):
    return append({"type": "job", "name": name, "kind": kind,
                   "gpus": int(gpus), "seconds": round(float(seconds), 1),
                   "gpu_hours": round(float(seconds) * int(gpus) / 3600.0, 6),
                   "images_decoded": int(images_decoded), "bucket": bucket,
                   "ok": bool(ok), "note": note})


def failure(name, what, gpus=0, seconds=0.0, images_decoded=0):
    return append({"type": "failure", "name": name, "what": what,
                   "gpus": int(gpus), "seconds": round(float(seconds), 1),
                   "gpu_hours": round(float(seconds) * int(gpus) / 3600.0, 6),
                   "images_decoded": int(images_decoded)})


def note(name, what):
    return append({"type": "note", "name": name, "what": what})


def read():
    if not os.path.exists(LEDGER):
        return []
    out = []
    with open(LEDGER) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                out.append({"type": "UNPARSEABLE", "raw": ln[:200]})
    return out


def summary(quiet=False):
    recs = read()
    gh = sum(r.get("gpu_hours", 0.0) for r in recs if r["type"] in ("job", "failure"))
    img = sum(r.get("images_decoded", 0) for r in recs if r["type"] in ("job", "failure"))
    by_bucket = {}
    for r in recs:
        if r["type"] in ("job", "failure") and r.get("images_decoded"):
            b = r.get("bucket") or "unbucketed"
            by_bucket[b] = by_bucket.get(b, 0) + r["images_decoded"]
    s = datetime.datetime.strptime(START, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    el = (datetime.datetime.now(datetime.timezone.utc) - s).total_seconds() / 3600.0
    out = {"gpu_hours_used": round(gh, 4), "gpu_hours_cap": CAPS["gpu_hours"],
           "gpu_hours_left": round(CAPS["gpu_hours"] - gh, 4),
           "images_decoded_total": img, "images_cap": CAPS["images_total"],
           "images_left": CAPS["images_total"] - img,
           "images_by_bucket": by_bucket,
           "elapsed_h": round(el, 3), "elapsed_cap_h": CAPS["elapsed_hours"],
           "elapsed_left_h": round(CAPS["elapsed_hours"] - el, 3),
           "n_jobs": sum(1 for r in recs if r["type"] == "job"),
           "n_failures": sum(1 for r in recs if r["type"] == "failure"),
           "n_unparseable": sum(1 for r in recs if r["type"] == "UNPARSEABLE")}
    if not quiet:
        print(json.dumps(out, indent=1))
    return out


def enforce(need_gpu_hours=0.0, need_images=0):
    """Hard pre-flight check. Raises SystemExit if a ceiling would be crossed."""
    s = summary(quiet=True)
    if s["gpu_hours_used"] + need_gpu_hours > CAPS["gpu_hours"]:
        raise SystemExit(f"REFUSED: would exceed GPU cap "
                         f"({s['gpu_hours_used']:.3f}+{need_gpu_hours} > {CAPS['gpu_hours']})")
    if s["images_decoded_total"] + need_images > CAPS["images_total"]:
        raise SystemExit(f"REFUSED: would exceed image cap "
                         f"({s['images_decoded_total']}+{need_images} > {CAPS['images_total']})")
    if s["elapsed_left_h"] <= 0:
        raise SystemExit("REFUSED: elapsed deadline passed")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    j = sub.add_parser("job")
    for a in ("name", "kind"):
        j.add_argument(a)
    j.add_argument("gpus", type=int); j.add_argument("seconds", type=float)
    j.add_argument("--images", type=int, default=0); j.add_argument("--bucket", default=None)
    j.add_argument("--ok", default="true"); j.add_argument("--note", default="")
    f = sub.add_parser("failure"); f.add_argument("name"); f.add_argument("what")
    f.add_argument("--gpus", type=int, default=0); f.add_argument("--seconds", type=float, default=0.0)
    f.add_argument("--images", type=int, default=0)
    n = sub.add_parser("note"); n.add_argument("name"); n.add_argument("what")
    sub.add_parser("summary")
    e = sub.add_parser("enforce"); e.add_argument("--gpu-hours", type=float, default=0.0)
    e.add_argument("--images", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "job":
        job(a.name, a.kind, a.gpus, a.seconds, a.images, a.ok.lower() == "true", a.note, a.bucket)
    elif a.cmd == "failure":
        failure(a.name, a.what, a.gpus, a.seconds, a.images)
    elif a.cmd == "note":
        note(a.name, a.what)
    elif a.cmd == "enforce":
        print(json.dumps(enforce(a.gpu_hours, a.images), indent=1)); sys.exit(0)
    summary()
