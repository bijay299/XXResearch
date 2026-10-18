#!/usr/bin/env python3
"""Persistent compute / image / job / failure ledgers.

GPU-hours are charged as (wall seconds of the job) x (number of GPU devices the job
occupied) / 3600 -- so two occupied GPUs for one hour costs 2.0, as the protocol requires.
Loading, warm-up, debugging, failures, activation prep, training, generation and GPU
scoring are all charged. CPU-only work is recorded with gpus=0 and charged 0.
"""
import json, os, time, fcntl, datetime, subprocess, sys

ROOT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
LED = f"{ROOT}/ledgers/ledger.json"
START = "2026-10-18T00:22:03Z"
CAPS = {"gpu_hours": 48.0, "images": 1600,
        "phase_integration_baseline": 20.0, "phase_development": 8.0,
        "phase_confirmation": 20.0, "unsuccessful_integration": 8.0}


def _blank():
    return {"start_utc": START, "caps": CAPS, "jobs": [], "images": [], "failures": []}


def _load():
    os.makedirs(os.path.dirname(LED), exist_ok=True)
    if not os.path.exists(LED):
        return _blank()
    with open(LED) as f:
        return json.load(f)


def _save(d):
    tmp = LED + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, LED)


def record_job(name, phase, gpus, seconds, note="", ok=True, images=0):
    d = _load()
    d["jobs"].append({"name": name, "phase": phase, "gpus": int(gpus),
                      "seconds": round(float(seconds), 1),
                      "gpu_hours": round(float(seconds) * int(gpus) / 3600.0, 4),
                      "ok": bool(ok), "images": int(images), "note": note,
                      "at_utc": datetime.datetime.now(datetime.timezone.utc)
                      .strftime("%Y-%m-%dT%H:%M:%SZ")})
    _save(d)
    return summary(quiet=True)


def record_failure(name, phase, what, gpus=0, seconds=0.0):
    d = _load()
    d["failures"].append({"name": name, "phase": phase, "what": what,
                          "gpu_hours": round(float(seconds) * int(gpus) / 3600.0, 4),
                          "at_utc": datetime.datetime.now(datetime.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ")})
    _save(d)


def record_images(arm, n, note=""):
    d = _load()
    d["images"].append({"arm": arm, "n": int(n), "note": note,
                        "at_utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ")})
    _save(d)


def summary(quiet=False):
    d = _load()
    gh = sum(j["gpu_hours"] for j in d["jobs"])
    by = {}
    for j in d["jobs"]:
        by[j["phase"]] = by.get(j["phase"], 0.0) + j["gpu_hours"]
    imgs = sum(i["n"] for i in d["images"])
    s = datetime.datetime.strptime(START, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    el = (datetime.datetime.now(datetime.timezone.utc) - s).total_seconds() / 3600.0
    out = {"gpu_hours_used": round(gh, 3), "gpu_hours_cap": CAPS["gpu_hours"],
           "gpu_hours_left": round(CAPS["gpu_hours"] - gh, 3),
           "by_phase": {k: round(v, 3) for k, v in by.items()},
           "images_used": imgs, "images_cap": CAPS["images"],
           "images_left": CAPS["images"] - imgs,
           "elapsed_h": round(el, 2), "elapsed_cap_h": 120,
           "elapsed_left_h": round(120 - el, 2),
           "n_jobs": len(d["jobs"]), "n_failures": len(d["failures"])}
    if not quiet:
        print(json.dumps(out, indent=1))
    return out


class Timer:
    """with Timer('name','phase',gpus=1) as t: ...   -> charges on exit, even on exception."""

    def __init__(self, name, phase, gpus=1, note="", images=0):
        self.name, self.phase, self.gpus = name, phase, gpus
        self.note, self.images = note, images

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, et, ev, tb):
        dt = time.time() - self.t0
        record_job(self.name, self.phase, self.gpus, dt,
                   note=self.note, ok=(et is None), images=self.images)
        if et is not None:
            record_failure(self.name, self.phase, f"{et.__name__}: {ev}", self.gpus, dt)
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        # ledger.py job NAME PHASE GPUS SECONDS [IMAGES] [NOTE]
        _, _, name, phase, gpus, secs, *rest = sys.argv
        imgs = int(rest[0]) if rest else 0
        note = rest[1] if len(rest) > 1 else ""
        record_job(name, phase, int(gpus), float(secs), note=note, images=imgs)
    elif len(sys.argv) > 1 and sys.argv[1] == "images":
        record_images(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
    summary()
