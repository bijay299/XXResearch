"""Append-only resource ledger + unprivileged watchdog for the CPR real-model pilot.

Covers parent/child jobs, setup, attempts, timestamps, device allocation,
GPU-seconds, CPU, memory, disk, decoded images and stop events.

Enforcement is best-effort and UNPRIVILEGED. Every control actually in force is
recorded; gaps are disclosed rather than claimed.
"""

import json
import os
import resource
import subprocess
import threading
import time

ROOT = os.environ.get("CPR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER_PATH = os.path.join(ROOT, "artifacts", "resource_ledger.jsonl")
COUNTER_PATH = os.path.join(ROOT, "artifacts", "counters.json")

# ---- the task ceilings (contract section 2) ----
CEILINGS = {
    "elapsed_days": 7.0,
    "science_stop_days": 6.0,
    "gpu_device_hours": 24.0,
    "gpu_hours_entry_max": 4.0,
    "gpu_hours_eval_reserve": 8.0,
    "decoded_images": 2200,
    "cpu_core_hours": 48.0,
    "workers": 8,
    "rss_gib": 128.0,
    "extra_storage_gib": 200.0,
    "min_free_gib_data": 50.0,
}

_LOCK = threading.Lock()
_T0 = None
_JOB = None
_DEVICES = ()
_GPU_T0 = None


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _cpu_core_s():
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _rss_kb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(s, c)


def _free_gib(path="/data"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def _dir_gib(path):
    try:
        out = subprocess.check_output(["du", "-sb", path], text=True, stderr=subprocess.DEVNULL)
        return int(out.split()[0]) / 2**30
    except Exception:
        return None


# --------------------------------------------------------------------------
# cumulative counters, shared across processes (parent and children)
# --------------------------------------------------------------------------

def _load_counters():
    if os.path.exists(COUNTER_PATH):
        try:
            with open(COUNTER_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"gpu_device_seconds": 0.0, "gpu_device_seconds_entry": 0.0,
            "gpu_device_seconds_eval": 0.0, "decoded_images": 0,
            "cpu_core_seconds": 0.0, "attempts": 0, "stop_events": 0}


def bump(**deltas):
    """Atomically add to the cross-process cumulative counters."""
    with _LOCK:
        c = _load_counters()
        for k, v in deltas.items():
            c[k] = c.get(k, 0) + v
        tmp = COUNTER_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(c, f, indent=2, sort_keys=True)
        os.replace(tmp, COUNTER_PATH)
        return c


def counters():
    return _load_counters()


def gpu_hours_used():
    return _load_counters()["gpu_device_seconds"] / 3600.0


def images_used():
    return _load_counters()["decoded_images"]


# --------------------------------------------------------------------------

def init(job, devices=()):
    """Start a ledger session for one job (parent or child)."""
    global _T0, _JOB, _DEVICES, _GPU_T0
    _T0 = time.time()
    _JOB = job
    _DEVICES = tuple(devices)
    _GPU_T0 = time.time() if devices else None
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    bump(attempts=1)
    record("job_start", {"job": job, "pid": os.getpid(), "ppid": os.getppid(),
                         "devices": list(_DEVICES),
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "free_gib_data": round(_free_gib(), 2)})


def snapshot():
    el = time.time() - _T0 if _T0 else 0.0
    gpu_s = (time.time() - _GPU_T0) * len(_DEVICES) if _GPU_T0 else 0.0
    return {"job": _JOB, "pid": os.getpid(),
            "elapsed_s": round(el, 3),
            "cpu_core_s": round(_cpu_core_s(), 3),
            "gpu_device_s_this_job": round(gpu_s, 3),
            "devices": list(_DEVICES),
            "rss_kb": _rss_kb(),
            "free_gib_data": round(_free_gib(), 2)}


def record(event, payload=None):
    entry = {"ts_utc": _now(), "event": event, **snapshot()}
    if payload is not None:
        entry["payload"] = payload
    with _LOCK:
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    return entry


def record_images(n, kind, detail=None):
    """Count newly DECODED images against the 2,200 ceiling."""
    c = bump(decoded_images=n)
    record("decoded_images", {"n": n, "kind": kind, "detail": detail,
                              "cumulative": c["decoded_images"],
                              "ceiling": CEILINGS["decoded_images"]})
    if c["decoded_images"] > CEILINGS["decoded_images"]:
        record("CEILING_BREACH", {"which": "decoded_images",
                                  "value": c["decoded_images"]})
        raise RuntimeError(f"decoded-image ceiling exceeded: {c['decoded_images']}")
    return c["decoded_images"]


def close(bucket=None):
    """End a job, folding its GPU-seconds into the cumulative counters."""
    snap = snapshot()
    gpu_s = snap["gpu_device_s_this_job"]
    d = {"gpu_device_seconds": gpu_s, "cpu_core_seconds": snap["cpu_core_s"]}
    if bucket == "entry":
        d["gpu_device_seconds_entry"] = gpu_s
    elif bucket == "eval":
        d["gpu_device_seconds_eval"] = gpu_s
    c = bump(**d)
    record("job_end", {"bucket": bucket, "cumulative": c,
                       "gpu_hours_total": round(c["gpu_device_seconds"] / 3600, 4)})
    return c


class Watchdog:
    """Real in-process watchdog: hard-kills on wall clock, CPU, RSS, disk or
    cumulative GPU-hours. Paired with external `timeout` and shell rlimits."""

    def __init__(self, max_wall_s, max_cpu_core_s=None, max_rss_kb=None,
                 max_gpu_hours=None, min_free_gib=None, poll_s=5.0):
        self.max_wall_s = max_wall_s
        self.max_cpu_core_s = max_cpu_core_s or CEILINGS["cpu_core_hours"] * 3600
        self.max_rss_kb = max_rss_kb or int(CEILINGS["rss_gib"] * 2**20)
        self.max_gpu_hours = max_gpu_hours or CEILINGS["gpu_device_hours"]
        self.min_free_gib = min_free_gib or CEILINGS["min_free_gib_data"]
        self.poll_s = poll_s
        self._stop = threading.Event()

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            s = snapshot()
            total_gpu_h = (_load_counters()["gpu_device_seconds"]
                           + s["gpu_device_s_this_job"]) / 3600.0
            breach = None
            if s["elapsed_s"] > self.max_wall_s:
                breach = f"wall {s['elapsed_s']}s > {self.max_wall_s}s"
            elif s["cpu_core_s"] > self.max_cpu_core_s:
                breach = f"cpu {s['cpu_core_s']}s > {self.max_cpu_core_s}s"
            elif s["rss_kb"] > self.max_rss_kb:
                breach = f"rss {s['rss_kb']}KiB > {self.max_rss_kb}KiB"
            elif total_gpu_h > self.max_gpu_hours:
                breach = f"cumulative GPU {total_gpu_h:.3f}h > {self.max_gpu_hours}h"
            elif s["free_gib_data"] < self.min_free_gib:
                breach = f"free /data {s['free_gib_data']}GiB < {self.min_free_gib}GiB"
            if breach:
                bump(stop_events=1)
                record("WATCHDOG_STOP", {"breach": breach})
                os._exit(97)

    def start(self):
        record("watchdog_start", {"max_wall_s": self.max_wall_s,
                                  "max_cpu_core_s": self.max_cpu_core_s,
                                  "max_rss_kb": self.max_rss_kb,
                                  "max_gpu_hours": self.max_gpu_hours,
                                  "min_free_gib": self.min_free_gib})
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        record("watchdog_stop")
