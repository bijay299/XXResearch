"""Append-only resource ledger and an in-process no-root watchdog.

Enforcement is best-effort and UNPRIVILEGED.  Every control actually in force
is recorded; gaps are disclosed rather than claimed as active.
"""

import json
import os
import resource
import threading
import time

LEDGER_PATH = None
_T0 = None
_PEAK_RSS_KB = 0
_LOCK = threading.Lock()


def _proc_tree_rss_kb():
    """Peak RSS of this process tree (self + reaped children), in KiB.

    ``ru_maxrss`` for RUSAGE_CHILDREN only covers children that have already
    been reaped -- a disclosed limitation.  This prototype runs ONE process, so
    the self value is the operative one.
    """
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(s, c)


def _cpu_seconds():
    """Process-tree CPU time (user+sys), self + reaped children."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def init(path):
    global LEDGER_PATH, _T0
    LEDGER_PATH = path
    _T0 = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write("")
    record("ledger_init", {"pid": os.getpid(),
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "omp_num_threads": os.environ.get("OMP_NUM_THREADS")})


def snapshot():
    global _PEAK_RSS_KB
    rss = _proc_tree_rss_kb()
    _PEAK_RSS_KB = max(_PEAK_RSS_KB, rss)
    return {
        "elapsed_s": round(time.time() - _T0, 3),
        "cpu_core_s": round(_cpu_seconds(), 3),
        "rss_kb": rss,
        "peak_rss_kb": _PEAK_RSS_KB,
    }


def record(event, payload=None):
    """Append one ledger line.  Append-only: the file is never rewritten."""
    entry = {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "event": event, **snapshot()}
    if payload is not None:
        entry["payload"] = payload
    with _LOCK:
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    return entry


class Watchdog:
    """Real in-process watchdog: hard-kills on wall-clock, CPU or RSS breach.

    Paired in ``run_suite.sh`` with an external ``timeout(1)`` and ``ulimit``
    so a hang inside a C extension is still bounded.
    """

    def __init__(self, max_wall_s, max_cpu_core_s, max_rss_kb, poll_s=2.0):
        self.max_wall_s = max_wall_s
        self.max_cpu_core_s = max_cpu_core_s
        self.max_rss_kb = max_rss_kb
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            snap = snapshot()
            breach = None
            if snap["elapsed_s"] > self.max_wall_s:
                breach = f"wall clock {snap['elapsed_s']}s > {self.max_wall_s}s"
            elif snap["cpu_core_s"] > self.max_cpu_core_s:
                breach = f"cpu {snap['cpu_core_s']}s > {self.max_cpu_core_s}s"
            elif snap["peak_rss_kb"] > self.max_rss_kb:
                breach = f"rss {snap['peak_rss_kb']}KiB > {self.max_rss_kb}KiB"
            if breach:
                record("WATCHDOG_STOP", {"breach": breach})
                os._exit(97)

    def start(self):
        record("watchdog_start", {"max_wall_s": self.max_wall_s,
                                  "max_cpu_core_s": self.max_cpu_core_s,
                                  "max_rss_kb": self.max_rss_kb})
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        record("watchdog_stop")
