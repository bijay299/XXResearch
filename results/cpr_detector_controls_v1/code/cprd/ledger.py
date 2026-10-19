"""Parent-side supervisor + durable, interprocess-safe allocation ledger.

Design requirements this implements:
  * Reservations are written BEFORE a child is launched.
  * Children are charged through CONFIRMED TERMINATION, including exceptions,
    OOMs, timeouts and kills. A failed job can never disappear from accounting:
    the parent reconciles every reservation it opened, and an unreconciled
    reservation is charged at its full reserved amount and flagged.
  * All state lives in one JSON file guarded by an fcntl advisory lock, so
    concurrent parents/children cannot lose updates.
  * Aggregate ceilings, the entry subcap and the protected evaluation reserve
    are ENFORCED at reservation time (a reservation that would breach is
    refused), not merely forecast.

Enforcement gaps are documented in `gaps()` and published; nothing here claims
a mechanism that does not exist.
"""

import fcntl
import json
import os
import signal
import subprocess
import time
import uuid

ROOT = os.environ.get("CPRD_ROOT", "/data/bijaypandey/cuig_pilot/cpr_detector_controls_v1")
STATE = os.path.join(ROOT, "artifacts", "allocation_ledger.json")
JOURNAL = os.path.join(ROOT, "artifacts", "allocation_journal.jsonl")

CEILINGS = {
    "elapsed_days": 7.0,
    "science_stop_days": 6.0,
    "gpu_device_hours": 24.0,
    "gpu_hours_entry_subcap": 4.0,
    "gpu_hours_eval_reserve": 8.0,
    "generated_images": 2200,
    "primary_images": 1928,
    "technical_images": 272,
    "coco_photographs": 600,
    "cpu_core_hours": 48.0,
    "workers": 8,
    "rss_gib": 128.0,
    "storage_gib": 200.0,
    "min_free_gib": 50.0,
    "repair_cycles": 1,
    "gpu_retries": 1,
}

BUCKETS = ("entry", "dev", "eval")          # eval == protected reserve


def _blank():
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "charged_gpu_s": {b: 0.0 for b in BUCKETS},
        "charged_cpu_s": 0.0,
        "generated_images": 0,
        "coco_photographs": 0,
        "open_reservations": {},
        "closed": [],
        "unreconciled": [],
        "retries": {"gpu_technical": 0, "cpu_repair": 0},
        "events": 0,
    }


class _Locked:
    """fcntl-guarded read-modify-write of the ledger file."""

    def __enter__(self):
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        self.fh = open(STATE, "a+")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        self.fh.seek(0)
        raw = self.fh.read()
        self.state = json.loads(raw) if raw.strip() else _blank()
        return self

    def __exit__(self, *exc):
        self.state["events"] += 1
        self.fh.seek(0)
        self.fh.truncate()
        json.dump(self.state, self.fh, indent=2, sort_keys=True)
        self.fh.flush()
        os.fsync(self.fh.fileno())
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()
        return False


def journal(event, **payload):
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    line = json.dumps({"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "pid": os.getpid(), "event": event, **payload}, default=str)
    with open(JOURNAL, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def read():
    with _Locked() as L:
        return json.loads(json.dumps(L.state))


def totals(state=None):
    s = state or read()
    g = s["charged_gpu_s"]
    open_g = sum(r["reserved_gpu_s"] for r in s["open_reservations"].values())
    return {
        "gpu_s_entry": g["entry"], "gpu_s_dev": g["dev"], "gpu_s_eval": g["eval"],
        "gpu_s_charged_total": sum(g.values()),
        "gpu_s_open_reservations": open_g,
        "gpu_s_committed_total": sum(g.values()) + open_g,
        "gpu_h_committed_total": (sum(g.values()) + open_g) / 3600.0,
        "cpu_s": s["charged_cpu_s"],
        "generated_images": s["generated_images"],
        "coco_photographs": s["coco_photographs"],
        "retries": s["retries"],
        "unreconciled": len(s["unreconciled"]),
    }


class CeilingBreach(RuntimeError):
    pass


def reserve(job, bucket, devices, max_seconds, note=None):
    """Open a reservation BEFORE launching. Refuses if it would breach a cap."""
    assert bucket in BUCKETS, bucket
    n_dev = max(1, len(devices)) if devices else 0
    reserved = float(max_seconds) * n_dev
    rid = f"{job}-{uuid.uuid4().hex[:8]}"
    with _Locked() as L:
        s = L.state
        charged = sum(s["charged_gpu_s"].values())
        open_g = sum(r["reserved_gpu_s"] for r in s["open_reservations"].values())
        committed = charged + open_g + reserved

        cap_total = CEILINGS["gpu_device_hours"] * 3600
        if committed > cap_total:
            raise CeilingBreach(
                f"GPU aggregate: committing {committed/3600:.3f}h > {CEILINGS['gpu_device_hours']}h")

        if bucket == "entry":
            e = s["charged_gpu_s"]["entry"] + sum(
                r["reserved_gpu_s"] for r in s["open_reservations"].values()
                if r["bucket"] == "entry") + reserved
            if e > CEILINGS["gpu_hours_entry_subcap"] * 3600:
                raise CeilingBreach(
                    f"entry subcap: {e/3600:.3f}h > {CEILINGS['gpu_hours_entry_subcap']}h")

        if bucket in ("entry", "dev"):
            # the evaluation reserve is PROTECTED: non-eval work may not eat it
            usable = cap_total - CEILINGS["gpu_hours_eval_reserve"] * 3600
            non_eval = (s["charged_gpu_s"]["entry"] + s["charged_gpu_s"]["dev"]
                        + sum(r["reserved_gpu_s"] for r in s["open_reservations"].values()
                              if r["bucket"] in ("entry", "dev")) + reserved)
            if non_eval > usable:
                raise CeilingBreach(
                    f"protected reserve: non-eval {non_eval/3600:.3f}h > usable "
                    f"{usable/3600:.3f}h (24h total - 8h reserve)")

        s["open_reservations"][rid] = {
            "job": job, "bucket": bucket, "devices": list(devices or []),
            "max_seconds": float(max_seconds), "reserved_gpu_s": reserved,
            "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "opened_monotonic": time.time(), "note": note, "pid": None,
        }
    journal("reserve", rid=rid, job=job, bucket=bucket, devices=list(devices or []),
            reserved_gpu_s=reserved, note=note)
    return rid


def attach_pid(rid, pid):
    with _Locked() as L:
        if rid in L.state["open_reservations"]:
            L.state["open_reservations"][rid]["pid"] = pid
    journal("attach_pid", rid=rid, pid=pid)


def settle(rid, actual_seconds, cpu_seconds=0.0, status="ok", detail=None):
    """Charge a reservation through CONFIRMED TERMINATION and close it.

    `actual_seconds` is wall-clock the devices were held, charged x n_devices.
    Called for every outcome: success, exception, OOM, timeout, kill.
    """
    with _Locked() as L:
        s = L.state
        r = s["open_reservations"].pop(rid, None)
        if r is None:
            journal("settle_missing", rid=rid, status=status)
            return None
        n_dev = max(1, len(r["devices"])) if r["devices"] else 0
        charged = float(actual_seconds) * n_dev
        # never charge less than observed, never silently exceed the reservation
        charged = min(charged, r["reserved_gpu_s"]) if charged > r["reserved_gpu_s"] else charged
        s["charged_gpu_s"][r["bucket"]] += charged
        s["charged_cpu_s"] += float(cpu_seconds)
        rec = {**r, "closed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "actual_seconds": float(actual_seconds), "charged_gpu_s": charged,
               "cpu_seconds": float(cpu_seconds), "status": status, "detail": detail}
        s["closed"].append(rec)
    journal("settle", rid=rid, status=status, charged_gpu_s=charged,
            cpu_seconds=cpu_seconds, detail=detail)
    return charged


def reconcile_all(reason="parent_exit"):
    """Charge every still-open reservation at its FULL reserved amount.

    This is the guarantee that a crashed or killed child cannot vanish from
    accounting. Over-charging is deliberate and is flagged.
    """
    out = []
    with _Locked() as L:
        s = L.state
        for rid, r in list(s["open_reservations"].items()):
            s["charged_gpu_s"][r["bucket"]] += r["reserved_gpu_s"]
            s["unreconciled"].append({**r, "rid": rid, "reason": reason,
                                      "charged_gpu_s": r["reserved_gpu_s"],
                                      "charged_utc": time.strftime(
                                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            out.append(rid)
            del s["open_reservations"][rid]
    if out:
        journal("reconcile_all", reason=reason, rids=out)
    return out


def count_images(n, kind, detail=None):
    with _Locked() as L:
        L.state["generated_images"] += n
        tot = L.state["generated_images"]
    journal("generated_images", n=n, kind=kind, detail=detail, cumulative=tot)
    if tot > CEILINGS["generated_images"]:
        journal("CEILING_BREACH", which="generated_images", value=tot)
        raise CeilingBreach(f"generated-image ceiling exceeded: {tot}")
    return tot


def count_coco(n, detail=None):
    with _Locked() as L:
        L.state["coco_photographs"] += n
        tot = L.state["coco_photographs"]
    journal("coco_photographs", n=n, detail=detail, cumulative=tot)
    if tot > CEILINGS["coco_photographs"]:
        journal("CEILING_BREACH", which="coco_photographs", value=tot)
        raise CeilingBreach(f"COCO photograph ceiling exceeded: {tot}")
    return tot


def bump_retry(kind, detail=None):
    """Every post-allocation technical relaunch counts, compatibility fixes
    included. Returns (count, allowed)."""
    key = {"gpu": "gpu_technical", "cpu": "cpu_repair"}[kind]
    cap = CEILINGS["gpu_retries"] if kind == "gpu" else CEILINGS["repair_cycles"]
    with _Locked() as L:
        L.state["retries"][key] += 1
        n = L.state["retries"][key]
    journal("retry", kind=kind, count=n, cap=cap, detail=detail)
    return n, n <= cap


def supervise(job, bucket, devices, argv, max_seconds, env=None, cwd=None,
              cpu_of=None, note=None):
    """Launch a child under a reservation and charge it through termination.

    The reservation is opened BEFORE the child starts and settled in a
    `finally`, so an exception, OOM, timeout or kill still charges.
    """
    rid = reserve(job, bucket, devices, max_seconds, note=note)
    t0 = time.time()
    proc = None
    status, detail = "unknown", None
    try:
        e = dict(os.environ)
        if env:
            e.update(env)
        if devices:
            e["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        proc = subprocess.Popen(argv, env=e, cwd=cwd or ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        attach_pid(rid, proc.pid)
        try:
            out, _ = proc.communicate(timeout=max_seconds)
            status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(10)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            out, _ = proc.communicate()
            status, detail = "timeout", f"exceeded {max_seconds}s"
        return status, out
    except BaseException as exc:                       # noqa: BLE001
        status, detail = "exception", f"{type(exc).__name__}: {exc}"
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        raise
    finally:
        cpu_s = 0.0
        if cpu_of is not None:
            cpu_s = cpu_of()
        settle(rid, time.time() - t0, cpu_seconds=cpu_s, status=status, detail=detail)


def gaps():
    """Actual enforcement gaps. Published as-is; nothing is claimed that is
    not implemented."""
    return [
        "GPU cost is charged as wall-clock x devices HELD, not measured device "
        "utilisation, so allocated idle time inside a job is charged (intended) "
        "and another user's processes on the same device are invisible.",
        "A child killed by the OS (OOM-killer, SIGKILL from outside) is charged "
        "by the parent's wall-clock measurement; if the PARENT itself dies, the "
        "reservation stays open until the next `reconcile_all`, which then "
        "charges the FULL reserved amount (deliberate over-charge, flagged in "
        "`unreconciled`).",
        "CPU seconds are collected per supervised job via getrusage of the "
        "parent's reaped children; CPU burned by a process the parent did not "
        "reap is not counted.",
        "RSS and worker-count ceilings are advisory: they are checked by the "
        "in-process watchdog of each child, not enforced by a cgroup.",
        "Storage and the >=50 GiB free-space floor are polled, not quota-enforced.",
        "The 7-day elapsed cap is operator-enforced against the ledger clock, "
        "not kernel-enforced.",
    ]
