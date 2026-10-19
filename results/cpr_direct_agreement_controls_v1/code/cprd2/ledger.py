"""Parent supervisor + durable allocation ledger for cpr_direct_agreement_controls_v1.

Guarantees (and their documented limits are listed by `coverage_and_gaps()`):
  * Every launch is preceded by a durable, fcntl-locked RESERVATION of
    (max_wall_seconds + termination_grace_seconds) x allocated_devices.
  * Every attempt is recorded in an append-only JSONL journal (reserve, launch,
    samples, termination request, confirmed termination, charge, refusals).
  * Charges are ACTUAL allocation: launch -> CONFIRMED process-tree termination
    or reaping, x allocated devices, for successes and failures alike.
    Charges are NEVER clipped to the reservation; an overrun is charged in full
    and recorded in `overruns`.
  * After a supervisor failure, `reconcile()` finds surviving children, kills
    and confirms their process trees, charges them, and closes the reservation
    BEFORE any further launch is allowed (`launch` refuses while stale
    reservations exist).
  * Caps are enforced at reservation time (GPU aggregate, entry/technical
    subcap, protected evaluation reserve, image counters, COCO photographs,
    aggregate workers, free space, science-stop clock).
"""
import fcntl
import json
import os
import resource
import shutil
import signal
import subprocess
import threading
import time
import uuid

import psutil

from cprd2.common import ROOT, utc

LEDGER_DIR = os.environ.get("CPRD2_LEDGER_DIR", os.path.join(ROOT, "artifacts", "ledger"))
STATE = os.path.join(LEDGER_DIR, "state.json")
ATTEMPTS = os.path.join(LEDGER_DIR, "attempts.jsonl")
MONITOR = os.path.join(LEDGER_DIR, "monitor.jsonl")

FIRST_ACTION_EPOCH = 1792431062.0          # 2026-10-19T17:31:02Z
CAPS = {
    "elapsed_days": 7.0, "science_stop_days": 6.0,
    "gpu_device_hours": 24.0, "gpu_entry_technical_hours": 4.0,
    "gpu_protected_eval_publication_hours": 8.0,
    "images_total": 2200, "images_primary": 1928, "images_technical": 272,
    "coco_photographs": 600,
    "cpu_core_hours": 48.0, "aggregate_workers": 8, "rss_gib": 128.0,
    "storage_gib": 200.0, "min_free_gib": 50.0,
    "repair_cycles": 2, "repair_engineering_minutes": 30,
}
# aggregate RSS kill threshold sits below the 128 GiB cap to leave margin for sampling lag
RSS_KILL_GIB = 120.0
BUCKETS = ("entry", "main", "reserve")      # entry = entry/qualification/technical; reserve = protected 8 h
POLL_S = 5.0
STORAGE_DIRS = [ROOT]                       # additional task storage = growth of these dirs


class CeilingBreach(RuntimeError):
    pass


def _caps_override():
    p = os.environ.get("CPRD2_CAPS_JSON")
    if p:
        CAPS.update(json.load(open(p)))


_caps_override()


def now():
    return time.time()


def _blank():
    return {
        "created_utc": utc(), "charged_gpu_s": {b: 0.0 for b in BUCKETS},
        "charged_cpu_s": 0.0, "images": {"primary": 0, "technical": 0,
                                         "reserved_primary": 0, "reserved_technical": 0},
        "coco_photographs": 0, "open": {}, "n_closed": 0, "overruns": [],
        "repair_cycles": [], "refusals": 0, "attempt_seq": 0,
        "peak": {"rss_gib": 0.0, "storage_gib_growth": 0.0, "threads": 0,
                 "min_free_gib": None},
        "storage_baseline_bytes": None,
    }


class _Locked:
    def __enter__(self):
        os.makedirs(LEDGER_DIR, exist_ok=True)
        self.fh = open(STATE, "a+")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        self.fh.seek(0)
        raw = self.fh.read()
        self.state = json.loads(raw) if raw.strip() else _blank()
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.fh.seek(0)
            self.fh.truncate()
            json.dump(self.state, self.fh, indent=1, sort_keys=True)
            self.fh.flush()
            os.fsync(self.fh.fileno())
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()
        return False


def journal(event, path=None, **payload):
    path = path or ATTEMPTS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps({"ts_utc": utc(), "epoch": round(now(), 3), "pid": os.getpid(),
                       "event": event, **payload}, default=str, sort_keys=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def read():
    with _Locked() as L:
        return json.loads(json.dumps(L.state))


def totals():
    s = read()
    g = s["charged_gpu_s"]
    open_res = {b: sum(r["reserved_gpu_s"] for r in s["open"].values() if r["bucket"] == b)
                for b in BUCKETS}
    return {"charged_gpu_h": {b: g[b] / 3600 for b in BUCKETS},
            "charged_gpu_h_total": sum(g.values()) / 3600,
            "open_reserved_gpu_h": {b: open_res[b] / 3600 for b in BUCKETS},
            "cpu_core_h": s["charged_cpu_s"] / 3600, "images": s["images"],
            "coco_photographs": s["coco_photographs"],
            "repair_cycles": len(s["repair_cycles"]), "n_open": len(s["open"]),
            "overruns": len(s["overruns"]), "peak": s["peak"]}


# ----------------------------------------------------------------- clocks
def science_deadline_epoch():
    return FIRST_ACTION_EPOCH + CAPS["science_stop_days"] * 86400


def hard_deadline_epoch():
    return FIRST_ACTION_EPOCH + CAPS["elapsed_days"] * 86400


# ----------------------------------------------------------------- GPU use check
def _gpu_foreign_users(devices):
    """Return pids using any requested device that are not in this supervisor's trees."""
    if not devices:
        return []
    try:
        idx = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        uuid2idx = {l.split(",")[1].strip(): int(l.split(",")[0]) for l in idx.strip().splitlines()}
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    except Exception as exc:           # a failed probe is a refusal, not a pass
        return [f"probe_failed:{exc}"]
    mine = set()
    for r in _LIVE.values():
        try:
            p = psutil.Process(r["pid"])
            mine.add(p.pid)
            mine.update(c.pid for c in p.children(recursive=True))
        except Exception:
            pass
    out = []
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        u, pid = [x.strip() for x in line.split(",")]
        if uuid2idx.get(u) in devices and int(pid) not in mine:
            out.append(int(pid))
    return out


# ----------------------------------------------------------------- reservations
_LIVE = {}                 # rid -> {"pid":..,"proc":psutil.Process, "t_launch":..}
_LIVE_LOCK = threading.Lock()


def _storage_bytes():
    tot = 0
    for d in STORAGE_DIRS:
        try:
            out = subprocess.check_output(["du", "-sb", d], text=True, timeout=120)
            tot += int(out.split()[0])
        except Exception:
            pass
    return tot


def reserve(job, bucket, devices, max_wall_s, grace_s, workers, kind="planned",
            images_primary=0, images_technical=0, cycle=None, note=None):
    assert bucket in BUCKETS, bucket
    n_dev = len(devices)
    reserved = (float(max_wall_s) + float(grace_s)) * n_dev
    rid = f"{job}-{uuid.uuid4().hex[:8]}"
    with _Locked() as L:
        s = L.state
        stale = [r for r, v in s["open"].items() if not _supervisor_alive(v)]
        if stale:
            s["refusals"] += 1
            raise CeilingBreach(f"stale reservations present; reconcile() first: {stale}")
        g = s["charged_gpu_s"]
        op = {b: sum(v["reserved_gpu_s"] for v in s["open"].values() if v["bucket"] == b)
              for b in BUCKETS}
        cap_tot = CAPS["gpu_device_hours"] * 3600
        if sum(g.values()) + sum(op.values()) + reserved > cap_tot:
            s["refusals"] += 1
            raise CeilingBreach("GPU aggregate cap would be exceeded")
        if bucket == "entry" and g["entry"] + op["entry"] + reserved > CAPS["gpu_entry_technical_hours"] * 3600:
            s["refusals"] += 1
            raise CeilingBreach("entry/technical subcap would be exceeded")
        protected = CAPS["gpu_protected_eval_publication_hours"] * 3600
        if bucket in ("entry", "main"):
            non_res = g["entry"] + g["main"] + op["entry"] + op["main"] + reserved
            if non_res > cap_tot - protected:
                s["refusals"] += 1
                raise CeilingBreach("protected evaluation/publication reserve would be breached")
        if bucket == "reserve":
            # reserve-bucket work may also consume unused non-reserve hours, never more than total
            pass
        if kind == "planned" and now() + float(max_wall_s) + float(grace_s) > science_deadline_epoch() \
                and bucket != "reserve":
            s["refusals"] += 1
            raise CeilingBreach("reservation would run past the day-6 science stop")
        if now() + float(max_wall_s) + float(grace_s) > hard_deadline_epoch():
            s["refusals"] += 1
            raise CeilingBreach("reservation would run past the 7-day cap")
        im = s["images"]
        if images_primary and im["primary"] + im["reserved_primary"] + images_primary > CAPS["images_primary"]:
            s["refusals"] += 1
            raise CeilingBreach("primary image cap would be exceeded")
        if images_technical and im["technical"] + im["reserved_technical"] + images_technical > CAPS["images_technical"]:
            s["refusals"] += 1
            raise CeilingBreach("technical image cap would be exceeded")
        if (im["primary"] + im["technical"] + im["reserved_primary"] + im["reserved_technical"]
                + images_primary + images_technical) > CAPS["images_total"]:
            s["refusals"] += 1
            raise CeilingBreach("total image cap would be exceeded")
        workers_now = sum(v["workers"] for v in s["open"].values())
        if workers_now + workers > CAPS["aggregate_workers"]:
            s["refusals"] += 1
            raise CeilingBreach("aggregate worker cap would be exceeded")
        free = shutil.disk_usage("/data").free / 2**30
        if free < CAPS["min_free_gib"]:
            s["refusals"] += 1
            raise CeilingBreach(f"free space {free:.1f} GiB below floor")
        s["images"]["reserved_primary"] += images_primary
        s["images"]["reserved_technical"] += images_technical
        s["attempt_seq"] += 1
        s["open"][rid] = {
            "job": job, "bucket": bucket, "devices": list(devices), "workers": workers,
            "max_wall_s": float(max_wall_s), "grace_s": float(grace_s),
            "reserved_gpu_s": reserved, "kind": kind, "cycle": cycle, "note": note,
            "images_primary": images_primary, "images_technical": images_technical,
            "reserved_utc": utc(), "reserved_epoch": now(), "supervisor_pid": os.getpid(),
            "supervisor_ctime": psutil.Process().create_time(), "pid": None, "pid_ctime": None,
            "launch_epoch": None, "seq": s["attempt_seq"],
        }
    journal("reserve", rid=rid, job=job, bucket=bucket, devices=list(devices), kind=kind,
            reserved_gpu_s=reserved, max_wall_s=max_wall_s, grace_s=grace_s, cycle=cycle,
            images_primary=images_primary, images_technical=images_technical, note=note)
    return rid


def _supervisor_alive(v):
    try:
        p = psutil.Process(v["supervisor_pid"])
        return abs(p.create_time() - v["supervisor_ctime"]) < 1.0 and p.is_running() \
            and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def release_unlaunched(rid, reason):
    """A reservation whose launch never happened holds no device: close it, charge 0, journal it."""
    with _Locked() as L:
        r = L.state["open"].pop(rid, None)
        if r:
            L.state["images"]["reserved_primary"] -= r["images_primary"]
            L.state["images"]["reserved_technical"] -= r["images_technical"]
            L.state["n_closed"] += 1
    journal("release_unlaunched", rid=rid, reason=reason, charged_gpu_s=0.0)


def settle(rid, end_epoch, cpu_s, status, detail=None, images_actual=None, end_observed=True):
    """Charge the reservation: (confirmed end - launch) x devices. NEVER clipped."""
    with _Locked() as L:
        s = L.state
        r = s["open"].pop(rid, None)
        if r is None:
            journal("settle_missing", rid=rid, status=status)
            return None
        launch = r["launch_epoch"] if r["launch_epoch"] is not None else r["reserved_epoch"]
        held = max(0.0, end_epoch - launch)
        n_dev = len(r["devices"])
        charged = held * n_dev
        if not end_observed:
            charged = max(charged, r["reserved_gpu_s"])
        s["charged_gpu_s"][r["bucket"]] += charged
        s["charged_cpu_s"] += float(cpu_s)
        over = charged - r["reserved_gpu_s"]
        if over > 1e-6:
            s["overruns"].append({"rid": rid, "job": r["job"], "over_gpu_s": over,
                                  "charged_gpu_s": charged, "reserved_gpu_s": r["reserved_gpu_s"]})
        im = s["images"]
        im["reserved_primary"] -= r["images_primary"]
        im["reserved_technical"] -= r["images_technical"]
        if images_actual:
            im["primary"] += int(images_actual.get("primary", 0))
            im["technical"] += int(images_actual.get("technical", 0))
        s["n_closed"] += 1
    journal("charge", rid=rid, job=r["job"], status=status, held_s=held, devices=r["devices"],
            charged_gpu_s=charged, reserved_gpu_s=r["reserved_gpu_s"], overrun_gpu_s=max(0.0, over),
            cpu_s=cpu_s, images_actual=images_actual, end_observed=end_observed, detail=detail,
            bucket=r["bucket"], kind=r["kind"], cycle=r["cycle"])
    return charged


# ----------------------------------------------------------------- process trees
def _tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def terminate_tree(root_pid, root_ctime, grace_s, sig_first=signal.SIGTERM):
    """TERM -> wait grace -> KILL, over the process group AND every descendant. Returns
    (confirmed_epoch, survivors_after, killed_hard)."""
    try:
        root = psutil.Process(root_pid)
        if abs(root.create_time() - root_ctime) > 1.0:
            return now(), [], False
    except psutil.NoSuchProcess:
        return now(), [], False
    procs = _tree(root)
    try:
        os.killpg(os.getpgid(root_pid), sig_first)
    except Exception:
        pass
    for p in procs:
        try:
            p.send_signal(sig_first)
        except Exception:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=grace_s)
    hard = False
    if alive:
        hard = True
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
        try:
            os.killpg(os.getpgid(root_pid), signal.SIGKILL)
        except Exception:
            pass
        gone2, alive = psutil.wait_procs(alive, timeout=30)
    return now(), [p.pid for p in alive], hard


def reconcile(reason="supervisor_restart", grace_s=20):
    """Close every reservation whose supervisor died: kill + confirm surviving children, charge."""
    closed = []
    with _Locked() as L:
        stale = {rid: dict(v) for rid, v in L.state["open"].items() if not _supervisor_alive(v)}
    for rid, v in stale.items():
        end_obs, survivors, hard = False, [], False
        end = now()
        if v["pid"] is not None:
            try:
                alive = abs(psutil.Process(v["pid"]).create_time() - v["pid_ctime"]) < 1.0
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                end, survivors, hard = terminate_tree(v["pid"], v["pid_ctime"], grace_s)
                end_obs = not survivors
        journal("reconcile", rid=rid, job=v["job"], reason=reason, had_live_child=bool(end_obs),
                survivors_after=survivors, killed_hard=hard)
        # child already gone (end time unobserved) -> charge the FULL reservation, flagged
        settle(rid, end, 0.0, "reconciled_after_supervisor_failure",
               detail={"reason": reason, "survivors_after": survivors, "hard_kill": hard},
               images_actual=_decode_counts_for(v["job"]) if v.get("images_primary") or v.get("images_technical") else None,
               end_observed=end_obs)
        closed.append(rid)
    return closed


def _decode_counts_for(job):
    p = os.path.join(ROOT, "artifacts", "decode_counts", f"{job}.jsonl")
    out = {"primary": 0, "technical": 0}
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
                out[d["kind"]] += int(d["n"])
            except Exception:
                pass
    return out


# ----------------------------------------------------------------- repair cycles
def open_repair_cycle(description, patch_commit, engineering_minutes):
    with _Locked() as L:
        s = L.state
        if len(s["repair_cycles"]) >= CAPS["repair_cycles"]:
            raise CeilingBreach("no third technical repair cycle is permitted")
        if engineering_minutes > CAPS["repair_engineering_minutes"]:
            raise CeilingBreach("repair cycle exceeds 30 engineering minutes")
        n = len(s["repair_cycles"]) + 1
        s["repair_cycles"].append({"cycle": n, "description": description,
                                   "patch_commit": patch_commit,
                                   "engineering_minutes": engineering_minutes, "opened_utc": utc()})
    journal("repair_cycle", cycle=n, description=description, patch_commit=patch_commit,
            engineering_minutes=engineering_minutes)
    return n


def add_coco(n, detail=None):
    with _Locked() as L:
        L.state["coco_photographs"] += n
        tot = L.state["coco_photographs"]
        if tot > CAPS["coco_photographs"]:
            raise CeilingBreach(f"COCO photograph cap exceeded: {tot}")
    journal("coco_photographs", n=n, cumulative=tot, detail=detail)
    return tot


# ----------------------------------------------------------------- supervised launch
def run_job(job, bucket, devices, argv, max_wall_s, grace_s=60, workers=2, env=None, cwd=None,
            log_path=None, kind="planned", images_primary=0, images_technical=0, cycle=None,
            note=None, rss_limit_gib=None, stdin_devnull=True):
    """Reserve -> launch -> monitor -> terminate/reap -> charge. Blocking. Returns a result dict."""
    reconcile_needed = [1 for v in read()["open"].values() if not _supervisor_alive(v)]
    if reconcile_needed:
        reconcile("pre_launch_check")
    rid = reserve(job, bucket, devices, max_wall_s, grace_s, workers, kind=kind,
                  images_primary=images_primary, images_technical=images_technical,
                  cycle=cycle, note=note)
    foreign = _gpu_foreign_users(devices)
    if foreign:
        release_unlaunched(rid, f"foreign GPU users {foreign}")
        raise CeilingBreach(f"requested devices {devices} in use by foreign processes {foreign}")
    e = dict(os.environ)
    if env:
        e.update(env)
    e["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices) if devices else ""
    e["CPRD2_JOB_ID"] = job
    e["CPRD2_RID"] = rid
    log_path = log_path or os.path.join(ROOT, "logs", f"{job}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "ab")
    t_launch = now()
    status, detail, rc = "unknown", None, None
    cpu_s, peak_rss, peak_thr = 0.0, 0.0, 0
    proc = None
    end_epoch = None
    hard = False
    rss_lim = rss_limit_gib or RSS_KILL_GIB
    try:
        proc = subprocess.Popen(argv, env=e, cwd=cwd or ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL if stdin_devnull else None,
                                start_new_session=True)
        pp = psutil.Process(proc.pid)
        with _Locked() as L:
            v = L.state["open"][rid]
            v["pid"], v["pid_ctime"], v["launch_epoch"] = proc.pid, pp.create_time(), t_launch
        with _LIVE_LOCK:
            _LIVE[rid] = {"pid": proc.pid}
        journal("launch", rid=rid, job=job, pid=proc.pid, argv=argv, devices=list(devices), log=log_path)
        deadline = t_launch + float(max_wall_s)
        last_du = 0.0
        while True:
            try:
                wpid, wstatus, ru = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:
                wpid, wstatus, ru = proc.pid, 0, None
            if wpid == proc.pid:
                rc = os.waitstatus_to_exitcode(wstatus)
                proc.returncode = rc
                if ru is not None:
                    cpu_s = max(cpu_s, ru.ru_utime + ru.ru_stime)
                status = "ok" if rc == 0 else f"exit_{rc}"
                # reap any descendants left behind by the child
                end_epoch, surv, hard = terminate_tree_leftovers(pp, grace_s)
                if surv:
                    detail = {"survivors_after_exit": surv}
                break
            tr = _tree(pp)
            rss = 0
            thr = 0
            cpu = 0.0
            for p in tr:
                try:
                    rss += p.memory_info().rss
                    thr += p.num_threads()
                    c = p.cpu_times()
                    cpu += c.user + c.system + c.children_user + c.children_system
                except Exception:
                    pass
            cpu_s = max(cpu_s, cpu)
            peak_rss = max(peak_rss, rss / 2**30)
            peak_thr = max(peak_thr, thr)
            agg = aggregate_rss_gib()
            reason = None
            if agg > rss_lim:
                reason = f"aggregate_rss_{agg:.1f}GiB>{rss_lim}"
            elif rss / 2**30 > rss_lim:
                reason = f"job_rss_{rss/2**30:.1f}GiB>{rss_lim}"
            elif now() > deadline:
                reason = f"timeout_{max_wall_s}s"
            elif shutil.disk_usage("/data").free / 2**30 < CAPS["min_free_gib"]:
                reason = "free_space_below_floor"
            if now() - last_du > 120:
                last_du = now()
                sb = _storage_bytes()
                _note_peaks(sb, peak_rss, peak_thr)
                journal("sample", path=MONITOR, rid=rid, job=job, rss_gib=round(rss / 2**30, 2),
                        agg_rss_gib=round(agg, 2), threads=thr, cpu_s=round(cpu_s, 1),
                        storage_gib=round(sb / 2**30, 2),
                        free_gib=round(shutil.disk_usage("/data").free / 2**30, 1))
            if reason:
                journal("termination_requested", rid=rid, job=job, reason=reason)
                end_epoch, surv, hard = terminate_tree(proc.pid, pp.create_time(), grace_s)
                try:
                    wpid, wstatus, ru = os.wait4(proc.pid, 0)
                    rc = os.waitstatus_to_exitcode(wstatus)
                    proc.returncode = rc
                    cpu_s = max(cpu_s, ru.ru_utime + ru.ru_stime)
                except ChildProcessError:
                    pass
                end_epoch = max(end_epoch, now()) if end_epoch else now()
                status = "killed_" + reason.split("_")[0]
                detail = {"reason": reason, "hard_kill": hard, "survivors": surv}
                break
            time.sleep(POLL_S)
        return {"rid": rid, "status": status, "returncode": rc, "peak_rss_gib": peak_rss,
                "cpu_s": cpu_s, "held_s": (end_epoch or now()) - t_launch, "log": log_path,
                "hard_kill": hard}
    except BaseException as exc:                              # noqa: BLE001
        status, detail = "supervisor_exception", f"{type(exc).__name__}: {exc}"
        if proc is not None:
            try:
                end_epoch, _, hard = terminate_tree(proc.pid, psutil.Process(proc.pid).create_time(), grace_s)
            except Exception:
                end_epoch = now()
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.pop(rid, None)
        logf.close()
        if proc is None:
            release_unlaunched(rid, "popen_failed")
        else:
            imgs = _decode_counts_for(job) if (images_primary or images_technical) else None
            settle(rid, end_epoch or now(), cpu_s, status, detail=detail, images_actual=imgs)


def terminate_tree_leftovers(root_proc, grace_s):
    """After the direct child exited: any orphaned descendants of this job are terminated."""
    left = []
    try:
        left = [p for p in psutil.process_iter(["pid", "ppid"])
                if p.info["ppid"] == 1 and _same_session(p, root_proc)]
    except Exception:
        pass
    if not left:
        return now(), [], False
    for p in left:
        try:
            p.terminate()
        except Exception:
            pass
    gone, alive = psutil.wait_procs(left, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    psutil.wait_procs(alive, timeout=30)
    return now(), [p.pid for p in alive if p.is_running()], bool(alive)


def _same_session(p, root_proc):
    try:
        return os.getsid(p.pid) == root_proc.pid
    except Exception:
        return False


def aggregate_rss_gib():
    tot = 0
    with _LIVE_LOCK:
        pids = [v["pid"] for v in _LIVE.values()]
    for pid in pids:
        try:
            for p in _tree(psutil.Process(pid)):
                try:
                    tot += p.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
    return tot / 2**30


def _note_peaks(storage_bytes, rss_gib, thr):
    with _Locked() as L:
        s = L.state
        if s["storage_baseline_bytes"] is None:
            s["storage_baseline_bytes"] = storage_bytes
        g = (storage_bytes - s["storage_baseline_bytes"]) / 2**30
        pk = s["peak"]
        pk["storage_gib_growth"] = max(pk["storage_gib_growth"], g)
        pk["rss_gib"] = max(pk["rss_gib"], rss_gib)
        pk["threads"] = max(pk["threads"], thr)
        fr = shutil.disk_usage("/data").free / 2**30
        pk["min_free_gib"] = fr if pk["min_free_gib"] is None else min(pk["min_free_gib"], fr)


def coverage_and_gaps():
    return {
        "monitored": [
            "GPU: wall-clock x allocated devices from launch to confirmed process-tree termination",
            "RSS: psutil sum over the launched process tree, every 5 s; aggregate over all live jobs; "
            "supervisor kills the tree above 120 GiB aggregate (cap 128 GiB)",
            "CPU: max(psutil process-tree cpu_times incl. reaped children, wait4 rusage of the direct child)",
            "threads: sampled process-tree num_threads, peak recorded (advisory; BLAS/OMP worker count is "
            "enforced only by OMP/MKL/OPENBLAS_NUM_THREADS=2 per job and <=4 concurrent jobs)",
            "storage: `du -sb` of the task root every 120 s while jobs run; growth vs. first sample recorded; "
            "free space on /data polled every 5 s (kill below 50 GiB)",
            "images: reserved before launch, actual decodes from per-job append-only decode counters",
        ],
        "gaps": [
            "Storage is polled at 120 s granularity and only for the task root; transient spikes between "
            "polls and files written outside the root (TMPDIR/XDG caches on /data) are not attributed. "
            "Reported storage peaks are lower bounds, not exact compliance claims.",
            "CPU core-hours exclude CPU used by processes that escaped the launched session and were not "
            "descendants at a sampling instant; reported CPU is a lower bound.",
            "Thread count is not a BLAS-worker count; the 8-worker aggregate is enforced by launch-time "
            "reservation of declared workers (2/job) plus env thread limits, not by cgroup.",
            "GPU memory (40 GB/device) is not monitored by the supervisor; OOM surfaces as a child failure "
            "and is charged like any failure.",
            "RSS counts resident file-backed mapped pages (safetensors mmap) as RSS; this overstates the "
            "private footprint (conservative for the cap).",
            "Other users' processes on the same machine are invisible to the accounting except through the "
            "pre-launch foreign-process probe on the requested devices.",
            "If the supervisor is killed uncleanly, children whose end time was unobserved are charged at "
            "their full reservation (conservative over-charge, flagged).",
        ],
    }
