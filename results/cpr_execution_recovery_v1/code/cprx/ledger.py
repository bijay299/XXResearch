"""Parent supervisor + durable allocation ledger for cpr_execution_recovery_v1.

New ledger. Opening balances carry the closed chain's KNOWN charges (the historical ledger is not rewritten; its
CPU accounting was incomplete, see AMENDMENT_01). Guarantees and their documented limits (see `coverage_and_gaps()`):

  * Every launch is preceded by a durable, fcntl-locked RESERVATION of (max_wall + grace) x allocated devices (GPU-s),
    (max_wall + grace) x allocated CPU cores (CPU-core-s), declared peak RSS, declared task storage and images.
  * Every attempt is recorded in an append-only fsync'd JSONL journal (reserve, launch, samples, termination, charge).
  * GPU charge = ACTUAL allocation: launch -> CONFIRMED process-tree termination/reaping, x devices, for successes and
    failures alike, never clipped (overruns are charged in full and recorded).
  * `reconcile()` kills + confirms + charges every reservation whose supervisor died; launches are refused while stale
    reservations exist (run_job reconciles first).
  * Enforced before launch: GPU aggregate / engineering sub-cap (incl. carried) / protected evaluation+publication reserve;
    CPU-core-hour cap (carried + charged + live + reserved); storage growth cap (declared + current); free-space floor;
    projected host RAM and aggregate RSS; image caps; deadlines (engineering window, day-6 science stop, 7-day cap).
  * Enforced during a run (5 s polls): process-tree RSS, aggregate RSS, free space, task storage growth (30 s du),
    CPU-core-hours (charged + live tree CPU), wall timeout. Breach -> TERM, grace, KILL of the whole tree, charge.
  * CPU concurrency is bounded by DISJOINT CPU-affinity slices of an 8-core pool (sched_setaffinity in the child) plus
    OMP/MKL/OPENBLAS/NUMEXPR thread env = slice size; effective core use is sampled from cpu_times deltas.
"""
import fcntl
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid

import psutil

from cprx.common import ROOT, REPO, utc

LEDGER_DIR = os.environ.get("CPRX_LEDGER_DIR", os.path.join(ROOT, "artifacts", "ledger"))
STATE = os.path.join(LEDGER_DIR, "state.json")
ATTEMPTS = os.path.join(LEDGER_DIR, "attempts.jsonl")
MONITOR = os.path.join(LEDGER_DIR, "monitor.jsonl")

# ---- clocks (UTC epochs) ----
CHAIN_START_EPOCH = 1792431062.0            # 2026-10-19T17:31:02Z  (chain start; one week cap)
ENGINEERING_END_EPOCH = 1792459755.0        # 2026-10-20T01:29:15Z  (6 h after the recovery prompt, 19:29:15Z)
SCIENCE_STOP_EPOCH = CHAIN_START_EPOCH + 6 * 86400
HARD_DEADLINE_EPOCH = CHAIN_START_EPOCH + 7 * 86400

CAPS = {
    "gpu_device_hours": 24.0,
    "gpu_engineering_hours": 4.0,                 # cumulative recovery-chain engineering incl. carried
    "gpu_protected_eval_publication_hours": 8.0,
    "images_total": 2200, "images_primary": 1928, "images_technical": 272,
    "cpu_core_hours": 48.0, "cpu_pool_cores": 8, "rss_gib": 128.0,
    "storage_gib": 200.0, "min_free_gib": 50.0,
}
RSS_KILL_GIB = 120.0                              # kill margin below 128 GiB for sampling lag
CPU_KILL_FRACTION = 0.97
STORAGE_KILL_FRACTION = 0.95
RAM_MARGIN_GIB = 24.0                             # host RAM that must remain available after the declared job RSS
LOAD_GRACE_S = 240.0                              # young jobs count at their declared RSS (model-load peak)
BUCKETS = ("eng", "main", "reserve")              # eng = engineering/technical; reserve = protected 8 h
POLL_S = 5.0
DU_EVERY_S = 30.0

OPENING = {
    "gpu_s_eng": 725.4364926815033,               # carried: old chain's supervised GPU subtotal (0.2015101369 GPU-h)
    "cpu_s": 2037.790186,                         # carried: old chain's supervised CPU subtotal (incomplete scope)
    "images": 0,
    "note": ("Carried from cpr_direct_agreement_controls_v1 supervised subtotals. Historical CPU accounting was "
             "incomplete (setup/diagnostics/verification CPU and peak storage were not metered); true historical "
             "totals are unknown and this opening balance is a LOWER BOUND, disclosed rather than estimated."),
}
STORAGE_DIRS = [ROOT, os.path.join(REPO, "results", "cpr_execution_recovery_v1")]


class CeilingBreach(RuntimeError):
    pass


def _caps_override():
    p = os.environ.get("CPRX_CAPS_JSON")
    if p:
        CAPS.update(json.load(open(p)))


_caps_override()


def now():
    return time.time()


def _pick_pool():
    try:
        pct = psutil.cpu_percent(interval=2.0, percpu=True)
        cand = sorted(range(0, 64), key=lambda i: (pct[i], i))[:CAPS["cpu_pool_cores"]]
        return sorted(cand)
    except Exception:
        return list(range(CAPS["cpu_pool_cores"]))


def _blank():
    return {
        "created_utc": utc(), "opening_balance": OPENING,
        "charged_gpu_s": {"eng": OPENING["gpu_s_eng"], "main": 0.0, "reserve": 0.0},
        "charged_cpu_s": OPENING["cpu_s"], "cpu_s_new": 0.0,
        "images": {"primary": 0, "technical": 0, "reserved_primary": 0, "reserved_technical": 0},
        "open": {}, "n_closed": 0, "overruns": [], "refusals": 0, "attempt_seq": 0,
        "cpu_pool": _pick_pool(),
        "peak": {"rss_gib": 0.0, "agg_rss_gib": 0.0, "storage_gib": 0.0, "threads": 0, "min_free_gib": None,
                 "effective_cores": 0.0},
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
    line = json.dumps({"ts_utc": utc(), "epoch": round(now(), 3), "pid": os.getpid(), "event": event, **payload},
                      default=str, sort_keys=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def read():
    with _Locked() as L:
        return json.loads(json.dumps(L.state))


def _open_sum(s, key, bucket=None):
    return sum(v[key] for v in s["open"].values() if bucket is None or v["bucket"] == bucket)


def totals():
    s = read()
    g = s["charged_gpu_s"]
    return {"charged_gpu_h": {b: g[b] / 3600 for b in BUCKETS}, "charged_gpu_h_total": sum(g.values()) / 3600,
            "open_reserved_gpu_h": {b: _open_sum(s, "reserved_gpu_s", b) / 3600 for b in BUCKETS},
            "cpu_core_h_total_incl_carried": s["charged_cpu_s"] / 3600, "cpu_core_h_new": s["cpu_s_new"] / 3600,
            "open_reserved_cpu_core_h": _open_sum(s, "reserved_cpu_s") / 3600, "images": s["images"],
            "n_open": len(s["open"]), "overruns": len(s["overruns"]), "refusals": s["refusals"], "peak": s["peak"],
            "cpu_pool": s["cpu_pool"]}


# ------------------------------------------------------------------ GPU/host probes
_LIVE = {}
_LIVE_LOCK = threading.Lock()


def _my_pids():
    mine = set()
    with _LIVE_LOCK:
        pids = [v["pid"] for v in _LIVE.values()]
    for pid in pids:
        try:
            p = psutil.Process(pid)
            mine.add(p.pid)
            mine.update(c.pid for c in p.children(recursive=True))
        except Exception:
            pass
    return mine


def _gpu_foreign(devices):
    """Foreign compute processes OR >1 GiB resident memory on a requested device (other users' work is respected)."""
    if not devices:
        return []
    try:
        idx = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                                      text=True)
        rows = [[x.strip() for x in l.split(",")] for l in idx.strip().splitlines()]
        uuid2idx = {r[1]: int(r[0]) for r in rows}
        used = {int(r[0]): float(r[2]) for r in rows}
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    except Exception as exc:
        return [f"probe_failed:{exc}"]
    mine = _my_pids()
    out = []
    for line in apps.strip().splitlines():
        if line.strip():
            u, pid = [x.strip() for x in line.split(",")]
            if uuid2idx.get(u) in devices and int(pid) not in mine:
                out.append(int(pid))
    live_devs = set()
    with _LIVE_LOCK:
        for v in _LIVE.values():
            live_devs.update(v.get("devices", []))
    for d in devices:
        if d not in live_devs and used.get(d, 0) > 1024:
            out.append(f"gpu{d}_mem_used_{used[d]:.0f}MiB")
    return out


def _supervisor_alive(v):
    try:
        p = psutil.Process(v["supervisor_pid"])
        return abs(p.create_time() - v["supervisor_ctime"]) < 1.0 and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _du_bytes(paths=None):
    tot = 0
    for d in (paths or STORAGE_DIRS):
        if not os.path.exists(d):
            continue
        try:
            tot += int(subprocess.check_output(["du", "-sb", d], text=True, timeout=300).split()[0])
        except Exception:
            pass
    return tot


def _live_rss_gib(young_declared=True):
    """Sum over live jobs of the process-tree RSS, counting young jobs at their declared peak."""
    tot = 0.0
    with _LIVE_LOCK:
        items = list(_LIVE.values())
    t = now()
    for v in items:
        cur = 0.0
        try:
            for p in _tree(psutil.Process(v["pid"])):
                try:
                    cur += p.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        cur /= 2**30
        tot += max(cur, v["declared_rss_gib"]) if (young_declared and t - v["t_launch"] < LOAD_GRACE_S) else cur
    return tot


# ------------------------------------------------------------------ reservations
def reserve(job, bucket, devices, max_wall_s, grace_s, cores, kind="planned", images_primary=0, images_technical=0,
            declared_rss_gib=8.0, declared_storage_gib=0.5, note=None):
    assert bucket in BUCKETS, bucket
    n_dev = len(devices)
    horizon = float(max_wall_s) + float(grace_s)
    reserved = horizon * n_dev
    reserved_cpu = horizon * cores
    rid = f"{job}-{uuid.uuid4().hex[:8]}"
    with _Locked() as L:
        s = L.state

        stale = [r for r, v in s["open"].items() if not _supervisor_alive(v)]
        msg = None
        g = s["charged_gpu_s"]
        capgpu = CAPS["gpu_device_hours"] * 3600
        protected = CAPS["gpu_protected_eval_publication_hours"] * 3600
        opb = {b: _open_sum(s, "reserved_gpu_s", b) for b in BUCKETS}
        im = s["images"]
        free = shutil.disk_usage("/data").free / 2**30
        used_cores = {c for v in s["open"].values() for c in v["cores"]}
        avail = [c for c in s["cpu_pool"] if c not in used_cores]
        sto = _du_bytes() / 2**30
        sto_open = _open_sum(s, "declared_storage_gib")
        ram_avail = psutil.virtual_memory().available / 2**30
        if stale:
            msg = f"stale reservations present; reconcile() first: {stale}"
        elif sum(g.values()) + sum(opb.values()) + reserved > capgpu:
            msg = "GPU aggregate cap would be exceeded"
        elif bucket == "eng" and g["eng"] + opb["eng"] + reserved > CAPS["gpu_engineering_hours"] * 3600:
            msg = "engineering GPU sub-cap (4 h cumulative incl. carried) would be exceeded"
        elif bucket in ("eng", "main") and (g["eng"] + g["main"] + opb["eng"] + opb["main"] + reserved) > capgpu - protected:
            msg = "protected 8 h evaluation/publication reserve would be breached"
        elif bucket == "eng" and now() + horizon > ENGINEERING_END_EPOCH:
            msg = "engineering reservation would run past the 6 h engineering window"
        elif bucket != "reserve" and kind == "planned" and now() + horizon > SCIENCE_STOP_EPOCH:
            msg = "reservation would run past the day-6 science stop"
        elif now() + horizon > HARD_DEADLINE_EPOCH:
            msg = "reservation would run past the 7-day cap"
        elif images_primary and im["primary"] + im["reserved_primary"] + images_primary > CAPS["images_primary"]:
            msg = "primary image cap would be exceeded"
        elif images_technical and im["technical"] + im["reserved_technical"] + images_technical > CAPS["images_technical"]:
            msg = "technical image cap would be exceeded"
        elif (im["primary"] + im["technical"] + im["reserved_primary"] + im["reserved_technical"]
              + images_primary + images_technical) > CAPS["images_total"]:
            msg = "total image cap would be exceeded"
        elif s["charged_cpu_s"] + _open_sum(s, "reserved_cpu_s") + reserved_cpu > CAPS["cpu_core_hours"] * 3600:
            msg = "CPU core-hour cap would be exceeded (carried + charged + open reservations + this reservation)"
        elif len(avail) < cores:
            msg = f"CPU pool exhausted: need {cores} cores, {len(avail)} free of {len(s['cpu_pool'])} (8-core aggregate cap)"
        elif free < CAPS["min_free_gib"] + declared_storage_gib:
            msg = f"free space {free:.1f} GiB would fall below floor"
        elif sto + sto_open + declared_storage_gib > CAPS["storage_gib"]:
            msg = f"task storage cap would be exceeded ({sto:.1f}+{sto_open:.1f}+{declared_storage_gib:.1f} GiB)"
        elif n_dev and ram_avail < declared_rss_gib + RAM_MARGIN_GIB + max(0.0, _live_rss_gib() - _live_rss_gib(False)):
            msg = f"projected host RAM insufficient: available {ram_avail:.0f} GiB, job {declared_rss_gib} GiB"
        elif _live_rss_gib() + declared_rss_gib > RSS_KILL_GIB:
            msg = f"projected aggregate RSS {_live_rss_gib():.0f}+{declared_rss_gib} GiB over {RSS_KILL_GIB} GiB"
        slice_ = avail[:cores]
        if msg:
            s["refusals"] += 1
        else:
            im["reserved_primary"] += images_primary
            im["reserved_technical"] += images_technical
            s["attempt_seq"] += 1
            s["open"][rid] = {
                "job": job, "bucket": bucket, "devices": list(devices), "cores": slice_, "workers": cores,
                "max_wall_s": float(max_wall_s), "grace_s": float(grace_s), "reserved_gpu_s": reserved,
                "reserved_cpu_s": reserved_cpu, "declared_rss_gib": declared_rss_gib,
                "declared_storage_gib": declared_storage_gib, "kind": kind, "note": note,
                "images_primary": images_primary, "images_technical": images_technical, "reserved_utc": utc(),
                "reserved_epoch": now(), "supervisor_pid": os.getpid(),
                "supervisor_ctime": psutil.Process().create_time(),
                "pid": None, "pid_ctime": None, "launch_epoch": None, "seq": s["attempt_seq"]}
    if msg:
        journal("refuse", job=job, bucket=bucket, reason=msg)
        raise CeilingBreach(msg)
    journal("reserve", rid=rid, job=job, bucket=bucket, devices=list(devices), cores=slice_, kind=kind,
            reserved_gpu_s=reserved, reserved_cpu_s=reserved_cpu, max_wall_s=max_wall_s, grace_s=grace_s,
            declared_rss_gib=declared_rss_gib, declared_storage_gib=declared_storage_gib,
            images_primary=images_primary, images_technical=images_technical, note=note)
    return rid


def release_unlaunched(rid, reason):
    with _Locked() as L:
        r = L.state["open"].pop(rid, None)
        if r:
            L.state["images"]["reserved_primary"] -= r["images_primary"]
            L.state["images"]["reserved_technical"] -= r["images_technical"]
            L.state["n_closed"] += 1
    journal("release_unlaunched", rid=rid, reason=reason, charged_gpu_s=0.0)


def settle(rid, end_epoch, cpu_s, status, detail=None, images_actual=None, end_observed=True):
    """Charge (confirmed end - launch) x devices; NEVER clipped. CPU = metered process-tree CPU seconds."""
    with _Locked() as L:
        s = L.state
        r = s["open"].pop(rid, None)
        if r is None:
            journal("settle_missing", rid=rid, status=status)
            return None
        launch = r["launch_epoch"] if r["launch_epoch"] is not None else r["reserved_epoch"]
        held = max(0.0, end_epoch - launch)
        charged = held * len(r["devices"])
        cpu_charge = float(cpu_s)
        if not end_observed:
            charged = max(charged, r["reserved_gpu_s"])
            cpu_charge = max(cpu_charge, r["reserved_cpu_s"])
        s["charged_gpu_s"][r["bucket"]] += charged
        s["charged_cpu_s"] += cpu_charge
        s["cpu_s_new"] += cpu_charge
        over = charged - r["reserved_gpu_s"]
        if over > 1e-6:
            s["overruns"].append({"rid": rid, "job": r["job"], "over_gpu_s": over, "charged_gpu_s": charged,
                                  "reserved_gpu_s": r["reserved_gpu_s"]})
        im = s["images"]
        im["reserved_primary"] -= r["images_primary"]
        im["reserved_technical"] -= r["images_technical"]
        if images_actual:
            im["primary"] += int(images_actual.get("primary", 0))
            im["technical"] += int(images_actual.get("technical", 0))
        s["n_closed"] += 1
    journal("charge", rid=rid, job=r["job"], status=status, held_s=held, devices=r["devices"], charged_gpu_s=charged,
            reserved_gpu_s=r["reserved_gpu_s"], overrun_gpu_s=max(0.0, over), cpu_s=cpu_charge,
            images_actual=images_actual, end_observed=end_observed, detail=detail, bucket=r["bucket"], kind=r["kind"])
    return charged


# ------------------------------------------------------------------ process trees
def _tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def terminate_tree(root_pid, root_ctime, grace_s, sig_first=signal.SIGTERM):
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
        settle(rid, end, 0.0, "reconciled_after_supervisor_failure",
               detail={"reason": reason, "survivors_after": survivors, "hard_kill": hard},
               images_actual=_decode_counts_for(v["job"]) if (v.get("images_primary") or v.get("images_technical")) else None,
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


# ------------------------------------------------------------------ supervised launch
def _tree_stats(pp):
    rss, thr, cpu = 0, 0, 0.0
    for p in _tree(pp):
        try:
            rss += p.memory_info().rss
            thr += p.num_threads()
            c = p.cpu_times()
            cpu += c.user + c.system + c.children_user + c.children_system
        except Exception:
            pass
    return rss / 2**30, thr, cpu


def run_job(job, bucket, devices, argv, max_wall_s, grace_s=60, cores=2, env=None, cwd=None, log_path=None,
            kind="planned", images_primary=0, images_technical=0, note=None, declared_rss_gib=8.0,
            declared_storage_gib=0.5):
    """Reserve -> launch (affinity-pinned) -> monitor/enforce -> terminate/reap -> charge. Blocking."""
    if [1 for v in read()["open"].values() if not _supervisor_alive(v)]:
        reconcile("pre_launch_check")
    rid = reserve(job, bucket, devices, max_wall_s, grace_s, cores, kind=kind, images_primary=images_primary,
                  images_technical=images_technical, declared_rss_gib=declared_rss_gib,
                  declared_storage_gib=declared_storage_gib, note=note)
    foreign = _gpu_foreign(devices)
    if foreign:
        release_unlaunched(rid, f"foreign GPU users {foreign}")
        raise CeilingBreach(f"requested devices {devices} in use by foreign processes {foreign}")
    with _Locked() as L:
        slice_ = list(L.state["open"][rid]["cores"])
    nthr = str(len(slice_))
    e = dict(os.environ)
    e.update({"OMP_NUM_THREADS": nthr, "MKL_NUM_THREADS": nthr, "OPENBLAS_NUM_THREADS": nthr, "NUMEXPR_NUM_THREADS": nthr,
              "CPRX_THREADS": nthr})
    if env:
        e.update(env)
    e["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices) if devices else ""
    e["CPRX_JOB_ID"], e["CPRX_RID"] = job, rid
    log_path = log_path or os.path.join(ROOT, "logs", f"{job}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "ab")
    t_launch = now()
    status, detail, rc = "unknown", None, None
    cpu_s, peak_rss, peak_thr, peak_eff = 0.0, 0.0, 0, 0.0
    proc = None
    end_epoch, hard = None, False
    s0 = read()
    carried_cpu = s0["charged_cpu_s"]
    other_live_cpu = 0.0

    def pin():
        os.sched_setaffinity(0, set(slice_))

    try:
        proc = subprocess.Popen(argv, env=e, cwd=cwd or ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=pin)
        pp = psutil.Process(proc.pid)
        with _Locked() as L:
            v = L.state["open"][rid]
            v["pid"], v["pid_ctime"], v["launch_epoch"] = proc.pid, pp.create_time(), t_launch
        with _LIVE_LOCK:
            _LIVE[rid] = {"pid": proc.pid, "devices": list(devices), "declared_rss_gib": declared_rss_gib, "t_launch": t_launch}
        journal("launch", rid=rid, job=job, pid=proc.pid, argv=argv, devices=list(devices), cores=slice_, log=log_path)
        deadline = t_launch + float(max_wall_s)
        last_du, last_cpu_t, last_cpu, last_heavy = 0.0, t_launch, 0.0, 0.0
        sto_gib = 0.0
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
                end_epoch, surv, hard = _reap_leftovers(pp, grace_s)
                if surv:
                    detail = {"survivors_after_exit": surv}
                break
            if now() - last_heavy < POLL_S:
                time.sleep(0.25)
                continue
            last_heavy = now()
            rss, thr, cpu = _tree_stats(pp)
            cpu_s = max(cpu_s, cpu)
            peak_rss, peak_thr = max(peak_rss, rss), max(peak_thr, thr)
            t = now()
            if t - last_cpu_t >= 20:
                eff = max(0.0, (cpu - last_cpu) / (t - last_cpu_t))
                peak_eff = max(peak_eff, eff)
                last_cpu_t, last_cpu = t, cpu
            agg = _live_rss_gib(young_declared=False)
            reason = None
            tot_cpu = read()["charged_cpu_s"]             # closed jobs only; live trees added below
            live_cpu_all = cpu_s + _other_live_cpu(rid)
            if agg > RSS_KILL_GIB:
                reason = f"aggregate_rss_{agg:.1f}GiB>{RSS_KILL_GIB}"
            elif t > deadline:
                reason = f"timeout_{max_wall_s}s"
            elif shutil.disk_usage("/data").free / 2**30 < CAPS["min_free_gib"]:
                reason = "free_space_below_floor"
            elif tot_cpu + live_cpu_all > CPU_KILL_FRACTION * CAPS["cpu_core_hours"] * 3600:
                reason = "cpu_core_hours_ceiling"
            elif bucket == "eng" and t > ENGINEERING_END_EPOCH + 60:
                reason = "engineering_window_end"
            elif bucket != "reserve" and kind == "planned" and t > SCIENCE_STOP_EPOCH:
                reason = "science_stop"
            elif t > HARD_DEADLINE_EPOCH:
                reason = "hard_deadline"
            if t - last_du > DU_EVERY_S:
                last_du = t
                sto_gib = _du_bytes() / 2**30
                if sto_gib > STORAGE_KILL_FRACTION * CAPS["storage_gib"] and reason is None:
                    reason = f"task_storage_{sto_gib:.1f}GiB"
                _note_peaks(sto_gib, peak_rss, agg, peak_thr, peak_eff)
                journal("sample", path=MONITOR, rid=rid, job=job, rss_gib=round(rss, 2), agg_rss_gib=round(agg, 2),
                        threads=thr, cpu_s=round(cpu_s, 1), eff_cores_20s=round(peak_eff, 2), storage_gib=round(sto_gib, 2),
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
        return {"rid": rid, "status": status, "returncode": rc, "peak_rss_gib": peak_rss, "cpu_s": cpu_s,
                "peak_effective_cores": peak_eff, "held_s": (end_epoch or now()) - t_launch, "log": log_path, "hard_kill": hard}
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


def _other_live_cpu(rid):
    """CPU seconds consumed so far by OTHER live supervised jobs in this supervisor process."""
    tot = 0.0
    with _LIVE_LOCK:
        items = [(k, v["pid"]) for k, v in _LIVE.items() if k != rid]
    for k, pid in items:
        try:
            tot += _tree_stats(psutil.Process(pid))[2]
        except Exception:
            pass
    return tot


def _reap_leftovers(root_proc, grace_s):
    left = []
    try:
        left = [p for p in psutil.process_iter(["pid", "ppid"]) if p.info["ppid"] == 1 and _same_session(p, root_proc)]
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


def _note_peaks(storage_gib, rss_gib, agg, thr, eff):
    with _Locked() as L:
        pk = L.state["peak"]
        pk["storage_gib"] = max(pk["storage_gib"], storage_gib)
        pk["rss_gib"] = max(pk["rss_gib"], rss_gib)
        pk["agg_rss_gib"] = max(pk["agg_rss_gib"], agg)
        pk["threads"] = max(pk["threads"], thr)
        pk["effective_cores"] = max(pk["effective_cores"], eff)
        fr = shutil.disk_usage("/data").free / 2**30
        pk["min_free_gib"] = fr if pk["min_free_gib"] is None else min(pk["min_free_gib"], fr)


def coverage_and_gaps():
    return {
        "enforced_and_metered": [
            "GPU: wall-clock x allocated devices from launch to CONFIRMED process-tree termination, failures charged, no clipping",
            "CPU: process-tree cpu_times (incl. reaped children) and wait4 rusage of the direct child, for EVERY supervised "
            "job including setup, diagnostics, scoring and verification run through run_job; charged on top of the carried subtotal",
            "CPU concurrency: disjoint sched_setaffinity slices of an 8-core pool (hard bound on simultaneously runnable cores) "
            "plus OMP/MKL/OPENBLAS/NUMEXPR thread env = slice size; effective cores sampled from cpu_times deltas (20 s windows)",
            "RSS: process-tree sum every 5 s, aggregate over live jobs, kill >120 GiB (cap 128); launch refused if projected "
            "aggregate RSS or host-available RAM is insufficient (young jobs counted at declared peak)",
            "storage: `du -sb` of the task root + the repo study dir every 30 s while jobs run, kill >95% of 200 GiB; launches "
            "refused when current + declared storage would exceed the cap; /data free space polled every 5 s (kill <50 GiB)",
            "images: reserved before launch; actual decodes from per-job append-only decode counters",
        ],
        "gaps": [
            "Unsupervised commands issued interactively by the operator (git, small file edits, report writing, curl) are not "
            "metered; they are negligible next to the supervised jobs but are NOT counted. All python/GPU/diagnostic jobs "
            "are launched through the supervisor.",
            "Storage is sampled every 30 s; transient spikes between samples are not captured. Files written outside the task "
            "root and repo study dir are not attributed (task jobs set TMPDIR/XDG/TRITON/INDUCTOR caches inside the task root).",
            "Shared pristine model and detector caches under hf_home are read-only inputs and are not counted as task storage; "
            "any cache additions made by task jobs (offline mode) would have to land under the task root to be counted.",
            "Historical (pre-recovery) CPU, peak storage and eight-worker enforcement were incomplete; the opening CPU balance is a "
            "lower bound for that scope.",
            "Affinity bounds runnable cores, but sampled effective-core use is advisory and cuda driver / kernel threads share the slice.",
            "GPU memory (40 GB/device) is not monitored by the supervisor; OOM surfaces as a child failure and is charged.",
            "RSS counts file-backed mapped pages (safetensors mmap): conservative for the cap.",
            "Other users' processes are invisible except through the pre-launch foreign-process / resident-memory probe.",
            "If the supervisor is killed uncleanly, children with unobserved end times are charged at their full reservation.",
        ],
    }
