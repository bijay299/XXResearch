#!/usr/bin/env python3
"""Supervisor-only mechanical smoke check. Uses a SCRATCH ledger (CPRD2_LEDGER_DIR) and tiny
caps so it neither touches the real ledger nor any GPU compute. 'Allocated devices' here are
logical reservations for sleeping children; no CUDA context is created."""
import json, os, signal, subprocess, sys, tempfile, time, textwrap
scratch = os.environ.setdefault("CPRD2_LEDGER_DIR", tempfile.mkdtemp(prefix="smoke_ledger_", dir="/data/bijaypandey/claude_tmp"))
caps = os.path.join(scratch, "caps.json")
os.makedirs(scratch, exist_ok=True)
json.dump({"gpu_device_hours": 0.2, "gpu_entry_technical_hours": 0.1,
           "gpu_protected_eval_publication_hours": 0.05, "images_total": 10, "images_primary": 6,
           "images_technical": 4, "aggregate_workers": 4}, open(caps, "w"))
os.environ["CPRD2_CAPS_JSON"] = caps
sys.path.insert(0, os.path.join(os.environ["CPRD2_ROOT"], "code"))
from cprd2 import ledger as LG
LG.POLL_S = 0.5
R = []
def chk(name, ok, **kw):
    R.append({"check": name, "pass": bool(ok), **kw}); print("PASS" if ok else "FAIL", name, kw, flush=True)

# 1. normal success is charged actual wall x devices
r = LG.run_job("smoke_ok", "entry", [0], [sys.executable, "-c", "import time; time.sleep(2)"], 20, grace_s=2, workers=2)
s = LG.read()
chk("success_charged_actual", r["status"] == "ok" and 1.5 < s["charged_gpu_s"]["entry"] < 6, charged=s["charged_gpu_s"]["entry"], held=r["held_s"])

# 2. failure (exit 3) is charged too
before = LG.read()["charged_gpu_s"]["entry"]
r = LG.run_job("smoke_fail", "entry", [0, 1], [sys.executable, "-c", "import time,sys; time.sleep(1); sys.exit(3)"], 20, grace_s=2, workers=2)
after = LG.read()["charged_gpu_s"]["entry"]
chk("failure_charged_x2_devices", r["status"] == "exit_3" and after - before >= 1.8, delta=after - before)

# 3. timeout: TERM-ignoring child with a setsid'd grandchild -> whole tree killed and confirmed
code = textwrap.dedent("""
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
gc = subprocess.Popen([sys.executable, '-c', 'import os,time; os.setsid(); open(os.environ["GCPID"],"w").write(str(os.getpid())); time.sleep(300)'], env=dict(os.environ))
time.sleep(300)
""")
gcp = os.path.join(scratch, "gc.pid")
t0 = time.time()
r = LG.run_job("smoke_timeout", "entry", [2], [sys.executable, "-c", code], 3, grace_s=2, workers=2, env={"GCPID": gcp})
gpid = int(open(gcp).read())
import psutil
alive = psutil.pid_exists(gpid) and psutil.Process(gpid).status() != psutil.STATUS_ZOMBIE
chk("timeout_kills_tree_and_charges", r["status"].startswith("killed_timeout") and not alive and r["held_s"] >= 3,
    held=r["held_s"], hard_kill=r["hard_kill"], grandchild_alive=alive)

# 4. charges are never clipped: settle with held >> reservation
rid = LG.reserve("smoke_overrun", "entry", [0], 1, 1, 2)
with LG._Locked() as L:
    L.state["open"][rid]["launch_epoch"] = time.time() - 30
    L.state["open"][rid]["pid"] = None
b = LG.read()["charged_gpu_s"]["entry"]
c = LG.settle(rid, time.time(), 0.0, "synthetic_overrun")
chk("no_clipping_overrun_charged_in_full", c >= 29.9 and any(o["job"] == "smoke_overrun" for o in LG.read()["overruns"]), charged=c, reserved=2.0)

# 5. cap refusal at reservation time (entry subcap 0.1 h = 360 s; request 4x110 s)
try:
    LG.reserve("smoke_cap", "entry", [0, 1, 2, 3], 100, 10, 2)
    chk("cap_refusal", False)
except LG.CeilingBreach as e:
    chk("cap_refusal", True, message=str(e)[:80])

# 6. image cap refusal
try:
    LG.reserve("smoke_img", "reserve", [], 10, 1, 2, images_primary=7)
    chk("image_cap_refusal", False)
except LG.CeilingBreach as e:
    chk("image_cap_refusal", True, message=str(e)[:80])

# 7. supervisor SIGKILL -> survivor child -> reconcile before next launch
sup = textwrap.dedent("""
import os, sys
sys.path.insert(0, os.path.join(os.environ['CPRD2_ROOT'], 'code'))
from cprd2 import ledger as LG
LG.run_job('smoke_orphan', 'reserve', [3], [sys.executable, '-c', 'import time; time.sleep(120)'], 200, grace_s=2, workers=2)
""")
p = subprocess.Popen([sys.executable, "-c", sup], env=dict(os.environ))
for _ in range(60):
    st = LG.read()["open"]
    if any(v["job"] == "smoke_orphan" and v["pid"] for v in st.values()):
        break
    time.sleep(0.5)
child_pid = [v["pid"] for v in LG.read()["open"].values() if v["job"] == "smoke_orphan"][0]
os.kill(p.pid, signal.SIGKILL); p.wait()
time.sleep(1)
survived = psutil.pid_exists(child_pid)
try:
    LG.reserve("smoke_blocked", "reserve", [], 5, 1, 2)
    blocked = False
except LG.CeilingBreach:
    blocked = True
closed = LG.reconcile("smoke_supervisor_killed", grace_s=2)
dead = (not psutil.pid_exists(child_pid)) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
chk("supervisor_failure_reconcile", survived and blocked and len(closed) == 1 and dead and not LG.read()["open"],
    child_survived_supervisor=survived, launches_blocked_until_reconcile=blocked, closed=len(closed), child_dead_after=dead)

# 8. RSS kill
rr = LG.run_job("smoke_rss", "reserve", [], [sys.executable, "-c", "x=bytearray(600*2**20); import time; time.sleep(60)"], 30, grace_s=2, workers=2, rss_limit_gib=0.3)
chk("rss_limit_kills", rr["status"].startswith("killed_job") or rr["status"].startswith("killed_aggregate"), status=rr["status"], peak_rss_gib=rr["peak_rss_gib"])

# 9. worker cap
rid1 = LG.reserve("w1", "reserve", [], 5, 1, 2); rid2 = LG.reserve("w2", "reserve", [], 5, 1, 2)
try:
    LG.reserve("w3", "reserve", [], 5, 1, 2); ok = False
except LG.CeilingBreach:
    ok = True
LG.release_unlaunched(rid1, "smoke"); LG.release_unlaunched(rid2, "smoke")
chk("aggregate_worker_cap", ok)

# 10. repair cycles capped at two
LG.open_repair_cycle("smoke1", "x", 10); LG.open_repair_cycle("smoke2", "y", 10)
try:
    LG.open_repair_cycle("smoke3", "z", 10); third = True
except LG.CeilingBreach:
    third = False
chk("no_third_repair_cycle", not third)

out = {"ledger_dir": scratch, "checks": R, "all_pass": all(c["pass"] for c in R),
       "totals": LG.totals(), "coverage_and_gaps": LG.coverage_and_gaps()}
json.dump(out, open(os.path.join(os.environ["CPRD2_ROOT"], "artifacts", "smoke_supervisor.json"), "w"), indent=1, default=str)
print("ALL_PASS" if out["all_pass"] else "SOME_FAIL")
sys.exit(0 if out["all_pass"] else 1)
