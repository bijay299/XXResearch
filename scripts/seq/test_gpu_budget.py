#!/usr/bin/env python3
"""CPU tests for scripts/seq/gpu_budget.py.

Two things are pinned down here, and they must not be confused.

1. THE CURRENT POLICY: RECORDING ONLY. The PI withdrew the four-GPU-hour
   ceiling, the frozen-test reserve and every budget-derived cutoff on
   2026-10-10. So: no admission is ever refused, no `max_wall_seconds` is ever
   handed back, a report never fails on a ceiling -- and usage, including failed
   and orphaned stages, is still recorded, prior spend is still carried forward,
   and `retire` preserves every historical entry exactly.

2. THE RETIRED CEILING MODE, kept deliberately. The defect it was built for was
   real and was verified: a 4.0 GPU-h ceiling with a 1.008 reserve ADMITTED TWO
   2.5-HOUR REQUESTS, because the old `check` summed only completed entries and
   nothing held an in-flight amount against the balance. Deleting that code
   along with the cap would delete the evidence that it was fixed, so the mode
   and its tests stay.

Covers: recording-only admission, retirement and record preservation, the
absence of any derived cutoff, concurrent admission, underestimated runtime,
interruption/crash reconciliation, reserve protection, and carry-forward.

No GPU. Fakes are sleeps and subprocesses.

    python scripts/seq/test_gpu_budget.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / "scripts" / "seq" / "gpu_budget.py"
PASS, FAIL = [], []


def check(desc: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(desc)
    print(f"  {'ok  ' if cond else 'FAIL'}  {desc}" + ("" if cond else f"  <- {detail}"))


def run(*args: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(BUDGET), *args],
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return p.returncode, p.stdout + p.stderr


def init(led: Path, ceiling=4.0, reserve=1.008, enforcement="ceiling",
         **kw) -> None:
    """Historical CEILING-mode ledger unless told otherwise."""
    args = ["init", "--ledger", str(led), "--enforcement", enforcement,
            "--ceiling", str(ceiling), "--reserve", str(reserve)]
    for k, v in kw.items():
        args += [f"--{k}", str(v)]
    rc, out = run(*args)
    assert rc == 0, out


def admit(led: Path, stage: str, need: float, gpus=1, pid=0,
          reserve=False) -> tuple[int, dict]:
    args = ["admit", "--ledger", str(led), "--stage", stage,
            "--need", str(need), "--gpus", str(gpus), "--pid", str(pid)]
    if reserve:
        args.append("--draws_on_reserve")
    rc, out = run(*args)
    j = None
    for line in out.splitlines():
        if line.startswith("{"):
            j = json.loads(line)
    return rc, (j or {})


def ledger(led: Path) -> dict:
    return json.loads(led.read_text())


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gpubudget_"))
    try:
        print("\nCURRENT POLICY: recording only -- nothing is refused, "
              "everything is recorded")
        led = tmp / "recording.json"
        rc, out = run("init", "--ledger", str(led))
        check("a new ledger defaults to recording only",
              rc == 0 and "RECORDING ONLY" in out, out.strip()[-160:])
        check("and says so on disk",
              ledger(led)["enforcement"] == "recording_only")
        # The exact request pair the old ceiling refused.
        rc1, d1 = admit(led, "big-1", 2.5, pid=os.getpid())
        rc2, d2 = admit(led, "big-2", 2.5, pid=os.getpid())
        check("two 2.5 GPU-h requests are both admitted now",
              rc1 == 0 and rc2 == 0
              and d1.get("decision") == d2.get("decision") == "RECORD",
              f"{d1.get('decision')} / {d2.get('decision')}")
        check("neither is given a runtime cutoff derived from its estimate",
              d1.get("max_wall_seconds") is None
              and d2.get("max_wall_seconds") is None,
              f"{d1.get('max_wall_seconds')} / {d2.get('max_wall_seconds')}")
        check("no balance is reported, rather than a negative one",
              d2.get("available_to_this_stage") is None
              and d2.get("ceiling") is None, str(d2))
        check("the estimate is marked informational on the open reservation",
              all(o.get("estimate_is_informational") for o in ledger(led)["open"]))
        # ... and the usage is still recorded, failures included.
        rc, out = run("settle", "--ledger", str(led), "--id", d1["reservation_id"],
                      "--wall_seconds", "9000", "--exit_code", "1")
        e = ledger(led)["entries"][-1]
        check("a FAILED stage is recorded at its actual occupancy",
              rc == 0 and abs(e["gpu_hours"] - 2.5) < 1e-9
              and "FAILED" in e["outcome"], str(e.get("gpu_hours")))
        check("exceeding the estimate is recorded, not an error",
              rc == 0 and e["estimate_was_informational"] is True)
        rc, out = run("report", "--ledger", str(led))
        check("the report exits 0 with no ceiling to breach", rc == 0)
        check("the report states the policy", "RECORDING ONLY" in out,
              out.strip()[:160])
        check("the report withholds nothing for the frozen test",
              "reserve withheld: none" in out, out.strip()[-200:])

        print("\nRETIRING an existing ceiling ledger preserves every record")
        led = tmp / "retire.json"
        init(led, ceiling=4.0, reserve=1.3474)
        rc, d = admit(led, "past", 1.2022, pid=os.getpid())
        run("settle", "--ledger", str(led), "--id", d["reservation_id"],
            "--wall_seconds", str(1.2022 * 3600), "--exit_code", "0")
        before = ledger(led)
        rc, out = run("retire", "--ledger", str(led),
                      "--reason", "PI withdrew the cap", "--authority", "PI")
        after = ledger(led)
        check("retire succeeds and says what it withdrew",
              rc == 0 and "RETIRED" in out, out.strip()[-160:])
        check("every entry survives byte for byte",
              after["entries"] == before["entries"])
        check("the recorded spend is unchanged",
              abs(after["spent_gpu_hours"] - before["spent_gpu_hours"]) < 1e-12)
        r = after["retirement"]
        check("the retired ceiling is stamped for the record",
              r["retired_ceiling_gpu_hours"] == 4.0
              and r["retired_reserve_gpu_hours"] == 1.3474, str(r))
        check("so is the state of the record at the switch",
              r["entries_at_retirement"] == 1
              and abs(r["spent_gpu_hours_at_retirement"] - 1.2022) < 1e-4, str(r))
        rc, d = admit(led, "after-retirement", 99.0, pid=os.getpid())
        check("a request the old ceiling would refuse is now admitted",
              rc == 0 and d["decision"] == "RECORD", str(d))
        check("prior spend is still carried in the record",
              abs(d["spent"] - 1.2022) < 1e-4, str(d.get("spent")))
        rc, out = run("retire", "--ledger", str(led), "--reason", "again")
        check("retiring twice is a no-op", rc == 0 and "already retired" in out,
              out.strip()[-120:])

        print("\nRETIRED CEILING MODE (kept for the record): the reproduced "
              "defect stays fixed")
        led = tmp / "race.json"
        init(led, ceiling=4.0, reserve=1.008)
        # The exact reproduction: two 2.5-hour requests against 4.0 with 1.008
        # withheld, i.e. 2.992 usable. The first must be admitted, the second
        # REFUSED while the first is still in flight.
        rc1, d1 = admit(led, "big-1", 2.5, pid=os.getpid())
        rc2, d2 = admit(led, "big-2", 2.5, pid=os.getpid())
        check("the first 2.5 GPU-h request is admitted",
              rc1 == 0 and d1.get("decision") == "ADMIT", str(d1))
        check("the second is REFUSED while the first is in flight",
              rc2 == 1 and d2.get("decision") == "REFUSE", str(d2))
        check("the refusal sees the in-flight amount",
              abs(d2.get("in_flight", 0) - 2.5) < 1e-9, str(d2.get("in_flight")))
        check("5.0 GPU-h was never admitted against a 4.0 ceiling",
              ledger(led)["committed_gpu_hours"] <= 4.0 + 1e-9,
              str(ledger(led)["committed_gpu_hours"]))
        check("the open reservation is durable on disk",
              len(ledger(led)["open"]) == 1)

        print("\nconcurrent admission is serialised by the lock")
        led = tmp / "concurrent.json"
        init(led, ceiling=1.0, reserve=0.0)
        # Ten threads each asking for 0.2 of a 1.0 ceiling: exactly five may win.
        with ThreadPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(lambda i: admit(led, f"c{i}", 0.2, pid=os.getpid()),
                              range(10)))
        admitted = [r for r in res if r[0] == 0]
        check("exactly five of ten concurrent 0.2 requests are admitted",
              len(admitted) == 5, f"{len(admitted)} admitted")
        check("committed never exceeds the ceiling",
              ledger(led)["committed_gpu_hours"] <= 1.0 + 1e-9,
              str(ledger(led)["committed_gpu_hours"]))
        check("every admission has a distinct reservation id",
              len({r[1]["reservation_id"] for r in admitted}) == len(admitted))

        print("\nin the retired mode the admission implied a runtime bound")
        led = tmp / "bound.json"
        init(led, ceiling=4.0, reserve=0.0)
        rc, d = admit(led, "one-gpu", 0.04, gpus=1, pid=os.getpid())
        check("a 0.04 GPU-h single-GPU admission bounds the wall clock at 144s",
              abs(d["max_wall_seconds"] - 144.0) < 0.5, str(d.get("max_wall_seconds")))
        rc, d2 = admit(led, "two-gpu", 0.04, gpus=2, pid=os.getpid())
        check("the same amount on two GPUs bounds it at 72s (device-hours)",
              abs(d2["max_wall_seconds"] - 72.0) < 0.5, str(d2.get("max_wall_seconds")))
        # The bound actually stops an over-running child.
        t0 = time.time()
        p = subprocess.run(["timeout", "--kill-after=2", "1", sys.executable, "-c",
                            "import time; time.sleep(30)"])
        el = time.time() - t0
        check("a runtime bound terminates an over-running child",
              p.returncode in (124, 137) and el < 5, f"rc={p.returncode} {el:.1f}s")
        # and it kills the whole process group, not just the direct child
        script = ("import subprocess,sys,time;"
                  "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                  "time.sleep(30)")
        t0 = time.time()
        p = subprocess.run(["timeout", "--kill-after=2", "1", sys.executable,
                            "-c", script], capture_output=True)
        check("the bound covers grandchildren (own process group)",
              p.returncode in (124, 137) and time.time() - t0 < 6,
              f"rc={p.returncode}")

        print("\nan UNDERESTIMATED stage is charged what it actually used")
        led = tmp / "under.json"
        init(led, ceiling=4.0, reserve=0.0)
        rc, d = admit(led, "under", 0.01, gpus=1, pid=os.getpid())
        rc, out = run("settle", "--ledger", str(led), "--id", d["reservation_id"],
                      "--wall_seconds", "720", "--exit_code", "0")
        e = ledger(led)["entries"][-1]
        check("the actual 0.2 GPU-h is charged, not the 0.01 reservation",
              abs(e["gpu_hours"] - 0.2) < 1e-9, str(e["gpu_hours"]))
        check("the overrun is flagged", e["exceeded_reservation"] is True)
        check("the reservation it broke is recorded beside it",
              abs(e["reserved_gpu_hours"] - 0.01) < 1e-9)
        check("the reservation is no longer in flight", ledger(led)["open"] == [])

        print("\na TIMED-OUT stage is charged and labelled")
        rc, d = admit(led, "timedout", 0.02, gpus=1, pid=os.getpid())
        run("settle", "--ledger", str(led), "--id", d["reservation_id"],
            "--wall_seconds", "72", "--exit_code", "124")
        e = ledger(led)["entries"][-1]
        check("a timeout is charged like any other occupancy",
              abs(e["gpu_hours"] - 0.02) < 1e-9)
        check("and is labelled as a runtime-bound hit",
              "TIMED OUT" in e["outcome"], e["outcome"])

        print("\nINTERRUPTION and crash: reconciled conservatively, never dropped")
        led = tmp / "crash.json"
        init(led, ceiling=4.0, reserve=0.0)
        # A reservation held by a process that no longer exists.
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        rc, d = admit(led, "crashed", 0.5, gpus=1, pid=dead.pid)
        check("the crashed stage's reservation is on the books",
              abs(ledger(led)["in_flight_gpu_hours"] - 0.5) < 1e-9)
        rc, out = run("reconcile", "--ledger", str(led))
        lg = ledger(led)
        e = next(x for x in lg["entries"] if x["stage"] == "crashed")
        check("it is settled rather than left open", lg["open"] == [])
        check("it is charged the FULL reservation, conservatively",
              abs(e["gpu_hours"] - 0.5) < 1e-9, str(e["gpu_hours"]))
        check("the charge basis is recorded", "unknowable" in e["charge_basis"])
        check("it is marked orphaned", e["reconciled"] is True
              and "ORPHANED" in e["outcome"])
        # A LIVE reservation must not be reconciled away.
        live = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
        try:
            rc, d = admit(led, "live", 0.1, gpus=1, pid=live.pid)
            rc, out = run("reconcile", "--ledger", str(led), "--strict")
            check("a live reservation is left alone", len(ledger(led)["open"]) == 1
                  and rc == 1, out.strip()[-120:])
        finally:
            live.kill(); live.wait()
        # admit() reconciles first, so a crashed predecessor is charged before
        # the next decision is taken.
        rc, out = run("reconcile", "--ledger", str(led))
        before = ledger(led)["spent_gpu_hours"]
        dead2 = subprocess.Popen([sys.executable, "-c", "pass"]); dead2.wait()
        admit(led, "crashed2", 0.3, pid=dead2.pid)
        rc, d = admit(led, "next", 0.1, pid=os.getpid())
        check("admission reconciles orphans before deciding",
              d.get("spent", 0) >= before + 0.3 - 1e-9,
              f"spent seen at admission: {d.get('spent')}")

        print("\nRESERVE protection")
        led = tmp / "reserve.json"
        init(led, ceiling=4.0, reserve=1.3474)
        rc, d = admit(led, "dev", 4.0 - 1.3474, pid=os.getpid())
        check("development work may use everything outside the reserve",
              rc == 0, str(d))
        rc, d = admit(led, "dev-more", 0.01, pid=os.getpid())
        check("but not one GPU-minute of the reserve itself",
              rc == 1 and d["decision"] == "REFUSE", str(d))
        rc, d = admit(led, "frozen_test", 1.3474, reserve=True, pid=os.getpid())
        check("the reserved stage CAN spend the reserve",
              rc == 0 and d["decision"] == "ADMIT", str(d))
        check("and that exhausts the ceiling exactly",
              abs(ledger(led)["committed_gpu_hours"] - 4.0) < 1e-6,
              str(ledger(led)["committed_gpu_hours"]))

        print("\nPRIOR SPEND is carried forward, never reset")
        old = tmp / "old.json"
        init(old, ceiling=4.0, reserve=1.008)
        rc, d = admit(old, "past", 1.2022, pid=os.getpid())
        run("settle", "--ledger", str(old), "--id", d["reservation_id"],
            "--wall_seconds", str(1.2022 * 3600), "--exit_code", "0")
        new = tmp / "new.json"
        rc, out = run("init", "--ledger", str(new), "--enforcement", "ceiling",
                      "--ceiling", "4.0",
                      "--reserve", "1.3474", "--carry_forward_from", str(old))
        check("a new ledger carries prior spend forward", rc == 0, out.strip()[-160:])
        lg = ledger(new)
        check("the carried spend is exactly the prior total",
              abs(lg["entries"][0]["gpu_hours"] - 1.2022) < 1e-6,
              str(lg["entries"][0]["gpu_hours"]))
        check("the carried entry records where it came from",
              "carried_forward_from" in lg["entries"][0])
        rc, d = admit(new, "would-overspend", 2.0, pid=os.getpid())
        check("carried spend constrains later admissions",
              rc == 1, f"available was {d.get('available_to_this_stage')}")
        # init must refuse to silently reset an existing ledger
        rc, out = run("init", "--ledger", str(new), "--enforcement", "ceiling",
                      "--ceiling", "4.0")
        check("init REFUSES to reset an existing ledger",
              rc == 1 and "refusing to reset" in out, out.strip()[-120:])
        # carrying forward from a ledger with open reservations must refuse
        rc, d = admit(old, "still-open", 0.1, pid=os.getpid())
        rc, out = run("init", "--ledger", str(tmp / "n2.json"),
                      "--enforcement", "ceiling", "--ceiling", "4.0",
                      "--carry_forward_from", str(old))
        check("carry-forward refuses while the source has open reservations",
              rc == 1 and "OPEN reservation" in out, out.strip()[-140:])

        print("\nCARRY-FORWARD under the current policy, too")
        old2 = tmp / "old_rec.json"
        rc, out = run("init", "--ledger", str(old2))
        rc, d = admit(old2, "past", 1.2022, pid=os.getpid())
        run("settle", "--ledger", str(old2), "--id", d["reservation_id"],
            "--wall_seconds", str(1.2022 * 3600), "--exit_code", "0")
        new2 = tmp / "new_rec.json"
        rc, out = run("init", "--ledger", str(new2),
                      "--carry_forward_from", str(old2))
        lg = ledger(new2)
        carried_total = sum(e.get("gpu_hours", 0.0) for e in lg["entries"])
        check("a recording-only ledger also carries prior spend forward",
              rc == 0 and abs(carried_total - 1.2022) < 1e-6, str(carried_total))
        check("and records where every carried hour came from",
              all("carried_forward_from" in e for e in lg["entries"]))
        check("and still refuses to be reset",
              run("init", "--ledger", str(new2))[0] == 1)

        print("\nreport surfaces in-flight and conservative charges")
        rc, out = run("report", "--ledger", str(tmp / "crash.json"))
        check("the report names orphaned charges",
              "orphaned and conservatively charged" in out, out.strip()[-200:])
        check("the report shows committed = spent + in flight",
              "committed" in out and "in flight" in out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL BUDGET CHECKS PASSED (CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
