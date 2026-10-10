#!/usr/bin/env python3
"""A hard GPU-hour ledger with ATOMIC DURABLE ADMISSION.

The approved ceiling is FOUR GPU-hours in total and it counts everything:
successful stages, failed attempts, and setup.

What was wrong with the first version, and why it mattered
----------------------------------------------------------
It exposed `check` (read the ledger, decide) and `record` (append the cost after
the stage returned). Nothing existed in between. So:

  * `check` summed only COMPLETED entries. A stage that had been admitted but
    had not yet finished was invisible to the balance.
  * The two trajectories, and later the two evaluation lanes, were admitted
    CONCURRENTLY. Both saw the same balance and both were allowed.
  * There was no runtime bound, so an under-estimated stage could run as long
    as it liked and the overrun was discovered only when it returned.
  * A crash or interruption between admission and recording left the cost
    unaccounted entirely.

Reproduced on CPU: a 4.0 ceiling with a 1.008 reserve admitted two 2.5-hour
requests, and the breach surfaced only after 5.0 hours had been recorded.

The fix
-------
`admit` is the only way to start work. Under an exclusive lock it writes a
DURABLE OPEN RESERVATION holding the full estimate against the balance, fsyncs
it, and returns a reservation id plus the wall-clock bound that estimate implies.
A concurrent `admit` therefore sees the reservation and is refused. `settle`
converts a reservation into a completed entry charging the ACTUAL occupancy, and
flags an overrun if the actual exceeded the reservation.

Balance arithmetic, with in-flight work included:

    committed = sum(completed.gpu_hours) + sum(open.reserved_gpu_hours)
    reserve_remaining = max(0, reserve - reserve-drawing committed)
    available(stage) = ceiling - committed
                       - (0 if stage draws on the reserve else reserve_remaining)

`reconcile` settles orphaned reservations CONSERVATIVELY: a reservation whose
process is gone is charged max(reserved, elapsed x gpus), because what it really
used is unknowable and the safe assumption is the larger figure. It is never
silently dropped.

GPU-hours are DEVICE-hours: two GPUs for 30 minutes is 1.0 GPU-hour, the unit
the approved cost model uses.

The ledger is the EXPERIMENT's, not an output directory's. Prior spend is
carried forward and never reset because a new output directory was created.

CPU only. This file touches no GPU; it only accounts for them.

    python scripts/seq/gpu_budget.py init    --ledger L --ceiling 4.0 --reserve 1.3474
    python scripts/seq/gpu_budget.py admit   --ledger L --stage s --need 0.04 --gpus 1 --pid $$
    python scripts/seq/gpu_budget.py settle  --ledger L --id <id> --wall_seconds 108 --exit_code 0
    python scripts/seq/gpu_budget.py reconcile --ledger L
    python scripts/seq/gpu_budget.py report  --ledger L

Exit codes: 0 ok/admitted, 1 refused or over ceiling or error, 2 usage.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import sys
import uuid
from pathlib import Path

H = 3600.0


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


class Ledger:
    """Exclusive lock held for the whole block; every mutation is fsynced."""

    def __init__(self, path: Path, create: bool = False):
        self.path, self.create, self.fh, self.data = path, create, None, {}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.fh = self.path.open("r+")
        elif self.create:
            self.fh = self.path.open("w+")
        else:
            raise FileNotFoundError(f"ledger {self.path} does not exist; run `init`")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        raw = self.fh.read()
        self.data = json.loads(raw) if raw.strip() else {}
        self.data.setdefault("entries", [])
        self.data.setdefault("open", [])
        return self

    def write(self) -> None:
        self.data["spent_gpu_hours"] = round(self.spent(), 6)
        self.data["in_flight_gpu_hours"] = round(self.open_total(), 6)
        self.data["committed_gpu_hours"] = round(self.committed(), 6)
        self.data["remaining_gpu_hours"] = round(self.ceiling() - self.committed(), 6)
        self.data["updated_utc"] = _now()
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(json.dumps(self.data, indent=2) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()

    # ---- arithmetic
    def ceiling(self) -> float:
        return float(self.data.get("ceiling_gpu_hours", 0.0))

    def reserve(self) -> float:
        return float(self.data.get("reserve_gpu_hours", 0.0))

    def spent(self) -> float:
        return sum(e.get("gpu_hours", 0.0) for e in self.data["entries"])

    def open_total(self) -> float:
        return sum(o.get("reserved_gpu_hours", 0.0) for o in self.data["open"])

    def committed(self) -> float:
        return self.spent() + self.open_total()

    def reserve_remaining(self) -> float:
        used = sum(e.get("gpu_hours", 0.0) for e in self.data["entries"]
                   if e.get("draws_on_reserve"))
        used += sum(o.get("reserved_gpu_hours", 0.0) for o in self.data["open"]
                    if o.get("draws_on_reserve"))
        return max(0.0, self.reserve() - used)

    def available(self, draws_on_reserve: bool) -> float:
        a = self.ceiling() - self.committed()
        if not draws_on_reserve:
            a -= self.reserve_remaining()
        return a


def cmd_init(a) -> int:
    p = Path(a.ledger)
    if p.exists() and not a.force:
        print(f"ledger {p} already exists; refusing to reset it. Prior spend "
              f"must be carried forward, never discarded.", file=sys.stderr)
        return 1
    carried, note = [], None
    if a.carry_forward_from:
        src = Path(a.carry_forward_from)
        old = json.loads(src.read_text())
        for e in old.get("entries", []):
            carried.append({**e, "carried_forward_from": str(src)})
        note = (f"{len(carried)} completed stage(s) totalling "
                f"{sum(e.get('gpu_hours', 0.0) for e in carried):.4f} GPU-h "
                f"carried forward from {src}")
        for o in old.get("open", []):
            print(f"WARNING: source ledger has an OPEN reservation "
                  f"({o.get('stage')}); reconcile it before carrying forward",
                  file=sys.stderr)
            return 1
    payload = {
        "_contract": (
            "Hard GPU-hour ledger with atomic durable admission. DEVICE-hours. "
            "Charges successes, FAILED attempts and setup alike. In-flight "
            "reservations count against the balance, so two concurrent "
            "admissions cannot spend the same GPU-hour. The confirmatory test "
            "stage is reserved. Prior spend is carried forward and is never "
            "reset because a new output directory was created."),
        "approved_by": a.approved_by,
        "experiment": a.experiment,
        "ceiling_gpu_hours": a.ceiling,
        "reserve_gpu_hours": a.reserve,
        "reserve_for": a.reserve_for,
        "reserve_basis": a.reserve_basis,
        "created_utc": _now(),
        "commit": a.commit,
        "carried_forward_note": note,
        "entries": carried,
        "open": [],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    tot = sum(e.get("gpu_hours", 0.0) for e in carried)
    print(f"ledger initialised: ceiling {a.ceiling} GPU-h, reserve {a.reserve} "
          f"for {a.reserve_for!r}, carried-forward spend {tot:.4f} GPU-h -> {p}")
    return 0


def cmd_admit(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        # Settle anything orphaned first, so a crashed predecessor's cost is on
        # the books before this decision is made.
        _reconcile(L, verbose=False)
        avail = L.available(a.draws_on_reserve)
        decision = {
            "stage": a.stage, "need_gpu_hours": a.need, "gpus": max(1, a.gpus),
            "ceiling": L.ceiling(), "spent": round(L.spent(), 4),
            "in_flight": round(L.open_total(), 4),
            "committed": round(L.committed(), 4),
            "reserve_withheld": 0.0 if a.draws_on_reserve
                                else round(L.reserve_remaining(), 4),
            "available_to_this_stage": round(avail, 4),
        }
        if a.need > avail + 1e-9:
            decision["decision"] = "REFUSE"
            print(json.dumps(decision))
            print(f"BUDGET REFUSED: {a.stage} needs {a.need} GPU-h; only "
                  f"{avail:.4f} available (committed {L.committed():.4f} of "
                  f"{L.ceiling()}, including {L.open_total():.4f} in flight; "
                  f"{L.reserve_remaining():.4f} withheld for "
                  f"{L.data.get('reserve_for')!r}). Nothing was launched.",
                  file=sys.stderr)
            return 1
        rid = uuid.uuid4().hex[:16]
        bound = a.need * H / max(1, a.gpus)
        L.data["open"].append({
            "id": rid, "stage": a.stage,
            "reserved_gpu_hours": a.need, "gpus": max(1, a.gpus),
            "gpu_index": a.gpu_index, "pid": a.pid,
            "draws_on_reserve": bool(a.draws_on_reserve),
            "max_wall_seconds": round(bound, 1),
            "started_utc": _now(),
        })
        L.write()
        decision.update({"decision": "ADMIT", "reservation_id": rid,
                         "max_wall_seconds": round(bound, 1)})
        print(json.dumps(decision))
    return 0


def cmd_settle(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        hit = next((o for o in L.data["open"] if o["id"] == a.id), None)
        if hit is None:
            # Already reconciled: a conservative charge was taken because the
            # owning process looked gone. A real settle is a MEASUREMENT and
            # supersedes that estimate, in either direction.
            prior = next((e for e in L.data["entries"]
                          if e.get("reservation_id") == a.id
                          and e.get("reconciled")), None)
            if prior is None:
                print(f"BUDGET ERROR: no open reservation {a.id}", file=sys.stderr)
                return 1
            gpus = max(1, int(prior.get("gpus_occupied", 1)))
            actual = (a.wall_seconds * gpus) / H
            prior.update({
                "gpu_hours": round(actual, 6),
                "wall_seconds": round(a.wall_seconds, 1),
                "exit_code": a.exit_code,
                "reconciled_then_settled": True,
                "charge_basis": ("measured at settle; supersedes the "
                                 "conservative orphan charge of "
                                 f"{prior.get('gpu_hours')} GPU-h"),
                "outcome": ("ok" if a.exit_code == 0
                            else "FAILED (still charged)"),
                "settled_utc": _now(),
            })
            L.write()
            print(json.dumps({**prior, "spent_total": round(L.spent(), 4),
                              "remaining": round(L.ceiling() - L.committed(), 4),
                              "over_ceiling": L.spent() > L.ceiling() + 1e-9}))
            return 0
        L.data["open"] = [o for o in L.data["open"] if o["id"] != a.id]
        gpus = max(1, int(hit.get("gpus", 1)))
        actual = (a.wall_seconds * gpus) / H
        over = actual > hit["reserved_gpu_hours"] + 1e-9
        entry = {
            "stage": hit["stage"], "reservation_id": a.id,
            "gpus_occupied": gpus, "gpu_index": hit.get("gpu_index", ""),
            "wall_seconds": round(a.wall_seconds, 1),
            "gpu_hours": round(actual, 6),
            "reserved_gpu_hours": hit["reserved_gpu_hours"],
            "exceeded_reservation": over,
            "exit_code": a.exit_code,
            "outcome": ("ok" if a.exit_code == 0 else
                        "TIMED OUT (runtime bound hit; still charged)"
                        if a.exit_code in (124, 137) else
                        "FAILED (still charged)"),
            "draws_on_reserve": bool(hit.get("draws_on_reserve")),
            "started_utc": hit.get("started_utc"),
            "settled_utc": _now(),
        }
        L.data["entries"].append(entry)
        L.write()
        out = {**entry, "spent_total": round(L.spent(), 4),
               "remaining": round(L.ceiling() - L.committed(), 4),
               "over_ceiling": L.spent() > L.ceiling() + 1e-9}
        print(json.dumps(out))
        if over:
            print(f"NOTE: {hit['stage']} used {actual:.4f} GPU-h against a "
                  f"{hit['reserved_gpu_hours']:.4f} reservation. The runtime "
                  f"bound should have prevented this; the actual figure is what "
                  f"is charged.", file=sys.stderr)
        if out["over_ceiling"]:
            print(f"CEILING EXCEEDED: spent {L.spent():.4f} of {L.ceiling()} "
                  f"GPU-h. No further stage may start.", file=sys.stderr)
            return 1
    return 0


def _reconcile(L: Ledger, verbose: bool = True) -> int:
    """Conservatively settle reservations whose process is gone."""
    n = 0
    keep = []
    for o in L.data["open"]:
        if _alive(o.get("pid")):
            keep.append(o)
            continue
        gpus = max(1, int(o.get("gpus", 1)))
        try:
            elapsed = (datetime.datetime.now(datetime.timezone.utc)
                       - _parse(o["started_utc"])).total_seconds()
        except Exception:
            elapsed = 0.0
        charge = max(o["reserved_gpu_hours"], (elapsed * gpus) / H)
        L.data["entries"].append({
            "stage": o["stage"], "reservation_id": o["id"],
            "gpus_occupied": gpus, "gpu_index": o.get("gpu_index", ""),
            "wall_seconds": round(elapsed, 1),
            "gpu_hours": round(charge, 6),
            "reserved_gpu_hours": o["reserved_gpu_hours"],
            "exit_code": None,
            "outcome": "ORPHANED (process gone before settling; still charged)",
            "charge_basis": ("max(reservation, elapsed x gpus): what it actually "
                             "used is unknowable, so the larger figure is taken"),
            "reconciled": True,
            "draws_on_reserve": bool(o.get("draws_on_reserve")),
            "started_utc": o.get("started_utc"),
            "settled_utc": _now(),
        })
        n += 1
        if verbose:
            print(f"reconciled orphaned reservation {o['id']} ({o['stage']}): "
                  f"charged {charge:.4f} GPU-h")
    L.data["open"] = keep
    if n:
        L.write()
    return n


def cmd_reconcile(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        n = _reconcile(L)
        still = len(L.data["open"])
        print(f"{n} orphaned reservation(s) settled; {still} still live")
        if still and a.strict:
            for o in L.data["open"]:
                print(f"  live: {o['stage']} pid={o.get('pid')} "
                      f"reserved={o['reserved_gpu_hours']}")
            return 1
    return 0


def cmd_report(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        _ = L.data
        e = L.data["entries"]
        by_stage: dict[str, float] = {}
        for x in e:
            by_stage[x["stage"]] = by_stage.get(x["stage"], 0.0) + x.get("gpu_hours", 0.0)
        failed = [x for x in e if x.get("exit_code") not in (0, None)]
        orphan = [x for x in e if x.get("reconciled")]
        carried = [x for x in e if x.get("carried_forward_from")]
        print(f"ceiling         : {L.ceiling():.4f} GPU-h "
              f"(approved: {L.data.get('approved_by')})")
        print(f"spent           : {L.spent():.4f} GPU-h over {len(e)} completed "
              f"stage(s); {len(failed)} failed and still charged; "
              f"{len(orphan)} orphaned and conservatively charged")
        if carried:
            print(f"  of which carried forward: {len(carried)} stage(s), "
                  f"{sum(x['gpu_hours'] for x in carried):.4f} GPU-h")
        print(f"in flight       : {L.open_total():.4f} GPU-h over "
              f"{len(L.data['open'])} open reservation(s)")
        print(f"committed       : {L.committed():.4f} GPU-h")
        print(f"reserve withheld: {L.reserve_remaining():.4f} GPU-h for "
              f"{L.data.get('reserve_for')!r}")
        print(f"remaining       : {L.ceiling() - L.committed():.4f} GPU-h "
              f"({L.available(False):.4f} usable before the reserve)")
        for s, v in sorted(by_stage.items(), key=lambda kv: -kv[1]):
            print(f"  {v:8.4f}  {s}")
        if L.spent() > L.ceiling() + 1e-9:
            print("STATUS: OVER CEILING")
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init")
    i.add_argument("--ledger", required=True)
    i.add_argument("--ceiling", type=float, required=True)
    i.add_argument("--reserve", type=float, default=0.0)
    i.add_argument("--reserve_for", default="frozen_test")
    i.add_argument("--reserve_basis", default="")
    i.add_argument("--approved_by", default="PI, four total GPU-hours")
    i.add_argument("--experiment", default="matched-effectiveness diagnostic")
    i.add_argument("--commit", default="")
    i.add_argument("--carry_forward_from", default=None)
    i.add_argument("--force", action="store_true")

    m = sub.add_parser("admit")
    m.add_argument("--ledger", required=True)
    m.add_argument("--stage", required=True)
    m.add_argument("--need", type=float, required=True)
    m.add_argument("--gpus", type=int, default=1)
    m.add_argument("--gpu_index", default="")
    m.add_argument("--pid", type=int, default=0)
    m.add_argument("--draws_on_reserve", action="store_true")

    s = sub.add_parser("settle")
    s.add_argument("--ledger", required=True)
    s.add_argument("--id", required=True)
    s.add_argument("--wall_seconds", type=float, required=True)
    s.add_argument("--exit_code", type=int, default=0)

    r = sub.add_parser("reconcile")
    r.add_argument("--ledger", required=True)
    r.add_argument("--strict", action="store_true")

    p = sub.add_parser("report")
    p.add_argument("--ledger", required=True)

    a = ap.parse_args()
    try:
        return {"init": cmd_init, "admit": cmd_admit, "settle": cmd_settle,
                "reconcile": cmd_reconcile, "report": cmd_report}[a.cmd](a)
    except Exception as e:
        print(f"BUDGET ERROR ({type(e).__name__}: {e})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
