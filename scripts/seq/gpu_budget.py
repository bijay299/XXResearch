#!/usr/bin/env python3
"""A hard GPU-hour ledger for the matched-effectiveness diagnostic.

The approved ceiling is FOUR GPU-hours in total, and it counts everything:
successful stages, failed attempts, and process setup. A budget that only
counted successes would let a crash loop spend the grant invisibly, so every
stage invocation is recorded with the wall time it actually occupied, whatever
its exit code.

GPU-hours here are DEVICE-hours: a stage occupying two GPUs for 30 minutes
spends 1.0 GPU-hour, not 0.5. That is the same unit the approved cost model uses
(protocol_cost_model.json), so the ceiling means what the PI approved.

The confirmatory test stage is RESERVED. Development work may not spend the
budget down to a point where the frozen test evaluation can no longer be
afforded -- that would leave the experiment with selection done and no
confirmatory measurement, which is the worst possible place to stop. So
``check`` subtracts the reserve for any stage not marked as drawing on it.

Concurrency: two trajectories train in parallel, so the ledger is read, modified
and written under an exclusive flock. Reads are locked too, so a check can never
see a half-written ledger.

CPU only. This file touches no GPU; it only accounts for them.

    python scripts/seq/gpu_budget.py init   --ledger L --ceiling 4.0 --reserve 1.008
    python scripts/seq/gpu_budget.py check  --ledger L --need 0.25 --stage train:U17
    python scripts/seq/gpu_budget.py record --ledger L --stage train:U17 --gpus 1 \
                                            --wall_seconds 830 --exit_code 0
    python scripts/seq/gpu_budget.py report --ledger L

Exit codes: 0 allowed / ok, 1 refused (would breach) or error, 2 usage.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import sys
from pathlib import Path

H = 3600.0


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Ledger:
    """Open the ledger under an exclusive lock for the duration of the block."""

    def __init__(self, path: Path, create: bool = False):
        self.path = path
        self.create = create
        self.fh = None
        self.data: dict = {}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self.path.exists() else ("w+" if self.create else None)
        if mode is None:
            raise FileNotFoundError(f"ledger {self.path} does not exist; run `init` first")
        self.fh = self.path.open(mode)
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        raw = self.fh.read()
        self.data = json.loads(raw) if raw.strip() else {}
        return self

    def write(self) -> None:
        self.data["spent_gpu_hours"] = round(self.spent(), 6)
        self.data["remaining_gpu_hours"] = round(self.remaining(), 6)
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

    # ---- accounting
    def spent(self) -> float:
        return sum(e.get("gpu_hours", 0.0) for e in self.data.get("entries", []))

    def ceiling(self) -> float:
        return float(self.data.get("ceiling_gpu_hours", 0.0))

    def reserve(self) -> float:
        """Reserve still held back, net of what the reserved stage already spent."""
        res = float(self.data.get("reserve_gpu_hours", 0.0))
        used = sum(e.get("gpu_hours", 0.0) for e in self.data.get("entries", [])
                   if e.get("draws_on_reserve"))
        return max(0.0, res - used)

    def remaining(self) -> float:
        return self.ceiling() - self.spent()

    def available(self, draws_on_reserve: bool) -> float:
        """Budget a stage may use: the reserve is withheld unless it IS the reserve."""
        return self.remaining() - (0.0 if draws_on_reserve else self.reserve())


def cmd_init(a) -> int:
    p = Path(a.ledger)
    if p.exists() and not a.force:
        print(f"ledger {p} already exists (pass --force to reset)", file=sys.stderr)
        return 1
    payload = {
        "_contract": (
            "Hard GPU-hour ledger. Device-hours, counting successes, FAILED "
            "attempts and setup alike. The confirmatory test stage is reserved "
            "so development work cannot consume it."),
        "approved_by": a.approved_by,
        "ceiling_gpu_hours": a.ceiling,
        "reserve_gpu_hours": a.reserve,
        "reserve_for": a.reserve_for,
        "created_utc": _now(),
        "commit": a.commit,
        "entries": [],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"ledger initialised: ceiling {a.ceiling} GPU-h, "
          f"reserve {a.reserve} GPU-h for {a.reserve_for!r} -> {p}")
    return 0


def cmd_check(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        avail = L.available(a.draws_on_reserve)
        ok = a.need <= avail + 1e-9
        print(json.dumps({
            "stage": a.stage, "need_gpu_hours": a.need,
            "ceiling": L.ceiling(), "spent": round(L.spent(), 4),
            "remaining": round(L.remaining(), 4),
            "reserve_withheld": 0.0 if a.draws_on_reserve else round(L.reserve(), 4),
            "available_to_this_stage": round(avail, 4),
            "decision": "ALLOW" if ok else "REFUSE",
        }))
        if not ok:
            print(f"BUDGET REFUSED: {a.stage} needs {a.need} GPU-h but only "
                  f"{avail:.4f} is available "
                  f"({L.remaining():.4f} remaining, "
                  f"{L.reserve():.4f} withheld for {L.data.get('reserve_for')!r}). "
                  f"Nothing was launched.", file=sys.stderr)
            return 1
    return 0


def cmd_record(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        gpu_hours = (a.wall_seconds * max(1, a.gpus)) / H
        entry = {
            "stage": a.stage,
            "gpus_occupied": max(1, a.gpus),
            "gpu_index": a.gpu_index,
            "wall_seconds": round(a.wall_seconds, 1),
            "gpu_hours": round(gpu_hours, 6),
            "exit_code": a.exit_code,
            "outcome": "ok" if a.exit_code == 0 else "FAILED (still charged)",
            "draws_on_reserve": bool(a.draws_on_reserve),
            "recorded_utc": _now(),
        }
        L.data.setdefault("entries", []).append(entry)
        L.write()
        over = L.spent() > L.ceiling() + 1e-9
        print(json.dumps({**entry, "spent_total": round(L.spent(), 4),
                          "remaining": round(L.remaining(), 4),
                          "over_ceiling": over}))
        if over:
            print(f"CEILING EXCEEDED: spent {L.spent():.4f} of "
                  f"{L.ceiling()} GPU-h. No further stage may start.",
                  file=sys.stderr)
            return 1
    return 0


def cmd_report(a) -> int:
    with Ledger(Path(a.ledger)) as L:
        e = L.data.get("entries", [])
        by_stage: dict[str, float] = {}
        for x in e:
            by_stage[x["stage"]] = by_stage.get(x["stage"], 0.0) + x.get("gpu_hours", 0.0)
        failed = [x for x in e if x.get("exit_code", 0) != 0]
        print(f"ceiling         : {L.ceiling():.3f} GPU-h "
              f"(approved: {L.data.get('approved_by')})")
        print(f"spent           : {L.spent():.4f} GPU-h over {len(e)} stage "
              f"invocation(s), {len(failed)} failed and still charged")
        print(f"reserve withheld: {L.reserve():.4f} GPU-h for "
              f"{L.data.get('reserve_for')!r}")
        print(f"remaining       : {L.remaining():.4f} GPU-h "
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
    i.add_argument("--approved_by", default="PI, four total GPU-hours")
    i.add_argument("--commit", default="")
    i.add_argument("--force", action="store_true")

    c = sub.add_parser("check")
    c.add_argument("--ledger", required=True)
    c.add_argument("--need", type=float, required=True)
    c.add_argument("--stage", required=True)
    c.add_argument("--draws_on_reserve", action="store_true")

    r = sub.add_parser("record")
    r.add_argument("--ledger", required=True)
    r.add_argument("--stage", required=True)
    r.add_argument("--wall_seconds", type=float, required=True)
    r.add_argument("--gpus", type=int, default=1)
    r.add_argument("--gpu_index", default="")
    r.add_argument("--exit_code", type=int, default=0)
    r.add_argument("--draws_on_reserve", action="store_true")

    p = sub.add_parser("report")
    p.add_argument("--ledger", required=True)

    a = ap.parse_args()
    try:
        return {"init": cmd_init, "check": cmd_check,
                "record": cmd_record, "report": cmd_report}[a.cmd](a)
    except Exception as e:
        print(f"BUDGET ERROR ({type(e).__name__}: {e})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
