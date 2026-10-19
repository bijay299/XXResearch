#!/usr/bin/env python3
"""Run one CPU-only script through the supervisor (charged to the CPU ledger; zero GPU)."""
import sys
from cprd2 import ledger as LG
job, wall, *argv = sys.argv[1:]
r = LG.run_job(job, "entry", [], [sys.executable] + argv, max_wall_s=float(wall), grace_s=60, workers=2, kind="planned")
print(r)
sys.exit(0 if r["status"] == "ok" else 1)
