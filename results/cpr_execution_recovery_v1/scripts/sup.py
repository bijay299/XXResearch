#!/usr/bin/env python3
"""Metered launcher CLI: every setup/diagnostic/verification job goes through the ledger.
  sup.py run --job J --bucket eng|main|reserve [--gpus 0,1] [--cores 2] [--wall S] [--rss G] [--sto G] [--img-primary N]
             [--img-technical N] -- cmd args...
  sup.py status | reconcile | gaps
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.environ["CPRX_ROOT"], "code"))
from cprx import ledger as L

ap = argparse.ArgumentParser()
ap.add_argument("cmd")
ap.add_argument("--job"); ap.add_argument("--bucket", default="eng")
ap.add_argument("--gpus", default=""); ap.add_argument("--cores", type=int, default=2)
ap.add_argument("--wall", type=float, default=600); ap.add_argument("--grace", type=float, default=30)
ap.add_argument("--rss", type=float, default=8.0); ap.add_argument("--sto", type=float, default=0.5)
ap.add_argument("--img-primary", type=int, default=0); ap.add_argument("--img-technical", type=int, default=0)
ap.add_argument("--kind", default="planned"); ap.add_argument("--note")
_i = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
a = ap.parse_args(sys.argv[1:_i])
a.rest = sys.argv[_i + 1:]
if a.cmd == "run":
    argv = a.rest
    devs = [int(x) for x in a.gpus.split(",") if x != ""]
    r = L.run_job(a.job, a.bucket, devs, argv, a.wall, grace_s=a.grace, cores=a.cores, kind=a.kind,
                  images_primary=a.img_primary, images_technical=a.img_technical, note=a.note,
                  declared_rss_gib=a.rss, declared_storage_gib=a.sto)
    print(json.dumps(r, default=str)); sys.exit(0 if r["status"] == "ok" else 1)
elif a.cmd == "status":
    print(json.dumps(L.totals(), indent=1, default=str))
elif a.cmd == "reconcile":
    print(L.reconcile("cli"))
elif a.cmd == "gaps":
    print(json.dumps(L.coverage_and_gaps(), indent=1))
