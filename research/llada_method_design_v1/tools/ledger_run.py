#!/usr/bin/env python3
"""Run one command with thread limits and append a resource record to the ledger.

Usage:  ledger_run.py --agent F --label "toy dev run" [--threads 1] -- <command ...>

Records wall time, child user+sys CPU seconds (RUSAGE_CHILDREN delta) and the
child max RSS reported by the kernel. CPU seconds are *observed* for this child
tree only; work done outside this wrapper is not captured (see ledger/README.md).
"""
import argparse, json, os, resource, subprocess, sys, time

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ledger", "resources.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(a.threads)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    t1 = time.time()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    rec = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "agent": a.agent, "label": a.label, "cmd": " ".join(cmd), "threads": a.threads,
        "rc": rc, "wall_s": round(t1 - t0, 2),
        "cpu_s": round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 2),
        "max_rss_mib": round(r1.ru_maxrss / 1024, 1),
    }
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[ledger] {json.dumps(rec)}", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
