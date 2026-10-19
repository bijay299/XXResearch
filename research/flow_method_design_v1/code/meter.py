"""Resource meter: run a command, append CPU-seconds / peak RSS / wall to results/ledger.jsonl.

Usage: python3 code/meter.py <label> -- <cmd...>
Enforces a per-call wall limit (METER_WALL_S, default 1800) and a process-tree RSS
address-space limit (METER_RSS_GIB, default 12) via RLIMIT_AS on the child.
"""
import json, os, resource, subprocess, sys, time

def main():
    argv = sys.argv[1:]
    label, cmd = argv[0], argv[argv.index("--") + 1:]
    wall = float(os.environ.get("METER_WALL_S", 1800))
    rss = float(os.environ.get("METER_RSS_GIB", 12)) * 2**30
    def lim():
        resource.setrlimit(resource.RLIMIT_AS, (int(rss * 2), int(rss * 2)))
    t0 = time.time(); status = "ok"
    try:
        p = subprocess.run(cmd, preexec_fn=lim, timeout=wall)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc, status = -9, "wall_timeout"
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    rec = dict(label=label, cmd=" ".join(cmd), rc=rc, status=status if rc == 0 or status != "ok" else "FAILED",
               wall_s=round(time.time() - t0, 3), cpu_s=round(ru.ru_utime + ru.ru_stime, 3),
               peak_rss_kib=ru.ru_maxrss, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ledger.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), file=sys.stderr)
    sys.exit(rc)

if __name__ == "__main__":
    main()
