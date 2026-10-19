#!/usr/bin/env python3
"""Write records/FREEZE_MANIFEST_<n>.json: sha256 of every code/script/config/record/patch file at freeze time."""
import hashlib, json, os, sys, time
ROOT = os.environ["CPRX_ROOT"]
n = sys.argv[1]
what = sys.argv[2]
files = {}
for d in ("code", "scripts", "configs", "records", "patches", "visual_audit"):
    for r, _, fs in os.walk(os.path.join(ROOT, d)):
        if "__pycache__" in r or "public" in r.split(os.sep):
            continue
        for f in fs:
            if f.startswith("FREEZE_MANIFEST") or f.endswith(".pyc") or (d == "visual_audit" and f != "sample.json"):
                continue
            p = os.path.join(r, f)
            files[os.path.relpath(p, ROOT)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"freeze": n, "what": what, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files}
json.dump(out, open(os.path.join(ROOT, "records", f"FREEZE_MANIFEST_{n}.json"), "w"), indent=1, sort_keys=True)
print(len(files), "files", out["utc"])
