#!/usr/bin/env python3
"""Mechanical re-check of the 38-row historical EVIDENCE_INDEX (9940e94) against
Git objects at its payload commit. Stdlib only; reads Git; writes one JSON file."""
import hashlib, json, re, subprocess
REPO = "/data/bijaypandey/xxresearch_standalone"
S = "results/cpr_detector_controls_v1"
REV, PAYLOAD = "9940e9403452e2b382c108c2d4d50bd3d054d707", "42614565cd79f911c194d9f7c8d1dea017fbe17c"
def git(*a): return subprocess.check_output(["git", "-C", REPO, *a])
idx = git("show", f"{REV}:{S}/EVIDENCE_INDEX.md").decode()
rows = re.findall(r"^\| (\d+) \| `([^`]+)` \| `([0-9a-f]{64})` \| ([\d,]+) \|", idx, re.M)
out, bad = [], 0
for n, path, sha, size in rows:
    b = git("show", f"{PAYLOAD}:{S}/{path}")
    ok = hashlib.sha256(b).hexdigest() == sha and len(b) == int(size.replace(",", ""))
    bad += (not ok)
    out.append({"n": int(n), "path": path, "sha256": sha, "bytes": len(b), "match": ok})
json.dump({"historical_index_sha256": hashlib.sha256(idx.encode()).hexdigest(),
           "payload_commit": PAYLOAD, "rows": len(rows), "mismatches": bad, "detail": out},
          open("historical_index_reverification.json", "w"), indent=1)
print(len(rows), "rows;", bad, "mismatches")
