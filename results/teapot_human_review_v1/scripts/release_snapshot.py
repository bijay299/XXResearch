#!/usr/bin/env python3
"""Snapshot the full identity of both questionnaire releases.

Read-only. Records asset IDs, names, sizes, digests, created/updated timestamps
and tag targets so that a before/after comparison can prove this maintenance
changed nothing respondent-facing. Download counters are recorded but excluded
from the comparison, because they legitimately change while reviewers work.

Usage: release_snapshot.py OUT.json [--download-digest]
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = "bijay299/XXResearch"
TAGS = ["teapot-human-review-v1-taskA", "teapot-human-review-v1-taskA-patch01"]
FROZEN = {
    "teapot-human-review-v1-taskA": {
        "zip": "teapot-human-review-v1-taskA.zip",
        "bytes": 13917868,
        "sha256": "45e3eb7aeb9443e35a4ab990969eb9aa450e2f36b2238192244050b320dbd005",
        "target": "28db2f48a209a0d072c3a294e8ca27a57d868831"},
    "teapot-human-review-v1-taskA-patch01": {
        "zip": "teapot-human-review-v1-taskA-patch01.zip",
        "bytes": 13923703,
        "sha256": "837770ae23e6b1fb56b7269a87e2a2bc23f40f6e2090e5e172ea57a8719ab7e7",
        "target": "ce3b95f1cf72bf6260f4210bcbc0649ebeb0ab3a"},
}


def api(path):
    r = urllib.request.Request(
        "https://api.github.com/repos/" + REPO + path,
        headers={"User-Agent": "thr-snapshot",
                 "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def main():
    out_path = sys.argv[1]
    do_digest = "--download-digest" in sys.argv

    snap = {"repo": REPO, "releases": {}, "tags": {}}
    for t in api("/tags?per_page=100"):
        snap["tags"][t["name"]] = t["commit"]["sha"]

    for tag in TAGS:
        rel = api(f"/releases/tags/{tag}")
        assets = []
        for a in sorted(rel["assets"], key=lambda x: x["name"]):
            assets.append({
                "id": a["id"], "name": a["name"], "size": a["size"],
                "content_type": a["content_type"], "state": a["state"],
                "created_at": a["created_at"], "updated_at": a["updated_at"],
                # recorded but EXCLUDED from the comparison
                "_download_count": a["download_count"],
            })
        entry = {
            "release_id": rel["id"],
            "tag_name": rel["tag_name"],
            "target_commitish": rel["target_commitish"],
            "draft": rel["draft"], "prerelease": rel["prerelease"],
            "created_at": rel["created_at"], "published_at": rel["published_at"],
            "body_sha256": hashlib.sha256((rel.get("body") or "").encode()).hexdigest(),
            "body_len": len(rel.get("body") or ""),
            "n_assets": len(assets), "assets": assets,
        }
        if do_digest:
            z = next((a for a in rel["assets"]
                      if a["name"] == FROZEN[tag]["zip"]), None)
            if z:
                r = urllib.request.Request(
                    z["browser_download_url"],
                    headers={"User-Agent": "thr-snapshot",
                             "Accept": "application/octet-stream"})
                h = hashlib.sha256()
                n = 0
                with urllib.request.urlopen(r, timeout=900) as resp:
                    while True:
                        b = resp.read(1 << 22)
                        if not b:
                            break
                        h.update(b)
                        n += len(b)
                entry["zip_downloaded"] = {
                    "name": z["name"], "bytes": n, "sha256": h.hexdigest(),
                    "matches_frozen": (n == FROZEN[tag]["bytes"]
                                       and h.hexdigest() == FROZEN[tag]["sha256"]),
                }
        entry["tag_target_matches_frozen"] = (
            rel["target_commitish"] == FROZEN[tag]["target"])
        snap["releases"][tag] = entry

    with open(out_path, "w") as fh:
        json.dump(snap, fh, indent=1)

    for tag, e in snap["releases"].items():
        print(f"{tag}")
        print(f"   release_id={e['release_id']} target={e['target_commitish'][:12]} "
              f"target_ok={e['tag_target_matches_frozen']} assets={e['n_assets']} "
              f"body_len={e['body_len']}")
        for a in e["assets"]:
            print(f"     id={a['id']} {a['name']:44} {a['size']:>10,} "
                  f"updated={a['updated_at']}")
        if "zip_downloaded" in e:
            z = e["zip_downloaded"]
            print(f"   ZIP digest: {z['bytes']:,} {z['sha256'][:16]}... "
                  f"matches_frozen={z['matches_frozen']}")
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
