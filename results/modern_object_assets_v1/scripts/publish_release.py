#!/usr/bin/env python3
"""Create the GitHub Release and upload the asset parts.

Uses the GitHub REST API with the token already in the user's git credential
store. The token is read at call time and never printed, logged, or placed in
argv/env. No paid service, no Git LFS, no large weights in ordinary Git.

Idempotent / resumable: an asset already present on the release with the right
size is skipped, so an interrupted upload can simply be re-run.

Usage:
  publish_release.py --tag TAG --target SHA --title T --notes-file F [--assets DIR]
  publish_release.py --list --tag TAG
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
REPO = "bijay299/XXResearch"
CRED = os.path.expanduser("~/.git-credentials")


def _token():
    with open(CRED) as fh:
        for line in fh:
            m = re.match(r"https://([^:]+):([^@]+)@github\.com", line.strip())
            if m:
                return m.group(2)
    raise SystemExit("no github.com credential found")


def req(method, path=None, data=None, url=None, raw=None, headers=None, retries=4):
    u = url or (API + path)
    hdrs = {"Authorization": "Bearer " + _token(),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "xxresearch-publish"}
    body = raw
    if data is not None:
        body = json.dumps(data).encode()
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)

    last = None
    for attempt in range(retries):
        try:
            r = urllib.request.Request(u, data=body, headers=hdrs, method=method)
            with urllib.request.urlopen(r, timeout=1800) as resp:
                payload = resp.read()
                return resp.status, (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = {"raw": payload[:400].decode("utf-8", "replace")}
            if e.code in (502, 503, 504, 429) and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                last = (e.code, parsed)
                continue
            return e.code, parsed
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                last = (0, {"error": str(e)})
                continue
            return 0, {"error": str(e)}
    return last or (0, {"error": "exhausted retries"})


def get_release(tag):
    st, d = req("GET", f"/repos/{REPO}/releases/tags/{urllib.parse.quote(tag)}")
    return d if st == 200 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", default=None, help="commit sha the tag points at")
    ap.add_argument("--title", default=None)
    ap.add_argument("--notes-file", default=None)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        rel = get_release(a.tag)
        if not rel:
            print("no such release:", a.tag)
            sys.exit(1)
        print(json.dumps({
            "tag": rel["tag_name"], "name": rel["name"], "draft": rel["draft"],
            "target_commitish": rel.get("target_commitish"),
            "html_url": rel["html_url"],
            "assets": [{"name": x["name"], "size": x["size"],
                        "download": x["browser_download_url"]}
                       for x in rel["assets"]],
        }, indent=2))
        return

    rel = get_release(a.tag)
    if rel is None:
        notes = ""
        if a.notes_file:
            with open(a.notes_file) as fh:
                notes = fh.read()
        payload = {"tag_name": a.tag, "name": a.title or a.tag,
                   "body": notes, "draft": False, "prerelease": False}
        if a.target:
            payload["target_commitish"] = a.target
        st, rel = req("POST", f"/repos/{REPO}/releases", data=payload)
        if st not in (200, 201):
            print("FAILED to create release:", st, json.dumps(rel)[:600])
            sys.exit(2)
        print(f"created release {a.tag} -> {rel['html_url']}")
    else:
        print(f"release {a.tag} already exists -> {rel['html_url']}")
        if a.notes_file:
            with open(a.notes_file) as fh:
                notes = fh.read()
            st, rel2 = req("PATCH", f"/repos/{REPO}/releases/{rel['id']}",
                           data={"body": notes, "name": a.title or a.tag})
            if st == 200:
                rel = rel2
                print("release notes updated")

    if not a.assets:
        return

    existing = {x["name"]: x for x in rel["assets"]}
    names = sorted(os.listdir(a.assets))
    upload_base = rel["upload_url"].split("{")[0]

    uploaded, skipped, failed = [], [], []
    for name in names:
        p = os.path.join(a.assets, name)
        if not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        if name in existing and existing[name]["size"] == size:
            print(f"skip (already uploaded, same size): {name}")
            skipped.append(name)
            continue
        if name in existing:
            print(f"replacing size-mismatched asset: {name}")
            req("DELETE", f"/repos/{REPO}/releases/assets/{existing[name]['id']}")

        with open(p, "rb") as fh:
            blob = fh.read()
        u = f"{upload_base}?name={urllib.parse.quote(name)}"
        t0 = time.time()
        st, d = req("POST", url=u, raw=blob,
                    headers={"Content-Type": "application/octet-stream",
                             "Content-Length": str(len(blob))})
        dt = time.time() - t0
        if st in (200, 201):
            print(f"uploaded {name}  {size/1024**3:.3f} GiB in {dt:.0f}s "
                  f"({size/1024**2/max(dt,1e-9):.1f} MiB/s)")
            uploaded.append(name)
        else:
            print(f"FAILED {name}: {st} {json.dumps(d)[:300]}")
            failed.append({"name": name, "status": st, "body": d})

    rel = get_release(a.tag)
    print(json.dumps({"tag": a.tag, "html_url": rel["html_url"],
                      "n_assets_on_release": len(rel["assets"]),
                      "uploaded": len(uploaded), "skipped": len(skipped),
                      "failed": failed}, indent=2))
    sys.exit(0 if not failed else 6)


if __name__ == "__main__":
    main()
