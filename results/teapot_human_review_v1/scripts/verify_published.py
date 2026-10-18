#!/usr/bin/env python3
"""Download the published Task-A ZIP into a FRESH directory and verify it.

Checks: hash, complete neutral-file inventory, offline function in a real
browser, and the absence of any embedded Task-B association or answer key.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
FRESH = os.path.join(ROOT, "verify", "fresh_download")
TAG = "teapot-human-review-v1-taskA"
ZIPNAME = "teapot-human-review-v1-taskA.zip"
_ident = json.load(open(os.path.join(ROOT, "publish", "results_tree",
                                     "TASKA_ZIP_IDENTITY.json")))
EXPECT_SHA = _ident["sha256"]
EXPECT_BYTES = _ident["bytes"]
ZROOT = "teapot-human-review-v1-taskA"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    shutil.rmtree(FRESH, ignore_errors=True)
    os.makedirs(FRESH)

    req = urllib.request.Request(
        f"https://api.github.com/repos/bijay299/XXResearch/releases/tags/{TAG}",
        headers={"User-Agent": "thr-verify", "Accept": "application/vnd.github+json"})
    rel = json.loads(urllib.request.urlopen(req, timeout=120).read())
    asset = next(a for a in rel["assets"] if a["name"] == ZIPNAME)

    dst = os.path.join(FRESH, ZIPNAME)
    r = urllib.request.Request(asset["browser_download_url"],
                               headers={"User-Agent": "thr-verify",
                                        "Accept": "application/octet-stream"})
    with urllib.request.urlopen(r, timeout=900) as resp, open(dst, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 22)

    got_sha, got_bytes = sha256_file(dst), os.path.getsize(dst)
    rep = {
        "tag": rel["tag_name"],
        "release_target_commitish": rel["target_commitish"],
        "release_html_url": rel["html_url"],
        "n_assets_on_release": len(rel["assets"]),
        "download_url": asset["browser_download_url"],
        "expected_sha256": EXPECT_SHA, "got_sha256": got_sha,
        "sha256_match": got_sha == EXPECT_SHA,
        "expected_bytes": EXPECT_BYTES, "got_bytes": got_bytes,
        "bytes_match": got_bytes == EXPECT_BYTES,
    }

    # ---- inventory ----
    ex = os.path.join(FRESH, "extracted")
    with zipfile.ZipFile(dst) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        z.extractall(ex)
    base = os.path.join(ex, ZROOT)
    pngs = sorted(f[:-4] for f in os.listdir(os.path.join(base, "images"))
                  if f.endswith(".png"))
    expected_ids = [f"A-{i:03d}" for i in range(1, 41)]
    support = sorted(os.path.basename(n) for n in names
                     if not n.startswith(f"{ZROOT}/images/"))

    rep["inventory"] = {
        "n_members": len(names),
        "n_images": len(pngs),
        "expected_images": 40,
        "all_40_present": pngs == expected_ids,
        "missing_ids": [i for i in expected_ids if i not in pngs],
        "unexpected_ids": [i for i in pngs if i not in expected_ids],
        "all_ids_neutral": all(re.fullmatch(r"A-\d{3}", i) for i in pngs),
        "support_files": support,
        "complete": pngs == expected_ids and set(support) >= {
            "index.html", "app.js", "items.js", "style.css", "items.json",
            "INSTRUCTIONS.md", "README.txt"},
    }

    # ---- no Task-B associations / answer keys / mapping ----
    bad = []
    for n in names:
        p = os.path.join(ex, n)
        if n.endswith(".png"):
            continue
        txt = open(p, "r", errors="ignore").read()
        for label, rx in [
            ("Task-B pair id", re.compile(r"\bB-\d{2}\b")),
            ("Task-B role", re.compile(r"_(reference|candidate)\.png")),
            ("case id", re.compile(r"\bCASE-\d{2}\b")),
            ("condition", re.compile(r"_(m0|edited)_seed")),
            ("eval seed", re.compile(r"\bseed ?7\d{3}\b", re.I)),
            ("answer key", re.compile(r'"(answer_key|ground_truth|gold)"')),
            ("mapping", re.compile(r'"(taskA_mapping|taskB_mapping|source_filename)"')),
        ]:
            if rx.search(txt):
                bad.append({"member": n, "kind": label})
    rep["no_taskB_or_answer_key"] = {
        "findings": bad, "clean": not bad,
        "taskB_members": [n for n in names if re.search(r"\bB-\d{2}\b", n)],
    }

    # ---- offline function in a real browser, from the FRESH extraction ----
    try:
        from playwright.sync_api import sync_playwright
        attempts = []
        with sync_playwright() as p:
            br = p.chromium.launch()
            ctx = br.new_context(viewport={"width": 1280, "height": 1000})
            page = ctx.new_page()

            def route(rt):
                if not rt.request.url.startswith("file://"):
                    attempts.append(rt.request.url)
                    return rt.abort()
                return rt.continue_()
            page.route("**/*", route)
            page.goto("file://" + os.path.join(base, "index.html"), wait_until="load")
            gate = page.is_visible("#gate h1")
            page.fill("#alias", "fresh-verify")
            page.select_option("#ordervar", "V1")
            for n in ("exp_pair", "exp_cond", "exp_coach"):
                page.check(f'input[name="{n}"][value="no"]')
            page.click("#startbtn")
            page.wait_for_selector("#app:not([hidden])", timeout=15000)
            iid = page.inner_text("#itemid")
            nat = page.eval_on_selector("#theimage", "e=>e.naturalWidth")
            foot = page.inner_text("#foot")
            ctx.close(); br.close()
        rep["browser"] = {
            "engine": "chromium (Chrome engine -- the one that broke before)",
            "opened_offline_from_fresh_extraction": bool(gate),
            "first_item_id": iid,
            "first_item_id_neutral": bool(re.fullmatch(r"A-\d{3}", iid)),
            "image_native_width": nat,
            "footer": foot,
            "network_requests_attempted": attempts,
            "no_network_required": not attempts,
            "ok": bool(gate) and nat == 512 and not attempts,
        }
    except Exception as e:
        rep["browser"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    ok = (rep["sha256_match"] and rep["bytes_match"]
          and rep["inventory"]["complete"] and rep["inventory"]["all_ids_neutral"]
          and rep["no_taskB_or_answer_key"]["clean"]
          and rep["browser"].get("ok"))
    rep["verdict"] = "PUBLISHED_PACKAGE_VERIFIED" if ok else "PUBLISHED_PACKAGE_PROBLEM"

    json.dump(rep, open(os.path.join(ROOT, "verify", "published_verification.json"), "w"),
              indent=1)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("inventory", "no_taskB_or_answer_key", "browser")},
                     indent=1))
    print("inventory complete:", rep["inventory"]["complete"],
          "| images:", rep["inventory"]["n_images"],
          "| neutral:", rep["inventory"]["all_ids_neutral"])
    print("no TaskB/answer key:", rep["no_taskB_or_answer_key"]["clean"])
    print("browser ok:", rep["browser"].get("ok"),
          "| no network:", rep["browser"].get("no_network_required"))
    print("VERDICT:", rep["verdict"])
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
