#!/usr/bin/env python3
"""Generate EVIDENCE_INDEX.md — the complete file/hash index. Run LAST.

Covers every tracked file in the published results directory and every live
release asset, each with its sha256 and a commit-pinned or direct download link.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

REPO = "bijay299/XXResearch"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--results-rel", default="results/modern_object_assets_v1")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    rdir = os.path.join(a.repo_dir, a.results_rel)

    tracked = subprocess.run(
        ["git", "-C", a.repo_dir, "ls-files", a.results_rel],
        capture_output=True, text=True, check=True).stdout.split()

    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}",
        headers={"User-Agent": "xxresearch-index",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        rel = json.loads(r.read())

    rv_path = os.path.join(rdir, "REMOTE_VERIFICATION.json")
    rv = json.load(open(rv_path)) if os.path.isfile(rv_path) else {}

    tree = f"https://github.com/{REPO}/blob/{a.commit}"
    L = []
    w = L.append

    w(f"# EVIDENCE_INDEX — modern_object_assets_v1\n")
    w("Complete file and hash index. Generated last, after remote verification.\n")
    w(f"* **Commit:** [`{a.commit}`](https://github.com/{REPO}/commit/{a.commit})")
    w(f"* **Branch:** `research/audit-and-matched-effectiveness-protocol`")
    w(f"* **Release tag:** [`{rel['tag_name']}`]({rel['html_url']}) → target `{rel['target_commitish']}`")
    w(f"* **Release assets:** {len(rel['assets'])}, "
      f"{sum(x['size'] for x in rel['assets']):,} bytes\n")

    if rv:
        w("## Remote verification\n")
        w(f"**Verdict: `{rv.get('verdict')}`** — downloaded into a fresh directory "
          f"({rv.get('n_downloaded')} assets, {rv.get('download_seconds')} s), "
          "every remote hash checked, reconstructed losslessly, remote checkpoint "
          "loaded against the documented pinned base.\n")
        w("| check | result |")
        w("|---|---|")
        w(f"| Assets downloaded | {rv.get('n_downloaded')} / {rv.get('n_assets_on_release')} |")
        w(f"| Remote hashes verified | {rv.get('n_parts_hash_ok')} / {rv.get('n_parts_checked')} |")
        w(f"| All remote hashes match | **{rv.get('all_remote_hashes_match')}** |")
        w(f"| Lossless reconstruction of the split checkpoint | **{rv.get('reconstruction_lossless')}** |")
        w(f"| Inner checkpoint files match published hashes | **{rv.get('all_inner_files_match')}** |")
        rr = rv.get("remote_reload", {})
        w(f"| Remote checkpoint loads via the documented pinned-base path | **{rr.get('overwrite_included_transformer')}** |")
        w(f"| Remote reload bitwise-identical to local post-merge | **{rr.get('bitwise_identical_to_local_postmerge')}** ({rr.get('n_value_mismatches')} mismatches) |")
        w(f"| Remote reloaded tensors finite | **{rr.get('all_finite')}** |")
        im = rv.get("images", {})
        w(f"| Primary PNG completeness | **{im.get('n_primary_png')} / {im.get('expected_primary_png')}** |")
        w(f"| All PNG hashes match | **{im.get('all_png_hashes_match')}** |")
        w(f"| Paired-input correspondence | **{im.get('paired_input_correspondence_ok')}** ({im.get('n_noise_tensors_verified')}/20 noise tensors re-verified) |")
        w(f"| Verification GPU-hours / decoded images | {rv.get('gpu_device_hours_used')} / {rv.get('decoded_images_used')} |")
        w("")

    w("## Release assets (live download links)\n")
    w("| asset | bytes | sha256 | download |")
    w("|---|---|---|---|")
    man = None
    for x in rel["assets"]:
        if x["name"] == "ASSET_MANIFEST.json":
            with urllib.request.urlopen(urllib.request.Request(
                    x["browser_download_url"],
                    headers={"User-Agent": "xxresearch-index"}), timeout=300) as r:
                man = json.loads(r.read())
    hashes = {}
    if man:
        for asset in man["assets"]:
            for p in asset["parts"]:
                hashes[p["name"]] = p["sha256"]
    for x in sorted(rel["assets"], key=lambda y: y["name"]):
        h = hashes.get(x["name"], "")
        hs = f"`{h[:16]}…`" if h else "_(self-describing manifest)_"
        w(f"| `{x['name']}` | {x['size']:,} | {hs} | [download]({x['browser_download_url']}) |")
    w("")

    if man:
        w("### Reconstructed logical artifacts\n")
        w("| logical artifact | parts | reconstructed bytes | reconstructed sha256 |")
        w("|---|---|---|---|")
        for asset in man["assets"]:
            w(f"| {asset['logical_name']} | {asset['n_parts']} | "
              f"{asset['reconstructed_bytes']:,} | `{asset['reconstructed_sha256']}` |")
        w("")

    w(f"## Repository files at `{a.commit[:12]}`\n")
    w("| file | bytes | sha256 |")
    w("|---|---|---|")
    total = 0
    for rp in sorted(tracked):
        p = os.path.join(a.repo_dir, rp)
        if not os.path.isfile(p):
            continue
        s = os.path.getsize(p)
        total += s
        rel_name = os.path.relpath(rp, a.results_rel)
        w(f"| [`{rel_name}`]({tree}/{rp}) | {s:,} | `{sha256_file(p)}` |")
    w("")
    w(f"**{len(tracked)} tracked files, {total:,} bytes.**\n")

    w("## Scope reminder\n")
    w("This index establishes **technical** delivery and provenance only: that "
      "the published artifacts are the ones produced, that they download and "
      "reconstruct losslessly, and that the remote checkpoint reloads to the "
      "same state. It establishes **nothing** about erasure efficacy — no "
      "detector, automated judge, human label or pass threshold was applied "
      "anywhere in this project.\n")

    out = os.path.join(rdir, "EVIDENCE_INDEX.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print("wrote", out, f"({os.path.getsize(out):,} bytes)")
    print(f"tracked files: {len(tracked)} | release assets: {len(rel['assets'])}")


if __name__ == "__main__":
    main()
