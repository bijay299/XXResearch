#!/usr/bin/env python3
"""Generate EVIDENCE_INDEX.md for teapot_human_review_v1. Run LAST."""
import argparse
import hashlib
import json
import os
import subprocess
import urllib.request

REPO = "bijay299/XXResearch"
RESULTS = "results/teapot_human_review_v1"


def sh(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    rdir = os.path.join(a.repo_dir, RESULTS)
    tracked = subprocess.run(["git", "-C", a.repo_dir, "ls-files", RESULTS],
                             capture_output=True, text=True, check=True).stdout.split()

    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}",
        headers={"User-Agent": "thr-index", "Accept": "application/vnd.github+json"})
    rel = json.loads(urllib.request.urlopen(req, timeout=120).read())

    def load(name):
        p = os.path.join(rdir, name)
        return json.load(open(p)) if os.path.isfile(p) else {}

    pv = load("verify/published_verification_patch01.json")
    pv_old = load("verify/published_verification.json")
    ma = load("verify/masking_audit.json")
    pb = load("verify/private_backup.json")
    bq = load("qa/qa_patch01_downloaded_report.json")
    vq = load("qa/validator_qa_report.json")
    sv = load("verify/source_verification_summary.json")
    tz = load("TASKA_ZIP_IDENTITY.json")

    tree = f"https://github.com/{REPO}/blob/{a.commit}"
    L = []
    w = L.append

    w("# EVIDENCE_INDEX — teapot_human_review_v1\n")
    w("Complete file and hash index. Generated last.\n")
    w("> File hashes and links below are pinned to the commit named here, which")
    w("> contains every indexed file. This index file itself is added in the")
    w("> immediately following commit, so it cannot list its own hash.\n")
    w(f"* **Commit:** [`{a.commit}`](https://github.com/{REPO}/commit/{a.commit})")
    w("* **Branch:** `research/audit-and-matched-effectiveness-protocol`")
    w(f"* **Release tag:** [`{rel['tag_name']}`]({rel['html_url']}) → target "
      f"`{rel.get('target_commitish')}`")
    w(f"* **Release assets:** {len(rel['assets'])}, "
      f"{sum(x['size'] for x in rel['assets']):,} bytes")
    w("* **Status:** `READY_FOR_INDEPENDENT_REVIEW` — **Task A only**. No "
      "participant, no label, no outcome.\n")

    w("## Verification summary\n")
    w("| check | result |")
    w("|---|---|")
    w(f"| Source images verified against accepted hashes | **{sv.get('pngs_verified')} / {sv.get('pngs_expected')}** |")
    w(f"| Source verdict | **`{sv.get('verdict')}`** |")
    w(f"| Pairs with byte-identical initial noise | **{sv.get('all_noise_identical_across_conditions')}** ({sv.get('distinct_noise_hashes')} distinct hashes) |")
    w(f"| Task-A bundle images | **{tz.get('n_images')} / 40**, {tz.get('n_members')} members |")
    w(f"| Mechanical validator QA | **{vq.get('n_pass')} / {vq.get('n_cases')}** pass |")
    w(f"| Focused browser QA (patch01) | **{bq.get('n_pass')} / {bq.get('n_checks')}** pass |")
    w(f"| Engines exercised | {', '.join(bq.get('engines') or [])} |")
    w(f"| Browser network requests attempted | **{len(bq.get('network_requests_attempted') or [])}** |")
    w(f"| Masking audit | **{'clean' if ma.get('clean') else 'NOT CLEAN'}** "
      f"({ma.get('scanned_public_files')} public files, "
      f"{(ma.get('zip') or {}).get('n_members')} ZIP members) |")
    w(f"| Task-B members in the published ZIP | **{len((ma.get('zip') or {}).get('taskB_members') or [])}** |")
    w(f"| Private backup | **`{pb.get('verdict')}`** ({pb.get('n_files')} files, host-performed) |")
    w(f"| Published-package verification (patch01) | **`{pv.get('verdict')}`** |")
    im = pv.get("images", {})
    w(f"| — all 40 image hashes match the accepted source | **{im.get('all_hashes_match_accepted_source')}** |")
    w(f"| — support files complete | **{pv.get('support_complete')}** |")
    w(f"| — private-association audit on downloaded bytes | **{(pv.get('private_audit') or {}).get('clean')}** |")
    b2 = pv.get("browser_qa_on_downloaded_bytes", {})
    w(f"| — browser QA re-run ON THE DOWNLOADED BYTES | **{b2.get('n_pass')} / {b2.get('n_checks')}** |")
    pa = (ma.get("preview_and_release_assets") or {})
    w(f"| Disallowed preview assets staged | **{len(pa.get('staged_disallowed') or [])}** |")
    lr = pa.get("live_releases") or {}
    w(f"| Disallowed preview assets live on either release | **{sum(len(v.get('disallowed_preview_assets') or []) for v in lr.values() if isinstance(v, dict))}** |")
    w(f"| — downloaded ZIP sha256 matches | **{pv.get('sha256_match')}** |")
    w("")

    w("## Release assets\n")
    w("| asset | bytes | download |")
    w("|---|---|---|")
    for x in sorted(rel["assets"], key=lambda y: y["name"]):
        w(f"| `{x['name']}` | {x['size']:,} | [download]({x['browser_download_url']}) |")
    w("")
    if tz:
        w(f"Reviewer bundle sha256: `{tz.get('sha256')}` "
          f"({tz.get('bytes'):,} bytes, deterministic build).\n")

    w(f"## Repository files at `{a.commit[:12]}`\n")
    w("| file | bytes | sha256 |")
    w("|---|---|---|")
    tot = 0
    for rp in sorted(tracked):
        p = os.path.join(a.repo_dir, rp)
        if not os.path.isfile(p):
            continue
        s = os.path.getsize(p)
        tot += s
        w(f"| [`{os.path.relpath(rp, RESULTS)}`]({tree}/{rp}) | {s:,} | `{sh(p)}` |")
    w("")
    w(f"**{len(tracked)} tracked files, {tot:,} bytes.**\n")

    w("## Scope reminder\n")
    w("This index establishes **preparation and integrity only**: that the "
      "package was built from verified source material, that it is complete and "
      "masked as described, that it functions offline, and that the published "
      "artifact is the one produced.\n")
    w("It establishes **nothing** about erasure efficacy, retention, quality, "
      "evaluator bias, novelty or paper readiness, and it does **not** "
      "establish that any participant exists, that any independent label was "
      "collected, or that the evaluation is scientifically valid. **No reviewer "
      "has been recruited or contacted and no response exists.** Task B is "
      "prepared but **not released**.\n")

    out = os.path.join(rdir, "EVIDENCE_INDEX.md")
    open(out, "w").write("\n".join(L))
    print("wrote", out, f"({os.path.getsize(out):,} bytes)")
    print(f"tracked files: {len(tracked)} | release assets: {len(rel['assets'])}")


if __name__ == "__main__":
    main()
