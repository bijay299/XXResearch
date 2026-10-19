#!/usr/bin/env python3
"""Post-publication mechanical checks for Amendment 01. Stdlib only. Reads Git objects, the
live remote refs and GitHub release/tag metadata; writes post_publication_verification.json.
Never imports torch/candidate/test modules; runs nothing from the study."""
import hashlib, json, os, re, subprocess, sys, time, urllib.request
REPO = "/data/bijaypandey/xxresearch_standalone"
S = "results/cpr_detector_controls_v1"
REV = "9940e9403452e2b382c108c2d4d50bd3d054d707"
A = sys.argv[1]
BR = "research/audit-and-matched-effectiveness-protocol"
START = "2026-10-19T16:41:31Z"
def git(*a, check=True): return subprocess.run(["git", "-C", REPO, *a], capture_output=True, check=check).stdout
tok = re.search(r"https://[^:]+:([^@]+)@github.com", open(os.path.expanduser("~/.git-credentials")).read()).group(1)
def api(path):
    r = urllib.request.Request("https://api.github.com/repos/bijay299/XXResearch/" + path,
                               headers={"Authorization": "token " + tok, "Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(r))
out = {"reviewed_head": REV, "amendment_payload_commit": A, "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
# protected paths: tree equality
prot = [f"{S}/{p}" for p in ("code", "configs", "artifacts", "RUN_MANIFEST.md", "RECONSTRUCTION.md")]
out["protected_paths_diff_REV_vs_A_empty"] = git("diff", "--stat", REV, A, "--", *prot).strip() == b""
rows = [l.split(None, 2) for l in open("protected_files_sha256_at_9940e94.txt") if not l.startswith("#")]
bad = []
for sha, size, path in rows:
    b = git("show", f"{A}:{S}/{path.strip()}")
    if hashlib.sha256(b).hexdigest() != sha or len(b) != int(size): bad.append(path.strip())
out["protected_files_checked_against_A_blobs"] = len(rows); out["protected_files_mismatches"] = bad
# allowed diff scope REV..A
changed = git("diff", "--name-status", REV, A).decode().strip().splitlines()
allowed = lambda p: (p in ("results/README.md",) or p in (f"{S}/FINAL_REPORT.md", f"{S}/ENFORCEMENT_GAPS.md",
    f"{S}/AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md", f"{S}/PRESERVATION_AND_ACCOUNTING.md")
    or p.startswith(f"{S}/amendment_01_records/"))
out["changed_REV_to_A"] = changed
out["changed_outside_allowed"] = [l for l in changed if not allowed(l.split("\t")[-1])]
def outside(rev):
    t = git("ls-tree", "-r", rev).decode().splitlines()
    t = [l for l in t if not l.split("\t")[1].startswith(S + "/") and l.split("\t")[1] != "results/README.md"]
    return hashlib.sha256("\n".join(t).encode()).hexdigest()
out["tree_outside_study_and_results_README_equal"] = outside(REV) == outside(A)
# remote
ls = git("ls-remote", "origin", f"refs/heads/{BR}").decode().split()
out["live_remote_sha_of_branch"] = ls[0] if ls else None
# release
rel = api("releases/417012770"); pre = json.load(open("release_417012770_pre_amendment.json"))
strip = lambda a: {k: a[k] for k in ("id", "name", "size", "digest", "created_at", "updated_at", "state", "content_type")}
prea = {a["id"]: {k: a.get(k) for k in ("id", "name", "size", "digest", "created_at", "updated_at", "state")} for a in pre["assets"]}
nowa = {a["id"]: {k: a.get(k) for k in ("id", "name", "size", "digest", "created_at", "updated_at", "state")} for a in rel["assets"]}
out["release"] = {"id": rel["id"], "tag": rel["tag_name"], "target": rel["target_commitish"], "target_unchanged": rel["target_commitish"] == pre["target_commitish"],
    "created_at_unchanged": rel["created_at"] == pre["created_at"], "published_at_unchanged": rel["published_at"] == pre["published_at"],
    "name_unchanged": rel["name"] == pre["name"], "draft": rel["draft"], "prerelease": rel["prerelease"],
    "assets_identical_to_pre_amendment": prea == nowa, "n_assets": len(rel["assets"])}
body = rel["body"].replace("\r\n", "\n"); orig = open("release_body_original.md", "rb").read().decode()
out["release"]["body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
out["release"]["body_ends_with_original_verbatim"] = body.endswith(orig)
out["release"]["original_body_sha256"] = hashlib.sha256(orig.encode()).hexdigest()
out["release"]["notice_pinned_to_A"] = A in body
# tags and other releases
tags = {l.split()[1]: l.split()[0] for l in git("ls-remote", "--tags", "origin").decode().splitlines() if "^{}" not in l}
pretags = {l.split()[1]: l.split()[0] for l in open("tags_pre_amendment.txt") if l.strip()}
out["tags_identical_to_pre_amendment"] = tags == pretags; out["n_tags"] = len(tags)
pr = json.load(open("all_releases_pre_amendment.json"))["releases"]
now = {r["id"]: r for r in api("releases?per_page=100")}
chk = {}
for r in pr:
    i = r["id"]
    if i == 417012770: continue
    c = now[i]
    chk[str(i)] = {"target_unchanged": c["target_commitish"] == r["target"],
                   "body_sha256_unchanged": hashlib.sha256(c["body"].encode()).hexdigest() == r["body_sha256_utf8"],
                   "assets_unchanged": sorted((a["id"], a["size"], a["digest"]) for a in c["assets"]) == sorted((a[0], a[2], a[3]) for a in r["assets"])}
out["other_releases"] = chk; out["n_releases_now"] = len(now)
json.dump(out, open("post_publication_verification.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("changed_REV_to_A",)}, indent=1, default=str)[:4000])
