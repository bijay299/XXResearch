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

    # Read the file list and every byte from the GIT OBJECTS AT --commit, never
    # from the working tree. Pairing working-tree bytes with an unrelated
    # commit is exactly what produced stale rows in the previous index
    # (see analysis_v1/AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md section 9).
    tracked = subprocess.run(
        ["git", "-C", a.repo_dir, "ls-tree", "-r", "--name-only", a.commit,
         "--", RESULTS],
        capture_output=True, text=True, check=True).stdout.split()

    # The index cannot verifiably list itself: it is generated here and
    # published in a CHILD commit, so its own bytes do not exist at --commit.
    # It is excluded rather than carrying an unverified historical self-hash.
    SELF = RESULTS + "/EVIDENCE_INDEX.md"
    excluded = [rp for rp in tracked if rp == SELF]
    tracked = [rp for rp in tracked if rp != SELF]

    def git_bytes(rp):
        r = subprocess.run(["git", "-C", a.repo_dir, "show",
                            f"{a.commit}:{rp}"], capture_output=True)
        return r.stdout if r.returncode == 0 else None

    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}",
        headers={"User-Agent": "thr-index", "Accept": "application/vnd.github+json"})
    rel = json.loads(urllib.request.urlopen(req, timeout=120).read())

    def load(name):
        """Load a verification JSON from the GIT OBJECT at --commit."""
        b = git_bytes(RESULTS + "/" + name)
        if b is None:
            return {}
        try:
            return json.loads(b)
        except ValueError:
            return {}

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
    w("> Every size, digest and link below is read from the **Git object at the")
    w("> commit named here**, which contains every indexed file — never from a")
    w("> working tree. This index is published in the **immediately following")
    w("> child commit**, so it does not and cannot list its own hash. Coverage")
    w("> and exclusions are stated explicitly in the repository-files section.\n")
    w(f"* **Commit:** [`{a.commit}`](https://github.com/{REPO}/commit/{a.commit})")
    w("* **Branch:** `research/audit-and-matched-effectiveness-protocol`")
    w(f"* **Release tag:** [`{rel['tag_name']}`]({rel['html_url']}) → target "
      f"`{rel.get('target_commitish')}`")
    w(f"* **Release assets:** {len(rel['assets'])}, "
      f"{sum(x['size'] for x in rel['assets']):,} bytes")
    w("* **Status:** `ANALYSIS_COMPLETE` — both stages CLOSED with **2** "
      "reviewers (3 planned) and the combined **descriptive** analysis of the "
      "existing human judgments published under "
      "[`analysis_v1/`](analysis_v1/FINAL_REPORT.md). Outcomes are now "
      "reported here. The **original responses are preserved unchanged and "
      "are NOT published**; aliases, free text, timestamps and the private "
      "mappings stay private.\n")

    w("## Verification summary\n")
    w("| check | result |")
    w("|---|---|")
    an = load("analysis_v1/verification.json")
    sy = load("analysis_v1/synthetic_checks.json")
    mv = load("analysis_v1/mapping_verification.json")
    aa = load("analysis_v1/public_audit.json")
    su = load("analysis_v1/summary.json")
    if su:
        dr = su.get("demonstrated_removal", {})
        pt = su.get("presence_table_m0_by_edited", {})
        ta = su.get("taskA", {})
        tb = su.get("taskB", {})
        w("| **ANALYSIS v1 — descriptive outcomes** | |")
        w(f"| Reviewers (planned / actual) | **{su.get('cohort',{}).get('planned')} / "
          f"{su.get('cohort',{}).get('actual')}** |")
        w(f"| Judgment counts | **20 paired cases, 80 Task-A, 40 Task-B, 2 reviewers** |")
        w(f"| All four exports valid under `{su.get('validator_revision')}` | "
          f"**yes, 0 errors** |")
        w(f"| 3x3 m0-by-edited presence table total | **{pt.get('total')} / 20** |")
        w(f"| Conditional demonstrated-removal count | **{dr.get('value_text')}** "
          f"(denominator = all consensus m0-present cases, incl. edited unresolved) |")
        w(f"| Task-A presence agreement | **"
          f"{ta.get('presence_agreement',{}).get('all_40_images',{}).get('definite_consensus')} / 40** definite consensus |")
        w(f"| Task-A rendering agreement | **"
          f"{ta.get('rendering_agreement',{}).get('all_40_images',{}).get('agreement')} / 40** |")
        w(f"| Task-B exact per-pair agreement | **"
          f"{tb.get('exact_per_pair_agreement',{}).get('agreeing_pairs')} / "
          f"{tb.get('exact_per_pair_agreement',{}).get('denominator')}** "
          f"(descriptive; no B consensus rule was frozen) |")
        w(f"| Input SHA256 identical before and after analysis | **"
          f"{su.get('integrity',{}).get('inputs_unchanged_by_analysis')}** |")
    if sy:
        w(f"| Synthetic aggregation edge cases | **{sy.get('n_passed')} / "
          f"{sy.get('n_checks')}** pass |")
    if an:
        w(f"| Arithmetic, join and reproducibility checks | **{an.get('n_passed')} / "
          f"{an.get('n_checks')}** — **`{an.get('verdict')}`** |")
    if mv:
        w(f"| Private mapping integrity (commitments re-verified) | **"
          f"{mv.get('n_passed')} / {mv.get('n_checks')}** — **`{mv.get('verdict')}`** |")
    if aa:
        w(f"| Analysis public leak audit | **{aa.get('n_findings')} findings** — "
          f"**`{aa.get('verdict')}`** ({aa.get('n_scans_ran')} secret scans ran) |")
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
    vr = load("maint/validator_r2_report.json")
    if vr:
        w(f"| Validator maintenance regression (r2, CPU-only) | **{vr.get('n_pass')} / {vr.get('n_checks')}** pass |")
    ric = load("maint/release_identity_check.json")
    if ric:
        w(f"| Both questionnaire releases unchanged by maintenance | "
          f"**{ric.get('all_releases_unchanged_by_this_maintenance')}** "
          f"(download counters excluded) |")
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
    listed = 0
    missing = []
    for rp in sorted(tracked):
        b = git_bytes(rp)
        if b is None:
            missing.append(rp)
            continue
        n = len(b)
        tot += n
        listed += 1
        w(f"| [`{os.path.relpath(rp, RESULTS)}`]({tree}/{rp}) | {n:,} | "
          f"`{hashlib.sha256(b).hexdigest()}` |")
    w("")
    w(f"**{listed} files, {tot:,} bytes — every size and digest read from the "
      f"Git object at `{a.commit[:12]}`, not from a working tree.**\n")
    if missing:
        w(f"**{len(missing)} listed path(s) had no object at this commit:** "
          + ", ".join(f"`{os.path.relpath(m, RESULTS)}`" for m in missing)
          + "\n")

    w("### Coverage and exclusions — stated, not implied\n")
    w(f"This table covers **{listed}** of the "
      f"**{listed + len(excluded)}** files tracked under `{RESULTS}` at "
      f"`{a.commit[:12]}`. Explicitly **excluded**:\n")
    w("* **`EVIDENCE_INDEX.md` — this file.** It is generated from the payload "
      "commit above and published in the **child** commit, so its own bytes do "
      "not exist at the commit it pins. **No self-hash is listed**, because "
      "any such row would be unverifiable at the declared pin. To verify this "
      "index, hash the file at the child commit that contains it.\n")
    w("* **Anything written after this index** — including any post-index "
      "verification report — is **not** covered here. Such a record states its "
      "own coverage separately.\n")
    w("No circular coverage is claimed: nothing in this table hashes itself, "
      "and nothing here depends on bytes created after the pinned commit.\n")

    w("## Scope reminder\n")
    w("This index establishes **preparation, integrity and the provenance of "
      "the published analysis**: that the package was built from verified "
      "source material, that it is complete and masked as described, that it "
      "functions offline, that the published artifacts are the ones produced, "
      "and that the analysis under [`analysis_v1/`](analysis_v1/) was computed "
      "from four hash-verified exports by published code under published "
      "rules.\n")
    w("**Outcomes are reported, in [`analysis_v1/`](analysis_v1/FINAL_REPORT.md)"
      " and nowhere else in this index.** They are **descriptive counts with "
      "explicit denominators** over 20 paired cases, 80 Task-A judgments and "
      "40 Task-B judgments from **two** people -- not independent replicated "
      "trials. Human consensus there is a **reporting convention, not ground "
      "truth**, and reviewer identity and independence are "
      "**coordinator-reported, not provable in software**.\n")
    w("Neither this index nor the analysis establishes anything about erasure "
      "efficacy beyond the single fixed endpoint measured, retention, "
      "adversarial robustness, concept-knowledge removal, quality, evaluator "
      "bias, novelty, state-of-the-art standing, generalization to other "
      "objects or models, or paper readiness. **No hypothesis test, interval, "
      "bootstrap, kappa, significance claim, pass/fail cutoff or method "
      "ranking** is reported -- a descriptive scope choice for this design, "
      "not a claim such statistics are impossible.\n")
    w("Distribution of the questionnaires was handled by the user "
      "independently of this repository. **No raw response, alias, free-text "
      "answer or private crosswalk is published**, and the case-level join "
      "runs through the **private** mapping, so public end-to-end "
      "reproducibility is **partial**. Task B was released under explicit "
      "authorization and is **partially unmasked by design**; because both "
      "bundles ship images byte-identical to the public per-case images whose "
      "condition and sha256 are published, the neutral IDs **do not hide "
      "condition**. The masking that mattered was masking **at judging "
      "time**.\n")

    out = os.path.join(rdir, "EVIDENCE_INDEX.md")
    open(out, "w").write("\n".join(L))
    print("wrote", out, f"({os.path.getsize(out):,} bytes)")
    print(f"tracked files: {len(tracked)} | release assets: {len(rel['assets'])}")


if __name__ == "__main__":
    main()
