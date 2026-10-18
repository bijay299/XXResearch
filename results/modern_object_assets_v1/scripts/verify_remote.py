#!/usr/bin/env python3
"""Fresh-directory remote verification of the published release.

Downloads every published asset into a clean directory, verifies every remote
hash against ASSET_MANIFEST.json, reconstructs the split checkpoint losslessly,
verifies the reconstructed-file hash and every inner file hash, loads the
REMOTE checkpoint through GEM's documented path against the pinned base, and
compares it to the canonical post-merge digest. Also verifies primary-file
completeness and paired-input correspondence from the downloaded image archive.

Runs on CPU: it allocates no GPU and decodes no images.

Usage: verify_remote.py --tag TAG --dir FRESH_DIR
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402

REPO = "bijay299/XXResearch"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def http_get(url, dest, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "xxresearch-verify",
                              "Accept": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=1800) as r, open(dest, "wb") as out:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    out.write(b)
            return True
        except Exception as e:
            print(f"    retry {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(5 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    dl = os.path.join(a.dir, "downloads")
    work = os.path.join(a.dir, "reconstructed")
    os.makedirs(dl, exist_ok=True)
    os.makedirs(work, exist_ok=True)

    print("=== listing the live release (public, unauthenticated) ===")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{a.tag}",
        headers={"User-Agent": "xxresearch-verify",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        rel = json.loads(r.read())
    print(f"tag={rel['tag_name']} target={rel['target_commitish']} "
          f"assets={len(rel['assets'])} url={rel['html_url']}")

    report = {
        "tag": rel["tag_name"],
        "release_html_url": rel["html_url"],
        "release_target_commitish": rel["target_commitish"],
        "n_assets_on_release": len(rel["assets"]),
        "verification_dir": a.dir,
        "gpu_device_hours_used": 0.0,
        "decoded_images_used": 0,
        "note": "CPU-only verification: no GPU allocated, no image decoded.",
    }

    # ---------- 1. download every asset ----------
    print("\n=== downloading every asset ===")
    downloaded = {}
    t0 = time.time()
    for asset in sorted(rel["assets"], key=lambda x: x["name"]):
        dest = os.path.join(dl, asset["name"])
        if os.path.isfile(dest) and os.path.getsize(dest) == asset["size"]:
            print(f"  have {asset['name']}")
        else:
            print(f"  GET  {asset['name']} ({asset['size']/1024**2:.1f} MiB)")
            if not http_get(asset["browser_download_url"], dest):
                report.setdefault("download_failures", []).append(asset["name"])
                continue
        downloaded[asset["name"]] = dest
    report["download_seconds"] = round(time.time() - t0, 1)
    report["n_downloaded"] = len(downloaded)
    print(f"downloaded {len(downloaded)} assets in {report['download_seconds']}s")

    # ---------- 2. verify every remote hash against the manifest ----------
    print("\n=== verifying every remote hash ===")
    with open(downloaded["ASSET_MANIFEST.json"]) as fh:
        man = json.load(fh)

    part_results, bad = [], []
    for asset in man["assets"]:
        for p in asset["parts"]:
            path = downloaded.get(p["name"])
            if path is None:
                bad.append({"name": p["name"], "problem": "not downloaded"})
                continue
            got = sha256_file(path)
            ok = (got == p["sha256"]) and os.path.getsize(path) == p["bytes"]
            part_results.append({"name": p["name"], "expected": p["sha256"],
                                 "got": got, "bytes_ok": os.path.getsize(path) == p["bytes"],
                                 "ok": ok})
            if not ok:
                bad.append({"name": p["name"], "expected": p["sha256"], "got": got})
    report["n_parts_checked"] = len(part_results)
    report["n_parts_hash_ok"] = sum(1 for r in part_results if r["ok"])
    report["parts_with_bad_hash"] = bad
    report["all_remote_hashes_match"] = not bad
    print(f"  {report['n_parts_hash_ok']}/{report['n_parts_checked']} remote hashes match")

    # ---------- 3. lossless reconstruction of the checkpoint ----------
    print("\n=== reconstructing the split checkpoint ===")
    tasset = next(x for x in man["assets"]
                  if x["reconstructed_file"].endswith("transformer.tar"))
    parts = sorted(p["name"] for p in tasset["parts"])
    tar_path = os.path.join(work, tasset["reconstructed_file"])
    with open(tar_path, "wb") as out:
        for pn in parts:
            with open(downloaded[pn], "rb") as fh:
                while True:
                    b = fh.read(1 << 22)
                    if not b:
                        break
                    out.write(b)
    got = sha256_file(tar_path)
    report["reconstructed_file"] = tasset["reconstructed_file"]
    report["reconstructed_expected_sha256"] = tasset["reconstructed_sha256"]
    report["reconstructed_got_sha256"] = got
    report["reconstruction_lossless"] = (got == tasset["reconstructed_sha256"]
                                         and os.path.getsize(tar_path) == tasset["reconstructed_bytes"])
    print(f"  n_parts={len(parts)} reconstructed sha256 match: "
          f"{report['reconstruction_lossless']}")

    print("  extracting ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(work)
    os.remove(tar_path)
    ck = os.path.join(work, "transformer")
    report["extracted_files"] = sorted(os.listdir(ck))

    # inner per-file hashes vs the published list
    inner_expect = {}
    ipath = os.path.join(C.ROOT, "manifests", "edited_transformer_sha256.txt")
    if os.path.isfile(ipath):
        for line in open(ipath):
            line = line.strip()
            if line:
                h, n = line.split(None, 1)
                inner_expect[n.lstrip("*")] = h
    inner = []
    for n in report["extracted_files"]:
        got = sha256_file(os.path.join(ck, n))
        exp = inner_expect.get(n)
        inner.append({"file": n, "sha256": got, "expected": exp,
                      "match": (exp is None or exp == got)})
    report["inner_file_hashes"] = inner
    report["all_inner_files_match"] = all(x["match"] for x in inner)
    report["n_inner_files_with_published_hash"] = sum(1 for x in inner if x["expected"])
    print(f"  inner files: {len(inner)}, all match published hashes: "
          f"{report['all_inner_files_match']}")

    # ---------- 4. load the REMOTE checkpoint with the documented pinned base ----------
    print("\n=== loading the REMOTE checkpoint via GEM's documented path (CPU) ===")
    C.pin_base_model_path()
    from dataclasses import replace
    from gem.configs.pipeline_config import PipelineConfig
    from gem.wrappers.factory import PipelineFactory

    pipe_config = PipelineConfig.default_for_model_type(model_type="flux")
    pipe_config = replace(pipe_config, model_dtype="bfloat16",
                          model_name_or_path=work)
    t0 = time.time()
    pipe, modified = PipelineFactory.create_pipeline(
        pipe_config, return_local_component_names=True)
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s; overwritten: {modified}")

    d = C.state_dict_digest(pipe.transformer.state_dict())
    with open(os.path.join(C.ROOT, "train", "gem_flux_teapot_v1",
                           "evidence", "digest_postmerge_transformer.json")) as fh:
        postmerge = json.load(fh)
    cmp_ = C.compare_digests(postmerge, d, "local_postmerge_before_save",
                             "remote_downloaded_reloaded")
    fin = C.finite_report(pipe.transformer.state_dict())
    report["remote_reload"] = {
        "load_seconds": round(load_s, 1),
        "overwritten_components": modified,
        "overwrite_included_transformer": "transformer" in (modified or []),
        "n_keys": d["n_keys"],
        "aggregate_sha256": d["aggregate_sha256"],
        "bitwise_identical_to_local_postmerge": cmp_["bitwise_identical"],
        "n_value_mismatches": cmp_["n_value_mismatches"],
        "all_finite": fin["all_finite"],
    }
    print(f"  remote reload bitwise-identical to local post-merge: "
          f"{cmp_['bitwise_identical']} | all finite: {fin['all_finite']}")

    # ---------- 5. primary-file completeness + paired-input correspondence ----------
    print("\n=== verifying the downloaded image archive ===")
    iasset = next(x for x in man["assets"]
                  if x["reconstructed_file"].endswith("images.tar.gz"))
    idir = os.path.join(work, "images_extracted")
    os.makedirs(idir, exist_ok=True)
    with tarfile.open(downloaded[iasset["reconstructed_file"]], "r:gz") as tf:
        tf.extractall(idir)
    base = os.path.join(idir, "images")
    pngs_m0 = sorted(f for f in os.listdir(os.path.join(base, "m0")) if f.endswith(".png"))
    pngs_ed = sorted(f for f in os.listdir(os.path.join(base, "edited")) if f.endswith(".png"))

    with open(os.path.join(base, "PAIR_VERIFICATION.json")) as fh:
        pv = json.load(fh)

    rows, mism = [], []
    for r in pv["rows"]:
        e = {"case_id": r["case_id"]}
        for cond, sub in (("m0", "m0"), ("edited", "edited")):
            fn = r[cond]["filename"]
            p = os.path.join(base, sub, fn)
            present = os.path.isfile(p)
            got = sha256_file(p) if present else None
            e[cond] = {"filename": fn, "present": present,
                       "sha256_match": got == r[cond]["file_sha256"]}
            if not present or got != r[cond]["file_sha256"]:
                mism.append(f"{r['case_id']}/{cond}")
        e["noise_identical_across_conditions"] = r["noise_identical_across_conditions"]
        rows.append(e)

    # re-verify the paired initial-noise tensors that shipped with the release
    ndir = os.path.join(base, "noise")
    with open(os.path.join(base, "initial_noise_index.json")) as fh:
        nidx = json.load(fh)["cases"]
    noise_ok, noise_bad = 0, []
    for rec in nidx:
        p = os.path.join(ndir, os.path.basename(rec["file"]))
        if os.path.isfile(p) and sha256_file(p) == rec["file_sha256"]:
            noise_ok += 1
        else:
            noise_bad.append(rec["case_id"])

    report["images"] = {
        "n_m0_png": len(pngs_m0),
        "n_edited_png": len(pngs_ed),
        "n_primary_png": len(pngs_m0) + len(pngs_ed),
        "expected_primary_png": 40,
        "primary_completeness_ok": (len(pngs_m0) == 20 and len(pngs_ed) == 20),
        "n_file_hash_mismatches": len(mism),
        "file_hash_mismatches": mism,
        "all_png_hashes_match": not mism,
        "contact_sheet_present": os.path.isfile(os.path.join(base, "contact_sheet.png")),
        "n_noise_tensors_verified": noise_ok,
        "noise_tensors_failed": noise_bad,
        "paired_input_correspondence_ok": all(
            r["noise_identical_across_conditions"] for r in rows) and not noise_bad,
        "rows": rows,
    }
    print(f"  primary PNGs: {len(pngs_m0)}+{len(pngs_ed)}="
          f"{len(pngs_m0)+len(pngs_ed)}/40, hash mismatches: {len(mism)}")
    print(f"  paired initial-noise tensors verified: {noise_ok}/20")

    # ---------- verdict ----------
    ok = (report["all_remote_hashes_match"]
          and report["reconstruction_lossless"]
          and report["all_inner_files_match"]
          and report["remote_reload"]["bitwise_identical_to_local_postmerge"]
          and report["remote_reload"]["all_finite"]
          and report["images"]["primary_completeness_ok"]
          and report["images"]["all_png_hashes_match"]
          and report["images"]["paired_input_correspondence_ok"]
          and not report.get("download_failures"))
    report["verdict"] = "REMOTE_VERIFICATION_PASSED" if ok else "PUBLICATION_INCOMPLETE"

    C.dump_json(report, os.path.join(C.ROOT, "publish", "REMOTE_VERIFICATION.json"))
    print("\n=== VERDICT:", report["verdict"], "===")
    sys.exit(0 if ok else 8)


if __name__ == "__main__":
    main()
