#!/usr/bin/env python3
"""Package the release assets: lossless <=1 GiB parts with full hash coverage.

GitHub caps each release asset at 2 GiB
(https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
and the three native safetensors shards are ~10 GiB each, so the native
transformer is published as an uncompressed tar split into 1 GiB parts. Nothing
is quantized or cast to fit hosting.

Produces, under publish/assets/:
  gem-flux-teapot-v1-transformer.tar.partNNN   (<= 1 GiB each)
  gem-flux-teapot-v1-lora-adapter.tar.gz
  gem-flux-teapot-v1-images.tar.gz
  gem-flux-teapot-v1-evidence.tar.gz
and publish/ASSET_MANIFEST.json with per-part and reconstructed-file hashes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moa_common as C  # noqa: E402

PART = 1024 ** 3  # 1 GiB


def sha256_stream(path, chunk=1 << 22):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def make_tar(src_dir, arcname, out_path, gzip=False):
    mode = "w:gz" if gzip else "w"
    with tarfile.open(out_path, mode) as tf:
        tf.add(src_dir, arcname=arcname, recursive=True)
    return out_path


def split_file(path, out_dir, base_name, part_size=PART):
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    with open(path, "rb") as fh:
        idx = 0
        while True:
            buf = fh.read(part_size)
            if not buf:
                break
            pname = f"{base_name}.part{idx:03d}"
            ppath = os.path.join(out_dir, pname)
            with open(ppath, "wb") as out:
                out.write(buf)
            h = hashlib.sha256(buf).hexdigest()
            parts.append({"name": pname, "bytes": len(buf), "sha256": h})
            print(f"  {pname}  {len(buf)/PART:.3f} GiB  {h[:16]}...")
            idx += 1
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training run dir (contains transformer/)")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "publish"))
    a = ap.parse_args()

    assets = os.path.join(a.out, "assets")
    staging = os.path.join(a.out, "staging")
    os.makedirs(assets, exist_ok=True)
    os.makedirs(staging, exist_ok=True)

    manifest = {"release_tag": "modern-object-assets-v1",
                "part_size_bytes": PART,
                "github_asset_limit_bytes": 2 * 1024 ** 3,
                "lossless": True,
                "quantized_or_cast_to_fit_hosting": False,
                "assets": []}

    # ---------- 1. native merged transformer (the author's own representation) ----------
    tdir = os.path.join(a.ckpt, "transformer")
    assert os.path.isdir(tdir), tdir
    files = sorted(os.listdir(tdir))
    print(f"[pack] transformer/ contains: {files}")
    tar_path = os.path.join(staging, "gem-flux-teapot-v1-transformer.tar")
    print("[pack] building uncompressed tar of transformer/ ...")
    make_tar(tdir, "transformer", tar_path, gzip=False)
    tar_sha, tar_bytes = sha256_stream(tar_path)
    print(f"[pack] tar: {tar_bytes/PART:.2f} GiB sha256={tar_sha}")
    print("[pack] splitting into 1 GiB parts ...")
    parts = split_file(tar_path, assets, "gem-flux-teapot-v1-transformer.tar")
    manifest["assets"].append({
        "logical_name": "native merged transformer (edited endpoint)",
        "reconstructed_file": "gem-flux-teapot-v1-transformer.tar",
        "reconstructed_bytes": tar_bytes,
        "reconstructed_sha256": tar_sha,
        "inner_files": files,
        "n_parts": len(parts),
        "parts": parts,
        "reconstruct": [
            "cat gem-flux-teapot-v1-transformer.tar.part* > gem-flux-teapot-v1-transformer.tar",
            f"sha256sum gem-flux-teapot-v1-transformer.tar   # expect {tar_sha}",
            "tar -xf gem-flux-teapot-v1-transformer.tar      # yields transformer/",
        ],
    })
    os.remove(tar_path)

    # ---------- 2. pre-merge native LoRA adapter (convenience) ----------
    ldir = os.path.join(a.ckpt, "lora_adapter")
    if os.path.isdir(ldir):
        lp = os.path.join(assets, "gem-flux-teapot-v1-lora-adapter.tar.gz")
        make_tar(ldir, "lora_adapter", lp, gzip=True)
        h, n = sha256_stream(lp)
        manifest["assets"].append({
            "logical_name": "pre-merge native PEFT LoRA adapter (convenience artifact)",
            "reconstructed_file": os.path.basename(lp),
            "reconstructed_bytes": n, "reconstructed_sha256": h, "n_parts": 1,
            "parts": [{"name": os.path.basename(lp), "bytes": n, "sha256": h}],
            "note": ("Published alongside, NOT instead of, the native merged "
                     "transformer. No claim is made that the unmerged adapter "
                     "and the merged weights are bit-identical; see "
                     "adapter_equivalence.json for the measured result."),
            "reconstruct": [f"tar -xzf {os.path.basename(lp)}"],
        })
        print(f"[pack] lora adapter: {n/1e6:.1f} MB sha256={h[:16]}...")

    # ---------- 3. the 40 primary PNGs ----------
    ip = os.path.join(assets, "gem-flux-teapot-v1-images.tar.gz")
    make_tar(a.images_dir, "images", ip, gzip=True)
    h, n = sha256_stream(ip)
    npng = sum(1 for _, _, fs in os.walk(a.images_dir) for f in fs if f.endswith(".png"))
    manifest["assets"].append({
        "logical_name": f"primary paired PNGs at original 512x512 resolution ({npng} files)",
        "reconstructed_file": os.path.basename(ip),
        "reconstructed_bytes": n, "reconstructed_sha256": h, "n_parts": 1,
        "n_png": npng,
        "parts": [{"name": os.path.basename(ip), "bytes": n, "sha256": h}],
        "reconstruct": [f"tar -xzf {os.path.basename(ip)}"],
    })
    print(f"[pack] images: {npng} PNGs, {n/1e6:.1f} MB sha256={h[:16]}...")

    # ---------- 4. evidence bundle ----------
    ev_stage = os.path.join(staging, "evidence")
    os.makedirs(ev_stage, exist_ok=True)
    for sub in ("manifests", "configs", "patches", "ledger"):
        src = os.path.join(C.ROOT, sub)
        if os.path.isdir(src):
            subprocess.run(["cp", "-r", src, ev_stage], check=True)
    for extra in (os.path.join(a.ckpt, "evidence"),):
        if os.path.isdir(extra):
            subprocess.run(["cp", "-r", extra, os.path.join(ev_stage, "run_evidence")],
                           check=True)
    sm = os.path.join(C.ROOT, "smoke", "run_smoke_01")
    if os.path.isdir(sm):
        subprocess.run(["cp", "-r", sm, os.path.join(ev_stage, "smoke_run")], check=True)
    ep = os.path.join(assets, "gem-flux-teapot-v1-evidence.tar.gz")
    make_tar(ev_stage, "evidence", ep, gzip=True)
    h, n = sha256_stream(ep)
    manifest["assets"].append({
        "logical_name": "evidence bundle (configs, manifests, digests, patches, ledger, logs)",
        "reconstructed_file": os.path.basename(ep),
        "reconstructed_bytes": n, "reconstructed_sha256": h, "n_parts": 1,
        "parts": [{"name": os.path.basename(ep), "bytes": n, "sha256": h}],
        "reconstruct": [f"tar -xzf {os.path.basename(ep)}"],
    })
    print(f"[pack] evidence: {n/1e6:.1f} MB sha256={h[:16]}...")

    # ---------- totals / limit checks ----------
    all_parts = [p for asset in manifest["assets"] for p in asset["parts"]]
    oversize = [p for p in all_parts if p["bytes"] > 2 * 1024 ** 3]
    manifest["total_assets_uploaded"] = len(all_parts)
    manifest["total_bytes_uploaded"] = sum(p["bytes"] for p in all_parts)
    manifest["all_under_github_2gib_limit"] = not oversize
    manifest["oversize_parts"] = oversize
    C.dump_json(manifest, os.path.join(a.out, "ASSET_MANIFEST.json"))

    print(f"\n[pack] {len(all_parts)} assets, "
          f"{manifest['total_bytes_uploaded']/PART:.2f} GiB total, "
          f"all under 2 GiB: {manifest['all_under_github_2gib_limit']}")
    subprocess.run(["rm", "-rf", staging], check=False)


if __name__ == "__main__":
    main()
