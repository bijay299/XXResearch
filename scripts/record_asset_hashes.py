#!/usr/bin/env python3
"""Record source, revision, local path, size and SHA-256 for downloaded assets.

Run this immediately after obtaining the benchmark assets. It updates the
`download` block of each asset in `assets/asset_manifest.json` and flips its
status from `not_downloaded` to `present_unverified`.

`present_unverified` is deliberate: hashing proves only *which bytes we hold*.

    A locally computed SHA-256 identifies a file. It does NOT independently
    prove that the file is the authors' checkpoint. Only a checksum published
    by the authors, or a copy obtained directly from them, can establish that.

Provenance therefore stays `unknown` until a maintainer confirms a checksum, and
behavioural correctness stays `not_validated` until `scripts/validate_assets.py`
passes. Neither is implied by a hash.

For the generator (a directory) the hash is a *tree digest*: SHA-256 over the
sorted list of `relative_path + per-file sha256`, so it is stable and
order-independent while still covering every file.

Usage:
    python scripts/record_asset_hashes.py
    python scripts/record_asset_hashes.py --source-url <url> --revision <rev>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def sha256_tree(root: Path) -> tuple[str, int, list[dict]]:
    """Order-independent digest over every file under `root`."""
    entries: list[dict] = []
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        digest, size = sha256_file(p)
        rel = p.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": digest, "size_bytes": size})
        total += size
    outer = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e["path"]):
        outer.update(f"{e['path']}:{e['sha256']}\n".encode())
    return outer.hexdigest(), total, entries


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(repo / "assets" / "asset_manifest.json"))
    ap.add_argument("--inventory", default=str(repo / "assets" / "asset_inventory.json"),
                    help="per-file digests for the generator directory")
    ap.add_argument("--source-url", default=None,
                    help="where the asset was actually obtained from")
    ap.add_argument("--revision", default=None,
                    help="revision/commit/folder-version of that source, if any")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    inventory: dict = {"generated_utc": now, "generator_files": []}
    changed = 0

    for name, asset in manifest["assets"].items():
        target = Path(asset["expected_local_path"])
        dl = asset["download"]

        if not target.exists():
            print(f"[not present] {name}: {target}")
            continue

        if target.is_dir():
            digest, size, entries = sha256_tree(target)
            dl["sha256"] = digest
            dl["sha256_kind"] = "tree-digest(sorted relpath:filesha256)"
            inventory["generator_files"] = entries
        else:
            digest, size = sha256_file(target)
            dl["sha256"] = digest
            dl["sha256_kind"] = "file"

        dl["status"] = "present_unverified"
        dl["local_path"] = str(target)
        dl["size_bytes"] = size
        dl["retrieved_at"] = now
        if args.source_url:
            dl["source_url"] = args.source_url
        if args.revision:
            dl["revision"] = args.revision

        # Hashing never establishes provenance or behaviour.
        asset["provenance"]["status"] = "unknown"
        asset["provenance"]["note"] = (
            "SHA-256 recorded locally; identifies bytes only. Equivalence to the authors' "
            "checkpoint remains unconfirmed pending a published checksum or direct copy.")
        asset.setdefault("validation", {})
        if asset["validation"].get("status") != "verified":
            asset["validation"]["status"] = "not_validated"

        changed += 1
        print(f"[recorded] {name}")
        print(f"             path   {target}")
        print(f"             size   {size:,} bytes")
        print(f"             sha256 {digest}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    Path(args.inventory).write_text(json.dumps(inventory, indent=2))
    print(f"\nupdated {manifest_path} ({changed} asset(s))")
    print(f"wrote   {args.inventory}")
    if changed:
        print("\nNOTE: status is 'present_unverified'. Run scripts/validate_assets.py "
              "before any scientific result is produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
