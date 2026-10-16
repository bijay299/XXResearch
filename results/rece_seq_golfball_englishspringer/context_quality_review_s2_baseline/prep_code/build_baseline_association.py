#!/usr/bin/env python3
"""Builds the PRIVATE baseline-association manifest: for each existing
R01-R20 neutral ID (from the v2 private mapping, read-only, NEVER
modified or regenerated here), associates the same case's pristine M0
reference image. Also copies the 40 existing edited A/B images AND the
20 new M0 reference images into the new baseline package's image
directory -- exact byte copies only, no re-encoding.

This does NOT rerandomize anything: pair order, A/B assignment, and
the v2 mapping file are read-only inputs. The v2 mapping's own
commitment is independently re-verified unchanged before use.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

HOST_ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
PUB_ROOT = Path("/data/bijaypandey/xxresearch_standalone/results/rece_seq_golfball_englishspringer")

V2_PRIVATE_MAPPING = Path("/data/bijaypandey/cuig_pilot/review_private/context_quality_review_s2_v2/PRIVATE_MAPPING_V2.json")
V2_MAPPING_REQUIRED_SHA256 = "bd360bc35c45d5eb68f5ff6e91aedb1db30cd70e10a9b55023bb2d330fbbc529"

BASELINE_SOURCE_VERIFICATION = HOST_ROOT / "context_quality_review_s2_baseline_SOURCE_VERIFICATION.json"
V2_SOURCE_VERIFICATION = HOST_ROOT / "context_quality_review_s2_v2_SOURCE_VERIFICATION.json"

PRIVATE_DIR = Path("/data/bijaypandey/cuig_pilot/review_private/context_quality_review_s2_baseline")
PACKAGE_IMAGES_DIR = HOST_ROOT / "context_quality_review_s2_baseline/reviewer_package/images"

V2_EDITED_IMAGES_DIR = HOST_ROOT / "context_quality_review_s2_v2/reviewer_package/images"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. verify the v2 mapping is unchanged before reading it
    actual_sha = sha256_file(V2_PRIVATE_MAPPING)
    assert actual_sha == V2_MAPPING_REQUIRED_SHA256, \
        f"STOP: v2 private mapping hash changed! expected {V2_MAPPING_REQUIRED_SHA256}, got {actual_sha}"
    v2_mapping = json.loads(V2_PRIVATE_MAPPING.read_text())

    baseline_verification = json.loads(BASELINE_SOURCE_VERIFICATION.read_text())
    assert baseline_verification["all_checks_passed"]
    ref_per_case = baseline_verification["per_case"]

    v2_verification = json.loads(V2_SOURCE_VERIFICATION.read_text())
    assert v2_verification["all_checks_passed"]

    PACKAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in PACKAGE_IMAGES_DIR.glob("*.png"):
        f.unlink()

    associations = []
    for pair in v2_mapping["pairs"]:
        rid = pair["neutral_id"]
        case_id = pair["case_id"]

        # copy the two existing edited images, byte-exact, same filenames as v2
        for side in ("A", "B"):
            src = V2_EDITED_IMAGES_DIR / f"{rid}_{side}.png"
            dst = PACKAGE_IMAGES_DIR / f"{rid}_{side}.png"
            shutil.copyfile(src, dst)
            dst_sha = sha256_file(dst)
            assert dst_sha == pair[f"{side}_sha256"], f"{rid}_{side}: byte copy hash mismatch"

        # copy the new M0 reference image
        ref = ref_per_case[str(case_id)]
        ref_src = Path(ref["published_path"])
        ref_dst = PACKAGE_IMAGES_DIR / f"{rid}_REFERENCE.png"
        shutil.copyfile(ref_src, ref_dst)
        ref_dst_sha = sha256_file(ref_dst)
        assert ref_dst_sha == ref["sha256"], f"{rid}_REFERENCE: byte copy hash mismatch"

        associations.append({
            "neutral_id": rid, "case_id": case_id,
            "reference_source_path": str(ref_src), "reference_filename": ref["filename"],
            "reference_sha256": ref_dst_sha, "reference_package_filename": f"{rid}_REFERENCE.png",
            "reference_prompt": ref["prompt"], "reference_seed": ref["seed"],
            # carried through for completeness, NOT re-derived or altered
            "A_endpoint": pair["A_endpoint"], "B_endpoint": pair["B_endpoint"],
            "A_sha256": pair["A_sha256"], "B_sha256": pair["B_sha256"],
            "shared_prompt": pair["shared_prompt"],
        })

    associations.sort(key=lambda a: a["neutral_id"])
    assert len(associations) == 20
    assert len(set(a["case_id"] for a in associations)) == 20

    private_assoc = {
        "purpose": "PRIVATE baseline-association manifest for context_quality_review_s2_baseline -- "
                   "associates each existing v2 R01-R20 neutral ID with its same-case M0 reference image. "
                   "Derived READ-ONLY from the v2 private mapping (never modified, never rerandomized). "
                   "Never publish this file. Kept outside the git checkout.",
        "derived_from_v2_mapping_sha256": actual_sha,
        "generated_at_unix": time.time(),
        "n_associations": len(associations),
        "associations": associations,
    }
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    assoc_path = PRIVATE_DIR / "BASELINE_ASSOCIATION_PRIVATE.json"
    assoc_path.write_text(json.dumps(private_assoc, indent=1))
    assoc_path.chmod(0o600)
    assoc_sha256 = sha256_file(assoc_path)

    # re-verify v2 mapping still unchanged after all this (paranoia check)
    actual_sha_after = sha256_file(V2_PRIVATE_MAPPING)
    assert actual_sha_after == V2_MAPPING_REQUIRED_SHA256, "STOP: v2 private mapping changed during this run!"

    commitment = {
        "purpose": "Public SHA256 commitment to the PRIVATE baseline-association manifest, published BEFORE "
                   "collecting any baseline-package response. Does NOT reveal the manifest, case IDs, or "
                   "endpoint identities.",
        "package_version": "context_quality_review_s2_baseline-v1",
        "baseline_association_sha256": assoc_sha256,
        "baseline_association_file_name": "BASELINE_ASSOCIATION_PRIVATE.json",
        "derived_from_v2_mapping_sha256_unchanged": actual_sha_after == V2_MAPPING_REQUIRED_SHA256,
        "v2_mapping_commitment_reused": V2_MAPPING_REQUIRED_SHA256,
        "committed_at_unix": time.time(),
        "n_associations_committed": len(associations),
    }
    commitment_path = HOST_ROOT / "context_quality_review_s2_baseline/BASELINE_ASSOCIATION_COMMITMENT.json"
    commitment_path.write_text(json.dumps(commitment, indent=1))

    print(f"Baseline package images built: {len(associations)} groups x 3 images = {len(associations)*3} files "
          f"written to {PACKAGE_IMAGES_DIR}")
    print(f"Private baseline-association manifest written to {assoc_path} (mode 0600, NOT published)")
    print(f"Public commitment hash: {assoc_sha256}")
    print(f"v2 mapping confirmed UNCHANGED before and after: {actual_sha == V2_MAPPING_REQUIRED_SHA256 == actual_sha_after}")


if __name__ == "__main__":
    main()
