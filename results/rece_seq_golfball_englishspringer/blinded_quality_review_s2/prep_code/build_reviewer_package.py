#!/usr/bin/env python3
"""Builds the blinded human review package for the 20 matched S2 springer
pairs (case IDs 60-79, S2_STAGE1_HISTORY vs. S2_HISTORY_BOTH_STAGES).

No GPU, no model computation, no image regeneration -- this only reads
already-published, already-verified original PNGs and copies their exact
bytes to neutral filenames. The pixels are never touched.

IMPORTANT, for anyone re-reading this published script: this script
generates its own random seed at runtime (via `secrets.randbits`) and
writes it ONLY to the private mapping file under review_private/, never
to stdout, never into any published artifact, and never hard-coded here.
Re-running this script produces a DIFFERENT randomization each time --
it does not reproduce any specific run's blinding.
"""
import hashlib
import json
import random
import secrets
import shutil
import time
from pathlib import Path

HOST_ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
PRIVATE_DIR = Path("/data/bijaypandey/cuig_pilot/review_private/blinded_quality_review_s2")
PACKAGE_IMAGES_DIR = HOST_ROOT / "blinded_quality_review_s2/reviewer_package/images"

SOURCE_VERIFICATION = HOST_ROOT / "blinded_quality_review_s2_SOURCE_VERIFICATION.json"

ENDPOINTS = ["S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    verification = json.loads(SOURCE_VERIFICATION.read_text())
    assert verification["all_checks_passed"], "source verification did not pass -- refusing to build the package"
    per_case = verification["per_case"]
    case_ids = sorted(int(k) for k in per_case.keys())
    assert case_ids == list(range(60, 80)), "expected exactly case IDs 60-79"

    # runtime-only seed, never hard-coded, never printed
    seed = secrets.randbits(64)
    rng = random.Random(seed)

    # 1. randomize which Q-slot each case_id gets (pair-position randomization)
    shuffled_case_ids = case_ids[:]
    rng.shuffle(shuffled_case_ids)
    neutral_ids = [f"Q{i+1:02d}" for i in range(20)]
    case_to_neutral = dict(zip(shuffled_case_ids, neutral_ids))

    # 2. randomize A/B assignment per pair, balanced 10/10
    ab_assignment = (["S2_STAGE1_HISTORY_is_A"] * 10) + (["S2_HISTORY_BOTH_STAGES_is_A"] * 10)
    rng.shuffle(ab_assignment)

    PACKAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in PACKAGE_IMAGES_DIR.glob("*.png"):
        f.unlink()

    pairs = []
    count_a = {"S2_STAGE1_HISTORY": 0, "S2_HISTORY_BOTH_STAGES": 0}
    for case_id in case_ids:
        qid = case_to_neutral[case_id]
        idx = shuffled_case_ids.index(case_id)
        a_rule = ab_assignment[idx]
        if a_rule == "S2_STAGE1_HISTORY_is_A":
            a_ep, b_ep = "S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES"
        else:
            a_ep, b_ep = "S2_HISTORY_BOTH_STAGES", "S2_STAGE1_HISTORY"
        count_a[a_ep] += 1

        a_src = Path(per_case[str(case_id)][a_ep]["path"])
        b_src = Path(per_case[str(case_id)][b_ep]["path"])
        a_dst = PACKAGE_IMAGES_DIR / f"{qid}_A.png"
        b_dst = PACKAGE_IMAGES_DIR / f"{qid}_B.png"

        # exact byte copy -- no re-encoding, no PIL re-save, no alteration
        shutil.copyfile(a_src, a_dst)
        shutil.copyfile(b_src, b_dst)

        a_bytes_sha = sha256_file(a_dst)
        b_bytes_sha = sha256_file(b_dst)
        assert a_bytes_sha == per_case[str(case_id)][a_ep]["sha256"], f"{qid} A: byte copy hash mismatch"
        assert b_bytes_sha == per_case[str(case_id)][b_ep]["sha256"], f"{qid} B: byte copy hash mismatch"

        pairs.append({
            "neutral_id": qid, "case_id": case_id,
            "A_endpoint": a_ep, "A_source_path": str(a_src), "A_sha256": a_bytes_sha,
            "A_package_filename": f"{qid}_A.png",
            "B_endpoint": b_ep, "B_source_path": str(b_src), "B_sha256": b_bytes_sha,
            "B_package_filename": f"{qid}_B.png",
        })

    assert count_a["S2_STAGE1_HISTORY"] == 10 and count_a["S2_HISTORY_BOTH_STAGES"] == 10, \
        f"A-position balance failed: {count_a}"

    pairs.sort(key=lambda p: p["neutral_id"])

    private_mapping = {
        "purpose": "PRIVATE mapping for blinded_quality_review_s2 -- neutral ID to case ID/endpoint/source/hash. "
                   "Never publish this file or its random_seed. Kept outside the git checkout.",
        "generated_at_unix": time.time(),
        "random_seed": seed,
        "randomization_method": "python random.Random(seed).shuffle() -- Fisher-Yates -- applied once to the "
                                 "20 case IDs for Q-slot assignment, and once to a 10x/10x balanced A-position "
                                 "label list, both seeded from the same runtime-generated secrets.randbits(64) seed",
        "n_pairs": len(pairs),
        "a_position_balance": count_a,
        "pairs": pairs,
    }

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = PRIVATE_DIR / "PRIVATE_MAPPING.json"
    mapping_path.write_text(json.dumps(private_mapping, indent=1))
    mapping_path.chmod(0o600)

    mapping_sha256 = sha256_file(mapping_path)

    commitment = {
        "purpose": "Public SHA256 commitment to the private mapping file, published BEFORE collecting responses. "
                   "Lets anyone later verify the mapping/seed revealed after unblinding was not altered after "
                   "the fact. Does NOT reveal the mapping or seed itself.",
        "private_mapping_sha256": mapping_sha256,
        "private_mapping_file_name": "PRIVATE_MAPPING.json",
        "committed_at_unix": time.time(),
        "n_pairs_committed": len(pairs),
    }
    commitment_path = HOST_ROOT / "blinded_quality_review_s2/MAPPING_COMMITMENT.json"
    commitment_path.write_text(json.dumps(commitment, indent=1))

    # print only non-sensitive confirmation -- no seed, no mapping, no case IDs
    print(f"Package built: {len(pairs)} pairs, {len(pairs)*2} neutral images written to {PACKAGE_IMAGES_DIR}")
    print(f"A-position balance: {count_a}")
    print(f"Private mapping written to {mapping_path} (mode 0600, NOT published)")
    print(f"Public commitment hash: {mapping_sha256}")


if __name__ == "__main__":
    main()
