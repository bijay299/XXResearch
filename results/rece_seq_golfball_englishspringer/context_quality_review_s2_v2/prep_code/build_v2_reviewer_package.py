#!/usr/bin/env python3
"""Builds the v2 (context-aware) blinded human review package for the
same 20 matched S2 springer pairs used in v1 (case IDs 60-79,
S2_STAGE1_HISTORY vs. S2_HISTORY_BOTH_STAGES). This is a SEPARATE
package from v1: new neutral IDs (R01-R20, not Q01-Q20), a fresh
independent randomization (new seed, new A/B assignment), and a
SEPARATE private mapping file under its own review_private subdirectory
-- it does not read, touch, or overwrite v1's PRIVATE_MAPPING.json.

No GPU, no model computation, no image regeneration -- exact byte
copies of the same already-verified source images used in v1. The only
new content is the exact shared generation prompt per case (taken
verbatim from the frozen panel manifest) and the new context-aware
protocol fields collected by the v2 viewer.

Published preparation code: generates its own fresh random seed at
runtime (secrets.randbits), never hard-coded, never printed, never
written anywhere but the private mapping file.
"""
import hashlib
import json
import random
import secrets
import shutil
import time
from pathlib import Path

HOST_ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
PRIVATE_DIR = Path("/data/bijaypandey/cuig_pilot/review_private/context_quality_review_s2_v2")
PACKAGE_IMAGES_DIR = HOST_ROOT / "context_quality_review_s2_v2/reviewer_package/images"
SOURCE_VERIFICATION = HOST_ROOT / "context_quality_review_s2_v2_SOURCE_VERIFICATION.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    verification = json.loads(SOURCE_VERIFICATION.read_text())
    assert verification["all_checks_passed"], "source verification did not pass -- refusing to build"
    per_case = verification["per_case"]
    case_prompts = verification["case_prompts_from_manifest"]
    case_ids = sorted(int(k) for k in per_case.keys())
    assert case_ids == list(range(60, 80))

    # SAFETY: never write into v1's private mapping location
    v1_private = Path("/data/bijaypandey/cuig_pilot/review_private/blinded_quality_review_s2/PRIVATE_MAPPING.json")
    v1_sha_before = sha256_file(v1_private) if v1_private.is_file() else None

    seed = secrets.randbits(64)
    rng = random.Random(seed)

    shuffled_case_ids = case_ids[:]
    rng.shuffle(shuffled_case_ids)
    neutral_ids = [f"R{i+1:02d}" for i in range(20)]  # R-prefix: distinct from v1's Q-prefix
    case_to_neutral = dict(zip(shuffled_case_ids, neutral_ids))

    ab_assignment = (["S2_STAGE1_HISTORY_is_A"] * 10) + (["S2_HISTORY_BOTH_STAGES_is_A"] * 10)
    rng.shuffle(ab_assignment)

    PACKAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in PACKAGE_IMAGES_DIR.glob("*.png"):
        f.unlink()

    pairs = []
    count_a = {"S2_STAGE1_HISTORY": 0, "S2_HISTORY_BOTH_STAGES": 0}
    for case_id in case_ids:
        rid = case_to_neutral[case_id]
        idx = shuffled_case_ids.index(case_id)
        a_rule = ab_assignment[idx]
        if a_rule == "S2_STAGE1_HISTORY_is_A":
            a_ep, b_ep = "S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES"
        else:
            a_ep, b_ep = "S2_HISTORY_BOTH_STAGES", "S2_STAGE1_HISTORY"
        count_a[a_ep] += 1

        a_src = Path(per_case[str(case_id)][a_ep]["path"])
        b_src = Path(per_case[str(case_id)][b_ep]["path"])
        a_dst = PACKAGE_IMAGES_DIR / f"{rid}_A.png"
        b_dst = PACKAGE_IMAGES_DIR / f"{rid}_B.png"

        shutil.copyfile(a_src, a_dst)
        shutil.copyfile(b_src, b_dst)

        a_sha = sha256_file(a_dst)
        b_sha = sha256_file(b_dst)
        assert a_sha == per_case[str(case_id)][a_ep]["sha256"]
        assert b_sha == per_case[str(case_id)][b_ep]["sha256"]

        pairs.append({
            "neutral_id": rid, "case_id": case_id,
            "shared_prompt": case_prompts[str(case_id)]["prompt"],  # same for A and B -- the point of v2
            "A_endpoint": a_ep, "A_source_path": str(a_src), "A_sha256": a_sha, "A_package_filename": f"{rid}_A.png",
            "B_endpoint": b_ep, "B_source_path": str(b_src), "B_sha256": b_sha, "B_package_filename": f"{rid}_B.png",
        })

    assert count_a["S2_STAGE1_HISTORY"] == 10 and count_a["S2_HISTORY_BOTH_STAGES"] == 10

    pairs.sort(key=lambda p: p["neutral_id"])

    private_mapping = {
        "purpose": "PRIVATE mapping for context_quality_review_s2_v2 -- SEPARATE from v1's mapping. "
                   "Neutral R-id to case ID/endpoint/source/hash/prompt. Never publish this file or its "
                   "random_seed. Kept outside the git checkout.",
        "package_version": "context_quality_review_s2_v2-v1",
        "generated_at_unix": time.time(),
        "random_seed": seed,
        "randomization_method": "python random.Random(seed).shuffle() -- Fisher-Yates -- applied once to the "
                                 "20 case IDs for R-slot assignment, and once to a 10x/10x balanced A-position "
                                 "label list, both seeded from a runtime-generated secrets.randbits(64) seed, "
                                 "independent of v1's seed and mapping",
        "n_pairs": len(pairs),
        "a_position_balance": count_a,
        "pairs": pairs,
    }

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = PRIVATE_DIR / "PRIVATE_MAPPING_V2.json"
    mapping_path.write_text(json.dumps(private_mapping, indent=1))
    mapping_path.chmod(0o600)
    mapping_sha256 = sha256_file(mapping_path)

    # safety assertion: v1's private mapping must be byte-identical to before this run
    v1_sha_after = sha256_file(v1_private) if v1_private.is_file() else None
    assert v1_sha_before == v1_sha_after, "REFUSING: v1 private mapping changed during v2 build -- this must never happen"

    commitment = {
        "purpose": "Public SHA256 commitment to the v2 private mapping file, published BEFORE collecting any "
                   "v2 responses. Separate from v1's commitment. Does NOT reveal the mapping or seed.",
        "package_version": "context_quality_review_s2_v2-v1",
        "private_mapping_sha256": mapping_sha256,
        "private_mapping_file_name": "PRIVATE_MAPPING_V2.json",
        "committed_at_unix": time.time(),
        "n_pairs_committed": len(pairs),
        "v1_mapping_untouched_check": {"sha256_before": v1_sha_before, "sha256_after": v1_sha_after, "unchanged": v1_sha_before == v1_sha_after},
    }
    commitment_path = HOST_ROOT / "context_quality_review_s2_v2/MAPPING_COMMITMENT_V2.json"
    commitment_path.write_text(json.dumps(commitment, indent=1))

    print(f"v2 package built: {len(pairs)} pairs, {len(pairs)*2} neutral images written to {PACKAGE_IMAGES_DIR}")
    print(f"A-position balance: {count_a}")
    print(f"v2 private mapping written to {mapping_path} (mode 0600, NOT published)")
    print(f"v2 public commitment hash: {mapping_sha256}")
    print(f"v1 mapping confirmed UNCHANGED by this run: {v1_sha_before == v1_sha_after}")


if __name__ == "__main__":
    main()
