#!/usr/bin/env python3
"""Re-verifies the same 40 source images used in v1 (case IDs 60-79,
S2_STAGE1_HISTORY vs S2_HISTORY_BOTH_STAGES) against the published
evidence index and completion records, AND extracts the exact shared
generation prompt for each case directly from the frozen panel manifest
-- not an assistant-written description of image content. No GPU, no
model work, no regeneration.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image

HOST_ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
PUB_ROOT = Path("/data/bijaypandey/xxresearch_standalone/results/rece_seq_golfball_englishspringer")

PANEL_MANIFEST = HOST_ROOT / "history_preservation_confirmation/panel/FRESH_PANEL_MANIFEST.json"
PANEL_SHA_EXPECTED = "cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2"

ENDPOINTS = {
    "S2_STAGE1_HISTORY": {
        "img_dir": HOST_ROOT / "stage1_history_intervention_s2/eval/S2_STAGE1_HISTORY/imgs",
        "pub_img_dir": "images/stage1_history_intervention_s2/original_images",
        "completion_record": HOST_ROOT / "stage1_history_intervention_s2/CASE_LEVEL_COMPLETION_RECORD.json",
    },
    "S2_HISTORY_BOTH_STAGES": {
        "img_dir": HOST_ROOT / "history_both_stages_s2/eval/S2_HISTORY_BOTH_STAGES/imgs",
        "pub_img_dir": "images/history_both_stages_s2/original_images",
        "completion_record": HOST_ROOT / "history_both_stages_s2/CASE_LEVEL_COMPLETION_RECORD.json",
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_evidence_index_hashes(path_substring: str) -> dict:
    idx_path = PUB_ROOT / "PUBLISHED_EVIDENCE_INDEX.md"
    out = {}
    for line in idx_path.read_text().splitlines():
        if path_substring in line and line.startswith("| `"):
            parts = line.split("`")
            pub_path = parts[1]
            sha = parts[3]
            fname = pub_path.rsplit("/", 1)[-1]
            out[fname] = sha
    return out


def main():
    report = {"checks": {}}

    panel_sha = sha256_file(PANEL_MANIFEST)
    report["checks"]["panel_manifest_hash"] = {"expected": PANEL_SHA_EXPECTED, "actual": panel_sha,
                                                 "match": panel_sha == PANEL_SHA_EXPECTED}
    assert panel_sha == PANEL_SHA_EXPECTED, "panel manifest hash mismatch"

    manifest = json.loads(PANEL_MANIFEST.read_text())
    rows_by_id = {r["case_id"]: r for r in manifest["rows"]}

    case_ids = list(range(60, 80))
    per_case = {}
    for ep_name, ep in ENDPOINTS.items():
        evidence_hashes = load_evidence_index_hashes(f"{ep['pub_img_dir']}/")
        completion = json.loads(ep["completion_record"].read_text())
        completion_by_case = {r["case_id"]: r for r in completion}

        for cid in case_ids:
            fname = f"{cid}_0.png"
            img_path = ep["img_dir"] / fname
            assert img_path.is_file(), f"{ep_name} case {cid}: missing image {img_path}"
            actual_hash = sha256_file(img_path)

            comp_row = completion_by_case.get(cid)
            assert comp_row is not None
            comp_hash_match = comp_row["sha256"] == actual_hash
            evidence_hash = evidence_hashes.get(fname)
            evidence_hash_match = evidence_hash == actual_hash

            with Image.open(img_path) as im:
                dims = im.size

            per_case.setdefault(cid, {})[ep_name] = {
                "path": str(img_path), "sha256": actual_hash,
                "completion_record_match": comp_hash_match,
                "evidence_index_match": evidence_hash_match,
                "dims_512x512": dims == (512, 512),
                "prompt_from_completion_record": comp_row["prompt"],
                "seed_from_completion_record": comp_row["seed"],
                "class": comp_row["class"], "family": comp_row["family"], "scene_index": comp_row["scene_index"],
            }
            assert comp_hash_match and evidence_hash_match and dims == (512, 512)

    # cross-check prompt/seed identity across endpoints AND against the panel manifest itself
    # (the manifest is the single source of truth for the "exact common generation prompt")
    mismatches = []
    case_prompts = {}
    for cid in case_ids:
        manifest_row = rows_by_id[cid]
        a = per_case[cid]["S2_STAGE1_HISTORY"]
        b = per_case[cid]["S2_HISTORY_BOTH_STAGES"]
        if a["prompt_from_completion_record"] != manifest_row["prompt"] or a["seed_from_completion_record"] != manifest_row["seed"]:
            mismatches.append(f"{cid}-A-vs-manifest")
        if b["prompt_from_completion_record"] != manifest_row["prompt"] or b["seed_from_completion_record"] != manifest_row["seed"]:
            mismatches.append(f"{cid}-B-vs-manifest")
        if a["prompt_from_completion_record"] != b["prompt_from_completion_record"] or a["seed_from_completion_record"] != b["seed_from_completion_record"]:
            mismatches.append(f"{cid}-A-vs-B")
        case_prompts[cid] = {
            "prompt": manifest_row["prompt"],  # taken from the frozen manifest itself, not from either endpoint's own copy
            "seed": manifest_row["seed"], "class": manifest_row["class"],
            "family": manifest_row["family"], "scene_index": manifest_row["scene_index"],
        }

    report["checks"]["prompt_seed_consistency"] = {"n_cases": len(case_ids), "mismatches": mismatches}
    assert not mismatches, f"prompt/seed mismatches: {mismatches}"

    report["per_case"] = per_case
    report["case_prompts_from_manifest"] = case_prompts
    report["all_checks_passed"] = True

    out_path = HOST_ROOT / "context_quality_review_s2_v2_SOURCE_VERIFICATION.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(f"ALL CHECKS PASSED. 20 cases, 40 images, all 512x512, hashes verified, prompts/seeds identical "
          f"across both endpoints and matching the frozen panel manifest exactly. Written to {out_path}")


if __name__ == "__main__":
    main()
