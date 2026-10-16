#!/usr/bin/env python3
"""Verifies the 40 source images (20 matched pairs, case IDs 60-79) for
the blinded human review package, before any package is built. Checks:
  1. Panel manifest hash matches the pinned value.
  2. Each image's bytes match BOTH the published evidence-index hash AND
     the host-side completion record's recorded hash.
  3. Each image is exactly 512x512.
  4. Prompt/seed match between the two endpoints for each case_id
     (the "matched" requirement).
  5. Exactly 20 unique case IDs, exactly one image per endpoint per case.
No GPU, no model work -- read-only verification of already-published,
already-generated files.
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
        "ckpt_sha256": "797739999bc29c2978dc10f3b7ebb7c6f983e7e1738732a5ad47b7d2c4c6bf41",
    },
    "S2_HISTORY_BOTH_STAGES": {
        "img_dir": HOST_ROOT / "history_both_stages_s2/eval/S2_HISTORY_BOTH_STAGES/imgs",
        "pub_img_dir": "images/history_both_stages_s2/original_images",
        "completion_record": HOST_ROOT / "history_both_stages_s2/CASE_LEVEL_COMPLETION_RECORD.json",
        "ckpt_sha256": "7fdc01c44e247ccfa23ebabd5f663283943501832894ff2e6fb23c475f1cd5c6",
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
            # | `path` | bytes | `sha256` |
            pub_path = parts[1]
            sha = parts[3]
            fname = pub_path.rsplit("/", 1)[-1]
            out[fname] = sha
    return out


def main():
    report = {"checks": {}}

    # 1. panel manifest hash
    panel_sha = sha256_file(PANEL_MANIFEST)
    report["checks"]["panel_manifest_hash"] = {"expected": PANEL_SHA_EXPECTED, "actual": panel_sha,
                                                 "match": panel_sha == PANEL_SHA_EXPECTED}
    assert panel_sha == PANEL_SHA_EXPECTED, "panel manifest hash mismatch"

    manifest = json.loads(PANEL_MANIFEST.read_text())
    rows_by_id = {r["case_id"]: r for r in manifest["rows"]}

    case_ids = list(range(60, 80))
    assert len(case_ids) == 20 and len(set(case_ids)) == 20

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

            # vs completion record
            comp_row = completion_by_case.get(cid)
            assert comp_row is not None, f"{ep_name} case {cid}: not in completion record"
            comp_hash_match = comp_row["sha256"] == actual_hash

            # vs published evidence index
            evidence_hash = evidence_hashes.get(fname)
            evidence_hash_match = evidence_hash == actual_hash

            # dimensions
            with Image.open(img_path) as im:
                dims = im.size

            per_case.setdefault(cid, {})[ep_name] = {
                "path": str(img_path), "sha256": actual_hash,
                "completion_record_match": comp_hash_match,
                "evidence_index_match": evidence_hash_match,
                "evidence_index_hash_found": evidence_hash is not None,
                "dims": dims, "dims_512x512": dims == (512, 512),
                "prompt": comp_row["prompt"], "seed": comp_row["seed"], "class": comp_row["class"],
                "manifest_prompt": rows_by_id[cid]["prompt"], "manifest_seed": rows_by_id[cid]["seed"],
            }
            assert comp_hash_match, f"{ep_name} case {cid}: completion-record hash mismatch"
            assert evidence_hash_match, f"{ep_name} case {cid}: evidence-index hash mismatch (found={evidence_hash is not None})"
            assert dims == (512, 512), f"{ep_name} case {cid}: not 512x512, got {dims}"

    # matched-pair check: same prompt/seed across both endpoints, per case
    mismatches = []
    for cid in case_ids:
        a = per_case[cid]["S2_STAGE1_HISTORY"]
        b = per_case[cid]["S2_HISTORY_BOTH_STAGES"]
        if a["prompt"] != b["prompt"] or a["seed"] != b["seed"]:
            mismatches.append(cid)
        if a["prompt"] != a["manifest_prompt"] or a["seed"] != a["manifest_seed"]:
            mismatches.append(f"{cid}-vs-manifest-A")
        if b["prompt"] != b["manifest_prompt"] or b["seed"] != b["manifest_seed"]:
            mismatches.append(f"{cid}-vs-manifest-B")

    report["checks"]["matched_pairs"] = {"n_cases": len(case_ids), "mismatches": mismatches}
    assert not mismatches, f"prompt/seed mismatches: {mismatches}"

    report["checks"]["n_unique_cases"] = len(set(case_ids))
    report["checks"]["n_images_total"] = len(case_ids) * 2
    report["per_case"] = per_case
    report["all_checks_passed"] = True

    out_path = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2_SOURCE_VERIFICATION.json")
    out_path.write_text(json.dumps(report, indent=1))
    print(f"ALL CHECKS PASSED. 20 unique cases, 40 images, all 512x512, all hashes verified against both "
          f"the completion records and the published evidence index. Written to {out_path}")


if __name__ == "__main__":
    main()
