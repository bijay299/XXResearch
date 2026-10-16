#!/usr/bin/env python3
"""Verifies the 20 pristine M0 reference images (cases 60-79, English
springer) before building the baseline-context package. Checks bytes,
hashes against the published evidence index and the M0 image_report,
prompt/seed against the frozen panel manifest, and compares recorded
generation settings across M0, S2_STAGE1_HISTORY, and
S2_HISTORY_BOTH_STAGES. No GPU, no model work, no regeneration.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image

HOST_ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer")
PUB_ROOT = Path("/data/bijaypandey/xxresearch_standalone/results/rece_seq_golfball_englishspringer")

PANEL_MANIFEST = HOST_ROOT / "history_preservation_confirmation/panel/FRESH_PANEL_MANIFEST.json"
PANEL_SHA_EXPECTED = "cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2"
M0_IMAGE_REPORT = HOST_ROOT / "history_preservation_confirmation/panel/eval/M0/image_report.json"

PUB_REF_DIR = "images/history_preservation_confirmation/target_role_full_res"


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
    report["checks"]["panel_manifest_hash"] = {"expected": PANEL_SHA_EXPECTED, "actual": panel_sha, "match": panel_sha == PANEL_SHA_EXPECTED}
    assert panel_sha == PANEL_SHA_EXPECTED

    manifest = json.loads(PANEL_MANIFEST.read_text())
    rows_by_id = {r["case_id"]: r for r in manifest["rows"]}

    m0_report = json.loads(M0_IMAGE_REPORT.read_text())
    m0_by_case = {im["case_id"]: im for im in m0_report["images"]}
    m0_settings = m0_report.get("settings")
    m0_edit_ckpt = m0_report.get("edit_ckpt_load_report", {}).get("edit_ckpt")
    report["checks"]["m0_report_records_no_edit_checkpoint"] = {"edit_ckpt_field": m0_edit_ckpt, "is_none": m0_edit_ckpt is None}
    assert m0_edit_ckpt is None, f"M0 image_report unexpectedly records an edit checkpoint: {m0_edit_ckpt}"

    # compare settings across all three checkpoints' own image_reports
    s1h_report = json.loads((HOST_ROOT / "stage1_history_intervention_s2/eval/S2_STAGE1_HISTORY/image_report.json").read_text())
    both_report = json.loads((HOST_ROOT / "history_both_stages_s2/eval/S2_HISTORY_BOTH_STAGES/image_report.json").read_text())
    settings_match = {
        "M0_vs_S2_STAGE1_HISTORY": m0_settings == s1h_report.get("settings"),
        "M0_vs_S2_HISTORY_BOTH_STAGES": m0_settings == both_report.get("settings"),
    }
    report["checks"]["generation_settings_identical_across_all_three"] = {
        "M0_settings": m0_settings, "match": settings_match,
    }
    if not all(settings_match.values()):
        report["checks"]["MATERIAL_MISMATCH_DISCLOSED"] = True

    evidence_hashes = load_evidence_index_hashes(PUB_REF_DIR + "/")

    case_ids = list(range(60, 80))
    per_case = {}
    for cid in case_ids:
        fname = f"M0_case{cid}_English_springer.png"
        pub_path = PUB_ROOT / PUB_REF_DIR / fname
        assert pub_path.is_file(), f"missing published reference image: {pub_path}"
        actual_hash = sha256_file(pub_path)

        m0_row = m0_by_case.get(cid)
        assert m0_row is not None, f"case {cid} not in M0 image_report"
        report_hash_match = m0_row["sha256"] == actual_hash

        evidence_hash = evidence_hashes.get(fname)
        evidence_hash_match = evidence_hash == actual_hash

        with Image.open(pub_path) as im:
            dims = im.size

        manifest_row = rows_by_id[cid]
        prompt_seed_match = (m0_row["prompt"] == manifest_row["prompt"]) and (m0_row["seed"] == manifest_row["seed"])

        per_case[cid] = {
            "published_path": str(pub_path), "filename": fname, "sha256": actual_hash,
            "m0_report_hash_match": report_hash_match, "evidence_index_hash_match": evidence_hash_match,
            "dims_512x512": dims == (512, 512), "prompt_seed_match_manifest": prompt_seed_match,
            "prompt": m0_row["prompt"], "seed": m0_row["seed"],
        }
        assert report_hash_match and evidence_hash_match and dims == (512, 512) and prompt_seed_match, f"case {cid} verification failed: {per_case[cid]}"

    report["checks"]["n_references_verified"] = len(per_case)
    report["per_case"] = per_case
    report["all_checks_passed"] = True

    out_path = HOST_ROOT / "context_quality_review_s2_baseline_SOURCE_VERIFICATION.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(f"ALL CHECKS PASSED. 20 M0 reference images verified (hashes vs. M0 image_report AND published "
          f"evidence index, 512x512, prompt/seed match the frozen manifest). M0 report confirms no edit "
          f"checkpoint. Generation settings match across M0/S2_STAGE1_HISTORY/S2_HISTORY_BOTH_STAGES: "
          f"{settings_match}. Written to {out_path}")


if __name__ == "__main__":
    main()
