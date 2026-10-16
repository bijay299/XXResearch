#!/usr/bin/env python3
"""Unblinds the selected human review response against the committed
private mapping, verifies it, joins Q01-Q20 to case IDs/endpoints, and
recomputes the reported counts independently from the raw response file.
No GPU, no model work, no new generation or classification -- this only
reads already-published/already-private files.
"""
import hashlib
import json
from pathlib import Path

PRIVATE_MAPPING_PATH = Path("/data/bijaypandey/cuig_pilot/review_private/blinded_quality_review_s2/PRIVATE_MAPPING.json")
REQUIRED_MAPPING_SHA256 = "bbf8a9eb8b6459bdb2e4ff33ed8638c5708fe4ecb10e58c09a02ecbdcb47689a"
RESPONSE_PATH = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2/responses/review_export_final_2026-10-16T14-06-23-096Z.json")
PANEL_MANIFEST_PATH = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_preservation_confirmation/panel/FRESH_PANEL_MANIFEST.json")
PANEL_SHA_EXPECTED = "cfb0f264b76620f47c292752515d545cb9ff058616b0c53d5c688753d628b8e2"
SOURCE_VERIFICATION_PATH = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2_SOURCE_VERIFICATION.json")

ASSISTANT_SPRINGER_DEGENERATE = {
    "S2_STAGE1_HISTORY": {77, 79},               # corrected, C31: 2/20
    "S2_HISTORY_BOTH_STAGES": {61, 63, 68, 71, 77, 79},  # C32: 6/20
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = {"steps": {}}

    # --- step 1: verify mapping hash before opening ---
    actual_mapping_sha = sha256_file(PRIVATE_MAPPING_PATH)
    out["steps"]["1_mapping_hash_verification"] = {
        "required": REQUIRED_MAPPING_SHA256, "actual": actual_mapping_sha,
        "match": actual_mapping_sha == REQUIRED_MAPPING_SHA256,
    }
    assert actual_mapping_sha == REQUIRED_MAPPING_SHA256, "STOP: mapping hash mismatch"

    mapping = json.loads(PRIVATE_MAPPING_PATH.read_text())
    pairs_by_qid = {p["neutral_id"]: p for p in mapping["pairs"]}

    # --- step 2: structural verification of the mapping itself ---
    case_ids = sorted(p["case_id"] for p in mapping["pairs"])
    out["steps"]["2_mapping_structure"] = {
        "n_pairs": len(mapping["pairs"]),
        "case_ids_are_60_to_79_unique": case_ids == list(range(60, 80)),
        "a_position_balance": mapping["a_position_balance"],
        "balance_is_10_10": mapping["a_position_balance"] == {"S2_STAGE1_HISTORY": 10, "S2_HISTORY_BOTH_STAGES": 10},
    }
    assert case_ids == list(range(60, 80))
    assert mapping["a_position_balance"] == {"S2_STAGE1_HISTORY": 10, "S2_HISTORY_BOTH_STAGES": 10}

    # --- step 3: cross-check mapping's recorded hashes against the panel manifest and prior source verification ---
    panel_sha = sha256_file(PANEL_MANIFEST_PATH)
    out["steps"]["3_panel_hash"] = {"expected": PANEL_SHA_EXPECTED, "actual": panel_sha, "match": panel_sha == PANEL_SHA_EXPECTED}
    assert panel_sha == PANEL_SHA_EXPECTED

    source_verification = json.loads(SOURCE_VERIFICATION_PATH.read_text())
    per_case = source_verification["per_case"]
    mismatches = []
    for p in mapping["pairs"]:
        cid = str(p["case_id"])
        for ep, sha_key in [(p["A_endpoint"], "A_sha256"), (p["B_endpoint"], "B_sha256")]:
            expected = per_case[cid][ep]["sha256"]
            actual = p[sha_key]
            if expected != actual:
                mismatches.append({"qid": p["neutral_id"], "endpoint": ep, "expected": expected, "actual": actual})
    out["steps"]["4_mapping_hashes_vs_source_verification"] = {"n_checked": 40, "mismatches": mismatches}
    assert not mismatches, f"mapping hash mismatches vs source verification: {mismatches}"

    # one image per endpoint per pair, exactly
    endpoint_pairs_ok = all(
        {p["A_endpoint"], p["B_endpoint"]} == {"S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES"}
        for p in mapping["pairs"]
    )
    out["steps"]["5_one_image_per_endpoint_per_pair"] = {"ok": endpoint_pairs_ok}
    assert endpoint_pairs_ok

    # --- step 6: verify response file integrity (re-check, not trusted) ---
    response_sha = sha256_file(RESPONSE_PATH)
    response = json.loads(RESPONSE_PATH.read_text())
    out["steps"]["6_response_integrity"] = {
        "sha256": response_sha,
        "packageVersion_ok": response["packageVersion"] == "blinded_quality_review_s2-v1",
        "protocolVersion_ok": response["protocolVersion"] == "FROZEN_PROTOCOL-v1",
        "isComplete": response["isComplete"], "n_rows": len(response["responses"]),
    }
    assert response["packageVersion"] == "blinded_quality_review_s2-v1"
    assert response["protocolVersion"] == "FROZEN_PROTOCOL-v1"
    assert len(response["responses"]) == 20

    # --- step 7: JOIN -- unblind ---
    joined = []
    resp_by_qid = {r["neutral_id"]: r for r in response["responses"]}
    for qid, p in pairs_by_qid.items():
        r = resp_by_qid[qid]
        a_ep, b_ep = p["A_endpoint"], p["B_endpoint"]
        row = {
            "neutral_id": qid, "case_id": p["case_id"],
            "A_endpoint": a_ep, "B_endpoint": b_ep,
            "a_degeneration": r["a_degeneration"], "b_degeneration": r["b_degeneration"],
            "a_local_defects": r["a_local_defects"], "b_local_defects": r["b_local_defects"],
            "pair_preference_raw": r["pair_preference"],  # A/B/none/uncertain -- "worse" image per protocol, NOT "preferred"
            "comment": r["comment"],
        }
        # endpoint-keyed judgments
        row[f"{a_ep}_degeneration"] = r["a_degeneration"]
        row[f"{b_ep}_degeneration"] = r["b_degeneration"]
        row[f"{a_ep}_local_defects"] = r["a_local_defects"]
        row[f"{b_ep}_local_defects"] = r["b_local_defects"]
        if r["pair_preference"] == "A":
            row["more_corrupted_endpoint"] = a_ep
        elif r["pair_preference"] == "B":
            row["more_corrupted_endpoint"] = b_ep
        else:
            row["more_corrupted_endpoint"] = r["pair_preference"]  # "none" or "uncertain"
        joined.append(row)
    joined.sort(key=lambda r: r["case_id"])

    # --- step 8: recompute reported counts independently ---
    def count(ep, field, val):
        return sum(1 for r in joined if r[f"{ep}_{field}"] == val)

    stage1h_degen = count("S2_STAGE1_HISTORY", "degeneration", "yes")
    bothstages_degen = count("S2_HISTORY_BOTH_STAGES", "degeneration", "yes")
    stage1h_local = count("S2_STAGE1_HISTORY", "local_defects", "yes")
    bothstages_local = count("S2_HISTORY_BOTH_STAGES", "local_defects", "yes")

    degen_cases = {
        "S2_STAGE1_HISTORY": sorted(r["case_id"] for r in joined if r["S2_STAGE1_HISTORY_degeneration"] == "yes"),
        "S2_HISTORY_BOTH_STAGES": sorted(r["case_id"] for r in joined if r["S2_HISTORY_BOTH_STAGES_degeneration"] == "yes"),
    }
    local_no_cases = {
        "S2_STAGE1_HISTORY": sorted(r["case_id"] for r in joined if r["S2_STAGE1_HISTORY_local_defects"] == "no"),
        "S2_HISTORY_BOTH_STAGES": sorted(r["case_id"] for r in joined if r["S2_HISTORY_BOTH_STAGES_local_defects"] == "no"),
    }

    more_corrupted_counts = {"S2_STAGE1_HISTORY": 0, "S2_HISTORY_BOTH_STAGES": 0, "none": 0, "uncertain": 0}
    more_corrupted_detail = []
    for r in joined:
        v = r["more_corrupted_endpoint"]
        more_corrupted_counts[v] = more_corrupted_counts.get(v, 0) + 1
        if v in ("S2_STAGE1_HISTORY", "S2_HISTORY_BOTH_STAGES"):
            more_corrupted_detail.append({"case_id": r["case_id"], "more_corrupted": v})

    # paired transition S2_STAGE1_HISTORY -> S2_HISTORY_BOTH_STAGES (degeneration)
    d1, d2 = degen_cases["S2_STAGE1_HISTORY"], degen_cases["S2_HISTORY_BOTH_STAGES"]
    s1, s2 = set(d1), set(d2)
    persisted = sorted(s1 & s2)
    recovered = sorted(s1 - s2)
    newly_damaged = sorted(s2 - s1)
    neither = sorted(set(range(60, 80)) - s1 - s2)

    out["steps"]["7_join_and_unblind"] = {"n_rows_joined": len(joined)}
    out["recomputed_counts"] = {
        "S2_STAGE1_HISTORY_degeneration_yes": f"{stage1h_degen}/20", "degenerate_cases": degen_cases["S2_STAGE1_HISTORY"],
        "S2_HISTORY_BOTH_STAGES_degeneration_yes": f"{bothstages_degen}/20", "degenerate_cases_both_stages": degen_cases["S2_HISTORY_BOTH_STAGES"],
        "S2_STAGE1_HISTORY_local_defects_yes": f"{stage1h_local}/20", "local_defects_no_cases": local_no_cases["S2_STAGE1_HISTORY"],
        "S2_HISTORY_BOTH_STAGES_local_defects_yes": f"{bothstages_local}/20", "local_defects_no_cases_both_stages": local_no_cases["S2_HISTORY_BOTH_STAGES"],
        "paired_degeneration_transition": {
            "persisted": persisted, "recovered": recovered, "newly_damaged": newly_damaged, "neither": neither,
            "n_persisted": len(persisted), "n_recovered": len(recovered), "n_newly_damaged": len(newly_damaged), "n_neither": len(neither),
        },
        "more_corrupted_counts": more_corrupted_counts,
        "more_corrupted_detail": more_corrupted_detail,
    }

    joined_path = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/UNBLINDED_JOINED_TABLE.json")
    joined_path.write_text(json.dumps(joined, indent=1))
    out["joined_table_path"] = str(joined_path)

    out["all_checks_passed"] = True
    out_path = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/blinded_quality_review_s2/closeout/UNBLINDING_VERIFICATION.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
