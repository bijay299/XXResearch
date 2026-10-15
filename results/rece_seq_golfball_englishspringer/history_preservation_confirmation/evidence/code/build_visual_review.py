#!/usr/bin/env python3
"""Builds the frozen initial visual-label file and the completion record
from the per-case judgments recorded during this session's full-512x512
review (both endpoints, all 200 images: 120 target + 80 retained-class).
Computes real image sha256 and file existence for every row. Written and
hashed BEFORE any classification is run for either endpoint, per the
task's mandatory execution order and classifier precondition.
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/preservation_content_control")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# --- recorded judgments from this session's full-resolution review ---
# degeneration: whole-frame pervasive mesh/checkerboard/sparkle corruption
#   signature established throughout this results line (not recognizable-
#   foreground-content-cancels-corruption; localized background noise on a
#   minority of the frame does NOT count).
# ball_present / dog_present: an actual, visually recognizable instance of
#   the object -- an ambiguous round blob/smudge near a flag or hole is
#   recorded as ambiguous and NOT counted as a positive.

GOLF_ARM_GOLFBALL = {  # case_id: (degenerate, ball_present_or_ambiguous)
    **{i: (False, "absent") for i in range(0, 8)},
    8: (False, "ambiguous_round_shape_not_counted"), 9: (False, "ambiguous_round_shape_not_counted"),
    10: (False, "absent"), 11: (False, "ambiguous_round_shape_not_counted"),
    12: (False, "ambiguous_round_shape_not_counted"), 13: (False, "absent"),
    14: (False, "absent"), 15: (False, "absent"),
    **{i: (False, "absent") for i in range(16, 39)},
    39: (True, "absent"),
}
assert len(GOLF_ARM_GOLFBALL) == 40

HORN_ARM_GOLFBALL = {
    **{i: (True, "absent") for i in range(0, 16)},
    16: (False, "absent"),
    **{i: (True, "absent") for i in range(17, 24)},
    **{i: (False, "absent") for i in range(24, 32)},
    **{i: (True, "absent") for i in range(32, 40)},
}
assert len(HORN_ARM_GOLFBALL) == 40

GOLF_ARM_SPRINGER = {  # case_id: (degenerate, dog_present, breed_note)
    80: (False, True, "tan/white floppy-eared, springer-plausible pattern, breed not confirmed"),
    81: (False, False, None), 82: (False, False, None), 83: (False, False, None),
    84: (False, True, "two reddish-brown dogs, breed uncertain (not classic springer liver/white)"),
    85: (False, False, None),
    86: (False, True, "single reddish-brown dog, breed uncertain"),
    87: (False, False, "structural glitch line across sky, contained, not whole-frame"),
    88: (False, False, None), 89: (False, False, None),
    90: (False, True, "curly-coated dog wearing a hat, breed uncertain"),
    91: (False, False, None),
    92: (False, True, "dog swimming, reddish/brown, breed uncertain"),
    93: (False, True, "dog swimming, golden/tan, breed uncertain"),
    94: (False, True, "dog jumping into water, brown, breed uncertain"),
    95: (False, False, None), 96: (False, False, None), 97: (False, False, None),
    98: (False, False, None), 99: (False, False, None),
}
assert len(GOLF_ARM_SPRINGER) == 20

HORN_ARM_SPRINGER = {
    80: (False, True, "tan/white floppy-eared, springer-plausible pattern, breed not confirmed"),
    81: (False, False, None), 82: (False, False, None), 83: (False, False, None),
    84: (False, True, "reddish-brown dog, breed uncertain"),
    85: (False, False, None),
    86: (False, True, "reddish-brown/white pattern, closer to classic liver/white springer coloring, breed not confirmed"),
    87: (False, False, "structural glitch artifacts across sky, contained, not whole-frame"),
    88: (False, False, None), 89: (False, False, None),
    90: (False, True, "curly-coated dog wearing a hat, breed uncertain"),
    91: (False, True, "close-up dog face, brown/white, breed uncertain"),
    92: (False, False, None), 93: (False, False, None), 94: (False, False, None),
    95: (False, False, None), 96: (False, False, None), 97: (False, False, None),
    98: (False, False, None), 99: (False, False, None),
}
assert len(HORN_ARM_SPRINGER) == 20

# retained classes: both arms fully reviewed this session, all 40/40 clean
# and recognizable for parachute (40-59) and French horn (60-79) in BOTH
# endpoints -- no degeneration, no ambiguity, no uncertain case requiring
# a second full-res pass (already viewed at full 512x512 the first time).
RETAINED_CLEAN = {i: (False, True) for i in range(40, 80)}  # (degenerate, recognizable)


def build_endpoint(arm, golfball, springer):
    rows = []
    img_dir = ROOT / "eval" / arm / "imgs"
    for case_id in range(0, 40):
        deg, ball = golfball[case_id]
        p = img_dir / f"{case_id}_0.png"
        rows.append({
            "endpoint": arm, "case_id": case_id, "class": "golf_ball",
            "image_path": str(p), "image_exists": p.is_file(),
            "image_sha256": sha256_file(p) if p.is_file() else None,
            "resolution_viewed": "full_512x512",
            "degenerate": deg, "target_object_status": ball,
        })
    for case_id in range(40, 60):
        deg, rec = RETAINED_CLEAN[case_id]
        p = img_dir / f"{case_id}_0.png"
        rows.append({
            "endpoint": arm, "case_id": case_id, "class": "parachute",
            "image_path": str(p), "image_exists": p.is_file(),
            "image_sha256": sha256_file(p) if p.is_file() else None,
            "resolution_viewed": "full_512x512",
            "degenerate": deg, "recognizable": rec,
        })
    for case_id in range(60, 80):
        deg, rec = RETAINED_CLEAN[case_id]
        p = img_dir / f"{case_id}_0.png"
        rows.append({
            "endpoint": arm, "case_id": case_id, "class": "French_horn",
            "image_path": str(p), "image_exists": p.is_file(),
            "image_sha256": sha256_file(p) if p.is_file() else None,
            "resolution_viewed": "full_512x512",
            "degenerate": deg, "recognizable": rec,
        })
    for case_id in range(80, 100):
        deg, dog, breed_note = springer[case_id]
        p = img_dir / f"{case_id}_0.png"
        rows.append({
            "endpoint": arm, "case_id": case_id, "class": "English_springer",
            "image_path": str(p), "image_exists": p.is_file(),
            "image_sha256": sha256_file(p) if p.is_file() else None,
            "resolution_viewed": "full_512x512",
            "degenerate": deg, "dog_present": dog, "breed_note": breed_note,
        })
    return rows


def main():
    t0 = time.time()
    rows_golf = build_endpoint("AB_RETAIN_GOLF", GOLF_ARM_GOLFBALL, GOLF_ARM_SPRINGER)
    rows_horn = build_endpoint("AB_RETAIN_HORN", HORN_ARM_GOLFBALL, HORN_ARM_SPRINGER)
    all_rows = rows_golf + rows_horn

    assert all(r["image_exists"] for r in all_rows), "missing image file(s) -- cannot freeze labels"
    assert all(r["image_sha256"] is not None for r in all_rows)

    target_rows = [r for r in all_rows if r["class"] in ("golf_ball", "English_springer")]
    retained_rows = [r for r in all_rows if r["class"] in ("parachute", "French_horn")]
    assert len(target_rows) == 120, f"expected 120 target rows, got {len(target_rows)}"
    assert len(retained_rows) == 80, f"expected 80 retained rows, got {len(retained_rows)}"
    assert len(all_rows) == 200

    completion = {
        "note": "Frozen BEFORE any classification run for AB_RETAIN_GOLF or AB_RETAIN_HORN, per the task's mandatory execution order and classifier precondition. Every row's resolution_viewed is full_512x512 -- all 200 images (120 target + 80 retained-class) were opened and visually judged individually at native 512x512 resolution this session; none were judged from a thumbnail/grid/hash/file-metadata substitute.",
        "written_at_unix": time.time(),
        "n_rows": len(all_rows), "n_target_rows": len(target_rows), "n_retained_rows": len(retained_rows),
        "all_images_exist": True, "all_images_hashed": True,
        "rows": all_rows,
    }
    out_path = ROOT / "CASE_LEVEL_COMPLETION_RECORD.json"
    out_path.write_text(json.dumps(completion, indent=2))
    completion_sha = sha256_file(out_path)

    # --- per-endpoint aggregate summary (initial labels, pre-classification) ---
    def summarize(arm_rows, arm):
        golf = [r for r in arm_rows if r["class"] == "golf_ball"]
        springer = [r for r in arm_rows if r["class"] == "English_springer"]
        parachute = [r for r in arm_rows if r["class"] == "parachute"]
        horn = [r for r in arm_rows if r["class"] == "French_horn"]
        return {
            "golf_ball": {
                "n": len(golf), "n_degenerate": sum(r["degenerate"] for r in golf),
                "degenerate_case_ids": sorted(r["case_id"] for r in golf if r["degenerate"]),
                "n_ball_present_unambiguous": sum(r["target_object_status"] == "present" for r in golf),
                "n_ambiguous_round_shape_not_counted": sum(r["target_object_status"] == "ambiguous_round_shape_not_counted" for r in golf),
            },
            "English_springer": {
                "n": len(springer), "n_degenerate": sum(r["degenerate"] for r in springer),
                "n_dog_present": sum(r["dog_present"] for r in springer),
                "dog_present_case_ids": sorted(r["case_id"] for r in springer if r["dog_present"]),
            },
            "parachute": {"n": len(parachute), "n_degenerate": sum(r["degenerate"] for r in parachute),
                          "n_recognizable": sum(r["recognizable"] for r in parachute)},
            "French_horn": {"n": len(horn), "n_degenerate": sum(r["degenerate"] for r in horn),
                            "n_recognizable": sum(r["recognizable"] for r in horn)},
        }

    initial_labels = {
        "note": "Frozen initial visual labels, this session's full-512x512 review, written and hashed BEFORE any classification run for either new endpoint. This is assistant (Claude) visual review, not independent human validation (CORRECTIONS.md C20).",
        "written_at_unix": time.time(),
        "completion_record_sha256": completion_sha,
        "AB_RETAIN_GOLF": summarize(rows_golf, "AB_RETAIN_GOLF"),
        "AB_RETAIN_HORN": summarize(rows_horn, "AB_RETAIN_HORN"),
    }
    labels_path = ROOT / "VISUAL_REVIEW_INITIAL.json"
    labels_path.write_text(json.dumps(initial_labels, indent=2))
    labels_sha = sha256_file(labels_path)

    gate = {
        "completion_record_path": str(out_path), "completion_record_sha256": completion_sha,
        "initial_labels_path": str(labels_path), "initial_labels_sha256": labels_sha,
        "n_target_rows_present": len(target_rows), "n_target_rows_required": 120,
        "classifier_precondition_met": len(target_rows) == 120 and len(all_rows) == 200,
        "elapsed_seconds": time.time() - t0,
    }
    (ROOT / "CLASSIFIER_PRECONDITION_GATE.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    print(json.dumps(initial_labels, indent=2))


if __name__ == "__main__":
    main()
