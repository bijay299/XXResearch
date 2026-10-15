#!/usr/bin/env python3
"""Compiles the frozen visual-review record for the 6 new endpoints
(S1_Stage1, S1_BLANK, S1_HISTORY, S2_Stage1, S2_BLANK, S2_HISTORY) on the
fresh 80-case panel. Judgments below were recorded during actual
full-512x512, case-by-case inspection of every historical-target and
new-target image (240 images per sequence x 2 sequences = ... ; 40
individual images per endpoint x 6 = 240 total individual) and grid-level
inspection (no escalation needed) of the two purely-retained-class blocks
per endpoint (240 total). Written and hashed BEFORE any classification is
run on these endpoints.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from review_lib import compile_endpoint, freeze_and_hash  # noqa: E402

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_preservation_confirmation/panel")


def clean_block(lo, hi, note="retained class, grid-reviewed, clean, no escalation"):
    return {c: {"degenerate": False, "object_present": True, "note": note, "resolution_viewed": "grid_160px"}
            for c in range(lo, hi)}


def suppressed_clean_block(lo, hi, note):
    return {c: {"degenerate": False, "object_present": False, "note": note, "resolution_viewed": "full_512x512"}
            for c in range(lo, hi)}


rows = []

# ================= S1: golf ball -> French horn =================
# golf ball (0-19, historical). Degenerate case sets recorded during review.
S1_STAGE1_GOLF_NONDEG = {17, 18}
S1_BLANK_GOLF_NONDEG = {18}
S1_HISTORY_GOLF_NONDEG = set()  # all 20 judged degenerate, incl. 16-19 (blocky/flat-shaded pixelation)

for name, nondeg in [("S1_Stage1", S1_STAGE1_GOLF_NONDEG), ("S1_BLANK", S1_BLANK_GOLF_NONDEG),
                      ("S1_HISTORY", S1_HISTORY_GOLF_NONDEG)]:
    j = {}
    for c in range(0, 20):
        deg = c not in nondeg
        j[c] = {"degenerate": deg, "object_present": False,
                 "note": ("pervasive sparkle/mesh/checkerboard or blocky-pixelation corruption, no golf ball" if deg
                           else "minor/localized texture only, not pervasive, no golf ball"),
                 "resolution_viewed": "full_512x512"}
    j.update(suppressed_clean_block(40, 60, "no French horn, coherent redirected content, no corruption"))  # new target
    j.update(clean_block(20, 40))  # parachute retained
    j.update(clean_block(60, 80))  # English springer retained
    rows += compile_endpoint(name, ROOT / "eval" / name / "image_report.json", j,
                              "S1 (golf ball -> French horn) endpoint on fresh panel")

# ================= S2: English springer -> golf ball =================
# English springer (60-79, historical).
S2_STAGE1_DOG_PRESENT = {62, 63, 66, 67}
S2_BLANK_DOG_PRESENT = {74, 77}
S2_HISTORY_DOG_PRESENT = {76, 77}

for name, dog_present in [("S2_Stage1", S2_STAGE1_DOG_PRESENT), ("S2_BLANK", S2_BLANK_DOG_PRESENT),
                           ("S2_HISTORY", S2_HISTORY_DOG_PRESENT)]:
    j = {}
    for c in range(60, 80):
        present = c in dog_present
        j[c] = {"degenerate": True, "object_present": present,
                 "note": ("dog-like animal visible but scene still pervasively corrupted (blur/mesh/distortion), not a clean recovery" if present
                           else "pervasive corruption (blur/mesh/painterly distortion), no dog"),
                 "resolution_viewed": "full_512x512"}
    j.update(suppressed_clean_block(0, 20, "no golf ball, coherent redirected content, no corruption"))  # new target
    j.update(clean_block(20, 40))  # parachute retained
    j.update(clean_block(40, 60))  # French horn retained
    rows += compile_endpoint(name, ROOT / "eval" / name / "image_report.json", j,
                              "S2 (English springer -> golf ball) endpoint on fresh panel")

assert len(rows) == 480, len(rows)

completion_path, labels_path, gate_path, gate = freeze_and_hash(rows, ROOT, "NEW_ENDPOINTS")
print("rows:", len(rows))
print(gate)
