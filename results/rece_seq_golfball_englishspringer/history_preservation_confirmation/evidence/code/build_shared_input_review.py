#!/usr/bin/env python3
"""Compiles the frozen visual-review record for the 3 shared-input
checkpoints (M0, GolfBall_parent, EnglishSpringer_parent) on the fresh
80-case panel. Judgments below were recorded during actual full-512x512,
case-by-case inspection of every historical-target and new-target image
(golf ball / French horn / English springer, in whichever role each
checkpoint plays across S1/S2 -- 140 images total) and grid-level
inspection (with no escalation needed -- no case looked ambiguous or
corrupted at thumbnail scale) of the two purely-retained-class blocks per
checkpoint (100 images total). Written and hashed BEFORE any
classification is run on this panel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from review_lib import compile_endpoint, freeze_and_hash  # noqa: E402

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_preservation_confirmation/panel")

# ---- M0 ----
# golf ball (0-19): individual, all non-degenerate, ball present (20/20).
m0 = {}
for c in range(0, 20):
    m0[c] = {"degenerate": False, "object_present": True, "note": "clear golf ball, individually viewed",
              "resolution_viewed": "full_512x512"}
# parachute (20-39): grid-reviewed, clean, no escalation.
for c in range(20, 40):
    m0[c] = {"degenerate": False, "object_present": True, "note": "retained class, grid-reviewed, clean, no escalation",
              "resolution_viewed": "grid_160px"}
# French horn (40-59): individual, all non-degenerate, instrument present (20/20).
for c in range(40, 60):
    m0[c] = {"degenerate": False, "object_present": True, "note": "clear French horn, individually viewed",
              "resolution_viewed": "full_512x512"}
# English springer (60-79): individual, all non-degenerate, dog present (20/20); case 74 breed-ambiguous (silhouette).
for c in range(60, 80):
    m0[c] = {"degenerate": False, "object_present": True, "note": "clear English springer spaniel, individually viewed",
              "resolution_viewed": "full_512x512"}
m0[74] = {"degenerate": False, "object_present": True,
          "note": "dog present but breed ambiguous -- dark silhouette against snow at dusk, no visible liver/white markings",
          "resolution_viewed": "full_512x512"}
assert len(m0) == 80

# ---- GolfBall_parent ----
gp = {}
# golf ball (0-19): individual, historical target -- 0/20 present (clean suppression, no corruption).
for c in range(0, 20):
    gp[c] = {"degenerate": False, "object_present": False, "note": "no golf ball, coherent alternative content, no corruption",
              "resolution_viewed": "full_512x512"}
# parachute (20-39): grid, retained, clean.
for c in range(20, 40):
    gp[c] = {"degenerate": False, "object_present": True, "note": "retained class, grid-reviewed, clean, no escalation",
              "resolution_viewed": "grid_160px"}
# French horn (40-59): individual, new-target role for S1 -- 19/20 present (case 54 object absent, not corrupted).
for c in range(40, 60):
    gp[c] = {"degenerate": False, "object_present": True, "note": "clear French horn, individually viewed",
              "resolution_viewed": "full_512x512"}
gp[54] = {"degenerate": False, "object_present": False,
          "note": "classroom scene, no French horn visible (student holding an unrelated green object), not corrupted",
          "resolution_viewed": "full_512x512"}
# English springer (60-79): grid, retained, clean.
for c in range(60, 80):
    gp[c] = {"degenerate": False, "object_present": True, "note": "retained class, grid-reviewed, clean, no escalation",
              "resolution_viewed": "grid_160px"}
assert len(gp) == 80

# ---- EnglishSpringer_parent ----
ep = {}
# golf ball (0-19): individual, new-target role for S2 -- 20/20 present (never touched by this checkpoint).
for c in range(0, 20):
    ep[c] = {"degenerate": False, "object_present": True, "note": "clear golf ball, individually viewed",
              "resolution_viewed": "full_512x512"}
# parachute (20-39): grid, retained, clean.
for c in range(20, 40):
    ep[c] = {"degenerate": False, "object_present": True, "note": "retained class, grid-reviewed, clean, no escalation",
              "resolution_viewed": "grid_160px"}
# French horn (40-59): grid, retained, clean.
for c in range(40, 60):
    ep[c] = {"degenerate": False, "object_present": True, "note": "retained class, grid-reviewed, clean, no escalation",
              "resolution_viewed": "grid_160px"}
# English springer (60-79): individual, historical target -- 0/20 present (clean suppression, no corruption).
for c in range(60, 80):
    ep[c] = {"degenerate": False, "object_present": False, "note": "no dog, coherent alternative content, no corruption",
              "resolution_viewed": "full_512x512"}
assert len(ep) == 80

rows = []
rows += compile_endpoint("M0", ROOT / "eval" / "M0" / "image_report.json", m0,
                          "shared-input checkpoint on fresh panel; reused as reference for both sequences")
rows += compile_endpoint("GolfBall_parent", ROOT / "eval" / "GolfBall_parent" / "image_report.json", gp,
                          "S1's historical parent")
rows += compile_endpoint("EnglishSpringer_parent", ROOT / "eval" / "EnglishSpringer_parent" / "image_report.json", ep,
                          "S2's historical parent")
assert len(rows) == 240

completion_path, labels_path, gate_path, gate = freeze_and_hash(rows, ROOT, "SHARED_INPUTS")
print("rows:", len(rows))
print(gate)
