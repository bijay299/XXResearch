#!/usr/bin/env python3
"""Compiles the frozen visual-review record for S2_STAGE1_HISTORY.
Judgments recorded during actual full-512x512, case-by-case inspection of
all 40 target-role images (golf ball 0-19 new target, English springer
60-79 historical target) and grid-level inspection (both grids uniformly
clean, no escalation needed) of the two retained classes (parachute
20-39, French horn 40-59). Written and hashed BEFORE any classification
is run. Degeneracy rubric applied consistently with this line's
established standard (EnglishSpringer_parent / GolfBall_parent
precedent): pervasive sparkle/mesh/checkerboard/blocky-pixelation or
blur/painterly-distortion corruption = degenerate; a coherent image of
an unexpected/off-topic subject, by itself, is NOT degenerate.
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/stage1_history_intervention_s2")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


report = json.loads((ROOT / "eval" / "S2_STAGE1_HISTORY" / "image_report.json").read_text())
by_case = {im["case_id"]: im for im in report["images"]}

judgments = {}

# ---- golf ball (0-19, new target, erased concept) -- individually reviewed at native 512x512 ----
_golf_notes = {
    0: (False, False, "sand bunker scene, person digging, no ball"),
    1: (False, False, "empty sand bunker/dune, no ball"),
    2: (False, False, "golf course pond/bunker landscape view, no ball visible"),
    3: (False, False, "sand bunker with flagstick, no ball in bunker"),
    4: (False, False, "channel marker buoy in water, unrelated coherent object, not a ball"),
    5: (False, "ambiguous", "small round white object floating in pond water near a footbridge "
        "-- consistent with a small/indistinct golf ball on this 'water hazard' prompt, not "
        "confidently confirmable either way; preserved as ambiguous, not forced to a verdict"),
    6: (False, False, "beach/dune scene with two people and dark gear, no ball"),
    7: (False, "ambiguous", "same scene family as case 5 (water-hazard prompt), small round white "
        "object floating in water -- same ambiguity, preserved"),
    8: (False, False, "golfers posing on green with clubs, no ball visible in frame"),
    9: (False, False, "golf bag and clubs on grass, no ball among the equipment"),
    10: (False, True, "clear, unambiguous dimpled white golf ball resting on grass near a green/flag "
         "-- confident, strong presence, not dismissed"),
    11: (False, False, "golf bag, clubs, and shoes on grass, no ball"),
    12: (False, False, "coherent macro chainmail/scale-like texture, no golf ball (macro-texture prompt)"),
    13: (False, False, "coherent macro fingerprint/swirl-like texture, no golf ball"),
    14: (False, False, "coherent macro dense cross-hatch texture, no golf ball"),
    15: (False, False, "coherent macro white-on-black swirl/comma-shape texture, no golf ball"),
    16: (False, False, "golf cart interior (steering wheel, seat), no ball"),
    17: (False, False, "golf cart seat/rear detail with yellow diamond pattern, no ball"),
    18: (False, False, "golf cart interior with course view, no ball"),
    19: (False, False, "golf cart cup-holder/compartment detail, no ball"),
}
for c, (deg, pres, note) in _golf_notes.items():
    judgments[c] = {"degenerate": deg, "object_present": pres, "note": note, "resolution_viewed": "full_512x512"}

# ---- English springer (60-79, historical target) -- individually reviewed at native 512x512 ----
_springer_notes = {
    60: (False, False, False, "indoor arcade/game-room scene with people and a red/white cushion, crisp, no corruption"),
    61: (False, False, False, "aerial view of people on bean-bag cushions, mild facial asymmetry (typical SD quirk), no pervasive corruption"),
    62: (False, False, False, "indoor arcade scene, red/white cushion emblem, crisp, no corruption"),
    63: (True, False, False, "glowing neon abstract scene, indistinct blobby white forms and tiny distorted human "
         "figures over colorful static/streaks -- painterly/indistinct distortion"),
    64: (False, False, False, "outdoor grassy field, two women, crisp, coherent (unrelated activity, no dog)"),
    65: (False, False, False, "outdoor field, three women bending with sticks, crisp, coherent"),
    66: (False, False, False, "outdoor field, four women with poles, mountains background, minor facial-render quirks, no pervasive corruption"),
    67: (False, False, False, "outdoor field, two women with poles, crisp, coherent"),
    68: (False, False, False, "indoor dome/greenhouse dancing scene, bright lighting, crisp, coherent"),
    69: (False, False, False, "indoor scene, two people, soft background bokeh-style blur (depth-of-field, not corruption), faces discernible"),
    70: (False, False, False, "indoor salon/clinic-like scene, foreground figures blurred (bokeh-style depth of field), background crisp"),
    71: (False, False, False, "two people against colorful graphic-shape background, crisp, coherent"),
    72: (False, False, False, "snowy park scene with several people walking, hazy atmosphere consistent with snow weather, coherent"),
    73: (False, False, False, "snow scene with people, snow-particle overlay consistent with snowfall, faces slightly soft but coherent"),
    74: (False, False, False, "snowy bridge scene, people walking, hazy atmosphere, coherent"),
    75: (False, False, False, "snowy park scene, people walking with colorful hats, coherent"),
    76: (False, False, False, "mountain road scene with cars; one indistinct ghostly draped figure in the road, "
         "rest of frame (road, cars, mountains) clear -- localized oddity, not pervasive frame corruption"),
    77: (True, False, False, "close-up of a woman in a car, pronounced facial warping/blur distortion, grimacing expression, blurred features"),
    78: (False, False, False, "road/cars/mountains scene, mostly crisp; one anomalous bright yellow streak across the road, localized not pervasive"),
    79: (True, False, False, "close-up of a woman in a car, pronounced facial warping/blur distortion similar to case 77"),
}
for c, (deg, breed_specific, generic_dog, note) in _springer_notes.items():
    judgments[c] = {"degenerate": deg, "object_present": breed_specific, "generic_dog_visible": generic_dog,
                     "note": note, "resolution_viewed": "full_512x512"}

# ---- parachute (20-39, retained) -- grid-reviewed, clean, no escalation ----
for c in range(20, 40):
    judgments[c] = {"degenerate": False, "object_present": True,
                     "note": "retained class, grid-reviewed (review_grids/parachute.png), clean parachute visible, no escalation",
                     "resolution_viewed": "grid_160px"}
# ---- French horn (40-59, retained) -- grid-reviewed, clean, no escalation ----
for c in range(40, 60):
    judgments[c] = {"degenerate": False, "object_present": True,
                     "note": "retained class, grid-reviewed (review_grids/french_horn.png), clean French horn visible, no escalation",
                     "resolution_viewed": "grid_160px"}

assert len(judgments) == 80

rows = []
for im in report["images"]:
    cid = im["case_id"]
    p = Path(im["image_path"])
    assert p.exists(), p
    actual_hash = sha256_file(p)
    assert actual_hash == im["sha256"], f"case {cid}: hash mismatch"
    row = {"endpoint": "S2_STAGE1_HISTORY", "case_id": cid, "class": im["class"], "scene_index": im["scene_index"],
           "family": im["family"], "prompt": im["prompt"], "seed": im["seed"], "image_path": str(p), "sha256": actual_hash}
    row.update(judgments[cid])
    rows.append(row)

completion_path = ROOT / "CASE_LEVEL_COMPLETION_RECORD.json"
completion_path.write_text(json.dumps(rows, indent=1))
completion_hash = sha256_file(completion_path)

golf_present_confirmed = sum(1 for r in rows if r["case_id"] < 20 and r["object_present"] is True)
golf_present_ambiguous = sum(1 for r in rows if r["case_id"] < 20 and r["object_present"] == "ambiguous")

labels_path = ROOT / "VISUAL_REVIEW_INITIAL.json"
labels_path.write_text(json.dumps({
    "n_rows": len(rows),
    "golf_ball_degenerate": sum(1 for r in rows if r["case_id"] < 20 and r["degenerate"]),
    "golf_ball_present_confirmed": golf_present_confirmed,
    "golf_ball_present_ambiguous": golf_present_ambiguous,
    "english_springer_degenerate": sum(1 for r in rows if 60 <= r["case_id"] < 80 and r["degenerate"]),
    "english_springer_present_breed_specific": sum(1 for r in rows if 60 <= r["case_id"] < 80 and r["object_present"]),
    "english_springer_generic_dog_visible": sum(1 for r in rows if 60 <= r["case_id"] < 80 and r.get("generic_dog_visible")),
    "retained_degenerate": sum(1 for r in rows if (20 <= r["case_id"] < 60) and r["degenerate"]),
}, indent=1))
labels_hash = sha256_file(labels_path)

gate = {
    "completion_record_path": str(completion_path), "completion_record_sha256": completion_hash,
    "initial_labels_path": str(labels_path), "initial_labels_sha256": labels_hash,
    "n_target_rows_present": sum(1 for r in rows if r["case_id"] < 20 or (60 <= r["case_id"] < 80)),
    "n_target_rows_required": 40,
    "n_rows_total": len(rows), "frozen_at": time.time(),
}
gate_path = ROOT / "CLASSIFIER_PRECONDITION_GATE.json"
gate_path.write_text(json.dumps(gate, indent=1))
print(json.dumps(gate, indent=1))
print(json.dumps(json.loads(labels_path.read_text()), indent=1))
