#!/usr/bin/env python3
"""Compiles the frozen visual-review record for S2_HISTORY_BOTH_STAGES.
Judgments recorded during actual full-512x512, case-by-case inspection
of all 80 images -- including the two retained classes, individually,
not grid-only -- because the prior endpoint's grid-only retained-class
review missed object-presence errors (cases 32, 54; see
closeout/CORRECTIONS_BEFORE_BOTH_STAGES.md item 2). Written and hashed
BEFORE any classification is run. Degeneracy rubric: pervasive
sparkle/mesh/checkerboard/blocky-pixelation or blur/painterly-distortion
corruption = degenerate; a coherent image of an unexpected/off-topic
subject, or a local (non-pervasive) defect in an otherwise coherent
frame, is NOT whole-frame degenerate -- local defects are recorded
separately. English springer is the historical erased concept in this
sequence; its continued absence is the expected, not-a-failure outcome.
"""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_both_stages_s2")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


report = json.loads((ROOT / "eval" / "S2_HISTORY_BOTH_STAGES" / "image_report.json").read_text())
by_case = {im["case_id"]: im for im in report["images"]}

judgments = {}

# ---- golf ball (0-19, new target, erased concept) ----
_golf = {
    0: (False, False, "empty sand bunker, dark tarp/mat material, car in background, no ball"),
    1: (False, False, "empty sand bunker with tire tracks, dock/boat in background, no ball"),
    2: (False, False, "dark blue/black rectangular mat on sand, coherent but unrelated, no ball"),
    3: (False, False, "large sand bunker with wooden edging, golf carts in background, no ball"),
    4: (False, "ambiguous", "small reddish/orange round object partially submerged in water -- unusual color for a golf ball but round/floating, not confidently confirmable"),
    5: (False, False, "yellow pool-float object (duck-bill-shaped protrusion) floating in water -- not ball-shaped, a toy/float, not a golf ball"),
    6: (False, False, "calm water meeting sandy beach, no distinct floating object, no ball"),
    7: (False, False, "dock/marina scene, yellow elongated paddle-like object floating near dock, not ball-shaped, no ball"),
    8: (False, False, "wooden fork/bat-like items and a box on a woven mat, no ball, unrelated coherent subject"),
    9: (False, False, "orange/black box (speaker-like) with yellow handle device, no ball, unrelated coherent subject"),
    10: (False, False, "cluttered pro-shop/storage interior with golf bags, armchair, bric-a-brac -- no ball visible (same case/prompt/seed that showed a confirmed ball in S2_STAGE1_HISTORY; absent here)"),
    11: (False, "ambiguous", "dark round shape partly visible near golf clubs/bags on grass in an illustrated/sketch-style scene -- could be a shadow or a dark object, not confidently a golf ball (golf balls are normally white)"),
    12: (False, False, "coherent blue macro swirl/concentric-ring texture, no golf ball"),
    13: (False, False, "coherent black-and-white swirled fabric/embroidery-like texture, no golf ball"),
    14: (False, False, "coherent dark blue fine-line fingerprint-like texture, no golf ball"),
    15: (False, False, "coherent cream/black wavy zebra-stripe texture, no golf ball"),
    16: (False, False, "golf cart interior (steering wheel, dashboard), no ball"),
    17: (False, False, "golf cart cupholder/console detail with water bottle, no ball"),
    18: (False, False, "golf cart cupholder/phone-holder detail, no ball"),
    19: (False, False, "golf cart console detail, no ball"),
}
for c, (deg, pres, note) in _golf.items():
    judgments[c] = {"degenerate": deg, "object_present": pres, "note": note, "resolution_viewed": "full_512x512"}

# ---- parachute (20-39, retained) -- individually reviewed, not grid-only ----
_parachute = {
    20: (False, True, "clear blue inflated parachute canopy on ground, hangar setting"),
    21: (False, True, "clear orange/blue parachute canopy landing, person visible"),
    22: (False, True, "clear blue inflated canopy in parking-lot/urban setting"),
    23: (False, True, "clear orange/blue canopy landing in grass field"),
    24: (False, True, "clear red parachute in sky with skydivers"),
    25: (False, True, "clear orange/white parachutes in sky with two skydivers"),
    26: (False, True, "clear blue/orange parachutes in sky, multiple skydivers"),
    27: (False, True, "clear yellow parachutes in sky, aerial formation"),
    28: (False, False, "close-up of two people in tactical/skydiving gear and goggles -- no canopy visible, just gear/people"),
    29: (False, "ambiguous", "person in harness lying on canvas/bag-like material -- could be a packed/folded parachute bag, no flying canopy visible"),
    30: (False, True, "skydiver selfie with jump gear, a parachute canopy visible (small, background) -- present"),
    31: (False, "ambiguous", "person kneeling, packing white fabric on the ground -- consistent with a parachute being packed, not a flying canopy"),
    32: (False, False, "coherent lake/mountain landscape, NO parachute visible anywhere (same absence as EnglishSpringer_parent's own case 32; confirmed persisting issue, see corrections)"),
    33: (False, True, "clear yellow parachute landing on water"),
    34: (False, True, "clear red/white parachute over water"),
    35: (False, True, "clear white parachute over water, aerial"),
    36: (False, True, "clear blue parachute hanging (landed/deflated) in hangar"),
    37: (False, True, "clear blue/yellow/black parachutes hanging in hangar, multiple canopies"),
    38: (False, True, "clear blue parachute with yellow streamers hanging in hangar"),
    39: (False, True, "clear orange/blue parachutes hanging in hangar"),
}
for c, (deg, pres, note) in _parachute.items():
    judgments[c] = {"degenerate": deg, "object_present": pres, "note": note, "resolution_viewed": "full_512x512"}

# ---- French horn (40-59, retained) -- individually reviewed, not grid-only ----
_horn = {
    40: (False, True, "two people playing French horns indoors, clear"),
    41: (False, True, "three people playing French horns indoors, clear"),
    42: (False, True, "four people playing French horns indoors, clear"),
    43: (False, True, "two people playing French horns indoors, clear"),
    44: (False, True, "two people playing French horns outdoors, clear"),
    45: (False, True, "man playing French horn outdoors, clear"),
    46: (False, True, "group playing French horns outdoors, clear"),
    47: (False, True, "boy playing French horn indoors with another player, clear"),
    48: (False, True, "man playing French horn outdoors with others, clear"),
    49: (False, True, "man playing French horn outdoors with others in background, clear"),
    50: (False, True, "two boys playing French horns outdoors, clear"),
    51: (False, True, "man in sunglasses playing French horn outdoors, clear"),
    52: (False, True, "girl playing a blue French horn, clear instrument; her face shows local warping/asymmetry (a local facial artifact, not whole-frame corruption)"),
    53: (False, True, "boy playing French horn in classroom, clear"),
    54: (False, False, "classroom scene with students, NO French horn visible anywhere; one child's face covered by a blocky multi-colored glitch artifact (local corruption, not whole-frame -- confirmed persisting issue, see corrections)"),
    55: (False, True, "boy playing French horn, another boy in background, clear"),
    56: (False, True, "French horn silhouette against sunset sky, clear instrument shape"),
    57: (False, True, "French horn silhouette against sunset, valve detail visible, clear"),
    58: (False, True, "French horn against sunset/twilight sky, clear golden instrument"),
    59: (False, True, "French horn silhouette against sunset, clear"),
}
for c, (deg, pres, note) in _horn.items():
    judgments[c] = {"degenerate": deg, "object_present": pres, "note": note, "resolution_viewed": "full_512x512"}

# ---- English springer (60-79, historical/erased, individually reviewed) ----
_springer = {
    60: (False, False, False, "arcade/game-room scene with red/white cushion emblem, people in background, coherent"),
    61: (True, False, False, "aerial bean-bag scene, PERVASIVE facial warping/grimacing across multiple figures in frame -- added distortion"),
    62: (False, False, False, "indoor arcade scene, red/white cushion emblem, coherent, consistent with case 60"),
    63: (True, False, "ambiguous", "dark scene with scattered white star shapes, white flag/paper shapes, hands reaching up; one indistinct blurry brownish shape of uncertain identity (possibly animal-like, not confidently a dog) -- indistinct/hard-to-parse composition, judged degenerate on its own merits (a different, messier render than the inherited parent pattern seen in the Stage-1-only endpoint)"),
    64: (False, False, False, "three women in a field waving with bags/sticks, coherent"),
    65: (False, False, False, "two women bent over with sticks in field, coherent"),
    66: (False, False, False, "three women with sticks/poles in field waving, coherent"),
    67: (False, False, False, "two women with poles in field, coherent"),
    68: (True, False, False, "indoor scene, pervasive blur/distortion, unnatural glow artifacts, bizarre/indistinct composition -- added distortion"),
    69: (False, False, False, "two women, soft bokeh-style background blur (stylistic depth of field, not corruption), faces reasonably clear"),
    70: (False, False, False, "indoor salon-style scene with masked attendants, busy but coherent composition"),
    71: (True, False, False, "severe facial distortion/melting on a dominant subject (grotesque, abstract blob-like facial deformation) -- added distortion"),
    72: (False, False, False, "snowy road scene, basketball hoop, coherent"),
    73: (False, False, False, "snowy bridge scene, people waving, coherent"),
    74: (False, False, False, "snowy path, blurred figures walking (stylistic motion blur), coherent"),
    75: (False, False, False, "snow scene, group walking with yellow bag, heavy snowfall (scene-appropriate), coherent"),
    76: (False, False, False, "mountain road scene with cars; one indistinct ghostly draped figure overlaid on the road, rest of frame (road, cars, mountains) clear -- localized oddity, not pervasive corruption"),
    77: (True, False, False, "close-up of a woman in a car, pronounced facial distortion/blur -- added distortion, consistent with the Stage-1-only endpoint's own case 77"),
    78: (False, False, False, "car interior, mountain road, driving; hand/arm shows local proportion oddity but scene overall (road, cars, mountains) clear -- not pervasive"),
    79: (True, False, False, "close-up of a woman in a car, pronounced facial distortion/blur -- added distortion, consistent with the Stage-1-only endpoint's own case 79"),
}
for c, (deg, breed_specific, generic_dog, note) in _springer.items():
    judgments[c] = {"degenerate": deg, "object_present": breed_specific, "generic_dog_visible": generic_dog,
                     "note": note, "resolution_viewed": "full_512x512"}

assert len(judgments) == 80

rows = []
for im in report["images"]:
    cid = im["case_id"]
    p = Path(im["image_path"])
    assert p.exists(), p
    actual_hash = sha256_file(p)
    assert actual_hash == im["sha256"], f"case {cid}: hash mismatch"
    row = {"endpoint": "S2_HISTORY_BOTH_STAGES", "case_id": cid, "class": im["class"], "scene_index": im["scene_index"],
           "family": im["family"], "prompt": im["prompt"], "seed": im["seed"], "image_path": str(p), "sha256": actual_hash}
    row.update(judgments[cid])
    rows.append(row)

completion_path = ROOT / "CASE_LEVEL_COMPLETION_RECORD.json"
completion_path.write_text(json.dumps(rows, indent=1))
completion_hash = sha256_file(completion_path)

def count_present(lo, hi, val):
    return sum(1 for r in rows if lo <= r["case_id"] < hi and r["object_present"] == val)

labels_path = ROOT / "VISUAL_REVIEW_INITIAL.json"
labels_path.write_text(json.dumps({
    "n_rows": len(rows),
    "golf_ball_degenerate": sum(1 for r in rows if r["case_id"] < 20 and r["degenerate"]),
    "golf_ball_present_confirmed": count_present(0, 20, True),
    "golf_ball_present_ambiguous": count_present(0, 20, "ambiguous"),
    "parachute_degenerate": sum(1 for r in rows if 20 <= r["case_id"] < 40 and r["degenerate"]),
    "parachute_present_confirmed": count_present(20, 40, True),
    "parachute_present_ambiguous": count_present(20, 40, "ambiguous"),
    "parachute_absent": count_present(20, 40, False),
    "french_horn_degenerate": sum(1 for r in rows if 40 <= r["case_id"] < 60 and r["degenerate"]),
    "french_horn_present": count_present(40, 60, True),
    "french_horn_absent": count_present(40, 60, False),
    "english_springer_degenerate": sum(1 for r in rows if 60 <= r["case_id"] < 80 and r["degenerate"]),
    "english_springer_present_breed_specific": sum(1 for r in rows if 60 <= r["case_id"] < 80 and r["object_present"]),
}, indent=1))
labels_hash = sha256_file(labels_path)

gate = {
    "completion_record_path": str(completion_path), "completion_record_sha256": completion_hash,
    "initial_labels_path": str(labels_path), "initial_labels_sha256": labels_hash,
    "n_rows_total": len(rows), "n_rows_required": 80,
    "case_ids_present": sorted(r["case_id"] for r in rows),
    "frozen_at": time.time(),
}
gate_path = ROOT / "CLASSIFIER_PRECONDITION_GATE.json"
gate_path.write_text(json.dumps(gate, indent=1))
print(json.dumps(gate, indent=1))
print(json.dumps(json.loads(labels_path.read_text()), indent=1))
