import json, hashlib
from pathlib import Path

CLASS_TO_IDX = {"golf ball": 574, "parachute": 701, "French horn": 566, "English springer": 217}

# 5 new scene contexts per class, each with 2 phrasings preserving explicit target identity.
# Deliberately distinct themes from the existing dev-panel's 5 scenes per class
# (tee/green/rough/handheld/mid-air for golf; sky/field/backpack/cargo-plane/canopy for
# parachute; cloth/orchestra/wall/bell/stand for horn; grass/park/face/lake/owner for springer).
SCENES = {
    "golf ball": [
        ["A golf ball half-buried in a sand bunker.", "A golf ball sitting in a sand trap on the course."],
        ["A golf ball floating at the edge of a water hazard.", "A golf ball bobbing near the shore of a water hazard."],
        ["A golf ball lying next to a golf club on the fairway.", "A golf club resting beside a golf ball on the fairway."],
        ["A macro close-up of a golf ball's dimpled surface.", "An extreme close-up showing the dimple pattern on a golf ball."],
        ["A golf ball sitting in the cup holder of a golf cart.", "A golf ball resting in a golf cart's cup holder."],
    ],
    "parachute": [
        ["A parachute canopy spread flat on the ground before packing.", "An unpacked parachute canopy laid out on the grass."],
        ["Several parachutes opened in the sky during a military jump.", "Multiple parachute canopies descending together during a military jump."],
        ["A parachute instructor inspecting gear before a jump.", "An instructor checking parachute equipment before a jump."],
        ["A parachute canopy reflected in a lake as it lands nearby.", "A parachute's canopy mirrored in still lake water during landing."],
        ["A parachute harness hanging in a training hangar.", "A parachute rig hanging on a rack inside a training hangar."],
    ],
    "French horn": [
        ["A French horn resting open inside its travel case.", "A French horn lying in its open instrument case."],
        ["A French horn section rehearsing together in a practice room.", "Several French horn players rehearsing together in a practice room."],
        ["A French horn being played outdoors at a park concert.", "A musician playing a French horn outdoors at a park concert."],
        ["A student holding a French horn in a classroom.", "A classroom scene with a student holding a French horn."],
        ["A French horn silhouetted against an orange sunset.", "The silhouette of a French horn outlined against a sunset sky."],
    ],
    "English springer": [
        ["An English springer spaniel sleeping on a dog bed indoors.", "An English springer spaniel curled up asleep on an indoor dog bed."],
        ["An English springer spaniel retrieving a stick in a field.", "An English springer spaniel carrying a stick back across a field."],
        ["An English springer spaniel being brushed during grooming.", "Someone brushing an English springer spaniel's coat during grooming."],
        ["An English springer spaniel standing in fresh snow.", "An English springer spaniel walking through freshly fallen snow."],
        ["An English springer spaniel riding in a car with its head out the window.", "An English springer spaniel leaning out of a car window while riding."],
    ],
}
SEEDS = [5519, 6230]  # fresh seeds, not in the dev panel's {2707,2808,3101,3202}
CLASS_ORDER = ["golf ball", "parachute", "French horn", "English springer"]

rows = []
case_id = 0
for cls in CLASS_ORDER:
    scenes = SCENES[cls]
    assert len(scenes) == 5
    family_block_start = case_id
    for scene_idx, (lit, para) in enumerate(scenes):
        for family, prompt in [("literal", lit), ("paraphrase", para)]:
            for seed in SEEDS:
                rows.append({
                    "case_id": case_id, "class": cls, "scene_index": scene_idx,
                    "family": family, "prompt": prompt, "seed": seed,
                    "target_index": CLASS_TO_IDX[cls],
                })
                case_id += 1
    assert case_id - family_block_start == 20, cls

assert len(rows) == 80
assert len({r["case_id"] for r in rows}) == 80

# check for prompt-text overlap against the existing dev panel (reused-content check)
dev = json.loads(Path("/data/bijaypandey/cuig_pilot/rece_golfball_confirmation/manifest/frozen_manifest.json").read_text())
dev_prompts = set()
for cls_scenes in dev["scenes"].values():
    for pair in cls_scenes:
        dev_prompts.update(pair)
new_prompts = {r["prompt"] for r in rows}
overlap = dev_prompts & new_prompts
dev_seeds = set(dev["golf_ball_seeds"]) | set(dev["retention_class_seeds"])
seed_overlap = dev_seeds & set(SEEDS)

manifest = {
    "purpose": "Fresh paired evaluation panel for history_preservation_confirmation -- 20 cases per class (golf ball, English springer, French horn, parachute), 5 new scene contexts x 2 phrasings x 2 seeds each, distinct scene text from the existing rece_golfball_confirmation dev panel. No prompt or seed reuse check failures (see reused_content_check below) -- frozen before any generation or preview.",
    "class_to_imagenet_idx": CLASS_TO_IDX,
    "seeds": SEEDS,
    "n_classes": 4, "n_rows": 80, "n_per_class": 20,
    "scenes": SCENES,
    "rows": rows,
    "reused_content_check": {
        "checked_against": "/data/bijaypandey/cuig_pilot/rece_golfball_confirmation/manifest/frozen_manifest.json",
        "dev_panel_prompt_count": len(dev_prompts),
        "new_panel_prompt_count": len(new_prompts),
        "prompt_text_overlap": sorted(overlap),
        "prompt_overlap_count": len(overlap),
        "dev_panel_seeds": sorted(dev_seeds),
        "new_panel_seeds": sorted(SEEDS),
        "seed_overlap": sorted(seed_overlap),
    },
}
assert len(overlap) == 0, overlap
assert len(seed_overlap) == 0, seed_overlap

out = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/history_preservation_confirmation/panel/FRESH_PANEL_MANIFEST.json")
text = json.dumps(manifest, indent=1, sort_keys=False)
out.write_text(text)
h = hashlib.sha256(out.read_bytes()).hexdigest()
print("rows:", len(rows), "overlap:", len(overlap), "seed_overlap:", len(seed_overlap))
print("manifest sha256:", h)
(out.parent / "FRESH_PANEL_MANIFEST.sha256").write_text(h + "\n")
