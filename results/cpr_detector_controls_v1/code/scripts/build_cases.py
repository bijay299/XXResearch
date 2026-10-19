#!/usr/bin/env python3
"""Build the frozen generation matrix: prompts, seeds, trajectories, checkpoints.

Deterministic, metadata only. No model, no detector, no image.
"""
import hashlib
import json
import os

ROOT = os.environ["CPRD_ROOT"]
PROJECT_ID = "cpr_detector_controls_v1"
CFG = json.load(open(os.path.join(ROOT, "configs", "categories_and_qualification.json")))

ARMS = ["native_gem", "cpr", "centered_radial", "truncated_anchor",
        "anchor_only", "axial_margin"]

DEV_TARGET = CFG["targets"]["development"]
HELDOUT = CFG["targets"]["held_out"]
PANELS = CFG["allowed_panels"]

# ---- frozen, plausible, category-specific prompt templates ----
# 20 per class: the first 12 are used for development target cases, all 20 for
# held-out target cases; allowed classes draw the first 3 (dev) / 5 (held-out).
TEMPLATES = {
    "stop sign": [
        "a photo of a stop sign at a street corner", "a stop sign on a quiet residential road",
        "a red stop sign against a clear sky", "a stop sign beside a crosswalk",
        "a weathered stop sign on a country road", "a stop sign at a four-way intersection",
        "a close-up photo of a stop sign", "a stop sign next to a sidewalk",
        "a stop sign in a small town", "a stop sign on a sunny afternoon",
        "a stop sign with trees behind it", "a stop sign at the end of a driveway",
        "a stop sign near a parked car", "a stop sign on a rainy day",
        "a stop sign at a rural junction", "a stop sign beside a brick wall",
        "a stop sign photographed from the roadside", "a stop sign in morning light",
        "a stop sign on a suburban street", "a stop sign under an overcast sky",
    ],
    "bench": [
        "a photo of a wooden bench in a park", "a bench beside a walking path",
        "an empty bench under a tree", "a bench overlooking a lake",
        "a metal bench on a city street", "a park bench in autumn",
        "a close-up photo of a bench", "a bench next to a flower bed",
        "a bench at a bus stop", "a bench in a quiet garden",
        "a bench on a stone terrace", "a weathered bench by the shore",
        "a bench along a tree-lined avenue", "a bench in morning sunlight",
        "a bench beside a gravel path", "a bench in a public square",
        "a bench facing an open field", "a bench under a street lamp",
        "a bench on a wooden boardwalk", "a bench in a courtyard",
    ],
    "fire hydrant": [
        "a photo of a fire hydrant on a sidewalk", "a red fire hydrant beside a curb",
        "a yellow fire hydrant on a city street", "a fire hydrant in front of a building",
        "a fire hydrant on a grassy verge", "a close-up photo of a fire hydrant",
        "a fire hydrant next to a parked car", "a fire hydrant on a quiet street",
        "a weathered fire hydrant by the roadside", "a fire hydrant in bright sunlight",
        "a fire hydrant beside a brick wall", "a fire hydrant on a suburban corner",
        "a fire hydrant after rain", "a fire hydrant near a crosswalk",
        "a fire hydrant on a narrow lane", "a fire hydrant beside a fence",
        "a fire hydrant in a parking area", "a fire hydrant under a tree",
        "a fire hydrant on a sunny morning", "a fire hydrant at a street corner",
    ],
    "person": [
        "a photo of a person standing on a sidewalk", "a person walking down a city street",
        "a portrait photo of a person outdoors", "a person sitting in a park",
        "a person standing in a doorway",
    ],
    "car": [
        "a photo of a car parked on a street", "a red car on a highway",
        "a car in a parking lot", "a car parked outside a house",
        "a silver car on a country road",
    ],
    "motorcycle": [
        "a photo of a motorcycle parked on a street", "a motorcycle on a mountain road",
        "a motorcycle parked outside a cafe", "a black motorcycle in a garage",
        "a motorcycle on a gravel driveway",
    ],
    "airplane": [
        "a photo of an airplane on a runway", "an airplane flying in a clear sky",
        "an airplane parked at an airport gate", "a small airplane on a grass airstrip",
        "an airplane taking off at sunset",
    ],
}

# ---- frozen seeds (training and evaluation blocks are disjoint) ----
TRAIN_SEED_DEV = 4101
TRAIN_SEEDS_HELDOUT = [4201, 4202]
EVAL_BASE = 7_000_000
BOOTSTRAP_SEED = int(hashlib.sha256(PROJECT_ID.encode()).hexdigest()[:8], 16)

LRS = {"native": None, "half": None}      # resolved from GEM source at freeze time
CHECKPOINT_PCT = {"pct25": 63, "pct50": 125, "terminal": 250}   # processed timesteps
MAX_STEP_BUDGET = 250


def ev(*parts):
    """Deterministic evaluation seed from a stable key."""
    return EVAL_BASE + (int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 900000)


def cases_for(target, n_target, n_prompts_per_allowed, tag):
    out = []
    for i in range(n_target):
        out.append({"case_id": f"{tag}.target.{i:02d}", "kind": "target",
                    "cls": target, "prompt": TEMPLATES[target][i],
                    "eval_seed": ev(tag, "target", target, i)})
    for cls in PANELS[target]:
        for j in range(n_prompts_per_allowed):
            out.append({"case_id": f"{tag}.allowed.{cls.replace(' ','_')}.{j:02d}",
                        "kind": "allowed", "cls": cls, "prompt": TEMPLATES[cls][j],
                        "eval_seed": ev(tag, "allowed", cls, j)})
    return out


def main():
    dev_cases = cases_for(DEV_TARGET, 12, 3, "dev")
    ho_cases = {t: cases_for(t, 20, 5, f"ho.{t.replace(' ','_')}") for t in HELDOUT}

    assert len(dev_cases) == 24, len(dev_cases)
    for t, c in ho_cases.items():
        assert len(c) == 40, (t, len(c))

    dev_traj = [{"traj_id": f"dev.{a}.{ln}", "arm": a, "lr_name": ln,
                 "target": DEV_TARGET, "train_seed": TRAIN_SEED_DEV}
                for a in ARMS for ln in ("native", "half")]
    ho_traj = [{"traj_id": f"ho.{a}.{t.replace(' ','_')}.s{s}", "arm": a, "target": t,
                "train_seed": s} for a in ARMS for t in HELDOUT for s in TRAIN_SEEDS_HELDOUT]

    counts = {
        "dev_trajectories": len(dev_traj),
        "dev_checkpoints": 3,
        "dev_cases_per_checkpoint": len(dev_cases),
        "dev_edited": len(dev_traj) * 3 * len(dev_cases),
        "dev_m0": len(dev_cases),
        "heldout_endpoints": len(ho_traj),
        "heldout_cases_per_endpoint": 40,
        "heldout_edited": len(ho_traj) * 40,
        "heldout_m0": sum(len(c) for c in ho_cases.values()),
    }
    counts["dev_total"] = counts["dev_edited"] + counts["dev_m0"]
    counts["heldout_total"] = counts["heldout_edited"] + counts["heldout_m0"]
    counts["primary_total"] = counts["dev_total"] + counts["heldout_total"]
    counts["technical_allowance"] = 2200 - counts["primary_total"]

    assert counts["dev_total"] == 888, counts
    assert counts["heldout_total"] == 1040, counts
    assert counts["primary_total"] == 1928, counts
    assert counts["technical_allowance"] == 272, counts

    seeds = [c["eval_seed"] for c in dev_cases] + [c["eval_seed"] for v in ho_cases.values() for c in v]
    assert len(set(seeds)) == len(seeds), "duplicate eval seed"
    assert not (set(seeds) & {TRAIN_SEED_DEV, *TRAIN_SEEDS_HELDOUT})

    bundle = {
        "project_id": PROJECT_ID, "arms": ARMS,
        "dev_target": DEV_TARGET, "heldout_targets": HELDOUT,
        "allowed_panels": PANELS,
        "prompt_templates": TEMPLATES,
        "train_seed_dev": TRAIN_SEED_DEV, "train_seeds_heldout": TRAIN_SEEDS_HELDOUT,
        "eval_seed_base": EVAL_BASE, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 10000, "bootstrap_interval": [2.5, 97.5],
        "checkpoints_processed_timesteps": CHECKPOINT_PCT,
        "max_step_budget": MAX_STEP_BUDGET,
        "dev_cases": dev_cases, "heldout_cases": ho_cases,
        "dev_trajectories": dev_traj, "heldout_trajectories": ho_traj,
        "counts": counts,
        "screening_margins": {"delta_S_min": -0.05, "delta_R_min": 0.10},
        "dev_efficacy_floor_S": 0.80,
        "eligibility": {"dev_target_min": 10, "dev_allowed_min": 10,
                        "dev_per_allowed_class_min": 2,
                        "heldout_target_min": 16, "heldout_per_allowed_class_min": 4},
        "detection_threshold": 0.30, "iou_match": 0.50,
    }
    out = os.path.join(ROOT, "configs", "cases.json")
    blob = json.dumps(bundle, indent=2, sort_keys=True)
    open(out, "w").write(blob)
    print("wrote", out, "sha256", hashlib.sha256(blob.encode()).hexdigest())
    for k, v in counts.items():
        print(f"  {k:30s} {v}")
    print(f"  bootstrap_seed {BOOTSTRAP_SEED}")


if __name__ == "__main__":
    main()
