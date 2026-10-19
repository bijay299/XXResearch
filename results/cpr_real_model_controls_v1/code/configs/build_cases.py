#!/usr/bin/env python3
"""Build the frozen case matrix. Deterministic; emits JSON only. No model use.

Object boundaries are frozen here. Allowed panels deliberately contain
RELATED BUT NON-TARGET objects and ordinary contexts, and deliberately EXCLUDE
near-synonyms of the target (which would make removal and preservation
contradictory).
"""
import hashlib
import json
import os

ROOT = os.environ["CPR_ROOT"]
OUT = os.path.join(ROOT, "configs")

ARMS = ["native_gem", "cpr", "centered_radial", "margin_truncated",
        "anchor_only", "axial_margin"]

# ---------------------------------------------------------------- objects
DEV_TARGET = "backpack"
HELDOUT_TARGETS = ["bicycle", "violin"]

# Near-synonyms excluded from every allowed panel for each target, recorded so
# the boundary is auditable rather than implicit.
EXCLUDED_NEAR_SYNONYMS = {
    "backpack": ["rucksack", "knapsack", "daypack", "bookbag", "packsack"],
    "bicycle":  ["bike", "cycle", "pushbike", "two-wheeler", "ten-speed"],
    "violin":   ["fiddle", "viola"],
}

DEV_ALLOWED = [
    "suitcase", "handbag", "umbrella", "laptop", "water bottle", "tent",
    "camera", "coffee mug", "skateboard", "running shoe", "wristwatch",
    "folding chair",
]
HELDOUT_ALLOWED = {
    # related-but-distinct first, then ordinary contexts
    "bicycle": ["motorcycle", "scooter", "wheelbarrow", "shopping cart",
                "skateboard", "wagon", "stroller", "lawn mower", "ladder",
                "park bench", "mailbox", "street lamp", "fire hydrant",
                "picnic table", "garden hose", "watering can", "toolbox",
                "bird house", "flower pot", "trash bin"],
    "violin":  ["cello", "guitar", "flute", "trumpet", "piano", "drum",
                "harp", "accordion", "music stand", "metronome",
                "sheet music", "bookshelf", "table lamp", "armchair",
                "vase of flowers", "wall clock", "teapot", "candlestick",
                "wicker basket", "ceramic bowl"],
}

# ---------------------------------------------------------------- templates
TARGET_TEMPLATES = [
    "a photo of a {o}", "a {o} on a wooden table", "a {o} in a sunlit room",
    "a close-up photo of a {o}", "a {o} on the grass", "a {o} against a plain wall",
    "a {o} on a city sidewalk", "a {o} in a studio photograph",
    "a {o} on a bed", "a {o} next to a window", "a {o} outdoors on a clear day",
    "a detailed photograph of a {o}",
    "a {o} on a tiled floor", "a {o} in a shop display",
    "a {o} on a park bench", "a {o} under soft lighting",
    "a {o} on a concrete step", "a {o} in a quiet hallway",
    "a {o} on a kitchen counter", "a {o} beside a doorway",
]
ALLOWED_TEMPLATES = list(TARGET_TEMPLATES)

# ---------------------------------------------------------------- seeds
SEED_BASE = 777000
TRAIN_SEED_DEV = 11
TRAIN_SEEDS_HELDOUT = [21, 22]
EVAL_SEED_BASE = 900000          # evaluation seeds are DISTINCT from training seeds
CALIB_SEED_BASE = 950000


def mk(case_id, kind, prompt, seed, panel, obj):
    return {"case_id": case_id, "kind": kind, "object": obj, "panel": panel,
            "prompt": prompt, "eval_seed": seed}


def dev_cases():
    cases = []
    for i in range(12):
        cases.append(mk(f"dev.target.{i:02d}", "target",
                        TARGET_TEMPLATES[i].format(o=DEV_TARGET),
                        EVAL_SEED_BASE + i, "dev_target", DEV_TARGET))
    for i, obj in enumerate(DEV_ALLOWED):
        cases.append(mk(f"dev.allowed.{i:02d}", "allowed",
                        ALLOWED_TEMPLATES[i].format(o=obj),
                        EVAL_SEED_BASE + 100 + i, "dev_allowed", obj))
    return cases


def heldout_cases():
    cases = []
    for t_i, tgt in enumerate(HELDOUT_TARGETS):
        for i in range(20):
            cases.append(mk(f"ho.{tgt}.target.{i:02d}", "target",
                            TARGET_TEMPLATES[i].format(o=tgt),
                            EVAL_SEED_BASE + 1000 + 100 * t_i + i,
                            f"ho_{tgt}_target", tgt))
        for i, obj in enumerate(HELDOUT_ALLOWED[tgt]):
            cases.append(mk(f"ho.{tgt}.allowed.{i:02d}", "allowed",
                            ALLOWED_TEMPLATES[i].format(o=obj),
                            EVAL_SEED_BASE + 2000 + 100 * t_i + i,
                            f"ho_{tgt}_allowed", obj))
    return cases


def calibration_cases():
    """64 SEPARATE development calibration cases for scorer qualification:
    32 intended target-PRESENT and 32 unrelated negatives. These are generated
    from pristine M0 only and are never part of the scientific matrix."""
    present_objs = [DEV_TARGET] * 16 + ["bicycle"] * 8 + ["violin"] * 8
    negatives = ["empty wooden table", "a cloudy sky", "a brick wall",
                 "a bowl of fruit", "an empty armchair", "a stack of books",
                 "a potted plant", "a kitchen sink", "a sandy beach",
                 "a forest path", "a plate of bread", "an open notebook",
                 "a glass of water", "a wooden fence", "a tiled bathroom",
                 "a desk with a lamp"]
    cases = []
    for i, o in enumerate(present_objs):
        cases.append({"case_id": f"cal.pos.{i:02d}", "kind": "calib_positive",
                      "object": o, "panel": "calibration",
                      "prompt": TARGET_TEMPLATES[i % len(TARGET_TEMPLATES)].format(o=o),
                      "eval_seed": CALIB_SEED_BASE + i,
                      "intended_present": True, "probe_object": o})
    for i, p in enumerate(negatives * 2):
        # negatives are probed for the DEV target and the two held-out targets
        probe = [DEV_TARGET, "bicycle", "violin"][i % 3]
        cases.append({"case_id": f"cal.neg.{i:02d}", "kind": "calib_negative",
                      "object": None, "panel": "calibration",
                      "prompt": f"a photo of {p}",
                      "eval_seed": CALIB_SEED_BASE + 500 + i,
                      "intended_present": False, "probe_object": probe})
    return cases


def main():
    dev = dev_cases()
    ho = heldout_cases()
    cal = calibration_cases()

    # learning rates: resolved native and half
    lrs = {"native": 1e-3, "half": 5e-4}
    dev_traj = [{"traj_id": f"dev.{arm}.{lrn}", "arm": arm, "lr": lr, "lr_name": lrn,
                 "target": DEV_TARGET, "train_seed": TRAIN_SEED_DEV}
                for arm in ARMS for lrn, lr in lrs.items()]
    ho_traj = [{"traj_id": f"ho.{arm}.{tgt}.s{s}", "arm": arm, "target": tgt,
                "train_seed": s}
               for arm in ARMS for tgt in HELDOUT_TARGETS for s in TRAIN_SEEDS_HELDOUT]

    # checkpoints: first completed optimizer update reaching 25% / 50% of the
    # 250-timestep allowance, plus the actual native terminal endpoint.
    checkpoints = {"pct25": 63, "pct50": 125, "terminal": 250}

    counts = {
        "dev_trajectories": len(dev_traj),                       # 12
        "dev_checkpoints_per_traj": 3,
        "dev_cases_per_checkpoint": len(dev),                    # 24
        "dev_edited_images": len(dev_traj) * 3 * len(dev),       # 864
        "dev_m0_images": len(dev),                               # 24
        "dev_total": len(dev_traj) * 3 * len(dev) + len(dev),    # 888
        "heldout_endpoints": len(ho_traj),                       # 24
        "heldout_cases_per_endpoint": 40,
        "heldout_edited_images": len(ho_traj) * 40,              # 960
        "heldout_m0_images": len(ho),                            # 80
        "heldout_total": len(ho_traj) * 40 + len(ho),            # 1040
        "calibration_images": len(cal),                          # 64
    }
    counts["primary_scientific_total"] = counts["dev_total"] + counts["heldout_total"]
    counts["declared_plus_calibration"] = counts["primary_scientific_total"] + counts["calibration_images"]
    counts["technical_retry_allowance"] = 2200 - counts["declared_plus_calibration"]

    assert counts["dev_total"] == 888, counts
    assert counts["heldout_total"] == 1040, counts
    assert counts["primary_scientific_total"] == 1928, counts
    assert counts["calibration_images"] == 64, counts
    assert counts["technical_retry_allowance"] == 208, counts

    # no allowed-panel object may be the target or one of its near-synonyms
    for tgt, panel in [(DEV_TARGET, DEV_ALLOWED)] + \
                      [(t, HELDOUT_ALLOWED[t]) for t in HELDOUT_TARGETS]:
        banned = {tgt} | set(EXCLUDED_NEAR_SYNONYMS[tgt])
        clash = banned & set(panel)
        assert not clash, f"allowed panel for {tgt} collides with {clash}"
    # dev allowed panel must not contain a held-out target (cross-contamination)
    assert not set(DEV_ALLOWED) & set(HELDOUT_TARGETS)

    # unique eval seeds everywhere; training and evaluation seeds disjoint
    all_seeds = [c["eval_seed"] for c in dev + ho + cal]
    assert len(set(all_seeds)) == len(all_seeds), "duplicate eval seed"
    assert not set(all_seeds) & set([TRAIN_SEED_DEV] + TRAIN_SEEDS_HELDOUT)

    bundle = {
        "arms": ARMS,
        "dev_target": DEV_TARGET,
        "heldout_targets": HELDOUT_TARGETS,
        "excluded_near_synonyms": EXCLUDED_NEAR_SYNONYMS,
        "dev_allowed_objects": DEV_ALLOWED,
        "heldout_allowed_objects": HELDOUT_ALLOWED,
        "learning_rates": lrs,
        "checkpoints_processed_timesteps": checkpoints,
        "train_seed_dev": TRAIN_SEED_DEV,
        "train_seeds_heldout": TRAIN_SEEDS_HELDOUT,
        "dev_cases": dev,
        "heldout_cases": ho,
        "calibration_cases": cal,
        "dev_trajectories": dev_traj,
        "heldout_trajectories": ho_traj,
        "counts": counts,
    }
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "cases.json")
    blob = json.dumps(bundle, indent=2, sort_keys=True)
    with open(p, "w") as f:
        f.write(blob)
    print(f"wrote {p}")
    print("sha256:", hashlib.sha256(blob.encode()).hexdigest())
    for k, v in counts.items():
        print(f"  {k:34s} {v}")


if __name__ == "__main__":
    main()
