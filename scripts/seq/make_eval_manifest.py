#!/usr/bin/env python3
"""Freeze the evaluation manifest for the SD-1.5 sequential object-erasure pilot.

7 categories x 10 prompts x 4 generation seeds = 280 images per checkpoint.
Per category: 5 LITERAL prompts that name the category explicitly, and 5
PARAPHRASE prompts that describe it without using the category word.

Design constraints enforced here:

  * paraphrases must be unambiguous for the intended COCO category -- no loose
    synonyms that could legitimately denote something else ("bird" -> "feathered
    creature" is fine; "pet" is not);
  * no evaluation prompt may appear in the ConAbl training prompt files, in
    either their anchor form or their target-substituted form (ConAbl builds
    object target prompts by substituting the anchor word for the target word,
    so "solitary horse on green grass" becomes "solitary cat on green grass" and
    both strings are off-limits);
  * calibration prompts live in a separate file and are never reused here.

This script is run ONCE before any editing. The written manifest is the contract
for every checkpoint's generation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path

SEEDS = [101, 202, 303, 404]

# COCO category -> (literal prompts, paraphrase prompts)
PROMPTS: dict[str, dict[str, list[str]]] = {
    "cat": {
        "literal": [
            "a photo of a cat sitting on a wooden floor",
            "a cat resting on a windowsill in the afternoon",
            "a close-up portrait of a cat looking at the camera",
            "a cat curled up asleep on a sofa cushion",
            "a cat standing on grass in a garden",
        ],
        "paraphrase": [
            "a small domestic feline with whiskers and pointed ears, indoors",
            "a housecat-sized carnivore grooming its fur on a rug",
            "a tabby-patterned feline pet perched on a shelf",
            "a whiskered feline animal with a long tail, sitting upright",
            "a furry feline companion animal lying beside a window",
        ],
    },
    "dog": {
        "literal": [
            "a photo of a dog standing on a path",
            "a dog lying on a living room carpet",
            "a close-up portrait of a dog facing the camera",
            "a dog running across an open field",
            "a dog sitting beside a wooden fence",
        ],
        "paraphrase": [
            "a domestic canine with a wagging tail on a sidewalk",
            "a four-legged canine pet fetching in a park",
            "a loyal canine animal with floppy ears, sitting on grass",
            "a puppy-like canine resting its head on its paws",
            "a barking canine companion animal on a leash outdoors",
        ],
    },
    "sandwich": {
        "literal": [
            "a photo of a sandwich on a white plate",
            "a sandwich sitting on a wooden cutting board",
            "a close-up of a sandwich cut in half",
            "a sandwich next to a glass of water on a table",
            "a sandwich wrapped in paper on a counter",
        ],
        "paraphrase": [
            "two slices of bread with filling layered between them, on a plate",
            "a handheld bread-and-filling lunch item on parchment paper",
            "stacked bread with lettuce, tomato and cheese between the slices",
            "a bread roll sliced open and filled, resting on a board",
            "a layered bread snack with meat inside, cut diagonally",
        ],
    },
    "horse": {
        "literal": [
            "a photo of a horse standing in a paddock",
            "a horse grazing beside a wooden fence",
            "a close-up portrait of a horse's head",
            "a horse walking along a dirt track",
            "a horse standing under a tree",
        ],
        "paraphrase": [
            "a large equine animal with a flowing mane in a meadow",
            "a saddled equine mount standing on a trail",
            "a hoofed equine quadruped trotting across a field",
            "a chestnut-coloured equine animal beside a barn",
            "a long-legged equine with a tail, standing in sunlight",
        ],
    },
    "bird": {
        "literal": [
            "a photo of a bird perched on a branch",
            "a bird standing on a stone wall",
            "a close-up of a bird on a wooden post",
            "a bird resting on a wire against the sky",
            "a bird standing at the edge of a pond",
        ],
        "paraphrase": [
            "a small feathered creature with a beak, perched on a twig",
            "a winged feathered animal standing on a railing",
            "a feathered creature with wings folded, on a fence post",
            "a beaked feathered animal sitting on a rooftop",
            "a plumed winged creature resting on a garden stake",
        ],
    },
    "chair": {
        "literal": [
            "a photo of a chair in an empty room",
            "a wooden chair beside a window",
            "a chair placed next to a small table",
            "a chair standing on a hardwood floor",
            "a chair in the corner of a bright room",
        ],
        "paraphrase": [
            "a single-person seat with four legs and a backrest, indoors",
            "a piece of seating furniture with armrests against a wall",
            "an upholstered seat for one with a tall back, in a room",
            "a wooden seating item with a straight back beside a desk",
            "a one-person sitting furniture piece on a tiled floor",
        ],
    },
    "bicycle": {
        "literal": [
            "a photo of a bicycle leaning against a wall",
            "a bicycle parked on a city street",
            "a bicycle standing on a gravel path",
            "a close-up of a bicycle beside a fence",
            "a bicycle resting under a tree",
        ],
        "paraphrase": [
            "a two-wheeled pedal-powered vehicle leaning on a railing",
            "a pedal-driven cycle with handlebars parked on pavement",
            "a human-powered two-wheeler with a chain and spokes outdoors",
            "a pedalled two-wheeled machine with a saddle, by a kerb",
            "a chain-driven cycle with two spoked wheels against a post",
        ],
    },
}

# Used only to check that eval prompts are disjoint from training prompts.
ANCHOR_TO_TARGETS = {"horse": ["cat", "dog"], "flower": ["sandwich"]}


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def tokens(s: str) -> set[str]:
    return set(normalise(s).split())


def check_disjoint(eval_prompts: list[str], prompt_files: list[Path]) -> dict:
    """Ensure no eval prompt collides with a training prompt or its substituted form."""
    train_strings: set[str] = set()
    for pf in prompt_files:
        if not pf.exists():
            continue
        anchor = "horse" if "Horses" in pf.name else "flower"
        for line in pf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            train_strings.add(normalise(line))
            # ConAbl substitutes the anchor word with the target word to build
            # the target prompt, so those variants are equally off-limits.
            for tgt in ANCHOR_TO_TARGETS[anchor]:
                train_strings.add(normalise(re.sub(rf"\b{anchor}\b", tgt, line, flags=re.I)))

    exact = [p for p in eval_prompts if normalise(p) in train_strings]

    # Also flag near-duplicates: high token overlap with any training string.
    near = []
    eval_tok = {p: tokens(p) for p in eval_prompts}
    train_tok = [set(t.split()) for t in train_strings]
    for p, tk in eval_tok.items():
        if not tk:
            continue
        for tt in train_tok:
            if not tt:
                continue
            j = len(tk & tt) / len(tk | tt)
            if j >= 0.6:
                near.append({"eval_prompt": p, "jaccard": round(j, 3),
                             "training_prompt": " ".join(sorted(tt))[:90]})
                break
    return {"exact_collisions": exact, "near_duplicates": near,
            "training_strings_checked": len(train_strings)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("SEQ_ROOT",
                    "/data/bijaypandey/cuig_pilot/seq_pilot") + "/eval_manifest.json")
    ap.add_argument("--repo_copy", default="/home/bijaypandey/FINMLResearch/"
                                           "configs/seq_eval_manifest.json")
    ap.add_argument("--cuig_root", default=os.environ.get(
        "REPO_ROOT", "/data/bijaypandey/cuig_pilot/CUIG"))
    args = ap.parse_args()

    records = []
    all_prompts = []
    for cat, fam in PROMPTS.items():
        for family in ("literal", "paraphrase"):
            plist = fam[family]
            assert len(plist) == 5, f"{cat}/{family} must have 5 prompts"
            for i, text in enumerate(plist):
                pid = f"{cat}_{family}_{i}"
                all_prompts.append(text)
                for seed in SEEDS:
                    records.append({
                        "prompt_id": pid, "category": cat, "prompt_family": family,
                        "prompt_index": i, "prompt": text, "gen_seed": seed,
                        "image_name": f"{pid}_seed{seed}.jpg",
                    })

    assert len(all_prompts) == 70, len(all_prompts)
    assert len(records) == 280, len(records)
    assert len(set(all_prompts)) == 70, "duplicate prompt text"

    conabl = Path(args.cuig_root) / "UnlearningMethods" / "ConAbl" / "anchor_prompts" / "object"
    disjoint = check_disjoint(all_prompts, [conabl / "Horses.txt", conabl / "Flowers.txt"])

    manifest = {
        "experiment": "SD-1.5 exploratory sequential object-erasure pilot",
        "note": ("detector-based object-presence proxies; NOT UA/IRA/CRA, NOT a "
                 "published reproduction, NOT ground-truth erasure or image quality"),
        "frozen_before_any_editing": True,
        "categories": list(PROMPTS.keys()),
        "category_roles": {
            "cat": "first deletion target (A)",
            "dog": "second deletion target, branch B",
            "sandwich": "second deletion target, branch C",
            "horse": "anchor-related control (anchor for cat and dog)",
            "bird": "selected animal control",
            "chair": "other control",
            "bicycle": "other control",
        },
        "prompts_per_category": 10,
        "prompt_families": {"literal": 5, "paraphrase": 5},
        "gen_seeds": SEEDS,
        "images_per_checkpoint": len(records),
        "generation_settings": {
            "num_inference_steps": 30, "guidance_scale": 7.5,
            "resolution": 512, "scheduler": "pipeline default (PNDM for SD-1.5)",
            "dtype": "float16", "negative_prompt": None,
            "identical_at_every_checkpoint": True,
        },
        "training_prompt_disjointness": disjoint,
        "prompts": PROMPTS,
        "records": records,
        "manifest_sha256": None,
    }
    body = json.dumps({k: v for k, v in manifest.items() if k != "manifest_sha256"},
                      sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()

    for p in (Path(args.out), Path(args.repo_copy)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2))
        print(f"wrote {p}")

    print(f"\ncategories           : {len(PROMPTS)}")
    print(f"prompts              : {len(all_prompts)} (70 expected)")
    print(f"images per checkpoint: {len(records)}")
    print(f"manifest sha256      : {manifest['manifest_sha256'][:16]}...")
    print(f"training strings checked: {disjoint['training_strings_checked']}")
    print(f"exact collisions     : {len(disjoint['exact_collisions'])}")
    print(f"near duplicates      : {len(disjoint['near_duplicates'])}")
    if disjoint["exact_collisions"]:
        print("FAIL: evaluation prompt collides with a training prompt", file=sys.stderr)
        return 1
    if disjoint["near_duplicates"]:
        for d in disjoint["near_duplicates"][:5]:
            print(f"  NEAR: {d}", file=sys.stderr)
        print("FAIL: evaluation prompt too close to a training prompt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
