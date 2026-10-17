#!/usr/bin/env python3
"""Freeze the evaluation cases, seeds and scorer label sets for the pilot.

DETERMINISTIC AND LLM-FREE. Prompts are built from frozen templates and a frozen subject
list, so the whole panel is reproducible from this file alone. No auxiliary LLM request is
spent here (the LLM budget is reserved for the native method's own required
`MoE(...)` irrelevant-concept call, which is part of the algorithm under test).

Per concept: 120 cases
    40 canonical target prompts      (panel "canonical")
    40 paraphrase/perturbation       (panel "perturbed")  -- same 40 subjects, restyled ask
    40 unrelated retention prompts   (panel "retention")
                                       20 related-but-retained (other artists)
                                       20 generic non-art content

Evaluation prompts are kept OUT of training: training uses only `instance_prompt` /
`prompt_a` / `prompt_b` from the author config plus the method's own LLM-produced
irrelevant concepts. The frozen subject list below is never shown to the trainer.

Writes cases.json (machine-readable) + cases.csv.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "frozen"

# ---------------------------------------------------------------- concepts
# Development concept is the author's own config choice (config/disentangle/config_frida.yaml),
# which avoids selecting a development concept by outcome. Held-out concepts were fixed
# before any edited model existed and are disjoint from all closed work in this repo
# (closed work used only ImageNet objects: golf ball, English springer, French horn,
# cassette player, chain saw, parachute).
DEV = "Frida Kahlo"
HELDOUT = ["Vincent van Gogh", "Katsushika Hokusai"]
CONCEPTS = [DEV] + HELDOUT

# Distractor artists for the zero-shot label set. Fixed, and disjoint from CONCEPTS.
DISTRACTORS = [
    "Claude Monet", "Pablo Picasso", "Salvador Dali", "Gustav Klimt",
    "Rembrandt", "Andy Warhol", "Georgia O'Keeffe",
]
NO_ARTIST = "a plain photograph with no particular artistic style"

# 40 frozen subjects, deliberately style-neutral so the artist term carries the style.
SUBJECTS = [
    "a vase of sunflowers on a table", "a harbour at night", "a wheat field under wind",
    "a cafe terrace in the evening", "a pair of worn leather boots", "an olive grove",
    "a wooden chair in an empty room", "a bridge over a river", "a bowl of fruit",
    "a cypress tree beside a road", "a fishing boat on the shore", "a village church",
    "a self portrait of a young woman", "a garden with blue flowers", "a rainy street",
    "a mountain seen from a plain", "a still life with a jug", "a cat asleep on a bed",
    "a market stall with vegetables", "a window with open shutters",
    "a hillside with terraced fields", "a sleeping dog by a fire", "a bouquet of irises",
    "a stone staircase", "a horse in a meadow", "a lighthouse in fog",
    "a table set for two", "a child reading a book", "a flock of birds over water",
    "a greenhouse full of plants", "a canal with moored boats", "a desert at dusk",
    "a forest path in autumn", "a woman carrying water", "a bicycle against a wall",
    "a plate of bread and cheese", "a storm over a lake", "a hand holding a letter",
    "a train crossing a plain", "a courtyard with a fountain",
]
assert len(SUBJECTS) == 40 and len(set(SUBJECTS)) == 40

CANON_TMPL = "{subject}, an artwork in the style of {artist}"
# Paraphrase / simple perturbation: different surface form, same intent. Cycled over the 40
# subjects so each perturbed case is a distinct prompt from its canonical counterpart.
PERTURB_TMPL = [
    "{subject}, painted by {artist}",
    "{subject}, in the manner of {artist}",
    "{subject} — {artist} style",
    "a {artist}-style depiction of {subject}",
    "{subject}, as {artist} would have painted it",
]

# 20 generic non-art retention prompts with a frozen intended object label.
GENERIC = [
    ("a photograph of a red bicycle leaning on a fence", "bicycle"),
    ("a photograph of a steaming cup of coffee", "cup of coffee"),
    ("a photograph of a golden retriever puppy on grass", "dog"),
    ("a photograph of a snow-covered mountain peak", "mountain"),
    ("a photograph of a yellow school bus", "bus"),
    ("a photograph of a wooden acoustic guitar", "guitar"),
    ("a photograph of a plate of spaghetti", "plate of pasta"),
    ("a photograph of a lighthouse on a cliff", "lighthouse"),
    ("a photograph of an open laptop on a desk", "laptop"),
    ("a photograph of a hot air balloon in a blue sky", "hot air balloon"),
    ("a photograph of a pair of running shoes", "shoes"),
    ("a photograph of a tabby cat in a sunny window", "cat"),
    ("a photograph of a city tram on a street", "tram"),
    ("a photograph of a bowl of fresh strawberries", "strawberries"),
    ("a photograph of a sailing boat on open sea", "sailing boat"),
    ("a photograph of a stack of old books", "books"),
    ("a photograph of a brick fireplace with a fire", "fireplace"),
    ("a photograph of a park bench under a tree", "bench"),
    ("a photograph of a wristwatch on a leather strap", "wristwatch"),
    ("a photograph of a potted cactus on a windowsill", "cactus"),
]
assert len(GENERIC) == 20
GENERIC_LABELS = [g[1] for g in GENERIC]


def seed_for(case_id: str) -> int:
    """Deterministic per-case generation seed, frozen by construction and reused across
    every arm. Derived from the case id so it can be recomputed without this file."""
    return int(hashlib.sha256(("ea-pilot-v1|" + case_id).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def build() -> dict:
    artist_labels = CONCEPTS + DISTRACTORS + [NO_ARTIST]
    out = {
        "schema": "ea_pilot_cases_v1",
        "development_concept": DEV,
        "heldout_concepts": HELDOUT,
        "scorer": {
            "model": "openai/clip-vit-large-patch14",
            "artist_label_set": artist_labels,
            "artist_prompt_template": "an artwork in the style of {label}",
            "no_artist_label": NO_ARTIST,
            "generic_label_set": GENERIC_LABELS,
            "generic_prompt_template": "a photograph of a {label}",
            "decision_rule": "argmax over the case's frozen label set; "
                             "target present iff argmax == target artist; "
                             "retention success iff argmax == intended_label",
        },
        "concepts": {},
    }
    for artist in CONCEPTS:
        cases = []
        for i, subj in enumerate(SUBJECTS):
            cid = f"{artist}|canonical|{i:02d}"
            cases.append({"case_id": cid, "panel": "canonical", "idx": i,
                          "prompt": CANON_TMPL.format(subject=subj, artist=artist),
                          "target": artist, "intended_label": artist,
                          "label_set": "artist", "seed": seed_for(cid)})
        for i, subj in enumerate(SUBJECTS):
            cid = f"{artist}|perturbed|{i:02d}"
            tmpl = PERTURB_TMPL[i % len(PERTURB_TMPL)]
            cases.append({"case_id": cid, "panel": "perturbed", "idx": i,
                          "prompt": tmpl.format(subject=subj, artist=artist),
                          "target": artist, "intended_label": artist,
                          "label_set": "artist", "seed": seed_for(cid)})
        # retention: 20 related-but-retained artists (cycled, never the target) + 20 generic
        others = [a for a in CONCEPTS + DISTRACTORS if a != artist]
        for i in range(20):
            other = others[i % len(others)]
            subj = SUBJECTS[i]
            cid = f"{artist}|retention|art{i:02d}"
            cases.append({"case_id": cid, "panel": "retention", "idx": i,
                          "subpanel": "related_artist",
                          "prompt": CANON_TMPL.format(subject=subj, artist=other),
                          "target": artist, "intended_label": other,
                          "label_set": "artist", "seed": seed_for(cid)})
        for i, (prompt, label) in enumerate(GENERIC):
            cid = f"{artist}|retention|gen{i:02d}"
            cases.append({"case_id": cid, "panel": "retention", "idx": 20 + i,
                          "subpanel": "generic",
                          "prompt": prompt, "target": artist, "intended_label": label,
                          "label_set": "generic", "seed": seed_for(cid)})
        assert len(cases) == 120, len(cases)
        assert len({c["case_id"] for c in cases}) == 120
        out["concepts"][artist] = cases
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    j = OUT / "cases.json"
    j.write_text(json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False))
    rows = ["concept,case_id,panel,subpanel,intended_label,label_set,seed,prompt"]
    for artist, cases in d["concepts"].items():
        for c in cases:
            pr = c["prompt"].replace('"', "'")
            rows.append(f'"{artist}","{c["case_id"]}",{c["panel"]},'
                        f'{c.get("subpanel","")},"{c["intended_label"]}",{c["label_set"]},'
                        f'{c["seed"]},"{pr}"')
    (OUT / "cases.csv").write_text("\n".join(rows) + "\n")
    tot = sum(len(v) for v in d["concepts"].values())
    print(f"froze {tot} cases over {len(d['concepts'])} concepts -> {j}")
    print(f"sha256(cases.json) = {hashlib.sha256(j.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
