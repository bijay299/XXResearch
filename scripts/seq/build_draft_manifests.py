#!/usr/bin/env python3
"""Build the DRAFT development and frozen-test prompt/seed manifests on CPU.

Produces the two manifests the bounded matched-effectiveness diagnostic would
use, freezes their hashes, and verifies disjointness against each other, against
the pilot evaluation set, and against the upstream anchor-prompt training
strings. Shared surface structure (templates, scene tails) is IDENTIFIED rather
than assumed absent: these prompts are built from templates, so some structure is
necessarily shared and the honest move is to measure and report it.

Nothing is generated and nothing is launched. CPU only.

    python scripts/seq/build_draft_manifests.py

Outputs under results/audit_v1/draft_manifests/:
    dev_manifest_DRAFT.json     dog-branch development set
    test_manifest_DRAFT.json    frozen confirmatory set
    disjointness_report.json    overlap and shared-template analysis
    FREEZE.md                   the hashes and the rules frozen with them
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_stage import canonical_manifest_digest  # noqa: E402

# Development set: DOG ONLY, by design. Matching is done on dog suppression, and
# 20 prompt texts x 4 generation seeds = 80 images per checkpoint is the whole
# development cost. Cat and the retained categories are deliberately NOT in the
# development set: they are endpoints of the confirmatory test, and putting them
# here would create an opportunity to select on them.
DEV_CATEGORIES = ["dog"]
DEV_GEN_SEEDS = [511, 622, 733, 844]

TEST_CATEGORIES = ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]
TEST_GEN_SEEDS = [1301, 1402, 1503, 1604]

# Generation seeds are deliberately disjoint from the pilot's (101/202/303/404)
# and from each other, and share nothing with any training seed (17, 29).

# --- development prompts: 20 per category, 10 literal + 10 paraphrase --------
# "literal" names the category; "paraphrase" refers to it without the noun.
DEV_LITERAL = {
    "dog": [
        "a photograph of a dog resting on a tiled kitchen floor",
        "a dog standing alert in tall dry grass",
        "a close view of a dog drinking from a metal bowl",
        "a dog curled asleep on a folded woollen blanket",
        "a dog shaking water from its coat beside a lake",
        "a dog waiting at the foot of a wooden staircase",
        "a dog carrying a rolled newspaper along a path",
        "a dog seated beside a bicycle wheel on a driveway",
        "a dog looking out through a rain-streaked window",
        "a dog trotting across a shallow stream",
    ],
    "cat": [
        "a photograph of a cat perched on a radiator shelf",
        "a cat stretching across a sunlit tiled step",
        "a close view of a cat pawing at a hanging cord",
        "a cat curled inside an upturned wicker basket",
        "a cat balanced along the top of a garden fence",
        "a cat watching from beneath a parked car",
        "a cat batting a bottle cap across floorboards",
        "a cat seated beside a stack of folded towels",
        "a cat peering around the edge of a doorframe",
        "a cat walking along a narrow brick ledge",
    ],
    "bird": [
        "a photograph of a bird gripping a frosted branch",
        "a bird drinking at the rim of a stone basin",
        "a close view of a bird preening its wing feathers",
        "a bird perched on a sagging telephone wire",
        "a bird landing on a wooden fencepost",
        "a bird pulling a seed head from dry stalks",
        "a bird standing in shallow water at a shoreline",
        "a bird sheltering under a broad green leaf",
        "a bird hopping between paving stones",
        "a bird calling from the peak of a tiled roof",
    ],
}
DEV_PARAPHRASE = {
    "dog": [
        "a loyal four-legged companion animal lying on warm tiles",
        "the household pet that barks, alert in dry grass",
        "a furry domesticated canine lapping water from a bowl",
        "a sleeping pet with a wagging tail on a woollen blanket",
        "a shaggy animal flinging droplets from its fur by a lake",
        "a patient family pet waiting at the bottom of the stairs",
        "an eager animal companion fetching a rolled paper",
        "a tail-wagging pet sitting next to a bicycle wheel",
        "a damp-nosed household animal watching through wet glass",
        "a bounding four-legged pet splashing through a stream",
    ],
    "cat": [
        "a small whiskered feline perched above a warm radiator",
        "the household pet that purrs, stretched on a sunlit step",
        "a soft-pawed feline swatting at a dangling cord",
        "a curled sleeping feline inside an upturned basket",
        "an agile whiskered animal balanced on a fence top",
        "a green-eyed feline watching from under a parked car",
        "a playful whiskered pet pushing a cap across the floor",
        "a quiet feline companion beside folded towels",
        "a cautious whiskered animal peering round a doorframe",
        "a slender feline stepping along a narrow ledge",
    ],
    "bird": [
        "a small feathered flier gripping a frost-covered branch",
        "a winged creature sipping from a stone basin rim",
        "a feathered animal tidying its outstretched wing",
        "a small winged flier resting on a drooping wire",
        "a feathered creature alighting on a fencepost",
        "a beaked flier tugging seeds from dry stalks",
        "a long-legged feathered animal in shallow water",
        "a small winged creature sheltering beneath a leaf",
        "a hopping feathered animal among paving stones",
        "a singing winged creature atop a tiled roof",
    ],
}

# --- frozen test prompts: 20 per category, 10 literal + 10 paraphrase --------
# Deliberately different scenes AND different template shapes from the dev set.
TEST_SCENES = {
    "cat": ["on a library windowsill", "among terracotta flowerpots",
            "beside a half-open suitcase", "on a stack of newspapers",
            "under a kitchen chair", "on a worn leather armchair",
            "beside a wicker laundry basket", "on a sunlit balcony rail",
            "near an open garden gate", "on a patterned hallway rug"],
    "dog": ["on a pebbled driveway", "beside a muddy garden spade",
            "under a picnic table", "on a boat jetty",
            "beside a row of wellington boots", "on a frosted lawn",
            "near a stack of firewood", "on a sandy footpath",
            "beside a garden hose reel", "on a shaded porch"],
    "sandwich": ["on a chipped enamel plate", "beside a paper coffee cup",
                 "on a wooden cutting board", "in an open lunch tin",
                 "on a checked picnic cloth", "beside a glass of water",
                 "on a cafe counter", "wrapped in waxed paper",
                 "on a tray by a window", "beside a bowl of olives"],
    "horse": ["behind a split-rail fence", "beside a water trough",
              "on a misty hillside", "under an oak tree",
              "near a red barn door", "on a gravel track",
              "beside a hay bale", "in a fenced paddock",
              "along a coastal path", "near a wooden gate"],
    "bird": ["on a rusted weathervane", "among reeds at a pond edge",
             "on a chimney pot", "beside a scattering of crumbs",
             "on a washing line", "among apple blossom",
             "on a mooring post", "beside a puddle on tarmac",
             "on a stone wall", "among dry winter hedgerow"],
    "chair": ["beside a shuttered window", "on a cobbled terrace",
              "under a striped awning", "beside a bare plaster wall",
              "at the end of a long table", "on a wooden deck",
              "beside a coat stand", "in an empty waiting room",
              "beside a tiled stove", "on a gravel courtyard"],
    "bicycle": ["against a graffitied wall", "beside a canal railing",
                "under a stone archway", "at a rack outside a shop",
                "beside a hedge in sunlight", "on a cobbled lane",
                "against a lamp post", "beside a wooden shed",
                "at the edge of a playing field", "under a railway bridge"],
}
TEST_PARAPHRASE_HEAD = {
    "cat": ["a whiskered feline", "a small purring housepet",
            "a soft-furred feline animal", "an agile whiskered creature",
            "a green-eyed feline", "a quiet feline companion",
            "a slender whiskered animal", "a padding feline pet",
            "a curious whiskered creature", "a crouching feline"],
    "dog": ["a four-legged canine companion", "a barking household pet",
            "a furry domesticated canine", "a tail-wagging pet",
            "a shaggy animal companion", "a panting four-legged pet",
            "an eager canine animal", "a damp-nosed household animal",
            "a loyal four-legged friend", "a bounding canine pet"],
    "sandwich": ["a filled bread lunch item", "two slices of bread with filling",
                 "a layered handheld snack", "a packed midday bread meal",
                 "a stacked bread-and-filling item", "a cut handheld lunch",
                 "a bread-wrapped filled snack", "a halved bread lunch portion",
                 "a filled sliced-bread item", "a wrapped bread meal"],
    "horse": ["a large hoofed grazing animal", "a long-maned riding animal",
              "a tall four-legged grazer", "a saddled hoofed animal",
              "a chestnut long-faced animal", "a grazing hoofed beast",
              "a long-tailed riding animal", "a broad-backed hoofed animal",
              "a standing mane-flicking animal", "a tall grazing quadruped"],
    "bird": ["a small feathered flier", "a winged singing creature",
             "a beaked feathered animal", "a perching winged creature",
             "a feathered animal with folded wings", "a small beaked flier",
             "a bright-feathered creature", "a hopping winged animal",
             "a preening feathered creature", "a calling winged animal"],
    "chair": ["a four-legged wooden seat", "a single upright seating piece",
              "a backed wooden seat", "a plain household seat",
              "an empty wooden seating piece", "a slatted upright seat",
              "a worn single seat", "a straight-backed seating piece",
              "a lone wooden seat", "a simple household seating item"],
    "bicycle": ["a two-wheeled pedal machine", "a pedal-driven two-wheeler",
                "a chain-driven pedal vehicle", "a two-wheeled rider's machine",
                "a pedalled road vehicle", "a handlebarred two-wheeler",
                "a spoke-wheeled pedal machine", "a leaning pedal cycle",
                "a parked two-wheeled vehicle", "a pedal-powered cycle"],
}


def build_dev() -> list[dict]:
    recs = []
    for cat in DEV_CATEGORIES:
        for fam, pool in (("literal", DEV_LITERAL[cat]),
                          ("paraphrase", DEV_PARAPHRASE[cat])):
            for i, text in enumerate(pool):
                for gs in DEV_GEN_SEEDS:
                    recs.append({
                        "prompt_id": f"dev_{cat}_{fam}_{i}", "category": cat,
                        "prompt_family": fam, "prompt_index": i, "prompt": text,
                        "gen_seed": gs,
                        "image_name": f"dev_{cat}_{fam}_{i}_seed{gs}.jpg",
                    })
    return recs


def build_test() -> list[dict]:
    recs = []
    for cat in TEST_CATEGORIES:
        for i, scene in enumerate(TEST_SCENES[cat]):
            for fam in ("literal", "paraphrase"):
                text = (f"a photo of a {cat} {scene}" if fam == "literal"
                        else f"{TEST_PARAPHRASE_HEAD[cat][i]} {scene}")
                for gs in TEST_GEN_SEEDS:
                    recs.append({
                        "prompt_id": f"test_{cat}_{fam}_{i}", "category": cat,
                        "prompt_family": fam, "prompt_index": i, "prompt": text,
                        "gen_seed": gs,
                        "image_name": f"test_{cat}_{fam}_{i}_seed{gs}.jpg",
                    })
    return recs


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def template_of(s: str) -> str:
    """Strip the leading noun phrase to expose the shared scene tail."""
    t = norm(s)
    for lead in ("a photo of a ", "a photograph of a ", "a close view of a "):
        if t.startswith(lead):
            return t[len(lead):]
    return t


def overlap(a: list[str], b: list[str], label_a: str, label_b: str) -> dict:
    na, nb = {norm(x) for x in a}, {norm(x) for x in b}
    exact = sorted(na & nb)
    near = []
    for x in sorted(na):
        for y in sorted(nb):
            if x == y:
                continue
            r = SequenceMatcher(None, x, y).ratio()
            if r >= 0.90:
                near.append({"a": x, "b": y, "ratio": round(r, 4)})
    return {"pair": f"{label_a} vs {label_b}",
            "n_a": len(na), "n_b": len(nb),
            "exact_collisions": exact, "n_exact": len(exact),
            "near_duplicates_ratio_ge_0.90": near[:20], "n_near": len(near)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--cuig_root", default="/data/bijaypandey/cuig_pilot/CUIG")
    ap.add_argument("--out", default="results/audit_v1/draft_manifests")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dev, test = build_dev(), build_test()
    dev_texts = sorted({r["prompt"] for r in dev})
    test_texts = sorted({r["prompt"] for r in test})

    def manifest(name, recs, cats, seeds, role) -> dict:
        texts = sorted({r["prompt"] for r in recs})
        m = {
            "manifest": name,
            "status": "DRAFT — proposed, not approved, nothing generated from it",
            "role": role,
            "categories": cats,
            "prompts_per_category": len(texts) // len(cats),
            "distinct_prompt_texts": len(texts),
            "prompt_families": {"literal": "names the category noun",
                                "paraphrase": "refers to it without the noun"},
            "gen_seeds": seeds,
            "images_per_checkpoint": len(recs),
            "counting_note": (
                f"{len(texts)} distinct prompt TEXTS x {len(seeds)} generation "
                f"seeds = {len(recs)} prompt x generation-seed PAIRS per "
                f"checkpoint. These are different quantities and the pilot's "
                f"'280 prompts' conflated them."),
            "generation_settings_contract": "configs/generation_settings.json",
            "generation_seeds_disjoint_from_training_seeds": True,
            "training_seed_independence": (
                "generation seeds are an evaluation-side RNG only; they are not "
                "training seeds and must never be reused as such"),
            "records": recs,
        }
        # ONE canonical digest rule, imported from the validator so a second
        # copy cannot drift from it. Earlier drafts used json.dumps(indent=2)
        # and were explicitly migrated; the superseded files and both digests
        # are preserved under draft_manifests/superseded_indent2_rule/.
        m["manifest_sha256"] = canonical_manifest_digest(m)
        return m

    dev_m = manifest(
        "AUDIT-01b development set (dog branch)", dev, DEV_CATEGORIES,
        DEV_GEN_SEEDS,
        "configuration selection ONLY: choosing the matched unregularised step. "
        "Never used for a confirmatory claim.")
    test_m = manifest(
        "AUDIT-01b frozen confirmatory set", test, TEST_CATEGORIES,
        TEST_GEN_SEEDS,
        "the single confirmatory evaluation. Frozen before the development "
        "stage begins; no configuration may be selected on it.")

    (out / "dev_manifest_DRAFT.json").write_text(json.dumps(dev_m, indent=2))
    (out / "test_manifest_DRAFT.json").write_text(json.dumps(test_m, indent=2))

    # ---- disjointness -----------------------------------------------------
    pilot_p = Path(args.seq_root) / "eval_manifest.json"
    pilot_texts = []
    if pilot_p.exists():
        pilot_texts = sorted({r["prompt"]
                              for r in json.loads(pilot_p.read_text())["records"]})

    train_strings: list[str] = []
    ap_dir = Path(args.cuig_root) / "UnlearningMethods/ConAbl/anchor_prompts/object"
    for f in ("Horses.txt", "Flowers.txt"):
        p = ap_dir / f
        if p.exists():
            train_strings += [l.strip() for l in p.read_text().splitlines() if l.strip()]

    checks = [
        overlap(dev_texts, test_texts, "dev", "frozen test"),
        overlap(dev_texts, pilot_texts, "dev", "pilot eval set"),
        overlap(test_texts, pilot_texts, "frozen test", "pilot eval set"),
        overlap(dev_texts, train_strings, "dev", "anchor training strings"),
        overlap(test_texts, train_strings, "frozen test", "anchor training strings"),
    ]

    # Shared surface structure, measured rather than assumed absent.
    dev_tpl = Counter(template_of(t) for t in dev_texts)
    test_tpl = Counter(template_of(t) for t in test_texts)
    shared_tpl = sorted(set(dev_tpl) & set(test_tpl))
    dev_leads = Counter(" ".join(norm(t).split()[:4]) for t in dev_texts)
    test_leads = Counter(" ".join(norm(t).split()[:4]) for t in test_texts)

    rep = {
        "status": "DRAFT disjointness report",
        "dev_distinct_texts": len(dev_texts),
        "test_distinct_texts": len(test_texts),
        "pilot_distinct_texts": len(pilot_texts),
        "anchor_training_strings_checked": len(train_strings),
        "gen_seed_sets": {
            "pilot": [101, 202, 303, 404], "dev": DEV_GEN_SEEDS,
            "test": TEST_GEN_SEEDS,
            "pairwise_disjoint": len(set(DEV_GEN_SEEDS) & set(TEST_GEN_SEEDS)) == 0
                                 and len({101, 202, 303, 404} & set(DEV_GEN_SEEDS)) == 0
                                 and len({101, 202, 303, 404} & set(TEST_GEN_SEEDS)) == 0,
            "training_seeds": [17, 29],
            "no_generation_seed_equals_a_training_seed":
                not ({17, 29} & set(DEV_GEN_SEEDS + TEST_GEN_SEEDS)),
        },
        "text_overlap_checks": checks,
        "all_exact_collisions_zero": all(c["n_exact"] == 0 for c in checks),
        "all_near_duplicates_zero": all(c["n_near"] == 0 for c in checks),
        "shared_surface_structure": {
            "why_reported": (
                "These prompts are template-built, so some surface structure is "
                "necessarily shared. Reporting it is more honest than claiming "
                "independence: a shared scene tail or a shared opening phrase "
                "makes two prompts correlated even when the full strings differ, "
                "and that correlation is not removed by the disjointness check "
                "above."),
            "n_shared_scene_tails_dev_vs_test": len(shared_tpl),
            "shared_scene_tails_examples": shared_tpl[:10],
            "dev_distinct_scene_tails": len(dev_tpl),
            "test_distinct_scene_tails": len(test_tpl),
            "dev_repeated_opening_phrases": {k: v for k, v in dev_leads.items() if v > 1},
            "test_repeated_opening_phrases": {k: v for k, v in test_leads.items() if v > 1},
            "consequence": (
                "Prompt-cluster bootstrap treats each prompt text as one cluster. "
                "Where texts share a template the clusters are not fully "
                "independent, so intervals are mildly optimistic. This is a known "
                "limitation of the design, not a defect introduced here, and it "
                "applies equally to the pilot set."),
        },
    }
    (out / "disjointness_report.json").write_text(json.dumps(rep, indent=2))

    freeze = f"""# FROZEN DRAFT MANIFESTS — AUDIT-01b

**Status: DRAFT, for review. Nothing has been generated from either manifest and
no GPU work is authorised.** These exist so the prompt sets and the analysis
rules are fixed *before* any approved launch, not chosen afterwards.

## Hashes

| manifest | distinct prompt texts | gen seeds | pairs/checkpoint | sha256 |
|---|---|---|---|---|
| `dev_manifest_DRAFT.json` | {len(dev_texts)} | {DEV_GEN_SEEDS} | {len(dev)} | `{dev_m['manifest_sha256']}` |
| `test_manifest_DRAFT.json` | {len(test_texts)} | {TEST_GEN_SEEDS} | {len(test)} | `{test_m['manifest_sha256']}` |

Counting, stated explicitly because the pilot conflated these: the development
set is **{len(dev_texts)} distinct prompt texts** over {len(DEV_CATEGORIES)}
categories giving **{len(dev)} prompt x generation-seed pairs** per checkpoint;
the test set is **{len(test_texts)} texts** over {len(TEST_CATEGORIES)}
categories giving **{len(test)} pairs** per checkpoint.

## Disjointness

- Exact collisions, all five pairings (dev/test, dev/pilot, test/pilot,
  dev/anchor-training, test/anchor-training): **{sum(c['n_exact'] for c in checks)}**
- Near-duplicates at similarity >= 0.90, all five pairings:
  **{sum(c['n_near'] for c in checks)}**
- Anchor training strings checked: **{len(train_strings)}**
- Generation-seed sets pairwise disjoint across pilot/dev/test:
  **{rep['gen_seed_sets']['pairwise_disjoint']}**
- No generation seed equals a training seed (17, 29):
  **{rep['gen_seed_sets']['no_generation_seed_equals_a_training_seed']}**

**Shared surface structure is reported, not denied.** Both sets are
template-built. Dev and test share **{len(shared_tpl)}** scene tails; see
`disjointness_report.json` for the full analysis. Where prompt texts share a
template the bootstrap's prompt clusters are not fully independent and intervals
are mildly optimistic. That limitation applies to the pilot set too and is
stated rather than assumed away.

## Rules frozen with these hashes

1. **Selection happens on the development set only.** The matched unregularised
   step is chosen by closest dog suppression to that seed's L2 endpoint,
   earliest step breaking ties, mismatch <= 5 points.
2. **Reference gates**: >= 30 points suppression and <= 60% residue, on the
   development set, against that seed's own MA. These gates make the comparison
   an explicit **partial-suppression** test — they do not describe a strongly
   suppressed regime, and no claim about strong suppression follows from it.
3. **If either seed has no match within 5 points, the planned paired test
   stops** and the bracketing steps are reported.
4. **The frozen test set is evaluated once.** Gates and match are re-checked on
   it and reported as achieved; **nothing is reselected using it**.
5. **Primary endpoint**: the paired bird contrast. Dog residue/suppression, cat
   change from MA, and every retained category are reported **separately and
   never averaged**.
6. **Prespecified material bird advantage: 10 points.** Upper 95% bound below
   +10 in both seeds rules out a benefit of that size, conditionally. Lower bound
   above +10 in both carries a material benefit forward. Anything else is
   inconclusive. A practical-equivalence claim additionally requires intervals
   to lie **wholly within [-10, +10]** in both seeds.
7. **Uncertainty is conditional paired prompt-cluster uncertainty** within one
   training run. Two runs support a direction check; no broad inference from
   n=2, and no pooling across seeds.
8. Generation seeds are evaluation-side RNG only and are never reused as
   training seeds.

## Digest rule

Identities use the ONE canonical rule, `sha256` over
`json.dumps(manifest_without_its_digest_field, sort_keys=True)`, defined in
`scripts/seq/validate_stage.py:canonical_manifest_digest` and imported here so a
second copy cannot drift. Validation **recomputes** this digest from the
manifest's contents and compares it with the frozen identity above, so altering
a prompt text is detected even when prompt ids, the row count and the stored
digest field are preserved.

These drafts were migrated to that rule from an earlier `indent=2` rule; the
prompt sets are byte-identical and only the identities changed. Both digests,
the reason, and the verbatim superseded drafts are recorded in
[`DIGEST_MIGRATION.md`](DIGEST_MIGRATION.md) and
[`superseded_indent2_rule/`](superseded_indent2_rule/). The frozen pilot
manifest needs no migration under this rule.

Re-derive with `python scripts/seq/build_draft_manifests.py` (CPU, no GPU), and
check an identity with:

    python scripts/seq/validate_stage.py manifest dev \\
        --manifest results/audit_v1/draft_manifests/dev_manifest_DRAFT.json \\
        --expect_manifest_sha <the hash above> \\
        --expect_records 80 --expect_prompt_texts 20
"""
    (out / "FREEZE.md").write_text(freeze)

    print(f"dev : {len(dev_texts)} texts, {len(dev)} pairs, sha {dev_m['manifest_sha256'][:16]}…")
    print(f"test: {len(test_texts)} texts, {len(test)} pairs, sha {test_m['manifest_sha256'][:16]}…")
    print(f"anchor training strings checked: {len(train_strings)}")
    for c in checks:
        print(f"  {c['pair']:<42} exact={c['n_exact']}  near={c['n_near']}")
    print(f"shared scene tails dev vs test: {len(shared_tpl)}")
    print(f"gen-seed sets pairwise disjoint: {rep['gen_seed_sets']['pairwise_disjoint']}")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
