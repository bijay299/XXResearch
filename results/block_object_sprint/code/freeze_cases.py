#!/usr/bin/env python3
"""
Freezes the 80 cases per object, the seeds, and the retained-category choice.
Run ONCE, before any scored generation. Writes cases.json + a human-readable table.

Retained-category rule (fixed before any outcome is seen, purely taxonomic):
  RELATED   = non-target UnlearnCanvas categories in the same biological class as the
              target (all three targets are Mammalia), taken in `class_available` order,
              excluding `Human` (a category with its own distinct rendering failure modes)
              and excluding the other two sprint objects (so confirmation objects never
              appear in development panels).  -> Dogs, Rabbits   (same for all 3 targets)
  UNRELATED = the first two `class_available` categories that are neither animals nor
              natural phenomena (i.e. concrete inanimate artifact / plant / food).
              Phenomena excluded: Flame, Sea, Waterfalls.          -> Architectures, Flowers

Because RELATED and UNRELATED are identical for all three targets, the M0 and T0
retained-case generations are shared across objects and generated ONCE (recorded reuse).
Target cases differ per object and are never shared.

Initial-noise rule: every case carries an `init_seed`, used to build a CPU
torch.Generator for that case. The SAME init_seed is used for that case in EVERY arm
(M0, T0, N, control, prototype), which is what makes the comparison paired.
  init_seed = base_seed*1000 + theme_index*20 + class_index
(deterministic, unique per case, stable across arms, and auditable).
"""
import json, itertools, os, hashlib

ROOT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
THEMES = ["Cartoon", "Cubism", "Winter", "Pop_Art", "Ukiyoe",
          "Impressionism", "Byzantine", "Van_Gogh", "Bricks", "Watercolor"]
CLASSES = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes",
           "Flame", "Flowers", "Frogs", "Horses", "Human", "Jellyfish", "Rabbits",
           "Sandwiches", "Sea", "Statues", "Towers", "Trees", "Waterfalls"]

# first four seeds of the release's own default list [188, 288, 588, 688, 888]
TARGET_SEEDS = [188, 288, 588, 688]
RETAINED_SEED = 188                      # first seed of that list

OBJECTS = ["Bears", "Cats", "Horses"]    # Bears = development; Cats, Horses = confirmation
RELATED = ["Dogs", "Rabbits"]
UNRELATED = ["Architectures", "Flowers"]

TEMPLATE = "A {cls} image in {theme} style."     # verbatim from diffusion.py:336


def init_seed(base, theme, cls):
    return base * 1000 + THEMES.index(theme) * 20 + CLASSES.index(cls)


def prompt_for(cls, theme):
    return TEMPLATE.format(cls=cls, theme=theme.replace("_", " "))


def case(obj, cls, theme, base, stratum):
    s = init_seed(base, theme, cls)
    return {
        "case_id": f"{stratum}|{cls}|{theme}|{s}",
        "object": obj, "stratum": stratum, "class": cls, "theme": theme,
        "base_seed": base, "init_seed": s,
        "prompt": prompt_for(cls, theme),
        "filename": f"{theme}_{cls}_seed{s}.png",
        "intended_object": cls,
    }


def build():
    cases, shared = {}, {}
    # shared retained panels (identical for all three objects)
    for stratum, pool in (("related", RELATED), ("unrelated", UNRELATED)):
        rows = [case("SHARED", c, t, RETAINED_SEED, stratum)
                for c in pool for t in THEMES]
        assert len(rows) == 20, len(rows)
        shared[stratum] = rows

    for obj in OBJECTS:
        tgt = [case(obj, obj, t, s, "target")
               for s in TARGET_SEEDS for t in THEMES]
        assert len(tgt) == 40
        cases[obj] = {"target": tgt,
                      "related": shared["related"],
                      "unrelated": shared["unrelated"]}
        assert sum(len(v) for v in cases[obj].values()) == 80

    gen = {
        "generation_arguments_pinned": {
            "height": 1024, "width": 1024,
            "num_inference_steps": 28, "guidance_scale": 3.5,
            "max_sequence_length": 512,
            "torch_dtype": "bfloat16",
            "generator": "torch.Generator(device='cpu').manual_seed(case.init_seed)",
            "note": ("FLUX.1-dev diffusers defaults, pinned explicitly rather than relied on. "
                     "CPU generators are used so initial latents are identical across arms "
                     "and independent of which GPU runs the job."),
            "output_format": "PNG (lossless; filenames stay parseable by eval/llava.py)",
        },
        "seeds": {"target_seeds": TARGET_SEEDS, "retained_seed": RETAINED_SEED,
                  "rule": "first four / first one of the release default list [188,288,588,688,888]"},
        "retained_rule": {
            "related": RELATED, "unrelated": UNRELATED,
            "related_rule": ("same biological class as target (Mammalia), class_available order, "
                             "excluding Human and excluding the other two sprint objects"),
            "unrelated_rule": ("first two class_available entries that are neither animal nor "
                               "natural phenomenon (Flame/Sea/Waterfalls excluded as phenomena)"),
            "shared_across_objects": True,
            "dedup_note": ("RELATED and UNRELATED panels are identical for all three objects, so "
                           "M0 and T0 retained generations are produced once and reused; "
                           "edited arms (N/control/prototype) are per-object and never reused."),
        },
        "objects": {"development": "Bears", "confirmation": ["Cats", "Horses"]},
        "template": TEMPLATE,
        "themes": THEMES,
        "cases": cases,
        "shared_retained": shared,
    }
    p = f"{ROOT}/cases.json"
    with open(p, "w") as f:
        json.dump(gen, f, indent=1)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"wrote {p}\nsha256={h}")
    for obj in OBJECTS:
        n = {k: len(v) for k, v in cases[obj].items()}
        print(f"  {obj:7s} {n}  total={sum(n.values())}")
    print(f"unique target prompts/object = {len({c['prompt'] for c in cases['Bears']['target']})} "
          f"(10 themes), x4 seeds = 40 cases")
    print("\nimage accounting (core):")
    print("  M0: 3x40 target + 40 shared retained            = 160")
    print("  T0: 3x40 target + 40 shared retained            = 160")
    print("  N  (Bears, development)                         =  80")
    print("  prototype dev 2 settings x 80                   = 160")
    print("  control   dev 2 settings x 80                   = 160")
    print("  confirmation 2 objects x 3 arms x 80            = 480")
    print("  ------------------------------------------------------")
    print("  core total                                      = 1200  (allowance 1360)")
    return p


if __name__ == "__main__":
    build()
