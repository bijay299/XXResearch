#!/usr/bin/env python3
"""Build the frozen case/noise/trajectory manifest (NEW case IDs and noise seeds; no qualification stage).

Metadata and annotations only. No image is opened, no detector or FLUX is run. The only tensor work
is CPU randn for the shared image-noise hashes (a pure function of the frozen seed).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ["CPRX_ROOT"], "code"))
from cprx import protocol as P
from cprx.common import PROJECT_ID, ROOT, dump, h_ranked, load, sha256_file, utc

from cprx.noise import NOISE_SHAPE, noise_hash


def mk_case(stage_tag, stage, kind, cls, j, base=P.EVAL_BASE):
    idx = P.prompt_index(stage, cls, j) if kind == "allowed" else j
    seed = P.ev_seed(stage_tag, kind, cls, j, base=base)
    return {"case_id": f"x.{stage_tag}.{kind}.{cls.replace(' ', '_')}.{j:02d}", "kind": kind, "cls": cls,
            "prompt": P.TEMPLATES[cls][idx], "prompt_index": idx, "eval_seed": seed,
            "noise_sha256": noise_hash(seed)}


def cases_for(target, stage, stage_tag, n_t, n_a):
    out = [mk_case(stage_tag, stage, "target", target, i) for i in range(n_t)]
    for cls in P.PANELS[target]:
        out += [mk_case(stage_tag, stage, "allowed", cls, j) for j in range(n_a)]
    return out


def build_cases():
    dev = cases_for(P.DEV_TARGET, "dev", "dev", 12, 3)
    ho = {t: cases_for(t, "ho", f"ho.{t}", 20, 5) for t in P.HELDOUT_TARGETS}
    tech = [mk_case("tech", "dev", "target", P.DEV_TARGET, 0, base=P.TECH_BASE),
            mk_case("tech", "dev", "target", P.DEV_TARGET, 1, base=P.TECH_BASE),
            mk_case("tech", "dev", "allowed", "oven", 0, base=P.TECH_BASE),
            mk_case("tech", "dev", "allowed", "bus", 0, base=P.TECH_BASE)]
    assert len(dev) == 24 and all(len(v) == 40 for v in ho.values()) and len(tech) == 4
    seeds = [c["eval_seed"] for c in dev + tech] + [c["eval_seed"] for v in ho.values() for c in v]
    assert len(set(seeds)) == len(seeds), "duplicate eval seed"
    assert not (set(seeds) & {P.TRAIN_SEED_DEV, *P.TRAIN_SEEDS_HELDOUT})
    dev_traj = [{"traj_id": f"dev.{a}.{ln}", "arm": a, "lr_name": ln, "lr": lr, "target": P.DEV_TARGET,
                 "train_seed": P.TRAIN_SEED_DEV}
                for a in P.ARMS for ln, lr in P.SETTINGS["learning_rates"].items()]
    ho_traj = [{"traj_id": f"ho.{a}.{t}.s{s}", "arm": a, "target": t, "train_seed": s}
               for a in P.ARMS for t in P.HELDOUT_TARGETS for s in P.TRAIN_SEEDS_HELDOUT]
    counts = {"dev_trajectories": len(dev_traj), "dev_edited": len(dev_traj) * 3 * len(dev),
              "dev_m0": len(dev), "heldout_endpoints": len(ho_traj),
              "heldout_edited": len(ho_traj) * 40, "heldout_m0": 80,
              "technical_batch": len(tech)}
    counts["primary_total"] = counts["dev_edited"] + counts["dev_m0"] + counts["heldout_edited"] + counts["heldout_m0"]
    assert counts["dev_edited"] == 864 and counts["heldout_edited"] == 960 and counts["primary_total"] == 1928, counts
    bundle = {"project_id": PROJECT_ID, "arms": P.ARMS, "dev_target": P.DEV_TARGET,
              "heldout_targets": P.HELDOUT_TARGETS, "panels": P.PANELS, "dev_cases": dev,
              "heldout_cases": ho, "technical_cases": tech, "dev_trajectories": dev_traj,
              "heldout_trajectories": ho_traj, "counts": counts,
              "train_seed_dev": P.TRAIN_SEED_DEV, "train_seeds_heldout": P.TRAIN_SEEDS_HELDOUT,
              "bootstrap_seed": P.BOOTSTRAP_SEED, "noise_shape_unpacked": list(NOISE_SHAPE),
              "noise_hash_definition": "sha256 of float32 bytes of the PACKED CPU torch.randn(1,16,64,64; "
                                       "torch.Generator('cpu').manual_seed(eval_seed))"}
    dump(bundle, os.path.join(ROOT, "configs", "cases.json"))
    print("cases manifest", counts, "bootstrap_seed", P.BOOTSTRAP_SEED)


if __name__ == "__main__":
    build_cases()
