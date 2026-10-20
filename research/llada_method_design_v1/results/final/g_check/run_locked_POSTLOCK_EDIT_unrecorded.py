"""Locked validation runner (VALIDATION_SPEC.md sections 2-4). Reads ONE JSON config and writes one JSON per unit.

A unit = (arm, world W, unlearning seed S); it contains one training run per LR of the arm's locked list L*, each
evaluated every `eval_every` steps on the full grid (P1-P3, H1-H6) with mechanism statistics (off-manifold fraction
of first-commit decision states, certificate B_hat +- s.e.) at every checkpoint, plus compute counters and wall time.
The unlearning seed changes ONLY the unlearning RNG (LoRA A init, batch sampling, rollouts); world, base/retrain
checkpoints and evaluation seeds are fixed by the config.

Modes (run each through tools/ledger_run.py, 1 thread per process):
  --refs                    reference evals (base, retrain, decode-time F-ban) per world -> <out>/refs/refs_wW.json
  --shard i/n               the i-th of n shards of the (arm, W, S, LR) job list -> <out>/runs/<arm>_wW_sS_lr<lr>.json
  --assemble                merge complete runs into <out>/units/<arm>_wW_sS.json and list missing jobs
  --list                    print the job list and exit
Filters: --world W, --unlearn-seed S, --arm NAME (restrict the job list; never change any setting).
Refuses to run while any LR list is the placeholder "LR_STAR".
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from mdmu.unlearn import ARMS, RunConfig, reference_evals, run  # noqa: E402


def load_cfg(path):
    cfg = json.load(open(path))
    for arm, spec in cfg["arms"].items():
        assert arm in ARMS, f"unknown arm {arm}"
    return cfg


def check_locked(cfg):
    bad = [a for a, s in cfg["arms"].items() if not isinstance(s["lrs"], list)]
    if bad:
        raise SystemExit(f"LR list not locked (placeholder) for arms: {bad}")


def ckpt_dir(cfg, W):
    return os.path.join(ROOT, cfg["substrate_dir"], f"w{W}_i{W}")


def run_config(cfg, arm, W, S, lr) -> RunConfig:
    kw = dict(cfg.get("common", {}))
    kw.update(cfg["eval"])
    kw.update(cfg["arms"][arm].get("overrides", {}))
    kw.update(arm=arm, lr=float(lr), steps=cfg["steps"], eval_every=cfg["eval_every"], world_seed=W, init_seed=W,
              run_seed=S, ckpt_dir=ckpt_dir(cfg, W), eval_at_start=0, threads=cfg.get("threads_per_process", 1))
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    unknown = set(kw) - fields
    assert not unknown, f"unknown RunConfig keys in config: {unknown}"
    return RunConfig(**kw)


def jobs(cfg, world=None, seed=None, arm=None):
    out = []
    for a, spec in cfg["arms"].items():
        if arm and a != arm:
            continue
        for W in cfg["worlds"]:
            if world is not None and W != world:
                continue
            for S in cfg["unlearn_seeds"]:
                if seed is not None and S != seed:
                    continue
                for lr in (spec["lrs"] if isinstance(spec["lrs"], list) else ["LR_STAR"]):
                    out.append((a, W, S, lr))
    return out


def run_path(out, a, W, S, lr):
    return os.path.join(out, "runs", f"{a}_w{W}_s{S}_lr{float(lr):g}.json")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def do_refs(cfg, out, world=None):
    os.makedirs(os.path.join(out, "refs"), exist_ok=True)
    for W in cfg["worlds"]:
        if world is not None and W != world:
            continue
        p = os.path.join(out, "refs", f"refs_w{W}.json")
        if os.path.exists(p):
            continue
        rc = run_config(cfg, next(iter(cfg["arms"])), W, 0, 1e-3)
        t0 = time.time()
        refs = reference_evals(W, W, rc)
        refs["wall_s"] = round(time.time() - t0, 1)
        refs["ckpt_sha256"] = {k: sha(os.path.join(ckpt_dir(cfg, W), f"{k}.pt")) for k in ("base", "retrain")}
        json.dump(refs, open(p, "w"), indent=1, default=str)
        print(f"refs world {W} done ({refs['wall_s']} s)", flush=True)


def do_shard(cfg, out, i, n, **filt):
    check_locked(cfg)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    for k, (a, W, S, lr) in enumerate(jobs(cfg, **filt)):
        if k % n != i:
            continue
        p = run_path(out, a, W, S, lr)
        if os.path.exists(p):
            continue
        rc = run_config(cfg, a, W, S, lr)
        t0 = time.time()
        res, _, _ = run(rc)
        res.pop("snapshots")                      # mechanism is computed in-run at every checkpoint
        res["wall_s"] = round(time.time() - t0, 1)
        res["unit"] = {"arm": a, "world": W, "unlearn_seed": S, "lr": float(lr)}
        tmp = p + ".tmp"
        json.dump(res, open(tmp, "w"))
        os.replace(tmp, p)                        # atomic: a killed process never leaves a partial run file
        print(f"done {a} w{W} s{S} lr={lr} wall={res['wall_s']}s", flush=True)


def do_assemble(cfg, out):
    check_locked(cfg)
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    missing = []
    for a, spec in cfg["arms"].items():
        for W in cfg["worlds"]:
            for S in cfg["unlearn_seeds"]:
                paths = [run_path(out, a, W, S, lr) for lr in spec["lrs"]]
                miss = [p for p in paths if not os.path.exists(p)]
                if miss:
                    missing += miss
                    continue
                unit = {"arm": a, "world": W, "unlearn_seed": S, "lrs": spec["lrs"],
                        "runs": [json.load(open(p)) for p in paths]}
                json.dump(unit, open(os.path.join(out, "units", f"{a}_w{W}_s{S}.json"), "w"))
    print(json.dumps({"missing_runs": [os.path.basename(m) for m in missing]}, indent=1))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "validation_locked.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--unlearn-seed", type=int, default=None)
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    assert all(W >= 2 for W in cfg["worlds"]) or cfg.get("allow_dev_worlds"), "final worlds are >= 2"
    out = a.out or os.path.join(ROOT, cfg["out_dir"])
    torch.set_num_threads(cfg.get("threads_per_process", 1))
    filt = dict(world=a.world, seed=a.unlearn_seed, arm=a.arm)
    if a.list:
        for j in jobs(cfg, **filt):
            print(*j)
        return
    if a.shard or a.assemble:
        check_locked(cfg)                         # refuse before creating anything while LR_STAR remains
    os.makedirs(out, exist_ok=True)
    if a.refs:
        do_refs(cfg, out, a.world)
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        do_shard(cfg, out, i, n, **filt)
    if a.assemble:
        do_assemble(cfg, out)


if __name__ == "__main__":
    main()
