"""Train the tiny base mask predictor ("original model", sees all facts) and the retrain reference
("retrain-without-forget", never sees forget facts) from the same initialisation with the LLaDA SFT loss.

CLI:  python -m mdmu.base_train --world-seed 0 --init-seed 0 --steps 3000 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import toyworld as tw
from .diffusion import sft_loss
from .evaluate import fact_accuracy, copy_accuracy, DECODING_CONFIGS
from .model import MaskPredictor


def build_pool(world: tw.World, include_forget: bool, fact_repeat: int = 1):
    facts = world.facts if include_forget else world.retain_facts
    ex = world.examples(facts) * fact_repeat + world.copy_train
    ids = torch.tensor([e.prompt + e.answer for e in ex], dtype=torch.long)
    return ids


def train(world: tw.World, include_forget: bool, init_seed: int, steps: int, batch: int = 128,
          lr: float = 3e-3, data_seed: int = 0, log_every: int = 500, model_kw=None):
    torch.manual_seed(init_seed)
    model = MaskPredictor(tw.V, tw.P_LEN + tw.A_LEN, **(model_kw or {}))
    pool = build_pool(world, include_forget)
    g = torch.Generator().manual_seed(data_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / steps))
    log = []
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, pool.shape[0], (batch,), generator=g)
        loss = sft_loss(model, pool[idx], tw.P_LEN, generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (s + 1) % log_every == 0:
            log.append({"step": s + 1, "loss": float(loss.detach()), "elapsed_s": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    return model, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", choices=["base", "retrain", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    world = tw.make_world(a.world_seed)
    summary = {}
    for name, inc in (("base", True), ("retrain", False)):
        if a.which not in (name, "both"):
            continue
        t0 = time.time()
        model, log = train(world, inc, a.init_seed, a.steps, data_seed=a.init_seed + 1000 * (not inc))
        model.eval()
        ev = {cfg: {"forget": fact_accuracy(model, world.forget_facts, cfg),
                    "neighbor": fact_accuracy(model, world.neighbor_facts, cfg),
                    "other": fact_accuracy(model, world.other_facts, cfg),
                    "copy": copy_accuracy(model, world.copy_eval, cfg)} for cfg in ("D1_lc_greedy", "D2_rand_T1")}
        torch.save({"state_dict": model.state_dict(), "cfg": model.cfg, "world_seed": a.world_seed,
                    "init_seed": a.init_seed, "steps": a.steps, "include_forget": inc},
                   os.path.join(a.out, f"{name}.pt"))
        summary[name] = {"train_log": log, "eval": ev, "wall_s": round(time.time() - t0, 1)}
        print(name, json.dumps(ev), flush=True)
    with open(os.path.join(a.out, "base_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
