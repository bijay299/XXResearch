"""Train the toy original model p0 (sees every fact) and the retrain reference (never sees any form of a forgotten
lives_in fact) from the same initialisation. Training mix (P §8): a fraction `pt_frac` of each batch uses the
PRETRAINING forward process (every position maskable, prompt included), the rest the LLaDA SFT process (prompt never
masked). Structure follows the closed sprint's `mdmu/base_train.py` (read-only reference); data and mix are new.

CLI:  python -m aslu.base_train --world-seed 0 --init-seed 0 --steps 6000 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import world as W
from .diffusion import pretrain_loss, sft_loss
from .model import MaskPredictor

MODEL_KW = dict(d=96, n_layers=3, n_heads=4, d_ff=288)


def pool_tensor(examples):
    return torch.tensor([pr + an for pr, an, _ in examples], dtype=torch.long)


def mixed_loss(model, ids, generator, pt_frac: float):
    b = ids.shape[0]
    n_pt = int(round(pt_frac * b))
    loss = 0.0
    if n_pt:
        loss = loss + pretrain_loss(model, ids[:n_pt], generator=generator) * (n_pt / b)
    if b - n_pt:
        loss = loss + sft_loss(model, ids[n_pt:], W.P_LEN, generator=generator) * ((b - n_pt) / b)
    return loss


def train(world: W.World, include_forget: bool, init_seed: int, steps: int, batch: int = 128, lr: float = 2e-3,
          pt_frac: float = 0.3, data_seed: int = 0, log_every: int = 500, model_kw=None):
    torch.manual_seed(init_seed)
    model = MaskPredictor(W.V, W.P_LEN + W.A_LEN, **(model_kw or MODEL_KW))
    pool = pool_tensor(W.build_training_set(world, include_forget))
    g = torch.Generator().manual_seed(data_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / steps))
    log, t0 = [], time.time()
    for s in range(steps):
        idx = torch.randint(0, pool.shape[0], (batch,), generator=g)
        loss = mixed_loss(model, pool[idx], g, pt_frac)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (s + 1) % log_every == 0:
            log.append({"step": s + 1, "loss": float(loss.detach()), "elapsed_s": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    model.eval()
    return model, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pt-frac", type=float, default=0.3)
    ap.add_argument("--which", choices=["base", "retrain", "both"], default="both")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    world = W.make_world(a.world_seed)
    for name, inc in (("base", True), ("retrain", False)):
        if a.which not in (name, "both"):
            continue
        t0 = time.time()
        model, log = train(world, inc, a.init_seed, a.steps, lr=a.lr, pt_frac=a.pt_frac,
                           data_seed=a.init_seed + 1000 * (not inc))
        torch.save({"state_dict": model.state_dict(), "cfg": model.cfg, "world_seed": a.world_seed,
                    "init_seed": a.init_seed, "steps": a.steps, "lr": a.lr, "pt_frac": a.pt_frac,
                    "include_forget": inc}, os.path.join(a.out, f"{name}.pt"))
        with open(os.path.join(a.out, f"{name}_trainlog.json"), "w") as f:
            json.dump({"log": log, "wall_s": round(time.time() - t0, 1)}, f, indent=1)


def load(path: str) -> MaskPredictor:
    ck = torch.load(path, map_location="cpu")
    m = MaskPredictor(W.V, W.P_LEN + W.A_LEN, **{k: v for k, v in ck["cfg"].items() if k not in ("vocab", "seq_len")})
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


if __name__ == "__main__":
    main()
