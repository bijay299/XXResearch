"""Qualify a toy checkpoint: does it know the facts it was trained on, and what does it say for strangers?

CLI: python -m aslu.qualify --ckpt <base.pt> --world-seed 0 [--out json]
"""
from __future__ import annotations

import argparse
import json

import torch

from . import world as W
from .base_train import load
from .evalkit import exact_match, label_rates, lives_disclosure, referent_dist


def qualify(model, world: W.World, n_stranger: int = 16, seed: int = 0):
    ppl, fg, rt = world.people, world.forget, world.retain_people
    out = {"lives_greedy_forget": lives_disclosure(model, world, fg),
           "lives_greedy_retain": lives_disclosure(model, world, rt),
           "lives_T1_forget": lives_disclosure(model, world, fg, cfg="D2_rand_T1", n=8, seed=seed)}
    out["born_em"] = exact_match(model, [(W.q_born(p.name), W.a_born(p.name, p.born)) for p in ppl], "D1_lc_greedy")
    out["work_em"] = exact_match(model, [(W.q_work(p.name), W.a_work(p.name, p.work)) for p in ppl], "D1_lc_greedy")
    out["job_em"] = exact_match(model, [(W.q_job(p.name), W.a_job(p.job)) for p in ppl], "D1_lc_greedy")
    out["forget_other_attr_em"] = {
        "born": exact_match(model, [(W.q_born(p.name), W.a_born(p.name, p.born)) for p in fg], "D1_lc_greedy"),
        "work": exact_match(model, [(W.q_work(p.name), W.a_work(p.name, p.work)) for p in fg], "D1_lc_greedy"),
        "job": exact_match(model, [(W.q_job(p.name), W.a_job(p.job)) for p in fg], "D1_lc_greedy")}
    out["ctok_em"] = exact_match(model, W.CTOK, "D1_lc_greedy")
    out["copy_eval_em"] = exact_match(model, [(W.q_copy(c), W.a_copy(c)) for c in world.copy_eval], "D1_lc_greedy")
    out["copy_train_em"] = exact_match(model, [(W.q_copy(c), W.a_copy(c)) for c in world.copy_train[:40]],
                                       "D1_lc_greedy")
    for nm, names in (("stranger_eval", world.novel_eval),):
        for tpl in (0, 2):
            d = referent_dist(model, [W.q_lives(n, tpl) for n in names], "D2_rand_T1", n_stranger, seed)
            r = label_rates(d)
            out[f"{nm}_Q{tpl}_T1"] = r
            out[f"{nm}_Q{tpl}_T1_TV_vs_prior"] = 0.5 * sum(abs(r.get(c, 0) - W.PRIOR[c]) for c in W.PRIOR) + \
                0.5 * sum(v for k, v in r.items() if k not in W.PRIOR)
        gq = referent_dist(model, [W.q_lives(n, 0) for n in names], "D1_lc_greedy")
        out[f"{nm}_Q0_greedy"] = label_rates(gq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--world-seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(1)
    res = qualify(load(a.ckpt), W.make_world(a.world_seed))
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
