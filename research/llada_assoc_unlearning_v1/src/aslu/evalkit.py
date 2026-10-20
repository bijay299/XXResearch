"""Raw-generation evaluation for the association-selective world (native sampler only; no logit hooks).

Every metric is computed from sampled/greedy answers mapped by κ (`world.referent`) — never from internal logits —
except the explicitly labelled likelihood diagnostics.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import torch

from . import world as W
from .diffusion import generate

DECODE = {
    "D1_lc_greedy": dict(steps=8, block_length=8, temperature=0.0, remasking="low_confidence"),
    "D2_rand_T1": dict(steps=8, block_length=8, temperature=1.0, remasking="random"),
    "H1_lc_block2": dict(steps=8, block_length=2, temperature=0.0, remasking="low_confidence"),
    "H2_lc_T07_4steps": dict(steps=4, block_length=8, temperature=0.7, remasking="low_confidence"),
}


@torch.no_grad()
def answers(model, prompts: Sequence[Sequence[int]], cfg: str, n: int = 1, seed: int = 0, chunk: int = 512):
    """Returns list (len(prompts)) of lists (n) of answer token lists."""
    c = DECODE[cfg]
    if c["temperature"] == 0.0 and c["remasking"] == "low_confidence":
        n = 1                                     # deterministic
    g = torch.Generator().manual_seed(seed)
    P = torch.tensor([list(p) for p in prompts for _ in range(n)], dtype=torch.long)
    outs = []
    for i in range(0, P.shape[0], chunk):
        x = generate(model, P[i:i + chunk], steps=c["steps"], gen_length=W.A_LEN, block_length=c["block_length"],
                     temperature=c["temperature"], remasking=c["remasking"], generator=g)
        outs.append(x[:, W.P_LEN:])
    X = torch.cat(outs).view(len(prompts), n, W.A_LEN).tolist()
    return X


def referent_dist(model, prompts, cfg, n=1, seed=0) -> List[Counter]:
    return [Counter(W.referent(a) for a in row) for row in answers(model, prompts, cfg, n, seed)]


def exact_match(model, pairs, cfg, seed=0) -> float:
    """pairs: list of (prompt, answer). Greedy-config exact string match of the full answer."""
    if not pairs:
        return float("nan")
    outs = answers(model, [p for p, _ in pairs], cfg, 1, seed)
    return sum(o[0] == list(a) for o, (_, a) in zip(outs, pairs)) / len(pairs)


def lives_disclosure(model, world: W.World, people, templates=(0, 1, 2, 3), cfg="D1_lc_greedy", n=1, seed=0):
    """Fraction of sampled lives_in answers whose referent equals the person's true city, per template group."""
    res = {}
    for tpl in templates:
        prompts = [W.q_lives(p.name, tpl) for p in people]
        d = referent_dist(model, prompts, cfg, n, seed)
        tot = sum(sum(c.values()) for c in d)
        res[f"Q{tpl}"] = sum(c[p.city] for c, p in zip(d, people)) / max(1, tot)
    return res


def label_rates(dists: List[Counter]) -> Dict[str, float]:
    tot = Counter()
    for c in dists:
        tot.update(c)
    n = sum(tot.values())
    return {k: v / n for k, v in sorted(tot.items())}
