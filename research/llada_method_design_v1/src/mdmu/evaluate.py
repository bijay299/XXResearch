"""Generation-based evaluation under several native LLaDA decoding configurations.

Leakage is measured on *generated answers* (never on likelihood proxies alone):
  leak_full : the forgotten value token sequence appears contiguously anywhere in the generated answer
  leak_any  : any key (value) token of the forgotten fact appears anywhere in the generated answer
Well-formedness (guards against "forgetting" by degeneration): answer is one of the two trained phrasings
[FIRST LAST ATTR IS v.. DOT EOS..] / [ATTR OF FIRST LAST IS v.. DOT EOS..] with the correct subject/attribute
tokens and value tokens of the right attribute type and arity (any value, including a wrong one).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from . import toyworld as tw
from .diffusion import generate

DECODING_CONFIGS: Dict[str, dict] = {
    # single block, one token per step, greedy, confidence-ordered commits (LLaDA default style)
    "D1_lc_greedy": dict(steps=8, block_length=8, temperature=0.0, remasking="low_confidence"),
    # single block, random remasking order, temperature-1 sampling
    "D2_rand_T1": dict(steps=8, block_length=8, temperature=1.0, remasking="random"),
    # semi-autoregressive blocks of 2, greedy, confidence-ordered
    "D3_lc_block2": dict(steps=8, block_length=2, temperature=0.0, remasking="low_confidence"),
    # fewer steps (2 tokens per step), temperature 0.5, confidence-ordered
    "D4_lc_fast_T05": dict(steps=4, block_length=8, temperature=0.5, remasking="low_confidence"),
    # ---- SPEC v0.1 section 3 grid (gen_length 8), notation (steps, block_length, remasking, T) ----
    # training pool Pi_train (states for FX-OP1 / FX-Mix rollouts come ONLY from these)
    "P1": dict(steps=8, block_length=8, temperature=0.0, remasking="low_confidence"),   # deployment default (= D1)
    "P2": dict(steps=8, block_length=8, temperature=1.0, remasking="random"),           # wide sequential (= D2)
    "P3": dict(steps=4, block_length=8, temperature=1.0, remasking="random"),           # wide parallel, 2 tok/step
    # held-out evaluation configs (never used for training states; reported separately)
    "H1": dict(steps=8, block_length=2, temperature=0.0, remasking="low_confidence"),   # semi-AR blocks (= D3)
    "H2": dict(steps=4, block_length=8, temperature=0.5, remasking="low_confidence"),   # (= D4)
    "H3": dict(steps=8, block_length=8, temperature=0.0, remasking="random"),
    "H4": dict(steps=8, block_length=8, temperature=1.0, remasking="low_confidence"),
    "H5": dict(steps=4, block_length=4, temperature=0.7, remasking="random"),           # 2 blocks, 2 tok/step
    "H6": dict(steps=2, block_length=8, temperature=1.0, remasking="random"),           # 4 tok/step
}
TRAIN_POOL = ["P1", "P2", "P3"]
HELDOUT = ["H1", "H2", "H3", "H4", "H5", "H6"]
EVAL_GRID = TRAIN_POOL + HELDOUT
# SPEC v1.1: primary forgetting statistic = max over H1-H5; H6 (4 tokens/step > pool max 2) is an extrapolation cell,
# reported on its own and inside the max over H1-H6.
HELDOUT_PRIMARY = ["H1", "H2", "H3", "H4", "H5"]
EXTRAPOLATION = ["H6"]
# A config is sampled repeatedly if its output is random: T>0 (Gumbel) OR random remasking (random commit order).
# (Unchanged for D1-D4: D2 was already stochastic via T=1; the remasking clause only matters for new entry H3.)
STOCHASTIC = {k for k, v in DECODING_CONFIGS.items() if v["temperature"] > 0 or v["remasking"] == "random"}


def _gen(model, prompts: torch.Tensor, cfg: str, n_samples: int, seed: int, ban: torch.Tensor | None = None):
    c = DECODING_CONFIGS[cfg]
    g = torch.Generator().manual_seed(seed)
    reps = n_samples if cfg in STOCHASTIC else 1
    p = prompts.repeat_interleave(reps, 0)
    hook = None
    if ban is not None:   # decode-time F-ban (INFERENCE-TIME REFERENCE ONLY, never a deployed method)
        bm = ban.repeat_interleave(reps, 0)[:, None, :]

        def hook(x, logits):
            return logits.masked_fill(bm.expand_as(logits), float("-inf"))
    out = generate(model, p, steps=c["steps"], gen_length=tw.A_LEN, block_length=c["block_length"],
                   temperature=c["temperature"], remasking=c["remasking"], generator=g, logits_hook=hook)
    return out[:, tw.P_LEN:].view(prompts.shape[0], reps, tw.A_LEN)


def _contains(seq: Sequence[int], sub: Sequence[int]) -> bool:
    n = len(sub)
    return any(list(seq[i:i + n]) == list(sub) for i in range(len(seq) - n + 1))


def well_formed(ans: Sequence[int], f: tw.Fact) -> bool:
    k = len(f.value)
    at = tw.STOI[f"ATTR_{f.attr}"]
    heads = ([f.first, f.last, at, tw.STOI["IS"]], [at, tw.STOI["OF"], f.first, f.last, tw.STOI["IS"]])
    head = next((h for h in heads if list(ans[:len(h)]) == h), None)
    if head is None:
        return False
    h = len(head)
    if h + k + 1 > len(ans):
        return False
    vals = ans[h:h + k]
    allowed = tw.VALUE_TOKENS[f.attr]
    if f.attr == "city":
        ok = vals[0] in allowed[: tw.N_CA] and vals[1] in allowed[tw.N_CA:]
    elif f.attr == "year":
        ok = vals[0] in allowed[: tw.N_YD] and vals[1] in allowed[tw.N_YD:]
    else:
        ok = vals[0] in allowed
    rest = list(ans[h + k:])
    return bool(ok) and rest == [tw.STOI["DOT"]] + [tw.EOS] * (len(rest) - 1)


@torch.no_grad()
def fact_eval(model, facts: List[tw.Fact], cfg: str, templates=(0, 1), n_samples: int = 8, seed: int = 0,
              ban_values: bool = False, return_per_fact: bool = False):
    """Returns dict with rates over (fact, template, sample): correct (= leak_full), leak_any, well_formed.
    ban_values=True: decode with each fact's value tokens banned (logits = -inf) for its own prompt (reference)."""
    if not facts:
        return {"n": 0}
    ex = [(f, t) for f in facts for t in templates]
    prompts = torch.tensor([tw.fact_prompt(f, t) for f, t in ex], dtype=torch.long)
    ban = None
    if ban_values:
        ban = torch.zeros(len(ex), tw.V, dtype=torch.bool)
        for r, (f, _) in enumerate(ex):
            ban[r, list(f.value)] = True
    outs = _gen(model, prompts, cfg, n_samples, seed, ban=ban)
    full = anyk = wf = tot = 0
    per_fact: Dict = {}
    for (f, _), o in zip(ex, outs):
        for ans in o.tolist():
            tot += 1
            c = _contains(ans, f.value)
            full += c
            anyk += any(t in ans for t in f.value)
            wf += well_formed(ans, f)
            pf = per_fact.setdefault(f, [0, 0])
            pf[0] += c
            pf[1] += 1
    r = {"n": tot, "leak_full": full / tot, "leak_any": anyk / tot, "well_formed": wf / tot}
    if return_per_fact:
        r["per_fact"] = per_fact   # fact -> [n_correct (= leak_full hits), n_samples]
    return r


def fact_accuracy(model, facts, cfg, **kw) -> float:
    r = fact_eval(model, facts, cfg, **kw)
    return r.get("leak_full", float("nan"))


@torch.no_grad()
def copy_accuracy(model, examples: List[tw.Example], cfg: str, n_samples: int = 4, seed: int = 0) -> float:
    prompts = torch.tensor([e.prompt for e in examples], dtype=torch.long)
    outs = _gen(model, prompts, cfg, n_samples, seed)
    hit = tot = 0
    for e, o in zip(examples, outs):
        for ans in o.tolist():
            tot += 1
            hit += ans == e.answer
    return hit / tot
