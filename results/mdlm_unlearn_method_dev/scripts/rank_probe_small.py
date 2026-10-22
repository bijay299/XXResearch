"""
Likelihood-ranking probe for the H18 small MDLM.

WHY. Greedy generation says forget recall is 0.1425 against a ~0.11 chance floor, i.e. no
memorisation. But greedy decoding is the harshest possible read, and this project has produced
several instrument bugs tonight (a probe whose prompts varied with the pool, a save that wrote
nothing, a decode that silently returned ''). So before concluding the model CANNOT memorise,
test the weaker claim: does it rank the correct value above the alternatives?

Scores every candidate in the attribute's closed pool by teacher-forced log-likelihood of the
value tokens, with exactly those tokens masked. Top-1 accuracy against a 1/|pool| chance floor
separates two very different diagnoses:

  ranking >> chance, generation at chance  -> the knowledge is there; the sampler or probe is
                                              the problem
  ranking at chance                        -> genuinely not memorised; the backbone needs more
                                              capacity, tokens, or fact exposure
"""
import argparse
import collections
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/mdlm_unlearn"
DATA = f"{P}/data/small"
sys.path.insert(0, f"{P}/scripts")
from train_small_mdlm import SmallMDLM, MASK_ID, NVOCAB   # noqa: E402


@torch.no_grad()
def score_candidates(model, tok, pre_ids, post_ids, cand_ids, dev, pad_to=32):
    """Length-normalised log p of each candidate's tokens, those tokens masked.

    Uses the EXACT token ids stored at corpus-build time, so the probe and the memo examples
    share one tokenisation path. Reconstructing from strings put a BPE leading-space mismatch in
    the middle of the measurement: training tokenised the value bare after a prefix ending in a
    space ('Ghent' = [38, 6925]) while the probe tokenised ' ' + value ([402, 6925]). Different
    first token, so the probe was scoring a continuation the model was never taught.
    """
    eos = tok.eos_token_id
    rows, spans = [], []
    for vt in cand_ids:
        seq = list(pre_ids) + list(vt) + list(post_ids)
        seq = seq[:pad_to] + [eos] * max(0, pad_to - len(seq))
        rows.append(seq)
        spans.append((len(pre_ids), len(pre_ids) + len(vt)))
    L = pad_to
    ids = torch.full((len(rows), L), MASK_ID, dtype=torch.long)
    tgt = torch.full((len(rows), L), -100, dtype=torch.long)
    for i, (r, (s, e)) in enumerate(zip(rows, spans)):
        ids[i, :L] = torch.tensor(r[:L])
        if e <= L:
            tgt[i, s:e] = torch.tensor(r[s:e])
            ids[i, s:e] = MASK_ID
    ids, tgt = ids.to(dev), tgt.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    lp = torch.log_softmax(logits.float(), -1)
    out = []
    for i in range(len(rows)):
        s, e = spans[i]
        e = min(e, L)
        tot = sum(float(lp[i, j, tgt[i, j]]) for j in range(s, e))
        out.append(tot / max(e - s, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    meta = json.load(open(f"{DATA}/facts_meta.json"))
    pools = {at: sorted({r[at] for r in meta["authors"]})
             for at in ("city", "genre", "award", "year")}
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    sd = torch.load(a.ckpt, map_location="cpu")
    cfg = sd.get("cfg", {})
    model = SmallMDLM(d=cfg.get("d", 768), layers=cfg.get("layers", 12),
                      heads=cfg.get("heads", 12), seq=cfg.get("seq", 512)).to(dev)
    miss, unexp = model.load_state_dict(sd["model"], strict=False)
    pn = {n for n, _ in model.named_parameters()}
    mp = [n for n in miss if n in pn]
    if mp or unexp:
        raise SystemExit(f"refusing a partial load: {mp[:5]} {unexp[:5]}")
    model.eval()
    print(f"[ckpt] {a.ckpt} step {sd.get('step')}", flush=True)

    # candidates tokenised BARE, exactly as the corpus builder tokenised values
    cand_tok = {at: {v: tok(v, add_special_tokens=False)["input_ids"] for v in vs}
                for at, vs in pools.items()}
    agg = collections.defaultdict(lambda: {"n": 0, "top1": 0, "rank_sum": 0})
    for i, pr in enumerate(meta["probes"]):
        cands = pools[pr["attr"]]
        cand_ids = [cand_tok[pr["attr"]][v] for v in cands]
        sc = score_candidates(model, tok, pr["pre_ids"], pr["post_ids"], cand_ids, dev,
                              pad_to=pr.get("memo_len", 32))
        order = sorted(range(len(cands)), key=lambda k: -sc[k])
        gold_i = cands.index(pr["answer"])
        rank = order.index(gold_i) + 1
        for key in (pr["split"], f"{pr['split']}/{pr['attr']}"):
            d = agg[key]
            d["n"] += 1
            d["top1"] += int(rank == 1)
            d["rank_sum"] += rank
        if i % 200 == 0:
            print(f"  .. {i}/{len(meta['probes'])}", flush=True)

    out = {"study": "h18_rank_probe", "label": a.label, "ckpt": a.ckpt,
           "step": sd.get("step"), "pool_sizes": {k: len(v) for k, v in pools.items()},
           "results": {}}
    for key, d in sorted(agg.items()):
        attr = key.split("/")[1] if "/" in key else None
        chance = 1.0 / len(pools[attr]) if attr else float(
            np.mean([1.0 / len(pools[at]) for at in pools]))
        out["results"][key] = {"n": d["n"], "top1": round(d["top1"] / d["n"], 4),
                               "chance": round(chance, 4),
                               "mean_rank": round(d["rank_sum"] / d["n"], 2),
                               "above_chance": round(d["top1"] / d["n"] - chance, 4)}
    json.dump(out, open(f"{P}/outputs/rank_{a.label}.json", "w"), indent=1)
    for k, v in out["results"].items():
        print(f"  {k:20s} top1={v['top1']:.4f} chance={v['chance']:.4f} "
              f"above={v['above_chance']:+.4f} mean_rank={v['mean_rank']}", flush=True)


if __name__ == "__main__":
    main()
