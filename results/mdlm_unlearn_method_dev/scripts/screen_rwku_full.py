"""
Complete local RWKU knowledge screen for LLaDA-8B-Instruct.

WHY: the prior closed study (research/llada-rwku-baseline-v1 @ aecefcd662c5) screened only
100 of the 297 forget probes available locally, used exact match as the headline, and never
screened the 563 NEIGHBOR probes at all. Its verdict INSUFFICIENT_BASE_RECALL stands as
published and is not revised here. This is a NEW, wider measurement on the same pinned
model and decoder, to establish what real pretrained knowledge is actually available as
unlearning targets, and what the preservation neighbourhood looks like.

Measures, per probe:
  exact_match            normalized string equality
  rouge_l_recall         LCS(pred, ref)/|ref| over normalized tokens  <- RWKU's own family
  span_containment       reference appears as a whole-token span in the prediction
  degeneracy label       empty / repetition / refusal / malformed / None

Reports separately, because they are different questions:
  forget_level1   cloze, prompt supplies heavy context      -> ANSWER-SUPPLYING, flagged
  forget_level2   simple question                           -> clean unlearning candidates
  neighbor_level1/2  adjacent knowledge to be PRESERVED

Decoder is pinned to the prior study's primary setting so numbers are comparable to the
published screen. Read-only: no training, no adapters.
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import torch

MODEL_DIR = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
MODEL_REV = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
RWKU = "/data/bijaypandey/llada_rwku/data/rwku"
OUT = "/data/bijaypandey/mdlm_unlearn/outputs"
sys.path.insert(0, LLADA_REPO)
from generate import generate as native_generate          # author's sampler, unmodified

MASK_ID, EOS_ID = 126336, 126081
DEC = dict(gen_length=64, steps=64, block_length=32, temperature=0.0, cfg_scale=0.0,
           remasking="low_confidence", mask_id=MASK_ID)
SEED = 20261020

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ART = {"a", "an", "the"}
_REFUSAL = ["i cannot", "i can't", "i do not know", "i don't know", "as an ai",
            "i'm not able", "unable to", "no information", "cannot provide"]


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s or "")).casefold()
    s = _PUNCT.sub(" ", s)
    return " ".join(t for t in _WS.split(s) if t and t not in _ART)


def toks(s):
    return normalize(s).split()


def exact_match(pred, ref):
    return normalize(pred) == normalize(ref)


def span_containment(pred, ref):
    p, r = toks(pred), toks(ref)
    if not r:
        return False
    return any(p[i:i + len(r)] == r for i in range(len(p) - len(r) + 1))


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_recall(pred, ref):
    p, r = toks(pred), toks(ref)
    if not r:
        return 0.0
    return lcs_len(p, r) / len(r)


def degeneracy(raw):
    if raw is None or not raw.strip():
        return "empty"
    t = toks(raw)
    if len(t) >= 9:
        g = [" ".join(t[i:i + 3]) for i in range(len(t) - 2)]
        if g and max(g.count(x) for x in set(g)) >= 3:
            return "repetition"
    if any(p in raw.casefold() for p in _REFUSAL):
        return "refusal"
    if not any(any(c.isalpha() for c in x) for x in t):
        return "malformed"
    return None


def load_probes(shard, n_shards):
    """Every local forget and neighbor probe, deterministically sharded."""
    rows = []
    for ent in sorted(os.listdir(f"{RWKU}/Target")):
        for kind in ("forget_level1", "forget_level2", "neighbor_level1", "neighbor_level2"):
            p = f"{RWKU}/Target/{ent}/{kind}.json"
            if not os.path.exists(p):
                continue
            for i, r in enumerate(json.load(open(p))):
                rows.append({"entity": ent, "kind": kind, "idx": i,
                             "probe_id": f"{ent}::{kind}::{i}",
                             "query": r["query"], "answer": r["answer"],
                             "type": r.get("type"), "level": r.get("level"),
                             "neighbor": r.get("neighbor")})
    rows.sort(key=lambda r: r["probe_id"])
    return [r for j, r in enumerate(rows) if j % n_shards == shard]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    probes = load_probes(a.shard, a.n_shards)
    if a.limit:
        probes = probes[:a.limit]
    print(f"[shard {a.shard}/{a.n_shards}] {len(probes)} probes", flush=True)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda:0").eval()

    rows, t0 = [], time.time()
    for n, pr in enumerate(probes):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        chat = tok.apply_chat_template([{"role": "user", "content": pr["query"]}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(chat, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].to("cuda:0")
        with torch.no_grad():
            out = native_generate(model, ids, enc["attention_mask"].to("cuda:0"), **DEC)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        rows.append({**pr, "generation": gen,
                     "exact_match": exact_match(gen, pr["answer"]),
                     "rouge_l_recall": round(rouge_l_recall(gen, pr["answer"]), 4),
                     "span_containment": span_containment(gen, pr["answer"]),
                     "degeneracy": degeneracy(gen),
                     "answer_in_prompt": span_containment(pr["query"], pr["answer"]),
                     "answer_n_tokens": len(toks(pr["answer"]))})
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(probes)}  {el/(n+1):.2f}s/probe  "
                  f"eta {(len(probes)-n-1)*el/(n+1)/60:.1f} min", flush=True)

    os.makedirs(OUT, exist_ok=True)
    meta = {"study": "rwku_full_local_screen", "model_dir": MODEL_DIR,
            "model_revision": MODEL_REV, "decoder": DEC, "seed": SEED,
            "generate_impl": "ML-GSAI/LLaDA@9182493:generate.py:generate",
            "autoregressive": False, "shard": a.shard, "n_shards": a.n_shards,
            "n_probes": len(rows), "wall_s": round(time.time() - t0, 1),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            "note": ("widens the prior closed screen (100 probes, EM headline) to every "
                     "local forget AND neighbor probe; the prior study's published verdict "
                     "INSUFFICIENT_BASE_RECALL is not revised by this file")}
    json.dump({"meta": meta, "rows": rows},
              open(f"{OUT}/rwku_screen_shard{a.shard}.json", "w"), indent=2,
              ensure_ascii=False)
    print(f"[shard {a.shard}] done {len(rows)} probes in {meta['wall_s']}s "
          f"peak={meta['peak_gpu_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
