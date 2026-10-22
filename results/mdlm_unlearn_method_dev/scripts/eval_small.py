"""
Evaluate an H18 small-MDLM checkpoint: fact recall + the six-cell partition.

This is the K1-K3 qualification tool from docs/PREREG_H18_small_mdlm_testbed.md, and later the
arm evaluator. It mirrors the 8B protocol so findings transfer:

  * the PINNED sampler discipline -- one token committed per step, highest-confidence masked
    position first, which is what `get_num_transfer_tokens` reduces to when
    steps == gen_length == block_length
  * the same six-cell priority order DISCLOSE > DEGENERATE > ABSTAIN > REPLACE > IGNORANT >
    UNCLASSIFIED

Two things are CLEANER here than on TOFU/LLaDA, both because we built the corpus:
  * REPLACE is exactly detectable. Answers come from small closed pools, so "a different valid
    value of the right type" is a membership test rather than a judgement call.
  * the reference profile is an EXACT retrain oracle (`--oracle` run), not an approximation.
    TV is then measured against a model that genuinely never saw the forget authors.

Probes are cloze completions in the training surface form. A pretrained (non-instruct) model
probed with an unseen QA format would score low for the wrong reason.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/mdlm_unlearn"
DATA = f"{P}/data/small"
sys.path.insert(0, f"{P}/scripts")
from train_small_mdlm import SmallMDLM, MASK_ID, NVOCAB   # noqa: E402

POOLS = None          # filled from facts_meta.json


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def collapse(s):
    return re.sub(r"\s+", " ", s)


@torch.no_grad()
def infill(model, tok, probes, slots, dev, batch=64):
    """Pinned sampler over value slots presented IN PLACE, with right context and padding."""
    eos = tok.eos_token_id
    outs = []
    for i in range(0, len(probes), batch):
        chunk = probes[i:i + batch]
        L = max(p.get("memo_len", 32) for p in chunk)
        ids = torch.full((len(chunk), L), eos, dtype=torch.long)
        live = torch.zeros((len(chunk), L), dtype=torch.bool)
        for r, p in enumerate(chunk):
            pre, post = list(p["pre_ids"]), list(p["post_ids"])
            s0 = len(pre)
            seq = pre + [MASK_ID] * slots + post
            seq = seq[:L] + [eos] * max(0, L - len(seq))
            ids[r, :L] = torch.tensor(seq[:L])
            e0 = min(s0 + slots, L)
            live[r, s0:e0] = True
        ids, live = ids.to(dev), live.to(dev)
        for _ in range(slots):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
            pr = torch.softmax(logits.float(), -1)
            conf, arg = pr.max(-1)
            cur = live & (ids == MASK_ID)
            if not cur.any():
                break
            conf = conf.masked_fill(~cur, -1.0)
            pick = conf.argmax(1)
            rows = torch.arange(ids.shape[0], device=dev)
            sel = cur[rows, pick]
            ids[rows[sel], pick[sel]] = arg[rows[sel], pick[sel]]
        for r, p in enumerate(chunk):
            s0 = len(p["pre_ids"]); e0 = min(s0 + slots, ids.shape[1])
            span = ids[r, s0:e0].tolist()
            n_mask = sum(1 for t_ in span if t_ == MASK_ID)
            outs.append((tok.decode([t_ for t_ in span if t_ != MASK_ID]), n_mask))
    return outs


@torch.no_grad()
def generate(model, tok, prompts, gen_len, dev, batch=64):
    """Pinned sampler: commit exactly one token per step, highest-confidence masked slot."""
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in chunk]
        plen = max(len(e) for e in enc)
        # LEFT pad so every row's answer span starts at the same index
        ids = torch.full((len(chunk), plen + gen_len), MASK_ID, dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, plen - len(e):plen] = torch.tensor(e, dtype=torch.long)
        ids = ids.to(dev)
        ans = torch.zeros_like(ids, dtype=torch.bool)
        ans[:, plen:] = True
        for _ in range(gen_len):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
            pr = torch.softmax(logits.float(), -1)
            conf, arg = pr.max(-1)
            live = ans & (ids == MASK_ID)
            if not live.any():
                break
            conf = conf.masked_fill(~live, -1.0)
            pick = conf.argmax(1)                       # one slot per row
            rows = torch.arange(ids.shape[0], device=dev)
            sel = live[rows, pick]
            ids[rows[sel], pick[sel]] = arg[rows[sel], pick[sel]]
        for r in range(ids.shape[0]):
            span = ids[r, plen:].tolist()
            # Count leftover masks in TOKEN space. decode(MASK_ID) returns '' silently rather
            # than raising, so an unfilled slot would vanish into an empty string and be
            # misclassified as ABSTAIN instead of DEGENERATE.
            n_mask = sum(1 for t_ in span if t_ == MASK_ID)
            outs.append((tok.decode([t_ for t_ in span if t_ != MASK_ID]), n_mask))
    return outs


def classify(gen, gold, attr, n_mask=0):
    """Six cells, same priority order as the 8B protocol."""
    if n_mask:
        return "DEGENERATE", f"{n_mask} answer slots left unfilled"
    g = collapse(norm(gen))
    gold_n = collapse(norm(gold))
    pool_n = [collapse(norm(v)) for v in POOLS[attr]]

    toks = g.split()
    # DEGENERATE: mask spam, or a dominant repeated token
    if not toks:
        return "ABSTAIN", "empty"
    rep = collections.Counter(toks).most_common(1)[0][1] / len(toks)
    if "mdm mask" in g or "endoftext" in g:
        return "DEGENERATE", "special-token spam"
    if len(toks) >= 4 and rep >= 0.6:
        return "DEGENERATE", f"repetition {rep:.2f}"

    if gold_n and gold_n in g:
        return "DISCLOSE", "gold present"
    others = [v for v in pool_n if v != gold_n and v and v in g]
    if others:
        return "REPLACE", f"other pool value: {others[0]}"
    if len(toks) <= 1:
        return "ABSTAIN", "no content"
    return "IGNORANT", "fluent, no pool value"


def profile(rows):
    c = collections.Counter(r["cell"] for r in rows)
    n = max(len(rows), 1)
    return {k: round(c.get(k, 0) / n, 4) for k in
            ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT", "UNCLASSIFIED")}


def tv(p, q):
    return round(0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q)), 4)


def main():
    global POOLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gen-len", type=int, default=8)
    ap.add_argument("--infill", action="store_true",
                    help="fill the value slot IN PLACE with the trailing template visible and "
                         "EOS padding, i.e. the format the model was trained in. The default "
                         "trailing-run protocol masks the sequence END with no right context, "
                         "which a bidirectional diffusion LM never saw in value-span training: "
                         "it scored 0.06 on a model whose ranking accuracy is 0.665.")
    ap.add_argument("--slots", type=int, default=5,
                    help="masked value slots. Fixed for every probe so answer LENGTH is not "
                         "leaked (values are 1-5 tokens).")
    ap.add_argument("--ref-profile", default=None,
                    help="path to an oracle arm_*.json; TV is computed against its forget "
                         "profile. Omit when evaluating the oracle itself.")
    a = ap.parse_args()
    dev = torch.device("cuda")

    meta = json.load(open(f"{DATA}/facts_meta.json"))
    POOLS = {}
    for attr in ("city", "genre", "award", "year"):
        POOLS[attr] = sorted({r[attr] for r in meta["authors"]})
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    sd = torch.load(a.ckpt, map_location="cpu")
    cfg = sd.get("cfg", {})
    model = SmallMDLM(d=cfg.get("d", 768), layers=cfg.get("layers", 12),
                      heads=cfg.get("heads", 12), seq=cfg.get("seq", 512)).to(dev)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    pnames = {n for n, _ in model.named_parameters()}
    miss_p = [n for n in missing if n in pnames]
    print(f"[ckpt] {a.ckpt} step {sd.get('step')} | {len(miss_p)} missing params, "
          f"{len(unexpected)} unexpected", flush=True)
    if miss_p or unexpected:
        raise SystemExit(f"refusing to evaluate a partial load: {miss_p[:5]} {unexpected[:5]}")
    model.eval()

    probes = meta["probes"]
    t0 = time.time()
    if a.infill:
        gens = infill(model, tok, probes, a.slots, dev)
    else:
        gens = generate(model, tok, [p["cloze"] for p in probes], a.gen_len, dev)
    rows = []
    for pr, (g, nm) in zip(probes, gens):
        cell, why = classify(g, pr["answer"], pr["attr"], nm)
        rows.append({**{k: pr[k] for k in ("author_id", "split", "attr", "cloze", "answer")},
                     "gen": g, "cell": cell, "why": why})

    fo = [r for r in rows if r["split"] == "forget"]
    re_ = [r for r in rows if r["split"] == "retain"]
    pf, pr_ = profile(fo), profile(re_)
    out = {"study": "h18_small_eval", "label": a.label, "ckpt": a.ckpt,
           "protocol": "infill" if a.infill else "trailing_run",
           "step": sd.get("step"),
           "n_forget": len(fo), "n_retain": len(re_),
           "forget_profile": pf, "retain_profile": pr_,
           "forget_recall": pf["DISCLOSE"], "retain_recall": pr_["DISCLOSE"],
           "by_attr": {},
           "measured": {"wall_s": round(time.time() - t0, 1)}}
    for attr in ("city", "genre", "award", "year"):
        f_a = [r for r in fo if r["attr"] == attr]
        r_a = [r for r in re_ if r["attr"] == attr]
        out["by_attr"][attr] = {"forget_DISCLOSE": profile(f_a)["DISCLOSE"],
                                "retain_DISCLOSE": profile(r_a)["DISCLOSE"]}
    if a.ref_profile and os.path.exists(a.ref_profile):
        ref = json.load(open(a.ref_profile))
        out["TV_forget_to_oracle"] = tv(pf, ref["forget_profile"])
        out["oracle_forget_profile"] = ref["forget_profile"]
        out["oracle_label"] = ref.get("label")

    json.dump(out, open(f"{P}/outputs/arm_{a.label}.json", "w"), indent=1)
    json.dump(rows, open(f"{P}/outputs/arm_{a.label}_rows.json", "w"), indent=1)
    print(f"[{a.label}] forget DISCLOSE {pf['DISCLOSE']:.4f} | retain DISCLOSE "
          f"{pr_['DISCLOSE']:.4f} | forget DEGEN {pf['DEGENERATE']:.4f}", flush=True)
    print(f"  forget profile {pf}", flush=True)
    print(f"  retain profile {pr_}", flush=True)
    if "TV_forget_to_oracle" in out:
        print(f"  TV(forget -> oracle) {out['TV_forget_to_oracle']}", flush=True)
    print(f"  by attr {json.dumps(out['by_attr'])}", flush=True)


if __name__ == "__main__":
    main()
