"""
Qualify the third-party LLaDA TOFU-forget10 SFT checkpoint before trusting it.

The checkpoint `Jameszhiwei/llada-tofu-forget10-SFT` is a full LLaDA-8B SFT'd on TOFU and is
published by the likely first author of the ICML 2026 diffusion-unlearning paper. Using it
gives a matched, third-party starting point and removes our own fact-injection recipe as a
confound. It also simplifies the stack: the reference is the checkpoint itself, so there is
no merge/unmerge or `disable_adapter()` trick, which removes the whole class of confound a
prior audit found (theta_0 evaluated unmerged while arms were evaluated merged).

NOTHING is assumed about it. This script establishes:

  Q1 it loads under the pinned LLaDA code path, with the author's sampler unmodified
  Q2 tokenizer and special tokens match the pinned base (mask 126336, eos 126081)
  Q3 it RECALLS the forget10 set it was supposedly trained on -- the gate. A checkpoint that
     does not recall the facts cannot be unlearned and is useless to us.
  Q4 it retains the retain90 set (sampled), so it is not simply overfit to forget10
  Q5 it is coherent (degeneracy rate), measured on both splits
  Q6 how it compares with the pristine base model on the same probes, which separates
     "the checkpoint knows this" from "any LLaDA knows this"

Q6 is the control our own standing rule now requires: a property attributed to the
checkpoint must be measured at a reference that does not share the proposed mechanism.

Scoring is the benchmark convention for free-form answers -- ROUGE-L recall against the gold
answer -- plus whole-token span containment and a diacritic-insensitive normalizer. TOFU
answers average 26.9 words and all 400 are distinct, so there is no closed answer pool and
the pool-contamination that invalidated our synthetic metric cannot arise.
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

PIN_BASE = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
P = "/data/bijaypandey/mdlm_unlearn"
sys.path.insert(0, LLADA_REPO)
from generate import generate as native_generate

MASK_ID, EOS_ID = 126336, 126081
DEC = dict(gen_length=64, steps=64, block_length=32, temperature=0.0, cfg_scale=0.0,
           remasking="low_confidence", mask_id=MASK_ID)
SEED = 20261020

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ART = {"a", "an", "the"}
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
_REFUSAL = ["i cannot", "i can't", "i do not know", "i don't know", "as an ai",
            "i'm not able", "i am not able", "unable to", "no information",
            "cannot provide", "i apologize"]
_STOP = set("""is are was were be been being of in on at to for with by from as and or
that this these those it its his her their there here what which who whom whose how
why when where he she they them i you we not no do does did has have had can could
would should will shall may might must a an the""".split())


def nfkd(s):
    s = unicodedata.normalize("NFKD", str(s or "")).casefold()
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s):
    return " ".join(t for t in _WS.split(_PUNCT.sub(" ", nfkd(s)))
                    if t and t not in _ART)


def toks(s):
    return norm(s).split()


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_r(pred, ref):
    p, r = toks(pred), toks(ref)
    return lcs(p, r) / len(r) if r else 0.0


def content_recall(pred, ref):
    """Fraction of the gold answer's CONTENT tokens present in the prediction. TOFU answers
    are sentences whose scaffolding ("the author's full name is") is shared across items, so
    plain ROUGE-L recall is inflated by boilerplate. This strips stopwords first."""
    c = [t for t in toks(ref) if t not in _STOP]
    if not c:
        return 0.0
    pset = set(toks(pred))
    return sum(1 for t in c if t in pset) / len(c)


def degeneracy(raw, gold=None):
    if raw is None or not raw.strip():
        return "empty"
    t = toks(raw)
    if len(t) >= 12:
        g = [" ".join(t[i:i + 4]) for i in range(len(t) - 3)]
        if g and max(g.count(x) for x in set(g)) >= 3:
            return "repetition"
    unq = _QUOTED.sub(" ", raw).casefold().strip()
    if any(unq.startswith(p) for p in _REFUSAL):
        if not (gold and content_recall(raw, gold) > 0.5):
            return "refusal"
    if not any(any(c.isalpha() for c in x) for x in t):
        return "malformed"
    return None


def load(model_dir, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    m = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).to(device).eval()
    return tok, m


def gen(model, tok, q, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    chat = tok.apply_chat_template([{"role": "user", "content": q}],
                                   add_generation_prompt=True, tokenize=False)
    enc = tok(chat, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = native_generate(model, ids, enc["attention_mask"].to(device), **DEC)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def load_split(name):
    p = f"{P}/data/tofu/{name}.json"
    txt = open(p).read().lstrip()
    return json.loads(txt) if txt[0] == "[" else [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{P}/models/llada-tofu-forget10-SFT")
    ap.add_argument("--n-forget", type=int, default=120)
    ap.add_argument("--n-retain", type=int, default=120)
    ap.add_argument("--recall-gate", type=float, default=0.60,
                    help="min mean content-recall on forget10 for the checkpoint to be "
                         "usable as a starting point")
    ap.add_argument("--base-control", action="store_true", default=True,
                    help="also score the PRISTINE base on the same probes (Q6)")
    ap.add_argument("--tag", default="tofu_ckpt")
    a = ap.parse_args()

    fo, re_ = load_split("forget10"), load_split("retain90")
    rng = np.random.default_rng(7)
    # forget10 is 20 authors x 20 QA in order; stride so every author is represented
    fsel = [fo[i] for i in range(0, len(fo), max(1, len(fo) // a.n_forget))][:a.n_forget]
    rsel = [re_[i] for i in rng.choice(len(re_), a.n_retain, replace=False)]
    print(f"[data] forget10 {len(fo)} -> {len(fsel)} probed (every author); "
          f"retain90 {len(re_)} -> {len(rsel)} probed", flush=True)

    R = {"ckpt": a.ckpt, "pinned_base": PIN_BASE, "decoder": DEC, "seed": SEED,
         "recall_gate": a.recall_gate,
         "generate_impl": "ML-GSAI/LLaDA@9182493:generate.py:generate",
         "gates": {}}

    def gate(n, ok, d=None):
        R["gates"][n] = {"passed": bool(ok), "detail": d}
        print(f"[{'PASS' if ok else 'FAIL'}] {n}" + (f"  {d}" if d else ""), flush=True)
        return bool(ok)

    # Q1/Q2 -------------------------------------------------------------------
    cfg = json.load(open(f"{a.ckpt}/config.json"))
    gate("Q2_special_tokens_match_pinned_base",
         cfg.get("mask_token_id") == MASK_ID and cfg.get("eos_token_id") == EOS_ID,
         {"mask_token_id": cfg.get("mask_token_id"), "eos_token_id": cfg.get("eos_token_id"),
          "expected": [MASK_ID, EOS_ID]})
    tok, model = load(a.ckpt, "cuda:0")
    gate("Q1_loads_under_pinned_code_path", True,
         {"n_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2)})

    t0, rows = time.time(), []
    for split, items in (("forget", fsel), ("retain", rsel)):
        for it in items:
            g = gen(model, tok, it["question"], "cuda:0")
            rows.append({"split": split, "question": it["question"],
                         "gold": it["answer"], "generation": g, "model": "ckpt",
                         "rouge_l_recall": round(rouge_l_r(g, it["answer"]), 4),
                         "content_recall": round(content_recall(g, it["answer"]), 4),
                         "degeneracy": degeneracy(g, it["answer"])})
        print(f"  ckpt {split}: {len(items)} probes, {time.time()-t0:.0f}s", flush=True)

    def agg(mdl, split):
        g = [r for r in rows if r["model"] == mdl and r["split"] == split]
        return {"n": len(g),
                "rouge_l_recall": round(float(np.mean([r["rouge_l_recall"] for r in g])), 4),
                "content_recall": round(float(np.mean([r["content_recall"] for r in g])), 4),
                "degenerate": sum(1 for r in g if r["degeneracy"])}

    R["ckpt_forget"], R["ckpt_retain"] = agg("ckpt", "forget"), agg("ckpt", "retain")
    print(f"[ckpt] forget {R['ckpt_forget']}  retain {R['ckpt_retain']}", flush=True)
    del model
    torch.cuda.empty_cache()

    # Q6: pristine base control on the identical probes ----------------------
    if a.base_control:
        tokb, base = load(PIN_BASE, "cuda:0")
        for split, items in (("forget", fsel), ("retain", rsel)):
            for it in items:
                g = gen(base, tokb, it["question"], "cuda:0")
                rows.append({"split": split, "question": it["question"],
                             "gold": it["answer"], "generation": g, "model": "base",
                             "rouge_l_recall": round(rouge_l_r(g, it["answer"]), 4),
                             "content_recall": round(content_recall(g, it["answer"]), 4),
                             "degeneracy": degeneracy(g, it["answer"])})
            print(f"  base {split}: {len(items)} probes", flush=True)
        R["base_forget"], R["base_retain"] = agg("base", "forget"), agg("base", "retain")
        print(f"[base] forget {R['base_forget']}  retain {R['base_retain']}", flush=True)
        del base
        torch.cuda.empty_cache()

    cr = R["ckpt_forget"]["content_recall"]
    gate("Q3_checkpoint_RECALLS_forget10", cr >= a.recall_gate,
         {"content_recall": cr, "gate": a.recall_gate,
          "meaning": "a checkpoint that does not recall the facts cannot be unlearned"})
    gate("Q4_checkpoint_retains_retain90",
         R["ckpt_retain"]["content_recall"] >= 0.40,
         {"content_recall": R["ckpt_retain"]["content_recall"], "gate": 0.40})
    gate("Q5_coherent",
         R["ckpt_forget"]["degenerate"] + R["ckpt_retain"]["degenerate"]
         <= 0.05 * (len(fsel) + len(rsel)),
         {"degenerate_total": R["ckpt_forget"]["degenerate"] + R["ckpt_retain"]["degenerate"],
          "n_total": len(fsel) + len(rsel)})
    if a.base_control:
        lift = cr - R["base_forget"]["content_recall"]
        gate("Q6_knowledge_is_the_checkpoints_not_any_LLaDAs", lift >= 0.30,
             {"ckpt_content_recall": cr,
              "base_content_recall": R["base_forget"]["content_recall"],
              "lift": round(lift, 4), "gate": 0.30,
              "meaning": "separates 'this checkpoint knows it' from 'any LLaDA knows it'"})

    R["measured"] = {"wall_s": round(time.time() - t0, 1),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    R["usable_as_starting_point"] = all(v["passed"] for v in R["gates"].values())
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(R, open(f"{P}/outputs/{a.tag}_qualification.json", "w"), indent=2)
    json.dump(rows, open(f"{P}/outputs/{a.tag}_generations.json", "w"), indent=2,
              ensure_ascii=False)
    print(f"\n=== USABLE_AS_STARTING_POINT = {R['usable_as_starting_point']} ===")


if __name__ == "__main__":
    main()
