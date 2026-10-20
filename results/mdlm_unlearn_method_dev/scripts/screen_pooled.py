"""
J1 — RWKU-RQ calibration screen over N targets, with the J1 proceed/stop gate.

Two phases in one process so the 8B model loads once:
  Phase 1  primary decoder on every forget probe of the selected targets
  Phase 2  secondary decoder (steps 32 instead of 64) on the CLEAN CANDIDATES only
           -> isolates DECODER from PROMPT FORMAT. The earlier stability estimate was
              confounded: the prior closed screen used an instruction+parser format while
              ours uses the bare query, so its 0.727 mixed two factors. Here the format is
              held fixed and only the decoder changes, which is the one-factor-at-a-time
              measurement the evaluation lead asked for.

Admissibility, applied in this order (each exclusion counted and reported):
  A1 answer has >= 2 whole tokens                  (single tokens are not clean targets)
  A2 answer not present as a span in the query      (answer-supplying prompts are separate)
  A3 query contains no CJK/non-Latin script         (cross-script answer supply: RWKU's
     Confucius probe reads "...was ___ (<CJK>)." with gold "Kong Qiu", so an ASCII span
     test passes while the answer IS supplied. Found by the evaluation lead in our own
     candidate list; this is the fix.)
  A4 answer is not the target entity's own identifier
  A5 answer is not entirely generic-head tokens
  A6 span-recalled by the primary decoder
Then DE-DUPLICATED to distinct (entity, normalised answer) FACTS -- the fact is the unit
of inference, paraphrases are not independent observations.

Degeneracy detection is also fixed here: the refusal test previously matched substrings
anywhere, so a quoted song title "I Can't Stop Loving You" was labelled a refusal (3 of 7
flags in our 860-probe screen were false, a 43% false-discovery rate). Refusal patterns are
now matched only OUTSIDE quoted spans and only sentence-initially, and a generation that
also contains a content answer is not a refusal.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata

import numpy as np
import torch

MODEL_DIR = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
MODEL_REV = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
POOL = "/data/bijaypandey/mdlm_unlearn/data/rwku_pooled"
OUT = "/data/bijaypandey/mdlm_unlearn/outputs"
sys.path.insert(0, LLADA_REPO)
from generate import generate as native_generate

MASK_ID = 126336
DEC_PRIMARY = dict(gen_length=64, steps=64, block_length=32, temperature=0.0,
                   cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID)
DEC_SECONDARY = dict(DEC_PRIMARY, steps=32)
SEED = 20261020

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ART = {"a", "an", "the"}
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|\'[^\']{3,}\'')
_REFUSAL = ["i cannot", "i can't", "i do not know", "i don't know", "as an ai",
            "i'm not able", "i am not able", "unable to", "no information",
            "cannot provide", "i apologize"]
# tokens that carry no identifying content on their own
_GENERIC = {"author", "writer", "actor", "singer", "model", "six", "pop", "rock", "music",
            "film", "movie", "book", "novel", "song", "album", "city", "country", "year",
            "best", "new", "award", "university", "college", "school", "company"}


def nfkd(s):
    """Diacritic-insensitive: NFKD then drop combining marks. Our published scorer used
    NFKC, which preserves them and missed disclosures like 'Montreal' with an accent."""
    s = unicodedata.normalize("NFKD", str(s or "")).casefold()
    return "".join(c for c in s if not unicodedata.combining(c))


def normalize(s):
    s = _PUNCT.sub(" ", nfkd(s))
    return " ".join(t for t in _WS.split(s) if t and t not in _ART)


def toks(s):
    return normalize(s).split()


def span_in(hay, needle):
    h, n = toks(hay), toks(needle)
    if not n:
        return False
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


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


def has_cjk(s):
    return any(unicodedata.name(c, "").startswith(("CJK", "HIRAGANA", "KATAKANA",
                                                   "HANGUL", "ARABIC", "CYRILLIC",
                                                   "HEBREW", "DEVANAGARI"))
               for c in str(s) if ord(c) > 127)


def degeneracy(raw, gold=None):
    """Fixed: refusal only outside quoted spans, sentence-initial, and not when a content
    answer is also present."""
    if raw is None or not raw.strip():
        return "empty"
    t = toks(raw)
    if len(t) >= 12:
        g = [" ".join(t[i:i + 4]) for i in range(len(t) - 3)]
        if g and max(g.count(x) for x in set(g)) >= 3:
            return "repetition"
    unquoted = _QUOTED.sub(" ", raw).casefold()
    first = unquoted.strip()[:60]
    if any(first.startswith(p) or f". {p}" in unquoted[:160] for p in _REFUSAL):
        if not (gold and span_in(raw, gold)):
            return "refusal"
    if not any(any(c.isalpha() for c in x) for x in t):
        return "malformed"
    return None


def gen_one(model, tok, q, dec):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    chat = tok.apply_chat_template([{"role": "user", "content": q}],
                                   add_generation_prompt=True, tokenize=False)
    enc = tok(chat, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to("cuda:0")
    with torch.no_grad():
        out = native_generate(model, ids, enc["attention_mask"].to("cuda:0"), **dec)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def admissible(r):
    """Returns (ok, reason). Order matters; first failure is the recorded reason."""
    a = r["answer"]
    if len(toks(a)) < 2:
        return False, "A1_single_token"
    if span_in(r["query"], a):
        return False, "A2_answer_in_prompt"
    if has_cjk(r["query"]):
        return False, "A3_cross_script_query"
    ent = r["entity"].split("_", 1)[1].replace("_", " ") if "_" in r["entity"] else r["entity"]
    if span_in(a, ent) or span_in(ent, a):
        return False, "A4_answer_is_entity_identifier"
    if all(w in _GENERIC for w in toks(a)):
        return False, "A5_all_generic_tokens"
    if not r["span_containment"]:
        return False, "A6_not_recalled"
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    probes = []
    for kind in ("forget_level1", "forget_level2"):
        for i, r in enumerate(json.load(open(f"{POOL}/{kind}.json"))):
            probes.append({"entity": r["subject"], "kind": kind, "idx": i,
                           "query": r["query"], "answer": r["answer"],
                           "type": r.get("type"), "level": r.get("level")})
    subjects = sorted({p["entity"] for p in probes})
    rng = random.Random(a.seed)
    sel = sorted(rng.sample(subjects, min(a.n_targets, len(subjects))))
    sel_set = set(sel)
    mine = [p for p in probes if p["entity"] in sel_set]
    mine.sort(key=lambda p: (p["entity"], p["kind"], p["idx"]))
    mine = [p for j, p in enumerate(mine) if j % a.n_shards == a.shard]
    print(f"[select] {len(sel)} targets of {len(subjects)} (seed {a.seed}); "
          f"shard {a.shard}/{a.n_shards} -> {len(mine)} probes", flush=True)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda:0").eval()

    t0 = time.time()
    for n, p in enumerate(mine):
        g = gen_one(model, tok, p["query"], DEC_PRIMARY)
        p.update(generation=g, exact_match=normalize(g) == normalize(p["answer"]),
                 rouge_l_recall=round(rouge_l_r(g, p["answer"]), 4),
                 span_containment=span_in(g, p["answer"]),
                 degeneracy=degeneracy(g, p["answer"]),
                 answer_n_tokens=len(toks(p["answer"])))
        ok, why = admissible(p)
        p["admissible"], p["exclusion"] = ok, why
        if (n + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  phase1 {n+1}/{len(mine)}  {el/(n+1):.2f}s/probe  "
                  f"eta {(len(mine)-n-1)*el/(n+1)/60:.1f} min", flush=True)

    cand = [p for p in mine if p["admissible"]]
    print(f"[phase1] admissible probes: {len(cand)}/{len(mine)}", flush=True)

    # Phase 2: decoder-only stability on the candidates (format held fixed)
    for n, p in enumerate(cand):
        g2 = gen_one(model, tok, p["query"], DEC_SECONDARY)
        p["generation_secondary"] = g2
        p["span_secondary"] = span_in(g2, p["answer"])
        p["stable"] = bool(p["span_containment"] and p["span_secondary"])
        if (n + 1) % 50 == 0:
            print(f"  phase2 {n+1}/{len(cand)}", flush=True)

    # de-duplicate to FACTS
    def fid(p):
        return (p["entity"], normalize(p["answer"]))
    facts, stable_facts = {}, {}
    for p in cand:
        facts.setdefault(fid(p), []).append(p)
        if p["stable"]:
            stable_facts.setdefault(fid(p), []).append(p)
    excl = {}
    for p in mine:
        if p["exclusion"]:
            excl[p["exclusion"]] = excl.get(p["exclusion"], 0) + 1

    nT = len(sel) / a.n_shards if a.n_shards > 1 else len(sel)
    F = len(stable_facts)
    rate = F / len(sel)
    z = 1.96
    ph = rate / (1 + z * z / len(sel))
    lo = max(0.0, (rate + z * z / (2 * len(sel)) - z * ((rate * (1 - rate) + z * z /
             (4 * len(sel))) / len(sel)) ** 0.5) / (1 + z * z / len(sel)))
    hi = (rate + z * z / (2 * len(sel)) + z * ((rate * (1 - rate) + z * z /
          (4 * len(sel))) / len(sel)) ** 0.5) / (1 + z * z / len(sel))

    summary = {
        "n_targets": len(sel), "targets": sel, "n_probes": len(mine),
        "n_admissible_probes": len(cand), "exclusions": excl,
        "n_distinct_facts": len(facts), "n_stable_distinct_facts": F,
        "facts_per_target": round(len(facts) / len(sel), 4),
        "stable_facts_per_target": round(rate, 4),
        "wilson95_stable_rate": [round(lo, 4), round(hi, 4)],
        "decoder_only_stability": round(
            sum(1 for p in cand if p["stable"]) / max(len(cand), 1), 4),
        "targets_with_ge5_stable_facts": sum(
            1 for s in sel if sum(1 for k in stable_facts if k[0] == s) >= 5),
        "pooled": {"exact_match": round(float(np.mean([p["exact_match"] for p in mine])), 4),
                   "rouge_l_recall": round(float(np.mean([p["rouge_l_recall"] for p in mine])), 4),
                   "span": round(float(np.mean([p["span_containment"] for p in mine])), 4),
                   "degenerate": sum(1 for p in mine if p["degeneracy"])},
        "meta": {"model_revision": MODEL_REV, "decoders": {"primary": DEC_PRIMARY,
                                                           "secondary": DEC_SECONDARY},
                 "seed": SEED, "select_seed": a.seed,
                 "wall_s": round(time.time() - t0, 1),
                 "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)},
    }
    gate = ("PROCEED_AS_SPECIFIED" if lo >= 0.55 else
            "PROCEED_NARROWED_30_70" if lo >= 0.35 else "STOP_TIER_A_ONLY")
    summary["J1_gate"] = {"statistic": "stable distinct facts per target",
                          "value": round(rate, 4), "wilson95": [round(lo, 4), round(hi, 4)],
                          "decision": gate,
                          "rule": "lo>=0.55 proceed; 0.35<=lo<0.55 narrow; lo<0.35 stop"}
    os.makedirs(OUT, exist_ok=True)
    tag = f"_s{a.shard}" if a.n_shards > 1 else ""
    json.dump(summary, open(f"{OUT}/j1_summary{tag}.json", "w"), indent=2)
    json.dump(mine, open(f"{OUT}/j1_probes{tag}.json", "w"), indent=2, ensure_ascii=False)
    json.dump({"facts": [{"entity": k[0], "answer_norm": k[1],
                          "n_probes": len(v), "stable": k in stable_facts,
                          "probe_ids": [f"{p['kind']}::{p['idx']}" for p in v]}
                         for k, v in sorted(facts.items())]},
              open(f"{OUT}/j1_facts{tag}.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in
                      ("n_targets", "n_probes", "n_admissible_probes", "exclusions",
                       "n_distinct_facts", "n_stable_distinct_facts",
                       "stable_facts_per_target", "wilson95_stable_rate",
                       "decoder_only_stability", "targets_with_ge5_stable_facts",
                       "pooled", "J1_gate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
