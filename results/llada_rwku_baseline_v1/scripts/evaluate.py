"""
Evaluation harness: native LLaDA masked-diffusion generation + frozen scoring.

Uses the author's generate() from ML-GSAI/LLaDA @ 9182493 (imported unmodified).
NOT an autoregressive pipeline, and NO autoregressive shifted likelihood is used
anywhere — LLaDA is bidirectional.

Scoring is imported from the frozen freeze_eval module; nothing is redefined here.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

W = "/data/bijaypandey/llada_rwku"
MODEL_DIR = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
sys.path.insert(0, f"{W}/repos/llada_official")
sys.path.insert(0, f"{W}/scripts")

from generate import generate                      # author's implementation
from freeze_eval import (ANSWER_ONLY_INSTRUCTION, CLOZE_INSTRUCTION,  # frozen
                         MMLU_INSTRUCTION, exact_match, parse_mmlu_letter,
                         parse_short_answer, rouge_l_recall, span_recall)

MASK_ID = 126336
F = f"{W}/artifacts/frozen"

# ── Frozen decoders (fixed before any model output) ─────────────────────────
DECODERS = {
    "primary":   dict(gen_length=64, steps=64, block_length=32, temperature=0.0,
                      cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID),
    "secondary": dict(gen_length=64, steps=32, block_length=32, temperature=0.0,
                      cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID),
    # MMLU answers are a single letter; a short budget is frozen separately so the
    # utility panel does not cost 8x its needed compute. Same deterministic
    # settings (temperature 0, cfg 0, low-confidence remasking).
    "mmlu":      dict(gen_length=8, steps=8, block_length=8, temperature=0.0,
                      cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID),
}
SEED = 20261020           # frozen before execution


def load_model(adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    return tok, model


@torch.no_grad()
def gen_one(model, tok, prompt_text, dec):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True, tokenize=False)
    enc = tok(templated, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")
    out = generate(model, ids, attn, **dec)
    gen_ids = out[0, ids.shape[1]:]
    raw = tok.decode(gen_ids, skip_special_tokens=True)
    n_new = int((gen_ids != tok.pad_token_id).sum().item())
    truncated = n_new >= dec["gen_length"]
    return raw, gen_ids.tolist(), truncated


def build_prompt(item):
    q = item["query"]
    return (CLOZE_INSTRUCTION if item["prompt_template"] == "cloze"
            else ANSWER_ONLY_INSTRUCTION).format(q=q)


def score_item(item, raw, truncated):
    pred, fail = parse_short_answer(raw)
    if fail is None and truncated:
        fail = "truncation"
    em = exact_match(pred, item["aliases"]) if pred else False
    return {
        "pred_raw": raw, "pred_parsed": pred, "failure": fail,
        "exact_match": bool(em),
        "span_recall": round(span_recall(pred, item["answer"]), 4),
        "rouge_l_recall": round(rouge_l_recall(pred, item["answer"]), 4),
        "truncated": bool(truncated),
    }


def eval_panel(model, tok, items, dec, tag, idkey):
    res = []
    for it in items:
        raw, gids, trunc = gen_one(model, tok, build_prompt(it), dec)
        r = score_item(it, raw, trunc)
        r.update({idkey: it[idkey], "query": it["query"], "answer": it["answer"],
                  "aliases": it["aliases"], "gen_token_ids": gids, "panel": tag})
        res.append(r)
    n = len(res) or 1
    summ = {
        "n": len(res),
        "exact_match_acc": round(sum(r["exact_match"] for r in res) / n, 4),
        "mean_span_recall": round(sum(r["span_recall"] for r in res) / n, 4),
        "mean_rouge_l_recall": round(sum(r["rouge_l_recall"] for r in res) / n, 4),
        "failures": {k: sum(1 for r in res if r["failure"] == k)
                     for k in ["empty", "refusal", "repetition", "truncation", "parse_failure"]},
    }
    return res, summ


def eval_mmlu(model, tok, panel, dec):
    res = []
    for it in panel:
        p = MMLU_INSTRUCTION.format(q=it["question"], a=it["choices"][0],
                                    b=it["choices"][1], c=it["choices"][2],
                                    d=it["choices"][3])
        raw, gids, trunc = gen_one(model, tok, p, dec)
        letter, fail = parse_mmlu_letter(raw)
        res.append({"mmlu_id": it["mmlu_id"], "task": it["task"],
                    "gold": it["answer_letter"], "pred_raw": raw,
                    "pred_letter": letter, "failure": fail,
                    "correct": bool(letter == it["answer_letter"])})
    n = len(res) or 1
    return res, {"n": len(res),
                 "accuracy": round(sum(r["correct"] for r in res) / n, 4),
                 "failures": {k: sum(1 for r in res if r["failure"] == k)
                              for k in ["empty", "parse_failure"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["screen", "full"])
    ap.add_argument("--entities", default="")
    ap.add_argument("--decoders", default="primary")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default="forget,neighbor,mmlu")
    args = ap.parse_args()

    panels_cfg = json.load(open(f"{F}/frozen_panels.json"))
    mmlu_panel = json.load(open(f"{F}/frozen_mmlu_panel.json"))
    ents = [e for e in (args.entities.split(",") if args.entities else list(panels_cfg))
            if e in panels_cfg]
    decs = args.decoders.split(",")
    want = set(args.panels.split(","))

    tok, model = load_model(args.adapter)
    out = {"mode": args.mode, "adapter": args.adapter, "seed": SEED,
           "model_dir": MODEL_DIR,
           "model_revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
           "generate_impl": "ML-GSAI/LLaDA@9182493:generate.py:generate",
           "autoregressive": False, "decoders": {k: DECODERS[k] for k in decs},
           "entities": {}, "mmlu": {}}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for dn in decs:
        dec = DECODERS[dn]
        for e in ents:
            p = panels_cfg[e]
            out["entities"].setdefault(e, {"subject": p["subject"]})
            if "forget" in want:
                r, s = eval_panel(model, tok, p["qualification_panel"], dec,
                                  f"forget::{dn}", "fact_id")
                out["entities"][e][f"forget_{dn}"] = {"summary": s, "items": r}
                print(f"[{dn}] {e:34s} forget EM={s['exact_match_acc']:.2f} "
                      f"span={s['mean_span_recall']:.2f} rougeL={s['mean_rouge_l_recall']:.2f} "
                      f"fail={ {k:v for k,v in s['failures'].items() if v} }", flush=True)
            if "neighbor" in want:
                r, s = eval_panel(model, tok, p["neighbor_nonconflicting_panel"], dec,
                                  f"neighbor::{dn}", "nb_id")
                out["entities"][e][f"neighbor_{dn}"] = {"summary": s, "items": r}
                print(f"[{dn}] {e:34s} neighbor(nonconf,n={s['n']}) EM={s['exact_match_acc']:.2f}", flush=True)
        if "mmlu" in want:
            r, s = eval_mmlu(model, tok, mmlu_panel, DECODERS["mmlu"])
            out["mmlu"][dn] = {"summary": s, "items": r}
            print(f"[{dn}] MMLU acc={s['accuracy']:.3f} (n={s['n']}) fail={ {k:v for k,v in s['failures'].items() if v} }", flush=True)

    out["wall_s"] = round(time.time() - t0, 1)
    out["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"[written] {args.out}\n[sha256] {h}\n[wall] {out['wall_s']}s peak={out['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
