"""eval_tofu.py — THE single TOFU evaluation path.

Used for all three things, so the pristine-base floor, the calibration scans and the
production gates are produced by identical code:

  * the pristine-base floor   (`--adapter none`)
  * calibration scans         (`--subset-per-split N`)
  * the qualification gates    (imported by `scripts/memorize_tofu.py`)

Scoring is `common_tofu.score_one`: DISTINCTIVE content recall as primary, with naive
content recall and ROUGE-L recall reported on every probe so the question-echo inflation
is visible. Degeneracy uses the frozen `output_failure` PLUS the character-level
repetition rule from `evaluate_partition.py` — the frozen rule tokenizes on whitespace and
needs >= 9 tokens, so a run-together degeneration is one token and escapes it entirely
(that hole scored an arm 1.000 IGNORANT in the matched-baseline batch).

Decoder: `common_tofu.TOFU_GEN_KW` (gen_length 96, steps 96, block_length 32,
temperature 0, low_confidence), seeded with the frozen `GEN_SEED` exactly as
`common_v2.gen_one` does, so generations are deterministic and reproducible.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common_v2 import (  # noqa: E402
    GEN_SEED, build_prompt, load_base, load_tokenizer, output_failure, sha256_file,
)
# the author's masked-diffusion sampler, taken from the FROZEN study module so this file
# uses the identical function object the synthetic path uses (common_big imports it from
# the pinned ML-GSAI/LLaDA@9182493 checkout). Read-only; nothing under spex/ is modified.
from common_big import native_generate  # noqa: E402
from common_tofu import (  # noqa: E402
    SCORING_DOC, STOPWORDS, TOFU_DECODER_DOC, TOFU_GEN_KW, aggregate, load_tofu,
    score_one, stopword_hash,
)
from evaluate_partition import char_repetition  # noqa: E402


@torch.no_grad()
def gen_one_tofu(model, tok, question, device="cuda:0"):
    """One native masked-diffusion generation under the TOFU decoder and the frozen seed."""
    torch.manual_seed(GEN_SEED)
    np.random.seed(GEN_SEED)
    enc = tok(build_prompt(tok, question), add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = native_generate(model, ids, attn, **TOFU_GEN_KW)
    g = out[0, ids.shape[1]:]
    return (tok.decode(g, skip_special_tokens=True),
            tok.decode(g, skip_special_tokens=False), g.tolist())


def degeneracy(gen):
    """Frozen label, widened by the character-level repetition rule."""
    lab = output_failure(gen)
    if lab is None:
        cr, ev = char_repetition(gen)
        if cr:
            return "repetitive_chars", ev
    return lab, None


def evaluate(model, tok, rows, device="cuda:0", log=print, every=50):
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        clean, raw, ids = gen_one_tofu(model, tok, r["question"], device)
        s = score_one(clean, r["question"], r["answer"])
        lab, ev = degeneracy(clean)
        out.append(dict(
            id=r["id"], split=r["split"], author=r["author"], question=r["question"],
            answer=r["answer"], gen=clean, gen_with_special=raw, gen_token_ids=ids,
            degeneracy=lab, degeneracy_evidence=ev, **s))
        if (i + 1) % every == 0:
            el = time.time() - t0
            log(f"[eval] {i+1}/{len(rows)} {el:.0f}s ({el/(i+1):.2f}s/gen)", flush=True)
    return out


def summarize(rows):
    def one(rs):
        if not rs:
            return {"n": 0}
        d = {
            "n": len(rs),
            "distinctive_content_recall": aggregate(rs, "distinctive_content_recall"),
            "naive_content_recall": aggregate(rs, "naive_content_recall"),
            "rouge_l_recall": aggregate(rs, "rouge_l_recall"),
            "n_items_with_empty_scorable_set": sum(1 for r in rs if r["scorable_is_empty"]),
            "mean_n_scorable": round(sum(r["n_scorable"] for r in rs) / len(rs), 2),
            "degeneracy": dict(collections.Counter(
                str(r["degeneracy"]) for r in rs if r["degeneracy"])),
            "n_degenerate": sum(1 for r in rs if r["degeneracy"]),
        }
        dm = d["distinctive_content_recall"]["mean"]
        nm = d["naive_content_recall"]["mean"]
        d["inflation_naive_minus_distinctive"] = (round(nm - dm, 4)
                                                  if (dm is not None and nm is not None) else None)
        return d
    S = {"scoring_doc": SCORING_DOC, "decoder_doc": TOFU_DECODER_DOC,
         "decoder": dict(TOFU_GEN_KW), "gen_seed": GEN_SEED,
         "stopwords_sha256": stopword_hash(), "n_stopwords": len(STOPWORDS),
         "forget": one([r for r in rows if r["split"] == "forget"]),
         "retain": one([r for r in rows if r["split"] == "retain"]),
         "overall": one(rows)}
    return S


def subset(rows, per_split):
    if not per_split:
        return rows
    out = []
    for sp in ("forget", "retain"):
        out += [r for r in rows if r["split"] == sp][:per_split]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="LoRA adapter dir, or 'none' for the pristine base")
    ap.add_argument("--subset-per-split", type=int, default=0)
    ap.add_argument("--label", default="eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = subset(load_tofu(), a.subset_per_split)
    tok = load_tokenizer()
    model = load_base("cuda:0", train=False)
    if a.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    per = evaluate(model, tok, rows, "cuda:0")
    R = {"label": a.label, "adapter": a.adapter, "n_probes": len(rows),
         "is_pristine_base": a.adapter == "none",
         "summary": summarize(per),
         "wall_s": round(time.time() - t0, 1),
         "s_per_generation": round((time.time() - t0) / max(1, len(rows)), 3),
         "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 3),
         "sources": {p: sha256_file(f"{HERE}/{p}") for p in
                     ("eval_tofu.py", "common_tofu.py", "common_v2.py")}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2, ensure_ascii=False)
    json.dump(per, open(a.out.replace(".json", "_probes.json"), "w"), indent=1,
              ensure_ascii=False)
    for sp in ("forget", "retain"):
        s = R["summary"][sp]
        print(f"[{a.label}] {sp:7s} n={s['n']} "
              f"DISTINCTIVE={s['distinctive_content_recall']['mean']} "
              f"naive={s['naive_content_recall']['mean']} "
              f"rougeL={s['rouge_l_recall']['mean']} "
              f"inflation=+{s['inflation_naive_minus_distinctive']} "
              f"degen={s['n_degenerate']}", flush=True)
    print(f"[done] {R['wall_s']}s ({R['s_per_generation']}s/gen) peak "
          f"{R['peak_gpu_gib']} GiB -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
