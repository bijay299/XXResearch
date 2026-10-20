"""COPY of /data/bijaypandey/llada_poc/scripts/evaluate_all.py for the SPEX baseline arms.
The ONLY edits vs the original: the `plan` list (now the 4 baseline arms x 2 checkpoints),
the output directory (OUT, under /data/bijaypandey/spex/outputs/baselines), and
`planned_generations` (len(plan) instead of the hardcoded 5). Scoring is untouched.

Stage 4 — evaluate every checkpoint on the 48 FIXED questions and preserve all outputs.

Checkpoints (all 5 published, not just the best):
  memorized_reference, sc_tf_step60, sc_tf_step120, fx_mix_step60, fx_mix_step120
48 questions = 24 original train questions + 24 paraphrases  =>  240 planned generations.

Reported per checkpoint:
  * forget-fact disclosure (lower = more forgetting)
  * retain-fact accuracy (higher = better retention)
  * the retained SUBSET SHARING forgotten answer words, separately from independent retain
  * empty / repetitive / refusal / malformed output counts
  * original-question and paraphrase results SEPARATELY
  * measured compute and memory

Automatic factual scoring (answer-presence, exact match) is kept strictly separate from
any qualitative judgment of coherence.
"""
import argparse
import json
import os
import sys
import time

import torch

W = "/data/bijaypandey/llada_poc"
SPEX = "/data/bijaypandey/spex"
OUT = f"{SPEX}/outputs/baselines"
sys.path.insert(0, f"{W}/scripts")
from common import (FACTS, SPEC, answer_present, exact_match, gen_one,  # noqa
                    load_base, load_tokenizer, output_failure)

DEV = "cuda:0"
SHARED = set(SPEC["shared_answer_design"]["shared_retain_fact_ids"])
INDEP = set(SPEC["shared_answer_design"]["independent_retain_fact_ids"])
FORGET_ANSWER_WORDS = sorted({w for f in FACTS if f["split"] == "forget"
                              for w in f["answer"].lower().split()})


def load_ckpt(name, memo_adapter, arm_adapter):
    """memorized_reference = base + memorization LoRA.
    Arm checkpoints = base + memorization LoRA MERGED, then the arm's unlearning LoRA."""
    from peft import PeftModel
    base = load_base(DEV, train=False)
    m = PeftModel.from_pretrained(base, memo_adapter)
    if name == "memorized_reference":
        return m.eval()
    merged = m.merge_and_unload()
    return PeftModel.from_pretrained(merged, arm_adapter).eval()


def evaluate(model, tok, tag):
    rows, t0 = [], time.time()
    for f in FACTS:
        for kind, q in (("original", f["train_question"]),
                        ("paraphrase", f["paraphrase_question"])):
            txt, raw_sp, ids_ = gen_one(model, tok, q, DEV)
            rows.append({
                "checkpoint": tag, "fact_id": f["fact_id"], "split": f["split"],
                "kind": kind, "entity": f["entity"], "relation": f["relation"],
                "question": q, "reference_answer": f["answer"], "aliases": f["aliases"],
                "shares_answer_across_splits": f["shares_answer_across_splits"],
                "generation": txt, "generation_with_special": raw_sp,
                "generation_token_ids": ids_,
                "answer_present": answer_present(txt, f["aliases"]),
                "exact_match": exact_match(txt, f["aliases"]),
                "output_failure": output_failure(txt),
                "mentions_any_forget_answer_word": any(
                    w in txt.lower().split() for w in FORGET_ANSWER_WORDS),
            })
    return rows, time.time() - t0


def summarize(rows, tag):
    def sub(pred):
        g = [r for r in rows if pred(r)]
        n = len(g) or 1
        return {"n": len(g),
                "answer_present_rate": round(sum(r["answer_present"] for r in g) / n, 4),
                "exact_match_rate": round(sum(r["exact_match"] for r in g) / n, 4)}

    fails = {k: sum(1 for r in rows if r["output_failure"] == k)
             for k in ["empty", "repetitive", "refusal", "malformed"]}
    s = {
        "checkpoint": tag, "n_generations": len(rows),
        "forget_disclosure": {
            "all": sub(lambda r: r["split"] == "forget"),
            "original": sub(lambda r: r["split"] == "forget" and r["kind"] == "original"),
            "paraphrase": sub(lambda r: r["split"] == "forget" and r["kind"] == "paraphrase"),
        },
        "retain_accuracy": {
            "all": sub(lambda r: r["split"] == "retain"),
            "original": sub(lambda r: r["split"] == "retain" and r["kind"] == "original"),
            "paraphrase": sub(lambda r: r["split"] == "retain" and r["kind"] == "paraphrase"),
            "shared_answer_subset": sub(lambda r: r["fact_id"] in SHARED),
            "independent_subset": sub(lambda r: r["fact_id"] in INDEP),
        },
        "output_failures": fails,
        "output_failure_rate": round(sum(fails.values()) / max(len(rows), 1), 4),
    }
    s["collateral_gap_shared_minus_independent"] = round(
        s["retain_accuracy"]["shared_answer_subset"]["answer_present_rate"]
        - s["retain_accuracy"]["independent_subset"]["answer_present_rate"], 4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo-adapter", default=f"{W}/adapters/memorized")
    ap.add_argument("--only", default="")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    # SPEX baselines: the memorized reference plus 4 published-baseline arms x
    # checkpoints [50, 150], frozen before any output was inspected.
    plan = [("memorized_reference", None)] + [
        (f"{arm}_step{st}", f"{SPEX}/adapters_baselines/{arm}/ckpt_{st}")
        for arm in ("npo", "mdu_t1", "emdu", "sc_tf") for st in (50, 150)]
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p[0] in keep]

    tok = load_tokenizer()
    all_rows, summaries, compute = [], [], {}
    for tag, ad in plan:
        if ad is not None and not os.path.isdir(ad):
            print(f"[skip] {tag}: adapter missing ({ad})")
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        t0 = time.time()
        model = load_ckpt(tag, args.memo_adapter, ad)
        rows, gen_s = evaluate(model, tok, tag)
        s = summarize(rows, tag)
        summaries.append(s)
        all_rows += rows
        compute[tag] = {"load_plus_eval_wall_s": round(time.time() - t0, 1),
                        "generation_wall_s": round(gen_s, 1),
                        "s_per_generation": round(gen_s / max(len(rows), 1), 3),
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        fd = s["forget_disclosure"]["all"]["answer_present_rate"]
        ra = s["retain_accuracy"]["all"]["answer_present_rate"]
        sh = s["retain_accuracy"]["shared_answer_subset"]["answer_present_rate"]
        ind = s["retain_accuracy"]["independent_subset"]["answer_present_rate"]
        print(f"[{tag:18s}] forget_disclosure={fd:.3f}  retain_acc={ra:.3f}  "
              f"(shared={sh:.3f} indep={ind:.3f})  failures={sum(s['output_failures'].values())}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    out = {"study": "llada_poc_v1", "n_checkpoints": len(summaries),
           "questions_per_checkpoint": 2 * len(FACTS),
           "planned_generations": len(plan) * 2 * len(FACTS),
           "actual_generations": len(all_rows),
           "forget_answer_words": FORGET_ANSWER_WORDS,
           "scoring_note": ("answer-presence and exact match are AUTOMATIC factual metrics; "
                            "coherence is judged qualitatively and reported separately"),
           "summaries": summaries, "compute": compute}
    json.dump(out, open(f"{OUT}/eval_summary{args.suffix}.json", "w"), indent=2)
    json.dump(all_rows, open(f"{OUT}/eval_generations{args.suffix}.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[written] {len(all_rows)} generations across {len(summaries)} checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
