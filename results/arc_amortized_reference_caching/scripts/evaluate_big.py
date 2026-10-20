"""Evaluate the 128-fact breaking-point arms, with PER-FACT pool-membership join.

The point: ARC trains only on pooled states. This evaluator reports forget disclosure
split by whether each fact was IN or ABSENT FROM the ARC pool, which is the falsifiable
per-fact prediction.
"""
import argparse, json, os, sys, time, torch
S = "/data/bijaypandey/spex"
sys.path.insert(0, f"{S}/scripts")
from common_big import FACTS, answer_present, exact_match, gen_one, load_base, load_tokenizer, output_failure
from peft import PeftModel
DEV = "cuda:0"
ap = argparse.ArgumentParser()
ap.add_argument("--adapter"); ap.add_argument("--tag", required=True)
ap.add_argument("--memo", default=f"{S}/adapters/memorized_big")
ap.add_argument("--pool-report", default=None, help="arm report.json, to join pool membership")
a = ap.parse_args()
inpool = None
if a.pool_report and os.path.exists(a.pool_report):
    pr = json.load(open(a.pool_report)).get("pool_membership")
    if pr: inpool = set(pr["facts_in_pool"])
tok = load_tokenizer()
base = load_base(DEV)
m = PeftModel.from_pretrained(base, a.memo)
if a.adapter:
    m = PeftModel.from_pretrained(m.merge_and_unload(), a.adapter)
m = m.eval()
rows, t0 = [], time.time()
for f in FACTS:
    for kind, q in (("original", f["train_question"]), ("paraphrase", f["paraphrase_question"])):
        txt, raw, ids = gen_one(m, tok, q, DEV)
        rows.append({"checkpoint": a.tag, "fact_id": f["fact_id"], "split": f["split"],
                     "kind": kind, "question": q, "reference_answer": f["answer"],
                     "aliases": f["aliases"], "generation": txt, "generation_token_ids": ids,
                     "answer_present": answer_present(txt, f["aliases"]),
                     "exact_match": exact_match(txt, f["aliases"]),
                     "output_failure": output_failure(txt),
                     "in_arc_pool": (None if inpool is None else f["fact_id"] in inpool)})
def sub(pred):
    g = [r for r in rows if pred(r)]
    n = len(g) or 1
    clean = [r for r in g if r["output_failure"] is None]
    return {"n": len(g),
            "disclosure": round(sum(r["answer_present"] for r in g) / n, 4),
            "clean_forgetting": round(sum(1 for r in clean if not r["answer_present"]) / n, 4),
            "degenerate": len(g) - len(clean)}
summ = {"checkpoint": a.tag, "n_generations": len(rows), "wall_s": round(time.time()-t0, 1),
        "forget_all": sub(lambda r: r["split"] == "forget"),
        "retain_all": sub(lambda r: r["split"] == "retain"),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
if inpool is not None:
    summ["forget_IN_pool"] = sub(lambda r: r["split"] == "forget" and r["in_arc_pool"])
    summ["forget_ABSENT_from_pool"] = sub(lambda r: r["split"] == "forget" and r["in_arc_pool"] is False)
json.dump(summ, open(f"{S}/outputs/big_eval_{a.tag}.json", "w"), indent=2)
json.dump(rows, open(f"{S}/outputs/big_gens_{a.tag}.json", "w"), indent=2, ensure_ascii=False)
print(f"[{a.tag}] forget={summ['forget_all']['disclosure']:.3f} "
      f"clean={summ['forget_all']['clean_forgetting']:.3f} "
      f"retain={summ['retain_all']['disclosure']:.3f} deg={summ['forget_all']['degenerate']}")
if inpool is not None:
    print(f"  IN pool    (n={summ['forget_IN_pool']['n']:3d}): disclosure={summ['forget_IN_pool']['disclosure']:.3f}")
    print(f"  ABSENT     (n={summ['forget_ABSENT_from_pool']['n']:3d}): disclosure={summ['forget_ABSENT_from_pool']['disclosure']:.3f}")
