"""Standalone qualification of the already-trained 128-fact memorization adapter.

Training completed and the adapter was saved; only the qualification eval crashed
(missing `aliases` in the generated facts file, since patched). This re-runs the
qualification without retraining, using the same frozen decoder and scoring.
"""
import json, sys, time, torch
S = "/data/bijaypandey/spex"
sys.path.insert(0, f"{S}/scripts")
from common_big import FACTS, MODEL_DIR, answer_present, exact_match, gen_one, load_base, load_tokenizer, output_failure
from peft import PeftModel
DEV = "cuda:0"
tok = load_tokenizer()
m = PeftModel.from_pretrained(load_base(DEV), f"{S}/adapters/memorized_big").eval()
rows, t0 = [], time.time()
for f in FACTS:
    for kind, q in (("train", f["train_question"]), ("paraphrase", f["paraphrase_question"])):
        txt, raw, ids = gen_one(m, tok, q, DEV)
        rows.append({"fact_id": f["fact_id"], "split": f["split"], "kind": kind,
                     "question": q, "answer": f["answer"], "aliases": f["aliases"],
                     "gen": txt, "gen_token_ids": ids,
                     "answer_present": answer_present(txt, f["aliases"]),
                     "exact_match": exact_match(txt, f["aliases"]),
                     "failure": output_failure(txt)})
def rec(sp, kd):
    g = [r for r in rows if r["split"] == sp and r["kind"] == kd]
    return round(sum(r["answer_present"] for r in g) / len(g), 4), len(g)
ft, nf = rec("forget", "train"); rt, nr = rec("retain", "train")
fp, _ = rec("forget", "paraphrase"); rp, _ = rec("retain", "paraphrase")
out = {"n_generations": len(rows), "wall_s": round(time.time()-t0, 1),
       "forget_train": ft, "retain_train": rt,
       "forget_paraphrase_REPORTED_ONLY": fp, "retain_paraphrase_REPORTED_ONLY": rp,
       "threshold": 0.80, "n_forget": nf, "n_retain": nr,
       "qualified": bool(ft >= 0.80 and rt >= 0.80),
       "note": "training was completed before the crash; this re-runs only qualification"}
json.dump(out, open(f"{S}/outputs/big_qualification.json", "w"), indent=2)
json.dump(rows, open(f"{S}/outputs/big_memorization_generations.json", "w"), indent=2, ensure_ascii=False)
print(f"[recall] forget_train={ft:.3f} ({nf})  retain_train={rt:.3f} ({nr})  threshold=0.80")
print(f"[recall] paraphrase (separate): forget={fp:.3f} retain={rp:.3f}")
print(f"[QUALIFIED] {out['qualified']}  ({len(rows)} generations, {out['wall_s']}s)")
sys.exit(0 if out["qualified"] else 1)
