"""
Is there anything to forget? Per-subject recall of REAL pretrained knowledge on RWKU probes.

WHY. Every result in this project is on one dataset (TOFU-style) and one backbone with
SYNTHETIC memorization injected by SFT. Transfer to real pretrained knowledge is untested, and
that is the single largest gap in the evidence. RWKU probes cover 200 real entities (Stephen
King, Taylor Swift, Warren Buffett ...) at 20 probes each, locally available, no download.

THE OBSTACLE, from our own closed study. `research/llada-rwku-baseline-v1` @ aecefcd662c5
concluded `INSUFFICIENT_BASE_RECALL`: LLaDA-8B-Instruct scored 22/100 EM on RWKU probes, which
is why the project moved to synthetic memorization in the first place. An unlearning experiment
on knowledge the model does not have measures nothing.

But 22/100 is an AGGREGATE over entities. If recall is concentrated -- the model knowing a few
famous entities well and most not at all -- then a high-recall subset is a legitimate real-
knowledge forget target, and the aggregate was hiding it. That is exactly the aggregate-vs-
per-item error this project has made before (the ARC study's aggregate-as-per-fact mistake), so
it is worth measuring per subject rather than assumed either way.

This scores the PRISTINE base model (no memorization adapter) per subject and reports the
distribution, so a forget set can be chosen on evidence. Inference only, no training.

Scored two ways, because they disagree and the disagreement matters:
  EM-ish  : the answer string appears in the generation (answer_present, the pinned alias check)
  ASCR    : fraction of author-salient content stems recovered, the metric used everywhere else

A subject is a usable forget target only if the model actually answers its probes. The
selection rule is stated before the numbers are seen: take subjects with mean EM >= 0.50, and
report how many there are. If fewer than 10 subjects clear it, real-knowledge transfer is not
available on this backbone and that is recorded as a limitation rather than worked around.
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

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, f"{P}/scripts")
sys.path.insert(0, LLADA_REPO)
from common_big import answer_present, build_prompt, load_base, load_tokenizer
from generate import generate as pinned_generate
from entity_mask_target import stems

MASK_ID = 126336
DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=None,
                    help="optional adapter; default None = the PRISTINE base model")
    ap.add_argument("--probes", default=f"{P}/data/rwku_pooled/forget_level1.json")
    ap.add_argument("--per-subject", type=int, default=8, help="probes scored per subject")
    ap.add_argument("--max-subjects", type=int, default=200)
    ap.add_argument("--gen-length", type=int, default=48)
    ap.add_argument("--em-threshold", type=float, default=0.50)
    ap.add_argument("--label", default="base")
    a = ap.parse_args()

    tok = load_tokenizer()
    data = json.load(open(a.probes))
    by = collections.defaultdict(list)
    for r in data:
        by[r["subject"]].append(r)
    subjects = sorted(by)[:a.max_subjects]
    print(f"[data] {len(data)} probes over {len(by)} subjects; scoring "
          f"{len(subjects)} subjects x up to {a.per_subject}", flush=True)

    model = load_base(DEV)
    if a.memo:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
    model = model.to(DEV).eval()
    stack = "base" + (f" -> {os.path.basename(a.memo)}" if a.memo else " (PRISTINE)")
    print(f"[stack] {stack}", flush=True)
    DEC = dict(steps=a.gen_length, gen_length=a.gen_length, block_length=a.gen_length,
               temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID)

    rows, t0 = [], time.time()
    for n, subj in enumerate(subjects):
        for r in by[subj][:a.per_subject]:
            q = r["query"]
            torch.manual_seed(20261021); np.random.seed(20261021)
            e = tok(build_prompt(tok, q), add_special_tokens=False, return_tensors="pt")
            ids = e["input_ids"].to(DEV)
            with torch.no_grad():
                out = pinned_generate(model, ids, e["attention_mask"].to(DEV), **DEC)
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ans = r["answer"] if isinstance(r["answer"], str) else " ".join(r["answer"])
            asal = [s for s in dict.fromkeys(stems(ans)) if s not in set(stems(q))]
            gt = set(stems(gen))
            rows.append({
                "subject": subj, "type": r.get("type"), "query": q, "answer": ans,
                "gen": gen,
                "em": float(answer_present(gen, [ans])),
                "ascr": (sum(1 for s in asal if s in gt) / len(asal)) if asal else None})
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(subjects)} subjects  {len(rows)} probes  "
                  f"{(time.time()-t0)/max(len(rows),1):.2f}s/probe", flush=True)

    per = collections.defaultdict(list)
    for r in rows:
        per[r["subject"]].append(r)
    subj_stats = {}
    for s, rs in per.items():
        asc = [x["ascr"] for x in rs if x["ascr"] is not None]
        subj_stats[s] = {"n": len(rs),
                         "em": round(float(np.mean([x["em"] for x in rs])), 4),
                         "ascr": round(float(np.mean(asc)), 4) if asc else None}
    usable = sorted([s for s, v in subj_stats.items() if v["em"] >= a.em_threshold],
                    key=lambda s: -subj_stats[s]["em"])
    overall_em = round(float(np.mean([r["em"] for r in rows])), 4)
    verdict = ("REAL_KNOWLEDGE_TRANSFER_AVAILABLE" if len(usable) >= 10
               else "INSUFFICIENT_BASE_RECALL_transfer_not_available_on_this_backbone")

    out = {"study": "rwku_per_subject_recall_of_pretrained_knowledge",
           "label": a.label, "stack": stack, "hparams": vars(a),
           "n_probes": len(rows), "n_subjects": len(subj_stats),
           "overall_em": overall_em,
           "prior_aggregate_finding": ("our closed study reported INSUFFICIENT_BASE_RECALL at "
                                       "22/100 EM aggregate; this measures PER SUBJECT because "
                                       "an aggregate hides concentration"),
           "selection_rule": (f"a subject is usable if mean EM >= {a.em_threshold}; if fewer "
                              "than 10 subjects clear it, real-knowledge transfer is recorded "
                              "as unavailable rather than worked around"),
           "n_usable_subjects": len(usable), "usable_subjects": usable[:40],
           "per_subject": subj_stats, "verdict": verdict,
           "measured": {"wall_s": round(time.time() - t0, 1),
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/rwku_recall_{a.label}.json", "w"), indent=2)
    json.dump(rows[:1200], open(f"{P}/outputs/rwku_recall_{a.label}_rows.json", "w"),
              indent=2, ensure_ascii=False)

    ems = sorted((v["em"] for v in subj_stats.values()), reverse=True)
    print(f"\n[overall] EM {overall_em:.4f} over {len(rows)} probes, "
          f"{len(subj_stats)} subjects")
    print(f"[distribution] EM deciles: "
          + " ".join(f"{np.percentile(ems, p):.2f}" for p in (90, 70, 50, 30, 10)))
    print(f"[usable] {len(usable)} subjects with mean EM >= {a.em_threshold}")
    for s in usable[:15]:
        v = subj_stats[s]
        print(f"   {s:38s} EM {v['em']:.3f}  ASCR {v['ascr']}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
