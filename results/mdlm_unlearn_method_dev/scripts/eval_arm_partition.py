"""
Evaluate a trained arm on the TOFU forget/retain probes and partition it against the
ignorance profile. The decisive measurement for any arm: where does it land, and how far is
it from a model that never saw the data?

Reference profiles already measured (`docs/FINDING7/FINDING8`):
  theta_0_tofu     DISCLOSE 0.667  DEGEN 0.000  ABSTAIN 0.000  REPLACE 0.125  IGNORANT 0.208
  ordering ORACLE  DISCLOSE 0.108  DEGEN 0.017  ABSTAIN 0.000  REPLACE 0.258  IGNORANT 0.617
  pristine base    DISCLOSE 0.030  DEGEN 0.000  ABSTAIN 0.113  REPLACE 0.470  IGNORANT 0.388

The oracle is the ceiling for any ordering-based arm; the pristine profile is the target.
TV distance to pristine is the profile-matching objective.
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
from common_big import build_prompt, load_base, load_tokenizer
from generate import generate as pinned_generate
from partition_tofu import partition_one

MASK_ID = 126336
DEV = "cuda:0"
STOP = set("""is are was were be been being of in on at to for with by from as and or that
this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should will
shall may might must a an the s t""".split())
REF = {
    "theta0_tofu": {"DISCLOSE": 0.667, "DEGENERATE": 0.000, "ABSTAIN": 0.000,
                    "REPLACE": 0.125, "IGNORANT": 0.208, "UNCLASSIFIED": 0.0},
    "ordering_oracle": {"DISCLOSE": 0.108, "DEGENERATE": 0.017, "ABSTAIN": 0.000,
                        "REPLACE": 0.258, "IGNORANT": 0.617, "UNCLASSIFIED": 0.0},
    "pristine_base": {"DISCLOSE": 0.030, "DEGENERATE": 0.000, "ABSTAIN": 0.113,
                      "REPLACE": 0.470, "IGNORANT": 0.388, "UNCLASSIFIED": 0.0},
}


def stems(s):
    ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]


def load_data(path):
    recs = json.loads(open(path).read())
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}
    out = []
    for r in recs:
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        out.append({**r, "salient": sal})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="arm adapter dir, or None for the backbone alone")
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--n-forget", type=int, default=120)
    ap.add_argument("--n-retain", type=int, default=80)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--data", default=f"{P}/data/tofu/tofu800.json",
                    help="dataset in TOFU format; default is the original path so prior "
                         "evaluations stay bit-reproducible")
    ap.add_argument("--label", required=True)
    ap.add_argument("--state-dict", default=None,
                    help="full 16-bit state dict (UNDO's save_16bit_model output). The student "
                         "already has memo+arm merged, was damaged and then distilled, so this "
                         "is the COMPLETE model: pass --memo none with it and apply no adapter.")
    a = ap.parse_args()
    if a.state_dict and (a.adapter or a.memo.lower() != "none"):
        ap.error("--state-dict is the complete model; use it with --memo none and no --adapter")

    tok = load_tokenizer()
    model = load_base(DEV)
    from peft import PeftModel
    if a.memo.lower() == "none":
        stack_memo = "pristine"        # RWKU: no memorization adapter
    else:
        model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
        stack_memo = os.path.basename(a.memo)
    stack = ["base", stack_memo]
    if a.state_dict:
        sd = torch.load(a.state_dict, map_location="cpu")
        pnames = {n for n, _ in model.named_parameters()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # DeepSpeed's save_16bit_model consolidates PARAMETERS only, so buffers (rotary
        # inv_freq and friends) are legitimately absent and keep their correct base values.
        # A missing PARAMETER is the dangerous case -- it would silently leave base weights in
        # place and report the base model as the arm -- so only that is fatal.
        missing_params = [n for n in missing if n in pnames]
        print(f"[state_dict] {a.state_dict}: loaded {len(sd)} tensors, "
              f"{len(missing)} missing ({len(missing_params)} of them parameters), "
              f"{len(unexpected)} unexpected", flush=True)
        if missing_params or unexpected:
            raise SystemExit(f"refusing to evaluate a partial load: "
                             f"missing_params={missing_params[:5]} unexpected={unexpected[:5]}")
        stack.append(f"{os.path.basename(a.state_dict)}(full state dict)")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        stack.append(f"{os.path.basename(os.path.dirname(a.adapter))}/"
                     f"{os.path.basename(a.adapter)}(merged)")
    model = model.to(DEV).eval()
    print(f"[stack] {' -> '.join(stack)}", flush=True)

    data = load_data(a.data)
    fo = [r for r in data if r["split"] == "forget" and len(r["salient"]) >= 3][:a.n_forget]
    rt = [r for r in data if r["split"] == "retain" and len(r["salient"]) >= 3][:a.n_retain]
    DEC = dict(steps=a.steps, gen_length=a.gen_length, block_length=a.gen_length,
               temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID)

    rows, t0 = [], time.time()
    for split, items in (("forget", fo), ("retain", rt)):
        for n, it in enumerate(items):
            torch.manual_seed(20261020); np.random.seed(20261020)
            e = tok(build_prompt(tok, it["question"]), add_special_tokens=False,
                    return_tensors="pt")
            ids = e["input_ids"].to(DEV)
            with torch.no_grad():
                out = pinned_generate(model, ids, e["attention_mask"].to(DEV), **DEC)
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            gt = set(stems(gen))
            asc = (sum(1 for s in it["salient"] if s in gt) / len(it["salient"])
                   if it["salient"] else None)
            rows.append(partition_one({"id": it["id"], "split": split,
                                       "author": it["author"], "gen": gen,
                                       "question": it["question"], "answer": it["answer"],
                                       "distinctive_content_recall": asc}))
            if (n + 1) % 40 == 0:
                print(f"  {split} {n+1}/{len(items)}  {(time.time()-t0)/len(rows):.2f}s/probe",
                      flush=True)

    def prof(split):
        g = [r for r in rows if r["split"] == split]
        c = collections.Counter(r["label"] for r in g)
        n = len(g) or 1
        return {k: round(c[k] / n, 4) for k in
                ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT", "UNCLASSIFIED")}

    pf, pr = prof("forget"), prof("retain")
    tv = lambda x, y: round(0.5 * sum(abs(x[k] - y.get(k, 0)) for k in x), 4)
    asc_f = [r["distinctive_content_recall"] for r in rows
             if r["split"] == "forget" and r["distinctive_content_recall"] is not None]
    asc_r = [r["distinctive_content_recall"] for r in rows
             if r["split"] == "retain" and r["distinctive_content_recall"] is not None]
    out = {"label": a.label, "stack": stack, "adapter": a.adapter, "decoder": DEC,
           "forget_profile": pf, "retain_profile": pr,
           "forget_ASCR_mean": round(float(np.mean(asc_f)), 4),
           "retain_ASCR_mean": round(float(np.mean(asc_r)), 4),
           "TV_forget_to": {k: tv(pf, v) for k, v in REF.items()},
           "reference_profiles": REF,
           "n_forget": len(asc_f), "n_retain": len(asc_r),
           "measured": {"wall_s": round(time.time() - t0, 1),
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/arm_{a.label}.json", "w"), indent=2)
    json.dump(rows, open(f"{P}/outputs/arm_{a.label}_rows.json", "w"), indent=2,
              ensure_ascii=False)
    print(f"\n[{a.label}] forget {pf}")
    print(f"[{a.label}] retain {pr}")
    print(f"[{a.label}] ASCR forget {out['forget_ASCR_mean']}  retain {out['retain_ASCR_mean']}")
    print(f"[{a.label}] TV to pristine {out['TV_forget_to']['pristine_base']}  "
          f"to oracle {out['TV_forget_to']['ordering_oracle']}  "
          f"to theta0 {out['TV_forget_to']['theta0_tofu']}")


if __name__ == "__main__":
    main()
