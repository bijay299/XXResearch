"""
Collect the decoder's VISITED states once from the frozen theta_0, with their index sets.

Finding 10: structural confidence is 0.995 on teacher-forced states (headroom 0.005) but
0.752 on the states the sampler actually visits (headroom 0.248) -- 53x more room. COMMIT
failed because it optimised a regime the decoder never enters, not because the ordering
mechanism is infeasible.

This writes a frozen pool of visited states so the ordering hinge can be trained on them.
Collecting OFFLINE from a frozen model is what keeps the arm honest: every index set
(which masked positions are fact-argmax, which are structural) is computed once at collection
time and stored, so it is theta-INDEPENDENT during training and the loss stays an ordinary
fixed scalar objective. On-policy collection would make the index sets move with theta and
turn the arm into a sequence of surrogates, which the design explicitly forbade.

CARRIED FORWARD FROM OUR OWN CLOSED ARC STUDY: a pre-sampled state pool has a coverage
failure mode. ARC held at 24 facts and broke at 128 because the pool could not cover the
state space, and the per-fact prediction made about it was refuted. So per-fact coverage is
measured and written into the manifest here, never assumed, and the pool is sized so every
fact contributes states.
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
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
import llada_fxmix as FX                      # PINNED, read-only
from common_big import load_base, load_tokenizer

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
STOP = set("""is are was were be been being of in on at to for with by from as and or that
this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should will
shall may might must a an the s t""".split())


def stems(s):
    ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]


def load_forget(tok):
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}
    out = []
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        if len(sal) < 3:
            continue
        ban = sorted({t for s in sal
                      for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                     | {t for s in sal
                        for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        out.append({**r, "salient": sal, "ban_ids": ban})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--per-fact", type=int, default=6,
                    help="states kept per fact, sampled evenly over the trajectory")
    ap.add_argument("--m-struct", type=int, default=4)
    ap.add_argument("--out", default=f"{P}/data/visited_pool")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = load_tokenizer()
    model = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV).eval()
    facts = load_forget(tok)
    print(f"[data] {len(facts)} forget facts with >=3 salient stems", flush=True)

    canvases, metas = [], []
    t0 = time.time()
    for n, f in enumerate(facts):
        ban = torch.tensor(f["ban_ids"], device=DEV)
        pid = FX.chat_prompt_ids(tok, f["question"])
        Pm, Am = FX.left_pad([pid], EOS_ID)
        Pm, Am = Pm.to(DEV), Am.to(DEV)
        plen = Pm.shape[1]
        kept = []

        def on_step(nstep, x, logits, cand, transfer, x0):
            mrow = (x[0] == MASK_ID)
            ar = torch.arange(x.shape[1], device=DEV)
            mrow = mrow & (ar >= plen)
            mi = mrow.nonzero().flatten()
            if mi.numel() < a.m_struct + 1:
                return
            amax = logits[0].argmax(-1)
            is_fact = torch.isin(amax[mi], ban)
            if is_fact.sum() == 0 or (~is_fact).sum() == 0:
                return
            kept.append({"step": int(nstep),
                         "canvas": x[0].detach().cpu().tolist(),
                         "fact_pos": mi[is_fact].detach().cpu().tolist(),
                         "struct_pos": mi[~is_fact].detach().cpu().tolist()})

        with torch.no_grad():
            FX.generate_traced(model, Pm, Am, steps=a.steps, gen_length=a.gen_length,
                               block_length=a.gen_length, temperature=0.0,
                               remasking="low_confidence", mask_id=MASK_ID, on_step=on_step)

        if not kept:
            continue
        # keep per_fact states spread evenly over the trajectory, not clustered early
        idx = np.linspace(0, len(kept) - 1, min(a.per_fact, len(kept))).astype(int)
        for i in sorted(set(idx.tolist())):
            k = kept[i]
            canvases.append(k["canvas"])
            metas.append({"fact_id": f["id"], "author": f["author"], "step": k["step"],
                          "prompt_len": plen, "n_fact_pos": len(k["fact_pos"]),
                          "n_struct_pos": len(k["struct_pos"]),
                          "fact_pos": k["fact_pos"], "struct_pos": k["struct_pos"],
                          "ban_ids": f["ban_ids"], "n_trajectory_states": len(kept)})
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{len(facts)}  {(time.time()-t0)/(n+1):.2f}s/fact  "
                  f"pool={len(canvases)}", flush=True)

    L = max(len(c) for c in canvases)
    arr = np.full((len(canvases), L), EOS_ID, dtype=np.int32)
    for i, c in enumerate(canvases):
        arr[i, :len(c)] = c
    np.save(f"{a.out}/canvases.npy", arr)

    # ---- coverage, measured per fact and reported, never assumed ----
    per_fact = collections.Counter(m["fact_id"] for m in metas)
    covered = len(per_fact)
    manifest = {
        "study": "visited_state_pool",
        "memo": a.memo, "hparams": vars(a),
        "n_states": len(metas), "canvas_len": int(L),
        "n_facts_total": len(facts), "n_facts_covered": covered,
        "coverage_fraction": round(covered / len(facts), 4),
        "facts_absent_from_pool": [f["id"] for f in facts if f["id"] not in per_fact],
        "states_per_fact": {"min": int(min(per_fact.values())),
                            "mean": round(float(np.mean(list(per_fact.values()))), 2),
                            "max": int(max(per_fact.values()))},
        "mean_fact_pos_per_state": round(float(np.mean([m["n_fact_pos"] for m in metas])), 2),
        "mean_struct_pos_per_state": round(float(np.mean([m["n_struct_pos"] for m in metas])), 2),
        "mean_trajectory_states_per_fact": round(
            float(np.mean([m["n_trajectory_states"] for m in metas])), 2),
        "coverage_note": ("ARC held at 24 facts and broke at 128 on exactly this failure "
                          "mode; coverage is measured here and the arm is to be run at more "
                          "than one pool size"),
        "measured": {"wall_s": round(time.time() - t0, 1),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)},
    }
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)
    json.dump(metas, open(f"{a.out}/states.json", "w"))
    print(f"\n[pool] {len(metas)} states, canvas {L} tokens")
    print(f"[coverage] {covered}/{len(facts)} facts = {manifest['coverage_fraction']:.4f}  "
          f"states/fact min {manifest['states_per_fact']['min']} "
          f"mean {manifest['states_per_fact']['mean']} "
          f"max {manifest['states_per_fact']['max']}")
    print(f"[positions] fact {manifest['mean_fact_pos_per_state']}  "
          f"struct {manifest['mean_struct_pos_per_state']} per state")
    print(f"[done] {manifest['measured']['wall_s']}s "
          f"peak={manifest['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
