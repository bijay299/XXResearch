"""
What are these anchors actually made of? All four prompt variants, one model load, no training.

WHY THIS EXISTS. I hypothesised that MDU's collapse to DEGENERATE 1.000 (empty strings on all
64 forget facts, measured in our own matched baseline) was caused by its anchor being
EOS-dominated: the all-masked prompt gives the unconditional prior, which on a masked answer
canvas should concentrate on EOS. The first measurement refuted that at theta_0:

    MDU all-masked anchor   eos_argmax 0.0927   entropy 2.5087   p_true 0.3909

Not EOS-dominated, and it still assigns 0.39 to the TRUE token with the entire prompt masked --
because the teacher-forced canvas reveals part of the answer, and the model reconstructs the
rest from it. That is the canvas-leak failure mode (F4 in `docs/H12_ESD_DESIGN.md`), and it
bites MDU's own anchor, not only the proposed one.

So the anchor-degeneracy story is wrong as stated and is not claimed. This script measures what
is actually true of each anchor before any replacement mechanism is proposed, because the next
hypothesis has to be built on the measurement rather than on the one I had already written down.

MEASURED, per prompt variant, on teacher-forced answer states and BINNED BY MASK RATE -- the
binning matters because the leak is expected to depend on how much of the answer is revealed,
and an aggregate would hide exactly that:

  eos_argmax      fraction of answer positions whose anchor argmax is EOS
  punct_argmax    ... whose argmax is pure punctuation (the other degeneracy channel: our
                  baselines emit '........' as well as '')
  fact_argmax     ... whose argmax is an author-salient fact token  (the leak)
  p_true          anchor probability of the actual answer token     (the leak, graded)
  entropy         anchor entropy in nats
  top1_p          anchor's max probability (how peaked it is -- relevant because MDU minimises
                  KL(p_student || p_anchor), a MODE-SEEKING direction: the student collapses
                  onto a mode of the anchor, so a peaked anchor and a broad one produce very
                  different students even at equal entropy)

VARIANTS
  natural   the ordinary prompt                              -- the pre-intervention reference
  mdu       every prompt token -> MASK_ID                    -- MDU's published anchor
  entity    only the entity-mention tokens -> MASK_ID
  control   an equal NUMBER of non-entity content tokens -> MASK_ID

All four are token-count aligned, so every variant is read at the SAME answer positions under
the SAME noise draw. No pre-registered verdict here: this is a measurement, not a test, and
labelling it with a verdict would invite reading a mechanism into whichever column looks good.
"""
import argparse
import collections
import json
import os
import string
import sys
import time

import numpy as np
import torch

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, f"{P}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
import llada_fxmix as FX                      # PINNED, read-only
from common_big import build_prompt, load_base, load_tokenizer
from entity_mask_target import (author_names, mask_n_content, mask_span,
                                mask_whole_prompt, stems)

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
VARIANTS = ("natural", "mdu", "entity", "control")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--n-facts", type=int, default=160)
    ap.add_argument("--draws", type=int, default=4, help="noise draws per fact")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--label", default="theta0")
    a = ap.parse_args()

    tok = load_tokenizer()
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
    NAME, _ = author_names(recs)
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}

    # pure-punctuation token ids, resolved lazily: the second degeneracy channel is the
    # '........' output our baselines produce alongside the empty string.
    punct_cache = {}

    def is_punct(t):
        if t not in punct_cache:
            s = tok.decode([t])
            punct_cache[t] = bool(s) and all(c in string.punctuation + " " for c in s)
        return punct_cache[t]

    facts = []
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        nm = NAME.get(r["author"])
        if len(sal) < 3 or not nm or nm.lower() not in r["question"].lower():
            continue
        nat = tok(build_prompt(tok, r["question"]), add_special_tokens=False)["input_ids"]
        em, k = mask_span(tok, r["question"], nm)
        if em is None:
            continue
        cm, _ = mask_n_content(tok, r["question"], k, nm)
        am, _ = mask_whole_prompt(tok, r["question"])
        if not (len(nat) == len(em) == len(cm) == len(am)):
            continue
        ban = sorted({t for s in sal
                      for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                     | {t for s in sal
                        for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        facts.append({**r, "name": nm, "salient": sal, "ban_ids": ban,
                      "prompts": {"natural": nat, "mdu": am, "entity": em, "control": cm}})
    facts = facts[:a.n_facts]
    print(f"[data] {len(facts)} aligned forget facts", flush=True)

    model = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV).eval()

    def canvas(items, which):
        pid = [it["prompts"][which] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    rows, t0 = [], time.time()
    for i in range(0, len(facts), a.batch):
        items = facts[i:i + a.batch]
        if not items:
            continue
        nat, attn, ast = canvas(items, "natural")
        for d in range(a.draws):
            g = torch.Generator(device=DEV).manual_seed(7000 + d)
            noisy, _, _ = FX.random_tf_states_llada(nat, ast, eps=1e-3, generator=g)
            ar = torch.arange(noisy.shape[1], device=DEV)[None]
            mp = (noisy == MASK_ID) & (ar >= ast[:, None])
            r_, c_ = mp.nonzero(as_tuple=True)
            if r_.numel() == 0:
                continue
            # mask rate per row, so readings can be binned by how much answer is revealed
            rate = (mp.sum(1).float() / a.gen_length).tolist()
            for v in VARIANTS:
                vf, _, ast2 = canvas(items, v)
                cv = torch.where(ar >= ast[:, None], noisy, vf)
                with torch.no_grad():
                    lg = model(cv, attention_mask=attn).logits[r_, c_].float()
                lp = torch.log_softmax(lg, -1)
                am_ = lp.argmax(-1)
                pt = lp.gather(1, nat[r_, c_].unsqueeze(1)).exp().squeeze(1)
                ent = -(lp.exp() * lp).sum(-1)
                top1 = lp.max(-1).values.exp()
                for j in range(r_.numel()):
                    b = int(r_[j])
                    t = int(am_[j])
                    rows.append({
                        "variant": v, "fact_id": items[b]["id"], "draw": d,
                        "mask_rate": round(rate[b], 3),
                        "eos": float(t == EOS_ID), "punct": float(is_punct(t)),
                        "fact": float(t in set(items[b]["ban_ids"])),
                        "p_true": float(pt[j]), "entropy": float(ent[j]),
                        "top1_p": float(top1[j])})
        if (i // a.batch) % 20 == 0:
            print(f"  {i+len(items)}/{len(facts)} facts  rows={len(rows)}  "
                  f"{(time.time()-t0)/max(i+len(items),1):.2f}s/fact", flush=True)

    K = ("eos", "punct", "fact", "p_true", "entropy", "top1_p")
    def agg(rs):
        return {k: round(float(np.mean([r[k] for r in rs])), 5) for k in K} | {"n": len(rs)}

    BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    out = {"study": "anchor_composition_by_prompt_variant_and_mask_rate",
           "label": a.label, "memo": a.memo, "hparams": vars(a), "n_facts": len(facts),
           "note": ("measurement, not a test; no pre-registered verdict, because labelling a "
                    "measurement with a verdict invites reading a mechanism into whichever "
                    "column looks favourable"),
           "refuted_hypothesis": ("that MDU's all-masked anchor is EOS-dominated and that this "
                                  "explains its DEGENERATE 1.000; the first reading gave "
                                  "eos_argmax 0.0927, entropy 2.5087, p_true 0.3909"),
           "overall": {v: agg([r for r in rows if r["variant"] == v]) for v in VARIANTS},
           "by_mask_rate": {}}
    for lo, hi in BINS:
        key = f"{lo:.2f}-{hi:.2f}"
        out["by_mask_rate"][key] = {
            v: agg([r for r in rows if r["variant"] == v and lo <= r["mask_rate"] < hi])
            for v in VARIANTS
            if any(r["variant"] == v and lo <= r["mask_rate"] < hi for r in rows)}
    out["measured"] = {"wall_s": round(time.time() - t0, 1), "n_rows": len(rows),
                       "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/anchor_probe_{a.label}.json", "w"), indent=2)

    hdr = f"{'variant':9s} {'eos':>7s} {'punct':>7s} {'fact':>7s} {'p_true':>7s} {'H':>7s} {'top1':>7s} {'n':>7s}"
    print("\n=== OVERALL ===\n" + hdr)
    for v in VARIANTS:
        d = out["overall"][v]
        print(f"{v:9s} {d['eos']:7.4f} {d['punct']:7.4f} {d['fact']:7.4f} "
              f"{d['p_true']:7.4f} {d['entropy']:7.4f} {d['top1_p']:7.4f} {d['n']:7d}")
    for key, blk in out["by_mask_rate"].items():
        print(f"\n=== mask rate {key} (fraction of answer still masked) ===\n" + hdr)
        for v in VARIANTS:
            if v not in blk:
                continue
            d = blk[v]
            print(f"{v:9s} {d['eos']:7.4f} {d['punct']:7.4f} {d['fact']:7.4f} "
                  f"{d['p_true']:7.4f} {d['entropy']:7.4f} {d['top1_p']:7.4f} {d['n']:7d}")


if __name__ == "__main__":
    main()
