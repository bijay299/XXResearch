"""
What SHAPE is genuine ignorance? The diagnostic that decides whether every anchor-matching
and suppression objective is aimed at the wrong target.

THE OBSERVATION. Measured at the generation level on the TOFU forget split, every objective we
have run trades disclosure for degeneracy, and every one of them RAISES answer-position entropy
to do it:

    theta_0      entropy 1.10   DISCLOSE 0.792  DEGENERATE 0.008
    MDU          entropy 2.31   DISCLOSE 0.067  DEGENERATE 0.150   retain ASCR 0.091
    EMDU         entropy 6.72   DISCLOSE 0.000  DEGENERATE 1.000
    entity arms  entropy ~3.9   DISCLOSE 0.02-0.05  DEGENERATE 0.69-0.75

Maximum-entropy loss, anchor matching at tau<1, and uniform-prior targets are all explicitly
entropy-RAISING. They encode an assumption: that a model which never learned a fact is
UNCERTAIN about it.

THE PRISTINE PROFILE CONTRADICTS THAT ASSUMPTION. A model that never saw the data sits at
DISCLOSE 0.030, REPLACE 0.470, IGNORANT 0.388, ABSTAIN only 0.113. It does not hedge. It
commits to a specific wrong answer. Our own substitution study measured the same thing from the
other side: p(true | stranger) = 0.228 against a 0.02 gate -- "confident confabulation".

If that is right, then genuine ignorance is NOT high entropy, and every objective above is
moving AWAY from the distribution it claims to approximate while appearing to succeed on any
metric that only asks whether the true answer's probability fell.

WHAT THIS MEASURES. Answer-position distributions on the SAME forget facts, same noise draws,
same positions, for three references and any number of trained arms:

  base            saw NEITHER forget nor retain -- pristine
  retain_ref      saw retain but NOT forget     -- the true retrain oracle for this split
  theta0          saw both                      -- what unlearning starts from
  <arms>          trained checkpoints

Per model: entropy, p_true, top1_p, eos/punct argmax rate, and the rank of the true token.
Binned by mask rate, because the whole trade-off turned out to be mask-rate dependent
(FINDING13).

PRE-REGISTERED READING, fixed before the numbers are seen:

  * retain_ref entropy at forget positions is CLOSE TO theta_0's (within ~0.5 nats) and well
    below the trained arms' -> genuine ignorance is low-entropy confident confabulation, the
    entropy-raising family is aimed at the wrong target shape, and the method should instead
    move the ARGMAX to a specific plausible alternative while PRESERVING confidence.
  * retain_ref entropy is HIGH, comparable to the trained arms -> ignorance really is
    uncertainty, the entropy-raising family is correctly aimed, and the bimodality must be
    explained some other way. This hypothesis then dies.

No training. Forward passes only.
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
from entity_mask_target import stems

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=120)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--arms", default="",
                    help="comma list of label=adapterdir stacked on theta0_tofu")
    ap.add_argument("--label", default="shape")
    a = ap.parse_args()

    tok = load_tokenizer()
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}
    facts = []
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        if len(sal) < 3:
            continue
        facts.append({**r, "salient": sal})
    facts = facts[:a.n_facts]
    print(f"[data] {len(facts)} forget facts", flush=True)

    punct_cache = {}

    def is_punct(t):
        if t not in punct_cache:
            s = tok.decode([t])
            punct_cache[t] = bool(s) and all(c in string.punctuation + " " for c in s)
        return punct_cache[t]

    # model spec: (label, memo_adapter_or_None, stacked_arm_or_None)
    specs = [("base_pristine", None, None),
             ("retain_ref", f"{P}/adapters/retain_ref_tofu", None),
             ("theta0", f"{P}/adapters/theta0_tofu", None)]
    for item in [x for x in a.arms.split(",") if x.strip()]:
        lab, d = item.split("=", 1)
        specs.append((lab, f"{P}/adapters/theta0_tofu", d))

    def canvas(items):
        pid = [tok(build_prompt(tok, it["question"]),
                   add_special_tokens=False)["input_ids"] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    rows, t0 = [], time.time()
    for lab, memo, arm in specs:
        model = load_base(DEV)
        from peft import PeftModel
        if memo:
            model = PeftModel.from_pretrained(model, memo).merge_and_unload()
        if arm:
            model = PeftModel.from_pretrained(model, arm).merge_and_unload()
        model = model.to(DEV).eval()
        print(f"[model] {lab}  memo={os.path.basename(memo) if memo else 'None'}  "
              f"arm={arm or 'None'}", flush=True)
        n0 = len(rows)
        for i in range(0, len(facts), a.batch):
            items = facts[i:i + a.batch]
            if not items:
                continue
            full, attn, ast = canvas(items)
            for d in range(a.draws):
                g = torch.Generator(device=DEV).manual_seed(4100 + d)
                noisy, _, _ = FX.random_tf_states_llada(full, ast, eps=1e-3, generator=g)
                ar = torch.arange(noisy.shape[1], device=DEV)[None]
                mp = (noisy == MASK_ID) & (ar >= ast[:, None])
                r_, c_ = mp.nonzero(as_tuple=True)
                if r_.numel() == 0:
                    continue
                rate = (mp.sum(1).float() / a.gen_length).tolist()
                with torch.no_grad():
                    lp = torch.log_softmax(
                        model(noisy, attention_mask=attn).logits[r_, c_].float(), -1)
                am_ = lp.argmax(-1)
                true_tok = full[r_, c_]
                pt = lp.gather(1, true_tok.unsqueeze(1)).exp().squeeze(1)
                ent = -(lp.exp() * lp).sum(-1)
                top1 = lp.max(-1).values.exp()
                # rank of the true token: how many tokens beat it
                rank = (lp > lp.gather(1, true_tok.unsqueeze(1))).sum(-1)
                for j in range(r_.numel()):
                    b = int(r_[j])
                    t = int(am_[j])
                    rows.append({"model": lab, "mask_rate": round(rate[b], 3),
                                 "entropy": float(ent[j]), "p_true": float(pt[j]),
                                 "top1_p": float(top1[j]),
                                 "eos": float(t == EOS_ID), "punct": float(is_punct(t)),
                                 "true_rank": int(rank[j]),
                                 "argmax_is_true": float(t == int(true_tok[j]))})
        print(f"  {len(rows)-n0} positions  {time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    K = ("entropy", "p_true", "top1_p", "eos", "punct", "argmax_is_true")
    def agg(rs):
        if not rs:
            return {}
        d = {k: round(float(np.mean([r[k] for r in rs])), 5) for k in K}
        d["true_rank_median"] = int(np.median([r["true_rank"] for r in rs]))
        d["n"] = len(rs)
        return d

    labs = [s[0] for s in specs]
    out = {"study": "shape_of_genuine_ignorance_at_answer_positions",
           "label": a.label, "n_facts": len(facts), "hparams": vars(a),
           "references": {
               "base_pristine": "saw NEITHER forget nor retain",
               "retain_ref": "saw retain but NOT forget -- the true retrain oracle",
               "theta0": "saw both -- what unlearning starts from"},
           "pre_registered_reading": (
               "retain_ref entropy close to theta_0 and well below the trained arms -> genuine "
               "ignorance is low-entropy confident confabulation and the entropy-raising family "
               "is aimed at the wrong target shape; retain_ref entropy high and comparable to "
               "the trained arms -> ignorance is uncertainty and this hypothesis dies"),
           "overall": {l: agg([r for r in rows if r["model"] == l]) for l in labs},
           "by_mask_rate": {}}
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        key = f"{lo:.2f}-{hi:.2f}"
        out["by_mask_rate"][key] = {
            l: agg([r for r in rows if r["model"] == l and lo <= r["mask_rate"] < hi])
            for l in labs}
    e_ref = out["overall"]["retain_ref"]["entropy"]
    e_t0 = out["overall"]["theta0"]["entropy"]
    arm_ents = [out["overall"][l]["entropy"] for l in labs
                if l not in ("base_pristine", "retain_ref", "theta0")]
    if arm_ents:
        verdict = ("IGNORANCE_IS_LOW_ENTROPY_entropy_raising_family_mis_aimed"
                   if abs(e_ref - e_t0) < 0.5 and e_ref < min(arm_ents) - 0.5
                   else "IGNORANCE_IS_UNCERTAINTY_hypothesis_dies")
    else:
        verdict = "NO_ARMS_SUPPLIED_reference_only"
    out["verdict"] = verdict
    out["measured"] = {"wall_s": round(time.time() - t0, 1), "n_rows": len(rows)}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/ignorance_shape_{a.label}.json", "w"), indent=2)

    hdr = (f"{'model':18s} {'H':>7s} {'p_true':>8s} {'top1':>7s} {'eos':>7s} {'punct':>7s} "
           f"{'argmax=true':>11s} {'rank':>6s} {'n':>7s}")
    print("\n=== OVERALL (answer positions, forget facts) ===\n" + hdr)
    for l in labs:
        d = out["overall"][l]
        if not d:
            continue
        print(f"{l:18s} {d['entropy']:7.4f} {d['p_true']:8.4f} {d['top1_p']:7.4f} "
              f"{d['eos']:7.4f} {d['punct']:7.4f} {d['argmax_is_true']:11.4f} "
              f"{d['true_rank_median']:6d} {d['n']:7d}")
    for key, blk in out["by_mask_rate"].items():
        print(f"\n=== mask rate {key} ===\n" + hdr)
        for l in labs:
            d = blk.get(l) or {}
            if not d:
                continue
            print(f"{l:18s} {d['entropy']:7.4f} {d['p_true']:8.4f} {d['top1_p']:7.4f} "
                  f"{d['eos']:7.4f} {d['punct']:7.4f} {d['argmax_is_true']:11.4f} "
                  f"{d['true_rank_median']:6d} {d['n']:7d}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
