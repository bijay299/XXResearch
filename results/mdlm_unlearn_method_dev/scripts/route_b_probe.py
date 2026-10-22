"""
TWO ROUTES — is there a second, untouched disclosure pathway, and can one anchor block both?

THE OBSERVATION, from our own data and missed at the time. Finding 13 measured the
PROMPT-MASKED anchor at 23,450 positions and found p_true = 0.651 at low mask rate. The prompt
is fully masked there, so that cannot be residual prompt knowledge. It is the model completing
the fact from the ANSWER's own partial content.

So disclosure in an MDLM has two routes:

    Route A   prompt -> fact                 the conditional knowledge
    Route B   partial answer -> rest         autoassociative self-completion

MDU (2605.18253), EMDU (2610.25447) and our AFC all use a PROMPT-masked anchor. They therefore
block Route A only -- and worse, the anchor itself self-completes, so the student is trained to
MATCH a target that exercises Route B. The objective preserves the pathway. Route B is exactly
what makes iterative decoding leaky: once a few fact tokens commit, the rest follows regardless
of what the prompt contributes.

THE CANDIDATE ANCHOR. Four constructions, of which only the fourth is new:

    natural        prompt visible,  answer as-is          the reference
    mdu            prompt MASKED,   answer as-is          MDU: blocks A, preserves B
    fam            prompt MASKED,   answer ALL masked     EMDU: blocks A and B, destroys form
    factmask       prompt MASKED,   answer FACT tokens masked, structural tokens visible   <- NEW

`factmask` should be fact-free (both routes blocked) AND well-formed (structural scaffold kept).
If so it resolves the anti-correlation that closed SSA, where no contiguous band of the noise
schedule had both properties.

WHY THIS IS A KILL TEST AND NOT A STUDY. No training. Measured per anchor, on the same states:

  p_true          anchor probability of the true token          -> leak, graded
  fact_argmax     argmax is an author-salient fact token        -> leak, hard
  punct / eos      argmax is punctuation / EOS                   -> formlessness
  entropy         nats                                           -> formlessness

PRE-REGISTERED KILL RULES, fixed before the numbers are read:
  K1 factmask p_true is not below mdu's by >= 0.15          -> it does not close Route B; DEAD.
  K2 factmask punct+eos exceeds fam's by more than 0.05      -> no better formed than FAM, so it
                                                                buys nothing over EMDU; DEAD.
  K3 factmask entropy exceeds 3.0 nats                       -> formless like the high-t anchor
                                                                (FINDING13 measured 4.41 there);
                                                                it will train a degenerate
                                                                student; DEAD.
  SURVIVES only if factmask is materially less leaky than mdu AND materially better formed than
  fam. Anything else and the line closes immediately rather than proceeding to training.

Reported binned by mask rate, because FINDING13 showed every anchor property in this family is
mask-rate dependent and an aggregate hides it.
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
from entity_mask_target import mask_whole_prompt, stems

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
VARIANTS = ("natural", "mdu", "fam", "factmask")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--data", default=f"{P}/data/tofu/tofu800.json")
    ap.add_argument("--n-facts", type=int, default=120)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--label", default="theta0")
    a = ap.parse_args()

    tok = load_tokenizer()
    recs = json.loads(open(a.data).read())
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}
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
        if len(sal) < 3:
            continue
        nat_p = tok(build_prompt(tok, r["question"]), add_special_tokens=False)["input_ids"]
        mk_p, _ = mask_whole_prompt(tok, r["question"])
        if len(mk_p) != len(nat_p):
            continue
        # the FACT token ids inside this answer: first token of each salient stem, both
        # space-prefixed and bare, exactly as every other arm in this project defines them
        # FULL token spans of each salient stem, not just first tokens. A first-token-only
        # set leaves the continuations of multi-token facts ("Hsiao Yun-Hwa") visible, so the
        # fact is only partially masked and completion is trivial -- which made the first run
        # of this probe undecisive rather than negative.
        ban = set()
        for s in sal:
            for v in (" " + s, s):
                ban.update(tok(v, add_special_tokens=False)["input_ids"])
        facts.append({**r, "salient": sal, "ban": ban,
                      "prompts": {"natural": nat_p, "masked": mk_p}})
    facts = facts[:a.n_facts]
    print(f"[data] {len(facts)} forget facts", flush=True)

    model = load_base(DEV)
    from peft import PeftModel
    if a.memo.lower() != "none":
        model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
    model = model.to(DEV).eval()

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
        mkp, _, _ = canvas(items, "masked")
        ar = torch.arange(nat.shape[1], device=DEV)[None]
        ansreg = ar >= ast[:, None]
        # per-row mask of FACT token positions inside the answer
        factpos = torch.zeros_like(nat, dtype=torch.bool)
        for b, it in enumerate(items):
            if it["ban"]:
                bt = torch.tensor(sorted(it["ban"]), device=DEV)
                factpos[b] = torch.isin(nat[b], bt) & ansreg[b]
        for d in range(a.draws):
            g = torch.Generator(device=DEV).manual_seed(8100 + d)
            noisy, _, _ = FX.random_tf_states_llada(nat, ast, eps=1e-3, generator=g)
            mp = (noisy == MASK_ID) & ansreg
            r_, c_ = mp.nonzero(as_tuple=True)
            if r_.numel() == 0:
                continue
            rate = (mp.sum(1).float() / a.gen_length).tolist()
            for v in VARIANTS:
                if v == "natural":
                    cv = noisy
                elif v == "mdu":
                    cv = torch.where(ansreg, noisy, mkp)
                elif v == "fam":
                    cv = torch.where(ansreg, torch.full_like(noisy, MASK_ID), mkp)
                else:   # factmask: prompt masked, FACT tokens in the answer masked too,
                        # structural answer tokens left visible
                    cv = torch.where(ansreg, noisy, mkp)
                    cv = torch.where(factpos, torch.full_like(cv, MASK_ID), cv)
                with torch.no_grad():
                    lp = torch.log_softmax(
                        model(cv, attention_mask=attn).logits[r_, c_].float(), -1)
                am_ = lp.argmax(-1)
                pt = lp.gather(1, nat[r_, c_].unsqueeze(1)).exp().squeeze(1)
                ent = -(lp.exp() * lp).sum(-1)
                for j in range(r_.numel()):
                    b = int(r_[j]); t = int(am_[j])
                    rows.append({"variant": v, "mask_rate": round(rate[b], 3),
                                 "p_true": float(pt[j]), "entropy": float(ent[j]),
                                 "fact": float(t in items[b]["ban"]),
                                 "punct": float(is_punct(t)), "eos": float(t == EOS_ID)})
        if (i // a.batch) % 20 == 0:
            print(f"  {i+len(items)}/{len(facts)} facts  rows={len(rows)}", flush=True)

    K = ("p_true", "fact", "punct", "eos", "entropy")
    agg = lambda rs: ({k: round(float(np.mean([r[k] for r in rs])), 5) for k in K}
                      | {"n": len(rs)}) if rs else {}
    out = {"study": "route_b_self_completion_and_fact_selective_anchor",
           "label": a.label, "memo": a.memo, "hparams": vars(a), "n_facts": len(facts),
           "routes": {"A": "prompt -> fact (what a prompt-masked anchor removes)",
                      "B": "partial answer -> rest (autoassociative self-completion)"},
           "overall": {v: agg([r for r in rows if r["variant"] == v]) for v in VARIANTS},
           "by_mask_rate": {}}
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        key = f"{lo:.2f}-{hi:.2f}"
        out["by_mask_rate"][key] = {
            v: agg([r for r in rows if r["variant"] == v and lo <= r["mask_rate"] < hi])
            for v in VARIANTS}
    o = out["overall"]
    kills = []
    if o["factmask"]["p_true"] > o["mdu"]["p_true"] - 0.15:
        kills.append("K1_does_not_close_route_B")
    if o["factmask"]["entropy"] >= o["fam"]["entropy"] - 1.0:
        kills.append("K2_no_better_formed_than_FAM_on_entropy")
    if o["factmask"]["entropy"] > 3.0:
        kills.append("K3_formless_will_train_a_degenerate_student")
    out["kills_fired"] = kills
    out["verdict"] = ("FACT_SELECTIVE_ANCHOR_SURVIVES" if not kills
                      else "DEAD: " + "; ".join(kills))
    out["route_B_evidence"] = {
        "mdu_anchor_p_true": o["mdu"]["p_true"],
        "mdu_anchor_fact_argmax": o["mdu"]["fact"],
        "meaning": ("the prompt is fully masked in the mdu anchor, so any residual p_true is "
                    "self-completion from the answer -- Route B. Finding 13 measured 0.651 at "
                    "low mask rate.")}
    out["measured"] = {"wall_s": round(time.time() - t0, 1), "n_rows": len(rows),
                       "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/routeb_{a.label}.json", "w"), indent=2)

    hdr = f"{'variant':10s} {'p_true':>8s} {'fact':>7s} {'punct':>7s} {'eos':>7s} {'entropy':>8s} {'n':>7s}"
    print("\n=== OVERALL ===\n" + hdr)
    for v in VARIANTS:
        d = o[v]
        print(f"{v:10s} {d['p_true']:8.4f} {d['fact']:7.4f} {d['punct']:7.4f} "
              f"{d['eos']:7.4f} {d['entropy']:8.4f} {d['n']:7d}")
    for key, blk in out["by_mask_rate"].items():
        print(f"\n=== mask rate {key} ===\n" + hdr)
        for v in VARIANTS:
            d = blk.get(v) or {}
            if d:
                print(f"{v:10s} {d['p_true']:8.4f} {d['fact']:7.4f} {d['punct']:7.4f} "
                      f"{d['eos']:7.4f} {d['entropy']:8.4f} {d['n']:7d}")
    print(f"\n=== VERDICT: {out['verdict']} ===")


if __name__ == "__main__":
    main()
