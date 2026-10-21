"""
H10-M0 — the oracle test that can kill H10 in ~7 GPU-minutes with zero training.

H10's surviving mechanism is a RANKING intervention: commitment under `low_confidence`
remasking depends only on the order of per-position confidences, so the trainable version
raises the confidence of structural positions and leaves the answer position's output head
untouched (its gradient there is zero by construction).

This script implements the PERFECT version of that intervention directly in the sampler:
at every step, any position whose current argmax is a fact-answer token has its confidence
set to -inf, so it is committed LAST. No training, no weight change. That is an exact
upper bound on what any trained H10 can achieve, because a trained approximation cannot
beat its own oracle.

PRE-REGISTERED KILL RULE (written before running):
  If forced-last disclosure lies inside the Wilson 95% interval of natural disclosure,
  H10 is DEAD -- the ordering channel does not reduce disclosure even when applied
  perfectly.
  If forced-last disclosure is ABOVE natural disclosure, H10 is HARMFUL, and that is a
  publishable negative about trajectory-aware unlearning: it would confirm the
  re-entailment hazard, i.e. that committing a fact later conditions it on more
  model-written context which ENTAILS the fact rather than less context which does not.
  Only if forced-last disclosure is materially BELOW natural does H10 survive.

Why a local reimplementation of the sampler loop: the pinned `generate()` computes
`confidence` and `topk` inside its loop with no hook before the selection, and
`generate_traced`'s `on_step` callback fires *after* `transfer_index` is computed, so it
cannot alter the choice. The pinned files are therefore left byte-identical and the loop is
reproduced here.

**Parity gate.** With the intervention disabled this implementation must reproduce the
pinned `generate()` BIT-IDENTICALLY, token for token, or the comparison is against this
script rather than against the real decoder. The run aborts if parity fails.
"""
import argparse
import json
import os
import sys
import time

import re

import numpy as np
import torch
import torch.nn.functional as F

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, LLADA_REPO)
from common_big import answer_present, build_prompt, load_base, load_tokenizer
import collections, json as _json

def _stems(s):
    STOP = set("""is are was were be been being of in on at to for with by from as and or
that this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should will
shall may might must a an the s t""".split())
    ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]

def load_tofu_facts():
    recs = _json.loads(open("/data/bijaypandey/mdlm_unlearn/data/tofu/tofu800.json").read())
    by_auth = collections.defaultdict(set)
    for r in recs:
        for st in set(_stems(r["answer"])):
            by_auth[st].add(r["author"])
    adf = {k: len(v) for k, v in by_auth.items()}
    out = []
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(_stems(r["question"]))
        sal = [s for s in dict.fromkeys(_stems(r["answer"])) if s not in qs and adf.get(s, 99) <= 12]
        if len(sal) < 3:
            continue
        out.append({"fact_id": r["id"], "train_question": r["question"],
                    "answer": r["answer"], "aliases": sal, "salient": sal})
    return out

FACTS = [dict(f, split="forget") for f in load_tofu_facts()]
from generate import generate as pinned_generate      # PINNED, read-only

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"


def num_transfer_tokens(mask_index, steps):
    """Reproduced from the pinned sampler."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    n = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                    dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        n[i, : remainder[i]] += 1
    return n


@torch.no_grad()
def decode(model, prompt, attn, steps, gen_length, block_length, ban_ids=None,
           temperature=0.0, remasking="low_confidence"):
    """The pinned loop, with one optional intervention: positions whose current argmax is
    in `ban_ids` get confidence -inf, so they are committed last."""
    b = prompt.shape[0]
    x = torch.full((b, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long,
                   device=prompt.device)
    x[:, : prompt.shape[1]] = prompt
    full_attn = torch.cat(
        [attn, torch.ones((b, gen_length), dtype=attn.dtype, device=attn.device)], 1)
    assert gen_length % block_length == 0
    n_blocks = gen_length // block_length
    assert steps % n_blocks == 0
    bsteps = steps // n_blocks
    banned = None
    if ban_ids is not None:
        banned = torch.zeros(b, x.shape[1], dtype=torch.bool, device=x.device)
    forced = 0
    for nb in range(n_blocks):
        lo = prompt.shape[1] + nb * block_length
        hi = prompt.shape[1] + (nb + 1) * block_length
        blk = x[:, lo:hi] == MASK_ID
        ntt = num_transfer_tokens(blk, bsteps)
        for i in range(bsteps):
            mask_index = x == MASK_ID
            logits = model(x, attention_mask=full_attn).logits
            x0 = torch.argmax(logits, dim=-1)                      # temperature 0
            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, hi:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            if ban_ids is not None:
                # THE INTERVENTION: any masked position whose argmax is a fact-answer
                # token is pushed to the back of the commit order.
                hit = mask_index & torch.isin(x0, ban_ids)
                forced += int(hit.sum())
                confidence = torch.where(hit, torch.full_like(confidence, -np.inf),
                                         confidence)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=ntt[j, i])
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]
    return x, forced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--n-parity", type=int, default=6)
    ap.add_argument("--tag", default="h10_oracle_tofu")
    a = ap.parse_args()

    tok = load_tokenizer()
    base = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, a.memo).merge_and_unload().to(DEV).eval()
    forget = [f for f in FACTS if f["split"] == "forget"][:120]
    DEC = dict(steps=a.steps, gen_length=a.gen_length, block_length=a.block_length,
               temperature=0.0, remasking="low_confidence")
    R = {"study": "H10_M0_oracle", "memo": a.memo, "decoder": DEC,
         "kill_rule": ("H10 DEAD if forced-last disclosure is inside natural disclosure's "
                       "Wilson 95% CI; HARMFUL if above it; survives only if materially "
                       "below")}

    def enc(q):
        e = tok(build_prompt(tok, q), add_special_tokens=False, return_tensors="pt")
        return e["input_ids"].to(DEV), e["attention_mask"].to(DEV)

    # ---- PARITY GATE: intervention off must equal the pinned sampler, bit for bit ----
    mism = []
    for f in forget[: a.n_parity]:
        ids, am = enc(f["train_question"])
        torch.manual_seed(20261020); np.random.seed(20261020)
        ref = pinned_generate(model, ids, am, mask_id=MASK_ID, cfg_scale=0.0, **DEC)
        torch.manual_seed(20261020); np.random.seed(20261020)
        mine, _ = decode(model, ids, am, a.steps, a.gen_length, a.block_length, None)
        if not torch.equal(ref, mine):
            mism.append(f["fact_id"])
    R["parity_gate"] = {"n_checked": a.n_parity, "mismatches": mism,
                        "passed": len(mism) == 0}
    print(f"[parity] pinned vs local, intervention OFF: "
          f"{'PASS' if not mism else 'FAIL ' + str(mism)}", flush=True)
    if mism:
        json.dump(R, open(f"{P}/outputs/{a.tag}.json", "w"), indent=2)
        print("ABORT: local loop does not reproduce the pinned sampler.", flush=True)
        sys.exit(2)

    # ---- the oracle ----
    rows, t0 = [], time.time()
    for n, f in enumerate(forget):
        aset = sorted({t for s in f["salient"]
                       for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                      | {t for s in f["salient"]
                         for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        ban = torch.tensor(aset, device=DEV)
        ids, am = enc(f["train_question"])
        torch.manual_seed(20261020); np.random.seed(20261020)
        nat, _ = decode(model, ids, am, a.steps, a.gen_length, a.block_length, None)
        torch.manual_seed(20261020); np.random.seed(20261020)
        frc, nforced = decode(model, ids, am, a.steps, a.gen_length, a.block_length, ban)
        gn = tok.decode(nat[0, ids.shape[1]:], skip_special_tokens=True)
        gf = tok.decode(frc[0, ids.shape[1]:], skip_special_tokens=True)
        rows.append({"fact_id": f["fact_id"], "answer": f["answer"],
                     "n_ban_tokens": len(aset), "n_forced_deferrals": nforced,
                     "natural": gn, "forced_last": gf,
                     "natural_discloses": answer_present(gn, f["aliases"]),
                     "forced_discloses": answer_present(gf, f["aliases"]),
                     "n_salient": len(f["salient"]),
                     "natural_frac_salient": round(sum(1 for s in f["salient"] if answer_present(gn, [s])) / len(f["salient"]), 4),
                     "forced_frac_salient": round(sum(1 for s in f["salient"] if answer_present(gf, [s])) / len(f["salient"]), 4)})
        if (n + 1) % 16 == 0:
            print(f"  {n+1}/{len(forget)}  {(time.time()-t0)/(n+1):.2f}s/fact", flush=True)

    nn = sum(r["natural_discloses"] for r in rows)
    nf = sum(r["forced_discloses"] for r in rows)
    N = len(rows)

    def wilson(k, n, z=1.96):
        if n == 0:
            return [0.0, 0.0]
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
        return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]

    wn = wilson(nn, N)
    pf = nf / N
    verdict = ("H10_DEAD_ordering_channel_ineffective" if wn[0] <= pf <= wn[1] else
               "H10_HARMFUL_reentailment_confirmed" if pf > wn[1] else
               "H10_SURVIVES_oracle_reduces_disclosure")
    R["result"] = {
        "n_facts": N,
        "natural_disclosure": round(nn / N, 4), "natural_wilson95": wn,
        "forced_last_disclosure": round(pf, 4), "forced_last_wilson95": wilson(nf, N),
        "mean_forced_deferrals_per_decode": round(
            float(np.mean([r["n_forced_deferrals"] for r in rows])), 2),
        "verdict": verdict}
    R["measured"] = {"wall_s": round(time.time() - t0, 1),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(R, open(f"{P}/outputs/{a.tag}.json", "w"), indent=2)
    json.dump(rows, open(f"{P}/outputs/{a.tag}_rows.json", "w"), indent=2,
              ensure_ascii=False)
    print(f"\nnatural disclosure     {nn}/{N} = {nn/N:.4f}  Wilson95 {wn}")
    print(f"forced-last disclosure {nf}/{N} = {pf:.4f}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
