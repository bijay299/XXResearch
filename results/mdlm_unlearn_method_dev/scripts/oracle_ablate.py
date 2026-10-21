"""
DECOMPOSE the ordering oracle: how much of its gain can a trained arm possibly reach?

The oracle (`h10_oracle_tofu.py`, published) sets the confidence of every fact-argmax masked
position to -inf at every step, and reaches DISCLOSE 0.108 from theta_0's 0.667. That was read
as "deferral works". But the intervention fires in two distinct regimes:

  NON-TERMINAL  at least one masked position in the block is NOT a fact-argmax, so a
                structural rival exists. Here -inf is equivalent to the structural rival
                simply having higher confidence -- which is exactly what a trained arm can
                produce by raising structural confidence. REACHABLE.

  TERMINAL      every remaining masked position is a fact-argmax. There is no structural
                rival. Setting them all to -inf does not defer anything; `topk` over a
                constant -inf vector falls back to INDEX ORDER, so the oracle fills the
                remaining fact slots left-to-right instead of highest-confidence-first.
                A trained arm cannot reproduce this: there is nothing to raise. NOT REACHABLE.

If the oracle's gain comes mostly from the terminal regime then the ordering line's headline
number is produced by a mechanism no training can access, and the honest move is to close the
line rather than spend GPU hours chasing a ceiling that is not a ceiling. If the gain survives
with the terminal regime switched off, the trainable target is sound.

Three arms in ONE run on the SAME facts and decoder, which is what this project's standing
rule requires: the property at theta_0, and at one intervention that does not share the
mechanism, in the same table, before any claim:

  natural              no intervention -- the pre-intervention reference
  oracle_full          the published intervention, both regimes
  oracle_nonterminal   the intervention ONLY where a structural rival exists

PARITY GATE: with the intervention off this loop must reproduce the pinned `generate()`
bit-identically, or the comparison is against this script rather than the real decoder.

PRE-REGISTERED READING, fixed before the numbers are seen:
  * oracle_nonterminal disclosure within ~0.05 of oracle_full  -> the reachable regime carries
    the gain; train the per-step hinge.
  * oracle_nonterminal disclosure reverting more than halfway from oracle_full back to
    natural -> the gain is mostly terminal index-order filling, the ordering line is closed
    for a reason no training can fix, and this is recorded as a negative result.
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
import torch.nn.functional as F

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, f"{P}/scripts")
sys.path.insert(0, LLADA_REPO)
from common_big import answer_present, build_prompt, load_base, load_tokenizer
from generate import generate as pinned_generate      # PINNED, read-only
from partition_tofu import partition_one

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
STOP = set("""is are was were be been being of in on at to for with by from as and or that
this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should will
shall may might must a an the s t""".split())


def stems(s):
    ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]


def load_facts():
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
                      for t in tok_g(" " + s)[:1]} | {t for s in sal for t in tok_g(s)[:1]})
        out.append({**r, "fact_id": r["id"], "salient": sal, "aliases": sal, "ban_ids": ban})
    return out


TOK = None


def tok_g(s):
    return TOK(s, add_special_tokens=False)["input_ids"]


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
           mode="full", remasking="low_confidence"):
    """The pinned loop with one optional intervention.

    mode='full'        -inf whenever the argmax is a fact token   (the published oracle)
    mode='nonterminal' -inf only when a structural rival exists in the block, i.e. only in
                       the regime a trained arm could reproduce
    """
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
    stat = {"forced": 0, "steps_intervened": 0, "steps_terminal": 0,
            "steps_terminal_skipped": 0, "steps_with_fact_argmax": 0}
    for nb in range(n_blocks):
        lo = prompt.shape[1] + nb * block_length
        hi = prompt.shape[1] + (nb + 1) * block_length
        blk = x[:, lo:hi] == MASK_ID
        ntt = num_transfer_tokens(blk, bsteps)
        for i in range(bsteps):
            mask_index = x == MASK_ID
            logits = model(x, attention_mask=full_attn).logits
            x0 = torch.argmax(logits, dim=-1)
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
                hit = mask_index & torch.isin(x0, ban_ids)
                # restrict attention to the active block, where commitment happens
                inblk = torch.zeros_like(hit)
                inblk[:, lo:hi] = True
                hit_b = hit & inblk
                cand_b = mask_index & inblk
                # TERMINAL: every masked position in the block is a fact-argmax
                terminal = (hit_b.sum(1) == cand_b.sum(1)) & (cand_b.sum(1) > 0)
                if hit_b.any():
                    stat["steps_with_fact_argmax"] += 1
                if terminal.any():
                    stat["steps_terminal"] += 1
                if mode == "nonterminal":
                    # withhold the intervention from rows in the terminal regime
                    hit = hit & (~terminal).unsqueeze(1)
                    if terminal.any():
                        stat["steps_terminal_skipped"] += 1
                if hit.any():
                    stat["steps_intervened"] += 1
                    stat["forced"] += int(hit.sum())
                    confidence = torch.where(hit, torch.full_like(confidence, -np.inf),
                                             confidence)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=ntt[j, i])
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]
    return x, stat


def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--n-facts", type=int, default=120)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=64)
    ap.add_argument("--label", default="ablate")
    a = ap.parse_args()

    TOK = load_tokenizer()
    tok = TOK
    model = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV).eval()
    facts = load_facts()[:a.n_facts]
    print(f"[data] {len(facts)} forget facts", flush=True)
    DEC = dict(steps=a.steps, gen_length=a.gen_length, block_length=a.block_length,
               temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID)

    # ---- PARITY GATE: intervention off must equal the pinned generate(), bit for bit ----
    f0 = facts[0]
    e = tok(build_prompt(tok, f0["question"]), add_special_tokens=False, return_tensors="pt")
    ids, am = e["input_ids"].to(DEV), e["attention_mask"].to(DEV)
    torch.manual_seed(20261021); np.random.seed(20261021)
    ref = pinned_generate(model, ids, am, **DEC)
    torch.manual_seed(20261021); np.random.seed(20261021)
    mine, _ = decode(model, ids, am, a.steps, a.gen_length, a.block_length, ban_ids=None)
    parity = bool(torch.equal(ref, mine))
    print(f"[parity gate] local loop == pinned generate(): {parity}", flush=True)
    assert parity, "PARITY FAILED -- aborting; the comparison would be against this script"

    rows, t0 = [], time.time()
    for n, f in enumerate(facts):
        ban = torch.tensor(f["ban_ids"], device=DEV)
        e = tok(build_prompt(tok, f["question"]), add_special_tokens=False,
                return_tensors="pt")
        ids, am = e["input_ids"].to(DEV), e["attention_mask"].to(DEV)
        g = {}
        for arm, kw in (("natural", dict(ban_ids=None, mode="full")),
                        ("oracle_full", dict(ban_ids=ban, mode="full")),
                        ("oracle_nonterminal", dict(ban_ids=ban, mode="nonterminal"))):
            torch.manual_seed(20261021); np.random.seed(20261021)
            out, st = decode(model, ids, am, a.steps, a.gen_length, a.block_length, **kw)
            g[arm] = {"text": tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True),
                      "stat": st}
        row = {"fact_id": f["fact_id"], "author": f["author"], "question": f["question"],
               "answer": f["answer"], "salient": f["salient"]}
        for arm in g:
            txt = g[arm]["text"]
            gt = set(stems(txt))
            asc = (sum(1 for s in f["salient"] if s in gt) / len(f["salient"])
                   if f["salient"] else None)
            row[arm] = {
                "gen": txt, "discloses": bool(answer_present(txt, f["aliases"])),
                "ascr": asc, "stat": g[arm]["stat"],
                "cell": partition_one({"id": f["fact_id"], "split": "forget",
                                       "author": f["author"], "gen": txt,
                                       "question": f["question"], "answer": f["answer"],
                                       "distinctive_content_recall": asc})["label"]}
        rows.append(row)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(facts)}  {(time.time()-t0)/(n+1):.2f}s/fact", flush=True)

    CELLS = ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT", "UNCLASSIFIED")
    PRIS = {"DISCLOSE": 0.030, "DEGENERATE": 0.000, "ABSTAIN": 0.113,
            "REPLACE": 0.470, "IGNORANT": 0.388, "UNCLASSIFIED": 0.0}
    summ = {}
    for arm in ("natural", "oracle_full", "oracle_nonterminal"):
        c = collections.Counter(r[arm]["cell"] for r in rows)
        prof = {k: round(c[k] / len(rows), 4) for k in CELLS}
        summ[arm] = {
            "profile": prof,
            "disclose_rate": round(float(np.mean([r[arm]["discloses"] for r in rows])), 4),
            "ascr_mean": round(float(np.mean([r[arm]["ascr"] for r in rows])), 4),
            "TV_to_pristine": round(0.5 * sum(abs(prof[k] - PRIS[k]) for k in CELLS), 4),
            "forced_deferrals_total": int(sum(r[arm]["stat"]["forced"] for r in rows)),
            "steps_terminal_total": int(sum(r[arm]["stat"]["steps_terminal"] for r in rows)),
            "steps_terminal_skipped_total":
                int(sum(r[arm]["stat"]["steps_terminal_skipped"] for r in rows)),
            "steps_intervened_total":
                int(sum(r[arm]["stat"]["steps_intervened"] for r in rows))}

    nat, orc, non = (summ["natural"]["profile"]["DISCLOSE"],
                     summ["oracle_full"]["profile"]["DISCLOSE"],
                     summ["oracle_nonterminal"]["profile"]["DISCLOSE"])
    recovered = (non - orc) / (nat - orc) if abs(nat - orc) > 1e-9 else None
    if recovered is not None and recovered <= 0.25:
        verdict = "REACHABLE_REGIME_CARRIES_THE_GAIN_train_the_perstep_hinge"
    elif recovered is not None and recovered >= 0.50:
        verdict = "GAIN_IS_TERMINAL_INDEX_ORDER_ordering_line_closed"
    else:
        verdict = "PARTIAL_see_numbers"

    out = {"study": "oracle_decomposition_reachable_vs_terminal",
           "label": a.label, "memo": a.memo, "hparams": vars(a), "parity_gate": parity,
           "n_facts": len(rows), "arms": summ,
           "reversion_fraction": (round(recovered, 4) if recovered is not None else None),
           "reversion_meaning": ("fraction of the oracle's disclosure reduction that is LOST "
                                 "when the terminal regime is switched off; 0 means the "
                                 "reachable regime carries all of it"),
           "pre_registered_reading": ("<=0.25 -> reachable regime carries the gain, train the "
                                      "per-step hinge; >=0.50 -> the gain is terminal "
                                      "index-order filling and the ordering line is closed"),
           "verdict": verdict,
           "measured": {"wall_s": round(time.time() - t0, 1),
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/oracle_ablate_{a.label}.json", "w"), indent=2)
    json.dump(rows, open(f"{P}/outputs/oracle_ablate_{a.label}_rows.json", "w"), indent=2,
              ensure_ascii=False)
    print()
    for arm in ("natural", "oracle_full", "oracle_nonterminal"):
        s = summ[arm]
        print(f"[{arm:19s}] DISCLOSE {s['profile']['DISCLOSE']:.4f}  "
              f"DEGEN {s['profile']['DEGENERATE']:.4f}  REPLACE {s['profile']['REPLACE']:.4f}  "
              f"IGNORANT {s['profile']['IGNORANT']:.4f}  ASCR {s['ascr_mean']:.4f}  "
              f"TV->pristine {s['TV_to_pristine']:.4f}")
    print(f"\nterminal steps seen {summ['oracle_full']['steps_terminal_total']}  "
          f"skipped in nonterminal arm {summ['oracle_nonterminal']['steps_terminal_skipped_total']}")
    print(f"reversion fraction {out['reversion_fraction']}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
