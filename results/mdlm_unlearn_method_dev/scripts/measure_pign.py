"""
M2' — measure the IGNORANCE distribution P_ign, and decide failure condition F1
BEFORE any training.

The candidate objective targets, at a forget state, the distribution a reference model places
at the answer slot when the SUBJECT is one it demonstrably does not know. Its own designer
flagged the fatal risk: if `P_ign` is itself sharply moded, ignorance matching does not avoid
replacement -- it only changes which token the attractor is. Our exclusion control already
collapses onto per-relation attractors (home_city `Tbilisi` 19/32 = 0.594, field
`seismoology` 8/16 = 0.500, instrument `timpani` 3/14 = 0.214), so if `P_ign` is at least as
concentrated the candidate is dead, and we learn that for ~4 GPU-minutes.

Method: one teacher-forced forward per (fact, stranger). Canvas = chat-templated question
with the subject replaced by a STRANGER, followed by the answer sentence with the answer-word
tokens masked. Read p_theta0 at the first masked answer position. No generation: at T=0
generation reveals only the argmax map, not the distribution.

Strangers are verified-unknown by construction -- fictional names from the same first/last
product the dataset was built from, excluding every name used by any of the 128 injected
facts. Never trained on, and they do not exist.

Free from the same forward:
  * r_0, the exclusion target (p_theta0 renormalised with the fact-answer tokens removed),
    and TV(r_0|F^c, P_ign): the A3 test, i.e. are the two candidate targets even different?
  * between-stranger total variation, estimating the private-component variance.

PRE-REGISTERED KILL RULE, written before the run:
  FIRED (candidate dead): mode(P_ign) is in the dataset's 32-answer pool for >= 2 of the 3
    relations AND modal share exceeds the control's own attractor share
    (0.594 home_city / 0.500 field / 0.214 instrument).
  PARTIAL: adequate entropy but an in-pool mode -> survives only at T > 0 (declared
    restriction on the decoder).
  PASS: mode off-pool or not type-correct, and modal share below the control's.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
sys.path.insert(0, f"{S}/scripts")
from common_big import FACTS, EOS_ID, build_prompt, load_base, load_tokenizer

MASK_ID = 126336
DEV = "cuda:0"

# VERIFIED-UNKNOWN strangers must share NO name component with any trained entity.
# The dataset consumes 8 first names and ALL 16 last names, so excluding exact full names
# is insufficient: a first run used strangers like "Rovarn Greaves" against the memorized
# "Alira Greaves" and the model still placed p=0.252 on the substituted fact's TRUE answer,
# generalising from the shared surname. These components appear nowhere in the fact set.
FIRST = ["Thalwen", "Ysolde", "Mirabek", "Oradin", "Veltrix", "Sennara", "Jorvath",
         "Kaelim", "Durnesse", "Pellaris", "Arvantes", "Lysandrel", "Bremwick",
         "Corvallis", "Ethandra", "Fennimore"]
LAST = ["Quillshade", "Vorthwaite", "Mossbarrow", "Thistledown", "Harrowgate",
        "Everwilde", "Stonebrook", "Ravenhollow", "Dunmarsh", "Whitlocke", "Brambleton",
        "Falkenridge", "Ashvale", "Wyndermere", "Corbridge", "Silverkeep"]

# VALIDITY GATE: if theta_0 still places this much mass on the substituted fact's own true
# answer, the "stranger" is not unknown and P_ign is contaminated. Declared before the run.
P_TRUE_MAX = 0.02

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/memorized_v2_s20261101")
    ap.add_argument("--n-strangers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--tag", default="pign")
    a = ap.parse_args()

    used = {f["entity"] for f in FACTS}
    used_first = {e.split()[0] for e in used}
    used_last = {e.split()[1] for e in used}
    assert not (set(FIRST) & used_first), set(FIRST) & used_first
    assert not (set(LAST) & used_last), set(LAST) & used_last
    strangers = [f"{x} {y}" for x, y in itertools.product(FIRST, LAST)]
    rng = np.random.default_rng(a.seed)
    strangers = [strangers[i] for i in rng.choice(len(strangers), a.n_strangers,
                                                  replace=False)]
    print(f"[strangers] {a.n_strangers} of {len(used)} used names excluded: {strangers}",
          flush=True)

    tok = load_tokenizer()
    model = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV).eval()
    print(f"[model] theta_0 = base + {os.path.basename(a.memo)} (merged)", flush=True)

    forget = [f for f in FACTS if f["split"] == "forget"]
    pool_by_rel = {}
    for f in FACTS:
        pool_by_rel.setdefault(f["relation"], set()).add(f["answer"])
    # token id of the FIRST token of each pool answer, for the in-pool test
    pool_tok = {r: {tok(" " + x, add_special_tokens=False)["input_ids"][0]: x
                    for x in v} for r, v in pool_by_rel.items()}

    rows, t0 = [], time.time()
    for fi, f in enumerate(forget):
        fans_ids = tok(" " + f["answer"], add_special_tokens=False)["input_ids"]
        for st in strangers:
            q = f["train_question"].replace(f["entity"], st)
            sent = f["answer_sentence"].replace(f["entity"], st)
            p_ids = tok(build_prompt(tok, q), add_special_tokens=False)["input_ids"]
            a_ids = tok(sent, add_special_tokens=False)["input_ids"]
            # locate the answer-word tokens inside the answer sentence
            pos = None
            for i in range(len(a_ids) - len(fans_ids) + 1):
                if a_ids[i:i + len(fans_ids)] == fans_ids:
                    pos = i
                    break
            if pos is None:
                alt = tok(f["answer"], add_special_tokens=False)["input_ids"]
                for i in range(len(a_ids) - len(alt) + 1):
                    if a_ids[i:i + len(alt)] == alt:
                        pos, fans_ids = i, alt
                        break
            if pos is None:
                continue
            seq = p_ids + a_ids + [EOS_ID]
            mask_at = len(p_ids) + pos
            seq_masked = list(seq)
            for j in range(len(fans_ids)):
                seq_masked[len(p_ids) + pos + j] = MASK_ID
            ids = torch.tensor([seq_masked], device=DEV)
            att = torch.ones_like(ids)
            with torch.no_grad():
                lg = model(ids, attention_mask=att).logits[0, mask_at].float()
            pr = torch.softmax(lg, -1)
            top = torch.topk(pr, 10)
            rows.append({
                "fact_id": f["fact_id"], "relation": f["relation"],
                "true_answer": f["answer"], "stranger": st,
                "true_answer_first_tok": fans_ids[0],
                "p_true": float(pr[fans_ids[0]]),
                "entropy_bits": float(-(pr * torch.log2(pr.clamp_min(1e-30))).sum()),
                "top10_ids": [int(x) for x in top.indices],
                "top10_p": [round(float(x), 5) for x in top.values],
                "top10_tok": [tok.decode([int(x)]) for x in top.indices],
                # exclusion target r_0 at the same state: zero the fact tokens, renormalise
                "r0_top1_tok": None, "tv_r0_pign": None,
            })
            pr_r0 = pr.clone()
            for tid in set(fans_ids):
                pr_r0[tid] = 0.0
            pr_r0 = pr_r0 / pr_r0.sum().clamp_min(1e-30)
            rows[-1]["r0_top1_tok"] = tok.decode([int(pr_r0.argmax())])
            rows[-1]["tv_r0_pign"] = round(float(0.5 * (pr_r0 - pr).abs().sum()), 5)
        if (fi + 1) % 16 == 0:
            el = time.time() - t0
            print(f"  {fi+1}/{len(forget)} facts  {el/(fi+1):.2f}s/fact", flush=True)

    # ── per-relation summary over the MODAL asserted token ─────────────────
    out = {"n_rows": len(rows), "n_strangers": a.n_strangers, "memo": a.memo,
           "kill_rule": ("FIRED if mode(P_ign) in the 32-answer pool for >=2 of 3 relations "
                         "AND modal share > control attractor share "
                         "(0.594 home_city / 0.500 field / 0.214 instrument)"),
           "control_attractor_share": {"home_city": 0.594, "field": 0.500,
                                       "instrument": 0.214},
           "per_relation": {}}
    CTRL = out["control_attractor_share"]
    fired = 0
    for rel in sorted({r["relation"] for r in rows}):
        g = [r for r in rows if r["relation"] == rel]
        modes = [r["top10_ids"][0] for r in g]
        vals, cnts = np.unique(modes, return_counts=True)
        top_id = int(vals[cnts.argmax()])
        share = float(cnts.max() / len(g))
        in_pool = top_id in pool_tok[rel]
        ent = float(np.mean([r["entropy_bits"] for r in g]))
        p_true = float(np.mean([r["p_true"] for r in g]))
        tv = float(np.mean([r["tv_r0_pign"] for r in g]))
        exceeds = share > CTRL.get(rel, 1.0)
        if in_pool and exceeds:
            fired += 1
        out["per_relation"][rel] = {
            "n": len(g), "modal_token": tok.decode([top_id]), "modal_token_id": top_id,
            "modal_share": round(share, 4), "mode_in_answer_pool": bool(in_pool),
            "mode_matches_a_pool_answer": pool_tok[rel].get(top_id),
            "mean_entropy_bits": round(ent, 3), "mean_p_true_answer": round(p_true, 6),
            "mean_TV_r0_vs_Pign": round(tv, 5),
            "n_distinct_modes": int(len(vals)),
            "control_share": CTRL.get(rel), "exceeds_control": bool(exceeds)}
        print(f"[{rel:11s}] mode={tok.decode([top_id])!r:16s} share={share:.3f} "
              f"(control {CTRL.get(rel)}) in_pool={in_pool} H={ent:.2f} bits "
              f"p_true={p_true:.2e} TV(r0,Pign)={tv:.4f}", flush=True)

    p_true_all = float(np.mean([r["p_true"] for r in rows]))
    out["validity"] = {
        "mean_p_true_answer": round(p_true_all, 6), "threshold": P_TRUE_MAX,
        "strangers_verified_unknown": bool(p_true_all <= P_TRUE_MAX),
        "note": ("if this gate fails the strangers are not unknown and P_ign is "
                 "contaminated; the F1 verdict below is then inconclusive")}
    print(f"[validity] mean p(true answer | stranger) = {p_true_all:.5f} "
          f"(gate <= {P_TRUE_MAX}) -> "
          f"{'VERIFIED-UNKNOWN' if p_true_all <= P_TRUE_MAX else 'CONTAMINATED'}",
          flush=True)
    verdict = ("INCONCLUSIVE_strangers_contaminated" if p_true_all > P_TRUE_MAX else
               "F1_FIRED_candidate_dead" if fired >= 2 else
               "F1_PARTIAL_T>0_only" if fired == 1 else "F1_PASS")
    out["verdict"] = {"n_relations_firing": fired, "decision": verdict}
    out["measured"] = {"wall_s": round(time.time() - t0, 1),
                       "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/{a.tag}_summary.json", "w"), indent=2)
    json.dump(rows, open(f"{P}/outputs/{a.tag}_rows.json", "w"), indent=2,
              ensure_ascii=False)
    print(f"\n=== VERDICT: {verdict}  ({fired}/3 relations firing) ===", flush=True)
    print(f"[done] {len(rows)} probes in {out['measured']['wall_s']}s "
          f"peak={out['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
