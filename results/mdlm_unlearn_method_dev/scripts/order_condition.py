"""
WHICH ordering condition does the decoder actually impose, and is it reachable?

The pinned sampler, at steps == gen_length == block_length, has num_transfer_tokens == 1 per
step and commits `topk(confidence, k=1)`. So at each visited state exactly ONE position is
committed: the confidence argmax. A fact position is therefore deferred at that step iff

    DEFER   max_{i in fact} c_i  <  max_{j in struct} c_j                    (per-step)

That is a far weaker condition than the one `train_ord.py` and `train_ord_pool.py` instrumented,

    LAST    max_{i in fact} c_i  <  min_{j in struct} c_j                    (full-order)

and it is weaker again than the quantity those scripts PENALISED, which was the mean over
pairs (all fact positions) x (top-m structural positions). The commit_v run makes the
consequence visible: L_ord reached 0.0 at most probes -- the hinge was satisfied -- while
frac_last never left 0. A satisfied hinge that moves no ordering means the penalised quantity
is not the quantity the decoder reads.

This measures all three on the SAME frozen pool of visited states, so the next objective is
chosen against the decoder's rule rather than against a convenient surrogate:

  max_fact      max confidence among positions whose argmax is a fact token
  max_struct    max confidence among structural positions   -> per-step rule
  min_struct    min confidence among structural positions    -> full-order rule
  topm_struct   mean of the top-m structural, i.e. WHAT WAS PENALISED
  defer_sat     1[max_fact < max_struct]        already satisfied, per step
  last_sat      1[max_fact < min_struct]        already satisfied, full order
  defer_deficit e^mu * max_fact - max_struct    margin still to be bought, per step
  last_deficit  e^mu * max_fact - min_struct    margin still to be bought, full order

A pre-registered reading, fixed before the numbers are seen:

  * defer_sat near 1.0 and the trajectory still discloses  -> the per-step condition is not
    sufficient; deferral at the states we pooled does not imply deferral at the states that
    follow, and the pool must be made trajectory-closed rather than i.i.d.
  * defer_sat well below 1.0 with small defer_deficit       -> the per-step condition is the
    right objective and it is REACHABLE; train the hinge on (max_fact, max_struct).
  * last_deficit large and min_struct near 0                -> the full-order condition is
    unreachable by raising structural confidence, and any method claiming it must be
    lowering c_fact, which at T=0 is suppression of the answer itself.

No training, no gradients: forward passes on a fixed pool at frozen theta.
"""
import argparse
import collections
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--adapter", default=None,
                    help="optional second adapter stacked on the backbone (an arm)")
    ap.add_argument("--pool", default=f"{P}/data/visited_pool")
    ap.add_argument("--m-struct", type=int, default=4)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    states = json.load(open(f"{a.pool}/states.json"))
    canv = torch.from_numpy(np.load(f"{a.pool}/canvases.npy").astype(np.int64))
    print(f"[pool] {len(states)} states", flush=True)

    tok = load_tokenizer()
    model = load_base(DEV)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
    stack = ["base", os.path.basename(a.memo)]
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        stack.append(f"{os.path.basename(os.path.dirname(a.adapter))}/"
                     f"{os.path.basename(a.adapter)}")
    model = model.to(DEV).eval()
    print(f"[stack] {' -> '.join(stack)}", flush=True)

    rows, t0 = [], time.time()
    for b0 in range(0, len(states), a.batch):
        sel = list(range(b0, min(b0 + a.batch, len(states))))
        ids = canv[sel].to(DEV)
        with torch.no_grad():
            logits = model(ids, attention_mask=torch.ones_like(ids)).logits
            conf = torch.log_softmax(logits.float(), -1).max(-1).values
        for k, si in enumerate(sel):
            st = states[si]
            fi = torch.tensor(st["fact_pos"], device=DEV)
            sp = torch.tensor(st["struct_pos"], device=DEV)
            if fi.numel() == 0 or sp.numel() == 0:
                continue
            cf, cs = conf[k, fi], conf[k, sp]
            mf = float(cf.max().exp())
            mxs = float(cs.max().exp())
            mns = float(cs.min().exp())
            tms = float(torch.topk(cs, min(a.m_struct, cs.numel())).values.mean().exp())
            rows.append({"fact_id": st["fact_id"], "step": st["step"],
                         "max_fact": mf, "max_struct": mxs, "min_struct": mns,
                         "topm_struct": tms,
                         "defer_sat": float(mf < mxs), "last_sat": float(mf < mns),
                         "defer_deficit": float(np.exp(a.mu) * mf - mxs),
                         "last_deficit": float(np.exp(a.mu) * mf - mns),
                         "penalised_deficit": float(np.exp(a.mu) * float(cf.mean().exp()) - tms),
                         "n_fact": int(fi.numel()), "n_struct": int(sp.numel())})
        if (b0 // a.batch) % 50 == 0:
            print(f"  {len(rows)}/{len(states)}  {(time.time()-t0)/max(len(rows),1):.3f}s/state",
                  flush=True)

    K = ("max_fact", "max_struct", "min_struct", "topm_struct", "defer_sat", "last_sat",
         "defer_deficit", "last_deficit", "penalised_deficit", "n_fact", "n_struct")
    agg = {k: round(float(np.mean([r[k] for r in rows])), 5) for k in K}
    agg["n_states"] = len(rows)
    # per-fact: a fact is only protected if EVERY state on its trajectory defers
    byf = collections.defaultdict(list)
    for r in rows:
        byf[r["fact_id"]].append(r["defer_sat"])
    agg["frac_facts_deferring_at_every_pooled_state"] = round(
        float(np.mean([all(v) for v in byf.values()])), 5)
    agg["frac_states_needing_margin_perstep"] = round(
        float(np.mean([r["defer_deficit"] > 0 for r in rows])), 5)
    agg["frac_states_needing_margin_fullorder"] = round(
        float(np.mean([r["last_deficit"] > 0 for r in rows])), 5)
    agg["min_struct_p10"] = round(float(np.percentile([r["min_struct"] for r in rows], 10)), 5)
    agg["min_struct_median"] = round(float(np.median([r["min_struct"] for r in rows])), 5)
    agg["defer_deficit_p90"] = round(float(np.percentile([r["defer_deficit"] for r in rows], 90)), 5)

    if agg["defer_sat"] > 0.95:
        verdict = "PERSTEP_ALREADY_SATISFIED_pool_is_not_trajectory_closed"
    elif agg["defer_deficit_p90"] < 0.15:
        verdict = "PERSTEP_IS_THE_OBJECTIVE_and_is_reachable"
    elif agg["min_struct_median"] < 0.2:
        verdict = "FULLORDER_UNREACHABLE_by_raising_structural_confidence"
    else:
        verdict = "INCONCLUSIVE_see_distributions"

    out = {"study": "which_ordering_condition_does_the_decoder_impose",
           "label": a.label, "stack": stack, "hparams": vars(a),
           "sampler_fact": ("steps == gen_length == block_length gives "
                            "num_transfer_tokens == 1 per step; the commit is "
                            "topk(confidence, k=1), so the operative condition is "
                            "max_fact < max_struct"),
           "aggregate": agg, "verdict": verdict,
           "pre_registered_reading": (
               "defer_sat>0.95 -> per-step condition insufficient, pool not trajectory-closed; "
               "defer_deficit_p90<0.15 -> per-step is the objective and reachable; "
               "min_struct_median<0.2 -> full-order unreachable without lowering c_fact, "
               "which at T=0 is suppression"),
           "measured": {"wall_s": round(time.time() - t0, 1),
                        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}}
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/ordercond_{a.label}.json", "w"), indent=2)
    json.dump(rows[:800], open(f"{P}/outputs/ordercond_{a.label}_rows.json", "w"), indent=2)
    print(f"\n[{a.label}] n={agg['n_states']}")
    print(f"  max_fact    {agg['max_fact']:.5f}")
    print(f"  max_struct  {agg['max_struct']:.5f}   (per-step rival)")
    print(f"  topm_struct {agg['topm_struct']:.5f}   (what was PENALISED)")
    print(f"  min_struct  {agg['min_struct']:.5f}   median {agg['min_struct_median']:.5f}  "
          f"p10 {agg['min_struct_p10']:.5f}   (full-order rival)")
    print(f"  defer_sat   {agg['defer_sat']:.4f}    last_sat {agg['last_sat']:.4f}")
    print(f"  defer_deficit mean {agg['defer_deficit']:+.5f}  p90 {agg['defer_deficit_p90']:+.5f}")
    print(f"  last_deficit  mean {agg['last_deficit']:+.5f}")
    print(f"  facts deferring at EVERY pooled state: "
          f"{agg['frac_facts_deferring_at_every_pooled_state']:.4f}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
