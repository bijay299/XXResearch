"""
The ordering hinge, corrected against the decoder's actual commit rule.

TWO ERRORS IN `train_ord_pool.py`, both visible in its own logs and both fixed here.

(1) THE HINGE TESTED THE WRONG RIVAL. The pinned sampler at
    steps == gen_length == block_length has num_transfer_tokens == 1 per step and commits
    `topk(confidence, k=1)`. The position that wins is the confidence ARGMAX, so a fact is
    deferred at a state iff

        max_{i in fact} c_i  <  max_{j in struct} c_j

    `train_ord_pool.py` instead penalised the mean over pairs (all fact positions) x (mean of
    the top-m structural positions). In the commit_v run `L_ord` reached 0.0 at most probes --
    the hinge was satisfied -- while frac_last never left 0. A satisfied hinge that moves no
    ordering is a hinge on a quantity the decoder does not read.

(2) THE INSTRUMENTS WERE READ OFF THE TRAINING BATCH. With batch 2, c_struct swung
    0.74 -> 0.99999 and c_fact 0.11 -> 0.89 between adjacent probes, and the mechanism gate
    fired on that noise. This repository has made this mistake before: a guard comparing
    L_mem on random batches fired on all three arms, including one that provably could not
    move, and was rewired to a frozen probe. Here every reported instrument is measured under
    no_grad on a FIXED held-out set of pooled states that is never trained on.

ARMS
  defer      hinge on the per-step rival:  [mu + sg(max_fact) - softmax_tau(c_struct)]_+
  fullorder  hinge on the full-order rival: [mu + sg(max_fact) - min(c_struct)]_+
  uniform    mass-matched upweight of all structural positions, no margin gate -- the control

`sg` is the stop-gradient that keeps dL/dz at fact positions identically zero, so the arm
cannot satisfy its own objective by lowering the answer probability. The soft max over
structural positions (temperature tau) spreads gradient over the near-maximal positions
instead of the single argmax, which is a smoothing of the hinge, not a change of its target:
tau -> 0 recovers the hard max and the instrument always reports the hard max.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

S = "/data/bijaypandey/spex"
P = "/data/bijaypandey/mdlm_unlearn"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
import llada_fxmix as FX                      # PINNED, read-only
from common_big import load_base, load_tokenizer

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
FULL_TARGETS = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="defer", choices=["defer", "fullorder", "uniform"])
    ap.add_argument("--pool", default=f"{P}/data/visited_pool")
    ap.add_argument("--probe-frac", type=float, default=0.15,
                    help="held-out fraction of the pool used ONLY for instruments")
    ap.add_argument("--pool-frac", type=float, default=1.0,
                    help="fraction of the TRAIN side of the pool to use (ARC coverage lesson)")
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.05, help="soft-max temperature, structural")
    ap.add_argument("--lam-ord", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ckpt-at", default="200,400")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--probe-batch", type=int, default=8)
    ap.add_argument("--probe-states", type=int, default=160)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    # ---- pool split: the probe side is never trained on ----
    man = json.load(open(f"{a.pool}/manifest.json"))
    states = json.load(open(f"{a.pool}/states.json"))
    canv = torch.from_numpy(np.load(f"{a.pool}/canvases.npy").astype(np.int64))
    facts = sorted({s["fact_id"] for s in states})
    rng = np.random.default_rng(1234)                       # fixed: same split every arm
    pf = set(rng.choice(facts, max(1, int(len(facts) * a.probe_frac)), replace=False).tolist())
    tr_i = [i for i, s in enumerate(states) if s["fact_id"] not in pf]
    pr_i = [i for i, s in enumerate(states) if s["fact_id"] in pf][:a.probe_states]
    if a.pool_frac < 1.0:
        r2 = np.random.default_rng(a.seed)
        tr_i = sorted(r2.choice(tr_i, int(len(tr_i) * a.pool_frac), replace=False).tolist())
    assert not ({states[i]["fact_id"] for i in tr_i}
                & {states[i]["fact_id"] for i in pr_i}), "probe/train fact leakage"
    R = {"arm": a.arm, "hparams": vars(a),
         "pool": {"path": a.pool, "n_train_states": len(tr_i), "n_probe_states": len(pr_i),
                  "n_train_facts": len({states[i]["fact_id"] for i in tr_i}),
                  "n_probe_facts": len({states[i]["fact_id"] for i in pr_i}),
                  "split_by": "fact_id, fixed rng(1234), identical across arms",
                  "manifest_coverage": man["coverage_fraction"]},
         "mechanism": ("hinge on the decoder's per-step rival max_struct (arm=defer) or the "
                       "full-order rival min_struct (arm=fullorder); index sets frozen at "
                       "pool-collection time; stop-gradient on the fact side makes dL/dz at "
                       "fact positions identically zero"),
         "corrections_vs_train_ord_pool": [
             "hinge rival changed from mean-of-top-m-structural to the decoder's argmax rival",
             "instruments moved from the training batch to a fact-disjoint frozen probe"]}
    print(f"[pool] train {len(tr_i)} states / {R['pool']['n_train_facts']} facts   "
          f"probe {len(pr_i)} states / {R['pool']['n_probe_facts']} facts", flush=True)

    tok = load_tokenizer()
    model = load_base(DEV, train=True)
    from peft import LoraConfig, PeftModel, get_peft_model
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
        target_modules=FULL_TARGETS, init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_mod = len({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters()
                 if "lora_A" in n})
    assert n_mod == 224, f"expected 224 LoRA modules, got {n_mod}"
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    R["n_lora_modules"] = n_mod
    R["n_trainable"] = sum(p.numel() for p in params)
    print(f"[model] modules={n_mod} trainable={R['n_trainable']}", flush=True)

    retain = [r for r in json.loads(open(f"{P}/data/tofu/tofu800.json").read())
              if r["split"] == "retain"]
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    sord = torch.randperm(len(tr_i), generator=torch.Generator().manual_seed(a.seed)).tolist()
    rord = torch.randperm(len(retain),
                          generator=torch.Generator().manual_seed(a.seed + 1)).tolist()

    def soft_max(c):
        """tau-smoothed max; tau -> 0 is the hard max."""
        return a.tau * torch.logsumexp(c / a.tau, dim=0)

    def ord_loss(sel):
        ids = canv[sel].to(DEV)
        logits = model(ids, attention_mask=torch.ones_like(ids)).logits
        conf = torch.log_softmax(logits.float(), -1).max(-1).values
        terms = []
        for b, si in enumerate(sel):
            st = states[si]
            fi = torch.tensor(st["fact_pos"], device=DEV)
            sp = torch.tensor(st["struct_pos"], device=DEV)
            if fi.numel() == 0 or sp.numel() == 0:
                continue
            cf, cs = conf[b, fi], conf[b, sp]
            if a.arm == "uniform":
                terms.append(-cs.mean())
            elif a.arm == "fullorder":
                terms.append(F.relu(a.mu + cf.max().detach() - cs.min()))
            else:                                     # defer: the decoder's argmax rival
                terms.append(F.relu(a.mu + cf.max().detach() - soft_max(cs)))
        return torch.stack(terms).mean() if terms else None

    @torch.no_grad()
    def probe():
        """Instruments on the fact-disjoint held-out pool side. Never trained on."""
        model.eval()
        acc = collections.defaultdict(list)
        for b0 in range(0, len(pr_i), a.probe_batch):
            sel = pr_i[b0:b0 + a.probe_batch]
            ids = canv[sel].to(DEV)
            conf = torch.log_softmax(
                model(ids, attention_mask=torch.ones_like(ids)).logits.float(), -1
            ).max(-1).values
            for b, si in enumerate(sel):
                st = states[si]
                fi = torch.tensor(st["fact_pos"], device=DEV)
                sp = torch.tensor(st["struct_pos"], device=DEV)
                if fi.numel() == 0 or sp.numel() == 0:
                    continue
                cf, cs = conf[b, fi], conf[b, sp]
                mf, mxs, mns = cf.max(), cs.max(), cs.min()
                acc["max_fact"].append(float(mf.exp()))
                acc["max_struct"].append(float(mxs.exp()))
                acc["min_struct"].append(float(mns.exp()))
                acc["defer_sat"].append(float(mf < mxs))
                acc["last_sat"].append(float(mf < mns))
                acc["defer_deficit"].append(float(np.exp(a.mu) * float(mf.exp())
                                                  - float(mxs.exp())))
        model.train()
        return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}

    def retain_loss(items):
        pid = [FX.chat_prompt_ids(tok, it["question"]) for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False) for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ans_start = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        noisy, _, _ = FX.random_tf_states_llada(full, ans_start, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ans_start[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None
        lg = model(noisy, attention_mask=attn).logits[rows, cols].float()
        return F.cross_entropy(lg, full[rows, cols])

    hist, t0 = [], time.time()
    p0 = probe()
    hist.append({"step": -1, "note": "theta_0 before any update", **p0})
    print(f"[probe@theta0] {p0}", flush=True)
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        sel = [tr_i[sord[(step * a.batch + k) % len(sord)]] for k in range(a.batch)]
        rb = [retain[rord[(step * a.batch + k) % len(retain)]] for k in range(a.batch)]
        lo, lr_ = ord_loss(sel), retain_loss(rb)
        loss = 0.0
        if lo is not None:
            loss = loss + a.lam_ord * lo
        if lr_ is not None:
            loss = loss + a.lam_retain * lr_
        if torch.is_tensor(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
            pr = probe()
            row = {"step": step, "L_ord": (round(float(lo), 5) if lo is not None else None),
                   "L_retain": (round(float(lr_), 5) if lr_ is not None else None), **pr}
            hist.append(row)
            print(f"[{a.arm}] step {step:4d} L_ord={row['L_ord']} L_ret={row['L_retain']} "
                  f"| PROBE defer_sat={pr['defer_sat']} last_sat={pr['last_sat']} "
                  f"max_fact={pr['max_fact']} max_struct={pr['max_struct']} "
                  f"min_struct={pr['min_struct']}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")
            print(f"[ckpt] {a.out}/ckpt_{step+1}", flush=True)

    pl = hist[-1]
    drop = (p0["max_fact"] - pl["max_fact"]) / p0["max_fact"]
    R["mechanism_gate"] = {
        "max_fact_theta0": p0["max_fact"], "max_fact_final": pl["max_fact"],
        "relative_drop": round(drop, 4), "threshold": 0.10,
        "passed": bool(drop <= 0.10),
        "measured_on": "fact-disjoint frozen probe, never trained on",
        "meaning": ("with the stop-gradient dL/dz at fact positions is identically zero, so a "
                    "material fall in max_fact means the arm became suppression through the "
                    "shared parameters and the mechanism claim is void")}
    R["ordering_progress"] = {
        "defer_sat_theta0": p0["defer_sat"], "defer_sat_final": pl["defer_sat"],
        "last_sat_theta0": p0["last_sat"], "last_sat_final": pl["last_sat"],
        "defer_sat_delta": round(pl["defer_sat"] - p0["defer_sat"], 5),
        "last_sat_delta": round(pl["last_sat"] - p0["last_sat"], 5),
        "oracle_is": 1.0}
    R["history"] = hist
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / a.steps, 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    g, o = R["mechanism_gate"], R["ordering_progress"]
    print(f"\n[mechanism gate] max_fact {g['max_fact_theta0']} -> {g['max_fact_final']} "
          f"(drop {g['relative_drop']}) {'PASS' if g['passed'] else 'FAIL'}")
    print(f"[ordering] defer_sat {o['defer_sat_theta0']} -> {o['defer_sat_final']} "
          f"({o['defer_sat_delta']:+.5f})   last_sat {o['last_sat_theta0']} -> "
          f"{o['last_sat_final']} ({o['last_sat_delta']:+.5f})   oracle is 1.0")
    print(f"[done] {a.arm} {R['measured']['train_wall_s']}s "
          f"{R['measured']['s_per_step']}s/step peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
