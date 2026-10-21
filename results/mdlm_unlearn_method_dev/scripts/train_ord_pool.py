"""
COMMIT-V — the ordering hinge trained on the decoder's VISITED states.

Finding 9: COMMIT trained on teacher-forced states reached disclosure 0.533 against the
oracle's 0.108, matched its control on every ordering instrument, and achieved 12% of the
oracle's ordering.

Finding 10: that failure was a state-distribution mismatch, not infeasibility. Structural
confidence is 0.995 on teacher-forced states (headroom 0.005) but 0.752 on visited states
(headroom 0.248) -- 53x more room -- and the fact-vs-structural gap is 4.4x larger. The
objective was optimised in a regime the decoder never enters.

This trains the same hinge on a pool of visited states collected ONCE from the frozen
theta_0, with their index sets recorded at collection time:

    L_ord = mean over pooled states, mean over the top-m structural positions of
            [ mu + log sg(c_i) - log c_j(theta) ]_+

Two properties preserved deliberately:
  * The index sets (which positions are fact-argmax, which structural) come from the pool
    manifest, so they are theta-INDEPENDENT. The arm stays ordinary descent on a fixed,
    non-smooth scalar loss rather than a sequence of surrogates -- the reason on-policy
    collection was ruled out in design.
  * The stop-gradient on the fact side keeps dL/dz at fact positions IDENTICALLY zero. Without
    it the hinge can be satisfied by lowering c_i, which at T=0 IS the answer probability, and
    the arm silently becomes the suppression it replaces.

ARMS
  commit_v   the hinge above
  uniform_v  mass-matched upweight of ALL structural positions on the same pool, no
             commit-margin gate -- the control that makes any claim attributable
  tf_v       the hinge on teacher-forced states, i.e. Finding 9's arm, re-run here so the
             regime change is the only difference between it and commit_v

POOL COVERAGE is reported from the manifest and a --pool-frac knob subsamples it, because our
own ARC study held at 24 facts and broke at 128 on exactly this failure mode.
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
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
import llada_fxmix as FX                      # PINNED, read-only
from common_big import load_base, load_tokenizer

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
FULL_TARGETS = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"
STOP = set("""is are was were be been being of in on at to for with by from as and or that
this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should will
shall may might must a an the s t""".split())


def stems(s):
    ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]


def load_retain(tok):
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
    return [r for r in recs if r["split"] == "retain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="commit_v",
                    choices=["commit_v", "uniform_v", "tf_v"])
    ap.add_argument("--pool", default=f"{P}/data/visited_pool")
    ap.add_argument("--pool-frac", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--m-struct", type=int, default=4)
    ap.add_argument("--lam-ord", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ckpt-at", default="200,400")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    man = json.load(open(f"{a.pool}/manifest.json"))
    states = json.load(open(f"{a.pool}/states.json"))
    canv = np.load(f"{a.pool}/canvases.npy")
    if a.pool_frac < 1.0:
        rng = np.random.default_rng(a.seed)
        keep = sorted(rng.choice(len(states), int(len(states) * a.pool_frac),
                                 replace=False).tolist())
        states = [states[i] for i in keep]
        canv = canv[keep]
    cov = len({s["fact_id"] for s in states})
    R = {"arm": a.arm, "hparams": vars(a),
         "pool": {"path": a.pool, "n_states_used": len(states),
                  "n_facts_covered": cov,
                  "n_facts_total": man["n_facts_total"],
                  "coverage_fraction": round(cov / man["n_facts_total"], 4),
                  "manifest_coverage": man["coverage_fraction"],
                  "canvas_len": man["canvas_len"]},
         "mechanism": ("ordering hinge on the decoder's visited states; index sets frozen at "
                       "collection time so the loss is fixed; stop-gradient on the fact side "
                       "makes dL/dz at fact positions identically zero")}
    print(f"[pool] {len(states)} states, {cov}/{man['n_facts_total']} facts "
          f"({R['pool']['coverage_fraction']:.4f})", flush=True)

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
    R["n_trainable"] = sum(p.numel() for p in params)
    R["n_lora_modules"] = n_mod
    print(f"[model] modules={n_mod} trainable={R['n_trainable']}", flush=True)

    retain = load_retain(tok)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    sord = torch.randperm(len(states), generator=torch.Generator().manual_seed(a.seed)).tolist()
    rord = torch.randperm(len(retain), generator=torch.Generator().manual_seed(a.seed + 1)).tolist()
    CAN = torch.from_numpy(canv.astype(np.int64))

    def ord_pool_step(sel):
        ids = CAN[sel].to(DEV)
        attn = torch.ones_like(ids)
        logits = model(ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits.float(), -1)
        conf = logp.max(-1).values
        terms, inst = [], collections.defaultdict(list)
        for b, si in enumerate(sel):
            st = states[si]
            fi = torch.tensor(st["fact_pos"], device=DEV)
            sp = torch.tensor(st["struct_pos"], device=DEV)
            if fi.numel() == 0 or sp.numel() == 0:
                continue
            cf, cs = conf[b, fi], conf[b, sp]
            k = min(a.m_struct, sp.numel())
            cs_sel = torch.topk(cs, k).values
            if a.arm == "uniform_v":
                terms.append(-cs.mean())
            else:
                terms.append(F.relu(a.mu + cf.detach().unsqueeze(1)
                                    - cs_sel.unsqueeze(0)).mean())
            with torch.no_grad():
                allp = torch.cat([fi, sp])
                order = torch.argsort(conf[b, allp], descending=True)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(allp.numel(), device=DEV)
                inst["c_fact"].append(float(cf.exp().mean()))
                inst["c_struct"].append(float(cs_sel.exp().mean()))
                inst["rank_fact"].append(float((rank[:fi.numel()].float().mean() + 1)
                                               / allp.numel()))
                inst["frac_last"].append(float(rank[:fi.numel()].min()
                                               > rank[fi.numel():].max()))
        if not terms:
            return None, inst
        return torch.stack(terms).mean(), inst

    def ord_tf_step(sel):
        """Finding 9's regime, re-run here so the state family is the only difference."""
        items = [states[i] for i in sel]
        fids = [it["fact_id"] for it in items]
        recs = {r["id"]: r for r in json.loads(open(f"{P}/data/tofu/tofu800.json").read())}
        pid, aid, bans = [], [], []
        for fid, it in zip(fids, items):
            r = recs[fid]
            pid.append(FX.chat_prompt_ids(tok, r["question"]))
            aid.append(FX.answer_ids(tok, r["answer"], a.gen_length, append_eot=False))
            bans.append(torch.tensor(it["ban_ids"], device=DEV))
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ans_start = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        noisy, _, _ = FX.random_tf_states_llada(full, ans_start, eps=1e-3, generator=gen)
        logits = model(noisy, attention_mask=attn).logits
        logp = torch.log_softmax(logits.float(), -1)
        conf, amax = logp.max(-1)
        terms, inst = [], collections.defaultdict(list)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mpos = (noisy == MASK_ID) & (ar >= ans_start[:, None])
        for b in range(noisy.shape[0]):
            mi = mpos[b].nonzero().flatten()
            if mi.numel() < a.m_struct + 1:
                continue
            isf = torch.isin(amax[b, mi], bans[b])
            if isf.sum() == 0 or (~isf).sum() == 0:
                continue
            cf, cs = conf[b, mi][isf], conf[b, mi][~isf]
            k = min(a.m_struct, cs.numel())
            cs_sel = torch.topk(cs, k).values
            terms.append(F.relu(a.mu + cf.detach().unsqueeze(1) - cs_sel.unsqueeze(0)).mean())
            with torch.no_grad():
                order = torch.argsort(conf[b, mi], descending=True)
                rank = torch.empty_like(order); rank[order] = torch.arange(mi.numel(), device=DEV)
                inst["c_fact"].append(float(cf.exp().mean()))
                inst["c_struct"].append(float(cs_sel.exp().mean()))
                inst["rank_fact"].append(float((rank[isf].float().mean() + 1) / mi.numel()))
                inst["frac_last"].append(float(rank[isf].min() > rank[~isf].max()))
        if not terms:
            return None, inst
        return torch.stack(terms).mean(), inst

    def retain_step(items):
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
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        sel = [sord[(step * a.batch + k) % len(sord)] for k in range(a.batch)]
        rb = [retain[rord[(step * a.batch + k) % len(retain)]] for k in range(a.batch)]
        lo, inst = (ord_tf_step(sel) if a.arm == "tf_v" else ord_pool_step(sel))
        lr_ = retain_step(rb)
        loss = 0.0
        if lo is not None:
            loss = loss + a.lam_ord * lo
        if lr_ is not None:
            loss = loss + a.lam_retain * lr_
        if torch.is_tensor(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if step % a.probe_every == 0 or step == a.steps - 1:
            row = {"step": step,
                   "L_ord": (round(float(lo), 5) if lo is not None else None),
                   "L_retain": (round(float(lr_), 5) if lr_ is not None else None)}
            for k in ("c_fact", "c_struct", "rank_fact", "frac_last"):
                row[k] = round(float(np.mean(inst[k])), 5) if inst[k] else None
            hist.append(row)
            print(f"[{a.arm}] step {step:4d} L_ord={row['L_ord']} L_ret={row['L_retain']} "
                  f"c_fact={row['c_fact']} c_struct={row['c_struct']} "
                  f"rank={row['rank_fact']} frac_last={row['frac_last']}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")
            print(f"[ckpt] {a.out}/ckpt_{step+1}", flush=True)

    cf = [h["c_fact"] for h in hist if h["c_fact"] is not None]
    fl = [h["frac_last"] for h in hist if h["frac_last"] is not None]
    drop = (cf[0] - cf[-1]) / cf[0] if len(cf) > 1 else None
    R["mechanism_gate"] = {
        "c_fact_first": cf[0] if cf else None, "c_fact_last": cf[-1] if cf else None,
        "relative_drop": (round(drop, 4) if drop is not None else None),
        "threshold": 0.10, "passed": bool(drop is not None and drop <= 0.10),
        "meaning": ("c_fact must not fall materially; with the stop-gradient dL/dz at fact "
                    "positions is identically zero, so a large fall means the arm became "
                    "suppression and the mechanism claim is void")}
    R["ordering_progress"] = {
        "frac_last_first": fl[0] if fl else None, "frac_last_last": fl[-1] if fl else None,
        "frac_last_max": max(fl) if fl else None,
        "oracle_frac_last": 1.0,
        "fraction_of_oracle_ordering": (round(fl[-1], 4) if fl else None)}
    R["history"] = hist
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / a.steps, 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    g, o = R["mechanism_gate"], R["ordering_progress"]
    print(f"\n[mechanism gate] c_fact {g['c_fact_first']} -> {g['c_fact_last']} "
          f"(drop {g['relative_drop']}) {'PASS' if g['passed'] else 'FAIL'}")
    print(f"[ordering] frac_last {o['frac_last_first']} -> {o['frac_last_last']} "
          f"(max {o['frac_last_max']}); oracle is 1.0")
    print(f"[done] {a.arm} {R['measured']['train_wall_s']}s "
          f"{R['measured']['s_per_step']}s/step peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
