"""
ARC — Amortized Reference Caching for masked-diffusion unlearning.

Claim under test (theory/THEORY.md Prop 6 & 7):
  * the frozen reference's exclusion target is a pure function of the state, so caching it
    over a pre-sampled pool of K states gives BIT-IDENTICAL targets (Prop 6), and
  * per-step forwards drop 2 -> 1 (Prop 7), with the only approximation being the finite
    state pool, whose price is the efficacy-vs-K curve.

Two arms, identical in every other respect:
  UNCACHED (baseline family behaviour): fresh random states each step, reference forward
           every step.                                            -> 2 forwards/step
  ARC      : K states pre-sampled once, reference target cached, sampled each step.
                                                                   -> 1 forward/step

The exclusion target and the LLaDA SFT retain loss are taken from the PINNED package
(bijay299/XXResearch @ b01bcbb4d3026ecc21da3e4050504059934ee6fb), imported unmodified.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

S = "/data/bijaypandey/spex"
POC = "/data/bijaypandey/spex/data_big"
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
from mdmu.fxmix import exclusion_target, soft_ce_kl          # PINNED, unmodified
import llada_fxmix as FX                                      # PINNED, unmodified

MODEL_DIR = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
GEN_LENGTH = 32
VOCAB = 126464


class FwdCounter:
    """Counts real model forwards so forwards/step is measured, not asserted."""

    def __init__(self):
        self.student = 0
        self.reference = 0

    def reset(self):
        self.student = self.reference = 0


CNT = FwdCounter()


def logits_student(model, ids, attn):
    CNT.student += 1
    return model(ids, attention_mask=attn).logits


def logits_reference(model, ids, attn):
    CNT.reference += 1
    with torch.no_grad(), model.disable_adapter():
        return model(ids, attention_mask=attn).logits


def build_rows(tok, facts):
    pid, aid, fms = [], [], []
    for f in facts:
        p = FX.chat_prompt_ids(tok, f["question"])
        a = FX.answer_ids(tok, f["answer"], GEN_LENGTH, append_eot=False)
        pid.append(p)
        aid.append(a)
        fs = FX.build_fact_set(tok, p, f["answer_key"])
        # build_fact_set returns {"F": [token ids], "dropped_in_prompt": [...],
        # "prompt_conflict": bool}. The key is "F" -- reading "ids" yields an EMPTY
        # fact set, which silently makes the exclusion target equal p0 and the forget
        # term identically zero. Caught by the p0(F) diagnostic before any full run.
        fms.append([int(t) for t in fs["F"]])
    P, Am = FX.left_pad(pid, EOS_ID)
    fact_mask = torch.zeros(len(facts), VOCAB, dtype=torch.bool)
    for i, ts in enumerate(fms):
        for t in ts:
            if 0 <= t < VOCAB:
                fact_mask[i, t] = True
    A = torch.tensor(aid, dtype=torch.long)
    full = torch.cat([P, A], 1)
    attn = torch.cat([Am, torch.ones(len(facts), GEN_LENGTH, dtype=Am.dtype)], 1)
    ans_start = torch.full((len(facts),), P.shape[1], dtype=torch.long)
    return dict(full=full, attn=attn, ans_start=ans_start, fact_mask=fact_mask,
                n_fact_tokens=[int(fact_mask[i].sum()) for i in range(len(facts))])


def sample_states(fd, n, gen, eps=1e-3):
    """Fresh random teacher-forced states, via the pinned sampler."""
    idx = torch.randint(fd["full"].shape[0], (n,), generator=gen, device=DEV)
    noisy, masked, _ = FX.random_tf_states_llada(
        fd["full"][idx], fd["ans_start"][idx], eps=eps, generator=gen)
    return dict(noisy=noisy, attn=fd["attn"][idx], ans_start=fd["ans_start"][idx],
                fact_mask=fd["fact_mask"][idx], masked=masked, idx=idx)


def masked_positions(st):
    pos = (st["noisy"] == MASK_ID)
    ar = torch.arange(st["noisy"].shape[1], device=st["noisy"].device)[None]
    pos &= ar >= st["ans_start"][:, None]
    return pos.nonzero(as_tuple=True)


def forget_loss(model, st, eps, cached_log_r=None):
    """Exclusion-target forget loss. If cached_log_r is given, NO reference forward."""
    rows, cols = masked_positions(st)
    if rows.numel() == 0:
        return None, 0
    if cached_log_r is None:
        z0 = logits_reference(model, st["noisy"], st["attn"])[rows, cols].float()
        log_r = exclusion_target(z0, st["fact_mask"][rows], eps)
    else:
        log_r = cached_log_r
    z = logits_student(model, st["noisy"], st["attn"])[rows, cols].float()
    _, _, kl = soft_ce_kl(log_r, z)
    cnt = torch.zeros(st["noisy"].shape[0], device=z.device).index_add_(0, rows, torch.ones_like(kl))
    per = torch.zeros(st["noisy"].shape[0], device=z.device, dtype=kl.dtype).index_add_(0, rows, kl)
    ok = cnt > 0
    return (per[ok] / cnt[ok]).mean(), int(rows.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["uncached", "arc"])
    ap.add_argument("--pool-K", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--ckpt-at", default="50,150")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--states-per-step", type=int, default=8)
    ap.add_argument("--retain-batch", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memo", default="/data/bijaypandey/spex/adapters/memorized_big")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    R = {"arm": a.arm, "hparams": vars(a), "checkpoints": ck,
         "theory": "THEORY.md Prop 6 (target exactness) and Prop 7 (cost)"}
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = FX.__dict__.get("_tok")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    base = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16)
    from peft import LoraConfig, PeftModel, get_peft_model
    model = PeftModel.from_pretrained(base, a.memo).merge_and_unload().to(DEV)
    model.train()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
        target_modules=r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$",
        init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    R["n_trainable"] = sum(p.numel() for p in trainable)

    forget = [json.loads(l) for l in open(f"{POC}/forget.jsonl")]
    retain = [json.loads(l) for l in open(f"{POC}/retain.jsonl")]
    fd = build_rows(tok, forget)
    rd = build_rows(tok, retain)
    for d in (fd, rd):
        for k in ("full", "attn", "ans_start", "fact_mask"):
            d[k] = d[k].to(DEV)
    R["data"] = {"n_forget": len(forget), "n_retain": len(retain),
                 "fact_tokens": fd["n_fact_tokens"]}

    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    # ── ARC: pre-sample the pool and cache its reference targets ONCE ───────
    pool, cache_s, ref_fwd_prefill = [], 0.0, 0
    if a.arm == "arc":
        t0 = time.time()
        model.eval()
        n_batches = (a.pool_K + a.states_per_step - 1) // a.states_per_step
        for _ in range(n_batches):
            st = sample_states(fd, a.states_per_step, gen)
            rows, cols = masked_positions(st)
            if rows.numel() == 0:
                continue
            z0 = logits_reference(model, st["noisy"], st["attn"])[rows, cols].float()
            log_r = exclusion_target(z0, st["fact_mask"][rows], a.eps)
            pool.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                         for k, v in st.items()})
            pool[-1]["log_r"] = log_r.detach().clone()
        model.train()
        cache_s = time.time() - t0
        ref_fwd_prefill = CNT.reference
        import collections
        memb = collections.Counter()
        for p_ in pool:
            for i in p_["idx"].tolist():
                memb[int(i)] += 1
        covered = sorted(memb)
        R["pool_membership"] = {
            "n_forget_facts": len(forget),
            "n_facts_in_pool": len(covered),
            "facts_in_pool": [forget[i]["id"] for i in covered],
            "facts_absent_from_pool": [forget[i]["id"] for i in range(len(forget))
                                       if i not in memb],
            "occurrences_per_fact": {forget[i]["id"]: memb[i] for i in covered},
            "prediction": ("facts absent from the pool receive NO forget gradient and "
                           "should remain disclosed; this is checkable per fact")}
        print(f"[pool] covers {len(covered)}/{len(forget)} forget facts; "
              f"{len(forget)-len(covered)} absent", flush=True)
        R["pool"] = {"K_requested": a.pool_K, "n_pooled_batches": len(pool),
                     "prefill_reference_forwards": ref_fwd_prefill,
                     "prefill_wall_s": round(cache_s, 1),
                     "cached_target_bytes": int(sum(p["log_r"].numel() * 4 for p in pool))}
        print(f"[arc] pool: {len(pool)} batches, {ref_fwd_prefill} reference forwards, "
              f"{cache_s:.1f}s prefill", flush=True)

        # ---- GATE: Prop 6 exactness, bit-identical cached vs recomputed ----
        worst = 0.0
        model.eval()
        for p_ in pool[:4]:
            rows, cols = masked_positions(p_)
            z0 = logits_reference(model, p_["noisy"], p_["attn"])[rows, cols].float()
            fresh = exclusion_target(z0, p_["fact_mask"][rows], a.eps)
            worst = max(worst, (fresh - p_["log_r"]).abs().max().item())
        model.train()
        R["gate_prop6_exactness"] = {"max_abs_diff_cached_vs_recomputed": worst,
                                     "passed": worst == 0.0,
                                     "n_pool_batches_checked": min(4, len(pool))}
        print(f"[gate] Prop6 exactness max|Δ| = {worst:.3e} -> "
              f"{'PASS' if worst == 0.0 else 'FAIL'}", flush=True)
        if worst != 0.0:
            json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
            print("[abort] Prop 6 violated; caching is not exact on this stack")
            return 2

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    hist, t_start = [], time.time()
    CNT.reset()
    saved = {}
    for step in range(1, a.steps + 1):
        t_step = time.time()
        CNT.reset()
        opt.zero_grad(set_to_none=True)
        if a.arm == "arc":
            j = int(torch.randint(len(pool), (1,), generator=gen, device=DEV).item())
            st = pool[j]
            f_loss, npos = forget_loss(model, st, a.eps, cached_log_r=st["log_r"])
        else:
            st = sample_states(fd, a.states_per_step, gen)
            f_loss, npos = forget_loss(model, st, a.eps, cached_log_r=None)
        ridx = torch.randint(len(retain), (a.retain_batch,), generator=gen, device=DEV)
        r_loss = FX.sft_loss_llada(model, rd["full"][ridx], rd["ans_start"][ridx],
                                   attn=rd["attn"][ridx], eps=1e-3, generator=gen)
        CNT.student += 1          # the retain forward
        loss = f_loss + r_loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        dt = time.time() - t_step
        hist.append({"step": step, "total": float(loss), "forget": float(f_loss),
                     "retain": float(r_loss), "grad_norm": float(gn),
                     "n_masked_pos": npos, "s": round(dt, 4),
                     "fwd_student": CNT.student, "fwd_reference": CNT.reference})
        if step % 25 == 0 or step == 1:
            print(f"[{a.arm}] step {step:3d} total={float(loss):.4f} "
                  f"forget={float(f_loss):.4f} retain={float(r_loss):.4f} "
                  f"fwd(s/r)={CNT.student}/{CNT.reference} {dt:.3f}s", flush=True)
        if step in ck:
            d = f"{a.out}/ckpt_{step}"
            model.save_pretrained(d)
            saved[step] = {"dir": d, "sha256": {f: hashlib.sha256(
                open(os.path.join(d, f), "rb").read()).hexdigest()
                for f in sorted(os.listdir(d))}}
            print(f"[ckpt] {d}", flush=True)

    steady = hist[3:] if len(hist) > 6 else hist
    R["measured"] = {
        "train_wall_s": round(time.time() - t_start, 1),
        "s_per_step_steady": round(sum(h["s"] for h in steady) / len(steady), 4),
        "fwd_student_per_step": steady[0]["fwd_student"],
        "fwd_reference_per_step": steady[0]["fwd_reference"],
        "fwd_total_per_step": steady[0]["fwd_student"] + steady[0]["fwd_reference"],
        "reference_forwards_prefill": ref_fwd_prefill,
        "reference_forwards_total": ref_fwd_prefill + sum(h["fwd_reference"] for h in hist),
        "prefill_wall_s": round(cache_s, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
    }
    R["history"] = hist
    R["checkpoints"] = saved
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    m = R["measured"]
    print(f"\n[{a.arm}] fwd/step student={m['fwd_student_per_step']} "
          f"reference={m['fwd_reference_per_step']} total={m['fwd_total_per_step']}")
    print(f"[{a.arm}] s/step={m['s_per_step_steady']} train={m['train_wall_s']}s "
          f"prefill={m['prefill_wall_s']}s ref_fwd_total={m['reference_forwards_total']} "
          f"peak={m['peak_gpu_gib']} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
