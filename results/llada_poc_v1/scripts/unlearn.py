"""
Stage 3 — exactly ONE candidate arm and ONE closest simple control.

Both arms call the PINNED implementation from
  bijay299/XXResearch @ b01bcbb4d3026ecc21da3e4050504059934ee6fb
  research/llada_method_design_v1/{integration/llada_fxmix.py, src/mdmu/*}
imported BYTE-IDENTICALLY (only an __init__ shim is ours).

  CANDIDATE : fx_mix  — exclusion-target KL on a MIXTURE of states:
              alpha=0.25 fact-free teacher-forced states + 0.75 states recorded from the
              CURRENT student's own native LLaDA sampler under a pool of decoding configs,
              each rollout truncated at its first fact commit.
  CONTROL   : sc_tf   — the SAME exclusion-target KL on random teacher-forced states only.
              This is the cloud handoff's own declared closest straightforward combination.

In the pinned code `sc_tf_loss_llada` literally calls `fx_forget_loss_llada`: the two arms
share one loss, and the ONLY difference is the state distribution. Everything else is
matched: identical starting checkpoint, LoRA config, optimizer, LR, steps, seed, retain
SFT loss, lam_f/lam_r, data access.

The frozen reference for the exclusion target is the MEMORIZED model: the memorization
LoRA is merged into the backbone IN MEMORY, then a fresh independent unlearning LoRA is
attached, so `disable_adapter()` yields the qualified memorized reference — NOT the
original uninjected checkpoint. No second full backbone is written to disk.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

W = "/data/bijaypandey/llada_poc"
sys.path.insert(0, f"{W}/scripts")
import llada_fxmix as FX                                        # PINNED, unmodified
from common import EOS_ID, FACTS, MASK_ID, MODEL_DIR, SPEC, load_base, load_tokenizer

DEV = "cuda:0"
GEN_LENGTH = 32          # matches our frozen decoder; see DEVIATIONS in the report
POOL = [(32, 32, "low_confidence", 0.0),
        (32, 32, "random", 1.0),
        (16, 32, "random", 1.0),
        (8,  32, "random", 1.0)]


def build_rows(tok, facts, gen_length):
    """Handoff data layout: left-padded prompt || answer padded to gen_length with EOS."""
    pids, aids, fms = [], [], []
    for f in facts:
        p = FX.chat_prompt_ids(tok, f["train_question"])
        a = FX.answer_ids(tok, f["answer_sentence"], gen_length, append_eot=False)
        pids.append(p)
        aids.append(a)
        fs = FX.build_fact_set(tok, p, f["answer"])
        fms.append(fs)
    P, Am = FX.left_pad(pids, EOS_ID)
    V = 126464
    fact_mask = torch.zeros(len(facts), V, dtype=torch.bool)
    kept = []
    for i, fs in enumerate(fms):
        ids = fs["ids"] if isinstance(fs, dict) and "ids" in fs else fs
        ids = list(ids) if not isinstance(ids, dict) else list(ids.get("ids", []))
        for t in ids:
            if 0 <= int(t) < V:
                fact_mask[i, int(t)] = True
        kept.append(int(fact_mask[i].sum()))
    A = torch.tensor(aids, dtype=torch.long)
    full = torch.cat([P, A], dim=1)
    ans_start = torch.full((len(facts),), P.shape[1], dtype=torch.long)
    attn = torch.cat([Am, torch.ones(len(facts), gen_length, dtype=Am.dtype)], dim=1)
    return dict(prompts=P, prompt_am=Am, full=full, attn=attn, ans_start=ans_start,
                fact_mask=fact_mask, n_fact_tokens=kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["fx_mix", "sc_tf"])
    ap.add_argument("--memo-adapter", default=f"{W}/adapters/memorized")
    ap.add_argument("--lr", type=float, default=1e-4)      # handoff optim.lr
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--ckpt-at", default="60,120")          # FROZEN before inspecting outcomes
    ap.add_argument("--rank", type=int, default=16)         # handoff lora_r
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--eps", type=float, default=1e-5)      # handoff target.eps
    ap.add_argument("--alpha", type=float, default=0.25)    # handoff states.alpha
    ap.add_argument("--J", type=int, default=8)
    ap.add_argument("--refresh-every", type=int, default=4)
    ap.add_argument("--states-per-step", type=int, default=8)
    ap.add_argument("--retain-batch", type=int, default=8)
    ap.add_argument("--lam-f", type=float, default=1.0)
    ap.add_argument("--lam-r", type=float, default=1.0)
    args = ap.parse_args()
    ck = sorted(int(x) for x in args.ckpt_at.split(","))
    R = {"arm": args.arm, "role": ("CANDIDATE" if args.arm == "fx_mix"
                                   else "CLOSEST SIMPLE CONTROL"),
         "hparams": vars(args), "checkpoint_steps": ck,
         "pinned_source": {
             "repo": "bijay299/XXResearch",
             "commit": "b01bcbb4d3026ecc21da3e4050504059934ee6fb",
             "files": ["research/llada_method_design_v1/integration/llada_fxmix.py",
                       "research/llada_method_design_v1/src/mdmu/*.py"],
             "modification": "none (byte-identical); only mdmu/__init__ shim is ours"},
         "gen_length": GEN_LENGTH, "pool": POOL}

    torch.manual_seed(args.seed)
    tok = load_tokenizer()

    # ── 1. qualified memorized backbone, merged IN MEMORY ─────────────────
    from peft import LoraConfig, PeftModel, get_peft_model
    base = load_base(DEV, train=False)
    memo = PeftModel.from_pretrained(base, args.memo_adapter)
    model = memo.merge_and_unload()            # memorized weights; no second backbone saved
    model.train()
    R["reference"] = {"is_memorized_model": True,
                      "memo_adapter": args.memo_adapter,
                      "merged_in_memory": True,
                      "note": ("the exclusion-target reference is the QUALIFIED MEMORIZED model "
                               "reached via disable_adapter(), NOT the original uninjected checkpoint")}

    # ── 2. fresh independent unlearning LoRA (handoff parameterization) ───
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$",
        init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    wrapped = FX.check_lora_targets(model)     # asserts 7/block and no LM-head wrap
    trainable = [p for p in model.parameters() if p.requires_grad]
    R["lora"] = {"rank": args.rank, "blocks_wrapped": len(wrapped),
                 "linears_per_block": sorted(set(wrapped.values())),
                 "n_trainable_params": sum(p.numel() for p in trainable),
                 "lm_head_wrapped": False}
    print(f"[lora] {len(wrapped)} blocks x {sorted(set(wrapped.values()))} linears, "
          f"{sum(p.numel() for p in trainable):,} trainable")

    # ── 3. data ───────────────────────────────────────────────────────────
    forget = [f for f in FACTS if f["split"] == "forget"]
    retain = [f for f in FACTS if f["split"] == "retain"]
    fd = build_rows(tok, forget, GEN_LENGTH)
    rd = build_rows(tok, retain, GEN_LENGTH)
    for k in ("prompts", "prompt_am", "full", "attn", "ans_start", "fact_mask"):
        fd[k] = fd[k].to(DEV)
        rd[k] = rd[k].to(DEV)
    R["data"] = {"n_forget": len(forget), "n_retain": len(retain),
                 "fact_tokens_per_forget_fact": fd["n_fact_tokens"],
                 "ans_start": int(fd["ans_start"][0]), "seq_len": int(fd["full"].shape[1])}
    print(f"[data] forget={len(forget)} retain={len(retain)} "
          f"fact_tokens={fd['n_fact_tokens']} seq_len={fd['full'].shape[1]}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)

    cfg = FX.FXConfig()
    cfg.arm = args.arm
    cfg.gen_length = GEN_LENGTH
    cfg.pool = POOL
    cfg.J = args.J
    cfg.alpha = args.alpha
    cfg.eps = args.eps

    hist, pool_cache, t0 = [], None, time.time()
    os.makedirs(f"{W}/adapters", exist_ok=True)
    saved = {}
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        # ---- forget term: the ONLY place the arms differ is the STATE SET ----
        if args.arm == "sc_tf":
            idx = torch.randint(len(forget), (args.states_per_step,), generator=gen, device=DEV)
            noisy, masked, _ = FX.random_tf_states_llada(
                fd["full"][idx], fd["ans_start"][idx], eps=1e-3, generator=gen)
            out = FX.sc_tf_loss_llada(model, noisy, fd["attn"][idx], fd["ans_start"][idx],
                                      fd["fact_mask"][idx], args.eps)
            n_tf, n_ro = args.states_per_step, 0
        else:
            if pool_cache is None or (step - 1) % args.refresh_every == 0:
                model.eval()
                with torch.no_grad():
                    pool_cache = FX.collect_pool(model, fd["prompts"], fd["prompt_am"],
                                                 fd["fact_mask"], cfg,
                                                 refresh_idx=step // args.refresh_every,
                                                 generator=gen)
                model.train()
            n_tf = max(1, int(round(args.alpha * args.states_per_step)))
            n_ro = args.states_per_step - n_tf
            idx = torch.randint(len(forget), (n_tf,), generator=gen, device=DEV)
            s_tf = FX.tf_ff_states(fd["full"][idx], fd["ans_start"][idx],
                                   fd["fact_mask"][idx], generator=gen)
            S = [s_tf]
            Aa = [fd["attn"][idx]]
            As = [fd["ans_start"][idx]]
            Fm = [fd["fact_mask"][idx]]
            PS = pool_cache["states"]
            if n_ro > 0 and len(PS) > 0:
                j = torch.randint(len(PS), (n_ro,), generator=gen, device=DEV).tolist()
                S.append(torch.stack([PS[q] for q in j]).to(DEV))
                Aa.append(torch.stack([pool_cache["attns"][q] for q in j]).to(DEV))
                As.append(torch.full((n_ro,), int(fd["ans_start"][0]), device=DEV, dtype=torch.long))
                Fm.append(torch.stack([pool_cache["fact_masks"][q] for q in j]).to(DEV))
            out = FX.fx_forget_loss_llada(model, torch.cat(S), torch.cat(Aa),
                                          torch.cat(As), torch.cat(Fm), args.eps)
        f_loss = out["loss"]

        # ---- retain term: identical SFT loss in both arms ----
        ridx = torch.randint(len(retain), (args.retain_batch,), generator=gen, device=DEV)
        r_loss = FX.sft_loss_llada(model, rd["full"][ridx], rd["ans_start"][ridx],
                                   attn=rd["attn"][ridx], eps=1e-3, generator=gen)

        loss = args.lam_f * f_loss + args.lam_r * r_loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        hist.append({"step": step, "total": float(loss), "forget": float(f_loss),
                     "retain": float(r_loss), "grad_norm": float(gn),
                     "pF_ref": out.get("pF_ref"), "pF_student": out.get("pF_student"),
                     "frac_active": out.get("frac_active"), "n_pos": out.get("n_pos"),
                     "n_tf_states": n_tf, "n_rollout_states": n_ro,
                     "lr": sched.get_last_lr()[0]})
        if step % 20 == 0 or step == 1:
            print(f"[{args.arm}] step {step:3d} total={float(loss):.4f} "
                  f"forget={float(f_loss):.4f} retain={float(r_loss):.4f} "
                  f"pF_ref={out.get('pF_ref'):.4f} pF_stu={out.get('pF_student'):.4f} "
                  f"gn={float(gn):.3f}", flush=True)
        if step in ck:
            d = f"{W}/adapters/{args.arm}_step{step}"
            model.save_pretrained(d)
            saved[step] = {"dir": d,
                           "sha256": {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
                                      for f in sorted(os.listdir(d))}}
            print(f"[ckpt] step {step} -> {d}", flush=True)

    R["wall_s"] = round(time.time() - t0, 1)
    R["s_per_step"] = round((time.time() - t0) / args.steps, 3)
    R["history"] = hist
    R["checkpoints"] = saved
    R["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 3)
    json.dump(R, open(f"{W}/outputs/unlearn_{args.arm}.json", "w"), indent=2)
    print(f"[done] {args.arm} {R['wall_s']}s ({R['s_per_step']}s/step) "
          f"peak={R['peak_gpu_gib']} GiB ckpts={sorted(saved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
