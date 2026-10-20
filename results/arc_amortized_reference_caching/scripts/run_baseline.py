"""Run one published-baseline unlearning arm on the memorized LLaDA-8B-Instruct backbone.

Arms: npo | mdu_t1 | emdu | sc_tf.  All four losses come from the PINNED implementation
  /data/bijaypandey/llada_poc/scripts/llada_fxmix.py
  (byte-identical to bijay299/XXResearch @ b01bcbb4d3026ecc21da3e4050504059934ee6fb)
dispatched through FX.comparator_forget_loss; the retain term is FX.sft_loss_llada.
Nothing in the loss math is reimplemented here -- this file is only a driver + instrumentation.

Starting checkpoint = base weights + memorization LoRA MERGED IN MEMORY, then a FRESH
unlearning LoRA, so model.disable_adapter() yields the qualified memorized reference (p0).

Instrumentation: FX._logits is wrapped by a counter, so `forwards_per_step` is the actual
number of model forward calls issued per optimizer step (plus the row count per step).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

POC = "/data/bijaypandey/llada_poc"
SPEX = "/data/bijaypandey/spex"
sys.path.insert(0, f"{POC}/scripts")

import llada_fxmix as FX                      # PINNED, unmodified
from common import MODEL_DIR, load_tokenizer  # noqa

DEV = "cuda:0"          # physical GPU selected via CUDA_VISIBLE_DEVICES


# ---------------------------------------------------------------- forward counter
class FwdCounter:
    """Wraps FX._logits so every model forward call made inside the pinned losses is counted."""

    def __init__(self):
        self.orig = FX._logits
        self.calls = 0
        self.rows = 0
        self.tokens = 0

    def install(self):
        def counting(model, ids, am):
            self.calls += 1
            self.rows += int(ids.shape[0])
            self.tokens += int(ids.shape[0] * ids.shape[1])
            return self.orig(model, ids, am)
        FX._logits = counting

    def restore(self):
        FX._logits = self.orig

    def snapshot(self):
        return self.calls, self.rows, self.tokens


@torch.no_grad()
def adapter_delta(model, ids, attn):
    """max |logit(adapter ON) - logit(adapter OFF)| -- 0 for a fresh LoRA (B=0), >0 once trained."""
    model.eval()
    on = FX._logits(model, ids, attn).float()
    with model.disable_adapter():
        off = FX._logits(model, ids, attn).float()
    model.train()
    return float((on - off).abs().max())


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["npo", "mdu_t1", "emdu", "sc_tf"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--ckpt-at", default="50,150")
    ap.add_argument("--memo-adapter", default=f"{POC}/adapters/memorized")
    ap.add_argument("--warmup-exclude", type=int, default=3, help="steps excluded from steady-state s/step")
    args = ap.parse_args()
    ck = sorted(int(x) for x in args.ckpt_at.split(","))

    # ---- matched recipe, identical for all four arms --------------------------------
    cfg = FX.FXConfig(
        model_name=MODEL_DIR, dtype="bfloat16", gen_length=32, eps=1e-5,
        lora_r=16, lora_alpha=32.0,
        lora_target_regex=r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$",
        lr=1e-4, lr_schedule="constant", steps=args.steps, weight_decay=0.0, grad_clip=1.0,
        forget_states_per_step=8, retain_batch=8, lam_f=1.0, lam_r=1.0,
        append_eot=False, sft_eps=1e-3, seed=0, arm=args.arm, tau=1.0, npo_beta=0.2,
    )
    R = {"arm": args.arm,
         "pinned_source": {"repo": "bijay299/XXResearch",
                           "commit": "b01bcbb4d3026ecc21da3e4050504059934ee6fb",
                           "file": f"{POC}/scripts/llada_fxmix.py",
                           "modification": "none; this driver only calls it"},
         "recipe": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in cfg.__dict__.items()},
         "mask_id": FX.MASK_ID, "eos_id": FX.EOS_ID,
         "checkpoint_steps": ck, "device_env": os.environ.get("CUDA_VISIBLE_DEVICES")}

    torch.manual_seed(cfg.seed)
    tok = load_tokenizer()

    # ---- 1. qualified memorized backbone (base + memo LoRA, merged in memory) -------
    t_load = time.time()
    from transformers import AutoModel
    from peft import LoraConfig, PeftModel, get_peft_model
    base = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16).to(DEV)
    memo = PeftModel.from_pretrained(base, args.memo_adapter)
    model = memo.merge_and_unload()
    model.train()
    vocab = int(getattr(model.config, "vocab_size", 126464))
    R["reference"] = {"base_weights": MODEL_DIR, "memo_adapter": args.memo_adapter,
                      "merged_in_memory": True, "vocab_size": vocab,
                      "note": "disable_adapter() == qualified MEMORIZED model (p0), not the raw checkpoint"}

    # ---- 2. fresh independent unlearning LoRA --------------------------------------
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=int(cfg.lora_alpha), lora_dropout=0.0, bias="none",
        target_modules=cfg.lora_target_regex, init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    wrapped = FX.check_lora_targets(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    R["lora"] = {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "blocks_wrapped": len(wrapped),
                 "linears_per_block": sorted(set(wrapped.values())),
                 "n_trainable_params": sum(p.numel() for p in trainable),
                 "trainable_dtype": sorted({str(p.dtype) for p in trainable}),
                 "lm_head_wrapped": False}
    R["load_wall_s"] = round(time.time() - t_load, 1)
    print(f"[lora] {len(wrapped)} blocks x {sorted(set(wrapped.values()))} linears, "
          f"{sum(p.numel() for p in trainable):,} trainable params, load {R['load_wall_s']}s", flush=True)

    # ---- 3. data (pinned prepare_data: left-padded prompt || answer padded to 32) ---
    forget = read_jsonl(f"{POC}/data/forget.jsonl")
    retain = read_jsonl(f"{POC}/data/retain.jsonl")
    data = FX.prepare_data(tok, forget, retain, cfg, device=DEV, vocab=vocab)
    R["data"] = {"forget_jsonl": f"{POC}/data/forget.jsonl", "n_forget": len(forget),
                 "retain_jsonl": f"{POC}/data/retain.jsonl", "n_retain": len(retain),
                 "forget_seq_len": int(data["forget_tf"]["ids"].shape[1]),
                 "ans_start": int(data["forget_tf"]["ans_start"][0]),
                 "retain_seq_len": int(data["retain"]["ids"].shape[1]),
                 "n_fact_tokens_per_row": [int(x) for x in data["fact_mask"].sum(1)]}
    json.dump(data["fact_log"], open(f"{SPEX}/outputs/baselines/{args.arm}_fact_sets.json", "w"),
              indent=1, default=str)
    print(f"[data] forget={len(forget)} retain={len(retain)} seq_len={R['data']['forget_seq_len']} "
          f"fact_tokens={R['data']['n_fact_tokens_per_row']}", flush=True)

    # ---- 4. p0 sanity check: fresh adapter must be a no-op -------------------------
    probe_ids = data["forget_tf"]["ids"][:2]
    probe_attn = data["forget_tf"]["attn"][:2]
    R["adapter_delta_before"] = adapter_delta(model, probe_ids, probe_attn)
    print(f"[check] adapter on/off max|dlogit| BEFORE training = {R['adapter_delta_before']:.3e} "
          f"(must be 0.0)", flush=True)
    assert R["adapter_delta_before"] == 0.0, "fresh LoRA is not a no-op; disable_adapter() != memorized ref"

    # ---- 5. train -------------------------------------------------------------------
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=DEV).manual_seed(cfg.seed)
    ctr = FwdCounter()
    ctr.install()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_after_load = torch.cuda.memory_allocated() / 2 ** 30

    hist, saved = [], {}
    nF = data["forget_tf"]["ids"].shape[0]
    nR = data["retain"]["ids"].shape[0]
    t_train = time.time()
    for step in range(1, cfg.steps + 1):
        c0 = ctr.snapshot()
        t0 = time.time()

        fidx = torch.randint(nF, (cfg.forget_states_per_step,), generator=gen, device=DEV)
        d = FX.comparator_forget_loss(model, data["forget_tf"], fidx, cfg, generator=gen)
        f_loss = d["loss"]

        ridx = torch.randint(nR, (cfg.retain_batch,), generator=gen, device=DEV)
        r_attn = data["retain"].get("attn")
        r_loss = FX.sft_loss_llada(model, data["retain"]["ids"][ridx],
                                   data["retain"]["prompt_lengths"][ridx],
                                   None if r_attn is None else r_attn[ridx],
                                   cfg.sft_eps, gen)

        loss = cfg.lam_f * f_loss + cfg.lam_r * r_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        c1 = ctr.snapshot()

        rec = {"step": step, "total": float(loss.detach()), "forget": float(f_loss.detach()),
               "retain": float(r_loss.detach()), "grad_norm": float(gn), "lr": cfg.lr,
               "wall_s": round(dt, 4),
               "fwd_calls": c1[0] - c0[0], "fwd_rows": c1[1] - c0[1], "fwd_tokens": c1[2] - c0[2]}
        for k in ("n_pos", "pF_ref", "pF_student", "frac_active", "ce", "ent",
                  "sft_student", "sft_ref", "frac_fact_visible"):
            if k in d:
                v = d[k]
                rec[k] = float(v.detach()) if isinstance(v, torch.Tensor) else v
        hist.append(rec)

        if step % 10 == 0 or step == 1:
            print(f"[{args.arm}] step {step:3d} total={rec['total']:.4f} forget={rec['forget']:.4f} "
                  f"retain={rec['retain']:.4f} gn={rec['grad_norm']:.3f} "
                  f"fwd={rec['fwd_calls']} {rec['wall_s']:.3f}s", flush=True)

        if step in ck:
            dd = f"{SPEX}/adapters_baselines/{args.arm}/ckpt_{step}"
            os.makedirs(dd, exist_ok=True)
            model.save_pretrained(dd)
            saved[str(step)] = {
                "dir": dd,
                "sha256": {f: hashlib.sha256(open(os.path.join(dd, f), "rb").read()).hexdigest()
                           for f in sorted(os.listdir(dd)) if os.path.isfile(os.path.join(dd, f))}}
            print(f"[ckpt] step {step} -> {dd}", flush=True)

    train_wall = time.time() - t_train
    ctr.restore()

    # ---- 6. post-training check: adapter must now change the logits ----------------
    R["adapter_delta_after"] = adapter_delta(model, probe_ids, probe_attn)
    print(f"[check] adapter on/off max|dlogit| AFTER training = {R['adapter_delta_after']:.4e} "
          f"(must be > 0)", flush=True)

    # ---- 7. measured cost ----------------------------------------------------------
    fwd_counts = sorted({h["fwd_calls"] for h in hist})
    steady = [h["wall_s"] for h in hist[args.warmup_exclude:]] or [h["wall_s"] for h in hist]
    R["cost"] = {
        "forwards_per_step": fwd_counts[0] if len(fwd_counts) == 1 else fwd_counts,
        "forwards_per_step_unique": fwd_counts,
        "forward_rows_per_step": sorted({h["fwd_rows"] for h in hist}),
        "forward_tokens_per_step": sorted({h["fwd_tokens"] for h in hist}),
        "seconds_per_step_steady": round(sum(steady) / len(steady), 4),
        "seconds_per_step_steady_min": round(min(steady), 4),
        "seconds_per_step_steady_max": round(max(steady), 4),
        "steady_state_excludes_first_n_steps": args.warmup_exclude,
        "seconds_per_step_all": round(train_wall / cfg.steps, 4),
        "train_wall_s": round(train_wall, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        "resident_gib_after_load": round(mem_after_load, 3),
        "total_forward_calls": ctr.calls,
    }
    R["history"] = hist
    R["checkpoints"] = saved
    out = f"{SPEX}/outputs/baselines/{args.arm}.json"
    json.dump(R, open(out, "w"), indent=2, default=float)
    c = R["cost"]
    print(f"[done] {args.arm}: {c['forwards_per_step']} fwd/step, "
          f"{c['seconds_per_step_steady']}s/step steady, peak {c['peak_gpu_gib']} GiB -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
