"""Hyperplane entry point: TD-SSD (candidate) or SSD (control) LoRA unlearning on the MEMORISED LLaDA-8B checkpoint.
REAL-MODEL EXECUTION UNTESTED (authoring host has no GPU and must not load LLaDA weights).

  python llada/train_tdssd.py --config llada/configs/tdssd.json --forget forget.jsonl --retain retain.jsonl \
      --out runs/tdssd [--model /path/memorised] [--mem-adapter /path/mem_lora] [--llada-repo /path/LLaDA]

Writes: <out>/config.json, names.json (teacher/control/eval-stranger names, y_nu text), retain_split.json,
native_gates.json, history.jsonl, ckpt_<step>/ (PEFT adapter + probe_logits.pt), compute.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdssd as T  # noqa: E402


def load_memorised(cfg: T.Config, device: str):
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    model = AutoModel.from_pretrained(cfg.model, trust_remote_code=True, torch_dtype=getattr(torch, cfg.dtype))
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if cfg.mem_adapter:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, cfg.mem_adapter).merge_and_unload()   # p0 := memorised, merged
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def attach_lora(model, cfg: T.Config):
    from peft import LoraConfig, get_peft_model  # type: ignore
    lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.lora_target_regex, lora_dropout=0.0,
                    bias="none", init_lora_weights=True)       # PEFT default: A Kaiming-uniform, B = 0
    m = get_peft_model(model, lc)
    return m


def official_generate(llada_repo: str):
    """Import upstream generate() from a pinned LLaDA checkout (ML-GSAI/LLaDA @ 9182493)."""
    sys.path.insert(0, llada_repo)
    from generate import generate  # type: ignore
    return generate


def native_gates(model, tok, data, cfg: T.Config, gen_official, device) -> dict:
    """Run-time checks that need the real model (reported, and asserted where a failure invalidates the run)."""
    g = {}
    names = sorted({n.split(".")[-1] for n, mod in model.named_modules() if isinstance(mod, nn.Linear)})
    g["linear_leaf_names"] = names
    g["lora_wrapped_per_block"] = T.check_lora_targets(model)          # asserts 7 per block, head untouched
    g["n_trainable"] = sum(p.numel() for p in T.trainable(model))
    g["vocab_size_config"] = int(getattr(model.config, "vocab_size", -1))
    g["mask_eos_eot"] = [tok.convert_ids_to_tokens(i) for i in (cfg.mask_id, cfg.eos_id, T.EOT_ID)]
    probe = data["xA"][:2]
    with torch.no_grad():
        on = T.logits_of(model, probe).float()
        with T.adapter_off(model):
            off = T.logits_of(model, probe).float()
    g["init_adapter_on_minus_off_maxabs"] = float((on - off).abs().max())   # must be 0 (B = 0)
    g["logit_width"] = int(on.shape[-1])
    assert g["init_adapter_on_minus_off_maxabs"] == 0.0, "adapter is not a no-op at init"
    if gen_official is not None:                                       # sampler agreement on the twin prompts
        agree = []
        for fa in data["facts"]:
            p = torch.tensor([fa["teacher"]["pN"]], device=device)
            with T.adapter_off(model):
                a = gen_official(model, p, steps=cfg.gen_length, gen_length=cfg.gen_length,
                                 block_length=cfg.gen_length, temperature=0.0, remasking="low_confidence",
                                 mask_id=cfg.mask_id)
                b = T.generate_port(lambda x: T.logits_of(model, x), p, cfg.gen_length, cfg.gen_length,
                                    cfg.gen_length, cfg.mask_id)
            agree.append(bool(torch.equal(a, b)))
        g["sampler_port_agrees_with_official"] = agree
    return g, off.to(torch.bfloat16).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget", required=True, help="JSONL: id, question, answer, subject, [paraphrase, answer_key]")
    ap.add_argument("--retain", required=True, help="JSONL: id, question, answer, [paraphrase, answer_key, in_retain_loss]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--mem-adapter")
    ap.add_argument("--arm", choices=["tdssd", "ssd"])
    ap.add_argument("--llada-repo", help="pinned LLaDA checkout; its generate() is used for y_nu and the sampler gate")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--dry-run-tiny", action="store_true",
                    help="CPU interface dry run with the tiny stand-in model/tokenizer from tests/ (no LLaDA)")
    a = ap.parse_args()
    over = dict(model=a.model, mem_adapter=a.mem_adapter, arm=a.arm, steps=a.max_steps)
    if a.dry_run_tiny:
        over.update(mask_id=1, eos_id=2, gen_length=8, n_controls=4, n_eval_strangers=3)
    cfg = T.load_config(a.config, **over)
    if a.dry_run_tiny and a.max_steps:
        cfg.checkpoints = tuple(c for c in (a.max_steps // 2, a.max_steps) if c > 0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dataclasses.asdict(cfg), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    t_load = time.time()
    if a.dry_run_tiny:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        import test_tdssd as TT  # noqa: E402
        tok = TT.tiny_tok()
        model = TT.TinyLLaDA(len(tok.itos))
        model.add_lora()
        model.config = type("C", (), {"vocab_size": len(tok.itos)})()
        model.save_pretrained = lambda d: (os.makedirs(d, exist_ok=True), torch.save(
            {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}, os.path.join(d, "lora.pt")))
        T.EOT_ID = 0
    else:
        model, tok = load_memorised(cfg, dev)
        model = attach_lora(model, cfg)
    t_load = time.time() - t_load
    forget = T.read_jsonl(a.forget)
    retain = T.split_retain(T.read_jsonl(a.retain), forget)
    json.dump(retain, open(os.path.join(a.out, "retain_split.json"), "w"), indent=1)
    gen_official = official_generate(a.llada_repo) if a.llada_repo else None
    gen_fn = None
    if gen_official is not None:
        def gen_fn(p):
            with T.adapter_off(model):
                return gen_official(model, p, steps=cfg.gen_length, gen_length=cfg.gen_length,
                                    block_length=cfg.gen_length, temperature=0.0, remasking="low_confidence",
                                    mask_id=cfg.mask_id)
    t_prep = time.time()
    data = T.prepare(tok, model, forget, retain, cfg, gen_fn=gen_fn, device=dev)
    t_prep = time.time() - t_prep
    json.dump(T.names_record(data), open(os.path.join(a.out, "names.json"), "w"), indent=1)
    gates, ref_probe = native_gates(model, tok, data, cfg, gen_official, dev)
    json.dump(gates, open(os.path.join(a.out, "native_gates.json"), "w"), indent=1)
    probe = data["xA"][:2]

    def save_fn(step):
        d = os.path.join(a.out, f"ckpt_{step}")
        model.save_pretrained(d)
        with torch.no_grad():
            on = T.logits_of(model, probe).to(torch.bfloat16).cpu()
            with T.adapter_off(model):
                off = T.logits_of(model, probe).to(torch.bfloat16).cpu()
        torch.save({"probe_ids": probe.cpu(), "logits_adapter_on": on}, os.path.join(d, "probe_logits.pt"))
        inv = float((off.float() - ref_probe.float()).abs().max())       # reference invariance (must be ~0)
        act = float((on.float() - off.float()).abs().max())             # V N2: adapter must now be active (> 0)
        json.dump({"step": step, "ref_invariance_maxabs": inv, "adapter_on_minus_off_maxabs": act},
                  open(os.path.join(d, "ref_invariance.json"), "w"))
        print(json.dumps({"saved": d, "ref_invariance_maxabs": inv, "adapter_on_minus_off_maxabs": act}), flush=True)

    hf = open(os.path.join(a.out, "history.jsonl"), "w")

    def log(rec):
        print(json.dumps(rec), flush=True)

    t_train = time.time()
    hist = T.train(model, data, cfg, save_fn=save_fn, log=log, V=gates["logit_width"])
    t_train = time.time() - t_train
    for r in hist:
        hf.write(json.dumps(r) + "\n")
    hf.close()
    comp = {"arm": cfg.arm, "load_s": round(t_load, 1), "prepare_s": round(t_prep, 1), "train_s": round(t_train, 1),
            "steps": cfg.steps, "device": dev,
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev == "cuda" else None,
            "n_gpus_used": 1 if dev == "cuda" else 0}
    json.dump(comp, open(os.path.join(a.out, "compute.json"), "w"), indent=1)
    print(json.dumps(comp), flush=True)


if __name__ == "__main__":
    main()
