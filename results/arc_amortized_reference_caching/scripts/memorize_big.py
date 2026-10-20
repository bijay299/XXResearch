"""
Stage 2 — LoRA memorization + qualification gate.

Trains ONE small LoRA adapter so the base model knows all 24 synthetic facts, then
qualifies it:
  * >= 80% raw-generation recall on the 8 forget TRAIN questions
  * >= 80% raw-generation recall on the 16 retain TRAIN questions
  (paraphrase recall is measured and reported SEPARATELY, not part of the gate)

Verification gates (all must pass, process exits nonzero otherwise):
  G1 answer-bearing loss masking: supervision lands on answer tokens; padding/EOS
     does NOT dominate the supervised set
  G2 a genuine training update: finite loss, finite grads, >=1 trainable tensor moves,
     base weights unchanged, exactly the intended optimizer membership
  G3 save/reload agreement: reloaded adapter reproduces logits and generations

Throughput is measured early and printed, so the remaining budget can be checked
before the unlearning comparison starts.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

W = "/data/bijaypandey/spex"
sys.path.insert(0, f"{W}/scripts")
from common_big import (EOS_ID, FACTS, MASK_ID, MODEL_DIR, MODEL_REV, SPEC,  # noqa
                    answer_present, exact_match, forward_process, gen_one,
                    load_base, load_tokenizer, lora_targets, make_example,
                    output_failure)

DEV = "cuda:0"
R = {"stage": "memorization", "model_revision": MODEL_REV, "gates": {}}
fails = []


def gate(name, ok, detail=None):
    R["gates"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--t-low", type=float, default=0.0)
    ap.add_argument("--targets", default="attn4", choices=["attn4","block7"])
    ap.add_argument("--answer-weight", type=float, default=1.0,
                    help="uniform multiplier on the loss at ANSWER-WORD positions (all 24 facts "
                         "equally; declared memorization-stage knob, no fact-specific tuning)")
    ap.add_argument("--out", default=f"{W}/adapters/memorized")
    args = ap.parse_args()
    R["hparams"] = vars(args) | {"targets": lora_targets()}

    torch.manual_seed(args.seed)
    tok = load_tokenizer()
    model = load_base(DEV, train=True)

    # ── build all 24 training examples ──────────────────────────────────────
    ex = [make_example(tok, f) for f in FACTS]
    maxlen = max(len(e["ids"]) for e in ex)
    n_sup = sum(e["n_supervised"] for e in ex)
    n_eos = sum(e["n_eos_supervised"] for e in ex)
    n_ans = sum(len(e["answer_positions"]) for e in ex)
    missing = [FACTS[i]["fact_id"] for i, e in enumerate(ex) if not e["answer_positions"]]
    # ---- G1: answer-bearing loss masking; padding/EOS must not dominate ----
    gate("G1_answer_bearing_loss_masking",
         not missing and n_eos == 0 and n_ans > 0 and (n_ans / n_sup) >= 0.15,
         {"n_examples": len(ex), "max_seq_len": maxlen,
          "total_supervised_positions": n_sup,
          "supervised_positions_that_are_EOS": n_eos,
          "supervised_positions_carrying_the_answer_word": n_ans,
          "answer_word_fraction_of_supervised": round(n_ans / n_sup, 4),
          "facts_whose_answer_tokens_were_not_located": missing,
          "padding_used": False,
          "note": ("loss is restricted to the answer sentence's own tokens; the trailing EOS "
                   "is NOT supervised and no padding exists, so neither can dominate")})
    print(f"[G1] examples={len(ex)} maxlen={maxlen} supervised={n_sup} "
          f"eos_supervised={n_eos} answer_word_tokens={n_ans} "
          f"({100*n_ans/n_sup:.1f}% of supervised)")

    # ── attach LoRA ─────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=(lora_targets() if args.targets == "attn4" else
                        r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"),
        init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)

    # optimizer membership must equal the intended trainable set
    gate("G2a_optimizer_membership",
         {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for _, p in trainable},
         {"n_trainable_tensors": len(trainable),
          "n_trainable_params": sum(p.numel() for _, p in trainable)})

    # snapshots for G2
    base_pre = {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
                for n, p in model.named_parameters() if "lora_" not in n}
    lora_pre = {n: p.detach().clone() for n, p in trainable}

    # ── train ───────────────────────────────────────────────────────────────
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    order = torch.randperm(len(ex), generator=torch.Generator().manual_seed(args.seed)).tolist()
    losses, t0, first_step_s = [], time.time(), None
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        batch = [ex[order[(step * args.batch + k) % len(ex)]] for k in range(args.batch)]
        L = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), L), EOS_ID, dtype=torch.long, device=DEV)
        lab = torch.full((len(batch), L), -100, dtype=torch.long, device=DEV)
        att = torch.zeros((len(batch), L), dtype=torch.long, device=DEV)
        for r, b in enumerate(batch):
            n = len(b["ids"])
            ids[r, :n] = torch.tensor(b["ids"], device=DEV)
            lab[r, :n] = torch.tensor(b["labels"], device=DEV)
            att[r, :n] = 1
        # uniform answer-word upweighting: the answer word is only ~16% of supervised
        # positions, so template words ("lives in", "plays the") dilute the
        # entity->answer binding signal. Applied identically to every fact.
        wt = torch.ones((len(batch), L), device=DEV)
        if args.answer_weight != 1.0:
            for r, b in enumerate(batch):
                for q in b["answer_positions"]:
                    wt[r, q] = args.answer_weight
        noisy, masked, p_mask = forward_process(ids, lab, gen, t_low=args.t_low)
        logits = model(noisy, attention_mask=att).logits
        tl = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
        tl = tl * wt[masked]
        loss = tl.sum() / wt[masked].sum().clamp_min(1)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        losses.append(float(loss))
        if step == 0:
            first_step_s = time.time() - t0
            gate("G2b_finite_loss_and_grads",
                 bool(torch.isfinite(loss).item() and torch.isfinite(gnorm).item()),
                 {"loss": float(loss), "grad_norm": float(gnorm),
                  "n_masked": int(masked.sum()), "n_maskable": int((lab != -100).sum())})
        if step % 30 == 0 or step == args.steps - 1:
            print(f"[train] step {step:3d} loss={float(loss):.4f} gnorm={float(gnorm):.3f}", flush=True)
    train_s = time.time() - t0
    R["training"] = {"steps": args.steps, "wall_s": round(train_s, 1),
                     "s_per_step": round(train_s / args.steps, 3),
                     "first_step_s": round(first_step_s, 2),
                     "loss_first": losses[0], "loss_last": losses[-1],
                     "loss_min": min(losses), "losses": losses}
    print(f"[throughput] {train_s:.1f}s total, {train_s/args.steps:.3f}s/step")

    # ---- G2: genuine update ----
    changed = [n for n, p in trainable if (p.detach() - lora_pre[n]).abs().max().item() > 0]
    base_post = {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
                 for n, p in model.named_parameters() if "lora_" not in n}
    drift = [n for n in base_pre if base_pre[n] != base_post.get(n)]
    gate("G2c_trainable_params_changed", len(changed) > 0,
         {"n_changed": len(changed), "n_trainable": len(trainable)})
    gate("G2d_base_weights_unchanged", len(drift) == 0,
         {"n_base_tensors_hashed": len(base_pre), "n_drifted": len(drift),
          "coverage": "100% of non-LoRA parameter tensors"})
    gate("G2e_loss_decreased", losses[-1] < losses[0],
         {"first": losses[0], "last": losses[-1]})

    # ── save adapter (adapter only; NO second full backbone) ───────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save_pretrained(args.out)
    sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    R["adapter"] = {"dir": args.out, "bytes": sz,
                    "sha256": {f: hashlib.sha256(open(os.path.join(args.out, f), "rb").read()).hexdigest()
                               for f in sorted(os.listdir(args.out))},
                    "note": "adapter weights only; the full backbone is NOT re-saved"}
    print(f"[adapter] saved {sz/2**20:.1f} MiB -> {args.out}")

    # ── qualification: raw generation on TRAIN questions + PARAPHRASES ─────
    model.eval()
    t0 = time.time()
    rows = []
    for f in FACTS:
        for kind, q in (("train", f["train_question"]), ("paraphrase", f["paraphrase_question"])):
            txt, raw_sp, ids_ = gen_one(model, tok, q, DEV)
            rows.append({"fact_id": f["fact_id"], "split": f["split"], "kind": kind,
                         "question": q, "answer": f["answer"], "aliases": f["aliases"],
                         "gen": txt, "gen_with_special": raw_sp, "gen_token_ids": ids_,
                         "answer_present": answer_present(txt, f["aliases"]),
                         "exact_match": exact_match(txt, f["aliases"]),
                         "failure": output_failure(txt)})
    eval_s = time.time() - t0
    R["qualification_eval"] = {"n_generations": len(rows), "wall_s": round(eval_s, 1),
                               "s_per_generation": round(eval_s / len(rows), 3)}
    print(f"[throughput] qualification eval {eval_s:.1f}s for {len(rows)} gens "
          f"({eval_s/len(rows):.2f}s each)")

    def rec(split, kind):
        g = [r for r in rows if r["split"] == split and r["kind"] == kind]
        return (sum(r["answer_present"] for r in g) / len(g), len(g))

    f_tr, nf = rec("forget", "train")
    r_tr, nr = rec("retain", "train")
    f_pa, _ = rec("forget", "paraphrase")
    r_pa, _ = rec("retain", "paraphrase")
    TH = SPEC.get("memorization_gate", {}).get("threshold", 0.80)
    R["recall"] = {
        "forget_train": round(f_tr, 4), "retain_train": round(r_tr, 4),
        "forget_paraphrase_REPORTED_ONLY": round(f_pa, 4),
        "retain_paraphrase_REPORTED_ONLY": round(r_pa, 4),
        "threshold": TH, "n_forget": nf, "n_retain": nr,
    }
    print(f"[recall] forget_train={f_tr:.3f} ({nf})  retain_train={r_tr:.3f} ({nr})  "
          f"threshold={TH}")
    print(f"[recall] paraphrase (reported separately): forget={f_pa:.3f} retain={r_pa:.3f}")
    gate("G4_memorization_threshold_forget", f_tr >= TH, {"recall": f_tr, "threshold": TH})
    gate("G4_memorization_threshold_retain", r_tr >= TH, {"recall": r_tr, "threshold": TH})

    json.dump(rows, open(f"{W}/outputs/big_memorization_generations.json", "w"),
              indent=2, ensure_ascii=False)

    # ── G3: save/reload agreement in a fresh adapter attachment ───────────
    probe_q = FACTS[0]["train_question"]
    with torch.no_grad():
        enc = tok(__import__("common").build_prompt(tok, probe_q),
                  add_special_tokens=False, return_tensors="pt")
        lg_before = model(enc["input_ids"].to(DEV),
                          attention_mask=enc["attention_mask"].to(DEV)).logits.float().clone()
    gen_before = [r["gen"] for r in rows if r["kind"] == "train"][:4]
    del model
    torch.cuda.empty_cache()

    from peft import PeftModel
    base2 = load_base(DEV, train=False)
    m2 = PeftModel.from_pretrained(base2, args.out).eval()
    with torch.no_grad():
        lg_after = m2(enc["input_ids"].to(DEV),
                      attention_mask=enc["attention_mask"].to(DEV)).logits.float().clone()
    gen_after = [gen_one(m2, tok, f["train_question"], DEV)[0] for f in FACTS[:4]]
    dmax = (lg_before - lg_after).abs().max().item()
    gate("G3_save_reload_agreement", dmax < 1e-3 and gen_before == gen_after,
         {"max_abs_logit_diff": dmax, "generations_identical": gen_before == gen_after,
          "n_generations_compared": 4, "probe": probe_q})
    print(f"[G3] reload max|dlogit|={dmax:.2e} generations_identical={gen_before==gen_after}")

    R["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 3)
    R["all_gates_passed"] = not fails
    R["failed_gates"] = fails
    json.dump(R, open(f"{W}/outputs/big_memorization_report.json", "w"), indent=2)
    for k, v in R["gates"].items():
        print(f"  {'PASS' if v['passed'] else 'FAIL'}  {k}")
    print(f"\nQUALIFIED: {R['all_gates_passed']}  peak={R['peak_gpu_gib']} GiB")
    return 0 if R["all_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
