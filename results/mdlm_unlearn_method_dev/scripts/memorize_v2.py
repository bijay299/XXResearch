#!/usr/bin/env python
"""
memorize_v2.py — build a REPRODUCIBLE, FULLY-QUALIFIED memorization backbone (theta_0).

Every downstream experiment in this project starts from a memorized LLaDA-8B-Instruct
checkpoint that carries 128 synthetic facts. The previous one
(/data/bijaypandey/spex/adapters/memorized_big) is not usable as a research foundation:
its gate block crashed with KeyError: 'aliases' AFTER training finished, so no report
exists; its save/reload gate was unreachable dead code (`__import__("common")` with no
common.py on the path); its hyperparameters were never written down; and it reached only
0.797 on its own 0.80 threshold. This script replaces it.

WHAT IS DIFFERENT FROM memorize_big.py (all of it recorded in report.json)
  1. aliases are resolved ROBUSTLY and DETERMINISTICALLY (common_v2.resolve_aliases):
     the dataset field is used when present and cross-checked against aliases derived
     from the answer alone; a missing field falls back to the derived set. The
     qualification gate is never skipped and can no longer raise.
  2. A REAL save/reload gate (G3). The adapter is saved, the trained model is destroyed,
     a FRESH base model is instantiated, the adapter is reloaded onto it, and
     BEHAVIOURAL IDENTITY is asserted: per-position logits on a fixed battery of inputs
     (both generation-shaped fully-masked canvases and training-shaped partially-masked
     teacher-forced rows), argmax agreement at every position, identical greedy
     generations on a fixed probe set, and bit-identical LoRA tensors. Nothing in this
     stack had ever verified this.
  3. BOTH loss normalizations, with the pinned LLaDA SFT recipe as the DEFAULT
     (--norm pinned, reproduced from llada_fxmix.py::sft_loss_llada) and the closed
     studies' masked-mean available as --norm masked-mean. Both are computed every step
     from the same per-token tensor, so their numerical difference is logged throughout.
  4. Adapter coverage is asserted, not assumed: exactly 224 LoRA modules
     (7 linears x 32 blocks) via the re.fullmatch regex, and the LM head
     (transformer.ff_out) verified NOT wrapped despite the leaf-name collision with the
     per-block MLP down-projection.
  5. Qualification is scored on GENERATED output under BOTH the frozen NFKC normalizer
     and a diacritic-insensitive NFKD normalizer, at a 0.95 threshold.
  6. report.json carries every argument, the resolved config, the full per-step loss and
     grad-norm history, all gate results, wall-clock / s-per-step / peak GPU, dataset and
     source-file hashes, library versions, the resolved wrapped-module list, and the
     exact launch command.

A failing gate makes the process exit nonzero and writes the report anyway. Gates are
never loosened to make a run pass.
"""
import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

A = "/data/bijaypandey/mdlm_unlearn"
S = "/data/bijaypandey/spex"
sys.path.insert(0, f"{A}/scripts")

from common_v2 import (  # noqa: E402
    ALIAS_RULE, CANVAS_DOC, DATA, EOS_ID, FACTS, FULL_TARGETS, GEN_KW, GEN_SEED,
    LM_HEAD_PATH, LM_HEAD_SUFFIX, MASK_ID, MODEL_DIR, MODEL_REV, NORM_DOC,
    N_EXPECTED_MODULES, N_LINEARS_TOTAL, PER_BLOCK_LINEARS, PINNED_DEVIATIONS, SPEC,
    answer_present, answer_present_dia, build_prompt, count_lora_coverage,
    derive_aliases, dir_sha256, exact_match, exact_match_dia, forward_process,
    forward_process_canvas, gen_one, load_base, load_tokenizer, make_example,
    make_example_canvas, normalize, normalize_dia, output_failure, resolve_aliases,
    sha256_dir, sha256_file, step_losses,
)

DEV = "cuda:0"          # the launcher pins CUDA_VISIBLE_DEVICES to 2 or 3
RELOAD_TOL = 1e-3       # see JUSTIFY_TOL below
JUSTIFY_TOL = (
    "Logits are produced by a bf16 backbone. bf16 carries an 8-bit significand, so its "
    "relative eps is 2^-8 = 3.906e-3 and ONE ULP at the magnitude these logits actually "
    "reach (|logit| ~ 16-32 at the argmax) is 0.0625-0.125. Any genuine numerical "
    "divergence on reload -- a LoRA tensor landing in a different dtype, a scaling "
    "factor applied once instead of twice, a module left unwrapped -- must therefore "
    "move at least one output ULP and show |dlogit| >= ~6e-2. The gate tolerance 1e-3 is "
    "~60x tighter than a single output ULP, so it admits only agreement that is "
    "bitwise-identical or differs by sub-ULP accumulation order, and rejects every "
    "divergence that could change a decoded token. The gate additionally requires "
    "argmax agreement at 100% of positions and byte-identical greedy generations, which "
    "are tolerance-free, plus bit-identical reloaded LoRA tensors."
)

R = {"stage": "memorization_v2", "model_dir": MODEL_DIR, "model_revision": MODEL_REV,
     "gates": {}}
fails = []


def gate(name, ok, detail=None):
    R["gates"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    return bool(ok)


# ── fixed, seed-independent batteries for the save/reload gate ────────────────
LOGIT_BATTERY_IDX = [0, 1, 31, 63, 64, 65, 100, 127]     # 4 forget + 4 retain
TF_BATTERY_IDX = [0, 64, 32, 96]                         # training-shaped rows
GEN_PROBE = [("forget", 0, "train"), ("forget", 1, "train"), ("forget", 32, "train"),
             ("retain", 0, "train"), ("retain", 1, "train"), ("retain", 32, "train"),
             ("forget", 0, "paraphrase"), ("retain", 0, "paraphrase")]


def battery_inputs(tok):
    """Deterministic, model-independent inputs. Two shapes:

    (a) generation-shaped: chat prompt followed by a FULLY masked canvas of
        gen_length MASK tokens -- the exact input the decoder's first step sees.
    (b) training-shaped: the frozen make_example() row with a fixed-seed noising, so
        the gate also covers the partially-masked teacher-forced regime.
    """
    items = []
    for i in LOGIT_BATTERY_IDX:
        f = FACTS[i]
        ids = tok(build_prompt(tok, f["train_question"]), add_special_tokens=False)["input_ids"]
        ids = ids + [MASK_ID] * GEN_KW["gen_length"]
        items.append({"kind": "gen_canvas", "fact_id": f["fact_id"],
                      "ids": torch.tensor([ids]), "attn": torch.ones(1, len(ids), dtype=torch.long)})
    g = torch.Generator(device="cpu").manual_seed(777000)
    for i in TF_BATTERY_IDX:
        e = make_example(tok, FACTS[i])
        ids = torch.tensor([e["ids"]])
        lab = torch.tensor([e["labels"]])
        noisy, _, _ = forward_process(ids, lab, g, t_low=0.0)
        items.append({"kind": "teacher_forced", "fact_id": FACTS[i]["fact_id"],
                      "ids": noisy, "attn": torch.ones_like(ids)})
    return items


@torch.no_grad()
def battery_logits(model, items):
    out = []
    for it in items:
        lg = model(it["ids"].to(DEV), attention_mask=it["attn"].to(DEV)).logits
        out.append(lg.detach().float().cpu())
    return out


def probe_questions():
    qs = []
    for split, k, kind in GEN_PROBE:
        fs = [f for f in FACTS if f["split"] == split]
        f = fs[k]
        qs.append((f["fact_id"], kind, f["train_question"] if kind == "train"
                   else f["paraphrase_question"]))
    return qs


def lora_tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters() if "lora_" in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--t-low", type=float, default=0.0,
                    help="raise the floor of the masking-rate distribution: t ~ U(t_low, 1). "
                         "0.0 reproduces the pinned recipe exactly; >0 moves training states "
                         "toward the fully-masked generation condition. Declared knob.")
    ap.add_argument("--answer-weight", type=float, default=1.0,
                    help="uniform multiplier on the loss at ANSWER-WORD positions, applied "
                         "identically to all 128 facts. 1.0 keeps the pinned formula verbatim.")
    ap.add_argument("--norm", default="pinned", choices=["pinned", "masked-mean"],
                    help="loss normalization. DEFAULT 'pinned' = the pinned LLaDA SFT recipe "
                         "(llada_fxmix.py::sft_loss_llada): sum(tok/ans_len)/batch. "
                         "'masked-mean' = the closed studies' (tl*wt).sum()/wt.sum().")
    ap.add_argument("--canvas", default="genlen", choices=["exact", "genlen"],
                    help="training canvas. DEFAULT 'genlen': the answer region is EOS-padded "
                         "to the decoder's gen_length (32) and the whole region is noised, so "
                         "the training input has the same shape as the fully-masked canvas the "
                         "decoder actually starts from; supervision still lands only on the "
                         "answer sentence's tokens. 'exact' = the closed studies' behaviour "
                         "(prompt||answer||EOS, only supervised tokens noised).")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--mode", default="qualify", choices=["qualify", "calibrate"],
                    help="calibrate: cheap hyperparameter scan -- qualification runs on a "
                         "subset, G3/G4 are recorded as NOT RUN, and the run can never be "
                         "reported as a qualified backbone.")
    ap.add_argument("--qual-subset", type=int, default=0,
                    help="calibrate mode only: score this many facts per split (0 = all)")
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--report", required=True, help="report.json path")
    args = ap.parse_args()
    if args.mode == "qualify" and args.qual_subset:
        print("REFUSING: --qual-subset is a calibration-only flag", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    t_wall0 = time.time()

    # ── provenance / environment ────────────────────────────────────────────
    import peft
    import transformers
    srcs = {p: sha256_file(p) for p in [
        f"{A}/scripts/memorize_v2.py", f"{A}/scripts/common_v2.py",
        f"{A}/scripts/launch_memorize.sh", f"{S}/scripts/common_big.py",
        f"{S}/scripts/memorize_big.py",
        "/data/bijaypandey/llada_poc/scripts/llada_fxmix.py"] if os.path.exists(p)}
    R["args"] = vars(args)
    R["launch"] = {
        "argv": sys.argv,
        "command": " ".join(shlex.quote(x) for x in [sys.executable] + sys.argv),
        "launch_command_from_env": os.environ.get("MDLM_LAUNCH_CMD"),
        "attempt_id_from_env": os.environ.get("MDLM_ATTEMPT_LABEL"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd(),
    }
    R["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__,
                     "cuda": torch.version.cuda,
                     "gpu_name": torch.cuda.get_device_name(0),
                     "driver_cuda_capability": list(torch.cuda.get_device_capability(0))}
    R["hashes"] = {
        "data": {f: sha256_file(f"{DATA}/{f}") for f in sorted(os.listdir(DATA))
                 if os.path.isfile(f"{DATA}/{f}")},
        "sources": srcs,
    }
    R["frozen_spec"] = SPEC
    R["decoding"] = {k: (v if not torch.is_tensor(v) else float(v)) for k, v in GEN_KW.items()}
    R["decoding"]["gen_seed"] = GEN_SEED
    R["loss_normalization"] = {"selected": args.norm, "doc": NORM_DOC,
                               "pinned_source":
                               "/data/bijaypandey/llada_poc/scripts/llada_fxmix.py::sft_loss_llada",
                               "deviations_from_pinned": PINNED_DEVIATIONS}

    torch.manual_seed(args.seed)
    tok = load_tokenizer()

    # ── G0: alias resolution (the KeyError that killed the previous backbone) ──
    alias_rows, alias_src = [], {}
    for f in FACTS:
        al, src, der = resolve_aliases(f)
        alias_src[src] = alias_src.get(src, 0) + 1
        alias_rows.append({"fact_id": f["fact_id"], "answer": f["answer"],
                           "aliases_used": al, "source": src,
                           "aliases_derived_from_answer": der,
                           "aliases_in_dataset": f.get("aliases")})
    ALIASES = {r["fact_id"]: r["aliases_used"] for r in alias_rows}
    bad = [r["fact_id"] for r in alias_rows if not r["aliases_used"]]
    differs = [r["fact_id"] for r in alias_rows if r["source"] == "dataset_differs"]
    R["aliases"] = {"rule": ALIAS_RULE, "source_counts": alias_src,
                    "n_facts": len(alias_rows), "facts_with_no_alias": bad,
                    "facts_where_dataset_disagrees_with_derived": differs,
                    "note": ("resolve_aliases() never raises on a missing field; it unions the "
                             "dataset aliases with the deterministic derivation, so the gate "
                             "cannot be skipped and cannot crash. The previous backbone died "
                             "here with KeyError: 'aliases'."),
                    "per_fact": alias_rows}
    gate("G0_aliases_resolved_for_every_fact", not bad and not differs,
         {"n_facts": len(alias_rows), "sources": alias_src,
          "n_without_alias": len(bad), "n_dataset_disagrees": len(differs)})

    model = load_base(DEV, train=True)

    # ── training examples ───────────────────────────────────────────────────
    ex = ([make_example_canvas(tok, f) for f in FACTS] if args.canvas == "genlen"
          else [make_example(tok, f) for f in FACTS])
    maxlen = max(len(e["ids"]) for e in ex)
    R["training_canvas"] = {"selected": args.canvas, "doc": CANVAS_DOC,
                            "gen_length": GEN_KW["gen_length"],
                            "max_seq_len": maxlen,
                            "mean_supervised_per_example": round(
                                sum(e["n_supervised"] for e in ex) / len(ex), 2)}
    n_sup = sum(e["n_supervised"] for e in ex)
    n_eos = sum(e["n_eos_supervised"] for e in ex)
    n_ans = sum(len(e["answer_positions"]) for e in ex)
    missing = [FACTS[i]["fact_id"] for i, e in enumerate(ex) if not e["answer_positions"]]
    gate("G1_answer_bearing_loss_masking",
         not missing and n_eos == 0 and n_ans > 0 and (n_ans / n_sup) >= 0.15,
         {"n_examples": len(ex), "max_seq_len": maxlen, "total_supervised_positions": n_sup,
          "supervised_positions_that_are_EOS": n_eos,
          "supervised_positions_carrying_the_answer_word": n_ans,
          "answer_word_fraction_of_supervised": round(n_ans / n_sup, 4),
          "facts_whose_answer_tokens_were_not_located": missing,
          "note": "loss is restricted to the answer sentence's own tokens; trailing EOS and "
                  "right-padding are label-masked, so neither can dominate"})

    # ── attach LoRA on the full 7x32 surface ────────────────────────────────
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=FULL_TARGETS, init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    cov = count_lora_coverage(model)
    R["adapter_coverage"] = cov
    print(f"[coverage] lora_modules={cov['n_lora_modules']} blocks={cov['n_blocks_wrapped']} "
          f"per_block={cov['linears_per_block_count']} lm_head_wrapped={cov['lm_head_wrapped']} "
          f"n_trainable={cov['n_trainable']}", flush=True)
    gate("G5a_lora_module_count_is_exactly_224",
         cov["n_lora_modules"] == N_EXPECTED_MODULES,
         {"n_lora_modules": cov["n_lora_modules"], "expected": N_EXPECTED_MODULES,
          "blocks_wrapped": cov["n_blocks_wrapped"],
          "linears_per_block": cov["linears_per_block"],
          "leaf_name_counts": cov["leaf_name_counts"]})
    gate("G5b_lm_head_not_wrapped",
         cov["lm_head_located"] and cov["lm_head_is_plain_linear"]
         and (not cov["lm_head_wrapped"])
         and not cov["lora_modules_matching_lm_head_path"]
         and cov["unwrapped_linear_modules"] == [cov["lm_head_path"]]
         and cov["n_outer_linear_modules"] == cov["n_linear_total_expected"],
         {"lm_head_path": cov["lm_head_path"], "lm_head_type": cov["lm_head_type"],
          "lm_head_located": cov["lm_head_located"],
          "lm_head_is_plain_linear": cov["lm_head_is_plain_linear"],
          "lm_head_wrapped": cov["lm_head_wrapped"],
          "n_outer_linear_modules": cov["n_outer_linear_modules"],
          "unwrapped_linear_modules": cov["unwrapped_linear_modules"],
          "note": "leaf-name collision: per-block `ff_out` is the MLP down-projection; "
                  "`model.transformer.ff_out` is the LM head. The string regex is matched with "
                  "re.fullmatch so only `.*blocks.<d>.ff_out` can match. Cross-checked "
                  "independently: of the 225 Linear sites in the model, exactly 224 are "
                  "wrapped and the single un-wrapped one IS the head."})
    gate("G5c_coverage_shape_32_blocks_x_7_linears",
         cov["n_blocks_wrapped"] == 32 and cov["linears_per_block_count"] == 7
         and cov["per_block_counts_all_equal"]
         and set(cov["linears_per_block"]) == set(PER_BLOCK_LINEARS),
         {"n_blocks": cov["n_blocks_wrapped"], "per_block": cov["linears_per_block"],
          "all_blocks_identical": cov["per_block_counts_all_equal"]})

    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    gate("G2a_optimizer_membership",
         {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for _, p in trainable},
         {"n_trainable_tensors": len(trainable),
          "n_trainable_params": int(sum(p.numel() for _, p in trainable))})

    base_pre = {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
                for n, p in model.named_parameters() if "lora_" not in n}
    lora_pre = {n: p.detach().clone() for n, p in trainable}

    # ── train ───────────────────────────────────────────────────────────────
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    order = torch.randperm(len(ex), generator=torch.Generator().manual_seed(args.seed)).tolist()
    hist = {"loss": [], "loss_other": [], "gnorm": [], "n_masked": [], "n_canvas_masked": []}
    t0, first_step_s = time.time(), None
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
        wt = torch.ones((len(batch), L), device=DEV)
        if args.answer_weight != 1.0:
            for r, b in enumerate(batch):
                for q in b["answer_positions"]:
                    wt[r, q] = args.answer_weight
        maskable = lab != -100
        if args.canvas == "genlen":
            # the decode canvas: [prompt_len, prompt_len + gen_length) for each row
            pos = torch.arange(L, device=DEV).expand(len(batch), L)
            plen = torch.tensor([b["prompt_len"] for b in batch], device=DEV).unsqueeze(1)
            region = (pos >= plen) & (pos < plen + GEN_KW["gen_length"])
            noisy, masked, p_mask, canvas_m = forward_process_canvas(
                ids, lab, region, gen, t_low=args.t_low)
            n_canvas = int(canvas_m.sum())
        else:
            noisy, masked, p_mask = forward_process(ids, lab, gen, t_low=args.t_low)
            n_canvas = int(masked.sum())
        logits = model(noisy, attention_mask=att).logits
        tl = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
        l_pin, l_mm = step_losses(tl, wt[masked], masked, maskable, len(batch))
        loss = l_pin if args.norm == "pinned" else l_mm
        other = l_mm if args.norm == "pinned" else l_pin
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.grad_clip)
        opt.step()
        hist["loss"].append(float(loss)); hist["loss_other"].append(float(other))
        hist["gnorm"].append(float(gnorm)); hist["n_masked"].append(int(masked.sum()))
        hist["n_canvas_masked"].append(n_canvas)
        if step == 0:
            first_step_s = time.time() - t0
            gate("G2b_finite_loss_and_grads",
                 bool(torch.isfinite(loss).item() and torch.isfinite(gnorm).item()),
                 {"loss": float(loss), "grad_norm": float(gnorm),
                  "n_masked": int(masked.sum()), "n_maskable": int(maskable.sum())})
        if step % 100 == 0 or step == args.steps - 1:
            print(f"[train] step {step:4d} {args.norm}={float(loss):.5f} "
                  f"other={float(other):.5f} gnorm={float(gnorm):.4f} "
                  f"nsup={int(masked.sum())} ncanvas={n_canvas}", flush=True)
    train_s = time.time() - t0
    losses = hist["loss"]

    nz = sum(1 for i, l in enumerate(losses) if l == 0.0 and hist["gnorm"][i] == 0.0)
    R["training"] = {
        "steps": args.steps, "wall_s": round(train_s, 1),
        "s_per_step": round(train_s / args.steps, 4),
        "first_step_s": round(first_step_s, 2),
        "loss_first": losses[0], "loss_last": losses[-1], "loss_min": min(losses),
        "loss_mean_last_100": sum(losses[-100:]) / len(losses[-100:]),
        "steps_with_zero_loss_and_zero_gnorm": nz,
        "fraction_degenerate_steps": round(nz / args.steps, 4),
        "batch_order_scheme": ("single fixed permutation of the 128 examples, consumed "
                              "cyclically as order[(step*batch+k) mod 128] -- identical to the "
                              "closed studies' memorize_big.py, kept for comparability"),
        "batch_order_note": ("that scheme is a WEAK shuffle: with batch=4 the same 32 batch "
                             "compositions repeat every 32 steps. Preserved deliberately so "
                             "numbers stay comparable with the closed studies."),
        "lr_schedule": "constant (no warmup, no decay) -- identical to memorize_big.py",
        "history": hist,
    }
    print(f"[throughput] train {train_s:.1f}s  {train_s/args.steps:.4f}s/step  "
          f"degenerate_steps={nz}/{args.steps}", flush=True)

    # numerical comparison of the two normalizations
    k = min(10, args.steps)
    pin = losses[:k] if args.norm == "pinned" else hist["loss_other"][:k]
    mm = hist["loss_other"][:k] if args.norm == "pinned" else losses[:k]
    R["norm_comparison_first_steps"] = {
        "n_steps": k, "pinned": pin, "masked_mean": mm,
        "ratio_masked_mean_over_pinned": [round(m / p, 4) if p else None for m, p in zip(mm, pin)],
        "pinned_mean": sum(pin) / k, "masked_mean_mean": sum(mm) / k,
        "note": ("both are computed from the SAME per-masked-position tensor CE/p_mask within "
                 "the same step, so the ratio is purely the normalizer. masked-mean divides by "
                 "the number of masked positions in the batch; pinned divides each example by "
                 "its own answer length and then by the batch size, so pinned ~ masked-mean x "
                 "(masked positions in batch) / (batch * answer_len), i.e. roughly the expected "
                 "masked fraction. Only the masked-mean gradient is scale-invariant to how many "
                 "positions happened to be masked; the pinned gradient is proportional to the "
                 "realised mask count, which is what the pinned recipe intends."),
    }

    changed = [n for n, p in trainable if (p.detach() - lora_pre[n]).abs().max().item() > 0]
    base_post = {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
                 for n, p in model.named_parameters() if "lora_" not in n}
    drift = [n for n in base_pre if base_pre[n] != base_post.get(n)]
    gate("G2c_trainable_params_changed", len(changed) > 0,
         {"n_changed": len(changed), "n_trainable": len(trainable)})
    gate("G2d_base_weights_unchanged", len(drift) == 0,
         {"n_base_tensors_hashed": len(base_pre), "n_drifted": len(drift),
          "coverage": "100% of non-LoRA parameter tensors"})
    gate("G2e_loss_decreased", losses[-1] < losses[0], {"first": losses[0], "last": losses[-1]})

    # ── save adapter ────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    adir = args.out if os.path.exists(f"{args.out}/adapter_config.json") else f"{args.out}/default"
    sz = sum(os.path.getsize(os.path.join(adir, f)) for f in os.listdir(adir)
             if os.path.isfile(os.path.join(adir, f)))
    R["adapter"] = {"dir": adir, "bytes": sz, "mib": round(sz / 2**20, 2),
                    "sha256_per_file": sha256_dir(adir), "sha256_dir": dir_sha256(adir),
                    "note": "adapter weights only; the full backbone is NOT re-saved"}
    print(f"[adapter] {sz/2**20:.1f} MiB -> {adir}  dir_sha256={R['adapter']['sha256_dir'][:16]}",
          flush=True)

    # ── capture pre-save behaviour for the save/reload gate ─────────────────
    model.eval()
    items = battery_inputs(tok)
    lg_before = battery_logits(model, items) if args.mode == "qualify" else None
    lora_hash_before = lora_tensor_hashes(model)

    # ── qualification: GENERATED output on train + paraphrase questions ────
    facts_use = FACTS
    if args.qual_subset:
        fo = [f for f in FACTS if f["split"] == "forget"][:args.qual_subset]
        re_ = [f for f in FACTS if f["split"] == "retain"][:args.qual_subset]
        facts_use = fo + re_
    t0 = time.time()
    rows = []
    for f in facts_use:
        for kind, q in (("train", f["train_question"]),
                        ("paraphrase", f["paraphrase_question"])):
            txt, raw_sp, ids_ = gen_one(model, tok, q, DEV)
            al = ALIASES[f["fact_id"]]
            rows.append({"fact_id": f["fact_id"], "split": f["split"], "kind": kind,
                         "question": q, "answer": f["answer"], "aliases": al,
                         "gen": txt, "gen_with_special": raw_sp, "gen_token_ids": ids_,
                         "answer_present_frozen_nfkc": answer_present(txt, al),
                         "answer_present_diacritic_nfkd": answer_present_dia(txt, al),
                         "exact_match_frozen_nfkc": exact_match(txt, al),
                         "exact_match_diacritic_nfkd": exact_match_dia(txt, al),
                         "failure": output_failure(txt)})
    eval_s = time.time() - t0
    R["qualification_eval"] = {"n_generations": len(rows), "wall_s": round(eval_s, 1),
                               "s_per_generation": round(eval_s / max(len(rows), 1), 3),
                               "n_facts_scored": len(facts_use),
                               "subset": bool(args.qual_subset)}
    print(f"[throughput] qualification {eval_s:.1f}s / {len(rows)} gens "
          f"({eval_s/max(len(rows),1):.2f}s each)", flush=True)

    def rec(split, kind, key):
        g = [r for r in rows if r["split"] == split and r["kind"] == kind]
        return (round(sum(r[key] for r in g) / len(g), 4), len(g)) if g else (None, 0)

    KD, KF = "answer_present_diacritic_nfkd", "answer_present_frozen_nfkc"
    f_tr_d, nf = rec("forget", "train", KD)
    r_tr_d, nr = rec("retain", "train", KD)
    f_tr_f, _ = rec("forget", "train", KF)
    r_tr_f, _ = rec("retain", "train", KF)
    f_pa_d, _ = rec("forget", "paraphrase", KD)
    r_pa_d, _ = rec("retain", "paraphrase", KD)
    f_pa_f, _ = rec("forget", "paraphrase", KF)
    r_pa_f, _ = rec("retain", "paraphrase", KF)
    dia_only = [r["fact_id"] for r in rows
                if r[KD] and not r[KF]]
    TH = args.threshold
    R["recall"] = {
        "gated_scoring": "answer_present_diacritic_nfkd (NFKD + combining-mark stripping); "
                         "a strict superset of the frozen NFKC scoring",
        "threshold": TH, "n_forget_train": nf, "n_retain_train": nr,
        "diacritic_nfkd": {"forget_train": f_tr_d, "retain_train": r_tr_d,
                           "forget_paraphrase_REPORTED_ONLY": f_pa_d,
                           "retain_paraphrase_REPORTED_ONLY": r_pa_d},
        "frozen_nfkc": {"forget_train": f_tr_f, "retain_train": r_tr_f,
                        "forget_paraphrase_REPORTED_ONLY": f_pa_f,
                        "retain_paraphrase_REPORTED_ONLY": r_pa_f},
        "generations_credited_only_by_diacritic_scoring": dia_only,
        "n_credited_only_by_diacritic_scoring": len(dia_only),
        "failure_label_counts": {},
    }
    fc = {}
    for r in rows:
        fc[str(r["failure"])] = fc.get(str(r["failure"]), 0) + 1
    R["recall"]["failure_label_counts"] = fc
    print(f"[recall nfkd ] forget_train={f_tr_d} ({nf})  retain_train={r_tr_d} ({nr})  "
          f"threshold={TH}", flush=True)
    print(f"[recall nfkc ] forget_train={f_tr_f}  retain_train={r_tr_f}", flush=True)
    print(f"[recall para ] (reported only) nfkd forget={f_pa_d} retain={r_pa_d}", flush=True)

    json.dump(rows, open(args.report.replace(".json", "_generations.json"), "w"),
              indent=2, ensure_ascii=False)

    if args.mode == "qualify":
        gate("G4a_memorization_threshold_forget", f_tr_d >= TH,
             {"recall_nfkd": f_tr_d, "recall_nfkc": f_tr_f, "threshold": TH, "n": nf})
        gate("G4b_memorization_threshold_retain", r_tr_d >= TH,
             {"recall_nfkd": r_tr_d, "recall_nfkc": r_tr_f, "threshold": TH, "n": nr})
    else:
        R["gates"]["G4_memorization_threshold"] = {
            "passed": None, "detail": {"status": "NOT RUN -- calibration mode",
                                       "recall_nfkd_forget": f_tr_d,
                                       "recall_nfkd_retain": r_tr_d}}

    # ── G3: REAL save/reload behavioural-identity gate ──────────────────────
    R["peak_gpu_gib_through_training_and_eval"] = round(
        torch.cuda.max_memory_allocated(0) / 2**30, 3)

    if args.mode == "qualify":
        gen_before = []
        for fid, kind, q in probe_questions():
            txt, _, gids = gen_one(model, tok, q, DEV)
            gen_before.append({"fact_id": fid, "kind": kind, "question": q,
                               "gen": txt, "ids": gids})
        del model, opt, lora_pre, trainable, logits, tl, loss, other, l_pin, l_mm
        del ids, lab, att, wt, noisy, masked, p_mask, maskable
        torch.cuda.empty_cache()

        from peft import PeftModel
        base2 = load_base(DEV, train=False)
        m2 = PeftModel.from_pretrained(base2, adir).eval()
        cov2 = count_lora_coverage(m2)
        lg_after = battery_logits(m2, items)
        lora_hash_after = lora_tensor_hashes(m2)
        gen_after = []
        for fid, kind, q in probe_questions():
            txt, _, gids = gen_one(m2, tok, q, DEV)
            gen_after.append({"fact_id": fid, "kind": kind, "question": q,
                              "gen": txt, "ids": gids})

        per_item = []
        for it, a, b in zip(items, lg_before, lg_after):
            d = (a - b).abs()
            per_item.append({
                "kind": it["kind"], "fact_id": it["fact_id"], "shape": list(a.shape),
                "max_abs_logit_diff": float(d.max()),
                "mean_abs_logit_diff": float(d.mean()),
                "bitwise_identical": bool(torch.equal(a, b)),
                "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
                "max_abs_logit_before": float(a.abs().max()),
            })
        dmax = max(p["max_abs_logit_diff"] for p in per_item)
        argmax_ok = all(p["argmax_agreement"] == 1.0 for p in per_item)
        bitwise = all(p["bitwise_identical"] for p in per_item)
        gens_identical = ([(g["gen"], g["ids"]) for g in gen_before]
                          == [(g["gen"], g["ids"]) for g in gen_after])
        lora_bit_identical = (lora_hash_before == lora_hash_after)
        same_cov = (cov2["n_lora_modules"] == cov["n_lora_modules"]
                    and cov2["wrapped_module_names"] == cov["wrapped_module_names"]
                    and cov2["unwrapped_linear_modules"] == cov["unwrapped_linear_modules"]
                    and not cov2["lm_head_wrapped"])

        R["save_reload"] = {
            "protocol": ("adapter saved; trained model DELETED and CUDA cache emptied; a FRESH "
                         "base model instantiated from the pinned checkpoint; adapter reloaded "
                         "with PeftModel.from_pretrained; behaviour compared on a fixed battery "
                         "of 8 generation-shaped fully-masked canvases and 4 training-shaped "
                         "teacher-forced rows, plus 8 greedy generations."),
            "tolerance": RELOAD_TOL, "tolerance_justification": JUSTIFY_TOL,
            "max_abs_logit_diff": dmax,
            "all_items_bitwise_identical": bitwise,
            "argmax_agreement_all_positions": argmax_ok,
            "generations_identical": gens_identical,
            "lora_tensors_bit_identical": lora_bit_identical,
            "n_lora_tensors_compared": len(lora_hash_before),
            "reloaded_coverage_matches": same_cov,
            "reloaded_n_lora_modules": cov2["n_lora_modules"],
            "reloaded_lm_head_wrapped": cov2["lm_head_wrapped"],
            "per_item": per_item,
            "generations_before": gen_before, "generations_after": gen_after,
        }
        print(f"[G3] max|dlogit|={dmax:.3e} bitwise={bitwise} argmax_all={argmax_ok} "
              f"gens_identical={gens_identical} lora_bits={lora_bit_identical}", flush=True)
        gate("G3_save_reload_behavioural_identity",
             dmax <= RELOAD_TOL and argmax_ok and gens_identical
             and lora_bit_identical and same_cov,
             {"max_abs_logit_diff": dmax, "tolerance": RELOAD_TOL,
              "bitwise_identical": bitwise, "argmax_agreement": argmax_ok,
              "generations_identical": gens_identical,
              "lora_tensors_bit_identical": lora_bit_identical,
              "reloaded_coverage_matches": same_cov,
              "n_logit_battery_items": len(items), "n_generation_probes": len(gen_before)})
        del m2, base2
    else:
        R["gates"]["G3_save_reload_behavioural_identity"] = {
            "passed": None, "detail": {"status": "NOT RUN -- calibration mode"}}

    R["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 3)
    R["wall_clock_total_s"] = round(time.time() - t_wall0, 1)
    R["failed_gates"] = fails
    R["all_gates_passed"] = not fails
    R["qualified_backbone"] = bool(args.mode == "qualify" and not fails)
    R["mode"] = args.mode
    json.dump(R, open(args.report, "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    for k2, v in R["gates"].items():
        st = "NOTRUN" if v["passed"] is None else ("PASS" if v["passed"] else "FAIL")
        print(f"  {st:6s}  {k2}")
    print(f"\nmode={args.mode}  QUALIFIED_BACKBONE={R['qualified_backbone']}  "
          f"peak={R['peak_gpu_gib']} GiB  total={R['wall_clock_total_s']}s")
    print(f"report -> {args.report}")
    if fails:
        print(f"FAILED GATES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
