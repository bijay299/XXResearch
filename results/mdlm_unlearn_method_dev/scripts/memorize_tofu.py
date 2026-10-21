#!/usr/bin/env python
"""
memorize_tofu.py — build a REPRODUCIBLE, FULLY-QUALIFIED TOFU backbone by SFT LoRA.

Two deliverables, one script, selected by --train-split:

  --train-split all      theta0_tofu       SFT on all 800 QA. Must recall BOTH splits.
  --train-split retain   retain_ref_tofu   SFT on the 400 RETAIN QA only. Must recall
                                           retain AND MUST NOT recall forget.

The second is the load-bearing one. The 128-fact synthetic set FAILS that ignorance gate
at recall 1.000: every forget fact shares both its surname and its answer with a retain
partner (`Alira Ashcroft -> Paris` / `Zelvira Ashcroft -> Paris`), so `surname -> answer`
is perfectly predictive and learnable from retain alone. On that data "forget while
retaining" is contradictory rather than merely hard. tofu800 is author-disjoint with
400/400 distinct answers and zero answer overlap, so the gate is a real test of whether
the confound is gone. If it fails here too, the dataset is not fixed and nothing should be
built on it.

RELATION TO THE SYNTHETIC BACKBONE. `scripts/memorize_v2.py` and `scripts/common_v2.py`
are a published artifact — their sha256 is recorded inside the three delivered backbone
reports — so neither is modified. This file is a port; everything data-independent
(`load_base`, `count_lora_coverage`, `forward_process_canvas`, `forward_process`,
`step_losses`, the sha256 helpers) is imported read-only from `common_v2`, which itself
imports the frozen `spex/scripts/common_big.py` read-only. Nothing under spex/,
llada_poc/ or anvil/ is touched.

WHAT IS CARRIED OVER UNCHANGED (the things that made the synthetic backbone work)
  1. The REAL save/reload gate (G3): adapter saved, trained model destroyed, a FRESH base
     instantiated, adapter reloaded, and BEHAVIOURAL IDENTITY asserted — per-position
     logits on a fixed battery of both generation-shaped fully-masked canvases and
     training-shaped teacher-forced rows, argmax agreement at every position, byte- and
     token-identical greedy generations, and bit-identical LoRA tensors.
  2. The 224-module assertion, with the LM head located BY MODULE KEY
     (`model.transformer.ff_out`) and required to be a plain nn.Linear carrying no LoRA
     and to be the UNIQUE unwrapped Linear of 225. A `getattr(model, "transformer")`
     chain returns None through LLaDAModelLM's inner `.model` nesting and would make the
     check pass vacuously.
  3. The pinned LLaDA SFT loss normalization as the DEFAULT (`--norm pinned`), with
     masked-mean computed every step from the same per-token tensor as a diagnostic.
  4. `--canvas genlen`: the answer region is EOS-padded to gen_length and the WHOLE region
     is noised, so training sees the fully-masked canvas the decoder starts from. Decisive
     on the synthetic set (0.844/0.812 -> 1.000/1.000) and expected to matter MORE here,
     because a TOFU answer has variable length so the model must also learn where to stop.
  5. A full report.json: every argument, the resolved config, the per-step loss/grad-norm
     history, all gate results, throughput, peak GPU, dataset and source hashes, library
     versions, the resolved wrapped-module list, and the exact launch command.

WHAT IS NEW
  * Scoring is DISTINCTIVE content recall (`common_tofu`), not alias span matching: TOFU
    answers restate the question, so plain content overlap credits a model that has never
    seen the data (naive 0.33 on the pristine base). Naive content recall and ROUGE-L
    recall are reported alongside on every probe so the inflation is visible.
  * The evaluator is `eval_tofu.evaluate`, the SAME function that measured the
    pristine-base floor the thresholds are justified against.

A failing gate makes the process exit nonzero and writes the report anyway. Gates are
never loosened to make a run pass.
"""
import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
import time

import torch
import torch.nn.functional as F

A = "/data/bijaypandey/mdlm_unlearn"
S = "/data/bijaypandey/spex"
sys.path.insert(0, f"{A}/scripts")

from common_v2 import (  # noqa: E402
    CANVAS_DOC, EOS_ID, FULL_TARGETS, LM_HEAD_PATH, MASK_ID, MODEL_DIR, MODEL_REV,
    NORM_DOC, N_EXPECTED_MODULES, N_LINEARS_TOTAL, PER_BLOCK_LINEARS, PINNED_DEVIATIONS,
    build_prompt, count_lora_coverage, dir_sha256, forward_process,
    forward_process_canvas, load_base, load_tokenizer, sha256_dir, sha256_file,
    step_losses,
)
from common_tofu import (  # noqa: E402
    SCORING_DOC, STOPWORDS, TOFU_DATA, TOFU_DECODER_DOC, TOFU_GEN_KW, TOFU_GEN_LENGTH,
    load_tofu, make_example_tofu, scorable_set, stopword_hash,
)
import eval_tofu as ET  # noqa: E402

DEV = "cuda:0"
RELOAD_TOL = 1e-3
JUSTIFY_TOL = (
    "Logits are produced by a bf16 backbone. bf16 carries an 8-bit significand, so its "
    "relative eps is 2^-8 = 3.906e-3 and ONE ULP at the magnitude these logits reach is "
    "~0.06-0.25. Any genuine divergence on reload -- a LoRA tensor in a different dtype, "
    "a scaling factor applied once instead of twice, a module left unwrapped -- must move "
    "at least one output ULP. The 1e-3 tolerance is ~60-250x tighter than a single ULP, so "
    "it admits only bitwise or sub-ULP agreement and rejects every divergence that could "
    "change a decoded token. The gate additionally requires three TOLERANCE-FREE "
    "conditions: argmax agreement at 100% of positions, byte- and token-identical greedy "
    "generations, and bit-identical reloaded LoRA tensors."
)

R = {"stage": "memorization_tofu", "model_dir": MODEL_DIR, "model_revision": MODEL_REV,
     "gates": {}}
fails = []


def gate(name, ok, detail=None):
    R["gates"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    return bool(ok)


# ── fixed, seed-independent batteries for the save/reload gate ────────────────
# 4 forget + 4 retain generation-shaped canvases, 4 training-shaped rows, 8 greedy probes.
LOGIT_BATTERY = [("forget", 0), ("forget", 1), ("forget", 199), ("forget", 399),
                 ("retain", 0), ("retain", 1), ("retain", 199), ("retain", 399)]
TF_BATTERY = [("forget", 0), ("forget", 200), ("retain", 0), ("retain", 200)]
GEN_PROBE = [("forget", 0), ("forget", 1), ("forget", 200), ("forget", 399),
             ("retain", 0), ("retain", 1), ("retain", 200), ("retain", 399)]


def by_split(rows):
    return {sp: [r for r in rows if r["split"] == sp] for sp in ("forget", "retain")}


def battery_inputs(tok, rows, canvas, gen_length):
    """Deterministic, model-independent inputs, in both shapes the model ever sees."""
    bs = by_split(rows)
    items = []
    for sp, k in LOGIT_BATTERY:
        r = bs[sp][k]
        ids = tok(build_prompt(tok, r["question"]), add_special_tokens=False)["input_ids"]
        ids = ids + [MASK_ID] * gen_length          # the decoder's first-step input
        items.append({"kind": "gen_canvas", "id": r["id"],
                      "ids": torch.tensor([ids]),
                      "attn": torch.ones(1, len(ids), dtype=torch.long)})
    g = torch.Generator(device="cpu").manual_seed(777000)
    for sp, k in TF_BATTERY:
        r = bs[sp][k]
        e = make_example_tofu(tok, r, gen_length=gen_length, canvas=canvas)
        ids = torch.tensor([e["ids"]])
        lab = torch.tensor([e["labels"]])
        if canvas == "genlen":
            pos = torch.arange(ids.shape[1]).unsqueeze(0)
            region = (pos >= e["prompt_len"]) & (pos < e["prompt_len"] + gen_length)
            noisy, _, _, _ = forward_process_canvas(ids, lab, region, g, t_low=0.0)
        else:
            noisy, _, _ = forward_process(ids, lab, g, t_low=0.0)
        items.append({"kind": "teacher_forced", "id": r["id"],
                      "ids": noisy, "attn": torch.ones_like(ids)})
    return items


@torch.no_grad()
def battery_logits(model, items):
    return [model(it["ids"].to(DEV), attention_mask=it["attn"].to(DEV))
            .logits.detach().float().cpu() for it in items]


def probe_questions(rows):
    bs = by_split(rows)
    return [(bs[sp][k]["id"], bs[sp][k]["question"]) for sp, k in GEN_PROBE]


def lora_tensor_hashes(model):
    return {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters() if "lora_" in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--t-low", type=float, default=0.0)
    ap.add_argument("--norm", default="pinned", choices=["pinned", "masked-mean"])
    ap.add_argument("--canvas", default="genlen", choices=["exact", "genlen"])
    ap.add_argument("--gen-length", type=int, default=TOFU_GEN_LENGTH)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, required=True,
                    help="distinctive-content-recall gate, both splits")
    ap.add_argument("--ignorance-ceiling", type=float, required=True,
                    help="max forget recall allowed for a retain-only reference")
    ap.add_argument("--mode", default="qualify", choices=["qualify", "calibrate"])
    ap.add_argument("--qual-subset", type=int, default=0,
                    help="calibration only: probes scored per split")
    ap.add_argument("--train-split", default="all", choices=["all", "forget", "retain"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
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
        f"{A}/scripts/memorize_tofu.py", f"{A}/scripts/common_tofu.py",
        f"{A}/scripts/eval_tofu.py", f"{A}/scripts/common_v2.py",
        f"{A}/scripts/evaluate_partition.py", f"{A}/scripts/launch_tofu.sh",
        f"{S}/scripts/common_big.py",
        "/data/bijaypandey/llada_poc/scripts/llada_fxmix.py"] if os.path.exists(p)}
    R["args"] = vars(args)
    R["launch"] = {
        "argv": sys.argv,
        "command": " ".join(shlex.quote(x) for x in [sys.executable] + sys.argv),
        "launch_command_from_env": os.environ.get("MDLM_LAUNCH_CMD"),
        "attempt_label_from_env": os.environ.get("MDLM_ATTEMPT_LABEL"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd()}
    R["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__,
                     "cuda": torch.version.cuda,
                     "gpu_name": torch.cuda.get_device_name(0),
                     "driver_cuda_capability": list(torch.cuda.get_device_capability(0))}
    R["hashes"] = {"data": {os.path.basename(TOFU_DATA): sha256_file(TOFU_DATA)},
                   "sources": srcs}
    R["decoding"] = dict(TOFU_GEN_KW)
    R["decoding"]["doc"] = TOFU_DECODER_DOC
    R["scoring"] = {"doc": SCORING_DOC, "stopwords_sha256": stopword_hash(),
                    "n_stopwords": len(STOPWORDS),
                    "stopwords": sorted(STOPWORDS)}
    R["loss_normalization"] = {
        "selected": args.norm, "doc": NORM_DOC,
        "pinned_source": "/data/bijaypandey/llada_poc/scripts/llada_fxmix.py::sft_loss_llada",
        "deviations_from_pinned": PINNED_DEVIATIONS}

    torch.manual_seed(args.seed)
    tok = load_tokenizer()
    ROWS = load_tofu()

    # ── G0: every probe must have scorable distinctive content ──────────────
    # The TOFU analogue of the synthetic set's alias gate. An item whose answer adds no
    # token the question did not already contain cannot distinguish knowledge from echo,
    # so it is excluded from the mean -- and the count must be visible, never silent.
    sc_counts = [len(scorable_set(r["question"], r["answer"])) for r in ROWS]
    empty = [r["id"] for r, n in zip(ROWS, sc_counts) if n == 0]
    R["scorable_content"] = {
        "n_items": len(ROWS),
        "mean_scorable_tokens": round(sum(sc_counts) / len(sc_counts), 3),
        "min_scorable_tokens": min(sc_counts), "max_scorable_tokens": max(sc_counts),
        "items_with_empty_scorable_set": empty,
        "n_with_empty_scorable_set": len(empty),
        "note": ("scorable(q,a) = content tokens of a MINUS all tokens of q. Items with an "
                 "empty set are excluded from the recall mean and counted here.")}
    gate("G0_every_probe_has_scorable_distinctive_content", not empty,
         {"n_items": len(ROWS), "n_empty": len(empty),
          "mean_scorable_tokens": R["scorable_content"]["mean_scorable_tokens"],
          "min_scorable_tokens": min(sc_counts)})

    # ── dataset structure: the confound the synthetic set had ───────────────
    fa = {r["author"] for r in ROWS if r["split"] == "forget"}
    ra = {r["author"] for r in ROWS if r["split"] == "retain"}
    fans = {r["answer"] for r in ROWS if r["split"] == "forget"}
    rans = {r["answer"] for r in ROWS if r["split"] == "retain"}
    R["dataset_structure"] = {
        "n_forget": sum(1 for r in ROWS if r["split"] == "forget"),
        "n_retain": sum(1 for r in ROWS if r["split"] == "retain"),
        "n_forget_authors": len(fa), "n_retain_authors": len(ra),
        "authors_in_both_splits": sorted(fa & ra),
        "author_disjoint": not (fa & ra),
        "n_distinct_forget_answers": len(fans), "n_distinct_retain_answers": len(rans),
        "n_answers_in_both_splits": len(fans & rans),
        "answer_disjoint": not (fans & rans),
        "why_this_matters": ("FINDING5: in the 128-fact synthetic set every forget fact "
                             "shared BOTH surname and answer with a retain partner, so "
                             "surname->answer was learnable from retain alone and a "
                             "retain-only reference recalled all 64 forget facts. The "
                             "ignorance gate below is the test of whether that is fixed.")}
    gate("G0b_splits_are_author_and_answer_disjoint",
         not (fa & ra) and not (fans & rans),
         {k: R["dataset_structure"][k] for k in
          ("n_forget_authors", "n_retain_authors", "author_disjoint",
           "n_answers_in_both_splits", "answer_disjoint")})

    model = load_base(DEV, train=True)

    # ── training examples ───────────────────────────────────────────────────
    TRAIN = ([r for r in ROWS if r["split"] == args.train_split]
             if args.train_split != "all" else list(ROWS))
    print(f"[train-split] {args.train_split}: training on {len(TRAIN)} of {len(ROWS)} QA",
          flush=True)
    ex = [make_example_tofu(tok, r, gen_length=args.gen_length, canvas=args.canvas)
          for r in TRAIN]
    maxlen = max(len(e["ids"]) for e in ex)
    n_sup = sum(e["n_supervised"] for e in ex)
    n_eos = sum(e["n_eos_supervised"] for e in ex)
    n_dist = sum(len(e["distinctive_positions"]) for e in ex)
    no_dist = [e["row_id"] for e in ex if not e["distinctive_positions"]]
    trunc = [e["row_id"] for e in ex if e["answer_len"] >= args.gen_length - 1]
    R["training_canvas"] = {
        "selected": args.canvas, "doc": CANVAS_DOC, "gen_length": args.gen_length,
        "max_seq_len": maxlen,
        "mean_supervised_per_example": round(n_sup / len(ex), 2),
        "mean_answer_tokens": round(sum(e["answer_len"] for e in ex) / len(ex), 2),
        "max_answer_tokens": max(e["answer_len"] for e in ex),
        "answers_truncated_by_canvas": trunc, "n_answers_truncated": len(trunc)}
    gate("G1_answer_bearing_loss_masking",
         n_eos == 0 and n_dist > 0 and not no_dist and not trunc,
         {"n_examples": len(ex), "max_seq_len": maxlen,
          "total_supervised_positions": n_sup,
          "supervised_positions_that_are_EOS": n_eos,
          "supervised_positions_carrying_distinctive_content": n_dist,
          "distinctive_fraction_of_supervised": round(n_dist / n_sup, 4),
          "examples_with_no_distinctive_supervised_position": no_dist,
          "n_answers_truncated_by_canvas": len(trunc),
          "note": ("supervision is restricted to the answer's own tokens; the EOS padding "
                   "that extends the answer region to gen_length is label-masked, so "
                   "neither padding nor EOS can dominate the loss. gen_length is chosen "
                   "from the measured answer-length distribution so no answer is "
                   "truncated.")})

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
          f"per_block={cov['linears_per_block_count']} "
          f"lm_head={cov['lm_head_path']} wrapped={cov['lm_head_wrapped']}", flush=True)
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
          "lm_head_located_by_module_key": cov["lm_head_located"],
          "lm_head_is_plain_linear": cov["lm_head_is_plain_linear"],
          "lm_head_wrapped": cov["lm_head_wrapped"],
          "n_outer_linear_modules": cov["n_outer_linear_modules"],
          "unwrapped_linear_modules": cov["unwrapped_linear_modules"],
          "expected_lm_head_path": LM_HEAD_PATH,
          "note": ("located BY MODULE KEY, not by a getattr chain: LLaDAModelLM nests the "
                   "backbone under an inner `.model`, so getattr(peft_model, 'transformer') "
                   "yields None and a naive check passes VACUOUSLY. Cross-checked "
                   f"independently: of the {N_LINEARS_TOTAL} Linear sites exactly 224 are "
                   "wrapped and the single unwrapped one IS the head.")})
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
        maskable = lab != -100
        if args.canvas == "genlen":
            pos = torch.arange(L, device=DEV).expand(len(batch), L)
            plen = torch.tensor([b["prompt_len"] for b in batch], device=DEV).unsqueeze(1)
            region = (pos >= plen) & (pos < plen + args.gen_length)
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
        if step % 200 == 0 or step == args.steps - 1:
            print(f"[train] step {step:5d} {args.norm}={float(loss):.5f} "
                  f"other={float(other):.5f} gnorm={float(gnorm):.4f} "
                  f"nsup={int(masked.sum())} ncanvas={n_canvas} "
                  f"({(time.time()-t0)/(step+1):.3f}s/step)", flush=True)
    train_s = time.time() - t0
    losses = hist["loss"]
    nz = sum(1 for i, l in enumerate(losses) if l == 0.0 and hist["gnorm"][i] == 0.0)
    R["training"] = {
        "steps": args.steps, "wall_s": round(train_s, 1),
        "s_per_step": round(train_s / args.steps, 4),
        "first_step_s": round(first_step_s, 2),
        "n_train_examples": len(ex), "batch": args.batch,
        "passes_over_training_set": round(args.steps * args.batch / len(ex), 2),
        "loss_first": losses[0], "loss_last": losses[-1], "loss_min": min(losses),
        "loss_mean_last_100": sum(losses[-100:]) / len(losses[-100:]),
        "steps_with_zero_loss_and_zero_gnorm": nz,
        "fraction_degenerate_steps": round(nz / args.steps, 4),
        "batch_order_scheme": ("single fixed permutation consumed cyclically as "
                               "order[(step*batch+k) mod n] -- the synthetic backbone's "
                               "scheme, preserved for comparability"),
        "lr_schedule": "constant (no warmup, no decay)",
        "history": hist}
    print(f"[throughput] train {train_s:.1f}s  {train_s/args.steps:.4f}s/step  "
          f"passes={R['training']['passes_over_training_set']}  "
          f"degenerate_steps={nz}/{args.steps}", flush=True)

    k = min(10, args.steps)
    pin = losses[:k] if args.norm == "pinned" else hist["loss_other"][:k]
    mm = hist["loss_other"][:k] if args.norm == "pinned" else losses[:k]
    R["norm_comparison_first_steps"] = {
        "n_steps": k, "pinned": pin, "masked_mean": mm,
        "ratio_masked_mean_over_pinned": [round(m / p, 4) if p else None
                                          for m, p in zip(mm, pin)],
        "pinned_mean": sum(pin) / k, "masked_mean_mean": sum(mm) / k,
        "note": ("both computed from the SAME per-masked-position tensor CE/p_mask inside "
                 "the same step, so the ratio is purely the normalizer")}

    changed = [n for n, p in trainable if (p.detach() - lora_pre[n]).abs().max().item() > 0]
    base_post = {n: hashlib.sha256(p.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
                 for n, p in model.named_parameters() if "lora_" not in n}
    drift = [n for n in base_pre if base_pre[n] != base_post.get(n)]
    gate("G2c_trainable_params_changed", len(changed) > 0,
         {"n_changed": len(changed), "n_trainable": len(trainable)})
    gate("G2d_base_weights_unchanged", len(drift) == 0,
         {"n_base_tensors_hashed": len(base_pre), "n_drifted": len(drift),
          "coverage": "100% of non-LoRA parameter tensors"})
    # G2e compares WINDOW MEANS, not single steps. On TOFU the per-step loss is genuinely
    # noisy because the number of supervised positions varies 3x between batches (answers
    # are 4-82 tokens) under the weak cyclic shuffle, so a single-step comparison fires at
    # random: calibration run p065 exited rc=1 with first=2.623 last=2.865 while its recall
    # had risen from the 0.0785 floor to 0.311. That is a gate carrying no information.
    # Changed BEFORE either production run was launched; the single-step values are still
    # reported so the original comparison remains visible.
    w = max(1, min(100, args.steps // 4))
    fm = sum(losses[:w]) / w
    lm = sum(losses[-w:]) / w
    gate("G2e_loss_decreased", lm < fm,
         {"window": w, "mean_first_window": round(fm, 6), "mean_last_window": round(lm, 6),
          "ratio_last_over_first": round(lm / fm, 4) if fm else None,
          "single_step_first": losses[0], "single_step_last": losses[-1],
          "single_step_comparison_would_pass": losses[-1] < losses[0],
          "note": ("window means, because the per-step loss is noisy: the supervised count "
                   "per batch varies ~3x with answer length. The single-step comparison is "
                   "reported but not gated.")})

    # ── save adapter ────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    adir = args.out if os.path.exists(f"{args.out}/adapter_config.json") else f"{args.out}/default"
    sz = sum(os.path.getsize(os.path.join(adir, f)) for f in os.listdir(adir)
             if os.path.isfile(os.path.join(adir, f)))
    R["adapter"] = {"dir": adir, "bytes": sz, "mib": round(sz / 2**20, 2),
                    "sha256_per_file": sha256_dir(adir), "sha256_dir": dir_sha256(adir),
                    "note": "adapter weights only; the full backbone is NOT re-saved"}
    print(f"[adapter] {sz/2**20:.1f} MiB -> {adir} "
          f"dir_sha256={R['adapter']['sha256_dir'][:16]}", flush=True)

    # ── capture pre-save behaviour for the save/reload gate ─────────────────
    model.eval()
    items = battery_inputs(tok, ROWS, args.canvas, args.gen_length)
    lg_before = battery_logits(model, items) if args.mode == "qualify" else None
    lora_hash_before = lora_tensor_hashes(model)

    # ── qualification: GENERATED output, scored by DISTINCTIVE content recall ──
    probes = ET.subset(ROWS, args.qual_subset) if args.qual_subset else ROWS
    t0 = time.time()
    per = ET.evaluate(model, tok, probes, DEV, every=100)
    eval_s = time.time() - t0
    R["evaluation"] = ET.summarize(per)
    R["evaluation"]["n_probes"] = len(probes)
    R["evaluation"]["wall_s"] = round(eval_s, 1)
    R["evaluation"]["s_per_generation"] = round(eval_s / max(1, len(probes)), 3)
    R["evaluation"]["subset"] = bool(args.qual_subset)
    json.dump(per, open(args.report.replace(".json", "_probes.json"), "w"),
              indent=1, ensure_ascii=False)

    FD = R["evaluation"]["forget"]["distinctive_content_recall"]["mean"]
    RD = R["evaluation"]["retain"]["distinctive_content_recall"]["mean"]
    FN = R["evaluation"]["forget"]["naive_content_recall"]["mean"]
    RN = R["evaluation"]["retain"]["naive_content_recall"]["mean"]
    nf = R["evaluation"]["forget"]["n"]
    nr = R["evaluation"]["retain"]["n"]
    print(f"[recall] DISTINCTIVE forget={FD} ({nf})  retain={RD} ({nr})", flush=True)
    print(f"[recall] naive       forget={FN}  retain={RN}  "
          f"(inflation +{R['evaluation']['forget']['inflation_naive_minus_distinctive']} / "
          f"+{R['evaluation']['retain']['inflation_naive_minus_distinctive']})", flush=True)
    print(f"[recall] rouge-L     forget="
          f"{R['evaluation']['forget']['rouge_l_recall']['mean']}  retain="
          f"{R['evaluation']['retain']['rouge_l_recall']['mean']}", flush=True)

    TH, IGN = args.threshold, args.ignorance_ceiling
    if args.mode == "qualify" and args.train_split == "retain":
        # THE LOAD-BEARING GATE. A retain-only reference must know retain and must NOT
        # know forget. Requiring forget recall here would be backwards. The synthetic set
        # fails this at 1.000 because surname->answer is learnable from retain alone.
        gate("G4a_IGNORANCE_of_forget_split", FD is not None and FD <= IGN,
             {"forget_distinctive_recall": FD, "ceiling": IGN, "n": nf,
              "forget_naive_recall": FN,
              "retain_distinctive_recall_for_contrast": RD,
              "meaning": ("a retain-only reference MUST NOT recall the forget split. This "
                          "is the gate the 128-fact synthetic set fails at 1.000, and it "
                          "is the test of whether tofu800 removes the confound.")})
        gate("G4b_recall_threshold_retain", RD is not None and RD >= TH,
             {"retain_distinctive_recall": RD, "threshold": TH, "n": nr,
              "retain_naive_recall": RN})
        R["ignorance_margin"] = {
            "retain_minus_forget_distinctive": (round(RD - FD, 4)
                                                if (RD is not None and FD is not None) else None),
            "meaning": ("the separation a retain-only reference achieves. On the synthetic "
                        "set this is ~0.000 by construction; a usable dataset must make it "
                        "large.")}
    elif args.mode == "qualify":
        gate("G4a_recall_threshold_forget", FD is not None and FD >= TH,
             {"forget_distinctive_recall": FD, "threshold": TH, "n": nf,
              "forget_naive_recall": FN})
        gate("G4b_recall_threshold_retain", RD is not None and RD >= TH,
             {"retain_distinctive_recall": RD, "threshold": TH, "n": nr,
              "retain_naive_recall": RN})
    else:
        R["gates"]["G4_recall_thresholds"] = {
            "passed": None,
            "detail": {"status": "NOT RUN -- calibration mode",
                       "forget_distinctive": FD, "retain_distinctive": RD}}

    # ── G3: REAL save/reload behavioural-identity gate ──────────────────────
    R["peak_gpu_gib_through_training_and_eval"] = round(
        torch.cuda.max_memory_allocated(0) / 2**30, 3)

    if args.mode == "qualify":
        gen_before = [{"id": i, "question": q, **dict(zip(("gen", "raw", "ids"),
                                                          ET.gen_one_tofu(model, tok, q, DEV)))}
                      for i, q in probe_questions(ROWS)]
        del model, opt, lora_pre, trainable, logits, tl, loss, other, l_pin, l_mm
        del ids, lab, att, wt, noisy, masked, p_mask, maskable
        torch.cuda.empty_cache()

        from peft import PeftModel
        base2 = load_base(DEV, train=False)
        m2 = PeftModel.from_pretrained(base2, adir).eval()
        cov2 = count_lora_coverage(m2)
        lg_after = battery_logits(m2, items)
        lora_hash_after = lora_tensor_hashes(m2)
        gen_after = [{"id": i, "question": q, **dict(zip(("gen", "raw", "ids"),
                                                         ET.gen_one_tofu(m2, tok, q, DEV)))}
                     for i, q in probe_questions(ROWS)]

        per_item = []
        for it, a_, b_ in zip(items, lg_before, lg_after):
            d = (a_ - b_).abs()
            per_item.append({
                "kind": it["kind"], "id": it["id"], "shape": list(a_.shape),
                "max_abs_logit_diff": float(d.max()),
                "mean_abs_logit_diff": float(d.mean()),
                "bitwise_identical": bool(torch.equal(a_, b_)),
                "argmax_agreement": float((a_.argmax(-1) == b_.argmax(-1)).float().mean()),
                "max_abs_logit_before": float(a_.abs().max())})
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
            "protocol": ("adapter saved; trained model DELETED and CUDA cache emptied; a "
                         "FRESH base instantiated from the pinned checkpoint; adapter "
                         "reloaded with PeftModel.from_pretrained; behaviour compared on 8 "
                         "generation-shaped fully-masked canvases and 4 training-shaped "
                         "teacher-forced rows, plus 8 greedy generations compared as both "
                         "decoded string and token-id list."),
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
            "generations_before": gen_before, "generations_after": gen_after}
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
              "n_logit_battery_items": len(items),
              "n_generation_probes": len(gen_before)})
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
    print(f"\nmode={args.mode} split={args.train_split} "
          f"QUALIFIED={R['qualified_backbone']} peak={R['peak_gpu_gib']} GiB "
          f"total={R['wall_clock_total_s']}s")
    print(f"report -> {args.report}")
    if fails:
        print(f"FAILED GATES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
