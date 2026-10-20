"""run_matched.py — one unlearning arm, trained and partition-evaluated under MATCHED conditions.

Deliverable 2 of the matched-baseline batch. Converts the unmatched framing table in
docs/FINDING2_ignorant_axis.md (our arms on 128 facts / 64 probes vs four reproduced
baselines on 24 facts / 8 probes) into a comparator table where every arm shares the
backbone, the data, the LoRA surface, the optimizer, the step budget, the checkpoint rule
and the evaluator.

PROVENANCE / WHAT IS NOT REIMPLEMENTED HERE
  * every forget objective comes from the PINNED package, imported read-only:
        llada_poc/scripts/llada_fxmix.py        FXConfig, prepare_data, comparator_forget_loss,
                                                sft_loss_llada, {mdu,emdu,npo,sc_tf}_loss_llada,
                                                fx_forget_loss_llada, build_fact_set, left_pad
        llada_poc/scripts/mdmu/fxmix.py         exclusion_target, soft_ce_kl (via the above)
    Nothing in the loss math is rewritten; this file is a driver + instrumentation, the
    same contract as spex/scripts/run_baseline.py, which it ports.
  * NOTE for anyone re-reading the prior audit: `sft_loss_llada` lives in
    scripts/llada_fxmix.py (line 475), NOT in scripts/mdmu/fxmix.py.
  * the evaluator is scripts/evaluate_partition.py; the frozen scorer/decoder/loaders come
    from spex/scripts/common_big.py through scripts/common_v2.py.
  * spex/, llada_poc/ and anvil/ are closed published studies and are never written to.

THE FIVE ARMS. Four are the published baselines ported from run_baseline.py (built for the
24-fact set); the fifth is our exclusion-target control, ported from spex/scripts/arc_big.py
(its `uncached` arm, already on 128 facts).

    mdu_t1  reverse-KL to a temperature-sharpened frozen anchor; anchor = reference at the
            same noisy state with the PROMPT masked; KL(p_theta || a).      hp = tau
    emdu    KL arguments swapped relative to MDU, KL(a || p_theta); anchor = reference on
            the FULLY masked sequence (FAM).                                hp = tau
    npo     -(2/beta) E log sigma(beta (L_sft(theta) - L_sft(ref))).        hp = beta
    sc_tf   exclusion target r_eps at the same state, target-first KL.      hp = eps
    excl    OUR exclusion-target control.                                   hp = eps

    *** DECLARED, AND IT MATTERS: `sc_tf` and `excl` call the SAME pinned function
    (fx_forget_loss_llada) on the SAME states. Under matched conditions the ONLY
    difference between them is how the fact set F is built:
        sc_tf : F = build_fact_set(prompt, item["answer"])      -- the whole answer SENTENCE
                    ("Alira Ashcroft lives in Paris.") as in run_baseline.py's prepare_data,
                    so F contains the entity's own name tokens and the template verbs.
        excl  : F = build_fact_set(prompt, item["answer_key"])  -- the answer ALONE
                    ("Paris") as in arc_big.py's build_rows.
    This was invisible in FINDING2 because the two arms were measured on different fact
    sets. It is reported as a measured fact-set size difference, not asserted. ***

MATCHED CONDITIONS, all asserted at runtime and recorded in the report:
    backbone      adapters/memorized_v2_s20261101 (1.000/1.000, save/reload gate max|dlogit|=0)
                  merged in memory, then a FRESH unlearning LoRA, so disable_adapter() is
                  exactly the qualified memorized reference (gate: max|dlogit| == 0 before
                  training, > 0 after).
    data          spex/data_big/{forget,retain}.jsonl -- 64 forget + 64 retain, laid out by
                  the pinned FX.prepare_data. ids/attn/ans_start are byte-identical across
                  all five arms (hashed into the report); only fact_mask differs, per above.
    lora          r=16, alpha=32, the 224-module regex. Asserted: exactly 224 LoraLayers,
                  the LM head located BY MODULE KEY (model.transformer.ff_out -- a getattr
                  chain on transformer.ff_out returns None through LLaDAModelLM's inner
                  .model nesting and would make the check pass vacuously) and required to
                  be a plain nn.Linear.
    optimizer     AdamW, lr 1e-4 constant, weight_decay 0.0, grad_clip 1.0
    step budget   150 steps, 8 forget states/step, 8 retain rows/step, lam_f = lam_r = 1
    checkpoint    one checkpoint, at the final step, evaluated; no mid-run selection
    evaluator     evaluate_partition.py on all 128 probes, train question, frozen decoder
    seed          0 -- ONE seed for this batch, by instruction. Multi-seed replication is a
                  later batch; no interval is claimed here.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, "/data/bijaypandey/llada_poc/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)

import llada_fxmix as FX                                     # PINNED, unmodified
import evaluate_partition as EP
from common_v2 import (
    DATA, FULL_TARGETS, MODEL_DIR, MODEL_REV, N_EXPECTED_MODULES, count_lora_coverage,
    load_tokenizer, sha256_file,
)

W = "/data/bijaypandey/mdlm_unlearn"
DEV = "cuda:0"                       # physical GPU chosen by CUDA_VISIBLE_DEVICES
VOCAB = 126464
MEMO_DEFAULT = f"{W}/adapters/memorized_v2_s20261101"

ARMS = ("reference", "mdu_t1", "emdu", "npo", "sc_tf", "excl")
HP_NAME = {"mdu_t1": "tau", "emdu": "tau", "npo": "npo_beta", "sc_tf": "eps",
           "excl": "eps", "reference": None}
FACTSET_SOURCE = {"sc_tf": "answer_sentence", "excl": "answer_key",
                  "mdu_t1": "unused", "emdu": "unused", "npo": "unused",
                  "reference": "unused"}

# ── the matched recipe. One dict, no per-arm branches other than the arm's own hp. ──
MATCHED = dict(
    dtype="bfloat16", gen_length=32, lora_r=16, lora_alpha=32.0,
    lora_target_regex=FULL_TARGETS, lr=1e-4, lr_schedule="constant", weight_decay=0.0,
    grad_clip=1.0, forget_states_per_step=8, retain_batch=8, lam_f=1.0, lam_r=1.0,
    append_eot=False, sft_eps=1e-3, seed=0, steps=150,
)


class FwdCounter:
    """Wraps FX._logits so forwards/step is MEASURED, not asserted. Splits student from
    reference by inspecting whether the adapter is currently disabled."""

    def __init__(self, model):
        self.orig = FX._logits
        self.model = model
        self.calls = self.student = self.reference = self.rows = self.tokens = 0

    def install(self):
        def counting(model, ids, am):
            self.calls += 1
            self.rows += int(ids.shape[0])
            self.tokens += int(ids.shape[0] * ids.shape[1])
            if torch.is_grad_enabled():
                self.student += 1
            else:
                self.reference += 1
            return self.orig(model, ids, am)
        FX._logits = counting

    def restore(self):
        FX._logits = self.orig

    def snap(self):
        return (self.calls, self.student, self.reference, self.rows, self.tokens)


@torch.no_grad()
def adapter_delta(model, ids, attn):
    """max |logit(adapter ON) - logit(adapter OFF)|. 0 for a fresh LoRA (B = 0)."""
    was = model.training
    model.eval()
    on = FX._logits(model, ids, attn).float()
    with model.disable_adapter():
        off = FX._logits(model, ids, attn).float()
    model.train(was)
    return float((on - off).abs().max())


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--hp", type=float, default=None,
                    help="the arm's principal hyperparameter (tau / npo_beta / eps)")
    ap.add_argument("--steps", type=int, default=MATCHED["steps"])
    ap.add_argument("--memo", default=MEMO_DEFAULT)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=f"{W}/adapters/matched")
    ap.add_argument("--report", required=True)
    ap.add_argument("--eval-limit", type=int, default=0, help="smoke: facts per split")
    ap.add_argument("--save-adapter", type=int, default=1)
    a = ap.parse_args()

    t_all = time.time()
    arm_is_ref = a.arm == "reference"
    hpn = HP_NAME[a.arm]
    if not arm_is_ref and a.hp is None:
        print(f"[fatal] arm {a.arm} needs --hp ({hpn})", flush=True)
        return 2

    # the pinned config object; cfg.arm selects the pinned loss dispatch
    cfg = FX.FXConfig(
        model_name=MODEL_DIR, dtype=MATCHED["dtype"], gen_length=MATCHED["gen_length"],
        eps=1e-5, lora_r=MATCHED["lora_r"], lora_alpha=MATCHED["lora_alpha"],
        lora_target_regex=MATCHED["lora_target_regex"], lr=MATCHED["lr"],
        lr_schedule=MATCHED["lr_schedule"], steps=a.steps,
        weight_decay=MATCHED["weight_decay"], grad_clip=MATCHED["grad_clip"],
        forget_states_per_step=MATCHED["forget_states_per_step"],
        retain_batch=MATCHED["retain_batch"], lam_f=MATCHED["lam_f"], lam_r=MATCHED["lam_r"],
        append_eot=MATCHED["append_eot"], sft_eps=MATCHED["sft_eps"], seed=MATCHED["seed"],
        arm=("sc_tf" if a.arm == "excl" else ("sc_tf" if arm_is_ref else a.arm)),
        tau=1.0, npo_beta=0.2,
    )
    if hpn == "tau":
        cfg.tau = a.hp
    elif hpn == "npo_beta":
        cfg.npo_beta = a.hp
    elif hpn == "eps":
        cfg.eps = a.hp

    R = {
        "label": a.label, "arm": a.arm, "hyperparameter": {"name": hpn, "value": a.hp},
        "matched_conditions": dict(MATCHED, steps=a.steps),
        "pinned_sources": {
            "llada_fxmix.py": sha256_file("/data/bijaypandey/llada_poc/scripts/llada_fxmix.py"),
            "mdmu/fxmix.py": sha256_file("/data/bijaypandey/llada_poc/scripts/mdmu/fxmix.py"),
            "spex/scripts/common_big.py": sha256_file("/data/bijaypandey/spex/scripts/common_big.py"),
            "spex/scripts/run_baseline.py": sha256_file("/data/bijaypandey/spex/scripts/run_baseline.py"),
            "spex/scripts/arc_big.py": sha256_file("/data/bijaypandey/spex/scripts/arc_big.py"),
            "modification": "none; imported read-only",
        },
        "driver_sources": {
            "scripts/run_matched.py": sha256_file(os.path.abspath(__file__)),
            "scripts/evaluate_partition.py": sha256_file(f"{HERE}/evaluate_partition.py"),
            "scripts/common_v2.py": sha256_file(f"{HERE}/common_v2.py"),
        },
        "resolved_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "factset_source": FACTSET_SOURCE[a.arm],
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mask_id": FX.MASK_ID, "eos_id": FX.EOS_ID,
        "gates": {},
    }
    torch.manual_seed(cfg.seed)

    # ── 1. qualified backbone: base + memo LoRA merged in memory ──────────────
    t0 = time.time()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel
    tok = load_tokenizer()
    base = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16).to(DEV)
    model = PeftModel.from_pretrained(base, a.memo).merge_and_unload()
    model.train()
    R["backbone"] = {"base": MODEL_DIR, "rev": MODEL_REV, "memo_adapter": a.memo,
                     "merged_in_memory": True,
                     "vocab_size": int(getattr(model.config, "vocab_size", VOCAB)),
                     "note": "disable_adapter() == the qualified MEMORIZED reference"}

    # ── 2. fresh unlearning LoRA + coverage gates ─────────────────────────────
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=int(cfg.lora_alpha), lora_dropout=0.0, bias="none",
        target_modules=cfg.lora_target_regex, init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    cov = count_lora_coverage(model)
    cov.pop("wrapped_module_names", None)
    R["lora_coverage"] = cov
    R["gates"]["G_lora_224"] = {
        "n_lora_modules": cov["n_lora_modules"], "expected": N_EXPECTED_MODULES,
        "passed": cov["n_lora_modules"] == N_EXPECTED_MODULES}
    R["gates"]["G_lm_head_not_wrapped"] = {
        "lm_head_path": cov["lm_head_path"], "lm_head_type": cov["lm_head_type"],
        "lm_head_located": cov["lm_head_located"],
        "lm_head_wrapped": cov["lm_head_wrapped"],
        "lm_head_is_plain_linear": cov["lm_head_is_plain_linear"],
        "unwrapped_linear_modules": cov["unwrapped_linear_modules"],
        "n_outer_linear_modules": cov["n_outer_linear_modules"],
        # NOT vacuous: the head must be FOUND by module key, be a plain nn.Linear, carry
        # no LoRA, and be the ONE unwrapped Linear in the whole model -- that last clause
        # is regex-independent, so a regex that silently missed the head cannot pass.
        "passed": bool(cov["lm_head_located"]
                       and cov["lm_head_is_plain_linear"]
                       and not cov["lm_head_wrapped"]
                       and cov["unwrapped_linear_modules"] == [cov["lm_head_path"]]),
        "check": ("located by module key AND plain nn.Linear AND not a LoraLayer AND the "
                  "unique unwrapped Linear in the model"),
    }
    for g in ("G_lora_224", "G_lm_head_not_wrapped"):
        print(f"[gate] {g}: {'PASS' if R['gates'][g]['passed'] else 'FAIL'} "
              f"{json.dumps({k: v for k, v in R['gates'][g].items() if k != 'passed'})}",
              flush=True)
    assert R["gates"]["G_lora_224"]["passed"], "LoRA module count != 224"
    assert R["gates"]["G_lm_head_not_wrapped"]["passed"], "LM head coverage gate failed"
    trainable = [p for p in model.parameters() if p.requires_grad]
    R["n_trainable_tensors"] = len(trainable)
    R["n_trainable_params"] = int(sum(p.numel() for p in trainable))
    R["load_wall_s"] = round(time.time() - t0, 1)

    # ── 3. data: the pinned layout, identical for every arm ──────────────────
    forget = read_jsonl(f"{DATA}/forget.jsonl")
    retain = read_jsonl(f"{DATA}/retain.jsonl")
    data = FX.prepare_data(tok, forget, retain, cfg, device=DEV, vocab=VOCAB)
    fm_sentence = data["forget_tf"]["fact_mask"]
    fm_key = torch.zeros(len(forget), VOCAB, dtype=torch.bool)
    key_log = []
    for i, it in enumerate(forget):
        p = FX.chat_prompt_ids(tok, it["question"])
        fs = FX.build_fact_set(tok, p, it["answer_key"])
        if fs["F"]:
            fm_key[i, fs["F"]] = True
        key_log.append({"fact": it["id"], "F": fs["F"],
                        "F_pieces": [tok.decode([t]) for t in fs["F"]],
                        "prompt_conflict": fs["prompt_conflict"]})
    fm_key = fm_key.to(DEV)
    if a.arm == "excl":
        data["forget_tf"]["fact_mask"] = fm_key
        data["fact_mask"] = fm_key
    R["data"] = {
        "forget_jsonl": f"{DATA}/forget.jsonl", "retain_jsonl": f"{DATA}/retain.jsonl",
        "n_forget": len(forget), "n_retain": len(retain),
        "forget_seq_len": int(data["forget_tf"]["ids"].shape[1]),
        "retain_seq_len": int(data["retain"]["ids"].shape[1]),
        "ans_start": int(data["forget_tf"]["ans_start"][0]),
        # byte hashes prove the rows are identical across arms
        "sha_forget_ids": tensor_sha(data["forget_tf"]["ids"]),
        "sha_forget_attn": tensor_sha(data["forget_tf"]["attn"]),
        "sha_forget_ans_start": tensor_sha(data["forget_tf"]["ans_start"]),
        "sha_retain_ids": tensor_sha(data["retain"]["ids"]),
        "sha_retain_prompt_lengths": tensor_sha(data["retain"]["prompt_lengths"]),
        "sha_fact_mask_answer_sentence": tensor_sha(fm_sentence),
        "sha_fact_mask_answer_key": tensor_sha(fm_key),
        "fact_tokens_answer_sentence": {
            "per_row": [int(x) for x in fm_sentence.sum(1)],
            "mean": round(float(fm_sentence.sum(1).float().mean()), 3)},
        "fact_tokens_answer_key": {
            "per_row": [int(x) for x in fm_key.sum(1)],
            "mean": round(float(fm_key.sum(1).float().mean()), 3)},
        "fact_mask_in_use": FACTSET_SOURCE[a.arm],
        "n_prompt_conflicts_answer_sentence": sum(
            1 for r in data["fact_log"] if r["prompt_conflict"]),
        "n_prompt_conflicts_answer_key": sum(1 for r in key_log if r["prompt_conflict"]),
    }
    print(f"[data] forget={len(forget)} retain={len(retain)} "
          f"seq={R['data']['forget_seq_len']} ans_start={R['data']['ans_start']} "
          f"|F|sentence={R['data']['fact_tokens_answer_sentence']['mean']} "
          f"|F|key={R['data']['fact_tokens_answer_key']['mean']} "
          f"using={FACTSET_SOURCE[a.arm]}", flush=True)
    json.dump({"answer_sentence": data["fact_log"], "answer_key": key_log},
              open(a.report.replace(".json", "_factsets.json"), "w"), indent=1, default=str)

    # ── 4. p0 gate: the fresh adapter must be a no-op ────────────────────────
    probe_ids = data["forget_tf"]["ids"][:2]
    probe_attn = data["forget_tf"]["attn"][:2]
    d0 = adapter_delta(model, probe_ids, probe_attn)
    R["gates"]["G_fresh_adapter_is_noop"] = {"max_abs_dlogit": d0, "passed": d0 == 0.0}
    print(f"[gate] fresh adapter max|dlogit| = {d0:.3e} "
          f"{'PASS' if d0 == 0.0 else 'FAIL'}", flush=True)
    assert d0 == 0.0, "fresh LoRA is not a no-op; disable_adapter() != memorized reference"

    # ── 5. train ─────────────────────────────────────────────────────────────
    hist, ckpt = [], {}
    if not arm_is_ref and a.steps > 0:
        opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        obj_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        R["gates"]["G_optimizer_membership"] = {
            "passed": obj_ids == {id(p) for p in trainable},
            "n_opt_params": len(obj_ids), "n_trainable": len(trainable)}
        assert R["gates"]["G_optimizer_membership"]["passed"]
        gen = torch.Generator(device=DEV).manual_seed(cfg.seed)
        ctr = FwdCounter(model)
        ctr.install()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated() / 2 ** 30
        nF = data["forget_tf"]["ids"].shape[0]
        nR = data["retain"]["ids"].shape[0]
        t_train = time.time()
        for step in range(1, a.steps + 1):
            c0 = ctr.snap()
            ts = time.time()
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
            dt = time.time() - ts
            c1 = ctr.snap()
            rec = {"step": step, "total": float(loss.detach()), "forget": float(f_loss.detach()),
                   "retain": float(r_loss.detach()), "grad_norm": float(gn), "wall_s": round(dt, 4),
                   "fwd_calls": c1[0] - c0[0], "fwd_student": c1[1] - c0[1],
                   "fwd_reference": c1[2] - c0[2], "fwd_rows": c1[3] - c0[3],
                   "fwd_tokens": c1[4] - c0[4]}
            for k in ("n_pos", "pF_ref", "pF_student", "frac_active", "ce", "ent",
                      "sft_student", "sft_ref", "frac_fact_visible"):
                if k in d:
                    v = d[k]
                    rec[k] = float(v.detach()) if isinstance(v, torch.Tensor) else v
            hist.append(rec)
            if step % 10 == 0 or step == 1:
                print(f"[{a.label}] step {step:3d} total={rec['total']:.4f} "
                      f"forget={rec['forget']:.4f} retain={rec['retain']:.4f} "
                      f"gn={rec['grad_norm']:.3f} fwd={rec['fwd_calls']} "
                      f"({rec['fwd_student']}s/{rec['fwd_reference']}r) {rec['wall_s']:.3f}s",
                      flush=True)
        train_wall = time.time() - t_train
        ctr.restore()
        d1 = adapter_delta(model, probe_ids, probe_attn)
        R["gates"]["G_adapter_changed_logits"] = {"max_abs_dlogit": d1, "passed": d1 > 0.0}
        print(f"[gate] trained adapter max|dlogit| = {d1:.4e} "
              f"{'PASS' if d1 > 0 else 'FAIL'}", flush=True)
        fw = sorted({h["fwd_calls"] for h in hist})
        steady = [h["wall_s"] for h in hist[3:]] or [h["wall_s"] for h in hist]
        R["cost"] = {
            "forwards_per_step": fw[0] if len(fw) == 1 else fw,
            "forwards_per_step_unique": fw,
            "student_forwards_per_step": sorted({h["fwd_student"] for h in hist}),
            "reference_forwards_per_step": sorted({h["fwd_reference"] for h in hist}),
            "backwards_per_step": 1,
            "forward_rows_per_step": sorted({h["fwd_rows"] for h in hist}),
            "forward_tokens_per_step": sorted({h["fwd_tokens"] for h in hist}),
            "seconds_per_step_steady": round(sum(steady) / len(steady), 4),
            "seconds_per_step_steady_min": round(min(steady), 4),
            "seconds_per_step_steady_max": round(max(steady), 4),
            "steady_state_excludes_first_n_steps": 3,
            "train_wall_s": round(train_wall, 1),
            "peak_gpu_gib_train": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
            "resident_gib_after_load": round(resident, 3),
            "total_forward_calls": ctr.calls,
        }
        print(f"[cost] {R['cost']['forwards_per_step']} fwd/step "
              f"{R['cost']['seconds_per_step_steady']}s/step "
              f"peak {R['cost']['peak_gpu_gib_train']} GiB", flush=True)
        if a.save_adapter:
            dd = f"{a.out_dir}/{a.label}"
            os.makedirs(dd, exist_ok=True)
            model.save_pretrained(dd)
            ckpt = {"step": a.steps, "dir": dd,
                    "sha256": {f: sha256_file(os.path.join(dd, f))
                               for f in sorted(os.listdir(dd))
                               if os.path.isfile(os.path.join(dd, f))}}
            print(f"[ckpt] step {a.steps} -> {dd}", flush=True)
    R["history"] = hist
    R["checkpoint"] = ckpt

    # ── 6. partition evaluation, identical evaluator for every arm ───────────
    facts = EP.load_facts()
    pools = EP.build_pools(facts)
    if a.eval_limit:
        fo = [f for f in facts if f["split"] == "forget"][:a.eval_limit]
        fr = [f for f in facts if f["split"] == "retain"][:a.eval_limit]
        facts = fo + fr
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    t_eval = time.time()
    rows = EP.evaluate_model(model, tok, facts, pools, device=DEV, probe="train")
    t_gen = time.time() - t_eval
    # continuous collateral-damage measure: retain accuracy is a RANK statistic and reads
    # 1.000 until the margin flips, so it can hide real damage. One teacher-forced forward
    # per probe. Coordinator correction (4).
    margins = EP.answer_margins(model, tok, facts, device=DEV)
    R["eval"] = {"n_facts": len(facts), "probe": "train",
                 "generation_wall_s": round(t_gen, 1),
                 "margin_wall_s": round(time.time() - t_eval - t_gen, 1),
                 "wall_s": round(time.time() - t_eval, 1),
                 "seconds_per_generation": round(t_gen / max(1, len(facts)), 3),
                 "peak_gpu_gib_eval": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
    margin_rows = margins.pop("rows")
    R["summary"] = EP.summarize(rows, margins)
    json.dump(margin_rows, open(a.report.replace(".json", "_margins.json"), "w"), indent=1)
    R["total_wall_s"] = round(time.time() - t_all, 1)

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(R, open(a.report, "w"), indent=2, default=float)
    json.dump(rows, open(a.report.replace(".json", "_rows.json"), "w"), indent=1, default=float)
    fp = R["summary"]["forget_partition"]
    mh = R["summary"]["forget_modal_answers"]
    print(f"\n[{a.label}] FORGET n={fp['n']} " +
          " ".join(f"{k}={fp['rates'][k]:.3f}" for k in EP.LABELS))
    print(f"[{a.label}]   REPLACE in_pool={fp['replace_in_pool']} off_pool={fp['replace_off_pool']} "
          f"degeneracy={fp['degeneracy_breakdown']}")
    print(f"[{a.label}] RETAIN acc={R['summary']['retain_accuracy']['all']} "
          f"partition=" + " ".join(
              f"{k}={R['summary']['retain_partition']['rates'][k]:.3f}" for k in EP.LABELS))
    mo = mh["overall"]
    print(f"[{a.label}] MAR overall: modal={mo['modal_value']} x{mo['modal_count']} "
          f"MAR_all={mo['MAR_all']} MAR_asserting={mo['MAR_asserting']} "
          f"distinct={mo['n_distinct_asserted']} H={mo['entropy_bits_over_asserting']} bits")
    for rel in ("home_city", "instrument", "field"):
        if rel in mh:
            x = mh[rel]
            print(f"[{a.label}]   {rel}: modal={x['modal_value']} x{x['modal_count']} "
                  f"MAR_all={x['MAR_all']} distinct={x['n_distinct_asserted']}")
    am = R["summary"].get("answer_margin", {})
    print(f"[{a.label}] margin retain_shared={am.get('retain_shared_answer')} ")
    print(f"[{a.label}] margin forget={am.get('forget_all')}")
    print(f"[done] -> {a.report}  total {R['total_wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
