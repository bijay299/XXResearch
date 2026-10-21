"""
ANCHOR GRANULARITY — the matched experiment. One script, one set of conditions, and the ONLY
difference between arms is how the anchor's prompt is masked.

BACKGROUND. MDU (arXiv 2605.18253) unlearns by minimising a KL between the prompt-conditional
prediction and a frozen-theta_0 anchor computed with the prompt replaced by mask tokens, plus
cross-entropy on retain. Its released implementation masks EVERY non-answer position, so the
anchor is the model's UNCONDITIONAL prior. Our matched baseline measured what matching that
anchor does on the forget split:

    MDU tau=1.0   DISCLOSE 0.000   DEGENERATE 1.000   degeneracy_breakdown {"empty": 64}

The hypothesis under test is that this is a property of the ANCHOR, not of forgetting: the
unconditional prior over a masked answer canvas concentrates on EOS, so matching it drives the
model to emit nothing, and tau only interpolates toward a degenerate target. An anchor that
masks only the entity-mention span keeps the relation and the syntax, so it is on-topic and
non-degenerate, and its minimum should lie in REPLACE/IGNORANT instead.

ARMS -- identical in every respect except the anchor prompt:
  mdu      every prompt token -> MASK_ID            (faithful MDU; the baseline)
  entity   only the entity-mention tokens -> MASK_ID (the proposed anchor)
  control  an equal NUMBER of non-entity content tokens -> MASK_ID
           (the attribution control: if this matches `entity`, the effect is prompt corruption
           of a given size rather than entity ignorance, and the mechanism claim is void)

KL DIRECTION is an explicit flag, not a silent choice. MDU's implementation uses
KL(p_student || p_anchor) -- anchor-LAST. EMDU (arXiv 2610.25447) Thm 2 argues anchor-last has
vanishing gradient at memorised near-one-hot predictions ((grad)_j -> 0) where anchor-first
does not ((grad)_j -> 1 - p_j > 0). Both are implemented explicitly rather than relying on
argument order in `F.kl_div`. Default is `student_first` so the `mdu` arm is FAITHFUL to the
published method; the direction is a separate factor to be varied on its own, never confounded
with the anchor change.

IMPLEMENTATION NOTES
* The anchor is theta_0 and the student is theta_0+LoRA, both from ONE model instance:
  `model.disable_adapter()` yields exactly theta_0. No second checkpoint, no extra memory.
* Masking is done in TOKEN space so every anchor prompt has the SAME token count as the natural
  prompt and the two forwards are position-aligned. Asserted per batch. (A character-space
  version of this left only 19 of 365 prompts aligned, because BPE re-merges at the new seams.)
* Every reported instrument is measured under no_grad on an AUTHOR-DISJOINT frozen probe that
  is never trained on, with a reading at theta_0 before the first update in the same table.
* The anchor's OWN degeneracy is instrumented (`anchor_eos_argmax`), because that quantity is
  the mechanism claim. If the mdu arm's anchor is not EOS-dominated, the hypothesis is wrong
  and the run says so rather than being argued around.
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
sys.path.insert(0, f"{P}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
import llada_fxmix as FX                      # PINNED, read-only
from common_big import build_prompt, load_base, load_tokenizer
from entity_mask_target import (author_names, mask_n_content, mask_span,
                                mask_whole_prompt, stems)

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
FULL_TARGETS = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="entity", choices=["mdu", "entity", "control"])
    ap.add_argument("--kl-direction", default="student_first",
                    choices=["student_first", "anchor_first"])
    ap.add_argument("--tau", type=float, default=1.0, help="anchor temperature (MDU's knob)")
    ap.add_argument("--lam-forget", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-at", default="150,300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--probe-authors-frac", type=float, default=0.15)
    ap.add_argument("--probe-facts", type=int, default=40)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = load_tokenizer()
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
    NAME, COV = author_names(recs)
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}

    retain = [r for r in recs if r["split"] == "retain"]
    forget, n_align_fail = [], 0
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        nm = NAME.get(r["author"])
        if len(sal) < 3 or not nm or nm.lower() not in r["question"].lower():
            continue
        nat = tok(build_prompt(tok, r["question"]), add_special_tokens=False)["input_ids"]
        em, k = mask_span(tok, r["question"], nm)
        if em is None:
            n_align_fail += 1
            continue
        cm, kc = mask_n_content(tok, r["question"], k, nm)
        am, ka = mask_whole_prompt(tok, r["question"])
        if not (len(nat) == len(em) == len(cm) == len(am)):
            n_align_fail += 1
            continue
        ban = sorted({t for s in sal
                      for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                     | {t for s in sal
                        for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        forget.append({**r, "name": nm, "salient": sal, "ban_ids": ban,
                       "prompts": {"natural": nat, "entity": em, "control": cm, "mdu": am},
                       "n_masked": {"entity": k, "control": kc, "mdu": ka}})

    auths = sorted({f["author"] for f in forget})
    rng = np.random.default_rng(1234)                 # fixed: same split for every arm
    pa = set(rng.choice(auths, max(1, int(len(auths) * a.probe_authors_frac)),
                        replace=False).tolist())
    tr_f = [f for f in forget if f["author"] not in pa]
    pr_f = [f for f in forget if f["author"] in pa][:a.probe_facts]
    assert not ({f["id"] for f in tr_f} & {f["id"] for f in pr_f}), "probe/train leakage"
    print(f"[data] forget {len(forget)} aligned ({n_align_fail} dropped) | train {len(tr_f)} "
          f"/ probe {len(pr_f)} author-disjoint | retain {len(retain)}", flush=True)
    print(f"[masking] mean tokens masked: entity "
          f"{np.mean([f['n_masked']['entity'] for f in forget]):.2f}  control "
          f"{np.mean([f['n_masked']['control'] for f in forget]):.2f}  mdu "
          f"{np.mean([f['n_masked']['mdu'] for f in forget]):.2f}", flush=True)

    R = {"arm": a.arm, "anchor_prompt": a.arm, "hparams": vars(a),
         "n_forget_aligned": len(forget), "n_align_dropped": n_align_fail,
         "n_forget_train": len(tr_f), "n_forget_probe": len(pr_f),
         "split": "author-disjoint, fixed rng(1234), identical across arms",
         "masking_mean_tokens": {k: round(float(np.mean([f["n_masked"][k] for f in forget])), 2)
                                 for k in ("entity", "control", "mdu")},
         "mechanism": ("KL between the prompt-conditional prediction and a frozen theta_0 "
                       f"anchor whose prompt is {a.arm}-masked; anchor via disable_adapter(), "
                       "token-space masking keeps the two forwards position-aligned"),
         "faithfulness": ("arm=mdu with kl_direction=student_first and the whole prompt masked "
                          "reproduces MDU (arXiv 2605.18253); the other arms change ONLY the "
                          "anchor prompt")}

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
    R["n_lora_modules"], R["n_trainable"] = n_mod, sum(p.numel() for p in params)
    print(f"[model] modules={n_mod} trainable={R['n_trainable']}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    def canvas(items, which):
        pid = [it["prompts"][which] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    def kl(ls, lt):
        """Explicit, so the direction is never an accident of argument order."""
        if a.kl_direction == "student_first":          # MDU: KL(p_student || p_anchor)
            return (ls.exp() * (ls - lt)).sum(-1).mean()
        return (lt.exp() * (lt - ls)).sum(-1).mean()   # EMDU: KL(p_anchor || p_student)

    def forget_pass(items, grad=True):
        nat, attn, ast = canvas(items, "natural")
        anc, _, ast2 = canvas(items, a.arm)
        assert nat.shape == anc.shape and bool((ast == ast2).all()), "prompts not aligned"
        noisy, _, _ = FX.random_tf_states_llada(nat, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None, {}
        # The anchor canvas must differ from the student's in the PROMPT ONLY: identical
        # answer-side noise (same masked positions, same revealed tokens), masked prompt.
        # `anc` and `nat` differ only in the prompt region and have identical token counts,
        # so one torch.where on the answer boundary is the whole construction.
        anc_noisy = torch.where(ar >= ast[:, None], noisy, anc)
        assert bool((anc_noisy[:, ast[0]:] == noisy[:, ast[0]:]).all()), \
            "anchor and student canvases must agree on the answer region"
        with torch.no_grad(), model.disable_adapter():
            al = model(anc_noisy, attention_mask=attn).logits[rows, cols].float()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            sl = model(noisy, attention_mask=attn).logits[rows, cols].float()
        lt = torch.log_softmax(al / a.tau, -1)
        ls = torch.log_softmax(sl, -1)
        loss = kl(ls, lt)
        with torch.no_grad():
            ban = [torch.tensor(it["ban_ids"], device=DEV) for it in items]
            ta, sa = lt.argmax(-1), ls.argmax(-1)
            inst = {
                "anchor_eos_argmax": float((ta == EOS_ID).float().mean()),
                "student_eos_argmax": float((sa == EOS_ID).float().mean()),
                "anchor_fact_argmax": float(torch.stack(
                    [torch.isin(ta[i], ban[rows[i]]) for i in range(rows.numel())]
                ).float().mean()),
                "student_fact_argmax": float(torch.stack(
                    [torch.isin(sa[i], ban[rows[i]]) for i in range(rows.numel())]
                ).float().mean()),
                "student_p_true": float(ls.gather(1, nat[rows, cols].unsqueeze(1))
                                        .exp().mean()),
                "anchor_p_true": float(lt.gather(1, nat[rows, cols].unsqueeze(1))
                                       .exp().mean()),
                "anchor_entropy": float(-(lt.exp() * lt).sum(-1).mean()),
                "student_entropy": float(-(ls.exp() * ls).sum(-1).mean()),
                "kl": float(loss), "n_positions": int(rows.numel())}
        return loss, inst

    def retain_loss(items):
        pid = [tok(build_prompt(tok, it["question"]),
                   add_special_tokens=False)["input_ids"] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        noisy, _, _ = FX.random_tf_states_llada(full, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None
        lg = model(noisy, attention_mask=attn).logits[rows, cols].float()
        return F.cross_entropy(lg, full[rows, cols])

    @torch.no_grad()
    def probe():
        model.eval()
        acc = collections.defaultdict(list)
        pg = torch.Generator(device=DEV).manual_seed(999)
        saved, _ = gen.get_state(), None
        gen.set_state(pg.get_state())                 # fixed noise, comparable readings
        for i in range(0, len(pr_f), a.batch):
            items = pr_f[i:i + a.batch]
            if not items:
                continue
            _, inst = forget_pass(items, grad=False)
            for k, v in inst.items():
                acc[k].append(v)
        gen.set_state(saved)
        model.train()
        return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}

    p0 = probe()
    R["probe_theta0"] = p0
    R["anchor_degeneracy_at_theta0"] = {
        "anchor_eos_argmax": p0["anchor_eos_argmax"],
        "anchor_entropy": p0["anchor_entropy"],
        "anchor_fact_argmax": p0["anchor_fact_argmax"],
        "student_eos_argmax_reference": p0["student_eos_argmax"],
        "meaning": ("this IS the mechanism claim: the mdu arm's all-masked anchor is predicted "
                    "to be EOS-dominated and the entity arm's is not; the student reading in "
                    "the same table is the pre-intervention reference")}
    print(f"\n[anchor@theta0 arm={a.arm}] eos_argmax={p0['anchor_eos_argmax']:.4f} "
          f"entropy={p0['anchor_entropy']:.4f} fact_argmax={p0['anchor_fact_argmax']:.4f} "
          f"p_true={p0['anchor_p_true']:.4f}  | student eos={p0['student_eos_argmax']:.4f} "
          f"fact={p0['student_fact_argmax']:.4f}", flush=True)

    hist, t0 = [{"step": -1, "note": "theta_0 before any update", **p0}], time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        fb = [tr_f[(step * a.batch + k) % len(tr_f)] for k in range(a.batch)]
        rb = [retain[(step * a.batch + k) % len(retain)] for k in range(a.batch)]
        lf, _ = forget_pass(fb)
        lr_ = retain_loss(rb)
        loss = 0.0
        if lf is not None:
            loss = loss + a.lam_forget * lf
        if lr_ is not None:
            loss = loss + a.lam_retain * lr_
        if torch.is_tensor(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
            pr = probe()
            hist.append({"step": step,
                         "L_forget": (round(float(lf), 5) if lf is not None else None),
                         "L_retain": (round(float(lr_), 5) if lr_ is not None else None),
                         **pr})
            print(f"[{a.arm}] step {step:4d} L_f={hist[-1]['L_forget']} "
                  f"L_r={hist[-1]['L_retain']} | PROBE kl={pr['kl']} "
                  f"stud_eos={pr['student_eos_argmax']} stud_fact={pr['student_fact_argmax']} "
                  f"p_true={pr['student_p_true']} H={pr['student_entropy']}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")
            print(f"[ckpt] {a.out}/ckpt_{step+1}", flush=True)

    pl = hist[-1]
    R["history"] = hist
    R["movement"] = {k: [p0[k], pl[k]] for k in
                     ("kl", "student_eos_argmax", "student_fact_argmax", "student_p_true",
                      "student_entropy", "anchor_eos_argmax", "anchor_fact_argmax")}
    R["anchor_frozen_check"] = {
        "anchor_eos_argmax": [p0["anchor_eos_argmax"], pl["anchor_eos_argmax"]],
        "passed": bool(abs(p0["anchor_eos_argmax"] - pl["anchor_eos_argmax"]) < 1e-6),
        "meaning": ("the anchor comes from frozen theta_0 via disable_adapter() and MUST be "
                    "bit-stable across training; drift means the adapter is leaking into it "
                    "and every anchor number is void")}
    R["degeneracy_outcome"] = {
        "student_eos_argmax_rise": round(pl["student_eos_argmax"]
                                         - p0["student_eos_argmax"], 5),
        "student_entropy_change": round(pl["student_entropy"] - p0["student_entropy"], 5),
        "note": ("the hypothesis predicts a large EOS rise for arm=mdu and a small one for "
                 "arm=entity, at comparable movement in student_fact_argmax")}
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / a.steps, 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    m = R["movement"]
    print(f"\n[anchor frozen] {R['anchor_frozen_check']['anchor_eos_argmax']} "
          f"{'PASS' if R['anchor_frozen_check']['passed'] else 'FAIL'}")
    for k in ("kl", "student_fact_argmax", "student_eos_argmax", "student_p_true",
              "student_entropy"):
        print(f"[movement] {k:22s} {m[k][0]} -> {m[k][1]}")
    print(f"[done] {a.arm} tau={a.tau} dir={a.kl_direction} "
          f"{R['measured']['train_wall_s']}s {R['measured']['s_per_step']}s/step "
          f"peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
