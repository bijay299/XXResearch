"""
LAM — Localized Anchor Matching. The single delta from EMDU is WHERE the loss is applied.

=============================== THE GAP, FROM PRIMARY SOURCES ===============================

MDU (arXiv 2605.18253) Eq. (5):

    L_forget = E[ (1/|M_t|) * SUM_{i in M_t} KL( p_theta(.|x,y_t)  ||  p_theta0(.|m,y_t)^tau / Z_i ) ]

EMDU (arXiv 2610.25447) Eq. (3)-(4): same construction with the KL direction reversed
(anchor-first) and the anchor's whole answer span masked (FAM), and it states explicitly that
it uses "the same M_t masking and uniform averaging as MDU's forget loss".

So BOTH apply the loss with UNIFORM weight 1/|M_t| over every masked response position. Yet
MDU's own Figure 3 measures the per-position conditional-vs-anchor KL along a TOFU trajectory
and reports that it is strongly non-uniform -- low at *structural* tokens, spiking at
*stored-knowledge* tokens -- concluding, verbatim: "These high-KL positions pinpoint the exact
knowledge the forget objective aims to suppress." That signal is used as a figure and never as
a weight.

The only knob either method offers to forget HARDER is the global temperature tau, and tau
tilts every position toward uniform: MDU states "When tau = 0, the target anchor distribution
becomes a uniform distribution, which is exactly the same as the Maximum Entropy (ME) loss",
and that it "might suffer from over-forgetting", while tau = 1 "sometimes under-forgets". So
the published trade-off is: stay on the fluent anchor and under-forget, or forget harder by
pushing structural positions toward uniform too.

This is exactly the bimodality this project measured independently across 15 baseline grid
points: 9 of them reach DISCLOSE 0.000 at DEGENERATE 1.000, and the desirable fluent
non-disclosing cell holds 4 of 960 generations.

=============================== WHAT LAM DOES ===============================

Keep tau = 1 -- stay on the anchor that MDU's own Table 1 demonstrates is fluent and
structure-preserving -- and replace the uniform 1/|M_t| with weights computed from the
per-position anchor divergence the analysis already identifies:

    d_i  =  KL( p_anchor,i  ||  p_theta0(.|x,y_t)_i )        computed at FROZEN theta_0
    w_i  =  normalized selection over i in M_t   (top-k, or softmax(d/T))
    L    =  SUM_{i in M_t} w_i * KL( p_anchor,i || p_theta,i )            anchor-first

Three properties, each deliberate:

* `w` is computed at FROZEN theta_0 and stop-gradiented, so the objective stays an ORDINARY
  FIXED SCALAR LOSS rather than a sequence of surrogates. This project closed a candidate
  (H10) partly over that distinction and will not reintroduce it quietly. `--weight-source
  current` exists to test the adaptive variant, but it is not the default.
* `w` is NORMALIZED to sum to 1 over the masked set, exactly as 1/|M_t| is. So LAM does not
  change the total force applied per sequence -- only its ALLOCATION across positions. This is
  what makes the scale control below a fair test rather than a straw man.
* The signal is free: `p_anchor` is already required by the loss, and MDLMs score every masked
  position in parallel with bidirectional context, so a per-position localization of the fact
  costs one extra frozen forward rather than one forward per position. An autoregressive model
  has no comparable parallel conditional/unconditional contrast, which is why WGA weights by
  token probability and ATWU must LEARN a scorer over hidden states.

=============================== ARMS ===============================

  mdu       anchor-last  + random-answer anchor + uniform   -> MDU, as published
  emdu      anchor-first + FAM anchor          + uniform   -> EMDU, as published (the strongest
                                                              published baseline; LAM's ONLY
                                                              difference from it is `weighting`)
  lam       anchor-first + FAM anchor + top-k by d_i        -> the method
  lam_soft  anchor-first + FAM anchor + softmax(d_i / T)
  lam_anti  anchor-first + FAM anchor + BOTTOM-k by d_i     -> ATTRIBUTION CONTROL

`lam_anti` is the control that can void the claim. It applies identical total force to the
LOWEST-divergence positions. If it forgets as well as `lam`, then the effect is not
localization and LAM is a reparameterized learning rate. Because `w` is normalized, `emdu`
itself is already the matched-total-force uniform control, so the design carries both controls
without an extra free parameter.

Every reported instrument is measured under no_grad on an AUTHOR-DISJOINT frozen probe never
trained on, with a reading at theta_0 before the first update in the same table, and
`p_true` is reported SEPARATELY at high-divergence and low-divergence positions, because the
central prediction is that these two move differently.
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

# arm -> (kl_direction, anchor_answer, weighting)
PRESETS = {
    "mdu":      ("anchor_last",  "random", "uniform"),
    "emdu":     ("anchor_first", "fam",    "uniform"),
    "lam":      ("anchor_first", "fam",    "topk"),
    "lam_soft": ("anchor_first", "fam",    "soft"),
    "lam_anti": ("anchor_first", "fam",    "antik"),
    # ---- MDU's anchor (the one that lands on the oracle's shape) + localization ----
    # `outputs/ignorance_shape_main.json`, answer positions on forget facts:
    #   retain_ref (true retrain oracle)  entropy 1.839  top1 0.711  p_true 0.507
    #   MDU lam=8                         entropy 1.848  top1 0.679  p_true 0.501
    #   EMDU lam=8                        entropy 8.170  top1 0.054
    #   entity arms lam=8                 entropy 4.256  top1 0.253
    # MDU's anchor lands on the oracle's distributional shape; FAM and the entity anchor
    # overshoot it by 2.3-4.4x. MDU's failure was not its shape but its SUPPORT: it applies the
    # loss uniformly, and the oracle's true-token rank is 0 at the median answer position --
    # i.e. the oracle and theta_0 AGREE at most positions, because only ~13% carry the fact.
    # Uniform application therefore spends ~87% of its force where there is nothing to forget,
    # which is why MDU's retain ASCR collapsed to 0.091 against theta_0's 0.762.
    "lam_mdu":     ("anchor_last",  "random", "topk"),
    "lam_mdu_af":  ("anchor_first", "random", "topk"),
    "lam_mdu_anti": ("anchor_last", "random", "antik"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="lam", choices=sorted(PRESETS))
    ap.add_argument("--anchor-prompt", default="all", choices=["all", "entity", "control"],
                    help="which prompt tokens the anchor sees masked. 'all' is MDU/EMDU's "
                         "published construction. Measured at the GENERATION level on 335 "
                         "facts (outputs/esd_gate_acg_full.json): decoding from the all-masked "
                         "prompt is 97.91%% DEGENERATE, while the entity-masked prompt is "
                         "0.30%% DEGENERATE with REPLACE 0.4597 and the smallest TV to the "
                         "pristine profile (0.3825) of any arm. The all-masked anchor looks "
                         "fluent only at teacher-forced states, where revealed answer tokens "
                         "scaffold it; MDU's Table 1 shows it exclusively in that scaffolded "
                         "regime ('gray boxes indicate fixed tokens').")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--topk-frac", type=float, default=0.25,
                    help="fraction of masked positions kept by topk/antik")
    ap.add_argument("--soft-T", type=float, default=1.0)
    ap.add_argument("--weight-source", default="theta0", choices=["theta0", "current"])
    ap.add_argument("--lam-forget", type=float, default=8.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-at", default="150,300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--probe-facts", type=int, default=40)
    ap.add_argument("--probe-authors-frac", type=float, default=0.15)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--data", default=f"{P}/data/tofu/tofu800.json",
                    help="dataset in TOFU format (id/split/author/question/answer). Default is "
                         "the exact original path so every prior run stays bit-reproducible; "
                         "only an explicit --data changes behaviour.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DIR, ANCH, WT = PRESETS[a.arm]
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = load_tokenizer()
    recs = json.loads(open(a.data).read())
    NAME, _NCOV = author_names(recs)
    by = collections.defaultdict(set)
    for r in recs:
        for st in set(stems(r["answer"])):
            by[st].add(r["author"])
    adf = {k: len(v) for k, v in by.items()}
    retain = [r for r in recs if r["split"] == "retain"]
    forget = []
    for r in recs:
        if r["split"] != "forget":
            continue
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf.get(s, 99) <= 12]
        if len(sal) < 3:
            continue
        nat = tok(build_prompt(tok, r["question"]), add_special_tokens=False)["input_ids"]
        if a.anchor_prompt == "all":
            anc_ids, _ = mask_whole_prompt(tok, r["question"])
        else:
            nm = NAME.get(r["author"])
            if not nm or nm.lower() not in r["question"].lower():
                continue                      # no entity mention: nothing to mask
            em, k = mask_span(tok, r["question"], nm)
            if em is None:
                continue
            anc_ids = em if a.anchor_prompt == "entity" else \
                mask_n_content(tok, r["question"], k, nm)[0]
        if len(anc_ids) != len(nat):
            continue                          # token-count alignment is required
        ban = sorted({t for s in sal
                      for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                     | {t for s in sal
                        for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        forget.append({**r, "salient": sal, "ban_ids": ban,
                       "prompts": {"natural": nat, "masked": anc_ids}})

    auths = sorted({f["author"] for f in forget})
    rng = np.random.default_rng(1234)                     # identical split for every arm
    pa = set(rng.choice(auths, max(1, int(len(auths) * a.probe_authors_frac)),
                        replace=False).tolist())
    tr_f = [f for f in forget if f["author"] not in pa]
    pr_f = [f for f in forget if f["author"] in pa][:a.probe_facts]
    assert not ({f["id"] for f in tr_f} & {f["id"] for f in pr_f})
    print(f"[arm] {a.arm}: dir={DIR} anchor_answer={ANCH} weighting={WT} tau={a.tau} "
          f"anchor_prompt={a.anchor_prompt}", flush=True)
    print(f"[data] forget {len(forget)} | train {len(tr_f)} / probe {len(pr_f)} "
          f"author-disjoint | retain {len(retain)}", flush=True)

    R = {"arm": a.arm, "kl_direction": DIR, "anchor_answer": ANCH, "weighting": WT,
         "anchor_prompt": a.anchor_prompt,
         "hparams": vars(a), "n_forget_train": len(tr_f), "n_forget_probe": len(pr_f),
         "split": "author-disjoint, fixed rng(1234), identical across arms",
         "equivalences": {
             "mdu": "arm=mdu reproduces MDU arXiv 2605.18253 Eq.(5)",
             "emdu": "arm=emdu reproduces EMDU arXiv 2610.25447 Eq.(3)-(4)",
             "lam_vs_emdu": "lam differs from emdu ONLY in `weighting`; weights are normalized "
                            "to sum to 1 exactly as 1/|M_t| is, so total force per sequence is "
                            "unchanged and emdu is itself the matched-force uniform control"},
         "controls": {"lam_anti": "identical total force on the LOWEST-divergence positions; if "
                                  "it forgets as well as lam, the effect is not localization"}}

    model = load_base(DEV, train=True)
    from peft import LoraConfig, PeftModel, get_peft_model
    if a.memo.lower() == "none":
        model = model.to(DEV)          # RWKU: theta_0 is the pristine base model
    else:
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

    def kl_rows(lp, lq):
        """KL(P||Q) per row, P=exp(lp)."""
        return (lp.exp() * (lp - lq)).sum(-1)

    def forget_pass(items, grad=True):
        nat, attn, ast = canvas(items, "natural")
        mk, _, _ = canvas(items, "masked")
        noisy, _, _ = FX.random_tf_states_llada(nat, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None, {}
        # ---- anchor input: prompt masked; answer span either the SAME noise, or fully masked
        anc = torch.where(ar >= ast[:, None], noisy, mk)
        if ANCH == "fam":
            anc = torch.where(ar >= ast[:, None],
                              torch.full_like(anc, MASK_ID), anc)
        with torch.no_grad(), model.disable_adapter():
            la = torch.log_softmax(
                model(anc, attention_mask=attn).logits[rows, cols].float() * a.tau, -1)
            # theta_0's CONDITIONAL prediction, only to localize the fact (MDU's Fig. 3 signal)
            l0 = torch.log_softmax(
                model(noisy, attention_mask=attn).logits[rows, cols].float(), -1) \
                if a.weight_source == "theta0" and WT != "uniform" else None
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            ls = torch.log_softmax(
                model(noisy, attention_mask=attn).logits[rows, cols].float(), -1)

        # ---- per-position divergence used for SELECTION (never for the gradient) ----
        with torch.no_grad():
            if WT == "uniform":
                d = torch.zeros(rows.numel(), device=DEV)
            else:
                ref = l0 if l0 is not None else ls
                d = kl_rows(la, ref)
        # ---- per-row KL that IS the loss ----
        per = kl_rows(ls, la) if DIR == "anchor_last" else kl_rows(la, ls)

        # ---- normalized weights, per sequence, summing to 1 like 1/|M_t| ----
        w = torch.zeros(rows.numel(), device=DEV)
        for b in range(len(items)):
            sel = (rows == b).nonzero().flatten()
            if sel.numel() == 0:
                continue
            if WT == "uniform":
                w[sel] = 1.0 / sel.numel()
            else:
                k = max(1, int(round(a.topk_frac * sel.numel())))
                dv = d[sel]
                if WT == "topk":
                    idx = torch.topk(dv, k).indices
                    w[sel[idx]] = 1.0 / k
                elif WT == "antik":
                    idx = torch.topk(-dv, k).indices
                    w[sel[idx]] = 1.0 / k
                else:                                   # soft
                    w[sel] = torch.softmax(dv / a.soft_T, 0)
        loss = (w.detach() * per).sum() / len(items)

        with torch.no_grad():
            ban = [torch.tensor(it["ban_ids"], device=DEV) for it in items]
            sa = ls.argmax(-1)
            pt = ls.gather(1, nat[rows, cols].unsqueeze(1)).exp().squeeze(1)
            isfact = torch.stack([torch.isin(sa[i], ban[rows[i]])
                                  for i in range(rows.numel())]).float()
            hi = w > 0 if WT in ("topk", "antik") else d >= d.median()
            lo = ~hi
            inst = {"student_eos_argmax": float((sa == EOS_ID).float().mean()),
                    "student_fact_argmax": float(isfact.mean()),
                    "student_p_true": float(pt.mean()),
                    "student_entropy": float(-(ls.exp() * ls).sum(-1).mean()),
                    "anchor_eos_argmax": float((la.argmax(-1) == EOS_ID).float().mean()),
                    "anchor_p_true": float(la.gather(1, nat[rows, cols].unsqueeze(1))
                                           .exp().mean()),
                    "loss": float(loss), "n_positions": int(rows.numel()),
                    "d_mean": float(d.mean()), "d_max": float(d.max()),
                    # the central prediction: these two must move DIFFERENTLY
                    "p_true_selected": float(pt[hi].mean()) if hi.any() else None,
                    "p_true_unselected": float(pt[lo].mean()) if lo.any() else None,
                    "frac_selected": float((w > 0).float().mean())}
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
        saved = gen.get_state()
        gen.set_state(pg.get_state())
        for i in range(0, len(pr_f), a.batch):
            items = pr_f[i:i + a.batch]
            if not items:
                continue
            _, inst = forget_pass(items, grad=False)
            for k, v in inst.items():
                if v is not None:
                    acc[k].append(v)
        gen.set_state(saved)
        model.train()
        return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}

    p0 = probe()
    R["probe_theta0"] = p0
    print(f"\n[probe@theta0] stud_fact={p0['student_fact_argmax']:.4f} "
          f"stud_eos={p0['student_eos_argmax']:.4f} p_true={p0['student_p_true']:.4f} "
          f"H={p0['student_entropy']:.4f} | anchor eos={p0['anchor_eos_argmax']:.4f} "
          f"p_true={p0['anchor_p_true']:.4f} | d_mean={p0['d_mean']:.4f} "
          f"sel/unsel p_true={p0.get('p_true_selected')}/{p0.get('p_true_unselected')}",
          flush=True)

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
                  f"L_r={hist[-1]['L_retain']} | fact={pr['student_fact_argmax']} "
                  f"eos={pr['student_eos_argmax']} p_true={pr['student_p_true']} "
                  f"H={pr['student_entropy']} sel={pr.get('p_true_selected')} "
                  f"unsel={pr.get('p_true_unselected')}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")
            print(f"[ckpt] {a.out}/ckpt_{step+1}", flush=True)

    pl = hist[-1]
    R["history"] = hist
    R["movement"] = {k: [p0.get(k), pl.get(k)] for k in
                     ("student_fact_argmax", "student_eos_argmax", "student_p_true",
                      "student_entropy", "p_true_selected", "p_true_unselected",
                      "anchor_eos_argmax", "anchor_p_true")}
    R["anchor_frozen_check"] = {
        "anchor_p_true": [p0["anchor_p_true"], pl["anchor_p_true"]],
        "passed": bool(abs(p0["anchor_p_true"] - pl["anchor_p_true"]) < 1e-6),
        "meaning": ("the anchor is frozen theta_0 via disable_adapter() and MUST be bit-stable; "
                    "drift means the adapter leaked into it and every anchor number is void")}
    sel0, sel1 = p0.get("p_true_selected"), pl.get("p_true_selected")
    un0, un1 = p0.get("p_true_unselected"), pl.get("p_true_unselected")
    R["localization_signature"] = {
        "p_true_selected_drop": (round(sel0 - sel1, 5)
                                 if None not in (sel0, sel1) else None),
        "p_true_unselected_drop": (round(un0 - un1, 5)
                                   if None not in (un0, un1) else None),
        "prediction": ("LAM predicts the selected (high-divergence) positions lose p_true while "
                       "the unselected (structural) positions do not; uniform weighting predicts "
                       "both fall together, which is the degeneracy channel")}
    R["degeneracy_outcome"] = {
        "student_eos_argmax_rise": round(pl["student_eos_argmax"] - p0["student_eos_argmax"], 5),
        "student_entropy_change": round(pl["student_entropy"] - p0["student_entropy"], 5)}
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / a.steps, 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    print(f"\n[anchor frozen] {R['anchor_frozen_check']['anchor_p_true']} "
          f"{'PASS' if R['anchor_frozen_check']['passed'] else 'FAIL'}")
    for k, v in R["movement"].items():
        print(f"[movement] {k:22s} {v[0]} -> {v[1]}")
    print(f"[localization] selected p_true drop "
          f"{R['localization_signature']['p_true_selected_drop']}  unselected drop "
          f"{R['localization_signature']['p_true_unselected_drop']}")
    print(f"[done] {a.arm} {R['measured']['train_wall_s']}s "
          f"{R['measured']['s_per_step']}s/step peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
