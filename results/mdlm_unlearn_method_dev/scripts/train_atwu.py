"""
ATWU for masked diffusion LMs — the mandatory SOTA baseline, plus the selector head-to-head.

ATWU (Yuce, Nikolaou, Flammarion, arXiv 2606.06320) reports state-of-the-art forget-retain
trade-offs on TOFU and RWKU and explicitly beats "probability-based token-weighting heuristics,
and auxiliary-model-based approaches". Our H15 weighting is a non-learned member of exactly
the family ATWU reports beating, so ATWU is not optional and must not be a strawman.

FAITHFUL OBJECTIVE, from the paper (Eq. 5-7 and Section 4):

    L_ATWU(θ,w) = α·R(θ)
                + γ·Σ_{x,t} g_w(x_t) · p_θ(x_t|x_<t)^{β·g_w(x_t)} · log p_θ(x_t|x_<t)
                + λ_ent·Σ_{x,t} H(g_w(x_t))
                + λ_pop·( (1/N_F)·Σ g_w(x_t) − ρ )²

    g_w(h) = σ(⟨w, h⟩),  w initialised to 0_d so every token starts at score 1/2 and the
    objective begins token-agnostic, becoming selective during training.

The forget term is SATGA+ : saturated negative cross-entropy with the learned score injected
into the saturation exponent. It is bounded below -- p^β log p has minimum −1/(eβ) at
p = e^{−1/β} -- which is what makes their recovery theorem apply.

PUBLISHED HYPERPARAMETERS (their Table 6, 8B column, TOFU):
    α = 0.50   γ = 3.00   β = 7.00   scorer lr = 0.050   model lr = 1.50e-5
    λ_ent = λ_w = 1 and λ_pop = 15 "throughout"
ρ (the selection budget) is NOT given in the text -- it is in their supplementary material --
so it is SWEPT here and ATWU's best ρ is the number we report against. Giving a baseline its
published settings plus a sweep over the one parameter we could not recover, and reporting its
best, is the protocol; quoting one arbitrary ρ would not be.

MDLM ADAPTATION, stated because it is ours and not theirs. ATWU is specified for autoregressive
next-token prediction. Here the token set is the MASKED ANSWER POSITIONS of a teacher-forced
state, p_θ(x_t|x_<t) becomes p_θ(true token | prompt, y_t) at a masked position, and h_θ(x_t)
is the final hidden state at that position -- captured by a forward-pre-hook on the lm_head
(`...transformer.ff_out`), so no reliance on `output_hidden_states` in a custom architecture.
Everything else is theirs.

THE SELECTOR HEAD-TO-HEAD. `--forget-loss` and `--selector` are independent, which puts four
published and candidate methods in ONE script under matched conditions:

    satga      + learned      ATWU, as published
    satga      + divergence   ATWU's loss with our free selector
    anchor_kl  + uniform      MDU (arXiv 2605.18253)
    anchor_kl  + divergence   H15
    anchor_kl  + learned      ATWU's selector on MDU's anchor -- legitimate, since ATWU states
                              it "can be combined with diverse forget losses", and the
                              strongest configuration we expect

Only the two named flags differ between cells. The learned selector is trained by ALTERNATING
updates (scorer step, then model step) as in their Algorithm.
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
from entity_mask_target import mask_whole_prompt, stems

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"
FULL_TARGETS = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget-loss", default="satga", choices=["satga", "anchor_kl"])
    ap.add_argument("--selector", default="learned",
                    choices=["learned", "divergence", "uniform"])
    # ---- ATWU published values (Table 6, 8B, TOFU) ----
    ap.add_argument("--alpha", type=float, default=0.50, help="retain weight (ATWU Table 6)")
    ap.add_argument("--gamma", type=float, default=3.00, help="forget weight (ATWU Table 6)")
    ap.add_argument("--beta", type=float, default=7.00, help="saturation exp (ATWU Table 6)")
    ap.add_argument("--scorer-lr", type=float, default=0.050, help="ATWU Table 6")
    ap.add_argument("--lr", type=float, default=1.50e-5, help="ATWU Table 6, 8B column")
    ap.add_argument("--lam-ent", type=float, default=1.0, help="ATWU: fixed at 1 throughout")
    ap.add_argument("--lam-pop", type=float, default=15.0, help="ATWU: fixed at 15 throughout")
    ap.add_argument("--rho", type=float, default=0.25,
                    help="selection budget; NOT recoverable from the paper text, so swept")
    ap.add_argument("--topk-frac", type=float, default=0.25,
                    help="divergence selector budget; matched to --rho for a fair head-to-head")
    ap.add_argument("--tau", type=float, default=1.0, help="anchor temperature (anchor_kl only)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-at", default="150,300")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--probe-facts", type=int, default=40)
    ap.add_argument("--probe-authors-frac", type=float, default=0.15)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = load_tokenizer()
    recs = json.loads(open(f"{P}/data/tofu/tofu800.json").read())
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
        am, _ = mask_whole_prompt(tok, r["question"])
        if len(am) != len(nat):
            continue
        ban = sorted({t for s in sal
                      for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                     | {t for s in sal
                        for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        forget.append({**r, "salient": sal, "ban_ids": ban,
                       "prompts": {"natural": nat, "masked": am}})
    auths = sorted({f["author"] for f in forget})
    rng = np.random.default_rng(1234)
    pa = set(rng.choice(auths, max(1, int(len(auths) * a.probe_authors_frac)),
                        replace=False).tolist())
    tr_f = [f for f in forget if f["author"] not in pa]
    pr_f = [f for f in forget if f["author"] in pa][:a.probe_facts]
    assert not ({f["id"] for f in tr_f} & {f["id"] for f in pr_f})
    print(f"[config] forget_loss={a.forget_loss} selector={a.selector} "
          f"alpha={a.alpha} gamma={a.gamma} beta={a.beta} rho={a.rho} "
          f"scorer_lr={a.scorer_lr} lr={a.lr}", flush=True)
    print(f"[data] forget {len(forget)} | train {len(tr_f)} / probe {len(pr_f)} | "
          f"retain {len(retain)}", flush=True)

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

    # ---- capture the final hidden state via a pre-hook on the lm_head ----
    lm_head, HID = None, {}
    for name, mod in model.named_modules():
        if name.endswith("transformer.ff_out") and isinstance(mod, torch.nn.Linear):
            lm_head = mod
            break
    assert lm_head is not None, "could not locate the lm_head (…transformer.ff_out)"
    d_model = lm_head.in_features
    lm_head.register_forward_pre_hook(lambda m, inp: HID.__setitem__("h", inp[0]))
    print(f"[scorer] lm_head in_features={d_model}", flush=True)

    # ATWU: w initialised to 0_d so every token starts at score 1/2
    w = torch.zeros(d_model, device=DEV, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    opt_w = torch.optim.AdamW([w], lr=a.scorer_lr, weight_decay=0.0)
    R = {"forget_loss": a.forget_loss, "selector": a.selector, "hparams": vars(a),
         "n_lora_modules": n_mod, "d_model": d_model,
         "n_forget_train": len(tr_f), "n_forget_probe": len(pr_f),
         "atwu_published_hparams": {
             "source": "arXiv 2606.06320 Table 6, 8B column, TOFU",
             "alpha": 0.50, "gamma": 3.00, "beta": 7.00, "scorer_lr": 0.050,
             "model_lr": 1.50e-5, "lam_ent": 1.0, "lam_pop": 15.0,
             "rho": "NOT in the paper text (supplementary only) -- swept here"},
         "mdlm_adaptation": ("token set = masked answer positions of a teacher-forced state; "
                             "p_theta(x_t|x_<t) -> p_theta(true token | prompt, y_t); "
                             "h_theta(x_t) = final hidden state at that position, captured by a "
                             "forward-pre-hook on the lm_head")}
    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    def canvas(items, which="natural"):
        pid = [it["prompts"][which] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    def forget_terms(items):
        """Returns (per-position forget loss, scores g, divergence d, rows, instruments)."""
        nat, attn, ast = canvas(items, "natural")
        noisy, _, _ = FX.random_tf_states_llada(nat, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return (None,) * 5
        logits = model(noisy, attention_mask=attn).logits
        h = HID["h"][rows, cols].float()                       # final hidden states
        lp = torch.log_softmax(logits[rows, cols].float(), -1)
        true_tok = nat[rows, cols]
        logp_true = lp.gather(1, true_tok.unsqueeze(1)).squeeze(1)

        # ---- selector ----
        if a.selector == "learned":
            g = torch.sigmoid(h @ w)                           # ATWU Eq. (6)
        elif a.selector == "uniform":
            g = torch.full_like(logp_true, 1.0)
        else:
            g = None                                           # set below from divergence

        d = None
        if a.forget_loss == "anchor_kl" or a.selector == "divergence":
            anc_full, _, _ = canvas(items, "masked")
            anc = torch.where(ar >= ast[:, None], noisy, anc_full)
            with torch.no_grad(), model.disable_adapter():
                la = torch.log_softmax(
                    model(anc, attention_mask=attn).logits[rows, cols].float() * a.tau, -1)
                l0 = torch.log_softmax(
                    model(noisy, attention_mask=attn).logits[rows, cols].float(), -1)
                d = (la.exp() * (la - l0)).sum(-1)             # KL(anchor || theta_0 cond)

        if a.selector == "divergence":
            g = torch.zeros_like(logp_true)
            for b in range(len(items)):
                sel = (rows == b).nonzero().flatten()
                if sel.numel() == 0:
                    continue
                k = max(1, int(round(a.topk_frac * sel.numel())))
                g[sel[torch.topk(d[sel], k).indices]] = 1.0

        # ---- forget loss, per position ----
        if a.forget_loss == "satga":
            p = logp_true.exp()
            # SATGA+ : score injected into the saturation exponent (ATWU Section 4)
            per = p.clamp_min(1e-9) ** (a.beta * g.detach() if a.selector != "learned"
                                        else a.beta * g) * logp_true
        else:
            per = (la.exp() * (la - lp)).sum(-1)               # anchor-first KL
        with torch.no_grad():
            ban = [torch.tensor(it["ban_ids"], device=DEV) for it in items]
            am_ = lp.argmax(-1)
            inst = {"student_fact_argmax": float(torch.stack(
                        [torch.isin(am_[i], ban[rows[i]]) for i in range(rows.numel())]
                    ).float().mean()),
                    "student_eos_argmax": float((am_ == EOS_ID).float().mean()),
                    "student_p_true": float(logp_true.exp().mean()),
                    "student_entropy": float(-(lp.exp() * lp).sum(-1).mean()),
                    "score_mean": float(g.mean()), "score_frac_above_half":
                        float((g > 0.5).float().mean()),
                    "n_positions": int(rows.numel())}
            if d is not None:
                # does the LEARNED selector agree with the free divergence selector?
                kk = max(1, int(round(a.rho * rows.numel())))
                hi_d = torch.zeros_like(d, dtype=torch.bool)
                hi_d[torch.topk(d, kk).indices] = True
                hi_g = torch.zeros_like(g, dtype=torch.bool)
                hi_g[torch.topk(g, kk).indices] = True
                inst["selector_overlap_with_divergence"] = float(
                    (hi_d & hi_g).sum() / max(kk, 1))
        return per, g, d, rows, inst

    def retain_canvas(items):
        """Retain records carry no prebuilt `prompts` key -- only forget records do -- so the
        prompt is rendered here. Same bug the smoke test caught in train_h16.py; this file had
        an independent copy of it."""
        pid = [tok(build_prompt(tok, it["question"]),
                   add_special_tokens=False)["input_ids"] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    def retain_loss(items):
        full, attn, ast = retain_canvas(items)
        noisy, _, _ = FX.random_tf_states_llada(full, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None
        lg = model(noisy, attention_mask=attn).logits[rows, cols].float()
        return F.cross_entropy(lg, full[rows, cols])

    def total(items_f, items_r, for_scorer):
        per, g, d, rows, inst = forget_terms(items_f)
        if per is None:
            return None, {}
        L = a.gamma * (g * per).sum() / max(rows.numel(), 1)
        if a.selector == "learned":
            gg = g.clamp(1e-6, 1 - 1e-6)
            H = -(gg * gg.log() + (1 - gg) * (1 - gg).log())
            L = L + a.lam_ent * H.mean() + a.lam_pop * (g.mean() - a.rho) ** 2
        if not for_scorer:
            lr_ = retain_loss(items_r)
            if lr_ is not None:
                L = L + a.alpha * lr_
                inst["L_retain"] = float(lr_)
        inst["L_total"] = float(L)
        return L, inst

    @torch.no_grad()
    def probe():
        model.eval()
        acc = collections.defaultdict(list)
        pg = torch.Generator(device=DEV).manual_seed(999)
        saved = gen.get_state(); gen.set_state(pg.get_state())
        for i in range(0, len(pr_f), a.batch):
            items = pr_f[i:i + a.batch]
            if not items:
                continue
            _, _, _, _, inst = forget_terms(items)
            if not inst:
                continue
            for k, v in inst.items():
                acc[k].append(v)
        gen.set_state(saved); model.train()
        return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}

    p0 = probe()
    R["probe_theta0"] = p0
    print(f"[probe@theta0] fact={p0['student_fact_argmax']:.4f} eos={p0['student_eos_argmax']:.4f} "
          f"p_true={p0['student_p_true']:.4f} H={p0['student_entropy']:.4f} "
          f"score={p0.get('score_mean')}", flush=True)

    hist, t0 = [{"step": -1, "note": "theta_0", **p0}], time.time()
    for step in range(a.steps):
        fb = [tr_f[(step * a.batch + k) % len(tr_f)] for k in range(a.batch)]
        rb = [retain[(step * a.batch + k) % len(retain)] for k in range(a.batch)]
        # ---- ATWU alternating optimization: scorer step, then model step ----
        if a.selector == "learned":
            opt_w.zero_grad(set_to_none=True)
            Lw, _ = total(fb, rb, for_scorer=True)
            if Lw is not None:
                Lw.backward()
                opt_w.step()
        opt.zero_grad(set_to_none=True)
        L, inst = total(fb, rb, for_scorer=False)
        if L is not None:
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
            pr = probe()
            hist.append({"step": step, **{k: inst.get(k) for k in ("L_total", "L_retain")},
                         **pr})
            print(f"[{a.forget_loss}/{a.selector}] step {step:4d} "
                  f"L={hist[-1].get('L_total')} | fact={pr['student_fact_argmax']} "
                  f"eos={pr['student_eos_argmax']} p_true={pr['student_p_true']} "
                  f"H={pr['student_entropy']} score={pr.get('score_mean')} "
                  f"overlap={pr.get('selector_overlap_with_divergence')}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")
            torch.save(w.detach().cpu(), f"{a.out}/ckpt_{step+1}/scorer_w.pt")
            print(f"[ckpt] {a.out}/ckpt_{step+1}", flush=True)

    pl = hist[-1]
    R["history"] = hist
    R["movement"] = {k: [p0.get(k), pl.get(k)] for k in
                     ("student_fact_argmax", "student_eos_argmax", "student_p_true",
                      "student_entropy", "score_mean", "score_frac_above_half",
                      "selector_overlap_with_divergence")}
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / a.steps, 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    for k, v in R["movement"].items():
        print(f"[movement] {k:34s} {v[0]} -> {v[1]}")
    print(f"[done] {a.forget_loss}/{a.selector} {R['measured']['train_wall_s']}s "
          f"{R['measured']['s_per_step']}s/step peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
