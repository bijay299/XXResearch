"""
H16 — retain-conflict projection. Act on the UPDATE DIRECTION, not on the loss's support.

WHY, from measurement (docs/FINDING15_localization_does_not_move_the_frontier.md):

H15 localized MDU's anchor loss to the positions a validated selector picks out. The selector
works -- it moves only the positions it selects, and its complement carries nothing forgettable
(the anti-control's p_true RISES and it reaches DISCLOSE 0.725 at retain 0.719, i.e. no
unlearning at all). But localization does not move the FRONTIER. At matched lam_forget:

    lam=1   MDU DISC 0.317 retain 0.524   |  H15 DISC 0.092 retain 0.085
    lam=2   MDU DISC 0.225 retain 0.403   |  H15 DISC 0.150 retain 0.094
    lam=8   MDU DISC 0.067 retain 0.091   |  H15 DISC 0.067 retain 0.199

The curves cross: H15 wins only at DISC 0.067 where both are unusable. In the usable region
(retain > 0.4) MDU is strictly better and H15 never gets there. Best point anywhere is plain
MDU at lam_forget=1: DISC 0.317, DEGENERATE 0.075, retain ASCR 0.524.

And lam_retain is not a lever: at lam_retain in {4,16} the forgetting is cancelled outright and
p_true ends ABOVE theta_0 (0.675, 0.701, 0.757 vs 0.606).

So position selection bounds the forget-side footprint and does nothing about the function
change induced on inputs the loss never sees. The damage transfers across INPUTS through the
shared parameters. H16 therefore removes the component of the forget update that conflicts with
retain, in parameter space:

    g_f  = grad of the forget loss        (MDU's anchor KL, optionally localized)
    g_r  = grad of the retain loss        (already computed every step)
    R    = orthonormal basis of the last k retain gradients
    g_f_perp = g_f - R (R^T g_f)
    step on  g_f_perp + lam_r * g_r

To first order a step along g_f_perp leaves the retain loss unchanged while still reducing the
forget loss by ||g_f_perp||^2. The forgetting that survives projection is exactly the forgetting
that does not require moving retain -- ATWU's retain-conflict criterion (arXiv 2606.06320 Eq. 2)
in PARAMETER space rather than at the token level.

RANK MATTERS AND IS SWEPT, NOT ASSUMED. Projecting one direction out of a 42M-dimensional LoRA
space removes almost nothing, and one minibatch gradient is a noisy estimate of the retain
direction. R is built from the last k retain gradients and re-orthonormalized, with k swept
including k=1 as the degenerate case.

ARMS
  proj     g_f_perp + lam_r g_r                                   the method
  pcgrad   project only when <g_f, g_r> < 0, else leave g_f       (PCGrad rule)
  anti     project ONTO R instead of off it                       ATTRIBUTION CONTROL: keeps
                                                                  only the retain-conflicting
                                                                  part, so it should forget
                                                                  little and preserve retain
                                                                  trivially
  none     g_f + lam_r g_r                                        == H15/MDU, the baseline here

NOVELTY, STATED PLAINLY: the projection machinery is standard in continual learning (orthogonal
gradient descent, GEM/A-GEM, PCGrad, O-LoRA) and nothing about it is new. What is not published
is the pairing with an anchor-matching forget loss on a masked diffusion LM, and the reason for
doing it is a measurement rather than an analogy. If H16 works, the contribution is the
diagnosis plus the combination, not the projection.

Q3 is the mechanism instrument: measured cosine between the forget and retain gradients, before
and after projection. If alignment cannot be driven toward zero, forget and retain are not
geometrically separable in this model and that closes a whole class of methods -- which makes
this run informative even if Q1 fails.
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


def flat(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


def unflat_add_(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = vec[i:i + n].view_as(p).clone()
        i += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="proj", choices=["proj", "pcgrad", "anti", "none"])
    ap.add_argument("--rank", type=int, default=8, help="retain-subspace rank k")
    ap.add_argument("--localize", action="store_true",
                    help="apply H15's validated position selector to the forget loss")
    ap.add_argument("--topk-frac", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam-forget", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-at", default="150,300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
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
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    tok = load_tokenizer()
    recs = json.loads(open(a.data).read())
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
    print(f"[config] mode={a.mode} k={a.rank} localize={a.localize} "
          f"lam_f={a.lam_forget} lam_r={a.lam_retain}", flush=True)
    print(f"[data] forget {len(forget)} | train {len(tr_f)} / probe {len(pr_f)} | "
          f"retain {len(retain)}", flush=True)

    model = load_base(DEV, train=True)
    from peft import LoraConfig, PeftModel, get_peft_model
    if a.memo.lower() == "none":
        model = model.to(DEV)          # RWKU: theta_0 is the pristine base model
    else:
        model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_rank, lora_alpha=2 * a.lora_rank, lora_dropout=0.0, bias="none",
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
    D = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    print(f"[model] modules={n_mod} trainable={D}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    R = {"mode": a.mode, "hparams": vars(a), "n_lora_modules": n_mod, "n_trainable": D,
         "n_forget_train": len(tr_f), "n_forget_probe": len(pr_f),
         "mechanism": ("remove the component of the forget update that lies in the span of the "
                       "last k retain gradients; ATWU's retain-conflict criterion in parameter "
                       "space rather than at the token level"),
         "novelty_note": ("the projection machinery is standard in continual learning (OGD, "
                          "GEM/A-GEM, PCGrad, O-LoRA); what is not published is the pairing "
                          "with an anchor-matching loss on an MDLM, motivated by measurement"),
         "frontier_to_beat": {"arm": "MDU lam_forget=1", "DISCLOSE": 0.317,
                              "DEGENERATE": 0.075, "retain_ASCR": 0.524,
                              "TV_to_pristine": 0.407}}

    def canvas(items, which="natural"):
        pid = [it["prompts"][which] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    def forget_loss(items, want_inst=False):
        nat, attn, ast = canvas(items, "natural")
        mk, _, _ = canvas(items, "masked")
        noisy, _, _ = FX.random_tf_states_llada(nat, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None, {}
        anc = torch.where(ar >= ast[:, None], noisy, mk)
        with torch.no_grad(), model.disable_adapter():
            la = torch.log_softmax(
                model(anc, attention_mask=attn).logits[rows, cols].float() * a.tau, -1)
            l0 = torch.log_softmax(
                model(noisy, attention_mask=attn).logits[rows, cols].float(), -1)
            d = (la.exp() * (la - l0)).sum(-1)
        ls = torch.log_softmax(
            model(noisy, attention_mask=attn).logits[rows, cols].float(), -1)
        per = (la.exp() * (la - ls)).sum(-1)                   # anchor-first KL
        w = torch.zeros_like(per)
        for b in range(len(items)):
            sel = (rows == b).nonzero().flatten()
            if sel.numel() == 0:
                continue
            if a.localize:
                k = max(1, int(round(a.topk_frac * sel.numel())))
                w[sel[torch.topk(d[sel], k).indices]] = 1.0 / k
            else:
                w[sel] = 1.0 / sel.numel()
        loss = (w.detach() * per).sum() / len(items)
        inst = {}
        if want_inst:
            with torch.no_grad():
                ban = [torch.tensor(it["ban_ids"], device=DEV) for it in items]
                am_ = ls.argmax(-1)
                inst = {"student_fact_argmax": float(torch.stack(
                            [torch.isin(am_[i], ban[rows[i]])
                             for i in range(rows.numel())]).float().mean()),
                        "student_eos_argmax": float((am_ == EOS_ID).float().mean()),
                        "student_p_true": float(
                            ls.gather(1, nat[rows, cols].unsqueeze(1)).exp().mean()),
                        "student_entropy": float(-(ls.exp() * ls).sum(-1).mean())}
        return loss, inst

    def retain_canvas(items):
        """Retain records carry no prebuilt `prompts` key -- only forget records do -- so the
        prompt is rendered here. The smoke test caught this as a KeyError in 18 seconds."""
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
            _, inst = forget_loss(items, want_inst=True)
            for k, v in inst.items():
                acc[k].append(v)
        gen.set_state(saved); model.train()
        return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}

    p0 = probe()
    R["probe_theta0"] = p0
    print(f"[probe@theta0] fact={p0['student_fact_argmax']:.4f} "
          f"eos={p0['student_eos_argmax']:.4f} p_true={p0['student_p_true']:.4f} "
          f"H={p0['student_entropy']:.4f}", flush=True)

    basis = []                      # list of flattened orthonormal retain directions
    hist, t0, cos_log = [{"step": -1, **p0}], time.time(), []
    for step in range(a.steps):
        # ---- forget gradient ----
        opt.zero_grad(set_to_none=True)
        lf, _ = forget_loss([tr_f[(step * a.batch + k) % len(tr_f)]
                             for k in range(a.batch)])
        if lf is None:
            continue
        (a.lam_forget * lf).backward()
        g_f = flat(params).detach().clone()

        # ---- retain gradient ----
        opt.zero_grad(set_to_none=True)
        lr_ = retain_loss([retain[(step * a.batch + k) % len(retain)]
                           for k in range(a.batch)])
        g_r = torch.zeros_like(g_f)
        if lr_ is not None:
            (a.lam_retain * lr_).backward()
            g_r = flat(params).detach().clone()

        # ---- Q3 instrument: alignment BEFORE projection ----
        nf, nr = g_f.norm(), g_r.norm()
        cos_before = float((g_f @ g_r) / (nf * nr + 1e-12)) if nf > 0 and nr > 0 else 0.0

        # ---- maintain the rank-k retain subspace ----
        if a.mode in ("proj", "anti") and nr > 0:
            v = g_r / nr
            for b in basis:
                v = v - (v @ b) * b
            nv = v.norm()
            if float(nv) > 1e-6:
                basis.append((v / nv).detach())
                if len(basis) > a.rank:
                    basis.pop(0)

        # ---- build the update ----
        g = g_f
        if a.mode == "proj" and basis:
            comp = torch.zeros_like(g_f)
            for b in basis:
                comp = comp + (g_f @ b) * b
            g = g_f - comp
        elif a.mode == "anti" and basis:
            comp = torch.zeros_like(g_f)
            for b in basis:
                comp = comp + (g_f @ b) * b
            g = comp                                  # keep ONLY the conflicting part
        elif a.mode == "pcgrad" and nr > 0 and float(g_f @ g_r) < 0:
            g = g_f - ((g_f @ g_r) / (nr ** 2)) * g_r

        cos_after = float((g @ g_r) / (g.norm() * nr + 1e-12)) if nr > 0 and g.norm() > 0 else 0.0
        cos_log.append((cos_before, cos_after))

        unflat_add_(params, g + g_r)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
            pr = probe()
            cb = float(np.mean([c[0] for c in cos_log[-a.probe_every:]]))
            ca = float(np.mean([c[1] for c in cos_log[-a.probe_every:]]))
            hist.append({"step": step, "L_forget": round(float(lf), 5),
                         "L_retain": (round(float(lr_), 5) if lr_ is not None else None),
                         "cos_before": round(cb, 5), "cos_after": round(ca, 5),
                         "basis_rank": len(basis), **pr})
            print(f"[{a.mode}] step {step:4d} L_f={hist[-1]['L_forget']} "
                  f"L_r={hist[-1]['L_retain']} | cos {cb:+.4f} -> {ca:+.4f} "
                  f"rank={len(basis)} | fact={pr['student_fact_argmax']} "
                  f"eos={pr['student_eos_argmax']} p_true={pr['student_p_true']} "
                  f"H={pr['student_entropy']}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")
            print(f"[ckpt] {a.out}/ckpt_{step+1}", flush=True)

    pl = hist[-1]
    R["history"] = hist
    R["movement"] = {k: [p0.get(k), pl.get(k)] for k in
                     ("student_fact_argmax", "student_eos_argmax", "student_p_true",
                      "student_entropy")}
    cb = float(np.mean([c[0] for c in cos_log])) if cos_log else None
    ca = float(np.mean([c[1] for c in cos_log])) if cos_log else None
    R["Q3_gradient_alignment"] = {
        "cos_forget_retain_before_projection": (round(cb, 5) if cb is not None else None),
        "cos_after_projection": (round(ca, 5) if ca is not None else None),
        "basis_rank_final": len(basis),
        "reduced": bool(cb is not None and abs(ca) < abs(cb) * 0.5),
        "meaning": ("if alignment cannot be driven toward zero, forget and retain gradients are "
                    "not geometrically separable in this model and the whole projection class "
                    "is closed, which makes this run informative even if Q1 fails")}
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / max(a.steps, 1), 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    q = R["Q3_gradient_alignment"]
    print(f"\n[Q3] cos(g_f,g_r) {q['cos_forget_retain_before_projection']} -> "
          f"{q['cos_after_projection']}  rank={q['basis_rank_final']}  "
          f"{'REDUCED' if q['reduced'] else 'NOT REDUCED'}")
    for k, v in R["movement"].items():
        print(f"[movement] {k:22s} {v[0]} -> {v[1]}")
    print(f"[done] {a.mode} {R['measured']['train_wall_s']}s "
          f"{R['measured']['s_per_step']}s/step peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
