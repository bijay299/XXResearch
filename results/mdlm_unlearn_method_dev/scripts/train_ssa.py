"""
SSA — Schedule-Split Anchoring. Restrict the forget objective to a band of the noise schedule.

THE ONE MDLM-SPECIFIC AXIS LEFT, and it comes from our own measurement rather than an analogy.

`FINDING13` measured the prompt-masked anchor at 23,450 positions and found it is **two
different targets at the two ends of the noise schedule**:

    mask rate      anchor p_true   punct argmax   entropy    character
    0.75-1.00        0.064            0.422        4.41      fact-free but FORMLESS
    0.50-0.75        0.412            0.259        2.15
    0.25-0.50        0.585            0.252        1.33
    0.00-0.25        0.651            0.288        0.82      well-formed but KNOWS THE FACT

MDU (Eq. 5) and EMDU (Eq. 3-4) both draw t ~ U[0,1] and weight every noise level EQUALLY, so
both average a target that is good-for-forgetting-bad-for-form against one that is the reverse.
Neither paper varies or ablates the schedule. There is no mask-rate axis in an autoregressive
model or in an image diffusion model, so this knob does not exist outside MDLMs.

WHY NOW. AFC's n=5 seeds destroyed its stability claim: TV sd 0.158 against MDU's 0.033, with
seed 3 producing DEGENERATE **0.508**. If degeneracy is imported from the formless high-mask-rate
anchor, restricting the forget loss away from that band should cut exactly that variance. That
is a falsifiable prediction about a measured failure, not a hoped-for improvement.

RELATION TO A CLOSED CANDIDATE, stated plainly. H6 ("noise-schedule shaping") was closed earlier
by `FINDING5_dataset_confound` -- closed for a DATA problem, on a dataset since replaced, not
because the mechanism was tested and failed. SSA touches the same axis. It is motivated by
evidence (FINDING13) that did not exist when H6 was closed, and it is run on the clean TOFU
backbone, but it is not a fresh axis and the write-up must not present it as one.

PARITY GATE. `FX.random_tf_states_llada` is pinned and read-only and draws t ~ U(0,1) with no
range argument, so the forward process is reproduced here with a t-range. With
`--t-lo 0 --t-hi 1` and the same generator state this local sampler must reproduce the pinned
one BIT-IDENTICALLY, or every number is against this script rather than the real forward
process. The run aborts if it does not.

ARMS (one flag apart, everything else identical to AFC lam=1):
  t in [0, 1]     == AFC, the baseline in this script
  t in [0, 0.5]   anchor is well-formed but knows more of the fact  -> gentler, predicted stabler
  t in [0.5, 1]   anchor is fact-free but formless                  -> predicted MORE degenerate
  t in [0.25,0.75] the middle band
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


def tf_states_band(full_ids, ans_start, t_lo, t_hi, eps=1e-3, generator=None):
    """The pinned LLaDA SFT forward process with t drawn from [t_lo, t_hi] instead of (0,1).

    Reproduces `FX.random_tf_states_llada` exactly when (t_lo, t_hi) == (0, 1): same two _rand
    draws, same order, same shapes, so the generator stream is identical."""
    b, l = full_ids.shape
    dev = full_ids.device
    t = FX._rand(b, dev, generator)
    t = t_lo + (t_hi - t_lo) * t                      # identity when (0, 1)
    p_mask = ((1 - eps) * t + eps)[:, None].expand(b, l)
    masked = FX._rand((b, l), dev, generator) < p_mask
    masked &= torch.arange(l, device=dev)[None] >= ans_start[:, None]
    return torch.where(masked, torch.full_like(full_ids, MASK_ID), full_ids), masked, p_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-lo", type=float, default=0.0)
    ap.add_argument("--t-hi", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam-forget", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-at", default="150,300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--probe-facts", type=int, default=40)
    ap.add_argument("--probe-authors-frac", type=float, default=0.15)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--data", default=f"{P}/data/tofu/tofu800.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    # ---------- PARITY GATE ----------
    gA = torch.Generator(device=DEV).manual_seed(4242)
    gB = torch.Generator(device=DEV).manual_seed(4242)
    fid = torch.randint(0, 1000, (3, 40), device=DEV)
    ast = torch.full((3,), 8, dtype=torch.long, device=DEV)
    ref, mref, pref = FX.random_tf_states_llada(fid, ast, eps=1e-3, generator=gA)
    mine, mmine, pmine = tf_states_band(fid, ast, 0.0, 1.0, eps=1e-3, generator=gB)
    parity = bool(torch.equal(ref, mine) and torch.equal(mref, mmine)
                  and torch.allclose(pref, pmine))
    print(f"[parity gate] band sampler == pinned random_tf_states_llada at (0,1): {parity}",
          flush=True)
    assert parity, "PARITY FAILED -- the local forward process differs from the pinned one"

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
    print(f"[ssa] t in [{a.t_lo}, {a.t_hi}] | forget train {len(tr_f)} / probe {len(pr_f)} | "
          f"retain {len(retain)}", flush=True)

    model = load_base(DEV, train=True)
    from peft import LoraConfig, PeftModel, get_peft_model
    if a.memo.lower() == "none":
        model = model.to(DEV)
    else:
        model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_rank, lora_alpha=2 * a.lora_rank, lora_dropout=0.0, bias="none",
        target_modules=FULL_TARGETS, init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32); p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_mod = len({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters()
                 if "lora_A" in n})
    assert n_mod == 224, f"expected 224 LoRA modules, got {n_mod}"
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    print(f"[model] modules={n_mod}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    R = {"study": "schedule_split_anchoring", "hparams": vars(a), "parity_gate": parity,
         "n_lora_modules": n_mod, "n_forget_train": len(tr_f), "n_forget_probe": len(pr_f),
         "mechanism": ("AFC's anchor-first KL on MDU's random-answer anchor, with the forget "
                       "loss restricted to mask rates t in [t_lo, t_hi]. MDU and EMDU both "
                       "weight t ~ U[0,1] uniformly and neither ablates the schedule."),
         "motivation": ("FINDING13: the anchor is fact-free but formless at high mask rate "
                        "(p_true 0.064, punct 0.422, 4.41 nats) and well-formed but "
                        "fact-knowing at low mask rate (p_true 0.651). AFC's n=5 seeds gave "
                        "TV sd 0.158 with one seed at DEGENERATE 0.508; if degeneracy is "
                        "imported from the high-t band, restricting away from it should cut "
                        "that variance."),
         "relation_to_closed_H6": ("H6 'noise-schedule shaping' was closed by "
                                   "FINDING5_dataset_confound -- a DATA problem on a dataset "
                                   "since replaced, not a tested mechanism failure. SSA is the "
                                   "same axis, motivated by evidence that postdates H6's "
                                   "closure, and is not presented as a fresh axis.")}

    def canvas(items, which="natural"):
        pid = [it["prompts"][which] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast_ = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast_

    def retain_canvas(items):
        pid = [tok(build_prompt(tok, it["question"]),
                   add_special_tokens=False)["input_ids"] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast_ = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast_

    def forget_loss(items, want_inst=False, band=True):
        nat, attn, ast_ = canvas(items, "natural")
        mk, _, _ = canvas(items, "masked")
        lo, hi = (a.t_lo, a.t_hi) if band else (0.0, 1.0)
        noisy, _, _ = tf_states_band(nat, ast_, lo, hi, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast_[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None, {}
        anc = torch.where(ar >= ast_[:, None], noisy, mk)
        with torch.no_grad(), model.disable_adapter():
            la = torch.log_softmax(
                model(anc, attention_mask=attn).logits[rows, cols].float() * a.tau, -1)
        ls = torch.log_softmax(
            model(noisy, attention_mask=attn).logits[rows, cols].float(), -1)
        loss = (la.exp() * (la - ls)).sum(-1).mean()        # anchor-first KL, uniform positions
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
                        "student_entropy": float(-(ls.exp() * ls).sum(-1).mean()),
                        "mask_rate": float(mp.float().sum() / (len(items) * a.gen_length))}
        return loss, inst

    def retain_loss(items):
        full, attn, ast_ = retain_canvas(items)
        noisy, _, _ = FX.random_tf_states_llada(full, ast_, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast_[:, None])
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
            _, inst = forget_loss(items, want_inst=True, band=False)   # probe spans all t
            for k, v in inst.items():
                acc[k].append(v)
        gen.set_state(saved); model.train()
        return {k: round(float(np.mean(v)), 5) for k, v in acc.items()}

    p0 = probe()
    R["probe_theta0"] = p0
    print(f"[probe@theta0] fact={p0['student_fact_argmax']:.4f} "
          f"eos={p0['student_eos_argmax']:.4f} p_true={p0['student_p_true']:.4f} "
          f"H={p0['student_entropy']:.4f}", flush=True)

    hist, t0 = [{"step": -1, **p0}], time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        lf, _ = forget_loss([tr_f[(step * a.batch + k) % len(tr_f)] for k in range(a.batch)])
        lr_ = retain_loss([retain[(step * a.batch + k) % len(retain)] for k in range(a.batch)])
        L = 0.0
        if lf is not None:
            L = L + a.lam_forget * lf
        if lr_ is not None:
            L = L + a.lam_retain * lr_
        if torch.is_tensor(L):
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
            pr = probe()
            hist.append({"step": step,
                         "L_forget": (round(float(lf), 5) if lf is not None else None),
                         "L_retain": (round(float(lr_), 5) if lr_ is not None else None),
                         **pr})
            print(f"[ssa t={a.t_lo}-{a.t_hi}] step {step:4d} L_f={hist[-1]['L_forget']} "
                  f"| fact={pr['student_fact_argmax']} eos={pr['student_eos_argmax']} "
                  f"p_true={pr['student_p_true']} H={pr['student_entropy']}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")

    pl = hist[-1]
    R["history"] = hist
    R["movement"] = {k: [p0.get(k), pl.get(k)] for k in
                     ("student_fact_argmax", "student_eos_argmax", "student_p_true",
                      "student_entropy")}
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "s_per_step": round((time.time() - t0) / max(a.steps, 1), 4),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    for k, v in R["movement"].items():
        print(f"[movement] {k:22s} {v[0]} -> {v[1]}")
    print(f"[done] t=[{a.t_lo},{a.t_hi}] {R['measured']['train_wall_s']}s "
          f"peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
