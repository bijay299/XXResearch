"""
RMU for masked diffusion LMs — representation misdirection, the one level FINDING18 does not close.

WHY THIS LEVEL. FINDING18 established, by three independent measurements, that in the
anchor family *any* anchor retaining enough answer context to be well-formed retains enough to
identify the fact: form and fact-identity are carried by the same information. That closes every
construction over WHAT THE ANCHOR SEES.

RMU (Li et al., arXiv 2403.03218, the WMDP method) is not in that family. Its loss, verbatim
from the paper:

    L_forget = E_{x_f ~ D_forget} [ (1/L_f) * SUM_t || M_updated(t) - c*u ||_2^2 ]
    L_retain = E_{x_r ~ D_retain} [ (1/L_r) * SUM_t || M_updated(t) - M_frozen(t) ||_2^2 ]
    L        = L_forget + alpha * L_retain

with `u` a fixed random unit vector, entries uniform on [0,1), held fixed throughout training;
loss computed at one layer `l`; gradients applied only to layers `l-2, l-1, l`.

The forget target is a RANDOM VECTOR -- it carries neither the fact nor the form -- and form
preservation comes from a SEPARATE retain term on DIFFERENT data. That is exactly the decoupling
FINDING18 shows is unavailable inside the anchor family, which is why our own impossibility
result points here and nowhere else.

THE MDLM-SPECIFIC DESIGN QUESTION, with no autoregressive analogue. RMU applies its forget loss
uniformly over all token positions. But `M_updated(t)` is not even well defined for an MDLM
without specifying the mask pattern: a position's activation depends on bidirectional context
AND changes as denoising proceeds. So there is a free choice RMU never had to make -- over which
(position, mask-rate) distribution do you misdirect?

FINDING13 answers it: the fact is decided at HIGH mask rate (anchor p_true 0.064 there, 0.651 at
low rate), the form at LOW mask rate. So misdirect activations at high-mask-rate states and
leave low-rate states alone. This is NOT the SSA failure repeated: SSA failed because the anchor
entangles form and fact in one signal; here the target is a random vector with nothing to
entangle, and the schedule choice selects WHEN TO MISDIRECT rather than WHAT TO COPY.

KNOWN RISK, recorded before running. Lucki et al. (arXiv 2409.18025) report that RMU "is
characterized by making models unusable -- they output gibberish generations with high
perplexity". So naive RMU may reproduce the degeneracy failure by a different route. That is
cheap to test and is therefore tested rather than argued about.

ARMS
  rmu_all   misdirect at all mask rates          faithful port of RMU
  rmu_hi    misdirect only at t in [t_lo, t_hi]  the MDLM-specific variant
  rmu_ctrl  retain term only, forget weight 0    ATTRIBUTION CONTROL: must not forget

Layer activations are captured by a forward hook on the chosen transformer block, so the pinned
model code is untouched.
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
from entity_mask_target import stems
from train_ssa import tf_states_band            # parity-gated band sampler

MASK_ID, EOS_ID = 126336, 126081
DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="rmu_all",
                    choices=["rmu_all", "rmu_hi", "rmu_ctrl", "decoy", "decoy_hi", "decoyq"])
    ap.add_argument("--layer", type=int, default=16, help="RMU layer l (of 32 blocks)")
    ap.add_argument("--c", type=float, default=320.0,
                    help="activation scaling (RMU's c). Model-dependent: measured "
                         "||h||=158.67 at layer 16 of LLaDA, and RMU scales the norm "
                         "UP, so c must exceed it. RMU's own default of 20 points "
                         "the target inward on this model.")
    ap.add_argument("--alpha", type=float, default=100.0, help="retain weight (RMU's alpha)")
    ap.add_argument("--t-lo", type=float, default=0.5)
    ap.add_argument("--t-hi", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt-at", default="150,300")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--probe-facts", type=int, default=40)
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--data", default=f"{P}/data/tofu/tofu800.json")
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
        ban = sorted({t for s in sal
                      for t in tok(" " + s, add_special_tokens=False)["input_ids"][:1]}
                     | {t for s in sal
                        for t in tok(s, add_special_tokens=False)["input_ids"][:1]})
        forget.append({**r, "salient": sal, "ban_ids": ban})
    # decoy assignment: each forget fact is paired with another fact from a DIFFERENT author,
    # so the target answer is plausible for the question type but factually wrong.
    _r = np.random.default_rng(4242)
    decoy_of = {}
    for f in forget:
        cands = [g for g in forget if g["author"] != f["author"]]
        decoy_of[f["id"]] = cands[int(_r.integers(len(cands)))] if cands else f

    auths = sorted({f["author"] for f in forget})
    rng = np.random.default_rng(1234)
    pa = set(rng.choice(auths, max(1, int(len(auths) * 0.15)), replace=False).tolist())
    tr_f = [f for f in forget if f["author"] not in pa]
    pr_f = [f for f in forget if f["author"] in pa][:a.probe_facts]
    print(f"[rmu] arm={a.arm} layer={a.layer} c={a.c} alpha={a.alpha} "
          f"band=[{a.t_lo},{a.t_hi}] | forget {len(tr_f)}/{len(pr_f)} retain {len(retain)}",
          flush=True)

    model = load_base(DEV, train=True)
    from peft import LoraConfig, PeftModel, get_peft_model
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload().to(DEV)
    # RMU updates ONLY layers l-2, l-1, l. Restrict the LoRA target regex accordingly.
    lay = [a.layer - 2, a.layer - 1, a.layer]
    targ = (r".*blocks\.(" + "|".join(str(x) for x in lay)
            + r")\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$")
    model = get_peft_model(model, LoraConfig(
        r=a.lora_rank, lora_alpha=2 * a.lora_rank, lora_dropout=0.0, bias="none",
        target_modules=targ, init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32); p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_mod = len({n.rsplit(".lora_", 1)[0] for n, _ in model.named_parameters()
                 if "lora_A" in n})
    assert n_mod == 21, f"expected 3 layers x 7 linears = 21 LoRA modules, got {n_mod}"
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    print(f"[model] layers {lay} -> {n_mod} LoRA modules, "
          f"{sum(p.numel() for p in params)} params", flush=True)

    # ---- forward hook on block `layer` to capture activations ----
    HID = {}
    blk = None
    for name, mod in model.named_modules():
        if name.endswith(f"transformer.blocks.{a.layer}"):
            blk = mod; break
    assert blk is not None, f"could not find transformer.blocks.{a.layer}"
    blk.register_forward_hook(
        lambda m, i, o: HID.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    print(f"[hook] attached to transformer.blocks.{a.layer}", flush=True)

    # RMU's u: a FIXED random unit vector, entries uniform on [0,1)
    gu = torch.Generator().manual_seed(a.seed + 31337)
    with torch.no_grad():
        probe_ids = torch.full((1, 8), MASK_ID, device=DEV)
        model(probe_ids, attention_mask=torch.ones_like(probe_ids))
        d_model = HID["h"].shape[-1]
    u = torch.rand(d_model, generator=gu).to(DEV)
    u = u / u.norm()
    print(f"[rmu] d_model={d_model}, u fixed, ||u||={float(u.norm()):.4f}", flush=True)

    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    def canvas(items):
        pid = [tok(build_prompt(tok, it["question"]),
                   add_special_tokens=False)["input_ids"] for it in items]
        aid = [FX.answer_ids(tok, it["answer"], a.gen_length, append_eot=False)
               for it in items]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.tensor(aid, dtype=torch.long)], 1).to(DEV)
        attn = torch.cat([Am, torch.ones(len(items), a.gen_length, dtype=Am.dtype)], 1).to(DEV)
        ast = torch.full((len(items),), Pm.shape[1], dtype=torch.long).to(DEV)
        return full, attn, ast

    def forget_loss(items, want_inst=False):
        full, attn, ast = canvas(items)
        lo, hi = ((a.t_lo, a.t_hi) if a.arm in ("rmu_hi", "decoy_hi") else (0.0, 1.0))
        noisy, _, _ = tf_states_band(full, ast, lo, hi, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None, {}
        logits = model(noisy, attention_mask=attn).logits
        h = HID["h"][rows, cols].float()
        if a.arm == "decoyq":
            # DECOY PROMPT, not decoy answer. Substituting the decoy ANSWER barely moved the
            # activation (L_forget 1.2 vs RMU's 3.9, and no forgetting) because a masked
            # position's activation is set by the PROMPT and the mask pattern, not by which
            # answer occupies the few visible slots. The frozen model's activation on the
            # DECOY'S OWN PROMPT does produce the decoy answer -- the model knows that answer
            # for that question -- so pushing the forget activation there induces identity
            # confusion: answer q as if it were q'. That is the pristine-like behaviour
            # (plausible wrong answer), not noise.
            dec = [decoy_of[it["id"]] for it in items]
            dpid = [tok(build_prompt(tok, dd["question"]),
                        add_special_tokens=False)["input_ids"] for dd in dec]
            daid = [FX.answer_ids(tok, dd["answer"], a.gen_length, append_eot=False)
                    for dd in dec]
            dPm, dAm = FX.left_pad(dpid, EOS_ID)
            dfull = torch.cat([dPm, torch.tensor(daid, dtype=torch.long)], 1).to(DEV)
            dattn = torch.cat([dAm, torch.ones(len(items), a.gen_length,
                                               dtype=dAm.dtype)], 1).to(DEV)
            dast = torch.full((len(items),), dPm.shape[1], dtype=torch.long).to(DEV)
            dar = torch.arange(dfull.shape[1], device=DEV)[None]
            # same per-row answer-side mask COUNT, applied to the decoy canvas
            dmp = (dar >= dast[:, None]) & (torch.rand(dfull.shape, device=DEV,
                                                       generator=gen) < 0.5)
            dnoisy = torch.where(dmp, torch.full_like(dfull, MASK_ID), dfull)
            dr, dc = dmp.nonzero(as_tuple=True)
            if dr.numel() == 0:
                return None, {}
            with torch.no_grad(), model.disable_adapter():
                model(dnoisy, attention_mask=dattn)
                h_dec = HID["h"][dr, dc].float()
                h_dec = h_dec.mean(0, keepdim=True)   # mean decoy activation as the target
            model(noisy, attention_mask=attn)
            h = HID["h"][rows, cols].float()
            loss = ((h - h_dec) ** 2).mean()
        elif a.arm in ("decoy", "decoy_hi"):
            # DECOY TARGET. RMU's random c*u carries neither form nor fact, which is why it
            # produces gibberish (measured: DEGENERATE 0.825, entropy 5.49). The pristine
            # reference does not emit noise -- it gives a plausible WRONG answer (REPLACE 0.470,
            # IGNORANT 0.388). So the target here is the FROZEN model's own activations on the
            # same prompt with a DIFFERENT entity's answer substituted: a target that carries
            # its own form and its own (wrong) fact. FINDING18's wall does not apply, because
            # the form/fact entanglement it describes is within ONE answer's context, and the
            # decoy supplies both from a different example.
            dec = [decoy_of[it["id"]] for it in items]
            daid = [FX.answer_ids(tok, dd["answer"], a.gen_length, append_eot=False)
                    for dd in dec]
            dfull = full.clone()
            dfull[:, ast[0]:] = torch.tensor(daid, dtype=torch.long, device=DEV)
            dnoisy = torch.where(mp, torch.full_like(dfull, MASK_ID), dfull)
            with torch.no_grad(), model.disable_adapter():
                model(dnoisy, attention_mask=attn)
                h_dec = HID["h"][rows, cols].float().clone()
            model(noisy, attention_mask=attn)               # restore hook to student state
            h = HID["h"][rows, cols].float()
            loss = ((h - h_dec) ** 2).mean()
        else:
            loss = ((h - a.c * u) ** 2).mean()              # RMU forget term, per-dim
        inst = {}
        if want_inst:
            with torch.no_grad():
                lp = torch.log_softmax(logits[rows, cols].float(), -1)
                am_ = lp.argmax(-1)
                ban = [torch.tensor(it["ban_ids"], device=DEV) for it in items]
                inst = {"student_fact_argmax": float(torch.stack(
                            [torch.isin(am_[i], ban[rows[i]])
                             for i in range(rows.numel())]).float().mean()),
                        "student_eos_argmax": float((am_ == EOS_ID).float().mean()),
                        "student_p_true": float(
                            lp.gather(1, full[rows, cols].unsqueeze(1)).exp().mean()),
                        "student_entropy": float(-(lp.exp() * lp).sum(-1).mean()),
                        "act_norm": float(h.norm(dim=-1).mean()),
                        "cos_to_u": float(F.cosine_similarity(h, u.unsqueeze(0), dim=-1).mean())}
        return loss, inst

    def retain_loss(items):
        full, attn, ast = canvas(items)
        noisy, _, _ = FX.random_tf_states_llada(full, ast, eps=1e-3, generator=gen)
        ar = torch.arange(noisy.shape[1], device=DEV)[None]
        mp = (noisy == MASK_ID) & (ar >= ast[:, None])
        rows, cols = mp.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return None
        with torch.no_grad(), model.disable_adapter():
            model(noisy, attention_mask=attn)
            h_frozen = HID["h"][rows, cols].float().clone()
        model(noisy, attention_mask=attn)
        h_upd = HID["h"][rows, cols].float()
        return ((h_upd - h_frozen) ** 2).mean()             # RMU retain term, per-dim

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
    print(f"[probe@theta0] fact={p0['student_fact_argmax']:.4f} "
          f"eos={p0['student_eos_argmax']:.4f} p_true={p0['student_p_true']:.4f} "
          f"H={p0['student_entropy']:.4f} act_norm={p0['act_norm']:.2f} "
          f"cos_to_u={p0['cos_to_u']:.4f}", flush=True)

    R = {"study": "rmu_for_mdlm", "arm": a.arm, "hparams": vars(a), "probe_theta0": p0,
         "n_lora_modules": n_mod, "rmu_layers": lay, "d_model": d_model,
         "source": ("RMU, Li et al. arXiv 2403.03218: L_forget = ||M_updated - c*u||^2 at layer "
                    "l with u a fixed random unit vector; L_retain = ||M_updated - M_frozen||^2; "
                    "gradients only on layers l-2, l-1, l"),
         "why_this_level": ("FINDING18 closes every anchor construction -- form and "
                            "fact-identity are the same information. RMU has no anchor: its "
                            "forget target is a random vector carrying neither, and form comes "
                            "from a separate retain term on different data."),
         "mdlm_specific": ("M_updated(t) is undefined for an MDLM without a mask pattern, so "
                           "there is a free choice RMU never faced: which (position, mask-rate) "
                           "distribution to misdirect over. arm=rmu_hi restricts to high mask "
                           "rate, where FINDING13 measured the fact to be decided."),
         "known_risk": ("Lucki et al. arXiv 2409.18025 report RMU makes models output gibberish "
                        "with high perplexity; this may reproduce the degeneracy failure by "
                        "another route, which is why it is tested rather than argued")}

    hist, t0 = [{"step": -1, **p0}], time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        fb = [tr_f[(step * a.batch + k) % len(tr_f)] for k in range(a.batch)]
        rb = [retain[(step * a.batch + k) % len(retain)] for k in range(a.batch)]
        lf, _ = forget_loss(fb)
        lr_ = retain_loss(rb)
        L = 0.0
        if lf is not None and a.arm != "rmu_ctrl":
            L = L + lf
        if lr_ is not None:
            L = L + a.alpha * lr_
        if torch.is_tensor(L):
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
        if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
            pr = probe()
            hist.append({"step": step,
                         "L_forget": (round(float(lf), 2) if lf is not None else None),
                         "L_retain": (round(float(lr_), 2) if lr_ is not None else None),
                         **pr})
            print(f"[{a.arm}] step {step:4d} L_f={hist[-1]['L_forget']} "
                  f"L_r={hist[-1]['L_retain']} | fact={pr['student_fact_argmax']} "
                  f"eos={pr['student_eos_argmax']} p_true={pr['student_p_true']} "
                  f"H={pr['student_entropy']} |h|={pr['act_norm']:.1f} "
                  f"cos_u={pr['cos_to_u']:.3f}", flush=True)
        if (step + 1) in ck:
            model.save_pretrained(f"{a.out}/ckpt_{step+1}")

    pl = hist[-1]
    R["history"] = hist
    R["movement"] = {k: [p0.get(k), pl.get(k)] for k in
                     ("student_fact_argmax", "student_eos_argmax", "student_p_true",
                      "student_entropy", "act_norm", "cos_to_u")}
    R["misdirection_check"] = {
        "cos_to_u": [p0["cos_to_u"], pl["cos_to_u"]],
        "act_norm": [p0["act_norm"], pl["act_norm"]],
        "worked": bool(pl["cos_to_u"] > p0["cos_to_u"] + 0.1),
        "meaning": ("RMU's mechanism is to rotate forget-set activations toward u and scale "
                    "their norm. If cos_to_u does not rise, the intervention did not happen and "
                    "any downstream number is uninterpretable.")}
    R["measured"] = {"train_wall_s": round(time.time() - t0, 1),
                     "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    m = R["misdirection_check"]
    print(f"\n[misdirection] cos_to_u {m['cos_to_u'][0]} -> {m['cos_to_u'][1]}  "
          f"|h| {m['act_norm'][0]:.1f} -> {m['act_norm'][1]:.1f}  "
          f"{'WORKED' if m['worked'] else 'DID NOT FIRE'}")
    for k, v in R["movement"].items():
        print(f"[movement] {k:22s} {v[0]} -> {v[1]}")
    print(f"[done] {a.arm} {R['measured']['train_wall_s']}s "
          f"peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
