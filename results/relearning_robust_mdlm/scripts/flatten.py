"""
ANVIL stage 2: flatten an already-unlearned model so relearning is slow.

Motivation is Lemma 1 / Prop 2 in theory/THEORY.md: the adversary's one-step recovery is
governed by ||grad L_mem|| at the unlearned point, NOT by the forget loss value. The
Phase-0 probe measured exactly this failure -- the control at 0.049 disclosure returns to
0.605 in 10 adversary steps.

Two constraints come from the theory and both are obeyed here, not negotiated:

  P3a  the gradient-norm penalty drives the fact straight back (L -> 0) when started
       inside the memorized basin. theta_0 IS that basin. So this is a SECOND STAGE
       applied to an already-unlearned model and never to theta_0.
  P3b  high loss and zero gradient coincide only in 1-D; raw ascent diverges in high
       dimensions. So nothing here does raw ascent -- the forget term is an ANCHOR to
       the stage-2 starting point, holding disclosure where it already is.

Objective (minimised):

    J(theta) = lam_G * ||grad L_mem(theta)||            <- the new term: flatten
             + lam_A * KL( p_theta || p_{theta_u} )      <- hold the output distribution
                                                            on forget AND retain states

theta_u is the frozen stage-2 start, reachable as `disable_adapter()` because the
unlearned weights are MERGED into the backbone and only a fresh LoRA trains. So the anchor
is exact and costs no extra stored model.

||grad L_mem|| is estimated by the finite-difference surrogate of Prop 4 -- note that it
approximates ||g||, NOT ||g||^2; the two are not interchangeable in the gradient and this
code penalises the norm, as labelled.

GUARD (pre-registered, P3.1): L_mem is monitored every step and the run ABORTS with rc=3
if it falls below its value at theta_u by more than --guard-tol. A run that reduces
disclosure while reducing the adversary's loss is recovering the fact, not forgetting it.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

S = "/data/bijaypandey/spex"
A = "/data/bijaypandey/anvil"
sys.path.insert(0, f"{S}/scripts")
sys.path.insert(0, "/data/bijaypandey/llada_poc/scripts")
from common_big import EOS_ID, FACTS, forward_process, load_base, load_tokenizer, lora_targets, make_example
from mdmu.fxmix import soft_ce_kl                      # PINNED, unmodified

DEV = "cuda:0"


class Counter:
    def __init__(self):
        self.fwd = 0
        self.bwd = 0


CNT = Counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["anvil", "anchor_only"],
                    help="anchor_only is the matched control: identical objective with "
                         "lam_G=0, so the ONLY difference is the gradient-norm term")
    ap.add_argument("--lam-G", type=float, default=1.0)
    ap.add_argument("--lam-A", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.01, help="finite-difference radius")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--ckpt-at", default="50,150")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"],
                    help="SGD by default, and this matters. With AdamW a near-zero "
                         "gradient still yields a step of ~lr per coordinate, so the "
                         "lam_G=0 control RANDOM-WALKS off theta_u instead of staying "
                         "put -- measured: L_mem 0.604 -> 0.180 in 10 steps with NO "
                         "penalty term, which fired the guard on the control itself. "
                         "Under SGD the control's gradient is exactly zero at theta_u, "
                         "so it is a true no-op and any motion in the anvil arm is "
                         "attributable to the gradient-norm term.")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--probe-batches", type=int, default=4,
                    help="frozen batches for the ||grad L_mem|| instrument")
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--guard-tol", type=float, default=0.10,
                    help="abort if L_mem drops this far below its value at theta_u")
    ap.add_argument("--unlearn-adapter", default=f"{S}/adapters/big_uncached/ckpt_300")
    ap.add_argument("--memo", default=f"{S}/adapters/memorized_big")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.arm == "anchor_only":
        a.lam_G = 0.0
    ck = sorted(int(x) for x in a.ckpt_at.split(","))
    R = {"arm": a.arm, "hparams": vars(a), "checkpoints": ck,
         "theory": "THEORY.md L1, P2 (mechanism); P3a (stage-2 only); P3b (anchor, not "
                   "raw ascent); P4 (surrogate penalises ||g||, not ||g||^2)"}
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    # ── stage-2 start point: unlearned weights MERGED, fresh LoRA trains ────
    tok = load_tokenizer()
    model = load_base(DEV, train=True)
    from peft import LoraConfig, PeftModel, get_peft_model
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
    model = PeftModel.from_pretrained(model, a.unlearn_adapter).merge_and_unload().to(DEV)
    R["stack"] = ["base", "memorized_big(merged)",
                  f"{os.path.basename(os.path.dirname(a.unlearn_adapter))}/"
                  f"{os.path.basename(a.unlearn_adapter)}(merged)"]
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
        target_modules=lora_targets(), init_lora_weights=True))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in trainable]
    opt = (torch.optim.SGD(params, lr=a.lr, momentum=0.0)
           if a.optimizer == "sgd" else
           torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0))
    R["n_trainable"] = sum(p.numel() for p in params)
    print(f"[stack] {' -> '.join(R['stack'])}  trainable={R['n_trainable']}", flush=True)

    forget = [f for f in FACTS if f["split"] == "forget"]
    retain = [f for f in FACTS if f["split"] == "retain"]
    ex_f = [make_example(tok, f) for f in forget]
    ex_r = [make_example(tok, f) for f in retain]
    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    def pack(batch):
        L = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), L), EOS_ID, dtype=torch.long, device=DEV)
        lab = torch.full((len(batch), L), -100, dtype=torch.long, device=DEV)
        att = torch.zeros((len(batch), L), dtype=torch.long, device=DEV)
        for r, b in enumerate(batch):
            n = len(b["ids"])
            ids[r, :n] = torch.tensor(b["ids"], device=DEV)
            lab[r, :n] = torch.tensor(b["labels"], device=DEV)
            att[r, :n] = 1
        return ids, lab, att

    def L_mem(ids, lab, att, noisy, masked, p_mask):
        """The ADVERSARY's objective, as a differentiable tensor."""
        CNT.fwd += 1
        logits = model(noisy, attention_mask=att).logits
        tl = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
        return tl.mean()

    @torch.no_grad()
    def anchor_logits(noisy, att):
        """p_{theta_u}: the frozen stage-2 start, via disable_adapter()."""
        CNT.fwd += 1
        with model.disable_adapter():
            return model(noisy, attention_mask=att).logits

    def flat_add(vecs, scale):
        with torch.no_grad():
            for p, v in zip(params, vecs):
                p.add_(v, alpha=scale)

    # ── FIXED gradient probe ────────────────────────────────────────────────
    # ||grad L_mem|| must be measured on the SAME states every time or the trajectory
    # conflates flattening with batch/mask noise (the per-step number below is on a
    # fresh random batch and is NOT comparable across steps). This probe freezes
    # --probe-batches batches and their masks once, and is the instrument the
    # Phase-2 gate is judged on.
    pgen = torch.Generator(device=DEV).manual_seed(12345)
    probe = []
    for j in range(a.probe_batches):
        bb = [ex_f[(j * a.batch + k) % len(ex_f)] for k in range(a.batch)]
        i2, l2, t2 = pack(bb)
        n2, m2, pm2 = forward_process(i2, l2, pgen)
        probe.append((i2, l2, t2, n2, m2, pm2))
    print(f"[probe] {len(probe)} frozen batches x {a.batch} facts, masks fixed", flush=True)

    def probe_grad_norm():
        """||grad L_mem|| and L_mem on the frozen probe states. Not counted in the
        training cost: this is instrumentation, and it is reported separately."""
        tot_sq, tot_l = 0.0, 0.0
        for (i2, l2, t2, n2, m2, pm2) in probe:
            lg = model(n2, attention_mask=t2).logits
            tl = F.cross_entropy(lg[m2].float(), i2[m2], reduction="none") / pm2[m2]
            lm = tl.mean()
            gv = torch.autograd.grad(lm, params)
            tot_sq += float(sum((x * x).sum() for x in gv))
            tot_l += float(lm)
        opt.zero_grad(set_to_none=True)
        return (tot_sq / len(probe)) ** 0.5, tot_l / len(probe)

    # ── guard baseline, on the FROZEN probe states at theta_u ───────────────
    # The first version of this guard compared L_mem on step N's RANDOM batch against a
    # baseline from batch 0. Because the mask rate is t ~ U(0,1), L_mem swings wildly
    # across batches (0.66 -> 0.18 with no real drift), so the guard fired at step 10 on
    # EVERY arm including lam_G = 0. The baseline and the guard now both read the frozen
    # probe, which is the only comparable measurement.
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        tot_l = 0.0
        for (i2, l2, t2, n2, m2, pm2) in probe:
            lg = model(n2, attention_mask=t2).logits
            tot_l += float((F.cross_entropy(lg[m2].float(), i2[m2], reduction="none")
                            / pm2[m2]).mean())
        L_mem_at_theta_u = tot_l / len(probe)
    model.train()
    R["L_mem_at_theta_u"] = round(L_mem_at_theta_u, 4)
    R["guard_definition"] = ("frozen-probe L_mem, compared only at probe steps; abort if "
                             "it falls more than --guard-tol below its value at theta_u")
    print(f"[guard] frozen-probe L_mem(theta_u) = {L_mem_at_theta_u:.4f}; abort below "
          f"{L_mem_at_theta_u - a.guard_tol:.4f}", flush=True)

    hist, t0, step_times = [], time.time(), []
    order_f = torch.randperm(len(ex_f), generator=torch.Generator().manual_seed(a.seed)).tolist()
    order_r = torch.randperm(len(ex_r), generator=torch.Generator().manual_seed(a.seed + 1)).tolist()

    for step in range(a.steps):
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        fb = [ex_f[order_f[(step * a.batch + k) % len(ex_f)]] for k in range(a.batch)]
        rb = [ex_r[order_r[(step * a.batch + k) % len(ex_r)]] for k in range(a.batch)]
        ids, lab, att = pack(fb)
        noisy, masked, p_mask = forward_process(ids, lab, gen)

        # ── term 1: gradient-norm penalty via the Prop-4 finite difference ──
        gnorm_val, pen = 0.0, None
        if a.lam_G > 0:
            g = torch.autograd.grad(L_mem(ids, lab, att, noisy, masked, p_mask), params,
                                    create_graph=False)
            CNT.bwd += 1
            gnorm_val = float(torch.sqrt(sum((x * x).sum() for x in g)))
            if gnorm_val > 1e-12:
                unit = [x / gnorm_val for x in g]
                flat_add(unit, a.rho)                       # theta + rho * ghat
                l_pert = L_mem(ids, lab, att, noisy, masked, p_mask)
                g_pert = torch.autograd.grad(l_pert, params, create_graph=False)
                CNT.bwd += 1
                flat_add(unit, -a.rho)                      # restore theta
                # d/dtheta ||g|| ~= (grad L(theta+rho ghat) - grad L(theta)) / rho
                for p, gp, g0 in zip(params, g_pert, g):
                    gr = (gp - g0) / a.rho
                    p.grad = gr.mul(a.lam_G) if p.grad is None else p.grad.add_(gr, alpha=a.lam_G)
                pen = gnorm_val
        else:
            # anchor_only takes NO gradient-norm step, but still MEASURES ||grad L_mem||
            # so the two arms are directly comparable on the quantity the theory says
            # matters. The measurement gradient is discarded before the anchor terms.
            gv = torch.autograd.grad(L_mem(ids, lab, att, noisy, masked, p_mask), params)
            CNT.bwd += 1
            gnorm_val = float(torch.sqrt(sum((x * x).sum() for x in gv)))
            opt.zero_grad(set_to_none=True)

        # ── term 2: anchor to theta_u on BOTH forget and retain states ──────
        anch = 0.0
        for (bb, tag) in ((fb, "forget"), (rb, "retain")):
            i2, l2, t2 = pack(bb)
            n2, m2, _ = forward_process(i2, l2, gen)
            with torch.no_grad():
                za = anchor_logits(n2, t2)[m2].float()
            CNT.fwd += 1
            zs = model(n2, attention_mask=t2).logits[m2].float()
            _, _, kl = soft_ce_kl(F.log_softmax(za, -1), zs)
            loss = a.lam_A * kl.mean()
            loss.backward()
            CNT.bwd += 1
            anch += float(loss)

        gn_upd = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step_times.append(time.time() - ts)

        # ── the guard ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            lm = float(F.cross_entropy(
                model(noisy, attention_mask=att).logits[masked].float(),
                ids[masked], reduction="none").div(p_mask[masked]).mean())
        model.train()
        pr = None
        if step % a.probe_every == 0 or step == a.steps - 1:
            pgn, plm = probe_grad_norm()
            pr = {"probe_grad_norm": round(pgn, 6), "probe_L_mem": round(plm, 4)}
            print(f"[probe] step {step:3d} ||gradL_mem||_frozen={pgn:.6f} "
                  f"L_mem_frozen={plm:.4f}", flush=True)
        hist.append({"step": step, "L_mem": round(lm, 4),
                     **(pr or {}),
                     "grad_norm_L_mem": round(gnorm_val, 5),
                     "anchor_kl": round(anch, 5), "update_norm": round(float(gn_upd), 4)})
        if step % 10 == 0 or step == a.steps - 1:
            print(f"[{a.arm}] step {step:3d} L_mem={lm:.4f} ||gradL_mem||={gnorm_val:.5f} "
                  f"anchor={anch:.5f}", flush=True)
        if pr is not None and pr["probe_L_mem"] < L_mem_at_theta_u - a.guard_tol:
            R["guard_fired"] = {"step": step, "probe_L_mem": pr["probe_L_mem"],
                                "threshold": L_mem_at_theta_u - a.guard_tol,
                                "meaning": "L_mem fell: the fact is coming back (P3a). "
                                           "Arm aborted by the pre-registered guard."}
            R["history"] = hist
            json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
            print(f"!! GUARD FIRED at step {step}: frozen-probe L_mem="
                  f"{pr['probe_L_mem']:.4f} < {L_mem_at_theta_u - a.guard_tol:.4f}",
                  flush=True)
            sys.exit(3)
        if (step + 1) in ck:
            d = f"{a.out}/ckpt_{step+1}"
            model.save_pretrained(d)
            print(f"[ckpt] {d}", flush=True)

    R["history"] = hist
    n = max(1, len(step_times) - 5)
    R["measured"] = {
        "train_wall_s": round(time.time() - t0, 1),
        "s_per_step_steady": round(sum(step_times[5:]) / n, 4),
        "fwd_total": CNT.fwd, "bwd_total": CNT.bwd,
        "fwd_per_step": round(CNT.fwd / a.steps, 3),
        "bwd_per_step": round(CNT.bwd / a.steps, 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
        "grad_norm_first": hist[0]["grad_norm_L_mem"],
        "grad_norm_last": hist[-1]["grad_norm_L_mem"],
        "probe_grad_norm_first": hist[0].get("probe_grad_norm"),
        "probe_grad_norm_last": next(h["probe_grad_norm"] for h in reversed(hist)
                                     if "probe_grad_norm" in h),
        "probe_L_mem_first": hist[0].get("probe_L_mem"),
        "probe_L_mem_last": next(h["probe_L_mem"] for h in reversed(hist)
                                 if "probe_L_mem" in h),
        "L_mem_first": hist[0]["L_mem"], "L_mem_last": hist[-1]["L_mem"]}
    json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
    print(f"[{a.arm}] done. ||gradL_mem|| {hist[0]['grad_norm_L_mem']:.5f} -> "
          f"{hist[-1]['grad_norm_L_mem']:.5f}   L_mem {hist[0]['L_mem']:.4f} -> "
          f"{hist[-1]['L_mem']:.4f}   {R['measured']['s_per_step_steady']}s/step")


if __name__ == "__main__":
    main()
