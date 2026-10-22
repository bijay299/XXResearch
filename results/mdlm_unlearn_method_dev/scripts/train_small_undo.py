"""
UNDO on the H18 backbone -- inside the regime where it was validated.

WHY HERE AND NOT ON 8B. UNDO (arXiv 2506.06278) is validated on a 100M model pretrained on 2B
tokens with the authors controlling the corpus. On LLaDA-8B our repair stalled at ~1.6 nats of
TRAINING loss, nowhere near its "within +-5% of teacher retain performance" criterion, and the
alpha values we could afford to repair were the ones the paper reports as LEAST robust. This
backbone is 124M with a corpus we built, so the method can actually be tested.

    theta_perturbed = (1 - alpha) * theta_suppressed + alpha * beta * N      N ~ Xavier
    then distil the SUPPRESSED TEACHER's outputs back into the damaged student.

The distillation pool COVERS the forget distribution (general text + all fact examples, forget
included), with the teacher's outputs as targets -- never gold answers. On 8B our pool contained
no forget prompts at all, so the teacher's suppression was never transferred where we measure it.

alpha=0 is the control: distil into an UNDAMAGED copy of the same teacher. If it is equally
robust, the damage does nothing and any gain is plain distillation.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
P = "/data/bijaypandey/mdlm_unlearn"; DATA = f"{P}/data/small"
sys.path.insert(0, f"{P}/scripts")
from train_small_mdlm import SmallMDLM, MASK_ID, NVOCAB   # noqa: E402


def xavier_like(p):
    t = torch.empty_like(p, dtype=torch.float32)
    if p.dim() >= 2: torch.nn.init.xavier_uniform_(t)
    else: torch.nn.init.normal_(t, std=0.02)
    return t.to(p.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the SUPPRESSED model (an unlearned arm)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--fact-frac", type=float, default=0.5)
    ap.add_argument("--retain-oversample", type=float, default=1.0,
                    help="relative weight of RETAIN fact rows in the distillation pool. Forget "
                         "rows must stay present -- on 8B their absence meant the teacher's "
                         "suppression was never transferred where we measure it -- but retain "
                         "repair may be the binding constraint, so oversampling retain is a "
                         "cheap lever on it (H19 arm B).")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protect", choices=["none","retain_fisher","random"], default="none",
                    help="SELECTIVE damage. UNDO perturbs every weight uniformly, yet FINDING19 "
                         "showed forget knowledge is localisable and the measured frontier shows "
                         "robustness costs retain at an accelerating rate (2.75 -> 1.63 per unit "
                         "retain as alpha goes 0.3 -> 0.5). retain_fisher scales each "
                         "parameter's perturbation by (1 - normalised retain importance), so "
                         "retain-critical weights are damaged less at the same mean damage.")
    ap.add_argument("--fisher-batches", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
    dev = torch.device("cuda")

    gs = sorted(f for f in os.listdir(DATA) if f.startswith("general_"))
    general = np.concatenate([np.load(f"{DATA}/{f}") for f in gs])
    unl = {s: np.load(f"{DATA}/unl_{s}.npz") for s in ("retain", "forget")}
    rep = max(1, int(round(a.retain_oversample)))
    facts = np.concatenate([unl["retain"]["ids"]] * rep + [unl["forget"]["ids"]])  # forget INCLUDED
    print(f"[data] general {len(general):,} tok | fact rows {len(facts)} "
          f"(forget included, retain x{rep})", flush=True)

    sd = torch.load(a.teacher, map_location="cpu"); cfg = sd.get("cfg", {})
    def build():
        m = SmallMDLM(d=cfg.get("d",768), layers=cfg.get("layers",12),
                      heads=cfg.get("heads",12), seq=cfg.get("seq",512))
        miss, unexp = m.load_state_dict(sd["model"], strict=False)
        pn = {n for n,_ in m.named_parameters()}; mp=[n for n in miss if n in pn]
        if mp or unexp: raise SystemExit(f"partial load: {mp[:5]} {unexp[:5]}")
        return m.to(dev)
    teacher = build().eval()
    for p_ in teacher.parameters(): p_.requires_grad_(False)
    student = build()

    # defined here, not after the damage step: the retain-Fisher pass below needs them
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)

    # retain-importance diagonal: E[(dL_retain/dtheta)^2] on RETAIN value spans only
    # random = the discriminating CONTROL: same per-parameter weight DISTRIBUTION as
    # retain_fisher but SHUFFLED within each tensor, so damage is equally non-uniform and equal
    # in mean while carrying NO retain information. If it matches retain_fisher, the gain comes
    # from non-uniformity alone and the Fisher signal is irrelevant -- the same logic as UNDO's
    # own alpha=0 control.
    imp = None
    if a.protect in ("retain_fisher", "random"):
        rt = np.load(f"{DATA}/unl_retain.npz")
        imp = {n_: torch.zeros_like(p_) for n_, p_ in student.named_parameters()}
        Lr = int(rt["ids"].shape[1])
        for _ in range(a.fisher_batches):
            idx = rng.integers(0, len(rt["ids"]), size=a.batch)
            ids_r = torch.from_numpy(rt["ids"][idx].astype(np.int64)).to(dev)
            ne = torch.from_numpy(rt["name_end"][idx].astype(np.int64)).to(dev)
            pl = torch.from_numpy(rt["plen"][idx].astype(np.int64)).to(dev)
            ar = torch.arange(Lr, device=dev)[None]
            ans = (ar >= ne[:, None]) & (ar < pl[:, None])
            tt = torch.rand(a.batch, device=dev, generator=gen).clamp_min(1e-3)
            m = (torch.rand((a.batch, Lr), device=dev, generator=gen) < tt[:, None]) & ans
            nz = torch.where(m, torch.full_like(ids_r, MASK_ID), ids_r)
            student.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = student(nz)
                ce = F.cross_entropy(lg.view(-1, NVOCAB).float(), ids_r.view(-1),
                                     reduction="none").view(ids_r.shape)
                lr_ = (ce * m).sum() / m.sum().clamp_min(1)
            lr_.backward()
            for n_, p_ in student.named_parameters():
                if p_.grad is not None: imp[n_] += p_.grad.detach().float() ** 2
        student.zero_grad(set_to_none=True)
        # normalise per tensor to [0,1] by its own 99th percentile, so scale differences
        # between tensors do not decide which ones get protected
        for n_ in imp:
            v = imp[n_].flatten()
            q = torch.quantile(v.float(), 0.99) if v.numel() < 1_000_000 else \
                torch.quantile(v.float()[torch.randperm(v.numel(), device=v.device)[:1_000_000]], 0.99)
            imp[n_] = (imp[n_] / q.clamp_min(1e-12)).clamp(0, 1)
        if a.protect == "random":
            for n_ in imp:
                flat = imp[n_].flatten()
                perm = torch.randperm(flat.numel(), device=flat.device,
                                      generator=torch.Generator(device=flat.device).manual_seed(a.seed))
                imp[n_] = flat[perm].view_as(imp[n_])
        tot = sum(float(x.mean()) for x in imp.values()) / max(len(imp), 1)
        print(f"[protect] {a.protect} over {a.fisher_batches} batches, mean importance {tot:.4f}",
              flush=True)

    nd, dsum, bsum = 0, 0.0, 0.0
    with torch.no_grad():
        for p_ in student.parameters():
            if p_.dim() < 1: continue
            b = p_.norm().item(); N = xavier_like(p_)
            if imp is not None:
                nm = dict(student.named_parameters())
                key = next((k for k, v in nm.items() if v is p_), None)
                w = (1.0 - imp[key]) if key in imp else torch.ones_like(p_)
                # renormalise so MEAN damage matches plain UNDO at the same alpha: the
                # comparison must be equal-damage, not less-damage
                w = w * (1.0 / w.mean().clamp_min(1e-6))
                al = (a.alpha * w).clamp(0, 0.95)
                p_.mul_(1.0 - al).add_(al * (a.beta * N))
            else:
                p_.mul_(1.0 - a.alpha).add_(a.alpha * a.beta * N)
            dsum += (p_.norm().item() - b) ** 2 ** 0.5 if False else abs(p_.norm().item()-b)
            bsum += b; nd += 1
    print(f"[damage] alpha={a.alpha} beta={a.beta} over {nd} tensors; "
          f"mean |d||p||/||p|| = {dsum/max(bsum,1e-9):.4f}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=0.0)
    L = int(facts.shape[1])

    # frozen probe: fixed rows, fixed mask, fixed rate -- comparable across steps AND arms
    pr_rows = torch.from_numpy(np.stack([general[o:o+L].astype(np.int64)
              for o in (0, 1_000_003, 5_000_011, 9_000_017)])).to(dev)
    pr_mask = (torch.rand(pr_rows.shape, device=dev,
               generator=torch.Generator(device=dev).manual_seed(777)) < 0.5)
    pr_noisy = torch.where(pr_mask, torch.full_like(pr_rows, MASK_ID), pr_rows)

    def batch():
        nf = int(round(a.batch * a.fact_frac)); ng = a.batch - nf
        rows = [facts[i].astype(np.int64) for i in rng.integers(0, len(facts), size=nf)]
        rows += [general[i:i+L].astype(np.int64)
                 for i in rng.integers(0, len(general)-L-1, size=ng)]
        return torch.from_numpy(np.stack(rows)).to(dev)

    t0, hist = time.time(), []
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step+1)/max(a.warmup,1))
        ids = batch(); B, T = ids.shape
        tt = torch.rand(B, device=dev, generator=gen).clamp_min(1e-3)
        m = torch.rand((B,T), device=dev, generator=gen) < tt[:,None]
        m[:,0] = True
        noisy = torch.where(m, torch.full_like(ids, MASK_ID), ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad(): tl = teacher(noisy)
            sl = student(noisy)
            lt = torch.log_softmax(tl.float(), -1); ls = torch.log_softmax(sl.float(), -1)
            per = (lt.exp() * (lt - ls)).sum(-1)          # forward KL, teacher first
            loss = (per * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()

        if step % a.log_every == 0 or step == a.steps-1:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tp = teacher(pr_noisy); spv = student(pr_noisy)
                a_ = torch.log_softmax(tp.float(),-1); b_ = torch.log_softmax(spv.float(),-1)
                fk = float(((a_.exp()*(a_-b_)).sum(-1) * pr_mask).sum()/pr_mask.sum())
            hist.append({"step": step, "frozen_KL": round(fk,5),
                         "step_KL": round(float(loss),5)})
            print(f"[undo a={a.alpha}] step {step:5d} frozen_KL={fk:.5f} "
                  f"(step={float(loss):.5f}) {(time.time()-t0)/(step+1):.3f}s/step", flush=True)

    path = f"{a.out}/model.pt"
    torch.save({"model": student.state_dict(), "step": a.steps, "cfg": cfg}, path)
    print(f"[ckpt] {path} ({os.path.getsize(path)/2**30:.2f} GiB)" if os.path.exists(path)
          else f"[ckpt] FAILED {path}", flush=True)
    json.dump({"study":"h18_undo","hparams":vars(a),
               "damage":{"alpha":a.alpha,"beta":a.beta,"protect":a.protect,"n_tensors":nd,
                         "mean_rel_norm_change":round(dsum/max(bsum,1e-9),5)},
               "distillation_pool":{"forget_included":True,
                   "note":"teacher outputs are the targets, never gold answers"},
               "history":hist,"measured":{"wall_s":round(time.time()-t0,1)}},
              open(f"{a.out}/report.json","w"), indent=1)
    print(f"[done] in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
