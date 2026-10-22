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
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
    dev = torch.device("cuda")

    gs = sorted(f for f in os.listdir(DATA) if f.startswith("general_"))
    general = np.concatenate([np.load(f"{DATA}/{f}") for f in gs])
    unl = {s: np.load(f"{DATA}/unl_{s}.npz") for s in ("retain", "forget")}
    facts = np.concatenate([unl[s]["ids"] for s in ("retain", "forget")])   # forget INCLUDED
    print(f"[data] general {len(general):,} tok | fact rows {len(facts)} (forget included)",
          flush=True)

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

    nd, dsum, bsum = 0, 0.0, 0.0
    with torch.no_grad():
        for p_ in student.parameters():
            if p_.dim() < 1: continue
            b = p_.norm().item(); N = xavier_like(p_)
            p_.mul_(1.0 - a.alpha).add_(a.alpha * a.beta * N)
            dsum += (p_.norm().item() - b) ** 2 ** 0.5 if False else abs(p_.norm().item()-b)
            bsum += b; nd += 1
    print(f"[damage] alpha={a.alpha} beta={a.beta} over {nd} tensors; "
          f"mean |d||p||/||p|| = {dsum/max(bsum,1e-9):.4f}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9,0.95), weight_decay=0.0)
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
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
               "damage":{"alpha":a.alpha,"beta":a.beta,"n_tensors":nd,
                         "mean_rel_norm_change":round(dsum/max(bsum,1e-9),5)},
               "distillation_pool":{"forget_included":True,
                   "note":"teacher outputs are the targets, never gold answers"},
               "history":hist,"measured":{"wall_s":round(time.time()-t0,1)}},
              open(f"{a.out}/report.json","w"), indent=1)
    print(f"[done] in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
