"""
Unlearning arms on the H18 backbone (see FINDING21_h18_backbone_qualified.md).

The point of this script is that the reference is now EXACT. On LLaDA-8B every arm was scored
against an approximated `pristine`; here theta_oracle was pretrained and fine-tuned with the
forget authors entirely absent and sits AT CHANCE on them (0.1100 vs 0.1135). So the objective
has ground truth:

    take theta_ref from 0.6975 -> 0.1100 forget recall while holding retain at 0.6725

ARMS, as the 2x2 of the two published anchor choices plus controls:

  mdu    anchor-LAST  KL + prompt-masked anchor      (arXiv 2605.18253 Eq. 5)
  emdu   anchor-FIRST KL + prompt+answer masked      (arXiv 2610.25447 Eq. 3-4, "FAM")
  afc    anchor-FIRST KL + prompt-masked anchor      the cell neither paper ran (ours, FINDING16)
  mdu_fam anchor-LAST KL + FAM                       the fourth cell, for completeness
  ga     gradient ascent on the forget value spans   the crude baseline

Every arm carries the same retain term: KL(frozen reference || student) on RETAIN examples,
weighted lam_retain. It is PRESERVATION, zero at initialisation -- a CE retain term kept
teaching the retain facts and pushed every arm's retain recall ABOVE theta_ref, which made
retain preservation unmeasurable.

Terminology fixed to the 8B work so results compare: "anchor-first" means KL(anchor || student),
"anchor-last" means KL(student || anchor). The anchor is always a FROZEN copy of theta_ref.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

P = "/data/bijaypandey/mdlm_unlearn"
DATA = f"{P}/data/small"
sys.path.insert(0, f"{P}/scripts")
from train_small_mdlm import SmallMDLM, MASK_ID, NVOCAB   # noqa: E402

PRESETS = {
    "mdu":     ("anchor_last",  "prompt"),
    "emdu":    ("anchor_first", "fam"),
    "afc":     ("anchor_first", "prompt"),
    "mdu_fam": ("anchor_last",  "fam"),
    "ga":      ("ascent",       "none"),
}


def load_unl(split):
    """Prompt/answer-split examples: prompt = [0, name_end), answer = [name_end, plen)."""
    d = np.load(f"{DATA}/unl_{split}.npz")
    return {k: d[k] for k in ("ids", "plen", "name_end", "vstart", "vend")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(PRESETS))
    ap.add_argument("--ref", default=f"{P}/adapters/small_ref/model.pt")
    ap.add_argument("--lam-forget", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--extra-mask", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    direction, anchor_mode = PRESETS[a.arm]

    fo, re_ = load_unl("forget"), load_unl("retain")
    print(f"[data] forget {len(fo['ids'])} / retain {len(re_['ids'])} value-span examples",
          flush=True)

    sd = torch.load(a.ref, map_location="cpu")
    cfg = sd.get("cfg", {})

    def build():
        m = SmallMDLM(d=cfg.get("d", 768), layers=cfg.get("layers", 12),
                      heads=cfg.get("heads", 12), seq=cfg.get("seq", 512))
        miss, unexp = m.load_state_dict(sd["model"], strict=False)
        pn = {n for n, _ in m.named_parameters()}
        mp = [n for n in miss if n in pn]
        if mp or unexp:
            raise SystemExit(f"refusing a partial load: {mp[:5]} {unexp[:5]}")
        return m.to(dev)

    student = build()
    anchor = build().eval()
    for p_ in anchor.parameters():
        p_.requires_grad_(False)
    print(f"[arm] {a.arm}: direction={direction} anchor={anchor_mode} "
          f"lam_f={a.lam_forget} lam_r={a.lam_retain}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    def draw(src, n):
        idx = rng.integers(0, len(src["ids"]), size=n)
        L = int(src["ids"].shape[1])
        ids = torch.from_numpy(src["ids"][idx].astype(np.int64)).to(dev)
        pl = torch.from_numpy(src["plen"][idx].astype(np.int64)).to(dev)
        ne = torch.from_numpy(src["name_end"][idx].astype(np.int64)).to(dev)
        ar = torch.arange(L, device=dev)[None]
        valid = ar < pl[:, None]
        ans = valid & (ar >= ne[:, None])          # answer region
        return ids, ans, valid

    def noisy_state(ids, ans):
        """Diffusion noising over the ANSWER region, prompt visible. The answer is ~9 tokens, so
        a state PARTIALLY reveals it -- which is what makes the prompt-masked anchor differ from
        full answer masking at all."""
        B = ids.shape[0]
        tt = torch.rand(B, device=dev, generator=gen).clamp_min(1e-3)
        pm = ((1 - 1e-3) * tt + 1e-3)[:, None]
        m = (torch.rand(ids.shape, device=dev, generator=gen) < pm) & ans
        return torch.where(m, torch.full_like(ids, MASK_ID), ids), m

    def anchor_input(state, ans, valid):
        """MDU: mask the PROMPT, keep the answer tokens the state revealed.
        FAM (EMDU): mask the prompt AND the whole answer."""
        s = state.clone()
        prompt = valid & ~ans
        if anchor_mode == "prompt":
            s = torch.where(prompt, torch.full_like(s, MASK_ID), s)
        elif anchor_mode == "fam":
            s = torch.where(prompt | ans, torch.full_like(s, MASK_ID), s)
        return s

    t0, hist = time.time(), []
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / max(a.warmup, 1))

        f_ids, f_ans, f_valid = draw(fo, a.batch // 2)
        r_ids, r_ans, r_valid = draw(re_, a.batch // 2)
        f_state, f_m = noisy_state(f_ids, f_ans)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_log = student(f_state)
            sel = f_m                            # masked ANSWER positions, as MDU specifies
            if direction == "ascent":
                ce = F.cross_entropy(s_log.view(-1, NVOCAB).float(), f_ids.view(-1),
                                     reduction="none").view(f_ids.shape)
                l_forget = -(ce * sel).sum() / sel.sum().clamp_min(1)
            else:
                with torch.no_grad():
                    a_log = anchor(anchor_input(f_state, f_ans, f_valid))
                sp = torch.log_softmax(s_log.float(), -1)
                ap_ = torch.log_softmax(a_log.float(), -1)
                if direction == "anchor_first":          # KL(anchor || student)
                    per = (ap_.exp() * (ap_ - sp)).sum(-1)
                else:                                   # KL(student || anchor)
                    per = (sp.exp() * (sp - ap_)).sum(-1)
                l_forget = (per * sel).sum() / sel.sum().clamp_min(1)

            # PRESERVATION, not extra training. A CE retain term kept teaching the retain facts
            # for 1500 steps, so every arm ended with retain recall ABOVE theta_ref (0.88-0.95
            # vs 0.6725) and retain preservation was not measurable. KL to the frozen reference
            # is zero at initialisation and only penalises DEVIATION.
            r_state, r_m = noisy_state(r_ids, r_ans)
            r_log = student(r_state)
            with torch.no_grad():
                r_ref = anchor(r_state)
            rp = torch.log_softmax(r_log.float(), -1)
            rq = torch.log_softmax(r_ref.float(), -1)
            per_r = (rq.exp() * (rq - rp)).sum(-1)       # KL(ref || student)
            l_retain = (per_r * r_m).sum() / r_m.sum().clamp_min(1)
            loss = a.lam_forget * l_forget + a.lam_retain * l_retain

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            hist.append({"step": step, "l_forget": round(float(l_forget), 5),
                         "l_retain": round(float(l_retain), 5),
                         "wall_s": round(time.time() - t0, 1)})
            print(f"[{a.arm}] step {step:5d} l_forget={float(l_forget):+.5f} "
                  f"l_retain={float(l_retain):.5f} {(time.time()-t0)/(step+1):.3f}s/step",
                  flush=True)

    path = f"{a.out}/model.pt"
    torch.save({"model": student.state_dict(), "step": a.steps, "cfg": cfg}, path)
    ok = os.path.exists(path)
    print(f"[ckpt] {path} ({os.path.getsize(path)/2**30:.2f} GiB)" if ok
          else f"[ckpt] FAILED {path}", flush=True)
    json.dump({"study": "h18_unlearn", "arm": a.arm, "direction": direction,
               "anchor": anchor_mode, "hparams": vars(a),
               "reference": {"theta_ref_forget_rank": 0.6975, "theta_ref_retain_rank": 0.6725,
                             "theta_oracle_forget_rank": 0.1100, "chance": 0.1135,
                             "note": "oracle is EXACT: forget authors absent from pretraining "
                                     "and fine-tune. Retain-side baseline is theta_ref, NOT the "
                                     "oracle, whose retain recall is higher only because it had "
                                     "half as many authors to memorise."},
               "history": hist,
               "measured": {"wall_s": round(time.time() - t0, 1)}},
              open(f"{a.out}/report.json", "w"), indent=1)
    print(f"[done] {a.arm} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
