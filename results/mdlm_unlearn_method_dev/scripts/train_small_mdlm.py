"""
Pretrain a small masked-diffusion LM for the H18 testbed
(see docs/PREREG_H18_small_mdlm_testbed.md).

WHY A NEW BACKBONE. UNDO is validated on a 100M model pretrained on 2B tokens with the authors
controlling the corpus, so the oracle exists by construction. On LLaDA-8B our distillation repair
stalled at ~1.6 nats of TRAINING loss, nowhere near UNDO's "within +-5% of teacher retain
performance" criterion, and the alpha values we can afford to repair are the ones UNDO reports as
LEAST robust. That combination makes LLaDA-8B unable to test the hypothesis either way.

What this backbone buys, which LLaDA never could:
  * an EXACT retrain oracle -- pretrain with the forget authors entirely absent, rather than
    approximating `pristine` with an adapter
  * UNDO inside its validated regime
  * arms cheap enough to attack properly (adversarial lr sweeps, many seeds)

OBJECTIVE. Identical in form to the 8B work so findings transfer: sample t ~ U(0,1], mask each
token iid with probability p_mask = (1-eps)t + eps, cross-entropy on masked positions, weighted
1/t per sequence. This is the LLaDA / MDLM loss.

MODEL. Bidirectional pre-LN transformer, GPT-2-small shape (12 layers, d=768, 12 heads), vocab
50257 + a [MASK] id, tied LM head. No causal mask -- a diffusion LM attends both ways.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = "/data/bijaypandey/mdlm_unlearn"
DATA = f"{P}/data/small"
VOCAB = 50257
MASK_ID = 50257                 # appended; embedding is sized VOCAB+1
NVOCAB = VOCAB + 1


class Block(nn.Module):
    def __init__(self, d, h, drop=0.0):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        self.drop = drop

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        # is_causal=False: a masked diffusion LM is bidirectional
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(o)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class SmallMDLM(nn.Module):
    def __init__(self, d=768, layers=12, heads=12, seq=512):
        super().__init__()
        self.tok = nn.Embedding(NVOCAB, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, NVOCAB, bias=False)
        self.head.weight = self.tok.weight            # tied
        self.seq = seq
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def load_streams(include_forget):
    """general shards + fact streams. Returns (general, facts) as uint16 arrays."""
    gs = sorted(f for f in os.listdir(DATA) if f.startswith("general_") and f.endswith(".npy"))
    if not gs:
        raise SystemExit(f"no general shards in {DATA}; run build_small_corpus.py first")
    general = np.concatenate([np.load(f"{DATA}/{f}") for f in gs])
    parts = [np.load(f"{DATA}/facts_retain.npy")]
    if include_forget:
        parts.append(np.load(f"{DATA}/facts_forget.npy"))
    facts = np.concatenate(parts)
    return general, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["ref", "oracle"],
                    help="ref trains on general+retain+forget; oracle OMITS forget entirely")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--fact-frac", type=float, default=0.05,
                    help="fraction of each batch drawn from the fact stream. The facts are only "
                         "~0.3M tokens against ~2e8 general, so without upsampling they are too "
                         "dilute to memorise, and K1 (recall >= 0.60) would fail for a reason "
                         "that has nothing to do with unlearning.")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--t-lo", type=float, default=0.0,
                    help="lower bound on the mask rate t. The 1/t weighting is unbiased but its "
                         "VARIANCE explodes as t->0: at t=1e-4 a single masked token with CE 10 "
                         "contributes 1e5 to a per-sequence sum where a typical t=0.5 sample "
                         "contributes ~2.6e3, a ~40x outlier. LLaDA uses the same weighting with "
                         "batches of millions of tokens; ours is 16K, so the outliers are not "
                         "averaged out and may drown the fact-association signal.")
    ap.add_argument("--t-hi", type=float, default=1.0)
    ap.add_argument("--pack-mode", action="store_true",
                    help="value-span supervision PACKED into full 512-token sequences. The L=32 "
                         "padded variant taught the binding (rank top1 0.155 -> 0.3175) but "
                         "wrecked generation (DISCLOSE 0.0175, IGNORANT 0.8975) because L=32 "
                         "with EOS padding is a different regime from the L=512 pretraining. "
                         "This keeps the pretraining length regime and uses no padding.")
    ap.add_argument("--memo-mode", action="store_true",
                    help="VALUE-SPAN memorisation phase. Random masking over document windows "
                         "spends nearly all its loss on template tokens, so loss fell to 2.61 "
                         "while recall stayed at chance. Here the value span is ALWAYS masked, "
                         "which trains the author->value binding directly. This is an SFT-style "
                         "phase, deliberately NOT the diffusion objective -- exactly as the 8B "
                         "theta0_tofu was a LoRA SFT on answers. Unlearning arms still use the "
                         "MDLM objective.")
    ap.add_argument("--memo-frac", type=float, default=0.5,
                    help="fraction of each batch from value-span examples; the rest is general "
                         "text at a fixed 0.5 mask rate, to hold fluency")
    ap.add_argument("--alt-general", type=int, default=0,
                    help="with --memo-mode: every Nth step runs a GENERAL-ONLY L=512 "
                         "random-masked batch instead of a memo batch. Isolation is what makes "
                         "the L=32 memo form work (rank top1 0.3175): the fact is the only "
                         "context, so the author name must be used. Packing the same supervision "
                         "into 512-token sequences put 31 other VISIBLE values of the same types "
                         "beside the target, and 'emit a plausible nearby value' removed the "
                         "pressure to bind (0.3175 -> 0.1225). Alternating whole batches keeps "
                         "isolation AND restores the trailing-run regime generation needs.")
    ap.add_argument("--lam-general", type=float, default=0.5,
                    help="weight on the fluency term. The two objectives are balanced by WEIGHT, "
                         "not by token count: a packed sequence yields ~2.4 value targets against "
                         "~1000 random-mask targets, so value tokens were ~3%% of the loss and "
                         "the binding signal vanished (rank top1 0.3175 at L=32 -> 0.1225 "
                         "packed). Value loss and fluency loss are now averaged SEPARATELY and "
                         "combined as L_value + lam_general * L_general.")
    ap.add_argument("--memo-extra", type=float, default=0.15,
                    help="extra random masking of non-value, non-pad tokens, so the model cannot "
                         "rely on the value being the only masked slot")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None,
                    help="initialise from an existing model.pt (two-phase: general-only base, "
                         "then a dedicated fact fine-tune). Mixing facts at 5% into pretraining "
                         "left recall AT CHANCE (0.090 forget vs the oracle's 0.0925) because "
                         "490M tokens for 124M params is ~4 tokens/param against UNDO's 20.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")

    include_forget = (a.role == "ref")
    general, facts = load_streams(include_forget)
    print(f"[data] role={a.role} general {len(general):,} tok | facts {len(facts):,} tok "
          f"| forget included={include_forget}", flush=True)

    pack = None
    if a.pack_mode:
        parts = [np.load(f"{DATA}/pack_retain.npz")]
        if include_forget:
            parts.append(np.load(f"{DATA}/pack_forget.npz"))
        pack = {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "vmask", "span")}
        print(f"[pack] {len(pack['ids'])} packed sequences, "
              f"{pack['vmask'].mean()*100:.1f}% value tokens "
              f"(forget included={include_forget})", flush=True)

    memo = None
    if a.memo_mode:
        parts = [np.load(f"{DATA}/memo_retain.npz")]
        if include_forget:
            parts.append(np.load(f"{DATA}/memo_forget.npz"))
        memo = {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "vstart", "vend", "plen")}
        print(f"[memo] {len(memo['ids'])} value-span examples "
              f"(forget included={include_forget})", flush=True)

    model = SmallMDLM(seq=a.seq).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")
        miss, unexp = model.load_state_dict(sd["model"], strict=False)
        pn = {n for n, _ in model.named_parameters()}
        mp = [n for n in miss if n in pn]
        if mp or unexp:
            raise SystemExit(f"refusing a partial init: missing_params={mp[:5]} "
                             f"unexpected={unexp[:5]}")
        print(f"[init] from {a.init} (step {sd.get('step')})", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] {nparam/1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    def sample_batch():
        n_f = int(round(a.batch * a.fact_frac))
        n_g = a.batch - n_f
        rows = []
        for src, n in ((general, n_g), (facts, n_f)):
            if n <= 0:
                continue
            hi = len(src) - a.seq - 1
            for i in rng.integers(0, hi, size=n):
                rows.append(src[i:i + a.seq].astype(np.int64))
        out = torch.from_numpy(np.stack(rows)).to(dev, non_blocking=True)
        return out

    def sample_batch_general_512():
        hi = len(general) - a.seq - 1
        rows = np.stack([general[i:i + a.seq].astype(np.int64)
                         for i in rng.integers(0, hi, size=a.batch)])
        return torch.from_numpy(rows).to(dev)

    def pack_batch():
        """Packed value-span sequences: value tokens ALWAYS masked, same L as pretraining."""
        n_p = int(round(a.batch * a.memo_frac))
        n_g = a.batch - n_p
        L = a.seq
        idx = rng.integers(0, len(pack["ids"]), size=n_p)
        ids_p = torch.from_numpy(pack["ids"][idx].astype(np.int64)).to(dev)
        sp = torch.from_numpy(pack["span"][idx].astype(np.int64)).to(dev)
        # Mask exactly ONE value span per sequence: every other value stays visible, so the
        # target fact has a clean readable context. Masking all of them at once removed that
        # context and collapsed the binding signal to chance.
        nspan = sp.amax(dim=1).clamp_min(1)
        pickk = (torch.rand(n_p, device=dev, generator=gen) * nspan).long() + 1
        mask_p = (sp == pickk[:, None])
        # Fact rows carry ONLY the value target now, so the value loss is not diluted. The
        # general rows below supply ordinary random masking, which is what puts mask runs at
        # sequence ENDS -- value spans always have right context, but generation masks a
        # trailing run with none, and a bidirectional model needs that regime in training.
        if n_g > 0:
            hi = len(general) - L - 1
            gi = np.stack([general[i:i + L].astype(np.int64)
                           for i in rng.integers(0, hi, size=n_g)])
            ids_g = torch.from_numpy(gi).to(dev)
            mask_g = torch.rand((n_g, L), device=dev, generator=gen) < 0.5
            ids_all = torch.cat([ids_p, ids_g]); mask_all = torch.cat([mask_p, mask_g])
            isval = torch.cat([torch.ones_like(mask_p), torch.zeros_like(mask_g)])
        else:
            ids_all, mask_all, isval = ids_p, mask_p, torch.ones_like(mask_p)
        return ids_all, mask_all, isval

    def memo_batch():
        """Value-span examples (value ALWAYS masked) mixed with general text for fluency."""
        n_m = int(round(a.batch * a.memo_frac))
        n_g = a.batch - n_m
        L = int(memo["ids"].shape[1])
        idx = rng.integers(0, len(memo["ids"]), size=n_m)
        mi = memo["ids"][idx].astype(np.int64)
        ids_m = torch.from_numpy(mi).to(dev)
        vs = torch.from_numpy(memo["vstart"][idx].astype(np.int64)).to(dev)
        ve = torch.from_numpy(memo["vend"][idx].astype(np.int64)).to(dev)
        pl = torch.from_numpy(memo["plen"][idx].astype(np.int64)).to(dev)
        ar = torch.arange(L, device=dev)[None]
        valid_m = ar < pl[:, None]                      # exclude padding from the loss
        mask_m = (ar >= vs[:, None]) & (ar < ve[:, None])
        extra = (torch.rand((n_m, L), device=dev, generator=gen) < a.memo_extra) & valid_m
        mask_m = mask_m | extra
        if n_g > 0:
            hi = len(general) - L - 1
            gi = np.stack([general[i:i + L].astype(np.int64)
                           for i in rng.integers(0, hi, size=n_g)])
            ids_g = torch.from_numpy(gi).to(dev)
            mask_g = torch.rand((n_g, L), device=dev, generator=gen) < 0.5
            valid_g = torch.ones((n_g, L), dtype=torch.bool, device=dev)
            return (torch.cat([ids_m, ids_g]), torch.cat([mask_m, mask_g]),
                    torch.cat([valid_m, valid_g]))
        return ids_m, mask_m, valid_m

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / max(a.warmup, 1)
        prog = min(1.0, (s - a.warmup) / max(a.steps - a.warmup, 1))
        # cosine decay from lr to 0.1*lr
        return 0.1 * a.lr + 0.9 * a.lr * 0.5 * (1 + math.cos(math.pi * prog))

    # A FROZEN probe batch. Fixing only the mask pattern is not enough: if the batch itself is
    # redrawn each time, successive probe values differ by which documents were sampled. That is
    # the comparability bug that already cost us a withdrawn claim on the 8B arms, so the probe
    # holds the data, the mask and the mask rate all fixed. Deterministic offsets, general text
    # only, so the value means the same thing in `ref` and `oracle` runs.
    _probe_rows = np.stack([general[o:o + a.seq].astype(np.int64)
                            for o in (0, 1_000_003, 5_000_011, 9_000_017,
                                      13_000_019, 17_000_023, 21_000_029, 25_000_031)])
    probe_ids = torch.from_numpy(_probe_rows).to(dev)
    probe_mask = (torch.rand(probe_ids.shape, device=dev,
                             generator=torch.Generator(device=dev).manual_seed(777)) < 0.5)
    probe_mask[:, 0] = True
    probe_noisy = torch.where(probe_mask, torch.full_like(probe_ids, MASK_ID), probe_ids)

    t0 = time.time()
    hist = []
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        use_general_only = (a.alt_general and a.memo_mode
                            and (step % a.alt_general) == (a.alt_general - 1))
        if use_general_only:
            ids = sample_batch_general_512()
            B, T = ids.shape
            tt = torch.rand(B, device=dev, generator=gen).clamp_min(1e-4)
            pm = (1 - a.eps) * tt + a.eps
            m = torch.rand((B, T), device=dev, generator=gen) < pm[:, None]
            m[:, 0] = True
            noisy = torch.where(m, torch.full_like(ids, MASK_ID), ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(noisy)
                ce = F.cross_entropy(logits.view(-1, NVOCAB).float(), ids.view(-1),
                                     reduction="none").view(B, T)
                loss = ((ce * m).sum(1) / tt / T).mean()
        elif a.pack_mode or a.memo_mode:
            ids, m, isval = pack_batch() if a.pack_mode else memo_batch()
            B, T = ids.shape
            noisy = torch.where(m, torch.full_like(ids, MASK_ID), ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(noisy)
                ce = F.cross_entropy(logits.view(-1, NVOCAB).float(), ids.view(-1),
                                     reduction="none").view(B, T)
                sv = m & isval.bool()
                sg = m & (~isval.bool())
                l_val = (ce * sv).sum() / sv.sum().clamp_min(1)
                l_gen = (ce * sg).sum() / sg.sum().clamp_min(1)
                loss = l_val + a.lam_general * l_gen
        else:
            ids = sample_batch()
            B, T = ids.shape
            # LLaDA / MDLM forward noising
            t = torch.rand(B, device=dev, generator=gen)
            t = (a.t_lo + (a.t_hi - a.t_lo) * t).clamp_min(1e-4)   # identity when (0,1)
            p_mask = (1 - a.eps) * t + a.eps
            m = torch.rand((B, T), device=dev, generator=gen) < p_mask[:, None]
            m[:, 0] = True                                  # never an all-clean row
            noisy = torch.where(m, torch.full_like(ids, MASK_ID), ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(noisy)
                ce = F.cross_entropy(logits.view(-1, NVOCAB).float(), ids.view(-1),
                                     reduction="none").view(B, T)
                # 1/t weighting, loss on masked positions only
                per = (ce * m).sum(1) / t
                loss = (per / T).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                # plain masked-token CE on the FROZEN probe batch at a fixed mask rate:
                # comparable across steps AND across runs, unlike the 1/t-weighted training loss
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(probe_noisy)
                pce = F.cross_entropy(lg.view(-1, NVOCAB).float(), probe_ids.view(-1),
                                      reduction="none").view(probe_ids.shape)
                probe = float((pce * probe_mask).sum() / probe_mask.sum().clamp_min(1))
            hist.append({"step": step, "loss": round(float(loss), 4),
                         "probe_ce_at_t0.5": round(probe, 4),
                         "wall_s": round(time.time() - t0, 1)})
            print(f"[{a.role}] step {step:6d} loss={float(loss):.4f} "
                  f"probe_CE@0.5={probe:.4f} lr={lr_at(step):.2e} "
                  f"{(time.time()-t0)/(step+1):.3f}s/step", flush=True)

        if (step + 1) % a.save_every == 0 or step == a.steps - 1:
            path = f"{a.out}/model.pt"
            torch.save({"model": model.state_dict(), "step": step + 1,
                        "cfg": {"d": 768, "layers": 12, "heads": 12, "seq": a.seq,
                                "nvocab": NVOCAB, "mask_id": MASK_ID}}, path)
            ok = os.path.exists(path)
            print(f"[ckpt] {path} step {step+1} "
                  f"({os.path.getsize(path)/2**30:.2f} GiB)" if ok
                  else f"[ckpt] FAILED {path}", flush=True)

    json.dump({"study": "h18_small_mdlm_pretrain", "role": a.role, "hparams": vars(a),
               "n_params": nparam,
               "data": {"general_tokens": int(len(general)), "fact_tokens": int(len(facts)),
                        "forget_included": include_forget},
               "oracle_note": ("role=oracle omits facts_forget ENTIRELY from pretraining, so the "
                               "retrain reference is exact rather than an approximation"),
               "history": hist,
               "measured": {"wall_s": round(time.time() - t0, 1)}},
              open(f"{a.out}/report.json", "w"), indent=1)
    print(f"[done] {a.role} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
