"""
Per-fact gradient geometry at the unlearned point -- the direct test of H1.

H1 claims the leak in E3 (8 leaked facts restore 43 unseen ones) is carried by a SHARED
low-dimensional direction. Evidence so far is indirect: the final unlearning edit has
effective rank 3.13 of 16 available (outputs/edit_spectrum.txt). But a low-rank edit can
also be a low-rank AVERAGE of diverse per-fact directions, which would NOT imply transfer.

This measures the thing directly: for each forget fact f, the gradient of the adversary's
objective L_mem(f) with respect to the adversary's own parameters, evaluated at theta_u.
Then the full 64x64 cosine matrix.

Interpretation, stated before running:
  * mean |cos| >> 0 between DISTINCT facts  -> shared direction; descending on 8 facts
    moves the other 56 along nearly the same ray; H1's mechanism is supported.
  * mean |cos| ~ 0                          -> per-fact directions are near-orthogonal and
    H1's mechanism is WRONG; E3 must be explained some other way (e.g. the adversary
    undoing a shared *bias* term, or template-level rather than fact-level transfer).
Both outcomes are informative and the second kills H1.

Controls that make the number readable, because a cosine needs a scale:
  * forget-forget cosines vs RETAIN-retain cosines vs forget-retain cosines. If everything
    is equally aligned, the alignment is a property of the loss/parameterization, not of
    the unlearning edit, and says nothing about the leak.
  * the same measurement at theta_0 (memorized, pre-unlearning). If alignment is already
    high at theta_0, it is intrinsic to the task, not created by unlearning.
  * alignment of each fact's gradient with the TOP SINGULAR DIRECTION of the actual edit,
    which links this measurement to outputs/edit_spectrum.txt.

Parameterization note, declared: gradients are taken w.r.t. a freshly attached LoRA's B
matrices. At LoRA init B=0, so dL/dA = 0 identically and all first-step signal lives in B;
this is exactly the subspace the adversary's first step moves, which is what E3's mechanism
is about. It is a RESTRICTION of the full weight-space gradient and is reported as such.
"""
import argparse
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
from common_big import (EOS_ID, FACTS, forward_process, load_base, load_tokenizer,
                        lora_targets, make_example)

DEV = "cuda:0"
# The gradient MUST be measured in the subspace where the unlearning edit actually lives.
# The edit (spex/adapters/big_uncached) spans 7 linears x 32 blocks = 224 modules, and the
# MLP projections carry 71.3% of its energy (ff_proj .316, up_proj .240, ff_out .157).
# lora_targets() covers only the 4 attention linears = 128 modules = 28.7% of the energy;
# using it would repeat the exact subspace-restriction defect found in the closed anvil
# study's noise_probe/flatten. Full coverage, matched to the edit:
FULL_TARGETS = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"

ap = argparse.ArgumentParser()
ap.add_argument("--point", required=True, choices=["theta_u", "theta_0"])
ap.add_argument("--unlearn-adapter", default=f"{S}/adapters/big_uncached/ckpt_300")
ap.add_argument("--memo", default=f"{S}/adapters/memorized_big")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--n-mask-draws", type=int, default=4,
                help="average each fact's gradient over this many mask draws, with the "
                     "SAME draws for every fact, so cosines are not mask noise")
ap.add_argument("--seed", type=int, default=31337)
a = ap.parse_args()

torch.manual_seed(a.seed)
tok = load_tokenizer()
model = load_base(DEV, train=True)
from peft import LoraConfig, PeftModel, get_peft_model

stack = ["base", "memorized_big(merged)"]
model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
if a.point == "theta_u":
    model = PeftModel.from_pretrained(model, a.unlearn_adapter).merge_and_unload()
    stack.append("big_uncached/ckpt_300(merged)")
model = model.to(DEV)
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
    target_modules=FULL_TARGETS, init_lora_weights=True))
for n, p in model.named_parameters():
    p.requires_grad_("lora_B" in n)          # B only: dL/dA == 0 at LoRA init
    if "lora_" in n:
        p.data = p.data.to(torch.float32)
B_params = [p for n, p in model.named_parameters() if p.requires_grad]
NP = sum(p.numel() for p in B_params)
print(f"[stack] {' -> '.join(stack)}", flush=True)
print(f"[grad space] {len(B_params)} lora_B tensors, {NP} params", flush=True)

forget = [f for f in FACTS if f["split"] == "forget"]
retain = [f for f in FACTS if f["split"] == "retain"]
recalled = set(json.load(open(f"{S}/outputs/big_recalled_facts.json"))["recalled_forget_facts"])
groups = {"forget": forget, "retain": retain}

# identical mask draws for every fact: generator reseeded per fact
def fact_grad(fact):
    ex = make_example(tok, fact)
    acc = None
    for d in range(a.n_mask_draws):
        g = torch.Generator(device=DEV).manual_seed(a.seed + 1000 * d)
        L = len(ex["ids"])
        ids = torch.tensor([ex["ids"]], device=DEV)
        lab = torch.tensor([ex["labels"]], device=DEV)
        att = torch.ones((1, L), dtype=torch.long, device=DEV)
        noisy, masked, p_mask = forward_process(ids, lab, g)
        logits = model(noisy, attention_mask=att).logits
        tl = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
        loss = tl.mean()
        gs = torch.autograd.grad(loss, B_params)
        v = torch.cat([x.reshape(-1) for x in gs])
        acc = v if acc is None else acc + v
    v = (acc / a.n_mask_draws).to(torch.float16).cpu()
    return v


t0 = time.time()
vecs, meta = [], []
for gname, facts in groups.items():
    for i, f in enumerate(facts):
        vecs.append(fact_grad(f))
        meta.append({"fact_id": f["fact_id"], "group": gname, "answer": f["answer"],
                     "relation": f["relation"],
                     "recalled": f["fact_id"] in recalled if gname == "forget" else None})
        if (len(vecs)) % 16 == 0:
            print(f"  {len(vecs)}/{len(forget)+len(retain)}  "
                  f"{(time.time()-t0)/len(vecs):.2f}s/fact", flush=True)

M = torch.stack(vecs).float()
nrm = M.norm(dim=1, keepdim=True)
print(f"[norms] mean={nrm.mean():.4e} min={nrm.min():.4e} max={nrm.max():.4e}")
Mn = M / nrm.clamp_min(1e-20)
C = (Mn @ Mn.T).numpy()

idx = {g: [i for i, m in enumerate(meta) if m["group"] == g] for g in groups}


def offdiag(ii, jj):
    vals = [C[i, j] for i in ii for j in jj if i != j]
    return (float(np.mean(vals)), float(np.mean(np.abs(vals))), float(np.std(vals)),
            len(vals))


res = {}
for nm, (ii, jj) in {"forget_forget": (idx["forget"], idx["forget"]),
                     "retain_retain": (idx["retain"], idx["retain"]),
                     "forget_retain": (idx["forget"], idx["retain"])}.items():
    m, am, sd, n = offdiag(ii, jj)
    res[nm] = {"mean_cos": round(m, 4), "mean_abs_cos": round(am, 4),
               "std": round(sd, 4), "n_pairs": n}
    print(f"[{nm:14s}] mean_cos={m:+.4f}  mean|cos|={am:.4f}  sd={sd:.4f}  n={n}")

# same-answer vs different-answer forget pairs: does the sharing ride on the answer token?
same = [C[i, j] for i in idx["forget"] for j in idx["forget"]
        if i != j and meta[i]["answer"] == meta[j]["answer"]]
diff = [C[i, j] for i in idx["forget"] for j in idx["forget"]
        if i != j and meta[i]["answer"] != meta[j]["answer"]]
samerel = [C[i, j] for i in idx["forget"] for j in idx["forget"]
           if i != j and meta[i]["relation"] == meta[j]["relation"]]
res["forget_same_answer"] = {"mean_cos": round(float(np.mean(same)), 4), "n": len(same)}
res["forget_diff_answer"] = {"mean_cos": round(float(np.mean(diff)), 4), "n": len(diff)}
res["forget_same_relation"] = {"mean_cos": round(float(np.mean(samerel)), 4), "n": len(samerel)}
print(f"[same answer   ] mean_cos={np.mean(same):+.4f} (n={len(same)})")
print(f"[diff answer   ] mean_cos={np.mean(diff):+.4f} (n={len(diff)})")
print(f"[same relation ] mean_cos={np.mean(samerel):+.4f} (n={len(samerel)})")

# spectrum of the forget-gradient matrix: the effective dimensionality of the gradient set
Mf = Mn[idx["forget"]]
sv = torch.linalg.svdvals(Mf).numpy()
en = sv**2
pr = float((en.sum() ** 2) / (en**2).sum())
res["forget_gradient_set"] = {
    "n_facts": len(idx["forget"]),
    "effective_rank_participation_ratio": round(pr, 3),
    "top1_energy_frac": round(float(en[0] / en.sum()), 4),
    "top3_energy_frac": round(float(en[:3].sum() / en.sum()), 4),
    "top10_energy_frac": round(float(en[:10].sum() / en.sum()), 4)}
print(f"[forget grad set] effective rank = {pr:.2f} of {len(idx['forget'])} facts; "
      f"top1={en[0]/en.sum():.4f} top3={en[:3].sum()/en.sum():.4f} "
      f"top10={en[:10].sum()/en.sum():.4f}")

out = {"point": a.point, "stack": stack, "hparams": vars(a),
       "grad_space": {"n_tensors": len(B_params), "n_params": NP,
                      "restriction": "lora_B only; dL/dA == 0 at LoRA init"},
       "n_facts": len(meta), "meta": meta, "summary": res,
       "measured": {"wall_s": round(time.time() - t0, 1),
                    "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3)}}
os.makedirs(f"{P}/outputs", exist_ok=True)
json.dump(out, open(f"{P}/outputs/grad_cosine_{a.point}.json", "w"), indent=2)
np.save(f"{P}/outputs/grad_cosine_matrix_{a.point}.npy", C)
print(f"[done] {a.point} in {out['measured']['wall_s']}s "
      f"peak={out['measured']['peak_gpu_gib']} GiB")
