"""
NOISE-FRAGILITY PROBE — how little does it take to bring the fact back?

Found by accident, then measured on purpose. A stage-2 run with lam_G = 0 and an anchor
that is identically zero at its own initialisation still dropped the adversary's loss
L_mem from 0.604 to 0.180 in 10 AdamW steps -- i.e. with no forget objective of any kind,
purely because AdamW's per-coordinate normalisation turns a zero gradient into a step of
size ~lr. That is a bug in that run, but it implies something testable about theta_u:

    the unlearned model may sit so close to the memorized one that an UNDIRECTED
    perturbation partially restores the fact -- no adversary, no leaked data, no
    gradient signal about the forget set at all.

This probe tests exactly that, with no optimiser and no training. A random LoRA is added
to the model with its effective update scaled to a fixed fraction of the norm of the real
unlearning edit, so the x-axis is in interpretable units: "a random perturbation this
large relative to the edit that did the unlearning."

Three models under the identical perturbation:
  theta_u   the unlearned control                 -> does the fact come back?
  theta_0   the memorized reference               -> ceiling (already discloses)
  base      never memorized                       -> FLOOR; nothing to come back
If theta_u's disclosure rises with noise while base's does not, the knowledge was present
and merely suppressed. If both rise together, the rise is generic damage, not recovery --
which is why base is here.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

S = "/data/bijaypandey/spex"
A = "/data/bijaypandey/anvil"
sys.path.insert(0, f"{S}/scripts")
from common_big import (FACTS, answer_present, gen_one, load_base, load_tokenizer,
                        lora_targets, output_failure)

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True, choices=["theta0", "unlearned", "base"])
ap.add_argument("--fracs", default="0,0.25,0.5,1.0,2.0",
                help="perturbation norm as a fraction of ||unlearning edit||_F")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--unlearn-adapter", default=f"{S}/adapters/big_uncached/ckpt_300")
ap.add_argument("--memo", default=f"{S}/adapters/memorized_big")
ap.add_argument("--n-forget", type=int, default=32)
ap.add_argument("--n-retain", type=int, default=16)
ap.add_argument("--seed", type=int, default=4242)
a = ap.parse_args()
DEV = "cuda:0"
FRACS = [float(x) for x in a.fracs.split(",")]

recalled = set(json.load(open(f"{S}/outputs/big_recalled_facts.json"))["recalled_forget_facts"])
forget = [f for f in FACTS if f["split"] == "forget" and f["fact_id"] in recalled][:a.n_forget]
retain = [f for f in FACTS if f["split"] == "retain"][:a.n_retain]

R = {"probe": "noise_fragility", "start": a.start, "hparams": vars(a),
     "n_forget_probed": len(forget), "n_retain_probed": len(retain)}

tok = load_tokenizer()
from peft import LoraConfig, PeftModel, get_peft_model

# ── reference scale: Frobenius norm of the REAL unlearning edit ─────────────
# effective update per target module is (alpha/r) * B @ A
sd = None
for fn in ("adapter_model.safetensors", "adapter_model.bin"):
    p = os.path.join(a.unlearn_adapter, fn)
    if os.path.exists(p):
        if fn.endswith("safetensors"):
            from safetensors.torch import load_file
            sd = load_file(p)
        else:
            sd = torch.load(p, map_location="cpu")
        break
assert sd is not None, f"no adapter weights in {a.unlearn_adapter}"
cfg = json.load(open(os.path.join(a.unlearn_adapter, "adapter_config.json")))
scal = cfg["lora_alpha"] / cfg["r"]
tot_sq, n_mod = 0.0, 0
for k in sd:
    if "lora_A" in k:
        kb = k.replace("lora_A", "lora_B")
        if kb in sd:
            dW = (sd[kb].float() @ sd[k].float()) * scal
            tot_sq += float((dW * dW).sum())
            n_mod += 1
EDIT_NORM = math.sqrt(tot_sq)
R["unlearning_edit"] = {"frobenius_norm": round(EDIT_NORM, 4), "n_modules": n_mod,
                        "lora_r": cfg["r"], "lora_alpha": cfg["lora_alpha"]}
print(f"[scale] ||unlearning edit||_F = {EDIT_NORM:.4f} over {n_mod} modules", flush=True)

# ── build the start point, once ─────────────────────────────────────────────
model = load_base(DEV)
if a.start in ("theta0", "unlearned"):
    model = PeftModel.from_pretrained(model, a.memo).merge_and_unload()
if a.start == "unlearned":
    model = PeftModel.from_pretrained(model, a.unlearn_adapter).merge_and_unload()
model = model.to(DEV)
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
    target_modules=lora_targets(), init_lora_weights=True))
model.eval()
lora_A = [(n, p) for n, p in model.named_parameters() if "lora_A" in n]
lora_B = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
print(f"[perturb] {len(lora_A)} random LoRA modules, r={a.rank}", flush=True)

g = torch.Generator(device="cpu").manual_seed(a.seed)
base_A = {n: torch.randn(p.shape, generator=g).to(p.device, p.dtype) for n, p in lora_A}
base_B = {n: torch.randn(p.shape, generator=g).to(p.device, p.dtype) for n, p in lora_B}


def set_perturbation(frac):
    """Scale the random LoRA so the total effective ||dW||_F = frac * EDIT_NORM."""
    with torch.no_grad():
        for n, p in lora_A:
            p.copy_(base_A[n])
        for n, p in lora_B:
            p.copy_(base_B[n])
        # current effective norm
        sc = 2.0  # lora_alpha / r = 2*rank/rank
        tot = 0.0
        Bd = dict(lora_B)
        for n, pA in lora_A:
            pB = Bd[n.replace("lora_A", "lora_B")]
            dW = (pB.float() @ pA.float()) * sc
            tot += float((dW * dW).sum())
        cur = math.sqrt(tot)
        if cur < 1e-12:
            return 0.0
        # dW is bilinear in (A,B): scaling both by s scales dW by s^2
        s = math.sqrt(frac * EDIT_NORM / cur) if frac > 0 else 0.0
        for _, p in lora_A:
            p.mul_(s)
        for _, p in lora_B:
            p.mul_(s)
        tot = 0.0
        for n, pA in lora_A:
            pB = Bd[n.replace("lora_A", "lora_B")]
            dW = (pB.float() @ pA.float()) * sc
            tot += float((dW * dW).sum())
        return math.sqrt(tot)


curve, rows, t0 = [], [], time.time()
for frac in FRACS:
    achieved = set_perturbation(frac)
    for group, facts in (("forget", forget), ("retain", retain)):
        for f in facts:
            txt, _, _ = gen_one(model, tok, f["train_question"], DEV)
            rows.append({"frac": frac, "group": group, "fact_id": f["fact_id"],
                         "generation": txt,
                         "answer_present": answer_present(txt, f["aliases"]),
                         "output_failure": output_failure(txt)})

    def sub(gr):
        r = [x for x in rows if x["frac"] == frac and x["group"] == gr]
        n = len(r) or 1
        clean = [x for x in r if x["output_failure"] is None]
        return {"n": len(r),
                "disclosure": round(sum(x["answer_present"] for x in r) / n, 4),
                "clean_forgetting": round(sum(1 for x in clean if not x["answer_present"]) / n, 4),
                "degenerate": len(r) - len(clean)}

    e = {"frac": frac, "perturbation_norm": round(achieved, 4),
         "forget": sub("forget"), "retain": sub("retain")}
    curve.append(e)
    print(f"[frac={frac:<5}] ||dW||={achieved:8.4f}  forget={e['forget']['disclosure']:.3f}  "
          f"retain={e['retain']['disclosure']:.3f}  deg={e['forget']['degenerate']}", flush=True)

R["curve"] = curve
R["measured"] = {"wall_s": round(time.time() - t0, 1),
                 "peak_gpu_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                 "n_generations": len(rows)}
json.dump(R, open(f"{A}/outputs/noise_{a.start}.json", "w"), indent=2)
json.dump(rows, open(f"{A}/outputs/noise_gens_{a.start}.json", "w"), indent=2,
          ensure_ascii=False)
print(f"[{a.start}] done in {R['measured']['wall_s']}s")
