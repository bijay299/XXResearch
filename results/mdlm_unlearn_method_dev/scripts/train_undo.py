"""
UNDO for masked diffusion LMs — damage the weights, then distil the teacher's outputs back.

WHY THIS AND NOT ANOTHER OBJECTIVE. FINDING18 and FINDING19 established two walls:

  output space      form and fact-identity are carried by the same information, so no anchor
                    construction is both well-formed and fact-free
  activation space  any forget target close enough to the data manifold to preserve fluency is
                    too close to the current activation to give a forgetting signal; only
                    off-manifold targets forget, and those produce gibberish (RMU: DISCLOSE
                    0.050, retain 0.704, DEGENERATE 0.825)

Both walls are about obtaining a forget TARGET from the model's own distributions. Their joint
prediction is that a working method must get its target from somewhere else -- a second model,
retraining, or data. UNDO (Lee, Foote, Infanger, ... Turner, arXiv 2506.06278) is exactly that
route, and it is the one direction the two walls do not block.

THE METHOD, verbatim from the paper:

    theta_perturbed = (1 - alpha) * theta_suppressed + alpha * beta * N        N ~ Xavier

    (i)  Unlearn  -- produce a behaviourally suppressed teacher with any unlearning method
    (ii) Noise    -- perturb its weights, damaging the network globally
    (iii) Distil  -- repair the damaged student on the TEACHER'S OUTPUTS

"the key idea is to globally damage the network to varying degrees"; behaviour transfers through
distillation while latent capability does not. At its strongest setting UNDO "matches the
robustness of a model retrained from scratch with perfect data filtering".

WHAT IS OURS AND WHAT IS THEIRS. The method is entirely theirs. Ours is: the MDLM instantiation,
the choice of teacher (AFC, our ablation result), and the test of whether the two walls' joint
prediction holds on a masked diffusion LM. If this works, the contribution is the diagnosis plus
the instantiation, not the method -- same honest framing as AFC.

DISTILLATION DATA. UNDO distils the teacher's OUTPUTS, so the inputs need only cover the
behaviour being preserved. Our evaluation measures TOFU-retain content recall and six-cell
fluency, not all of LLaDA's capabilities, so the inputs are drawn from what is already on disk:
TOFU retain (400 QA), RWKU probes (~11k prompts over 200 real entities), and optionally a
HuggingFace text corpus via --hf-corpus if broader coverage proves necessary. Using local data
first is deliberate: it keeps the first run cheap and makes the data requirement an empirical
question rather than an assumption.

COMPUTE. Full-parameter distillation of 8B on 2x40GB needs sharding: ZeRO-3 with optimizer and
parameter offload to CPU (503GB RAM available, 486 free). Launch with:

    deepspeed --num_gpus 2 scripts/train_undo.py ...

The teacher is the frozen unlearned model, kept in bf16 inference mode on each rank.
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

MASK_ID, EOS_ID = 126336, 126081


def xavier_like(p):
    """Xavier init as UNDO specifies for N. Falls back to normal for 1-D tensors."""
    t = torch.empty_like(p, dtype=torch.float32)
    if p.dim() >= 2:
        torch.nn.init.xavier_uniform_(t)
    else:
        torch.nn.init.normal_(t, std=0.02)
    return t.to(p.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=f"{P}/adapters/h16_none_f1/ckpt_300",
                    help="the UNLEARNED adapter -- the suppressed teacher (default: AFC l=1 s0)")
    ap.add_argument("--memo", default=f"{P}/adapters/theta0_tofu")
    ap.add_argument("--alpha", type=float, default=0.3, help="UNDO mixing coefficient")
    ap.add_argument("--beta", type=float, default=0.1, help="UNDO noise scale")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--data", default=f"{P}/data/tofu/tofu800.json")
    ap.add_argument("--rwku", default=f"{P}/data/rwku_tofu_format.json")
    ap.add_argument("--hf-corpus", default=None,
                    help="optional HF dataset id for broader distillation coverage")
    ap.add_argument("--hf-rows", type=int, default=4000)
    ap.add_argument("--probe-every", type=int, default=150)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0
    if is_main:
        os.makedirs(a.out, exist_ok=True)

    def log(*m):
        if is_main:
            print(*m, flush=True)

    # Measured 3014%/2992% CPU (~30 cores each) with 255 on the box: the
    # DeepSpeedCPUAdam step over 8B params was thread-starved, not inherently slow.
    os.environ.setdefault("OMP_NUM_THREADS", "60")
    import deepspeed
    deepspeed.init_distributed()
    dev = torch.device(f"cuda:{a.local_rank}")
    torch.cuda.set_device(dev)

    tok = load_tokenizer()

    # ---------- distillation inputs: local first, HF only if asked ----------
    prompts = []
    recs = json.loads(open(a.data).read())
    prompts += [r["question"] for r in recs if r["split"] == "retain"]
    if os.path.exists(a.rwku):
        prompts += [r["question"] for r in json.loads(open(a.rwku).read())]
    if a.hf_corpus:
        from datasets import load_dataset
        ds = load_dataset(a.hf_corpus, split="train", streaming=True)
        n = 0
        for row in ds:
            txt = row.get("text") or row.get("content") or ""
            if len(txt) > 200:
                prompts.append(txt[:400]); n += 1
            if n >= a.hf_rows:
                break
        log(f"[data] +{n} rows from {a.hf_corpus}")
    log(f"[data] {len(prompts)} distillation prompts "
        f"(TOFU retain + RWKU{' + HF' if a.hf_corpus else ''})")

    # ---------- teacher: frozen unlearned model, bf16 inference ----------
    from peft import PeftModel
    teacher = load_base(dev)
    teacher = PeftModel.from_pretrained(teacher, a.memo).merge_and_unload()
    teacher = PeftModel.from_pretrained(teacher, a.teacher).merge_and_unload()
    teacher = teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log(f"[teacher] base -> {os.path.basename(a.memo)} -> {a.teacher} (frozen)")

    # ---------- student: the SAME weights, then globally damaged ----------
    student = load_base(dev, train=True)
    student = PeftModel.from_pretrained(student, a.memo).merge_and_unload()
    student = PeftModel.from_pretrained(student, a.teacher).merge_and_unload()
    # merge_and_unload leaves parameters frozen from PEFT, so DeepSpeed saw an empty parameter
    # list ("max() arg is an empty sequence"). UNDO repairs the FULL model, so everything trains.
    for p_ in student.parameters():
        p_.requires_grad_(True)
    g = torch.Generator().manual_seed(a.seed + 777)
    with torch.no_grad():
        n_damaged, delta_sum, base_sum = 0, 0.0, 0.0
        for p in student.parameters():
            if p.dim() < 1:
                continue
            before = p.detach().float().norm().item()
            N = xavier_like(p)
            p.mul_(1.0 - a.alpha).add_(a.alpha * a.beta * N)   # UNDO perturbation
            after = p.detach().float().norm().item()
            base_sum += before; delta_sum += abs(after - before); n_damaged += 1
    log(f"[damage] alpha={a.alpha} beta={a.beta} over {n_damaged} tensors; "
        f"mean |d||p|| / ||p|| = {delta_sum / max(base_sum, 1e-9):.4f}")
    # LLaDA's custom model code raises "LLaDAModelLM does not support gradient checkpointing",
    # and the model files are pinned read-only in this project, so activation memory is carried
    # rather than recomputed. ZeRO-3 offloads params and optimizer state to CPU, so the GPU
    # cost here is activations only; if that does not fit, the lever is batch size, not a third
    # GPU (activations are per-rank and do not shard under data parallelism).
    try:
        student.gradient_checkpointing_enable()
    except ValueError as e:
        log(f"[note] gradient checkpointing unavailable: {e}")
    if hasattr(student, "config"):
        student.config.use_cache = False

    ds_cfg = {
        "train_micro_batch_size_per_gpu": a.batch,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e8,
            "stage3_prefetch_bucket_size": 3e7,
        },
        "optimizer": {"type": "AdamW",
                      "params": {"lr": a.lr, "betas": [0.9, 0.95], "weight_decay": 0.0}},
        "gradient_clipping": 1.0,
        "steps_per_print": 10000,
    }
    engine, _, _, _ = deepspeed.initialize(model=student, config=ds_cfg,
                                           model_parameters=[p_ for p_ in student.parameters() if p_.requires_grad])
    log(f"[deepspeed] ZeRO-3 + CPU offload, {torch.cuda.device_count()} visible GPUs, "
        f"world={int(os.environ.get('WORLD_SIZE', 1))}")

    gen = torch.Generator(device=dev).manual_seed(a.seed + rank)

    def batch_canvas(qs):
        pid = [tok(build_prompt(tok, q), add_special_tokens=False)["input_ids"][:128] for q in qs]
        Pm, Am = FX.left_pad(pid, EOS_ID)
        full = torch.cat([Pm, torch.full((len(qs), a.gen_length), MASK_ID,
                                         dtype=torch.long)], 1).to(dev)
        attn = torch.cat([Am, torch.ones(len(qs), a.gen_length, dtype=Am.dtype)], 1).to(dev)
        ast = torch.full((len(qs),), Pm.shape[1], dtype=torch.long).to(dev)
        return full, attn, ast

    def distil_step(qs):
        """UNDO distils on the teacher's OUTPUTS. Inputs are prompts with a partially masked
        answer canvas, so the student matches the teacher across the denoising schedule rather
        than at one mask rate -- the MDLM instantiation of 'distil on outputs'."""
        full, attn, ast = batch_canvas(qs)
        t = torch.rand(len(qs), device=dev, generator=gen)
        pm = ((1 - 1e-3) * t + 1e-3)[:, None].expand_as(full)
        ar = torch.arange(full.shape[1], device=dev)[None]
        keep = (torch.rand(full.shape, device=dev, generator=gen) >= pm) & (ar >= ast[:, None])
        noisy = full.clone()                       # answer starts fully masked
        with torch.no_grad():
            tl = teacher(noisy, attention_mask=attn).logits
        # Guarantee >=1 masked answer position per row. A low-t draw reveals every answer
        # token (P ~ int (1-t)^64 dt ~ 1/65 per row per step); the mask RNG is seeded
        # seed+rank, so that fires on ONE rank only, which then skipped backward()/step()
        # and desynchronised ZeRO-3 -- measured as an _ALLGATHER_BASE timeout with rank0
        # at SeqNum 3931 against rank1 at 3910.
        force = ast + torch.randint(0, a.gen_length, (len(qs),), device=dev, generator=gen)
        keep[torch.arange(len(qs), device=dev), force] = False
        # reveal the teacher's own argmax at `keep` positions, so the student sees states
        # the teacher actually produces
        noisy = torch.where(keep, tl.argmax(-1), full)
        with torch.no_grad():
            tl = teacher(noisy, attention_mask=attn).logits
        sl = engine(noisy, attention_mask=attn).logits
        m = (noisy == MASK_ID) & (ar >= ast[:, None])
        r_, c_ = m.nonzero(as_tuple=True)
        if r_.numel() == 0:
            # Must never skip the step: every rank has to issue the same collectives, so
            # return a real zero-gradient loss instead of None.
            return sl.float().sum() * 0.0
        lt = torch.log_softmax(tl[r_, c_].float(), -1)
        ls = torch.log_softmax(sl[r_, c_].float(), -1)
        return (lt.exp() * (lt - ls)).sum(-1).mean()        # forward KL, teacher first

    order = np.random.default_rng(a.seed).permutation(len(prompts))
    t0 = time.time()
    hist = []
    for step in range(a.steps):
        qs = [prompts[order[(step * a.batch + k) % len(order)]] for k in range(a.batch)]
        L = distil_step(qs)
        engine.backward(L)
        engine.step()
        if is_main and ((step + 1) % a.probe_every == 0 or step == 0):
            hist.append({"step": step, "kl": round(float(L), 5),
                         "wall_s": round(time.time() - t0, 1)})
            log(f"[undo] step {step:4d} KL(teacher||student)={float(L):.5f} "
                f"{(time.time()-t0)/(step+1):.2f}s/step "
                f"peak={torch.cuda.max_memory_allocated(dev)/2**30:.1f} GiB")
        if (step + 1) % a.save_every == 0:
            engine.save_16bit_model(a.out, f"undo_step{step+1}.bin")
            log(f"[ckpt] {a.out}/undo_step{step+1}.bin")

    if is_main:
        R = {"study": "undo_for_mdlm", "hparams": vars(a),
             "method_source": ("UNDO, Lee et al. arXiv 2506.06278: theta_perturbed = "
                               "(1-alpha)*theta_suppressed + alpha*beta*N, then distil the "
                               "teacher's outputs into the damaged student"),
             "why": ("FINDING18 and FINDING19 show both walls concern getting a forget target "
                     "from the model's own distributions; their joint prediction is that a "
                     "working method must source it elsewhere. UNDO is that route and is the "
                     "one direction the walls do not block."),
             "ours_vs_theirs": ("the method is entirely theirs; ours is the MDLM instantiation, "
                                "the teacher choice (AFC), and the test of the walls' joint "
                                "prediction"),
             "damage": {"alpha": a.alpha, "beta": a.beta, "n_tensors": n_damaged,
                        "mean_rel_norm_change": round(delta_sum / max(base_sum, 1e-9), 5)},
             "history": hist,
             "measured": {"wall_s": round(time.time() - t0, 1),
                          "s_per_step": round((time.time() - t0) / max(a.steps, 1), 3),
                          "peak_gpu_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}}
        json.dump(R, open(f"{a.out}/report.json", "w"), indent=2)
        log(f"[done] {R['measured']['wall_s']}s {R['measured']['s_per_step']}s/step "
            f"peak={R['measured']['peak_gpu_gib']} GiB")


if __name__ == "__main__":
    main()
