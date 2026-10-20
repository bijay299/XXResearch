"""
Full-parameter feasibility probe for the EMDU/MDU comparison on 2x A100-40GB.

The paper uses full-parameter training on A100 80GB. Our envelope is two 40 GB
devices plus 491 GiB host RAM. This probe MEASURES whether the documented plan
below actually fits, and what one optimizer update costs, so the go/no-go is based
on measurement rather than arithmetic.

DOCUMENTED SHARDING / OFFLOAD PLAN (role-parallel + CPU-offloaded optimizer):
  cuda:0  student, full-parameter, bf16 params (16 GiB) + bf16 grads (16 GiB)
  cuda:1  frozen anchor, bf16 (16 GiB), no_grad, never updated
  host    AdamW state in fp32: master + exp_avg + exp_avg_sq (~96 GiB of 491 GiB)
  step    grads -> CPU, AdamW on CPU in fp32, updated params -> GPU
  batch   microbatch 1 x grad-accum 4 = effective batch 4 (paper's batch 4, accum 1)
  activations  LLaDA native activation checkpointing on the student

No LoRA, no quantization, no shortened recipe. If this does not fit the authorized
envelope, the comparison is reported as PAPER_ALIGNED_RUN_BLOCKED.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MD = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
OUT = "/data/bijaypandey/llada_emdu/artifacts/feasibility_probe.json"
MASK_ID = 126336
SEQ = 512
N_UPDATES = 3            # measure a few real updates, then extrapolate
ACCUM = 4                # microbatch 1 x 4 = effective batch 4
REMAINING_GPU_S = 21277  # carried-forward budget at probe time
TRAJECTORIES = 6         # 3 subjects x 2 methods
UPDATES_PER_TRAJ = 40    # paper's selected LLaDA RWKU step count

R = {"plan": "role-parallel 2-GPU + CPU-offloaded fp32 AdamW; no LoRA/quantization/shortened recipe",
     "seq_len": SEQ, "microbatch": 1, "accum": ACCUM, "effective_batch": ACCUM}


def gib(b):
    return round(b / 2**30, 3)


def main():
    assert torch.cuda.device_count() >= 2, "need 2 visible devices"
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)

    t0 = time.time()
    student = AutoModel.from_pretrained(MD, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16).to("cuda:0")
    student.train()
    anchor = AutoModel.from_pretrained(MD, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda:1").eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    R["load_s"] = round(time.time() - t0, 1)

    # activation checkpointing on the student
    ac = None
    for name in ("set_activation_checkpointing", "gradient_checkpointing_enable"):
        if hasattr(student, name):
            try:
                getattr(student, name)("whole_layer") if name == "set_activation_checkpointing" \
                    else getattr(student, name)()
                ac = name
                break
            except Exception as e:
                try:
                    getattr(student, name)()
                    ac = name
                    break
                except Exception:
                    R.setdefault("ac_errors", []).append(f"{name}: {type(e).__name__}: {e}")
    R["activation_checkpointing"] = ac or "NOT ENABLED"

    n_params = sum(p.numel() for p in student.parameters())
    R["n_params"] = n_params
    R["mem_after_load"] = {"cuda0_gib": gib(torch.cuda.memory_allocated(0)),
                           "cuda1_gib": gib(torch.cuda.memory_allocated(1))}
    print(f"[load] {R['load_s']}s  params={n_params:,}  ac={R['activation_checkpointing']}")
    print(f"[mem] after load cuda0={R['mem_after_load']['cuda0_gib']} "
          f"cuda1={R['mem_after_load']['cuda1_gib']} GiB")

    # ── CPU-offloaded fp32 AdamW state ────────────────────────────────────
    t0 = time.time()
    names = [n for n, _ in student.named_parameters()]
    master, m_buf, v_buf = {}, {}, {}
    try:
        for n, p in student.named_parameters():
            master[n] = p.detach().to("cpu", torch.float32).pin_memory()
            m_buf[n] = torch.zeros_like(master[n])
            v_buf[n] = torch.zeros_like(master[n])
    except RuntimeError as e:
        R["optimizer_state_alloc_error"] = f"{type(e).__name__}: {e}"
        R["fits"] = False
        R["blocker"] = "could not allocate fp32 AdamW state in host RAM"
        json.dump(R, open(OUT, "w"), indent=2)
        print("BLOCKED:", R["blocker"])
        return 2
    R["opt_state_alloc_s"] = round(time.time() - t0, 1)
    R["opt_state_host_gib"] = gib(3 * 4 * n_params)
    print(f"[opt] fp32 master+m+v on host = {R['opt_state_host_gib']} GiB "
          f"(alloc {R['opt_state_gib'] if False else R['opt_state_alloc_s']}s)")

    # ── synthetic batch at the real sequence length ───────────────────────
    ids = torch.randint(1000, 120000, (1, SEQ), device="cuda:0")
    labels = ids.clone()
    labels[:, :32] = -100            # 32-token prompt region
    attn = torch.ones_like(ids)

    lr, beta1, beta2, eps, wd = 1e-5, 0.9, 0.999, 1e-8, 0.0
    step_times = []
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)

    for step in range(N_UPDATES):
        ts = time.time()
        student.zero_grad(set_to_none=True)
        for micro in range(ACCUM):
            g = torch.Generator(device="cuda:0").manual_seed(1234 + step * 10 + micro)
            maskable = labels != -100
            t = 1e-3 + (1 - 1e-3) * torch.rand(1, device="cuda:0", generator=g)
            p_mask = t.unsqueeze(1).expand(1, SEQ)
            masked = (torch.rand((1, SEQ), device="cuda:0", generator=g) < p_mask) & maskable
            noised = torch.where(masked, MASK_ID, ids)

            logits_c = student(noised, attention_mask=attn).logits
            # anchor on the OTHER device, prompt masked (MDU-style probe shape)
            q_mask = (~maskable) & attn.bool()
            noised_u = torch.where(q_mask, MASK_ID, noised)
            with torch.no_grad():
                lu = anchor(noised_u.to("cuda:1"),
                            attention_mask=attn.to("cuda:1")).logits.to("cuda:0")
            log_pc = F.log_softmax(logits_c.float(), -1)
            log_pt = F.log_softmax(lu.float(), -1).detach()
            kl = (log_pc.exp() * (log_pc - log_pt)).sum(-1)
            loss = (kl * masked.float()).sum() / masked.sum().clamp_min(1) / ACCUM
            loss.backward()
            del logits_c, lu, log_pc, log_pt, kl

        # CPU-offloaded AdamW
        tstep = time.time()
        with torch.no_grad():
            for n, p in student.named_parameters():
                if p.grad is None:
                    continue
                gcpu = p.grad.detach().to("cpu", torch.float32)
                mb, vb, ms = m_buf[n], v_buf[n], master[n]
                mb.mul_(beta1).add_(gcpu, alpha=1 - beta1)
                vb.mul_(beta2).addcmul_(gcpu, gcpu, value=1 - beta2)
                bc1 = 1 - beta1 ** (step + 1)
                bc2 = 1 - beta2 ** (step + 1)
                denom = (vb / bc2).sqrt_().add_(eps)
                ms.addcdiv_(mb / bc1, denom, value=-lr)
                p.copy_(ms.to(p.device, p.dtype, non_blocking=True))
        opt_s = time.time() - tstep
        torch.cuda.synchronize(0)
        dt = time.time() - ts
        step_times.append({"step": step, "total_s": round(dt, 2),
                           "optimizer_s": round(opt_s, 2),
                           "fwdbwd_s": round(dt - opt_s, 2),
                           "loss": float(loss.item() * ACCUM)})
        print(f"[step {step}] total={dt:.1f}s (fwd/bwd {dt-opt_s:.1f}s, opt {opt_s:.1f}s) "
              f"loss={loss.item()*ACCUM:.5f}", flush=True)

    R["step_times"] = step_times
    R["peak_mem"] = {"cuda0_gib": gib(torch.cuda.max_memory_allocated(0)),
                     "cuda1_gib": gib(torch.cuda.max_memory_allocated(1))}
    # use the last steady-state step, not the first (warmup/alloc)
    steady = step_times[-1]["total_s"]
    R["steady_update_s"] = steady
    total_train_wall = steady * UPDATES_PER_TRAJ * TRAJECTORIES
    R["projection"] = {
        "updates_per_trajectory": UPDATES_PER_TRAJ, "trajectories": TRAJECTORIES,
        "total_updates": UPDATES_PER_TRAJ * TRAJECTORIES,
        "steady_s_per_update": steady,
        "training_wall_s": round(total_train_wall),
        "training_gpu_s_at_2_devices": round(total_train_wall * 2),
        "remaining_gpu_s_at_probe": REMAINING_GPU_S,
        "fits_training_only": total_train_wall * 2 < REMAINING_GPU_S,
        "gpu_s_left_for_eval": round(REMAINING_GPU_S - total_train_wall * 2),
    }
    R["mem_fits"] = (R["peak_mem"]["cuda0_gib"] < 40 and R["peak_mem"]["cuda1_gib"] < 40)
    R["fits"] = bool(R["mem_fits"] and R["projection"]["fits_training_only"])
    json.dump(R, open(OUT, "w"), indent=2)

    print(f"\n[peak] cuda0={R['peak_mem']['cuda0_gib']} cuda1={R['peak_mem']['cuda1_gib']} GiB "
          f"-> mem_fits={R['mem_fits']}")
    pr = R["projection"]
    print(f"[projection] {pr['total_updates']} updates x {steady:.1f}s = {pr['training_wall_s']}s wall "
          f"= {pr['training_gpu_s_at_2_devices']} GPU-s on 2 devices")
    print(f"[budget] remaining {REMAINING_GPU_S} GPU-s -> training fits: {pr['fits_training_only']}; "
          f"left for eval: {pr['gpu_s_left_for_eval']} GPU-s")
    print(f"[FEASIBLE] {R['fits']}")
    return 0 if R["fits"] else 3


if __name__ == "__main__":
    sys.exit(main())
