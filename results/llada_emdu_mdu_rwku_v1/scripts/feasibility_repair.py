"""
Feasibility probe, ATTEMPT 2 (the one permitted technical repair).

Attempt 1 (preserved: logs/b0002_*, artifacts/feasibility_probe.json) measured
43.9-67.8 s per full-parameter update, of which 98% was the CPU-offloaded AdamW,
and 9,078 CPU-core-s per update. Both the GPU-second and CPU-core-hour budgets
projected over.

This repair replaces the naive per-tensor CPU AdamW (which allocated fp32
temporaries per parameter) with an in-place `_foreach` implementation using
preallocated buffers and no temporaries, and measures the SAME update at several
CPU thread counts. The question it answers is not "is my code faster" but:

    does ANY point on the thread-count trade-off curve fit BOTH the remaining
    GPU-second budget and the 20 CPU-core-hour cap, for 240 updates?

Because the optimizer runs on CPU while both GPUs sit allocated and idle,
core-seconds ~ threads x wall while GPU-seconds ~ 2 x wall. Fewer threads saves
core-hours but costs GPU-seconds, and vice versa. This measures both ends.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel

MD = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
OUT = "/data/bijaypandey/llada_emdu/artifacts/feasibility_repair.json"
MASK_ID = 126336
SEQ, ACCUM = 512, 4
THREAD_SETTINGS = [255, 32, 8]
TOTAL_UPDATES = 240                 # 3 subjects x 2 methods x 40 updates
REMAINING_GPU_S = 20729             # at repair time
CPU_CAP_CORE_S = 20 * 3600
CPU_ALREADY_CORE_S = 27233          # consumed by attempt 1


def gib(b):
    return round(b / 2**30, 3)


def main():
    R = {"attempt": 2, "repair": "in-place _foreach CPU AdamW, preallocated buffers, no temporaries",
         "attempt1_preserved": "artifacts/feasibility_probe.json",
         "seq_len": SEQ, "accum": ACCUM, "total_updates_planned": TOTAL_UPDATES,
         "remaining_gpu_s": REMAINING_GPU_S, "cpu_already_core_s": CPU_ALREADY_CORE_S}

    student = AutoModel.from_pretrained(MD, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16).to("cuda:0")
    student.train()
    anchor = AutoModel.from_pretrained(MD, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda:1").eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    params = [p for p in student.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    R["n_params"] = n

    # Preallocated CPU state; NOT pinned (attempt 1 spent 56.6 s pinning 89.6 GiB)
    master = [p.detach().to("cpu", torch.float32) for p in params]
    m_buf = [torch.zeros_like(x) for x in master]
    v_buf = [torch.zeros_like(x) for x in master]
    g_buf = [torch.zeros_like(x) for x in master]      # reused grad landing buffer
    d_buf = [torch.zeros_like(x) for x in master]      # reused denom buffer
    R["cpu_state_gib"] = gib(5 * 4 * n)

    ids = torch.randint(1000, 120000, (1, SEQ), device="cuda:0")
    labels = ids.clone(); labels[:, :32] = -100
    attn = torch.ones_like(ids)
    lr, b1, b2, eps = 1e-5, 0.9, 0.999, 1e-8

    def one_update(step):
        student.zero_grad(set_to_none=True)
        for micro in range(ACCUM):
            g = torch.Generator(device="cuda:0").manual_seed(99 + step * 10 + micro)
            maskable = labels != -100
            t = 1e-3 + (1 - 1e-3) * torch.rand(1, device="cuda:0", generator=g)
            pm = t.unsqueeze(1).expand(1, SEQ)
            masked = (torch.rand((1, SEQ), device="cuda:0", generator=g) < pm) & maskable
            noised = torch.where(masked, MASK_ID, ids)
            lc = student(noised, attention_mask=attn).logits
            q = (~maskable) & attn.bool()
            au = torch.where(q, MASK_ID, noised)
            with torch.no_grad():
                lu = anchor(au.to("cuda:1"), attention_mask=attn.to("cuda:1")).logits.to("cuda:0")
            lpc = F.log_softmax(lc.float(), -1); lpt = F.log_softmax(lu.float(), -1).detach()
            kl = (lpc.exp() * (lpc - lpt)).sum(-1)
            ((kl * masked.float()).sum() / masked.sum().clamp_min(1) / ACCUM).backward()
            del lc, lu, lpc, lpt, kl
        torch.cuda.synchronize(0)
        t_opt = time.time()
        with torch.no_grad():
            # D2H grads into preallocated fp32 buffers (no new allocations)
            for gb, p in zip(g_buf, params):
                gb.copy_(p.grad, non_blocking=False)
            bc1 = 1 - b1 ** (step + 1); bc2 = 1 - b2 ** (step + 1)
            torch._foreach_mul_(m_buf, b1)
            torch._foreach_add_(m_buf, g_buf, alpha=1 - b1)
            torch._foreach_mul_(v_buf, b2)
            torch._foreach_addcmul_(v_buf, g_buf, g_buf, value=1 - b2)
            for db, vb in zip(d_buf, v_buf):
                torch.sqrt(vb, out=db)
            torch._foreach_div_(d_buf, bc2 ** 0.5)
            torch._foreach_add_(d_buf, eps)
            torch._foreach_addcdiv_(master, m_buf, d_buf, value=-lr / bc1)
            for p, ms in zip(params, master):
                p.copy_(ms, non_blocking=False)
        torch.cuda.synchronize(0)
        return time.time() - t_opt

    results = []
    torch.cuda.reset_peak_memory_stats(0); torch.cuda.reset_peak_memory_stats(1)
    step = 0
    for nt in THREAD_SETTINGS:
        torch.set_num_threads(nt)
        one_update(step); step += 1                    # warm
        t0 = time.time(); opt_s = one_update(step); step += 1
        wall = time.time() - t0
        core_s = nt * opt_s                            # optimizer is the threaded part
        proj_wall = wall * TOTAL_UPDATES
        proj_gpu = proj_wall * 2                       # 2 devices allocated throughout
        proj_core = core_s * TOTAL_UPDATES + CPU_ALREADY_CORE_S
        rec = {"threads": nt, "update_wall_s": round(wall, 2), "optimizer_s": round(opt_s, 2),
               "optimizer_core_s_est": round(core_s),
               "proj_training_wall_s": round(proj_wall),
               "proj_training_gpu_s": round(proj_gpu),
               "proj_total_cpu_core_s": round(proj_core),
               "proj_total_cpu_core_h": round(proj_core / 3600, 1),
               "gpu_budget_ok": proj_gpu < REMAINING_GPU_S,
               "gpu_s_left_for_eval": round(REMAINING_GPU_S - proj_gpu),
               "cpu_budget_ok": proj_core < CPU_CAP_CORE_S,
               "both_ok": bool(proj_gpu < REMAINING_GPU_S and proj_core < CPU_CAP_CORE_S)}
        results.append(rec)
        print(f"[threads={nt:>3}] update={wall:.1f}s (opt {opt_s:.1f}s) -> "
              f"240 updates = {proj_gpu:.0f} GPU-s (ok={rec['gpu_budget_ok']}, "
              f"eval left {rec['gpu_s_left_for_eval']}), "
              f"{rec['proj_total_cpu_core_h']} core-h (ok={rec['cpu_budget_ok']}) "
              f"=> BOTH={rec['both_ok']}", flush=True)

    R["tradeoff"] = results
    R["peak_mem"] = {"cuda0_gib": gib(torch.cuda.max_memory_allocated(0)),
                     "cuda1_gib": gib(torch.cuda.max_memory_allocated(1))}
    R["memory_fits"] = R["peak_mem"]["cuda0_gib"] < 40 and R["peak_mem"]["cuda1_gib"] < 40
    R["any_setting_fits_both"] = any(r["both_ok"] for r in results)
    R["best_speedup_vs_attempt1"] = round(43.87 / min(r["update_wall_s"] for r in results), 2)
    R["verdict"] = ("FEASIBLE" if R["any_setting_fits_both"] else "PAPER_ALIGNED_RUN_BLOCKED")
    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\n[peak] {R['peak_mem']} memory_fits={R['memory_fits']}")
    print(f"[speedup vs attempt 1] {R['best_speedup_vs_attempt1']}x")
    print(f"[VERDICT] {R['verdict']}")
    return 0 if R["any_setting_fits_both"] else 3


if __name__ == "__main__":
    sys.exit(main())
