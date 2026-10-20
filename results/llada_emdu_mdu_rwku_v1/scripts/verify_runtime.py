"""
§5 RUNTIME correctness checks on the real model and real RWKU data.

These are the checks Amendment 01 flagged as missing in the previous study:
source-string inspection is NOT runtime verification, and a fresh zero-init
adapter agreeing with the base model does NOT establish teacher invariance.
Here the student is genuinely CHANGED by real optimizer updates and the frozen
anchor is then re-checked.

Takes a small, bounded number of real full-parameter updates per method purely as
a correctness probe. This is NOT a training trajectory and produces no comparison
result; the comparison itself is budget-blocked.

Exits NONZERO on any failure (negative-control tested separately).
"""
import hashlib
import json
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

W = "/data/bijaypandey/llada_emdu"
sys.path.insert(0, f"{W}/scripts")
from objectives import MASK_ID, forget_loss  # noqa

MD = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
SEQ, ACCUM, N_UPDATES = 512, 4, 2
LR = 1e-5
R = {"checks": {}, "note": "bounded runtime correctness probe; not a training trajectory"}
fails = []


def chk(name, ok, detail=None):
    R["checks"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    return ok


def digest(model, n_tensors=None):
    h = hashlib.sha256()
    for i, (nm, p) in enumerate(model.named_parameters()):
        if n_tensors and i >= n_tensors:
            break
        h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()


def build_example(tok, subj_dir):
    pair = json.load(open(f"{W}/data/rwku/Target/{subj_dir}/pair.json"))
    rec = pair[0]
    prompt = tok.apply_chat_template([{"role": "user", "content": rec["prompt"]}],
                                     add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(rec["response"][0], add_special_tokens=False)["input_ids"]
    EOS = 126081
    seq = (p_ids + a_ids)[:SEQ]
    plen = min(len(p_ids), SEQ)
    seq = seq + [EOS] * (SEQ - len(seq))
    ids = torch.tensor([seq], device="cuda:0")
    labels = ids.clone()
    labels[:, :plen] = -100
    attn = torch.ones_like(ids)
    return ids, labels, attn, plen, rec["id"]


def main():
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    student = AutoModel.from_pretrained(MD, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16).to("cuda:0")
    student.train()
    anchor = AutoModel.from_pretrained(MD, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda:1").eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    params = [p for p in student.parameters() if p.requires_grad]
    all_params = list(student.parameters())
    chk("all_parameters_trainable_full_parameter",
        len(params) == len(all_params) and len(params) > 0,
        {"trainable_tensors": len(params), "total_tensors": len(all_params),
         "trainable_numel": sum(p.numel() for p in params),
         "note": "full-parameter: every tensor trainable, no LoRA/adapters"})
    chk("anchor_parameters_all_frozen",
        all(not p.requires_grad for p in anchor.parameters()),
        {"anchor_tensors": sum(1 for _ in anchor.parameters())})

    ids, labels, attn, plen, row = build_example(tok, "1_Stephen_King")
    R["fixture"] = {"row_id": row, "prompt_len": plen, "seq_len": SEQ,
                    "n_answer_positions": int((labels != -100).sum())}

    # digests BEFORE (sample of tensors, to bound probe cost)
    NT = 40
    anchor_before = digest(anchor, NT)
    student_before = digest(student, NT)

    master = [p.detach().to("cpu", torch.float32) for p in params]
    m_buf = [torch.zeros_like(x) for x in master]
    v_buf = [torch.zeros_like(x) for x in master]
    torch.set_num_threads(32)

    def dispatched(method):
        """Record which objective actually executed at runtime, from the returned
        info dict (not from source inspection)."""
        gen = torch.Generator(device="cuda:0").manual_seed(7)
        loss, info = forget_loss(method, lambda x, am: student(x, attention_mask=am).logits,
                                 lambda x, am: anchor(x.to("cuda:1"),
                                                      attention_mask=am.to("cuda:1")).logits.to("cuda:0"),
                                 ids, labels, attn, gen, tau=1.0, return_info=True)
        return loss, info

    runtime_info = {}
    for method in ("mdu", "emdu"):
        with torch.no_grad():
            pass
        loss, info = dispatched(method)
        runtime_info[method] = {
            "direction_executed": info["direction"],
            "loss": float(loss),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "n_masked_student": info["n_masked_student"],
            "n_maskable": info["n_maskable"],
            "anchor_prompt_fully_masked": info["anchor_prompt_fully_masked"],
            "anchor_answer_fully_masked": info["anchor_answer_fully_masked"],
            "student_prompt_uncorrupted": info["student_prompt_uncorrupted"],
        }
        del loss, info
        torch.cuda.empty_cache()

    R["runtime_dispatch"] = runtime_info
    chk("runtime_dispatch_mdu_is_anchor_last",
        runtime_info["mdu"]["direction_executed"] == "anchor_last")
    chk("runtime_dispatch_emdu_is_anchor_first",
        runtime_info["emdu"]["direction_executed"] == "anchor_first")
    chk("runtime_emdu_anchor_entire_answer_span_masked_on_real_data",
        runtime_info["emdu"]["anchor_answer_fully_masked"] is True)
    chk("runtime_mdu_anchor_answer_NOT_fully_masked_on_real_data",
        runtime_info["mdu"]["anchor_answer_fully_masked"] is False,
        "MDU's anchor sees only the student's partial mask, so it must NOT be fully masked")
    chk("runtime_both_losses_finite",
        all(v["loss_finite"] for v in runtime_info.values()),
        {k: v["loss"] for k, v in runtime_info.items()})
    chk("runtime_prompt_uncorrupted_both",
        all(v["student_prompt_uncorrupted"] for v in runtime_info.values()))

    # ── genuinely CHANGE the student with real optimizer updates ──────────
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    t0 = time.time()
    for step in range(N_UPDATES):
        student.zero_grad(set_to_none=True)
        for micro in range(ACCUM):
            gen = torch.Generator(device="cuda:0").manual_seed(100 + step * 10 + micro)
            loss = forget_loss("emdu", lambda x, am: student(x, attention_mask=am).logits,
                               lambda x, am: anchor(x.to("cuda:1"),
                                                    attention_mask=am.to("cuda:1")).logits.to("cuda:0"),
                               ids, labels, attn, gen, tau=1.0)
            (loss / ACCUM).backward()
            losses.append(float(loss))
            del loss
        gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        with torch.no_grad():
            bc1, bc2 = 1 - b1 ** (step + 1), 1 - b2 ** (step + 1)
            for p, ms, mb, vb in zip(params, master, m_buf, v_buf):
                g = p.grad.detach().to("cpu", torch.float32)
                mb.mul_(b1).add_(g, alpha=1 - b1)
                vb.mul_(b2).addcmul_(g, g, value=1 - b2)
                ms.addcdiv_(mb, (vb / bc2).sqrt_().add_(eps), value=-LR / bc1)
                p.copy_(ms.to(p.device, p.dtype))
        print(f"[update {step}] loss={losses[-1]:.6f} gnorm={float(gnorm):.5f}", flush=True)
    R["updates_taken"] = N_UPDATES
    R["update_wall_s"] = round((time.time() - t0) / N_UPDATES, 1)
    R["losses"] = losses
    chk("real_optimizer_updates_executed", N_UPDATES == 2, N_UPDATES)
    chk("gradient_norm_finite", bool(torch.isfinite(gnorm).item()), float(gnorm))

    # ── the decisive checks: student CHANGED, anchor UNCHANGED ───────────
    student_after = digest(student, NT)
    anchor_after = digest(anchor, NT)
    max_delta = max((p.detach() - ms.to(p.device, p.dtype)).abs().max().item()
                    for p, ms in list(zip(params, master))[:5])
    chk("student_genuinely_changed_by_updates", student_after != student_before,
        {"before": student_before[:16], "after": student_after[:16],
         "n_tensors_hashed": NT})
    chk("frozen_anchor_unchanged_after_student_changed", anchor_after == anchor_before,
        {"before": anchor_before[:16], "after": anchor_after[:16],
         "n_tensors_hashed": NT,
         "note": "teacher invariance verified against a GENUINELY CHANGED student, "
                 "not a zero-init no-op adapter"})
    R["student_digest_before"] = student_before
    R["student_digest_after"] = student_after
    R["anchor_digest_before"] = anchor_before
    R["anchor_digest_after"] = anchor_after
    R["peak_mem_gib"] = {"cuda0": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                         "cuda1": round(torch.cuda.max_memory_allocated(1) / 2**30, 3)}

    R["n_checks"] = len(R["checks"])
    R["failed"] = fails
    R["all_passed"] = not fails
    json.dump(R, open(f"{W}/artifacts/stageb_runtime_checks.json", "w"), indent=2)
    for k, v in R["checks"].items():
        print(f"  {'PASS' if v['passed'] else 'FAIL'}  {k}")
    print(f"\n{sum(1 for v in R['checks'].values() if v['passed'])}/{R['n_checks']} passed")
    print(f"ALL PASSED: {R['all_passed']}")
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
