"""
LLaDA-8B-Instruct: exactly ONE LoRA optimizer step (feasibility gate).

Starts from the pristine downloaded checkpoint (rev 08b83a6f...), attaches a
disposable LoRA adapter, and performs one optimizer update on a tiny benign
prompt->answer fixture using the author's masked-response SFT recipe from the
pinned GUIDELINES.md.

SCOPE: this is a LoRA feasibility test. It does NOT establish that
full-parameter MDU training fits this hardware.

Settings frozen in RUN_MANIFEST.md (sha256 3c9fee37...) before execution.
"""
import hashlib
import json
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

W = "/data/bijaypandey/llada_smoke"
MODEL_DIR = f"{W}/model/LLaDA-8B-Instruct"

MASK_ID = 126336
EOS_ID = 126081
SEED = 1234
SEQ_LEN = 128           # <= 128 tokens
LORA_R = 8
LORA_ALPHA = 16
LR = 1e-4
TARGETS = ["q_proj", "k_proj", "v_proj", "attn_out"]   # ff_out excluded: LM-head name clash

# Tiny predefined benign fixture (batch size 1).
FIXTURE = {
    "prompt": "What color is a clear daytime sky?",
    "answer": "A clear daytime sky is blue.",
}


def forward_process(input_ids, eps=1e-3, generator=None):
    """Verbatim from pinned GUIDELINES.md (LLaDA pre-training / SFT noising)."""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device, generator=generator)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device, generator=generator) < p_mask
    noisy_batch = torch.where(masked_indices, MASK_ID, input_ids)
    return noisy_batch, masked_indices, p_mask


def tensor_sha(t):
    return hashlib.sha256(
        t.detach().to(torch.float32).cpu().numpy().tobytes()
    ).hexdigest()


def hash_base_params(model):
    """SHA-256 of every non-LoRA (base) parameter. Full coverage, one tensor
    at a time to bound host memory."""
    out = {}
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        out[n] = tensor_sha(p)
    return out


def main():
    report = {
        "task": "llada_one_step_lora_feasibility",
        "scope_note": "LoRA feasibility only; does NOT establish full-parameter MDU training fits this hardware.",
        "model_dir": MODEL_DIR,
        "model_revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "sft_recipe_source": "ML-GSAI/LLaDA@9182493720ed723ef8031210d85959364e51cbe0:GUIDELINES.md#SFT",
    }

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")

    # ---- build the fixture sequence ----
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": FIXTURE["prompt"]}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(templated, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(FIXTURE["answer"], add_special_tokens=False)["input_ids"] + [EOS_ID]
    prompt_len = len(prompt_ids)
    seq = prompt_ids + answer_ids
    assert len(seq) <= SEQ_LEN, f"fixture too long: {len(seq)}"
    # Pad the ANSWER region with EOS, per GUIDELINES SFT data format.
    seq = seq + [EOS_ID] * (SEQ_LEN - len(seq))
    input_ids = torch.tensor([seq], device="cuda")
    report["fixture"] = dict(
        **FIXTURE, templated_prompt=templated, prompt_len=prompt_len,
        answer_len_tokens=len(answer_ids), seq_len=SEQ_LEN,
        batch_size=1, input_ids=seq,
    )

    # ---- LoRA ----
    lcfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        target_modules=TARGETS, init_lora_weights=True,
    )
    model = get_peft_model(model, lcfg)

    # Base frozen; LoRA params in fp32 for a stable optimizer step.
    n_lora, n_base_trainable = 0, 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
            n_lora += 1
        else:
            if p.requires_grad:
                n_base_trainable += 1
            p.requires_grad_(False)

    trainable_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    report["lora"] = dict(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        target_modules=TARGETS, excluded_ff_out_reason="name collides with top-level LM head model.transformer.ff_out",
        init_lora_weights="True (PEFT default: A Kaiming-uniform, B zeros)",
        optimizer="AdamW", lr=LR, betas=[0.9, 0.999], weight_decay=0.0,
        base_dtype="bfloat16", lora_dtype="float32",
        n_lora_tensors=n_lora,
        n_base_tensors_requires_grad_before_freeze=n_base_trainable,
        n_trainable_params=n_trainable_params,
        n_total_params=n_total_params,
        trainable_pct=round(100 * n_trainable_params / n_total_params, 4),
        n_trainable_tensors=len(trainable_names),
    )
    print(f"[lora] trainable {n_trainable_params:,} / {n_total_params:,} "
          f"({100*n_trainable_params/n_total_params:.4f}%) in {len(trainable_names)} tensors")

    # ---- GATE 4: optimizer membership == intended trainable set ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.999), weight_decay=0.0,
    )
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    intended_ids = {id(p) for n, p in model.named_parameters() if p.requires_grad}
    gate4 = (opt_ids == intended_ids)
    report["gate4_optimizer_membership"] = dict(
        passed=bool(gate4), n_in_optimizer=len(opt_ids), n_intended=len(intended_ids),
        set_equal=bool(gate4),
        method="set equality of python id() over optimizer.param_groups vs params with requires_grad=True",
    )

    # ---- snapshot BEFORE ----
    print("[hash] snapshotting all base params (pre)...")
    t0 = time.time()
    base_pre = hash_base_params(model)
    print(f"[hash] {len(base_pre)} base tensors in {time.time()-t0:.1f}s")
    lora_pre = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # ---- masked-response SFT loss (pinned GUIDELINES recipe) ----
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    noisy_batch, _, p_mask = forward_process(input_ids, eps=1e-3, generator=gen)

    # Do not add noise to the prompt -> prompt tokens stay uncorrupted.
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(
        noisy_batch.size(0), noisy_batch.size(1)
    )
    prompt_mask_bool = token_positions < torch.tensor([prompt_len], device="cuda").unsqueeze(1)
    noisy_batch[prompt_mask_bool] = input_ids[prompt_mask_bool]

    # Answer length = number of non-prompt positions, INCLUDING padded EOS.
    prompt_mask_int = prompt_mask_bool.to(torch.int64)
    answer_lengths = torch.sum((1 - prompt_mask_int), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

    masked_indices = (noisy_batch == MASK_ID)

    # prompt region must be fully uncorrupted, supervised mask must be nonempty
    prompt_intact = bool(torch.equal(noisy_batch[prompt_mask_bool], input_ids[prompt_mask_bool]))
    n_supervised = int(masked_indices.sum().item())
    n_masked_in_prompt = int((masked_indices & prompt_mask_bool).sum().item())
    assert prompt_intact, "prompt tokens were corrupted"
    assert n_supervised > 0, "supervised mask is empty"
    assert n_masked_in_prompt == 0, "mask leaked into prompt region"

    logits = model(noisy_batch).logits
    token_loss = F.cross_entropy(
        logits[masked_indices].to(torch.float32), input_ids[masked_indices], reduction="none"
    ) / p_mask[masked_indices]
    loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]

    report["loss_computation"] = dict(
        recipe="GUIDELINES.md SFT: CE/p_mask, normalized by answer_lengths, summed, /batch",
        eps=1e-3, p_mask_value=float(p_mask[0, 0].item()),
        n_supervised_positions=n_supervised,
        n_masked_positions_in_prompt=n_masked_in_prompt,
        prompt_region_uncorrupted=prompt_intact,
        answer_lengths=int(answer_lengths[0, 0].item()),
        seed=SEED,
        loss_value=float(loss.item()),
        loss_finite=bool(torch.isfinite(loss).item()),
        cross_entropy_in_fp32=True,
    )
    print(f"[loss] {loss.item():.6f} finite={torch.isfinite(loss).item()} "
          f"supervised={n_supervised} p_mask={p_mask[0,0].item():.4f} "
          f"answer_len={answer_lengths[0,0].item()}")

    # ---- backward + exactly ONE optimizer step ----
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    n_with_grad = sum(1 for g in grads.values() if g is not None)
    all_grads_finite = all(bool(torch.isfinite(g).all().item()) for g in grads.values() if g is not None)
    gnorm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=float("inf")
    )
    n_nonzero_grad = sum(
        1 for g in grads.values() if g is not None and bool((g != 0).any().item())
    )
    report["gate1_finite_loss_and_grads"] = dict(
        passed=bool(torch.isfinite(loss).item() and all_grads_finite and torch.isfinite(gnorm).item()),
        loss_finite=bool(torch.isfinite(loss).item()),
        all_grads_finite=all_grads_finite,
        grad_norm=float(gnorm.item()), grad_norm_finite=bool(torch.isfinite(gnorm).item()),
        n_trainable_with_grad=n_with_grad,
        n_trainable_with_nonzero_grad=n_nonzero_grad,
    )
    print(f"[grad] norm={gnorm.item():.6f} finite={all_grads_finite} "
          f"with_grad={n_with_grad} nonzero_grad={n_nonzero_grad}")

    opt.step()
    steps_taken = 1
    opt.zero_grad(set_to_none=True)

    # ---- GATE 5: exactly one update ----
    adam_steps = sorted({
        int(s["step"].item()) for s in opt.state.values() if "step" in s
    })
    report["gate5_exactly_one_update"] = dict(
        passed=bool(steps_taken == 1 and adam_steps == [1]),
        optimizer_step_calls=steps_taken,
        adamw_internal_step_counters=adam_steps,
        method="opt.step() invoked once; AdamW per-param internal 'step' state inspected",
    )
    print(f"[step] calls={steps_taken} adamw_step_counters={adam_steps}")

    # ---- GATE 2: >=1 intended trainable param changed, finite & nonzero ----
    changed, unchanged = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = (p.detach() - lora_pre[n]).abs().max().item()
        rec = dict(name=n, max_abs_delta=d, finite=bool(torch.isfinite(p).all().item()))
        (changed if d > 0 else unchanged).append(rec)
    changed.sort(key=lambda r: -r["max_abs_delta"])
    gate2 = len(changed) > 0 and all(r["finite"] for r in changed) and all(
        r["max_abs_delta"] > 0 and r["max_abs_delta"] < float("inf") for r in changed
    )
    report["gate2_trainable_changed"] = dict(
        passed=bool(gate2),
        n_changed=len(changed), n_unchanged=len(unchanged),
        max_abs_delta_overall=changed[0]["max_abs_delta"] if changed else 0.0,
        top5_changed=changed[:5],
        unchanged_sample=unchanged[:5],
        note=("Zero-initialized lora_B legitimately yields zero gradient for its "
              "partner lora_A on the first step; not all LoRA tensors are required to change."),
        n_changed_lora_A=sum(1 for r in changed if "lora_A" in r["name"]),
        n_changed_lora_B=sum(1 for r in changed if "lora_B" in r["name"]),
        n_unchanged_lora_A=sum(1 for r in unchanged if "lora_A" in r["name"]),
        n_unchanged_lora_B=sum(1 for r in unchanged if "lora_B" in r["name"]),
    )
    print(f"[gate2] changed={len(changed)} unchanged={len(unchanged)} "
          f"max_delta={changed[0]['max_abs_delta'] if changed else 0}")

    # ---- GATE 3: base params frozen and unchanged (FULL coverage) ----
    print("[hash] snapshotting all base params (post)...")
    base_post = hash_base_params(model)
    drifted = [n for n in base_pre if base_pre[n] != base_post.get(n)]
    base_requires_grad = [n for n, p in model.named_parameters()
                          if "lora_" not in n and p.requires_grad]
    gate3 = (len(drifted) == 0 and len(base_requires_grad) == 0
             and set(base_pre) == set(base_post))
    report["gate3_base_frozen"] = dict(
        passed=bool(gate3),
        method=("SHA-256 over float32-cast raw bytes of EVERY non-LoRA parameter tensor, "
                "compared pre-step vs post-step; plus requires_grad audit of all base tensors"),
        coverage="100% of base parameter tensors (all non-LoRA named_parameters)",
        n_base_tensors_hashed=len(base_pre),
        n_drifted=len(drifted), drifted_names=drifted[:10],
        n_base_tensors_requires_grad_after_freeze=len(base_requires_grad),
    )
    print(f"[gate3] hashed={len(base_pre)} drifted={len(drifted)} "
          f"base_requires_grad={len(base_requires_grad)}")

    peak = torch.cuda.max_memory_allocated()
    report["resources"] = dict(
        peak_gpu_mem_bytes=int(peak), peak_gpu_mem_gib=round(peak / 2**30, 3),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
    )

    gates = ["gate1_finite_loss_and_grads", "gate2_trainable_changed",
             "gate3_base_frozen", "gate4_optimizer_membership",
             "gate5_exactly_one_update"]
    report["all_gates_passed"] = all(report[g]["passed"] for g in gates)
    report["gate_summary"] = {g: report[g]["passed"] for g in gates}

    # ---- save adapter only (no base weights) ----
    adapter_dir = f"{W}/artifacts/lora_adapter_one_step"
    model.save_pretrained(adapter_dir)
    report["adapter_dir"] = adapter_dir

    with open(f"{W}/artifacts/lora_step_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[peak_gpu_mem] {peak/2**30:.3f} GiB")
    print(f"[ALL GATES PASSED] {report['all_gates_passed']}")
    print(f"[gates] {report['gate_summary']}")


if __name__ == "__main__":
    main()
