"""
LLaDA-8B-Instruct native masked-diffusion inference smoke test.

Uses the author's masked-diffusion `generate()` from the pinned LLaDA repo
(ML-GSAI/LLaDA @ 9182493720ed723ef8031210d85959364e51cbe0), imported UNMODIFIED.
This is NOT an autoregressive generation pipeline.

Scope: inference smoke test only. Not benchmark accuracy, not paper reproduction.
Settings are frozen in RUN_MANIFEST.md (sha256 3c9fee37...) before execution.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

W = "/data/bijaypandey/llada_smoke"
MODEL_DIR = f"{W}/model/LLaDA-8B-Instruct"
sys.path.insert(0, f"{W}/repos/llada_official")

# The author's implementation, imported unmodified.
from generate import generate  # noqa: E402

MASK_ID = 126336
EOS_ID = 126081
SEED = 42

# Frozen panel of 4 short benign prompts. Fixed before execution.
# No prompt replacement, no repeated sampling.
PROMPTS = [
    "What is the capital of France?",
    "Name three primary colors.",
    "Explain in one sentence what photosynthesis is.",
    "What is 17 plus 25?",
]

DECODE = dict(
    steps=64,
    gen_length=64,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=MASK_ID,
)


class MaskScheduleProbe(torch.nn.Module):
    """Wraps the model to record how many positions are still [MASK] at each
    denoising step, by observing the input the author's generate() passes in.
    Does not alter model behaviour or the generation algorithm."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.counts = []

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    def forward(self, x, attention_mask=None):
        self.counts.append(int((x == MASK_ID).sum().item()))
        return self.model(x, attention_mask=attention_mask)

    def __call__(self, x, attention_mask=None):
        return self.forward(x, attention_mask=attention_mask)


def main():
    t_load0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    load_s = time.time() - t_load0

    assert tokenizer.pad_token_id != MASK_ID, "pad id must differ from mask id"
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    print(f"[load] {load_s:.1f}s  dtype={next(model.parameters()).dtype}")
    print(f"[ids] mask={MASK_ID} eos={EOS_ID} pad={tokenizer.pad_token_id} "
          f"vocab={model.config.vocab_size}")

    results = []
    torch.cuda.reset_peak_memory_stats()

    for i, raw_prompt in enumerate(PROMPTS):
        # Fixed seed per prompt, set before generation.
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Instruct model: apply chat template.
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tokenizer(templated, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")        # batch size 1
        attn = enc["attention_mask"].to("cuda")
        assert input_ids.shape[0] == 1, "batch size must be 1"

        probe = MaskScheduleProbe(model)
        t0 = time.time()
        with torch.no_grad():
            out = generate(probe, input_ids, attn, **DECODE)
        torch.cuda.synchronize()
        runtime_s = time.time() - t0

        gen_ids = out[0, input_ids.shape[1]:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        decoded_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)

        # ---- schedule / validity checks ----
        counts = probe.counts
        # Masks remaining must be non-increasing across the trajectory.
        monotone = all(counts[j] >= counts[j + 1] for j in range(len(counts) - 1))
        n_mask_left = int((gen_ids == MASK_ID).sum().item())
        nonempty = len(decoded.strip()) > 0

        res = dict(
            index=i,
            prompt=raw_prompt,
            templated_prompt=templated,
            prompt_token_ids=input_ids[0].tolist(),
            prompt_len=int(input_ids.shape[1]),
            output_token_ids=gen_ids.tolist(),
            decoded=decoded,
            decoded_with_special=decoded_raw,
            seed=SEED,
            decoding=DECODE,
            runtime_s=round(runtime_s, 3),
            forward_calls=len(counts),
            mask_count_trajectory=counts,
            schedule_monotone_non_increasing=monotone,
            masks_remaining_in_output=n_mask_left,
            output_nonempty=nonempty,
            passed=bool(monotone and n_mask_left == 0 and nonempty),
        )
        results.append(res)
        print(f"\n--- prompt {i}: {raw_prompt!r}")
        print(f"    runtime={runtime_s:.2f}s forwards={len(counts)} "
              f"monotone={monotone} masks_left={n_mask_left} nonempty={nonempty}")
        print(f"    traj[:6]={counts[:6]} ... traj[-3:]={counts[-3:]}")
        print(f"    output: {decoded!r}")

    peak_mem = torch.cuda.max_memory_allocated()
    summary = dict(
        task="llada_native_masked_diffusion_inference_smoke",
        generate_impl="ML-GSAI/LLaDA@9182493720ed723ef8031210d85959364e51cbe0:generate.py:generate",
        autoregressive=False,
        model_dir=MODEL_DIR,
        model_revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",
        dtype="bfloat16",
        seed=SEED,
        decoding=DECODE,
        n_prompts=len(PROMPTS),
        model_load_s=round(load_s, 2),
        total_gen_s=round(sum(r["runtime_s"] for r in results), 2),
        peak_gpu_mem_bytes=int(peak_mem),
        peak_gpu_mem_gib=round(peak_mem / 2**30, 3),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        all_passed=all(r["passed"] for r in results),
        results=results,
    )
    out_path = f"{W}/artifacts/inference_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[peak_gpu_mem] {peak_mem/2**30:.3f} GiB")
    print(f"[all_passed] {summary['all_passed']}")
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
