#!/usr/bin/env python3
"""Smoke test + measured forecast + one correctness check, on ONE GPU.

Checks:
 1. FLUX.1-dev loads in bf16 on a single 40GB device; reports peak memory.
 2. The two BLOCK intervention sites are the modules the paper/code say they are,
    with the shapes the transcoders must match.
 3. CORRECTNESS CHECK for a compute-saving refactor: the pooled (CLIP) embedding used by
    the pooled-text transcoder is produced by FluxPipeline._get_clip_prompt_embeds alone,
    so collecting 330k pooled pairs does NOT need T5-XXL. We assert bit-identity against
    the release's encode_prompt() path on a sample before relying on it.
 4. One generation at the pinned arguments, timed, to forecast the image budget.
"""
import os, sys, time, json, argparse
import torch

REPO = "/data/bijaypandey/cuig_pilot/block_object_sprint/code/block-concept-removal"
sys.path.insert(0, REPO)
sys.path.insert(0, "/data/bijaypandey/cuig_pilot/block_object_sprint/code")
os.chdir(REPO)

from diffusers import FluxPipeline

FLUX = os.environ["FLUX_PATH"]
OUT = "/data/bijaypandey/cuig_pilot/block_object_sprint/artifacts/smoke"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=1)
    ap.add_argument("--check-n", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    dev = "cuda:0"

    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    rep["load_seconds"] = round(time.time() - t0, 1)
    rep["peak_mem_after_load_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # ---- 2. intervention sites
    ce = pipe.transformer.context_embedder
    te = pipe.transformer.time_text_embed.text_embedder
    rep["context_embedder"] = {"type": type(ce).__name__,
                               "in": int(ce.in_features), "out": int(ce.out_features),
                               "dtype": str(ce.weight.dtype)}
    rep["text_embedder"] = {"type": type(te).__name__, "repr": str(te),
                            "in": int(te.linear_1.in_features),
                            "hidden": int(te.linear_1.out_features),
                            "out": int(te.linear_2.out_features),
                            "named_children": [(n, type(m).__name__) for n, m in te.named_children()]}
    print("context_embedder:", rep["context_embedder"])
    print("text_embedder   :", rep["text_embedder"])

    # ---- 3. correctness check for the T5-skip refactor
    prompts = [f"a photo of object number {i} in Van Gogh style." for i in range(a.check_n)]
    with torch.no_grad():
        pe, pooled_ref, _ = pipe.encode_prompt(
            prompts, None, device=dev, num_images_per_prompt=1, max_sequence_length=512)
        pooled_fast = pipe._get_clip_prompt_embeds(
            prompt=prompts, device=dev, num_images_per_prompt=1)
    same = torch.equal(pooled_ref, pooled_fast)
    rep["pooled_refactor_bit_identical"] = bool(same)
    rep["pooled_refactor_max_abs_diff"] = float((pooled_ref.float() - pooled_fast.float()).abs().max())
    rep["pooled_shape"] = list(pooled_ref.shape)
    rep["t5_context_shape"] = list(pe.shape)
    print(f"pooled refactor bit-identical: {same}  maxdiff={rep['pooled_refactor_max_abs_diff']:.3e}")
    print(f"t5 context shape {list(pe.shape)}  pooled shape {list(pooled_ref.shape)}")
    assert same, "T5-skip refactor is NOT bit-identical; do not use it"

    # how many valid T5 tokens does a typical benchmark prompt have? (sizes the context corpus)
    toks = pipe.tokenizer_2(prompts[:8], padding="max_length", max_length=512,
                            truncation=True, return_tensors="pt")
    rep["valid_t5_tokens_per_prompt_sample"] = toks["attention_mask"].sum(1).tolist()

    # ---- 4. timed generation at pinned args
    cases = json.load(open("/data/bijaypandey/cuig_pilot/block_object_sprint/cases.json"))
    ga = cases["generation_arguments_pinned"]
    c = cases["cases"]["Bears"]["target"][0]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(a.n_gen):
        g = torch.Generator(device="cpu").manual_seed(c["init_seed"])
        with torch.no_grad():
            img = pipe(c["prompt"], height=ga["height"], width=ga["width"],
                       num_inference_steps=ga["num_inference_steps"],
                       guidance_scale=ga["guidance_scale"],
                       max_sequence_length=ga["max_sequence_length"],
                       generator=g).images[0]
    dt = time.time() - t0
    img.save(f"{OUT}/smoke_{c['filename']}")
    rep["gen_seconds_per_image"] = round(dt / max(1, a.n_gen), 2)
    rep["peak_mem_gen_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rep["smoke_images_decoded"] = a.n_gen
    rep["gen_args"] = ga
    print(f"generation: {rep['gen_seconds_per_image']}s/image  peak {rep['peak_mem_gen_GB']}GB")
    print(f"forecast: 1200 core images = {1200*rep['gen_seconds_per_image']/3600:.2f} GPU-h on 1 device")

    json.dump(rep, open(f"{OUT}/smoke_report.json", "w"), indent=1)
    print(f"\nwrote {OUT}/smoke_report.json")


if __name__ == "__main__":
    main()
