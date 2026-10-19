# Small fixes applied during bring-up (preserved)

All are path/API/memory/instrumentation fixes. **No algorithmic change.** None
consumed the permitted implementation-repair cycle or the GPU retry.

| # | Symptom | Fix | Numerically inert? |
|---|---|---|---|
| P1 | T5 tokenizer: `Cannot instantiate this tokenizer from a slow version` | installed `sentencepiece==0.2.0`, `protobuf==4.25.3` into the **pilot-owned** `pydeps/` overlay; shared venv untouched. Cause: the FLUX tokenizer_2 config sets `add_prefix_space`, which forces a slow→fast conversion even though `tokenizer.json` exists | yes — enables the stock conversion path |
| P2 | `AttributeError: 'str' object has no attribute 'type'` in `randn_tensor` | store `self.device` as a `torch.device`, not a string | yes |
| P3 | CUDA OOM at 39.35/39.38 GiB in the student forward | call `transformer.train()` (as native GEM's `wrapper.train()` does), which is what actually activates diffusers' gradient checkpointing; and offload **T5-XXL only** after the native pre-encode | yes — recompute + placement only; same arithmetic, same RNG draws |
| P4 | `Cannot generate a cpu tensor from a generator of type cuda` | revert the VAE/CLIP offload: diffusers derives the pipeline execution device from the first registered `nn.Module`, so those must stay resident. Only T5-XXL is offloaded | yes |
| P5 | RoPE shape error 1536 vs 768 in the bring-up fixed-input probe | build the probe from `pipe.prepare_latents` so it has the TRUE packed length (1024), not GEM's `calculate_image_seq_len` value (256, which only ever feeds the scheduler shift) | yes — probe construction only; the training path was already correct |
| P6 | `fixed_input_forward_bit_identical: false` (max diff 0.07) | run the probe in **eval mode**: LoRA dropout p=0.1 is stochastic in train mode, so the comparison was meaningless. After the fix: bit-identical, max diff **0.0**, with the perturbation control confirming test power | yes — test correctness only |
| P7 | repeated bring-up attempts re-decoded the same M0 image | skip decoding when the declared M0 PNG already exists, protecting the image ceiling | yes |
