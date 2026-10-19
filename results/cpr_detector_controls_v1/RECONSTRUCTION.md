# Reconstruction — CPR detector-defined controls screen v1

**Status: `ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`.**
Production never started. **There is no selected adapter and there are no
generated images**, because no development selection stage was reached and
**zero images were generated**.

| Asset | What it is |
|---|---|
| `entry_gate_peft_adapter_init.pt` | the **native PEFT** `qk_dual` adapter state captured by the CPU bring-up gate (76 modules, 152 tensors, FP32 on a BF16 backbone) at its native initialisation — LoRA A non-zero, B zero. An entry-gate artifact for verifying adapter-state restoration. **Not a trained erasure model.** |
| `qualification_raw_detections.tar.gz` | both detectors' raw per-image boxes, scores and per-class gate counts over the 280 qualification image/class pairs |
| `allocation_ledger.tar.gz` | the durable allocation ledger and its append-only journal |
| `frozen_configs.tar.gz` | the frozen case matrix and the category/qualification selection |
| `SHA256SUMS` | hashes of every asset above |

## Restoring the adapter state

```python
import torch
from gem.configs.pipeline_config import PipelineConfig
from gem.configs import AdaptationConfig
from gem.wrappers.factory import PipelineFactory

pc = PipelineConfig(_allow_init=True, model_type="flux",
                    model_name_or_path="<pinned FLUX.1-dev 3de623fc snapshot dir>")
pipe = PipelineFactory.create_pipeline(pc, model_dtype=torch.bfloat16)
w = PipelineFactory.wrap_pipeline(pipe)
w = PipelineFactory.prepare_adaptation(
        AdaptationConfig(adapter_mode="lora", subset_name="qk_dual"), w)

blob = torch.load("entry_gate_peft_adapter_init.pt", map_location="cpu")
with torch.no_grad():
    for n, p in w.named_parameters():
        if n in blob["adapter"]:
            p.copy_(blob["adapter"][n].to(p.device, p.dtype))
```

Effective-weight restoration is what is verified. **Merged/unmerged BF16 bit
equality is NOT assumed or claimed.**

## Licensing and redistribution

Backbone `black-forest-labs/FLUX.1-dev@3de623fc`, FLUX.1 [dev]
**Non-Commercial License** — **no base weights are redistributed**; the only
weight-like asset here is a ~28 MiB LoRA adapter at its initialisation.
Detectors `PekingU/rtdetr_r50vd@df939e66` and
`facebook/detr-resnet-50@70120ba8` (both Apache-2.0) are **not** redistributed.
For COCO, **image ids, annotation provenance and reproduction instructions are
published instead of the photographs themselves**
(`instances_val2017.json` sha256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`).
GEM (`c328fc6`) is third-party work by its authors; only attributed diffs are
published, never vendored source. No credentials are published.
