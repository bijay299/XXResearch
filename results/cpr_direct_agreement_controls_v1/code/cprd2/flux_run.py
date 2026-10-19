"""FLUX + native PEFT helpers: build, adapter I/O, deterministic generation. Imported only by job scripts."""
import hashlib
import json
import os
import time

import numpy as np
import torch

from cprd2 import protocol as P
from cprd2.common import ROOT, sha256_file, utc
from cprd2.noise import noise_hash, packed_noise

REVISIONS = {"gem": "c328fc6bec687efe60223498d1b6bed2e5fbcffd",
             "flux": "black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"}


def build(device, init_seed, grad_ckpt=True):
    """Fresh pristine pipeline -> native wrapper -> native PEFT qk_dual LoRA (seeded init)."""
    from gem.configs import AdaptationConfig
    from gem.configs.pipeline_config import PipelineConfig
    from gem.utils.seed import seed_everything
    from gem.wrappers.factory import PipelineFactory
    pc = PipelineConfig(_allow_init=True, model_type="flux", model_name_or_path=os.environ["CPRD_BASE_MODEL_PATH__FLUX"])
    t = time.time()
    pipe = PipelineFactory.create_pipeline(pc, model_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    wrapper = PipelineFactory.wrap_pipeline(pipe)
    seed_everything(int(init_seed))                 # deterministic LoRA-A init per trajectory seed
    wrapper = PipelineFactory.prepare_adaptation(
        AdaptationConfig(adapter_mode="lora", subset_name=P.SETTINGS["adapter"]["subset_name"]), wrapper)
    if grad_ckpt:
        pipe.transformer.enable_gradient_checkpointing()
    return pipe, wrapper, round(time.time() - t, 1)


def lora_params(wrapper):
    return {n: p for n, p in wrapper.named_parameters() if "lora_" in n and p.requires_grad}


def peft_state(wrapper):
    from peft import get_peft_model_state_dict
    return {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in get_peft_model_state_dict(wrapper).items()}


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


def inventory(wrapper):
    names = list(lora_params(wrapper))
    mods = sorted({n.split(".lora_")[0] for n in names})
    base_trainable = [n for n, p in wrapper.named_parameters() if p.requires_grad and "lora_" not in n]
    dropouts = sorted({float(m.p) for n, m in wrapper.named_modules()
                       if "lora_dropout" in n and hasattr(m, "p")})
    return {"n_lora_tensors": len(names), "n_lora_modules": len(mods), "n_base_trainable": len(base_trainable),
            "trainable_params": int(sum(p.numel() for p in wrapper.parameters() if p.requires_grad)),
            "lora_dropout_p": dropouts, "lora_dtypes": sorted({str(p.dtype) for p in lora_params(wrapper).values()}),
            "first_module": mods[0] if mods else None}


@torch.no_grad()
def base_digest(wrapper):
    """Cheap sensitive digest of every frozen (non-LoRA) parameter: float64 sum, sum of squares, abs-max."""
    h = hashlib.sha256()
    n = 0
    for name, p in wrapper.named_parameters():
        if "lora_" in name:
            continue
        x = p.detach().double()
        h.update(name.encode())
        h.update(np.array([float(x.sum()), float((x * x).sum()), float(x.abs().max())], dtype=np.float64).tobytes())
        n += 1
    return {"sha256": h.hexdigest(), "n_params": n}


def rng_digest():
    d = {"python": hashlib.sha256(repr(__import__("random").getstate()).encode()).hexdigest()[:16],
         "numpy": hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest()[:16],
         "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        d["torch_cuda"] = hashlib.sha256(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())).hexdigest()[:16]
    return d


def save_adapter(wrapper, out_dir, meta):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = peft_state(wrapper)
    path = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(sd, path, metadata={"project": "cpr_direct_agreement_controls_v1"})
    cfg = wrapper.peft_config["default"].to_dict()
    cfg = {k: (sorted(v) if isinstance(v, set) else v) for k, v in cfg.items()}
    meta = dict(meta, saved_utc=utc(), n_tensors=len(sd), tensor_state_sha256=state_hash(sd),
                file_sha256=sha256_file(path), peft_config=cfg, gem_commit=REVISIONS["gem"],
                flux=REVISIONS["flux"], reconstruct=(
                    "PipelineFactory.create_pipeline(flux, bf16) -> wrap_pipeline -> prepare_adaptation(lora qk_dual r16 a16 "
                    "do0.1) -> peft.set_peft_model_state_dict(wrapper, safetensors.load_file(adapter_model.safetensors))"))
    with open(os.path.join(out_dir, "adapter_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    return meta


@torch.no_grad()
def zero_adapter(wrapper):
    for p in lora_params(wrapper).values():
        p.zero_()


@torch.no_grad()
def load_adapter(wrapper, adapter_dir):
    """Zero first, load, then verify the restored state equals the file bit for bit."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    zero_adapter(wrapper)
    zeroed = state_hash(peft_state(wrapper))
    res = set_peft_model_state_dict(wrapper, sd)
    got = state_hash(peft_state(wrapper))
    want = state_hash({k: v.to(torch.float32).cpu() for k, v in sd.items()})
    meta = json.load(open(os.path.join(adapter_dir, "adapter_meta.json")))
    ok = (got == want == meta["tensor_state_sha256"]) and got != zeroed
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if not ok or unexpected:
        raise RuntimeError(f"adapter restoration failed: got={got[:12]} want={want[:12]} unexpected={unexpected[:3]}")
    return got


class DecodeCounter:
    def __init__(self, vae):
        self.n = 0
        self._orig = vae.decode
        vae.decode = self._wrap
        self._vae = vae

    def _wrap(self, *a, **k):
        self.n += 1
        return self._orig(*a, **k)

    def restore(self):
        self._vae.decode = self._orig


def fresh_scheduler(pipe):
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)


@torch.no_grad()
def generate_case(pipe, wrapper, case, adapter_on, device, out_path):
    """One 512x512, 28-step, guidance-3.5 image from the frozen shared noise. Returns metadata."""
    ev = P.SETTINGS["evaluation"]
    nz = packed_noise(case["eval_seed"])
    nh = noise_hash(case["eval_seed"])
    if nh != case["noise_sha256"]:
        raise RuntimeError(f"noise hash mismatch for {case['case_id']}")
    fresh_scheduler(pipe)
    wrapper.eval()
    t = time.time()
    if adapter_on:
        out = pipe(prompt=case["prompt"], height=ev["height"], width=ev["width"],
                   num_inference_steps=ev["num_inference_steps"], guidance_scale=ev["guidance_scale"],
                   latents=nz.to(device=device, dtype=torch.bfloat16), max_sequence_length=ev["max_sequence_length"],
                   output_type="pil")
    else:
        with wrapper.disable_adapter():
            out = pipe(prompt=case["prompt"], height=ev["height"], width=ev["width"],
                       num_inference_steps=ev["num_inference_steps"], guidance_scale=ev["guidance_scale"],
                       latents=nz.to(device=device, dtype=torch.bfloat16),
                       max_sequence_length=ev["max_sequence_length"], output_type="pil")
    img = out.images[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG")
    return {"case_id": case["case_id"], "path": os.path.relpath(out_path, ROOT), "seconds": round(time.time() - t, 2),
            "png_sha256": sha256_file(out_path), "size": list(img.size), "adapter_on": bool(adapter_on)}


def log_decodes(job_id, kind, n):
    d = os.path.join(ROOT, "artifacts", "decode_counts")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{job_id}.jsonl"), "a") as f:
        f.write(json.dumps({"kind": kind, "n": int(n), "utc": utc()}) + "\n")
