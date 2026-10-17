#!/usr/bin/env python3
"""Native-EraseAnything trainer for the efficiency pilot, with instrumentation.

Arm A calls the AUTHOR's loss functions (utils.calc_loss.calculate_upper_loss /
calculate_lower_loss) unchanged. Only the training loop, device placement, RNG control and
timing are ours; the objective, update rule, LoRA targets, optimizer, scheduler and step
counts come from the author's config.

PLACEMENT (the paper used one 80GB A100; we have 4x40GB, which is not one pool):
    device0 : FluxTransformer2DModel (bf16, ~23.8GB) + VAE (~0.17GB)
    device1 : CLIP text encoder (~0.25GB) + T5-XXL text encoder (~9.5GB)
Text embeddings are computed on device1 and moved to device0, exactly as the author's
`compute_text_embeddings` does (train_flux_lora.py:299-301). `torch.cuda.set_device(device0)`
is set so the author's bare `.cuda()` in calc_loss resolves to the transformer's device.

ATTENTION SETTINGS (see patches/apply_ea_patch.py):
    --attn author   : explicit materialised softmax in EVERY dual-stream block (author-faithful)
    --attn shared   : explicit softmax ONLY in the block whose maps are consumed (last
                      dual-stream block); all other blocks use fused SDPA. The consumed
                      attention matrix is bit-identical; the model sample agrees only within
                      floating-point tolerance. This is SHARED ENGINEERING, applied to all arms.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import yaml

EA = pathlib.Path(__file__).resolve().parent.parent / "source" / "ea"
sys.path.insert(0, str(EA))
os.chdir(EA)  # author modules use relative config/image paths

from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel  # noqa: E402
from diffusers import AutoencoderKL, FluxTransformer2DModel  # noqa: E402
from diffusers.models.attention_processor import EA_STATE  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from lora_dataset import LoraDataset, collate_data_fn  # noqa: E402
from tools.prompt_process import encode_prompt  # noqa: E402
from tools.scheduler_process import CustomFlowMatchEulerDiscreteScheduler  # noqa: E402
import utils.calc_loss as CL  # noqa: E402


# --------------------------------------------------------------- instrumentation
class Prof:
    """Low-overhead phase timer. Synchronises only at phase boundaries."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.t = {}
        self.n = {}
        self.fwd = 0          # transformer forward calls
        self.peak_bytes = 0

    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.t[name] = self.t.get(name, 0.0) + dt
            self.n[name] = self.n.get(name, 0) + 1

    def snapshot_peak(self):
        for d in range(torch.cuda.device_count()):
            try:
                self.peak_bytes = max(self.peak_bytes, torch.cuda.max_memory_allocated(d))
            except Exception:
                pass

    def as_dict(self):
        return {"seconds_by_phase": self.t, "calls_by_phase": self.n,
                "transformer_forwards": self.fwd,
                "peak_mem_gb": round(self.peak_bytes / 1024**3, 3)}


def count_forwards(transformer, prof):
    """Count transformer forward calls without altering numerics."""
    orig = transformer.forward

    def wrapped(*a, **k):
        prof.fwd += 1
        return orig(*a, **k)

    transformer.forward = wrapped
    return orig


# --------------------------------------------------------------- frozen LLM concepts
def frozen_irrelevant_concepts(key_word: str, K: int, frozen_dir: pathlib.Path,
                               ledger_note: str) -> list[str]:
    """The native method needs K 'irrelevant concepts' from an LLM
    (train_flux_lora.py:373 -> tools.ir_concept.MoE). One call PER CONCEPT is made and
    frozen to disk; every later run of the same concept reuses the frozen response, so the
    20-request auxiliary-LLM cap covers all arms and seeds.
    """
    frozen_dir.mkdir(parents=True, exist_ok=True)
    safe = key_word.replace(" ", "_").replace("/", "_")
    f = frozen_dir / f"irrelevant_{safe}.json"
    if f.exists():
        d = json.loads(f.read_text())
        print(f"[llm] reusing frozen response for {key_word!r}: {d['concepts']}")
        return d["concepts"]

    from tools.ir_concept import UniversalModelCaller, MoE
    key = os.environ.get("OPENAI_API_KEY")
    assert key, "OPENAI_API_KEY not in environment"
    # EA_PILOT_FIX: the author hardcodes an Azure deployment we do not have
    # (train_flux_lora.py:73-82, endpoint research-01-02.openai.azure.com). We use the
    # standard OpenAI endpoint with the same model name and the same MoE prompt.
    api_keys = {"gpt": {"azure": False, "api_key": key, "end_point": None,
                        "api_version": None}, "claude": None, "kimi": None, "qwen": None}
    caller = UniversalModelCaller(api_keys)
    t0 = time.time()
    concepts = MoE(caller, key_word, K, model_list=["gpt"])
    d = {"key_word": key_word, "K": K, "concepts": list(concepts),
         "provider": "openai", "model": "gpt-4o", "requests": 1,
         "elapsed_s": round(time.time() - t0, 2), "note": ledger_note,
         "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    f.write_text(json.dumps(d, indent=1))
    print(f"[llm] froze response for {key_word!r}: {d['concepts']}")
    return list(concepts)


# --------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept", required=True, help="key_word / artist to erase")
    ap.add_argument("--instance-prompt", required=True)
    ap.add_argument("--prompt-b", required=True, help="neutral prompt (author prompt_b)")
    ap.add_argument("--instance-data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="A", choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--attn", default="shared", choices=["author", "shared"])
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="if >0, stop after N steps (profiling only)")
    ap.add_argument("--frozen-dir", default=str(pathlib.Path(__file__).resolve().parent.parent / "frozen"))
    # arm C / B knobs are declared here but their values are frozen by the run manifest
    ap.add_argument("--early-stop-infonce", action="store_true",
                    help="ordinary control: stop each InfoNCE trajectory at the consumed index")
    ap.add_argument("--algo", default="none",
                    help="arm C mechanism id (set by the frozen manifest)")
    ap.add_argument("--algo-param", type=float, default=0.0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    for k, v in cfg.items():
        setattr(a, k, v) if not hasattr(a, k) or getattr(a, k) is None else None
    # explicit overrides that must win over the yaml
    args = argparse.Namespace(**cfg)
    args.instance_prompt = a.instance_prompt
    args.key_word = a.concept
    args.prompt_b = a.prompt_b
    args.instance_data_dir = a.instance_data_dir
    args.output_dir = a.out
    args.max_train_steps = a.steps or cfg["max_train_steps"]
    args.cache_latents = False

    devs = [int(d) for d in a.devices.split(",")]
    d0 = torch.device(f"cuda:{devs[0]}")
    d1 = torch.device(f"cuda:{devs[1] if len(devs) > 1 else devs[0]}")
    torch.cuda.set_device(d0)

    # reproducible: training seed controls LoRA init, data order, noise, timestep draws
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    EA_STATE["explicit_attn"] = True
    EA_STATE["maps_only_last"] = (a.attn == "shared")

    prof = Prof()
    meta = {"arm": a.arm, "concept": a.concept, "seed": a.seed, "attn": a.attn,
            "algo": a.algo, "algo_param": a.algo_param,
            "early_stop_infonce": bool(a.early_stop_infonce),
            "devices": devs, "steps": args.max_train_steps,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    model_id = cfg["pretrained_model_name_or_path"]
    rev = os.environ.get("FLUX_DEV_REVISION")
    assert rev, "FLUX_DEV_REVISION must pin the model revision"
    wd = torch.bfloat16 if cfg["mixed_precision"] == "bf16" else torch.float32

    with prof("prep_model_load"):
        tok1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=rev)
        tok2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2", revision=rev)
        te1 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", revision=rev,
                                            torch_dtype=wd).to(d1).eval()
        te2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", revision=rev,
                                             torch_dtype=wd).to(d1).eval()
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=rev,
                                            torch_dtype=wd).to(d0).eval()
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", revision=rev, torch_dtype=wd).to(d0)
        sched = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler",
                                                                      revision=rev)
        import copy as _copy
        sched_copy = _copy.deepcopy(sched)
        for m in (transformer, vae, te1, te2):
            m.requires_grad_(False)

    lora_cfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"], init_lora_weights="gaussian",
                          target_modules=["attn.add_k_proj", "attn.add_q_proj"])
    transformer.add_adapter(lora_cfg)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    meta["lora_tensors"] = len(lora_params)
    meta["lora_param_count"] = int(sum(p.numel() for p in lora_params))
    # fidelity checks: backbone frozen, adapter trainable, finite init
    assert meta["lora_param_count"] > 0
    assert all(torch.isfinite(p).all() for p in lora_params)
    n_frozen = sum(1 for p in transformer.parameters() if not p.requires_grad)
    meta["frozen_tensors"] = n_frozen
    print(f"[fidelity] trainable LoRA tensors {len(lora_params)} "
          f"({meta['lora_param_count']} params); frozen backbone tensors {n_frozen}")

    opt = torch.optim.AdamW([{"params": lora_params, "lr": float(cfg["learning_rate"])}],
                            betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
                            weight_decay=float(cfg["adam_weight_decay"]),
                            eps=float(cfg["adam_epsilon"]))
    criteria = torch.nn.MSELoss()

    with prof("prep_dataset"):
        ds = LoraDataset(instance_data_root=args.instance_data_dir,
                         instance_prompt=args.instance_prompt, key_word=args.key_word,
                         tokenizer_t5=tok2, size=cfg["resolution"], repeats=cfg["repeats"],
                         center_crop=cfg["center_crop"])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=cfg["train_batch_size"], shuffle=True,
            collate_fn=lambda ex: collate_data_fn(ex, cfg["with_prior_preservation"]),
            num_workers=0)

    lr_sched = get_scheduler(cfg["lr_scheduler"], optimizer=opt,
                             num_warmup_steps=cfg["lr_warmup_steps"],
                             num_training_steps=args.max_train_steps,
                             num_cycles=cfg["lr_num_cycles"], power=cfg["lr_power"])

    tokenizers = [tok1, tok2]
    text_encoders = [te1, te2]

    def compute_text_embeddings(prompt, _te, _tk):
        with prof("text_encode"):
            with torch.no_grad():
                pe, ppe, tid = encode_prompt(_te, _tk, prompt, cfg["max_sequence_length"])
            return pe.to(d0), ppe.to(d0), tid.to(d0)

    with prof("prep_llm_irrelevant"):
        ir = frozen_irrelevant_concepts(args.key_word, 3, pathlib.Path(a.frozen_dir),
                                        f"native MoE call, arm-independent, concept={args.key_word}")
    assert len(ir) == 3, f"MoE returned {len(ir)}"

    count_forwards(transformer, prof)

    # arm B/C hooks live in ea_arms; arm A uses the author path untouched.
    # NOTE: install() REBINDS CL.calculate_lower_loss, so the loop must resolve it at call
    # time -- binding it to a local before install() silently kept the author version
    # (observed: identical 435 forwards and no saving).
    if a.arm in ("B", "C") or a.early_stop_infonce or a.algo != "none":
        import ea_arms
        ea_arms.install(CL, prof, arm=a.arm, early_stop_infonce=a.early_stop_infonce,
                        algo=a.algo, algo_param=a.algo_param)
    CL_upper = CL.calculate_upper_loss

    transformer.train()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    losses = []
    gstep = 0
    t_enc_ddpm = None
    t_train0 = time.time()
    stop_at = a.profile_steps or args.max_train_steps

    while gstep < stop_at:
        for batch in dl:
            if gstep >= stop_at:
                break
            torch.manual_seed(a.seed * 100003 + gstep)  # per-step reproducible draws
            if gstep % 2 == 0 or float(cfg["lamb3"]) + float(cfg["lamb4"]) == 0.0:
                with prof("step_upper"):
                    up, t_enc_ddpm = CL_upper(
                        args, batch, compute_text_embeddings, text_encoders, tokenizers,
                        transformer, sched_copy, batch["prompts"], vae, criteria,
                        negative_guidance=cfg["negative_guidance"], weight_dtype=wd,
                        neg_prompts=str(args.prompt_b), start_guidance=3, ddim_steps=28,
                        lamb1=float(cfg["lamb1"]), lamb2=float(cfg["lamb2"]))
                    loss = float(cfg["lamb1"]) * up[0] + float(cfg["lamb2"]) * up[1]
                    rec = {"step": gstep, "kind": "upper",
                           "esd": float(up[0].detach()), "attn": float(up[1].detach())}
            else:
                with prof("step_lower"):
                    lo = CL.calculate_lower_loss(
                        args, batch, compute_text_embeddings, text_encoders, tokenizers,
                        transformer, sched_copy, batch["prompts"], vae, criteria,
                        negative_guidance=cfg["negative_guidance"], weight_dtype=wd,
                        t_enc_ddpm=t_enc_ddpm, start_guidance=3, ddim_steps=28, K=3,
                        ir_concept_lst=ir)
                    loss = float(cfg["lamb3"]) * lo[0] + float(cfg["lamb4"]) * lo[1]
                    rec = {"step": gstep, "kind": "lower",
                           "lora": float(lo[0].detach()), "infoNCE": float(lo[1].detach())}
            rec["loss"] = float(loss.detach())
            if not math.isfinite(rec["loss"]):
                raise SystemExit(f"NON-FINITE LOSS at step {gstep}: {rec}")
            with prof("backward"):
                loss.backward()
            with prof("optimizer"):
                gn = torch.nn.utils.clip_grad_norm_(lora_params, 1e9).item()  # measure only
                opt.step(); opt.zero_grad(set_to_none=True); lr_sched.step()
            rec["grad_norm"] = gn
            losses.append(rec)
            prof.snapshot_peak()
            gstep += 1
            if gstep % 10 == 0 or gstep == 1:
                print(f"[{a.arm}|{a.concept}|s{a.seed}] step {gstep}/{stop_at} "
                      f"{rec['kind']} loss {rec['loss']:.4f} gn {gn:.3e} "
                      f"fwd {prof.fwd} peak {prof.peak_bytes/1024**3:.1f}GB", flush=True)

    meta["train_wall_s"] = round(time.time() - t_train0, 3)
    with prof("io_save_adapter"):
        sd = get_peft_model_state_dict(transformer)
        sd = {f"transformer.{k}": v.to(torch.float32).contiguous() for k, v in sd.items()}
        save_file(sd, os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"))

    meta.update(prof.as_dict())
    meta["n_devices_occupied"] = len(set(devs))
    meta["gpu_device_seconds"] = round(meta["train_wall_s"] * len(set(devs)), 3)
    meta["loss_trace"] = losses
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pathlib.Path(os.path.join(args.output_dir, "run_meta.json")).write_text(
        json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "loss_trace"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
