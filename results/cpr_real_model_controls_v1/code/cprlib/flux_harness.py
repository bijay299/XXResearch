"""FLUX.1-dev harness reproducing the pinned native GEM training procedure.

Source-derived from `multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd`
(`gem/operators/erasure/erasure_gem.py`, `gem/wrappers/pipeline_wrapper_flux.py`,
`gem/wrappers/base_pipeline_wrapper.py`), re-implemented here. No GEM module is
imported. This is a SOURCE-DERIVED OBJECT EXPERIMENT, not a reproduction of the
authors' benchmark numbers.

Resolved native values for FLUX (registry has only `gem` and `gem:sd3`; FLUX
uses the base `GEMConfig`):

    learning_rate                       1e-3
    gamma                               100.0
    max_step_budget                     250        (processed timesteps)
    partial_denoising_end_at_timestep   10         -> 11 latents per trajectory
    partial_denoising_mode              "student"
    partial_denoising_guidance_scale    3.0
    use_inference_timesteps             True
    partial_denoising_inference_timesteps  None -> pipe num_inference_steps = 28
    predict_noise guidance_scale        1.0        (GEM calls it without the arg)
    anchors                             [EMPTY_CONCEPT] where EMPTY_CONCEPT == ""
    optimizer                           AdamW(lr=learning_rate)
    max_parallel_timesteps              1          (frozen by this contract)

Contract-frozen deviations from MODEL_CONFIGS, recorded deliberately:
  * backbone dtype BF16 (MODEL_CONFIGS["flux"]["model_dtype"] is "float16");
    the loss is computed in FP32.
  * eta = 1 (GEMConfig default is also 1.0).
  * 512x512 generation, 28 inference steps, guidance 3.5 for evaluation images.
"""

import contextlib
import os

import numpy as np
import torch

from cprlib import lora as LORA
from cprlib import losses as LS

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
EMPTY_CONCEPT = ""                      # gem/operators/utils.py:3

# ---- frozen native / contract constants ----
NATIVE_LR = 1e-3
GAMMA = 100.0
ETA = 1.0
MAX_STEP_BUDGET = 250                   # processed timesteps, NOT optimizer updates
PARTIAL_END_AT = 10                     # -> 11 latents (callback fires after steps 0..10)
PARTIAL_GUIDANCE = 3.0
PARTIAL_MODE = "student"
TRAIN_PREDICT_GUIDANCE = 1.0            # predict_noise default in the GEM source
INTERNAL_STEPS = 28                     # pipe_config.num_inference_steps fallback
MAX_PARALLEL_TIMESTEPS = 1
IMAGE_SIZE = 512
EVAL_STEPS = 28
EVAL_GUIDANCE = 3.5


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class FluxHarness:
    def __init__(self, device="cuda:0", dtype=torch.bfloat16, gradient_checkpointing=True):
        from diffusers import FluxPipeline
        # diffusers' randn_tensor compares `generator.device.type` against
        # `device.type`, so this must be a torch.device, not a string.
        self.device = torch.device(device)
        self.dtype = dtype
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.transformer.requires_grad_(False)
        if gradient_checkpointing:
            # numerically inert (recompute); purely a memory/compute trade
            self.transformer.enable_gradient_checkpointing()
        self.wrapped = None

    # ---------------- memory placement (numerically inert) ----------------
    # The backbone, T5-XXL, CLIP and the VAE together exceed a 40 GB card once
    # training activations are added. GEM pre-encodes every embedding once, so
    # the text encoders are not needed during training, and trajectory
    # generation returns latents (no VAE decode). Moving those modules off the
    # device changes no arithmetic, no RNG draw and no result -- only placement.
    def offload_text_encoders(self):
        # Only T5-XXL (~9.5 GB) is offloaded. The VAE and the small CLIP encoder
        # stay resident because diffusers derives the pipeline's execution
        # device from the first registered nn.Module; moving them to CPU would
        # make the pipeline try to sample latents on CPU from a CUDA generator.
        self.pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    def set_train_mode(self):
        """Native GEM calls `wrapper.train()` each iteration. This also makes
        diffusers' gradient checkpointing take effect (it is gated on
        `self.training`) and enables the LoRA dropout the native config sets."""
        self.transformer.train()
        torch.cuda.empty_cache()

    def set_eval_mode(self):
        self.transformer.eval()
        torch.cuda.empty_cache()

    # ---------------- adaptation ----------------
    def attach_lora(self, seed):
        self.wrapped = LORA.inject(self.transformer, seed=seed)
        return self.wrapped

    def disable_adapter(self):
        if self.wrapped is None:
            return contextlib.nullcontext()
        return LORA.disable_adapter(self.wrapped)

    # ---------------- text ----------------
    @torch.no_grad()
    def encode_prompt(self, prompt):
        """Returns (prompt_embeds, pooled_prompt_embeds, text_ids), detached."""
        pe, ppe, tid = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=self.device, num_images_per_prompt=1)
        return (pe.detach(), ppe.detach(), tid.detach())

    # ---------------- scheduler ----------------
    def image_seq_len(self, size=IMAGE_SIZE):
        return (size // self.vae_scale_factor // 2) * (size // self.vae_scale_factor // 2)

    def set_timesteps(self, image_seq_len, num_timesteps):
        assert 64 <= image_seq_len <= 16384, f"bad image_seq_len {image_seq_len}"
        sigmas = np.linspace(1.0, 1 / num_timesteps, num_timesteps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15))
        self.scheduler.set_timesteps(num_timesteps, sigmas=sigmas, mu=mu, device=self.device)
        self.pipe._num_timesteps = len(self.scheduler.timesteps)

    # ---------------- core forward ----------------
    def predict_noise(self, timestep_idx, latents, embeds, guidance_scale=TRAIN_PREDICT_GUIDANCE,
                      size=IMAGE_SIZE):
        """Faithful to pipeline_wrapper_flux.predict_noise."""
        prompt_embeds, pooled, text_ids = embeds
        batch_size = len(latents)
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipe.prepare_latents(
            batch_size, num_channels_latents, size, size,
            latents.dtype, self.device, None, latents)
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=self.device)
            guidance = guidance.expand(latents.shape[0]).to(latents.dtype)
        else:
            guidance = None
        timestep = self.scheduler.timesteps[timestep_idx].expand(
            latents.shape[0]).to(latents.dtype) / 1000
        return self.transformer(
            hidden_states=latents,
            timestep=timestep.to(self.device),
            guidance=guidance,
            pooled_projections=pooled.to(latents.dtype).expand(latents.shape[0], -1),
            encoder_hidden_states=prompt_embeds.to(latents.dtype).expand(
                latents.shape[0], -1, -1),
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False)[0]

    # ---------------- trajectory ----------------
    def partial_trajectory(self, embeds, generator, end_at=PARTIAL_END_AT,
                           guidance=PARTIAL_GUIDANCE, size=IMAGE_SIZE,
                           num_inference_steps=INTERNAL_STEPS):
        """Collect every latent produced up to `end_at`, inclusive.

        The native callback appends AFTER each step and stops once
        `step >= end_at`, so steps 0..end_at inclusive are collected:
        `n_latents == end_at + 1 == 11`. No image is decoded (the native
        caller discards it, and automatic training previews are disabled).
        """
        isl = self.image_seq_len(size)
        self.set_timesteps(isl, num_inference_steps)
        latents_per_step = []

        def cb(pipe, step, timestep, kw):
            with torch.no_grad():
                latents_per_step.append(kw["latents"].clone().detach())
            if step >= end_at:
                raise StopIteration
            return {}

        pe, ppe, _ = embeds
        with torch.no_grad():
            try:
                self.pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe,
                          height=size, width=size,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance, generator=generator,
                          callback_on_step_end=cb, output_type="latent")
            except StopIteration:
                pass
        for t in latents_per_step:
            if not torch.isfinite(t).all():
                raise RuntimeError("partially denoised latents are not finite")
        return latents_per_step

    # ---------------- training ----------------
    def train_arm(self, arm, target_prompt, lr, train_seed, checkpoint_at=(),
                  on_checkpoint=None, log=None, max_iters=None):
        """One trajectory for one arm, reproducing the native GEM loop.

        `checkpoint_at` is a set of PROCESSED-TIMESTEP counts at which the
        first completed optimizer update at or beyond that count triggers
        `on_checkpoint`. The budget is never exceeded to manufacture one.
        """
        assert self.wrapped is not None, "attach_lora() first"
        params = LORA.trainable_parameters(self.wrapped)
        optimizer = torch.optim.AdamW(params, lr=lr)

        neutral = self.encode_prompt(EMPTY_CONCEPT)          # native pre-encode
        target_embeds = self.encode_prompt(target_prompt)
        anchor_embeds = self.encode_prompt(EMPTY_CONCEPT)    # anchors == [EMPTY_CONCEPT]
        del neutral
        # every embedding this loop needs is now pre-encoded (as in native GEM)
        self.offload_text_encoders()
        self.set_train_mode()

        max_timestep = PARTIAL_END_AT
        expected_n_iterations = 1 + MAX_STEP_BUDGET // max_timestep
        step_budget = MAX_STEP_BUDGET
        train_generator = torch.Generator(device=self.device).manual_seed(train_seed)

        processed = 0
        updates = 0
        history = []
        pending = sorted(checkpoint_at)

        for global_step in range(expected_n_iterations):
            # --- partial denoising with the STUDENT (adapter enabled) ---
            ctx = self.disable_adapter() if PARTIAL_MODE == "teacher" else contextlib.nullcontext()
            self.transformer.eval()          # sampler runs without dropout
            with ctx:
                traj = self.partial_trajectory(target_embeds, train_generator)
            self.transformer.train()         # native wrapper.train() per iteration
            n_latents = len(traj)

            optimizer.zero_grad()
            traj_loss = 0.0
            arm_failed = None

            for batch_start in range(0, n_latents, MAX_PARALLEL_TIMESTEPS):
                batch_end = min(batch_start + MAX_PARALLEL_TIMESTEPS, n_latents)
                stacked = torch.stack(traj[batch_start:batch_end], dim=1).squeeze(0)
                self.set_timesteps(stacked.shape[1], INTERNAL_STEPS)
                idx = torch.arange(batch_start, batch_start + stacked.shape[0])

                with torch.no_grad(), self.disable_adapter():
                    anchor_score = self.predict_noise(idx, stacked, anchor_embeds)
                predicted_target_score = self.predict_noise(idx, stacked, target_embeds)
                with torch.no_grad(), self.disable_adapter():
                    target_score = self.predict_noise(idx, stacked, target_embeds)

                try:
                    loss = LS.arm_loss(arm, predicted_target_score, anchor_score,
                                       target_score, eta=ETA, gamma=GAMMA,
                                       n_latents=n_latents)
                except LS.AxialDomainViolation as exc:
                    arm_failed = f"AXIAL_DOMAIN_VIOLATION at global_step {global_step}, " \
                                 f"batch {batch_start}: {exc}"
                    break

                loss.backward()
                step_budget -= stacked.shape[0]
                processed += stacked.shape[0]
                traj_loss += float(loss.detach())
                del anchor_score, predicted_target_score, target_score, stacked
                if step_budget <= 0:
                    break

            if arm_failed:
                return {"status": "ARM_FAILED", "reason": arm_failed,
                        "processed_timesteps": processed, "optimizer_updates": updates,
                        "history": history}

            for p in params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    return {"status": "ARM_FAILED",
                            "reason": f"nonfinite gradient at global_step {global_step}",
                            "processed_timesteps": processed,
                            "optimizer_updates": updates, "history": history}

            optimizer.step()
            updates += 1
            history.append({"global_step": global_step, "processed_timesteps": processed,
                            "optimizer_updates": updates, "traj_loss": traj_loss,
                            "n_latents": n_latents})
            if log:
                log(history[-1])

            while pending and processed >= pending[0]:
                mark = pending.pop(0)
                if on_checkpoint:
                    on_checkpoint(mark, processed, updates)

            if step_budget <= 0:
                break
            if max_iters is not None and (global_step + 1) >= max_iters:
                # timing probe only -- never used to produce a production endpoint
                return {"status": "PROFILE_TRUNCATED", "processed_timesteps": processed,
                        "optimizer_updates": updates, "history": history,
                        "remaining_budget": step_budget}

        return {"status": "OK", "processed_timesteps": processed,
                "optimizer_updates": updates, "history": history,
                "remaining_budget": step_budget}

    # ---------------- evaluation generation ----------------
    @torch.no_grad()
    def generate(self, prompt_embeds_tuple, seed, size=IMAGE_SIZE,
                 steps=EVAL_STEPS, guidance=EVAL_GUIDANCE):
        """One image from a fixed seed. Returns (PIL image, initial-noise hash)."""
        pe, ppe, _ = prompt_embeds_tuple
        gen = torch.Generator(device=self.device).manual_seed(seed)
        isl = self.image_seq_len(size)
        self.set_timesteps(isl, steps)
        out = self.pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe,
                        height=size, width=size, num_inference_steps=steps,
                        guidance_scale=guidance, generator=gen,
                        output_type="pil")
        return out.images[0]

    @torch.no_grad()
    def initial_noise_hash(self, seed, size=IMAGE_SIZE):
        """Hash of the initial latent noise for one evaluation seed.

        Proves pairing across conditions: matching seeds alone do not establish
        identical starting noise.
        """
        import hashlib
        num_channels_latents = self.transformer.config.in_channels // 4
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents, _ = self.pipe.prepare_latents(
            1, num_channels_latents, size, size, self.dtype, self.device, gen, None)
        arr = latents.detach().to(torch.float32).cpu().contiguous().numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()
