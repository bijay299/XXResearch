#!/usr/bin/env python3
"""Arm B (ordinary cost-saving control) and arm C (the one proposed algorithmic change).

Both act only on `calculate_lower_loss`, because that is where the measured cost is
(see profile/ -- the lower step runs K+1 = 4 full 28-step no-grad trajectories and consumes
ONE attention map from each).

-------------------------------------------------------------------------------
ARM B -- `early_stop_infonce`: ORDINARY, EXACT, BIT-IDENTICAL
-------------------------------------------------------------------------------
The author draws `attn_map_rand_idx = random.randint(0, ddim_steps-1)` (calc_loss.py:572)
and then uses only `attn_map_lst[attn_map_rand_idx]` from each of the K+1 trajectories
(calc_loss.py:595, 598). The attention map produced at sampler index j depends only on
z_0..z_j, so the steps after j cannot change it. Stopping each trajectory at j therefore
returns the SAME map, bit for bit, and saves (28-(j+1))/28 of that trajectory's forwards.

The only reordering is that the index must be drawn BEFORE sampling instead of between the
negative and positive trajectories. Nothing between those two points consumes Python's
`random` stream (the trajectories use torch RNG: randn_tensor and
CustomFlowMatchEulerDiscreteScheduler.set_train_timesteps), so the drawn value is the same
one the author would have drawn. This is pure removal of redundant computation: it is
reported as ENGINEERING, not as the proposed algorithm.

Expected saving on those trajectories: 1 - E[j+1]/28 = 1 - 14.5/28 = 48.2%.

-------------------------------------------------------------------------------
ARM C -- `shared_prefix`: THE ONE PROPOSED ALGORITHMIC CHANGE
-------------------------------------------------------------------------------
Native: each of the K+1 concepts c gets its own trajectory driven by its own text
embedding e_c from the shared start code, and its attention map is read at index j from
*its own* latent:

    z_j^{(c)} = (f_{j-1} o ... o f_0)(z_0 ; e_c),      A^{(c)} = Attn(z_j^{(c)}, e_c)

so the cost is (K+1)(j+1) transformer forwards and the K+1 maps are evaluated at K+1
DIFFERENT visual states.

Proposed: run ONE prefix of length j+1 driven by a single frozen "carrier" embedding e*,
then read each concept's attention map by ONE extra forward at the COMMON latent:

    z_j^*  = (f_{j-1} o ... o f_0)(z_0 ; e*),          A^{(c)} = Attn(z_j^*, e_c)

Cost: (j+1) + (K+1) forwards instead of (K+1)(j+1).

EXACT / APPROXIMATE / STOCHASTIC:
  * exact      : the extra per-concept forward itself (one true transformer forward each).
  * changed    : the EVALUATION POINT. A^{(c)} is now read at z_j^* instead of z_j^{(c)}.
                 This is an explicit change to the objective's sampling distribution and is
                 declared as such -- it is NOT a numerically equivalent refactor.
  * stochastic : z_0, the timestep schedule and j are drawn exactly as in the native method.

CLAIM (the reason to expect it to work, not a proof of effectiveness): the InfoNCE term
contrasts attention over the removed-token columns across concepts. Native evaluates those
columns at K+1 independently-drifted latents, so each logit mixes the text-conditioning
difference with between-trajectory drift. Sharing the prefix is a common-random-numbers
coupling: it removes the between-trajectory component from every pairwise comparison, so
the contrast isolates the conditioning difference. Coupling reduces the variance of a
difference of dependent terms whenever the removed component is positively correlated
across arms; it does not change what the contrast is trying to measure.

CONDITIONS UNDER WHICH IT SHOULD FAIL (stated in advance): if the attention map over the
removed-token columns depends strongly on the concept-specific latent rather than on the
text conditioning, then evaluating all concepts at a carrier latent collapses the contrast
(all A^{(c)} become near-identical) and the InfoNCE gradient vanishes or mis-signs. The
worked check in derivation_note.md gives the degenerate case explicitly.

Setting `algo_param` selects the carrier embedding e*:
    0 -> the anchor/current training prompt embedding (emb_p)
    1 -> the negative/synonym embedding (emb_neg)
Exactly two development settings of this one mechanism are permitted, and these are they.
"""
from __future__ import annotations

import random

import torch
from diffusers import FluxPipeline

_STATE = {}


def install(CL, prof, arm="A", early_stop_infonce=False, algo="none", algo_param=0.0):
    """Monkeypatch utils.calc_loss.calculate_lower_loss with the selected variant."""
    _STATE.update(prof=prof, arm=arm, early=bool(early_stop_infonce),
                  algo=algo, param=int(algo_param))
    orig = CL.calculate_lower_loss
    _STATE["orig_lower"] = orig
    CL.calculate_lower_loss = _make_lower(CL)
    print(f"[arms] installed arm={arm} early_stop_infonce={early_stop_infonce} "
          f"algo={algo} param={int(algo_param)}")


def _make_lower(CL):
    from utils.esd_utils import latent_sample
    from utils.infoNCE import calculate_steer_loss

    def lower(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer,
              noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype,
              t_enc_ddpm, start_guidance=3, ddim_steps=28, K=3, ir_concept_lst=()):
        prof = _STATE["prof"]
        early = _STATE["early"]
        algo = _STATE["algo"]
        param = _STATE["param"]

        vcfg_shift = vae.config.shift_factor
        vcfg_scale = vae.config.scaling_factor
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

        with prof("lower_vae_encode"):
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
            model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - vcfg_shift) * vcfg_scale
        model_input = model_input.to(dtype=weight_dtype)

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2,
            transformer.device, weight_dtype)
        noise = torch.randn_like(model_input)

        emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(prompts, text_encoders, tokenizers)
        noisy = noise_scheduler.add_noise(model_input, noise, t_enc_ddpm)
        packed = FluxPipeline._pack_latents(
            noisy, batch_size=model_input.shape[0], num_channels_latents=model_input.shape[1],
            height=model_input.shape[2], width=model_input.shape[3])

        if transformer.config.guidance_embeds:
            guidance = torch.tensor([args.guidance_scale], device=transformer.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        with prof("lower_fwd_grad"):
            model_pred, attn_maps = transformer(
                hidden_states=packed.to(dtype=weight_dtype, device=transformer.device),
                timestep=t_enc_ddpm / 1000,
                guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
                pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
                encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
                txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
                img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
                return_dict=False)[0:2]

        model_pred = FluxPipeline._unpack_latents(
            model_pred, height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor, vae_scale_factor=vae_scale_factor)

        target = noise - model_input
        loss_lora = torch.mean(((model_pred.float() - target.float()) ** 2)
                               .reshape(target.shape[0], -1), 1)[0]
        total = [loss_lora]

        start_code = torch.randn_like(model_input)
        sg = torch.tensor([3], device=transformer.device).expand(model_input.shape[0])

        emb_neg, pooled_emb_neg, text_ids_neg = compute_text_embeddings(
            batch["synonym_words"], text_encoders, tokenizers)
        if len(ir_concept_lst) != K:
            raise Exception("ir_concept_lst length mismatch")
        pos_embs = [compute_text_embeddings(ir_concept_lst[i], text_encoders, tokenizers)
                    for i in range(K)]

        # ---- draw the consumed index FIRST (see module docstring: same value) ----
        j = random.randint(0, int(ddim_steps) - 1)
        n_steps = (j + 1) if (early or algo == "shared_prefix") else int(ddim_steps)

        def traj_map(emb, pooled, tids):
            with torch.no_grad():
                _, _, lst = latent_sample(
                    transformer, noise_scheduler, 1, model_input.shape[1], 512, 512,
                    emb.to(transformer.device), pooled.to(transformer.device),
                    tids.to(transformer.device), sg, int(n_steps), vae_scale_factor,
                    latents=start_code, return_attn=True)
            return lst[j if n_steps == int(ddim_steps) else -1]

        if algo == "shared_prefix":
            # ONE prefix with the carrier embedding, then one forward per concept at z_j*
            carrier = (emb_p, pooled_emb_p, text_ids_p) if param == 0 else \
                      (emb_neg, pooled_emb_neg, text_ids_neg)
            with prof("lower_shared_prefix"):
                with torch.no_grad():
                    z_star, lids, _ = latent_sample(
                        transformer, noise_scheduler, 1, model_input.shape[1], 512, 512,
                        carrier[0].to(transformer.device), carrier[1].to(transformer.device),
                        carrier[2].to(transformer.device), sg, int(n_steps),
                        vae_scale_factor, latents=start_code, return_attn=True)

            def map_at_shared(emb, pooled, tids):
                with torch.no_grad():
                    _, m = transformer(
                        hidden_states=z_star,
                        timestep=(t_enc_ddpm / 1000).to(transformer.device),
                        guidance=guidance, pooled_projections=pooled.to(transformer.device),
                        encoder_hidden_states=emb.to(transformer.device),
                        txt_ids=tids.to(transformer.device), img_ids=lids,
                        return_dict=False)[0:2]
                return m

            with prof("lower_shared_reads"):
                attn_neg = map_at_shared(emb_neg, pooled_emb_neg, text_ids_neg)
                pos_lst = [map_at_shared(*pos_embs[i]) for i in range(K)]
        else:
            with prof("lower_infonce_traj"):
                attn_neg = traj_map(emb_neg, pooled_emb_neg, text_ids_neg)
                pos_lst = [traj_map(*pos_embs[i]) for i in range(K)]

        ri = batch["remove_indices"][0]
        info_neg = attn_neg[..., ri][:, 0, ...].permute(0, 2, 1)
        info_pos_lst = [p[..., ri][:, 0, ...].permute(0, 2, 1) for p in pos_lst]
        info_center = attn_maps[..., ri][:, 0, ...].permute(0, 2, 1)
        total.append(calculate_steer_loss(info_center, info_neg, info_pos_lst, temperature=0.07))
        _STATE["last_j"] = j
        _STATE["last_nsteps"] = n_steps
        return total

    return lower
