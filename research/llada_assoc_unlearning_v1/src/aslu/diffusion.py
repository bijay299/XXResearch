# Copied read-only from research/llada_method_design_v1/src/mdmu/diffusion.py at commit b01bcbb (closed sprint; reused as reference
# code under the new assignment; not modified there). Local changes in this copy are marked "ASLU:".
"""LLaDA-faithful masked-diffusion training loss and native sampler, ported from the pinned official source
(github.com/ML-GSAI/LLaDA @ 9182493720ed723ef8031210d85959364e51cbe0: GUIDELINES.md "SFT", generate.py, get_log_likelihood.py).

Differences from upstream (deliberate, semantics-preserving):
- explicit torch.Generator for reproducibility instead of the global RNG;
- `mask_id` is the toy vocabulary's MASK id; EOS/EoT-specific logit options are omitted (not used by the toy);
- CFG is supported exactly as upstream (prompt positions replaced by MASK in the unconditional pass).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .constants import MASK  # ASLU: decoupled from the closed sprint's toy world

# Batched top-k selection in `generate` (semantics-preserving speed-up; set False to use the upstream per-row loop).
_BATCHED_TOPK = True


# ----------------------------------------------------------------------------------------------- training loss
def forward_process_sft(ids: torch.Tensor, prompt_len: int, eps: float = 1e-3,
                        generator: Optional[torch.Generator] = None):
    """LLaDA SFT forward process: t~U(0,1), p_mask=(1-eps)t+eps per sequence; answer tokens (incl. EOS padding)
    masked iid w.p. p_mask; prompt never masked. Returns (noisy, masked_indices, p_mask[b,l])."""
    b, l = ids.shape
    t = torch.rand(b, generator=generator)
    p_mask = ((1 - eps) * t + eps)[:, None].expand(b, l)
    masked = torch.rand((b, l), generator=generator) < p_mask
    masked[:, :prompt_len] = False
    noisy = torch.where(masked, torch.full_like(ids, MASK), ids)
    return noisy, masked, p_mask


def sft_loss(model, ids: torch.Tensor, prompt_len: int, generator=None, eps: float = 1e-3,
             reduction: str = "mean"):
    """Exactly the GUIDELINES.md SFT loss: sum_masked CE/p_mask/answer_len, averaged over the batch."""
    noisy, masked, p_mask = forward_process_sft(ids, prompt_len, eps, generator)
    logits = model(noisy)
    ans_len = ids.shape[1] - prompt_len
    tok = F.cross_entropy(logits[masked], ids[masked], reduction="none") / p_mask[masked]
    if reduction == "none":
        per = torch.zeros(ids.shape[0])
        per.index_add_(0, masked.nonzero()[:, 0], tok / ans_len)
        return per
    return tok.sum() / ans_len / ids.shape[0]


# ----------------------------------------------------------------------------------------------- native sampler
def add_gumbel_noise(logits, temperature, generator=None):
    """Upstream: logits.exp() / (-log U)^temperature in float64 (argmax == Gumbel-max sample of softmax(logits/T))."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand(logits.shape, dtype=torch.float64, generator=generator)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, : remainder[i]] += 1
    return num


@torch.no_grad()
def generate(model, prompt: torch.Tensor, steps: int, gen_length: int, block_length: int,
             temperature: float = 0.0, cfg_scale: float = 0.0, remasking: str = "low_confidence",
             generator: Optional[torch.Generator] = None,
             logits_hook: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
             return_trace: bool = False):
    """Port of upstream `generate` (batched prompts of equal length, no padding).

    logits_hook(x, logits) -> logits, if given, is applied after the model call (used ONLY by reference-side
    utilities such as constrained proposal sampling; never by the deployed/evaluated model).
    return_trace: also return the list of states x before each step (for on-policy state collection).
    """
    b, plen = prompt.shape
    x = torch.full((b, plen + gen_length), MASK, dtype=torch.long)
    x[:, :plen] = prompt
    prompt_index = x != MASK
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    trace = []
    for nb in range(num_blocks):
        bs, be = plen + nb * block_length, plen + (nb + 1) * block_length
        block_mask_index = x[:, bs:be] == MASK
        num_transfer = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            if return_trace:
                trace.append(x.clone())
            mask_index = x == MASK
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK
                logits_all = model(torch.cat([x, un_x], 0))
                logits, un_logits = torch.chunk(logits_all, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)
            if logits_hook is not None:
                logits = logits_hook(x, logits)
            logits_with_noise = add_gumbel_noise(logits, temperature, generator)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, dtype=torch.float64, generator=generator)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, be:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            ks = num_transfer[:, i]
            if _BATCHED_TOPK and bool((ks == ks[0]).all()):
                # F (wave 3) fast path: identical per-row top-k when every row commits the same k (always true for
                # equal-length all-mask starts); equivalence vs the upstream per-row loop is tested in test_fxmix.py.
                k = int(ks[0])
                if k > 0:
                    _, sel = torch.topk(confidence, k=k, dim=1)
                    transfer.scatter_(1, sel, True)
            else:
                for j in range(b):
                    k = int(num_transfer[j, i])
                    if k > 0:
                        _, sel = torch.topk(confidence[j], k=k)
                        transfer[j, sel] = True
            x[transfer] = x0[transfer]
    return (x, trace) if return_trace else x


# ----------------------------------------------------------------------------------------------- likelihood
@torch.no_grad()
def answer_nll_mc(model, ids: torch.Tensor, prompt_len: int, mc: int = 32, generator=None):
    """Per-example MC estimate of the LLaDA ELBO NLL bound (get_log_likelihood semantics: masked count
    k~U{1..L}, CE/p_mask summed). Returns tensor (b,) of NLL upper-bound estimates (nats per sequence)."""
    b, l = ids.shape
    L = l - prompt_len
    tot = torch.zeros(b)
    for _ in range(mc):
        k = torch.randint(1, L + 1, (b,), generator=generator)
        perm = torch.argsort(torch.rand(b, L, generator=generator), dim=1)
        is_mask = perm < k[:, None]
        masked = torch.zeros(b, l, dtype=torch.bool)
        masked[:, prompt_len:] = is_mask
        noisy = torch.where(masked, torch.full_like(ids, MASK), ids)
        logits = model(noisy)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none").view(b, l)
        p_mask = (k.float() / L)[:, None]
        tot += (ce * masked / p_mask).sum(1)
    return tot / mc


# ----------------------------------------------------------------------------------------------- ASLU: pretraining loss
def forward_process_pretrain(ids: torch.Tensor, eps: float = 1e-3, generator: Optional[torch.Generator] = None,
                             maskable: Optional[torch.Tensor] = None):
    """ASLU: LLaDA PRETRAINING forward process (GUIDELINES.md "Pre-training"): every position (prompt and answer alike)
    masked iid w.p. p_mask = (1-eps)t+eps, t~U(0,1). `maskable` (b,l) bool optionally excludes padding."""
    b, l = ids.shape
    t = torch.rand(b, generator=generator)
    p_mask = ((1 - eps) * t + eps)[:, None].expand(b, l)
    masked = torch.rand((b, l), generator=generator) < p_mask
    if maskable is not None:
        masked &= maskable
    noisy = torch.where(masked, torch.full_like(ids, MASK), ids)
    return noisy, masked, p_mask


def pretrain_loss(model, ids: torch.Tensor, generator=None, eps: float = 1e-3, maskable=None):
    """ASLU: GUIDELINES.md pretraining loss: sum_masked CE/p_mask divided by (b*l)."""
    noisy, masked, p_mask = forward_process_pretrain(ids, eps, generator, maskable)
    logits = model(noisy)
    tok = F.cross_entropy(logits[masked], ids[masked], reduction="none") / p_mask[masked]
    return tok.sum() / (ids.shape[0] * ids.shape[1])
