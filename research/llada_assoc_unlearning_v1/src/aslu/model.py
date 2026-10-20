# Copied read-only from research/llada_method_design_v1/src/mdmu/model.py at commit b01bcbb (closed sprint; reused as reference
# code under the new assignment; not modified there). Local changes in this copy are marked "ASLU:".
"""Tiny bidirectional masked transformer (LLaDA-style mask predictor, no causal mask) + LoRA.

Architecture mirrors the LLaDA block structure in miniature: pre-RMSNorm, multi-head self-attention with
q/k/v/o projections and NO causal mask, SwiGLU MLP (gate/up/down), learned absolute positions (LLaDA uses RoPE;
positions here are a fixed 16-token window so learned embeddings are adequate), untied output head.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B A x, W frozen. A ~ N(0, 1/in) (Kaiming-style), B = 0 (so the adapter starts at 0)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, generator: torch.Generator | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, generator=generator) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.enabled = True  # False => adapter-disabled pass (= frozen reference theta_0), cf. PEFT disable_adapter()

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

    def delta_weight(self) -> torch.Tensor:
        return self.scale * self.B @ self.A


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.n_heads = n_heads
        self.attn_norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.ff_norm = RMSNorm(d)
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(b, l, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(h).view(b, l, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(h).view(b, l, self.n_heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bidirectional
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, l, d))
        h = self.ff_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class MaskPredictor(nn.Module):
    def __init__(self, vocab: int, seq_len: int, d: int = 64, n_layers: int = 2, n_heads: int = 4, d_ff: int = 192):
        super().__init__()
        self.cfg = dict(vocab=vocab, seq_len=seq_len, d=d, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids (b, l) -> logits (b, l, V). Mirrors `model(input_ids).logits` of LLaDA."""
        x = self.tok(ids) + self.pos.weight[: ids.shape[1]][None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def add_lora(model: MaskPredictor, r: int = 4, alpha: float = 8.0, targets: Iterable[str] = DEFAULT_LORA_TARGETS,
             layers: Iterable[int] | None = None, seed: int = 0) -> List[LoRALinear]:
    """Freeze all base parameters and wrap target linears of the selected blocks with LoRA. Returns the adapters."""
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    layers = list(range(len(model.blocks))) if layers is None else list(layers)
    out = []
    for li in layers:
        blk = model.blocks[li]
        for name in targets:
            lin = getattr(blk, name)
            assert isinstance(lin, nn.Linear), f"{name} already wrapped"
            lo = LoRALinear(lin, r, alpha, g)
            setattr(blk, name, lo)
            out.append(lo)
    return out


class lora_disabled:
    """Context manager: run `model` with every LoRALinear adapter switched off (reference pass p_theta0).
    Mirrors PEFT's `with model.disable_adapter():` used by the LLaDA integration."""

    def __init__(self, model: nn.Module):
        self.mods = [m for m in model.modules() if isinstance(m, LoRALinear)]

    def __enter__(self):
        self.prev = [m.enabled for m in self.mods]
        for m in self.mods:
            m.enabled = False
        return self

    def __exit__(self, *exc):
        for m, e in zip(self.mods, self.prev):
            m.enabled = e
        return False


def merged_copy(model: nn.Module) -> nn.Module:
    """Deep copy with every enabled LoRALinear folded into a plain nn.Linear (W + scale*B@A): identical function,
    faster inference (used for evaluation; mirrors PEFT merge_and_unload for deployment)."""
    import copy
    m = copy.deepcopy(model)
    for mod in list(m.modules()):
        for name, ch in list(mod.named_children()):
            if isinstance(ch, LoRALinear):
                lin = nn.Linear(ch.base.in_features, ch.base.out_features, bias=False)
                w = ch.base.weight.detach().clone()
                if ch.enabled:
                    w = w + ch.delta_weight().detach()
                lin.weight = nn.Parameter(w, requires_grad=False)
                setattr(mod, name, lin)
    m.eval()
    return m


def lora_params(adapters: List[LoRALinear]) -> List[nn.Parameter]:
    return [p for a in adapters for p in (a.A, a.B)]


def count_params(params: Iterable[torch.Tensor]) -> int:
    return sum(p.numel() for p in params)


def state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
