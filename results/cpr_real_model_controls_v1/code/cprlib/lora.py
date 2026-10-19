"""Source-derived native GEM `qk_dual` LoRA for FLUX, implemented afresh.

Resolved from the pinned GEM source
(`multimodal-ai-lab/GEM@c328fc6bec687efe60223498d1b6bed2e5fbcffd`):

* `GEMConfig.get_default_adaptation_config()` ->
  `AdaptationConfig(adapter_mode="lora", subset_name="qk_dual")`
* `AdaptationConfig.__post_init__` LoRA defaults ->
  `r=16, lora_alpha=16, lora_dropout=0.1, bias="none"`
* `pipeline_wrapper_flux._get_modules_for_subset("qk_dual")` ->
  `find(startswith=["transformer"], exclude=["single_transformer_blocks"],
        endswith=["add_q_proj", "add_k_proj", "to_q", "to_k"])`
  matching `Linear`/`Conv2d`/`LoRACompatible*` modules only, where the
  name-token rules are: first token == "transformer"; no name TOKEN equal to
  "single_transformer_blocks"; last token in the endswith set.

PEFT is deliberately NOT a dependency here (it is absent from the pinned
read-only environment and installing into a shared venv is not permitted), so
the standard LoRA arithmetic PEFT would apply is reproduced directly:

    y = base(x) + dropout(x) @ A^T @ B^T * (lora_alpha / r)
    A ~ kaiming_uniform_(a=sqrt(5)),  B = 0   (so the adapter starts inert)

`disable_adapter()` reproduces PEFT's context manager used by the GEM training
loop for both teacher forward passes.
"""

import contextlib
import hashlib
import math

import torch
import torch.nn as nn

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
QK_DUAL_ENDSWITH = ("add_q_proj", "add_k_proj", "to_q", "to_k")
QK_DUAL_EXCLUDE_TOKEN = "single_transformer_blocks"


class LoRALinear(nn.Module):
    """Wraps a frozen base Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # FP32 adapter parameters on a BF16 backbone: the update is computed in
        # FP32 and cast back, which keeps the optimizer numerically sane.
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._disabled = False

    def forward(self, x):
        out = self.base(x)
        if self._disabled:
            return out
        h = self.lora_dropout(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(h.to(torch.float32), self.lora_A), self.lora_B)
        return out + (delta * self.scaling).to(out.dtype)


def select_qk_dual(transformer):
    """Resolve the native `qk_dual` target modules under a `transformer` root.

    Names are produced relative to the transformer, so the GEM rule
    "first token == 'transformer'" is satisfied by construction; the
    exclude/endswith token rules are applied exactly as the source does.
    """
    targets = {}
    for name, mod in transformer.named_modules():
        tokens = name.split(".")
        if QK_DUAL_EXCLUDE_TOKEN in tokens:
            continue
        if tokens[-1] not in QK_DUAL_ENDSWITH:
            continue
        if mod.__class__.__name__ not in ("Linear", "LoRACompatibleLinear"):
            continue
        targets[name] = mod
    return targets


def inject(transformer, seed):
    """Replace each target Linear with a LoRALinear. Returns the wrapper map."""
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)                     # couples adapter init across arms
    targets = select_qk_dual(transformer)
    wrapped = {}
    for name in sorted(targets):
        parent_path, _, leaf = name.rpartition(".")
        parent = transformer.get_submodule(parent_path) if parent_path else transformer
        base = getattr(parent, leaf)
        lin = LoRALinear(base)
        lin.to(device=base.weight.device)
        setattr(parent, leaf, lin)
        wrapped[name] = lin
    torch.random.set_rng_state(gen_state)
    for n, p in transformer.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)
    return wrapped


@contextlib.contextmanager
def disable_adapter(wrapped):
    """PEFT-equivalent `disable_adapter()` used for both teacher forwards."""
    for m in wrapped.values():
        m._disabled = True
    try:
        yield
    finally:
        for m in wrapped.values():
            m._disabled = False


def trainable_parameters(wrapped):
    ps = []
    for name in sorted(wrapped):
        ps.append(wrapped[name].lora_A)
        ps.append(wrapped[name].lora_B)
    return ps


# --------------------------------------------------------------------------
# compact serialization: only the adapter tensors, never a transformer copy
# --------------------------------------------------------------------------

def state_dict(wrapped):
    """Canonical compact representation, identical in form across all arms."""
    sd = {}
    for name in sorted(wrapped):
        m = wrapped[name]
        sd[f"{name}.lora_A"] = m.lora_A.detach().to(torch.float32).cpu().clone()
        sd[f"{name}.lora_B"] = m.lora_B.detach().to(torch.float32).cpu().clone()
    return sd


def metadata(wrapped):
    return {"r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "bias": LORA_BIAS, "scaling": LORA_ALPHA / LORA_R,
            "subset_name": "qk_dual",
            "endswith": list(QK_DUAL_ENDSWITH),
            "exclude_token": QK_DUAL_EXCLUDE_TOKEN,
            "n_modules": len(wrapped),
            "module_names": sorted(wrapped),
            "init": "lora_A kaiming_uniform_(a=sqrt(5)); lora_B zeros",
            "dtype": "float32 adapter on bfloat16 backbone"}


def load_state_dict(wrapped, sd, strict=True):
    """Reload a compact snapshot. Verifies keys, shapes, dtypes and finiteness."""
    report = {"loaded": 0, "missing": [], "unexpected": [], "nonfinite": [],
              "shape_mismatch": []}
    expected = set()
    for name in sorted(wrapped):
        expected.update({f"{name}.lora_A", f"{name}.lora_B"})
    report["unexpected"] = sorted(set(sd) - expected)
    report["missing"] = sorted(expected - set(sd))
    if strict and (report["missing"] or report["unexpected"]):
        raise ValueError(f"adapter key mismatch: {report}")
    for name in sorted(wrapped):
        m = wrapped[name]
        for leaf, param in (("lora_A", m.lora_A), ("lora_B", m.lora_B)):
            key = f"{name}.{leaf}"
            if key not in sd:
                continue
            t = sd[key]
            if tuple(t.shape) != tuple(param.shape):
                report["shape_mismatch"].append((key, tuple(t.shape), tuple(param.shape)))
                continue
            if not torch.isfinite(t).all():
                report["nonfinite"].append(key)
                continue
            with torch.no_grad():
                param.copy_(t.to(param.dtype).to(param.device))
            report["loaded"] += 1
    if strict and (report["shape_mismatch"] or report["nonfinite"]):
        raise ValueError(f"adapter reload failed: {report}")
    return report


def state_hash(sd):
    """Order-independent-by-sorted-key hash of the canonical representation."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].to(torch.float32).cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def effective_delta(wrapped, name):
    """The effective weight update B @ A * scaling for one module (FP32)."""
    m = wrapped[name]
    return (m.lora_B @ m.lora_A) * m.scaling
