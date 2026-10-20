"""FX-Mix on LLaDA-8B-Instruct (SPEC v1.1) — integration against the pinned interfaces.

STATUS: dry-run tested on CPU against a tiny random model exposing `.logits` (tests/test_integration_dryrun.py).
**Real-model execution: UNTESTED** (no GPU, HuggingFace blocked on the authoring host).

Pinned interfaces (github.com/ML-GSAI/LLaDA @ 9182493720ed723ef8031210d85959364e51cbe0, generate.py / GUIDELINES.md):
  model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16); model(x, attention_mask=..).logits
  MASK 126336, EOS 126081 (answer padding), EoT 126348; vocab 126464 (UNVERIFIED: read from model.config at run time)
  SFT layout: prompt = chat template + generation prompt (never masked); answer = text + EOS padding to the fixed
  answer length; loss = sum_masked CE/p_mask / answer_len(incl. EOS) / batch.
LoRA target names: **UNVERIFIED** — the HF `modeling_llada.py` is not inspectable here. The only local evidence is the
third-party copy in dllm (@ca176752, dllm/pipelines/llada/models/modeling_llada.py, class LLaDALlamaBlock):
  q_proj, k_proj, v_proj, attn_out (= o), ff_proj (= gate, activation applied), up_proj, ff_out (= down);
  and the LM head is ALSO named `transformer.ff_out` -> a suffix match on "ff_out" would wrap the head. We therefore
  target a block-anchored regex and `check_lora_targets` asserts exactly 7 wrapped linears per block, head untouched.
  (dllm's own GRPO LLaDA example lists Llama names o_proj/gate_proj/down_proj, which this copy does not define.)

Everything heavy (transformers, peft, yaml) is imported lazily so this file imports on the CPU host.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mdmu.fxmix import exclusion_target, soft_ce_kl  # noqa: E402  (model-agnostic, log-space, tested)

MASK_ID, EOS_ID, EOT_ID = 126336, 126081, 126348
LORA_TARGET_REGEX_UNVERIFIED = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"
LORA_TARGET_NAMES_UNVERIFIED = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")


# ============================================================================================ config
@dataclasses.dataclass
class FXConfig:
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"   # Hyperplane: the TOFU-SFT'd checkpoint path
    dtype: str = "bfloat16"
    gen_length: int = 128
    eps: float = 1e-5                    # SPEC v1.3 LLaDA default; report fixed_point_report(...) from theta0 rollouts
    alpha: float = 0.25                  # TF_ff share of forget states
    J: int = 8                           # pre-tau states per rollout (+ decision state)
    refresh_every: int = 4               # R
    rollout_prompts_per_refresh: int = 8  # forget prompts rolled out per refresh (toy: all 24); caller samples them
    pool: Tuple[Tuple[int, int, str, float], ...] = (
        (128, 128, "low_confidence", 0.0), (128, 128, "random", 1.0), (64, 128, "random", 1.0),
        (32, 128, "random", 1.0))        # SPEC v1 pool + v1.1 entry (32,128,rnd,1)
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_target_regex: str = LORA_TARGET_REGEX_UNVERIFIED
    lr: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 10
    steps: int = 300
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    forget_states_per_step: int = 8
    retain_batch: int = 8
    lam_f: float = 1.0
    lam_r: float = 1.0
    append_eot: bool = False             # GUIDELINES SFT layout has no EoT before EOS padding (UNVERIFIED for TOFU-SFT)
    sft_eps: float = 1e-3                # LLaDA forward-process eps (GUIDELINES)
    seed: int = 0


def load_config(path: str) -> FXConfig:
    """YAML -> FXConfig (uses PyYAML if available, else a minimal parser for the flat file we ship)."""
    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(open(path))
    except ImportError:
        raw = _mini_yaml(open(path).read())
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    kw = {}
    for f in dataclasses.fields(FXConfig):
        if f.name in flat:
            v = flat[f.name]
            if f.name == "pool":
                v = tuple(tuple(e) for e in v)
            kw[f.name] = v
    return FXConfig(**kw)


def _mini_yaml(text: str) -> dict:
    """Parses `key: value` lines, one level of indentation, `- [a, b, c, d]` list items, comments. Not general YAML."""
    import ast
    root: dict = {}
    cur = root
    last_key = None
    for line in text.splitlines():
        line = line.split(" #")[0].rstrip() if not line.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        s = line.strip()
        if s.startswith("- "):
            item = s[2:].strip()
            cur.setdefault(last_key, [])
            cur[last_key].append(ast.literal_eval(item.replace("low_confidence", "'low_confidence'")
                                                  .replace("random", "'random'")) if item.startswith("[") else
                                 _scalar(item))
            continue
        k, _, v = s.partition(":")
        k, v = k.strip(), v.strip()
        if ind == 0:
            cur = root
        if v == "":
            if ind == 0:
                root[k] = {}
                cur = root[k]
            else:
                cur[k] = []
            last_key = k
        else:
            cur[k] = _scalar(v)
            last_key = k
    return root


def _scalar(v: str):
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v.strip("'\"")


# ============================================================================================ sampler port
def add_gumbel_noise(logits, temperature, generator=None):
    """Upstream add_gumbel_noise (float64). generator=None reproduces upstream RNG use exactly (torch.rand_like)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    if generator is None:
        noise = torch.rand_like(logits, dtype=torch.float64)
    else:
        noise = torch.rand(logits.shape, dtype=torch.float64, generator=generator, device=logits.device)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, : remainder[i]] += 1
    return num


@torch.no_grad()
def generate_traced(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
                    cfg_scale=0., remasking="low_confidence", mask_id=MASK_ID, logits_eos_inf=False,
                    confidence_eos_eot_inf=False, generator=None, fact_mask: Optional[torch.Tensor] = None,
                    record: bool = False, stop_when_all_leaked: bool = False,
                    on_step: Optional[Callable] = None):
    """Line-by-line port of upstream `generate` (end-think boost omitted: it is a no-op at its default 0 and is out of
    scope, SPEC v1 section 1) with optional, side-effect-free extras:
      fact_mask (b, V) bool: detect the first step tau whose committed set contains an F token, per row;
      record: keep the ANSWER part of the state before each step (prompt is constant);
      stop_when_all_leaked: stop once every row has committed an F token (state collection only: the final x is then
        incomplete, so never use it for evaluation);
      on_step(n, x, logits, candidate_mask, transfer_index, x0): callback (e.g. certificate accumulation), called
        before the commit with the logits used for selection.
    With extras off and generator=None the RNG calls and outputs are identical to upstream (tested)."""
    b, plen = prompt.shape
    dev = model.device
    x = torch.full((b, plen + gen_length), mask_id, dtype=torch.long).to(dev)
    x[:, :plen] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((b, gen_length), dtype=attention_mask.dtype,
                                                               device=dev)], dim=-1)
    prompt_index = x != mask_id
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    trace: List[torch.Tensor] = []
    first_commit = torch.full((b,), -1, dtype=torch.long)
    n_global = 0
    for num_block in range(num_blocks):
        block_mask_index = x[:, plen + num_block * block_length: plen + (num_block + 1) * block_length] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            if stop_when_all_leaked and fact_mask is not None and bool((first_commit >= 0).all()):
                return x, {"trace": trace, "first_commit": first_commit, "n_steps_run": n_global}
            if record:
                trace.append(x[:, plen:].clone())
            mask_index = x == mask_id
            block_start, block_end = plen + num_block * block_length, plen + (num_block + 1) * block_length
            candidate_mask_index = mask_index.clone()
            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                am_ = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                logits = model(x_, attention_mask=am_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            if logits_eos_inf:
                logits[:, :, EOS_ID] = -torch.inf
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature, generator=generator)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if confidence_eos_eot_inf:
                # upstream chained assignment, kept verbatim: writes -inf into logits_with_noise[..., EOS] AND
                # logits[..., EoT] (the latter changes the low-confidence p below)
                logits_with_noise[:, :, EOS_ID] = logits[:, :, EOT_ID] = -torch.inf
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                if generator is None:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device, generator=generator)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, plen + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            if on_step is not None:
                on_step(n_global, x, logits, candidate_mask_index, transfer_index, x0)
            if fact_mask is not None:
                hit = (transfer_index & torch.gather(fact_mask.to(dev), 1, x0)).any(1).cpu()
                first_commit = torch.where((first_commit < 0) & hit, torch.full_like(first_commit, n_global),
                                           first_commit)
            x[transfer_index] = x0[transfer_index]
            n_global += 1
    return x, {"trace": trace, "first_commit": first_commit, "n_steps_run": n_global}


def select_rollout_states(trace: List[torch.Tensor], first_commit: torch.Tensor, J: int,
                          generator: Optional[torch.Generator] = None) -> List[Tuple[int, int, bool]]:
    """SPEC v1 (E5): per row keep J states sampled uniformly from the pre-tau states PLUS the decision state s_tau
    when it exists. Returns (row, step, is_decision) triples; states are trace[step][row] (answer part)."""
    out = []
    n_rec = len(trace)
    for r in range(first_commit.shape[0]):
        tau = int(first_commit[r])
        pre = list(range(tau)) if tau >= 0 else list(range(n_rec))
        k = min(J, len(pre))
        pick = [pre[i] for i in torch.randperm(len(pre), generator=generator)[:k].tolist()]
        out += [(r, n, False) for n in sorted(pick)]
        if tau >= 0:
            out.append((r, tau, True))
    return out


# ============================================================================================ data layout
def chat_prompt_ids(tokenizer, question: str) -> List[int]:
    """Instruct prompt exactly as upstream generate.py main(): chat template with generation prompt, no extra specials."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                         tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def answer_ids(tokenizer, answer: str, gen_length: int, append_eot: bool = False) -> List[int]:
    """GUIDELINES SFT answer: answer tokens (+ optional EoT, UNVERIFIED) + EOS padding to the fixed gen_length."""
    a = tokenizer(answer, add_special_tokens=False)["input_ids"] + ([EOT_ID] if append_eot else [])
    if len(a) >= gen_length:
        a = a[: gen_length - 1]
    return a + [EOS_ID] * (gen_length - len(a))


def sft_batch(tokenizer, pairs: Sequence[Tuple[str, str]], gen_length: int, append_eot: bool = False):
    """Retain/SFT batch in the GUIDELINES layout: prompt || answer || EOS padding, right-padded with EOS to a common
    length (the extra EOS are answer tokens, as in GUIDELINES); returns (input_ids, prompt_lengths)."""
    rows, pl = [], []
    for q, a in pairs:
        p = chat_prompt_ids(tokenizer, q)
        rows.append(p + answer_ids(tokenizer, a, gen_length, append_eot))
        pl.append(len(p))
    Lm = max(len(r) for r in rows)
    ids = torch.tensor([r + [EOS_ID] * (Lm - len(r)) for r in rows], dtype=torch.long)
    return ids, torch.tensor(pl, dtype=torch.long)


def left_pad(seqs: Sequence[Sequence[int]], pad_id: int):
    """Left padding as recommended upstream; returns (ids, attention_mask). pad_id must differ from MASK_ID."""
    assert pad_id != MASK_ID
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        am[i, L - len(s):] = 1
    return ids, am


# ============================================================================================ fact sets
DEFAULT_STOPLIST = {"the", "a", "an", "of", "in", "on", "and", "or", "is", "was", "are", "were", "to", "for", "by",
                    "with", "as", "at", "from", "his", "her", "their", "he", "she", "they", "it", "its", "this",
                    "that", "has", "have", "had", "be", "been", "who", "which", "also", "known", "born", "s"}


def build_fact_set(tokenizer, prompt_ids: Sequence[int], answer: str, fact_strings: Optional[Sequence[str]] = None,
                   stoplist=DEFAULT_STOPLIST, frequent_ids: Optional[set] = None) -> Dict:
    """F(c, y*) per fact (SPEC v1 section 1; APPROXIMATION A2): token ids of the fact's distinctive surface pieces
    (fact_strings if given, else the answer) that are not in the prompt, not special, not punctuation/whitespace,
    not stop-list words, not in `frequent_ids` (e.g. top-N retain-answer tokens); plus tokenisation variants
    (leading space / no space / capitalised / lower-case; ALL tokens of each variant's encoding). Variant ids that
    occur in the prompt are dropped and reported ("conditioning supplies the answer" items are reported separately)."""
    special = set(getattr(tokenizer, "all_special_ids", []) or []) | {MASK_ID, EOS_ID, EOT_ID}
    prompt_set = set(prompt_ids)
    frequent_ids = frequent_ids or set()
    sources = list(fact_strings) if fact_strings else [answer]

    def ok_piece(tid):
        if tid in special or tid in frequent_ids:
            return False
        s = tokenizer.decode([tid]).strip()
        if not s or all(not ch.isalnum() for ch in s):
            return False
        return s.lower() not in stoplist

    words = []
    for src in sources:
        words += re.findall(r"\w[\w'\-\.]*\w|\w", src)
    cand = set()
    for src in sources:
        cand |= {t for t in tokenizer(src, add_special_tokens=False)["input_ids"] if ok_piece(t)}
    for w in words:
        if w.lower() in stoplist:
            continue
        for v in {w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()}:
            cand |= {t for t in tokenizer(v, add_special_tokens=False)["input_ids"] if ok_piece(t)}
    in_prompt = sorted(cand & prompt_set)
    F_ids = sorted(cand - prompt_set)
    return {"F": F_ids, "dropped_in_prompt": in_prompt, "prompt_conflict": bool(in_prompt)}


# ============================================================================================ LoRA
def apply_lora(model, cfg: FXConfig, backend: str = "peft"):
    """Wrap the 7 block linears with LoRA (A Gaussian, B = 0). backend 'peft' (Hyperplane) or 'builtin' (dry run).
    Returns a model exposing `.disable_adapter()` (context manager) for the reference pass."""
    if backend == "peft":
        from peft import LoraConfig, get_peft_model  # type: ignore
        lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.lora_target_regex,
                        lora_dropout=0.0, bias="none", init_lora_weights="gaussian")
        # NB PEFT 'gaussian' init: A ~ N(0, 1/r), B = 0  (SPEC: A Gaussian, B = 0)
        m = get_peft_model(model, lc)
        check_lora_targets(m)
        return m
    return _BuiltinLoRA(model, cfg)


class _BuiltinLoRA(nn.Module):
    """Minimal PEFT stand-in for the CPU dry run (uses mdmu.model.LoRALinear)."""

    def __init__(self, model, cfg: FXConfig):
        super().__init__()
        from mdmu.model import LoRALinear
        self.base_model = model
        for p in model.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(cfg.seed)
        rx = re.compile(cfg.lora_target_regex)
        for name, mod in list(model.named_modules()):
            for cname, ch in list(mod.named_children()):
                full = f"{name}.{cname}" if name else cname
                if isinstance(ch, nn.Linear) and rx.match(full):
                    setattr(mod, cname, LoRALinear(ch, cfg.lora_r, cfg.lora_alpha, g))
        self.adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
        check_lora_targets(self)

    @property
    def device(self):
        return self.base_model.device

    def forward(self, *a, **k):
        return self.base_model(*a, **k)

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = [m.enabled for m in self.adapters]
        for m in self.adapters:
            m.enabled = False
        try:
            yield
        finally:
            for m, e in zip(self.adapters, prev):
                m.enabled = e


def check_lora_targets(model) -> Dict[str, int]:
    """Assert 7 LoRA-wrapped linears per transformer block and an unwrapped LM head. Prints the wrapped names."""
    wrapped = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") or (hasattr(mod, "A") and hasattr(mod, "B") and hasattr(mod, "base")):
            mblk = re.search(r"blocks\.(\d+)\.", name)
            key = mblk.group(1) if mblk else "NON-BLOCK"
            wrapped.setdefault(key, []).append(name.split(".")[-1])
    assert "NON-BLOCK" not in wrapped, f"LoRA wrapped a non-block module (LM head?): {wrapped.get('NON-BLOCK')}"
    for blk, names in wrapped.items():
        assert len(set(names)) == 7, f"block {blk}: wrapped {sorted(set(names))} (expected 7 linears)"
    return {k: len(set(v)) for k, v in wrapped.items()}


def trainable_params(model) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ============================================================================================ losses
def _logits(model, ids, am):
    return model(ids, attention_mask=am).logits


def fx_forget_loss_llada(model, states: torch.Tensor, attn: torch.Tensor, ans_start: torch.Tensor,
                         fact_mask: torch.Tensor, eps: float) -> Dict:
    """(E1) forget term on LLaDA: mean over states of the mean over masked ANSWER positions of
    KL(r_eps || p_theta); r_eps from the adapter-disabled reference at the same state, computed in log space from
    fp32 logits gathered at the masked answer positions only (|V| = 126,464 => never materialise fp32 for all)."""
    pos = (states == MASK_ID)
    ar = torch.arange(states.shape[1], device=states.device)[None]
    pos &= ar >= ans_start[:, None]
    rows, cols = pos.nonzero(as_tuple=True)
    with torch.no_grad(), model.disable_adapter():
        z0 = _logits(model, states, attn)[rows, cols].float()
    log_r = exclusion_target(z0, fact_mask[rows], eps)
    z = _logits(model, states, attn)[rows, cols].float()
    ce, ent, kl = soft_ce_kl(log_r, z)
    cnt = torch.zeros(states.shape[0], device=states.device).index_add_(0, rows, torch.ones_like(kl))
    per = torch.zeros(states.shape[0], device=states.device, dtype=kl.dtype).index_add_(0, rows, kl)
    valid = cnt > 0
    loss = (per[valid] / cnt[valid]).mean()
    with torch.no_grad():
        p0F = (F.softmax(z0, -1) * fact_mask[rows]).sum(-1)
        pF = (F.softmax(z.detach(), -1) * fact_mask[rows]).sum(-1)
    return {"loss": loss, "ce": float(ce.detach().mean()), "ent": float(ent.mean()), "pF_ref": float(p0F.mean()),
            "pF_student": float(pF.mean()), "frac_active": float((p0F > eps).float().mean()), "n_pos": int(rows.numel())}


def sft_loss_llada(model, input_ids: torch.Tensor, prompt_lengths: torch.Tensor, attn=None, eps: float = 1e-3,
                   generator=None):
    """GUIDELINES.md SFT loss (retain term): t~U, p_mask=(1-eps)t+eps, prompt restored, masked = noisy == MASK,
    loss = sum CE/p_mask/answer_len(incl. EOS) / batch. (Right-padded with EOS as in GUIDELINES; with left-padded
    prompts pass attn and prompt_lengths = index of the first answer token.)"""
    b, l = input_ids.shape
    dev = input_ids.device
    t = torch.rand(b, device=dev, generator=generator)
    p_mask = ((1 - eps) * t + eps)[:, None].repeat(1, l)
    masked = torch.rand((b, l), device=dev, generator=generator) < p_mask
    noisy = torch.where(masked, torch.full_like(input_ids, MASK_ID), input_ids)
    pos = torch.arange(l, device=dev).expand(b, l)
    prompt_mask = pos < prompt_lengths[:, None]
    noisy[prompt_mask] = input_ids[prompt_mask]
    ans_len = (~prompt_mask).sum(-1, keepdim=True).repeat(1, l)
    mi = noisy == MASK_ID
    logits = _logits(model, noisy, attn)
    tok = F.cross_entropy(logits[mi].float(), input_ids[mi], reduction="none") / p_mask[mi]
    return torch.sum(tok / ans_len[mi]) / b


def tf_ff_states(full_ids: torch.Tensor, ans_start: torch.Tensor, fact_mask: torch.Tensor, generator=None):
    """(E4) on LLaDA rows (left-padded prompt || answer padded to gen_length with EOS): t~U(0,1); mask non-fact answer
    positions iid w.p. t; ALWAYS mask answer positions holding an F token."""
    b, l = full_ids.shape
    dev = full_ids.device
    t = torch.rand(b, device=dev, generator=generator)
    masked = torch.rand((b, l), device=dev, generator=generator) < t[:, None]
    masked |= torch.gather(fact_mask, 1, full_ids)
    masked &= torch.arange(l, device=dev)[None] >= ans_start[:, None]
    return torch.where(masked, torch.full_like(full_ids, MASK_ID), full_ids)


# ============================================================================================ rollout pool / step
@torch.no_grad()
def collect_pool(model, prompts: torch.Tensor, prompt_am: torch.Tensor, fact_mask: torch.Tensor, cfg: FXConfig,
                 refresh_idx: int, generator=None) -> Dict:
    """Rollouts of the CURRENT student (adapter ON, no grad) on forget prompts; config of prompt i at refresh k =
    pool[(i + k) % K] (stratified uniform w); first-F-commit truncation; J pre-tau states + decision state."""
    K = len(cfg.pool)
    S, A, M, leak, fwd = [], [], [], {}, 0
    for k, (steps, block, rm, T) in enumerate(cfg.pool):
        rows = [i for i in range(prompts.shape[0]) if (i + refresh_idx) % K == k]
        if not rows:
            continue
        P, Am, Fm = prompts[rows], prompt_am[rows], fact_mask[rows]
        x, info = generate_traced(model, P, Am, steps=steps, gen_length=cfg.gen_length, block_length=block,
                                  temperature=T, remasking=rm, generator=generator, fact_mask=Fm, record=True,
                                  stop_when_all_leaked=True)
        fwd += info["n_steps_run"] * len(rows)
        picks = select_rollout_states(info["trace"], info["first_commit"], cfg.J, generator)
        for r, n, _ in picks:
            S.append(torch.cat([P[r], info["trace"][n][r]]))
            A.append(torch.cat([Am[r], torch.ones(cfg.gen_length, dtype=Am.dtype, device=Am.device)]))
            M.append(Fm[r])
        leak[str((steps, block, rm, T))] = float((info["first_commit"] >= 0).float().mean())
    return {"states": torch.stack(S), "attn": torch.stack(A), "fact_mask": torch.stack(M),
            "ans_start": torch.full((len(S),), prompts.shape[1], dtype=torch.long), "leak_rate": leak,
            "rollout_fwd_rows": fwd}


def train_step(model, opt, sched, pool: Dict, forget_tf: Dict, retain: Dict, cfg: FXConfig, generator=None) -> Dict:
    """One FX-Mix step: round(alpha*B) TF_ff states + rest from the pool; forget KL + lam_r * SFT retain; clip; step.
    forget_tf: {"ids","attn","ans_start","fact_mask"} answer-padded forget rows; retain: {"ids","prompt_lengths","attn"}."""
    B = cfg.forget_states_per_step
    n_tf = int(round(cfg.alpha * B))
    dev = pool["states"].device
    ti = torch.randint(0, forget_tf["ids"].shape[0], (n_tf,), generator=generator).to(dev)
    pi = torch.randint(0, pool["states"].shape[0], (B - n_tf,), generator=generator).to(dev)
    s_tf = tf_ff_states(forget_tf["ids"][ti], forget_tf["ans_start"][ti], forget_tf["fact_mask"][ti], generator)
    # pool and TF rows can differ in length (left-padded prompts): right-align by left-padding to a common length
    states, attn, ans_start, fm = _concat_left([s_tf, pool["states"][pi]], [forget_tf["attn"][ti], pool["attn"][pi]],
                                               [forget_tf["ans_start"][ti], pool["ans_start"][pi]],
                                               [forget_tf["fact_mask"][ti], pool["fact_mask"][pi]])
    d = fx_forget_loss_llada(model, states, attn, ans_start, fm, cfg.eps)
    ri = torch.randint(0, retain["ids"].shape[0], (cfg.retain_batch,), generator=generator).to(dev)
    r_attn = retain.get("attn")
    l_ret = sft_loss_llada(model, retain["ids"][ri], retain["prompt_lengths"][ri],
                           None if r_attn is None else r_attn[ri], cfg.sft_eps, generator)
    loss = cfg.lam_f * d["loss"] + cfg.lam_r * l_ret
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable_params(model), cfg.grad_clip)
    opt.step()
    if sched is not None:
        sched.step()
    d = {k: (float(v.detach()) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
    d.update(retain=float(l_ret.detach()), grad_norm=float(gn))
    d["counts"] = {"student_fwd": states.shape[0] + cfg.retain_batch,
                   "student_bwd": states.shape[0] + cfg.retain_batch, "ref_fwd": states.shape[0]}
    return d


def _concat_left(states, attns, starts, fms):
    L = max(s.shape[1] for s in states)
    S, A, St = [], [], []
    for s, a, st in zip(states, attns, starts):
        pad = L - s.shape[1]
        S.append(F.pad(s, (pad, 0), value=EOS_ID))          # attention 0 on the pad; any non-MASK id
        A.append(F.pad(a, (pad, 0), value=0))
        St.append(st + pad)
    return torch.cat(S), torch.cat(A), torch.cat(St), torch.cat(fms)


def make_optimizer(model, cfg: FXConfig):
    opt = torch.optim.AdamW(trainable_params(model), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        def f(s):
            if s < cfg.warmup_steps:
                return (s + 1) / cfg.warmup_steps
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)
    return opt, None


# ============================================================================================ diagnostics
def implied_fixed_point_leak(eps: float, n_hat: float) -> float:
    """SPEC v1: at the fixed point p(F|s,i) = eps on n_hat decision positions, leak <= 1 - (1 - eps)^n_hat."""
    return 1.0 - (1.0 - eps) ** n_hat


def tempered_fact_prob(logits: torch.Tensor, fact_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """pi_T(F|s,i): softmax(z/T)(F) for T>0, 1{argmax in F} for T=0 (SPEC v1.3 certificate)."""
    z = logits.double()
    fm = fact_mask[:, None, :].to(z.device)
    if temperature == 0:
        return torch.gather(fm.expand_as(z), -1, z.argmax(-1, keepdim=True)).squeeze(-1).double()
    return (F.softmax(z / temperature, -1) * fm).sum(-1)


@torch.no_grad()
def certificate(model, prompts, prompt_am, fact_mask, config, gen_length, generator=None) -> Dict:
    """B_hat_pi (SPEC v1.3) on fresh rollouts: mean over rollouts of sum_{n<=tau} sum_{i in E_n} pi_T(F|s_n,i)
    (x k_n/|E_n| for random remasking), with MC standard error, next to the observed leak rate."""
    steps, block, rm, T = config
    b = prompts.shape[0]
    bound = torch.zeros(b, dtype=torch.float64)
    alive = torch.ones(b, dtype=torch.bool)

    def on_step(n, x, logits, cand, transfer, x0):
        term = (tempered_fact_prob(logits, fact_mask, T) * cand).sum(1).cpu()
        if rm == "random":
            term = term * transfer.sum(1).double().cpu() / cand.sum(1).clamp_min(1).double().cpu()
        bound.add_(torch.where(alive, term, torch.zeros_like(term)))
        hit = (transfer & torch.gather(fact_mask.to(x0.device), 1, x0)).any(1).cpu()
        alive.mul_(~hit)
    x, info = generate_traced(model, prompts, prompt_am, steps=steps, gen_length=gen_length, block_length=block,
                              temperature=T, remasking=rm, generator=generator, fact_mask=fact_mask, on_step=on_step)
    leak = info["first_commit"] >= 0
    return {"B_hat": float(bound.mean()), "B_hat_se": float(bound.std() / math.sqrt(b)) if b > 1 else float("nan"),
            "leak_rate": float(leak.float().mean())}


@torch.no_grad()
def fixed_point_report(model, prompts, prompt_am, fact_mask, config: Tuple[int, int, str, float], gen_length: int,
                       eps: float, generator=None) -> Dict:
    """SPEC v1.3 helper on theta0 rollouts (adapter disabled):
      n_hat      = mean # committed positions whose reference p0(F) at commit time exceeds eps (target clamps them
                   at eps)  -> implied fixed-point leak 1 - (1-eps)^n_hat  (upper version with n_hat = gen_length);
      n_hat_elig = mean sum_n |E_n| (x k_n/|E_n| for random remasking), the certificate's position count
                   -> certificate floor eps * n_hat_elig (B_hat cannot certify below this at the fixed point)."""
    steps, block, rm, T = config
    b = prompts.shape[0]
    n_act = torch.zeros(b)
    n_elig = torch.zeros(b, dtype=torch.float64)

    def on_step(n, x, logits, cand, transfer, x0):
        pF = (F.softmax(logits.float(), -1) * fact_mask[:, None, :].to(logits.device)).sum(-1)
        n_act.add_(((pF > eps) & transfer).sum(1).float().cpu())
        e = cand.sum(1).double().cpu()
        if rm == "random":
            e = transfer.sum(1).double().cpu()            # |E_n| * k_n/|E_n|
        n_elig.add_(e)
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with ctx:
        generate_traced(model, prompts, prompt_am, steps=steps, gen_length=gen_length, block_length=block,
                        temperature=T, remasking=rm, generator=generator, on_step=on_step)
    n_hat, n_hat_elig = float(n_act.mean()), float(n_elig.mean())
    return {"eps": eps, "n_hat": n_hat, "implied_leak": implied_fixed_point_leak(eps, n_hat),
            "implied_leak_upper": implied_fixed_point_leak(eps, gen_length), "n_hat_elig": n_hat_elig,
            "certificate_floor": eps * n_hat_elig}


@torch.no_grad()
def f_closure_audit(model, tokenizer, prompts, prompt_am, fact_mask: torch.Tensor,
                    surface_forms: Sequence[Sequence[str]], config, gen_length: int, top_k: int = 5,
                    min_len: int = 2, stoplist=DEFAULT_STOPLIST, generator=None) -> Dict:
    """SPEC v1.3 F-closure audit: at theta0 rollout decision states (first F commit), take the top-k tokens of
    p0 restricted to F^c at the committing position(s); add every token whose decoded (stripped, case-folded) string
    is a prefix or substring of any surface form of that fact (len >= min_len, alphanumeric, not a stop word).
    Returns the augmented fact_mask and a log of added ids per row (log it in the run record)."""
    steps, block, rm, T = config
    new_mask = fact_mask.clone()
    added: Dict[int, List[Tuple[int, str]]] = {}
    forms = [[f.lower().strip() for f in fs] for fs in surface_forms]

    def on_step(n, x, logits, cand, transfer, x0):
        hit_pos = transfer & torch.gather(fact_mask.to(x0.device), 1, x0)
        for r, i in zip(*hit_pos.nonzero(as_tuple=True)):
            r, i = int(r), int(i)
            z = logits[r, i].float().clone()
            z[fact_mask[r].to(z.device)] = -torch.inf
            for t in torch.topk(z, top_k).indices.tolist():
                sstr = tokenizer.decode([t]).strip().lower()
                if len(sstr) < min_len or not any(ch.isalnum() for ch in sstr) or sstr in stoplist:
                    continue
                if any(f.startswith(sstr) or sstr in f for f in forms[r]) and not bool(new_mask[r, t]):
                    new_mask[r, t] = True
                    added.setdefault(r, []).append((t, sstr))
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with ctx:
        generate_traced(model, prompts, prompt_am, steps=steps, gen_length=gen_length, block_length=block,
                        temperature=T, remasking=rm, generator=generator, fact_mask=fact_mask, on_step=on_step)
    return {"fact_mask": new_mask, "added": added, "n_added": sum(len(v) for v in added.values())}


def load_llada(cfg: FXConfig, device: str = "cuda"):
    """Real-model loader (UNTESTED here). Prints the linear module names so the LoRA regex can be confirmed."""
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True,
                                      torch_dtype=getattr(torch, cfg.dtype)).to(device)
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    assert tok.pad_token_id != MASK_ID
    names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)})
    print("[llada_fxmix] nn.Linear leaf names:", names)
    vocab = model.config.vocab_size if hasattr(model.config, "vocab_size") else None
    print("[llada_fxmix] vocab_size:", vocab, "(expected 126464; MASK 126336, EOS 126081, EoT 126348)")
    return model, tok
