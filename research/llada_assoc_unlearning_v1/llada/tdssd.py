"""TD-SSD (twin-differenced stranger-swap distillation) and its control SSD on LLaDA-8B with LoRA.

EXPERIMENTAL ALGORITHM (combination/adaptation; benefit and novelty unestablished). Design: agents/O_PROPOSAL.md +
agents/O_FINAL.md; selection: DECISION_LOG.md #7; method summary: METHOD.md.

Per forget fact (question q_A about subject s_A, answer a_A):
  * stranger nu_A: a name with the same token count, substituted for s_A in q_A -> q_nu; sigma = token swap s_A<->nu_A.
  * y_nu = native greedy generation of the FROZEN MEMORISED checkpoint p0 (adapter disabled) on q_nu (once, offline);
    ybar = sigma(y_nu) (stranger name tokens mapped back to the subject's).
  * each step: t~U(0,1), p_m=(1-eps)t+eps, the same iid mask M over the answer region of both rows
        x_A  = [q_A  || mask_M(ybar)]      (student, adapter on)
        x_nu = [q_nu || mask_M(y_nu)]      (twin; student with adapter on, and teacher p0 with adapter off, no grad)
    T_j = softmax(z0(x_nu)_j), target t_j(v) = T_j(sigma(v)).
  * L_F = W * sum_{j in M} KL(t_j || softmax(zmix_j)),  zmix = z_A - z_nu + sg(z_nu)   (arm "tdssd")
                                                         zmix = z_A                    (arm "ssd", control)
    L_T = W * sum_{j in M} KL(T_j || softmax(z_nu_j))   (twin anchor; both arms)
    L_R = W * sum_{j in M_R} KL(R_j || softmax(z_theta(x_R)_j)), R_j = softmax(z0(x_R)_j)  (retain; both arms)
    W = 1/(p_m * |answer region|) ("sft", LLaDA SFT weighting) or 1/|M| ("mean"); L = L_F + lam_T L_T + lam_R L_R.
The loss VALUE of L_F is identical in both arms; only the gradient differs: dL_F/dtheta = (J_A - J_nu)^T e (tdssd)
vs J_A^T e (ssd), e = softmax(z_A) - t.

Heavy dependencies (transformers, peft) are imported lazily: the module imports and is unit-tested on CPU with a tiny
stand-in model (tests/test_tdssd.py). REAL-MODEL EXECUTION IS UNTESTED (Cloud must not load LLaDA weights).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import random
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# LLaDA-8B special ids (pinned LLaDA repo @ 9182493: generate.py / GUIDELINES.md). Verified at run time where possible.
MASK_ID, EOS_ID, EOT_ID = 126336, 126081, 126348
# LoRA targets: UNVERIFIED against the HF checkpoint (names from the dllm vendored modeling_llada.py @ ca17675).
# Block-anchored so the LM head (also named ff_out in that copy) is never wrapped; check_lora_targets() asserts it.
LORA_TARGET_REGEX = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"

FIRST_NAMES = ["James", "Mary", "Robert", "Linda", "Michael", "Susan", "David", "Karen", "Daniel", "Nancy", "Paul",
               "Laura", "Mark", "Sarah", "Kevin", "Helen", "Brian", "Anna", "George", "Emma", "Peter", "Julia",
               "Thomas", "Grace", "Henry", "Alice", "Frank", "Ruth", "Simon", "Clara"]
LAST_NAMES = ["Smith", "Johnson", "Brown", "Miller", "Wilson", "Moore", "Taylor", "Clark", "Lewis", "Walker", "Hall",
              "Young", "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter", "Turner", "Parker",
              "Evans", "Morris", "Cook", "Bell", "Murphy", "Bailey", "Cooper", "Reed"]


# ============================================================================================ config
@dataclasses.dataclass
class Config:
    model: str = "GSAI-ML/LLaDA-8B-Instruct"      # or a local path of the MEMORISED checkpoint (merged)
    mem_adapter: Optional[str] = None              # optional: PEFT adapter holding the memorisation; merged before use
    arm: str = "tdssd"                             # "tdssd" (candidate) | "ssd" (control)
    dtype: str = "bfloat16"
    gen_length: int = 32                           # answer region length (training states and y_nu generation)
    weighting: str = "sft"                         # "sft": 1/(p_m |a|)  |  "mean": 1/|M|
    eps: float = 1e-3
    lam_T: float = 1.0
    lam_R: float = 1.0
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_target_regex: str = LORA_TARGET_REGEX
    lr: float = 1e-4
    warmup_steps: int = 10
    steps: int = 200
    checkpoints: Tuple[int, ...] = (100, 200)
    grad_clip: float = 1.0
    retain_batch: int = 8
    seed: int = 0
    stranger_seed: int = 1234
    log_every: int = 5
    mask_id: int = MASK_ID
    eos_id: int = EOS_ID


def load_config(path: str, **overrides) -> Config:
    d = json.load(open(path))
    d.update({k: v for k, v in overrides.items() if v is not None})
    if "checkpoints" in d:
        d["checkpoints"] = tuple(int(x) for x in d["checkpoints"])
    return Config(**d)


# ============================================================================================ data adapter
def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def chat_prompt_ids(tokenizer, question: str) -> List[int]:
    """Instruct prompt as upstream generate.py: chat template + generation prompt, no extra special tokens."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                         tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def answer_ids(tokenizer, answer: str, L: int, eos_id: int) -> List[int]:
    """GUIDELINES SFT answer region: answer tokens + EOS padding to L (truncated to L-1 tokens + EOS)."""
    a = tokenizer(answer, add_special_tokens=False)["input_ids"][: L - 1]
    return a + [eos_id] * (L - len(a))


def _swap_pairs(a_ids: Sequence[int], b_ids: Sequence[int]) -> Optional[List[Tuple[int, int]]]:
    """Positions where two equal-length token lists differ -> [(a_tok, b_tok)]; None if lengths differ."""
    if len(a_ids) != len(b_ids):
        return None
    return [(x, y) for x, y in zip(a_ids, b_ids) if x != y]


def _consistent_bijection(pairs: Sequence[Tuple[int, int]]) -> Optional[Dict[int, int]]:
    """Involutive swap map {a:b, b:a}; None if any token is mapped inconsistently or sits on both sides."""
    m: Dict[int, int] = {}
    for a, b in pairs:
        if m.get(a, b) != b or m.get(b, a) != a:
            return None
        m[a], m[b] = b, a
    A, B = {a for a, _ in pairs}, {b for _, b in pairs}
    if A & B:
        return None
    return m


def find_stranger(tokenizer, item: Dict, forbidden_tokens: set, rng: random.Random,
                  names: Optional[Sequence[str]] = None) -> Optional[Dict]:
    """Pick a stranger name for a forget item: substituting it for the subject in the question must give a prompt of
    the same token length whose differing tokens form a consistent involutive swap, also for the bare and the
    space-prefixed name (answer-side tokenisations); stranger tokens must not occur in `forbidden_tokens`."""
    subj = item["subject"]
    q = item["question"]
    assert subj in q, f"subject {subj!r} not found verbatim in question {q!r}"
    pA = chat_prompt_ids(tokenizer, q)
    cands = list(names) if names is not None else [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(cands)
    for nu in cands:
        if nu == subj or nu in q:
            continue
        pN = chat_prompt_ids(tokenizer, q.replace(subj, nu))
        pairs = _swap_pairs(pA, pN)
        if not pairs:
            continue
        for variant in (subj, " " + subj):
            vp = _swap_pairs(tokenizer(variant, add_special_tokens=False)["input_ids"],
                             tokenizer(variant.replace(subj, nu), add_special_tokens=False)["input_ids"])
            if vp is None:
                pairs = None
                break
            pairs = pairs + vp
        if pairs is None:
            continue
        m = _consistent_bijection(pairs)
        if m is None:
            continue
        nu_toks = {b for _, b in pairs}
        if nu_toks & forbidden_tokens:
            continue
        return {"stranger": nu, "q_nu": q.replace(subj, nu), "pA": pA, "pN": pN, "swap": m,
                "subj_tokens": sorted({a for a, _ in pairs}), "nu_tokens": sorted(nu_toks)}
    return None


def swap_perm(swap: Dict[int, int], V: int, device=None) -> torch.Tensor:
    """sigma as a vocabulary permutation (an involution): perm[v] = sigma(v)."""
    perm = torch.arange(V, device=device)
    for a, b in swap.items():
        perm[a] = b
    return perm


def split_retain(retain: List[Dict], forget: List[Dict]) -> List[Dict]:
    """Mark each retain item: shares_answer_word (a content word >= 3 chars shared with any forget answer) and
    in_retain_loss. If the data does not set "in_retain_loss", every second shared-answer item (in file order) and every
    second non-shared item are held out of L_R (predeclared, O_PROPOSAL §5)."""
    def words(s):
        return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3}
    fw = set().union(*[words(f.get("answer_key", f["answer"])) for f in forget]) if forget else set()
    k = {True: 0, False: 0}
    out = []
    for r in retain:
        r = dict(r)
        sh = bool(words(r.get("answer_key", r["answer"])) & fw)
        r["shares_answer_word"] = sh
        if "in_retain_loss" not in r:
            r["in_retain_loss"] = (k[sh] % 2 == 0)
            k[sh] += 1
        out.append(r)
    return out


# ============================================================================================ native sampler (port)
@torch.no_grad()
def generate_port(model_fn: Callable, prompt: torch.Tensor, steps: int, gen_length: int, block_length: int,
                  mask_id: int = MASK_ID) -> torch.Tensor:
    """Greedy (temperature 0, low_confidence) port of upstream `generate` (LLaDA @ 9182493), no CFG, no attention
    mask. Used for y_nu when the official generate is not importable; agreement is a native gate (HYPERPLANE_PLAN)."""
    b, plen = prompt.shape
    x = torch.full((b, plen + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :plen] = prompt
    nb = gen_length // block_length
    spb = steps // nb
    for k in range(nb):
        bs, be = plen + k * block_length, plen + (k + 1) * block_length
        mnum = (x[:, bs:be] == mask_id).sum(1, keepdim=True)
        num = torch.zeros(b, spb, dtype=torch.int64, device=x.device) + mnum // spb
        for i in range(b):
            num[i, : int(mnum[i] % spb)] += 1
        for i in range(spb):
            mi = x == mask_id
            logits = model_fn(x)
            x0 = logits.argmax(-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            conf[:, be:] = -math.inf
            x0 = torch.where(mi, x0, x)
            conf = torch.where(mi, conf, torch.full_like(conf, -math.inf))
            tr = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(b):
                _, sel = torch.topk(conf[j], k=int(num[j, i]))
                tr[j, sel] = True
            x[tr] = x0[tr]
    return x


# ============================================================================================ model helpers
def logits_of(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    return out.logits if hasattr(out, "logits") else out


@contextlib.contextmanager
def adapter_off(model):
    """Frozen reference p0 = the memorised checkpoint with the unlearning adapter disabled (PEFT disable_adapter)."""
    with model.disable_adapter():
        yield


def check_lora_targets(model, expected_per_block: int = 7) -> Dict[str, int]:
    wrapped: Dict[str, set] = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") or (hasattr(mod, "A") and hasattr(mod, "B") and hasattr(mod, "base")):
            mb = re.search(r"blocks\.(\d+)\.", name)
            wrapped.setdefault(mb.group(1) if mb else "NON-BLOCK", set()).add(name.split(".")[-1])
    assert wrapped, "no LoRA-wrapped modules found"
    assert "NON-BLOCK" not in wrapped, f"LoRA wrapped a non-block module: {wrapped.get('NON-BLOCK')}"
    for blk, ns in wrapped.items():
        assert len(ns) == expected_per_block, f"block {blk}: wrapped {sorted(ns)}"
    return {k: len(v) for k, v in wrapped.items()}


def trainable(model) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ============================================================================================ data preparation
def prepare(tokenizer, model, forget: List[Dict], retain: List[Dict], cfg: Config, gen_fn: Optional[Callable] = None,
            device="cpu") -> Dict:
    """Builds strangers, offline twin answers y_nu (p0, adapter off, native greedy generate) and padded tensors.
    gen_fn(prompt_ids (1,P)) -> (1, P+L) token ids; default = generate_port with steps = block = gen_length."""
    L = cfg.gen_length
    rng = random.Random(cfg.stranger_seed)
    all_text = " ".join([f["question"] + " " + f["answer"] for f in forget] + [r["question"] + " " + r["answer"]
                                                                               for r in retain])
    forbidden = set(tokenizer(all_text, add_special_tokens=False)["input_ids"])
    used: set = set()
    facts = []
    for f in forget:
        st = find_stranger(tokenizer, f, forbidden | used, rng)
        if st is None:
            raise RuntimeError(f"no admissible stranger name for forget item {f.get('id')}: {f['subject']!r}")
        used |= set(st["nu_tokens"])
        facts.append({**f, **st})
    if gen_fn is None:
        def gen_fn(p):
            with adapter_off(model):
                return generate_port(lambda x: logits_of(model, x), p, steps=L, gen_length=L, block_length=L,
                                     mask_id=cfg.mask_id)
    for fa in facts:
        out = gen_fn(torch.tensor([fa["pN"]], dtype=torch.long, device=device))
        y_nu = out[0, len(fa["pN"]):].tolist()
        assert len(y_nu) == L
        fa["y_nu"] = y_nu
        fa["ybar"] = [fa["swap"].get(t, t) for t in y_nu]
        fa["y_nu_text"] = tokenizer.decode([t for t in y_nu if t != cfg.eos_id])
        fa["n_nu_tokens_in_ynu"] = int(sum(t in set(fa["nu_tokens"]) for t in y_nu))      # restated stranger name
        fa["n_subj_tokens_in_ynu"] = int(sum(t in set(fa["subj_tokens"]) for t in y_nu))  # (a3) collision count
    # forget tensors: right-pad every row with EOS to a common length (GUIDELINES SFT layout; no attention mask)
    plen = [len(fa["pA"]) for fa in facts]
    Ltot = max(plen) + L
    A_rows, N_rows = [], []
    for fa in facts:
        A_rows.append(fa["pA"] + fa["ybar"] + [cfg.eos_id] * (Ltot - len(fa["pA"]) - L))
        N_rows.append(fa["pN"] + fa["y_nu"] + [cfg.eos_id] * (Ltot - len(fa["pN"]) - L))
    rtrain = [r for r in retain if r["in_retain_loss"]]
    R_rows, R_plen = [], []
    if rtrain:
        rp = [chat_prompt_ids(tokenizer, r["question"]) for r in rtrain]
        LR = max(len(p) for p in rp) + L
        for p, r in zip(rp, rtrain):
            R_rows.append(p + answer_ids(tokenizer, r["answer"], L, cfg.eos_id) + [cfg.eos_id] * (LR - len(p) - L))
            R_plen.append(len(p))
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return {"facts": facts, "xA": t(A_rows), "xN": t(N_rows), "plen": t(plen),
            "xR": t(R_rows) if R_rows else None, "R_plen": t(R_plen) if R_rows else None,
            "swaps": [fa["swap"] for fa in facts]}


# ============================================================================================ losses
def sample_answer_mask(b: int, l: int, plen: torch.Tensor, eps: float, generator=None, device="cpu"):
    """LLaDA SFT forward process on the answer region [plen_i, l): t~U(0,1), p_m=(1-eps)t+eps, iid mask; prompt never
    masked. Returns (mask (b,l) bool, p_m (b,), region_len (b,))."""
    t = torch.rand(b, generator=generator).to(device)
    pm = (1 - eps) * t + eps
    u = torch.rand((b, l), generator=generator).to(device)
    pos = torch.arange(l, device=device)[None]
    region = pos >= plen[:, None]
    mask = (u < pm[:, None]) & region
    return mask, pm, region.sum(1)


def kl_rows(log_target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax(logits)) per row in fp32; target given in log space; 0*log0 := 0."""
    lt = log_target.float()
    tgt = lt.exp()
    logp = F.log_softmax(logits.float(), -1)
    return torch.where(tgt > 0, tgt * (lt - logp), torch.zeros_like(tgt)).sum(-1)


def _weights(mask: torch.Tensor, pm: torch.Tensor, region: torch.Tensor, weighting: str) -> torch.Tensor:
    """Per-position weight W (b,l) so that L = sum_rows sum_j W * KL_j / b."""
    if weighting == "sft":
        w = 1.0 / (pm * region.float())
        return mask.float() * w[:, None]
    if weighting == "mean":
        cnt = mask.sum(1).clamp(min=1).float()
        return mask.float() / cnt[:, None]
    raise ValueError(weighting)


def forget_losses(model, xA: torch.Tensor, xN: torch.Tensor, plen: torch.Tensor, perms: torch.Tensor, cfg: Config,
                  generator=None) -> Dict:
    """L_F (arm-dependent gradient) and L_T. xA/xN: (B,l) clean rows; perms: (B,V) sigma per fact."""
    B, l = xA.shape
    dev = xA.device
    mask, pm, region = sample_answer_mask(B, l, plen, cfg.eps, generator, dev)
    nA = torch.where(mask, torch.full_like(xA, cfg.mask_id), xA)
    nN = torch.where(mask, torch.full_like(xN, cfg.mask_id), xN)
    rows, cols = mask.nonzero(as_tuple=True)
    W = _weights(mask, pm, region, cfg.weighting)[rows, cols]
    with torch.no_grad(), adapter_off(model):
        z0N = logits_of(model, nN)[rows, cols].float()
    logT = F.log_softmax(z0N, -1)                                    # teacher on the twin, detached
    log_t = torch.gather(logT, -1, perms[rows])                      # t_j(v) = T_j(sigma(v))
    z = logits_of(model, torch.cat([nA, nN], 0))                     # one student forward, adapter on
    zA, zN = z[:B][rows, cols].float(), z[B:][rows, cols].float()
    zmix = zA - zN + zN.detach() if cfg.arm == "tdssd" else zA
    if cfg.arm not in ("tdssd", "ssd"):
        raise ValueError(cfg.arm)
    klF = kl_rows(log_t, zmix)
    klT = kl_rows(logT, zN)
    LF = (W * klF).sum() / B
    LT = (W * klT).sum() / B
    with torch.no_grad():
        D_F = float(klF.mean()) if klF.numel() else 0.0
    return {"L_F": LF, "L_T": LT, "D_F_masked_mean": D_F, "n_pos": int(rows.numel())}


def retain_loss(model, xR: torch.Tensor, R_plen: torch.Tensor, cfg: Config, generator=None) -> Dict:
    B, l = xR.shape
    mask, pm, region = sample_answer_mask(B, l, R_plen, cfg.eps, generator, xR.device)
    nR = torch.where(mask, torch.full_like(xR, cfg.mask_id), xR)
    rows, cols = mask.nonzero(as_tuple=True)
    W = _weights(mask, pm, region, cfg.weighting)[rows, cols]
    with torch.no_grad(), adapter_off(model):
        logR = F.log_softmax(logits_of(model, nR)[rows, cols].float(), -1)
    z = logits_of(model, nR)[rows, cols].float()
    return {"L_R": (W * kl_rows(logR, z)).sum() / B}


def total_loss(model, data: Dict, perms: torch.Tensor, cfg: Config, generator=None, retain_idx=None) -> Dict:
    out = forget_losses(model, data["xA"], data["xN"], data["plen"], perms, cfg, generator)
    loss = out["L_F"] + cfg.lam_T * out["L_T"]
    if data.get("xR") is not None and cfg.lam_R > 0:
        idx = retain_idx if retain_idx is not None else torch.arange(data["xR"].shape[0])
        r = retain_loss(model, data["xR"][idx], data["R_plen"][idx], cfg, generator)
        out.update(r)
        loss = loss + cfg.lam_R * r["L_R"]
    out["loss"] = loss
    return out


# ============================================================================================ training
def make_optimizer(model, cfg: Config):
    opt = torch.optim.AdamW(trainable(model), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))
    return opt, sched


def train(model, data: Dict, cfg: Config, save_fn: Optional[Callable[[int], None]] = None,
          log: Callable = print, V: Optional[int] = None) -> List[Dict]:
    g = torch.Generator().manual_seed(cfg.seed)
    dev = data["xA"].device
    V = V or int(logits_of(model, data["xA"][:1]).shape[-1])
    perms = torch.stack([swap_perm(s, V, dev) for s in data["swaps"]])
    opt, sched = make_optimizer(model, cfg)
    hist = []
    nR = 0 if data.get("xR") is None else data["xR"].shape[0]
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        ridx = torch.randperm(nR, generator=g)[: cfg.retain_batch] if nR else None
        out = total_loss(model, data, perms, cfg, g, ridx)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable(model), cfg.grad_clip)
        opt.step()
        sched.step()
        rec = {"step": step, "loss": float(out["loss"].detach()), "L_F": float(out["L_F"].detach()),
               "L_T": float(out["L_T"].detach()), "L_R": float(out["L_R"].detach()) if "L_R" in out else None,
               "D_F_masked_mean": out["D_F_masked_mean"], "grad_norm": float(gn), "elapsed_s": round(time.time() - t0, 2)}
        if torch.cuda.is_available():
            rec["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        hist.append(rec)
        if step % cfg.log_every == 0 or step == 1:
            log(rec)
        if save_fn is not None and step in cfg.checkpoints:
            save_fn(step)
    return hist
