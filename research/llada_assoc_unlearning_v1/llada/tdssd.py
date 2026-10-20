"""TD-SSD (twin-differenced stranger-swap distillation) and its control SSD on LLaDA-8B with LoRA.

EXPERIMENTAL ALGORITHM — a combination/adaptation, benefit and novelty unestablished. Frozen spec: agents/O_FINAL.md §2
(development revision #1, DECISION_LOG #8). Method summary: METHOD.md.

Per forget fact (question q_A about subject s_A):
  * teacher stranger nu_A (fixed): same-token-length name substituted for s_A; sigma_nu = token swap s_A <-> nu_A.
  * y_nu = native greedy generation of the FROZEN MEMORISED checkpoint p0 (adapter disabled) on q_nu, once, offline;
    ybar = sigma_nu(y_nu) (restated stranger name mapped back to the subject).
  * control names C_A: up to 16 same-token-length names; sigma_k = token swap s_A <-> c_k.
  * each step: t~U(0,1), p_m=(1-eps)t+eps, iid mask M over the answer region (EOS padding included; one position if M
    would be empty), prompt never masked;
        x_A = [q_A || mask_M(ybar)]          student (adapter on)
        x_T = [q_nu || mask_M(y_nu)]         teacher p0 (adapter off, no grad):  t_j = softmax(z0(x_T)_j)[sigma_nu]
        x_c = sigma_k(x_A), k ~ U(C_A)       control canvas, redrawn every step, same M
  * arm "tdssd" (X):  zmix = z_A - z_c + sg(z_c);  L = L_F + lam_R L_R                    (lam_T = 0)
    arm "ssd"   (Z):  zmix = z_A;                  L = L_F + lam_T L_T + lam_R L_R
        L_F = mean_rows mean_{j in M} KL(t_j || softmax(zmix_j))
        L_T = mean_rows mean_{j in M} KL(p0(.|x_c)_j || softmax(z_c_j))      (rotating-anchor control)
        L_R = mean_rows mean_{j in M_R} KL(p0(.|x_R)_j || softmax(z(x_R)_j))  (retain, both arms)
The VALUE of L_F is the same functional in both arms; in X its gradient is (J_A - J_c)^T e, in Z J_A^T e,
e_j = (softmax(z_A)_j - t_j)/|M|.

transformers/peft are imported lazily (train_tdssd.py); this module is unit-tested on CPU with a tiny stand-in model
(tests/test_tdssd.py). REAL-MODEL EXECUTION IS UNTESTED (the authoring host must not load LLaDA weights).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import random
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# LLaDA-8B special ids (pinned LLaDA repo @ 9182493: generate.py / GUIDELINES.md). Checked at run time (native gates).
MASK_ID, EOS_ID, EOT_ID = 126336, 126081, 126348
# LoRA targets: UNVERIFIED against the HF checkpoint (names from the dllm vendored modeling_llada.py @ ca17675).
# Block-anchored so the LM head (also named ff_out in that copy) is never wrapped; check_lora_targets() asserts it.
LORA_TARGET_REGEX = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"

FIRST_NAMES = ["James", "Mary", "Robert", "Linda", "Michael", "Susan", "David", "Karen", "Daniel", "Nancy", "Paul",
               "Laura", "Mark", "Sarah", "Kevin", "Helen", "Brian", "Anna", "George", "Emma", "Peter", "Julia",
               "Thomas", "Grace", "Henry", "Frank", "Ruth", "Simon", "Clara", "Oliver"]
LAST_NAMES = ["Smith", "Johnson", "Brown", "Miller", "Wilson", "Moore", "Taylor", "Clark", "Lewis", "Walker", "Hall",
              "Young", "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter", "Turner", "Parker",
              "Evans", "Morris", "Cook", "Bell", "Murphy", "Bailey", "Cooper", "Reed"]


# ============================================================================================ config
@dataclasses.dataclass
class Config:
    model: str = "GSAI-ML/LLaDA-8B-Instruct"      # or a local path of the MEMORISED checkpoint (merged)
    mem_adapter: Optional[str] = None              # optional PEFT adapter holding the memorisation; merged before use
    arm: str = "tdssd"                             # "tdssd" (candidate X) | "ssd" (control Z)
    dtype: str = "bfloat16"
    gen_length: int = 32                           # answer region length L_a (multiple of 8)
    weighting: str = "mean"                        # frozen: mean over masked positions ("sft" kept for diagnostics)
    eps: float = 1e-3
    lam_T: float = 1.0                             # used by "ssd" only (X has lam_T = 0 by construction)
    lam_R: float = 1.0
    n_controls: int = 16
    n_eval_strangers: int = 8
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_target_regex: str = LORA_TARGET_REGEX
    lr: float = 1e-4
    warmup_steps: int = 10
    steps: int = 150
    checkpoints: Tuple[int, ...] = (50, 150)
    grad_clip: float = 1.0
    retain_batch: int = 8
    seed: int = 0
    name_seed: int = 1234
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
    if len(a_ids) != len(b_ids):
        return None
    return [(x, y) for x, y in zip(a_ids, b_ids) if x != y]


def _involution(pairs: Sequence[Tuple[int, int]]) -> Optional[Dict[int, int]]:
    """Product of transpositions {a:b, b:a}; None if any token maps inconsistently or sits on both sides."""
    m: Dict[int, int] = {}
    for a, b in pairs:
        if m.get(a, b) != b or m.get(b, a) != a:
            return None
        m[a], m[b] = b, a
    if {a for a, _ in pairs} & {b for _, b in pairs}:
        return None
    return m


def accept_name(tokenizer, text: str, subj: str, name: str, forbidden: set) -> Optional[Dict]:
    """Name acceptance (O_FINAL §2): substitute at string level, re-tokenise the chat prompt; equal length, only
    name-span tokens differ, the same holds for the bare and space-prefixed name (answer-side tokenisations), the
    swap is a consistent involution, subject tokens do not occur outside the name spans, and the new name's tokens
    are not in `forbidden`."""
    if name == subj or name in text or subj not in text:
        return None
    pA = chat_prompt_ids(tokenizer, text)
    pN = chat_prompt_ids(tokenizer, text.replace(subj, name))
    pairs = _swap_pairs(pA, pN)
    if not pairs:
        return None
    for variant in (subj, " " + subj):
        vp = _swap_pairs(tokenizer(variant, add_special_tokens=False)["input_ids"],
                         tokenizer(variant.replace(subj, name), add_special_tokens=False)["input_ids"])
        if vp is None:
            return None
        pairs = pairs + vp
    m = _involution(pairs)
    if m is None:
        return None
    subj_toks, name_toks = {a for a, _ in pairs}, {b for _, b in pairs}
    if name_toks & forbidden:
        return None
    n_span = sum(1 for a, b in zip(pA, pN) if a != b)
    if sum(t in subj_toks for t in pA) != n_span:          # a subject token occurs outside the name span
        return None
    return {"name": name, "pA": pA, "pN": pN, "swap": m, "subj_tokens": sorted(subj_toks),
            "name_tokens": sorted(name_toks)}


def apply_swap(ids: torch.Tensor, swap: Dict[int, int]) -> torch.Tensor:
    """Token-wise sigma on a tensor (MASK and every token outside the swap are unchanged)."""
    out = ids.clone()
    for a, b in swap.items():
        out[ids == a] = b
    return out


def swap_perm(swap: Dict[int, int], V: int, device=None) -> torch.Tensor:
    """sigma as a vocabulary permutation (an involution): perm[v] = sigma(v)."""
    perm = torch.arange(V, device=device)
    for a, b in swap.items():
        perm[a] = b
    return perm


def split_retain(retain: List[Dict], forget: List[Dict]) -> List[Dict]:
    """shares_answer_word: a content word (>= 3 chars) shared with any forget answer_key/answer. If the data does not
    set "in_retain_loss", every second item within each stratum (shared / not shared, file order) is held out of L_R
    (8/8 stratified split for 16 retain facts; predeclared, O_FINAL §2)."""
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
    mask. Fallback when the official generate is not importable; agreement is a native gate."""
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
def name_candidates(seed: int) -> List[str]:
    c = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    random.Random(seed).shuffle(c)
    return c


def prepare(tokenizer, model, forget: List[Dict], retain: List[Dict], cfg: Config, gen_fn: Optional[Callable] = None,
            device="cpu") -> Dict:
    """Name pools (teacher nu_A per fact, controls C_A per fact, eval strangers E; disjoint), offline twin answers y_nu
    (p0, adapter off, native greedy generate, steps = block = gen_length) and padded tensors.
    gen_fn(prompt_ids (1,P)) -> (1, P+L) token ids."""
    L = cfg.gen_length
    if gen_fn is None:
        def gen_fn(p):
            with adapter_off(model):
                return generate_port(lambda x: logits_of(model, x), p, steps=L, gen_length=L, block_length=L,
                                     mask_id=cfg.mask_id)
    all_text = " ".join([f["question"] + " " + f["answer"] + " " + f.get("paraphrase", "") for f in forget] +
                        [r["question"] + " " + r["answer"] + " " + r.get("paraphrase", "") for r in retain])
    forbidden = set(tokenizer(all_text, add_special_tokens=False)["input_ids"])
    cands = name_candidates(cfg.name_seed)
    taken: set = set()
    facts, rejected = [], []
    for f in forget:                                                   # teacher names
        fa = None
        for nm in cands:
            if nm in taken:
                continue
            st = accept_name(tokenizer, f["question"], f["subject"], nm, forbidden)
            if st is None:
                continue
            out = gen_fn(torch.tensor([st["pN"]], dtype=torch.long, device=device))
            y_nu = out[0, len(st["pN"]):].tolist()
            assert len(y_nu) == L
            if any(t in set(st["subj_tokens"]) for t in y_nu):         # (a3) subject token in the twin answer
                rejected.append({"id": f.get("id"), "name": nm, "reason": "subject token in y_nu"})
                continue
            fa = {**f, "teacher": st, "y_nu": y_nu, "ybar": [st["swap"].get(t, t) for t in y_nu],
                  "y_nu_text": tokenizer.decode([t for t in y_nu if t != cfg.eos_id]),
                  "n_name_tokens_in_ynu": int(sum(t in set(st["name_tokens"]) for t in y_nu))}
            taken.add(nm)
            break
        if fa is None:
            raise RuntimeError(f"no admissible teacher stranger for forget item {f.get('id')}: {f['subject']!r}")
        facts.append(fa)
    for fa in facts:                                                   # control names (rotating pool)
        ctrl = []
        ybar_set = set(fa["ybar"])
        for nm in cands:
            if len(ctrl) >= cfg.n_controls:
                break
            if nm in taken:
                continue
            st = accept_name(tokenizer, fa["question"], fa["subject"], nm, forbidden | ybar_set)
            if st is not None:
                ctrl.append(st)
        if not ctrl:
            raise RuntimeError(f"no admissible control name for forget item {fa.get('id')}")
        fa["controls"] = ctrl
        taken |= {c["name"] for c in ctrl}
    eval_names = [nm for nm in cands if nm not in taken][: cfg.n_eval_strangers]
    plen = [len(fa["teacher"]["pA"]) for fa in facts]
    Ltot = max(plen) + L
    A_rows, T_rows = [], []
    for fa in facts:
        pad = [cfg.eos_id] * (Ltot - len(fa["teacher"]["pA"]) - L)    # GUIDELINES layout: right EOS padding
        A_rows.append(fa["teacher"]["pA"] + fa["ybar"] + pad)
        T_rows.append(fa["teacher"]["pN"] + fa["y_nu"] + pad)
    rtrain = [r for r in retain if r["in_retain_loss"]]
    R_rows, R_plen = [], []
    if rtrain:
        rp = [chat_prompt_ids(tokenizer, r["question"]) for r in rtrain]
        LR = max(len(p) for p in rp) + L
        for p, r in zip(rp, rtrain):
            R_rows.append(p + answer_ids(tokenizer, r["answer"], L, cfg.eos_id) + [cfg.eos_id] * (LR - len(p) - L))
            R_plen.append(len(p))
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return {"facts": facts, "xA": t(A_rows), "xT": t(T_rows), "plen": t(plen),
            "teacher_swaps": [fa["teacher"]["swap"] for fa in facts],
            "control_swaps": [[c["swap"] for c in fa["controls"]] for fa in facts],
            "eval_names": eval_names, "rejected_names": rejected,
            "xR": t(R_rows) if R_rows else None, "R_plen": t(R_plen) if R_rows else None}


def names_record(data: Dict) -> Dict:
    return {"facts": [{"id": fa.get("id"), "subject": fa["subject"], "teacher_name": fa["teacher"]["name"],
                       "y_nu_text": fa["y_nu_text"], "n_name_tokens_in_ynu": fa["n_name_tokens_in_ynu"],
                       "control_names": [c["name"] for c in fa["controls"]]} for fa in data["facts"]],
            "eval_names": data["eval_names"], "rejected_names": data["rejected_names"]}


# ============================================================================================ losses
def sample_answer_mask(b: int, l: int, plen: torch.Tensor, eps: float, generator=None, device="cpu"):
    """LLaDA SFT forward process on the answer region [plen_i, l): t~U(0,1), p_m=(1-eps)t+eps, iid mask; the prompt is
    never masked; a row whose mask would be empty gets one uniformly drawn answer position. Returns
    (mask (b,l) bool, p_m (b,), region_len (b,))."""
    t = torch.rand(b, generator=generator).to(device)
    pm = (1 - eps) * t + eps
    u = torch.rand((b, l), generator=generator).to(device)
    pos = torch.arange(l, device=device)[None]
    region = pos >= plen[:, None]
    mask = (u < pm[:, None]) & region
    fill = torch.rand(b, generator=generator).to(device)
    return ensure_nonempty(mask, region, plen, fill), pm, region.sum(1)


def ensure_nonempty(mask: torch.Tensor, region: torch.Tensor, plen: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    """A row with no masked answer position gets exactly one, at plen + floor(fill * |region|) (O_FINAL §2)."""
    mask = mask.clone()
    for i in range(mask.shape[0]):
        if not bool(mask[i].any()):
            n = int(region[i].sum())
            mask[i, int(plen[i]) + min(n - 1, int(fill[i] * n))] = True
    return mask


def kl_rows(log_target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax(logits)) per row in fp32; target given in log space; 0*log0 := 0."""
    lt = log_target.float()
    tgt = lt.exp()
    logp = F.log_softmax(logits.float(), -1)
    return torch.where(tgt > 0, tgt * (lt - logp), torch.zeros_like(tgt)).sum(-1)


def _weights(mask: torch.Tensor, pm: torch.Tensor, region: torch.Tensor, weighting: str) -> torch.Tensor:
    """Per-position weight W (b,l) with L = sum_rows sum_j W * KL_j / b."""
    if weighting == "mean":
        return mask.float() / mask.sum(1).clamp(min=1).float()[:, None]
    if weighting == "sft":
        return mask.float() * (1.0 / (pm * region.float()))[:, None]
    raise ValueError(weighting)


def forget_losses(model, data: Dict, perms_T: torch.Tensor, cfg: Config, generator=None,
                  ctrl_idx: Optional[List[int]] = None) -> Dict:
    """L_F (arm-dependent gradient) and, for arm "ssd", the rotating-anchor L_T on the control canvas."""
    if cfg.arm not in ("tdssd", "ssd"):
        raise ValueError(cfg.arm)
    xA, xT, plen = data["xA"], data["xT"], data["plen"]
    B, l = xA.shape
    dev = xA.device
    mask, pm, region = sample_answer_mask(B, l, plen, cfg.eps, generator, dev)
    nA = torch.where(mask, torch.full_like(xA, cfg.mask_id), xA)
    nT = torch.where(mask, torch.full_like(xT, cfg.mask_id), xT)
    if ctrl_idx is None:
        ctrl_idx = [int(torch.randint(len(cs), (1,), generator=generator)) for cs in data["control_swaps"]]
    nC = torch.stack([apply_swap(nA[i], data["control_swaps"][i][k]) for i, k in enumerate(ctrl_idx)])
    rows, cols = mask.nonzero(as_tuple=True)
    W = _weights(mask, pm, region, cfg.weighting)[rows, cols]
    with torch.no_grad(), adapter_off(model):
        log_t = torch.gather(F.log_softmax(logits_of(model, nT)[rows, cols].float(), -1), -1, perms_T[rows])
        log_c0 = F.log_softmax(logits_of(model, nC)[rows, cols].float(), -1) if cfg.arm == "ssd" else None
    z = logits_of(model, torch.cat([nA, nC], 0))                     # one student forward (adapter on), 2B rows
    zA, zC = z[:B][rows, cols].float(), z[B:][rows, cols].float()
    zmix = zA - zC + zC.detach() if cfg.arm == "tdssd" else zA
    klF = kl_rows(log_t, zmix)
    out = {"L_F": (W * klF).sum() / B, "ctrl_idx": ctrl_idx, "n_pos": int(rows.numel()),
           "D_F_state_mean": float(klF.detach().mean())}
    if cfg.arm == "ssd":
        out["L_T"] = (W * kl_rows(log_c0, zC)).sum() / B
    return out


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


def total_loss(model, data: Dict, perms_T: torch.Tensor, cfg: Config, generator=None, retain_idx=None) -> Dict:
    out = forget_losses(model, data, perms_T, cfg, generator)
    loss = out["L_F"]
    if cfg.arm == "ssd":
        loss = loss + cfg.lam_T * out["L_T"]
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
    perms_T = torch.stack([swap_perm(s, V, dev) for s in data["teacher_swaps"]])
    opt, sched = make_optimizer(model, cfg)
    hist = []
    nR = 0 if data.get("xR") is None else data["xR"].shape[0]
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        ridx = torch.randperm(nR, generator=g)[: cfg.retain_batch] if nR else None
        out = total_loss(model, data, perms_T, cfg, g, ridx)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable(model), cfg.grad_clip)
        opt.step()
        sched.step()
        rec = {"step": step, "loss": float(out["loss"].detach()), "L_F": float(out["L_F"].detach()),
               "L_T": float(out["L_T"].detach()) if "L_T" in out else None,
               "L_R": float(out["L_R"].detach()) if "L_R" in out else None,
               "D_F_state_mean": out["D_F_state_mean"], "grad_norm": float(gn), "ctrl_idx": out["ctrl_idx"],
               "elapsed_s": round(time.time() - t0, 2)}
        if torch.cuda.is_available():
            rec["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        hist.append(rec)
        if step % cfg.log_every == 0 or step == 1:
            log(rec)
        if save_fn is not None and step in cfg.checkpoints:
            save_fn(step)
    return hist
