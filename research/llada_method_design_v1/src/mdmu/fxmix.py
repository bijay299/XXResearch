"""FX-Mix reference implementation (SPEC v0.1, section 2) + control losses (SPEC section 4) on the CPU substrate.

Equation labels (assigned here, in order of appearance in SPEC v0.1 section 2; G must re-map if SPEC v1 renumbers):
  (E1) L(theta) = lam_f * E_{(c,y*,F)} E_{s~mu} [ (1/|M(s)|) sum_{i in M(s)} KL(r_eps(.|s,i) || p_theta(.|s,i)) ]
                  + lam_r * L_ret(theta)
  (E2) r_eps(.|s,i) = argmin_{q: q(F) <= eps} KL(q || p0(.|s,i))          (I-projection of the frozen reference)
  (E3) closed form: p0(F) <= eps  =>  r = p0;  else  r(v) = (1-eps) p0(v)/(1-p0(F)) (v not in F),
                    r(f) = eps p0(f)/p0(F) (f in F);  == softmax(z0 - gamma 1_F),
                    gamma = log[p0(F)(1-eps) / (eps (1-p0(F)))];  eps = 0  =>  logits[F] = -inf.
  (E4) TF_ff(c,y*): t~U(0,1); mask each non-fact answer position iid w.p. t; ALWAYS mask j with y*_j in F.
  (E5) ROLL_k(theta_bar, c): states before each step of native `generate` under config pi_k, truncated at the first
       step whose commit contains a token of F (that decision state kept); J states per rollout, decision state
       always kept.  mu = alpha TF_ff + (1-alpha) sum_k w_k ROLL_k, fact-free states only.
Controls (SPEC section 4): NPO (beta=0.2, MDU App. B form), MDU (tau in {0,1}), SC (r_0 on MDU's random TF states).

Conventions: a "state" is a full (prompt || answer) id row in which masked answer positions hold MASK; the prompt is
never masked; M(s) = all masked answer positions (SPEC: "Positions: all masked answer positions"). Fact sets are
passed either as a list of id collections (one per row) or as a boolean (rows, V) mask.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .diffusion import forward_process_sft, generate
from .toyworld import MASK

FactSpec = Union[torch.Tensor, Sequence[Sequence[int]]]


# ============================================================================================ fact-set helpers
def fact_mask_from_sets(fact_sets: Sequence[Sequence[int]], vocab: int) -> torch.Tensor:
    """List of per-row fact-id collections -> bool (rows, V) mask."""
    m = torch.zeros(len(fact_sets), vocab, dtype=torch.bool)
    for r, fs in enumerate(fact_sets):
        if len(fs):
            m[r, torch.as_tensor(list(fs), dtype=torch.long)] = True
    return m


def rows_fact_mask(fact_sets: FactSpec, rows: int, vocab: int) -> torch.Tensor:
    """Bool (rows, V) mask from either a bool mask (returned as is) or a list of per-row id collections."""
    if isinstance(fact_sets, torch.Tensor) and fact_sets.dtype == torch.bool:
        assert fact_sets.shape == (rows, fact_sets.shape[1])
        return fact_sets
    assert len(fact_sets) == rows
    return fact_mask_from_sets(fact_sets, vocab)


def _at_least_fp32(z: torch.Tensor) -> torch.Tensor:
    """SPEC v1.1: the target is computed from (at least) fp32 reference logits (bf16/fp16 are upcast; fp64 kept)."""
    return z.float() if z.dtype in (torch.float16, torch.bfloat16) else z


def _as_mask(fact_ids, ref_logits: torch.Tensor) -> torch.Tensor:
    """Normalise a fact spec to a bool mask broadcastable to ref_logits (..., V)."""
    V = ref_logits.shape[-1]
    if isinstance(fact_ids, torch.Tensor) and fact_ids.dtype == torch.bool:
        assert fact_ids.shape[-1] == V
        return fact_ids
    ids = torch.as_tensor(list(fact_ids) if not isinstance(fact_ids, torch.Tensor) else fact_ids, dtype=torch.long)
    m = torch.zeros(V, dtype=torch.bool)
    if ids.numel():
        m[ids] = True
    return m  # shared F for every row


# ============================================================================================ (E2)/(E3) target
@torch.no_grad()
def exclusion_target(ref_logits: torch.Tensor, fact_ids, eps: float = 1e-3, return_info: bool = False):
    """Closed-form I-projection r_eps (E2)/(E3) of p0 = softmax(ref_logits) onto {q : q(F) <= eps}.

    Returns LOG-probabilities log r (..., V) (entries -inf where r = 0, i.e. on F when eps = 0).
    Everything is computed in log space from logits; nothing is ever divided by 1 - p0(F):
        log r(v) = log(1-eps) + z0(v) - logsumexp_{u not in F} z0(u)     (v not in F)
        log r(f) = log(eps)   + z0(f) - logsumexp_{u in F} z0(u)         (f in F)
    so p0(F) -> 1 (even p0(F) == 1 in float) stays finite. Rows with p0(F) <= eps (incl. F empty) get r = p0.
    """
    assert 0.0 <= eps < 1.0
    z = _at_least_fp32(ref_logits)
    fm = _as_mask(fact_ids, z).to(z.device)
    if fm.shape != z.shape:
        fm = _broadcast_rows(fm, z)
    neg_inf = torch.tensor(-math.inf, dtype=z.dtype)
    lse_all = torch.logsumexp(z, -1, keepdim=True)
    lse_F = torch.logsumexp(torch.where(fm, z, neg_inf), -1, keepdim=True)        # -inf if F empty
    lse_nF = torch.logsumexp(torch.where(fm, neg_inf, z), -1, keepdim=True)
    assert torch.isfinite(lse_nF).all(), "fact set covers the whole vocabulary: target undefined"
    log_p0 = z - lse_all
    log_eps = math.log(eps) if eps > 0 else -math.inf
    log_pF = lse_F - lse_all
    feasible = log_pF <= log_eps                                                   # p0(F) <= eps  =>  r = p0
    log_r_nF = math.log1p(-eps) + z - lse_nF
    if eps > 0:
        log_r_F = log_eps + z - lse_F
    else:
        log_r_F = torch.full_like(z, -math.inf)
    log_r = torch.where(fm, log_r_F, log_r_nF)
    log_r = torch.where(feasible, log_p0, log_r)
    if not return_info:
        return log_r
    gamma = torch.where(feasible, torch.zeros_like(log_pF),
                        (lse_F - lse_nF) + math.log1p(-eps) - (log_eps if eps > 0 else -math.inf))
    return log_r, {"log_p0F": log_pF.squeeze(-1), "gamma": gamma.squeeze(-1), "active": (~feasible).squeeze(-1)}


def _broadcast_rows(fm: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """fm (N, V) with z (N, L, V): insert the position axis."""
    if fm.dim() == 1:
        return fm.expand_as(z)
    while fm.dim() < z.dim():
        fm = fm.unsqueeze(-2)
    return fm.expand_as(z)


def soft_ce_kl(log_r: torch.Tensor, logits: torch.Tensor):
    """Per-row CE(r, softmax(logits)) = -sum r log p, entropy H(r), KL = CE - H (grad of KL == grad of CE).
    r = 0 entries contribute 0 (xlogy convention); log p is finite for finite logits."""
    log_p = F.log_softmax(_at_least_fp32(logits), -1)
    r = log_r.exp().to(log_p.dtype)
    ce = -(r * log_p).sum(-1)
    ent = -torch.special.xlogy(r, r).sum(-1)
    return ce, ent, ce - ent.detach()


# ============================================================================================ (E4) TF_ff states
def fact_free_tf_states(ids: torch.Tensor, prompt_len: int, fact_ids: FactSpec,
                        generator: Optional[torch.Generator] = None, vocab: Optional[int] = None,
                        force_fact_mask: bool = True):
    """(E4): t~U(0,1) per row; every non-fact answer position masked iid w.p. t; positions j with y*_j in F ALWAYS
    masked. Returns (states, masked) with states fact-free by construction (no visible token in F)."""
    b, l = ids.shape
    if vocab is None:
        vocab = int(ids.max()) + 1 if isinstance(fact_ids, torch.Tensor) and fact_ids.dtype == torch.bool \
            else max([int(ids.max())] + [int(f) for fs in fact_ids for f in fs]) + 1
    fm = rows_fact_mask(fact_ids, b, vocab)
    is_fact = torch.gather(fm, 1, ids)
    t = torch.rand(b, generator=generator)
    masked = torch.rand((b, l), generator=generator) < t[:, None]
    if force_fact_mask:                     # (E4); False = T5c ablation (plain random masks, F may be visible)
        masked |= is_fact
    masked[:, :prompt_len] = False
    states = torch.where(masked, torch.full_like(ids, MASK), ids)
    return states, masked


def is_fact_free(states: torch.Tensor, fact_mask: torch.Tensor) -> torch.Tensor:
    """(rows,) bool: no visible (non-MASK) token of the row belongs to its fact set."""
    vis = states != MASK
    hit = torch.gather(fact_mask, 1, states.clamp_max(fact_mask.shape[1] - 1))
    return ~(hit & vis).any(1)


# ============================================================================================ (E5) rollout states
@torch.no_grad()
def collect_rollout_states(model, prompts: torch.Tensor, fact_sets: FactSpec, config: dict, J: int,
                           generator: Optional[torch.Generator] = None, gen_length: int = 8,
                           vocab: Optional[int] = None, truncate: bool = True):
    """(E5) Run native `generate` (pinned semantics, return_trace=True) with the CURRENT student (no grad) under one
    decoding config for every prompt row; truncate each rollout at the first step tau whose commit contains a token
    of that row's F (the pre-commit decision state s_tau is kept, nothing after it); keep J states sampled uniformly
    without replacement from the pre-tau states PLUS s_tau when it exists (SPEC v1 section 2: a design choice, not
    an unbiased sample of the visitation measure). Returns dict:
      states (N, l) long, owner (N,) row index into prompts, decision (N,) bool, leak (b,) bool (free diagnostic:
      the rollout committed an F token = leaked under this config), n_kept (b,), fwd_rows (# model row-forwards run).
    """
    b, plen = prompts.shape
    x, trace = generate(model, prompts, steps=config["steps"], gen_length=gen_length,
                        block_length=config["block_length"], temperature=config["temperature"],
                        remasking=config["remasking"], generator=generator, return_trace=True)
    if vocab is None:
        vocab = fact_sets.shape[1] if isinstance(fact_sets, torch.Tensor) else \
            max([int(x.max())] + [int(f) for fs in fact_sets for f in fs]) + 1
    fm = rows_fact_mask(fact_sets, b, vocab)
    seq = torch.stack(trace + [x], 0)                                  # (N+1, b, l): s_0..s_{N-1}, final
    nsteps = len(trace)
    out_s, out_o, out_d = [], [], []
    leak = torch.zeros(b, dtype=torch.bool)
    n_kept = torch.zeros(b, dtype=torch.long)
    for r in range(b):
        first = None
        for n in range(nsteps):
            committed = (seq[n, r] == MASK) & (seq[n + 1, r] != MASK)
            toks = seq[n + 1, r][committed]
            if toks.numel() and bool(fm[r, toks].any()):
                first = n
                break
        leak[r] = first is not None
        if not truncate:                                               # T5b ablation: keep every pre-step state
            first = None
        last = nsteps - 1 if first is None else first                 # keep s_0..s_last
        n_kept[r] = last + 1
        cand = list(range(last + 1))
        if first is not None:                  # SPEC v1: J uniform pre-tau states PLUS the decision state s_tau
            rest = cand[:-1]
            k = min(J, len(rest))
            pick = [rest[i] for i in torch.randperm(len(rest), generator=generator)[:k].tolist()] + [first]
        else:
            pick = [cand[i] for i in torch.randperm(len(cand), generator=generator)[:min(J, len(cand))].tolist()]
        for n in sorted(pick):
            out_s.append(seq[n, r])
            out_o.append(r)
            out_d.append(first is not None and n == first)
    states = torch.stack(out_s, 0)
    owner = torch.tensor(out_o, dtype=torch.long)
    if truncate:
        assert bool(is_fact_free(states, fm[owner]).all()), "rollout state with a visible fact token"
    return {"states": states, "owner": owner, "decision": torch.tensor(out_d), "leak": leak, "n_kept": n_kept,
            "fwd_rows": b * nsteps * (2 if config.get("cfg_scale", 0) > 0 else 1)}


# ============================================================================================ (E1) forget loss
def _masked_mean(per_pos: torch.Tensor, pos: torch.Tensor):
    """Mean over states of the mean over that state's positions (states with no position get weight 0)."""
    cnt = pos.sum(1)
    per_state = (per_pos * pos).sum(1) / cnt.clamp_min(1)
    valid = cnt > 0
    return per_state[valid].mean() if valid.any() else per_pos.sum() * 0.0


def fx_forget_loss(model, ref_fn: Callable[[torch.Tensor], torch.Tensor], states: torch.Tensor,
                   fact_mask: torch.Tensor, eps: float = 1e-3, prompt_len: int = 8, positions=None) -> Dict:
    """(E1) forget term: mean_s mean_{i in M(s)} KL(r_eps(.|s,i) || p_theta(.|s,i)) with r_eps from the frozen
    reference evaluated at the SAME state s. fact_mask: bool (N, V), row n = F of the forget item owning state n.
    Returns dict: loss (KL, differentiable), ce (CE to soft target), ent (target entropy), diagnostics."""
    pos = (states == MASK) if positions is None else positions
    pos = pos.clone()
    pos[:, :prompt_len] = False
    with torch.no_grad():
        z0 = ref_fn(states)
        log_r, info = exclusion_target(z0, fact_mask[:, None, :].expand_as(z0), eps, return_info=True)
    z = model(states)
    ce, ent, kl = soft_ce_kl(log_r, z)
    loss = _masked_mean(kl, pos)
    with torch.no_grad():
        pF = (F.softmax(z.float(), -1) * fact_mask[:, None, :]).sum(-1)
        p0F = info["log_p0F"].exp()
        d = {"ce": float(_masked_mean(ce, pos)), "ent": float(_masked_mean(ent, pos)),
             "pF_student": float(pF[pos].mean()), "pF_ref": float(p0F[pos].mean()),
             "frac_active": float(info["active"][pos].float().mean())}
    d["loss"] = loss
    return d


# ============================================================================================ controls
def _per_example_sft(logits, ids, masked, p_mask, ans_len):
    """Per-example LLaDA SFT loss (sum_masked CE/p_mask/answer_len) — the MDLM substitute for -log p(y|x)."""
    ce = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
    per = torch.zeros(ids.shape[0], dtype=ce.dtype)
    per.index_add_(0, masked.nonzero()[:, 0], ce / ans_len)
    return per


def npo_loss(model, ref_fn, ids: torch.Tensor, prompt_len: int, beta: float = 0.2, generator=None) -> Dict:
    """MDU App. B NPO for MDLMs: -(2/beta) E log sigma(beta (L_sft(theta) - L_sft^ref)), same noisy state for both."""
    noisy, masked, p_mask = forward_process_sft(ids, prompt_len, generator=generator)
    ans_len = ids.shape[1] - prompt_len
    lt = _per_example_sft(model(noisy), ids, masked, p_mask, ans_len)
    with torch.no_grad():
        lr_ = _per_example_sft(ref_fn(noisy), ids, masked, p_mask, ans_len)
    loss = -(2.0 / beta) * F.logsigmoid(beta * (lt - lr_)).mean()
    return {"loss": loss, "sft_student": float(lt.mean()), "sft_ref": float(lr_.mean())}


def mdu_loss(model, ref_fn, ids: torch.Tensor, prompt_len: int, tau: float = 1.0, generator=None) -> Dict:
    """MDU: random TF state y_t (LLaDA SFT forward process); target a = softmax(tau * z_anchor), anchor = frozen
    reference at the same y_t with PROMPT positions -> MASK; loss = mean_s mean_{i in M_t} KL(p_theta || a)."""
    noisy, masked, _ = forward_process_sft(ids, prompt_len, generator=generator)
    with torch.no_grad():
        un = noisy.clone()
        un[:, :prompt_len] = MASK
        log_a = F.log_softmax(tau * ref_fn(un).float(), -1)
    log_p = F.log_softmax(model(noisy).float(), -1)
    kl = (log_p.exp() * (log_p - log_a)).sum(-1)
    return {"loss": _masked_mean(kl, masked), "n_masked": int(masked.sum())}


def sc_tf_loss(model, ref_fn, ids: torch.Tensor, prompt_len: int, fact_mask: torch.Tensor, eps: float = 1e-3,
               generator=None) -> Dict:
    """SC-TF (SPEC update 04:5x, D 3.4/T3): exclusion target r_eps at the SAME state (prompt visible) on MDU's random
    TF states (LLaDA SFT forward process; fact tokens may be visible), all masked positions, target-first
    KL(r_eps || p_theta) (= MDU `--null_anchor_kl_dir reverse` with the anchor replaced by r_eps)."""
    noisy, masked, _ = forward_process_sft(ids, prompt_len, generator=generator)
    d = fx_forget_loss(model, ref_fn, noisy, fact_mask, eps=eps, prompt_len=prompt_len, positions=masked)
    d["frac_fact_visible_states"] = float((~is_fact_free(noisy, fact_mask)).float().mean())
    return d


@torch.no_grad()
def mdu_traj_rollout_state(model, ids: torch.Tensor, prompt_len: int, T: int = 8, stop_step: Optional[int] = None,
                           generator: Optional[torch.Generator] = None):
    """Port of MDU `_traj_rollout_na_loss` state construction (unlearn_mdu_llada.py L1005-1060): CURRENT model,
    greedy argmax, low-confidence top-k with n_unmask = ceil(remaining / (T - step)), ONE block over the answer
    region, stop at ONE random step in [1, T-1] per batch; no fact filter / truncation. Returns (state, stop_step,
    fwd_rows)."""
    b, l = ids.shape
    T = max(int(T), 2)
    if stop_step is None:
        stop_step = int(torch.randint(1, T, (1,), generator=generator))
    x = ids.clone()
    x[:, prompt_len:] = MASK
    fwd = 0
    for step in range(stop_step):
        mask_idx = x == MASK
        if not mask_idx.any():
            break
        n_unmask = (mask_idx.sum(1).float() / max(T - step, 1)).ceil().long().clamp(min=1)
        probs = F.softmax(model(x).float(), -1)
        fwd += b
        conf, pred = probs.max(-1)
        for i in range(b):
            pos = mask_idx[i].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            k = int(min(int(n_unmask[i]), len(pos)))
            _, top = conf[i, pos].topk(k, largest=True)
            ch = pos[top]
            x[i, ch] = pred[i, ch]
    return x, stop_step, fwd


def sc_loss(model, ref_fn, ids: torch.Tensor, prompt_len: int, fact_mask: torch.Tensor, eps: float = 1e-3,
            T: int = 8, generator=None) -> Dict:
    """SC (D 3.4/T4): SC-TF target on the MDU rollout-flag state (one random-stop greedy rollout state of the current
    model per batch, refreshed every step, no fact filter: post-leak states are trained)."""
    s, stop, fwd = mdu_traj_rollout_state(model, ids, prompt_len, T=T, generator=generator)
    d = fx_forget_loss(model, ref_fn, s, fact_mask, eps=eps, prompt_len=prompt_len)
    d["stop_step"] = stop
    d["rollout_fwd_rows"] = fwd
    d["frac_fact_visible_states"] = float((~is_fact_free(s, fact_mask)).float().mean())
    return d


# ============================================================================================ certificate (SPEC v1 5)
@torch.no_grad()
def hazard_certificate(model, prompts: torch.Tensor, fact_mask: torch.Tensor, config: dict,
                       generator: Optional[torch.Generator] = None, gen_length: int = 8) -> Dict:
    """Post-hoc certificate B_hat_pi (Proposition 1) on fresh native rollouts of `model` under one config:
        B_hat = mean over rollouts of sum_{n <= tau} sum_{i in E_n} min(1, rho(s_n, i)) * (k_n/|E_n| if random)
    rho = p(F|s,i) / max_v p(v|s,i) from the model's own logits at the visited state (valid for T in [0,1], cfg 0);
    E_n = masked positions of the current block; tau = first F-commit step (last step if none). Logits are captured
    from the sampler itself (logits_hook returning them unchanged), so no extra forwards are needed.
    Returns per-rollout arrays: bound, leak, decision state (or None), plus their means."""
    cap = []

    def hook(x, logits):
        cap.append(logits.detach().float())
        return logits

    x, trace = generate(model, prompts, steps=config["steps"], gen_length=gen_length,
                        block_length=config["block_length"], temperature=config["temperature"],
                        remasking=config["remasking"], generator=generator, logits_hook=hook, return_trace=True)
    b, plen = prompts.shape
    seq = trace + [x]
    N = len(trace)
    spb = N // (gen_length // config["block_length"])
    bound = torch.zeros(b, dtype=torch.float64)
    alive = torch.ones(b, dtype=torch.bool)          # no F commit strictly before step n
    leak = torch.zeros(b, dtype=torch.bool)
    dec_state = [None] * b
    for n in range(N):
        s, nxt = seq[n], seq[n + 1]
        nb = n // spb
        lo, hi = plen + nb * config["block_length"], plen + (nb + 1) * config["block_length"]
        elig = torch.zeros_like(s, dtype=torch.bool)
        elig[:, lo:hi] = s[:, lo:hi] == MASK
        p = F.softmax(cap[n].double(), -1)
        rho = (p * fact_mask[:, None, :]).sum(-1) / p.max(-1).values
        term = (rho.clamp(max=1.0) * elig).sum(1)
        if config["remasking"] == "random":
            k = ((s == MASK) & (nxt != MASK)).sum(1).double()
            term = term * k / elig.sum(1).clamp_min(1).double()
        bound += torch.where(alive, term, torch.zeros_like(term))
        committed = (s == MASK) & (nxt != MASK)
        hitF = (committed & torch.gather(fact_mask, 1, nxt)).any(1)
        for r in torch.nonzero(alive & hitF).flatten().tolist():
            dec_state[r] = s[r].clone()
        leak |= alive & hitF
        alive &= ~hitF
    return {"bound": bound, "leak": leak, "decision_states": dec_state,
            "bound_mean": float(bound.mean()), "leak_rate": float(leak.float().mean())}
