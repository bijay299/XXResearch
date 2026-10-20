"""
MDU null-anchor objective, reimplemented against the PINNED native functions.

Authoritative source: leegeoru/MDU @ e5e8e8274ff8f56f16e978803d9e902359eb5272
  src/unlearn_mdu_llada.py
    - _null_anchor_kl           (lines 947-1004)
    - _random_sft_null_anchor_loss (lines 1102-1147)
    - compute_loss dispatch     (lines 1334-1398)

REAL TRAINING PATH for our configuration, traced in compute_loss:
    match_mode == "random" and loss_type == "null_anchor"
      and not null_anchor_traj_rollout
    -> _random_sft_null_anchor_loss
and the result is ADDED (`loss = loss + forget_nll`), not negated, because
null_anchor is minimized directly. With alpha == 0 the retain branch is skipped
entirely (`if len(retain_idx) > 0 and self.alpha != 0`).
NOTE: the other null_anchor branch (line 864) uses `model` under no_grad as its
anchor; it is DEAD CODE for our config. Ours uses a genuinely frozen ref model.

RECONSTRUCTED FROM dllm (NOT AVAILABLE, revision unverified):
  MDUTrainer subclasses dllm.core.trainers.MDLMTrainer, so `self.scheduler`,
  `self.time_epsilon`, `_postprocess_outputs` and `right_shift_logits` live in
  dllm, which MDU neither vendors nor pins. Reconstruction used here:
    scheduler(t)   = 1 - t            (MDLM linear/absorbing schedule)
      => p_mask    = 1 - scheduler(t) = t
    time_epsilon   = 1e-3
    _postprocess_outputs = identity, because right_shift_logits=False
      (no autoregressive shift; LLaDA is bidirectional)
  Cross-check: this makes p_mask ~ U(1e-3, 1), which coincides with LLaDA's own
  GUIDELINES.md forward_process (p_mask = (1-eps)t + eps, eps=1e-3). Two
  independent sources agree, but the dllm identity remains unverified.
"""
import torch
import torch.nn.functional as F

MASK_ID = 126336

# Verified objective configuration (explicitly set; the pinned launcher omits these)
CONFIG = dict(
    loss_type="null_anchor",
    match_mode="random",
    null_anchor_tau=0.5,
    null_anchor_eta=0.0,
    null_anchor_kl_dir="forward",
    null_anchor_traj_rollout=False,
    null_anchor_exclude_special=False,
    alpha=0.0,
    right_shift_logits=False,
    time_epsilon=1e-3,
    scheduler="linear: scheduler(t)=1-t  =>  p_mask=t",
)


def null_anchor_kl(logits_c, logits_u, tau, eta=0.0, kl_dir="forward",
                   exclude_special=False, special_ids=None):
    """Port of MDU _null_anchor_kl (eta=0, forward, exclude_special=False path).

    target_logits = tau * logits_u.detach()      # tau MULTIPLIES anchor logits
    returns per-token KL(p_c || sg(p_target)), shape (B, T)
    """
    if eta == 0.0:
        target_logits = logits_u.detach()
    else:
        target_logits = (logits_u - eta * (logits_c - logits_u)).detach()
    # tau multiplies the anchor logits. It is NOT a temperature divisor.
    target_logits = tau * target_logits

    logits_c_eff = logits_c.float()
    target_logits = target_logits.float()
    if exclude_special and special_ids is not None and len(special_ids):
        ids = torch.as_tensor(special_ids, device=target_logits.device, dtype=torch.long)
        logits_c_eff = logits_c_eff.clone()
        target_logits = target_logits.clone()
        logits_c_eff.index_fill_(-1, ids, float("-inf"))
        target_logits.index_fill_(-1, ids, float("-inf"))

    log_pc = F.log_softmax(logits_c_eff, dim=-1)
    log_pt = F.log_softmax(target_logits, dim=-1).detach()

    def _safe_kl(p, log_p, log_q):
        term = log_p - log_q
        term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
        return (p * term).sum(dim=-1)

    if kl_dir == "reverse":
        pt = log_pt.exp()
        return _safe_kl(pt, log_pt, log_pc)
    if kl_dir == "js":
        pc, pt = log_pc.exp(), log_pt.exp()
        pm = 0.5 * (pc + pt)
        log_pm = pm.clamp_min(1e-12).log()
        return 0.5 * (_safe_kl(pc, log_pc, log_pm) + _safe_kl(pt, log_pt, log_pm))
    pc = log_pc.exp()
    return _safe_kl(pc, log_pc, log_pt)


def sample_mask(input_ids, labels, generator, time_epsilon=1e-3):
    """Port of the noising in _random_sft_null_anchor_loss.
    maskable = answer positions only (labels != -100); prompt is never corrupted.
    """
    b, l = input_ids.shape
    dev = input_ids.device
    maskable = labels != -100
    t = time_epsilon + (1 - time_epsilon) * torch.rand(b, device=dev, generator=generator)
    # scheduler(t) = 1 - t  =>  p_mask = t   (reconstructed; see module docstring)
    p_mask = t.unsqueeze(1).expand(b, l)
    masked = (torch.rand((b, l), device=dev, generator=generator) < p_mask) & maskable
    noised = torch.where(masked, MASK_ID, input_ids)
    return noised, masked, maskable, p_mask


def null_anchor_loss(student_fwd, anchor_fwd, input_ids, labels, attention_mask,
                     generator, tau=0.5, eta=0.0, kl_dir="forward",
                     time_epsilon=1e-3, exclude_special=False, special_ids=None):
    """Port of _random_sft_null_anchor_loss.

    cond   : student( [prompt | partially-masked answer] )        grad ON
    anchor : frozen( [MASK x prompt | same partially-masked answer] )  no_grad
    loss   : mean over MASKED ANSWER positions of KL(p_c || sg(p_anchor^tau))
    Minimized directly (the caller adds it; it is not negated).
    """
    noised, masked, maskable, _ = sample_mask(input_ids, labels, generator, time_epsilon)

    logits_c = student_fwd(noised, attention_mask)

    # anchor input: ALSO mask the prompt positions
    if attention_mask is not None:
        q_mask = (~maskable) & attention_mask.bool()
    else:
        q_mask = ~maskable
    noised_u = torch.where(q_mask, MASK_ID, noised)
    with torch.no_grad():
        logits_u = anchor_fwd(noised_u, attention_mask)

    kl = null_anchor_kl(logits_c, logits_u, tau=tau, eta=eta, kl_dir=kl_dir,
                        exclude_special=exclude_special, special_ids=special_ids)
    denom = masked.sum().clamp_min(1)
    loss = (kl * masked.float()).sum() / denom
    return loss, dict(n_masked=int(masked.sum()), n_maskable=int(maskable.sum()),
                      noised=noised, noised_u=noised_u, masked=masked,
                      maskable=maskable, q_mask=q_mask,
                      logits_c=logits_c, logits_u=logits_u, kl=kl)
