"""
MDU and EMDU forget objectives for LLaDA (masked diffusion).

MDU  — leegeoru/MDU @ e5e8e8274ff8f56f16e978803d9e902359eb5272, native
       `_null_anchor_kl` + `_random_sft_null_anchor_loss`.
EMDU — arXiv:2610.25447v1 (Garber & Tirer), PAPER-BASED REIMPLEMENTATION.
       No author code or checkpoints exist: the paper says "the checkpoints we
       will release" (future tense) and gives no EMDU repository.

The method difference is EXACTLY these two axes and nothing else:

                     anchor input (answer region)      KL direction
    MDU    same partially-masked answer as student     KL(student || anchor)   "anchor-last"
    EMDU   ENTIRE answer span masked                   KL(anchor || student)   "anchor-first"

In both, the student sees the original prompt and a randomly masked answer, the
anchor is frozen and sees a fully masked prompt, tau MULTIPLIES anchor logits,
alpha=0 (forget-only, no retain corpus).

Everything else — data order, masking draws, optimizer, precision, update budget,
reduction, special-token handling, padding — is shared.
"""
import torch
import torch.nn.functional as F

MASK_ID = 126336

CONFIG = dict(
    tau=1.0,                      # paper's RWKU setting; multiplies anchor logits
    alpha=0.0,                    # forget-only, no retain corpus
    eta=0.0,                      # MDU null_anchor_eta
    exclude_special=False,        # MDU null_anchor_exclude_special
    right_shift_logits=False,     # no AR shift; LLaDA is bidirectional
    time_epsilon=1e-3,
    scheduler="linear: scheduler(t)=1-t => p_mask=t",
)


def sample_answer_mask(input_ids, labels, generator, time_epsilon=1e-3):
    """Shared noising. maskable = answer positions only (labels != -100), so the
    prompt is never corrupted for the student."""
    b, l = input_ids.shape
    dev = input_ids.device
    maskable = labels != -100
    t = time_epsilon + (1 - time_epsilon) * torch.rand(b, device=dev, generator=generator)
    p_mask = t.unsqueeze(1).expand(b, l)
    masked = (torch.rand((b, l), device=dev, generator=generator) < p_mask) & maskable
    noised = torch.where(masked, MASK_ID, input_ids)
    return noised, masked, maskable


def _kl_per_token(logits_c, logits_u, tau, direction, eta=0.0,
                  exclude_special=False, special_ids=None):
    """Per-token KL between the student conditional and the tau-scaled frozen anchor.

    direction="anchor_last"  -> KL(p_student || sg(p_anchor))   [MDU]
    direction="anchor_first" -> KL(sg(p_anchor) || p_student)   [EMDU]

    The anchor is detached in BOTH directions: it is a frozen target, never a
    gradient path. Casting follows MDU's native code, which calls .float(), so the
    KL and its gradient are computed in FP32.
    """
    if eta == 0.0:
        target_logits = logits_u.detach()
    else:
        target_logits = (logits_u - eta * (logits_c - logits_u)).detach()
    target_logits = target_logits / tau  # DELIBERATE BREAK

    logits_c_eff = logits_c.float()
    target_logits = target_logits.float()
    if exclude_special and special_ids:
        ids = torch.as_tensor(special_ids, device=target_logits.device, dtype=torch.long)
        logits_c_eff = logits_c_eff.clone()
        target_logits = target_logits.clone()
        logits_c_eff.index_fill_(-1, ids, float("-inf"))
        target_logits.index_fill_(-1, ids, float("-inf"))

    log_pc = F.log_softmax(logits_c_eff, dim=-1)
    log_pt = F.log_softmax(target_logits, dim=-1).detach()

    def _safe_kl(p, log_p, log_q):
        term = torch.nan_to_num(log_p - log_q, nan=0.0, posinf=0.0, neginf=0.0)
        return (p * term).sum(dim=-1)

    if direction == "anchor_last":               # MDU
        pc = log_pc.exp()
        return _safe_kl(pc, log_pc, log_pt)
    if direction == "anchor_first":              # EMDU
        pt = log_pt.exp()
        return _safe_kl(pt, log_pt, log_pc)
    raise ValueError(direction)


def forget_loss(method, student_fwd, anchor_fwd, input_ids, labels, attention_mask,
                generator, tau=1.0, eta=0.0, time_epsilon=1e-3,
                exclude_special=False, special_ids=None, return_info=False):
    """One forget-loss value for `method` in {"mdu","emdu"}.

    Student input (both methods):
        [ original prompt | randomly masked answer ]
    Anchor input:
        MDU : [ MASK x prompt | the SAME randomly masked answer ]
        EMDU: [ MASK x prompt | ENTIRE answer span masked ]
    Reduction (both): mean over the student's MASKED ANSWER positions, i.e.
        sum(kl * masked) / masked.sum().clamp_min(1)
    Batch reduction: the mean above is already per-batch-element-pooled across the
    batch's masked positions (a single pooled mean, matching MDU's native code,
    NOT a per-sequence mean then averaged).
    """
    assert method in ("mdu", "emdu")
    noised, masked, maskable = sample_answer_mask(
        input_ids, labels, generator, time_epsilon)

    logits_c = student_fwd(noised, attention_mask)

    # Anchor: prompt fully masked in BOTH methods.
    if attention_mask is not None:
        q_mask = (~maskable) & attention_mask.bool()
    else:
        q_mask = ~maskable
    if method == "mdu":
        # anchor sees the SAME partially masked answer as the student
        anchor_in = torch.where(q_mask, MASK_ID, noised)
    else:
        # EMDU: anchor sees the ENTIRE answer span masked.
        # The STUDENT's answer masking is unchanged (still random).
        anchor_in = torch.where(q_mask | maskable, MASK_ID, input_ids)

    with torch.no_grad():
        logits_u = anchor_fwd(anchor_in, attention_mask)

    direction = "anchor_last" if method == "mdu" else "anchor_first"
    kl = _kl_per_token(logits_c, logits_u, tau=tau, direction=direction, eta=eta,
                       exclude_special=exclude_special, special_ids=special_ids)
    denom = masked.sum().clamp_min(1)
    loss = (kl * masked.float()).sum() / denom

    if not return_info:
        return loss
    return loss, dict(
        method=method, direction=direction, tau=tau,
        n_masked_student=int(masked.sum()), n_maskable=int(maskable.sum()),
        noised=noised, anchor_in=anchor_in, masked=masked, maskable=maskable,
        q_mask=q_mask, logits_c=logits_c, logits_u=logits_u, kl=kl,
        anchor_answer_fully_masked=bool((anchor_in[maskable] == MASK_ID).all()),
        anchor_prompt_fully_masked=bool((anchor_in[q_mask] == MASK_ID).all()),
        student_prompt_uncorrupted=bool(torch.equal(noised[q_mask], input_ids[q_mask])),
    )
