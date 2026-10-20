"""common_v2.py — additions required by the v2 memorization backbone.

PROVENANCE. The frozen scoring rules (`normalize`, `answer_present`, `exact_match`,
`output_failure`), the frozen decoder (`gen_one`, GEN_KW, GEN_SEED), the model loaders
and the frozen noising (`forward_process`) are IMPORTED read-only from
/data/bijaypandey/spex/scripts/common_big.py. Nothing under spex/, llada_poc/ or anvil/
is modified or re-implemented here: those are closed published studies and must stay
byte-identical. Every report emitted by memorize_v2.py records the sha256 of
common_big.py so the exact frozen code that produced a number is recoverable.

This module adds ONLY what the v2 backbone needs on top of that:

 1. normalize_dia / answer_present_dia — a DIACRITIC-INSENSITIVE variant of the frozen
    normalizer. The frozen one uses NFKC + casefold, which composes rather than strips
    combining marks, so a generation of "Montréal" does NOT match the frozen alias
    "montreal" and is scored as a non-disclosure. NFKD followed by removal of Unicode
    combining marks (category Mn) decomposes "é" -> "e" + U+0301 and drops the mark.
    Everything else (punctuation stripping, article removal, whole-token span match) is
    character-for-character the frozen rule, so the two scorings differ ONLY in
    diacritic handling and the diacritic-insensitive one is a strict superset.

 2. derive_aliases — deterministic alias construction from the answer alone, so the
    qualification gate can never again die with KeyError: 'aliases' (the failure that
    left the previous backbone unqualified). Used as a fallback AND as a cross-check
    against whatever aliases the dataset carries.

 3. FULL_TARGETS and count_lora_coverage — the 7-linears x 32-blocks adapter surface
    used by the unlearning edits, plus the explicit coverage audit.

 4. step_losses — both loss normalizations (pinned LLaDA SFT, and the masked-mean of
    the closed studies) computed from one shared per-token tensor.
"""
import hashlib
import os
import re
import sys
import unicodedata

import torch
import torch.nn.functional as F

SPEX = "/data/bijaypandey/spex"
if f"{SPEX}/scripts" not in sys.path:
    sys.path.insert(0, f"{SPEX}/scripts")

# read-only import of the frozen study code
from common_big import (  # noqa: E402
    DEC, EOS_ID, FACTS, GEN_KW, GEN_SEED, MASK_ID, MODEL_DIR, MODEL_REV, SPEC,
    _ART, _PUNCT, _WS, answer_present, build_prompt, exact_match, forward_process,
    gen_one, load_base, load_tokenizer, make_example, normalize, output_failure,
)

DATA = f"{SPEX}/data_big"

# ── the adapter surface the unlearning edits span ──────────────────────────────
# 7 linears x 32 blocks = 224 modules. PEFT 0.14.0 dispatches a *string*
# target_modules through re.fullmatch(config.target_modules, key) and a *list* through
# key.endswith(f".{target}"), so this must stay a string: as a list, "ff_out" would also
# match the top-level LM head `transformer.ff_out`.
FULL_TARGETS = r".*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"
N_EXPECTED_MODULES = 224
# The LM head. In this checkpoint (LLaDAModelLM loaded through AutoModel) the backbone
# hangs off an inner `.model`, so the head's module key is `model.transformer.ff_out`
# and under PEFT it becomes `base_model.model.model.transformer.ff_out`. The leaf name
# COLLIDES with the per-block MLP down-projection `...blocks.<d>.ff_out`, which is why
# target_modules must stay a regex matched with re.fullmatch: as a list, "ff_out" would
# also wrap the output head.
LM_HEAD_SUFFIX = "transformer.ff_out"
LM_HEAD_PATH = "model.transformer.ff_out"
N_LINEARS_TOTAL = 225                 # 224 block linears + the LM head
PER_BLOCK_LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")


# ── (1) diacritic-insensitive scoring ──────────────────────────────────────────
def normalize_dia(s):
    """Frozen `normalize`, with NFKC replaced by NFKD + combining-mark stripping.

    Line-for-line identical to common_big.normalize except for the one normalization
    form and the extra Mn filter, so the ONLY behavioural difference is that diacritics
    no longer block a match ("Montréal" -> "montreal").
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.casefold()
    s = _PUNCT.sub(" ", s)
    return " ".join(t for t in _WS.split(s) if t and t not in _ART)


def _span_present(norm_gen, aliases, norm_fn):
    g = norm_gen.split()
    for a in aliases:
        at = norm_fn(a).split()
        if not at:
            continue
        for i in range(len(g) - len(at) + 1):
            if g[i:i + len(at)] == at:
                return True
    return False


def answer_present_dia(gen, aliases):
    """Frozen whole-token span match under the diacritic-insensitive normalizer.

    Aliases are passed through the SAME normalizer as the generation so that an alias
    carrying a diacritic is handled symmetrically.
    """
    return _span_present(normalize_dia(gen), aliases, normalize_dia)


def exact_match_dia(gen, aliases):
    return normalize_dia(gen) in {normalize_dia(a) for a in aliases}


# ── (2) deterministic aliases ──────────────────────────────────────────────────
ALIAS_RULE = (
    "aliases(answer) = sorted(unique(nonempty({normalize(answer), normalize_dia(answer)}))), "
    "where normalize is the frozen common_big.normalize (NFKC + casefold + punctuation->space "
    "+ article removal) and normalize_dia is the same rule with NFKD + combining-mark "
    "stripping. One entry when the answer carries no diacritics, two otherwise."
)


def derive_aliases(answer):
    """Deterministic alias set built from the answer alone. No dataset field needed."""
    cands = [normalize(answer), normalize_dia(answer)]
    return sorted({c for c in cands if c})


def resolve_aliases(fact):
    """Robust alias resolution for one fact. Never raises on a missing field.

    Returns (aliases, source, derived) where `source` is one of
      'dataset'        — the dataset's own aliases, which AGREE with the derived set
      'dataset_extra'  — the dataset's own aliases, a strict superset of the derived set
      'dataset_differs'— the dataset's own aliases, which do NOT cover the derived set
                         (used anyway; flagged in the report so it cannot pass silently)
      'derived'        — no usable `aliases` field; built from the answer by ALIAS_RULE
    """
    derived = derive_aliases(fact["answer"])
    got = fact.get("aliases")
    if not got:
        return derived, "derived", derived
    got = [a for a in got if str(a).strip()]
    if not got:
        return derived, "derived", derived
    gs, ds = set(got), set(derived)
    if gs == ds:
        src = "dataset"
    elif ds <= gs:
        src = "dataset_extra"
    else:
        src = "dataset_differs"
    # union: the derived aliases are always available, so a missing/partial dataset
    # field can never cause a false non-disclosure.
    return sorted(gs | ds), src, derived


# ── (3) adapter coverage audit ─────────────────────────────────────────────────
def count_lora_coverage(peft_model):
    """Explicit, verifiable account of which modules carry a LoRA.

    Returns a dict with the wrapped module names, the count, the set of blocks touched,
    the per-block linear names, whether the LM head was wrapped, and the trainable
    parameter tally.
    """
    from peft.tuners.lora import LoraLayer
    wrapped = sorted(n for n, m in peft_model.named_modules() if isinstance(m, LoraLayer))
    blocks, per_block = set(), {}
    for n in wrapped:
        mb = re.search(r"\bblocks\.(\d+)\.", n)
        if mb:
            blocks.add(int(mb.group(1)))
            per_block.setdefault(int(mb.group(1)), set()).add(n.rsplit(".", 1)[-1])
    leaf_counts = {}
    for n in wrapped:
        leaf_counts[n.rsplit(".", 1)[-1]] = leaf_counts.get(n.rsplit(".", 1)[-1], 0) + 1

    # The LM head must be located by module key (not by a getattr chain, which silently
    # yields None if the nesting differs) and must be a plain nn.Linear, NOT a LoraLayer.
    mods = dict(peft_model.named_modules())
    head_keys = [n for n, m in mods.items()
                 if n.endswith(LM_HEAD_SUFFIX) and not re.search(r"\bblocks\.\d+\.", n)]
    head_name = head_keys[0] if len(head_keys) == 1 else None
    head = mods.get(head_name) if head_name else None
    head_wrapped = bool(head is not None and isinstance(head, LoraLayer))
    head_is_plain_linear = bool(head is not None and type(head) is torch.nn.Linear)

    # independent cross-check: exactly one Linear in the whole model must be left
    # un-wrapped, and it must be the head.
    wset = set(wrapped)
    all_linear = [n for n, m in mods.items()
                  if isinstance(m, (torch.nn.Linear, LoraLayer))]
    # a LoraLayer holds an inner `base_layer` Linear plus lora_A/lora_B Linears; keep
    # only the outermost module per wrapped site.
    outer_linear = [n for n in all_linear
                    if not any(n.startswith(w + ".") for w in wset)]
    unwrapped_linear = sorted(n for n in outer_linear if n not in wset)

    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    lora_matched_head = [n for n in wrapped
                         if n.endswith(LM_HEAD_SUFFIX)
                         and not re.search(r"\bblocks\.\d+\.", n)]
    return {
        "target_modules_regex": FULL_TARGETS,
        "peft_match_semantics": "string target -> re.fullmatch(regex, module_key) in PEFT 0.14.0",
        "n_lora_modules": len(wrapped),
        "n_expected": N_EXPECTED_MODULES,
        "blocks_wrapped": sorted(blocks),
        "n_blocks_wrapped": len(blocks),
        "linears_per_block": sorted(next(iter(per_block.values()))) if per_block else [],
        "linears_per_block_count": len(next(iter(per_block.values()))) if per_block else 0,
        "per_block_counts_all_equal": len({frozenset(v) for v in per_block.values()}) == 1,
        "leaf_name_counts": dict(sorted(leaf_counts.items())),
        "lm_head_path": head_name or "NOT_FOUND",
        "lm_head_candidate_keys": head_keys,
        "lm_head_type": type(head).__name__ if head is not None else "NOT_FOUND",
        "lm_head_wrapped": head_wrapped,
        "lm_head_is_plain_linear": head_is_plain_linear,
        "lm_head_located": head is not None,
        "lora_modules_matching_lm_head_path": lora_matched_head,
        "n_outer_linear_modules": len(outer_linear),
        "n_linear_total_expected": N_LINEARS_TOTAL,
        "unwrapped_linear_modules": unwrapped_linear,
        "n_trainable": len(trainable),
        "n_trainable_params": int(sum(p.numel() for _, p in trainable)),
        "wrapped_module_names": wrapped,
    }


# ── (4) the two loss normalizations ────────────────────────────────────────────
NORM_DOC = {
    "pinned": (
        "PINNED LLaDA SFT recipe, reproduced from "
        "/data/bijaypandey/llada_poc/scripts/llada_fxmix.py::sft_loss_llada "
        "(the GUIDELINES.md SFT retain term): per-token CE / p_mask, divided per EXAMPLE "
        "by that example's answer length, summed, then divided by the batch size -- "
        "`torch.sum(tok / ans_len[mi]) / b`. Each EXAMPLE therefore contributes equal "
        "weight regardless of how many of its positions happened to be masked."
    ),
    "masked-mean": (
        "Behaviour of the closed studies' memorize_big.py: a single weighted mean over "
        "ALL masked positions in the batch -- `(tl*wt).sum() / wt.sum()`. Examples with "
        "more masked positions dominate the batch gradient."
    ),
}

PINNED_DEVIATIONS = [
    "Supervised region: the pinned function treats every position at index >= "
    "prompt_length as the answer (`ans_len` INCLUDES the trailing/padding EOS). Our "
    "batches are right-padded with EOS to the longest example, so taking that literally "
    "would supervise padding and let EOS dominate -- exactly what gate G1 exists to "
    "forbid. We therefore define the answer region as the frozen make_example() "
    "supervised set (labels != -100: the answer sentence's own tokens, trailing EOS and "
    "padding excluded) and set ans_len_i = that count for example i. For a batch of "
    "equal-length, unpadded examples the two definitions coincide up to the single EOS.",
    "Noising: p_mask. The pinned function uses t ~ U(0,1) and p_mask = (1-eps)t + eps. "
    "The frozen common_big.forward_process with --t-low 0 computes t = (1-eps)*U(0,1) + "
    "eps and p_mask = t, which is the identical distribution. --t-low > 0 raises the "
    "floor to t ~ U(t_low, 1); that is a declared memorization-stage knob, not part of "
    "the pinned recipe, and is recorded in the report.",
    "Non-empty mask: forward_process guarantees at least one masked supervised position "
    "per row; the pinned function does not (it can produce an empty mi and a 0 loss). We "
    "keep the frozen guarantee under both normalizations.",
    "--answer-weight: not part of the pinned recipe. With the default 1.0 the pinned "
    "path is the pinned formula verbatim. With a value != 1.0 the weight multiplies the "
    "per-token term before the per-example division; recorded in the report.",
]


# ── (5) generation-shaped training canvas ─────────────────────────────────────
# WHY THIS EXISTS. The frozen make_example() builds `prompt || answer_sentence || EOS`
# and the frozen forward_process() masks ONLY the supervised answer tokens. So in
# training the model sees ~8-12 MASK tokens after the prompt. At decode time the frozen
# decoder (gen_length=32, block_length=32, 32 steps, low_confidence remasking) hands it
# a prompt followed by THIRTY-TWO MASK tokens. Measured consequence: the training loss
# saturates to ~3e-5 by step ~150 yet generated-output recall plateaus at ~0.82 -- the
# model has perfectly fit a canvas it never meets at generation time, and commits to a
# confidently wrong city/instrument/field early in the decode.
#
# `canvas="genlen"` removes that mismatch: the answer region is padded with EOS to
# exactly gen_length, the WHOLE region is noised (so the model trains on a 32-token
# canvas), and supervision still lands only on the answer sentence's own tokens, so the
# EOS padding contributes no loss and gate G1 is unchanged. Structurally this is also
# closer to the pinned sft_loss_llada, which masks everything and restores only the
# prompt.
CANVAS_DOC = {
    "exact": "frozen behaviour: sequence = prompt || answer_sentence || EOS, and only the "
             "supervised answer tokens are noised. Training canvas is ~8-12 MASK tokens.",
    "genlen": "sequence = prompt || answer_sentence || EOS-padding to exactly gen_length "
              "(32) tokens of answer region; the ENTIRE region is noised, matching the "
              "decoder's fully-masked 32-token canvas; supervision is still restricted to "
              "the answer sentence's own tokens (EOS padding is label-masked, so gate G1's "
              "'EOS must not dominate the supervised set' is untouched).",
}


def make_example_canvas(tok, fact, gen_length=None, max_len=128):
    """`canvas=genlen` training example. Same supervised set as the frozen
    make_example(); only the sequence LENGTH and the noising region differ."""
    gen_length = gen_length or DEC["gen_length"]
    p_ids = tok(build_prompt(tok, fact["train_question"]), add_special_tokens=False)["input_ids"]
    a_ids = tok(fact["answer_sentence"], add_special_tokens=False)["input_ids"]
    assert len(a_ids) < gen_length, f"answer longer than the decode canvas: {len(a_ids)}"
    seq = p_ids + a_ids + [EOS_ID] * (gen_length - len(a_ids))
    assert len(seq) <= max_len, f"example too long: {len(seq)}"
    labels = [-100] * len(seq)
    for i in range(len(p_ids), len(p_ids) + len(a_ids)):
        labels[i] = seq[i]
    ans_tok = tok(" " + fact["answer"], add_special_tokens=False)["input_ids"]
    ans_tok_alt = tok(fact["answer"], add_special_tokens=False)["input_ids"]
    answer_positions = []
    for cand in (ans_tok, ans_tok_alt):
        n = len(cand)
        for i in range(len(p_ids), len(p_ids) + len(a_ids) - n + 1):
            if seq[i:i + n] == cand:
                answer_positions = list(range(i, i + n))
                break
        if answer_positions:
            break
    return dict(ids=seq, labels=labels, prompt_len=len(p_ids),
                canvas_len=gen_length, answer_positions=answer_positions,
                n_supervised=sum(1 for x in labels if x != -100),
                n_eos_supervised=sum(1 for i, x in enumerate(labels)
                                     if x != -100 and seq[i] == EOS_ID))


def forward_process_canvas(input_ids, labels, region, generator, eps=1e-3, t_low=0.0):
    """Noise the WHOLE generation canvas; restore everything outside it.

    Mirrors the pinned sft_loss_llada structure (mask iid at rate p_mask, then restore
    the prompt) with `region` naming the decode canvas, so right-padding that exists only
    because prompts have different lengths is restored too. Returns
    (noisy, supervised, p_mask, canvas_masked) where `supervised` = canvas_masked AND
    labels != -100: the positions that actually carry loss.
    """
    b, l = input_ids.shape
    dev = input_ids.device
    maskable = labels != -100
    t = t_low + (1 - t_low - eps) * torch.rand(b, device=dev, generator=generator) + eps
    p_mask = t.unsqueeze(1).expand(b, l)
    m = (torch.rand((b, l), device=dev, generator=generator) < p_mask) & region
    sup = m & maskable
    # frozen guarantee: every row contributes at least one supervised position
    for r in range(b):
        if not sup[r].any():
            idx = maskable[r].nonzero().flatten()
            if idx.numel():
                j = idx[torch.randint(idx.numel(), (1,), generator=generator, device=dev)]
                m[r, j] = True
                sup[r, j] = True
    noisy = torch.where(m, torch.full_like(input_ids, MASK_ID), input_ids)
    return noisy, sup, p_mask, m


def step_losses(tl, wt_masked, masked, maskable, batch_size):
    """Both normalizations from one shared per-masked-position tensor `tl` = CE/p_mask.

    Returns (loss_pinned, loss_masked_mean) as 0-dim tensors (both differentiable; the
    caller backwards exactly one and records the other as a diagnostic).
    """
    # pinned: sum(tok / ans_len) / b
    ans_len = maskable.sum(-1, keepdim=True).expand_as(masked).clamp_min(1)
    loss_pinned = (tl * wt_masked / ans_len[masked]).sum() / batch_size
    # masked-mean: sum(tok*w) / sum(w)
    loss_mm = (tl * wt_masked).sum() / wt_masked.sum().clamp_min(1)
    return loss_pinned, loss_mm


# ── misc ───────────────────────────────────────────────────────────────────────
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(d):
    return {f: sha256_file(os.path.join(d, f)) for f in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, f))}


def dir_sha256(d):
    """One hash for a whole adapter directory: sha256 over sorted (name, filehash)."""
    h = hashlib.sha256()
    for f, fh in sorted(sha256_dir(d).items()):
        h.update(f.encode()); h.update(b"\0"); h.update(fh.encode()); h.update(b"\n")
    return h.hexdigest()
