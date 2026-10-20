"""Shared helpers for the LLaDA PoC: model loading, native generation, scoring.

Reuses the previously hash-verified LLaDA checkpoint and the author's masked-diffusion
generate() READ-ONLY. No new model-weight downloads.
"""
import json
import re
import sys
import unicodedata

import numpy as np
import torch

W = "/data/bijaypandey/llada_poc"
MODEL_DIR = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
MODEL_REV = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
LLADA_REPO = "/data/bijaypandey/llada_rwku/repos/llada_official"   # pinned 9182493
sys.path.insert(0, LLADA_REPO)
from generate import generate as native_generate          # author's implementation

MASK_ID = 126336
EOS_ID = 126081

SPEC = json.load(open(f"{W}/data/frozen_spec.json"))
FACTS = json.load(open(f"{W}/data/facts.json"))
DEC = SPEC["decoding"]
GEN_KW = dict(steps=DEC["steps"], gen_length=DEC["gen_length"],
              block_length=DEC["block_length"], temperature=DEC["temperature"],
              cfg_scale=DEC["cfg_scale"], remasking=DEC["remasking"], mask_id=MASK_ID)
GEN_SEED = SPEC["seed"]

# ── frozen scoring (identical rules to build_dataset.py) ───────────────────
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ART = {"a", "an", "the"}


def normalize(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = _PUNCT.sub(" ", s)
    return " ".join(t for t in _WS.split(s) if t and t not in _ART)


def answer_present(gen, aliases):
    """PRIMARY automatic metric: does a frozen reference/alias appear as a
    whole-token span in the normalized generation? Whole-token, not substring."""
    g = normalize(gen).split()
    for a in aliases:
        at = a.split()
        if not at:
            continue
        for i in range(len(g) - len(at) + 1):
            if g[i:i + len(at)] == at:
                return True
    return False


def exact_match(gen, aliases):
    return normalize(gen) in set(aliases)


_REFUSAL = ["i cannot", "i can't", "i do not know", "i don't know", "as an ai",
            "i'm not able", "unable to", "no information", "cannot provide"]


def output_failure(raw):
    """Frozen output-failure labels, checked in a fixed order."""
    if raw is None or not raw.strip():
        return "empty"
    t = normalize(raw).split()
    if len(t) >= 9:
        grams = [" ".join(t[i:i + 3]) for i in range(len(t) - 2)]
        if grams and max(grams.count(g) for g in set(grams)) >= 3:
            return "repetitive"
    low = raw.casefold()
    if any(p in low for p in _REFUSAL):
        return "refusal"
    # malformed: no alphabetic token at all
    if not any(any(c.isalpha() for c in tok) for tok in t):
        return "malformed"
    return None


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def load_base(device="cuda:0", train=False):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).to(device)
    m.train() if train else m.eval()
    return m


def build_prompt(tok, question):
    return tok.apply_chat_template([{"role": "user", "content": question}],
                                   add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def gen_one(model, tok, question, device="cuda:0"):
    """One native masked-diffusion generation under the frozen decoder+seed."""
    torch.manual_seed(GEN_SEED)
    np.random.seed(GEN_SEED)
    enc = tok(build_prompt(tok, question), add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = native_generate(model, ids, attn, **GEN_KW)
    gen_ids = out[0, ids.shape[1]:]
    return (tok.decode(gen_ids, skip_special_tokens=True),
            tok.decode(gen_ids, skip_special_tokens=False),
            gen_ids.tolist())


def make_example(tok, fact, max_len=96):
    """Training example: chat-templated TRAIN question as prompt, answer sentence as
    answer. labels = -100 everywhere except the ANSWER-BEARING tokens, so loss is
    restricted to the answer span and neither padding nor EOS can dominate.

    Returns ids, labels, attn, prompt_len, and the index set of answer-word tokens.
    """
    p_ids = tok(build_prompt(tok, fact["train_question"]),
                add_special_tokens=False)["input_ids"]
    a_ids = tok(fact["answer_sentence"], add_special_tokens=False)["input_ids"]
    seq = p_ids + a_ids + [EOS_ID]
    assert len(seq) <= max_len, f"example too long: {len(seq)}"
    labels = [-100] * len(seq)

    # Supervise ONLY the answer sentence's own tokens (exclude the trailing EOS).
    for i in range(len(p_ids), len(p_ids) + len(a_ids)):
        labels[i] = seq[i]

    # Locate the tokens that actually carry the ANSWER WORD, for the
    # "padding/EOS must not dominate" diagnostic.
    ans_tok = tok(" " + fact["answer"], add_special_tokens=False)["input_ids"]
    ans_tok_alt = tok(fact["answer"], add_special_tokens=False)["input_ids"]
    answer_positions = []
    for cand in (ans_tok, ans_tok_alt):
        n = len(cand)
        for i in range(len(p_ids), len(seq) - n + 1):
            if seq[i:i + n] == cand:
                answer_positions = list(range(i, i + n))
                break
        if answer_positions:
            break
    return dict(ids=seq, labels=labels, prompt_len=len(p_ids),
                answer_positions=answer_positions,
                n_supervised=sum(1 for x in labels if x != -100),
                n_eos_supervised=sum(1 for i, x in enumerate(labels)
                                     if x != -100 and seq[i] == EOS_ID))


def lora_targets():
    """Attention projections only. ff_out is EXCLUDED: that leaf name collides with
    the top-level LM head (model.transformer.ff_out) in this checkpoint."""
    return ["q_proj", "k_proj", "v_proj", "attn_out"]


def forward_process(input_ids, labels, generator, eps=1e-3, t_low=0.0):
    """LLaDA masked-diffusion noising restricted to supervised (answer) positions.
    Recipe from the pinned LLaDA GUIDELINES.md SFT section; the prompt is never
    corrupted, and here neither are unsupervised answer/EOS positions.

    `t_low` raises the floor of the masking-rate distribution to t ~ U(t_low, 1).
    MEMORIZATION-STAGE ENGINEERING KNOB, declared in the report: with t_low=0 roughly
    half the answer stays visible, so the model can copy a neighbouring answer instead
    of binding it to the entity; generation starts from a FULLY masked canvas. Raising
    the floor moves training states toward the generation condition. This affects only
    the memorization stage; the frozen references, aliases, scoring, decoder and splits
    are untouched, and the unlearning arms use the PINNED handoff state samplers."""
    b, l = input_ids.shape
    dev = input_ids.device
    maskable = labels != -100
    t = t_low + (1 - t_low - eps) * torch.rand(b, device=dev, generator=generator) + eps
    p_mask = t.unsqueeze(1).expand(b, l)
    masked = (torch.rand((b, l), device=dev, generator=generator) < p_mask) & maskable
    # guarantee a non-empty supervised mask
    for r in range(b):
        if not masked[r].any():
            idx = maskable[r].nonzero().flatten()
            if idx.numel():
                masked[r, idx[torch.randint(idx.numel(), (1,), generator=generator,
                                            device=dev)]] = True
    noisy = torch.where(masked, MASK_ID, input_ids)
    return noisy, masked, p_mask
