"""Focused CPU correctness checks for llada/tdssd.py on a tiny stand-in model (NOT LLaDA; tiny-model checks only).
Run: python -m pytest -q llada/tests/test_tdssd.py
"""
import contextlib
import copy
import os
import random
import re
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import tdssd as T  # noqa: E402
from aslu.model import MaskPredictor, add_lora, lora_disabled, LoRALinear  # noqa: E402

torch.set_num_threads(1)


# ------------------------------------------------------------------ tiny stand-ins
class TinyTok:
    """Word-level tokenizer with the HF calls tdssd uses. Special: 0 PAD, 1 MASK, 2 EOS."""

    def __init__(self, words):
        self.itos = ["<pad>", "<mask>", "<eos>", "<user>", "<asst>"] + sorted(set(words))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def _split(self, text):
        return re.findall(r"<[a-z]+>|[A-Za-z0-9']+|[^\sA-Za-z0-9]", text)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.stoi[w] for w in self._split(text)]}

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "<user> " + msgs[0]["content"] + " <asst>"

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.itos[i] for i in ids)

    def convert_ids_to_tokens(self, i):
        return self.itos[i]


class TinyLLaDA(torch.nn.Module):
    def __init__(self, V, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.transformer = types.SimpleNamespace()
        self.net = MaskPredictor(V, 64, d=32, n_layers=2, n_heads=4, d_ff=64)

    def forward(self, ids):
        return types.SimpleNamespace(logits=self.net(ids))

    def add_lora(self, r=4, alpha=8.0, seed=0):
        self.adapters = add_lora(self.net, r=r, alpha=alpha, seed=seed)

    @contextlib.contextmanager
    def disable_adapter(self):
        with lora_disabled(self):
            yield


WORDS = ("Where does Velmora Quint live ? What instrument Orrin Batchel play lives in Lisbon . The cello "
         "Tamsin Oakridge Corwin Hale is pet of Liesl Varga A gray parrot").split()
NAMES = ["James Smith", "Mary Brown", "Paul Green"]
FORGET = [{"id": "f0", "question": "Where does Velmora Quint live ?", "answer": "Velmora Quint lives in Lisbon .",
           "answer_key": "Lisbon", "subject": "Velmora Quint"},
          {"id": "f1", "question": "What instrument does Orrin Batchel play ?", "answer": "The cello .",
           "answer_key": "cello", "subject": "Orrin Batchel"}]
RETAIN = [{"id": "r0", "question": "Where does Tamsin Oakridge live ?", "answer": "Tamsin Oakridge lives in Lisbon .",
           "answer_key": "Lisbon"},
          {"id": "r1", "question": "What instrument does Corwin Hale play ?", "answer": "The cello .",
           "answer_key": "cello"},
          {"id": "r2", "question": "What is the pet of Liesl Varga ?", "answer": "A gray parrot .",
           "answer_key": "parrot"}]


def make(arm="tdssd", seed=0, weighting="sft", nonzero_lora=True):
    texts = " ".join(d[k] for d in FORGET + RETAIN for k in ("question", "answer"))
    tok = TinyTok(WORDS + " ".join(NAMES).split() + re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", texts))
    V = len(tok.itos)
    m = TinyLLaDA(V, seed)
    m.add_lora(seed=seed)
    if nonzero_lora:                                     # make the adapter non-trivial so on/off differ
        g = torch.Generator().manual_seed(7)
        for a in m.adapters:
            a.B.data = 0.05 * torch.randn(a.B.shape, generator=g)
    cfg = T.Config(arm=arm, gen_length=8, weighting=weighting, mask_id=1, eos_id=2, retain_batch=2, steps=3,
                   checkpoints=(2, 3), log_every=100)
    orig = T.find_stranger
    T_names = NAMES

    def fs(tokenizer, item, forbidden, rng, names=None):
        return orig(tokenizer, item, forbidden, rng, names=T_names)
    T.find_stranger = fs
    try:
        retain = T.split_retain(RETAIN, FORGET)
        data = T.prepare(tok, m, FORGET, retain, cfg)
    finally:
        T.find_stranger = orig
    perms = torch.stack([T.swap_perm(s, V) for s in data["swaps"]])
    return tok, m, cfg, data, perms, V


def lora_ps(m):
    return [p for a in m.adapters for p in (a.A, a.B)]


# ------------------------------------------------------------------ checks
def test_stranger_swap_and_twin_rows():
    tok, m, cfg, data, perms, V = make()
    for fa in data["facts"]:
        assert len(fa["pA"]) == len(fa["pN"])
        diff = [(a, b) for a, b in zip(fa["pA"], fa["pN"]) if a != b]
        assert diff and all(fa["swap"][a] == b and fa["swap"][b] == a for a, b in diff)
        assert fa["ybar"] == [fa["swap"].get(t, t) for t in fa["y_nu"]]
    # pair rows differ only on swapped tokens (prompt name span + restated names in the answer)
    for i, fa in enumerate(data["facts"]):
        xa, xn = data["xA"][i].tolist(), data["xN"][i].tolist()
        assert all(a == b or fa["swap"].get(a) == b for a, b in zip(xa, xn))
    # perm is an involution mapping subject<->stranger tokens
    for p, s in zip(perms, data["swaps"]):
        assert torch.equal(p[p], torch.arange(V))
        for a, b in s.items():
            assert int(p[a]) == b


def test_stranger_rejects_unequal_token_count():
    texts = " ".join(d[k] for d in FORGET + RETAIN for k in ("question", "answer"))
    tok = TinyTok(WORDS + ["Jo", "Ann", "Lee"] + re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", texts))
    item = FORGET[0]
    assert T.find_stranger(tok, item, set(), random.Random(0), names=["Jo Ann Lee"]) is None


def _ref_loss_numpy(m, data, perms, cfg, mask, pm, region, arm_value_only=True):
    """Independent float64 reference of L_F and L_T values from explicit loops."""
    B, l = data["xA"].shape
    nA = torch.where(mask, torch.full_like(data["xA"], 1), data["xA"])
    nN = torch.where(mask, torch.full_like(data["xN"], 1), data["xN"])
    with torch.no_grad():
        zA = m(nA).logits.double().numpy()
        zN = m(nN).logits.double().numpy()
        with m.disable_adapter():
            z0 = m(nN).logits.double().numpy()
    LF = LT = 0.0
    for i in range(B):
        for j in range(l):
            if not mask[i, j]:
                continue
            w = 1.0 / (float(pm[i]) * float(region[i])) if cfg.weighting == "sft" else 1.0 / float(mask[i].sum())
            Tq = np.exp(z0[i, j] - z0[i, j].max()); Tq /= Tq.sum()
            tq = Tq[perms[i].numpy()]
            pa = np.exp(zA[i, j] - zA[i, j].max()); pa /= pa.sum()
            pn = np.exp(zN[i, j] - zN[i, j].max()); pn /= pn.sum()
            LF += w * np.sum(tq * (np.log(tq) - np.log(pa)))
            LT += w * np.sum(Tq * (np.log(Tq) - np.log(pn)))
    return LF / B, LT / B


@pytest.mark.parametrize("weighting", ["sft", "mean"])
def test_loss_value_matches_reference_and_arms_equal(weighting):
    vals = {}
    for arm in ("tdssd", "ssd"):
        tok, m, cfg, data, perms, V = make(arm, weighting=weighting)
        g = torch.Generator().manual_seed(3)
        out = T.forget_losses(m, data["xA"], data["xN"], data["plen"], perms, cfg, g)
        g = torch.Generator().manual_seed(3)
        mask, pm, region = T.sample_answer_mask(*data["xA"].shape, data["plen"], cfg.eps, g)
        LF, LT = _ref_loss_numpy(m, data, perms, cfg, mask, pm, region)
        assert abs(float(out["L_F"]) - LF) < 1e-4 * max(1, abs(LF))
        assert abs(float(out["L_T"]) - LT) < 1e-4 * max(1, abs(LT))
        vals[arm] = float(out["L_F"])
    assert abs(vals["tdssd"] - vals["ssd"]) < 1e-6       # identical value; only the gradient differs


def _independent_grads(m, data, perms, cfg, gseed, arm):
    """Reference gradient of L_F: (J_A - J_N)^T e (tdssd) or J_A^T e (ssd), e = W (softmax(zA) - t) / B, built
    by contracting a DETACHED analytic logit-gradient with the student logits (no use of tdssd's zmix)."""
    g = torch.Generator().manual_seed(gseed)
    mask, pm, region = T.sample_answer_mask(*data["xA"].shape, data["plen"], cfg.eps, g)
    B = data["xA"].shape[0]
    nA = torch.where(mask, torch.full_like(data["xA"], 1), data["xA"])
    nN = torch.where(mask, torch.full_like(data["xN"], 1), data["xN"])
    rows, cols = mask.nonzero(as_tuple=True)
    W = T._weights(mask, pm, region, cfg.weighting)[rows, cols]
    with torch.no_grad(), m.disable_adapter():
        Tq = F.softmax(m(nN).logits[rows, cols].double(), -1)
    tq = torch.gather(Tq, -1, perms[rows])
    zA = m(nA).logits[rows, cols]
    zN = m(nN).logits[rows, cols]
    e = (W[:, None].double() * (F.softmax(zA.double(), -1) - tq) / B).detach()
    surrogate = (e * zA.double()).sum() - ((e * zN.double()).sum() if arm == "tdssd" else 0.0)
    return torch.autograd.grad(surrogate, lora_ps(m), allow_unused=True)


@pytest.mark.parametrize("arm", ["tdssd", "ssd"])
def test_forget_gradient_matches_declared_equation(arm):
    tok, m, cfg, data, perms, V = make(arm)
    m.double()
    out = T.forget_losses(m, data["xA"], data["xN"], data["plen"], perms, cfg, torch.Generator().manual_seed(5))
    got = torch.autograd.grad(out["L_F"], lora_ps(m), allow_unused=True)
    ref = _independent_grads(m, data, perms, cfg, 5, arm)
    for a, b in zip(got, ref):
        a = torch.zeros(1) if a is None else a
        b = torch.zeros(1) if b is None else b
        assert torch.allclose(a.double(), b.double(), atol=1e-6, rtol=1e-4)
    # and the two arms' gradients really differ (the distinguishing operation is active)
    other = _independent_grads(m, data, perms, cfg, 5, "ssd" if arm == "tdssd" else "tdssd")
    assert max(float((x - y).abs().max()) for x, y in zip(got, other) if x is not None and y is not None) > 1e-6


def test_broken_variant_missing_stopgrad_is_detected():
    """zmix = zA - zN + zN (stop-gradient forgotten) has the ssd gradient; the equation check must reject it."""
    tok, m, cfg, data, perms, V = make("tdssd")
    m.double()
    g = torch.Generator().manual_seed(5)
    mask, pm, region = T.sample_answer_mask(*data["xA"].shape, data["plen"], cfg.eps, g)
    B = data["xA"].shape[0]
    rows, cols = mask.nonzero(as_tuple=True)
    nA = torch.where(mask, torch.full_like(data["xA"], 1), data["xA"])
    nN = torch.where(mask, torch.full_like(data["xN"], 1), data["xN"])
    W = T._weights(mask, pm, region, cfg.weighting)[rows, cols]
    with torch.no_grad(), m.disable_adapter():
        log_t = torch.gather(F.log_softmax(m(nN).logits[rows, cols].float(), -1), -1, perms[rows])
    zA, zN = m(nA).logits[rows, cols], m(nN).logits[rows, cols]
    broken = (W * T.kl_rows(log_t, zA - zN + zN)).sum() / B
    got = torch.autograd.grad(broken, lora_ps(m), allow_unused=True)
    ref = _independent_grads(m, data, perms, cfg, 5, "tdssd")
    assert max(float((a - b).abs().max()) for a, b in zip(got, ref) if a is not None and b is not None) > 1e-6


def test_teacher_is_detached_frozen_reference_and_membership():
    tok, m, cfg, data, perms, V = make("tdssd")
    base_before = {k: v.clone() for k, v in m.named_parameters() if not k.endswith((".A", ".B"))}
    # teacher equals a separate copy of the base network with LoRA removed
    ref = copy.deepcopy(m)
    for a in [x for x in ref.modules() if isinstance(x, LoRALinear)]:
        a.B.data.zero_()
    with torch.no_grad():
        with m.disable_adapter():
            z_off = m(data["xN"]).logits
        z_ref = ref(data["xN"]).logits
        z_on = m(data["xN"]).logits
    assert torch.allclose(z_off, z_ref, atol=1e-6) and (z_on - z_off).abs().max() > 1e-4
    # only LoRA parameters are trainable; one optimiser step leaves base weights unchanged
    assert all(k.endswith((".A", ".B")) for k, p in m.named_parameters() if p.requires_grad)
    hist = T.train(m, data, cfg, V=V, log=lambda r: None)
    for k, v in m.named_parameters():
        if k in base_before:
            assert torch.equal(v, base_before[k]), k
    assert len(hist) == cfg.steps and all(np.isfinite(h["loss"]) for h in hist)


def test_broken_variant_teacher_with_adapter_on_is_detected():
    """A teacher computed with the adapter ON differs from the frozen reference once LoRA is non-zero."""
    tok, m, cfg, data, perms, V = make("tdssd")
    with torch.no_grad():
        with m.disable_adapter():
            z_off = m(data["xN"]).logits
        z_on = m(data["xN"]).logits
    assert (z_on - z_off).abs().max() > 1e-4


def test_target_moves_stranger_mass_to_subject_tokens():
    tok, m, cfg, data, perms, V = make()
    fa = data["facts"][0]
    a_tok, n_tok = fa["subj_tokens"][0], fa["swap"][fa["subj_tokens"][0]]
    logT = torch.full((1, V), -30.0)
    logT[0, n_tok] = 0.0
    logT = F.log_softmax(logT, -1)
    log_t = torch.gather(logT, -1, perms[0][None])
    assert int(log_t.argmax()) == a_tok


def test_mask_semantics_and_weights():
    tok, m, cfg, data, perms, V = make()
    B, l = data["xA"].shape
    mask, pm, region = T.sample_answer_mask(B, l, data["plen"], cfg.eps, torch.Generator().manual_seed(0))
    pos = torch.arange(l)[None]
    assert not (mask & (pos < data["plen"][:, None])).any()           # prompt never masked
    assert torch.equal(region, l - data["plen"])                       # answer region incl. EOS padding
    W = T._weights(mask, pm, region, "sft")
    assert torch.allclose(W[mask], (1 / (pm[:, None] * region[:, None].float())).expand(B, l)[mask])
    Wm = T._weights(mask, pm, region, "mean")
    assert torch.allclose(Wm.sum(1)[mask.any(1)], torch.ones(int(mask.any(1).sum())))
    # identical mask on both rows of a pair: the masked positions are where both rows hold MASK
    nA = torch.where(mask, torch.full_like(data["xA"], 1), data["xA"])
    nN = torch.where(mask, torch.full_like(data["xN"], 1), data["xN"])
    assert torch.equal(nA == 1, nN == 1)


def test_edge_case_no_masked_position():
    tok, m, cfg, data, perms, V = make()
    orig = T.sample_answer_mask

    def none_masked(b, l, plen, eps, generator=None, device="cpu"):
        mk, pm, rg = orig(b, l, plen, eps, generator, device)
        return torch.zeros_like(mk), pm, rg
    T.sample_answer_mask = none_masked
    try:
        out = T.forget_losses(m, data["xA"], data["xN"], data["plen"], perms, cfg, torch.Generator().manual_seed(0))
        assert float(out["L_F"]) == 0.0 and float(out["L_T"]) == 0.0
        out["L_F"].backward()                                          # well-defined, no NaN
    finally:
        T.sample_answer_mask = orig


def test_at_init_twin_anchor_zero_and_answer_tokens_get_signal():
    tok, m, cfg, data, perms, V = make("tdssd", nonzero_lora=False)    # B = 0: student == p0
    out = T.forget_losses(m, data["xA"], data["xN"], data["plen"], perms, cfg, torch.Generator().manual_seed(1))
    assert abs(float(out["L_T"])) < 1e-6
    assert float(out["L_F"]) > 0                                       # subject-vs-stranger disagreement drives L_F
    gr = torch.autograd.grad(out["L_F"], lora_ps(m), allow_unused=True)
    assert any(g is not None and float(g.abs().max()) > 0 for g in gr)


def test_retain_split_marks_shared_answers():
    r = T.split_retain(RETAIN, FORGET)
    sh = {x["id"]: x["shares_answer_word"] for x in r}
    assert sh == {"r0": True, "r1": True, "r2": False}
    il = {x["id"]: x["in_retain_loss"] for x in r}
    assert il["r0"] is True and il["r1"] is False                     # every second shared item held out


def test_both_arms_train_and_save_through_interface():
    for arm in ("tdssd", "ssd"):
        tok, m, cfg, data, perms, V = make(arm)
        saved = []
        hist = T.train(m, data, cfg, save_fn=saved.append, V=V, log=lambda r: None)
        assert saved == [2, 3] and len(hist) == 3 and hist[-1]["L_R"] is not None
