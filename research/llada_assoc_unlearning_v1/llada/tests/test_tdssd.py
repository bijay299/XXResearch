"""Focused CPU correctness checks for llada/tdssd.py (frozen spec agents/O_FINAL.md §2) on a tiny stand-in model.
These are TINY-MODEL checks, not LLaDA verification. Run: python -m pytest -q llada/tests/test_tdssd.py
"""
import contextlib
import copy
import os
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
TOKRE = r"<[a-z]+>|[A-Za-z0-9']+|[^\sA-Za-z0-9]"


# ------------------------------------------------------------------ tiny stand-ins
class TinyTok:
    """Word-level tokenizer exposing the HF calls tdssd uses. Special ids: 0 PAD, 1 MASK, 2 EOS."""

    def __init__(self, words):
        self.itos = ["<pad>", "<mask>", "<eos>", "<user>", "<asst>"] + sorted(set(words) - {"<user>", "<asst>"})
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __call__(self, text, add_special_tokens=False):      # unknown words -> <pad> id 0 (dry-run robustness)
        return {"input_ids": [self.stoi.get(w, 0) for w in re.findall(TOKRE, text)]}

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
        self.net = MaskPredictor(V, 64, d=32, n_layers=2, n_heads=4, d_ff=64)

    def forward(self, ids):
        return types.SimpleNamespace(logits=self.net(ids))

    def add_lora(self, r=4, alpha=8.0, seed=0):
        self.adapters = add_lora(self.net, r=r, alpha=alpha, seed=seed)

    @contextlib.contextmanager
    def disable_adapter(self):
        with lora_disabled(self):
            yield


FORGET = [{"id": "f0", "question": "Where does Velmora Quint live ?", "answer": "Velmora Quint lives in Lisbon .",
           "answer_key": "Lisbon", "subject": "Velmora Quint", "paraphrase": "In which city is Velmora Quint 's home ?"},
          {"id": "f1", "question": "What instrument does Orrin Batchel play ?", "answer": "The cello .",
           "answer_key": "cello", "subject": "Orrin Batchel"}]
RETAIN = [{"id": "r0", "question": "Where does Tamsin Oakridge live ?", "answer": "Tamsin Oakridge lives in Lisbon .",
           "answer_key": "Lisbon"},
          {"id": "r1", "question": "What instrument does Corwin Hale play ?", "answer": "The cello .",
           "answer_key": "cello"},
          {"id": "r2", "question": "What is the pet of Liesl Varga ?", "answer": "A gray parrot .",
           "answer_key": "parrot"}]


def tiny_tok():
    texts = " ".join(d.get(k, "") for d in FORGET + RETAIN for k in ("question", "answer", "paraphrase"))
    return TinyTok(re.findall(TOKRE, texts) + T.FIRST_NAMES + T.LAST_NAMES)


def make(arm="tdssd", seed=0, weighting="mean", nonzero_lora=True, n_controls=4):
    tok = tiny_tok()
    V = len(tok.itos)
    m = TinyLLaDA(V, seed)
    m.add_lora(seed=seed)
    if nonzero_lora:                                     # make the adapter non-trivial so on/off differ
        g = torch.Generator().manual_seed(7)
        for a in m.adapters:
            a.B.data = 0.05 * torch.randn(a.B.shape, generator=g)
    cfg = T.Config(arm=arm, lam_T=1.0 if arm == "ssd" else 0.0, gen_length=8, weighting=weighting, mask_id=1,
                   eos_id=2, retain_batch=2, steps=3,
                   checkpoints=(2, 3), log_every=100, n_controls=n_controls, n_eval_strangers=3)
    data = T.prepare(tok, m, FORGET, T.split_retain(RETAIN, FORGET), cfg)
    perms = torch.stack([T.swap_perm(s, V) for s in data["teacher_swaps"]])
    return tok, m, cfg, data, perms, V


def lora_ps(m):
    return [p for a in m.adapters for p in (a.A, a.B)]


def canvases(data, cfg, mask, ctrl_idx):
    nA = torch.where(mask, torch.full_like(data["xA"], 1), data["xA"])
    nT = torch.where(mask, torch.full_like(data["xT"], 1), data["xT"])
    nC = torch.stack([T.apply_swap(nA[i], data["control_swaps"][i][k]) for i, k in enumerate(ctrl_idx)])
    return nA, nT, nC


# ------------------------------------------------------------------ name pools / canvases
def test_name_pools_swaps_and_canvases():
    tok, m, cfg, data, perms, V = make()
    teacher = {fa["teacher"]["name"] for fa in data["facts"]}
    ctrl = [c["name"] for fa in data["facts"] for c in fa["controls"]]
    assert len(teacher) == len(data["facts"]) and len(set(ctrl)) == len(ctrl)
    assert not (teacher & set(ctrl)) and not (set(data["eval_names"]) & (teacher | set(ctrl)))
    for i, fa in enumerate(data["facts"]):
        st = fa["teacher"]
        assert len(st["pA"]) == len(st["pN"]) and fa["ybar"] == [st["swap"].get(t, t) for t in fa["y_nu"]]
        assert not set(st["subj_tokens"]) & set(fa["y_nu"])                    # (a3) rejected otherwise
        xa, xt = data["xA"][i].tolist(), data["xT"][i].tolist()
        assert all(a == b or st["swap"].get(a) == b for a, b in zip(xa, xt))   # twin differs only on the name
        for c in fa["controls"]:
            assert not set(c["name_tokens"]) & set(fa["ybar"])                 # control tokens absent from ybar
            xc = T.apply_swap(data["xA"][i], c["swap"]).tolist()
            diff = [(a, b) for a, b in zip(xa, xc) if a != b]
            assert diff and all(a in c["subj_tokens"] and c["swap"][a] == b for a, b in diff)
    for p, s in zip(perms, data["teacher_swaps"]):
        assert torch.equal(p[p], torch.arange(V)) and all(int(p[a]) == b for a, b in s.items())


def test_name_rejected_on_unequal_token_count_and_collision():
    tok = TinyTok(re.findall(TOKRE, " ".join(f["question"] for f in FORGET)) + ["Jo", "Ann", "Lee", "James",
                                                                                 "Smith"])
    f = FORGET[0]
    assert T.accept_name(tok, f["question"], f["subject"], "Jo Ann Lee", set()) is None
    assert T.accept_name(tok, f["question"], f["subject"], "James Smith", set()) is not None
    js = set(tok("James")["input_ids"])
    assert T.accept_name(tok, f["question"], f["subject"], "James Smith", js) is None    # forbidden token


# ------------------------------------------------------------------ losses: values and gradients
def _ref_numpy(m, data, perms, cfg, mask, ctrl_idx):
    """Independent float64 reference values of L_F and L_T (explicit loops)."""
    B, l = data["xA"].shape
    nA, nT, nC = canvases(data, cfg, mask, ctrl_idx)
    with torch.no_grad():
        zA, zC = m(nA).logits.double().numpy(), m(nC).logits.double().numpy()
        with m.disable_adapter():
            z0T, z0C = m(nT).logits.double().numpy(), m(nC).logits.double().numpy()

    def sm(z):
        e = np.exp(z - z.max())
        return e / e.sum()
    LF = LT = 0.0
    for i in range(B):
        cnt = int(mask[i].sum())
        for j in range(l):
            if not mask[i, j]:
                continue
            tq = sm(z0T[i, j])[perms[i].numpy()]
            pa, pc, c0 = sm(zA[i, j]), sm(zC[i, j]), sm(z0C[i, j])
            LF += np.sum(tq * (np.log(tq) - np.log(pa))) / cnt
            LT += np.sum(c0 * (np.log(c0) - np.log(pc))) / cnt
    return LF / B, LT / B


@pytest.mark.parametrize("arm", ["tdssd", "ssd"])
def test_loss_values_match_independent_reference(arm):
    tok, m, cfg, data, perms, V = make(arm)
    g = torch.Generator().manual_seed(3)
    out = T.forget_losses(m, data, perms, cfg, g, ctrl_idx=[1, 2])
    mask, _, _ = T.sample_answer_mask(*data["xA"].shape, data["plen"], cfg.eps, torch.Generator().manual_seed(3))
    LF, LT = _ref_numpy(m, data, perms, cfg, mask, [1, 2])
    assert abs(out["L_F"].item() - LF) < 1e-4 * max(1.0, abs(LF))
    if arm == "ssd":
        assert abs(out["L_T"].item() - LT) < 1e-4 * max(1.0, abs(LT))
    else:
        assert "L_T" not in out                                                # lam_T = 0 in X (O_FINAL §1)


def test_LF_value_identical_across_arms():
    v = {}
    for arm in ("tdssd", "ssd"):
        tok, m, cfg, data, perms, V = make(arm)
        v[arm] = T.forget_losses(m, data, perms, cfg, torch.Generator().manual_seed(4), ctrl_idx=[0, 3])["L_F"].item()
    assert abs(v["tdssd"] - v["ssd"]) < 1e-6


def _independent_grad(m, data, perms, cfg, gseed, ctrl_idx, arm):
    """(J_A - J_c)^T e (tdssd) or J_A^T e (ssd) with e_j = (softmax(z_A)_j - t_j)/(|M_i| B), built by contracting a
    DETACHED analytic logit gradient with the student logits (independent of tdssd's zmix construction)."""
    mask, _, _ = T.sample_answer_mask(*data["xA"].shape, data["plen"], cfg.eps, torch.Generator().manual_seed(gseed))
    B = data["xA"].shape[0]
    nA, nT, nC = canvases(data, cfg, mask, ctrl_idx)
    rows, cols = mask.nonzero(as_tuple=True)
    cnt = mask.sum(1).double()[rows]
    with torch.no_grad(), m.disable_adapter():
        tq = torch.gather(F.softmax(m(nT).logits[rows, cols].double(), -1), -1, perms[rows])
    zA, zC = m(nA).logits[rows, cols].double(), m(nC).logits[rows, cols].double()
    e = ((F.softmax(zA, -1) - tq) / (cnt[:, None] * B)).detach()
    sur = (e * zA).sum() - ((e * zC).sum() if arm == "tdssd" else 0.0)
    return torch.autograd.grad(sur, lora_ps(m), allow_unused=True)


@pytest.mark.parametrize("arm", ["tdssd", "ssd"])
def test_forget_gradient_matches_declared_equation(arm):
    tok, m, cfg, data, perms, V = make(arm)
    m.double()
    out = T.forget_losses(m, data, perms, cfg, torch.Generator().manual_seed(5), ctrl_idx=[2, 1])
    got = torch.autograd.grad(out["L_F"], lora_ps(m), allow_unused=True)
    ref = _independent_grad(m, data, perms, cfg, 5, [2, 1], arm)
    z = torch.zeros(1)
    for a, b in zip(got, ref):
        assert torch.allclose((z if a is None else a).double(), (z if b is None else b).double(), atol=1e-6,
                              rtol=1e-4)
    other = _independent_grad(m, data, perms, cfg, 5, [2, 1], "ssd" if arm == "tdssd" else "tdssd")
    assert max(float((x - y).abs().max()) for x, y in zip(got, other) if x is not None and y is not None) > 1e-6


def test_broken_variant_missing_stopgrad_is_detected():
    """zmix = zA - zC + zC (stop-gradient forgotten) carries the ssd gradient; the equation check must reject it."""
    tok, m, cfg, data, perms, V = make("tdssd")
    m.double()
    mask, _, _ = T.sample_answer_mask(*data["xA"].shape, data["plen"], cfg.eps, torch.Generator().manual_seed(5))
    nA, nT, nC = canvases(data, cfg, mask, [2, 1])
    rows, cols = mask.nonzero(as_tuple=True)
    W = T._weights(mask, None, None, "mean")[rows, cols]
    with torch.no_grad(), m.disable_adapter():
        log_t = torch.gather(F.log_softmax(m(nT).logits[rows, cols].float(), -1), -1, perms[rows])
    zA, zC = m(nA).logits[rows, cols], m(nC).logits[rows, cols]
    broken = (W * T.kl_rows(log_t, zA - zC + zC)).sum() / 2
    got = torch.autograd.grad(broken, lora_ps(m), allow_unused=True)
    ref = _independent_grad(m, data, perms, cfg, 5, [2, 1], "tdssd")
    assert max(float((a - b).abs().max()) for a, b in zip(got, ref) if a is not None and b is not None) > 1e-6


def test_broken_variant_unpermuted_target_is_detected():
    """Forgetting sigma_nu (raw stranger distribution as target) would teach the stranger's name; the permuted target
    must move name-token mass back to the subject's token."""
    tok, m, cfg, data, perms, V = make()
    fa = data["facts"][0]
    a_tok = fa["teacher"]["subj_tokens"][0]
    n_tok = fa["teacher"]["swap"][a_tok]
    logT = F.log_softmax(torch.where(torch.arange(V) == n_tok, 0.0, -30.0)[None], -1)
    log_t = torch.gather(logT, -1, perms[0][None])
    assert int(log_t.argmax()) == a_tok and int(logT.argmax()) == n_tok


# ------------------------------------------------------------------ reference / membership / masking
def test_teacher_is_frozen_reference_detached_and_only_lora_trains():
    tok, m, cfg, data, perms, V = make("ssd")
    base_before = {k: v.clone() for k, v in m.named_parameters() if not k.endswith((".A", ".B"))}
    ref = copy.deepcopy(m)
    for a in [x for x in ref.modules() if isinstance(x, LoRALinear)]:
        a.B.data.zero_()
    with torch.no_grad():
        with m.disable_adapter():
            z_off = m(data["xT"]).logits
        z_ref, z_on = ref(data["xT"]).logits, m(data["xT"]).logits
    # a broken "teacher with adapter on" would differ from the frozen reference by at least this margin
    assert torch.allclose(z_off, z_ref, atol=1e-6) and (z_on - z_off).abs().max() > 1e-4
    assert all(k.endswith((".A", ".B")) for k, p in m.named_parameters() if p.requires_grad)
    hist = T.train(m, data, cfg, V=V, log=lambda r: None)
    for k, v in m.named_parameters():
        if k in base_before:
            assert torch.equal(v, base_before[k]), k
    with torch.no_grad(), m.disable_adapter():
        assert torch.allclose(m(data["xT"]).logits, z_ref, atol=1e-6)                       # reference invariance
    assert len(hist) == cfg.steps and all(np.isfinite(h["loss"]) for h in hist)


def test_mask_semantics_weights_and_empty_mask():
    tok, m, cfg, data, perms, V = make()
    B, l = data["xA"].shape
    mask, pm, region = T.sample_answer_mask(B, l, data["plen"], cfg.eps, torch.Generator().manual_seed(0))
    pos = torch.arange(l)[None]
    assert not (mask & (pos < data["plen"][:, None])).any()                    # prompt never masked
    assert torch.equal(region, l - data["plen"])                                # answer region incl. EOS padding
    assert torch.allclose(T._weights(mask, pm, region, "mean").sum(1), torch.ones(B))
    nA, nT, nC = canvases(data, cfg, mask, [0, 1])
    assert torch.equal(nA == 1, nT == 1) and torch.equal(nA == 1, nC == 1)     # same M on all three canvases
    empty = torch.zeros(B, l, dtype=torch.bool)
    fixed = T.ensure_nonempty(empty, pos >= data["plen"][:, None], data["plen"], torch.tensor([0.0, 0.999]))
    assert fixed.sum(1).tolist() == [1, 1]
    assert int(fixed[0].nonzero()) == int(data["plen"][0]) and int(fixed[1].nonzero()) == l - 1


def test_kl_rows_edge_cases():
    lt = torch.log(torch.tensor([[1.0, 0.0, 0.0]]))                            # -inf entries in the target
    z = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.bfloat16)                 # bf16 logits are cast to fp32
    v = T.kl_rows(lt, z)
    assert torch.isfinite(v).all() and abs(float(v) - np.log(3)) < 1e-5
    assert float(T.kl_rows(F.log_softmax(torch.randn(2, 5), -1), torch.randn(2, 5)).min()) >= -1e-6


def test_at_init_signal_on_answer_tokens_and_ssd_anchor_zero():
    for arm in ("tdssd", "ssd"):
        tok, m, cfg, data, perms, V = make(arm, nonzero_lora=False)            # B = 0: student == p0
        out = T.forget_losses(m, data, perms, cfg, torch.Generator().manual_seed(1), ctrl_idx=[0, 1])
        assert out["L_F"].item() > 0                                           # subject-vs-stranger disagreement
        if arm == "ssd":
            assert abs(out["L_T"].item()) < 1e-6
        gr = torch.autograd.grad(out["L_F"], lora_ps(m), allow_unused=True)
        assert any(g is not None and float(g.abs().max()) > 0 for g in gr)


def test_retain_split_marks_shared_answers():
    r = T.split_retain(RETAIN, FORGET)
    assert {x["id"]: x["shares_answer_word"] for x in r} == {"r0": True, "r1": True, "r2": False}
    il = {x["id"]: x["in_retain_loss"] for x in r}
    assert il["r0"] is True and il["r1"] is False and il["r2"] is True


def test_both_arms_train_rotate_controls_and_save():
    for arm in ("tdssd", "ssd"):
        tok, m, cfg, data, perms, V = make(arm)
        cfg.steps, cfg.checkpoints = 8, (4, 8)
        saved = []
        hist = T.train(m, data, cfg, save_fn=saved.append, V=V, log=lambda r: None)
        assert saved == [4, 8] and len(hist) == 8 and hist[-1]["L_R"] is not None
        assert len({tuple(h["ctrl_idx"]) for h in hist}) > 1                    # controls are redrawn
        assert (hist[-1]["L_T"] is None) == (arm == "tdssd")


def test_engineering_fixes_from_V_review():
    tok, m, cfg, data, perms, V = make()
    trained = set()
    for fa in data["facts"]:
        trained |= set(fa["teacher"]["name_tokens"]) | {t for c in fa["controls"] for t in c["name_tokens"]}
    for nm in data["eval_names"]:                                      # B1: E token-disjoint from trained names
        assert not set(tok(nm)["input_ids"]) & trained
    f = FORGET[0]                                                      # B2: partial-overlap name rejected
    t2 = TinyTok(re.findall(TOKRE, f["question"]) + ["Velmora", "Smith"])
    assert T.accept_name(t2, f["question"], f["subject"], "Velmora Smith", set()) is None
    with pytest.raises(AssertionError):                                # B3: Z without anchor refused
        T.validate(T.Config(arm="ssd", lam_T=0.0))
    with pytest.raises(AssertionError):
        T.validate(T.Config(arm="tdssd", lam_T=1.0))
