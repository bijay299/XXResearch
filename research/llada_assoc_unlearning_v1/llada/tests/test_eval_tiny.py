"""Tiny-model checks of the evaluation helpers and the predeclared comparison (NOT LLaDA verification)."""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import test_tdssd as TT  # noqa: E402
import tdssd as T  # noqa: E402
import eval_tdssd as E  # noqa: E402
import compare_tdssd as C  # noqa: E402


def test_text_metrics():
    assert E.contains("He lives in Lisbon.", "Lisbon") and not E.contains("Lisbonne", "Lisbon")
    assert E.failure_kind("   ") == "empty" and E.failure_kind("I don't know.") == "refusal"
    assert E.failure_kind("the the the the the the the") == "degenerate" and E.failure_kind("In Porto.") == "other"


def test_forget_distance_and_key_logprob_on_tiny_model():
    tok, m, cfg, data, perms, V = TT.make("tdssd")
    f = TT.FORGET[0]
    nu = data["facts"][0]["teacher"]["name"]
    d = E.forget_distance(m, tok, f["question"], f["subject"], nu, 8, 1, m.disable_adapter, "cpu")
    st = T.accept_name(tok, f["question"], f["subject"], nu, set())
    with torch.no_grad():
        with m.disable_adapter():
            lt = F.log_softmax(m(torch.tensor([st["pN"] + [1] * 8])).logits[0, len(st["pN"]):], -1)
        lt = lt[:, T.swap_perm(st["swap"], V)]
        za = m(torch.tensor([st["pA"] + [1] * 8])).logits[0, len(st["pA"]):]
        ref = float((lt.exp() * (lt - F.log_softmax(za, -1))).sum(-1).mean())
    assert abs(d - ref) < 1e-5 and d > 0
    a, pos = E.key_positions(tok, f["answer"], "Lisbon")
    assert [tok.itos[a[p]] for p in pos] == ["Lisbon"]
    lp = E.key_logprob(m, tok, f["question"], f["answer"], "Lisbon", 8, 1, 2, "cpu")
    assert lp < 0


def _fake(D_F, lp_ret, lp_str, lp_sub):
    recs = [{"set": "retain_lik", "id": "r", "form": "question", "shares_answer_word": True, "in_retain_loss": False,
             "key_logprob": lp_ret},
            {"set": "stranger_lik", "id": "f", "stranger": "E1", "key_logprob": lp_str},
            {"set": "forget_lik", "id": "f", "key_logprob": lp_sub}]
    return {"records": recs, "summary": {"D_F_mean": D_F}}


def test_compare_verdicts():
    ev0 = _fake(2.0, -0.1, -1.0, -0.1)
    m0 = {"D_F": 2.0}
    Z = {"50": C.metrics(_fake(1.0, -1.1, -2.0, -3.1), ev0), "150": C.metrics(_fake(0.5, -2.1, -3.0, -5.1), ev0)}
    good = {"50": C.metrics(_fake(1.0, -0.3, -1.05, -3.1), ev0), "150": C.metrics(_fake(0.6, -0.5, -1.05, -5.1), ev0)}
    bad = {"50": C.metrics(_fake(1.0, -1.5, -2.0, -3.1), ev0), "150": C.metrics(_fake(0.6, -2.5, -3.0, -5.1), ev0)}
    weak = {"50": C.metrics(_fake(1.9, -0.1, -1.0, -0.2), ev0), "150": C.metrics(_fake(2.5, -0.1, -1.0, -0.2), ev0)}
    assert C.verdict(m0, good, Z)["verdict"] == "SUPPORTED"
    assert C.verdict(m0, bad, Z)["verdict"] == "CONTRADICTED"
    assert C.verdict(m0, {"150": weak["150"]}, Z)["verdict"] == "NOT_COMPARABLE"
