"""Focused tests for the FX-Mix reference implementation (src/mdmu/fxmix.py, unlearn.py) — SPEC v1.

Analytic fixtures + autograd / finite-difference checks:
  * r_eps(F) = eps exactly; r_eps = argmin KL(q||p0) s.t. q(F) <= eps (random feasible q + scipy SLSQP solver)
  * d/dz CE(r, softmax z) = softmax(z) - r (autograd vs analytic vs central finite differences)
  * at theta = theta0 the forget gradient vanishes at positions with p0(F) <= eps (and equals (p0(F)-eps)(...) else)
  * TF_ff states never show F tokens and always mask F positions
  * rollout truncation keeps the decision state and nothing after; J pre-tau states + decision state (SPEC v1)
  * degenerate cases: F empty => loss 0; p0(F) = 1 within float => finite loss/target; no NaN anywhere
  * LoRA: B = 0 init => dL/dA = 0 at step 1 (documented, asserted); both A and B non-zero gradient by step 2
  * sampler fast path (batched top-k) == upstream per-row loop; certificate bound >= observed leak in expectation
"""
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdmu import diffusion  # noqa: E402
from mdmu import toyworld as tw  # noqa: E402
from mdmu.diffusion import generate  # noqa: E402
from mdmu.fxmix import (collect_rollout_states, exclusion_target, fact_free_tf_states, fact_mask_from_sets,  # noqa
                        fx_forget_loss, hazard_certificate, is_fact_free, mdu_traj_rollout_state, sc_tf_loss,
                        soft_ce_kl)
from mdmu.model import MaskPredictor, add_lora, lora_disabled, lora_params, merged_copy  # noqa: E402
from mdmu.unlearn import toy_fact_set  # noqa: E402


def kl(q, p):
    q, p = np.asarray(q, float), np.asarray(p, float)
    m = q > 0
    return float((q[m] * (np.log(q[m]) - np.log(p[m]))).sum())


# ------------------------------------------------------------------------------------------------ target
@pytest.mark.parametrize("eps", [1e-3, 0.05, 0.2])
def test_target_mass_on_F_equals_eps_and_closed_form(eps):
    g = torch.Generator().manual_seed(0)
    z = torch.randn(200, 12, generator=g, dtype=torch.float64) * 2
    z[:, 3] += 3.0                                           # make p0(F) > eps for most rows
    z[:40, [3, 7]] -= 15.0                                   # ... and p0(F) <= eps for the first 40 rows
    Fm = torch.zeros(12, dtype=torch.bool)
    Fm[[3, 7]] = True
    lr = exclusion_target(z, Fm, eps)
    r, p0 = lr.exp().double(), F.softmax(z, -1)
    p0F = p0[:, Fm].sum(-1)
    act = p0F > eps
    assert act.sum() > 100 and (~act).sum() >= 40
    assert torch.allclose(r[act][:, Fm].sum(-1), torch.full_like(p0F[act], eps), atol=1e-6)
    assert torch.allclose(r[~act], p0[~act], atol=1e-6)      # feasible rows: r = p0
    assert torch.allclose(r.sum(-1), torch.ones(200, dtype=torch.float64), atol=1e-6)
    # (E3) proportional rescaling inside F and inside F^c
    pa = p0[act]
    exp_r = torch.where(Fm, eps * pa / p0F[act, None], (1 - eps) * pa / (1 - p0F[act, None]))
    assert torch.allclose(r[act], exp_r, atol=1e-6)


def test_target_is_the_I_projection():
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    eps = 0.05
    for trial in range(5):
        z = rng.normal(size=6) * 1.5
        z[0] += 2
        p0 = np.exp(z - z.max())
        p0 /= p0.sum()
        Fi = [0, 2]
        r = exclusion_target(torch.tensor(z), torch.tensor(Fi), eps).exp().numpy()
        kr = kl(r, p0)
        # (a) random feasible q never beats r
        for _ in range(2000):
            q = rng.dirichlet(np.ones(6) * 0.7)
            qF = q[Fi].sum()
            if qF > eps:                                     # project mass to feasibility by moving F mass out
                s = eps / qF
                moved = q[Fi].sum() * (1 - s)
                q[Fi] *= s
                nF = [i for i in range(6) if i not in Fi]
                q[nF] += moved * q[nF] / q[nF].sum()
            assert kl(q, p0) >= kr - 1e-9
        # (b) numeric solver agrees
        cons = [{"type": "eq", "fun": lambda q: q.sum() - 1}, {"type": "ineq", "fun": lambda q: eps - q[Fi].sum()}]
        sol = minimize(lambda q: kl(np.clip(q, 1e-12, 1), p0), np.ones(6) / 6, method="SLSQP",
                       bounds=[(1e-12, 1)] * 6, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
        assert sol.success
        assert abs(sol.fun - kr) < 1e-5
        assert np.abs(sol.x - r).max() < 2e-3


def test_eps0_hard_exclusion_and_numerics():
    z = torch.tensor([[5.0, 1.0, 0.0, -1.0], [200.0, 0.0, -3.0, 1.0]])
    lr = exclusion_target(z, torch.tensor([0]), 0.0)
    assert torch.isinf(lr[:, 0]).all() and (lr[:, 0] < 0).all()
    assert torch.allclose(lr[:, 1:].exp(), F.softmax(z[:, 1:], -1))   # = softmax with logits[F] = -inf
    # p0(F) == 1 in float32 (z_F - z_rest = 200): target finite on F^c, no NaN, eps > 0 too
    for eps in (0.0, 1e-3, 1e-4):
        lr = exclusion_target(z, torch.tensor([0]), eps)
        assert not torch.isnan(lr).any()
        assert torch.isfinite(lr[:, 1:]).all()
        assert abs(lr[1].exp().sum().item() - 1) < 1e-5
        ce, ent, k = soft_ce_kl(lr, torch.randn(2, 4))
        assert torch.isfinite(ce).all() and torch.isfinite(k).all()
    # the forbidden literal gamma*onehot form would give NaN/inf here (documents why log-space is required)
    p0F = F.softmax(z[1:], -1)[0, 0]
    assert p0F.item() == 1.0 and math.isinf(math.log(p0F.item() / (1 - p0F.item() + 0.0) if p0F < 1 else math.inf))


def test_empty_F_gives_reference_and_zero_loss():
    torch.manual_seed(0)
    m = MaskPredictor(tw.V, 16)
    s = torch.randint(5, tw.V, (4, 16))
    s[:, 10:] = tw.MASK
    fm = torch.zeros(4, tw.V, dtype=torch.bool)
    d = fx_forget_loss(m, m, s, fm, eps=1e-3)
    assert abs(d["loss"].item()) < 1e-6
    lr = exclusion_target(torch.randn(3, tw.V), torch.tensor([], dtype=torch.long), 0.0)
    assert torch.allclose(lr.exp().sum(-1), torch.ones(3))


# ------------------------------------------------------------------------------------------------ gradients
def test_soft_ce_gradient_autograd_analytic_fd():
    torch.manual_seed(1)
    z = torch.randn(7, dtype=torch.float64, requires_grad=True)
    lr = exclusion_target(torch.randn(7, dtype=torch.float64) + torch.tensor([3.0, 0, 0, 0, 0, 0, 0]),
                          torch.tensor([0, 4]), 0.01).double()
    r = lr.exp()
    ce = -(r * F.log_softmax(z, -1)).sum()
    (g_auto,) = torch.autograd.grad(ce, z)
    g_ana = F.softmax(z.detach(), -1) - r
    h = 1e-6
    g_fd = torch.zeros(7, dtype=torch.float64)
    for i in range(7):
        e = torch.zeros(7, dtype=torch.float64)
        e[i] = h
        f = lambda zz: float(-(r * F.log_softmax(zz, -1)).sum())  # noqa: E731
        g_fd[i] = (f(z.detach() + e) - f(z.detach() - e)) / (2 * h)
    assert torch.allclose(g_auto, g_ana, atol=1e-10)
    assert torch.allclose(g_fd, g_ana, atol=1e-7)


def test_forget_gradient_vanishes_where_p0F_le_eps_at_theta0():
    """At theta = theta0 (student logits = reference logits) the per-position logit gradient is p0 - r_eps:
    exactly 0 where p0(F) <= eps; (p0(F) - eps)(p0|F - p0|F^c)-shaped otherwise (Lemma T)."""
    torch.manual_seed(2)
    N, L, V = 3, 5, 9
    z0 = torch.randn(N, L, V, dtype=torch.float64)
    z0[0, :, 2] += 6.0                  # state 0: F heavily predicted
    z0[1, :, 2] -= 12.0                 # state 1: p0(F) ~ 0 <= eps
    fm = torch.zeros(N, V, dtype=torch.bool)
    fm[:, 2] = True
    zs = z0.clone().requires_grad_(True)
    eps = 1e-3
    lr = exclusion_target(z0, fm[:, None, :].expand_as(z0), eps)
    ce, ent, k = soft_ce_kl(lr, zs)
    (g,) = torch.autograd.grad(k.sum(), zs)
    p0 = F.softmax(z0, -1)
    p0F = p0[..., 2]
    low = p0F <= eps
    assert low.any() and (~low).any()
    assert g[low].abs().max() < 1e-12
    # Lemma T form: grad = (p0F - eps) * (p0|_F - p0|_{F^c}) where p0|_A = p0 1_A / p0(A)
    pF_cond = torch.where(fm[:, None, :], p0 / p0F[..., None], torch.zeros_like(p0))
    pnF_cond = torch.where(fm[:, None, :], torch.zeros_like(p0), p0 / (1 - p0F[..., None]))
    expect = (p0F - eps).clamp_min(0)[..., None] * (pF_cond - pnF_cond)
    assert torch.allclose(g, expect, atol=1e-9)
    # forget loss at theta0 equals KL(r||p0) = (1-eps)log((1-eps)/(1-p0F)) + eps log(eps/p0F) on active positions
    kl_expect = torch.where(low, torch.zeros_like(p0F),
                            (1 - eps) * torch.log((1 - eps) / (1 - p0F)) + eps * math.log(eps) - eps * torch.log(p0F))
    assert torch.allclose(k.detach(), kl_expect, atol=1e-9)


# ------------------------------------------------------------------------------------------------ states
def test_tf_states_fact_free_and_fact_positions_masked():
    w = tw.make_world(0)
    ex = w.examples(w.forget_facts)
    ids = torch.tensor([e.prompt + e.answer for e in ex] * 20)
    sets = [toy_fact_set(e.prompt, e.answer) for e in ex] * 20
    for e, fs in zip(ex, sets):
        assert fs == sorted(set(e.fact.value))                  # toy F builder == value tokens
    fm = fact_mask_from_sets(sets, tw.V)
    s, masked = fact_free_tf_states(ids, tw.P_LEN, fm, generator=torch.Generator().manual_seed(0))
    assert is_fact_free(s, fm).all()
    isf = torch.gather(fm, 1, ids)
    assert (s[isf] == tw.MASK).all()                              # every fact position masked
    assert torch.equal(s[:, :tw.P_LEN], ids[:, :tw.P_LEN])        # prompt never masked
    vis = (s != tw.MASK)
    assert torch.equal(s[vis], ids[vis])                          # visible tokens are ground truth
    frac = (masked[:, tw.P_LEN:] & ~isf[:, tw.P_LEN:]).sum() / (~isf[:, tw.P_LEN:]).sum()
    assert 0.4 < frac.item() < 0.6                                # t ~ U(0,1): E[mask rate] = 1/2
    s2, _ = fact_free_tf_states(ids, tw.P_LEN, fm, generator=torch.Generator().manual_seed(0), force_fact_mask=False)
    assert not is_fact_free(s2, fm).all()                         # T5c ablation does show facts sometimes


class ScriptedModel(torch.nn.Module):
    """Logits that make greedy low-confidence decode commit answer positions left to right, emitting `tokens`."""

    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def forward(self, ids):
        b, l = ids.shape
        out = torch.zeros(b, l, tw.V)
        for j, t in enumerate(self.tokens):
            out[:, tw.P_LEN + j, t] = 10.0 - j                    # earlier positions more confident
        return out


def test_rollout_truncation_keeps_decision_state_only():
    f_tok = tw.STOI["CA3"]
    toks = [tw.STOI["FN0"], tw.STOI["LN0"], tw.STOI["IS"], f_tok, tw.STOI["DOT"], tw.EOS, tw.EOS, tw.EOS]
    m = ScriptedModel(toks)
    prompts = torch.full((2, tw.P_LEN), 5)
    fm = torch.zeros(2, tw.V, dtype=torch.bool)
    fm[0, f_tok] = True                                           # row 0 leaks at step 3; row 1 has no F
    cfg = dict(steps=8, block_length=8, temperature=0.0, remasking="low_confidence")
    out = collect_rollout_states(m, prompts, fm, cfg, J=100, generator=torch.Generator().manual_seed(0))
    s0 = out["states"][out["owner"] == 0]
    assert out["leak"].tolist() == [True, False]
    assert s0.shape[0] == 4                                       # s_0..s_3 (decision state = s_3), nothing after
    assert out["decision"][out["owner"] == 0].tolist() == [False, False, False, True]
    dec = s0[-1]
    assert (dec[tw.P_LEN:tw.P_LEN + 3] != tw.MASK).all() and (dec[tw.P_LEN + 3:] == tw.MASK).all()
    assert out["states"][out["owner"] == 1].shape[0] == 8
    assert is_fact_free(out["states"], fm[out["owner"]]).all()
    # J = 2: two uniform pre-tau states PLUS the decision state (SPEC v1)
    out = collect_rollout_states(m, prompts, fm, cfg, J=2, generator=torch.Generator().manual_seed(0))
    o0 = out["owner"] == 0
    assert int(o0.sum()) == 3 and bool(out["decision"][o0][-1])
    assert int((out["owner"] == 1).sum()) == 2
    # T5b: no truncation keeps post-leak states
    out = collect_rollout_states(m, prompts, fm, cfg, J=100, truncate=False)
    assert int((out["owner"] == 0).sum()) == 8
    # states match the sampler trace exactly (T0b)
    x, trace = generate(m, prompts, 8, 8, 8, return_trace=True)
    full = collect_rollout_states(m, prompts, fm, cfg, J=100)
    assert torch.equal(full["states"][full["owner"] == 1], torch.stack(trace)[:, 1])


def test_mdu_traj_state_schedule():
    torch.manual_seed(0)
    m = MaskPredictor(tw.V, 16)
    ids = torch.randint(5, tw.V, (3, 16))
    s, stop, fwd = mdu_traj_rollout_state(m, ids, 8, T=8, stop_step=3)
    assert ((s[:, 8:] != tw.MASK).sum(1) == 3).all() and fwd == 9 and torch.equal(s[:, :8], ids[:, :8])


# ------------------------------------------------------------------------------------------------ LoRA
def test_lora_gradients_A_zero_at_step1_both_nonzero_by_step2():
    """B = 0 at init => dL/dA = scale * B^T (dL/dy) x^T = 0 at step 1 (documented, not a bug); after one AdamW step B != 0,
    so both A and B receive non-zero gradient at step 2. Reference (adapter-disabled) pass is unaffected by training."""
    torch.manual_seed(0)
    m = MaskPredictor(tw.V, 16)
    ad = add_lora(m, r=8, alpha=16)
    opt = torch.optim.AdamW(lora_params(ad), lr=1e-2, weight_decay=0.0)
    s = torch.randint(5, tw.V, (4, 16))
    s[:, 10:] = tw.MASK
    fm = torch.zeros(4, tw.V, dtype=torch.bool)
    fm[:, 30] = True
    with torch.no_grad():
        ref0 = m(s).clone()

    def ref_fn(x):
        with lora_disabled(m):
            return m(x)
    for step in (1, 2):
        d = fx_forget_loss(m, ref_fn, s, fm, eps=0.0)
        opt.zero_grad()
        d["loss"].backward()
        gA = sum(a.A.grad.abs().sum().item() for a in ad)
        gB = sum(a.B.grad.abs().sum().item() for a in ad)
        if step == 1:
            assert gA == 0.0 and gB > 0
        else:
            assert all(a.A.grad.abs().sum() > 0 for a in ad) and all(a.B.grad.abs().sum() > 0 for a in ad)
        opt.step()
    with torch.no_grad():
        assert torch.allclose(ref_fn(s), ref0, atol=1e-6)          # adapter-off pass == theta0
        assert not torch.allclose(m(s), ref0, atol=1e-6)
        assert torch.allclose(merged_copy(m)(s), m(s), atol=1e-5)   # merged copy == adapter model


# ------------------------------------------------------------------------------------------------ sampler/cert
@pytest.mark.parametrize("cfg", [dict(steps=8, block_length=8, temperature=0.0, remasking="low_confidence"),
                                 dict(steps=4, block_length=4, temperature=0.7, remasking="random"),
                                 dict(steps=2, block_length=8, temperature=1.0, remasking="random")])
def test_batched_topk_equals_upstream_loop(cfg):
    torch.manual_seed(3)
    m = MaskPredictor(tw.V, 16)
    p = torch.randint(5, tw.V, (64, 8))
    a = generate(m, p, gen_length=8, generator=torch.Generator().manual_seed(1), **cfg)
    diffusion._BATCHED_TOPK = False
    try:
        b = generate(m, p, gen_length=8, generator=torch.Generator().manual_seed(1), **cfg)
    finally:
        diffusion._BATCHED_TOPK = True
    assert torch.equal(a, b)


def test_certificate_upper_bounds_leak_in_expectation():
    torch.manual_seed(4)
    m = MaskPredictor(tw.V, 16)
    with torch.no_grad():
        m.head.weight[30] += 0.15 * m.head.weight.std() * 10      # fact token moderately likely
    p = torch.randint(5, tw.V, (400, 8))
    fm = torch.zeros(400, tw.V, dtype=torch.bool)
    fm[:, 30] = True
    for cfg in (dict(steps=8, block_length=8, temperature=1.0, remasking="random"),
                dict(steps=4, block_length=8, temperature=0.5, remasking="low_confidence"),
                dict(steps=8, block_length=2, temperature=0.0, remasking="low_confidence")):
        c = hazard_certificate(m, p, fm, cfg, generator=torch.Generator().manual_seed(0))
        assert c["bound_mean"] + 0.03 >= c["leak_rate"]
        assert (c["bound"] >= 0).all()
        assert all((s is None) == (not l) for s, l in zip(c["decision_states"], c["leak"].tolist()))


def test_sc_tf_allows_fact_visible_states():
    w = tw.make_world(0)
    ex = w.examples(w.forget_facts)
    ids = torch.tensor([e.prompt + e.answer for e in ex])
    fm = fact_mask_from_sets([list(e.fact.value) for e in ex], tw.V)
    torch.manual_seed(0)
    m = MaskPredictor(tw.V, 16)
    d = sc_tf_loss(m, m, ids, tw.P_LEN, fm, eps=1e-3, generator=torch.Generator().manual_seed(0))
    assert 0 < d["frac_fact_visible_states"] < 1 and torch.isfinite(d["loss"])
