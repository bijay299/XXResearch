"""AdamW: equivalence with an independent straightforward loop, purity of peek, contracts."""
import math

import numpy as np
import pytest

from vau import adamw as aw
from vau.contracts import ContractViolation


def reference_adamw(theta, grads, lr, b1, b2, eps, wd):
    """Independent textbook loop (per element, python floats), Loshchilov-Hutter / torch AdamW docs."""
    theta = [float(x) for x in theta]
    m = [0.0] * len(theta); v = [0.0] * len(theta)
    out = []
    for t, g in enumerate(grads, start=1):
        for i in range(len(theta)):
            theta[i] = theta[i] - lr * wd * theta[i]
            m[i] = b1 * m[i] + (1 - b1) * g[i]
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            theta[i] = theta[i] - lr * mh / (math.sqrt(vh) + eps)
        out.append((list(theta), list(m), list(v)))
    return out


@pytest.mark.parametrize("b1,b2,wd,eps", [(0.9, 0.999, 0.01, 1e-8), (0.3, 0.9, 0.1, 1e-3), (0.0, 0.5, 0.0, 1e-8)])
def test_matches_reference_loop(b1, b2, wd, eps):
    rng = np.random.default_rng(0)
    P, T = 7, 40
    theta = rng.normal(size=P)
    grads = [rng.normal(size=P) * 10.0 ** rng.uniform(-4, 2, size=P) for _ in range(T)]
    hp = aw.AdamWHP(lr=3e-3, beta1=b1, beta2=b2, eps=eps, weight_decay=wd)
    ref = reference_adamw(theta, grads, 3e-3, b1, b2, eps, wd)
    st, th = aw.init_state(P), theta.copy()
    for t, g in enumerate(grads):
        th, st = aw.step(st, g, th, hp)
        rt, rm, rv = (np.array(a) for a in ref[t])
        assert np.allclose(th, rt, rtol=1e-12, atol=1e-15)
        assert np.allclose(st.m, rm, rtol=1e-12, atol=1e-300)
        assert np.allclose(st.v, rv, rtol=1e-12, atol=1e-300)
        assert st.t == t + 1


def test_peek_is_pure_and_D_relation():
    rng = np.random.default_rng(1)
    P = 5
    hp = aw.AdamWHP()
    st = aw.AdamState(rng.normal(size=P), rng.uniform(size=P), 3)
    m0, v0 = st.m.copy(), st.v.copy()
    g, th = rng.normal(size=P), rng.normal(size=P)
    th0 = th.copy()
    d0, tent, D = aw.peek(st, g, th, hp)
    assert np.array_equal(st.m, m0) and np.array_equal(st.v, v0) and st.t == 3 and np.array_equal(th, th0)
    assert tent.t == 4
    assert np.allclose(d0, -hp.lr * hp.weight_decay * th - hp.lr * aw.mhat(tent, hp) * D, rtol=1e-13)
    assert np.allclose(D, 1.0 / (np.sqrt(tent.v / (1 - hp.beta2 ** 4)) + hp.eps), rtol=1e-13)


def test_first_step_is_lr_sign():
    rng = np.random.default_rng(2)
    g = rng.normal(size=6); th = rng.normal(size=6)
    hp = aw.AdamWHP(lr=1e-3, weight_decay=0.0)
    d0, _, _ = aw.peek(aw.init_state(6), g, th, hp)
    assert np.allclose(d0, -1e-3 * np.sign(g), rtol=1e-6)


def test_contracts():
    hp = aw.AdamWHP()
    st = aw.init_state(3)
    with pytest.raises(ContractViolation):
        aw.peek(st, np.array([1.0, np.nan, 0.0]), np.zeros(3), hp)
    with pytest.raises(ContractViolation):
        aw.peek(st, np.zeros(4), np.zeros(3), hp)
    with pytest.raises(ContractViolation):
        aw.peek(st, np.zeros(3, dtype=np.float32), np.zeros(3), hp)
    with pytest.raises(ContractViolation):
        aw.AdamWHP(lr=-1.0)
    with pytest.raises(ContractViolation):
        aw.AdamWHP(beta2=1.0)
    _, tent, _ = aw.peek(st, np.ones(3), np.zeros(3), hp)
    with pytest.raises(ContractViolation):
        aw.commit(tent, tent)      # not a successor
