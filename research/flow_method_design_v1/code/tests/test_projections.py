"""Projector invariants: constraint, minimiser (vs scipy), metric-norm contraction, H-progress, sketch."""
import numpy as np
import pytest
import scipy.linalg as sla
from scipy.optimize import minimize

from vau import adamw as aw
from vau.contracts import ContractViolation, Ops
from vau.projections import (euclid_project, metric_project, ridge_mu, sketch_probe,
                             weighted_null_project)


def _case(seed, P=12, k=4, decades=2.0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(k, P))
    D = 10.0 ** rng.uniform(-decades / 2, decades / 2, size=P)
    d0 = rng.normal(size=P)
    return rng, S, D, d0


@pytest.mark.parametrize("seed", range(5))
def test_constraint_exact_mu0(seed):
    _, S, D, d0 = _case(seed)
    d1, _ = metric_project(d0, S, D, mu=0.0)
    assert np.linalg.norm(S @ d1) <= 1e-12 * np.linalg.norm(S @ d0)
    e1, _ = euclid_project(d0, S, mu=0.0)
    assert np.linalg.norm(S @ e1) <= 1e-12 * np.linalg.norm(S @ d0)


@pytest.mark.parametrize("seed", range(5))
def test_ridge_residual_identity(seed):
    _, S, D, d0 = _case(seed)
    d1, info = metric_project(d0, S, D)
    G0 = (S * D) @ S.T
    mu = ridge_mu(G0)
    assert info["mu"] == mu
    pred = mu * np.linalg.solve(G0 + mu * np.eye(len(G0)), S @ d0)
    assert np.allclose(S @ d1, pred, rtol=1e-6, atol=1e-14)
    assert np.linalg.norm(S @ d1) <= 1e-4 * np.linalg.norm(S @ d0)


@pytest.mark.parametrize("seed", range(4))
def test_minimiser_matches_scipy(seed):
    _, S, D, d0 = _case(seed, P=8, k=3, decades=1.0)
    d1, _ = metric_project(d0, S, D, mu=0.0)
    obj = lambda x: np.sum((x - d0) ** 2 / D)
    res = minimize(obj, np.zeros_like(d0), jac=lambda x: 2 * (x - d0) / D, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda x: S @ x, "jac": lambda x: S}],
                   options={"ftol": 1e-14, "maxiter": 500})
    assert res.success
    assert np.allclose(d1, res.x, atol=1e-6)
    # independent KKT block solve
    P, k = len(d0), S.shape[0]
    K = np.block([[np.diag(2 / D), S.T], [S, np.zeros((k, k))]])
    sol = sla.solve(K, np.concatenate([2 * d0 / D, np.zeros(k)]))
    assert np.allclose(d1, sol[:P], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_metric_norm_contracts(seed):
    _, S, D, d0 = _case(seed, decades=4.0)
    d1, _ = metric_project(d0, S, D)
    n = lambda x: np.sqrt(np.sum(x * x / D))
    assert n(d1) <= n(d0) * (1 + 1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_H_progress_nonnegative_when_mhat_equals_g(seed):
    """Step 1 (mhat = g exactly by bias correction), wd = 0: -g.Delta1 = lr||(I-Q) D^1/2 g||^2 >= 0."""
    rng, S, _, _ = _case(seed)
    g = rng.normal(size=S.shape[1]) * 10.0 ** rng.uniform(-3, 1, size=S.shape[1])
    hp = aw.AdamWHP(weight_decay=0.0)
    d0, tent, D = aw.peek(aw.init_state(len(g)), g, np.zeros_like(g), hp)
    assert np.allclose(aw.mhat(tent, hp), g, rtol=1e-12)
    d1, _ = metric_project(d0, S, D, mu=0.0)
    assert -g @ d1 >= -1e-15 * abs(g @ d0)


def test_degenerate_and_contracts():
    x = np.ones(4)
    y, info = euclid_project(x, np.zeros((2, 4)))
    assert info["degenerate"] and np.array_equal(y, x)
    with pytest.raises(ContractViolation):
        weighted_null_project(x, np.ones((2, 5)), np.ones(4))
    with pytest.raises(ContractViolation):
        weighted_null_project(x, np.ones((2, 4)), -np.ones(4))
    with pytest.raises(ContractViolation):
        weighted_null_project(np.array([1, np.inf, 0, 0.0]), np.ones((2, 4)), np.ones(4))
    with pytest.raises(ContractViolation):   # rank-deficient, mu = 0
        metric_project(x, np.ones((2, 4)), np.ones(4), mu=0.0)


def test_sketch_equals_sigma_times_J():
    from conftest import make, perturbed_theta
    task, rng = make("mlp")
    th = perturbed_theta(task, rng)
    x = task.retained_batch(rng, 3)
    J = task.model.jacobian(x, th)
    ops = Ops()
    S = sketch_probe(task.model, x, th, 5, np.random.default_rng(7), ops)
    Sigma = np.random.default_rng(7).standard_normal((64, 3 * task.model.out_dim))[:5] / np.sqrt(5)
    assert np.allclose(S, Sigma @ J, rtol=1e-12, atol=1e-13)
    assert ops.calls["vjp"] == 5 and ops.samples["vjp"] == 15 and ops.calls["probe_build"] == 1
