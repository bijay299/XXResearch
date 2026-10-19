"""Finite-difference checks (central differences, FP64) of every analytic gradient / VJP / JVP."""
import numpy as np
import pytest
from conftest import central_fd, make, perturbed_theta

from vau import losses
from vau.contracts import ContractViolation, Ops
from vau.models import FlowInput

RTOL = 1e-6


def _rel(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12)


@pytest.mark.parametrize("name", ["gem", "cpr", "anchor"])
def test_loss_grad_fd(name):
    rng = np.random.default_rng(1)
    v, a, b = (rng.normal(size=(5, 3, 2)) for _ in range(3))
    eta = 0.0 if name == "anchor" else 0.7
    fn = losses.LOSSES[name]
    f = lambda vv: fn(vv.reshape(v.shape), a, b, 7, eta=eta, gamma=3.0)[0]
    loss, g = fn(v, a, b, 7, eta=eta, gamma=3.0)
    assert loss > 0
    assert _rel(g.ravel(), central_fd(f, v.ravel())) < RTOL


def test_gem_pairwise_distance_semantics():
    x = np.zeros((2, 9)); y = np.zeros((2, 9))
    assert np.allclose(losses.pairwise_distance(x, y), 1e-6 * 3.0, rtol=0, atol=1e-18)
    # at v == b, D- = eps*sqrt(d) and its gradient is the all-ones direction / sqrt(d)
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(1, 9)), rng.normal(size=(1, 9))
    _, g = losses.gem_loss(b.copy(), a, b, 1, eta=1.0, gamma=1.0)
    u = (b - a + 1e-6) / np.linalg.norm(b - a + 1e-6)
    assert np.allclose(g, u - np.ones((1, 9)) / 3.0)


def test_gem_relu_inactive_zero_grad():
    a = np.zeros((1, 4)); b = np.full((1, 4), 5.0); v = np.zeros((1, 4)) + 0.01
    loss, g = losses.gem_loss(v, a, b, 1)        # D+ small, D- large -> hinge off
    assert loss == 0.0 and np.all(g == 0.0)


def test_penalty_grad_fd():
    rng = np.random.default_rng(3)
    fr, fr0 = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    _, g = losses.retention_penalty(fr, fr0, 2.5)
    fd = central_fd(lambda x: losses.retention_penalty(x.reshape(4, 3), fr0, 2.5)[0], fr.ravel())
    assert _rel(g.ravel(), fd) < RTOL


def test_model_jvp_vjp_jacobian_fd(family):
    task, rng = make(family)
    m = task.model
    th = perturbed_theta(task, rng)
    x = task.retained_batch(rng, 3)
    J_fd = central_fd(lambda t: m.forward(x, t).ravel(), th)          # (N*d, P)
    J = m.jacobian(x, th)
    assert _rel(J, J_fd) < RTOL
    w = rng.normal(size=m.P)
    assert _rel(m.jvp(x, th, w).ravel(), J_fd @ w) < RTOL
    U = rng.normal(size=(m.n(x), m.out_dim))
    assert _rel(m.vjp(x, th, U), J_fd.T @ U.ravel()) < RTOL
    # adjoint identity, to rounding
    assert abs(U.ravel() @ m.jvp(x, th, w).ravel() - m.vjp(x, th, U) @ w) < 1e-10 * (1 + abs(m.vjp(x, th, U) @ w))


def test_erasure_grad_fd(family):
    """Full-chain erasure gradient (teachers constant, evaluation points fixed as in GEM's no_grad rollout)."""
    task, rng = make(family)
    th = perturbed_theta(task, rng, 0.1)
    eb = task.erasure_batch(rng)
    xs, xa = task.erasure_inputs(th, eb, Ops())
    loss, g = task.loss_and_grad_at(th, xs, xa, Ops())
    assert loss > 0
    fd = central_fd(lambda t: task.loss_and_grad_at(t, xs, xa, Ops())[0], th)
    assert _rel(g, fd) < RTOL


def test_jacobian_guard():
    task, rng = make("flow")
    x = task.retained_batch(rng, 4)
    with pytest.raises(ContractViolation):
        task.model.jacobian(x, task.model.init_theta(rng), max_entries=100)


def test_flow_input_contract():
    task, rng = make("flow")
    m = task.model
    th = m.init_theta(rng)
    good = task.retained_batch(rng, 2)
    with pytest.raises(ContractViolation):
        m.forward(FlowInput(good.z[:, :2], good.sigma, good.prompt), th)
    with pytest.raises(ContractViolation):
        bad = good.z.copy(); bad[0, 0, 0] = np.nan
        m.forward(FlowInput(bad, good.sigma, good.prompt), th)
    with pytest.raises(ContractViolation):
        m.forward(FlowInput(good.z, good.sigma, np.array([0, 99])), th)
    with pytest.raises(ContractViolation):
        m.forward(good, th[:-1])
