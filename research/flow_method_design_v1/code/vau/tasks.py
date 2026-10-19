"""GEM-style erasure tasks on the three model families (synthetic, no images).

A Task bundles a model, the prompt structure (anchor id 0, target id 1, retained ids 2..), the GEM loss
hyper-parameters, and samplers for erasure and retained batches. The erasure gradient follows GEM:
student = adapter ON on the TARGET prompt, teachers = adapter DISABLED on anchor / target prompts,
teachers constant (detached), loss = gamma * sum_i relu(D+ - eta D-) / n_latents over the n_latents
evaluation points, gradient through the student prediction only (rollout under no_grad).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import losses
from .contracts import ContractViolation, Ops
from .models import FlowInput, LinearModel, LoraMLP, TinyFlowDiT, flux_sigmas, rollout_prefix

ANCHOR, TARGET = 0, 1


@dataclass
class Task:
    model: object
    family: str
    n_latents: int = 11
    eta: float = 1.0
    gamma: float = 100.0
    loss: str = "gem"
    n_retained_prompts: int = 4

    # ---- inputs (family specific) --------------------------------------------------------------
    def erasure_batch(self, rng):
        raise NotImplementedError

    def erasure_inputs(self, theta, eb, ops):
        """(x_student, x_anchor): the target-prompt and anchor-prompt evaluation points."""
        raise NotImplementedError

    def retained_batch(self, rng, n):
        raise NotImplementedError

    # ---- shared GEM loss / gradient -------------------------------------------------------------
    def loss_and_grad_at(self, theta, xs, xa, ops: Ops):
        """GEM loss and gradient at fixed evaluation inputs (latents treated as constants)."""
        m = self.model
        n = m.n(xs)
        if n != self.n_latents:
            raise ContractViolation(f"erasure batch has {n} points, n_latents={self.n_latents}")
        v = m.forward(xs, theta); ops.add("forward", n)
        a = m.forward(xa, None); ops.add("teacher", n)       # anchor teacher, adapter disabled
        b = m.forward(xs, None); ops.add("teacher", n)       # target teacher, adapter disabled
        loss, gv = losses.LOSSES[self.loss](v, a, b, self.n_latents, eta=self.eta, gamma=self.gamma)
        g = m.vjp(xs, theta, gv); ops.add("vjp", n)
        return loss, g

    def erasure_loss_grad(self, theta, eb, ops: Ops):
        xs, xa = self.erasure_inputs(theta, eb, ops)
        return self.loss_and_grad_at(theta, xs, xa, ops)

    def outputs(self, x, theta):
        return self.model.forward(x, theta)


class _VectorTask(Task):
    """Families (a), (b): input x = concat(z, prompt_embedding[prompt])."""

    def __init__(self, model, family, z_dim, emb, **kw):
        super().__init__(model=model, family=family, **kw)
        self.z_dim, self.emb = z_dim, emb

    def _x(self, z, pid):
        return np.concatenate([z, self.emb[pid]], axis=1)

    def erasure_batch(self, rng):
        return rng.normal(size=(self.n_latents, self.z_dim))

    def erasure_inputs(self, theta, eb, ops):
        n = eb.shape[0]
        return self._x(eb, np.full(n, TARGET)), self._x(eb, np.full(n, ANCHOR))

    def retained_batch(self, rng, n):
        z = rng.normal(size=(n, self.z_dim))
        return self._x(z, rng.integers(2, 2 + self.n_retained_prompts, size=n))


class FlowTask(Task):
    """Family (c): GEM erasure on a student rollout prefix.

    erasure batch = initial noise z0; inputs = the end_idx+1 states of a student rollout (adapter ON,
    current theta, no grad) on the TARGET prompt, the state after step k paired with sigma_k (GEM
    EG:214-215 pairing). Retained queries: z ~ N(0, I), sigma uniform over the GEM prefix sigmas,
    retained prompt ids (synthetic stand-in for retained-prompt latents).
    """

    def __init__(self, model, n_steps=28, end_idx=10, mu=0.630, **kw):
        super().__init__(model=model, family="flow", n_latents=end_idx + 1, **kw)
        self.sigmas = flux_sigmas(n_steps, mu)
        self.end_idx = end_idx

    def erasure_batch(self, rng):
        return rng.normal(size=(self.model.n_img, self.model.c))

    def erasure_inputs(self, theta, eb, ops):
        states, n_fwd = rollout_prefix(self.model, theta, eb, TARGET, self.sigmas, self.end_idx)
        for _ in range(n_fwd):
            ops.add("rollout_forward", 1)
        n = states.shape[0]
        sig = self.sigmas[:n].copy()
        return (FlowInput(states, sig, np.full(n, TARGET)), FlowInput(states, sig, np.full(n, ANCHOR)))

    def retained_batch(self, rng, n):
        z = rng.normal(size=(n, self.model.n_img, self.model.c))
        sig = rng.choice(self.sigmas[: self.end_idx + 1], size=n)
        return FlowInput(z, sig, rng.integers(2, 2 + self.n_retained_prompts, size=n))


def make_task(family, rng, **kw):
    """Build a task of family 'linear' | 'mlp' | 'flow' with frozen random base weights from rng."""
    n_ret = kw.pop("n_retained_prompts", 4)
    if family in ("linear", "mlp"):
        z_dim, e_dim = kw.pop("z_dim", 4), kw.pop("e_dim", 4)
        emb = rng.normal(size=(2 + n_ret, e_dim))
        if family == "linear":
            model = LinearModel(rng, z_dim + e_dim, kw.pop("d_out", 4), kw.pop("P", 40),
                                aniso_decades=kw.pop("aniso_decades", 0.0))
        else:
            model = LoraMLP(rng, z_dim + e_dim, kw.pop("d_hidden", 16), kw.pop("d_out", 4), r=kw.pop("r", 2))
        return _VectorTask(model, family, z_dim, emb, n_retained_prompts=n_ret, **kw)
    if family == "flow":
        mk = {k: kw.pop(k) for k in ("c", "n_img", "n_txt", "dm", "r") if k in kw}
        model = TinyFlowDiT(rng, n_prompts=2 + n_ret, **mk)
        return FlowTask(model, n_retained_prompts=n_ret, **kw)
    raise ContractViolation(f"unknown family {family!r}")
