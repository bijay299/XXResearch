"""GEM-style erasure tasks on the three model families (synthetic, no images).

A Task bundles a model, the prompt structure, the GEM loss hyper-parameters and samplers for erasure and
retained inputs. The erasure gradient follows GEM: student = adapter ON on the TARGET prompt, teachers =
adapter DISABLED on anchor / target prompts, teachers constant (detached), loss = gamma * sum_i
relu(D+ - eta D-) / n_latents, gradient through the student prediction only (rollout under no_grad).

Prompt ids: 0 = anchor, 1 = target, 2 .. 1+n_ret = retained, then 1 rare-subgroup prompt, then (flow)
n_heldout held-out retained prompts.

Vector families (a, b): input x = feat_scale * (concat(z, e[prompt]) + shift * u). The last `n_concept`
embedding coordinates are "concept" coordinates: nonzero for the target (and the rare prompt, which has
cosine `rare_cos` = 0.8 with the target embedding), exactly zero for anchor and retained prompts. Thus the
retained Jacobian row space is spanned by the non-concept input coordinates only, which is what the
structured (G2) linear construction controls.

Knobs (eval_protocol.json): aniso_decades (a: parameter column scales; b: input-feature scales spread over
d decades, the G1 substitute), kappa/phi (G2, model a), rare_mass/rare_cos (R_rare, population mixture),
shift (R_shift), contrast_scale (G4: ||b-a|| scaled by contrast_scale relative to the reference anchor),
eps_n (G4), gain (G5 tanh gain), gauge steps via model.gauge_step (G5), k/K/ridge on the arms.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import losses
from .contracts import ContractViolation, Ops
from .models import (FlowInput, LinearModel, LoraMLP, TinyFlowDiT, euler, flux_sigmas, rollout_prefix,
                     structured_T)

ANCHOR, TARGET = 0, 1


def _cos_mix(rng, t, c):
    """Vector with cosine c to t and the same norm."""
    t = t.ravel()
    r = rng.normal(size=t.shape)
    r -= (r @ t) / (t @ t) * t
    out = c * t / np.linalg.norm(t) + np.sqrt(1 - c * c) * r / np.linalg.norm(r)
    return out * np.linalg.norm(t)


def input_hashes(x):
    """sha256 per input row (I10 disjointness check). Works for arrays and FlowInput."""
    if isinstance(x, FlowInput):
        rows = [np.concatenate([x.z[i].ravel(), [x.sigma[i], float(x.prompt[i])]]) for i in range(x.z.shape[0])]
    else:
        rows = list(x)
    return [hashlib.sha256(np.ascontiguousarray(r, dtype=np.float64).tobytes()).hexdigest() for r in rows]


@dataclass
class Task:
    model: object
    family: str
    n_latents: int = 11
    eta: float = 1.0
    gamma: float = 100.0
    loss: str = "gem"
    eps_n: float = losses.EPS_N
    n_retained_prompts: int = 4
    rare_mass: float = 0.005

    @property
    def rare_id(self):
        return 2 + self.n_retained_prompts

    def _retained_ids(self, rng, n, rare):
        if rare is True:
            return np.full(n, self.rare_id)
        ids = rng.integers(2, 2 + self.n_retained_prompts, size=n)
        if rare is None and self.rare_mass > 0:
            ids = np.where(rng.random(n) < self.rare_mass, self.rare_id, ids)
        elif rare not in (None, False):
            raise ContractViolation("rare must be None (population mixture), False or True")
        return ids

    def erasure_batch(self, rng):
        raise NotImplementedError

    def erasure_inputs(self, theta, eb, ops):
        raise NotImplementedError

    def retained_batch(self, rng, n, **kw):
        raise NotImplementedError

    def loss_and_grad_at(self, theta, xs, xa, ops: Ops):
        """GEM loss and gradient at fixed evaluation inputs (latents treated as constants)."""
        m = self.model
        n = m.n(xs)
        if n != self.n_latents:
            raise ContractViolation(f"erasure batch has {n} points, n_latents={self.n_latents}")
        v = m.forward(xs, theta); ops.add("forward", n)
        a = m.forward(xa, None); ops.add("teacher", n)       # anchor teacher, adapter disabled
        b = m.forward(xs, None); ops.add("teacher", n)       # target teacher, adapter disabled
        loss, gv = losses.LOSSES[self.loss](v, a, b, self.n_latents, eta=self.eta, gamma=self.gamma,
                                            eps_n=self.eps_n)
        g = m.vjp(xs, theta, gv); ops.add("vjp", n)
        return loss, g

    def erasure_loss_grad(self, theta, eb, ops: Ops):
        xs, xa = self.erasure_inputs(theta, eb, ops)
        return self.loss_and_grad_at(theta, xs, xa, ops)


class VectorTask(Task):
    """population='generic': retained x = concat(z, e[retained prompt]);
    population='covered': retained x = c @ W, c ~ N(0, I_q), W (q x d_in) fixed with zero concept coordinates
    (rank-q retained subspace; protocol probe_coverage.population_axis). Rare-subgroup samples are
    concat(z, e[rare]) in both populations.
    target_mode='z' (default): target inputs concat(z, e[target]); 'concept' (G3): target inputs vary only in
    the concept coordinates (z = 0, shared embedding coordinates 0), x = concat(0, 0, e_c[target] + noise),
    anchor = the same input with the concept coordinates set to the anchor's (0)."""

    def __init__(self, model, family, z_dim, emb, feat_scale, shift_dir, n_concept=2, population="generic",
                 cover_basis=None, target_mode="z", **kw):
        super().__init__(model=model, family=family, **kw)
        self.z_dim, self.emb, self.feat_scale, self.shift_dir = z_dim, emb, feat_scale, shift_dir
        self.n_concept, self.population, self.cover_basis, self.target_mode = n_concept, population, cover_basis, target_mode
        if population not in ("generic", "covered") or target_mode not in ("z", "concept"):
            raise ContractViolation("bad population / target_mode")

    def _x(self, z, pid, shift=0.0):
        return self.feat_scale * (np.concatenate([z, self.emb[pid]], axis=1) + shift * self.shift_dir)

    def erasure_batch(self, rng):
        return rng.normal(size=(self.n_latents, self.z_dim if self.target_mode == "z" else self.n_concept))

    def target_inputs(self, eb):
        """(x_target, x_anchor) for a target set eb (z vectors, or concept noise in 'concept' mode)."""
        n = eb.shape[0]
        if self.target_mode == "z":
            return self._x(eb, np.full(n, TARGET)), self._x(eb, np.full(n, ANCHOR))
        d_in = self.z_dim + self.emb.shape[1]
        xt = np.zeros((n, d_in)); xa = np.zeros((n, d_in))
        xt[:, d_in - self.n_concept:] = self.emb[TARGET, -self.n_concept:] + eb
        xa[:, d_in - self.n_concept:] = self.emb[ANCHOR, -self.n_concept:] + eb
        return self.feat_scale * xt, self.feat_scale * xa

    def erasure_inputs(self, theta, eb, ops):
        return self.target_inputs(eb)

    def retained_batch(self, rng, n, rare=None, shift=0.0):
        """rare=None: population mixture (rare prompt w.p. rare_mass); True: R_rare; False: no rare.
        shift: R_shift magnitude along the fixed unit direction (units of the z sd = 1)."""
        z = rng.normal(size=(n, self.z_dim))
        ids = self._retained_ids(rng, n, rare)
        x = np.concatenate([z, self.emb[ids]], axis=1)
        if self.population == "covered":
            c = rng.normal(size=(n, self.cover_basis.shape[0]))
            xc = c @ self.cover_basis
            x = np.where((ids == self.rare_id)[:, None], x, xc)
        return self.feat_scale * (x + shift * self.shift_dir)


class FlowTask(Task):
    """Family (c): GEM erasure on a student rollout prefix.

    Erasure batch = initial noise z0; inputs = the end_idx+1 states of a student rollout (adapter ON, current
    theta, no grad) on the TARGET prompt, the state after step k paired with sigma_k (GEM EG:214-215).
    Retained queries (probes, penalty, gate; eval_protocol observables.model_c_retained_output): velocity at
    (state of a BASE retained trajectory entering step k, sigma_k), k uniform in 0..n_steps-1.
    """

    def __init__(self, model, n_steps=28, end_idx=10, mu=0.630, n_heldout=4, **kw):
        kw.setdefault("rare_mass", 0.0)     # protocol: model c has no rare subgroup in the population
        super().__init__(model=model, family="flow", n_latents=end_idx + 1, **kw)
        self.sigmas = flux_sigmas(n_steps, mu)
        self.n_steps, self.end_idx, self.n_heldout = n_steps, end_idx, n_heldout

    @property
    def heldout_ids(self):
        return np.arange(self.rare_id + 1, self.rare_id + 1 + self.n_heldout)

    def erasure_batch(self, rng):
        return rng.normal(size=(self.model.n_img, self.model.c))

    def erasure_inputs(self, theta, eb, ops):
        """eb = z0 (prefix scheme, TARGET prompt) or a tuple (z0, target_id, idx): idx None -> GEM prefix
        steps 0..end_idx; idx = sorted step indices (GD-D 'uniform 11-of-28'), rollout to max(idx)."""
        z0, tid, idx = (eb, TARGET, None) if not isinstance(eb, tuple) else eb
        end = self.end_idx if idx is None else int(np.max(idx))
        states, n_fwd = rollout_prefix(self.model, theta, z0, tid, self.sigmas, end)
        for _ in range(n_fwd):
            ops.add("rollout_forward", 1)
        if idx is None:
            sel = np.arange(states.shape[0])
        else:
            sel = np.asarray(idx)
            if len(sel) != self.n_latents:
                raise ContractViolation("uniform scheme needs n_latents indices")
        st, sig = states[sel], self.sigmas[sel].copy()
        n = st.shape[0]
        return FlowInput(st, sig, np.full(n, tid)), FlowInput(st, sig, np.full(n, ANCHOR))

    def retained_batch(self, rng, n, rare=None, heldout=False, shift=0.0, ids=None):
        """Base-trajectory retained queries (one-off data generation; not charged to any arm).
        ids: optional pool of prompt ids to draw conditions from."""
        if shift != 0.0:
            raise ContractViolation("model c has no R_shift (protocol substitute: held-out conditions)")
        if ids is not None:
            ids = rng.choice(np.asarray(ids), size=n)
        else:
            ids = rng.choice(self.heldout_ids, size=n) if heldout else self._retained_ids(rng, n, rare)
        z0 = rng.normal(size=(n, self.model.n_img, self.model.c))
        k = rng.integers(0, self.n_steps, size=n)
        states = euler(self.model, None, z0, ids, self.sigmas, int(k.max()))
        return FlowInput(states[k, np.arange(n)], self.sigmas[k].copy(), ids)


def make_task(family, rng, **kw):
    """Build a task of family 'linear' | 'mlp' | 'flow'; all fixture randomness from `rng` (stream 0)."""
    n_ret = kw.pop("n_retained_prompts", 4)
    contrast = kw.pop("contrast_scale", 1.0)
    rare_cos = kw.pop("rare_cos", 0.8)
    if family in ("linear", "mlp"):
        z_dim, e_dim, n_c = kw.pop("z_dim", 4), kw.pop("e_dim", 4), kw.pop("n_concept", 2)
        d_in, d_out = z_dim + e_dim, kw.pop("d_out", 4)
        emb = rng.normal(size=(3 + n_ret, e_dim))
        emb[[ANCHOR] + list(range(2, 2 + n_ret)), e_dim - n_c:] = 0.0     # concept coords: target/rare only
        emb[2 + n_ret] = _cos_mix(rng, emb[TARGET], rare_cos)            # rare subgroup prompt
        emb[ANCHOR] = emb[TARGET] + contrast * (emb[ANCHOR] - emb[TARGET])
        shift_dir = rng.normal(size=d_in); shift_dir /= np.linalg.norm(shift_dir)
        aniso = kw.pop("aniso_decades", 0.0)
        population, q = kw.pop("population", "generic"), kw.pop("q", 2)
        cover = np.zeros((q, d_in))
        cover[:, : d_in - n_c] = rng.normal(size=(q, d_in - n_c))
        T_concept = kw.pop("T_concept", None)
        if family == "linear":
            P = kw.pop("P", 40)
            kappa, phi = kw.pop("kappa", None), kw.pop("phi", 0.0)
            T = None
            if kappa is not None:
                n_sh = d_in - n_c
                Ts = structured_T(rng, n_sh, d_out, P, kappa, phi)
                if T_concept is None:
                    Tc = rng.normal(size=(n_c, d_out, P)) / np.sqrt(d_in)
                elif callable(T_concept):            # G3: concept rows designed relative to the retained block
                    Tc = T_concept(Ts.reshape(n_sh * d_out, P), rng).reshape(n_c, d_out, P)
                else:
                    Tc = T_concept
                T = np.concatenate([Ts, Tc])
            model = LinearModel(rng, d_in, d_out, P, aniso_decades=aniso, T=T)
            feat = np.ones(d_in)
        else:
            model = LoraMLP(rng, d_in, kw.pop("d_hidden", 16), d_out, r=kw.pop("r", 2), gain=kw.pop("gain", 1.5))
            feat = 10.0 ** rng.uniform(-aniso / 2, aniso / 2, size=d_in)
        return VectorTask(model, family, z_dim, emb, feat, shift_dir, n_concept=n_c, population=population,
                          cover_basis=cover, n_retained_prompts=n_ret, **kw)
    if family == "flow":
        mk = {k: kw.pop(k) for k in ("c", "n_img", "n_txt", "dm", "r") if k in kw}
        n_held = kw.pop("n_heldout", 4)
        model = TinyFlowDiT(rng, n_prompts=3 + n_ret + n_held, **mk)
        E = model.E
        E[2 + n_ret] = _cos_mix(rng, E[TARGET], rare_cos).reshape(E[TARGET].shape)
        E[ANCHOR] = E[TARGET] + contrast * (E[ANCHOR] - E[TARGET])
        return FlowTask(model, n_retained_prompts=n_ret, n_heldout=n_held, **kw)
    raise ContractViolation(f"unknown family {family!r}")
