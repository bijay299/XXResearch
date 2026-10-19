"""Three model families with flattened LoRA parameters theta (base weights frozen).

Every model exposes:
    P, out_dim, n(x)               parameter count, per-sample output size, batch size of input x
    init_theta(rng)                LoRA init (peft default: A kaiming-uniform(a=sqrt(5)) i.e.
                                   U(-1/sqrt(d_in), 1/sqrt(d_in)); B = 0) -- or 0 for the linear model
    forward(x, theta)              (N, out_dim); theta=None means ADAPTER DISABLED (teacher)
    jvp(x, theta, w)               J w, (N, out_dim)       analytic, matrix-free
    vjp(x, theta, U)               J^T U summed over samples, (P,)
    jacobian(x, theta, max_entries) dense J, (N*out_dim, P), only for small cases (guarded)

LoRA convention (peft/torch Linear): y = x W_eff^T, W_eff = W + s * B @ A, A (r, d_in), B (d_out, r),
s = lora_alpha / r (GEM default 16/16 = 1). peft applies LoRA dropout to x before A; this numpy
surrogate has NO dropout (see README_E.md, infidelity list).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContractViolation, check_array, check_vec

DEFAULT_MAX_JAC_ENTRIES = 2_000_000


class LoraLayout:
    """Packs/unpacks a list of LoRA (A, B) pairs into one flat theta; order per module is [A, B]."""

    def __init__(self, modules, r, scaling=1.0):
        self.modules = list(modules)        # [(name, d_out, d_in)]
        self.r, self.s = int(r), float(scaling)
        self.slices, off = {}, 0
        for name, do, di in self.modules:
            nA, nB = self.r * di, do * self.r
            self.slices[name] = (slice(off, off + nA), slice(off + nA, off + nA + nB), do, di)
            off += nA + nB
        self.P = off

    def unpack(self, theta):
        out = {}
        for name, (sa, sb, do, di) in self.slices.items():
            out[name] = (theta[sa].reshape(self.r, di), theta[sb].reshape(do, self.r))
        return out

    def pack(self, grads):
        theta = np.empty(self.P)
        for name, (sa, sb, _, _) in self.slices.items():
            gA, gB = grads[name]
            theta[sa], theta[sb] = gA.ravel(), gB.ravel()
        return theta

    def init(self, rng):
        theta = np.zeros(self.P)
        for name, (sa, _, _, di) in self.slices.items():
            bound = 1.0 / np.sqrt(di)        # kaiming_uniform_(a=sqrt(5)): gain*sqrt(3/fan_in) = 1/sqrt(fan_in)
            theta[sa] = rng.uniform(-bound, bound, size=sa.stop - sa.start)
        return theta                          # B block stays exactly 0

    def w_eff(self, W, AB):
        A, B = AB
        return W + self.s * (B @ A)

    def dW(self, AB, dAB):
        (A, B), (dA, dB) = AB, dAB
        return self.s * (dB @ A + B @ dA)

    def grad_AB(self, gW, AB):
        A, B = AB
        return self.s * (B.T @ gW), self.s * (gW @ A.T)


def lora_gauge_step(layout, theta, k_norm, rng):
    """First-order-neutral LoRA gauge direction (C-X2): per module dB = B K, dA = -K A with K (r x r)
    Gaussian rescaled to Frobenius norm k_norm. B dA + dB A = 0, exact map change -s B K^2 A."""
    ab = layout.unpack(theta)
    out = {}
    for name, (A, B) in ab.items():
        K = rng.normal(size=(layout.r, layout.r))
        K *= k_norm / np.linalg.norm(K)
        out[name] = (-K @ A, B @ K)
    return layout.pack(out)


class Model:
    P: int
    out_dim: int

    def n(self, x):
        raise NotImplementedError

    def check_input(self, x):
        raise NotImplementedError

    def _check(self, x, theta, allow_none=True):
        self.check_input(x)
        if theta is None and allow_none:
            return
        check_vec("theta", theta, self.P)

    def jacobian(self, x, theta, max_entries=DEFAULT_MAX_JAC_ENTRIES):
        self._check(x, theta, allow_none=False)
        N = self.n(x)
        if N * self.out_dim * self.P > max_entries:
            raise ContractViolation(f"dense Jacobian too large: {N}*{self.out_dim}*{self.P} > {max_entries}")
        J = np.empty((N * self.out_dim, self.P))
        e = np.zeros(self.P)
        for p in range(self.P):
            e[p] = 1.0
            J[:, p] = self.jvp(x, theta, e).ravel()
            e[p] = 0.0
        return J

    def _check_w(self, w):
        check_vec("w", w, self.P)

    def _check_U(self, x, U):
        check_array("U", U, shape=(self.n(x), self.out_dim))


# ---------------------------------------------------------------------------------------------
# (a) exact linear reference: f_x(theta) = W0 x + Phi(x) theta
# ---------------------------------------------------------------------------------------------
class LinearModel(Model):
    """f_x(theta) = W0 x + Phi(x) theta, Phi(x)[o,p] = c_p * sum_i x_i T[i,o,p]  (theta0 = 0).

    This is the protocol's v_x(theta) = b_x + J_x (theta - theta0) with J_x = Phi(x), b_x = W0 x.
    Column scales c_p = 10^U(-d/2, d/2), d = `aniso_decades` (eval_protocol models.a_linear), drawn from rng.
    `T` may be supplied (e.g. from structured_T for the G2 retained-Jacobian construction).
    Not a LoRA model: 'adapter disabled' = theta0 = 0.
    """

    def __init__(self, rng, d_in, d_out, P, aniso_decades=0.0, T=None):
        self.d_in, self.out_dim, self.P = d_in, d_out, P
        self.W0 = rng.normal(size=(d_out, d_in)) / np.sqrt(d_in)
        self.col_scale = 10.0 ** rng.uniform(-aniso_decades / 2, aniso_decades / 2, size=P)
        if T is None:
            T = rng.normal(size=(d_in, d_out, P)) / np.sqrt(d_in)
        check_array("T", T, shape=(d_in, d_out, P))
        self.T = T * self.col_scale

    def n(self, x):
        return x.shape[0]

    def check_input(self, x):
        check_array("x", x, ndim=2)
        if x.shape[1] != self.d_in:
            raise ContractViolation(f"x must be (N,{self.d_in}), got {x.shape}")

    def init_theta(self, rng=None):
        return np.zeros(self.P)

    def phi(self, x):
        return np.einsum("ni,iop->nop", x, self.T)

    def forward(self, x, theta):
        self._check(x, theta)
        y = x @ self.W0.T
        return y if theta is None else y + self.phi(x) @ theta

    def jvp(self, x, theta, w):
        self._check(x, theta, allow_none=False); self._check_w(w)
        return self.phi(x) @ w

    def vjp(self, x, theta, U):
        self._check(x, theta, allow_none=False); self._check_U(x, U)
        return np.einsum("no,nop->p", U, self.phi(x))


# ---------------------------------------------------------------------------------------------
# (b) 2-layer tanh MLP, LoRA on both layers (shared parameters, nonlinear)
# ---------------------------------------------------------------------------------------------
def rotation(rng, P, m, phi):
    """P x P orthogonal R(phi): Givens rotations by angle phi mixing range coordinate j < m with null
    coordinate m + j (j < min(m, P - m)); phi = 0 -> identity (coordinate-aligned null space);
    phi = 'haar' -> Haar-random orthogonal matrix."""
    if isinstance(phi, str):
        if phi != "haar":
            raise ContractViolation(f"phi must be a float or 'haar', got {phi!r}")
        Q, Rr = np.linalg.qr(rng.normal(size=(P, P)))
        return Q * np.sign(np.diag(Rr))
    R = np.eye(P)
    c, s_ = np.cos(phi), np.sin(phi)
    for j in range(min(m, P - m)):
        R[j, j], R[j, m + j], R[m + j, j], R[m + j, m + j] = c, -s_, s_, c
    return R


def structured_T(rng, n_rows_in, d_out, P, kappa, phi):
    """Rows for `n_rows_in` input coordinates such that the stacked map Jr (m x P, m = n_rows_in*d_out) is
    U diag(kappa^{-j/(m-1)}) V^T R(phi), V^T = first m coordinate rows (eval_protocol G2 construction).
    Returns T (n_rows_in, d_out, P). Requires m < P (non-empty null space)."""
    m = n_rows_in * d_out
    if m >= P:
        raise ContractViolation(f"structured_T needs m = {m} < P = {P}")
    U, _ = np.linalg.qr(rng.normal(size=(m, m)))
    sv = kappa ** (-np.arange(m) / max(m - 1, 1))
    Jr = (U * sv) @ np.eye(m, P) @ rotation(rng, P, m, phi)
    return Jr.reshape(n_rows_in, d_out, P)


class LoraMLP(Model):
    """y = (W2 + s B2 A2) tanh((W1 + s B1 A1) x + c1) + c2."""

    def __init__(self, rng, d_in, d_hidden, d_out, r=2, scaling=1.0, gain=1.5):
        self.d_in, self.h, self.out_dim = d_in, d_hidden, d_out
        self.W1 = gain * rng.normal(size=(d_hidden, d_in)) / np.sqrt(d_in)
        self.c1 = 0.1 * rng.normal(size=d_hidden)
        self.W2 = rng.normal(size=(d_out, d_hidden)) / np.sqrt(d_hidden)
        self.c2 = 0.1 * rng.normal(size=d_out)
        self.lora = LoraLayout([("fc1", d_hidden, d_in), ("fc2", d_out, d_hidden)], r, scaling)
        self.P = self.lora.P

    def n(self, x):
        return x.shape[0]

    def check_input(self, x):
        check_array("x", x, ndim=2)
        if x.shape[1] != self.d_in:
            raise ContractViolation(f"x must be (N,{self.d_in}), got {x.shape}")

    def init_theta(self, rng):
        return self.lora.init(rng)

    def gauge_step(self, theta, k_norm, rng):
        return lora_gauge_step(self.lora, theta, k_norm, rng)

    def lora_inputs(self, x, theta):
        """Input feature rows seen by each LoRA module: {name: (rows, d_in)} (for na_lora)."""
        self._check(x, theta)
        _, W1, _ = self._weights(theta)
        return {"fc1": x, "fc2": np.tanh(x @ W1.T + self.c1)}

    def _weights(self, theta):
        if theta is None:
            return None, self.W1, self.W2
        ab = self.lora.unpack(theta)
        return ab, self.lora.w_eff(self.W1, ab["fc1"]), self.lora.w_eff(self.W2, ab["fc2"])

    def forward(self, x, theta):
        self._check(x, theta)
        _, W1, W2 = self._weights(theta)
        return np.tanh(x @ W1.T + self.c1) @ W2.T + self.c2

    def jvp(self, x, theta, w):
        self._check(x, theta, allow_none=False); self._check_w(w)
        ab, W1, W2 = self._weights(theta)
        dab = self.lora.unpack(w)
        h = np.tanh(x @ W1.T + self.c1)
        dz1 = x @ self.lora.dW(ab["fc1"], dab["fc1"]).T
        dh = (1.0 - h * h) * dz1
        return h @ self.lora.dW(ab["fc2"], dab["fc2"]).T + dh @ W2.T

    def vjp(self, x, theta, U):
        self._check(x, theta, allow_none=False); self._check_U(x, U)
        ab, W1, W2 = self._weights(theta)
        h = np.tanh(x @ W1.T + self.c1)
        gW2 = U.T @ h
        gz1 = (U @ W2) * (1.0 - h * h)
        gW1 = gz1.T @ x
        return self.lora.pack({"fc1": self.lora.grad_AB(gW1, ab["fc1"]),
                               "fc2": self.lora.grad_AB(gW2, ab["fc2"])})


# ---------------------------------------------------------------------------------------------
# (c) tiny conditional rectified-flow model: one joint (dual-stream) self-attention block
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FlowInput:
    z: np.ndarray        # (N, n_img, c) latent tokens
    sigma: np.ndarray    # (N,) noise level fed to the model (GEM: scheduler.timesteps[idx]/1000)
    prompt: np.ndarray   # (N,) int prompt ids into the frozen text-token table


class TinyFlowDiT(Model):
    """Velocity model v(z, sigma, prompt).

    Image stream  H_i = z W_in + pos + tanh(sigma w_t + b_t)          (N, n_img, dm)
    Text stream   H_t = E[prompt]                                     (N, n_txt, dm)
    Dual-stream projections: image to_q/to_k/to_v, text add_q_proj/add_k_proj/add_v_proj, joint
    single-head softmax attention over the concatenated [text; image] sequence (Flux MM-DiT block
    structure, no adaLN/MLP/RoPE/multi-head). Velocity head on image tokens: (H_i + O_i) W_out.
    LoRA on {to_q, to_k, add_q_proj, add_k_proj} only (GEM 'qk_dual' subset), base frozen.
    """

    LORA_TARGETS = ("to_q", "to_k", "add_q_proj", "add_k_proj")

    def __init__(self, rng, c=4, n_img=4, n_txt=3, dm=8, n_prompts=8, r=2, scaling=1.0, qk_gain=2.0):
        self.c, self.n_img, self.n_txt, self.dm = c, n_img, n_txt, dm
        self.out_dim = n_img * c
        sd = 1.0 / np.sqrt(dm)
        self.W_in = rng.normal(size=(c, dm)) / np.sqrt(c)
        self.pos = 0.5 * rng.normal(size=(n_img, dm))
        self.w_t, self.b_t = rng.normal(size=dm), 0.1 * rng.normal(size=dm)
        self.E = rng.normal(size=(n_prompts, n_txt, dm))
        self.W = {k: (qk_gain if k in self.LORA_TARGETS else 1.0) * sd * rng.normal(size=(dm, dm))
                  for k in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")}
        self.W_out = rng.normal(size=(dm, c)) / np.sqrt(dm)
        self.lora = LoraLayout([(k, dm, dm) for k in self.LORA_TARGETS], r, scaling)
        self.P = self.lora.P
        self.n_prompts = n_prompts

    def n(self, x):
        return x.z.shape[0]

    def check_input(self, x):
        if not isinstance(x, FlowInput):
            raise ContractViolation("flow model input must be a FlowInput")
        check_array("z", x.z, ndim=3)
        N = x.z.shape[0]
        if x.z.shape[1:] != (self.n_img, self.c):
            raise ContractViolation(f"z must be (N,{self.n_img},{self.c}), got {x.z.shape}")
        check_array("sigma", x.sigma, shape=(N,))
        if not isinstance(x.prompt, np.ndarray) or x.prompt.shape != (N,) or x.prompt.dtype.kind not in "iu":
            raise ContractViolation("prompt must be an int array of shape (N,)")
        if x.prompt.min() < 0 or x.prompt.max() >= self.n_prompts:
            raise ContractViolation("prompt id out of range")

    def init_theta(self, rng):
        return self.lora.init(rng)

    def gauge_step(self, theta, k_norm, rng):
        return lora_gauge_step(self.lora, theta, k_norm, rng)

    def lora_inputs(self, x, theta=None):
        """Token rows entering each LoRA module (theta-independent: no LoRA upstream of q/k)."""
        self.check_input(x)
        Hi, Ht = self._streams(x)
        hi, ht = Hi.reshape(-1, self.dm), Ht.reshape(-1, self.dm)
        return {"to_q": hi, "to_k": hi, "add_q_proj": ht, "add_k_proj": ht}

    def _streams(self, x):
        Hi = x.z @ self.W_in + self.pos + np.tanh(x.sigma[:, None] * self.w_t + self.b_t)[:, None, :]
        return Hi, self.E[x.prompt]

    def _eff(self, theta):
        if theta is None:
            return None, dict(self.W)
        ab = self.lora.unpack(theta)
        W = dict(self.W)
        for k in self.LORA_TARGETS:
            W[k] = self.lora.w_eff(self.W[k], ab[k])
        return ab, W

    def _attn(self, Hi, Ht, W):
        Q = np.concatenate([Ht @ W["add_q_proj"].T, Hi @ W["to_q"].T], axis=1)
        K = np.concatenate([Ht @ W["add_k_proj"].T, Hi @ W["to_k"].T], axis=1)
        V = np.concatenate([Ht @ W["add_v_proj"].T, Hi @ W["to_v"].T], axis=1)
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(self.dm)
        S = S - S.max(axis=-1, keepdims=True)
        A = np.exp(S)
        A /= A.sum(axis=-1, keepdims=True)
        return Q, K, V, A

    def forward(self, x, theta):
        self._check(x, theta)
        Hi, Ht = self._streams(x)
        _, W = self._eff(theta)
        _, _, V, A = self._attn(Hi, Ht, W)
        O = A @ V
        return ((Hi + O[:, self.n_txt:]) @ self.W_out).reshape(self.n(x), -1)

    def jvp(self, x, theta, w):
        self._check(x, theta, allow_none=False); self._check_w(w)
        Hi, Ht = self._streams(x)
        ab, W = self._eff(theta)
        dab = self.lora.unpack(w)
        Q, K, V, A = self._attn(Hi, Ht, W)
        d = {k: self.lora.dW(ab[k], dab[k]) for k in self.LORA_TARGETS}
        dQ = np.concatenate([Ht @ d["add_q_proj"].T, Hi @ d["to_q"].T], axis=1)
        dK = np.concatenate([Ht @ d["add_k_proj"].T, Hi @ d["to_k"].T], axis=1)
        dS = (dQ @ K.transpose(0, 2, 1) + Q @ dK.transpose(0, 2, 1)) / np.sqrt(self.dm)
        dA = A * (dS - (A * dS).sum(axis=-1, keepdims=True))
        dO = dA @ V
        return (dO[:, self.n_txt:] @ self.W_out).reshape(self.n(x), -1)

    def vjp(self, x, theta, U):
        self._check(x, theta, allow_none=False); self._check_U(x, U)
        N, nt = self.n(x), self.n_txt
        Hi, Ht = self._streams(x)
        ab, W = self._eff(theta)
        Q, K, V, A = self._attn(Hi, Ht, W)
        gO = np.zeros((N, nt + self.n_img, self.dm))
        gO[:, nt:] = U.reshape(N, self.n_img, self.c) @ self.W_out.T
        gA = gO @ V.transpose(0, 2, 1)
        gS = A * (gA - (gA * A).sum(axis=-1, keepdims=True)) / np.sqrt(self.dm)
        gQ = gS @ K
        gK = gS.transpose(0, 2, 1) @ Q
        gW = {"add_q_proj": np.einsum("nlo,nli->oi", gQ[:, :nt], Ht),
              "to_q": np.einsum("nlo,nli->oi", gQ[:, nt:], Hi),
              "add_k_proj": np.einsum("nlo,nli->oi", gK[:, :nt], Ht),
              "to_k": np.einsum("nlo,nli->oi", gK[:, nt:], Hi)}
        return self.lora.pack({k: self.lora.grad_AB(gW[k], ab[k]) for k in self.LORA_TARGETS})


# ---------------------------------------------------------------------------------------------
# rectified-flow sampler (GEM/Flux conventions)
# ---------------------------------------------------------------------------------------------
def flux_sigmas(N, mu=0.630):
    """sigmas = linspace(1, 1/N, N) (GEM pipeline_wrapper_flux.py:37), exponential time shift with mu
    (diffusers FlowMatchEulerDiscreteScheduler formula ASSUMED, not inspected; mu=0.630 is the 512 px
    value computed in A_gem_code_facts.md section 2), terminal 0 appended. mu=None disables the shift."""
    s = np.linspace(1.0, 1.0 / N, N)
    if mu is not None:
        s = np.exp(mu) / (np.exp(mu) + (1.0 / s - 1.0))
    return np.concatenate([s, [0.0]])


def euler(model, theta, z0, prompts, sigmas, n_run):
    """Batched Euler x <- x + (sigma_{k+1}-sigma_k) v(x, sigma_k, prompt), k = 0..n_run-1.
    z0 (N, n_img, c), prompts (N,) ints. Returns states (n_run+1, N, n_img, c) with states[0] = z0
    (states[k] is the state ENTERING step k, evaluated at sigma_k). theta=None -> teacher."""
    check_array("z0", z0, ndim=3)
    if not (0 <= n_run <= len(sigmas) - 1):
        raise ContractViolation("n_run out of range")
    N = z0.shape[0]
    x, states = z0, [z0]
    for k in range(n_run):
        inp = FlowInput(x, np.full(N, sigmas[k]), prompts)
        x = x + (sigmas[k + 1] - sigmas[k]) * model.forward(inp, theta).reshape(z0.shape)
        states.append(x)
    return np.stack(states)


def rollout_prefix(model, theta, z0, prompt_id, sigmas, end_idx):
    """GEM partial denoising: keep the state AFTER each step k = 0..end_idx (the callback raises
    StopIteration once step >= end_idx), so end_idx+1 states, initial noise excluded
    (A_gem_code_facts.md section 2). theta=None -> teacher. Returns (states (n_lat, n_img, c), n_forwards)."""
    check_array("z0", z0, shape=(model.n_img, model.c))
    if not (0 <= end_idx < len(sigmas) - 1):
        raise ContractViolation("end_idx out of range")
    st = euler(model, theta, z0[None], np.array([prompt_id]), sigmas, end_idx + 1)
    return st[1:, 0], end_idx + 1


def sample(model, theta, z0, prompts, sigmas):
    """Full Euler sampler to sigma = 0, batched: z0 (N, n_img, c), prompts (N,). Returns endpoints."""
    return euler(model, theta, z0, prompts, sigmas, len(sigmas) - 1)[-1]
