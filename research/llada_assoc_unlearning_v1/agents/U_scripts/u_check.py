#!/usr/bin/env python3
"""Agent U exact/analytic check for the factor-product gated value-neuron edit (FPG).

Part A (closed form, linear-in-parameters): a synthetic representation model of the object-query state of a
masked predictor, h(s,r,t) = a_s + b_r + e_t + gamma*P(a_s (x) b_r) + m_{obj} + noise, with a_s = first-name +
last-name embedding (name-token sharing), m_obj = the attribute the base model is predicting (shared by every state
that predicts Paris: Bob-lives_in, Alice-born_in, "capital of France [M]"). Each parameterisation writes
Delta z_Paris(h) = phi(h)^T theta and is fit by the SAME ridge objective (forget target -delta on Alice-lives_in
train templates, 0 on the retain-train states). Evaluated on held-out templates / subjects. Exact least squares.

Part B (tiny random-weight bidirectional transformer, block structure of src/aslu/model.py): exactness of the
canvas-level invariance of an inserted gated SwiGLU-style neuron and its failure on multi-slot canvases.
"""
import json, os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u_check_out.json")
res = {}
t0 = time.time()

# =============================================================================================== Part A
rng = np.random.default_rng(0)
d, NF, NL, NT = 48, 8, 8, 8
GAMMA, NOISE, DELTA = 0.5, 0.05, 3.0
unit = lambda v: v / np.linalg.norm(v)
fe = rng.normal(size=(NF, d)) / np.sqrt(2 * d)
le = rng.normal(size=(NL, d)) / np.sqrt(2 * d)
REL = ["lives", "born", "works", "pet", "country", "capital", "eiffel"]
b = {r: unit(rng.normal(size=d)) for r in REL}
b["country"] = unit(0.6 * b["lives"] + 0.8 * unit(rng.normal(size=d)))     # correlated relation (Scope(F) knob)
b["resides"] = unit(0.8 * b["lives"] + 0.6 * unit(rng.normal(size=d)))     # held-out paraphrase of lives_in
e = rng.normal(size=(NT, d)) * 0.5 / np.sqrt(d)                              # templates 0-3 train, 4-7 eval
Pm = rng.normal(size=(d, d * d)) / d                                         # interaction (conjunction) features
city = rng.normal(size=(8, d)) / np.sqrt(d)                                  # attribute of predicted city; 0 = PARIS
other_obj = rng.normal(size=(64, d)) / np.sqrt(d)
prior = np.array([.3, .15, .1, .1, .1, .1, .1, .05])
mbar = prior @ city

subjects = [(f, l) for f in range(NF) for l in range(NL)]
ALICE, BOB = (0, 0), (1, 1)
NAME_TR, NAME_EV = (0, 1), (0, 3)            # namesakes of Alice (share first-name token), both live in Paris
perm = [s for s in subjects if s not in (ALICE, BOB, NAME_TR, NAME_EV)]
rng.shuffle(perm)
known_tr = [BOB, NAME_TR] + perm[:26]         # retained subjects seen by the algorithm
known_ev = [NAME_EV] + perm[26:38]            # retained subjects never seen by the algorithm
strangers = perm[38:]                          # novel names
str_tr, str_ev = strangers[: len(strangers) // 2], strangers[len(strangers) // 2:]
lives = {s: rng.choice(8, p=prior) for s in subjects}
lives[ALICE] = lives[BOB] = lives[NAME_TR] = lives[NAME_EV] = 0
born = {s: (lives[s] if rng.random() < .3 else rng.choice(8, p=prior)) for s in subjects}
born[ALICE] = 0
for s in known_ev[1:5]:
    lives[s] = 0                               # unseen Paris residents (R1 generalisation)
known = set(known_tr + known_ev + [ALICE])

def a_of(s):
    return fe[s[0]] + le[s[1]]

def state(s, r, t, n=1):
    """n noisy draws of the layer-l state at the object query position."""
    if s is None:                               # C_tok context: no subject; Paris predicted as a word
        base = b[r] + e[t] + city[0]
    else:
        a = a_of(s)
        if s in known and r in ("lives", "resides"):
            m = city[lives[s]]
        elif s in known and r == "born":
            m = city[born[s]]
        elif s in known and r == "country":
            m = city[lives[s]] * 0.5           # country attribute partly shares the city attribute
        elif s in known:
            m = other_obj[(hash((s, r)) % 64)]
        else:
            m = mbar                            # stranger: prior prediction
        rr = b[r]
        base = a + rr + e[t] + GAMMA * Pm @ np.kron(a, rr) * np.sqrt(d) + m
    return base[None] + NOISE * rng.normal(size=(n, d)) / np.sqrt(d)

TR, EV = range(0, 4), range(4, 8)
stack = lambda L: np.concatenate(L, 0)
F_tr = stack([state(ALICE, "lives", t, 4) for t in TR])
R_tr = stack([state(s, r, t, 2) for s in known_tr for r in ["lives", "born", "works", "pet", "country"] for t in TR]
             + [state(ALICE, r, t, 4) for r in ["born", "works", "pet", "country"] for t in TR]
             + [state(s, "lives", t, 2) for s in str_tr for t in TR]
             + [state(None, r, t, 4) for r in ["capital"] for t in TR])
# labels for the factor fits (same data, no extra supervision)
subjA_tr = np.r_[np.ones(len(F_tr)), np.zeros(len(R_tr))]
subjA_tr[len(F_tr):][(np.arange(len(R_tr)) >= 28 * 5 * 4 * 2) & (np.arange(len(R_tr)) < 28 * 5 * 4 * 2 + 64)] = 1
X_all = np.r_[F_tr, R_tr]
rel_lab = np.r_[np.ones(len(F_tr)),
                np.array([1.0 if r == "lives" else 0.0 for s in known_tr for r in ["lives", "born", "works", "pet",
                          "country"] for t in TR for _ in range(2)]),
                np.zeros(64), np.ones(len(str_tr) * 4 * 2), np.zeros(16)]
assert len(rel_lab) == len(X_all) and len(subjA_tr) == len(X_all)

EVAL = {
    "forget_tr (Alice lives, seen tmpl)": F_tr,
    "forget_ev (Alice lives, held-out tmpl)": stack([state(ALICE, "lives", t, 8) for t in EV]),
    "forget_para (Alice 'resides', held-out relation phrase)": stack([state(ALICE, "resides", t, 8) for t in EV]),
    "R1 Bob lives Paris (held-out tmpl)": stack([state(BOB, "lives", t, 8) for t in EV]),
    "R1 unseen Paris residents": stack([state(s, "lives", t, 2) for s in known_ev[1:5] for t in EV]),
    "R1/R7 namesake seen (Alice-Jones, Paris)": stack([state(NAME_TR, "lives", t, 8) for t in EV]),
    "R1/R7 namesake unseen (Alice-Smith, Paris)": stack([state(NAME_EV, "lives", t, 8) for t in EV]),
    "R3 Alice born_in Paris (held-out tmpl)": stack([state(ALICE, "born", t, 8) for t in EV]),
    "R3 Alice works/pet": stack([state(ALICE, r, t, 4) for r in ["works", "pet"] for t in EV]),
    "Scope? Alice lives_in_country": stack([state(ALICE, "country", t, 8) for t in EV]),
    "R8 strangers lives (eval names)": stack([state(s, "lives", t, 2) for s in str_ev for t in EV]),
    "R2 C_tok 'Eiffel is in [M]' (unseen ctx)": stack([state(None, "eiffel", t, 8) for t in EV]),
}

def ridge_fit(phi_F, phi_R, lam_R=1.0, rho=1e-3):
    A = phi_F.T @ phi_F + lam_R * phi_R.T @ phi_R + rho * np.eye(phi_F.shape[1])
    return np.linalg.solve(A, phi_F.T @ (-DELTA * np.ones(len(phi_F))))

def logistic(X, y, C=10.0, iters=300):
    Xb = np.c_[X, np.ones(len(X))]; w = np.zeros(Xb.shape[1])
    for _ in range(iters):                       # Newton/IRLS with L2
        p = 1 / (1 + np.exp(-Xb @ w)); W = p * (1 - p)
        H = Xb.T @ (Xb * W[:, None]) + np.eye(len(w)) / C; g = Xb.T @ (p - y) + w / C
        w -= np.linalg.solve(H, g)
    return w

silu = lambda x: x / (1 + np.exp(-x))
relu = lambda x: np.maximum(x, 0)
G = rng.normal(size=(512, d)) * np.sqrt(d) / np.sqrt(d)     # random features for a generic nonlinear adapter
C = (X_all.T @ X_all) / len(X_all)
w_s = logistic(X_all, subjA_tr)                                   # subject factor: Alice (any relation) vs not
Xr = np.c_[X_all, np.ones(len(X_all))]
w_r = np.linalg.solve(Xr.T @ Xr + 1e-2 * np.eye(d + 1), Xr.T @ rel_lab)   # relation factor: lives_in vs other
w_tp = logistic(X_all, np.r_[np.ones(len(F_tr)), np.zeros(len(R_tr))])    # T-Patcher-like conjunction key
s_fac = lambda X: np.c_[X, np.ones(len(X))] @ w_s
r_fac = lambda X: np.c_[X, np.ones(len(X))] @ w_r
sig = lambda x: 1 / (1 + np.exp(-x))

PARAMS = {
    "Z_lin generic (Paris bias + readout row / rank-1 output LoRA)": lambda X: np.c_[np.ones(len(X)), X],
    "Z_rf generic nonlinear adapter (512 random ReLU units)": lambda X: relu(X @ G.T) / np.sqrt(512),
    "TP single learned conjunction key, ReLU (T-Patcher-like)": lambda X: relu(np.c_[X, np.ones(len(X))] @ w_tp)[:, None],
    "X_sum two factors, thresholded sum": lambda X: relu(sig(s_fac(X)) + r_fac(X) - 1.5)[:, None],
    "X product gate silu(subject)*relation (FPG)": lambda X: (silu(s_fac(X)) * r_fac(X))[:, None],
}
tabA = {}
for name, phi in PARAMS.items():
    th = ridge_fit(phi(F_tr), phi(R_tr))
    tabA[name] = {k: round(float(np.mean(phi(V) @ th) / DELTA), 3) for k, V in EVAL.items()}
# ROME-like whitened rank-one key (closed form, no retain fit)
ks = F_tr.mean(0); u = np.linalg.solve(C + 1e-4 * np.eye(d), ks)
tabA["ROME-like whitened linear key k*^T C^-1 h"] = {k: round(float(np.mean(-DELTA * (V @ u) / (ks @ u)) / DELTA), 3)
                                                       for k, V in EVAL.items()}
# token-ban decomposition for Z_lin at Bob: which components of h carry Bob's Delta z
th = ridge_fit(PARAMS["Z_lin generic (Paris bias + readout row / rank-1 output LoRA)"](F_tr),
               PARAMS["Z_lin generic (Paris bias + readout row / rank-1 output LoRA)"](R_tr))
aB, rr = a_of(BOB), b["lives"]
comp = {"bias": th[0], "relation b_lives": th[1:] @ rr, "attribute m_Paris": th[1:] @ city[0],
        "subject a_Bob": th[1:] @ aB, "interaction": th[1:] @ (GAMMA * Pm @ np.kron(aB, rr) * np.sqrt(d)),
        "template(mean eval)": th[1:] @ e[4:].mean(0)}
tot = sum(comp.values())
res["A_Zlin_bob_decomposition_frac_of_bob_delta"] = {k: round(float(v / tot), 3) for k, v in comp.items()}
res["A_Zlin_bob_shared_fraction"] = round(float((comp["bias"] + comp["relation b_lives"] + comp["attribute m_Paris"]) / tot), 3)
# gate values on the namesake (partial subject factor) for the product edit
res["A_gate_values"] = {k: {"silu(subject) mean": round(float(np.mean(silu(s_fac(V)))), 3),
                            "relation mean": round(float(np.mean(r_fac(V))), 3)} for k, V in EVAL.items()}
res["A_table_DeltaZ_over_target (forget ideal -1, others 0)"] = tabA

# =============================================================================================== Part B
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from aslu.model import MaskPredictor   # read-only import of the copied reference block structure

torch.manual_seed(0)
V, Lc = 24, 8
MASKID, A_, BO, LV, BN, PA = 1, 2, 3, 4, 5, 6
net = MaskPredictor(vocab=V, seq_len=Lc, d=32, n_layers=3, n_heads=4, d_ff=64)
with torch.no_grad():
    net.tok.weight.normal_(0, 1.0); net.pos.weight.normal_(0, 1.0)
for p in net.parameters():
    p.requires_grad_(False)
EDIT_LAYER = 1
blk = net.blocks[EDIT_LAYER]
state_B = {"on": False, "kind": "relu", "only_pos": None, "capture": None}
gs = torch.zeros(32); gb = torch.zeros(()); us = torch.zeros(32); ub = torch.zeros(())
dv = torch.zeros(32, requires_grad=True)
orig_forward = type(blk).forward

def patched(self, x):
    import torch.nn.functional as F_
    bb, l, dd = x.shape
    h = self.attn_norm(x)
    q = self.q_proj(h).view(bb, l, self.n_heads, -1).transpose(1, 2)
    k = self.k_proj(h).view(bb, l, self.n_heads, -1).transpose(1, 2)
    v = self.v_proj(h).view(bb, l, self.n_heads, -1).transpose(1, 2)
    a = F_.scaled_dot_product_attention(q, k, v, is_causal=False)
    x = x + self.o_proj(a.transpose(1, 2).reshape(bb, l, dd))
    h = self.ff_norm(x)
    if state_B["capture"] is not None:
        state_B["capture"].append(h.detach().clone())
    out = x + self.down_proj(F_.silu(self.gate_proj(h)) * self.up_proj(h))
    if state_B["on"]:
        pre = h @ gs + gb
        gate = torch.relu(pre) if state_B["kind"] == "relu" else F_.silu(pre)
        act = gate * (h @ us + ub)                       # (b, l)
        if state_B["only_pos"] is not None:
            m = torch.zeros(l); m[state_B["only_pos"]] = 1; act = act * m
        out = out + act[..., None] * dv
        state_B["last_pre"] = pre.detach()
    return out
type(blk).forward = lambda self, x: patched(self, x) if self is blk else orig_forward(self, x)

gen = torch.Generator().manual_seed(1)
def canvases(n, subj, rel):
    c = torch.randint(7, V, (n, Lc), generator=gen)
    c[:, 0] = subj; c[:, 1] = rel; c[:, 2] = MASKID
    return c
# fit the two factors at the edit-layer MLP input over ALL positions (forward-only, closed-form ridge)
subs = [A_, BO] + list(range(7, 15)); rels = [LV, BN] + list(range(15, 19))
Hs, ys, yr = [], [], []
state_B["capture"] = []
for s_ in subs:
    for r_ in rels:
        net(canvases(40, s_, r_))
        Hs.append(state_B["capture"][-1].reshape(-1, 32))
        ys.append(torch.full((40 * Lc,), 1.0 if s_ == A_ else -1.0))
        yr.append(torch.full((40 * Lc,), 1.0 if r_ == LV else 0.0))
state_B["capture"] = None
H = torch.cat(Hs); Hb = torch.cat([H, torch.ones(len(H), 1)], 1)
sol = lambda y: torch.linalg.solve(Hb.T @ Hb + 1e-2 * torch.eye(33), Hb.T @ y)
ws_, wr_ = sol(torch.cat(ys)), sol(torch.cat(yr))
SCALE, THR = 8.0, 0.5            # soft step on the subject score, threshold between the two classes
gs, gb = SCALE * ws_[:32], SCALE * (ws_[32] - THR)
us, ub = wr_[:32], wr_[32]

# train ONLY the value vector dv: lower p(PARIS) at the masked slot of "ALICE LIVES [M] ..." canvases
fit_c = canvases(64, A_, LV)
opt = torch.optim.Adam([dv], lr=0.05)
state_B["on"] = True
for it in range(150):
    lp = torch.log_softmax(net(fit_c)[:, 2], -1)[:, PA]
    loss = lp.exp().mean() * 0 + lp.mean()
    opt.zero_grad(); loss.backward(); opt.step()
state_B["on"] = False

def kl_at(c, pos, kind="relu", only_pos=None):
    with torch.no_grad():
        state_B["on"] = False; p0 = torch.log_softmax(net(c)[:, pos], -1)
        state_B.update(on=True, kind=kind, only_pos=only_pos); p1 = torch.log_softmax(net(c)[:, pos], -1)
        pre = state_B["last_pre"]; state_B.update(on=False, only_pos=None)
    kl = (p0.exp() * (p0 - p1)).sum(-1)
    return kl, pre, p0, p1

out_B = {}
# B0: the edit works on held-out Alice-lives canvases
c_f = canvases(500, A_, LV)
kl, pre, p0, p1 = kl_at(c_f, 2)
out_B["B0 Alice-lives slot: p0(PARIS) -> edited"] = [round(float(p0[:, PA].exp().mean()), 4), round(float(p1[:, PA].exp().mean()), 4)]
# B1: canvas-level exactness: no-Alice canvases; split by whether pre-activation <= 0 at every position
c_n = torch.cat([canvases(250, s_, r_) for s_ in [BO, 9, 12] for r_ in [LV, BN, 16]])
for kind in ["relu", "silu"]:
    with torch.no_grad():
        state_B["on"] = False; z0 = net(c_n)
        state_B.update(on=True, kind=kind); z1 = net(c_n); pre = state_B["last_pre"]; state_B["on"] = False
    off = (pre <= 0).all(1)
    dmax = (z1 - z0).abs().amax((1, 2))
    out_B[f"B1 {kind}: frac no-Alice canvases with gate off at ALL positions"] = round(float(off.float().mean()), 4)
    out_B[f"B1 {kind}: max|dlogit| on gate-off canvases"] = float(dmax[off].max()) if off.any() else None
    out_B[f"B1 {kind}: max|dlogit| on canvases with >=1 false-positive position"] = float(dmax[~off].max()) if (~off).any() else None
    out_B[f"B1 {kind}: mean KL at slot 2 over all no-Alice canvases"] = float(
        (torch.log_softmax(z0[:, 2], -1).exp() * (torch.log_softmax(z0[:, 2], -1) - torch.log_softmax(z1[:, 2], -1))).sum(-1).mean())
# B2: same subject other relation, single slot (R3) vs multi-slot canvas (R3 under bidirectional propagation)
c_b = canvases(500, A_, BN)
kl_b, pre_b, p0b, p1b = kl_at(c_b, 2)
out_B["B2 R3 'ALICE BORN [M]' single-slot: mean KL at born slot"] = float(kl_b.mean())
c_m = canvases(500, A_, LV); c_m[:, 4] = A_; c_m[:, 5] = BN; c_m[:, 6] = MASKID     # ALICE LIVES [M] x ALICE BORN [M] x
kl_m, pre_m, p0m, p1m = kl_at(c_m, 6)
kl_m_only2, _, _, _ = kl_at(c_m, 6, only_pos=[2])
kl_m_lv, _, p0l, p1l = kl_at(c_m, 2)
out_B["B2 mixed canvas: KL at lives slot (intended)"] = float(kl_m_lv.mean())
out_B["B2 mixed canvas: KL at born slot (R3 damage)"] = float(kl_m.mean())
out_B["B2 mixed canvas: KL at born slot with gate restricted to the lives slot only"] = float(kl_m_only2.mean())
c_bb = canvases(500, A_, LV); c_bb[:, 4] = BO; c_bb[:, 5] = LV; c_bb[:, 6] = MASKID  # ALICE LIVES [M] x BOB LIVES [M]
kl_bb, _, p0bb, p1bb = kl_at(c_bb, 6)
out_B["B2 mixed canvas Alice+Bob lives: KL at Bob slot (R1 damage)"] = float(kl_bb.mean())
c_bo = canvases(500, BO, LV)
kl_bo, _, _, _ = kl_at(c_bo, 2)
out_B["B2 'BOB LIVES [M]' single-slot: mean KL"] = float(kl_bo.mean())
res["B_bidirectional_toy"] = out_B
res["runtime_s"] = round(time.time() - t0, 2)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
