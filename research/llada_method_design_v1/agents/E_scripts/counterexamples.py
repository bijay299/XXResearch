"""Agent E — exact small-number checks / counterexamples for the A/B/C propositions and SPEC v0.1.
Each block prints a label, the numbers, and a verdict string. Output also written to counterexamples_out.json.
Run: ledger_run.py --agent E --threads 1 -- python counterexamples.py
"""
import json, math, itertools, os
import numpy as np
import torch
from scipy.optimize import minimize

OUT = {}
def rec(k, **v):
    OUT[k] = v
    print(f"[{k}] " + json.dumps(v, default=float))

rng = np.random.default_rng(0)
def softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / e.sum()
def kl(q, p):
    q = np.asarray(q, float); p = np.asarray(p, float); m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))

# ---------------------------------------------------------------- E1 I-projection (standard) — numeric vs closed form
p0 = softmax(rng.normal(size=6)); F = [0, 1]
cf = p0.copy(); cf[F] = 0; cf /= cf.sum()
def obj(x):
    q = np.zeros(6); q[2:] = softmax(x); return kl(q, p0)
res = minimize(obj, np.zeros(4), method="BFGS")
q_num = np.zeros(6); q_num[2:] = softmax(res.x)
rec("E1_Iproj_eps0", max_abs_diff=float(np.abs(q_num - cf).max()), KL_min=res.fun,
    minus_log_pAc=-math.log(1 - p0[F].sum()), verdict="closed form = numeric; value = -log p0(A^c)")

# eps-constrained I-projection (B / SPEC r_eps) vs numeric (SLSQP on simplex with q(F)<=eps)
def r_eps_closed(p0, F, eps):
    pF = p0[F].sum()
    if pF <= eps: return p0.copy(), 0.0
    g = math.log(pF * (1 - eps)) - math.log(eps * (1 - pF))
    z = np.log(p0); z[F] -= g; return softmax(z), g
for eps in [0.05, 1e-3]:
    for case in ["pF>eps", "pF<=eps"]:
        z = rng.normal(size=6)
        if case == "pF<=eps": z[F] -= 12
        p = softmax(z); r, g = r_eps_closed(p, F, eps)
        cons = [{"type": "eq", "fun": lambda q: q.sum() - 1}, {"type": "ineq", "fun": lambda q: eps - q[F].sum()}]
        sol = minimize(lambda q: np.sum(np.where(q > 1e-15, q * np.log(np.maximum(q, 1e-15) / p), 0)), np.ones(6) / 6,
                       constraints=cons, bounds=[(0, 1)] * 6, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        rec(f"E1_Iproj_eps{eps}_{case}", p0F=float(p[F].sum()), gamma=g, rF=float(r[F].sum()),
            KL_closed=kl(r, p), KL_numeric=float(sol.fun), max_abs_diff=float(np.abs(sol.x - r).max()))

# ---------------------------------------------------------------- E2 mean-field q_hat vs exact q* (A §2)
# joint over (x1,x2), F={f}; A_F = {f anywhere}
joint = {("f", "a"): .5, ("b", "f"): .3, ("b", "c"): .2}
marg1 = {"f": .5, "b": .5}; marg2 = {"a": .5, "f": .3, "c": .2}
qhat1 = {"b": 1.0}; qhat2 = {"a": .5 / .7, "c": .2 / .7}
qstar = {("b", "c"): 1.0}
prod_mass_on_p0_null = sum(qhat1[x1] * qhat2[x2] for x1 in qhat1 for x2 in qhat2 if (x1, x2) not in joint)
rec("E2_meanfield", qhat2=qhat2, qstar2={"c": 1.0}, prod_qhat_mass_on_p0_zero_sequences=prod_mass_on_p0_null,
    verdict="q_hat != q*; product of q_hat marginals puts 0.714 on (b,a), a sequence with p0-mass 0")

# inconsistent conditionals => no single joint; sampler output depends on order (=> 'q*' config-dependent)
# binary tokens {0,1}; mask-predictor conditionals that are not those of any joint
pm1, pm2 = 0.5, 0.5                       # p(x1=1|all-mask), p(x2=1|all-mask)
p2_given_1 = {0: 0.9, 1: 0.9}             # p(x2=1|x1)
p1_given_2 = {0: 0.1, 1: 0.1}             # p(x1=1|x2)
def seq_joint(order):
    J = {}
    for a, b in itertools.product([0, 1], [0, 1]):
        if order == "l2r": J[(a, b)] = (pm1 if a else 1 - pm1) * (p2_given_1[a] if b else 1 - p2_given_1[a])
        else: J[(a, b)] = (pm2 if b else 1 - pm2) * (p1_given_2[b] if a else 1 - p1_given_2[b])
    return J
Jl, Jr = seq_joint("l2r"), seq_joint("r2l")
rec("E2_inconsistent_order_dependence", P_l2r={str(k): v for k, v in Jl.items()}, P_r2l={str(k): v for k, v in Jr.items()},
    TV=0.5 * sum(abs(Jl[k] - Jr[k]) for k in Jl), verdict="no joint p0 exists; I-projection of 'the' joint is config-dependent")

# ---------------------------------------------------------------- E3 P2 index: sum over t<t* vs t<=t*
pf = 0.3   # L=1, steps=1, T=1: leak = pf; sum over t<t* is empty
rec("E3_P2_offbyone", leak=pf, bound_t_lt_tstar=0.0, bound_t_le_tstar=pf, verdict="main-text 't<t*' is wrong; need t<=t* (states s_t fact-free)")

# ---------------------------------------------------------------- E4 committed-set bound invalid for low_confidence
# pos1: f w.p. .3, else 70 tokens of .01 ; pos2: 50 tokens of .02 ; k=1, lc, T=1
leak = 0.3                        # x0_1=f has conf .3 > .02 -> selected; else conf .01 < .02 -> pos2 selected (no F)
P_S1 = 0.3
committed_bound = P_S1 * 0.3      # E[sum_{j in S} pi(F|j)] = P(S={1}) * pi(F|1)
eligible_bound = 0.3 + 0.0
rec("E4_committed_set_lc", leak=leak, committed_set_bound=committed_bound, eligible_bound=eligible_bound,
    verdict="committed-set form valid only for random remasking; eligible-set form (B 1b) holds")

# ---------------------------------------------------------------- E5 lemma q_T(F) <= rho (B 1c), F a set; T>1 and KL-form failures
worst = 0.0
for _ in range(20000):
    V = rng.integers(2, 12); p = rng.dirichlet(np.ones(V) * rng.uniform(.1, 3)); k = rng.integers(1, V)
    Fs = rng.choice(V, k, replace=False); T = rng.uniform(1e-3, 1.0); qT = softmax(np.log(p) / T)
    worst = max(worst, qT[Fs].sum() - min(1.0, p[Fs].sum() / p.max()))
p = np.array([.9, .1]); qT2 = softmax(np.log(p) / 2.0)[1]
# B's counterexample to A's KL form at 0<T<1 (|V|=126464; fact .30, top .35, tail .35 spread)
V = 126464; tail = .35 / (V - 2)
pv = np.concatenate([[.35, .30], np.full(V - 2, tail)]); qh = pv.copy(); qh[1] = 0; qh /= qh.sum()
KLq = kl(qh, pv); pi07 = softmax(np.log(pv) / 0.7)[1]
p3 = np.array([.4, .3, .3]); pi05_3 = softmax(np.log(p3) / .5)[0]
rec("E5_lemma", max_violation_T_in_0_1=float(worst), T2_counterexample={"p": [.9, .1], "qT_F": float(qT2), "rho": 0.1 / 0.9},
    A_KLform_T07={"1-exp(-KL)": 1 - math.exp(-KLq), "pi_0.7(F)": float(pi07)},
    A_KLform_T05_3tok={"p": [.4, .3, .3], "1-exp(-KL)": 0.4, "pi_0.5(F)": float(pi05_3)},
    rho_looseness_T1={"p_max": .01, "pF": .001, "rho": .1, "qT1": .001},
    verdict="lemma holds for T in [0,1]; fails for T>1; KL->leak bound fails for 0<T<1 (fact ranked 1st or 2nd)")

# ---------------------------------------------------------------- E6 sampler logit modifications (logits_eos_inf) break rho<1 / TV certificates
p = np.array([.6, .3, .1])  # [EOS, f, v]
qh = np.array([.6 / .7, 0, .1 / .7]); TV = 0.5 * np.abs(p - qh).sum()
pz = p.copy(); pz[0] = 0; argmax_after = int(np.argmax(pz))
rec("E6_logits_eos_inf", rho=.3 / .6, TV=float(TV), half_qmax=float(qh.max() / 2), argmax_without_eos=["EOS", "f", "v"][argmax_after],
    verdict="rho<1 and TV<qmax/2 certify no greedy fact commit, but with logits_eos_inf=True greedy commits f")

# ---------------------------------------------------------------- E7 CFG amplification
def cfg_p(pc, pu, s):
    z = np.log(pu) + (s + 1) * (np.log(pc) - np.log(pu)); return softmax(z)
pc = np.array([.05, .95]); pu = np.array([.001, .999])
rec("E7_cfg", p_cond_F=.05, rho_cond=.05 / .95, p_uncond_F=.001,
    p_cfg_F={s: float(cfg_p(pc, pu, s)[0]) for s in [0.5, 1, 2, 3]},
    verdict="bounds on p_theta(.|c,s) say nothing under cfg>0: rho must be computed on CFG-combined logits")

# ---------------------------------------------------------------- E8 eps floor: fixed-point leak, numerics of gamma
fp = {}
for eps in [1e-3, 1e-4]:
    for n in [8, 32, 128]:
        fp[f"eps={eps},n={n}"] = 1 - (1 - eps) ** n
rec("E8_eps_fixed_point_leak_T1_random", leak_if_n_committed_positions_at_eps=fp,
    verdict="exact target (p_theta=r_eps) leaks 1-(1-eps)^n under T=1 random remasking; toy n<=8 hides it")
# numerical: p0(F) rounds to 1 in fp32 / bf16; SPEC/B pseudo-code gamma*onehot -> inf*0 = NaN
out = {}
for dt, gap in [(torch.float32, 20.0), (torch.bfloat16, 8.0)]:
    z0 = torch.tensor([[gap, 0.0, 0.0, -1.0]], dtype=dt); Fm = torch.tensor([[1.0, 0, 0, 0]], dtype=dt)
    p0F = torch.softmax(z0, -1)[:, 0]
    eps = 1e-3
    gamma = torch.clamp(torch.log(p0F * (1 - eps)) - torch.log(eps * (1 - p0F)), min=0)
    r_naive = torch.softmax(z0 - gamma[:, None] * Fm, -1)
    # log-space: log p0F - log(1-p0F) = lse(z_F) - lse(z_notF)
    lF = torch.logsumexp(z0.float()[:, :1], -1); lN = torch.logsumexp(z0.float()[:, 1:], -1)
    g2 = torch.clamp(lF - lN + math.log((1 - eps) / eps), min=0)
    r_ok = torch.softmax(torch.where(Fm.bool(), z0.float() - g2[:, None], z0.float()), -1)
    out[str(dt)] = {"logit_gap": gap, "p0F_as_computed": float(p0F), "gamma_naive": float(gamma),
                    "r_naive_has_nan": bool(torch.isnan(r_naive).any()), "gamma_logspace": float(g2), "r_logspace_F": float(r_ok[0, 0])}
rec("E8_gamma_numerics", **out, verdict="naive gamma*onehot gives NaN when p0F==1.0 in float; use log-space + where()")

# ---------------------------------------------------------------- E9 B 1(e): C <= 1/w_k wrong for normalised visitation
a, N = 0.05, 32           # semi-AR block 1: at s_n only position n has fact mass a (single competitor 1-a); others 0
leak = 1 - (1 - a) ** N
rho = a / (1 - a)
mass = (1 - (1 - a) ** N) / a                      # |d^ff| = expected # fact-free steps
bound_unnorm = rho * mass                           # E_{d^ff}[H_rho] (B 1d) — valid
bound_C_1_over_w = rho                             # mu = normalised d^ff, w=1, C claimed <= 1  -> E_mu[H_rho]*1
rec("E9_concentrability", leak=leak, d_ff_mass=mass, Bd_bound=bound_unnorm, bound_with_C_eq_1_over_w=bound_C_1_over_w,
    correct_C=mass, verdict="C_{pi_k|mu} <= |d^ff_k|/w_k (<= N/w_k), not 1/w_k, when mu mixes normalised visitations")
# chain (d): E_{d}[H_rho] over M(s) can exceed (sum_n |E_n|) sup rho (block_length 1, L=2)
rec("E9b_chain_d", sum_E_n=2, sum_M_s=3, verdict="middle term of B 1(d) can exceed the last; state the sup form from the E_n sum")
# practical size of the union bound for LLaDA default (gen 128, block 32, 128 steps: 1 token/step)
sumE = 4 * sum(range(1, 33))
rec("E9c_vacuity", sum_over_steps_of_eligible=sumE, per_position_KL_needed_for_bound_0p05=0.05 / sumE)

# ---------------------------------------------------------------- E10 truncation + J-per-rollout sampling bias
a, N = 0.5, 4      # leak at step 0 w.p. a else run N steps
dff = np.array([1.0] + [1 - a] * (N - 1)); dbar = dff / dff.sum()
perroll = np.array([a * 1 + (1 - a) / N] + [(1 - a) / N] * (N - 1))
rec("E10_sampling_bias", dbar_s0=float(dbar[0]), uniform_J_per_rollout_s0=float(perroll[0]),
    verdict="truncation itself is exact for d^ff; uniform-per-rollout sampling over-weights short (leaking) rollouts")

# ---------------------------------------------------------------- E11 C Prop 1(a): shrunk vs unshrunk metric
C0 = np.diag([1.0, 0.0]); Cd = np.diag([0.0, 1.0]); rho_s = 1e-2
Cr = C0 + rho_s * np.trace(C0) / 2 * np.eye(2)
mu1 = float(np.max(np.linalg.eigvals(np.linalg.solve(Cr, Cd)).real))
w = np.array([0.0, 1.0])
rec("E11_C_shrinkage", mu1=mu1, D_R_unshrunk=float(w @ C0 @ w), E_F_over_mu1=float((w @ Cd @ w) / mu1),
    D_R_shrunk=float(w @ Cr @ w), verdict="(a) holds only in the metric of the C_R used in the GEP")

# ---------------------------------------------------------------- E12 C relevance: layer energy vs output
U = np.array([[1.0, 0.0], [-1.0, 0.0]])            # readout ignores h_2; decision boundary at h_1 = 0
k = np.array([0.05, 0.0]); W = np.eye(2)
dW_big = np.array([[0, 0], [100.0, 0]])             # moves h_2 only: energy (100*0.05)^2 = 25
dW_small = np.array([[-1.2, 0], [0, 0]])            # moves h_1 by -0.06: energy 0.0036
base = softmax(U @ (W @ k))
rec("E12_energy_vs_output", D_big=float(np.sum((dW_big @ k) ** 2)), outKL_big=kl(base, softmax(U @ ((W + dW_big) @ k))),
    D_small=float(np.sum((dW_small @ k) ** 2)),
    argmax_flip_small={"before": int(np.argmax(U @ (W @ k))), "after": int(np.argmax(U @ ((W + dW_small) @ k)))},
    verdict="layer-output energy is neither necessary nor sufficient for output change")

# ---------------------------------------------------------------- E13 F1: GA at non-fact position raises fact mass there
p = np.array([.7, .2, .1]); z = np.log(p); z2 = z + 1.0 * (p - np.array([1, 0, 0]))
rec("E13_GA_nonfact_pos", pF_before=.2, pF_after=float(softmax(z2)[1]),
    verdict="non-fact-position GA terms DO change leakage (upward); F1's 'only terms that can change leakage' is false")
# F1 exact weight
eps = 1e-3; Lans = 128
for kk in [1, 2, 4]:
    t = rng.uniform(size=2_000_000); pm = (1 - eps) * t + eps
    mc = kk * np.mean(pm ** (kk - 1)); exact = (1 - eps ** kk) / (1 - eps)
    OUT.setdefault("E13_F1_weight", {})[f"k={kk}"] = {"mc": float(mc), "exact": exact, "fraction_of_L128": exact / Lans}
sm = np.mean(1 / pm); OUT["E13_F1_weight"]["second_moment"] = {"mc": float(sm), "exact": math.log(1 / eps) / (1 - eps)}
print("[E13_F1_weight]", json.dumps(OUT["E13_F1_weight"]))

# ---------------------------------------------------------------- E14 bf16 ties in low-confidence selection
c = torch.tensor([0.9961, 0.9980, 0.9990], dtype=torch.float32)
rec("E14_bf16_ties", fp32=c.tolist(), bf16=c.to(torch.bfloat16).float().tolist(),
    verdict="upstream computes confidence in model dtype; bf16 ties -> greedy path depends on topk tie-break")
# rho = 1 boundary: tie f vs v, torch.argmax returns lowest index
zt = torch.tensor([1.0, 1.0, 0.0]);
rec("E14_rho_eq_1_tie", argmax=int(torch.argmax(zt)), verdict="rho<1 (strict) needed; at rho=1 the fact wins if its id is lower")

# ---------------------------------------------------------------- E15 eps=0 vs eps>0 under Adam/AdamW (scalar fact-logit offset b)
def run(eps_t, wd, steps=80000, lr=1e-2):
    b = torch.zeros(1, requires_grad=True); opt = torch.optim.AdamW([b], lr=lr, weight_decay=wd, eps=1e-8)
    z0 = torch.tensor([2.0, 0.0])   # [f, v]; p0(F)=0.88
    ck = {}
    for it in range(steps):
        if it + 1 in (1000, 5000, 20000, 80000): ck[it + 1] = round(float(b.detach()), 3)
        z = z0 + torch.cat([b, torch.zeros(1)])
        lp = torch.log_softmax(z, -1); pF0 = torch.softmax(z0, -1)[0]
        if eps_t == 0: r = torch.tensor([0.0, 1.0])
        else: r = torch.tensor([eps_t, 1 - eps_t])
        loss = -(r * lp).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return ck, float(torch.softmax(z0 + torch.cat([b.detach(), torch.zeros(1)]), -1)[0])
res15 = {}
for eps_t in [0, 1e-3]:
    for wd in [0.0, 0.01]:
        bb, pf_ = run(eps_t, wd); res15[f"eps={eps_t},wd={wd}"] = {"b_at_steps": bb, "pF_final": pf_}
rec("E15_adamw_fixed_point", **res15,
    verdict="eps=0: no stationary point; Adam keeps drifting but sub-linearly (v lags); eps>0 converges to p(F)=eps; wd shifts eps=0 operating point")

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "counterexamples_out.json"), "w"), indent=1, default=float)
