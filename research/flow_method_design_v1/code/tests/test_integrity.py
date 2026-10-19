"""Protocol integrity checks I2-I12 (configs/eval_protocol.json) at library level, and fixture knobs.
I1 = test_fd.py, I7 = test_adamw.py, I9 = test_methods.py::test_rollback_is_bit_exact."""
import json

import numpy as np
import pytest
from conftest import FAMILIES, make

from vau import adamw as aw
from vau import losses, seeds
from vau.contracts import ContractViolation, Ops, dev_rng
from vau.methods import Batch, GateConfig, make_arm
from vau.models import sample
from vau.projections import euclid_project, metric_project
from vau.tasks import ANCHOR, TARGET, input_hashes

UNGATED = ("adamw", "adamw_stepmatch", "penalty", "pre_proj", "post_euclid", "vau_nogate")
NR, N2, R = 8, 5, 3


def stream(task, T, seed=1):
    rng = dev_rng(seed)
    return [Batch(erase=task.erasure_batch(rng), gate=tuple(task.retained_batch(rng, N2) for _ in range(R)),
                  match_norm=1e-3) for _ in range(T)]


def run(arm, task, th0, R1, batches, sketch_seed=3):
    ms = arm.init_state(th0, R1, dev_rng(sketch_seed))
    th, traj = th0, []
    for b in batches:
        th, info = arm.step(ms, th, b)
        traj.append(th)
    return np.array(traj), ms


def fixture(family, **kw):
    task, rng = make(family, **kw)
    return task, task.model.init_theta(rng), task.retained_batch(rng, NR)


def gate(tau2):
    return GateConfig(tau2=tau2, M=1e3, n2=N2, delta=0.1, T=20, R=R)


# ---- I2: no probes + no gate == AdamW bitwise ---------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", ["vau_nogate", "vau", "pre_proj", "post_euclid"])
def test_I2_no_probe_no_gate_equals_adamw(family, name):
    task, th0, R1 = fixture(family)
    bs = stream(task, 20)
    ref, _ = run(make_arm("adamw", task), task, th0, R1, bs)
    arm = make_arm(name, task, gate=gate(np.inf), k=0)
    tr, ms = run(arm, task, th0, R1, bs)
    assert ms.ops.calls["nonprogress_zeroed"] == 0       # the reject rule did not fire on this stream
    assert np.max(np.abs(tr - ref)) <= 1e-12 and np.array_equal(tr, ref)


# ---- I3: constant D -> metric projection == Euclidean projection --------------------------------
@pytest.mark.parametrize("c", [1.0, 3.7e-4, 2e3])
def test_I3_constant_D_equals_euclid(c):
    rng = np.random.default_rng(4)
    S, x = rng.normal(size=(5, 20)), rng.normal(size=20)
    a, _ = metric_project(x, S, np.full(20, c))
    b, _ = euclid_project(x, S)
    assert np.max(np.abs(a - b)) <= 1e-12 * max(1.0, np.max(np.abs(b)))


# ---- I4: exact J residual ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["vau_nogate", "post_euclid"])
def test_I4_exact_J_residual(name):
    task, th0, R1 = fixture("linear", aniso_decades=2.0)
    R1 = R1[:4]     # 16 exact rows <= rank 24 of the retained row space (ridge 0 needs full row rank)
    b = stream(task, 1)[0]
    out = {}
    for ridge in (0.0, 1e-6):
        arm = make_arm(name, task, probe="exact", K=1, ridge_rel=ridge)
        ms = arm.init_state(th0, R1, dev_rng(3))
        prop = arm.propose(ms, th0, b)
        d0, _, _ = aw.peek(ms.adam, prop.g, th0, arm.hp)
        J = ms.probe
        out[ridge] = np.linalg.norm(J @ prop.delta) / np.linalg.norm(J @ d0)
    assert out[0.0] <= 1e-8
    # NOTE: at the frozen ridge 1e-6*tr(G0)/k the residual is mu (G0+mu I)^-1 S Delta0, typically ~1e-6..1e-3
    # relative, so I4's 1e-8 is only attainable with ridge_rel = 0 (reported in README_E.md).
    assert out[1e-6] <= 1e-2


# ---- I5: lr = 0 -> every arm leaves theta unchanged -------------------------------------------------
@pytest.mark.parametrize("family", ["linear", "flow"])
@pytest.mark.parametrize("name", list(UNGATED) + ["vau", "gated_adamw", "earlystop_penalty"])
def test_I5_lr_zero(family, name):
    task, th0, R1 = fixture(family)
    arm = make_arm(name, task, hp=aw.AdamWHP(lr=0.0), gate=gate(1.0), k=4, K=2)
    tr, _ = run(arm, task, th0, R1, stream(task, 4))
    assert all(np.array_equal(t, th0) for t in tr)


# ---- I6: gate with tau = inf == ungated arm bitwise ------------------------------------------------
@pytest.mark.parametrize("family", ["mlp", "flow"])
@pytest.mark.parametrize("name", UNGATED)
def test_I6_infinite_budget_gate_is_identity(family, name):
    task, th0, R1 = fixture(family)
    bs = stream(task, 6)
    a, _ = run(make_arm(name, task, k=4, K=2), task, th0, R1, bs)
    gname = "vau" if name == "vau_nogate" else f"gated_{name}"
    b, ms = run(make_arm(gname, task, gate=gate(np.inf), k=4, K=2), task, th0, R1, bs)
    assert np.array_equal(a, b) and ms.ops.calls["rejected_step"] == 0


# ---- I8: identical update rules -> identical trajectories (common random numbers) -----------------
@pytest.mark.parametrize("name", UNGATED)
def test_I8_same_rule_same_trajectory(name):
    task, th0, R1 = fixture("mlp")
    bs = stream(task, 6)
    a, _ = run(make_arm(name, task, k=4, K=2), task, th0, R1, bs)
    b, _ = run(make_arm(name, task, k=4, K=2), task, th0, R1, bs)
    assert np.array_equal(a, b)


def test_sketch_stream_shared_across_k():
    """Sketch rows are drawn at k_max and truncated: S(k=4) equals the first 4 rows' directions of S(k=8)."""
    task, th0, R1 = fixture("mlp")
    S = {}
    for k in (4, 8):
        arm = make_arm("vau_nogate", task, k=k, K=1)
        ms = arm.init_state(th0, R1, dev_rng(3))
        arm._probe(ms, th0)
        S[k] = ms.probe
    assert np.allclose(S[4] * np.sqrt(4), S[8][:4] * np.sqrt(8), rtol=1e-12, atol=1e-14)


# ---- I10: disjointness helper ------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_I10_input_hashes_disjoint(family):
    task, rng = make(family)
    sets = [task.retained_batch(rng, 32) for _ in range(3)]
    hs = [set(input_hashes(s)) for s in sets]
    assert all(len(h) == 32 for h in hs)
    assert not (hs[0] & hs[1]) and not (hs[0] & hs[2]) and not (hs[1] & hs[2])


# ---- I11: validation seeds refused without a matching FREEZE.json ---------------------------------
def test_I11_seed_guard(tmp_path):
    assert seeds.check_seed(5) == 5
    for bad in (100, 6999, 7012, -1, 1.5, True):
        with pytest.raises(ContractViolation):
            seeds.check_seed(bad)
    missing = str(tmp_path / "FREEZE.json")
    with pytest.raises(ContractViolation):
        seeds.check_seed(7000, freeze_path=missing)
    fake = tmp_path / "FREEZE_bad.json"
    fake.write_text(json.dumps({"sha256_eval_protocol": "0", "sha256_candidate": "0",
                                "sha256_code_vau_tree": seeds.vau_tree_hash()}))
    with pytest.raises(ContractViolation):
        seeds.streams("G1", 7003, freeze_path=str(fake))
    a = seeds.streams("G1", 3)["sketch"].normal(size=4)
    b = seeds.streams("G1", 3)["sketch"].normal(size=4)
    c = seeds.streams("G2", 3)["sketch"].normal(size=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ---- I12: model c at LoRA = 0 reproduces base exactly ---------------------------------------------
def test_I12_flow_zero_lora_endpoint_equals_base():
    task, th0, _ = fixture("flow")
    rng = dev_rng(2)
    z0 = rng.normal(size=(6, task.model.n_img, task.model.c))
    pr = np.array([TARGET, ANCHOR, 2, 3, 4, 5])
    assert np.array_equal(sample(task.model, th0, z0, pr, task.sigmas), sample(task.model, None, z0, pr, task.sigmas))


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_v_equals_b_at_init(family):
    task, th0, _ = fixture(family)
    xs, _ = task.erasure_inputs(th0, task.erasure_batch(dev_rng(1)), Ops())
    assert np.array_equal(task.model.forward(xs, th0), task.model.forward(xs, None))


# ---- fixture knobs ----------------------------------------------------------------------------------
def test_anisotropy_column_scales():
    task, _, _ = fixture("linear", aniso_decades=3.0, P=200)
    lg = np.log10(task.model.col_scale)
    assert lg.min() >= -1.5 and lg.max() <= 1.5 and lg.max() - lg.min() > 2.8
    mlp, _, _ = fixture("mlp", aniso_decades=2.0)
    lf = np.log10(mlp.feat_scale)
    assert lf.min() >= -1.0 and lf.max() <= 1.0


@pytest.mark.parametrize("phi", [0.0, np.pi / 8, "haar"])
def test_structured_retained_jacobian(phi):
    task, _, _ = fixture("linear", kappa=1e2, phi=phi, P=40)
    n_sh = task.model.d_in - 2
    Jr = task.model.T[:n_sh].reshape(n_sh * task.model.out_dim, -1)
    m = Jr.shape[0]
    sv = np.linalg.svd(Jr, compute_uv=False)
    assert np.isclose(sv[0] / sv[-1], 1e2, rtol=1e-8)
    if phi == 0.0:
        assert np.all(Jr[:, m:] == 0.0)             # coordinate-aligned null space
    else:
        assert np.linalg.norm(Jr[:, m:]) > 1e-3 * np.linalg.norm(Jr)
    # retained inputs never touch the concept coordinates -> their Jacobian lies in row space of Jr
    rng = dev_rng(4)
    x = task.retained_batch(rng, 6, rare=False)
    assert np.all(x[:, -2:] == 0.0)
    J = task.model.jacobian(x, np.zeros(task.model.P))
    resid = J - J @ np.linalg.pinv(Jr) @ Jr
    assert np.linalg.norm(resid) <= 1e-10 * np.linalg.norm(J)


def test_rare_subgroup_and_shift():
    task, _, _ = fixture("mlp")
    e_t = task.emb[TARGET]
    e_r = task.emb[task.rare_id]
    assert np.isclose(e_r @ e_t / np.linalg.norm(e_r) / np.linalg.norm(e_t), 0.8, rtol=1e-12)
    x = task.retained_batch(dev_rng(5), 7, rare=True)
    assert np.allclose(x[:, task.z_dim:] / task.feat_scale[task.z_dim:], e_r)
    mix = task.retained_batch(dev_rng(5), 20000)
    frac = np.mean(np.all(np.isclose(mix[:, task.z_dim:], e_r * task.feat_scale[task.z_dim:]), axis=1))
    assert 0.002 < frac < 0.009                         # population mass 0.005
    a = task.retained_batch(dev_rng(6), 5, shift=1.0)
    b = task.retained_batch(dev_rng(6), 5, shift=0.0)
    assert np.allclose((a - b) / task.feat_scale, task.shift_dir) and np.isclose(np.linalg.norm(task.shift_dir), 1)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("c", [1e-2, 1e-6])
def test_contrast_scale_and_eps0(family, c):
    ref, th_r, _ = fixture(family)
    task, th0, _ = fixture(family, contrast_scale=c)
    def L(t, th):
        xs, xa = t.erasure_inputs(th, t.erasure_batch(dev_rng(1)), Ops())
        return np.linalg.norm(t.model.forward(xs, None) - t.model.forward(xa, None), axis=1)
    ratio = np.median(L(task, th0) / L(ref, th_r))
    if family == "linear":
        assert np.isclose(ratio, c, rtol=1e-6)            # exact for the linear base map
    else:
        assert c / 3 < ratio < 3 * c                      # nonlinear base: ||b-a|| ~ c (first order)
    task.eps_n = 0.0                                     # unshifted norm, zero subgradient at v == b
    loss, g = task.erasure_loss_grad(th0, task.erasure_batch(dev_rng(1)), Ops())
    assert np.isfinite(loss) and np.all(np.isfinite(g))


def test_eps0_zero_subgradient_at_v_equals_b():
    a = np.zeros((1, 4)); b = np.ones((1, 4))
    _, g = losses.gem_loss(b.copy(), a, b, 1, eps_n=0.0, gamma=1.0)
    assert np.allclose(g, b / 2.0)                       # only the attraction term; D- term is 0 at v == b


@pytest.mark.parametrize("family", ["mlp", "flow"])
def test_gauge_step_first_order_neutral_but_drifts(family):
    task, rng = make(family)
    th = task.model.init_theta(rng) + 0.3 * rng.normal(size=task.model.P)
    gdir = task.model.gauge_step(th, 0.3, dev_rng(8))
    x = task.retained_batch(dev_rng(9), 6)
    jv = task.model.jvp(x, th, gdir)
    assert np.linalg.norm(jv) <= 1e-12 * np.linalg.norm(task.model.jvp(x, th, rng.normal(size=task.model.P)))
    drift = np.linalg.norm(task.model.forward(x, th + gdir) - task.model.forward(x, th))
    assert drift > 1e-3


def test_inject_hook_and_gate_sees_it():
    task, th0, R1 = fixture("mlp")
    bs = stream(task, 2)
    hook = lambda ms, th, d: d + 0.5
    arm = make_arm("vau", task, gate=gate(np.inf), k=4, inject=hook)
    base = make_arm("vau", task, gate=gate(np.inf), k=4)
    a, _ = run(arm, task, th0, R1, bs[:1])
    b, _ = run(base, task, th0, R1, bs[:1])
    assert np.allclose(a[0] - b[0], 0.5)


def test_earlystop_and_build_once():
    task, th0, R1 = fixture("linear")
    arm = make_arm("earlystop_penalty", task, gate=gate(-1.0))
    tr, ms = run(arm, task, th0, R1, stream(task, 3))      # forward: 1 erasure + 1 penalty R1, then stopped
    assert ms.ops.calls["gate_try"] == 1 and ms.ops.calls["rejected_step"] == 1 and ms.calls == 3
    assert ms.ops.calls["forward"] == 2 and all(np.array_equal(t, th0) for t in tr)
    once = make_arm("vau_nogate", task, k=4, K=None)
    _, ms = run(once, task, th0, R1, stream(task, 5))
    assert ms.ops.calls["probe_build"] == 1


# ---- native torch file: static checks only (torch not installed) ------------------------------------
def test_native_files_static():
    import ast
    import os
    nat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native")
    src = open(os.path.join(nat, "vau_torch.py")).read()
    assert src.lstrip('"').startswith("STATIC, NOT EXECUTED")
    tree = ast.parse(src)                                  # syntax only; never imported
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"adamw_peek", "weighted_null_project", "VAUAdamW", "build_probe", "rho_hat", "step",
            "state_dict", "load_state_dict"} <= names
    patch = open(os.path.join(nat, "gem_integration.patch")).read()
    assert patch.startswith("diff --git a/gem/operators/erasure/erasure_gem.py b/gem/operators/erasure/erasure_gem.py")
    assert "+                optimizer.step()                    # native GEM (EG:288)" in patch


def test_native_arithmetic_mirror_by_transliteration():
    """Execute the two pure functions of vau_torch.py with a numpy shim standing in for torch tensors and
    compare against the numpy package (checks the transcription, not torch itself)."""
    import ast
    import math
    import os
    nat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native")
    tree = ast.parse(open(os.path.join(nat, "vau_torch.py")).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "adamw_peek"]
    ns = {"math": math}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "vau_torch_adamw_peek", "exec"), ns)

    class T(np.ndarray):                                   # .sqrt() like a torch tensor
        def sqrt(self):
            return np.sqrt(self)
    rng = np.random.default_rng(11)
    m, v, g, th = (rng.normal(size=9).view(T) for _ in range(4))
    v = np.abs(v).view(T)
    d_t, m_t, v_t, t_t, D_t = ns["adamw_peek"](m, v, 4, g, th, 1e-3, 0.9, 0.999, 1e-8, 0.01)
    d_n, st, D_n = aw.peek(aw.AdamState(np.asarray(m), np.asarray(v), 4), np.asarray(g), np.asarray(th), aw.AdamWHP())
    assert np.array_equal(np.asarray(d_t), d_n) and np.array_equal(np.asarray(D_t), D_n) and t_t == st.t
