"""Arms: LoRA zero-init, rollback bit-exactness, gate accept/reject/backtrack, op counters, contracts."""
import numpy as np
import pytest
from conftest import FAMILIES, make

from vau.contracts import dev_rng

from vau import adamw as aw
from vau.contracts import ContractViolation, Ops
from vau.methods import ARMS, Batch, GateConfig, Gated, make_arm

ALL_ARMS = ("adamw", "adamw_stepmatch", "penalty", "pre_proj", "post_euclid", "vau_nogate", "vau")
NR, K_SK, K_REF, N2, R = 8, 4, 2, 5, 3


def setup(family="linear", seed=0, **kw):
    task, rng = make(family, seed, **kw)
    th0 = task.model.init_theta(rng)
    R1 = task.retained_batch(rng, NR)
    return task, rng, th0, R1


def batch(task, rng, n_gate=R, same_gate=False, match_norm=None):
    if same_gate:
        g = task.retained_batch(rng, N2)
        gates = (g,) * n_gate
    else:
        gates = tuple(task.retained_batch(rng, N2) for _ in range(n_gate))
    return Batch(erase=task.erasure_batch(rng), gate=gates, match_norm=match_norm)


def gate_cfg(tau2):
    return GateConfig(tau2=tau2, M=1e3, n2=N2, delta=0.1, T=10, R=R)


def arm_of(name, task, tau2=1e12, **kw):
    cfg = dict(k=K_SK, K=K_REF)
    if name == "penalty":
        cfg["lam"] = 2.0
    cfg.update(kw)
    return make_arm(name, task, gate=gate_cfg(tau2), **cfg)


# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["mlp", "flow"])
def test_lora_zero_init_A_gradient_exactly_zero(family):
    task, rng, th0, _ = setup(family)
    _, g = task.erasure_loss_grad(th0, task.erasure_batch(rng), Ops())
    lay = task.model.lora
    for name, (sa, sb, _, _) in lay.slices.items():
        assert np.all(th0[sb] == 0.0)
        assert np.all(g[sa] == 0.0), name          # exactly zero, not merely small
    assert np.any(np.concatenate([g[sb] for (_, sb, _, _) in lay.slices.values()]) != 0.0)
    # step 1 AdamW: A moves only by decoupled weight decay
    d0, _, _ = aw.peek(aw.init_state(len(g)), g, th0, aw.AdamWHP())
    for sa, _, _, _ in lay.slices.values():
        assert np.allclose(d0[sa], -1e-3 * 1e-2 * th0[sa], rtol=1e-12, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("inner", ["vau_nogate", "adamw", "penalty", "pre_proj"])
def test_rollback_is_bit_exact(family, inner):
    task, rng, th0, R1 = setup(family)
    name = "vau" if inner == "vau_nogate" else f"gated_{inner}"
    arm = arm_of(name, task, tau2=1e12)
    ms = arm.init_state(th0, R1, dev_rng(0))
    th, _ = arm.step(ms, th0, batch(task, rng))             # accepted step -> nontrivial state
    m_b, v_b, t_b, th_b = ms.adam.m.copy(), ms.adam.v.copy(), ms.adam.t, th.copy()
    adam_obj = ms.adam
    arm.gate = gate_cfg(-1.0)                                # impossible budget -> must reject
    th2, info = arm.step(ms, th, batch(task, rng))
    assert not info["accepted"] and info["s"] == 0.0 and len(info["rho_hat"]) == R
    assert np.array_equal(th2, th_b) and th2 is not th
    assert ms.adam is adam_obj and ms.adam.t == t_b
    assert np.array_equal(ms.adam.m, m_b) and np.array_equal(ms.adam.v, v_b)
    assert ms.ops.calls["rejected_step"] == 1 and ms.calls == 2


def test_gate_accepts_within_budget_and_commits():
    task, rng, th0, R1 = setup("mlp")
    arm = arm_of("vau", task, tau2=1e12)
    ms = arm.init_state(th0, R1, dev_rng(0))
    th1, info = arm.step(ms, th0, batch(task, rng))
    assert info["accepted"] and info["s"] == 1.0 and ms.adam.t == 1 and not np.array_equal(th1, th0)


def test_gate_backtracks_to_half_on_linear_model():
    """Linear model, same gate batch, no clipping: rho(s) = s^2 rho(1) exactly."""
    task, rng, th0, R1 = setup("linear")
    probe_arm = make_arm("adamw", task)
    ms0 = probe_arm.init_state(th0, R1, dev_rng(0))
    b = batch(task, rng, same_gate=True)
    prop = probe_arm.propose(ms0, th0, b)
    g = Gated(make_arm("adamw", task), gate_cfg(1.0))
    f = lambda th: np.mean(np.sum((task.model.forward(b.gate[0], th) - task.model.forward(b.gate[0], th0)) ** 2, 1))
    rho1 = f(th0 + prop.delta)
    assert np.isclose(f(th0 + 0.5 * prop.delta), 0.25 * rho1, rtol=1e-10)
    cfg = gate_cfg(1.0)
    tau2 = cfg.slack + 0.5 * rho1                     # fails at s=1, passes at s=1/2
    g.gate = GateConfig(tau2=tau2, M=cfg.M, n2=N2, delta=cfg.delta, T=cfg.T, R=R)
    ms = g.init_state(th0, R1, dev_rng(0))
    th1, info = g.step(ms, th0, b)
    assert info["s"] == 0.5 and info["accepted"]
    assert np.allclose(th1, th0 + 0.5 * prop.delta, rtol=0, atol=1e-15)
    assert ms.adam.t == 1


def _expected(name, family, steps, accepted=True):
    e = {"forward": [steps, 11 * steps], "teacher": [2 * steps, 22 * steps], "vjp": [steps, 11 * steps]}
    ev = {"step": steps, "solve": 0, "probe_build": 0, "gate_try": 0, "rejected_step": 0}
    inner = {"vau": "vau_nogate"}.get(name, name.replace("gated_", ""))
    if family == "flow":
        e["rollout_forward"] = [11 * steps, 11 * steps]
    if inner in ("pre_proj", "post_euclid", "vau_nogate"):
        builds = len([c for c in range(steps) if c % K_REF == 0])
        e["forward"][0] += builds; e["forward"][1] += builds * NR
        e["vjp"][0] += builds * K_SK; e["vjp"][1] += builds * K_SK * NR
        ev["solve"], ev["probe_build"] = steps, builds
    if inner == "penalty":
        e["teacher"][0] += 1; e["teacher"][1] += NR
        e["forward"][0] += steps; e["forward"][1] += steps * NR
        e["vjp"][0] += steps; e["vjp"][1] += steps * NR
    if name == "vau" or name.startswith("gated_"):
        tries = steps if accepted else steps * R
        e["gate_forward"] = [tries, tries * N2]; e["gate_base"] = [tries, tries * N2]
        ev["gate_try"] = tries
        ev["rejected_step"] = 0 if accepted else steps
    return e, ev


@pytest.mark.parametrize("family", ["linear", "flow"])
@pytest.mark.parametrize("name", list(ALL_ARMS) + ["gated_adamw", "gated_post_euclid"])
def test_operation_counters(family, name):
    steps = 3
    task, rng, th0, R1 = setup(family)
    arm = arm_of(name, task, ref_norms=[1e-3] * steps) if name == "adamw_stepmatch" else arm_of(name, task)
    ms = arm.init_state(th0, R1, dev_rng(0))
    th = th0
    infos = []
    for _ in range(steps):
        th, info = arm.step(ms, th, batch(task, rng))
        infos.append(info)
    e, ev = _expected(name, family, steps)
    for op in ("forward", "teacher", "vjp", "jvp", "rollout_forward", "gate_forward", "gate_base"):
        exp = e.get(op, [0, 0])
        assert [ms.ops.calls[op], ms.ops.samples[op]] == exp, (op, ms.ops.snapshot())
    for op, val in ev.items():
        assert ms.ops.calls[op] == val, (op, ms.ops.snapshot())
    assert ms.ops.calls["nonprogress_zeroed"] == sum(i.get("nonprogress_zeroed", False) for i in infos)


def test_counters_when_rejecting():
    task, rng, th0, R1 = setup("linear")
    arm = arm_of("vau", task, tau2=-1.0)
    ms = arm.init_state(th0, R1, dev_rng(0))
    th = th0
    for _ in range(2):
        th, _ = arm.step(ms, th, batch(task, rng))
    e, ev = _expected("vau", "linear", 2, accepted=False)
    assert [ms.ops.calls["gate_forward"], ms.ops.samples["gate_forward"]] == e["gate_forward"]
    assert ms.ops.calls["rejected_step"] == 2 and ms.adam.t == 0 and np.array_equal(th, th0)


def test_stepmatch_norm():
    task, rng, th0, R1 = setup("mlp")
    arm = arm_of("adamw_stepmatch", task, metric="euclid")
    ms = arm.init_state(th0, R1, dev_rng(0))
    th1, info = arm.step(ms, th0, batch(task, rng, match_norm=2.5e-4))
    assert np.isclose(np.linalg.norm(th1 - th0), 2.5e-4, rtol=1e-12)
    armH = arm_of("adamw_stepmatch", task)                             # protocol default: H-norm
    msH = armH.init_state(th0, R1, dev_rng(0))
    _, infoH = armH.step(msH, th0, batch(task, rng, match_norm=0.7))
    assert np.isclose(infoH["step_Hnorm"], 0.7, rtol=1e-12)
    with pytest.raises(ContractViolation):
        arm.step(ms, th1, batch(task, rng))                 # no target available


def test_pre_proj_state_uses_projected_gradient():
    task, rng, th0, R1 = setup("linear")
    arm = arm_of("pre_proj", task)
    ms = arm.init_state(th0, R1, dev_rng(0))
    b = batch(task, rng)
    arm.step(ms, th0, b)
    _, g = task.erasure_loss_grad(th0, b.erase, Ops())
    S = ms.probe
    assert np.linalg.norm(S @ (ms.adam.m / 0.1)) <= 1e-4 * np.linalg.norm(S @ g)


def test_vau_nogate_exact_probe_zero_first_order_drift_linear():
    """Linear model + exact J (K=1): retained drift on R1 vanishes to ridge level; AdamW's does not."""
    task, rng, th0, R1 = setup("linear")
    base = lambda th: task.model.forward(R1, th) - task.model.forward(R1, th0)
    arms = {n: arm_of(n, task, probe="exact", K=1) for n in ("vau_nogate", "post_euclid", "adamw")}
    drift = {}
    for n, arm in arms.items():
        ms = arm.init_state(th0, R1, dev_rng(0))
        r = np.random.default_rng(5)
        th = th0
        for _ in range(4):
            th, _ = arm.step(ms, th, batch(task, r))
        drift[n] = np.linalg.norm(base(th))
    assert drift["vau_nogate"] <= 1e-4 * drift["adamw"]
    assert drift["post_euclid"] <= 1e-4 * drift["adamw"]


def test_invalid_inputs_rejected():
    task, rng, th0, R1 = setup("flow")
    arm = arm_of("vau", task)
    ms = arm.init_state(th0, R1, dev_rng(0))
    b = batch(task, rng)
    with pytest.raises(ContractViolation):
        arm.step(ms, th0[:-1], b)
    bad = th0.copy(); bad[3] = np.nan
    with pytest.raises(ContractViolation):
        arm.step(ms, bad, b)
    with pytest.raises(ContractViolation):
        eb = b.erase.copy(); eb[0, 0] = np.inf
        arm.step(ms, th0, Batch(erase=eb, gate=b.gate))
    with pytest.raises(ContractViolation):
        arm.step(ms, th0, Batch(erase=b.erase, gate=b.gate[:1]))       # too few gate batches
    with pytest.raises(ContractViolation):
        arm.step(ms, th0, Batch(erase=b.erase, gate=(task.retained_batch(rng, N2 + 1),) * R))
    with pytest.raises(ContractViolation):
        arm.init_state(th0, R1, 0)                                     # sketch rng must be a Generator
    with pytest.raises(ContractViolation):
        make_arm("vau", task)                                          # gate config required
    with pytest.raises(ContractViolation):
        make_arm("nope", task)
    with pytest.raises(ContractViolation):
        GateConfig(tau2=1.0, M=0.0, n2=5, delta=0.1, T=10)
    assert ms.adam.t == 0 and ms.calls == 0                            # nothing was committed


def test_arm_registry_complete():
    assert set(ARMS) | {"vau"} == set(ALL_ARMS)
    task, _, _, _ = setup("linear")
    assert make_arm("earlystop_penalty", task, gate=gate_cfg(1.0)).name == "earlystop_penalty"
    assert make_arm("gated_vau_nogate", task, gate=gate_cfg(1.0)).name == "vau"   # identical rule
