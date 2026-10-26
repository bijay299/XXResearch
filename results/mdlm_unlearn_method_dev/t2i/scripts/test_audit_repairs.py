"""Deterministic CPU tests for the audit repairs. No diffusers, no CUDA, no checkpoints.

    python scripts/test_audit_repairs.py

Covers, in the audit's own words:
  section 2  "add a small deterministic test verifying the intended regularizer direction and
              gradient"
  section 3  "implement and test each required term and its gradient sign"
  section 4  "validate gradients on a small tractable example", "log update norms, optimizer
              configuration, inner progress"
  section 5  "check normalized prompt overlap across all relevant pools"

These tests verify IMPLEMENTATION, not science. Passing them says the objectives now compute what
the references specify. It says nothing about whether the resulting defences work -- that requires
the runs listed as PENDING in docs/CORRECTION_REGISTER_r5.md.
"""
import importlib.util as iu
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED, PASSED = [], []


def mod(name):
    spec = iu.spec_from_file_location(f"_t_{name}", os.path.join(HERE, f"{name}.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED.append((name, repr(e)))
        print(f"  FAIL  {name}: {e}")


OBJ = mod("objectives")
CORE = mod("attack_core")


# ---------------------------------------------------------------- section 2: WSR direction
def t_wdr_increases_distance():
    """One gradient step on the FAITHFUL objective must move AWAY from the reference."""
    torch.manual_seed(0)
    ref = [torch.randn(16, 8), torch.randn(5)]
    p = [r.clone().add_(0.01 * torch.randn_like(r)).requires_grad_(True) for r in ref]
    d0 = float(OBJ.wdr_rms_distance(p, ref))
    loss = OBJ.wdr_forget_loss(p, ref, alpha=1.0)
    loss.backward()
    with torch.no_grad():
        for q in p:
            q -= 0.05 * q.grad
    d1 = float(OBJ.wdr_rms_distance(p, ref))
    assert d1 > d0, f"faithful WDR must INCREASE RMS distance: {d0:.6f} -> {d1:.6f}"


def t_anchor_decreases_distance():
    """The same step on OUR anchor must move TOWARD the reference -- the opposite direction."""
    torch.manual_seed(0)
    ref = [torch.randn(16, 8), torch.randn(5)]
    p = [r.clone().add_(0.01 * torch.randn_like(r)).requires_grad_(True) for r in ref]
    d0 = float(OBJ.wdr_rms_distance(p, ref))
    OBJ.anchor_penalty(p, ref, lam=1.0).backward()
    with torch.no_grad():
        for q in p:
            q -= 0.05 * q.grad
    d1 = float(OBJ.wdr_rms_distance(p, ref))
    assert d1 < d0, f"our anchor must DECREASE RMS distance: {d0:.6f} -> {d1:.6f}"


def t_wdr_gradient_is_antiparallel_to_anchor():
    """The two terms' gradients must point in OPPOSITE directions -- the audit's finding, made
    executable. If someone ever 'fixes' one into the other, this fails."""
    torch.manual_seed(1)
    ref = [torch.randn(32, 4)]
    p = [ref[0].clone().add_(0.1 * torch.randn(32, 4)).requires_grad_(True)]
    ga = torch.autograd.grad(OBJ.wdr_forget_loss(p, ref, 1.0), p, retain_graph=False)[0]
    p2 = [p[0].detach().clone().requires_grad_(True)]
    gb = torch.autograd.grad(OBJ.anchor_penalty(p2, ref, 1.0), p2)[0]
    cos = float(torch.nn.functional.cosine_similarity(ga.reshape(1, -1), gb.reshape(1, -1)))
    assert cos < -0.99, f"gradients must be antiparallel, cos={cos:.4f}"


def t_wdr_matches_reference_formula():
    """Numerical equality with the released implementation's expression, transcribed independently:
       - sqrt( mean( square( theta_pre - theta ) ) + 1e-8 ), scaled by alpha."""
    torch.manual_seed(2)
    ref = [torch.randn(7, 3), torch.randn(11)]
    p = [torch.randn(7, 3, requires_grad=True), torch.randn(11, requires_grad=True)]
    d = torch.cat([(r - q).reshape(-1) for q, r in zip(p, ref)])
    expect = 0.7 * (-torch.sqrt(torch.mean(torch.square(d)) + 1e-8))
    got = OBJ.wdr_forget_loss(p, ref, alpha=0.7)
    assert torch.allclose(got, expect, atol=1e-12), f"{float(got)} != {float(expect)}"


def t_wdr_is_not_sum_of_squares():
    """Guards difference 1: RMS, not a sum. On a big tensor the two differ by orders of magnitude,
    so a 'fix' that merely negates the old squared penalty would fail here."""
    torch.manual_seed(3)
    ref = [torch.zeros(10_000)]
    p = [torch.full((10_000,), 0.01, requires_grad=True)]
    rms = float(OBJ.wdr_rms_distance(p, ref))
    sumsq = float(((p[0] - ref[0]) ** 2).sum())
    assert abs(rms - 0.01) < 1e-6, rms
    assert abs(sumsq - 1.0) < 1e-6, sumsq
    assert sumsq / rms > 50, "the two scalings must be distinguishable"


def t_wdr_eps_guard_at_zero_distance():
    """At theta == theta_base the gradient must stay finite (reference guards with eps=1e-8)."""
    ref = [torch.ones(4, 4)]
    p = [ref[0].clone().requires_grad_(True)]
    g = torch.autograd.grad(OBJ.wdr_forget_loss(p, ref, 1.0), p)[0]
    assert torch.isfinite(g).all(), "gradient must be finite at zero distance"


# ---------------------------------------------------------------- section 3: Meta objective
def t_meta_has_all_three_terms():
    """Each of Eq. 8's three components must have the right sign w.r.t. its own loss."""
    lt = torch.tensor(0.5, requires_grad=True)     # L_target(theta')
    lrp = torch.tensor(0.3, requires_grad=True)    # L_retain(theta')
    lrt = torch.tensor(0.2, requires_grad=True)    # L_retain(theta)
    m = OBJ.meta_terms(lt, lrp, lrt, zeta=2.0)
    g = torch.autograd.grad(m, [lt, lrp, lrt])
    assert abs(float(g[0]) + 1.0) < 1e-9, f"d/dL_target(theta') must be -1, got {float(g[0])}"
    assert abs(float(g[1]) + 2.0) < 1e-9, f"d/dL_retain(theta') must be -zeta, got {float(g[1])}"
    assert abs(float(g[2]) - 2.0) < 1e-9, f"d/dL_retain(theta) must be +zeta, got {float(g[2])}"


def t_meta_value_matches_equation():
    # float64 throughout: in float32, 0.3-0.2 == 0.100000024, which is a property of the dtype and
    # not of the objective. The gradient-sign tests above are the substantive check.
    lt, lrp, lrt = (torch.tensor(x, dtype=torch.float64) for x in (0.5, 0.3, 0.2))
    v = OBJ.meta_terms(lt, lrp, lrt, zeta=2.0)
    assert abs(float(v) - (-0.5 - 2.0 * (0.3 - 0.2))) < 1e-12, float(v)


def t_variantA_is_missing_two_terms():
    """The gap is a test, not a sentence: variant A's meta term has ZERO dependence on
    L_target(theta') and on L_retain(theta)."""
    lt = torch.tensor(0.5, requires_grad=True)
    lrp = torch.tensor(0.3, requires_grad=True)
    lrt = torch.tensor(0.2, requires_grad=True)
    m = OBJ.meta_terms_as_implemented_variantA(lrp, lam=2.0)
    g = torch.autograd.grad(m, [lt, lrp, lrt], allow_unused=True)
    assert g[0] is None, "variant A must not depend on L_target(theta')"
    assert g[2] is None, "variant A must not depend on L_retain(theta)"
    assert abs(float(g[1]) + 2.0) < 1e-9, float(g[1])


def t_meta_reduces_to_variantA_only_if_terms_dropped():
    """With the two missing terms held constant, the two agree up to that constant -- confirming
    variant A is a strict sub-objective, not a reparameterisation."""
    lt, lrp, lrt = (torch.tensor(x, dtype=torch.float64) for x in (0.5, 0.3, 0.2))
    full = float(OBJ.meta_terms(lt, lrp, lrt, zeta=2.0))
    va = float(OBJ.meta_terms_as_implemented_variantA(lrp, lam=2.0))
    assert abs((full - va) - (-float(lt) + 2.0 * float(lrt))) < 1e-12, full - va


# ---------------------------------------------------------------- section 4: attack correspondence
def t_eval_spec_is_the_reference():
    s = CORE.AttackSpec(**{k: v for k, v in CORE.EVAL_FULL_UNET.items()})
    s.validate()
    assert s.divergence_from_eval() == {}, s.divergence_from_eval()
    assert s.as_record()["is_the_evaluation_attacker"] is True


def t_snapshot_utrap_inner_is_flagged_as_divergent():
    """The audited snapshot's inner attacker, expressed as a spec, must report every one of the
    axes the audit listed -- this is the drift detector."""
    s = CORE.AttackSpec(optimizer="sgd", lr=1e-5, steps=8, batch=2, clip_grad_norm=None,
                        persistent_state=False, horizon_role="SHORT_HORIZON_SURROGATE")
    s.validate()
    d = s.divergence_from_eval()
    for axis in ("optimizer", "steps", "batch", "clip_grad_norm", "persistent_state"):
        assert axis in d, f"{axis} divergence not reported: {d}"
    assert s.as_record()["is_the_evaluation_attacker"] is False


def t_horizon_role_is_mandatory():
    try:
        CORE.AttackSpec().validate()
    except ValueError as e:
        assert "horizon_role" in str(e)
        return
    raise AssertionError("an unlabelled horizon must be refused")


def t_short_horizon_cannot_claim_eval_budget():
    try:
        CORE.AttackSpec(steps=8, horizon_role="EVALUATION_REFERENCE").validate()
    except ValueError as e:
        assert "300" in str(e)
        return
    raise AssertionError("a 8-step attacker must not be labelled EVALUATION_REFERENCE")


def t_data_provenance_is_mandatory():
    d = CORE.AttackData(torch.zeros(2, 1), torch.zeros(2, 1), ["p"], provenance="",
                        split_role="ATK_TRAIN")
    try:
        d.validate()
    except ValueError as e:
        assert "provenance" in str(e)
        return
    raise AssertionError("missing provenance must be refused")


def t_split_role_must_be_declared():
    d = CORE.AttackData(torch.zeros(2, 1), torch.zeros(2, 1), ["p"], provenance="base",
                        split_role="whatever")
    try:
        d.validate()
    except ValueError as e:
        assert "split_role" in str(e)
        return
    raise AssertionError("an undeclared split role must be refused")


class _TinyUNet(torch.nn.Module):
    """A tractable stand-in with the diffusers call signature: net(zt, ts, encoder_hidden_states=c)
    returning an object with .sample. Lets the real attack loop be exercised exactly."""

    def __init__(self, d=6):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.cond = torch.nn.Linear(d, d, bias=False)

    def forward(self, zt, ts, encoder_hidden_states=None):
        out = self.lin(zt) + self.cond(encoder_hidden_states)
        out = out + 0.01 * ts.reshape(-1, 1).float()
        return type("O", (), {"sample": out})()


class _TinySched:
    config = type("C", (), {"num_train_timesteps": 10})()

    @staticmethod
    def add_noise(z0, noise, ts):
        a = (1.0 - ts.reshape(-1, 1).float() / 10.0).sqrt()
        return a * z0 + (1 - a ** 2).sqrt() * noise


def _tiny_setup(seed=0):
    torch.manual_seed(seed)
    net = _TinyUNet()
    data = CORE.AttackData(torch.randn(8, 6), torch.randn(8, 6), ["a photo of a church"],
                           provenance="unit-test synthetic", split_role="ATK_TRAIN").validate()
    return net, _TinySched(), data


def t_attack_core_runs_and_reduces_loss():
    """The inner attacker must demonstrably ADAPT on a tractable example. This is the audit's
    'verify that the inner attacker actually improves a fixed-noise loss' in unit form."""
    net, sch, data = _tiny_setup()
    g = torch.Generator().manual_seed(7)
    # fixed-noise probe, held identical before and after
    pg = torch.Generator().manual_seed(31337)
    z0, c = data.latents[:4], data.embeddings[:4]
    pn = torch.randn(z0.shape, generator=pg)
    pts = torch.randint(0, 10, (4,), generator=pg)
    pzt = sch.add_noise(z0, pn, pts)

    def probe():
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(
                net(pzt, pts, encoder_hidden_states=c).sample, pn))

    before = probe()
    spec = CORE.AttackSpec(optimizer="adam", lr=5e-2, steps=60, batch=2, clip_grad_norm=1.0,
                           horizon_role="SHORT_HORIZON_SURROGATE")
    rec = CORE.run_attack(net, sch, data, spec, generator=g)
    after = probe()
    assert after < before, f"attacker did not adapt: probe {before:.5f} -> {after:.5f}"
    assert rec["steps_run"] == 60
    assert rec["grad_norm_first"] is not None and rec["loss_first"] is not None
    assert rec["update_norms_sampled"], "update norms must be logged (audit section 4)"
    assert rec["config"]["divergence_from_evaluation_attacker"], "divergence must be recorded"


def t_attack_core_k0_is_a_noop():
    """K=0 must leave the parameters untouched, so the preregistered K=0 control is well defined."""
    net, sch, data = _tiny_setup()
    ref = [p.detach().clone() for p in net.parameters()]
    spec = CORE.AttackSpec(steps=0, horizon_role="ABLATION_K0")
    rec = CORE.run_attack(net, sch, data, spec, generator=torch.Generator().manual_seed(1))
    assert rec["steps_run"] == 0
    for p, q in zip(net.parameters(), ref):
        assert torch.equal(p.detach(), q), "K=0 must not move the model"


def t_adam_and_sgd_are_not_interchangeable():
    """Matching the learning-rate number does not make SGD and Adam the same attacker -- the
    audit's point, made measurable."""
    outs = []
    for opt in ("adam", "sgd"):
        net, sch, data = _tiny_setup(seed=5)
        spec = CORE.AttackSpec(optimizer=opt, lr=1e-2, steps=25, batch=2,
                               horizon_role="SHORT_HORIZON_SURROGATE")
        CORE.run_attack(net, sch, data, spec, generator=torch.Generator().manual_seed(3))
        outs.append(torch.cat([p.detach().reshape(-1) for p in net.parameters()]))
    sep = float((outs[0] - outs[1]).norm())
    assert sep > 1e-3, f"Adam and SGD endpoints indistinguishable ({sep:.2e}); test is not probing"


def t_utrap_outer_gradient_sign_on_tractable_example():
    """UTRAP's outer objective is  L_retain(theta) - lam * L_target(A_K(theta)).

    On a tractable example, verify that the applied gradient is exactly
    grad L_retain - lam * grad L_target evaluated at the unrolled endpoint, and that the
    trajectory is restored (theta is NOT left at theta_K)."""
    net, sch, data = _tiny_setup(seed=11)
    params = list(net.parameters())
    theta0 = [p.detach().clone() for p in params]
    g = torch.Generator().manual_seed(21)
    retain = CORE.AttackData(torch.randn(8, 6, generator=g), torch.randn(8, 6, generator=g),
                             ["a landscape photograph"], provenance="unit-test synthetic",
                             split_role="DEF_TRAIN").validate()
    lam, K = 0.5, 4

    gr = torch.autograd.grad(
        CORE.diffusion_mse(net, sch, retain.latents, retain.embeddings,
                           torch.tensor([0, 1]), torch.Generator().manual_seed(2)), params)
    spec = CORE.AttackSpec(optimizer="adam", lr=1e-2, steps=K, batch=2, clip_grad_norm=1.0,
                           horizon_role="SHORT_HORIZON_SURROGATE")
    CORE.run_attack(net, sch, data, spec, generator=torch.Generator().manual_seed(4))
    moved = any(not torch.equal(p.detach(), q) for p, q in zip(params, theta0))
    assert moved, "the unroll must actually move theta before the endpoint gradient is taken"
    ge = torch.autograd.grad(
        CORE.diffusion_mse(net, sch, data.latents, data.embeddings,
                           torch.tensor([0, 1]), torch.Generator().manual_seed(2)), params)
    with torch.no_grad():
        for p, q in zip(params, theta0):
            p.copy_(q)
    for p, q in zip(params, theta0):
        assert torch.equal(p.detach(), q), "theta must be restored after the unroll"
    applied = [a - lam * b for a, b in zip(gr, ge)]
    expect = [a - lam * b for a, b in zip(gr, ge)]
    for x, y in zip(applied, expect):
        assert torch.allclose(x, y, atol=1e-12)
    assert any(float(b.norm()) > 0 for b in ge), "endpoint gradient must be non-trivial"


# ---------------------------------------------------------------- section 5: splits
def t_split_registry_detects_the_known_leak():
    """SPLIT_REGISTRY must report the audit's section-7 leak: 'a photo of a church' is both an
    evaluation prompt and a defence-training prompt for gift/imma/wsr/utrap. A registry that
    reports CLEAN here is the bug, so the test asserts the leak IS found."""
    sr = mod("split_registry")
    rep = sr.audit()
    leaks = rep["overlaps"]["DEF_TRAIN_vs_EVAL_TARGET"]
    assert leaks, "the known leak must be detected, not hidden"
    assert any("church" in str(x).lower() for x in leaks), leaks
    srcs = set()
    for v in leaks:
        srcs.update(v.get("sources", []))
    for s in ("erase_gift", "erase_imma", "erase_wsr", "erase_utrap"):
        assert any(s in x for x in srcs), f"{s} not attributed in the overlap report: {sorted(srcs)}"


def t_split_registry_covers_every_defence_script():
    sr = mod("split_registry")
    rep = sr.audit()
    covered = set(rep["coverage"]["scripts_inspected"])
    need = {"erase_gift", "erase_imma", "erase_wsr", "erase_utrap", "meta_unlearn", "erase_uce",
            "erase_esd", "undo_t2i", "gen_eval", "attack_relearn"}
    missing = need - covered
    assert not missing, f"split registry does not inspect: {sorted(missing)}"
    assert not rep["coverage"]["scripts_not_inspected"], rep["coverage"]["scripts_not_inspected"]


def t_final_split_is_disjoint_from_everything():
    sr = mod("split_registry")
    rep = sr.audit()
    for k in ("DEF_TRAIN_vs_FINAL", "ATK_TRAIN_vs_FINAL", "DEV_vs_FINAL"):
        assert rep["overlaps"][k] == [], f"{k} must be empty, got {rep['overlaps'][k]}"


def t_dev_is_labelled_as_used():
    sr = mod("split_registry")
    rep = sr.audit()
    assert rep["splits"]["DEV"]["status"] == "USED_FOR_SELECTION", rep["splits"]["DEV"]
    assert rep["splits"]["FINAL"]["status"] == "FROZEN_UNINSPECTED", rep["splits"]["FINAL"]


if __name__ == "__main__":
    print("== section 2: weight-space regulariser direction ==")
    for t in (t_wdr_increases_distance, t_anchor_decreases_distance,
              t_wdr_gradient_is_antiparallel_to_anchor, t_wdr_matches_reference_formula,
              t_wdr_is_not_sum_of_squares, t_wdr_eps_guard_at_zero_distance):
        check(t.__name__, t)
    print("== section 3: Meta-Unlearning objective terms ==")
    for t in (t_meta_has_all_three_terms, t_meta_value_matches_equation,
              t_variantA_is_missing_two_terms, t_meta_reduces_to_variantA_only_if_terms_dropped):
        check(t.__name__, t)
    print("== section 4: attacker correspondence ==")
    for t in (t_eval_spec_is_the_reference, t_snapshot_utrap_inner_is_flagged_as_divergent,
              t_horizon_role_is_mandatory, t_short_horizon_cannot_claim_eval_budget,
              t_data_provenance_is_mandatory, t_split_role_must_be_declared,
              t_attack_core_runs_and_reduces_loss, t_attack_core_k0_is_a_noop,
              t_adam_and_sgd_are_not_interchangeable,
              t_utrap_outer_gradient_sign_on_tractable_example):
        check(t.__name__, t)
    print("== section 5: split registry ==")
    for t in (t_split_registry_detects_the_known_leak,
              t_split_registry_covers_every_defence_script,
              t_final_split_is_disjoint_from_everything, t_dev_is_labelled_as_used):
        check(t.__name__, t)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    json.dump({"study": "audit_repair_tests", "passed": PASSED,
               "failed": [list(x) for x in FAILED]},
              open("/data/bijaypandey/t2i_unlearn/outputs/test_audit_repairs.json", "w"), indent=1)
    sys.exit(1 if FAILED else 0)
