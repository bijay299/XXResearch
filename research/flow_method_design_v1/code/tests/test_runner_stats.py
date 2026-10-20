"""Runner statistics and matched-efficacy interpolation on synthetic inputs with known answers."""
import itertools
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_validation as rv  # noqa: E402


def test_crossing_and_interpolation():
    ts, E = [0, 1, 2, 3], [0.0, 0.4, 1.0, 1.0]
    i, w = rv.crossing(ts, E, 0.7)
    assert i == 2 and math.isclose(w, 0.5)
    assert rv.interp_at([0.0, 2.0, 4.0, 9.0], (2, 0.5)) == 3.0
    assert rv.crossing(ts, [0.9, 1, 1, 1], 0.8) == (0, 1.0)
    assert rv.crossing(ts, [0.0, 0.1, 0.2, 0.3], 0.8) is None
    assert rv.crossing(ts, [0.0, math.nan, 1.0, 1.0], 0.8) is None          # failed run: search stops
    assert rv.interp_at([1.0, math.inf], (1, 0.5)) == math.inf


def _run(E, rho, status="ok"):
    t = list(range(len(E)))
    return dict(status=status, traj=dict(t=t, E_sel=E, rho_sel=rho, rho_test=[2 * r for r in rho],
                                         E_test=E, FE=[10.0 * i for i in t]))


def test_matched_and_select_config():
    r1 = _run([0.0, 0.5, 1.0], [0.0, 1.0, 3.0])        # crosses 0.8 at w = 0.6 -> rho_sel 2.2
    r2 = _run([0.0, 0.9, 1.0], [0.0, 0.5, 0.6])        # crosses 0.8 at w = 8/9 -> rho_sel 4/9
    r3 = _run([0.0, 0.1, 0.2], [0.0, 0.0, 0.0])        # never
    m = rv.matched(r1, 0.8)
    assert math.isclose(m["rho_sel"], 2.2) and math.isclose(m["rho_test"], 4.4) and math.isclose(m["FE"], 16.0)
    assert math.isclose(m["t_cross"], 1.6)
    best = rv.select_config([r1, r2, r3], 0.8)
    assert best[1] is r2 and math.isclose(best[0]["rho_sel"], 0.5 * 8 / 9)
    assert rv.select_config([r3], 0.8) is None


def test_log_ratio_rules():
    assert rv.log_ratio(None, None) is None
    assert rv.log_ratio(None, 1.0) == math.log(100)
    assert rv.log_ratio(1.0, None) == -math.log(100)
    assert math.isclose(rv.log_ratio(0.5, 1.0), math.log(0.5))
    assert math.isclose(rv.log_ratio(0.0, 1e-12), 0.0)                  # floor 1e-12


def test_permutation_test_exact():
    x = -np.arange(1, 13, dtype=float)
    assert rv.perm_test_one_sided(x) == 1 / 4096
    assert rv.perm_test_one_sided(-x) == 1.0
    y = [-1.0, -2.0, 3.5]
    obs = np.mean(y)
    brute = np.mean([np.mean(np.array(s) * y) <= obs + 1e-15 for s in itertools.product([-1, 1], repeat=3)])
    assert rv.perm_test_one_sided(y) == brute


def test_bootstrap_ci():
    lo, hi = rv.bootstrap_gmr_ci([math.log(0.8)] * 12)
    assert math.isclose(lo, 0.8) and math.isclose(hi, 0.8)
    x = np.random.default_rng(0).normal(size=12)
    assert rv.bootstrap_gmr_ci(x) == rv.bootstrap_gmr_ci(x)            # seed 424242 -> reproducible
    lo, hi = rv.bootstrap_gmr_ci(x)
    assert lo < math.exp(np.mean(x)) < hi


def test_holm_and_binomial():
    adj = rv.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)
    assert math.isclose(rv.binom_sf(1, 1, 0.05), 0.05)
    assert rv.binom_sf(0, 10, 0.3) == pytest.approx(1.0)
    assert math.isclose(rv.binom_sf(2, 3, 0.5), 0.5)


def test_fallback_and_family_stats():
    n = 12
    fail = [(False, False)] * 6 + [(True, True)] * 6
    ok = [(True, True)] * 12
    assert rv.fallback_level({0.8: fail, 0.5: ok, 0.2: ok}, n) == 0.5
    assert rv.fallback_level({0.8: fail, 0.5: fail, 0.2: ok}, n) == 0.2
    assert rv.fallback_level({0.8: ok, 0.5: ok, 0.2: ok}, n) == 0.8
    st = rv.family_stats({s: math.log(0.5) for s in range(12)}, 12, 9)
    assert math.isclose(st["GMR"], 0.5) and st["wins"] == 12 and st["p"] == 1 / 4096


def test_spearman_and_fe():
    assert math.isclose(rv.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(rv.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    ops = {"samples": dict(forward=11, vjp=11, gate_forward=256, probe_vjp=16 * 32, probe_forward=32,
                           rollout_forward=11, teacher=22)}
    assert rv.fe(ops, "a") == 11 + 22 + 256 + 3 * 16 * 32
    assert rv.fe(ops, "c") == 11 + 22 + 256 + 3 * 16 * 32 + 11 + 22
    ops2 = {"samples": dict(forward=11, vjp=11, gate_forward=0, probe_vjp=0, probe_forward=32, rollout_forward=0,
                            teacher=0)}
    assert rv.fe(ops2, "b") == 11 + 22 + 32                             # na_lora: one feature forward


def test_compare_vs_best_control():
    def m(rho):
        return dict(rho_sel=rho, rho_test=rho, E_test=0.8, FE=1.0)
    cm = {0: {0.8: {"vau": m(0.5), "penalty": m(1.0), "_best": ("penalty", m(1.0)), "_crash": {}}},
          1: {0.8: {"vau": None, "penalty": m(1.0), "_best": ("penalty", m(1.0)), "_crash": {}}}}
    lr = rv.compare(cm, 0.8)
    assert math.isclose(lr[0], math.log(0.5)) and lr[1] == math.log(100)
