#!/usr/bin/env python3
"""Single entry point that executes configs/eval_protocol.json (v1.1 + amendment A2).

    python code/run_validation.py --phase dev          # integrity, calibration (dev seeds 0..7), timing, projection
    python code/run_validation.py --phase validation   # refuses unless configs/FREEZE.json matches (see below)

The command re-executes itself under code/meter.py, so the whole phase is metered into results/ledger.jsonl.
Validation REFUSES to run unless configs/FREEZE.json exists and records sha256 of configs/eval_protocol.json,
configs/candidate.json, the code/vau tree and this file that equal the current files (vau/seeds.py:freeze_ok),
and carries the calibration (lr0, lambda0, tau0^2 per calibration key) and the measured timings.

Output: results/<phase>/raw/<family>/<cell>__s<seed>.jsonl.gz (one JSON line per run: config, checkpoint
trajectory, per-step log, gate-coverage record, FE), results/<phase>/summary.json (coverage, matched-efficacy
tables, family statistics, falsifiers F1-F5, verdict), results/<phase>/units.jsonl (per-unit CPU).
Every interpretation of an ambiguous protocol rule is listed in DECISIONS below and copied into summary.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import glob
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPRINT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from vau import adamw as aw  # noqa: E402
from vau import seeds as sd  # noqa: E402
from vau import tasks as tk  # noqa: E402
from vau.contracts import ContractViolation, Ops  # noqa: E402
from vau.methods import Batch, GateConfig, NotApplicable, h_norm, make_arm  # noqa: E402
from vau.models import FlowInput, TinyFlowDiT, euler, sample  # noqa: E402

PROTOCOL_PATH = os.path.join(SPRINT, "configs", "eval_protocol.json")
CANDIDATE_PATH = os.path.join(SPRINT, "configs", "candidate.json")
FREEZE_PATH = os.path.join(SPRINT, "configs", "FREEZE.json")

DECISIONS = [
    "D1 erasure minibatch (models a, b): 11 inputs per step drawn without replacement from target_train (32) with "
    "stream 2; n_latents = 11 (mirrors GEM's 11 prefix latents).",
    "D2 model a/b sizes: a: P=300, d_out=8, d_in=8 (z 4 + embedding 4, 2 concept coords); b: d_in=8, hidden 16, "
    "d_out=8, r=4 (P=192), tanh gain 1 unless the cell sets it; c: c=4, n_img=4, n_txt=3, dm=8, r=2 (P=128).",
    "D3 covered population: models a, b: retained inputs in a rank-q span (q=2, q*d_out=16=k) with zero concept "
    "coordinates; model c: retained conditioning tables in a rank-2 span. The rare subgroup (mass 0.005) keeps its "
    "generic form in both populations.",
    "D4 drift units: every rho (and the gate statistic and tau^2) is normalised by Z = mean L^2 over target_test at "
    "theta0, a fixture constant fixed before training; model c's gate acts on velocities and is normalised by "
    "Z_v = mean velocity contrast ||b-a||^2 on target-sel velocity queries; tau0 for model c is calibrated on the "
    "velocity drift rho_vsel so tau is in the gate's own units.",
    "D5 E_axial: q_i = clip(((b-v).e)/(L/2), 0, 1), i.e. axial progress measured from the target towards the anchor "
    "(0 at theta0, 1 at the bisector); the JSON wording '(v-a).e from the anchor' would give 1 at theta0.",
    "D6 gate batches: one pool of T*R_max batches per unit (stream 4), consumed sequentially by gate TRIES (the j-th "
    "try of every arm uses the j-th batch); null steps consume none; base outputs cached (not charged).",
    "D7 FE units: a/b: forward + 2*backward samples of the student on erasure and penalty inputs, 3 per sketch-row "
    "VJP sample (k*3*n1 per refresh), n1 for na_lora's single feature forward, n2 per gate try; teachers are not "
    "charged for a/b (protocol adamw_step_ab = 3*n_target_batch); model c additionally charges rollout and the two "
    "teacher forwards per latent (66 per iteration).",
    "D8 rho floor: rho values are floored at 1e-12 before log ratios (na_lora can give exactly 0).",
    "D9 calibration reference cells: G1 d=2 sketch generic; G2 kappa=1e2 phi=pi/8; G3 cos psi=0.1; G4 contrast "
    "1e-2 (model a) and the model-b cell; G5 gain 1 generic (also used by G6, G7); G9 itself; G8 generic (also GD_D); "
    "GD_B uses G1's lr0. If native adamw never reaches E_sel 0.8 the calibration level falls back to 0.5 then 0.2; "
    "lr0 = argmin |t_cross - 23| (t_cross = inf if unreached; ties -> larger final E).",
    "D10 G3: anisotropy 0 (column scaling would break the designed null/range split), kappa=1e2, phi=Haar; target "
    "inputs vary only in the concept coordinates whose Jacobian rows are cos(psi)*null + sin(psi)*range directions "
    "of the retained block; rich cell: generic T with P=40 < rank, so null(J_r) = {0}.",
    "D11 G7 uses the G5 stream code so the non-probe controls are reused from G5 gain-1 covered exactly (same seeds, "
    "same fixture); G6 is computed from the G5 gain-1 and G1 d=2 sketch runs with rho_shift as Q1 metric.",
    "D12 F1 is evaluated on the covered-population exact-J cells (d=2 OR d=3), ridge 1e-6, wd 0.01, lr0; generic "
    "cells and the ratio vs post_euclid are reported. P = E_test at step 10, or at the first step at which either "
    "arm's E_test reaches 0.95.",
    "D13 F4 is triggered if GMR(vau / best of {penalty+gate, penalty_earlystop}) >= 1.0 in G5 OR in G8 (the value "
    "claim needs both); evaluated on COVERED primary cells, else on all primary cells.",
    "D14 penalty_oracle: penalty with 28 lambda = lambda0*10^linspace(-3,3,28) x 4 lr in the G1 exact-J covered "
    "cells; 'flip' = G1's Q1 qualification differs when the oracle runs join the best-control pool.",
    "D15 GD_B: linear student v = b + delta + J theta with M = J J^T = diag(1 axial, kappa transverse), offset "
    "|delta| = eps (eps/L cell) orthogonal to the axis, eps_n = 0 for both losses; r_end at the first step with zero "
    "loss (or the last step); SGD lr 1e-3*L up to 20000 steps (gradient-flow proxy, the falsification basis) and "
    "AdamW at G1's lr0 for 60 steps (reported).",
    "D16 GD_D: rho_late = mean_{k>=11} ||v_student - v_base|| / mean_{k<=10} ||v_student - v_base|| on the target "
    "conditions along the trained prefix student's own rollouts; Spearman over (seed, arm) pairs.",
    "D17 coverage drift along vau runs: computed in one vau run per unit (lr0, first tau value) with a 64-row "
    "held-out sketch; theta0 coverage uses the protocol's 256 rows.",
    "D18 mechanism logs (beta_t, gamma_t, delta_t, imbalance, residual): every model-b run on full R_test; every "
    "model-c run on a fixed 256-query subset of the R_test velocity queries (cost); gamma_t = ratio of sums over "
    "layers and inputs.",
    "D19 G9 measurement runs: lr 1e-3 and each arm's first secondary value, T=23; the safety grid uses G9's own lr0.",
    "D20 failure: exception, NaN/Inf, or CPU time > 10x the frozen median per-run time of the model (dev: 120 s) "
    "marks the run failed; checkpoints after the failure have E = 0 and rho = +inf.",
    "D21 matched selection ties/unreached: an arm whose configs never reach e is unreached (rho = inf); a cell's "
    "fallback level applies to all arms of that cell.",
    "D22 vau_blocksplit on model a = one block (D/mean D), reported as identical to vau up to the ridge floor.",
    "D23 dev phase statistics are computed with the seeds that were run (thresholds such as '>= 9 of 12' are scaled "
    "as fractions of the number of seeds); they are DEVELOPMENT ONLY and never a verdict.",
]

# ------------------------------------------------------------------------------------------------------------
# protocol constants
# ------------------------------------------------------------------------------------------------------------
ARMS16 = ["vau", "vau_nogate", "vau_blocksplit", "adamw", "adamw_stepmatch", "penalty", "pre_proj", "post_euclid",
          "post_euclid_reject", "na_lora", "adamw+gate", "penalty+gate", "pre_proj+gate", "post_euclid_reject+gate",
          "penalty_earlystop", "na_lora+gate"]
ABLATIONS = ["vau_nogate", "vau_blocksplit"]
CONTROLS = [a for a in ARMS16 if a not in ["vau"] + ABLATIONS]
PENALTY_VARIANTS = ["penalty", "penalty+gate", "penalty_earlystop"]
NA_VARIANTS = ["na_lora", "na_lora+gate"]
IMPL = {"adamw+gate": "gated_adamw", "penalty+gate": "gated_penalty", "pre_proj+gate": "gated_pre_proj",
        "post_euclid_reject+gate": "gated_post_euclid_reject", "penalty_earlystop": "earlystop_penalty",
        "na_lora+gate": "gated_na_lora", "penalty_oracle": "penalty", "vau_tau0_paired": "vau"}
GATED = {"vau", "vau_blocksplit", "adamw+gate", "penalty+gate", "pre_proj+gate", "post_euclid_reject+gate",
         "penalty_earlystop", "na_lora+gate", "vau_tau0_paired"}
TAU_ARMS = {"vau", "vau_blocksplit", "adamw+gate", "penalty+gate", "pre_proj+gate", "post_euclid_reject+gate",
            "na_lora+gate"}
RIDGE_ARMS = {"pre_proj", "post_euclid", "post_euclid_reject", "vau_nogate"}
PROBE_ARMS = ["vau", "vau_nogate", "vau_blocksplit", "pre_proj", "post_euclid", "post_euclid_reject",
              "pre_proj+gate", "post_euclid_reject+gate", "adamw_stepmatch"]
LR_EXP = [-1.0, -0.5, 0.0, 0.5]
K_DEF, KREF_DEF, N1, R_MAX, DELTA_GATE = 16, 5, 32, 4, 0.05
SIZES = dict(target_train=32, target_sel=128, target_test=512, R1=32, R2=256, R_sel=512, R_test=2048,
             R_shift=1024, R_rare=512, population=20000)
MODEL_A = dict(P=300, d_out=8, z_dim=4, e_dim=4, n_concept=2)
MODEL_B = dict(d_hidden=16, d_out=8, r=4, z_dim=4, e_dim=4, n_concept=2)
MODEL_C = dict(c=4, n_img=4, n_txt=3, dm=8, r=2)
RHO_FLOOR = 1e-12
LOG100 = math.log(100.0)


def r7(x):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return float(f"{x:.7g}")


def unr7(x):
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[x]
    return x


# ------------------------------------------------------------------------------------------------------------
# cells
# ------------------------------------------------------------------------------------------------------------
def family_cells(family, ladder=()):
    """List of cell dicts: id, model ('a'|'b'|'c'), task kwargs, probe settings, role flags."""
    C = []
    if family == "G1":
        for d in (0, 1, 2, 3):
            for probe in ("exact", "sketch"):
                pops = ["generic"] + (["covered"] if d >= 2 else [])
                for pop in pops:
                    C.append(dict(id=f"d{d}_{probe}_{pop}", model="a", kw=dict(aniso_decades=float(d), population=pop),
                                  probe=probe, k=K_DEF, K=1 if probe == "exact" else KREF_DEF, primary=d >= 2,
                                  safety=d == 0, f1=(probe == "exact" and d >= 2), oracle=(probe == "exact" and d >= 2
                                                                                          and pop == "covered")))
    elif family == "G2":
        for kappa in (1e2, 1e6):
            for phi_name, phi in (("0", 0.0), ("pi8", math.pi / 8), ("pi4", math.pi / 4), ("haar", "haar")):
                C.append(dict(id=f"k{kappa:.0e}_phi{phi_name}", model="a",
                              kw=dict(aniso_decades=2.0, kappa=kappa, phi=phi), probe="sketch", k=K_DEF, K=KREF_DEF,
                              primary=phi_name != "0", safety=phi_name == "0"))
    elif family == "G3":
        for c in (1.0, 0.1, 0.01, 0.0):
            C.append(dict(id=f"cospsi{c:g}", model="a", kw=dict(aniso_decades=0.0, kappa=1e2, phi="haar",
                                                                target_mode="concept", cospsi=c),
                          probe="sketch", k=K_DEF, K=KREF_DEF, primary=False, safety=True))
        C.append(dict(id="rich", model="a", kw=dict(aniso_decades=0.0, P=40), probe="sketch", k=K_DEF, K=KREF_DEF,
                      primary=False, safety=True))
    elif family == "G4":
        for c in (1.0, 1e-2, 1e-4, 1e-6):
            C.append(dict(id=f"a_L{c:g}_epsnat", model="a", kw=dict(contrast_scale=c), probe="sketch", k=K_DEF,
                          K=KREF_DEF, primary=False, safety=True))
        for c in (1e-4, 1e-6):
            C.append(dict(id=f"a_L{c:g}_eps0", model="a", kw=dict(contrast_scale=c), eps_n=0.0, probe="sketch",
                          k=K_DEF, K=KREF_DEF, primary=False, safety=True))
        C.append(dict(id="b_L0.0001_epsnat", model="b", kw=dict(contrast_scale=1e-4, gain=1.0), probe="sketch",
                      k=K_DEF, K=KREF_DEF, primary=False, safety=True, calib="G4b"))
    elif family == "G5":
        for gain, pop in ((0.5, "generic"), (1.0, "generic"), (1.0, "covered"), (2.0, "generic")):
            if "L4" in ladder and gain == 0.5:
                continue
            C.append(dict(id=f"gain{gain:g}_{pop}", model="b", kw=dict(gain=gain, population=pop), probe="sketch",
                          k=K_DEF, K=KREF_DEF, primary=True, safety=False, gauge=(gain == 1.0 and pop == "generic")))
    elif family == "G7":
        for K, k in ((1, 32), (5, 32), (20, 8), (None, 32)):
            if "L3" in ladder and (K, k) == (20, 8):
                continue
            C.append(dict(id=f"K{K if K else 'inf'}_k{k}", model="b", kw=dict(gain=1.0, population="covered"),
                          probe="sketch", k=k, K=K, primary=(K, k) in ((1, 32), (5, 32)), safety=False,
                          stream_family="G5", calib="G5", reuse_cell="gain1_covered"))
    elif family == "G9":
        C.append(dict(id="zeroinit", model="b", kw=dict(gain=1.0, population="generic"), probe="sketch", k=K_DEF,
                      K=KREF_DEF, primary=False, safety=True, T=23, measure=True))
    elif family == "G8":
        for pop in ("generic", "covered"):
            C.append(dict(id=pop, model="c", kw=dict(population=pop), probe="sketch", k=K_DEF, K=KREF_DEF,
                          primary=True, safety=False))
    elif family == "GD_B":
        for r in (1e-6, 1e-4, 1e-2):
            for kap in (1, 10, 100):
                C.append(dict(id=f"eps{r:g}_kappa{kap}", model="gdb", eps_over_L=r, kappa=kap))
    elif family == "GD_D":
        C.append(dict(id="prefix_vs_uniform", model="c", kw=dict(population="generic"), probe="sketch", k=K_DEF,
                      K=KREF_DEF, calib="G8"))
    else:
        raise ContractViolation(f"unknown family {family}")
    for c in C:
        c.setdefault("calib", family)
        c.setdefault("stream_family", family)
    return C


def model_T(model, cell, ladder):
    if cell.get("T"):
        return cell["T"]
    if model == "c":
        return 24 if "L5" in ladder else 28
    return 40 if "L2" in ladder else 60


def checkpoints(model, T):
    if model == "a":
        return list(range(0, T + 1))
    if model == "b":
        return sorted(set(list(range(0, T + 1, 2)) + [T]))
    return sorted(set(list(range(0, T + 1, 4)) + [min(23, T), T]))


# ------------------------------------------------------------------------------------------------------------
# unit (fixture + data) construction
# ------------------------------------------------------------------------------------------------------------
class Unit:
    pass


def _g3_concept_fn(cospsi):
    def fn(Jr, rng):
        P = Jr.shape[1]
        U, s, Vt = np.linalg.svd(Jr, full_matrices=True)
        r = int(np.sum(s > 1e-10 * s.max()))
        Rb, Nb = Vt[:r], Vt[r:]
        rows = []
        n_rows = 2 * MODEL_A["d_out"]
        for _ in range(n_rows):
            nn = rng.normal(size=Nb.shape[0]) @ Nb
            rr = rng.normal(size=Rb.shape[0]) @ Rb
            row = cospsi * nn / np.linalg.norm(nn) + math.sqrt(max(0.0, 1 - cospsi ** 2)) * rr / np.linalg.norm(rr)
            rows.append(row)
        return np.array(rows) * np.linalg.norm(Jr) / math.sqrt(Jr.shape[0])
    return fn


def _hinge_obs(v, a, b, Lfloor):
    dpv, dnv = np.linalg.norm(v - a, axis=1), np.linalg.norm(v - b, axis=1)
    Lraw = np.linalg.norm(b - a, axis=1)
    L = np.maximum(Lraw, Lfloor)
    p = 1.0 - np.maximum(dpv - dnv, 0.0) / L
    e = (b - a) / np.where(Lraw > 0, Lraw, 1.0)[:, None]
    q = np.clip(np.sum((b - v) * e, axis=1) / (L / 2.0), 0.0, 1.0)
    w = v - a
    r = np.linalg.norm(w - np.sum(w * e, axis=1)[:, None] * e, axis=1) / L
    return p, q, r


def build_unit(family, cell, seed, ladder, freeze_path=FREEZE_PATH):
    U = Unit()
    U.family, U.cell, U.seed, U.ladder = family, cell, seed, tuple(ladder)
    U.model_kind = cell["model"]
    st = sd.streams(cell["stream_family"], seed, freeze_path=freeze_path)
    U.T = model_T(U.model_kind, cell, ladder)
    U.ck = checkpoints(U.model_kind, U.T)
    if U.model_kind in ("a", "b"):
        _build_vector(U, cell, st)
    else:
        _build_flow(U, cell, st, ladder)
    U.streams = {k: st[k] for k in ("train", "sketch")}
    U.gauge_rng = copy.deepcopy(st["fixture"])
    U.cov_rng = st["spare"]
    return U


def _build_vector(U, cell, st):
    kw = dict(cell["kw"])
    fam = "linear" if U.model_kind == "a" else "mlp"
    base = dict(MODEL_A) if fam == "linear" else dict(MODEL_B)
    if fam == "linear" and "P" in kw:
        base["P"] = kw.pop("P")
    cospsi = kw.pop("cospsi", None)
    if cospsi is not None:
        kw["T_concept"] = _g3_concept_fn(cospsi)
    task = tk.make_task(fam, st["fixture"], **base, **kw)
    if "eps_n" in cell:
        task.eps_n = cell["eps_n"]
    m = task.model
    U.task, U.model = task, m
    U.theta0 = m.init_theta(st["fixture"])
    sp = st["splits"]
    dim = task.z_dim if task.target_mode == "z" else task.n_concept
    tsets = {k: sp.normal(size=(SIZES[k], dim)) for k in ("target_train", "target_sel", "target_test")}
    U.target_train = tsets["target_train"]
    U.tgt = {}
    for k in ("target_sel", "target_test"):
        xt, xa = task.target_inputs(tsets[k])
        U.tgt[k] = dict(xs=xt, xa=xa, a=m.forward(xa, None), b=m.forward(xt, None))
    Lt = np.linalg.norm(U.tgt["target_test"]["b"] - U.tgt["target_test"]["a"], axis=1)
    U.Z = float(np.mean(Lt ** 2))
    if not U.Z > 0:
        raise ContractViolation("degenerate fixture: Z = 0")
    for k in U.tgt:
        L = np.linalg.norm(U.tgt[k]["b"] - U.tgt[k]["a"], axis=1)
        U.tgt[k]["Lfloor"] = 1e-3 * float(np.median(L))
    U.R1 = task.retained_batch(sp, SIZES["R1"])
    U.ret = {}
    for k, kw2 in (("R_sel", {}), ("R_test", {}), ("R_shift", dict(shift=1.0)), ("R_rare", dict(rare=True))):
        x = task.retained_batch(sp, SIZES[k], **kw2)
        U.ret[k] = dict(x=x, f0=m.forward(x, U.theta0))
    U.ret["R1"] = dict(x=U.R1, f0=m.forward(U.R1, U.theta0))
    U.n2 = SIZES["R2"]
    g = st["gate"]
    U.gate_pool = []
    for _ in range(U.T * R_MAX):
        x = task.retained_batch(g, U.n2)
        U.gate_pool.append((x, m.forward(x, U.theta0)))
    pr = st["population"]
    xp = task.retained_batch(pr, SIZES["population"])
    U.pop = dict(x=xp, f0=m.forward(xp, U.theta0), rare=np.all(np.isclose(
        xp[:, task.z_dim:], task.feat_scale[task.z_dim:] * task.emb[task.rare_id]), axis=1))
    U.mech_x = U.ret["R_test"]["x"]
    U.mech_f0 = None


def _build_flow(U, cell, st, ladder):
    rng = st["fixture"]
    n_tt, n_ts, n_te, n_rp, n_rs, n_rt = 4, 4, 8, 8, 8, 32
    n_pr = 1 + n_tt + n_ts + n_te + n_rp + n_rs + n_rt
    model = TinyFlowDiT(rng, n_prompts=n_pr, **MODEL_C)
    ids = np.arange(n_pr)
    U.ids = dict(target_train=ids[1:1 + n_tt], target_sel=ids[1 + n_tt:1 + n_tt + n_ts],
                 target_test=ids[1 + n_tt + n_ts:1 + n_tt + n_ts + n_te])
    o = 1 + n_tt + n_ts + n_te
    U.ids.update(R_probe=ids[o:o + n_rp], R_sel=ids[o + n_rp:o + n_rp + n_rs], R_test=ids[o + n_rp + n_rs:])
    E = model.E
    Et = E[1].copy()
    for i in np.concatenate([U.ids["target_train"], U.ids["target_sel"], U.ids["target_test"]]):
        E[i] = Et + 0.3 * rng.normal(size=Et.shape)
    if cell["kw"]["population"] == "covered":
        W = rng.normal(size=(2,) + Et.shape)
        for i in np.concatenate([U.ids["R_probe"], U.ids["R_sel"], U.ids["R_test"]]):
            E[i] = np.tensordot(rng.normal(size=2), W, axes=1)
    task = tk.FlowTask(model, n_steps=28, end_idx=10, mu=0.630, n_heldout=0, n_retained_prompts=n_rp)
    U.task, U.model = task, model
    U.theta0 = model.init_theta(rng)
    sp, ev = st["splits"], st["eval_noise"]
    n_noise_test = 32 if "L1" in ladder else 64
    U.R1 = task.retained_batch(sp, SIZES["R1"], ids=U.ids["R_probe"])
    U.vel = {}
    for k, pool, n in (("R_sel", "R_sel", 512), ("R_test", "R_test", 2048)):
        x = task.retained_batch(sp, n, ids=U.ids[pool])
        U.vel[k] = dict(x=x, f0=model.forward(x, U.theta0))
    U.vel["R1"] = dict(x=U.R1, f0=model.forward(U.R1, U.theta0))
    # target velocity contrast for Z_v (target-sel velocity queries on base trajectories)
    xq = task.retained_batch(sp, 512, ids=U.ids["target_sel"])
    xa = FlowInput(xq.z, xq.sigma, np.zeros_like(xq.prompt))
    U.Zv = float(np.mean(np.sum((model.forward(xq, None) - model.forward(xa, None)) ** 2, axis=1)))
    sig = task.sigmas

    def endpoints(theta, z, p):
        return model.sample_fast(theta, z, p, sig).reshape(z.shape[0], -1)
    U.endpoints = endpoints
    U.tgt = {}
    for k, nn in (("target_sel", 16), ("target_test", n_noise_test)):
        tids = np.repeat(U.ids[k], nn)
        z = ev.normal(size=(len(tids), model.n_img, model.c))
        U.tgt[k] = dict(z=z, p=tids, a=endpoints(None, z, np.zeros_like(tids)), b=endpoints(None, z, tids))
        L = np.linalg.norm(U.tgt[k]["b"] - U.tgt[k]["a"], axis=1)
        U.tgt[k]["Lfloor"] = 1e-3 * float(np.median(L))
    U.Z = float(np.mean(np.sum((U.tgt["target_test"]["b"] - U.tgt["target_test"]["a"]) ** 2, axis=1)))
    U.ret = {}
    for k, nn in (("R_sel", 16), ("R_test", n_noise_test)):
        rids = np.repeat(U.ids[k], nn)
        z = ev.normal(size=(len(rids), model.n_img, model.c))
        U.ret[k] = dict(z=z, p=rids, f0=endpoints(U.theta0, z, rids))
    U.n2 = 64
    g = st["gate"]
    U.gate_pool = []
    for _ in range(U.T * R_MAX):
        x = task.retained_batch(g, U.n2, ids=U.ids["R_probe"])
        U.gate_pool.append((x, model.forward(x, U.theta0)))
    U.pop = None
    sub = np.arange(256)
    xt = U.vel["R_test"]["x"]
    U.mech_x = FlowInput(xt.z[sub], xt.sigma[sub], xt.prompt[sub])
    U.target_train_ids = U.ids["target_train"]


# ------------------------------------------------------------------------------------------------------------
# evaluation at a checkpoint
# ------------------------------------------------------------------------------------------------------------
def evaluate(U, theta):
    """Checkpoint observables; the theta0 evaluation is cached per unit (identical for every arm)."""
    if np.array_equal(theta, U.theta0) and getattr(U, "_ev0", None) is not None:
        return dict(U._ev0)
    out = _evaluate(U, theta)
    if np.array_equal(theta, U.theta0):
        U._ev0 = dict(out)
    return out


def _evaluate(U, theta):
    m, out = U.model, {}
    if U.model_kind in ("a", "b"):
        if not hasattr(U, "_evcat"):     # one forward per checkpoint over all evaluation inputs
            keys = ["target_sel", "target_test", "R_sel", "R_test", "R_shift", "R_rare", "R1"]
            xs = [U.tgt[k]["xs"] if k.startswith("target") else U.ret[k]["x"] for k in keys]
            U._evcat = (np.concatenate(xs), dict(zip(keys, np.cumsum([0] + [x.shape[0] for x in xs])[:-1])),
                        dict(zip(keys, [x.shape[0] for x in xs])))
        X, off, nn = U._evcat
        F = m.forward(X, theta)
        part = {k: F[off[k]:off[k] + nn[k]] for k in off}
        for k, name in (("target_sel", "sel"), ("target_test", "test")):
            T = U.tgt[k]
            p, q, r = _hinge_obs(part[k], T["a"], T["b"], T["Lfloor"])
            out[f"E_{name}"], out[f"Eax_{name}"] = float(p.mean()), float(q.mean())
            if name == "test":
                out["Epass_test"], out["coll_test"] = float(np.mean(p >= 1 - 1e-12)), float(r.mean())
        for k in ("R_sel", "R_test", "R_shift", "R_rare", "R1"):
            R = U.ret[k]
            d = np.sum((part[k] - R["f0"]) ** 2, axis=1) / U.Z
            out[f"rho_{k[2:] if k != 'R1' else 'R1'}"] = float(d.mean())
            if k == "R_test":
                out["rho_max"], out["rho_q99"] = float(d.max()), float(np.quantile(d, 0.99))
        out["rho_R1_gap"] = out["rho_R1"] - out["rho_test"]
    else:
        if not hasattr(U, "_evcat"):
            parts = [U.tgt["target_sel"], U.tgt["target_test"], U.ret["R_sel"], U.ret["R_test"]]
            U._evcat = (np.concatenate([q["z"] for q in parts]), np.concatenate([q["p"] for q in parts]),
                        np.cumsum([0] + [q["z"].shape[0] for q in parts]))
        zc, pc, cut = U._evcat
        allend = U.endpoints(theta, zc, pc)
        ends = {k: allend[cut[i]:cut[i + 1]] for i, k in enumerate(("target_sel", "target_test", "R_sel", "R_test"))}
        for k, name in (("target_sel", "sel"), ("target_test", "test")):
            T = U.tgt[k]
            p, q, r = _hinge_obs(ends[k], T["a"], T["b"], T["Lfloor"])
            out[f"E_{name}"], out[f"Eax_{name}"] = float(p.mean()), float(q.mean())
            if name == "test":
                out["Epass_test"], out["coll_test"] = float(np.mean(p >= 1 - 1e-12)), float(r.mean())
        for k, name in (("R_sel", "sel"), ("R_test", "test")):
            R = U.ret[k]
            d = np.sum((ends[k] - R["f0"]) ** 2, axis=1) / U.Z
            out[f"rho_{name}"] = float(d.mean())
            if name == "test":
                out["rho_max"], out["rho_q99"] = float(d.max()), float(np.quantile(d, 0.99))
        for k, name in (("R_sel", "vsel"), ("R_test", "vtest"), ("R1", "vR1")):
            V = U.vel[k]
            out[f"rho_{name}"] = float(np.mean(np.sum((m.forward(V["x"], theta) - V["f0"]) ** 2, axis=1)) / U.Zv)
        out["rho_R1_gap"] = out["rho_vR1"] - out["rho_vtest"]
    return out


def gate_units(U):
    """(Z_unit) multiplying normalised tau^2 into the gate's raw units."""
    return U.Z if U.model_kind in ("a", "b") else U.Zv


# ------------------------------------------------------------------------------------------------------------
# FE accounting (protocol forward_equivalent_units; decision D7)
# ------------------------------------------------------------------------------------------------------------
def fe(ops_snap, model_kind, init_teacher=0):
    s = ops_snap["samples"]
    probe = 3 * s["probe_vjp"] if s["probe_vjp"] > 0 else s["probe_forward"]
    val = s["forward"] + 2 * s["vjp"] + s["gate_forward"] + probe
    if model_kind == "c":
        val += s["rollout_forward"] + s["teacher"] - init_teacher
    return float(val)


# ------------------------------------------------------------------------------------------------------------
# configs per arm (tuning budget)
# ------------------------------------------------------------------------------------------------------------
def arm_configs(arm, calib, g8=False):
    lr0, lam0, tau0 = calib["lr0"], calib["lambda0"], calib["tau0_sq"]
    lrs = [lr0 * 10 ** e for e in LR_EXP]
    if g8:
        centre = {"adamw": dict(wd=0.01), "adamw_stepmatch": dict(c=1.0), "penalty": dict(lam=lam0),
                  "penalty_earlystop": dict(lam=lam0, tau2=tau0), "na_lora": dict(energy=0.999),
                  "na_lora+gate": dict(energy=0.999, tau2=tau0), "penalty+gate": dict(lam=lam0, tau2=tau0)}
        sec = [centre.get(arm, dict(ridge=1e-4) if arm in RIDGE_ARMS else dict(tau2=tau0) if arm in TAU_ARMS else {})]
    else:
        sec = {
            "adamw": [dict(wd=0.0), dict(wd=0.01)],
            "adamw_stepmatch": [dict(c=0.5), dict(c=1.0)],
            "penalty": [dict(lam=lam0 * 0.3), dict(lam=lam0 * 3)],
            "penalty_earlystop": [dict(lam=lam0 * 0.3, tau2=tau0), dict(lam=lam0 * 3, tau2=tau0)],
            "na_lora": [dict(energy=0.99), dict(energy=0.999)],
            "na_lora+gate": [dict(energy=0.999, tau2=tau0 * 0.5), dict(energy=0.999, tau2=tau0 * 1.5)],
            "penalty+gate": [dict(lam=lam0, tau2=tau0 * 0.5), dict(lam=lam0, tau2=tau0 * 1.5)],
        }.get(arm)
        if sec is None:
            if arm in RIDGE_ARMS:
                sec = [dict(ridge=1e-6), dict(ridge=1e-2)]
            elif arm in TAU_ARMS:
                sec = [dict(tau2=tau0 * 0.5), dict(tau2=tau0 * 1.5)]
            else:
                raise ContractViolation(f"no secondary grid for {arm}")
    return [dict(lr=lr, **s) for lr in lrs for s in sec]


def build_arm(U, arm, cfg, inject=None):
    impl = IMPL.get(arm, arm)
    hp = aw.AdamWHP(lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01))
    kw = dict(probe=U.cell.get("probe", "sketch"), k=U.cell.get("k", K_DEF), K=U.cell.get("K", KREF_DEF))
    if "ridge" in cfg:
        kw["ridge_rel"] = cfg["ridge"]
    if "lam" in cfg:
        kw["lam"] = cfg["lam"]
    if "energy" in cfg:
        kw["energy"] = cfg["energy"]
    if impl == "adamw_stepmatch":
        kw.update(ref_norms=cfg["ref_norms"], c=cfg["c"], metric="H")
    if inject is not None:
        kw["inject"] = inject
    gate = None
    if arm in GATED:
        tau2 = cfg["tau2"] * gate_units(U)
        gate = GateConfig(tau2=tau2, M=2 * tau2, n2=U.n2, delta=DELTA_GATE, T=U.T, R=R_MAX)
    return make_arm(impl, U.task, hp=hp, gate=gate, **kw)


# ------------------------------------------------------------------------------------------------------------
# one run
# ------------------------------------------------------------------------------------------------------------
class RunFailure(Exception):
    pass


def _erasure_batch(U, rng, scheme="prefix"):
    if U.model_kind in ("a", "b"):
        return U.target_train[rng.choice(U.target_train.shape[0], U.task.n_latents, replace=False)]
    z0 = rng.normal(size=(U.model.n_img, U.model.c))
    tid = int(rng.choice(U.target_train_ids))
    idx = None if scheme == "prefix" else np.sort(rng.choice(28, U.task.n_latents, replace=False))
    return (z0, tid, idx)


def _lora_split(U, vec):
    lay = getattr(U.model, "lora", None)
    if lay is None:
        return None
    ia = np.concatenate([np.arange(sa.start, sa.stop) for sa, _, _, _ in lay.slices.values()])
    ib = np.concatenate([np.arange(sb.start, sb.stop) for _, sb, _, _ in lay.slices.values()])
    return vec[ia], vec[ib]


def _mechanism(U, theta, d_applied, info, f_prev, f_new):
    """Per-step G9 quantities on the mechanism set (decision D18)."""
    m, x = U.model, U.mech_x
    df = f_new - f_prev
    num = float(np.sum(df ** 2))
    jv = m.jvp(x, theta, d_applied) if np.any(d_applied) else np.zeros_like(df)
    rec = dict(delta=num / m.n(x) / gate_units(U), first_order=float(np.sum(jv ** 2)) / m.n(x) / gate_units(U),
               beta=float(np.sum((df - jv) ** 2)) / num if num > 0 else None)
    lay = getattr(m, "lora", None)
    if lay is not None and np.any(d_applied):
        ab, dab = lay.unpack(theta), lay.unpack(d_applied)
        feats = m.lora_inputs(x, theta)
        bil = tot = 0.0
        for name, (A, B) in ab.items():
            dA, dB = dab[name]
            X = feats[name]
            bil += float(np.sum((X @ (lay.s * dB @ dA).T) ** 2))
            tot += float(np.sum((X @ (lay.s * (dB @ A + B @ dA + dB @ dA)).T) ** 2))
        rec["gamma"] = bil / tot if tot > 0 else None
        D = info.get("_D")
        if D is not None:
            sp = _lora_split(U, D)
            rec["DA_over_DB"] = float(np.max(sp[0]) / np.median(sp[1]))
        spd = _lora_split(U, d_applied)
        nb = float(np.linalg.norm(spd[1]))
        rec["dA_over_dB"] = float(np.linalg.norm(spd[0])) / nb if nb > 0 else None
    d0 = info.get("_delta0")
    if d0 is not None:
        n0 = float(np.linalg.norm(d0))
        d1 = info.get("_delta1_raw")
        rec["d1_over_d0"] = float(np.linalg.norm(d1)) / n0 if n0 > 0 else None
        S = info.get("_S")
        if S is not None and S.shape[0] > 0:
            s0 = float(np.linalg.norm(S @ d0))
            rec["residual"] = float(np.linalg.norm(S @ d1)) / s0 if s0 > 0 else None
    return rec


def _held_sketch(U, theta, rows, rng):
    """J_held = Omega J(R_test) (rows x P), Omega iid N(0,1) (stream 7)."""
    m = U.model
    x = U.ret["R_test"]["x"] if U.model_kind in ("a", "b") else U.vel["R_test"]["x"]
    N, d = m.n(x), m.out_dim
    Om = rng.standard_normal((rows, N, d))
    if hasattr(m, "vjp_rows"):
        return m.vjp_rows(x, Om)
    return np.stack([m.vjp(x, theta, Om[i]) for i in range(rows)])


def _rowspace(S, tol=1e-10):
    if S.shape[0] == 0 or not np.any(S):
        return np.zeros((S.shape[1], 0))
    _, s, Vt = np.linalg.svd(S, full_matrices=False)
    return Vt[s > tol * s.max()].T


def _cprobe(Jh, S, w=None):
    if w is not None:
        Jh, S = Jh * np.sqrt(w), S * np.sqrt(w)
    Q = _rowspace(S)
    den = float(np.sum(Jh ** 2))
    return float(np.sum((Jh @ Q) ** 2)) / den if den > 0 else 1.0


def run_arm(U, arm, cfg, time_limit, inject=None, coverage_track=False, f3=True):
    rec = dict(family=U.family, cell=U.cell["id"], seed=U.seed, arm=arm,
               cfg={k: (r7(v) if isinstance(v, float) else v) for k, v in cfg.items() if k != "ref_norms"},
               status="ok")
    t0 = time.process_time()
    try:
        arm_obj = build_arm(U, arm, cfg, inject=inject)
    except NotApplicable:
        rec["status"] = "NOT_APPLICABLE"
        return rec
    train = copy.deepcopy(U.streams["train"])
    sketch = copy.deepcopy(U.streams["sketch"])
    ms = arm_obj.init_state(U.theta0, U.R1, sketch)
    init_teacher = ms.ops.samples["teacher"]
    th = arm_obj.initial_theta(ms)
    if arm in ("na_lora", "na_lora+gate"):
        rec["na_no_capacity"] = bool(ms.cache.get("no_capacity"))
        rec["na_feature_ranks"] = {k: list(v) for k, v in ms.cache.get("feature_ranks", {}).items()}
    traj = {"t": []}

    def push(t, obs, fe_val):
        traj["t"].append(t)
        for kk, vv in obs.items():
            traj.setdefault(kk, []).append(r7(vv))
        traj.setdefault("FE", []).append(r7(fe_val))
    push(0, evaluate(U, th), 0.0)
    steps = {k: [] for k in ("s", "acc", "null", "tries", "Hnorm")}
    mech, gate_ptr = [], 0
    f3rec = dict(n_acc=0, viol=False, viol_unclipped=False, viol_rare=False, max_pop=0.0) if (
        f3 and arm in GATED and U.pop is not None) else None
    cov = [] if coverage_track else None
    if cov is not None:
        cov_rng = np.random.default_rng(np.random.SeedSequence([U.seed, 7, 64]))
        Jh_rows = 64
    want_mech = U.model_kind in ("b", "c")
    f_prev = U.model.forward(U.mech_x, th) if want_mech else None
    builds_before = 0
    scheme = cfg.get("scheme", "prefix")
    try:
        for t in range(1, U.T + 1):
            eb = _erasure_batch(U, train, scheme)
            gb = tuple(U.gate_pool[gate_ptr:gate_ptr + R_MAX]) if arm in GATED else ()
            if arm in GATED and len(gb) < R_MAX:
                raise RunFailure("gate pool exhausted")
            b = Batch(erase=eb, gate=gb)
            th_new, info = arm_obj.step(ms, th, b)
            if not np.all(np.isfinite(th_new)):
                raise RunFailure("nonfinite parameters")
            tries = len(info.get("rho_hat", []) or [])
            gate_ptr += tries
            steps["s"].append(r7(info["s"])); steps["acc"].append(bool(info["accepted"]))
            steps["null"].append(bool(info.get("nonprogress_zeroed", False))); steps["tries"].append(tries)
            steps["Hnorm"].append(r7(info.get("step_Hnorm", 0.0)))
            d_app = th_new - th
            if want_mech:
                f_new = U.model.forward(U.mech_x, th_new)
                mr = _mechanism(U, th, d_app, info, f_prev, f_new)
                mech.append({k: r7(v) for k, v in mr.items()})
                f_prev = f_new
            if f3rec is not None and info["accepted"] and info["s"] > 0:
                gtau = arm_obj.gate.tau2
                d = np.sum((U.model.forward(U.pop["x"], th_new) - U.pop["f0"]) ** 2, axis=1)
                clipped = float(np.mean(np.minimum(d, arm_obj.gate.M)))
                f3rec["n_acc"] += 1
                f3rec["max_pop"] = max(f3rec["max_pop"], clipped / gtau)
                f3rec["viol"] |= clipped > gtau
                f3rec["viol_unclipped"] |= float(np.mean(d)) > gtau
                if np.any(U.pop["rare"]):
                    f3rec["viol_rare"] |= float(np.mean(d[U.pop["rare"]])) > gtau
            if cov is not None and ms.ops.calls["probe_build"] > builds_before and ms.probe is not None:
                Jh = _held_sketch(U, th, Jh_rows, cov_rng)
                cov.append([t, r7(_cprobe(Jh, ms.probe))])
            builds_before = ms.ops.calls["probe_build"]
            th = th_new
            if t in U.ck:
                push(t, evaluate(U, th), fe(ms.ops.snapshot(), U.model_kind, init_teacher))
            if time.process_time() - t0 > time_limit:
                raise RunFailure(f"timeout > {time_limit:.1f} s")
    except (RunFailure, ContractViolation, FloatingPointError, np.linalg.LinAlgError, ValueError) as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    except Exception as e:  # any crash is recorded, never retried
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:300] + " | " + traceback.format_exc()[-300:]
    rec["traj"], rec["steps"] = traj, steps
    if mech:
        rec["mech"] = mech
    if f3rec is not None:
        f3rec["max_pop"] = r7(f3rec["max_pop"])
        rec["f3"] = f3rec
    if cov is not None:
        rec["cov_refresh"] = cov
    rec["ops"] = ms.ops.snapshot()
    rec["cpu_s"] = r7(time.process_time() - t0)
    return rec


# ------------------------------------------------------------------------------------------------------------
# coverage at theta0
# ------------------------------------------------------------------------------------------------------------
def coverage_theta0(U):
    m = U.model
    rng = U.cov_rng
    Jh = _held_sketch(U, U.theta0, 256, rng)
    cell = U.cell
    ops = Ops()
    if cell.get("probe") == "exact":
        from vau.projections import exact_probe
        S = exact_probe(m, U.R1, U.theta0, ops)
    else:
        from vau.projections import sketch_probe
        S = sketch_probe(m, U.R1, U.theta0, cell.get("k", K_DEF), copy.deepcopy(U.streams["sketch"]), ops)
    out = dict(C_probe=_cprobe(Jh, S))
    # D-weighted (D from vau's step-1 AdamW state at theta0 on the first erasure batch)
    _, g = U.task.erasure_loss_grad(U.theta0, _erasure_batch(U, copy.deepcopy(U.streams["train"])), Ops())
    _, _, D = aw.peek(aw.init_state(m.P), g, U.theta0, aw.AdamWHP())
    out["C_probe_D"] = _cprobe(Jh, S, D)
    sv = np.linalg.svd(Jh, compute_uv=False)
    cum = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-300)
    out["eff_rank_99"] = int(np.searchsorted(cum, 0.99) + 1)
    out["k_eff"] = int(_rowspace(S).shape[1]) if cell.get("probe") == "exact" else int(S.shape[0])
    if hasattr(m, "lora_inputs"):
        xt = U.ret["R_test"]["x"] if U.model_kind == "b" else U.vel["R_test"]["x"]
        Ft, F1 = m.lora_inputs(xt, U.theta0), m.lora_inputs(U.R1, U.theta0)
        num = den = 0.0
        for name in Ft:
            Q = _rowspace(F1[name])
            num += float(np.sum((Ft[name] @ Q) ** 2))
            den += float(np.sum(Ft[name] ** 2))
        out["C_feat"] = num / den if den > 0 else 1.0
    return {k: r7(v) if isinstance(v, float) else v for k, v in out.items()}


def disjointness(U):
    """I10: no input hash in two sets (target sets, R1, eval sets, gate pool, population)."""
    sets = {}
    if U.model_kind in ("a", "b"):
        sets["target_sel"], sets["target_test"] = U.tgt["target_sel"]["xs"], U.tgt["target_test"]["xs"]
        for k in U.ret:
            sets[k] = U.ret[k]["x"]
        sets["gate"] = np.concatenate([x for x, _ in U.gate_pool])
        sets["population"] = U.pop["x"]
        # rare-subgroup rows carry identical embeddings but distinct z; hash full rows
    else:
        for k in U.vel:
            sets["vel_" + k] = U.vel[k]["x"]
        g = U.gate_pool
        sets["gate"] = FlowInput(np.concatenate([x.z for x, _ in g]), np.concatenate([x.sigma for x, _ in g]),
                                 np.concatenate([x.prompt for x, _ in g]))
    seen, dup = {}, 0
    for name, x in sets.items():
        for h in set(tk.input_hashes(x)):
            if h in seen and seen[h] != name:
                dup += 1
            seen[h] = name
    return dup


# ------------------------------------------------------------------------------------------------------------
# a unit = (family, cell, seed): all arms x configs
# ------------------------------------------------------------------------------------------------------------
def unit_arms(family, cell):
    if family == "G7":
        return PROBE_ARMS
    if family == "GD_D":
        return []
    return ARMS16


def run_unit(spec):
    """Worker: build fixture, run every arm/config, write raw gz JSONL; returns a small summary."""
    t0 = time.process_time()
    family, cell, seed = spec["family"], spec["cell"], spec["seed"]
    out_path = spec["out_path"]
    try:
        if family == "GD_B":
            recs = run_gdb_unit(cell, seed, spec["calib"]["G1"])
        else:
            U = build_unit(family, cell, seed, spec["ladder"], spec.get("freeze_path", FREEZE_PATH))
            recs = [dict(kind="unit", family=family, cell=cell["id"], seed=seed, coverage=coverage_theta0(U),
                         disjoint_dups=disjointness(U), Z=r7(U.Z), T=U.T)]
            calib = spec["calib"][cell["calib"]]
            limit = spec["time_limit"][U.model_kind]
            if family == "GD_D":
                recs += run_gdd_unit(U, calib, limit)
            else:
                g8 = family == "G8"
                arms = unit_arms(family, cell)
                if "only_arms" in spec:
                    arms = [a for a in arms if a in spec["only_arms"]]
                paired = {}
                if "adamw_stepmatch" in arms:
                    for lr in [calib["lr0"] * 10 ** e for e in LR_EXP]:
                        r = run_arm(U, "vau_tau0_paired", dict(lr=lr, tau2=calib["tau0_sq"]), limit, f3=False)
                        paired[r7(lr)] = [unr7(h) if a else 0.0 for h, a in zip(r["steps"]["Hnorm"], r["steps"]["acc"])]
                        paired[r7(lr)] += [0.0] * (U.T - len(paired[r7(lr)]))
                        recs.append(r)
                tracked = False
                for arm in arms:
                    for cfg in arm_configs(arm, calib, g8=g8):
                        if arm == "adamw_stepmatch":
                            cfg["ref_norms"] = paired[r7(cfg["lr"])]
                        track = (arm == "vau" and not tracked and abs(cfg["lr"] - calib["lr0"]) < 1e-15 * cfg["lr"]
                                 + 1e-18)
                        tracked |= track
                        recs.append(run_arm(U, arm, cfg, limit, coverage_track=track))
                if cell.get("oracle"):
                    for lr in [calib["lr0"] * 10 ** e for e in LR_EXP]:
                        for lam in calib["lambda0"] * 10 ** np.linspace(-3, 3, 28):
                            recs.append(run_arm(U, "penalty_oracle", dict(lr=lr, lam=float(lam)), limit, f3=False))
                if cell.get("gauge"):
                    recs += run_gauge(U, calib, limit)
                if cell.get("measure"):
                    for arm in ARMS16:
                        cfg = dict(arm_configs(arm, calib)[0])
                        cfg["lr"] = 1e-3
                        if arm == "adamw_stepmatch":
                            r = run_arm(U, "vau_tau0_paired", dict(lr=1e-3, tau2=calib["tau0_sq"]), limit, f3=False)
                            cfg["ref_norms"] = [unr7(h) if a else 0.0 for h, a in zip(r["steps"]["Hnorm"],
                                                                                    r["steps"]["acc"])]
                            cfg["ref_norms"] += [0.0] * (U.T - len(cfg["ref_norms"]))
                        rr = run_arm(U, arm, cfg, limit, f3=False)
                        rr["kind"] = "g9_measure"
                        recs.append(rr)
        status = "ok"
    except Exception as e:
        recs = [dict(kind="unit_failure", family=family, cell=cell["id"], seed=seed,
                     error=f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}")]
        status = "unit_failed"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return dict(family=family, cell=cell["id"], seed=seed, status=status, cpu_s=time.process_time() - t0,
                n_runs=sum(1 for r in recs if "arm" in r), path=out_path)


def run_gauge(U, calib, limit):
    """G5 gauge subtest: inject dB = B K, dA = -K A (||K||_F = 0.3) into vau_nogate and vau at lr0, tau0; record
    the true R_test drift of every accepted step against the gate decision (pass = no accepted step over tau^2)."""
    out = []
    for arm in ("vau_nogate", "vau"):
        cfg = dict(lr=calib["lr0"], tau2=calib["tau0_sq"], ridge=1e-6)
        if arm == "vau_nogate":
            cfg.pop("tau2")
        out.append(dict(kind="gauge", family=U.family, cell=U.cell["id"], seed=U.seed, arm=arm,
                        gauge_check=gauge_replay(U, arm, cfg, limit)))
    return out


def gauge_replay(U, arm, cfg, limit):
    rng = copy.deepcopy(U.gauge_rng)
    records = []
    arm_obj = build_arm(U, arm, cfg, inject=lambda ms, th, d: d + U.model.gauge_step(th, 0.3, rng))
    ms = arm_obj.init_state(U.theta0, U.R1, copy.deepcopy(U.streams["sketch"]))
    th = arm_obj.initial_theta(ms)
    train = copy.deepcopy(U.streams["train"])
    x, f0 = U.ret["R_test"]["x"], U.ret["R_test"]["f0"]
    ptr, n_bad, n_acc = 0, 0, 0
    for t in range(1, U.T + 1):
        b = Batch(erase=_erasure_batch(U, train), gate=tuple(U.gate_pool[ptr:ptr + R_MAX]))
        th_new, info = arm_obj.step(ms, th, b)
        ptr += len(info.get("rho_hat", []) or [])
        d = np.sum((U.model.forward(x, th_new) - f0) ** 2, axis=1) / U.Z
        true = float(np.mean(d))
        if info["accepted"]:
            n_acc += 1
            if arm == "vau" and true > cfg["tau2"]:
                n_bad += 1
        records.append(r7(true))
        th = th_new
    return dict(true_drift=records, accepted=n_acc, accepted_over_budget=n_bad,
                pass_=(n_bad == 0) if arm == "vau" else None)


# ------------------------------------------------------------------------------------------------------------
# GD_B and GD_D diagnostics
# ------------------------------------------------------------------------------------------------------------
def run_gdb_unit(cell, seed, calib_g1):
    st = sd.streams("GD_B", seed)
    rng = st["fixture"]
    d = 8
    a = rng.normal(size=d)
    e = rng.normal(size=d); e /= np.linalg.norm(e)
    L = 1.0
    b = a + L * e
    Q, _ = np.linalg.qr(np.column_stack([e, rng.normal(size=(d, d - 1))]))
    Q[:, 0] = e
    J = Q @ np.diag([1.0] + [math.sqrt(cell["kappa"])] * (d - 1)) @ Q.T
    perp = rng.normal(size=d); perp -= (perp @ e) * e; perp /= np.linalg.norm(perp)
    delta = cell["eps_over_L"] * L * perp
    recs = []

    def r_of(v):
        w = v - a
        return float(np.linalg.norm(w - (w @ e) * e))

    def gem(v):
        dp, dn = np.linalg.norm(v - a), np.linalg.norm(v - b)
        h = dp - dn
        if h <= 0:
            return 0.0, np.zeros(d)
        return h, (v - a) / dp - ((v - b) / dn if dn > 0 else 0.0)

    def axial(v):
        alpha = (b - v) @ e
        h = L - 2.0 * alpha
        if h <= 0:
            return 0.0, np.zeros(d)
        return h, 2.0 * e

    for loss_name, fn in (("gem", gem), ("axial", axial)):
        for opt in ("sgd", "adamw"):
            th = np.zeros(d)
            state = aw.init_state(d)
            hp = aw.AdamWHP(lr=calib_g1["lr0"])
            steps = 20000 if opt == "sgd" else 60
            v = b + delta
            for t in range(steps):
                v = b + delta + J @ th
                val, gv = fn(v)
                if val == 0.0:
                    break
                g = J.T @ gv
                if opt == "sgd":
                    th = th - 1e-3 * L * g
                else:
                    d0, state, _ = aw.peek(state, g, th, hp)
                    th = th + d0
            v = b + delta + J @ th
            recs.append(dict(kind="gdb", family="GD_B", cell=cell["id"], seed=seed, loss=loss_name, opt=opt,
                             r_end=r7(r_of(v) / L), steps=t + 1, loss_end=r7(fn(v)[0])))
    return recs


def run_gdd_unit(U, calib, limit):
    """GD_D: adamw and vau (lr0, tau0) under prefix (T=23) and uniform 11-of-28 (T=18) schemes."""
    out = []
    for arm in ("adamw", "vau"):
        for scheme, T in (("prefix", 23), ("uniform", 18)):
            U.T, U.ck = T, [0, T]
            cfg = dict(lr=calib["lr0"], scheme=scheme)
            if arm == "vau":
                cfg["tau2"] = calib["tau0_sq"]
            r = run_arm(U, arm, cfg, limit, f3=False)
            r["kind"] = "gdd"
            r["scheme"] = scheme
            if scheme == "prefix" and r["status"] == "ok":
                r["rho_late"] = r7(rho_late(U, arm, cfg, limit))
            out.append(r)
    return out


def rho_late(U, arm, cfg, limit):
    """Re-train the prefix student (same streams) and measure ||v_student - v_base|| along its own rollouts."""
    arm_obj = build_arm(U, arm, cfg)
    ms = arm_obj.init_state(U.theta0, U.R1, copy.deepcopy(U.streams["sketch"]))
    th = arm_obj.initial_theta(ms)
    train = copy.deepcopy(U.streams["train"])
    ptr = 0
    for _ in range(U.T):
        b = Batch(erase=_erasure_batch(U, train), gate=tuple(U.gate_pool[ptr:ptr + R_MAX]))
        th, info = arm_obj.step(ms, th, b)
        ptr += len(info.get("rho_hat", []) or [])
    T = U.tgt["target_test"]
    z, p = T["z"][:128], T["p"][:128]
    sig = U.task.sigmas
    x, gaps = z, []
    for k in range(28):
        inp = FlowInput(x, np.full(len(p), sig[k]), p)
        vs, vb = U.model.forward(inp, th), U.model.forward(inp, None)
        gaps.append(float(np.mean(np.linalg.norm(vs - vb, axis=1))))
        x = x + (sig[k + 1] - sig[k]) * vs.reshape(x.shape)
    early, late = np.mean(gaps[:11]), np.mean(gaps[11:])
    return late / early if early > 0 else float("nan")


# ------------------------------------------------------------------------------------------------------------
# statistics (unit-tested in code/tests/test_runner_stats.py)
# ------------------------------------------------------------------------------------------------------------
def crossing(ts, E, e):
    """First checkpoint pair (i-1, i) with E[i-1] < e <= E[i]; returns (i, w) with w = (e-E[i-1])/(E[i]-E[i-1]),
    (0, 1.0) if E[0] >= e, or None if never reached. NaN/None entries end the search (failed run)."""
    for i, v in enumerate(E):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return None
        if v >= e:
            if i == 0:
                return (0, 1.0)
            prev = E[i - 1]
            w = (e - prev) / (v - prev) if v != prev else 1.0
            return (i, float(w))
    return None


def interp_at(series, cr):
    i, w = cr
    if i == 0:
        return float(series[0])
    a, b = float(series[i - 1]), float(series[i])
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return a + w * (b - a)


METRICS = ("rho_sel", "rho_test", "rho_max", "rho_q99", "rho_rare", "rho_shift", "E_test", "Eax_test", "FE",
           "rho_vsel", "rho_vtest")


def matched(run, e, observable="E_sel"):
    """Interpolated metrics of one run at the first crossing of `observable` = e, or None."""
    if run.get("status") != "ok":
        tr = run.get("traj", {})
    else:
        tr = run["traj"]
    if not tr or observable not in tr:
        return None
    E = [unr7(v) for v in tr[observable]]
    cr = crossing(tr["t"], E, e)
    if cr is None:
        return None
    out = {}
    for k in METRICS:
        if k in tr:
            out[k] = interp_at([unr7(v) for v in tr[k]], cr)
    out["t_cross"] = tr["t"][cr[0] - 1] + cr[1] * (tr["t"][cr[0]] - tr["t"][cr[0] - 1]) if cr[0] > 0 else tr["t"][0]
    return out


def select_config(runs, e, observable="E_sel"):
    """Config minimising interpolated rho_sel at the crossing; None if no config reaches e."""
    best = None
    for r in runs:
        m = matched(r, e, observable)
        if m is None or not math.isfinite(m.get("rho_sel", math.inf)):
            continue
        if best is None or m["rho_sel"] < best[0]["rho_sel"]:
            best = (m, r)
    return best


def log_ratio(x_vau, x_ctl):
    """log(x_vau/x_ctl) with the protocol's tie/inf rule; None when both are unreached."""
    if x_vau is None and x_ctl is None:
        return None
    if x_vau is None:
        return LOG100
    if x_ctl is None:
        return -LOG100
    return math.log(max(x_vau, RHO_FLOOR) / max(x_ctl, RHO_FLOOR))


def perm_test_one_sided(x):
    """Exact sign-flip test of H1: mean(x) < 0. p = #{s: mean(s*x) <= mean(x)} / 2^n."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return 1.0
    obs = x.mean()
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    means = (signs * x).mean(axis=1)
    return float(np.mean(means <= obs + 1e-15))


def bootstrap_gmr_ci(x, n_boot=10000, seed=424242):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    g = np.exp(x[idx].mean(axis=1))
    return (float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975)))


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def binom_sf(k, n, p):
    """P(X >= k), X ~ Bin(n, p)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return math.nan
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fallback_level(cell_pairs, n_seeds):
    """cell_pairs: {e: [(vau_reached, best_reached) per seed]} -> first level in 0.8, 0.5, 0.2 at which NOT
    (both fail in >= half of the seeds)."""
    for e in (0.8, 0.5, 0.2):
        both_fail = sum(1 for v, c in cell_pairs[e] if not v and not c)
        if both_fail < math.ceil(n_seeds / 2):
            return e
    return 0.2


def family_stats(per_seed, n_seeds_total, thr_wins):
    """per_seed: {seed: log ratio} -> GMR, p, CI, wins."""
    x = [v for v in per_seed.values() if v is not None]
    if not x:
        return dict(n=0, GMR=None, p=None, ci=None, wins=0)
    return dict(n=len(x), GMR=float(np.exp(np.mean(x))), p=perm_test_one_sided(x), ci=bootstrap_gmr_ci(x),
                wins=int(sum(1 for v in x if v < 0)), wins_needed=thr_wins)


# ------------------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------------------
def load_family(outdir, family):
    recs = []
    for p in sorted(glob.glob(os.path.join(outdir, "raw", family, "*.jsonl.gz"))):
        with gzip.open(p, "rt") as f:
            recs += [json.loads(line) for line in f]
    return recs


def cell_matched(recs, e_levels=(0.8, 0.5, 0.2), extra_controls=None, controls=CONTROLS, observable="E_sel"):
    """recs of one (family, cell): returns {seed: {e: {arm: matched-or-None, 'best': (arm, m)}}}."""
    by = {}
    for r in recs:
        if r.get("kind") in ("unit", "gauge", "g9_measure", "gdd", "gdb", "unit_failure") or "arm" not in r:
            continue
        if r["status"] == "NOT_APPLICABLE":
            continue
        by.setdefault(r["seed"], {}).setdefault(r["arm"], []).append(r)
    if extra_controls:
        for seed, arms in extra_controls.items():
            for a, rs in arms.items():
                by.setdefault(seed, {}).setdefault(a, []).extend(rs)
    out = {}
    for seed, arms in by.items():
        out[seed] = {}
        for e in e_levels:
            res = {}
            for a, rs in arms.items():
                if a == "vau_tau0_paired":
                    continue
                sel = select_config(rs, e, observable)
                res[a] = None if sel is None else sel[0]
            best = None
            for a in controls + (["penalty_oracle"] if "penalty_oracle" in arms and "penalty_oracle" in controls
                                 else []):
                m = res.get(a)
                if m is not None and (best is None or m["rho_sel"] < best[1]["rho_sel"]):
                    best = (a, m)
            res["_best"] = best
            res["_crash"] = {a: any(r["status"] == "failed" for r in rs) for a, rs in arms.items()}
            out[seed][e] = res
    return out


def _metric(m, key):
    return None if m is None else m.get(key)


def compare(cm, levels, key="rho_test", ctl="_best", ctl_arms=None):
    """Per-seed log ratio vau vs best control (or vs the best of ctl_arms) at each seed's cell level."""
    out = {}
    for seed, byl in cm.items():
        e = levels
        res = byl[e]
        if ctl_arms is not None:
            cands = [(a, res.get(a)) for a in ctl_arms if res.get(a) is not None]
            best = min(cands, key=lambda am: am[1]["rho_sel"]) if cands else None
        else:
            best = res["_best"]
        out[seed] = log_ratio(_metric(res.get("vau"), key), None if best is None else best[1].get(key))
    return out


def analyse(outdir, families, n_seeds_expected, phase):
    S = dict(phase=phase, decisions=DECISIONS, families={}, coverage={}, label=(
        "DEVELOPMENT ONLY - not a verdict" if phase == "dev" else "VALIDATION"))
    cell_cache = {}
    seeds_frac = lambda k12: k12 / 12.0  # noqa: E731
    all_f3 = []
    for fam in families:
        recs = load_family(outdir, fam)
        if not recs:
            S["families"][fam] = dict(status="UNRUN")
            continue
        fs = dict(status="RUN", cells={})
        cells = {c["id"]: c for c in family_cells(fam)}
        by_cell = {}
        for r in recs:
            by_cell.setdefault(r.get("cell"), []).append(r)
        if fam == "GD_B":
            S["families"][fam] = analyse_gdb(recs)
            continue
        if fam == "GD_D":
            S["families"][fam] = analyse_gdd(recs)
            continue
        for cid, rs in by_cell.items():
            units = [r for r in rs if r.get("kind") == "unit"]
            cov = [u["coverage"] for u in units]
            if cov:
                medC = float(np.median([c["C_probe"] for c in cov]))
                medk = float(np.median([c["k_eff"] for c in cov]))
                medr = float(np.median([c["eff_rank_99"] for c in cov]))
                cls = "COVERED" if (medC >= 0.9 and medk >= medr) else ("PARTIAL" if medC >= 0.5 else "UNCOVERED")
                S["coverage"][f"{fam}/{cid}"] = dict(C_probe=medC, C_probe_D=float(np.median([c["C_probe_D"]
                                                                                             for c in cov])),
                                                     eff_rank_99=medr, k=medk, cls=cls,
                                                     C_feat=(float(np.median([c["C_feat"] for c in cov]))
                                                             if "C_feat" in cov[0] else None),
                                                     disjoint_dups=int(sum(u["disjoint_dups"] for u in units)))
            extra = None
            if fam == "G7":
                src = cell_cache.get(("G5", cells[cid]["reuse_cell"]), {}).get("raw", [])
                extra = {}
                for r in src:
                    if r.get("arm") in ARMS16 and r["arm"] not in PROBE_ARMS and r.get("kind") is None:
                        extra.setdefault(r["seed"], {}).setdefault(r["arm"], []).append(r)
            cm = cell_matched(rs, extra_controls=extra)
            cma = cell_matched(rs, extra_controls=extra, observable="Eax_sel")
            cmo = (cell_matched(rs, controls=CONTROLS + ["penalty_oracle"], extra_controls=extra)
                   if cells[cid].get("oracle") else None)
            for r in rs:
                if "f3" in r:
                    all_f3.append((fam, r["f3"], r["arm"], r["status"]))
            n_seeds = len(cm)
            pairs = {e: [(cm[s][e].get("vau") is not None, cm[s][e]["_best"] is not None) for s in cm]
                     for e in (0.8, 0.5, 0.2)}
            lvl = fallback_level(pairs, n_seeds) if n_seeds else 0.8
            cell_cache[(fam, cid)] = dict(cm=cm, cma=cma, cmo=cmo, level=lvl, raw=rs if fam == "G5" else [],
                                          cell=cells[cid])
            fs["cells"][cid] = dict(level=lvl, n_seeds=n_seeds, cls=S["coverage"].get(f"{fam}/{cid}", {}).get("cls"),
                                    table=_cell_table(cm, lvl))
            if fam == "G9":
                fs["mechanism"] = analyse_g9(rs)
            if fam == "G5" and cells[cid].get("gauge"):
                fs["gauge"] = [dict(arm=r["arm"], seed=r["seed"], **r.get("gauge_check", {}))
                               for r in rs if r.get("kind") == "gauge"]
        S["families"][fam] = fs
    # G6 derived from G5 gain-1 and G1 d=2 sketch cells
    g6_src = [("G5", "gain1_generic"), ("G5", "gain1_covered"), ("G1", "d2_sketch_generic"),
              ("G1", "d2_sketch_covered")]
    if all(k in cell_cache for k in g6_src[:2]) or any(k in cell_cache for k in g6_src):
        S["families"]["G6"] = dict(status="RUN", sources=[f"{a}/{b}" for a, b in g6_src if (a, b) in cell_cache])
        for a, b in g6_src:
            if (a, b) in cell_cache:
                cell_cache[("G6", f"{a}_{b}")] = dict(cell_cache[(a, b)], g6=True)
                S["coverage"][f"G6/{a}_{b}"] = S["coverage"].get(f"{a}/{b}")
    S["F3"] = analyse_f3(all_f3)
    S["stats"] = family_level(S, cell_cache, phase)
    S["verdict"] = verdict(S)
    return S


def _cell_table(cm, lvl):
    tab = {}
    for seed, byl in cm.items():
        res = byl[lvl]
        tab[str(seed)] = {a: (None if m is None else {k: r7(v) for k, v in m.items()})
                          for a, m in res.items() if not a.startswith("_")}
        b = res["_best"]
        tab[str(seed)]["_best"] = None if b is None else b[0]
    return tab


def analyse_f3(all_f3):
    runs = [x for x in all_f3 if x[0] in ("G1", "G2", "G3", "G4", "G5", "G7", "G9") and x[1]["n_acc"] > 0]
    n = len(runs)
    k = sum(1 for x in runs if x[1]["viol"])
    p = binom_sf(k, n, DELTA_GATE) if n else 1.0
    return dict(n_runs=n, n_violating=k, frac=(k / n if n else None), binom_p=p,
                triggered=bool(n and k / n > DELTA_GATE and p < 0.01),
                unclipped_frac=(sum(1 for x in runs if x[1]["viol_unclipped"]) / n if n else None),
                rare_frac=(sum(1 for x in runs if x[1]["viol_rare"]) / n if n else None))


def analyse_g9(rs):
    meas = [r for r in rs if r.get("kind") == "g9_measure" and r.get("mech")]
    out = {}
    for r in meas:
        steps = r["mech"][:8]
        out.setdefault(r["arm"], []).append(dict(
            delta_1_8=float(np.mean([unr7(s["delta"]) for s in steps if s.get("delta") is not None] or [math.nan])),
            gamma_1_8=float(np.nanmean([unr7(s["gamma"]) if s.get("gamma") is not None else math.nan
                                        for s in steps] or [math.nan])),
            DA_DB_step1=steps[0].get("DA_over_DB") if steps else None))
    summ = {a: dict(delta_1_8=float(np.median([v["delta_1_8"] for v in vs])),
                    gamma_1_8=float(np.nanmedian([v["gamma_1_8"] for v in vs])),
                    DA_DB_step1=vs[0]["DA_DB_step1"]) for a, vs in out.items()}
    pe = summ.get("post_euclid_reject", {}).get("delta_1_8")
    for a in summ:
        summ[a]["ratio_vs_post_euclid_reject"] = (summ[a]["delta_1_8"] / pe) if pe else None
    return summ


def analyse_gdb(recs):
    g = {}
    for r in recs:
        if r.get("kind") != "gdb":
            continue
        g.setdefault((r["cell"], r["opt"], r["seed"]), {})[r["loss"]] = unr7(r["r_end"])
    cells = {}
    for (cell, opt, seed), d in g.items():
        if "gem" in d and "axial" in d:
            cells.setdefault((cell, opt), []).append(d["gem"] / max(d["axial"], 1e-300))
    out = {f"{c}|{o}": dict(median_ratio=float(np.median(v)), n=len(v)) for (c, o), v in cells.items()}
    key = [k for k in out if ("eps1e-06" in k or "eps0.0001" in k) and "kappa1|" in k and k.endswith("sgd")]
    fals = any(out[k]["median_ratio"] < 2 for k in key)
    return dict(cells=out, P2_falsified=fals, P2_cells=key, criterion="diagnostic only")


def analyse_gdd(recs):
    by = {}
    for r in recs:
        if r.get("kind") == "gdd" and r["status"] == "ok":
            by.setdefault((r["seed"], r["arm"]), {})[r["scheme"]] = r
    xs, ys = [], []
    rows = []
    for (seed, arm), d in by.items():
        if "prefix" in d and "uniform" in d:
            ep = unr7(d["prefix"]["traj"]["E_test"][-1]); eu = unr7(d["uniform"]["traj"]["E_test"][-1])
            rl = unr7(d["prefix"].get("rho_late", math.nan))
            xs.append(ep - eu); ys.append(1 - rl)
            rows.append(dict(seed=seed, arm=arm, E_prefix=ep, E_uniform=eu, rho_late=rl))
    rho = spearman(xs, ys)
    return dict(rows=rows, spearman=rho, P_D5_prediction=(rho >= 0.5) if math.isfinite(rho) else None,
                P_D5_falsified=(rho < 0.2) if math.isfinite(rho) else None, criterion="diagnostic only")


ADV = ["G1", "G2", "G5", "G6", "G7", "G8"]
SAFETY = {"G1": "d0", "G2": "phi0", "G3": "all", "G4": "all", "G6": "rho_rare", "G9": "all"}


def _cells_of(cell_cache, fam, pred):
    return [(cid, v) for (f, cid), v in cell_cache.items() if f == fam and pred(cid, v)]


def _per_seed_mean(cells, key, ctl_arms=None, axial=False, cm_key="cm", invert=False):
    acc = {}
    for cid, v in cells:
        cm = v[cm_key]
        if cm is None:
            continue
        lr = compare(cm, v["level"], key=key, ctl_arms=ctl_arms)
        for s, x in lr.items():
            if x is not None:
                acc.setdefault(s, []).append(-x if invert else x)
    return {s: float(np.mean(v)) for s, v in acc.items()}


def family_level(S, cell_cache, phase):
    out = {}
    cov = S["coverage"]

    def covered(fam, cid):
        src = f"{fam}/{cid}"
        return (cov.get(src) or {}).get("cls") == "COVERED"

    def primary(fam, cid, v):
        c = v["cell"]
        if fam == "G6":
            return True
        return bool(c.get("primary"))

    pvals = {}
    for fam in ADV:
        cells = _cells_of(cell_cache, fam, lambda cid, v: primary(fam, cid, v))
        if not cells:
            out[fam] = dict(status="UNRUN")
            continue
        cov_cells = [(cid, v) for cid, v in cells if covered(fam, cid)]
        key = "rho_shift" if fam == "G6" else "rho_test"
        if fam == "G8":
            key = "rho_test"
        res = dict(primary_cells=[c for c, _ in cells], covered_cells=[c for c, _ in cov_cells])
        if not cov_cells:
            res["status"] = "NOT_TESTABLE"
            res["generic_report"] = family_stats(_per_seed_mean(cells, key), 12, 9)
            out[fam] = res
            continue
        n = max(len(v["cm"]) for _, v in cov_cells)
        q1 = family_stats(_per_seed_mean(cov_cells, key), n, math.ceil(9 * n / 12))
        res["Q1_stats"] = q1
        pvals[fam] = q1["p"] if q1["p"] is not None else 1.0
        q2 = {k: family_stats(_per_seed_mean(cov_cells, k), n, 0)["GMR"] for k in ("rho_max", "rho_q99", "rho_rare")}
        res["Q2"] = q2
        gaps = []
        for cid, v in cov_cells:
            for s, byl in v["cm"].items():
                r = byl[v["level"]]
                if r.get("vau") is not None and r["_best"] is not None:
                    gaps.append(abs(r["vau"]["E_test"] - r["_best"][1]["E_test"]))
        q3_ax = family_stats(_per_seed_mean(cov_cells, key, cm_key="cma"), n, 0)["GMR"]
        res["Q3"] = dict(median_abs_dE=float(np.median(gaps)) if gaps else None, GMR_axial=q3_ax)
        crashes_vau = len({s for _, v in cov_cells for s, byl in v["cm"].items()
                           if byl[v["level"]]["_crash"].get("vau")})
        crashes_ctl = {}
        for _, v in cov_cells:
            for s, byl in v["cm"].items():
                for a, c in byl[v["level"]]["_crash"].items():
                    if c and a != "vau":
                        crashes_ctl.setdefault(a, set()).add(s)
        res["Q4"] = dict(vau_crash_seeds=crashes_vau, max_control_crash_seeds=max([len(x) for x in crashes_ctl.values()]
                                                                                  or [0]))
        # F2 / F4 / F5 inputs
        res["F2"] = family_stats(_per_seed_mean(cov_cells, key, ctl_arms=PENALTY_VARIANTS, invert=True), n,
                                 math.ceil(8 * n / 12))
        res["F4_GMR"] = family_stats(_per_seed_mean(cov_cells, key, ctl_arms=["penalty+gate", "penalty_earlystop"]),
                                     n, 0)["GMR"]
        res["F5_GMR"] = family_stats(_per_seed_mean(cov_cells, key, ctl_arms=NA_VARIANTS), n, 0)["GMR"]
        res["compute"] = compute_rule(cov_cells)
        res["n_seeds"] = n
        out[fam] = res
    adj = holm(pvals)
    for fam in ADV:
        r = out.get(fam, {})
        if "Q1_stats" not in r:
            continue
        q1, n = r["Q1_stats"], r["n_seeds"]
        r["Q1_holm_p"] = adj[fam]
        Q1 = (q1["GMR"] is not None and q1["GMR"] <= 0.90 and adj[fam] < 0.05 and q1["ci"][1] < 1.0
              and q1["wins"] >= math.ceil(9 * n / 12))
        Q2 = all(v is None or v <= 1.05 for v in r["Q2"].values())
        Q3 = (r["Q3"]["median_abs_dE"] is not None and r["Q3"]["median_abs_dE"] <= 0.03
              and r["Q3"]["GMR_axial"] is not None and r["Q3"]["GMR_axial"] <= 0.95)
        Q4 = r["Q4"]["vau_crash_seeds"] <= 1 and r["Q4"]["max_control_crash_seeds"] <= 1 and not S["F3"]["triggered"]
        r.update(Q1=Q1, Q2=Q2, Q3=Q3, Q4=Q4, qualifies=bool(Q1 and Q2 and Q3 and Q4), status="TESTABLE")
        f2 = r["F2"]
        r["F2_triggered"] = bool(f2["GMR"] is not None and f2["GMR"] <= 0.90 and f2["wins"] >= math.ceil(8 * n / 12))
    # F5 / F4 fall back to generic cells when no COVERED cell exists in G5 / G8
    for fam in ("G5", "G8"):
        r = out.get(fam, {})
        if r.get("status") == "NOT_TESTABLE":
            cells = _cells_of(cell_cache, fam, lambda cid, v: primary(fam, cid, v))
            r["F5_GMR"] = family_stats(_per_seed_mean(cells, "rho_test", ctl_arms=NA_VARIANTS), 12, 0)["GMR"]
            r["F4_GMR"] = family_stats(_per_seed_mean(cells, "rho_test", ctl_arms=["penalty+gate",
                                                                                   "penalty_earlystop"]), 12, 0)["GMR"]
            r["compute"] = compute_rule(cells)
    # safety
    for fam, sel in SAFETY.items():
        if fam == "G1":
            cells = _cells_of(cell_cache, fam, lambda cid, v: v["cell"].get("safety"))
        elif fam == "G2":
            cells = _cells_of(cell_cache, fam, lambda cid, v: v["cell"].get("safety"))
        elif fam == "G6":
            cells = _cells_of(cell_cache, "G6", lambda cid, v: True)
        else:
            cells = _cells_of(cell_cache, fam, lambda cid, v: True)
        if not cells:
            out[f"safety_{fam}"] = dict(status="UNRUN")
            continue
        key = "rho_rare" if fam == "G6" else "rho_test"
        n = max(len(v["cm"]) for _, v in cells)
        st = family_stats(_per_seed_mean(cells, key), n, 0)
        st_max = family_stats(_per_seed_mean(cells, "rho_max"), n, 0) if fam == "G6" else None
        crash = any(byl[v["level"]]["_crash"].get("vau") for _, v in cells for byl in v["cm"].values())
        viol = any(x for x in [])
        ok = (st["GMR"] is not None and st["GMR"] <= 1.10 and st["ci"][1] <= 1.25 and not crash
              and not S["F3"]["triggered"] and not viol)
        if fam == "G6" and st_max and st_max["GMR"] is not None:
            ok = ok and st_max["GMR"] <= 1.10
        out[f"safety_{fam}"] = dict(stats=st, rho_max_stats=st_max, vau_crash=crash, passes=bool(ok))
    # F1
    out["F1"] = falsifier_f1(cell_cache)
    # penalty oracle flip (G1)
    out["oracle_flip"] = oracle_flip(cell_cache, out)
    return out


def compute_rule(cells):
    vb, va, per_arm = [], [], {}
    for _, v in cells:
        for s, byl in v["cm"].items():
            r = byl[v["level"]]
            fv = _metric(r.get("vau"), "FE")
            fb = None if r["_best"] is None else r["_best"][1].get("FE")
            fa = _metric(r.get("adamw"), "FE")
            if fv is not None and fb:
                vb.append(fv / fb)
            if fv is not None and fa:
                va.append(fv / fa)
            for a, m in r.items():
                if not a.startswith("_") and m is not None and "FE" in m:
                    per_arm.setdefault(a, []).append(m["FE"])
    return dict(vau_over_best_median=float(np.median(vb)) if vb else None,
                vau_over_adamw_median=float(np.median(va)) if va else None,
                FE_median_per_arm={a: float(np.median(x)) for a, x in per_arm.items()})


def falsifier_f1(cell_cache):
    res = {}
    for d in (2, 3):
        for pop in ("covered", "generic"):
            key = ("G1", f"d{d}_exact_{pop}")
            if key not in cell_cache:
                continue
            rs = cell_cache[key]
            ratios, ratios_pe = [], []
            raw_seeds = {}
            for s in rs["cm"]:
                raw_seeds[s] = True
            res[f"d{d}_{pop}"] = dict(ratios=f1_ratios(rs, "post_euclid_reject"),
                                      ratios_vs_post_euclid=f1_ratios(rs, "post_euclid"))
    trig = False
    for d in (2, 3):
        r = res.get(f"d{d}_covered")
        if r:
            vals = [v for v in r["ratios"].values() if v is not None]
            n = len(vals)
            if n and sum(1 for v in vals if v < 1.2) >= math.ceil(8 * n / 12):
                trig = True
    res["triggered"] = trig
    return res


F1_RUNS = {}


def f1_ratios(cell_entry, comparator):
    out = {}
    for seed, pair in cell_entry.get("f1", {}).items():
        a, b = pair.get("vau_nogate"), pair.get(comparator)
        out[seed] = (a / b) if (a is not None and b not in (None, 0)) else None
    return out


def oracle_flip(cell_cache, out):
    cells = [(cid, v) for (f, cid), v in cell_cache.items() if f == "G1" and v.get("cmo") is not None]
    if not cells or "G1" not in out or "Q1_stats" not in out.get("G1", {}):
        return dict(evaluated=False)
    alt = []
    for cid, v in cells:
        alt.append((cid, dict(v, cm=v["cmo"])))
    q1 = family_stats(_per_seed_mean(alt, "rho_test"), 12, 9)
    base_q1 = out["G1"]["Q1"]
    n = out["G1"]["n_seeds"]
    alt_q1 = (q1["GMR"] is not None and q1["GMR"] <= 0.90 and q1["p"] < 0.05 and q1["ci"][1] < 1.0
              and q1["wins"] >= math.ceil(9 * n / 12))
    return dict(evaluated=True, oracle_GMR=q1["GMR"], flip=bool(base_q1 != alt_q1))


def verdict(S):
    st = S["stats"]
    reasons_no, ready_fail = [], []
    if st["F1"]["triggered"]:
        reasons_no.append("F1")
    for fam in ("G1", "G8"):
        if st.get(fam, {}).get("F2_triggered"):
            reasons_no.append(f"F2 in {fam}")
    if S["F3"]["triggered"]:
        reasons_no.append("F3")
    g5, g8 = st.get("G5", {}), st.get("G8", {})
    if g5.get("F5_GMR") is not None and g8.get("F5_GMR") is not None and g5["F5_GMR"] >= 0.95 and g8["F5_GMR"] >= 0.95:
        reasons_no.append("F5")
    testable = [f for f in ADV if st.get(f, {}).get("status") == "TESTABLE"]
    if not any(st[f]["Q1_stats"]["GMR"] is not None and st[f]["Q1_stats"]["GMR"] <= 0.95 for f in testable):
        reasons_no.append("no testable advantage family with GMR <= 0.95")
    safety_fail = [f for f in SAFETY if st.get(f"safety_{f}", {}).get("passes") is False]
    if len(safety_fail) >= 2:
        reasons_no.append(f"safety failures {safety_fail}")
    if reasons_no:
        return dict(verdict="NO_SUPPORTED_CANDIDATE", reasons=reasons_no, label=S["label"])
    if st["oracle_flip"].get("flip"):
        return dict(verdict="HYPOTHESIS_UNRESOLVED", reasons=["penalty_oracle flip"], label=S["label"])
    for fam in ("G1", "G8"):
        if not st.get(fam, {}).get("qualifies"):
            ready_fail.append(f"{fam} not testable/qualifying")
    if sum(1 for f in ("G2", "G5", "G6", "G7") if st.get(f, {}).get("qualifies")) < 2:
        ready_fail.append("< 2 of G2/G5/G6/G7 qualify")
    if safety_fail or any(st.get(f"safety_{f}", {}).get("status") == "UNRUN" for f in SAFETY):
        ready_fail.append("safety")
    for fam in ADV:
        if st.get(fam, {}).get("F2_triggered"):
            ready_fail.append(f"F2 {fam}")
    if any((st.get(f, {}).get("F4_GMR") or 0) >= 1.0 for f in ("G5", "G8")):
        ready_fail.append("F4")
    for fam in ("G5", "G8"):
        c = st.get(fam, {}).get("compute", {})
        if c.get("vau_over_best_median") is None or c["vau_over_best_median"] > 4.0:
            ready_fail.append(f"compute rule {fam}")
    if any(S["families"].get(f, {}).get("status") == "UNRUN" for f in ["G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9",
                                                                     "GD_B", "GD_D"]):
        ready_fail.append("UNRUN family")
    if ready_fail:
        return dict(verdict="HYPOTHESIS_UNRESOLVED", reasons=ready_fail, label=S["label"])
    return dict(verdict="READY_FOR_SMALL_REAL_MODEL_TEST", reasons=[], label=S["label"],
                condition="FLUX C_probe >= 0.90 measured before training (REAL_MODEL_PROTOCOL C9)")


# ------------------------------------------------------------------------------------------------------------
# F1 needs P = E_test at step 10 for vau_nogate / post_euclid(_reject) at lr0, ridge 1e-6, wd 0.01 (ungated)
# ------------------------------------------------------------------------------------------------------------
def attach_f1(cell_cache_entry, rs, lr0):
    out = {}
    for r in rs:
        if r.get("arm") not in ("vau_nogate", "post_euclid_reject", "post_euclid") or r["status"] != "ok":
            continue
        c = r["cfg"]
        if abs(unr7(c["lr"]) - lr0) > 1e-9 * lr0 or abs(unr7(c.get("ridge", 1e-6)) - 1e-6) > 1e-12:
            continue
        tr = r["traj"]
        out.setdefault(r["seed"], {})[r["arm"]] = tr
    res = {}
    for seed, d in out.items():
        res[seed] = {}
        stop = 10
        for a, tr in d.items():
            E = [unr7(v) for v in tr["E_test"]]
            for i, t in enumerate(tr["t"]):
                if E[i] >= 0.95:
                    stop = min(stop, t)
                    break
        for a, tr in d.items():
            ts = tr["t"]
            res[seed][a] = unr7(tr["E_test"][ts.index(stop)]) if stop in ts else None
    cell_cache_entry["f1"] = res


# ------------------------------------------------------------------------------------------------------------
# calibration (dev seeds 0..7 only)
# ------------------------------------------------------------------------------------------------------------
CALIB_CELLS = {"G1": ("G1", "d2_sketch_generic"), "G2": ("G2", "k1e+02_phipi8"), "G3": ("G3", "cospsi0.1"),
               "G4": ("G4", "a_L0.01_epsnat"), "G4b": ("G4", "b_L0.0001_epsnat"), "G5": ("G5", "gain1_generic"),
               "G9": ("G9", "zeroinit"), "G8": ("G8", "generic")}


def calib_job(spec):
    key, seed, stage, params = spec["key"], spec["seed"], spec["stage"], spec["params"]
    fam, cid = CALIB_CELLS[key]
    cell = [c for c in family_cells(fam) if c["id"] == cid][0]
    U = build_unit(fam, cell, seed, ())
    out = []
    obs = "rho_vsel" if U.model_kind == "c" else "rho_sel"
    if stage == "lr":
        for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            r = run_arm(U, "adamw", dict(lr=lr, wd=0.01), 600, f3=False)
            out.append(dict(key=key, seed=seed, stage="lr", lr=lr, E=r["traj"]["E_sel"], t=r["traj"]["t"],
                            rho=r["traj"][obs], status=r["status"], cpu=r["cpu_s"]))
    else:
        for lam in 10.0 ** np.arange(-3, 4):
            r = run_arm(U, "penalty", dict(lr=params["lr0"], lam=float(lam)), 600, f3=False)
            out.append(dict(key=key, seed=seed, stage="lam", lam=float(lam), E=r["traj"]["E_sel"], t=r["traj"]["t"],
                            rho=r["traj"][obs], status=r["status"], cpu=r["cpu_s"]))
    return out


def _cross_t(rec, e):
    E = [unr7(v) for v in rec["E"]]
    cr = crossing(rec["t"], E, e)
    if cr is None:
        return math.inf, None
    t = rec["t"][cr[0] - 1] + cr[1] * (rec["t"][cr[0]] - rec["t"][cr[0] - 1]) if cr[0] > 0 else rec["t"][0]
    return t, interp_at([unr7(v) for v in rec["rho"]], cr)


def calibrate(pool, keys, seeds=range(8)):
    res = {}
    lr_jobs = [dict(key=k, seed=s, stage="lr", params={}) for k in keys for s in seeds]
    lr_out = [x for chunk in pool.map(calib_job, lr_jobs) for x in chunk]
    for k in keys:
        rows = [r for r in lr_out if r["key"] == k]
        level = None
        for e in (0.8, 0.5, 0.2):
            med = {lr: float(np.median([_cross_t(r, e)[0] for r in rows if r["lr"] == lr]))
                   for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)}
            if any(math.isfinite(v) for v in med.values()):
                level = e
                break
        if level is None:
            level = 0.2
            lr0 = max((1e-4, 3e-4, 1e-3, 3e-3, 1e-2), key=lambda lr: np.median(
                [unr7(r["E"][-1]) for r in rows if r["lr"] == lr]))
        else:
            lr0 = min(med, key=lambda lr: (abs(med[lr] - 23), -lr))
        rho_ad = float(np.median([_cross_t(r, level)[1] if _cross_t(r, level)[1] is not None else math.inf
                                  for r in rows if r["lr"] == lr0]))
        res[k] = dict(lr0=lr0, level=level, adamw_rho_at_level=rho_ad, t_cross_median=med if level else None)
    lam_jobs = [dict(key=k, seed=s, stage="lam", params=dict(lr0=res[k]["lr0"])) for k in keys for s in seeds]
    lam_out = [x for chunk in pool.map(calib_job, lam_jobs) for x in chunk]
    for k in keys:
        rows = [r for r in lam_out if r["key"] == k]
        target = res[k]["adamw_rho_at_level"] / 2
        best, bestd = None, math.inf
        lam_med = {}
        for lam in 10.0 ** np.arange(-3, 4):
            vals = [_cross_t(r, res[k]["level"])[1] for r in rows if abs(r["lam"] - lam) < 1e-12 * lam]
            med = float(np.median([v if v is not None else math.inf for v in vals]))
            lam_med[float(lam)] = med
            dist = abs(math.log(max(med, 1e-300) / max(target, 1e-300))) if math.isfinite(med) and target > 0 \
                else math.inf
            if dist < bestd:
                best, bestd = float(lam), dist
        res[k]["lambda0"] = best if best is not None else 1.0
        res[k]["penalty_rho_median"] = lam_med
        tau = 0.25 * res[k]["adamw_rho_at_level"]
        res[k]["tau0_sq"] = tau if math.isfinite(tau) and tau > 0 else 1e-3
        res[k]["cpu_s"] = float(sum(r["cpu"] for r in rows) + sum(r["cpu"] for r in lr_out if r["key"] == k))
    return res


# ------------------------------------------------------------------------------------------------------------
# integrity (I1-I15)
# ------------------------------------------------------------------------------------------------------------
I_MAP = {
    "I1": ["test_fd.py::"], "I2": ["test_I2_"], "I3": ["test_I3_"], "I4": ["test_I4_"], "I5": ["test_I5_"],
    "I6": ["test_I6_"], "I7": ["test_adamw.py::test_matches_reference_loop", "test_adamw.py::test_peek"],
    "I8": ["test_I8_", "test_sketch_stream_shared"], "I9": ["test_rollback_is_bit_exact"], "I10": ["test_I10_"],
    "I11": ["test_I11_"], "I12": ["test_I12_", "test_exact_v_equals_b"], "I13": ["test_adam_counter"],
    "I14": ["test_null_step", "test_mu_floor", "test_degenerate_and_contracts"],
    "I15": ["test_na_lora_exact_retention", "test_I15_", "test_factorwise"],
}


def integrity(outdir):
    import xml.etree.ElementTree as ET
    xml = os.path.join(outdir, "pytest_junit.xml")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(HERE, "tests"), f"--junitxml={xml}"],
                       capture_output=True, text=True, cwd=SPRINT)
    tests = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        name = tc.get("classname", "") .replace(".", "/") + ".py::" + tc.get("name", "")
        name = name.split("tests/")[-1]
        failed = any(ch.tag in ("failure", "error") for ch in tc)
        skipped = any(ch.tag == "skipped" for ch in tc)
        tests.append((name, "fail" if failed else ("skip" if skipped else "pass")))
    res = {}
    for iid, pats in I_MAP.items():
        sel = [t for t in tests if any(pp in t[0] for pp in pats)]
        res[iid] = dict(n=len(sel), passed=sum(1 for t in sel if t[1] == "pass"),
                        failed=[t[0] for t in sel if t[1] == "fail"],
                        skipped=sum(1 for t in sel if t[1] == "skip"))
    # runner-level checks on dev seed 0
    rl = {}
    for fam, cid in (("G1", "d2_sketch_covered"), ("G5", "gain1_generic"), ("G8", "generic")):
        cell = [c for c in family_cells(fam) if c["id"] == cid][0]
        U = build_unit(fam, cell, 0, ())
        rl[f"I10_{fam}"] = dict(dups=disjointness(U))
        calib = dict(lr0=1e-3, lambda0=1.0, tau0_sq=1.0)
        bad = []
        for arm in ARMS16:
            cfg = dict(arm_configs(arm, calib)[0])
            cfg["lr"] = 0.0
            if arm == "adamw_stepmatch":
                cfg["ref_norms"] = [0.0] * U.T
            U.T, U.ck = 3, [0, 1, 2, 3]
            r = run_arm(U, arm, cfg, 300, f3=False)
            if r["status"] == "NOT_APPLICABLE":
                continue
            tr = r["traj"]
            Es = [unr7(v) for k in ("E_sel", "E_test") for v in tr[k]]
            rhos = [unr7(v) for k in tr for v in tr[k] if k.startswith("rho_") and k != "rho_R1_gap"]
            if r["status"] != "ok" or any(v != 0.0 for v in Es) or any(v != 0.0 for v in rhos):
                bad.append(dict(arm=arm, status=r["status"], maxE=max(abs(v) for v in Es),
                                maxrho=max(abs(v) for v in rhos)))
        rl[f"I5_runner_{fam}"] = dict(failures=bad)
        if fam == "G8":
            ob = evaluate(U, U.theta0)
            rl["I12_runner"] = dict(E_sel=ob["E_sel"], E_test=ob["E_test"], rho_test=ob["rho_test"],
                                    ok=ob["E_test"] == 0.0 and ob["rho_test"] == 0.0 and ob["E_sel"] == 0.0)
    try:
        check_freeze(require=True)
        rl["I11_runner"] = dict(ok=False, note="validation was NOT refused")
    except SystemExit as e:
        rl["I11_runner"] = dict(ok=True, refused=str(e))
    res["runner_level"] = rl
    ok = (all(not v["failed"] and v["n"] > 0 for k, v in res.items() if k.startswith("I"))
          and all(v.get("dups", 0) == 0 for k, v in rl.items() if k.startswith("I10"))
          and all(not v["failures"] for k, v in rl.items() if k.startswith("I5"))
          and rl["I12_runner"]["ok"] and rl["I11_runner"]["ok"])
    res["all_pass"] = bool(ok)
    res["pytest_rc"] = p.returncode
    res["pytest_tail"] = p.stdout[-400:]
    return res


# ------------------------------------------------------------------------------------------------------------
# FREEZE guard
# ------------------------------------------------------------------------------------------------------------
def check_freeze(require=True):
    if not sd.freeze_ok(FREEZE_PATH):
        raise SystemExit("REFUSED: configs/FREEZE.json missing or its hashes (eval_protocol.json, candidate.json, "
                         "code/vau tree, run_validation.py) do not match the current files")
    with open(FREEZE_PATH) as f:
        fz = json.load(f)
    for k in ("calibration", "timings"):
        if k not in fz:
            raise SystemExit(f"REFUSED: FREEZE.json lacks '{k}'")
    return fz


# ------------------------------------------------------------------------------------------------------------
# phases
# ------------------------------------------------------------------------------------------------------------
FAMILY_ORDER = ["G1", "G2", "G3", "G4", "G5", "G7", "G9", "G8", "GD_B", "GD_D"]


def unit_specs(families, seeds, calib, ladder, outdir, time_limit, cell_filter=None):
    specs = []
    for fam in families:
        for cell in family_cells(fam, ladder):
            if cell_filter and fam in cell_filter and cell["id"] not in cell_filter[fam]:
                continue
            for s in seeds:
                specs.append(dict(family=fam, cell=cell, seed=s, calib=calib, ladder=list(ladder),
                                  time_limit=time_limit,
                                  out_path=os.path.join(outdir, "raw", fam, f"{cell['id']}__s{s}.jsonl.gz")))
    return specs


def execute(pool, specs, outdir, budget_core_h=None):
    done, cpu, stopped = [], 0.0, False
    with open(os.path.join(outdir, "units.jsonl"), "a") as log:
        for res in pool.imap_unordered(run_unit, specs):
            done.append(res)
            cpu += res["cpu_s"]
            log.write(json.dumps(res) + "\n"); log.flush()
            if budget_core_h is not None and cpu / 3600 > budget_core_h:
                stopped = True
                pool.terminate()
                break
    return done, cpu, stopped


def post_f1(outdir, families, calib):
    """F1 inputs are attached in analysis from raw G1 records (needs lr0)."""
    return


def run_analysis(outdir, families, phase, calib):
    S = analyse_with_f1(outdir, families, phase, calib)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=lambda o: r7(o) if isinstance(o, float) else str(o))
    return S


def analyse_with_f1(outdir, families, phase, calib):
    # wrap analyse so the G1 cell cache gets the F1 trajectories
    global attach_hook
    S = analyse(outdir, families, None, phase)
    if "G1" in families:
        recs = load_family(outdir, "G1")
        ccache = {}
        for cid in ("d2_exact_covered", "d3_exact_covered", "d2_exact_generic", "d3_exact_generic"):
            rs = [r for r in recs if r.get("cell") == cid]
            if rs:
                ent = {}
                attach_f1(ent, rs, calib["G1"]["lr0"])
                ccache[("G1", cid)] = dict(ent, cm={})
        S["stats"]["F1"] = falsifier_f1(ccache)
        S["verdict"] = verdict(S)
    return S


def projection(units_done, calib_cpu, ladder):
    """Validation cost from measured dev unit CPU: per family median unit CPU x (#cells x 12 seeds)."""
    per_fam = {}
    for u in units_done:
        per_fam.setdefault(u["family"], []).append(u["cpu_s"])
    proj, rows = 0.0, {}
    for fam in FAMILY_ORDER:
        if fam not in per_fam:
            continue
        n_cells = len(family_cells(fam, ladder))
        med = float(np.median(per_fam[fam]))
        ch = med * n_cells * 12 / 3600
        rows[fam] = dict(median_unit_cpu_s=med, n_cells=n_cells, core_h=ch)
        proj += ch
    return proj, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "validation"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-integrity", action="store_true")
    ap.add_argument("--dev-seeds", default="0,1")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit("at most 4 workers")
    if os.environ.get("VAU_METERED") != "1":
        env = dict(os.environ, VAU_METERED="1", METER_WALL_S=os.environ.get("METER_WALL_S", "21600"))
        cmd = [sys.executable, os.path.join(HERE, "meter.py"), f"E_run_{args.phase}", "--", sys.executable,
               os.path.abspath(__file__)] + sys.argv[1:]
        os.execve(sys.executable, cmd, env)
    outdir = os.path.join(SPRINT, "results", args.phase)
    os.makedirs(outdir, exist_ok=True)
    if args.phase == "validation":
        fz = check_freeze(require=True)
        calib, ladder = fz["calibration"], tuple(fz.get("ladder", []))
        tl = {m: 10 * float(fz["timings"][m]["median_run_cpu_s"]) for m in ("a", "b", "c")}
        seeds = list(sd.VALIDATION)
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            if not args.analyse_only:
                specs = unit_specs(FAMILY_ORDER, seeds, calib, ladder, outdir, tl)
                done, cpu, stopped = execute(pool, specs, outdir, budget_core_h=6.0)
        S = run_analysis(outdir, FAMILY_ORDER, "validation", calib)
        print(json.dumps(S["verdict"], indent=1))
        return
    # ---------------- dev phase ----------------
    ctx = mp.get_context("fork")
    t_all = time.time()
    dev_seeds = [int(s) for s in args.dev_seeds.split(",")]
    for s in dev_seeds + list(range(8)):
        sd.check_seed(s)
        if s not in sd.DEV:
            raise SystemExit("dev phase uses development seeds only")
    if not args.skip_integrity and not args.analyse_only:
        integ = integrity(outdir)
        with open(os.path.join(outdir, "integrity.json"), "w") as f:
            json.dump(integ, f, indent=1)
        print("integrity all_pass:", integ["all_pass"], flush=True)
    with ctx.Pool(args.workers) as pool:
        cal_path = os.path.join(outdir, "calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                calib = json.load(f)
        else:
            t0 = time.time()
            calib = calibrate(pool, list(CALIB_CELLS))
            calib["_meta"] = dict(seeds=list(range(8)), wall_s=time.time() - t0)
            with open(cal_path, "w") as f:
                json.dump(calib, f, indent=1)
        print("calibration:", {k: (v["lr0"], v["lambda0"], r7(v["tau0_sq"])) for k, v in calib.items()
                               if not k.startswith("_")}, flush=True)
        ladder = ()
        if not args.analyse_only:
            # one representative cell per family (its first primary or safety cell), dev seeds
            pick = {"G1": ["d2_exact_covered", "d2_sketch_covered", "d0_sketch_generic", "d2_exact_generic",
                           "d2_sketch_generic"],
                    "G2": ["k1e+06_phipi4", "k1e+06_phi0"], "G3": ["cospsi0.1"], "G4": ["a_L1e-06_eps0",
                                                                                         "b_L0.0001_epsnat"],
                    "G5": ["gain1_generic", "gain1_covered"], "G7": ["K5_k32"], "G9": ["zeroinit"],
                    "G8": ["generic", "covered"], "GD_B": None, "GD_D": None}
            specs = unit_specs(FAMILY_ORDER, dev_seeds, calib, ladder, outdir,
                               {"a": 120.0, "b": 120.0, "c": 300.0}, cell_filter={k: v for k, v in pick.items() if v})
            done, cpu, _ = execute(pool, specs, outdir)
            with open(os.path.join(outdir, "dev_units.json"), "w") as f:
                json.dump(done, f, indent=1)
        else:
            with open(os.path.join(outdir, "dev_units.json")) as f:
                done = json.load(f)
    S = run_analysis(outdir, FAMILY_ORDER, "dev", calib)
    # compute projection and ladder
    run_cpu = {}
    for fam in FAMILY_ORDER:
        for r in load_family(outdir, fam):
            if "cpu_s" in r and "arm" in r and r.get("status") == "ok":
                mk = "c" if fam in ("G8", "GD_D") else ("a" if fam in ("G1", "G2", "G3") or
                                                        (fam == "G4" and r["cell"].startswith("a_")) else "b")
                run_cpu.setdefault(mk, []).append(unr7(r["cpu_s"]))
    ladder_applied = []
    proj, rows = projection(done, None, tuple(ladder_applied))
    rungs = ["L1", "L2", "L3", "L4", "L5"]
    scale = {"L1": ("G8", 0.6), "L2": ("ab", 40 / 60), "L3": ("G7", 3 / 4), "L4": ("G5", 3 / 4), "L5": ("G8", 24 / 28)}
    proj_now, rows_now = proj, json.loads(json.dumps(rows))
    for rung in rungs:
        if proj_now <= 4.0:
            break
        ladder_applied.append(rung)
        tgt, f = scale[rung]
        for fam, rr in rows_now.items():
            hit = (tgt == "ab" and fam in ("G1", "G2", "G3", "G4", "G5", "G7", "G9")) or fam == tgt
            if hit:
                rr["core_h"] *= f
        proj_now = sum(rr["core_h"] for rr in rows_now.values())
    cp = dict(label="DEVELOPMENT measurement (dev seeds); projection for the 12 validation seeds",
              median_run_cpu_s={m: float(np.median(v)) for m, v in run_cpu.items()},
              per_family=rows_now, projected_core_h_before_ladder=proj, projected_core_h=proj_now,
              ladder_applied=ladder_applied, budget_hard_stop_core_h=6.0,
              note="ladder effects estimated by scaling (L1 eval noises ~0.6 of model-c cost, L2 T 60->40, L3/L4 drop "
                   "one cell, L5 iterations 28->24); re-measure after applying",
              wall_s_dev_phase=time.time() - t_all,
              calibration_cpu_s=float(sum(v.get("cpu_s", 0) for k, v in calib.items() if not k.startswith("_"))))
    with open(os.path.join(outdir, "compute_projection.json"), "w") as f:
        json.dump(cp, f, indent=1)
    print(json.dumps(dict(projected_core_h=proj_now, ladder=ladder_applied, verdict_DEV=S["verdict"]), indent=1))


if __name__ == "__main__":
    main()
